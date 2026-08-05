//! Main-world LOD controller: a retained chunk octree around the camera,
//! advanced in atomic epochs.
//!
//! Every LOD change is planned as an *epoch*: one batch of splits, merges
//! and seam remeshes whose meshes are generated hidden (or held, for
//! in-place remeshes) and committed in a single frame only when every
//! member is drawable. Between commits the shown configuration never
//! changes, so every shown mesh's seam mask is consistent with its shown
//! neighbors at every frame — cracks are structurally impossible, not just
//! transiently unlikely. (GDVoxelTerrain serializes the same way — its LOD
//! camera is frozen while a build or any meshing is in flight; the atomic
//! commit is the strict version of that.)
//!
//! Seam masks derive from the *post-epoch shown configuration*, not from
//! the desired LOD field: the field (a pure function of the sticky anchor)
//! only chooses which splits/merges an epoch attempts, and a ±1-adjacency
//! fixpoint filters out transitions the shown tree isn't ready for.

use std::sync::Arc;

use bevy::platform::collections::{HashMap, HashSet};

use bevy::math::DVec3;
use bevy::prelude::*;
use voxel_core::csg::CsgOp;
use voxel_core::ChunkKey;
use voxel_render::{ChunkCommand, ChunkCommandQueue, ChunkReadyChannel, SharedRenderStats};

/// Optional hook supplying planning-layer CSG ops for a requested chunk
/// (already AABB-culled to the chunk). Installed by the app/worldgen.
#[derive(Resource, Default)]
pub struct ChunkOpsProvider(pub Option<Arc<dyn Fn(ChunkKey) -> Vec<CsgOp> + Send + Sync>>);

fn request(
    queue: &ChunkCommandQueue,
    provider: &ChunkOpsProvider,
    key: ChunkKey,
    show_on_ready: bool,
    hold: bool,
    face_mask: u32,
) {
    let ops = provider
        .0
        .as_ref()
        .map(|f| f(key))
        .filter(|v| !v.is_empty())
        .map(Arc::new);
    queue.push(ChunkCommand::Request {
        key,
        show_on_ready,
        hold,
        ops,
        face_mask,
    });
}

/// Face directions in mask order (+x, -x, +y, -y, +z, -z).
const FACE_DIRS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// The LOD field: does the field want this chunk refined? A pure function
/// of (chunk, quantized camera anchor). Advisory only — it drives which
/// transitions an epoch attempts; seam masks come from the shown tree.
fn split_wanted(config: &LodConfig, anchor: DVec3, key: ChunkKey) -> bool {
    key.level > 0 && aabb_distance(anchor, key) < config.split_k * key.edge_m()
}

/// LOD configuration.
#[derive(Resource)]
pub struct LodConfig {
    /// Coarsest chunk level. Edge = 32 · 2^max_level meters.
    pub max_level: u8,
    /// Radius, in top-level chunks, kept loaded around the camera.
    pub top_radius: i32,
    /// Vertical range of top-level chunks (inclusive).
    pub top_y: (i32, i32),
    /// Split when camera distance < split_k × edge.
    pub split_k: f64,
    /// Merge when camera distance > merge_k × parent edge. Must exceed
    /// `split_k` for hysteresis.
    pub merge_k: f64,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            // Top chunks of 8.2 km; ring radius 3 → ~25 km view distance.
            max_level: 8,
            top_radius: 3,
            top_y: (-1, 0),
            split_k: 2.5,
            merge_k: 3.0,
        }
    }
}

/// Set to request a full streaming rebuild (e.g. after a hot-reloaded
/// level changes generation parameters): every chunk is freed and the tree
/// restarts from the top-level ring with the current providers.
#[derive(Resource, Default)]
pub struct StreamingRebuild(pub bool);

/// One planned batch of LOD transitions, committed atomically.
struct Epoch {
    /// Shown leaf → the 8 children replacing it.
    splits: Vec<(ChunkKey, [ChunkKey; 8])>,
    /// Hidden parent → the 8 shown leaves it replaces.
    merges: Vec<(ChunkKey, [ChunkKey; 8])>,
    /// Every mesh the commit waits for → the seam mask it must carry
    /// (empty chunks report `u32::MAX` and satisfy any expectation).
    waits: HashMap<ChunkKey, u32>,
    /// Requests not yet issued, trickled a budget per frame (safe: nothing
    /// swaps until commit, so deferral can't show stale seams).
    /// (key, mask, hold) — hold marks in-place remeshes of shown chunks.
    to_request: Vec<(ChunkKey, u32, bool)>,
}

/// Generation requests issued per frame while an epoch is in flight.
const EPOCH_REQUEST_BUDGET: usize = 64;

/// Structural changes attempted per epoch. Caps bound the planning burst
/// and the epoch's generation load; refinement just takes more epochs.
const EPOCH_MAX_SPLITS: usize = 24;
const EPOCH_MAX_MERGES: usize = 24;

#[derive(Resource, Default)]
struct LodTree {
    /// Currently-shown chunks. Changes only at epoch commit (plus additive
    /// top-ring arrivals and evictions between epochs).
    leaves: HashSet<ChunkKey>,
    /// The single in-flight epoch, if any.
    epoch: Option<Epoch>,
    /// Latest drawable mesh per chunk, with the seam mask it was built
    /// with (u32::MAX = empty, satisfies any expectation).
    ready: HashMap<ChunkKey, u32>,
    /// Top-level cells whose subtree is live.
    top_cells: HashSet<IVec3>,
    /// Seam mask of each shown chunk's committed (or in-flight requested)
    /// mesh; the remesh scan compares against post-epoch masks.
    sent_masks: HashMap<ChunkKey, u32>,
    /// Quantized camera anchor the LOD field is evaluated at. Sticky —
    /// this is the hysteresis; it is only read when planning an epoch.
    anchor: Option<DVec3>,
}

/// The shown configuration a plan would produce: current leaves minus
/// `removed` (split parents, merge children) plus `added` (split children,
/// merge parents). All seam masks and adjacency checks evaluate against
/// this — the state the commit will make real.
/// Do the two chunks' closed axis-aligned boxes share at least a point
/// (face-, edge- or corner-touch)? Coordinates in level-0 cells.
fn boxes_touch(a: ChunkKey, b: ChunkKey) -> bool {
    let (amin, amax) = key_box(a);
    let (bmin, bmax) = key_box(b);
    amin.cmple(bmax).all() && bmin.cmple(amax).all()
}

fn key_box(k: ChunkKey) -> (bevy::math::I64Vec3, bevy::math::I64Vec3) {
    let min = k.pos.as_i64vec3() << (k.level as i64);
    let max = (k.pos + IVec3::ONE).as_i64vec3() << (k.level as i64);
    (min, max)
}

struct PostState<'a> {
    leaves: &'a HashSet<ChunkKey>,
    added: HashSet<ChunkKey>,
    removed: HashSet<ChunkKey>,
}

impl PostState<'_> {
    fn current(leaves: &HashSet<ChunkKey>) -> PostState<'_> {
        PostState {
            leaves,
            added: HashSet::new(),
            removed: HashSet::new(),
        }
    }

    fn plan<'a>(
        leaves: &'a HashSet<ChunkKey>,
        splits: &[ChunkKey],
        merges: &[ChunkKey],
    ) -> PostState<'a> {
        let mut added = HashSet::new();
        let mut removed = HashSet::new();
        for p in splits {
            removed.insert(*p);
            added.extend(p.children());
        }
        for p in merges {
            added.insert(*p);
            removed.extend(p.children());
        }
        PostState {
            leaves,
            added,
            removed,
        }
    }

    fn is_leaf(&self, key: ChunkKey) -> bool {
        self.added.contains(&key) || (self.leaves.contains(&key) && !self.removed.contains(&key))
    }

    /// Level of the leaf covering `key`'s region at `key.level` or above.
    fn covering_level(&self, max_level: u8, key: ChunkKey) -> Option<u8> {
        let mut k = key;
        loop {
            if self.is_leaf(k) {
                return Some(k.level);
            }
            if k.level >= max_level {
                return None;
            }
            k = k.parent();
        }
    }

    /// Any leaf strictly finer than `min_level` inside `region` that
    /// actually touches `target`'s box (shares at least a corner)? Leaves
    /// deep inside the region can't produce a seam with `target`, so they
    /// must not veto its transitions.
    fn has_touching_finer_than(&self, region: ChunkKey, target: ChunkKey, min_level: u8) -> bool {
        if !boxes_touch(region, target) {
            return false;
        }
        if self.is_leaf(region) {
            return region.level < min_level;
        }
        if region.level == 0 {
            return false;
        }
        region
            .children()
            .iter()
            .any(|c| self.has_touching_finer_than(*c, target, min_level))
    }

    /// May `region` be shown at `level` without a ≥2-level jump anywhere in
    /// its 26-neighborhood? (Seams only bridge one level.)
    fn adjacency_ok(&self, max_level: u8, region: ChunkKey, level: u8) -> bool {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let n = ChunkKey::new(region.level, region.pos + IVec3::new(dx, dy, dz));
                    if let Some(l) = self.covering_level(max_level, n) {
                        if l > level + 1 {
                            return false;
                        }
                    }
                    if level >= 2 && self.has_touching_finer_than(n, region, level - 1) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Add to `out` every leaf covering or inside `key`'s region that
    /// touches `target`'s box. Descends only into touching subtrees.
    fn collect_touching_leaves(
        &self,
        max_level: u8,
        key: ChunkKey,
        target: ChunkKey,
        out: &mut HashSet<ChunkKey>,
    ) {
        let mut k = key;
        loop {
            if self.is_leaf(k) {
                if boxes_touch(k, target) {
                    out.insert(k);
                }
                return;
            }
            if k.level >= max_level {
                break;
            }
            k = k.parent();
        }
        self.descend_touching(key, target, out);
    }

    fn descend_touching(&self, region: ChunkKey, target: ChunkKey, out: &mut HashSet<ChunkKey>) {
        if !boxes_touch(region, target) {
            return;
        }
        if self.is_leaf(region) {
            out.insert(region);
            return;
        }
        if region.level == 0 {
            return;
        }
        for c in region.children() {
            self.descend_touching(c, target, out);
        }
    }

    /// Seam mask: one coarser-neighbor bit per direction of the
    /// 26-neighborhood, in scan order (dz, dy, dx in -1..=1, center
    /// skipped) — twin of `snap_to_parity` in the mesh shader. Diagonal
    /// (edge/corner) coarser neighbors must snap too, or pinholes open at
    /// junctions where only the face neighbors snap.
    fn seam_mask(&self, max_level: u8, key: ChunkKey) -> u32 {
        let mut mask = 0u32;
        let mut idx = 0;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let n = ChunkKey::new(key.level, key.pos + IVec3::new(dx, dy, dz));
                    if matches!(self.covering_level(max_level, n), Some(l) if l > key.level) {
                        mask |= 1 << idx;
                    }
                    idx += 1;
                }
            }
        }
        mask
    }
}

pub struct VoxelStreamingPlugin;

impl Plugin for VoxelStreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LodConfig>()
            .init_resource::<ChunkOpsProvider>()
            .init_resource::<StreamingRebuild>()
            .init_resource::<LodTree>()
            .add_systems(Update, (lod_tick, hud_stats));
    }
}

fn aabb_distance(camera: DVec3, key: ChunkKey) -> f64 {
    let min = key.min_corner_m();
    let max = min + DVec3::splat(key.edge_m());
    camera.distance(camera.clamp(min, max))
}

/// The top-level ancestor cell of a key.
fn top_ancestor(key: ChunkKey, max_level: u8) -> IVec3 {
    let mut k = key;
    while k.level < max_level {
        k = k.parent();
    }
    k.pos
}

/// Plan the next epoch: field-wanted splits/merges filtered to transitions
/// the shown tree can absorb (±1 fixpoint), plus every seam remesh the
/// resulting configuration requires. Returns None when nothing changes.
fn plan_epoch(tree: &LodTree, config: &LodConfig, anchor: DVec3) -> Option<Epoch> {
    let mut splits: Vec<ChunkKey> = tree
        .leaves
        .iter()
        .filter(|l| split_wanted(config, anchor, **l))
        .copied()
        .collect();
    // Nearest-first, capped: with post-state masks a cap only slows
    // refinement (the fixpoint keeps every intermediate config seam-legal),
    // and it bounds both the planning burst and the per-epoch GPU load.
    splits.sort_by(|a, b| {
        aabb_distance(anchor, *a)
            .total_cmp(&aabb_distance(anchor, *b))
    });
    splits.truncate(EPOCH_MAX_SPLITS);

    let mut sibling_count: HashMap<ChunkKey, u8> = HashMap::new();
    for leaf in &tree.leaves {
        if leaf.level >= config.max_level {
            continue;
        }
        *sibling_count.entry(leaf.parent()).or_default() += 1;
    }
    let mut merges: Vec<ChunkKey> = sibling_count
        .into_iter()
        .filter(|(p, c)| *c == 8 && !split_wanted(config, anchor, *p))
        .map(|(p, _)| p)
        .collect();
    merges.sort_by(|a, b| {
        aabb_distance(anchor, *b)
            .total_cmp(&aabb_distance(anchor, *a))
    });
    merges.truncate(EPOCH_MAX_MERGES);

    // Force-split closure: a split whose children would sit two levels
    // finer than a touching shown leaf must not be vetoed — the field has
    // legitimate diagonal 2-jumps, so a veto clamps refinement there
    // forever and the clamp cascades inward (each frozen region vetoes its
    // finer neighbor's split, all the way to the camera; ops-gated content
    // then never generates). Restricted-octree rule instead: split the
    // too-coarse covering leaf in the same epoch, whether or not the field
    // wants it. The merge veto below keeps forced splits stable.
    loop {
        let post = PostState::plan(&tree.leaves, &splits, &merges);
        let mut forced: Vec<ChunkKey> = Vec::new();
        for p in &splits {
            let child_level = p.level - 1;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let n = ChunkKey::new(p.level, p.pos + IVec3::new(dx, dy, dz));
                        let Some(l) = post.covering_level(config.max_level, n) else {
                            continue;
                        };
                        if l <= child_level + 1 {
                            continue;
                        }
                        let mut k = n;
                        while k.level < l {
                            k = k.parent();
                        }
                        if !splits.contains(&k) && !forced.contains(&k) {
                            forced.push(k);
                        }
                    }
                }
            }
        }
        if forced.is_empty() {
            break;
        }
        // A forced split overrides any merge that involves the same
        // region (as the merge parent or one of its children).
        merges.retain(|m| !forced.iter().any(|f| *m == *f || *m == f.parent()));
        splits.extend(forced);
    }

    // ±1 veto for merges: coarsening waits until the neighborhood can
    // absorb it (this is what keeps forced splits from flapping back).
    loop {
        let n0 = merges.len();
        let post = PostState::plan(&tree.leaves, &splits, &merges);
        let keep_merges: Vec<ChunkKey> = merges
            .iter()
            .filter(|p| post.adjacency_ok(config.max_level, **p, p.level))
            .copied()
            .collect();
        merges = keep_merges;
        if merges.len() == n0 {
            break;
        }
    }
    if splits.is_empty() && merges.is_empty() {
        return None;
    }

    let post = PostState::plan(&tree.leaves, &splits, &merges);
    let mut epoch = Epoch {
        splits: Vec::new(),
        merges: Vec::new(),
        waits: HashMap::new(),
        to_request: Vec::new(),
    };
    for p in splits {
        let children = p.children();
        for c in children {
            let m = post.seam_mask(config.max_level, c);
            epoch.waits.insert(c, m);
            epoch.to_request.push((c, m, false));
        }
        epoch.splits.push((p, children));
    }
    for p in merges {
        let m = post.seam_mask(config.max_level, p);
        epoch.waits.insert(p, m);
        epoch.to_request.push((p, m, false));
        epoch.merges.push((p, p.children()));
    }
    // Seam remeshes: every kept leaf whose mask changes under the
    // post-epoch configuration regenerates (held) and commits with it. A
    // mask can only change for leaves touching a region whose level
    // changes, so enumerate those from the changed regions (26 neighbor
    // keys each, descending only into touching subtrees) — an all-leaves
    // scan is an O(leaves × changes) planning burst.
    let mut candidates: HashSet<ChunkKey> = HashSet::new();
    for (r, _) in epoch.splits.iter().chain(epoch.merges.iter()) {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let n = ChunkKey::new(r.level, r.pos + IVec3::new(dx, dy, dz));
                    post.collect_touching_leaves(config.max_level, n, *r, &mut candidates);
                }
            }
        }
    }
    for leaf in candidates {
        if !tree.leaves.contains(&leaf) || post.removed.contains(&leaf) {
            continue;
        }
        let m = post.seam_mask(config.max_level, leaf);
        if tree.sent_masks.get(&leaf) != Some(&m) {
            epoch.waits.insert(leaf, m);
            epoch.to_request.push((leaf, m, true));
        }
    }
    Some(epoch)
}

fn commit_epoch(tree: &mut LodTree, queue: &ChunkCommandQueue, epoch: Epoch) {
    for (parent, children) in &epoch.splits {
        tree.leaves.remove(parent);
        tree.ready.remove(parent);
        tree.sent_masks.remove(parent);
        queue.push(ChunkCommand::Free(*parent));
        tree.leaves.extend(children.iter().copied());
    }
    for (parent, children) in &epoch.merges {
        tree.leaves.insert(*parent);
        for c in children {
            tree.leaves.remove(c);
            tree.ready.remove(c);
            tree.sent_masks.remove(c);
            queue.push(ChunkCommand::Free(*c));
        }
    }
    // One command batch: every member becomes visible / swaps its held
    // mesh in the same frame the replaced chunks are freed.
    for key in epoch.waits.keys() {
        queue.push(ChunkCommand::Commit(*key));
    }
}

#[allow(clippy::too_many_arguments)]
fn lod_tick(
    config: Res<LodConfig>,
    mut tick_worst: Local<(f32, f32)>,
    mut tree: ResMut<LodTree>,
    queue: Res<ChunkCommandQueue>,
    ops_provider: Res<ChunkOpsProvider>,
    mut rebuild: ResMut<StreamingRebuild>,
    ready_rx: Res<ChunkReadyChannel>,
    mut field: ResMut<voxel_render::FieldParams>,
    stats: Res<SharedRenderStats>,
    cameras: Query<&Transform, (With<Camera3d>, Without<voxel_render::HelperCamera>)>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let tick_start = std::time::Instant::now();
    let camera = camera.translation.as_dvec3();
    let tree = &mut *tree;

    // Sticky quantized anchor. Only read when planning an epoch, so it may
    // move freely while one is in flight.
    let anchor_moved = match tree.anchor {
        Some(a) if camera.distance(a) < 48.0 => false,
        _ => {
            tree.anchor = Some(camera);
            true
        }
    };
    let anchor = tree.anchor.unwrap();
    if anchor_moved {
        field.anchor = anchor.as_vec3();
        field.dist_scale = (config.split_k * 32.0) as f32;
        field.max_vs = (1u32 << config.max_level) as f32;
    }

    // 0. Full rebuild: free every requested chunk and restart from the top
    //    ring (used when generation parameters hot-reload).
    if rebuild.0 {
        rebuild.0 = false;
        let mut requested: HashSet<ChunkKey> = tree.leaves.iter().copied().collect();
        if let Some(epoch) = &tree.epoch {
            requested.extend(epoch.waits.keys().copied());
        }
        for key in requested {
            queue.push(ChunkCommand::Free(key));
        }
        *tree = LodTree::default();
    }

    // 1. Absorb readiness notifications.
    for (key, mask) in ready_rx.rx.try_iter() {
        tree.ready.insert(key, mask);
    }

    // 2. Top-level ring maintenance. Additions are purely additive (their
    //    faces are equal-level or unstreamed) and show as soon as ready.
    let top_edge = ChunkKey::new(config.max_level, IVec3::ZERO).edge_m();
    let center_x = (camera.x / top_edge).floor() as i32;
    let center_z = (camera.z / top_edge).floor() as i32;
    let r = config.top_radius;
    for dz in -r..=r {
        for dx in -r..=r {
            for y in config.top_y.0..=config.top_y.1 {
                let cell = IVec3::new(center_x + dx, y, center_z + dz);
                if tree.top_cells.insert(cell) {
                    let key = ChunkKey::new(config.max_level, cell);
                    let mask = PostState::current(&tree.leaves).seam_mask(config.max_level, key);
                    tree.sent_masks.insert(key, mask);
                    request(&queue, &ops_provider, key, true, false, mask);
                    tree.leaves.insert(key);
                }
            }
        }
    }
    // Evictions wait for the epoch: freeing a member would wedge its
    // commit, and the stale ring is far behind the camera anyway.
    if tree.epoch.is_none() {
        let keep = r + 1;
        let stale: Vec<IVec3> = tree
            .top_cells
            .iter()
            .filter(|c| (c.x - center_x).abs() > keep || (c.z - center_z).abs() > keep)
            .copied()
            .collect();
        for cell in stale {
            tree.top_cells.remove(&cell);
            free_subtree(tree, &queue, cell, config.max_level);
        }
    }

    // 3. Plan the next epoch when none is in flight.
    if tree.epoch.is_none() {
        let t = std::time::Instant::now();
        tree.epoch = plan_epoch(tree, &config, anchor);
        let ms = t.elapsed().as_secs_f32() * 1000.0;
        if ms > 8.0 && std::env::var("VOXEL_LOG_FPS").is_ok() {
            info!(
                "plan_epoch {ms:.1} ms ({} waits)",
                tree.epoch.as_ref().map_or(0, |e| e.waits.len())
            );
        }
    }

    // 4. Advance the in-flight epoch: trickle its generation requests,
    //    then commit atomically once every member is drawable.
    if let Some(mut epoch) = tree.epoch.take() {
        let n = epoch.to_request.len().min(EPOCH_REQUEST_BUDGET);
        for (key, mask, hold) in epoch.to_request.drain(..n) {
            tree.sent_masks.insert(key, mask);
            request(&queue, &ops_provider, key, false, hold, mask);
        }
        let done = epoch.to_request.is_empty()
            && epoch
                .waits
                .iter()
                .all(|(k, m)| matches!(tree.ready.get(k), Some(&r) if r == *m || r == u32::MAX));
        if done {
            commit_epoch(tree, &queue, epoch);
        } else {
            tree.epoch = Some(epoch);
        }
    }

    // VOXEL_VALIDATE_SEAMS=1: every ~30 frames, verify that every shown
    // chunk's requested mask matches the mask the current shown
    // configuration demands, and that its drawable mesh carries it.
    // Any mismatch is a crack on screen — log precisely which chunk.
    if std::env::var("VOXEL_VALIDATE_SEAMS").is_ok() {
        static FRAME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 30 == 0 {
            // Ground truth: compare each DRAWN mesh's mask (exported by the
            // render world) against what the current shown configuration
            // demands. Any mismatch is a potential crack on screen.
            let drawn: Vec<(ChunkKey, u32)> = stats
                .0
                .lock()
                .map(|s| s.drawn_masks.clone())
                .unwrap_or_default();
            let post = PostState::current(&tree.leaves);
            let mut stale = 0u32;
            for (key, mask) in &drawn {
                if !tree.leaves.contains(key) {
                    continue; // freed/swapped since export
                }
                let want = post.seam_mask(config.max_level, *key);
                if *mask != want {
                    stale += 1;
                    if stale <= 4 {
                        info!(
                            "seam-validate: DRAWN mask stale for {key:?}: want {want:#x} drawn {mask:#x}"
                        );
                    }
                }
            }
            if stale > 0 {
                info!(
                    "seam-validate: {stale} stale drawn of {} drawn meshes (epoch {})",
                    drawn.len(),
                    if tree.epoch.is_some() { "in-flight" } else { "idle" },
                );
            }
        }
    }

    if std::env::var("VOXEL_LOG_FPS").is_ok() {
        let ms = tick_start.elapsed().as_secs_f32() * 1000.0;
        tick_worst.0 += ms;
        tick_worst.1 = tick_worst.1.max(ms);
        if tick_worst.0 > 250.0 {
            info!("lod_tick worst {:.1} ms in window", tick_worst.1);
            *tick_worst = (0.0, 0.0);
        }
    }
}

/// Free every requested chunk whose subtree hangs under `cell`. Only
/// called between epochs, so no in-flight members are touched.
fn free_subtree(tree: &mut LodTree, queue: &ChunkCommandQueue, cell: IVec3, max_level: u8) {
    let in_subtree = |key: &ChunkKey| top_ancestor(*key, max_level) == cell;

    let mut to_free: HashSet<ChunkKey> = HashSet::new();
    tree.leaves.retain(|k| {
        let stale = in_subtree(k);
        if stale {
            to_free.insert(*k);
        }
        !stale
    });
    for key in to_free {
        tree.ready.remove(&key);
        tree.sent_masks.remove(&key);
        queue.push(ChunkCommand::Free(key));
    }
    tree.ready.retain(|k, _| !in_subtree(k));
    tree.sent_masks.retain(|k, _| !in_subtree(k));
}

fn hud_stats(
    stats: Res<SharedRenderStats>,
    tree: Res<LodTree>,
    time: Res<Time>,
    mut slow_accum: Local<(f32, u32, f32)>,
    hud: Option<ResMut<voxel_debug::DebugHudExtra>>,
) {
    // VOXEL_LOG_FPS=1: log worst frame time every 2 s so perf regressions
    // are measurable in headless eval runs, not just felt interactively.
    if std::env::var("VOXEL_LOG_FPS").is_ok() {
        let dt = time.delta_secs();
        slow_accum.0 += dt;
        slow_accum.1 += 1;
        slow_accum.2 = slow_accum.2.max(dt);
        if slow_accum.0 >= 2.0 {
            info!(
                "fps avg {:.0} | worst frame {:.0} ms",
                slow_accum.1 as f32 / slow_accum.0,
                slow_accum.2 * 1000.0
            );
            *slow_accum = (0.0, 0, 0.0);
        }
    }
    let Some(mut hud) = hud else {
        return;
    };
    let Ok(s) = stats.0.lock() else {
        return;
    };
    hud.0.push(format!(
        "chunks: {} tracked | {} meshed | {} drawn | {} culled | {} pending",
        s.tracked, s.meshed, s.drawn, s.culled, s.awaiting
    ));
    let occ: Vec<String> = s
        .slab_occupancy
        .iter()
        .map(|(free, total)| format!("{}/{}", total - free, total))
        .collect();
    hud.0.push(format!(
        "arena free: {} | slab used: [{}]",
        s.arena_free,
        occ.join(", ")
    ));

    // Leaf histogram by level (finest first).
    let mut histo: HashMap<u8, usize> = HashMap::new();
    for leaf in &tree.leaves {
        *histo.entry(leaf.level).or_default() += 1;
    }
    let mut levels: Vec<u8> = histo.keys().copied().collect();
    levels.sort_unstable();
    let parts: Vec<String> = levels
        .iter()
        .map(|l| format!("L{l}:{}", histo[l]))
        .collect();
    let epoch = match &tree.epoch {
        Some(e) => format!("{} waiting", e.waits.len()),
        None => "idle".to_string(),
    };
    hud.0.push(format!(
        "leaves: {} [{}] | epoch: {}",
        tree.leaves.len(),
        parts.join(" "),
        epoch,
    ));
}

#[cfg(test)]
mod epoch_invariants {
    use super::*;
    use voxel_core::seed::Rng;

    fn cfg() -> LodConfig {
        LodConfig::default()
    }

    fn rand_anchor(rng: &mut Rng) -> DVec3 {
        DVec3::new(
            (rng.next_f32() as f64 - 0.5) * 40000.0,
            (rng.next_f32() as f64) * 2000.0,
            (rng.next_f32() as f64 - 0.5) * 40000.0,
        )
    }

    /// The field's leaf level at a world position: descend from the top
    /// while the field wants refinement.
    fn leaf_at(config: &LodConfig, anchor: DVec3, p: DVec3) -> ChunkKey {
        let top_edge = ChunkKey::new(config.max_level, IVec3::ZERO).edge_m();
        let cell = (p / top_edge).floor();
        let mut k = ChunkKey::new(
            config.max_level,
            IVec3::new(cell.x as i32, cell.y as i32, cell.z as i32),
        );
        while split_wanted(config, anchor, k) {
            let mut next = k.children()[0];
            for c in k.children() {
                let min = c.min_corner_m();
                let max = min + DVec3::splat(c.edge_m());
                if p.x >= min.x
                    && p.x < max.x
                    && p.y >= min.y
                    && p.y < max.y
                    && p.z >= min.z
                    && p.z < max.z
                {
                    next = c;
                }
            }
            k = next;
        }
        k
    }

    /// Fresh tree with the top-level ring shown, like startup.
    fn top_ring_tree(config: &LodConfig, anchor: DVec3) -> LodTree {
        let mut tree = LodTree::default();
        let top_edge = ChunkKey::new(config.max_level, IVec3::ZERO).edge_m();
        let cx = (anchor.x / top_edge).floor() as i32;
        let cz = (anchor.z / top_edge).floor() as i32;
        for dz in -config.top_radius..=config.top_radius {
            for dx in -config.top_radius..=config.top_radius {
                for y in config.top_y.0..=config.top_y.1 {
                    let cell = IVec3::new(cx + dx, y, cz + dz);
                    tree.top_cells.insert(cell);
                    tree.leaves.insert(ChunkKey::new(config.max_level, cell));
                }
            }
        }
        let masks: Vec<(ChunkKey, u32)> = tree
            .leaves
            .iter()
            .map(|l| {
                (
                    *l,
                    PostState::current(&tree.leaves).seam_mask(config.max_level, *l),
                )
            })
            .collect();
        tree.sent_masks.extend(masks);
        tree
    }

    /// Plan and apply epochs toward `anchor` until quiescent, asserting
    /// the shown configuration's crack-freedom along the way. The
    /// per-epoch structural caps mean convergence takes many small
    /// epochs; the consistency scan is expensive, so it samples commits.
    fn chain_epochs(config: &LodConfig, tree: &mut LodTree, anchor: DVec3) {
        for round in 0..600 {
            let Some(epoch) = plan_epoch(tree, config, anchor) else {
                assert_consistent(config, &tree.leaves, &tree.sent_masks);
                return;
            };
            let _ = round;
            assert!(!epoch.waits.is_empty());
            for (key, mask, _) in &epoch.to_request {
                tree.sent_masks.insert(*key, *mask);
            }
            for (parent, children) in &epoch.splits {
                tree.leaves.remove(parent);
                tree.sent_masks.remove(parent);
                tree.leaves.extend(children.iter().copied());
            }
            for (parent, children) in &epoch.merges {
                tree.leaves.insert(*parent);
                for c in children {
                    tree.leaves.remove(c);
                    tree.sent_masks.remove(c);
                }
            }
            if round % 7 == 0 {
                assert_consistent(config, &tree.leaves, &tree.sent_masks);
            }
        }
        panic!("epochs did not quiesce");
    }

    /// The 26 neighborhood directions in seam-mask scan order.
    fn scan_dirs() -> Vec<IVec3> {
        let mut dirs = Vec::new();
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx != 0 || dy != 0 || dz != 0 {
                        dirs.push(IVec3::new(dx, dy, dz));
                    }
                }
            }
        }
        dirs
    }

    /// Assert the crack-freedom invariants of a shown configuration whose
    /// meshes carry `masks`: neighborhood levels within ±1, and the snap
    /// bit set exactly on the finer side of every unequal pair.
    fn assert_consistent(
        config: &LodConfig,
        leaves: &HashSet<ChunkKey>,
        masks: &HashMap<ChunkKey, u32>,
    ) {
        let post = PostState::current(leaves);
        for leaf in leaves {
            let mask = *masks.get(leaf).expect("shown leaf without a mask");
            assert_eq!(
                mask,
                post.seam_mask(config.max_level, *leaf),
                "shown mesh mask inconsistent with shown neighbors for {leaf:?}"
            );
            for (i, d) in scan_dirs().iter().enumerate() {
                let n = ChunkKey::new(leaf.level, leaf.pos + *d);
                if let Some(l) = post.covering_level(config.max_level, n) {
                    assert!(
                        (l as i32 - leaf.level as i32) <= 1,
                        "2-level jump shown across the neighborhood: {leaf:?} vs level {l}"
                    );
                }
                if leaf.level >= 2 {
                    assert!(
                        !post.has_touching_finer_than(n, *leaf, leaf.level - 1),
                        "touching leaves jump 2 levels (finer side): {leaf:?} dir {d:?}"
                    );
                }
                // Reciprocity: if we snap toward n, the covering coarser
                // neighbor is a shown leaf and must not snap back toward
                // our region (both sides snapping = a real crack).
                if (mask >> i) & 1 == 1 {
                    let cover = n.parent();
                    assert!(post.is_leaf(cover), "snap toward a non-leaf neighbor");
                    let n_mask = *masks.get(&cover).expect("neighbor without a mask");
                    for (j, e) in scan_dirs().iter().enumerate() {
                        let back = ChunkKey::new(cover.level, cover.pos + *e);
                        // Any direction of the coarser leaf that points at
                        // our region must not carry a snap bit.
                        if back == ChunkKey::new(leaf.level, leaf.pos).parent() {
                            assert_eq!(
                                (n_mask >> j) & 1,
                                0,
                                "both sides of a seam snap: {leaf:?} vs {cover:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Refining from a fresh top ring stays consistent after every epoch
    /// commit and quiesces.
    #[test]
    fn refinement_from_scratch_is_consistent() {
        let config = cfg();
        let mut rng = Rng::new(0xE7A1);
        for _ in 0..12 {
            let anchor = rand_anchor(&mut rng);
            let mut tree = top_ring_tree(&config, anchor);
            chain_epochs(&config, &mut tree, anchor);
        }
    }

    /// Moving the anchor mid-flight (small nudges and teleports) replans
    /// through consistent configurations only.
    #[test]
    fn anchor_moves_stay_consistent() {
        let config = cfg();
        let mut rng = Rng::new(0x51EA);
        for _ in 0..8 {
            let a1 = rand_anchor(&mut rng);
            let mut tree = top_ring_tree(&config, a1);
            chain_epochs(&config, &mut tree, a1);
            // Mix of long jumps and small nudges.
            let a2 = if rng.next_f32() < 0.5 {
                a1 + DVec3::new(
                    (rng.next_f32() as f64 - 0.5) * 40000.0,
                    0.0,
                    (rng.next_f32() as f64 - 0.5) * 40000.0,
                )
            } else {
                a1 + DVec3::new(
                    (rng.next_f32() as f64 - 0.5) * 400.0,
                    0.0,
                    (rng.next_f32() as f64 - 0.5) * 400.0,
                )
            };
            chain_epochs(&config, &mut tree, a2);
        }
    }

    /// Face-adjacent leaf levels of the pure field never differ by more
    /// than one (the parity snap only bridges one level).
    #[test]
    fn field_neighbors_within_one_level() {
        let config = cfg();
        let mut rng = Rng::new(0xADD1);
        for _ in 0..2000 {
            let anchor = rand_anchor(&mut rng);
            let p = rand_anchor(&mut rng);
            let a = leaf_at(&config, anchor, p);
            let edge = a.edge_m();
            let center = a.min_corner_m() + DVec3::splat(edge * 0.5);
            for d in FACE_DIRS {
                let q = center + DVec3::new(d.x as f64, d.y as f64, d.z as f64) * edge;
                let b = leaf_at(&config, anchor, q);
                assert!(
                    (a.level as i32 - b.level as i32).abs() <= 1,
                    "leaf levels jump >1 across a face: {a:?} vs {b:?}"
                );
            }
        }
    }
}
