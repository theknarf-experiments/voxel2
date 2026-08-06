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

/// (key, mask, hold, ops) rows queued for the chunk pipeline.
type RequestList = Vec<(ChunkKey, u32, bool, Option<Arc<Vec<CsgOp>>>)>;

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

/// The LOD field: does the field want this chunk refined? A pure function
/// of (chunk, quantized camera anchor). Advisory only — it drives which
/// transitions an epoch attempts; seam masks come from the shown tree.
fn split_wanted(config: &LodConfig, anchor: DVec3, key: ChunkKey) -> bool {
    key.level > 0 && aabb_distance(anchor, key) < config.split_k * key.edge_m()
}

/// LOD configuration.
#[derive(Resource, Clone)]
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

/// The fully-converged starting configuration for a fresh world: the
/// exact final LOD per region (no intermediate levels are ever
/// generated), revealed in one atomic commit — the alternative,
/// coarse-first refinement, reads as "broken, then less broken" on a
/// cold start and wastes the transient rungs.
struct GenesisPlan {
    top_cells: HashSet<IVec3>,
    leaves: HashSet<ChunkKey>,
    sent_masks: HashMap<ChunkKey, u32>,
    waits: HashMap<ChunkKey, u32>,
    to_request: RequestList,
}

/// Simulate epoch refinement to its fixpoint on a scratch tree (pure,
/// runs in the planning task) and emit the converged configuration.
fn plan_genesis(
    config: &LodConfig,
    anchor: DVec3,
    provider: Option<&(dyn Fn(ChunkKey) -> Vec<CsgOp> + Send + Sync)>,
) -> GenesisPlan {
    let t0 = std::time::Instant::now();
    // 1. Pure field descent: the exact configuration the field wants.
    fn descend(config: &LodConfig, anchor: DVec3, k: ChunkKey, out: &mut HashSet<ChunkKey>) {
        if split_wanted(config, anchor, k) {
            for c in k.children() {
                descend(config, anchor, c, out);
            }
        } else {
            out.insert(k);
        }
    }
    let mut leaves: HashSet<ChunkKey> = HashSet::new();
    let mut top_cells: HashSet<IVec3> = HashSet::new();
    let top_edge = ChunkKey::new(config.max_level, IVec3::ZERO).edge_m();
    let cx = (anchor.x / top_edge).floor() as i32;
    let cz = (anchor.z / top_edge).floor() as i32;
    for dz in -config.top_radius..=config.top_radius {
        for dx in -config.top_radius..=config.top_radius {
            for y in config.top_y.0..=config.top_y.1 {
                let cell = IVec3::new(cx + dx, y, cz + dz);
                top_cells.insert(cell);
                descend(config, anchor, ChunkKey::new(config.max_level, cell), &mut leaves);
            }
        }
    }
    // 2. ±1 clamp: the field allows 2-level jumps across diagonals; split
    //    the coarser side until every touching pair is within one level —
    //    the same fixpoint the runtime's force-split closure converges to.
    loop {
        let mut force: HashSet<ChunkKey> = HashSet::new();
        {
            let post = PostState::current(&leaves);
            for leaf in &leaves {
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let n =
                                ChunkKey::new(leaf.level, leaf.pos + IVec3::new(dx, dy, dz));
                            if let Some(l) = post.covering_level(config.max_level, n) {
                                if l > leaf.level + 1 {
                                    let mut k = n;
                                    while k.level < l {
                                        k = k.parent();
                                    }
                                    force.insert(k);
                                }
                            }
                        }
                    }
                }
            }
        }
        if force.is_empty() {
            break;
        }
        for k in force {
            leaves.remove(&k);
            leaves.extend(k.children());
        }
    }
    // 3. Masks, waits and nearest-first requests.
    let post = PostState::current(&leaves);
    let mut sent_masks = HashMap::new();
    let mut waits = HashMap::new();
    let ops_for = |key: ChunkKey| -> Option<Arc<Vec<CsgOp>>> {
        provider.and_then(|f| {
            let v = f(key);
            if v.is_empty() {
                None
            } else {
                Some(Arc::new(v))
            }
        })
    };
    let mut to_request = Vec::new();
    for leaf in &leaves {
        let mask = post.seam_mask(config.max_level, *leaf);
        sent_masks.insert(*leaf, mask);
        waits.insert(*leaf, mask);
        to_request.push((*leaf, mask, false, ops_for(*leaf)));
    }
    to_request.sort_by(|a, b| {
        aabb_distance(anchor, a.0).total_cmp(&aabb_distance(anchor, b.0))
    });
    info!(
        "genesis: planned {} leaves in {:.1}s",
        leaves.len(),
        t0.elapsed().as_secs_f32()
    );
    GenesisPlan {
        top_cells,
        leaves,
        sent_masks,
        waits,
        to_request,
    }
}

/// One planned batch of LOD transitions, committed atomically.
struct Epoch {
    /// When planning finished — a stall timer, not a profiler.
    born: std::time::Instant,
    /// Shown leaf → the 8 children replacing it.
    /// Shown leaf → the full field-wanted descendant set replacing it in
    /// one transition. Deep: a leaf at L4 that the field wants at L0 goes
    /// straight there — the intermediate rungs are never generated (the
    /// ±1 rule constrains what is SHOWN together, and commits are atomic).
    splits: Vec<(ChunkKey, Vec<ChunkKey>)>,
    /// Hidden parent → the 8 shown leaves it replaces.
    merges: Vec<(ChunkKey, [ChunkKey; 8])>,
    /// Every mesh the commit waits for → the seam mask it must carry
    /// (empty chunks report `u32::MAX` and satisfy any expectation).
    waits: HashMap<ChunkKey, u32>,
    /// Requests not yet issued, trickled a budget per frame (safe: nothing
    /// swaps until commit, so deferral can't show stale seams).
    /// (key, mask, hold, ops) — hold marks in-place remeshes of shown
    /// chunks; ops are precomputed by the planning task so provider cost
    /// never lands on the main thread.
    to_request: RequestList,
}

/// Generation requests issued per frame while an epoch is in flight.
const EPOCH_REQUEST_BUDGET: usize = 64;

/// Requests per frame during the genesis bootstrap (nothing is shown
/// yet, so bursting is safe; the GPU batches at its own budget).
const GENESIS_REQUEST_BUDGET: usize = 256;

/// Structural changes attempted per epoch. The split cap bounds the
/// planning burst and generation load; the merge cap is much higher —
/// splits get amplified by the force-split closure, and if merges can't
/// keep pace the leaf population ratchets upward until the slabs
/// exhaust (each merge is only one generation, and they free memory).
const EPOCH_MAX_SPLITS: usize = 24;
const EPOCH_MAX_MERGES: usize = 128;

/// Leaf-count governor: above this, epochs stop proposing new splits
/// (forced splits only follow proposed ones, so none happen either) and
/// merge-only epochs drain the tree back under the cap.
const LEAF_SOFT_CAP: usize = 16_000;

/// After an aborted (stalled) epoch, plan merge-only epochs for this
/// long — the stall is almost always slab exhaustion, and merges free
/// slots while splits would consume them.
const ABORT_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(5);

/// An epoch stuck longer than this has a member that cannot generate
/// (typically slab exhaustion): abort it and coarsen instead of wedging
/// the pipeline forever.
const EPOCH_STALL_LIMIT: std::time::Duration = std::time::Duration::from_secs(20);

/// Live epoch-machine probe for remote debugging (voxel/status).
#[derive(Resource, Default, Clone)]
pub struct StreamProbe {
    /// Genesis committed: the world exists and planning caches are warm
    /// (vegetation streamers wait for this before their first build).
    pub world_ready: bool,
    /// Planning chunks generated by a READ instead of by the ensure-load
    /// pass — should stay at 0 once genesis has committed.
    pub read_generated: usize,
    /// Rolling 2-second frame telemetry (hosts log or display it).
    pub fps: f32,
    pub worst_frame_ms: f32,
    pub slab_free: [u32; 4],
    pub leaves: usize,
    pub planning: bool,
    pub replan_needed: bool,
    pub epoch_waits: usize,
    pub epoch_to_request: usize,
    pub epoch_age_s: f32,
}

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
    /// In-flight async planning task (pure function of a tree snapshot;
    /// runs on the compute pool so the main thread never blocks on it).
    planning: Option<bevy::tasks::Task<Option<Epoch>>>,
    /// Cold-start bootstrap: the converged configuration generating
    /// hidden, revealed atomically when complete.
    genesis_planning: Option<bevy::tasks::Task<GenesisPlan>>,
    genesis: Option<GenesisPlan>,
    /// Something changed since the last plan (commit, anchor move, ring
    /// churn) — gates snapshotting so idle frames don't clone the tree.
    replan_needed: bool,
    /// Merge-only planning until this instant (set on epoch abort).
    split_cooldown_until: Option<std::time::Instant>,
    /// The in-flight plan was spawned with splits suppressed (cooldown).
    /// An empty result then means "try again later", not "converged" —
    /// without this the machine idles at a coarse tree until the camera
    /// happens to move 48 m (observed live after a teleport abort).
    plan_split_capped: bool,
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
        splits: &[(ChunkKey, Vec<ChunkKey>)],
        merges: &[ChunkKey],
    ) -> PostState<'a> {
        let mut added = HashSet::new();
        let mut removed = HashSet::new();
        for (p, descendants) in splits {
            removed.insert(*p);
            added.extend(descendants.iter().copied());
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
            .init_resource::<StreamProbe>()
            .add_systems(Update, (lod_tick, log_fps));
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
/// Pure over the snapshot — safe to run off-thread.
#[cfg(test)]
fn plan_epoch(tree: &LodTree, config: &LodConfig, anchor: DVec3) -> Option<Epoch> {
    plan_epoch_snapshot(
        &tree.leaves,
        &tree.sent_masks,
        config,
        anchor,
        None,
        EPOCH_MAX_SPLITS,
    )
}

#[allow(clippy::too_many_arguments)]
fn plan_epoch_snapshot(
    leaves: &HashSet<ChunkKey>,
    sent_masks: &HashMap<ChunkKey, u32>,
    config: &LodConfig,
    anchor: DVec3,
    provider: Option<&(dyn Fn(ChunkKey) -> Vec<CsgOp> + Send + Sync)>,
    split_cap: usize,
) -> Option<Epoch> {
    // Deep splits: each wanted leaf is replaced by its full field-wanted
    // descendant set in one transition — flying into a region goes
    // straight to the target LOD; the intermediate rungs are never
    // generated. Nearest-first, budgeted by NEW CHUNK COUNT (a deep split
    // near the camera can be large): the chunk the player stands in
    // always converges first.
    fn descend(config: &LodConfig, anchor: DVec3, k: ChunkKey, out: &mut Vec<ChunkKey>) {
        if split_wanted(config, anchor, k) {
            for c in k.children() {
                descend(config, anchor, c, out);
            }
        } else {
            out.push(k);
        }
    }
    let mut wanted: Vec<ChunkKey> = leaves
        .iter()
        .filter(|l| split_wanted(config, anchor, **l))
        .copied()
        .collect();
    wanted.sort_by(|a, b| aabb_distance(anchor, *a).total_cmp(&aabb_distance(anchor, *b)));
    // Governor: a bloated tree (fast flight ratchets leaves upward)
    // stops splitting; merges drain the population and free slab memory.
    if leaves.len() > LEAF_SOFT_CAP {
        wanted.clear();
    }
    let chunk_budget = split_cap * 8;
    let mut splits: Vec<(ChunkKey, Vec<ChunkKey>)> = Vec::new();
    let mut budget_used = 0usize;
    for p in wanted {
        if budget_used >= chunk_budget {
            break;
        }
        let mut descendants = Vec::new();
        descend(config, anchor, p, &mut descendants);
        budget_used += descendants.len();
        splits.push((p, descendants));
    }

    let mut sibling_count: HashMap<ChunkKey, u8> = HashMap::new();
    for leaf in leaves {
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
        let post = PostState::plan(leaves, &splits, &merges);
        let mut forced: Vec<ChunkKey> = Vec::new();
        for (_, descendants) in &splits {
            for d in descendants {
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let n = ChunkKey::new(d.level, d.pos + IVec3::new(dx, dy, dz));
                            let Some(l) = post.covering_level(config.max_level, n) else {
                                continue;
                            };
                            if l <= d.level + 1 {
                                continue;
                            }
                            let mut k = n;
                            while k.level < l {
                                k = k.parent();
                            }
                            if !forced.contains(&k) && !splits.iter().any(|(p, _)| *p == k) {
                                forced.push(k);
                            }
                        }
                    }
                }
            }
        }
        if forced.is_empty() {
            break;
        }
        // A forced split overrides any merge that involves the same
        // region (as the merge parent or one of its children). Forced
        // splits are single-level (the field did not ask for them).
        merges.retain(|m| !forced.iter().any(|f| *m == *f || *m == f.parent()));
        for k in forced {
            splits.push((k, k.children().to_vec()));
        }
    }

    // ±1 veto for merges: coarsening waits until the neighborhood can
    // absorb it (this is what keeps forced splits from flapping back).
    loop {
        let n0 = merges.len();
        let post = PostState::plan(leaves, &splits, &merges);
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

    let post = PostState::plan(leaves, &splits, &merges);
    let ops_for = |key: ChunkKey| -> Option<Arc<Vec<CsgOp>>> {
        provider.and_then(|f| {
            let v = f(key);
            if v.is_empty() {
                None
            } else {
                Some(Arc::new(v))
            }
        })
    };
    let mut epoch = Epoch {
        born: std::time::Instant::now(),
        splits: Vec::new(),
        merges: Vec::new(),
        waits: HashMap::new(),
        to_request: Vec::new(),
    };
    for (p, descendants) in splits {
        for c in &descendants {
            let m = post.seam_mask(config.max_level, *c);
            epoch.waits.insert(*c, m);
            epoch.to_request.push((*c, m, false, None));
        }
        epoch.splits.push((p, descendants));
    }
    for p in merges {
        let m = post.seam_mask(config.max_level, p);
        epoch.waits.insert(p, m);
        epoch.to_request.push((p, m, false, None));
        epoch.merges.push((p, p.children()));
    }
    // Seam remeshes: every kept leaf whose mask changes under the
    // post-epoch configuration regenerates (held) and commits with it. A
    // mask can only change for leaves touching a region whose level
    // changes, so enumerate those from the changed regions (26 neighbor
    // keys each, descending only into touching subtrees) — an all-leaves
    // scan is an O(leaves × changes) planning burst.
    let mut candidates: HashSet<ChunkKey> = HashSet::new();
    let changed_regions: Vec<ChunkKey> = epoch
        .splits
        .iter()
        .map(|(p, _)| *p)
        .chain(epoch.merges.iter().map(|(p, _)| *p))
        .collect();
    for r in &changed_regions {
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
        if !leaves.contains(&leaf) || post.removed.contains(&leaf) {
            continue;
        }
        let m = post.seam_mask(config.max_level, leaf);
        if sent_masks.get(&leaf) != Some(&m) {
            epoch.waits.insert(leaf, m);
            epoch.to_request.push((leaf, m, true, None));
        }
    }
    // Fill each chunk's ops from resident planning, off the main thread.
    for (key, _, _, ops) in epoch.to_request.iter_mut() {
        *ops = ops_for(*key);
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
    world: Res<crate::planning::WorldQuery>,
    mut rebuild: ResMut<StreamingRebuild>,
    ready_rx: Res<ChunkReadyChannel>,
    mut field: ResMut<voxel_render::FieldParams>,
    stats: Res<SharedRenderStats>,
    mut probe: ResMut<StreamProbe>,
    sources: crate::StreamSourceQuery,
) {
    let Ok(source) = sources.single() else {
        return; // no streaming source tagged yet
    };
    let camera = source.translation();
    let tick_start = std::time::Instant::now();
    let camera = camera.as_dvec3();
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
        tree.replan_needed = true;
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
        // Mid-genesis rebuild: genesis chunks were requested but are in
        // neither leaves nor an epoch — forgetting them leaks hidden
        // render chunks (and their slab allocs) forever.
        if let Some(plan) = &tree.genesis {
            requested.extend(plan.waits.keys().copied());
        }
        for key in requested {
            queue.push(ChunkCommand::Free(key));
        }
        *tree = LodTree::default();
        tree.replan_needed = true;
    }

    // 1. Absorb readiness notifications.
    for (key, mask) in ready_rx.rx.try_iter() {
        tree.ready.insert(key, mask);
    }

    // 1b. Cold start: an empty tree bootstraps through genesis — the
    //    converged configuration generates hidden and reveals in one
    //    atomic commit. No intermediate LODs are generated at all, and
    //    the screen goes "loading -> world" instead of morphing through
    //    refinement rungs.
    if tree.leaves.is_empty() || tree.genesis.is_some() || tree.genesis_planning.is_some() {
        if let Some(task) = &mut tree.genesis_planning {
            if let Some(plan) = bevy::tasks::block_on(bevy::tasks::futures_lite::future::poll_once(task))
            {
                tree.genesis_planning = None;
                tree.genesis = Some(plan);
            }
        } else if let Some(mut plan) = tree.genesis.take() {
            let n = plan.to_request.len().min(GENESIS_REQUEST_BUDGET);
            for (key, mask, _, ops) in plan.to_request.drain(..n) {
                // Every genesis chunk's mask comes from the one final
                // configuration, so any subset shown together is
                // seam-consistent: stream each in the moment it is
                // drawable. Requests are nearest-first, so the chunk the
                // player stands in appears first and the world grows
                // outward.
                queue.push(ChunkCommand::Request {
                    key,
                    show_on_ready: true,
                    hold: false,
                    ops,
                    face_mask: mask,
                });
            }
            let done = plan.to_request.is_empty()
                && plan
                    .waits
                    .iter()
                    .all(|(k, m)| matches!(tree.ready.get(k), Some(&r) if r == *m || r == u32::MAX));
            if done {
                tree.top_cells = plan.top_cells;
                tree.leaves = plan.leaves;
                tree.sent_masks = plan.sent_masks;
                tree.replan_needed = true;
                info!(
                    "genesis: world revealed ({} chunks, {} planning chunks cached, {} read-generated)",
                    tree.leaves.len(),
                    world.stats().resident_chunks,
                    world.stats().reads_missed,
                );
            } else {
                tree.genesis = Some(plan);
            }
        } else {
            let config = config.clone();
            let provider = ops_provider.0.clone();
            tree.genesis_planning = Some(bevy::tasks::AsyncComputeTaskPool::get().spawn(
                async move {
                    plan_genesis(&config, anchor, provider.as_deref())
                },
            ));
        }
        return;
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
                    tree.replan_needed = true;
                }
            }
        }
    }
    // Evictions wait for the epoch AND any in-flight planning: freeing a
    // member (or a plan-snapshot leaf) would wedge the commit. The stale
    // ring is far behind the camera anyway.
    if tree.epoch.is_none() && tree.planning.is_none() {
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

    // 3. Plan the next epoch off-thread when none is in flight. Planning
    //    is pure over a snapshot; the shown tree can only change under it
    //    additively (ring arrivals, which don't affect masks of existing
    //    leaves) — commits and evictions are gated while it runs.
    if tree.epoch.is_none() {
        if let Some(task) = &mut tree.planning {
            if let Some(result) =
                bevy::tasks::block_on(bevy::tasks::futures_lite::future::poll_once(task))
            {
                tree.planning = None;
                tree.epoch = result;
                if tree.epoch.is_none() && tree.plan_split_capped {
                    tree.replan_needed = true;
                }
            }
        } else if tree.replan_needed {
            tree.replan_needed = false;
            let leaves = tree.leaves.clone();
            let sent_masks = tree.sent_masks.clone();
            let config = config.clone();
            let provider = ops_provider.0.clone();
            let split_cap = if tree
                .split_cooldown_until
                .is_none_or(|until| std::time::Instant::now() >= until)
            {
                EPOCH_MAX_SPLITS
            } else {
                0
            };
            tree.plan_split_capped = split_cap == 0;
            tree.planning = Some(bevy::tasks::AsyncComputeTaskPool::get().spawn(async move {
                plan_epoch_snapshot(
                    &leaves,
                    &sent_masks,
                    &config,
                    anchor,
                    provider.as_deref(),
                    split_cap,
                )
            }));
        }
    }

    // 4. Advance the in-flight epoch: trickle its generation requests,
    //    then commit atomically once every member is drawable.
    if let Some(mut epoch) = tree.epoch.take() {
        let n = epoch.to_request.len().min(EPOCH_REQUEST_BUDGET);
        for (key, mask, hold, ops) in epoch.to_request.drain(..n) {
            // A wait may only be satisfied by a report that ARRIVES after
            // this request: a stale entry (held mesh cancelled by an
            // earlier abort, or an old empty classification) would let
            // the epoch commit against a mesh that no longer exists —
            // permanent crack plus an orphaned held alloc.
            tree.ready.remove(&key);
            tree.sent_masks.insert(key, mask);
            queue.push(ChunkCommand::Request {
                key,
                show_on_ready: false,
                hold,
                ops,
                face_mask: mask,
            });
        }
        let done = epoch.to_request.is_empty()
            && epoch
                .waits
                .iter()
                .all(|(k, m)| matches!(tree.ready.get(k), Some(&r) if r == *m || r == u32::MAX));
        if done {
            commit_epoch(tree, &queue, epoch);
            // Refinement cascades: the new configuration may want more.
            tree.replan_needed = true;
        } else if epoch.born.elapsed() > EPOCH_STALL_LIMIT {
            // A member cannot generate — almost always slab exhaustion.
            // Wedging forever would also block the merges that free slabs:
            // abort, then coarsen for a cooldown before splitting again.
            warn!(
                "epoch stalled {}s with {} waits — aborting; merge-only for {}s",
                EPOCH_STALL_LIMIT.as_secs(),
                epoch.waits.len(),
                ABORT_COOLDOWN.as_secs(),
            );
            for key in epoch.waits.keys() {
                if tree.leaves.contains(key) {
                    // In-place remesh: drop the held result (if any) and
                    // forget the sent mask so a later epoch re-requests it.
                    // The ready entry goes too — the held mesh it reported
                    // is being cancelled.
                    queue.push(ChunkCommand::CancelHold(*key));
                    tree.ready.remove(key);
                    tree.sent_masks.remove(key);
                } else {
                    // Hidden replacement chunk: free it outright.
                    queue.push(ChunkCommand::Free(*key));
                    tree.ready.remove(key);
                    tree.sent_masks.remove(key);
                }
            }
            tree.split_cooldown_until = Some(std::time::Instant::now() + ABORT_COOLDOWN);
            tree.replan_needed = true;
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
        if FRAME
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .is_multiple_of(30)
        {
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

    probe.read_generated = world.stats().reads_missed;
    probe.world_ready =
        tree.genesis.is_none() && tree.genesis_planning.is_none() && !tree.leaves.is_empty();
    probe.leaves = tree.leaves.len();
    probe.planning = tree.planning.is_some();
    probe.replan_needed = tree.replan_needed;
    probe.epoch_waits = tree.epoch.as_ref().map_or(0, |e| e.waits.len());
    probe.epoch_to_request = tree.epoch.as_ref().map_or(0, |e| e.to_request.len());
    probe.epoch_age_s = tree
        .epoch
        .as_ref()
        .map_or(0.0, |e| e.born.elapsed().as_secs_f32());

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


/// Frame telemetry for hosts and tools: the engine measures, the host
/// decides whether to log or draw it.
fn log_fps(
    stats: Res<SharedRenderStats>,
    tree: Res<LodTree>,
    time: Res<Time>,
    mut probe: ResMut<StreamProbe>,
    mut window: Local<(f32, u32, f32)>,
) {
    let dt = time.delta_secs();
    window.0 += dt;
    window.1 += 1;
    window.2 = window.2.max(dt);
    if window.0 >= 2.0 {
        probe.fps = window.1 as f32 / window.0;
        probe.worst_frame_ms = window.2 * 1000.0;
        *window = (0.0, 0, 0.0);
    }
    probe.leaves = tree.leaves.len();
    if let Ok(s) = stats.0.lock() {
        probe.slab_free = s.slab_occupancy.map(|(free, _)| free);
    }
}

#[cfg(test)]
mod epoch_invariants {
    /// Face directions in mask order (+x, -x, +y, -y, +z, -z).
    const FACE_DIRS: [IVec3; 6] = [
        IVec3::new(1, 0, 0),
        IVec3::new(-1, 0, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 0, 1),
        IVec3::new(0, 0, -1),
    ];

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
            for (key, mask, _, _) in &epoch.to_request {
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
