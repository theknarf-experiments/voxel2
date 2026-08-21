//! The LOD field, as pure functions of a chunk and the camera anchor.
//!
//! Which chunks exist, which level each is at, and which of its neighbors
//! are coarser are all decided one chunk at a time from
//! `(config, anchor, key)` — nothing here consults a tree, a plan or a
//! shown configuration. That is what lets [`crate::lod_layers`] hand the
//! whole question of residency to the dependency graph: a top dependency
//! per level, filtered by [`resident_clamped`].
//!
//! It replaced an epoch machine — a planner that proposed batches of
//! splits and merges, generated them hidden, and committed each batch in
//! one frame once every member was drawable. That machine was how
//! seam-consistency was kept before the field had a closed form; its
//! fixpoint now lives in this file's tests, as the specification these
//! functions are pinned to.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use bevy::math::DVec3;
use bevy::prelude::*;
use voxel_core::ChunkKey;
use voxel_render::SharedRenderStats;

use crate::chunkgen::ChunkGenPlugin;

/// Does the field want this chunk refined? The one rule everything else
/// here is built from: a chunk splits inside `split_k` of its own edge —
/// scaled by `2^levels` where a detail volume overlaps it, so a feature
/// thinner than a distant voxel is sampled finely enough to survive.
///
/// The scale keeps the property the descent rests on: a volume touching a
/// chunk touches its parent, so the parent's bias is at least the child's,
/// and a split wanted at the child is always wanted at the parent too.
fn split_wanted(config: &LodConfig, anchor: DVec3, key: ChunkKey) -> bool {
    key.level > 0
        && aabb_distance(anchor, key)
            < config.split_k * (1u64 << detail_bias(config, key).min(32)) as f64 * key.edge_m()
}

/// A region the field refines beyond what distance alone asks for: chunks
/// overlapping the box act up to `levels` octree levels closer than they
/// are. The bias fades twice over, and both fades are load-bearing.
///
/// ACROSS SPACE it fades one level per chunk edge of distance from the
/// box. A cliff-edge bias puts field leaves `levels` apart across one
/// face, and closing that gap needs forced splits that cascade through
/// other forced splits — a fixpoint the closed-form clamp cannot see
/// (measured: it strands 2-level jumps exactly one ring out). Faded, the
/// field never jumps more than one level across a face, which is the same
/// contract the plain distance field gives the clamp.
///
/// ACROSS SCALE it is full only for chunks no larger than the volume and
/// loses one level per level above. One less per level is the most a
/// parent may lag its child (thresholds double per level, so the descent
/// stays monotone), and it is also what makes the volume LOCAL: rings are
/// a chunk's own edges, so a flat bias let a 50 m volume re-level terrain
/// `levels` coarse edges — tens of kilometers — away (measured: +1725
/// resident chunks, most of them a vista the author never pointed at).
/// Capped, the total influence is bounded near `2^(levels-1)` times the
/// volume's own extent, and so is the range: past the distance where the
/// plain leaf outgrows `levels` above the volume's scale, the volume is
/// inert. Refining a landmark for a farther eye is a bigger `levels`, at
/// a cost admission control counts like everything else.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetailVolume {
    pub min: DVec3,
    pub max: DVec3,
    /// Extra refinement levels inside the volume, at the volume's scale.
    pub levels: u8,
}

/// The bias `volume` can give a chunk of `level` before the spatial fade:
/// full at the level whose edge first covers the volume and below, one
/// less per level above, zero once the chunk outscales it by `levels`.
fn scale_cap(volume: &DetailVolume, level: u8) -> u8 {
    let extent = (volume.max - volume.min).max_element();
    let mut fit = 0u8;
    while ChunkKey::new(fit, IVec3::ZERO).edge_m() < extent && fit < 32 {
        fit += 1;
    }
    volume.levels.saturating_sub(level.saturating_sub(fit))
}

/// How far, in Chebyshev meters, a volume's refinement can put a leaf of
/// `level` from the volume's box — zero when it cannot put one there at
/// all, which is what lets a dependency skip the level entirely. A leaf
/// of level L is extra only if its PARENT carried bias, so it sits within
/// the parent's fade reach — its scale-capped bias in parent edges — plus
/// the parent's own edge and a ring of clamp grading. The volume-anchored
/// dependency boxes are sized from this, and
/// `a_volume_refines_at_distance_and_only_nearby` pins it — a leaf outside
/// the reach is a chunk residency would silently clip.
pub fn detail_reach_m(volume: &DetailVolume, level: u8) -> f64 {
    let parent = scale_cap(volume, level + 1);
    if parent == 0 {
        return 0.0;
    }
    (2.0 * f64::from(parent) + 4.0) * ChunkKey::new(level, IVec3::ZERO).edge_m()
}

/// The largest faded bias of any volume near this chunk's box; zero almost
/// everywhere, so the common case is one empty-slice check.
///
/// Distance is Chebyshev — the clamp's neighborhoods are 26-connected, so
/// a touching pair of chunks is within one edge on EVERY axis, which is
/// what bounds the bias step across any touching pair to one level.
///
/// Monotone up the tree, which the descent relies on: a parent's box is
/// closer to the volume than its child's and its edge is twice as long, so
/// its ring gap is no larger and its bias no smaller.
fn detail_bias(config: &LodConfig, key: ChunkKey) -> u8 {
    if config.detail.is_empty() {
        return 0;
    }
    let min = key.min_corner_m();
    let max = min + DVec3::splat(key.edge_m());
    let edge = key.edge_m();
    config
        .detail
        .iter()
        .map(|v| {
            let cap = scale_cap(v, key.level);
            if cap == 0 {
                return 0;
            }
            let gap = (v.min - max).max(min - v.max).max(DVec3::ZERO);
            let rings = (gap.max_element() / edge).ceil() as u64;
            u64::from(cap).saturating_sub(rings) as u8
        })
        .max()
        .unwrap_or(0)
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
    /// Split when camera distance < split_k × edge. The one constant the
    /// field is made of.
    pub split_k: f64,
    /// Historically: merge when camera distance > merge_k × parent edge.
    /// Nothing reads it any more — a chunk merges where its parent stops
    /// splitting, and the hysteresis is the sticky anchor, not a second
    /// constant. Kept because it is still level data, and because the
    /// residency measurement uses it to reproduce the sizing it rejected.
    /// Ensure the levels INSIDE the ops horizon finest-first.
    ///
    /// Only these levels wait on `chunk_covered`, and coverage is regional
    /// — a level's box shrinks with its edge, so the finest is covered
    /// soonest and the coarsest last. Whichever gated level a pass reaches
    /// first is the one it blocks on, so leading with the finest blocks
    /// for the shortest time and the coarser levels then find their own
    /// coverage already arrived.
    ///
    /// NOT a free win, which is why it is level data and not a rule:
    /// leading with thousands of small chunks also puts the coarse levels'
    /// GPU work behind them. Which effect dominates is a property of the
    /// level, and the two shipped extremes disagree by 0.3 s in OPPOSITE
    /// directions (6 runs each): megastructure 1.47 coarse / 1.18 fine,
    /// planet 1.52 coarse / 1.66 fine.
    ///
    /// Default false — plain coarsest-first, which is also what the levels
    /// OUTSIDE the horizon always use, since those need no planning at all
    /// and are what the pipeline should chew on while the planners run.
    pub gated_finest_first: bool,
    pub merge_k: f64,
    /// Regions refined beyond what distance alone asks for. Level data,
    /// shared rather than copied because the config is cloned per world
    /// and per admission-control probe.
    pub detail: Arc<Vec<DetailVolume>>,
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
            gated_finest_first: false,
            detail: Arc::new(Vec::new()),
        }
    }
}

/// Set to request a full streaming rebuild (e.g. after a hot-reloaded
/// level changes generation parameters): the LOD graph is dropped, which
/// destroys every chunk it holds, and rebuilt against the new world.
#[derive(Resource, Default)]
pub struct StreamingRebuild(pub bool);

/// Streaming telemetry for the HUD and remote (voxel/status).
#[derive(Resource, Default, Clone)]
pub struct StreamProbe {
    /// Planning chunks generated by a READ instead of by the ensure-load
    /// pass. Anything but 0 means a consumer's working set is not covered.
    pub reads_missed: usize,
    /// Rolling 2-second frame telemetry (hosts log or display it).
    pub fps: f32,
    pub worst_frame_ms: f32,
    /// Mesh pages free, and the peak used over the session. Peaks, not
    /// the present: what a world costs depends on where the camera is.
    /// Chunks the interval bound proved empty, and chunks that reached
    /// the generator anyway (with how many of those carried planning ops,
    /// which skips the bound entirely). Cumulative.
    pub pruned: usize,
    pub unpruned: usize,
    pub unpruned_with_ops: usize,
    pub slab_free_pages: u32,
    pub slab_peak_pages: u32,
    /// Chunks the slab can hold, measured from what they have cost.
    pub slab_capacity_chunks: usize,
    /// Resident voxel chunks. Residency is exactly the shown set, so this
    /// is also what is drawn.
    pub resident: usize,
    /// A generation pass is running.
    pub generating: bool,
    /// Chunks that never became drawable inside their timeout — slab
    /// exhaustion, and a hole for as long as it lasts.
    pub stalled: usize,
    /// The world matches where the camera is: nothing generating, nothing
    /// left in the pipeline.
    pub settled: bool,
    /// How long the CURRENT unsettled stretch has lasted, in seconds; 0
    /// when settled.
    pub settling_s: f32,
    /// How long the last one took.
    ///
    /// Every anchor move starts a stretch and the world catching up ends
    /// it, so this is the answer to "how long after I move does the world
    /// agree with me" — the number that says whether generation is keeping
    /// up, which a frame rate does not.
    pub last_settle_s: f32,
    /// The worst stretch since start.
    pub worst_settle_s: f32,
}

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

/// Streaming: the chunk generation service, the LOD layers that drive it,
/// and the telemetry both feed.
pub struct VoxelStreamingPlugin;

impl Plugin for VoxelStreamingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((ChunkGenPlugin, crate::lod_layers::LodLayersPlugin))
            .init_resource::<LodConfig>()
            .init_resource::<StreamingRebuild>()
            .init_resource::<StreamProbe>()
            .add_systems(Update, log_fps);
    }
}

/// Frame telemetry for hosts and tools: the engine measures, the host
/// decides whether to log or draw it.
fn log_fps(
    stats: Res<SharedRenderStats>,
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
    if let Ok(s) = stats.0.lock() {
        probe.slab_free_pages = s.slab_total_pages.saturating_sub(s.slab_used_pages);
        probe.slab_peak_pages = s.slab_peak_pages;
        probe.slab_capacity_chunks = s.slab_capacity_chunks;
    }
}

fn aabb_distance(camera: DVec3, key: ChunkKey) -> f64 {
    let min = key.min_corner_m();
    let max = min + DVec3::splat(key.edge_m());
    camera.distance(camera.clamp(min, max))
}

/// Does the CLAMPED field split this chunk? A pure function of (chunk,
/// anchor), and the closed form of what the epoch machine reaches by
/// iterating a fixpoint over a shown set.
///
/// The field's own rule allows two shown leaves to touch across a corner
/// with two levels between them, which a seam cannot bridge; the machine
/// splits the coarser side until nothing does. Written as a fixpoint that
/// is a property of a whole configuration — which is exactly why it cannot
/// be a residency rule, since residency has to be decidable one chunk at a
/// time. Written this way it is decidable: descend the touching part of
/// each neighbor's subtree and ask how fine the field wants it. Only
/// touching subtrees are visited, so the walk is a handful of nodes per
/// level, not a subtree.
fn split_clamped(config: &LodConfig, anchor: DVec3, key: ChunkKey) -> bool {
    if split_wanted(config, anchor, key) {
        return true;
    }
    if key.level < 2 {
        return false; // nothing can be two levels finer than level 1
    }
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let n = ChunkKey::new(key.level, key.pos + IVec3::new(dx, dy, dz));
                if field_wants_touching_finer(config, anchor, n, key, key.level - 1) {
                    return true;
                }
            }
        }
    }
    false
}

/// Does the field want a leaf strictly finer than `min_level` inside
/// `region`, touching `target`? Twin of `PostState::has_touching_finer_than`
/// against the field rather than against a shown set.
fn field_wants_touching_finer(
    config: &LodConfig,
    anchor: DVec3,
    region: ChunkKey,
    target: ChunkKey,
    min_level: u8,
) -> bool {
    if !boxes_touch(region, target) {
        return false;
    }
    if !split_wanted(config, anchor, region) {
        return region.level < min_level;
    }
    if region.level == 0 {
        return false;
    }
    region
        .children()
        .iter()
        .any(|c| field_wants_touching_finer(config, anchor, *c, target, min_level))
}

/// Is this chunk inside the streamed volume — the ring of top-level cells
/// the world consists of? Outside it there is nothing at all, which is a
/// different thing from "coarser".
fn in_top_ring(config: &LodConfig, anchor: DVec3, key: ChunkKey) -> bool {
    let mut k = key;
    while k.level < config.max_level {
        k = k.parent();
    }
    let edge = k.edge_m();
    let cx = (anchor.x / edge).floor() as i32;
    let cz = (anchor.z / edge).floor() as i32;
    (k.pos.x - cx).abs() <= config.top_radius
        && (k.pos.z - cz).abs() <= config.top_radius
        && k.pos.y >= config.top_y.0
        && k.pos.y <= config.top_y.1
}

/// Seam mask of a chunk under the clamped field: one bit per direction of
/// the 26-neighborhood, in scan order, set where the neighbor is COARSER.
/// Twin of `PostState::seam_mask` and of `snap_to_parity` in the mesh
/// shader — but a pure function of (chunk, anchor) rather than of a shown
/// configuration, which is what lets a chunk be built without consulting
/// a tree.
///
/// A neighbor is coarser exactly when the chunk covering its region is: if
/// the neighbor's PARENT is not split, the leaf there is at least one
/// level up.
pub fn seam_mask_at(config: &LodConfig, anchor: DVec3, key: ChunkKey) -> u32 {
    if key.level >= config.max_level {
        return 0; // nothing is coarser than the top
    }
    // The 26 neighbors have at most 8 distinct parents — three consecutive
    // coordinates halve to at most two — and `split_clamped` is a subtree
    // descent, so asking it once per neighbor asks the same question 3x
    // over.
    let mut seen: [(IVec3, bool); 8] = [(IVec3::ZERO, false); 8];
    let mut count = 0usize;
    let mut mask = 0u32;
    let mut idx = 0;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let n = ChunkKey::new(key.level, key.pos + IVec3::new(dx, dy, dz));
                // Nothing streams outside the top ring, so there is no
                // neighbor there to snap to — the world simply ends, and
                // a snap bit would pull vertices toward a mesh that does
                // not exist.
                if in_top_ring(config, anchor, n) {
                    let parent = n.parent();
                    let split = match seen[..count].iter().find(|(p, _)| *p == parent.pos) {
                        Some((_, split)) => *split,
                        None => {
                            let split = split_clamped(config, anchor, parent);
                            seen[count] = (parent.pos, split);
                            count += 1;
                            split
                        }
                    };
                    if !split {
                        mask |= 1 << idx;
                    }
                }
                idx += 1;
            }
        }
    }
    mask
}

/// Where a LOD level is resident, as a predicate on one chunk: this level
/// could be the finest covering it — its own split radius does not swallow
/// it, its parent's does, and it is inside the streamed ring.
///
/// A top dependency is a box, and the plan sized these levels as a box
/// with a hole. Measured against the configuration that ships, that box
/// holds 2.35x the chunks the field draws (2.14x on the megastructure),
/// because a box is not an annulus — its diagonal reaches 1.7x further
/// than its face. On the megastructure, where 78% of chunks carry
/// geometry, 2.14x is ~5.1k meshed against 3,656 slots: the same wall
/// nested balls hit. This holds 1.00x. Residency is a predicate, not a
/// shape.
pub fn resident_clamped(config: &LodConfig, anchor: DVec3, key: ChunkKey) -> bool {
    in_top_ring(config, anchor, key)
        && !split_clamped(config, anchor, key)
        && (key.level >= config.max_level || split_clamped(config, anchor, key.parent()))
}

/// How many chunks this configuration keeps resident, counted exactly by
/// descending the field.
///
/// Residency is a pure function of (config, anchor), so this is not an
/// estimate — it is the same set the LOD graph will ask for. That is what
/// lets the slab be checked against demand BEFORE a world is loaded,
/// rather than discovering the shortfall as chunks pile up with nowhere
/// to go.
///
/// The count barely moves with the anchor (the field is radial and the
/// top ring recentres), so one sample is representative.
pub fn resident_count(config: &LodConfig, anchor: DVec3) -> usize {
    count_leaves(config, anchor, &|_| true)
}

/// How many resident chunks could actually hold a mesh.
///
/// Residency is not demand: on the shipped planet most resident chunks
/// are sky or deep rock and are classified empty without ever taking a
/// slab slot. [`can_hold_surface`] is the same conservative test the LOD
/// graph filters with, so this is an upper bound on the slots a world
/// will ask for — which is exactly what admission control needs.
///
/// Conservative in the safe direction: it can only over-count, never
/// under.
pub fn meshable_count(
    config: &LodConfig,
    generator: &voxel_worldgen::Generator,
    anchor: DVec3,
) -> usize {
    count_leaves(config, anchor, &|key| can_hold_surface(generator, key))
}

/// The resident leaf set, counted by descending the field, keeping the
/// leaves `keep` accepts.
///
/// One walk rather than two: residency and meshability differ only in
/// what they do at a leaf, and both claim to be the set the LOD graph
/// will ask for. Two copies of the descent are two chances for one of
/// them to stop being that.
///
/// PARALLEL over the top ring, because this is admission control and it
/// runs before a world can stream: on the shipped planet it was 736 ms of
/// a 1.57 s load, all of it one thread evaluating an interval bound per
/// candidate chunk. The cells of the ring are independent — `keep` is a
/// pure function of a key — so they split with no coordination beyond the
/// counter.
fn count_leaves(
    config: &LodConfig,
    anchor: DVec3,
    keep: &(dyn Fn(ChunkKey) -> bool + Sync),
) -> usize {
    fn descend(
        config: &LodConfig,
        anchor: DVec3,
        key: ChunkKey,
        keep: &(dyn Fn(ChunkKey) -> bool + Sync),
        out: &mut usize,
    ) {
        if key.level > 0 && split_clamped(config, anchor, key) {
            for child in key.children() {
                descend(config, anchor, child, keep, out);
            }
        } else if keep(key) {
            *out += 1;
        }
    }
    let top_edge = ChunkKey::new(config.max_level, IVec3::ZERO).edge_m();
    let cx = (anchor.x / top_edge).floor() as i32;
    let cz = (anchor.z / top_edge).floor() as i32;
    let mut cells = Vec::new();
    for dz in -config.top_radius..=config.top_radius {
        for dx in -config.top_radius..=config.top_radius {
            for y in config.top_y.0..=config.top_y.1 {
                cells.push(IVec3::new(cx + dx, y, cz + dz));
            }
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let workers = threads.min(cells.len()).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let total = std::sync::atomic::AtomicUsize::new(0);
    let cells = &cells;
    let run = || {
        let mut count = 0usize;
        while let Some(cell) = cells.get(next.fetch_add(1, Ordering::Relaxed)) {
            descend(
                config,
                anchor,
                ChunkKey::new(config.max_level, *cell),
                keep,
                &mut count,
            );
        }
        total.fetch_add(count, Ordering::Relaxed);
    };
    std::thread::scope(|scope| {
        for _ in 1..workers {
            scope.spawn(run);
        }
        run();
    });
    total.load(Ordering::Relaxed)
}

/// Cap a world's detail until the meshes it will ask for fit in the slab
/// slots left over from the worlds already loaded.
///
/// This is the mechanism `FAR_MAX_LEVEL` was a hand-tuned instance of: a
/// level seen only through a portal does not need its finest LODs, and
/// which levels it can afford is a function of what else is loaded, not a
/// constant somebody picked.
///
/// Returns the config to use and its demand. Capping the coarsest level
/// shrinks the streamed ring, so the count falls monotonically and the
/// loop terminates; a world that cannot fit even at `min_level` is
/// admitted anyway, at `min_level`, because refusing to load it is worse
/// than a few deferred chunks now that deferral is safe.
pub fn fit_to_budget(
    config: &LodConfig,
    generator: &voxel_worldgen::Generator,
    anchor: DVec3,
    available: usize,
    min_level: u8,
    // What `config` already measured at, if the caller has it. Admission
    // control computes exactly this to divide the slab and then asked for
    // it again here — a second full walk of the LOD tree, which on the
    // planet is most of a second.
    known: Option<usize>,
) -> (LodConfig, usize) {
    let mut fitted = config.clone();
    let mut demand = known.unwrap_or_else(|| meshable_count(&fitted, generator, anchor));
    loop {
        if demand <= available || fitted.max_level <= min_level {
            return (fitted, demand);
        }
        fitted.max_level -= 1;
        demand = meshable_count(&fitted, generator, anchor);
    }
}

/// Could this chunk contain a surface at all?
///
/// Evaluating the generator on the chunk's box instead of at points
/// answers that for the cost of a dozen interval operations, where
/// finding out by generating it costs a 38³ density pass and a GPU round
/// trip. On the shipped planet 11,177 of 13,083 resident chunks exist
/// only to be classified empty — sky and deep rock, either side of a
/// surface that is nowhere near them.
///
/// Answers for the GENERATOR only. Planning carves into the world after
/// it, so a caller has to account for that too: either by knowing the
/// chunk has no ops, or by only asking past the ops horizon.
///
/// The box is padded by the density apron, because samples reach outside
/// the chunk and a surface just beyond it still puts geometry inside.
pub fn can_hold_surface(generator: &voxel_worldgen::Generator, key: ChunkKey) -> bool {
    let apron = 4.0 * key.voxel_size_m() as f32;
    let min = key.min_corner_m().as_vec3() - Vec3::splat(apron);
    let max = min + Vec3::splat(key.edge_m() as f32 + 2.0 * apron);
    uniform_sign(
        generator,
        min,
        max,
        key.voxel_size_m() as f32,
        *PRUNE_SPLITS,
    )
    .is_none()
}

/// How many times [`can_hold_surface`] may halve a box it cannot decide.
///
/// A bound loosens with the size of the box it covers, so a box that
/// cannot be decided whole can often be decided in pieces — this is the
/// interval-arithmetic half of the same argument the LOD field makes about
/// detail. Three levels is 8³ = 512 evaluations in the worst case, against
/// ONE 38³ density pass and a GPU round trip if the answer is wrong, so
/// the trade is lopsided by orders of magnitude and the cost only lands on
/// chunks that were marginal in the first place.
/// Octree depth `can_hold_surface` subdivides to before giving up.
///
/// Re-measured 2026-08-12 after the pipeline got roughly twice as fast:
/// the balance is CPU pruning against GPU density passes, and moving the
/// GPU side moved the knee. 3 was right when a density pass was dearer;
/// now 5 is, and 6 buys nothing back. Planet settle, 3 samples at a
/// loading budget of 192: splits 3 -> 1.76 s, 4 -> 1.77, 5 -> 1.65,
/// 6 -> 1.69.
///
/// Env-overridable for the same reason the frame budgets are: where the
/// knee sits is a property of the machine as much as of the code.
static PRUNE_SPLITS: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
    std::env::var("VOXEL_PRUNE_SPLITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}); // Measured on the planet: 3 splits prune 37% of resident chunks and
    // settle in 1.79 s; 4 prunes 39% and settles in 2.27; 5 prunes 40% and
    // settles in 2.56. The bound is loose because the LOD field is already
    // surface-hugging, so what reaches it is marginal by construction —
    // subdividing asks a harder question more times, and the CPU it spends
    // is not repaid by the GPU passes it saves.

/// `Some(true)` if the box is entirely solid, `Some(false)` if entirely
/// air, `None` if a surface could cross it.
///
/// Sub-boxes TILE the parent exactly, so "every piece is air" is a proof
/// that the whole is air. Pieces that disagree prove the opposite — a
/// surface lies between them — and one undecided piece is enough to give
/// up on the whole box.
fn uniform_sign(
    generator: &voxel_worldgen::Generator,
    min: Vec3,
    max: Vec3,
    voxel_size: f32,
    splits: u32,
) -> Option<bool> {
    // No bound at all means the program can put solid anywhere: subdividing
    // an unbounded op only asks the same unanswerable question 8 times.
    let sdf = generator.range(min, max, voxel_size)?;
    if sdf.is_positive() {
        return Some(false);
    }
    if sdf.is_negative() {
        return Some(true);
    }
    if splits == 0 {
        return None;
    }
    let mid = (min + max) * 0.5;
    let mut solid: Option<bool> = None;
    for octant in 0..8u32 {
        let axis = |bit: u32, lo: f32, mid: f32, hi: f32| {
            if octant & (1 << bit) == 0 {
                (lo, mid)
            } else {
                (mid, hi)
            }
        };
        let (x0, x1) = axis(0, min.x, mid.x, max.x);
        let (y0, y1) = axis(1, min.y, mid.y, max.y);
        let (z0, z1) = axis(2, min.z, mid.z, max.z);
        let piece = uniform_sign(
            generator,
            Vec3::new(x0, y0, z0),
            Vec3::new(x1, y1, z1),
            voxel_size,
            splits - 1,
        )?;
        match solid {
            None => solid = Some(piece),
            Some(seen) if seen == piece => {}
            _ => return None,
        }
    }
    solid
}

/// How far out a level's chunks can still be resident, in meters — the box
/// a predicate has to be evaluated over.
///
/// A chunk of level L is drawn only if its parent split, which happens
/// within `split_k·2E` of the parent's near face. The chunk sits in one
/// octant of that parent, so its own nearest point is at worst half a
/// parent diagonal — `√3·E` — further out. Hence `2·split_k·E + √3·E`,
/// which on the shipped planet is 6.73E against a measured worst case of
/// 6.7E: the bound is tight, not padded.
///
/// Too small a box silently clips corner chunks the field wants, which is
/// a hole that only appears at some camera positions. Too large costs
/// predicate evaluations here and residency in every consumer that sizes
/// itself from this.
///
/// Detail volumes are deliberately NOT in this bound: their refined sets
/// are covered by pinned per-volume dependencies sized from
/// [`detail_reach_m`], on the LOD side and in every planning consumer.
/// Folding them in here would grow every camera-following box by
/// `2^levels` per axis instead.
pub fn resident_reach(config: &LodConfig, level: u8) -> f64 {
    const SQRT_3: f64 = 1.732_050_807_568_877_2;
    let edge = ChunkKey::new(level, IVec3::ZERO).edge_m();
    2.0 * config.split_k * edge + SQRT_3 * edge
}

/// The box a level's residency predicate must be evaluated over, as a
/// `TopDep` size.
///
/// Lives beside the rules it must not clip: [`resident_reach`] for a
/// level that is bounded by distance, and the ring's own cell count and
/// vertical band for the top level, which is bounded by neither.
pub fn level_span(config: &LodConfig, level: u8) -> IVec3 {
    if level < config.max_level {
        return IVec3::splat(2 * resident_reach(config, level).ceil() as i32);
    }
    let edge = ChunkKey::new(level, IVec3::ZERO).edge_m().ceil() as i32;
    // The band is absolute, not centred on the camera, so the box has to
    // reach it from wherever the camera is; a cell of slack on each side
    // costs only predicate evaluations.
    IVec3::new(
        2 * edge * (config.top_radius + 1),
        4 * edge * (config.top_y.1 - config.top_y.0 + 2),
        2 * edge * (config.top_radius + 1),
    )
}

#[cfg(test)]
mod residency_shape {
    use super::*;
    use bevy::platform::collections::{HashMap, HashSet};

    /// THE SPECIFICATION: the configuration the retired epoch machine
    /// converged to, by iterating a fixpoint over a whole shown set.
    ///
    /// It is kept because the closed-form field in this module is only
    /// trustworthy against something independent, and this is what it used
    /// to be. The machine that ran it is gone; the definition it computed is
    /// not allowed to change.
    fn converged_leaves(config: &LodConfig, anchor: DVec3) -> HashSet<ChunkKey> {
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
        let top_edge = ChunkKey::new(config.max_level, IVec3::ZERO).edge_m();
        let cx = (anchor.x / top_edge).floor() as i32;
        let cz = (anchor.z / top_edge).floor() as i32;
        for dz in -config.top_radius..=config.top_radius {
            for dx in -config.top_radius..=config.top_radius {
                for y in config.top_y.0..=config.top_y.1 {
                    let cell = IVec3::new(cx + dx, y, cz + dz);
                    descend(
                        config,
                        anchor,
                        ChunkKey::new(config.max_level, cell),
                        &mut leaves,
                    );
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
        leaves
    }

    /// A shown configuration, as the epoch machine modelled one: a leaf
    /// set, from which the level covering any region and the seam mask of
    /// any leaf follow.
    struct PostState<'a> {
        leaves: &'a HashSet<ChunkKey>,
    }

    impl PostState<'_> {
        fn current(leaves: &HashSet<ChunkKey>) -> PostState<'_> {
            PostState { leaves }
        }

        fn is_leaf(&self, key: ChunkKey) -> bool {
            self.leaves.contains(&key)
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
        fn has_touching_finer_than(
            &self,
            region: ChunkKey,
            target: ChunkKey,
            min_level: u8,
        ) -> bool {
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

    fn planet() -> LodConfig {
        LodConfig {
            max_level: 11,
            top_radius: 3,
            top_y: (-1, 0),
            split_k: 2.5,
            merge_k: 3.0,
            ..Default::default()
        }
    }

    fn mega() -> LodConfig {
        LodConfig {
            max_level: 8,
            top_radius: 2,
            top_y: (-3, 3),
            split_k: 1.6,
            merge_k: 2.1,
            ..Default::default()
        }
    }

    fn top_ring(config: &LodConfig, anchor: DVec3) -> HashSet<ChunkKey> {
        let edge = ChunkKey::new(config.max_level, IVec3::ZERO).edge_m();
        let cx = (anchor.x / edge).floor() as i32;
        let cz = (anchor.z / edge).floor() as i32;
        let mut out = HashSet::new();
        for dz in -config.top_radius..=config.top_radius {
            for dx in -config.top_radius..=config.top_radius {
                for y in config.top_y.0..=config.top_y.1 {
                    out.insert(ChunkKey::new(
                        config.max_level,
                        IVec3::new(cx + dx, y, cz + dz),
                    ));
                }
            }
        }
        out
    }

    /// Chunks of one level a top dependency would hold under the field
    /// predicate. The top level has no parent, so it keeps today's ring.
    fn resident_level(config: &LodConfig, anchor: DVec3, level: u8) -> HashSet<ChunkKey> {
        if level == config.max_level {
            return top_ring(config, anchor);
        }
        let reach = resident_reach(config, level);
        keys_within(anchor, level, reach, |key| resident_at(config, anchor, key))
    }

    /// The field predicate WITHOUT the clamp, kept only to measure what
    /// the clamp is worth: it misses the forced splits, which are
    /// two-level jumps, which are pinholes.
    fn resident_at(config: &LodConfig, anchor: DVec3, key: ChunkKey) -> bool {
        let split_here =
            key.level > 0 && farthest_corner(anchor, key) < config.split_k * key.edge_m();
        let parent = key.parent();
        let parent_split = key.level >= config.max_level
            || aabb_distance(anchor, parent) < config.split_k * parent.edge_m();
        in_top_ring(config, anchor, key) && !split_here && parent_split
    }

    /// Distance from `p` to the farthest corner of a chunk's box.
    fn farthest_corner(p: DVec3, key: ChunkKey) -> f64 {
        let min = key.min_corner_m();
        let max = min + DVec3::splat(key.edge_m());
        (p - min).abs().max((p - max).abs()).length()
    }

    /// One predicate covers every level, the top one included: outside the
    /// ring there is no world.
    fn resident_level_clamped(config: &LodConfig, anchor: DVec3, level: u8) -> HashSet<ChunkKey> {
        let reach = resident_reach(config, level);
        keys_within(anchor, level, reach, |key| {
            resident_clamped(config, anchor, key)
        })
    }

    /// Chunks a box-with-hole top dependency would hold, the sizing the
    /// plan specified — kept as the measurement it lost to.
    fn boxed_level(config: &LodConfig, anchor: DVec3, level: u8) -> HashSet<ChunkKey> {
        if level == config.max_level {
            return top_ring(config, anchor);
        }
        let edge = ChunkKey::new(level, IVec3::ZERO).edge_m();
        let hole = if level == 0 {
            0.0
        } else {
            config.split_k * edge
        };
        keys_within(anchor, level, 2.0 * config.merge_k * edge, |key| {
            let min = key.min_corner_m();
            let max = min + DVec3::splat(edge);
            let in_hole = min.cmpge(anchor - DVec3::splat(hole)).all()
                && max.cmple(anchor + DVec3::splat(hole)).all();
            !in_hole
        })
    }

    /// Every key of one level within `reach` metres of the anchor that
    /// `keep` accepts. The three shapes above differ only in `keep`; the
    /// box they scan is the same arithmetic three times over.
    fn keys_within(
        anchor: DVec3,
        level: u8,
        reach: f64,
        keep: impl Fn(ChunkKey) -> bool,
    ) -> HashSet<ChunkKey> {
        let edge = ChunkKey::new(level, IVec3::ZERO).edge_m();
        let lo = ((anchor - DVec3::splat(reach)) / edge).floor();
        let hi = ((anchor + DVec3::splat(reach)) / edge).ceil();
        let mut out = HashSet::new();
        for z in lo.z as i32..hi.z as i32 {
            for y in lo.y as i32..hi.y as i32 {
                for x in lo.x as i32..hi.x as i32 {
                    let key = ChunkKey::new(level, IVec3::new(x, y, z));
                    if keep(key) {
                        out.insert(key);
                    }
                }
            }
        }
        out
    }

    fn measure(
        config: &LodConfig,
        anchor: DVec3,
        shape: fn(&LodConfig, DVec3, u8) -> HashSet<ChunkKey>,
        drawn: &HashSet<ChunkKey>,
    ) -> (usize, usize) {
        let levels: Vec<HashSet<ChunkKey>> = (0..=config.max_level)
            .map(|l| shape(config, anchor, l))
            .collect();
        let resident = levels.iter().map(HashSet::len).sum();
        let uncovered = drawn
            .iter()
            .filter(|k| !levels[k.level as usize].contains(*k))
            .count();
        (resident, uncovered)
    }

    /// The residency each shape would cost, against the configuration the
    /// shipped levels use. This is the measurement the LOD-as-layers
    /// conversion is sized from; the ratios are asserted so that changing
    /// the shape, or `split_k`, cannot quietly move them.
    #[test]
    fn the_field_is_a_cheaper_shape_than_a_box() {
        for (name, config, ceiling) in [("planet", planet(), 1.3), ("mega", mega(), 1.3)] {
            for anchor in [
                DVec3::new(-27570.0, 80.0, -36770.0),
                DVec3::new(1234.0, 600.0, -800.0),
            ] {
                let drawn = converged_leaves(&config, anchor);
                let (field, field_missed) = measure(&config, anchor, resident_level, &drawn);
                let (boxed, boxed_missed) = measure(&config, anchor, boxed_level, &drawn);
                println!(
                    "{name} @{:?}: drawn {} — field {field} ({:.2}x, {field_missed} missed), \
                     box {boxed} ({:.2}x, {boxed_missed} missed)",
                    anchor.as_ivec3(),
                    drawn.len(),
                    field as f64 / drawn.len() as f64,
                    boxed as f64 / drawn.len() as f64,
                );
                assert!(
                    (field as f64) < ceiling * drawn.len() as f64,
                    "{name}: field residency {field} exceeds {ceiling}x of {}",
                    drawn.len(),
                );
            }
        }
    }

    /// The closed-form clamped field IS the configuration the epoch
    /// machine converges to by iterating a fixpoint. Descending with
    /// `split_clamped` from the top ring reproduces `plan_genesis`'s leaf
    /// set exactly — which is what lets residency be decided one chunk at
    /// a time.
    #[test]
    fn the_closed_form_clamp_reproduces_the_fixpoint() {
        for (name, config) in [("planet", planet()), ("mega", mega())] {
            for anchor in [
                DVec3::new(-27570.0, 80.0, -36770.0),
                DVec3::new(1234.0, 600.0, -800.0),
                DVec3::new(0.0, 0.0, 0.0),
            ] {
                let expect = converged_leaves(&config, anchor);
                let mut got: HashSet<ChunkKey> = HashSet::new();
                fn descend(
                    config: &LodConfig,
                    anchor: DVec3,
                    k: ChunkKey,
                    out: &mut HashSet<ChunkKey>,
                ) {
                    if split_clamped(config, anchor, k) {
                        for c in k.children() {
                            descend(config, anchor, c, out);
                        }
                    } else {
                        out.insert(k);
                    }
                }
                for top in top_ring(&config, anchor) {
                    descend(&config, anchor, top, &mut got);
                }
                let extra: Vec<&ChunkKey> = got.difference(&expect).take(3).collect();
                let short: Vec<&ChunkKey> = expect.difference(&got).take(3).collect();
                assert_eq!(
                    got.len(),
                    expect.len(),
                    "{name} @{:?}: closed form has {} leaves, fixpoint {} — extra {extra:?}, \
                     missing {short:?}",
                    anchor.as_ivec3(),
                    got.len(),
                    expect.len(),
                );
                assert_eq!(got, expect, "{name} @{:?}", anchor.as_ivec3());
            }
        }
    }

    /// The mask a chunk gets from the field alone is the mask the epoch
    /// machine derives from its shown configuration. Both feed the same
    /// `snap_to_parity` in the mesh shader, so any disagreement is a crack.
    #[test]
    fn the_field_mask_is_the_shown_mask() {
        for (name, config) in [("planet", planet()), ("mega", mega())] {
            for anchor in [
                DVec3::new(-27570.0, 80.0, -36770.0),
                DVec3::new(1234.0, 600.0, -800.0),
            ] {
                let leaves = converged_leaves(&config, anchor);
                let post = PostState::current(&leaves);
                for leaf in &leaves {
                    assert_eq!(
                        seam_mask_at(&config, anchor, *leaf),
                        post.seam_mask(config.max_level, *leaf),
                        "{name} @{:?}: {leaf:?}",
                        anchor.as_ivec3(),
                    );
                }
            }
        }
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
    ///
    /// These are the retired epoch machine's own assertions about a
    /// configuration it was about to show.
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

    /// The configuration the closed form produces satisfies every
    /// crack-freedom invariant the epoch machine's own tests assert of a
    /// shown configuration — not just "same leaf set as genesis".
    ///
    /// `plan_genesis` clamps with a same-level neighbor test; the running
    /// machine additionally vetoes a merge whose region would TOUCH
    /// something two levels finer. Matching the first does not imply
    /// matching the second, and a corner where it does not is a pinhole.
    #[test]
    fn the_closed_form_is_crack_free_by_the_machine_s_own_rules() {
        for (name, config) in [("planet", planet()), ("mega", mega())] {
            // One anchor per configuration: the scan is O(leaves x 26 x
            // touching subtree) and this is already the slowest test here.
            for anchor in [DVec3::new(-29840.0, 2400.0, -36767.0)] {
                let mut leaves: HashSet<ChunkKey> = HashSet::new();
                fn descend(
                    config: &LodConfig,
                    anchor: DVec3,
                    k: ChunkKey,
                    out: &mut HashSet<ChunkKey>,
                ) {
                    if split_clamped(config, anchor, k) {
                        for c in k.children() {
                            descend(config, anchor, c, out);
                        }
                    } else {
                        out.insert(k);
                    }
                }
                for top in top_ring(&config, anchor) {
                    descend(&config, anchor, top, &mut leaves);
                }
                let masks: HashMap<ChunkKey, u32> = leaves
                    .iter()
                    .map(|k| (*k, seam_mask_at(&config, anchor, *k)))
                    .collect();
                let _ = name;
                assert_consistent(&config, &leaves, &masks);
            }
        }
    }

    /// A configuration with a detail volume 1.5 km from the anchor —
    /// far enough that distance alone leaves its chunks coarse. `levels`
    /// must out-run the scale cap to matter from there: an 80 m volume
    /// fits level 5, the plain leaf at 1.5 km is level 7, so anything
    /// under 3 is inert at this range by design.
    fn biased() -> (LodConfig, DetailVolume) {
        let volume = DetailVolume {
            min: DVec3::new(-26040.0, 60.0, -36810.0),
            max: DVec3::new(-25960.0, 140.0, -36730.0),
            levels: 4,
        };
        let mut config = planet();
        config.detail = std::sync::Arc::new(vec![volume]);
        (config, volume)
    }

    fn clamped_leaves(config: &LodConfig, anchor: DVec3) -> HashSet<ChunkKey> {
        fn descend(config: &LodConfig, anchor: DVec3, k: ChunkKey, out: &mut HashSet<ChunkKey>) {
            if split_clamped(config, anchor, k) {
                for c in k.children() {
                    descend(config, anchor, c, out);
                }
            } else {
                out.insert(k);
            }
        }
        let mut leaves = HashSet::new();
        for top in top_ring(config, anchor) {
            descend(config, anchor, top, &mut leaves);
        }
        leaves
    }

    /// With a volume, the field's face-adjacent leaves can differ by MORE
    /// than the two levels a plain field allows at corners, so the clamp
    /// has to close multi-level gaps it never used to see. The fixpoint is
    /// the independent definition of "closed": the closed form must still
    /// reproduce it exactly.
    #[test]
    fn the_closed_form_clamp_reproduces_the_fixpoint_with_detail() {
        let (config, _) = biased();
        let anchor = DVec3::new(-27570.0, 80.0, -36770.0);
        let expect = converged_leaves(&config, anchor);
        let got = clamped_leaves(&config, anchor);
        let extra: Vec<&ChunkKey> = got.difference(&expect).take(3).collect();
        let short: Vec<&ChunkKey> = expect.difference(&got).take(3).collect();
        assert_eq!(
            got, expect,
            "detail: closed form disagrees with fixpoint — extra {extra:?}, missing {short:?}",
        );
    }

    /// The biased configuration passes every crack-freedom assertion the
    /// epoch machine made of a shown set, masks included.
    #[test]
    fn detail_volumes_stay_crack_free() {
        let (config, _) = biased();
        let anchor = DVec3::new(-27570.0, 80.0, -36770.0);
        let leaves = clamped_leaves(&config, anchor);
        let masks: HashMap<ChunkKey, u32> = leaves
            .iter()
            .map(|k| (*k, seam_mask_at(&config, anchor, *k)))
            .collect();
        assert_consistent(&config, &leaves, &masks);
    }

    /// What a volume buys and what it costs: the leaf over the volume is
    /// `levels` finer than distance alone gives, and every leaf the volume
    /// adds sits within [`detail_reach_m`] of the volume's box at its own
    /// scale — the bound the volume-anchored dependency boxes are sized
    /// from. A leaf outside that reach is refinement leaking to chunks the
    /// volume does not touch.
    #[test]
    fn a_volume_refines_at_distance_and_only_nearby() {
        let (config, volume) = biased();
        let plain = planet();
        let anchor = DVec3::new(-27570.0, 80.0, -36770.0);
        let center = (volume.min + volume.max) * 0.5;
        let leaf_covering = |config: &LodConfig, p: DVec3| -> ChunkKey {
            let mut k = ChunkKey::containing(p, config.max_level);
            while split_clamped(config, anchor, k) {
                k = *k
                    .children()
                    .iter()
                    .find(|c| ChunkKey::containing(p, c.level) == **c)
                    .unwrap();
            }
            k
        };
        let with = leaf_covering(&config, center);
        let without = leaf_covering(&plain, center);
        assert_eq!(
            u32::from(with.level) + u32::from(volume.levels),
            u32::from(without.level),
            "volume did not buy exactly {} levels: {with:?} vs {without:?}",
            volume.levels,
        );
        let biased_leaves = clamped_leaves(&config, anchor);
        let plain_leaves = clamped_leaves(&plain, anchor);
        for leaf in biased_leaves.difference(&plain_leaves) {
            let min = leaf.min_corner_m();
            let max = min + DVec3::splat(leaf.edge_m());
            let gap = (volume.min - max).max(min - volume.max).max(DVec3::ZERO);
            assert!(
                gap.max_element() <= detail_reach_m(&volume, leaf.level),
                "leaf {leaf:?} refined {:.0} m from the volume",
                gap.max_element(),
            );
        }
        let extra = biased_leaves.len() - plain_leaves.len();
        assert!(
            extra > 0 && extra < 2000,
            "one small volume changed residency by {extra} leaves"
        );
    }

    /// Every leaf the clamped field wants sits inside some dependency's
    /// box, under the EXACT runtime geometry: camera boxes from
    /// [`level_span`] centred on the truncated anchor, volume boxes from
    /// [`detail_reach_m`] centred on the truncated volume centre, both
    /// rounded to chunk indices the way `chunk_range` rounds. An
    /// uncovered leaf is a chunk residency silently clips — a hole that
    /// only appears at some camera positions. Shipped-planet scale on
    /// purpose: the fixpoint tests run at `max_level` 11, and a clipped
    /// corner three levels up would never show there.
    #[test]
    fn every_wanted_leaf_is_inside_a_dependency_box() {
        let volume = DetailVolume {
            min: DVec3::new(-27585.0, 130.0, -36715.0),
            max: DVec3::new(-27535.0, 185.0, -36665.0),
            levels: 4,
        };
        let config = LodConfig {
            max_level: 14,
            top_radius: 3,
            top_y: (-1, 0),
            detail: std::sync::Arc::new(vec![volume]),
            ..Default::default()
        };
        let camera = DVec3::new(-26100.0, 180.0, -36700.0);
        let anchor = camera.as_ivec3().as_dvec3();
        // Inclusive chunk-index range of a focus-centred box, exactly as
        // `TopDep::bounds` + `chunk_range` compute it.
        let indices = |focus: IVec3, size: IVec3, edge: f64| -> (IVec3, IVec3) {
            let size = size.max(IVec3::ONE);
            let min = focus - size / 2;
            let max = min + size;
            let axis = |lo: i32, hi: i32| {
                (
                    (f64::from(lo) / edge).floor() as i32,
                    (f64::from(hi) / edge).ceil() as i32 - 1,
                )
            };
            let (x0, x1) = axis(min.x, max.x);
            let (y0, y1) = axis(min.y, max.y);
            let (z0, z1) = axis(min.z, max.z);
            (IVec3::new(x0, y0, z0), IVec3::new(x1, y1, z1))
        };
        let contains =
            |(lo, hi): (IVec3, IVec3), pos: IVec3| lo.cmple(pos).all() && pos.cmple(hi).all();
        let holes: Vec<ChunkKey> = clamped_leaves(&config, anchor)
            .into_iter()
            .filter(|key| {
                let edge = key.edge_m();
                if contains(
                    indices(camera.as_ivec3(), level_span(&config, key.level), edge),
                    key.pos,
                ) {
                    return false;
                }
                if key.level >= config.max_level {
                    return true; // the ring box must have covered it
                }
                let reach = 2.0 * detail_reach_m(&volume, key.level);
                if reach == 0.0 {
                    return true;
                }
                let span = ((volume.max - volume.min) + DVec3::splat(reach))
                    .ceil()
                    .as_ivec3();
                let center = ((volume.min + volume.max) * 0.5).as_ivec3();
                !contains(indices(center, span, edge), key.pos)
            })
            .collect();
        assert!(
            holes.is_empty(),
            "{} leaves outside every dependency box, e.g. {:?}",
            holes.len(),
            &holes[..holes.len().min(5)],
        );
    }

    /// Residency under the clamped predicate covers every chunk the epoch
    /// machine draws — the gap the plain field predicate leaves — and
    /// still costs a fraction of what a box costs.
    #[test]
    fn the_clamped_predicate_covers_what_the_field_alone_misses() {
        for (name, config) in [("planet", planet()), ("mega", mega())] {
            for anchor in [
                DVec3::new(-27570.0, 80.0, -36770.0),
                DVec3::new(1234.0, 600.0, -800.0),
            ] {
                let drawn = converged_leaves(&config, anchor);
                let (resident, missed) = measure(&config, anchor, resident_level_clamped, &drawn);
                println!(
                    "{name} @{:?}: drawn {} — clamped field {resident} ({:.2}x, {missed} missed)",
                    anchor.as_ivec3(),
                    drawn.len(),
                    resident as f64 / drawn.len() as f64,
                );
                assert_eq!(missed, 0, "{name}: shown chunks outside residency");
                assert!(
                    (resident as f64) < 1.3 * drawn.len() as f64,
                    "{name}: {resident}"
                );
            }
        }
    }
}

#[cfg(test)]
mod field_invariants {
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

#[cfg(test)]
mod pruning {
    use super::*;
    use voxel_core::seed::Rng;

    /// A chunk we refuse to generate must genuinely have no surface in
    /// it. This is the property the whole optimisation rests on, and the
    /// coverage eval cannot prove it: a wrongly pruned chunk is a hole
    /// that appears only when the camera happens to look at it.
    ///
    /// So it is checked directly — densely sample the chunk's box with
    /// the REAL evaluator, including the density apron, and assert the
    /// SDF never changes sign in anything we skipped.
    /// Densely sample a chunk's box, apron included, with the REAL
    /// evaluator. Returns whether it holds both solid and air — i.e.
    /// whether a surface crosses it.
    fn has_surface(generator: &voxel_worldgen::Generator, key: ChunkKey) -> bool {
        let vs = key.voxel_size_m() as f32;
        let min = key.min_corner_m().as_vec3() - Vec3::splat(4.0 * vs);
        let span = key.edge_m() as f32 + 8.0 * vs;
        let (mut solid, mut air) = (false, false);
        const N: i32 = 10;
        for iz in 0..=N {
            for iy in 0..=N {
                for ix in 0..=N {
                    let p = min + Vec3::new(ix as f32, iy as f32, iz as f32) * (span / N as f32);
                    let (d, _) = voxel_worldgen::program::eval(generator.ops(), 0, p, vs);
                    if d <= 0.0 {
                        solid = true;
                    } else {
                        air = true;
                    }
                    if solid && air {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// A chunk we refuse to generate must genuinely have no surface in
    /// it. This is the property the whole optimisation rests on, and the
    /// coverage eval cannot prove it: a wrongly pruned chunk is a hole
    /// that shows only when the camera happens to look at it.
    ///
    /// Swept rather than sampled. Random chunks over a whole world are
    /// almost all deep sky, where any bound is right and a broken one
    /// passes — verified: halving the noise bound went unnoticed until
    /// this walked the column THROUGH the ground at every level, which is
    /// where a bound is load-bearing.
    /// Diagnostic: how often the bound actually DECIDES, per level, on
    /// the shipped planet, for chunks that genuinely have no surface.
    /// A bound that is always conservative but never decisive prunes
    /// nothing, and the streamer pays a density pass for each one.
    #[test]
    #[ignore]
    fn how_decisive_is_the_bound() {
        let generator = voxel_worldgen::Generator::new(
            voxel_worldgen::program::planet_program(),
            0,
            Vec3::new(0.55, 0.5, 0.32),
        );
        let mut rng = Rng::new(0xC0DE_5EED);
        println!("{:>4} {:>8} {:>8} {:>8}", "lvl", "empty", "pruned", "%");
        for level in 0..12u8 {
            let (mut empty, mut pruned) = (0, 0);
            for _ in 0..40 {
                let xz = bevy::math::Vec2::new(
                    (rng.next_f32() - 0.5) * 60_000.0,
                    (rng.next_f32() - 0.5) * 60_000.0,
                );
                let ground = generator.height(xz, 1.0);
                let edge = ChunkKey::new(level, IVec3::ZERO).edge_m() as f32;
                let cy = (ground / edge).floor() as i32;
                for dy in -6..=6 {
                    let key = ChunkKey::new(
                        level,
                        IVec3::new(
                            (xz.x / edge).floor() as i32,
                            cy + dy,
                            (xz.y / edge).floor() as i32,
                        ),
                    );
                    if has_surface(&generator, key) {
                        continue;
                    }
                    empty += 1;
                    if !can_hold_surface(&generator, key) {
                        pruned += 1;
                    }
                }
            }
            println!(
                "{level:>4} {empty:>8} {pruned:>8} {:>7.0}%",
                100.0 * f64::from(pruned) / f64::from(empty.max(1))
            );
        }
    }

    #[test]
    fn nothing_pruned_had_a_surface_in_it() {
        let generator = voxel_worldgen::Generator::new(
            voxel_worldgen::program::planet_program(),
            0,
            Vec3::new(0.55, 0.5, 0.32),
        );
        let mut rng = Rng::new(0xC0DE_5EED);
        let mut pruned = 0;
        let mut marginal = 0;
        for _ in 0..24 {
            let xz = bevy::math::Vec2::new(
                (rng.next_f32() - 0.5) * 60_000.0,
                (rng.next_f32() - 0.5) * 60_000.0,
            );
            let ground = generator.height(xz, 1.0);
            for level in 0..12u8 {
                let edge = ChunkKey::new(level, IVec3::ZERO).edge_m() as f32;
                let cy = (ground / edge).floor() as i32;
                // The column through the surface, both ways out of it.
                for dy in -6..=6 {
                    let key = ChunkKey::new(
                        level,
                        IVec3::new(
                            (xz.x / edge).floor() as i32,
                            cy + dy,
                            (xz.y / edge).floor() as i32,
                        ),
                    );
                    if can_hold_surface(&generator, key) {
                        continue;
                    }
                    pruned += 1;
                    if dy.abs() <= 2 {
                        marginal += 1;
                    }
                    assert!(
                        !has_surface(&generator, key),
                        "pruned {key:?} but a surface crosses it — that is a hole"
                    );
                }
            }
        }
        assert!(pruned > 500, "only {pruned} chunks pruned; not a test");
        assert!(
            marginal > 20,
            "only {marginal} pruned chunks were near the ground; the sweep is not reaching \
             the cases where the bound is load-bearing"
        );
    }
}

#[cfg(test)]
mod residency_budget {
    use super::*;

    /// The count must be the residency PREDICATE's own set, or admission
    /// control is checking a different number from the one the graph will
    /// ask for.
    #[test]
    fn the_count_is_the_predicate_it_claims_to_be() {
        let config = LodConfig {
            max_level: 4,
            top_radius: 1,
            top_y: (0, 0),
            ..Default::default()
        };
        let anchor = DVec3::new(120.0, 40.0, -60.0);
        // Enumerate every key the predicate could accept, by descending
        // the whole top ring to level 0 and testing each node.
        fn walk(config: &LodConfig, anchor: DVec3, key: ChunkKey, out: &mut usize) {
            if resident_clamped(config, anchor, key) {
                *out += 1;
            }
            if key.level > 0 {
                for child in key.children() {
                    walk(config, anchor, child, out);
                }
            }
        }
        let top_edge = ChunkKey::new(config.max_level, IVec3::ZERO).edge_m();
        let cx = (anchor.x / top_edge).floor() as i32;
        let cz = (anchor.z / top_edge).floor() as i32;
        let mut expected = 0;
        for dz in -config.top_radius..=config.top_radius {
            for dx in -config.top_radius..=config.top_radius {
                for y in config.top_y.0..=config.top_y.1 {
                    let cell = IVec3::new(cx + dx, y, cz + dz);
                    walk(
                        &config,
                        anchor,
                        ChunkKey::new(config.max_level, cell),
                        &mut expected,
                    );
                }
            }
        }
        assert_eq!(resident_count(&config, anchor), expected);
    }

    /// Cheap enough to run when a world loads, on the configurations that
    /// ship. If this ever stops being true the answer is a cached bound,
    /// not a slower load.
    #[test]
    fn counting_the_shipped_configs_is_fast() {
        let config = LodConfig::default();
        let start = std::time::Instant::now();
        let count = resident_count(&config, DVec3::new(0.0, 100.0, 0.0));
        let elapsed = start.elapsed();
        assert!(count > 1_000, "the default config is not trivial: {count}");
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "counting residency took {elapsed:?} for {count} chunks",
        );
        println!("default config: {count} resident chunks in {elapsed:?}");
    }

    /// Fewer levels means fewer chunks — the property capping relies on.
    #[test]
    fn capping_the_level_reduces_the_count() {
        let anchor = DVec3::new(0.0, 100.0, 0.0);
        let mut last = usize::MAX;
        for max_level in (4..=8u8).rev() {
            let config = LodConfig {
                max_level,
                ..Default::default()
            };
            let count = resident_count(&config, anchor);
            assert!(
                count < last,
                "L{max_level} = {count} did not shrink below {last}"
            );
            last = count;
        }
    }
}
