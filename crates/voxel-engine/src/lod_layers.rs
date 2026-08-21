//! LOD levels as layers.
//!
//! One layer instance per level, whose chunks ARE the voxel chunks — not a
//! second materialisation on top of them. Residency is the dependency
//! graph's job: a top dependency per level, shaped by the LOD field
//! itself, and `create`/`destroy` request and free the chunk.
//!
//! Three properties come out of the structure rather than from machinery:
//!
//! - **Ready before swap.** `create` blocks until the chunk is drawable,
//!   and [`LayerGraph::process_tops`] runs every ensure before any
//!   release. A chunk replacing another therefore exists, meshed, before
//!   the one it replaces is freed. What overlaps is coplanar; what never
//!   happens is a gap.
//! - **No plan.** Residency is a pure function of (chunk, anchor), so
//!   there is nothing to plan, budget, trickle or commit — and nothing
//!   that can be planned against a configuration that has since moved.
//! - **Seams without a tree.** A mask is `seam_mask_at(config, anchor,
//!   key)`, which is proven equal to the mask the epoch machine derives
//!   from its shown configuration. A chunk can be built knowing only
//!   where the camera is.
//!
//! What is left over is the one thing that genuinely is not a function of
//! a chunk's coordinate: its mask follows the camera, so when the anchor
//! moves, chunks that stay resident need their meshes rebuilt in place.
//! That is NOT a lifecycle event and deliberately does not go through the
//! graph — see [`refresh_masks`].

use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bevy::math::DVec3;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use voxel_core::csg::CsgOp;
use voxel_core::ChunkKey;
use voxel_layers::{ChunkCtx, CoordFilter, Layer, LayerChunk, LayerGraph, LayerRuntime, TopDep};

use crate::chunkgen::ChunkGen;
use crate::streaming::{
    can_hold_surface, detail_reach_m, level_span, resident_clamped, seam_mask_at, LodConfig,
    StreamProbe, StreamingRebuild,
};

/// How long a `create` waits for its chunk to become drawable before
/// giving up on it.
///
/// The only thing that takes this long is slab exhaustion, and the layer
/// model has no answer for it — residency is what the field says it is,
/// so there is no coarsening to fall back on the way an aborted epoch had.
/// Giving up leaves a hole and counts it, which is at least a number
/// somebody can read; blocking forever would wedge the generation thread
/// and present as a frozen world.
const CREATE_TIMEOUT: Duration = Duration::from_secs(10);

/// Layer-thread workers.
///
/// This used to be a count of chunks IN FLIGHT, because `create` blocked
/// on the GPU round trip and a thread was the only way to have another
/// chunk outstanding. It no longer waits — the pass waits, once — so what
/// is left in a create is CPU: resolving ops and deciding whether the
/// generator can put a surface in the box. That wants cores, not
/// hundreds of threads, and 256 of them competing for ten cores measured
/// slightly WORSE than 32 as well as burying the machine.
const LOD_WORKERS: usize = 32;

/// Instance name of a level's layer.
fn instance(level: u8) -> String {
    format!("voxel:l{level}")
}

/// The camera anchor every level shares.
///
/// One value, published once for all levels. Each level quantizes it to
/// its own grid internally, but they must all be reading the SAME point:
/// two levels disagreeing about where the field is centred is a chunk one
/// of them thinks it covers and the other thinks it does not, which is a
/// hole or a double-cover depending on the sign.
#[derive(Default)]
struct Anchor([AtomicI32; 3]);

impl Anchor {
    fn load(&self) -> IVec3 {
        IVec3::new(
            self.0[0].load(Ordering::Relaxed),
            self.0[1].load(Ordering::Relaxed),
            self.0[2].load(Ordering::Relaxed),
        )
    }

    fn store(&self, v: IVec3) {
        self.0[0].store(v.x, Ordering::Relaxed);
        self.0[1].store(v.y, Ordering::Relaxed);
        self.0[2].store(v.z, Ordering::Relaxed);
    }
}

/// Everything a level's chunks share.
///
/// One object rather than a handful of parallel `Arc`s: the anchor, the
/// record of what is shown and the service they build through are read
/// together on every pass, and a level holding five clones of them was
/// five chances for one to be forgotten.
struct LodShared {
    /// Which world these levels stream. Every key they build carries it,
    /// so one `ChunkGen` and one GPU arena serve every loaded world.
    world: voxel_core::WorldId,
    chunks: ChunkGen,
    config: LodConfig,
    /// The world itself, for asking whether a chunk can hold a surface
    /// before paying to find out.
    generator: Arc<voxel_worldgen::Generator>,
    /// Where the camera has asked the field to be centred, and where THIS
    /// pass froze it. The pass reads the dependencies' focuses one at a
    /// time, so without a snapshot a publish landing mid-read leaves two
    /// levels centred a step apart — and a seam between them that neither
    /// side owns.
    requested: Anchor,
    anchor: Anchor,
    state: Mutex<LodState>,
    /// How many chunks are shown, published separately from the set.
    ///
    /// The count is read every frame to fill in a probe counter, and the
    /// set is held by the worker for as long as a residency pass takes,
    /// so locking to ask for a `len()` parked the MAIN thread behind the
    /// pass. Measured over a 90 s flight: 67 blocks totalling 1.0 s, worst
    /// 33 ms — the largest single cause of frame spikes while moving, and
    /// none of it was work.
    resident: AtomicUsize,
    /// Chunks the pipeline could not place. A hole for as long as it lasts.
    stalled: AtomicUsize,
    /// Chunks the interval bound proved empty, and chunks that reached
    /// the generator anyway. The ratio is what says whether the bound is
    /// doing anything HERE — it can be decisive in the abstract and never
    /// consulted, because a chunk carrying planning ops skips it.
    pruned: AtomicUsize,
    unpruned: AtomicUsize,
    /// Of the unpruned, how many were blocked by carrying ops.
    had_ops: AtomicUsize,
    /// An authored edit waiting to be applied to the chunks that care.
    ///
    /// Drained by the residency pass rather than by whoever wrote it: a
    /// rebuilt chunk is only asked for by `create`, and the one moment it
    /// is settled and revealed is `between`. Invalidating from a system
    /// instead frees the chunk and leaves its replacement queued forever
    /// — a hole, until the camera happens to move.
    restale: Mutex<Option<Restale>>,
}

/// A level edit that moved authored geometry, and what it takes to find
/// the chunks it moved. See [`crate::fingerprint`].
pub struct Restale {
    pub seed: u32,
    pub was_placed: Vec<CsgOp>,
    pub now_placed: Vec<CsgOp>,
}

impl LodShared {
    /// Republish the resident count, with the lock already held. Called
    /// by everything that changes the shown set: a count that can go
    /// stale past a frame is worse than one nobody publishes.
    fn publish_resident(&self, state: &LodState) {
        self.resident.store(state.shown.len(), Ordering::Relaxed);
    }
}

/// What a resident chunk was built from.
#[derive(Clone)]
struct ShownChunk {
    /// The mask its mesh carries — what the refresh scan compares against.
    mask: u32,
    /// The ops that shaped it. Kept so a seam-only rebuild does not have
    /// to ask the planning graph again for an answer that cannot have
    /// changed: same chunk, same coordinate, same ops.
    ops: Option<Arc<Vec<CsgOp>>>,
}

#[derive(Default)]
struct LodState {
    shown: HashMap<ChunkKey, ShownChunk>,
    /// Chunks this pass has built and not yet revealed.
    pending: Vec<ChunkKey>,
    /// Builds this pass has asked for and not yet waited on, and what each
    /// becomes once it arrives. A `create` only ASKS — see `settle_builds`.
    asked: crate::chunkgen::ChunkBatch,
    building: HashMap<ChunkKey, ShownChunk>,
}

/// One LOD level.
struct VoxelLod {
    level: u8,
    shared: Arc<LodShared>,
}

impl Layer for VoxelLod {
    type Chunk = LodChunk;
    const NAME: &'static str = "voxel-lod";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::splat(ChunkKey::new(self.level, IVec3::ZERO).edge_m())
    }
}

/// A voxel chunk, owned by the level that wants it there.
#[derive(Default)]
struct LodChunk {
    key: Option<ChunkKey>,
}

impl LayerChunk for LodChunk {
    type Layer = VoxelLod;

    fn create(&mut self, ctx: &ChunkCtx<'_, VoxelLod>) {
        let shared = &ctx.layer().shared;
        let key = ChunkKey::in_world(shared.world, ctx.layer().level, ctx.coord());
        let ops = shared.chunks.ops_for(key);
        // With the chunk's own ops in hand the question is exact: nothing
        // planned here, and a generator that cannot put a surface in the
        // box, means there is nothing to build. Skipping costs a dozen
        // interval operations and saves a 38³ density pass and the GPU
        // round trip the pass is waited on — which is what actually
        // bounds how fast a world can appear.
        if ops.is_none() && !can_hold_surface(&shared.generator, key) {
            shared.pruned.fetch_add(1, Ordering::Relaxed);
            self.key = None;
            return;
        }
        shared.unpruned.fetch_add(1, Ordering::Relaxed);
        if ops.is_some() {
            shared.had_ops.fetch_add(1, Ordering::Relaxed);
        }
        let mask = seam_mask_at(&shared.config, shared.anchor.load().as_dvec3(), key);
        // ASK, do not wait. Waiting here made the round trip the unit of
        // work: the framework runs a level's creates in a pool and joins
        // it, so every level drained the pipeline to empty and refilled it
        // — measured as `awaiting` swinging 200 -> 0 fifteen times a pass,
        // one GPU round trip of idle GPU each. The pass already has a
        // single moment where everything it built is revealed together, so
        // that is where the waiting belongs.
        //
        // Hidden until then: shown the moment it is drawable, a new chunk
        // would be drawn against neighbors whose masks this pass has not
        // refreshed yet — a hairline along the boundary it just moved.
        {
            let mut state = shared.state.lock().unwrap();
            state
                .asked
                .add(&shared.chunks, key, mask, false, ops.clone());
            state.building.insert(key, ShownChunk { mask, ops });
        }
        // Owned from here on, whether or not it arrives: `destroy` still
        // has to free it.
        self.key = Some(key);
    }

    fn destroy(&mut self, ctx: &ChunkCtx<'_, VoxelLod>) {
        let shared = &ctx.layer().shared;
        if let Some(key) = self.key.take() {
            {
                let mut state = shared.state.lock().unwrap();
                state.shown.remove(&key);
                shared.publish_resident(&state);
            }
            shared.chunks.free(key);
        }
    }
}

/// One world's running LOD graph: a layer per level, and the thread that
/// keeps their top dependencies satisfied.
pub struct WorldLod {
    runtime: LayerRuntime,
    shared: Arc<LodShared>,
    /// Sticky anchor: the field is only re-centred when the camera has
    /// moved this far. The quantization IS the hysteresis.
    published: Option<DVec3>,
    /// How many leading top dependencies follow the camera; the rest are
    /// pinned on detail volumes.
    camera_tops: usize,
    /// The pinned dependencies' focuses. Their boxes never move — only
    /// their FILTER's answer does, so a re-centred anchor touches them
    /// instead of moving them.
    volume_tops: Vec<IVec3>,
}

/// Every world the engine is streaming.
///
/// A portal shows two levels at once, so residency cannot be a property
/// of "the" world: each has its own field, its own anchor and its own
/// generator, and they share one chunk service because the world rides in
/// the chunk key.
#[derive(Resource, Default)]
pub struct LodLayers {
    worlds: Vec<WorldLod>,
}

impl LodLayers {
    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    /// Rebuild only the chunks an authored edit changed.
    ///
    /// Queued for the residency pass and answered there — see
    /// [`restale`]. `false` if this world is not streaming yet, in which
    /// case there is nothing built to be stale.
    pub fn restale(&self, world: voxel_core::WorldId, edit: Restale) -> bool {
        let Some(lod) = self.worlds.iter().find(|w| w.shared.world == world) else {
            return false;
        };
        *lod.shared.restale.lock().unwrap() = Some(edit);
        // Nothing MOVED, so no dependency would ask for the pass that
        // settles what this is about to rebuild.
        lod.runtime.force_pass();
        true
    }

    /// Sum over worlds — what the HUD and the settle metric report.
    pub fn resident(&self) -> usize {
        self.worlds.iter().map(WorldLod::resident).sum()
    }

    pub fn stalled(&self) -> usize {
        self.worlds.iter().map(WorldLod::stalled).sum()
    }

    /// (proved empty, reached the generator, of those blocked by ops).
    pub fn prune_counts(&self) -> (usize, usize, usize) {
        self.worlds.iter().fold((0, 0, 0), |a, w| {
            let b = w.prune_counts();
            (a.0 + b.0, a.1 + b.1, a.2 + b.2)
        })
    }

    pub fn is_generating(&self) -> bool {
        self.worlds.iter().any(WorldLod::is_generating)
    }

    /// Every world has caught up with the focus.
    pub fn is_idle(&self) -> bool {
        self.worlds.iter().all(WorldLod::is_idle)
    }

    /// Re-centre each world's field on where THAT world is being looked
    /// at from, falling back to the camera.
    ///
    /// A world seen through a portal is not seen from the camera: the far
    /// side of the opening can be anywhere, and streaming it around the
    /// near camera resides chunks nowhere near what the portal shows —
    /// the portal looks out onto nothing, correctly and uselessly.
    pub fn follow(&mut self, camera: DVec3, focus: &WorldFocus) {
        for world in &mut self.worlds {
            world.follow(focus.at(world.shared.world, camera));
        }
    }
}

/// Where each world is being looked at from, when that is not the
/// camera. Indexed by world; `None` means "follow the camera".
#[derive(Resource, Default)]
pub struct WorldFocus(pub Vec<Option<DVec3>>);

impl WorldFocus {
    /// Where world `id` is being looked at from.
    ///
    /// ONE answer, for every consumer that follows a world: the LOD
    /// graphs and the planning graphs must centre on the same point or
    /// planning is resident where the chunks are not, and the chunks
    /// stream in featureless.
    pub fn at(&self, id: voxel_core::WorldId, camera: DVec3) -> DVec3 {
        self.0
            .get(usize::from(id))
            .copied()
            .flatten()
            .unwrap_or(camera)
    }
}

/// How far the camera moves before the field is re-centred.
pub const ANCHOR_STEP: f64 = 48.0;

impl WorldLod {
    fn new(
        world: voxel_core::WorldId,
        config: LodConfig,
        chunks: ChunkGen,
        generator: Arc<voxel_worldgen::Generator>,
    ) -> Self {
        let shared = Arc::new(LodShared {
            world,
            chunks,
            resident: AtomicUsize::new(0),
            generator,
            requested: Anchor::default(),
            anchor: Anchor::default(),
            state: Mutex::new(LodState::default()),
            stalled: AtomicUsize::new(0),
            restale: Mutex::new(None),
            pruned: AtomicUsize::new(0),
            unpruned: AtomicUsize::new(0),
            had_ops: AtomicUsize::new(0),
            config: config.clone(),
        });
        let mut graph = LayerGraph::new(u64::from(world)).with_threads(LOD_WORKERS);
        let mut tops = Vec::new();
        // Coarsest FIRST, but only OUTSIDE the ops horizon. Tops are
        // ensured one after another, so whichever is processed first
        // decides what the pipeline chews on while planning is still
        // running — and the coarse levels are exactly the ones that need
        // no planning at all.
        //
        // Inside the horizon that rule is backwards. There every chunk
        // waits on `chunk_covered`, and coverage is REGIONAL: a level's box
        // shrinks with its edge, so the finest levels are covered first and
        // the coarsest — largest box — last. Led with the coarsest, that
        // one level parked EVERY worker in the gate's 1 ms sleep for
        // 0.3-0.5 s (32-68 chunks reach the gate on a cold start, not the
        // one this used to claim) while the finer levels, whose regions
        // were already covered, had not been asked for yet. That wait is
        // the only one in the pass that is genuinely idle rather than
        // contended, which is why filling it works where overlapping the
        // rest of the pass does not.
        let mut gated: Vec<TopDep> = Vec::new();
        // Volume-anchored dependencies, appended after every camera
        // dependency: a detail volume keeps chunks resident far outside
        // the camera-following boxes, and growing those to cover it would
        // multiply their coordinate count by 8 per bias level. The
        // volume's own box is small and static; only the filter's answer
        // moves with the camera, which is what `touch` re-asks for.
        let mut volumes: Vec<(TopDep, IVec3)> = Vec::new();
        for level in (0..=config.max_level).rev() {
            graph.register_as(
                &instance(level),
                VoxelLod {
                    level,
                    shared: shared.clone(),
                },
            );
            // The box is only what the predicate is evaluated over; the
            // predicate is what decides.
            let at = shared.clone();
            let filter: CoordFilter = Arc::new(move |coord: IVec3| {
                let key = ChunkKey::in_world(at.world, level, coord);
                if !resident_clamped(&at.config, at.anchor.load().as_dvec3(), key) {
                    return false;
                }
                // Past the ops horizon the generator is the whole story,
                // so a chunk it cannot put a surface in need not exist at
                // all. Nearer than that, planning may still carve one, and
                // `create` decides with the chunk's actual ops in hand.
                (key.edge_m() as f32) <= crate::planning::OPS_HORIZON_EDGE_M
                    || can_hold_surface(&at.generator, key)
            });
            let dep = TopDep::new(&instance(level), level_span(&config, level))
                .with_filter(filter.clone());
            if level < config.max_level {
                // The top ring is the world's own extent; a volume adds
                // nothing there. Zero reach is a level the volume cannot
                // refine at all — the scale cap makes that most of them.
                for v in config.detail.iter() {
                    let reach = 2.0 * detail_reach_m(v, level);
                    if reach == 0.0 {
                        continue;
                    }
                    let span = ((v.max - v.min) + DVec3::splat(reach)).ceil().as_ivec3();
                    volumes.push((
                        TopDep::new(&instance(level), span).with_filter(filter.clone()),
                        ((v.min + v.max) * 0.5).as_ivec3(),
                    ));
                }
            }
            if (ChunkKey::in_world(world, level, IVec3::ZERO).edge_m() as f32)
                <= crate::planning::OPS_HORIZON_EDGE_M
            {
                gated.push(dep);
            } else {
                tops.push(dep);
            }
        }
        // Descending so far; the level decides whether the gated half
        // wants the other order.
        if config.gated_finest_first {
            gated.reverse();
        }
        tops.extend(gated);
        let camera_tops = tops.len();
        let volume_tops: Vec<IVec3> = volumes.iter().map(|(_, center)| *center).collect();
        tops.extend(volumes.into_iter().map(|(dep, _)| dep));
        // `before` freezes the focus this pass works from. `between` runs
        // after every ensure and before any release — the only moment when
        // the new configuration is resident and the old one is still there
        // to be drawn while its replacements are built.
        let before: voxel_layers::BetweenPasses = {
            let shared = shared.clone();
            Arc::new(move |_| shared.anchor.store(shared.requested.load()))
        };
        let between: voxel_layers::BetweenPasses = {
            let shared = shared.clone();
            Arc::new(move |_| {
                restale(&shared);
                settle_builds(&shared);
                refresh_masks(&shared);
                // One reveal per pass: everything this pass generated
                // becomes visible together, with every mask already
                // agreeing, and only then is the old configuration
                // released.
                let pending: Vec<ChunkKey> =
                    shared.state.lock().unwrap().pending.drain(..).collect();
                for key in pending {
                    shared.chunks.commit(key);
                }
            })
        };
        let runtime = LayerRuntime::start_with(Arc::new(graph), tops, Some(before), Some(between));
        Self {
            runtime,
            shared,
            published: None,
            camera_tops,
            volume_tops,
        }
    }

    /// Re-centre the field, if the camera has moved far enough to matter.
    fn follow(&mut self, camera: DVec3) {
        if self
            .published
            .is_some_and(|a| camera.distance(a) < ANCHOR_STEP)
        {
            return;
        }
        // One pass runs against one focus. Moving it mid-pass would let
        // levels processed early and late disagree about where the field
        // is centred, which is a seam nobody owns; the next frame
        // publishes instead.
        if self.runtime.is_generating() {
            return;
        }
        self.published = Some(camera);
        // Published, not applied: the next pass snapshots it at its head,
        // and everything that pass does works from that one value.
        self.shared.requested.store(camera.as_ivec3());
        for i in 0..self.camera_tops {
            self.runtime.top(i).set_focus(camera.as_ivec3());
        }
        for (i, center) in self.volume_tops.iter().enumerate() {
            let top = self.runtime.top(self.camera_tops + i);
            // The first call activates the pinned dependency; after that
            // it is a no-op, and `touch` is what carries the anchor move.
            top.set_focus(*center);
            top.touch();
        }
    }

    /// Resident chunks. Residency is exactly the shown set — this is its
    /// size, published as it changes so asking never waits on the pass
    /// that is changing it.
    pub fn resident(&self) -> usize {
        self.shared.resident.load(Ordering::Relaxed)
    }

    pub fn stalled(&self) -> usize {
        self.shared.stalled.load(Ordering::Relaxed)
    }

    fn prune_counts(&self) -> (usize, usize, usize) {
        (
            self.shared.pruned.load(Ordering::Relaxed),
            self.shared.unpruned.load(Ordering::Relaxed),
            self.shared.had_ops.load(Ordering::Relaxed),
        )
    }

    pub fn is_generating(&self) -> bool {
        self.runtime.is_generating()
    }

    /// Residency has caught up with the focus. Says nothing about the
    /// render pipeline, which is the other half of settled.
    pub fn is_idle(&self) -> bool {
        self.runtime.is_idle()
    }
}

/// Rebuild, in place, every resident chunk whose mask the field has
/// changed, and swap them all at once.
///
/// A chunk's content is a pure function of its coordinate; its SEAM is
/// not, because it is meshed against the LOD of its neighbors and that
/// follows the camera. So this is not a lifecycle event — the chunk is the
/// same chunk, and only its mesh is stale. Going through the graph's
/// `invalidate` would make it one: destroy frees the slab, so every
/// refresh would delete the chunk and rebuild it, which is a hole for the
/// length of a GPU round-trip. (Measured: it is exactly the ragged
/// contour the coverage eval reports along a LOD boundary.)
///
/// Instead the pipeline regenerates in place while the old mesh keeps
/// drawing, and every rebuilt mesh is HELD until all of them are ready.
/// Half a swapped set is a crack, so the set swaps together.
/// Rebuild chunks that are already SHOWN, without a hole.
///
/// `hold` keeps the drawn mesh in place until its replacement is built,
/// and `commit` is the swap — ready-before-swap for a chunk replacing
/// ITSELF, which is the same rule the field follows for a chunk replacing
/// its neighbour. `LayerGraph::invalidate` cannot do this: it frees the
/// mesh first, so the hole lasts as long as the round trip.
fn rebuild_shown(shared: &LodShared, want: Vec<(ChunkKey, ShownChunk)>) {
    if want.is_empty() {
        return;
    }
    let mut batch = crate::chunkgen::ChunkBatch::default();
    for (key, w) in &want {
        batch.add(&shared.chunks, *key, w.mask, true, w.ops.clone());
    }
    // The lock is NOT held across the wait: a create finishing meanwhile
    // has to be able to record itself.
    let outcome = batch.wait(CREATE_TIMEOUT);
    let want: HashMap<ChunkKey, ShownChunk> = want.into_iter().collect();
    let mut state = shared.state.lock().unwrap();
    for key in outcome.built {
        shared.chunks.commit(key);
        state.shown.insert(key, want[&key].clone());
    }
    shared.publish_resident(&state);
    // A stalled chunk keeps what it had, so the next scan tries again.
    shared
        .stalled
        .fetch_add(outcome.stalled.len(), Ordering::Relaxed);
}

/// Rebuild the chunks an authored edit changed, and only those.
///
/// Every shown chunk within reach of the edit is fingerprinted against
/// the world as it was and as it is; the ones whose number moved are
/// rebuilt from the ops they now have.
fn restale(shared: &LodShared) {
    let Some(edit) = shared.restale.lock().unwrap().take() else {
        return;
    };
    // Where the edit could possibly have landed. Fingerprinting is not
    // free — it bounds the region axes over every chunk's box — and an
    // edit leaves almost every chunk alone, so the cheap question is
    // asked first. Sound because only the AUTHORED ops differ here
    // (`only_authored_moved` guarantees it): a chunk this box misses sees
    // the same two op sets either way, so its two prints cannot differ.
    let Some(moved) = crate::fingerprint::touched(&edit.was_placed, &edit.now_placed) else {
        return; // the lists differ in nothing that occupies space
    };
    let near: Vec<(ChunkKey, u32)> = {
        let state = shared.state.lock().unwrap();
        state
            .shown
            .iter()
            .filter(|(key, _)| crate::fingerprint::read_box(**key).touches(moved))
            .map(|(key, shown)| (*key, shown.mask))
            .collect()
    };
    // Out of the lock: asking for a chunk's ops is a spatial query over
    // the planning graph, not a lookup.
    let want: Vec<(ChunkKey, ShownChunk)> = near
        .into_iter()
        .filter(|(key, _)| {
            // The generator this world is streaming — an authored edit
            // cannot have changed it, which is why this path exists.
            let ops = shared.generator.ops();
            let was = crate::fingerprint::of(*key, edit.seed, ops, &edit.was_placed);
            let now = crate::fingerprint::of(*key, edit.seed, ops, &edit.now_placed);
            was != now
        })
        .map(|(key, mask)| {
            let ops = shared.chunks.ops_for(key);
            (key, ShownChunk { mask, ops })
        })
        .collect();
    info!("level edit: rebuilt {} chunks", want.len());
    rebuild_shown(shared, want);
}

/// Wait for everything this pass asked for, once.
///
/// A stalled chunk is NOT recorded as carrying its mask, or the refresh
/// scan would see it as already correct and never retry it, turning a
/// transient slab stall into a permanent crack.
fn settle_builds(shared: &LodShared) {
    // The lock is NOT held across the wait: this runs between ensure and
    // release, but a create racing in from a nested ensure still has to be
    // able to record itself.
    let (mut asked, building) = {
        let mut state = shared.state.lock().unwrap();
        (
            std::mem::take(&mut state.asked),
            std::mem::take(&mut state.building),
        )
    };
    if asked.is_empty() {
        return;
    }
    let outcome = asked.wait(CREATE_TIMEOUT);
    let mut state = shared.state.lock().unwrap();
    for key in outcome.built {
        if let Some(shown) = building.get(&key) {
            state.shown.insert(key, shown.clone());
            state.pending.push(key);
        }
    }
    shared.publish_resident(&state);
    drop(state);
    if !outcome.stalled.is_empty() {
        let n = shared
            .stalled
            .fetch_add(outcome.stalled.len(), Ordering::Relaxed);
        if n < 8 {
            warn!(
                "lod: {} chunks never became drawable — slabs are full",
                outcome.stalled.len()
            );
        }
    }
}

fn refresh_masks(shared: &LodShared) {
    let at = shared.anchor.load().as_dvec3();
    let stale: Vec<(ChunkKey, ShownChunk)> = {
        let state = shared.state.lock().unwrap();
        state
            .shown
            .iter()
            .filter_map(|(key, shown)| {
                if !resident_clamped(&shared.config, at, *key) {
                    return None; // about to be released
                }
                let want = seam_mask_at(&shared.config, at, *key);
                (want != shown.mask).then(|| {
                    (
                        *key,
                        ShownChunk {
                            mask: want,
                            ops: shown.ops.clone(),
                        },
                    )
                })
            })
            .collect()
    };
    if stale.is_empty() {
        return;
    }
    if std::env::var_os("VOXEL_LOG_REFRESH").is_some() {
        let shown = shared.state.lock().unwrap().shown.len();
        info!("REFRESH {} of {shown} chunks", stale.len());
    }
    rebuild_shown(shared, stale);
}

/// Build a LOD graph per registered world once configuration exists, and
/// rebuild them when a hot reload changes generation.
fn build_lod_layers(
    mut layers: ResMut<LodLayers>,
    worlds: Res<crate::Worlds>,
    chunks: Res<ChunkGen>,
    mut rebuild: ResMut<StreamingRebuild>,
) {
    if rebuild.0 && !layers.is_empty() {
        rebuild.0 = false;
        // Dropping a graph destroys every chunk it owns, which frees every
        // slab: a rebuild needs no separate teardown pass.
        layers.worlds.clear();
        return;
    }
    rebuild.0 = false;
    // Any REGISTERED world without a graph gets one, rather than building
    // the set once: a world can arrive long after startup — opening a
    // portal loads the far level on the spot — and a one-shot build would
    // register it, stream nothing, and show an empty opening.
    for world in worlds.iter() {
        if layers.worlds.iter().any(|w| w.shared.world == world.id) {
            continue;
        }
        layers.worlds.push(WorldLod::new(
            world.id,
            world.config.clone(),
            chunks.clone(),
            world.generator.clone(),
        ));
    }
}

/// Publish the camera to every level, and report.
#[allow(clippy::too_many_arguments)]
fn follow_lod_focus(
    mut layers: ResMut<LodLayers>,
    focus: Res<WorldFocus>,
    mut probe: ResMut<StreamProbe>,
    worlds: Res<crate::Worlds>,
    sources: crate::StreamSourceQuery,
    stats: Res<voxel_render::SharedRenderStats>,
    time: Res<Time>,
    mut settling: Local<f32>,
) {
    if layers.is_empty() {
        return;
    }
    if let Ok(source) = sources.single() {
        layers.follow(source.translation().as_dvec3(), &focus);
    }
    probe.resident = layers.resident();
    probe.generating = layers.is_generating();
    probe.stalled = layers.stalled();
    let (pruned, unpruned, had_ops) = layers.prune_counts();
    probe.pruned = pruned;
    probe.unpruned = unpruned;
    probe.unpruned_with_ops = had_ops;
    // Summed over worlds: a miss in ANY of them is a consumer reading
    // outside what a top dependency covers, and the number has to stay 0
    // whichever world it happened in.
    probe.reads_missed = worlds.iter().map(|w| w.query.reads_missed()).sum();

    // Settled: residency agrees with the focus AND the pipeline has
    // drained AND the planners have caught up. Any of the three alone
    // lies — the graph is idle the moment it stops asking, while the
    // chunks it asked for are still meshing.
    //
    // The planners are in here because a chunk only waits for the
    // planning IT stands in, through the ops gate. Nothing waits for a
    // population: no chunk depends on a prop, so scatter layers could
    // still be generating when the terrain had drained, and `settled`
    // said the world was ready with none of its props in it. The
    // megastructure reported 1.22 s with `rubble` created ZERO times and
    // kept building for another second.
    let awaiting = stats.0.lock().map_or(0, |s| s.awaiting);
    let planned = worlds.iter().all(|w| w.query.is_idle());
    let settled = layers.is_idle() && awaiting == 0 && !layers.is_generating() && planned;
    if settled {
        if *settling > 0.0 {
            probe.last_settle_s = *settling;
            probe.worst_settle_s = probe.worst_settle_s.max(*settling);
            *settling = 0.0;
        }
    } else {
        *settling += time.delta_secs();
    }
    probe.settled = settled;
    probe.settling_s = *settling;
}

/// Installs the LOD-as-layers streaming path.
pub struct LodLayersPlugin;

impl Plugin for LodLayersPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LodLayers>()
            .init_resource::<WorldFocus>()
            .add_systems(
                Update,
                (build_lod_layers, follow_lod_focus)
                    .chain()
                    .in_set(crate::WorldFocusSet::Follow),
            );
    }
}
