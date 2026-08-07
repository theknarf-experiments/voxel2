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
    level_span, resident_clamped, seam_mask_at, LodConfig, StreamProbe, StreamingRebuild,
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

/// Layer-thread workers. Each one spends its time waiting on the GPU
/// rather than computing, so this is a count of chunks in flight, not of
/// cores in use — sized to keep the pipeline's per-frame budget fed.
const LOD_WORKERS: usize = 48;

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
    chunks: ChunkGen,
    config: LodConfig,
    /// Where the camera has asked the field to be centred, and where THIS
    /// pass froze it. The pass reads the dependencies' focuses one at a
    /// time, so without a snapshot a publish landing mid-read leaves two
    /// levels centred a step apart — and a seam between them that neither
    /// side owns.
    requested: Anchor,
    anchor: Anchor,
    state: Mutex<LodState>,
    /// Chunks the pipeline could not place. A hole for as long as it lasts.
    stalled: AtomicUsize,
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

    fn create(&mut self, ctx: &ChunkCtx<'_, VoxelLod>, _level: u32) {
        let shared = &ctx.layer().shared;
        let key = ChunkKey::new(ctx.layer().level, ctx.coord());
        let mask = seam_mask_at(&shared.config, shared.anchor.load().as_dvec3(), key);
        let ops = shared.chunks.ops_for(key);
        // Hidden until the pass swaps: shown the moment it is drawable, a
        // new chunk would be drawn against neighbors whose masks this pass
        // has not refreshed yet — a hairline along the boundary it just
        // moved.
        let mut batch = shared.chunks.batch();
        batch.add(key, mask, false, ops.clone());
        // The chunk is owned either way — `destroy` still has to free it —
        // but a stalled chunk is NOT recorded as carrying its mask, or the
        // refresh scan would see it as already correct and never retry it,
        // turning a transient slab stall into a permanent crack.
        self.key = Some(key);
        if batch.wait(CREATE_TIMEOUT).stalled.is_empty() {
            let mut state = shared.state.lock().unwrap();
            state.shown.insert(key, ShownChunk { mask, ops });
            state.pending.push(key);
        } else {
            let n = shared.stalled.fetch_add(1, Ordering::Relaxed);
            if n < 8 {
                warn!("lod: {key:?} never became drawable — slabs are full");
            }
        }
    }

    fn destroy(&mut self, ctx: &ChunkCtx<'_, VoxelLod>, _level: u32) {
        let shared = &ctx.layer().shared;
        if let Some(key) = self.key.take() {
            shared.state.lock().unwrap().shown.remove(&key);
            shared.chunks.free(key);
        }
    }
}

/// The running LOD graph: one layer per level, and the thread that keeps
/// their top dependencies satisfied.
#[derive(Resource)]
pub struct LodLayers {
    runtime: LayerRuntime,
    shared: Arc<LodShared>,
    /// Sticky anchor: the field is only re-centred when the camera has
    /// moved this far. The quantization IS the hysteresis.
    published: Option<DVec3>,
}

/// How far the camera moves before the field is re-centred.
pub const ANCHOR_STEP: f64 = 48.0;

impl LodLayers {
    fn new(config: LodConfig, chunks: ChunkGen) -> Self {
        let shared = Arc::new(LodShared {
            chunks,
            requested: Anchor::default(),
            anchor: Anchor::default(),
            state: Mutex::new(LodState::default()),
            stalled: AtomicUsize::new(0),
            config: config.clone(),
        });
        let mut graph = LayerGraph::new(0).with_threads(LOD_WORKERS);
        let mut tops = Vec::new();
        for level in 0..=config.max_level {
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
                resident_clamped(
                    &at.config,
                    at.anchor.load().as_dvec3(),
                    ChunkKey::new(level, coord),
                )
            });
            tops.push(
                TopDep::at_level(&instance(level), 0, level_span(&config, level))
                    .with_filter(filter),
            );
        }
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
        let runtime =
            LayerRuntime::start_with(Arc::new(graph), tops, Some(before), Some(between));
        Self {
            runtime,
            shared,
            published: None,
        }
    }

    /// Re-centre the field, if the camera has moved far enough to matter.
    fn follow(&mut self, camera: DVec3) -> bool {
        if self.published.is_some_and(|a| camera.distance(a) < ANCHOR_STEP) {
            return false;
        }
        // One pass runs against one focus. Moving it mid-pass would let
        // levels processed early and late disagree about where the field
        // is centred, which is a seam nobody owns; the next frame
        // publishes instead.
        if self.runtime.is_generating() {
            return false;
        }
        self.published = Some(camera);
        // Published, not applied: the next pass snapshots it at its head,
        // and everything that pass does works from that one value.
        self.shared.requested.store(camera.as_ivec3());
        for i in 0..self.runtime.tops() {
            self.runtime.top(i).set_focus(camera.as_ivec3());
        }
        true
    }

    /// Resident chunks. Residency is exactly the shown set.
    pub fn resident(&self) -> usize {
        self.shared.state.lock().unwrap().shown.len()
    }

    pub fn stalled(&self) -> usize {
        self.shared.stalled.load(Ordering::Relaxed)
    }

    pub fn is_generating(&self) -> bool {
        self.runtime.is_generating()
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
    let mut batch = shared.chunks.batch();
    for (key, want) in &stale {
        batch.add(*key, want.mask, true, want.ops.clone());
    }
    // The lock is NOT held across the wait: a create finishing meanwhile
    // has to be able to record itself.
    let outcome = batch.wait(CREATE_TIMEOUT);
    let rebuilt: HashMap<ChunkKey, ShownChunk> = stale.into_iter().collect();
    let mut state = shared.state.lock().unwrap();
    for key in outcome.built {
        shared.chunks.commit(key);
        state.shown.insert(key, rebuilt[&key].clone());
    }
    // A stalled chunk keeps its OLD mask, so the next scan tries again.
    shared
        .stalled
        .fetch_add(outcome.stalled.len(), Ordering::Relaxed);
}

/// Build the LOD graph once the level's configuration exists, and rebuild
/// it when a hot reload changes generation.
fn build_lod_layers(
    mut commands: Commands,
    existing: Option<ResMut<LodLayers>>,
    config: Res<LodConfig>,
    chunks: Res<ChunkGen>,
    mut field: ResMut<voxel_render::FieldParams>,
    mut rebuild: ResMut<StreamingRebuild>,
) {
    let stale = existing.is_some() && rebuild.0;
    if stale {
        rebuild.0 = false;
        // Dropping the graph destroys every chunk, which frees every
        // slab: a rebuild needs no separate teardown pass.
        commands.remove_resource::<LodLayers>();
        return;
    }
    if existing.is_none() {
        rebuild.0 = false;
        // The density band's scale is a constant of the configuration, so
        // it is set with the graph rather than on every anchor move.
        field.dist_scale = (config.split_k * 32.0) as f32;
        field.max_vs = (1u32 << config.max_level) as f32;
        commands.insert_resource(LodLayers::new(config.clone(), chunks.clone()));
    }
}

/// Publish the camera to every level, and report.
fn follow_lod_focus(
    layers: Option<ResMut<LodLayers>>,
    mut field: ResMut<voxel_render::FieldParams>,
    mut probe: ResMut<StreamProbe>,
    world: Res<crate::planning::WorldQuery>,
    sources: crate::StreamSourceQuery,
) {
    let Some(mut layers) = layers else {
        return;
    };
    if let Ok(source) = sources.single() {
        let camera = source.translation().as_dvec3();
        if layers.follow(camera) {
            field.anchor = camera.as_vec3();
        }
    }
    probe.resident = layers.resident();
    probe.generating = layers.is_generating();
    probe.stalled = layers.stalled();
    probe.reads_missed = world.reads_missed();
}

/// Installs the LOD-as-layers streaming path.
pub struct LodLayersPlugin;

impl Plugin for LodLayersPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (build_lod_layers, follow_lod_focus).chain());
    }
}
