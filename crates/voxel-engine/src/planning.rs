//! The engine↔host planning contract.
//!
//! Planning layers are the host's code. LayerProcGen's framework provides
//! dependency management, threading and spatial organisation, and nothing
//! else — the concrete layers (terrain features, paths, structures,
//! vegetation) belong to the game. `voxel-layers` is that framework here;
//! a host builds its layers on it and hands the engine one
//! [`WorldPlanner`], which the engine asks for the ops shaping each chunk.
//!
//! The engine keeps only the policy it must own to preserve seams: where
//! the ops horizon sits, how far the density apron reaches, and that the
//! whole query is per chunk rather than per op. Everything about what the
//! ops *mean* is on the host's side of this trait.

use std::sync::Arc;

use bevy::{
    math::{Vec2, Vec3},
    prelude::*,
};
use voxel_core::{csg::CsgOp, ChunkKey};

pub use voxel_core::patch::{Marker, PatchSet, RibbonSeg};

/// Chunks coarser than this never receive planning ops at all: structures
/// are subpixel there and haze covers the hard-cut ring. Both the engine's
/// per-chunk query and a planner's pre-generation pass read it, so the
/// resident planning set matches exactly what can render.
pub const OPS_HORIZON_EDGE_M: f32 = 1000.0;

/// A host's planning stack, seen from the engine.
///
/// Everything except [`ops_in`](WorldPlanner::ops_in) has a default: a
/// planner that only carves geometry implements one method. Queries are
/// called from async generation tasks and must be `Send + Sync`.
pub trait WorldPlanner: Send + Sync + 'static {
    /// All ops overlapping the box, as served to a chunk of the given
    /// edge. A planner that gates features by chunk edge MUST apply the
    /// gate per chunk, never per op: a per-op gate desynchronizes
    /// neighboring LODs and cracks every seam.
    fn ops_in(&self, min: Vec3, max: Vec3, chunk_edge_m: f32) -> Vec<CsgOp>;

    /// Pre-generate the planning every chunk in `keys` is about to query,
    /// before anything reads it. Generation is dependency-driven: without
    /// this, the first read generates on whatever thread reads it.
    /// `voxctl status` → `stream.read_generated` staying ~0 is the check.
    fn prepare(&self, _keys: &[ChunkKey]) {}

    /// Segments props must keep off (roadbeds, ribbon beds) in the xz box.
    fn clearance_in(&self, _min: Vec2, _max: Vec2) -> Vec<[Vec2; 2]> {
        Vec::new()
    }

    /// Ribbon surface segments overlapping the xz box.
    fn ribbons_in(&self, _min: Vec2, _max: Vec2) -> Vec<RibbonSeg> {
        Vec::new()
    }

    /// Markers overlapping the xz box, optionally of one kind.
    fn markers_in(&self, _min: Vec2, _max: Vec2, _kind: Option<&str>) -> Vec<Marker> {
        Vec::new()
    }

    /// Blended weights at a point for a named biome field: (name, weight).
    /// Empty if the planner has no such field.
    fn biomes_at(&self, _field: &str, _p: Vec2) -> Vec<(String, f32)> {
        Vec::new()
    }

    /// The layer managers backing this planner, so the engine can roll
    /// their caches with the camera and report their stats.
    fn layer_managers(&self) -> Vec<Arc<voxel_layers::LayerManager>> {
        Vec::new()
    }
}

/// A source of ops for a world-space box, independent of any layer stack
/// (authored placements, editor brushes).
pub type OpsSource = Arc<dyn Fn(Vec3, Vec3) -> Vec<CsgOp> + Send + Sync>;

/// The one facade over everything the world knows on the CPU: the
/// generator (heights, fields, shadows) and the host's planning. Every
/// consumer — the chunk ops provider, prop scattering, debug overlays,
/// gameplay — reads through this, which is what keeps the engine from
/// growing a side channel per feature.
#[derive(Resource, Clone, Default)]
pub struct WorldQuery {
    planner: Option<Arc<dyn WorldPlanner>>,
    sources: Vec<OpsSource>,
    generator: Arc<voxel_worldgen::Generator>,
}

impl WorldQuery {
    pub fn new(generator: Arc<voxel_worldgen::Generator>) -> Self {
        Self {
            planner: None,
            sources: Vec::new(),
            generator,
        }
    }

    /// Install the host's planning stack.
    pub fn with_planner(mut self, planner: Arc<dyn WorldPlanner>) -> Self {
        self.planner = Some(planner);
        self
    }

    /// Add a stack-independent op source, served after the planner's.
    pub fn with_source(mut self, source: OpsSource) -> Self {
        self.sources.push(source);
        self
    }

    /// True when nothing can produce ops — the streamer then skips the
    /// provider entirely.
    pub fn is_empty(&self) -> bool {
        self.planner.is_none() && self.sources.is_empty()
    }

    /// The world's generator: heights, slopes, fields, shadows. Hosts
    /// sample the world through this.
    pub fn generator(&self) -> &Arc<voxel_worldgen::Generator> {
        &self.generator
    }

    /// All ops overlapping the box, as served to a chunk of the given edge.
    pub fn ops_in(&self, min: Vec3, max: Vec3, chunk_edge_m: f32) -> Vec<CsgOp> {
        let mut out = Vec::new();
        for source in &self.sources {
            out.extend(source(min, max));
        }
        if let Some(planner) = &self.planner {
            out.extend(planner.ops_in(min, max, chunk_edge_m));
        }
        out
    }

    /// The ops shaping one chunk. This is engine policy, not the host's:
    /// past the ops horizon the SDF genuinely loses the ops (a hard-cut
    /// seam by doctrine), and the query is padded by the density apron —
    /// samples extend 2 voxels below and 3 above the 32-cell core, so an
    /// op grazing only the apron still shapes this chunk. Culling it would
    /// desynchronize the seam with the neighbor that keeps it.
    pub fn chunk_ops(&self, key: ChunkKey) -> Vec<CsgOp> {
        let edge = key.edge_m() as f32;
        if edge > OPS_HORIZON_EDGE_M {
            return Vec::new();
        }
        let pad = 4.0 * key.voxel_size_m() as f32;
        let min = key.min_corner_m().as_vec3() - Vec3::splat(pad);
        let max = key.min_corner_m().as_vec3() + Vec3::splat(edge + pad);
        self.ops_in(min, max, edge)
    }

    /// Pre-generate the planning `keys` will query. See
    /// [`WorldPlanner::prepare`].
    pub fn prepare(&self, keys: &[ChunkKey]) {
        if std::env::var_os("VOXEL_NO_PREPARE").is_some() {
            return; // A/B kill switch: fall back to read-driven generation
        }
        if let Some(planner) = &self.planner {
            planner.prepare(keys);
        }
    }

    /// Cut ops (carved voids) overlapping the box: spawners consult this
    /// so props never seat on heightfield ground that a cave mouth or
    /// doorway has carved away.
    pub fn cuts_in(&self, min: Vec3, max: Vec3) -> Vec<CsgOp> {
        let mut ops = self.ops_in(min, max, 0.0);
        ops.retain(|op| op.kind & 1 == 1);
        ops
    }

    pub fn clearance_in(&self, min: Vec2, max: Vec2) -> Vec<[Vec2; 2]> {
        self.planner
            .as_ref()
            .map_or_else(Vec::new, |p| p.clearance_in(min, max))
    }

    pub fn ribbons_in(&self, min: Vec2, max: Vec2) -> Vec<RibbonSeg> {
        self.planner
            .as_ref()
            .map_or_else(Vec::new, |p| p.ribbons_in(min, max))
    }

    pub fn markers_in(&self, min: Vec2, max: Vec2, kind: Option<&str>) -> Vec<Marker> {
        self.planner
            .as_ref()
            .map_or_else(Vec::new, |p| p.markers_in(min, max, kind))
    }

    pub fn biomes_at(&self, field: &str, p: Vec2) -> Vec<(String, f32)> {
        self.planner
            .as_ref()
            .map_or_else(Vec::new, |p2| p2.biomes_at(field, p))
    }

    /// The layer managers behind this world, for cache rolling and stats.
    pub fn layer_managers(&self) -> Vec<Arc<voxel_layers::LayerManager>> {
        self.planner
            .as_ref()
            .map_or_else(Vec::new, |p| p.layer_managers())
    }
}

/// Layer managers backing the world query, resolved once so the engine can
/// roll their caches with the camera (they grow unboundedly otherwise) and
/// report cache stats in the HUD.
#[derive(Resource, Default, Clone)]
pub struct PlanningLayers(pub Vec<Arc<voxel_layers::LayerManager>>);

/// The engine's top dependency: pre-generate planning for a batch of
/// chunk keys in the async planning task, so the per-chunk ops queries
/// that follow are cache reads.
pub fn ops_prepare(world: &WorldQuery) -> crate::streaming::ChunkOpsPrepare {
    let world = world.clone();
    crate::streaming::ChunkOpsPrepare(Some(Arc::new(move |keys: &[ChunkKey]| {
        world.prepare(keys);
    })))
}

/// The per-chunk ops provider for a world query.
pub fn ops_provider(world: &WorldQuery) -> crate::streaming::ChunkOpsProvider {
    if world.is_empty() {
        return crate::streaming::ChunkOpsProvider(None);
    }
    let world = world.clone();
    crate::streaming::ChunkOpsProvider(Some(Arc::new(move |key: ChunkKey| world.chunk_ops(key))))
}

/// Rolling eviction for the planning-layer caches: every few seconds, drop
/// cached layer chunks far outside the region any chunk request can reach.
/// Everything is regenerable, so the only cost of evicting too eagerly is
/// regeneration.
pub fn roll_planning_caches(
    layers: Res<PlanningLayers>,
    time: Res<Time>,
    mut last: Local<f32>,
    sources: crate::StreamSourceQuery,
) {
    if time.elapsed_secs() - *last < 5.0 {
        return;
    }
    *last = time.elapsed_secs();
    let Ok(source) = sources.single() else {
        return; // no streaming source tagged yet
    };
    let p = source.translation();
    const KEEP_M: i32 = 8_000;
    let keep = voxel_layers::IAabb::new(
        bevy::math::IVec3::new(p.x as i32 - KEEP_M, i32::MIN / 2, p.z as i32 - KEEP_M),
        bevy::math::IVec3::new(p.x as i32 + KEEP_M, i32::MAX / 2, p.z as i32 + KEEP_M),
    );
    for mgr in &layers.0 {
        mgr.evict_outside(keep);
    }
}
