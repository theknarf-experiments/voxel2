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

    /// Publish where the streaming source is. A planner decides for
    /// itself what follows it — how far each of its layers stays
    /// resident is a declaration it owns, not a region the engine
    /// computes on its behalf.
    fn set_focus(&self, _focus: IVec3) {}

    /// Residency and health, for the HUD and the eval. Builds a snapshot
    /// of every layer — for a display, not for a frame.
    fn stats(&self) -> PlanningStats {
        PlanningStats::default()
    }

    /// Reads that found no resident chunk. Separate from [`Self::stats`]
    /// because the engine watches it every frame and the snapshot costs a
    /// read lock and a `String` per layer instance.
    fn reads_missed(&self) -> usize {
        0
    }

    /// Block until residency has caught up with the focus. For loading
    /// screens and tests — never call it from a frame.
    fn wait_idle(&self) {}

    /// The host's own per-world state, if it has any. The engine never
    /// looks inside; this is how a host's systems read what its own layers
    /// published, without the engine learning what a river is.
    fn host_ctx(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        None
    }

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

    /// Names of the biome fields this planner answers `biomes_at` for.
    fn biome_fields(&self) -> Vec<String> {
        Vec::new()
    }

    /// Blended weights at a point for a named biome field: (name, weight).
    /// Empty if the planner has no such field.
    fn biomes_at(&self, _field: &str, _p: Vec2) -> Vec<(String, f32)> {
        Vec::new()
    }

}

/// What a planner reports about itself.
#[derive(Debug, Default, Clone)]
pub struct PlanningStats {
    /// Chunks currently held resident. With a dependency graph this is
    /// the exact transitive closure of the top dependencies, so it should
    /// be flat while the camera orbits — a sawtooth means something is
    /// releasing and regenerating.
    pub resident_chunks: usize,
    /// Reads that found no resident chunk. Anything but 0 means a
    /// consumer's working set is not covered by a top dependency, or a
    /// layer under-declared its padding.
    pub reads_missed: usize,
    /// A generation pass is running.
    pub generating: bool,
    /// Per-layer residency and cost, dearest first.
    pub layers: Vec<voxel_layers::LayerStats>,
}

/// A source of ops for a world-space box, independent of any layer stack
/// (authored placements, editor brushes).
pub type OpsSource = Arc<dyn Fn(Vec3, Vec3) -> Vec<CsgOp> + Send + Sync>;

/// How a host supplies its planning. The engine calls this at startup and
/// again on every hot reload that changes generation, so a host's layers
/// rebuild with the world instead of going stale.
///
/// It builds rather than being built because the generator does not exist
/// until the engine has read the level, and layers need it.
pub trait HostPlanning: Send + Sync + 'static {
    /// Check the level's planning data before anything is built.
    /// Authoring errors surface HERE with a message, never as a panic
    /// mid-generation: boot fails loudly on an invalid shipped level,
    /// hot reload warns and keeps the running world.
    fn validate(&self, _level: &crate::level::LevelDef) -> Result<(), String> {
        Ok(())
    }

    /// Build this level's planner, or `None` if it declares no layers.
    fn build(
        &self,
        level: &crate::level::LevelDef,
        seed: u64,
        generator: &Arc<voxel_worldgen::Generator>,
    ) -> Option<Arc<dyn WorldPlanner>>;
}

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

    /// Publish where the streaming source is, so the planner's top
    /// dependencies can follow it.
    pub fn set_focus(&self, focus: IVec3) {
        if let Some(planner) = &self.planner {
            planner.set_focus(focus);
        }
    }

    pub fn stats(&self) -> PlanningStats {
        self.planner.as_ref().map_or_else(PlanningStats::default, |p| p.stats())
    }

    /// See [`WorldPlanner::reads_missed`].
    pub fn reads_missed(&self) -> usize {
        self.planner.as_ref().map_or(0, |p| p.reads_missed())
    }

    /// Hold while making INTROSPECTION reads — a debug overlay asking what
    /// the world has, rather than a consumer asserting its working set is
    /// covered. Absent chunks are not counted against `reads_missed`.
    pub fn peek(&self) -> voxel_layers::Peek {
        voxel_layers::peek()
    }

    /// The host's per-world layer state, downcast to `C`.
    pub fn host_ctx<C: 'static>(&self) -> Option<&C> {
        self.planner
            .as_ref()
            .and_then(|p| p.host_ctx())
            .and_then(|c| c.downcast_ref::<C>())
    }

    /// Block until residency has caught up. See [`WorldPlanner::wait_idle`].
    pub fn wait_idle(&self) {
        if let Some(planner) = &self.planner {
            planner.wait_idle();
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

    pub fn biome_fields(&self) -> Vec<String> {
        self.planner
            .as_ref()
            .map_or_else(Vec::new, |p| p.biome_fields())
    }

    pub fn biomes_at(&self, field: &str, p: Vec2) -> Vec<(String, f32)> {
        self.planner
            .as_ref()
            .map_or_else(Vec::new, |p2| p2.biomes_at(field, p))
    }

}

/// The per-chunk ops provider for a world query.
///
/// It waits for planning to be resident before reading. Until the voxel
/// chunks are layers themselves — depending on the planning layers, so the
/// framework orders this for us — the streamer has no declared dependency
/// on planning and would otherwise read an empty graph on the frame the
/// world starts, baking featureless chunks that nothing ever revisits.
///
/// This runs in the async planning task, never on a frame, and the wait is
/// one atomic load once residency has caught up.
pub fn ops_provider(world: &WorldQuery) -> crate::chunkgen::ChunkOpsProvider {
    if world.is_empty() {
        return crate::chunkgen::ChunkOpsProvider(None);
    }
    let world = world.clone();
    crate::chunkgen::ChunkOpsProvider(Some(Arc::new(move |key: ChunkKey| {
        // Past the ops horizon `chunk_ops` returns nothing whatever
        // planning says, so waiting for residency is waiting to be told
        // the answer is empty — and on a cold start that wait IS the cold
        // start. The coarse levels are most of the world's area, so they
        // can be streaming while the planners are still running.
        if key.edge_m() as f32 > OPS_HORIZON_EDGE_M {
            return Vec::new();
        }
        // This planner belongs to the level `LevelPlugin` loaded, which is
        // world 0. Serving it to every world asked world 0's planning
        // graph about coordinates in another world, where nothing is
        // resident: 40,474 `reads_missed` the moment a portal opened, and
        // world 0's roads and ruins carved into the far world. A world
        // with its own planning will bring its own provider.
        if key.world != 0 {
            return Vec::new();
        }
        world.wait_idle();
        world.chunk_ops(key)
    })))
}

/// Publish the streaming source's position to the world's planner every
/// frame. The planner's own quantization decides whether that is a change
/// worth acting on, so this is a store, never a wait.
pub fn follow_stream_source(world: Res<WorldQuery>, sources: crate::StreamSourceQuery) {
    let Ok(source) = sources.single() else {
        return; // no streaming source tagged yet
    };
    world.set_focus(source.translation().as_ivec3());
}
