//! Engine glue: Bevy plugins for chunk streaming lifecycle, generation
//! budgets, edit application, and persistence.

pub mod chunkgen;
pub mod layers;
pub mod level;
pub mod planning;
pub mod streaming;
pub mod scatter;

use bevy::prelude::*;

/// Tag the entity whose position drives streaming: chunk LOD, prop and
/// ribbon tiles, planning pre-generation. Usually the player or the
/// camera, but the engine never assumes which — a game owns its own
/// scene. LayerProcGen calls this a generation source.
///
/// ```ignore
/// commands.spawn((Camera3d::default(), VoxelStreamSource));
/// ```
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct VoxelStreamSource;

/// The streaming anchor's transform. Systems take this instead of
/// querying for a camera.
pub type StreamSourceQuery<'w, 's> = Query<'w, 's, &'static GlobalTransform, With<VoxelStreamSource>>;

pub use level::{LevelDef, LevelPlugin};
pub use chunkgen::{ChunkGen, ChunkOpsProvider};
pub use layers::{MainThreadBudget, MainThreadQueue, VoxelLayersPlugin};
pub use planning::{Marker, PatchSet, PlanningStats, RibbonSeg, WorldPlanner, WorldQuery};
pub use streaming::{LodConfig, VoxelStreamingPlugin};
pub use scatter::{Placement, PlacementInputs, ScatterInstance};
pub use voxel_core::ChunkKey;

/// Everything needed for a streamed voxel world. The world itself is data:
/// a generator program in [`voxel_render::WorldProgram`], normally installed
/// by a [`LevelPlugin`].
#[derive(Default)]
pub struct VoxelEnginePlugin;

impl Plugin for VoxelEnginePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((voxel_render::VoxelChunksPlugin, VoxelStreamingPlugin));
    }
}
