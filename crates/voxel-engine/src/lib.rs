//! Engine glue: Bevy plugins for chunk streaming lifecycle, generation
//! budgets, edit application, and persistence.

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
pub use layers::{MainThreadBudget, MainThreadQueue, VoxelLayersPlugin, WorldLayers};
pub use planning::{Marker, PatchSet, PlanningLayers, RibbonSeg, WorldPlanner, WorldQuery};
pub use streaming::{LodConfig, VoxelStreamingPlugin};
pub use scatter::{Placement, ScatterInstance, ScatterPlugin};
pub use voxel_core::ChunkKey;

/// Everything needed for a streamed voxel world. The world itself is data:
/// a generator program in [`voxel_render::WorldProgram`], normally installed
/// by a [`LevelPlugin`].
pub struct VoxelEnginePlugin {
    pub vegetation: bool,
}

impl Default for VoxelEnginePlugin {
    fn default() -> Self {
        Self { vegetation: true }
    }
}

impl Plugin for VoxelEnginePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((voxel_render::VoxelChunksPlugin, VoxelStreamingPlugin));
        // Scatter streams the level's prop classes as entities the host
        // dresses; a game that spawns its own props can turn it off.
        if self.vegetation {
            app.add_plugins(ScatterPlugin);
        }
    }
}
