//! Engine glue: Bevy plugins for chunk streaming lifecycle, generation
//! budgets, edit application, and persistence.

pub mod level;
pub mod remote;
pub mod streaming;
pub mod vegetation;
pub mod water_mesh;

use bevy::prelude::*;

pub use level::{LevelDef, LevelPlugin};
pub use streaming::{LodConfig, VoxelStreamingPlugin};
pub use vegetation::VegetationPlugin;
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
        // Vegetation systems gate themselves on the level's feature toggle
        // at runtime (so a hot-reload can switch worlds).
        if self.vegetation {
            app.add_plugins(VegetationPlugin);
        }
    }
}
