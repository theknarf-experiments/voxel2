//! Engine glue: Bevy plugins for chunk streaming lifecycle, generation
//! budgets, edit application, and persistence.

pub mod level;
pub mod streaming;
pub mod vegetation;

use bevy::prelude::*;

pub use level::{LevelDef, LevelPlugin};
pub use streaming::{LodConfig, VoxelStreamingPlugin};
pub use vegetation::VegetationPlugin;
pub use voxel_core::ChunkKey;
pub use voxel_render::WorldKind;

/// Everything needed for a streamed voxel world.
pub struct VoxelEnginePlugin {
    pub world: WorldKind,
    pub vegetation: bool,
}

impl Default for VoxelEnginePlugin {
    fn default() -> Self {
        Self {
            world: WorldKind::default(),
            vegetation: true,
        }
    }
}

impl Plugin for VoxelEnginePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            voxel_render::VoxelChunksPlugin { world: self.world },
            VoxelStreamingPlugin,
        ));
        // Organic vegetation only grows on planets.
        if self.vegetation && self.world == WorldKind::Planet {
            app.add_plugins(VegetationPlugin);
        }
    }
}
