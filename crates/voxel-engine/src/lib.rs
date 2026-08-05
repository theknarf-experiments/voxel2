//! Engine glue: Bevy plugins for chunk streaming lifecycle, generation
//! budgets, edit application, and persistence.

pub mod streaming;
pub mod vegetation;

use bevy::prelude::*;

pub use streaming::{LodConfig, VoxelStreamingPlugin};
pub use voxel_core::ChunkKey;
pub use vegetation::VegetationPlugin;
pub use voxel_render::WorldKind;

/// Everything needed for a streamed voxel world.
#[derive(Default)]
pub struct VoxelEnginePlugin {
    pub world: WorldKind,
}

impl Plugin for VoxelEnginePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            voxel_render::VoxelChunksPlugin { world: self.world },
            VoxelStreamingPlugin,
        ));
        // Organic vegetation only grows on planets.
        if self.world == WorldKind::Planet {
            app.add_plugins(VegetationPlugin);
        }
    }
}
