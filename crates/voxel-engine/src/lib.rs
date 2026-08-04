//! Engine glue: Bevy plugins for chunk streaming lifecycle, generation
//! budgets, edit application, and persistence.

pub mod streaming;

use bevy::prelude::*;

pub use streaming::{LodConfig, VoxelStreamingPlugin};

/// Everything needed for a streamed voxel terrain world.
pub struct VoxelEnginePlugin;

impl Plugin for VoxelEnginePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((voxel_render::VoxelChunksPlugin, VoxelStreamingPlugin));
    }
}
