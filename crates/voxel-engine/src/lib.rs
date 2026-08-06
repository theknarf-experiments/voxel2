//! Engine glue: Bevy plugins for chunk streaming lifecycle, generation
//! budgets, edit application, and persistence.

pub mod level;
pub mod streaming;
pub mod vegetation;
pub mod river_water;

use bevy::prelude::*;

/// Tag the entity whose position drives streaming: chunk LOD, prop and
/// water tiles, planning pre-generation. Usually the player or the
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

/// Tests that install or read the process-global generator program must
/// hold this lock (several test modules race otherwise).
#[cfg(test)]
pub(crate) static PROGRAM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
