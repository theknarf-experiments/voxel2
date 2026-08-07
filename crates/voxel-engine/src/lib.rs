//! Engine glue: Bevy plugins for chunk streaming lifecycle, generation
//! budgets, edit application, and persistence.

pub mod chunkgen;
pub mod layers;
pub mod lod_layers;
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
pub use lod_layers::WorldFocus;
pub use scatter::{Placement, PlacementInputs, ScatterInstance};
pub use voxel_core::{ChunkKey, WorldId};

/// One world the engine streams.
///
/// A portal shows two levels at once, so "the world" stops being a
/// singleton: each has its own LOD field, its own anchor and its own
/// generator. They share one chunk service and one GPU arena, because the
/// world rides in [`ChunkKey`].
pub struct StreamedWorld {
    pub id: WorldId,
    pub config: LodConfig,
    pub generator: std::sync::Arc<voxel_worldgen::Generator>,
}

/// The worlds to stream. `LevelPlugin` registers the level it loaded as
/// world 0; a host adds more.
#[derive(Resource, Default)]
pub struct StreamedWorlds(pub Vec<StreamedWorld>);

impl StreamedWorlds {
    /// Register a world and return its id.
    pub fn add(
        &mut self,
        config: LodConfig,
        generator: std::sync::Arc<voxel_worldgen::Generator>,
    ) -> WorldId {
        let id = self.0.len() as WorldId;
        assert!(
            (id as usize) < voxel_render::MAX_WORLDS,
            "at most {} worlds fit one program buffer",
            voxel_render::MAX_WORLDS,
        );
        self.0.push(StreamedWorld {
            id,
            config,
            generator,
        });
        id
    }
}

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
