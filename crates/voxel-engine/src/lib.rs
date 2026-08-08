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
pub use chunkgen::ChunkGen;
pub use layers::{MainThreadBudget, MainThreadQueue, VoxelLayersPlugin};
pub use planning::{Marker, PatchSet, PlanningStats, RibbonSeg, WorldPlanner, WorldQuery};
pub use streaming::{LodConfig, VoxelStreamingPlugin};
pub use lod_layers::WorldFocus;
pub use scatter::{Placement, PlacementInputs, ScatterInstance};
pub use voxel_core::{ChunkKey, WorldId};

/// One loaded world, and everything that is true of it.
///
/// A portal shows two levels at once, so "the world" is not a singleton:
/// each has its own LOD field, its own anchor, its own generator and its
/// own planning. They share one chunk service and one GPU arena, because
/// the world rides in [`ChunkKey`].
///
/// One record rather than a resource per aspect. The state a world needs
/// used to live in five places that a caller had to append to in step,
/// and the only thing keeping the indices aligned was that two call sites
/// happened to agree.
pub struct World {
    pub id: WorldId,
    /// The definition this world was loaded from. Kept so a hot reload
    /// can diff against it.
    pub level: LevelDef,
    pub seed: u64,
    pub config: LodConfig,
    pub generator: std::sync::Arc<voxel_worldgen::Generator>,
    /// Heights, fields, shadows and the host's planning, for THIS world.
    pub query: WorldQuery,
}

/// Every loaded world, indexed BY world id.
///
/// Grown only through [`WorldLoader::load`], which is what keeps this and
/// [`voxel_render::RenderWorlds`] the same length with the same ids.
#[derive(Resource, Default)]
pub struct Worlds(Vec<World>);

impl Worlds {
    pub fn get(&self, id: WorldId) -> Option<&World> {
        self.0.get(usize::from(id))
    }

    pub fn get_mut(&mut self, id: WorldId) -> Option<&mut World> {
        self.0.get_mut(usize::from(id))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &World> {
        self.0.iter()
    }

    /// This world's facade, or world 0's if it does not exist. Callers
    /// that genuinely mean "whatever the player is looking at" want this;
    /// there is always a world 0 once a level has loaded.
    pub fn query(&self, id: WorldId) -> Option<&WorldQuery> {
        self.get(id).or_else(|| self.0.first()).map(|w| &w.query)
    }

    /// One ops provider per world, indexed by id, for [`ChunkGen`].
    ///
    /// A world with nothing to plan gets `None` rather than world 0's
    /// provider. Serving one world's planner to another asked its graph
    /// about coordinates where nothing is resident — worlds share
    /// coordinates, so it answered with the wrong level's roads and ruins
    /// and counted 40,474 missed reads doing it.
    pub fn ops_providers(&self) -> Vec<Option<chunkgen::OpsFn>> {
        self.0
            .iter()
            .map(|w| planning::ops_provider(&w.query))
            .collect()
    }
}

/// The one way to load a world.
///
/// Registration touches two sets — the simulation's [`Worlds`] and the
/// renderer's [`voxel_render::RenderWorlds`] — and a world is only usable
/// when its id means the same thing in both. Handing out a loader rather
/// than the sets themselves is what makes appending to one of them alone
/// impossible to write.
#[derive(bevy::ecs::system::SystemParam)]
pub struct WorldLoader<'w> {
    worlds: ResMut<'w, Worlds>,
    render: ResMut<'w, voxel_render::RenderWorlds>,
    planner: Res<'w, level::HostPlanner>,
}

impl WorldLoader<'_> {
    /// Load a level as a new world and return its id.
    ///
    /// `config` is separate from the level's own `lod` block because a
    /// world can be worth streaming at less than its authored detail — a
    /// level seen only through a portal does not need its finest LODs.
    pub fn load(&mut self, level: LevelDef, seed: u64, config: LodConfig) -> WorldId {
        let (program, generator) = level::build_generator(&level, seed);
        let query = level::build_world_query(&level, seed, &generator, self.planner.0.as_ref());

        let render_id = self.render.register(voxel_render::RenderWorld {
            program,
            materials: voxel_render::material_table(
                level.materials.iter().map(|m| (m.id(), m.pack())),
            ),
            ..default()
        });
        let id = self.worlds.0.len() as WorldId;
        assert_eq!(
            id, render_id,
            "world registries disagree: simulation would be {id}, renderer said {render_id}",
        );
        self.worlds.0.push(World {
            id,
            level,
            seed,
            config,
            generator,
            query,
        });
        id
    }
}

/// Push each world's ops provider into the chunk service whenever the set
/// of worlds changes. A world's planner is reached by `key.world`, so
/// nothing has to know which world a generation thread is serving.
fn sync_ops_providers(worlds: Res<Worlds>, chunks: Res<ChunkGen>) {
    if worlds.is_changed() {
        chunks.set_ops_providers(worlds.ops_providers());
    }
}

/// Everything needed for a streamed voxel world. The world itself is data:
/// a generator program in [`voxel_render::WorldProgram`], normally installed
/// by a [`LevelPlugin`].
#[derive(Default)]
pub struct VoxelEnginePlugin;

impl Plugin for VoxelEnginePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((voxel_render::VoxelChunksPlugin, VoxelStreamingPlugin))
            .init_resource::<Worlds>()
            .add_systems(PreUpdate, sync_ops_providers);
    }
}
