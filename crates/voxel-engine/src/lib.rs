//! Engine glue: Bevy plugins for chunk streaming lifecycle, generation
//! budgets, edit application, and persistence.

pub mod chunkgen;
pub mod graph;
pub mod layers;
pub mod level;
pub mod lod_layers;
pub mod planning;
pub mod scatter;
pub mod schema;
pub mod streaming;

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
pub type StreamSourceQuery<'w, 's> =
    Query<'w, 's, &'static GlobalTransform, With<VoxelStreamSource>>;

pub use chunkgen::ChunkGen;
pub use layers::{MainThreadBudget, MainThreadQueue, VoxelLayersPlugin};
pub use level::{LevelDef, LevelPlugin};
pub use lod_layers::WorldFocus;
pub use planning::{Marker, PatchSet, PlanningStats, RibbonSeg, WorldPlanner, WorldQuery};
pub use scatter::{Placement, PlacementInputs, ScatterInstance};
pub use streaming::{LodConfig, VoxelStreamingPlugin};
pub use voxel_core::{ChunkKey, WorldId};

/// Where each world is looked at from is PUBLISHED before it is
/// CONSUMED, in the same frame.
///
/// A host decides this — a portal moves the point a world is seen from —
/// and the engine's LOD and planning graphs both follow it. Consuming a
/// stale focus for one frame is not a hiccup: the fallback is the camera,
/// and after stepping through a portal the camera is in another world's
/// coordinates entirely, so the world you just left re-centres 46 km away
/// and regenerates its whole planning graph.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorldFocusSet {
    /// Publish `WorldFocus`. The host's.
    Publish,
    /// Read it: LOD residency and planning residency.
    Follow,
}

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
    /// What the host ASKED for. Kept because the slab is shared: when
    /// another world loads, this one is re-fitted from its authored
    /// detail rather than from whatever it was last capped to, so
    /// capping never ratchets downwards.
    pub authored: LodConfig,
    pub config: LodConfig,
    pub generator: std::sync::Arc<voxel_worldgen::Generator>,
    /// Heights, fields, shadows and the host's planning, for THIS world.
    pub query: WorldQuery,
    /// Slab slots this world's residency was admitted against — what the
    /// NEXT world's budget is reduced by.
    pub slab_demand: usize,
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
pub struct WorldLoader<'w, 's> {
    worlds: ResMut<'w, Worlds>,
    render: ResMut<'w, voxel_render::RenderWorlds>,
    planner: Res<'w, level::HostPlanner>,
    rebuild: ResMut<'w, streaming::StreamingRebuild>,
    /// Where the camera is. Demand is measured THERE, not at the origin.
    sources: StreamSourceQuery<'w, 's>,
    /// What the slab has actually cost, reported by the render world.
    /// Optional: a headless host registers worlds with no renderer at
    /// all, and admission falls back to the configured page budget.
    stats: Option<Res<'w, voxel_render::SharedRenderStats>>,
    /// The page budget, for the first admission — before any chunk has
    /// been meshed there is no evidence to measure.
    slab_config: Option<Res<'w, voxel_render::slab::SlabConfig>>,
}

impl WorldLoader<'_, '_> {
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
            authored: config.clone(),
            config,
            generator,
            query,
            slab_demand: 0,
        });
        // A new world changes everyone's share, so everyone is re-fitted.
        if self.rebalance() {
            self.rebuild.0 = true;
        }
        id
    }

    /// Divide the slab between the loaded worlds and re-fit each to its
    /// share. Returns true if a world ALREADY streaming had to change,
    /// which needs its LOD graph rebuilt.
    ///
    /// Shared, not first-come-first-served. Admitting each world against
    /// "whatever the last one left" gave the launched level its full
    /// authored detail and the next one the scraps — with three levels
    /// loaded the third got 173 of 3656 slots and streamed 22 chunks,
    /// which is a horizon and nothing else. Every world now gets an equal
    /// share, and a world that needs less than its share hands the
    /// surplus to the ones that need more.
    ///
    /// Re-fitting is from the AUTHORED config every time, so a world that
    /// was capped while three were loaded goes back up when one is
    /// dropped, and repeated loads cannot ratchet it down.
    fn rebalance(&mut self) -> bool {
        // MEASURED, not declared. The slab is a page pool, so how many
        // chunks it holds depends on how many pages a chunk costs — ~1
        // for terrain, several times that in dense interior geometry —
        // and that is a property of the level, not of the allocator. It
        // reports the figure from its own peak; before the render world
        // has run there is nothing to report and the page count is the
        // optimistic bound.
        let budget = self
            .slab_config
            .as_deref()
            .copied()
            .unwrap_or_default()
            .total_pages as usize;
        let capacity = self
            .stats
            .as_ref()
            .and_then(|s| s.0.lock().ok().map(|s| s.slab_capacity_chunks))
            .filter(|c| *c > 0)
            .unwrap_or(budget);
        // WHERE THE CAMERA IS, not the origin. Residency is radial, so I
        // assumed the count barely moved with the anchor and measured at
        // the origin — it moves by a THIRD on both shipped levels (1.37x
        // and 1.31x at the demo's own start positions). Measuring in the
        // wrong place let two worlds be admitted into 3568 of 3656 slots
        // when they wanted 5310 where the player actually stood, which
        // filled every slab class and stopped the world settling at all.
        let anchor = self
            .sources
            .single()
            .map_or(bevy::math::DVec3::ZERO, |t| t.translation().as_dvec3());
        let n = self.worlds.0.len().max(1);
        let mut budget = vec![capacity / n; n];

        // Water-fill: whoever wants less than an equal share releases the
        // difference to whoever wants more. Twice is enough for the sizes
        // in play and cannot loop.
        let full: Vec<usize> = self
            .worlds
            .0
            .iter()
            .map(|w| streaming::meshable_count(&w.authored, &w.generator, anchor))
            .collect();
        for _ in 0..2 {
            // TAKE the surplus back before handing it out, or the budgets
            // sum to more than the slab and every world is "admitted"
            // into space that does not exist.
            let mut surplus = 0;
            for i in 0..n {
                if full[i] < budget[i] {
                    surplus += budget[i] - full[i];
                    budget[i] = full[i];
                }
            }
            let hungry: Vec<usize> = (0..n).filter(|&i| full[i] > budget[i]).collect();
            if surplus == 0 || hungry.is_empty() {
                break;
            }
            let each = surplus / hungry.len();
            for i in hungry {
                budget[i] += each;
            }
        }
        debug_assert!(
            budget.iter().sum::<usize>() <= capacity,
            "shares {budget:?} exceed the {capacity} slots they divide",
        );

        let mut disturbed = false;
        for (i, world) in self.worlds.0.iter_mut().enumerate() {
            let (fitted, demand) = streaming::fit_to_budget(
                &world.authored,
                &world.generator,
                anchor,
                budget[i],
                MIN_STREAMED_LEVEL,
            );
            let was = world.config.max_level;
            if world.slab_demand != 0 && fitted.max_level != was {
                disturbed = true;
            }
            if fitted.max_level == world.authored.max_level {
                info!(
                    "world {i} at its authored L{}: {demand} of {} slots",
                    fitted.max_level, budget[i],
                );
            } else {
                info!(
                    "world {i} capped to L{} (from L{}): {demand} of {} slots",
                    fitted.max_level, world.authored.max_level, budget[i],
                );
            }
            world.config = fitted;
            world.slab_demand = demand;
        }
        disturbed
    }
}

/// Coarsest a world may be capped to. Below this a level is a few boxes
/// on an empty horizon and not worth loading at all — better to admit it
/// over budget and let deferral absorb the difference, which is safe.
const MIN_STREAMED_LEVEL: u8 = 4;

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
pub struct VoxelEnginePlugin {
    /// The mesh slab's budget. Passed straight through to
    /// `voxel_render::VoxelChunksPlugin` — a game with denser chunks
    /// than this demo's raises it here.
    pub slab: voxel_render::slab::SlabConfig,
}

impl Plugin for VoxelEnginePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            voxel_render::VoxelChunksPlugin { slab: self.slab },
            VoxelStreamingPlugin,
        ))
        .insert_resource(self.slab)
        .init_resource::<Worlds>()
        .configure_sets(Update, WorldFocusSet::Follow.after(WorldFocusSet::Publish))
        .add_systems(PreUpdate, sync_ops_providers);
    }
}
