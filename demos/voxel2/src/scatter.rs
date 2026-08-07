//! Prop and ground-cover populations, as layers.
//!
//! The engine decides *where* things go — [`tile_placements`] is a pure
//! function of a tile and its inputs. Deciding *when they exist* is a
//! residency question, which makes it a layer's job, and this game's.
//!
//! What this replaces: two tile maps keyed by (class, tile), a radius and
//! a keep-radius per class, a staleness scan, a per-frame budget and a
//! rebuild flag wired through level reload. All of it re-implemented, less
//! well, what a declared dependency and a top dependency already say.
//!
//! Both outputs go through a sink of placements. A population that draws
//! entities has them reconciled against the sink on the main thread —
//! chunks generate on workers and cannot spawn — and one that draws points
//! feeds `voxel_render::ScatterPoints` in bulk.

use std::sync::Arc;

use bevy::math::DVec3;
use bevy::prelude::*;
use voxel_engine::{
    level::{LevelDef, ScatterDef, ScatterOutput},
    scatter::{tile_placements, Placement, PlacementInputs, ScatterInstance},
};
use voxel_layers::{ChunkCtx, Dep, Layer, LayerChunk, LayerGraph, TopDep};

use crate::planning::world::{Sink, WorldCtx};

/// Clearance the planning stack reserved; a tile has to see the beds that
/// cross into it, not just the ones that start in it.
const GATE_PAD_M: i32 = 64;

/// How far above and below a population looks for carved ground. Cave
/// mouths reach the surface from a long way down, and this layer is planar
/// so it has to say how deep it cares.
const GATE_Y_M: i32 = 4096;

/// One population. Registered once per class the level declares.
pub struct ScatterPopulation {
    def: ScatterDef,
    /// Emit instances whose carved ground and clearance gate placement.
    emit_sources: Vec<String>,
    /// (instance, biome name, biome count) this population is gated on.
    biome: Option<(String, String, usize)>,
    /// That biome's position in its field's table, resolved once.
    biome_index: Option<usize>,
}

#[derive(Default)]
pub struct ScatterChunk {
    pub placements: Vec<Placement>,
}

impl Layer for ScatterPopulation {
    type Chunk = ScatterChunk;
    const NAME: &'static str = "scatter";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(self.def.tile_m as f64, 0.0, self.def.tile_m as f64)
    }

    fn dependencies(&self, _level: u32) -> Vec<Dep> {
        let pad = IVec3::new(GATE_PAD_M, GATE_Y_M, GATE_PAD_M);
        let mut deps: Vec<Dep> = self
            .emit_sources
            .iter()
            .map(|name| Dep::named(name, pad))
            .collect();
        if let Some((instance, _, _)) = &self.biome {
            // Biome weights blend across a wide influence window.
            let biome_pad = crate::planning::layers::BIOME_INFLUENCE_CELLS;
            deps.push(Dep::named(instance, IVec3::new(biome_pad, 0, biome_pad)));
        }
        deps
    }
}

impl LayerChunk for ScatterChunk {
    type Layer = ScatterPopulation;

    fn create(&mut self, ctx: &ChunkCtx<'_, ScatterPopulation>, _level: u32) {
        let layer = ctx.layer();
        let own = ctx.chunk_bounds();
        let pad = IVec3::new(GATE_PAD_M, GATE_Y_M, GATE_PAD_M);

        // Everything the gates read, from declared dependencies only.
        let mut clearance = Vec::new();
        let mut cut_ops = Vec::new();
        for source in &layer.emit_sources {
            ctx.get_named::<crate::planning::layers::EmitPatches>(source, voxel_layers::dep_bounds(own, pad))
                .for_each(|_, chunk| {
                    clearance.extend(chunk.patches.clearance.iter().copied());
                    cut_ops.extend(chunk.patches.ops.iter().filter(|op| op.kind & 1 == 1).copied());
                });
        }

        let biome_sites = layer.biome.as_ref().map(|(instance, _, _)| {
            let biome_pad = crate::planning::layers::BIOME_INFLUENCE_CELLS;
            let mut sites = Vec::new();
            ctx.get_named::<crate::planning::layers::BiomeField>(
                instance,
                voxel_layers::dep_bounds(own, IVec3::new(biome_pad, 0, biome_pad)),
            )
            .for_each(|_, c| sites.push((c.site, c.biome)));
            sites
        });
        let biome_weight = move |xz: Vec2| -> f32 {
            let (Some(sites), Some((_, name, count))) = (&biome_sites, &layer.biome) else {
                return 1.0;
            };
            let _ = name;
            let weights = crate::planning::layers::biome_weights_from(sites, *count, xz);
            layer
                .biome_index
                .and_then(|i| weights.get(i).copied())
                .unwrap_or(1.0)
        };

        let world = ctx.context::<WorldCtx>();
        let inputs = PlacementInputs {
            generator: &world.generator,
            clearance,
            cut_ops,
            biome_weight: Box::new(biome_weight),
        };
        let tile = IVec2::new(ctx.coord().x, ctx.coord().z);
        self.placements = tile_placements(&layer.def, &inputs, tile);
    }

    fn destroy(&mut self, _ctx: &ChunkCtx<'_, ScatterPopulation>, _level: u32) {
        self.placements.clear();
    }
}

/// What a population *draws*, separated from what it *is*.
///
/// Residency of the data and residency of the meshes are different
/// questions: the far impostor ring reads tree placements kilometres out,
/// and none of those should become near-mesh entities. Two layers, two top
/// dependencies, and neither has to know why the other wants the data.
pub struct ScatterDraw {
    source: String,
    tile_m: f32,
    sink: Sink<Placement>,
}

#[derive(Default)]
pub struct ScatterDrawChunk;

impl Layer for ScatterDraw {
    type Chunk = ScatterDrawChunk;
    const NAME: &'static str = "scatter-draw";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(self.tile_m as f64, 0.0, self.tile_m as f64)
    }

    fn dependencies(&self, _level: u32) -> Vec<Dep> {
        vec![Dep::named(&self.source, IVec3::ZERO)]
    }
}

impl LayerChunk for ScatterDrawChunk {
    type Layer = ScatterDraw;

    fn create(&mut self, ctx: &ChunkCtx<'_, ScatterDraw>, _level: u32) {
        let layer = ctx.layer();
        let mut placements = Vec::new();
        ctx.get_named::<ScatterPopulation>(&layer.source, ctx.chunk_bounds())
            .for_each(|_, chunk| placements.extend(chunk.placements.iter().cloned()));
        layer.sink.put(ctx.coord(), placements);
    }

    fn destroy(&mut self, ctx: &ChunkCtx<'_, ScatterDraw>, _level: u32) {
        ctx.layer().sink.take(ctx.coord());
    }
}

/// Registered populations, so the reconciling systems know what to draw.
#[derive(Resource, Default)]
pub struct Populations(pub Vec<PopulationHandle>);

pub struct PopulationHandle {
    pub class: Arc<str>,
    pub output: ScatterOutput,
    pub sink: Sink<Placement>,
    /// Entities currently spawned per chunk, for entity populations.
    spawned: std::collections::HashMap<IVec3, Vec<Entity>>,
    seen_generation: u64,
}

/// Register one layer per scatter class and return their top dependencies.
pub fn register(
    graph: &mut LayerGraph,
    level: &LevelDef,
    emit_sources: Vec<String>,
    biome_tables: &[(String, Vec<String>)],
) -> (Vec<TopDep>, Populations) {
    let mut tops = Vec::new();
    let mut handles = Vec::new();
    for def in &level.scatter {
        let biome = def.biome.as_ref().and_then(|reference| {
            let (instance, name) = reference.rsplit_once(':')?;
            let table = biome_tables.iter().find(|(n, _)| n == instance)?;
            Some((instance.to_string(), name.to_string(), table.1.len()))
        });
        let biome_index = biome.as_ref().and_then(|(instance, name, _)| {
            biome_tables
                .iter()
                .find(|(n, _)| n == instance)
                .and_then(|(_, table)| table.iter().position(|b| b == name))
        });
        let sink = Sink::default();
        let population = ScatterPopulation {
            def: def.clone(),
            emit_sources: emit_sources.clone(),
            biome,
            biome_index,
        };
        // The population's own radius, as declared in the level, becomes
        // the size of its top dependency — the one number that decides how
        // far it exists, instead of a radius plus a keep-radius plus a
        // pre-warm reach that all had to agree.
        let reach = (def.radius_tiles as f32 + 0.5) * def.tile_m;
        graph.register_as(&def.class, population);
        let draw = format!("{}:draw", def.class);
        graph.register_as(
            &draw,
            ScatterDraw {
                source: def.class.clone(),
                tile_m: def.tile_m,
                sink: sink.clone(),
            },
        );
        tops.push(TopDep::at_level(
            &draw,
            0,
            IVec3::new((2.0 * reach) as i32, 0, (2.0 * reach) as i32),
        ));
        handles.push(PopulationHandle {
            class: Arc::from(def.class.as_str()),
            output: def.output,
            sink,
            spawned: std::collections::HashMap::new(),
            seen_generation: u64::MAX,
        });
    }
    (tops, Populations(handles))
}

/// Bring the world in line with what the layers published: spawn and
/// despawn entities per chunk, and hand point populations to the renderer
/// in bulk.
fn reconcile(
    mut commands: Commands,
    points: Res<voxel_render::ScatterPoints>,
    mut populations: ResMut<Populations>,
) {
    for population in &mut populations.0 {
        let generation = population.sink.generation();
        if generation == population.seen_generation {
            continue;
        }
        population.seen_generation = generation;
        match population.output {
            ScatterOutput::Points => {
                let merged: Vec<voxel_render::ScatterPoint> = population
                    .sink
                    .collect()
                    .iter()
                    .map(|p| voxel_render::ScatterPoint {
                        pos: p.position.to_array(),
                        hash: p.seed as u32,
                    })
                    .collect();
                points.set_class(&population.class, merged);
            }
            ScatterOutput::Entities => {
                let live = population.sink.keys();
                population.spawned.retain(|coord, entities| {
                    if live.contains(coord) {
                        return true;
                    }
                    for entity in entities.drain(..) {
                        commands.entity(entity).despawn();
                    }
                    false
                });
                for coord in live {
                    if population.spawned.contains_key(&coord) {
                        continue;
                    }
                    let Some(placements) = population.sink.get(coord) else {
                        continue;
                    };
                    let entities = placements
                        .into_iter()
                        .map(|p| {
                            commands
                                .spawn((
                                    Transform::from_translation(p.position)
                                        .with_rotation(p.rotation)
                                        .with_scale(Vec3::splat(p.scale)),
                                    Visibility::default(),
                                    ScatterInstance {
                                        class: population.class.clone(),
                                        variant: p.variant,
                                        scale: p.scale,
                                        seed: p.seed,
                                    },
                                ))
                                .id()
                        })
                        .collect();
                    population.spawned.insert(coord, entities);
                }
            }
        }
    }
}

/// Draws the level's scatter populations.
pub struct ScatterPlugin;

impl Plugin for ScatterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Populations>()
            .add_systems(Update, (adopt_populations, reconcile).chain());
    }
}

/// The planner builds the populations while the level plugin builds; this
/// picks them up once the world query exists.
fn adopt_populations(
    world: Option<Res<voxel_engine::WorldQuery>>,
    mut populations: ResMut<Populations>,
    mut taken: Local<bool>,
) {
    if *taken {
        return;
    }
    let Some(world) = world else { return };
    let Some(ctx) = world.host_ctx::<WorldCtx>() else {
        return;
    };
    let found = ctx.populations.lock().unwrap().take();
    if let Some(found) = found {
        *populations = found;
        *taken = true;
    }
}
