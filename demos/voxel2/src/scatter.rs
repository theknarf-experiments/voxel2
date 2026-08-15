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
    level::{ScatterDef, ScatterOutput},
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
///
/// Only for a population that names no altitude band. One that DOES is
/// asking about a slab it already described — see [`ScatterPopulation::gate_pad`].
const GATE_Y_M: i32 = 4096;

/// One population. Registered once per class the level declares.
pub struct ScatterPopulation {
    def: ScatterDef,
    /// Emit instances whose carved ground and clearance gate placement.
    emit_sources: Vec<String>,
    /// The material of the region this population is gated on, resolved
    /// through its wired `biomes` node once. `None` grows anywhere.
    biome: Option<u32>,
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

    fn dependencies(&self) -> Vec<Dep> {
        let pad = self.gate_pad();
        let deps: Vec<Dep> = self
            .emit_sources
            .iter()
            .map(|name| Dep::named(name, pad))
            .collect();
        // A region gate reads no layer: the bands are in the generator
        // program, so the weight costs a couple of noise samples and no
        // residency at all.
        deps
    }
}

impl ScatterPopulation {
    /// How far outside its own tile this population reads its emit
    /// sources. ONE definition, because `dependencies` declares it and
    /// `create` reads with it, and a layer that reads past what it
    /// declared is the framework's loudest assert.
    ///
    /// The vertical half is the interesting one, and it is LEVEL data
    /// (`gate_y_m`) because nothing here can derive it: what a population
    /// must see is how far the ops its sources emit REACH, which is a
    /// fact about those structures. Deriving it from the population's own
    /// `altitude` was tried and is wrong — the megastructure's rubble
    /// lives in [-140, 140] and still needs more than 1024 m, because the
    /// shafts that carve it are kilometres long.
    ///
    /// The default stays "as far as anything could reach". Overstating it
    /// is expensive: the megastructure was pulling +/-31 tiles of a 132 m
    /// layer per rubble tile, and every consumer of those followed.
    fn gate_pad(&self) -> IVec3 {
        let y = self.def.gate_y_m.map_or(GATE_Y_M, |m| m as i32);
        IVec3::new(GATE_PAD_M, y, GATE_PAD_M)
    }
}

impl LayerChunk for ScatterChunk {
    type Layer = ScatterPopulation;

    fn create(&mut self, ctx: &ChunkCtx<'_, ScatterPopulation>) {
        let layer = ctx.layer();
        let own = ctx.chunk_bounds();
        let pad = layer.gate_pad();

        // Everything the gates read, from declared dependencies only.
        let mut clearance = Vec::new();
        let mut cut_ops = Vec::new();
        for source in &layer.emit_sources {
            ctx.get_named::<crate::planning::layers::EmitPatches>(
                source,
                voxel_layers::dep_bounds(own, pad),
            )
            .for_each(|_, chunk| {
                clearance.extend(chunk.patches.clearance.iter().copied());
                cut_ops.extend(
                    chunk
                        .patches
                        .ops
                        .iter()
                        .filter(|op| op.kind & 1 == 1)
                        .copied(),
                );
            });
        }

        let world = ctx.context::<WorldCtx>();
        // The region gate and the ground colour are the same ops, so a
        // population that grows in the forest grows exactly where the
        // ground is forest-coloured.
        let gate_generator = world.generator.clone();
        let zero_generator = world.generator.clone();
        let gate_material = layer.biome;
        let gate_weight = move |xz: Vec2| -> f32 {
            match gate_material {
                Some(m) => gate_generator.surface_material_weight(xz, 8.0, m),
                None => 1.0,
            }
        };

        let inputs = PlacementInputs {
            generator: &world.generator,
            clearance,
            cut_ops,
            gate_weight: Box::new(gate_weight),
            gate_is_zero_over: Box::new(move |lo, hi| match gate_material {
                Some(m) => zero_generator.material_weight_is_zero_over(lo, hi, 8.0, m),
                None => false,
            }),
        };
        let tile = IVec2::new(ctx.coord().x, ctx.coord().z);
        self.placements = tile_placements(&layer.def, &inputs, tile);
    }

    fn destroy(&mut self, _ctx: &ChunkCtx<'_, ScatterPopulation>) {
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

    fn dependencies(&self) -> Vec<Dep> {
        vec![Dep::named(&self.source, IVec3::ZERO)]
    }
}

impl LayerChunk for ScatterDrawChunk {
    type Layer = ScatterDraw;

    fn create(&mut self, ctx: &ChunkCtx<'_, ScatterDraw>) {
        let layer = ctx.layer();
        let mut placements = Vec::new();
        ctx.get_named::<ScatterPopulation>(&layer.source, ctx.chunk_bounds())
            .for_each(|_, chunk| placements.extend(chunk.placements.iter().cloned()));
        layer.sink.put(ctx.instance_key(), ctx.coord(), placements);
    }

    fn destroy(&mut self, ctx: &ChunkCtx<'_, ScatterDraw>) {
        ctx.layer().sink.take(ctx.instance_key(), ctx.coord());
    }
}

/// The promoted slice of a point population: the same placements the far
/// draw publishes, elected by their own seed and re-seated finely, drawn
/// as entities near the camera.
///
/// This is what makes the near trees and the far impostors AGREE: both
/// read one data layer, so every entity stands exactly where a point
/// stands, at the same variant. Election is per placement
/// (`seed / 2^32 < near.per_tile / per_tile`), so which points get
/// entities never depends on the camera, the tile's survivor count or
/// each other — walking toward a wood promotes the same trees every time.
pub struct ScatterNearDraw {
    source: String,
    def: ScatterDef,
    sink: Sink<Placement>,
}

#[derive(Default)]
pub struct ScatterNearDrawChunk;

impl Layer for ScatterNearDraw {
    type Chunk = ScatterNearDrawChunk;
    const NAME: &'static str = "scatter-near";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(self.def.tile_m as f64, 0.0, self.def.tile_m as f64)
    }

    fn dependencies(&self) -> Vec<Dep> {
        vec![Dep::named(&self.source, IVec3::ZERO)]
    }
}

impl LayerChunk for ScatterNearDrawChunk {
    type Layer = ScatterNearDraw;

    fn create(&mut self, ctx: &ChunkCtx<'_, ScatterNearDraw>) {
        let layer = ctx.layer();
        let Some(near) = &layer.def.near else {
            return;
        };
        let world = ctx.context::<WorldCtx>();
        let mut placements = Vec::new();
        ctx.get_named::<ScatterPopulation>(&layer.source, ctx.chunk_bounds())
            .for_each(|_, chunk| {
                for p in &chunk.placements {
                    let elected = (p.seed as u32 as u64 * layer.def.per_tile as u64) >> 32
                        < near.per_tile as u64;
                    if !elected {
                        continue;
                    }
                    // The population seated this point at its own coarse
                    // `detail_vs`; an entity stands on finely meshed
                    // ground, so refine the seat by the DELTA between the
                    // two heights. The delta, not a fresh seat: sink and
                    // altitude falloff are already in the stored y.
                    let mut p = *p;
                    let xz = Vec2::new(p.position.x, p.position.z);
                    p.position.y += world.generator.height(xz, near.detail_vs)
                        - world.generator.height(xz, layer.def.detail_vs);
                    placements.push(p);
                }
            });
        layer.sink.put(ctx.instance_key(), ctx.coord(), placements);
    }

    fn destroy(&mut self, ctx: &ChunkCtx<'_, ScatterNearDraw>) {
        ctx.layer().sink.take(ctx.instance_key(), ctx.coord());
    }
}

/// Registered populations, so the reconciling systems know what to draw.
#[derive(Resource, Default)]
pub struct Populations(pub Vec<PopulationHandle>);

pub struct PopulationHandle {
    /// The world this population decorates. Scatter is scene content and
    /// scene content belongs to a world — a portal shows another level's
    /// trees, not this one's moved sideways.
    pub world: voxel_engine::WorldId,
    pub class: Arc<str>,
    /// Where tooling reads this handle's live size. Distinct from the
    /// class since a population with a near slice is TWO handles of one
    /// class, and they must not overwrite each other's count.
    count_key: String,
    pub output: ScatterOutput,
    pub sink: Sink<Placement>,
    /// Entities currently spawned per contributing chunk, for entity
    /// populations. Keyed like the sink: instance AND coordinate.
    spawned: std::collections::HashMap<crate::planning::world::PartKey, Vec<Entity>>,
    seen_generation: u64,
}

/// The material of the region a population's `"instance:member"` gate
/// names, or `None` for one that grows anywhere.
///
/// A region IS a generator band, so this resolves to the same material id
/// the ground is painted with — which is what lets anything that draws the
/// population where its instances are not (the far-field ground paint) ask
/// the same question the placer asked.
pub fn gate_material(
    def: &ScatterDef,
    biome_tables: &[(String, Vec<(String, u32)>)],
) -> Option<u32> {
    let reference = def.gate.as_ref()?;
    let (instance, name) = reference.rsplit_once(':')?;
    let table = biome_tables.iter().find(|(n, _)| n == instance)?;
    table.1.iter().find(|(n, _)| n == name).map(|(_, m)| *m)
}

/// Register one layer per population and return their top dependencies.
pub fn register(
    graph: &mut LayerGraph,
    populations: &[ScatterDef],
    emit_sources: Vec<String>,
    biome_tables: &[(String, Vec<(String, u32)>)],
) -> (Vec<TopDep>, Populations) {
    let mut tops = Vec::new();
    let mut handles = Vec::new();
    for def in populations {
        let biome = gate_material(def, biome_tables);
        let sink = Sink::default();
        let population = ScatterPopulation {
            def: def.clone(),
            emit_sources: emit_sources.clone(),
            biome,
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
        tops.push(TopDep::new(
            &draw,
            IVec3::new((2.0 * reach) as i32, 0, (2.0 * reach) as i32),
        ));
        handles.push(PopulationHandle {
            // Stamped when the world adopts them; the planner that built
            // these does not know which world it belongs to.
            world: 0,
            class: Arc::from(def.class.as_str()),
            count_key: def.class.clone(),
            output: def.output,
            sink,
            spawned: std::collections::HashMap::new(),
            seen_generation: u64::MAX,
        });

        // The promoted near slice: a second draw over the SAME data
        // layer, at its own radius, spawning entities. See
        // [`ScatterNearDraw`] for why this is a subset and not a second
        // population.
        if let (Some(near), ScatterOutput::Points) = (&def.near, def.output) {
            let near_sink = Sink::default();
            let near_name = format!("{}:near", def.class);
            graph.register_as(
                &near_name,
                ScatterNearDraw {
                    source: def.class.clone(),
                    def: def.clone(),
                    sink: near_sink.clone(),
                },
            );
            let reach = (near.radius_tiles as f32 + 0.5) * def.tile_m;
            tops.push(TopDep::new(
                &near_name,
                IVec3::new((2.0 * reach) as i32, 0, (2.0 * reach) as i32),
            ));
            handles.push(PopulationHandle {
                world: 0,
                class: Arc::from(def.class.as_str()),
                count_key: near_name,
                output: ScatterOutput::Entities,
                sink: near_sink,
                spawned: std::collections::HashMap::new(),
                seen_generation: u64::MAX,
            });
        }
    }
    (tops, Populations(handles))
}

/// Bring the world in line with what the layers published: spawn and
/// despawn entities per chunk, and hand point populations to the renderer
/// in bulk.
///
/// Not budgeted with `timed!`: this spawns in bulk whenever a population
/// arrives, so overrunning a frame is what it DOES on a hot reload, and a
/// warning that fires on the normal case says nothing.
fn reconcile(
    mut commands: Commands,
    points: Res<voxel_render::ScatterPoints>,
    mut populations: ResMut<Populations>,
    worlds: Res<voxel_engine::Worlds>,
    props: Res<crate::WorldProps>,
) {
    for population in &mut populations.0 {
        let generation = population.sink.generation();
        if generation == population.seen_generation {
            continue;
        }
        population.seen_generation = generation;
        match population.output {
            ScatterOutput::Points => {
                // If the class has a prop table, each point's silhouette
                // bit is written from its VARIANT's model, so an impostor
                // and the entity it may be promoted to are the same
                // species. Bit 23 is the top of the shader's shape byte —
                // set means the waisted (broadleaf) silhouette.
                let variant_models = props
                    .0
                    .get(&population.world)
                    .and_then(|t| t.0.get(&*population.class))
                    .map(|class| {
                        class
                            .variants
                            .iter()
                            .map(|v| v.model == crate::props::Model::Broadleaf)
                            .collect::<Vec<bool>>()
                    });
                let merged = population.sink.collect_map(|p| {
                    let mut hash = p.seed as u32;
                    if let Some(wide) = variant_models
                        .as_ref()
                        .and_then(|models| models.get(p.variant as usize).copied())
                    {
                        hash = (hash & !(1 << 23)) | (u32::from(wide) << 23);
                    }
                    voxel_render::ScatterPoint {
                        pos: p.position.to_array(),
                        hash,
                    }
                });
                let n = merged.len();
                points.set_class(population.world, &population.class, merged);
                record_count(&worlds, population, n);
            }
            ScatterOutput::Entities => {
                let live = population.sink.keys();
                population.spawned.retain(|part, entities| {
                    if live.contains(part) {
                        return true;
                    }
                    for entity in entities.drain(..) {
                        commands.entity(entity).despawn();
                    }
                    false
                });
                for part in live {
                    if population.spawned.contains_key(&part) {
                        continue;
                    }
                    let Some(placements) = population.sink.get(part) else {
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
                                    // Only visible from its own world,
                                    // and from a far view of it.
                                    crate::OfWorld::scene(population.world),
                                ))
                                .id()
                        })
                        .collect();
                    population.spawned.insert(part, entities);
                }
                let n = population.spawned.values().map(Vec::len).sum();
                record_count(&worlds, population, n);
            }
        }
    }
}

/// Publish a population's live size where tooling can read it.
fn record_count(worlds: &voxel_engine::Worlds, population: &PopulationHandle, n: usize) {
    if let Some(ctx) = worlds
        .query(population.world)
        .and_then(|w| w.host_ctx::<WorldCtx>())
    {
        ctx.placements
            .lock()
            .unwrap()
            .insert(population.count_key.clone(), n);
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
/// Adopt the populations of EVERY loaded world, once each.
///
/// Not a one-shot: a world can arrive long after startup — opening a
/// portal loads one on the spot — and a `taken` flag meant only the
/// launched level was ever decorated. Its trees showed through a portal
/// and the other level's did not.
fn adopt_populations(
    mut commands: Commands,
    worlds: Res<voxel_engine::Worlds>,
    points: Res<voxel_render::ScatterPoints>,
    mut populations: ResMut<Populations>,
) {
    for world in worlds.iter() {
        let Some(ctx) = world.query.host_ctx::<WorldCtx>() else {
            continue;
        };
        // Having something to take IS the signal, so there is no "already
        // adopted" flag: a hot reload rebuilds the world query and leaves
        // a fresh set here, and a flag meant the old planner's handles
        // stayed in the list forever, holding sinks nothing would ever
        // fill again. Editing a population then did nothing at all.
        let Some(found) = ctx.populations.lock().unwrap().take() else {
            continue;
        };
        retire_populations(&mut commands, &points, &mut populations, world.id);
        populations.0.extend(found.0.into_iter().map(|mut p| {
            p.world = world.id;
            p
        }));
    }
}

/// Drop one world's populations, taking their entities and points with
/// them. What replaces them may not have the same classes at all.
fn retire_populations(
    commands: &mut Commands,
    points: &voxel_render::ScatterPoints,
    populations: &mut Populations,
    world: voxel_engine::WorldId,
) {
    populations.0.retain_mut(|population| {
        if population.world != world {
            return true;
        }
        for entities in population.spawned.values_mut() {
            for entity in entities.drain(..) {
                commands.entity(entity).despawn();
            }
        }
        // A class the new level dropped would otherwise keep drawing the
        // old planner's last published points forever.
        points.set_class(world, &population.class, Vec::new());
        false
    });
}
