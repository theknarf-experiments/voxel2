//! This demo's planning: JSON-authored LayerProcGen layers.
//!
//! `voxel-layers` is the framework — dependency management, threaded
//! chunk generation, spatial organisation — and the concrete layers are
//! the game's, so they live here. [`layers`] holds the layer
//! implementations, [`structure`] the grammar one of them builds with,
//! and [`schema`] the JSON vocabulary that composes them. The engine sees
//! only [`StackPlanning`] through [`HostPlanning`].

pub mod layers;
pub mod schema;
pub mod structure;
pub mod world;

use std::sync::Arc;

use bevy::prelude::*;
use voxel_core::csg::CsgOp;
use voxel_engine::{
    level::LevelDef,
    planning::{HostPlanning, Marker, PlanningStats, RibbonSeg, WorldPlanner, OPS_HORIZON_EDGE_M},
};
use voxel_layers::{LayerGraph, LayerRuntime, TopDep, TopHandle};

pub use schema::{PlanningDef, StackLayerDef};
use schema::{validate_level, EmitDef};

/// This demo's planning: interpret the level's `planning` block.
pub struct StackPlanning;

/// Vertical band xz-facade queries cover: enough for any current world
/// (the deepest LOD tree spans ~±2.5 km), small enough that volumetric
/// emit layers don't enumerate thousands of 132 m y-rows per query.
const FACADE_Y_M: f32 = 2_560.0;

/// The level stack as a [`WorldPlanner`]: one `LayerManager` holding every
/// layer the level declared, plus the bookkeeping needed to answer a query
/// without generating layers that cannot contribute to it.
///
/// This is *a* host's planner, not the engine's: it interprets the
/// `stack`/`structures` blocks of a level file. A game with hand-written
/// layers implements [`WorldPlanner`] itself and never touches this.
#[derive(Clone, Default)]
pub struct StackPlanner {
    /// The level's planning stack: one graph for every layer, plus the
    /// thread keeping its top dependencies satisfied.
    stack: Option<Arc<LayerRuntime>>,
    /// Handles for the top dependencies that follow the camera.
    tops: Vec<TopHandle>,
    /// What this game's layers share, so its systems can read what they
    /// published.
    ctx: Option<Arc<world::WorldCtx>>,
    /// Emit instances and what each one can produce.
    emitters: Vec<Emitter>,
    /// Biome layers: (instance name, ordered biome names).
    biome_tables: Vec<(String, Vec<String>)>,
    /// A real camera position has been published at least once. Before
    /// that the graph is idle only because nothing has been asked of it.
    focused: Arc<std::sync::atomic::AtomicBool>,
}

/// One emit layer as the facade sees it. `produces` keeps a query from
/// touching — and therefore GENERATING — layers that cannot answer it: a
/// level with no ribbon emitter must not pull its structure planning into
/// existence through `ribbons_in`.
#[derive(Clone)]
struct Emitter {
    name: String,
    /// Carve-horizon gate in chunk-edge meters.
    gate: Option<f32>,
    /// Residency this layer's data is held to regardless of the gate.
    keep_m: Option<f32>,
    /// Its ribbons lie on the ground rather than at a level.
    seated: bool,
    ribbons: bool,
    clearance: bool,
    markers: bool,
}

impl Emitter {
    fn new(name: String, gate: Option<f32>, keep_m: Option<f32>, emit: &EmitDef) -> Self {
        let seated = matches!(emit, EmitDef::PathRibbon { .. });
        let (ribbons, clearance, markers) = match emit {
            EmitDef::Ribbon { .. } => (true, true, false),
            // Ribbons yes; clearance no — the carved road beside it
            // already declares that, and props must not be kept off a
            // road twice.
            EmitDef::PathRibbon { .. } => (true, false, false),
            EmitDef::PathSlabs { clearance, .. } => (false, *clearance, false),
            EmitDef::SiteStructure { marker, .. } | EmitDef::SiteStructure3 { marker, .. } => {
                (false, false, marker.is_some())
            }
            EmitDef::WormCuts | EmitDef::Tubes { .. } => (false, false, false),
        };
        Self {
            name,
            keep_m,
            seated,
            gate,
            ribbons,
            clearance,
            markers,
        }
    }
}

impl WorldPlanner for StackPlanner {
    /// Gated emitters drop out wholesale for coarse chunks — the gate is
    /// per chunk, never per op.
    fn ops_in(&self, min: Vec3, max: Vec3, chunk_edge_m: f32) -> Vec<CsgOp> {
        let mut out = Vec::new();
        if let Some(rt) = &self.stack {
            let mgr = rt.graph();
            for e in &self.emitters {
                if e.gate.is_none_or(|g| chunk_edge_m <= g) {
                    out.extend(layers::patches_in(mgr, &e.name, min, max).ops);
                }
            }
        }
        out
    }

    fn clearance_in(&self, min: bevy::math::Vec2, max: bevy::math::Vec2) -> Vec<[bevy::math::Vec2; 2]> {
        let (min3, max3) = (
            Vec3::new(min.x, -FACADE_Y_M, min.y),
            Vec3::new(max.x, FACADE_Y_M, max.y),
        );
        let mut out = Vec::new();
        if let Some(rt) = &self.stack {
            let mgr = rt.graph();
            for e in self.emitters.iter().filter(|e| e.clearance) {
                out.extend(layers::patches_in(mgr, &e.name, min3, max3).clearance);
            }
        }
        out
    }

    fn ribbons_in(&self, min: bevy::math::Vec2, max: bevy::math::Vec2) -> Vec<RibbonSeg> {
        let (min3, max3) = (
            Vec3::new(min.x, -FACADE_Y_M, min.y),
            Vec3::new(max.x, FACADE_Y_M, max.y),
        );
        let mut out = Vec::new();
        if let Some(rt) = &self.stack {
            let mgr = rt.graph();
            for e in self.emitters.iter().filter(|e| e.ribbons) {
                out.extend(layers::patches_in(mgr, &e.name, min3, max3).ribbons);
            }
        }
        out
    }

    fn biome_fields(&self) -> Vec<String> {
        self.biome_tables.iter().map(|(n, _)| n.clone()).collect()
    }

    fn biomes_at(&self, instance: &str, p: bevy::math::Vec2) -> Vec<(String, f32)> {
        let Some(rt) = &self.stack else {
            return Vec::new();
        };
        let mgr = rt.graph();
        let Some(table) = self.biome_tables.iter().find_map(|(n, t)| {
            (n == instance).then_some(t)
        }) else {
            return Vec::new();
        };
        let w = layers::biome_weights_at(mgr, instance, table.len(), p);
        table.iter().cloned().zip(w).collect()
    }

    fn markers_in(
        &self,
        min: bevy::math::Vec2,
        max: bevy::math::Vec2,
        kind: Option<&str>,
    ) -> Vec<Marker> {
        let (min3, max3) = (
            Vec3::new(min.x, -FACADE_Y_M, min.y),
            Vec3::new(max.x, FACADE_Y_M, max.y),
        );
        let mut out = Vec::new();
        if let Some(rt) = &self.stack {
            let mgr = rt.graph();
            for e in self.emitters.iter().filter(|e| e.markers) {
                out.extend(
                    layers::patches_in(mgr, &e.name, min3, max3)
                        .markers
                        .into_iter()
                        .filter(|m| kind.is_none_or(|k| m.kind == k)),
                );
            }
        }
        out
    }

    fn set_focus(&self, focus: bevy::math::IVec3) {
        for top in &self.tops {
            top.set_focus(focus);
        }
        self.focused.store(true, std::sync::atomic::Ordering::Release);
    }

    fn host_ctx(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        self.ctx
            .as_ref()
            .map(|c| c.as_ref() as &(dyn std::any::Any + Send + Sync))
    }

    /// Block until planning reflects a REAL camera position.
    ///
    /// "Idle" alone is true before anything has been asked for, which is
    /// the state the graph is in for the first frames — so a chunk
    /// generated in that window would wait for nothing, read an empty
    /// graph, and bake itself featureless. It presents as a burst of
    /// `reads_missed` at startup and nowhere else, and only sometimes,
    /// because it is a race between the first published camera position
    /// and the first chunk.
    fn wait_idle(&self) {
        let Some(rt) = &self.stack else { return };
        while !self.focused.load(std::sync::atomic::Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        rt.wait_idle();
    }

    fn reads_missed(&self) -> usize {
        self.stack.as_ref().map_or(0, |rt| rt.graph().reads_missed())
    }

    fn stats(&self) -> PlanningStats {
        self.stack.as_ref().map_or_else(PlanningStats::default, |rt| PlanningStats {
            resident_chunks: rt.graph().resident_chunks(),
            reads_missed: rt.graph().reads_missed(),
            generating: rt.is_generating(),
            layers: rt.graph().layer_stats(),
        })
    }
}

/// The LOD field is evaluated at a camera anchor quantized to this, so a
/// chunk can be up to this much further out than its nominal band. The
/// engine owns the step; this is the same number seen from the host.
const ANCHOR_SLOP_M: f32 = voxel_engine::lod_layers::ANCHOR_STEP as f32;

/// The coarsest LOD level whose chunk edge does not exceed `cap`.
///
/// A chunk is [`voxel_core::CHUNK_CELLS`] cells of [`voxel_core::BASE_VOXEL_M`]
/// at level 0 and doubles per level, so a carve horizon admits a specific
/// level rather than the horizon value itself. Derived from the constants
/// rather than written down: assuming 32 m here instead of 3.2 m
/// under-sized every reach by a factor of ten, and an empirical slack was
/// quietly covering for it.
fn largest_leaf_level(cap: f32) -> u8 {
    let base = (voxel_core::CHUNK_CELLS as f64 * voxel_core::BASE_VOXEL_M) as f32;
    let mut level = 0u8;
    while base * (1u32 << (level + 1)) as f32 <= cap {
        level += 1;
    }
    level
}

impl StackPlanner {
    /// Build the planner a level's `stack` block describes, or `None` if
    /// it declares no layers.
    pub fn new(
        def: &PlanningDef,
        level: &LevelDef,
        seed: u64,
        generator: &Arc<voxel_worldgen::Generator>,
    ) -> Option<Self> {
        if def.stack.is_empty() {
            return None;
        }
        // Every layer into ONE graph, in author order.
        let ctx = Arc::new(world::WorldCtx::new(generator.clone()));
        let mut graph = LayerGraph::with_context(seed, ctx.clone());
        let mut emitters = Vec::new();
        for layer in &def.stack {
            layer.register(&def.stack, &def.structures, &mut graph);
            if let StackLayerDef::Emit {
                name,
                max_chunk_edge_m,
                keep_m,
                emit,
                ..
            } = layer
            {
                emitters.push(Emitter::new(
                    name.clone(),
                    *max_chunk_edge_m,
                    *keep_m,
                    emit,
                ));
            }
        }
        let biome_tables: Vec<(String, Vec<String>)> = def
            .stack
            .iter()
            .filter_map(|d| match d {
                StackLayerDef::Biomes { name, table, .. } => Some((
                    name.clone(),
                    table.iter().map(|(n, _)| n.clone()).collect(),
                )),
                _ => None,
            })
            .collect();

        // One top dependency per emit layer, sized to the furthest thing
        // that can ask it for ops.
        //
        // A chunk of edge E stays a leaf only within a bounded distance
        // of the camera, so an emitter gated at edge G is never queried
        // from further than that. The carve-horizon gate stops being a
        // per-query filter and becomes a residency size — which is the
        // whole point of expressing this as a dependency graph.
        let lod = voxel_engine::LodConfig::from(&level.lod);
        let mut deps = Vec::new();
        for e in &emitters {
            // How far a leaf of edge E can be and still be drawn is the
            // engine's own bound — `resident_reach`, which the engine
            // derives and tests, rather than a second derivation here that
            // can drift from it. (It did: this used to size from `merge_k`
            // and a hysteresis gap that no longer exists, and came out
            // SHORTER than the engine's bound.) On top of that, distance
            // is measured to the chunk's near face so its far face is
            // another E out; the ops query adds the density apron and the
            // element pad; and the LOD field is evaluated at an anchor
            // quantized to ANCHOR_SLOP_M.
            //
            // So a carve-horizon gate is not a query filter any more — it
            // is a residency size, in every axis.
            let leaf = largest_leaf_level(e.gate.unwrap_or(OPS_HORIZON_EDGE_M));
            let edge = voxel_core::ChunkKey::new(leaf, bevy::math::IVec3::ZERO).edge_m() as f32;
            let ops_reach = voxel_engine::streaming::resident_reach(&lod, leaf) as f32
                + edge
                + edge / 8.0
                + layers::ELEM_PAD_M
                + ANCHOR_SLOP_M;
            // What the CARVE needs, or what a reader of the data needs,
            // whichever is further. They are different questions: carving
            // stops paying once a chunk's voxels dwarf the feature, while
            // a map or a coarse representation still wants to know where
            // the feature is. Sizing residency from the gate alone would
            // make "on the map at 40 km" cost a carve op per chunk out to
            // 40 km — measured at 132 -> 360 ops per chunk when tried.
            let reach = ops_reach.max(e.keep_m.unwrap_or(0.0));

            // Nothing else needs sizing in here any more: every other
            // consumer of these layers — ribbon surfaces, scatter
            // populations, the far forest — declares a dependency and the
            // graph takes the union.
            let size = bevy::math::IVec3::splat((2.0 * reach) as i32);
            if std::env::var_os("VOXEL_LOG_LAYERS").is_some() {
                info!("top dep {:?}: size {size:?}", e.name);
            }
            deps.push(TopDep::at_level(&e.name, 0, size));
        }
        // Presentation layers: this game's own, sitting on the emit
        // layers and turning what they plan into something it can draw.
        // Only LEVELLED ribbons become geometry. A seated ribbon is
        // ground, and the ground draws itself — see `surface_paint`. A
        // second ribbon layer for them would also be a dependency on
        // every seated emit at ITS view distance, which dragged the fine
        // road network across 80 km: 505k planning chunks and 5 fps.
        let levelled: Vec<String> = emitters
            .iter()
            .filter(|e| e.ribbons && !e.seated)
            .map(|e| e.name.clone())
            .collect();
        if let Some(top) = crate::ribbons::register(
            &mut graph,
            level,
            "ribbon-surface",
            levelled,
            crate::ribbons::RIBBON_NEAR_TILE_M,
            crate::ribbons::RIBBON_NEAR_VIEW_M,
        ) {
            deps.push(top);
        }
        let emit_names: Vec<String> = emitters.iter().map(|e| e.name.clone()).collect();
        let (scatter_tops, populations) =
            crate::scatter::register(&mut graph, level, emit_names, &biome_tables);
        deps.extend(scatter_tops);
        *ctx.populations.lock().unwrap() = Some(populations);
        deps.extend(crate::props::register_far_forest(&mut graph));

        let runtime = Arc::new(LayerRuntime::start(Arc::new(graph), deps));
        let tops = (0..runtime.tops()).map(|i| runtime.top(i)).collect();
        Some(Self {
            stack: Some(runtime),
            ctx: Some(ctx),
            tops,
            emitters,
            biome_tables,
            focused: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }
}

impl HostPlanning for StackPlanning {
    fn validate(&self, level: &LevelDef) -> Result<(), String> {
        validate_level(level)
    }

    fn build(
        &self,
        level: &LevelDef,
        seed: u64,
        generator: &Arc<voxel_worldgen::Generator>,
    ) -> Option<Arc<dyn WorldPlanner>> {
        let def = match PlanningDef::of(level) {
            Ok(def) => def,
            Err(e) => {
                error!("planning: {e}");
                return None;
            }
        };
        StackPlanner::new(&def, level, seed, generator).map(|p| Arc::new(p) as Arc<dyn WorldPlanner>)
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bevy::math::Vec3;
    use voxel_engine::{
        level::LevelDef,
        planning::{HostPlanning, WorldQuery},
    };

    use voxel_core::csg::CsgOp;

    use super::schema::{validate_level, validate_stack, StackLayerDef};
    use super::{PlanningDef, StackPlanning};

    fn shipped(name: &str) -> LevelDef {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../levels/");
        LevelDef::from_json(&std::fs::read_to_string(format!("{path}{name}")).unwrap()).unwrap()
    }

    fn validate_stack_of(level: &LevelDef) -> Result<(), String> {
        let def = PlanningDef::of(level).unwrap();
        validate_stack(&def.stack, &def.structures)
    }

    fn world_for(def: &LevelDef) -> WorldQuery {
        seeded_world_for(def, 0)
    }

    fn seeded_world_for(def: &LevelDef, seed: u64) -> WorldQuery {
        world_at(def, seed, Vec3::ZERO)
    }

    /// A world whose residency is focused where the test reads. Reads no
    /// longer generate, so a test declares its working set like the game.
    fn world_at(def: &LevelDef, seed: u64, focus: Vec3) -> WorldQuery {
        let generator = Arc::new(def.generator(seed));
        let world = WorldQuery::new(generator.clone());
        let world = match StackPlanning.build(def, seed, &generator) {
            Some(planner) => world.with_planner(planner),
            None => world,
        };
        world.set_focus(focus.as_ivec3());
        world.wait_idle();
        world
    }

    /// The whole point of per-world generator state: two levels with
    /// different programs and seeds, live in one process at the same
    /// time, each sampling its own world.
    #[test]
    fn two_worlds_coexist_in_one_process() {
        let planet = shipped("planet.json");
        let a = seeded_world_for(&planet, 0);
        let b = seeded_world_for(&planet, 0x5eed);
        let mega = shipped("megastructure.json");
        let m = world_for(&mega);

        // Same query, three different worlds — interleaved, so a global
        // "last one installed wins" would show up here.
        let probes = [
            bevy::math::Vec2::new(-27000.0, -38000.0),
            bevy::math::Vec2::new(1200.0, -400.0),
            bevy::math::Vec2::ZERO,
        ];
        for p in probes {
            let (ha, hb) = (a.generator().height(p, 1.0), b.generator().height(p, 1.0));
            assert_ne!(ha, hb, "reseeded world returned the same height at {p:?}");
            // Re-sampling after touching the others must not drift.
            assert_eq!(ha, a.generator().height(p, 1.0));
            let _ = m.generator().height(p, 1.0);
            assert_eq!(hb, b.generator().height(p, 1.0));
        }
        assert_eq!(a.generator().seed(), 0);
        assert_eq!(b.generator().seed(), 0x5eed);
        // Mega has no height op at all; its height mirror stays inert
        // while the planets keep working.
        assert_eq!(m.generator().height(probes[0], 1.0), 0.0);
        assert_ne!(a.generator().height(probes[0], 1.0), 0.0);
    }

    #[test]
    fn mega_stack_serves_pockets_and_tubes_through_world_query() {
        let mega = shipped("megastructure.json");
        let world = world_for(&mega);
        let min = Vec3::new(-1500.0, -260.0, -1500.0);
        let max = Vec3::new(1500.0, 260.0, 1500.0);
        let ops = world.ops_in(min, max, 12.8);
        assert!(!ops.is_empty(), "mega stack served no ops");
        // Room shells and tube shells: adds and cuts both present.
        assert!(ops.iter().any(|op| op.kind & 1 == 0), "no shell adds");
        assert!(ops.iter().any(|op| op.kind & 1 == 1), "no bore/room cuts");
        // Markers reach the facade.
        let min2 = bevy::math::Vec2::new(min.x, min.z);
        let max2 = bevy::math::Vec2::new(max.x, max.z);
        assert!(
            !world.markers_in(min2, max2, Some("pocket")).is_empty(),
            "no pocket markers"
        );
        // A specific marker's room shell exists near the site.
        let m = &world.markers_in(min2, max2, Some("pocket"))[0];
        let near = world.ops_in(
            m.pos - Vec3::splat(40.0),
            m.pos + Vec3::splat(40.0),
            12.8,
        );
        assert!(
            !near.is_empty(),
            "no ops within 40 m of pocket marker at {:?}",
            m.pos
        );
    }

    #[test]
    fn stack_validation_catches_authoring_errors() {
        let parse = |json: &str| -> Vec<StackLayerDef> { serde_json::from_str(json).unwrap() };
        // The shipped stacks validate.
        let planet = shipped("planet.json");
        validate_stack_of(&planet).unwrap();
        let mega = shipped("megastructure.json");
        validate_stack_of(&mega).unwrap();

        let cases: &[(&str, &str)] = &[
            // Unknown recipe name.
            (
                r#"[{"kind":"scatter","name":"s","chance":1.0},
                    {"kind":"emit","name":"e","source":"s","pad_m":0.0,
                     "emit":{"type":"site_structure","structure":"castle"}}]"#,
                "unknown structure",
            ),
            // Biome ref to a missing layer.
            (
                r#"[{"kind":"scatter","name":"s","chance":1.0,"biome":"nope:forest"}]"#,
                "not found in stack",
            ),
            // Biome name missing from the table.
            (
                r#"[{"kind":"biomes","name":"b","table":[["forest",1.0]]},
                    {"kind":"scatter","name":"s","chance":1.0,"biome":"b:desert"}]"#,
                "not in layer",
            ),
            // Source declared later (registration order).
            (
                r#"[{"kind":"connect","name":"c","source":"s"},
                    {"kind":"scatter","name":"s","chance":1.0}]"#,
                "not declared earlier",
            ),
            // Source of the wrong kind for the emit.
            (
                r#"[{"kind":"scatter","name":"s","chance":1.0},
                    {"kind":"emit","name":"e","source":"s","pad_m":0.0,
                     "emit":{"type":"worm_cuts"}}]"#,
                "expected Worm",
            ),
            // Duplicate names.
            (
                r#"[{"kind":"scatter","name":"s","chance":1.0},
                    {"kind":"scatter","name":"s","chance":0.5}]"#,
                "duplicate",
            ),
            // Empty biome table.
            (r#"[{"kind":"biomes","name":"b","table":[]}]"#, "empty table"),
            // Volumetric source with a collapsed emit y axis.
            (
                r#"[{"kind":"scatter3","name":"s","chance":1.0},
                    {"kind":"emit","name":"e","source":"s","pad_m":0.0,
                     "emit":{"type":"site_structure3","structure":"ruin"}}]"#,
                "cell_y_m",
            ),
        ];
        for (json, expect) in cases {
            let err = validate_stack(&parse(json), &PlanningDef::of(&planet).unwrap().structures).unwrap_err();
            assert!(
                err.contains(expect),
                "error {err:?} missing {expect:?} for {json}"
            );
        }
    }

    #[test]
    fn spawner_biome_refs_are_validated() {
        let mut planet = shipped("planet.json");
        validate_level(&planet).unwrap();
        if let Some(def) = planet.scatter.first_mut() {
            def.biome = Some("biomes:forrest".into());
        }
        let err = validate_level(&planet).unwrap_err();
        assert!(err.contains("forrest"), "typo not caught: {err}");
    }

    #[test]
    fn planet_stack_serves_gated_ops_through_world_query() {
        let planet = shipped("planet.json");
        // A land region large enough to hold every feature kind.
        let world = world_at(&planet, 0, Vec3::new(-27000.0, 0.0, -38000.0));
        let min = Vec3::new(-31096.0, -200.0, -42096.0);
        let max = Vec3::new(-22904.0, 500.0, -33904.0);
        let fine = world.ops_in(min, max, 12.8);
        assert!(!fine.is_empty(), "stack served no ops");
        let has_sphere_cuts = |ops: &[CsgOp]| {
            ops.iter()
                .any(|op| op.kind == voxel_core::csg::CSG_KIND_SPHERE_CUT)
        };
        assert!(has_sphere_cuts(&fine), "no cave cuts at fine LOD");
        // Coarse chunks must not see gated emitters (carve horizon).
        let coarse = world.ops_in(min, max, 500.0);
        assert!(!has_sphere_cuts(&coarse), "cave cuts leaked past the gate");
        // Clearance + water flow through the same facade.
        let min2 = bevy::math::Vec2::new(min.x, min.z);
        let max2 = bevy::math::Vec2::new(max.x, max.z);
        assert!(!world.clearance_in(min2, max2).is_empty(), "no clearance");
        assert!(!world.ribbons_in(min2, max2).is_empty(), "no ribbon segments");
        assert!(
            !world.markers_in(min2, max2, Some("ruin")).is_empty(),
            "no ruin markers"
        );
        assert!(
            !world.markers_in(min2, max2, Some("dungeon")).is_empty(),
            "no dungeon markers"
        );
        // Biomes blend through the facade: partition of unity, both
        // regions dominant somewhere across a wide sweep.
        let mut dominant = [false; 2];
        for gz in 0..12 {
            for gx in 0..12 {
                let t = bevy::math::Vec2::new(gx as f32 / 11.0, gz as f32 / 11.0);
                let p = min2 + (max2 - min2) * t;
                let w = world.biomes_at("biomes", p);
                assert_eq!(w.len(), 2);
                let sum: f32 = w.iter().map(|(_, v)| v).sum();
                assert!((sum - 1.0).abs() < 1e-4);
                for (b, (_, v)) in w.iter().enumerate() {
                    if *v > 0.7 {
                        dominant[b] = true;
                    }
                }
            }
        }
        assert!(dominant[0] && dominant[1], "biomes not regional: {dominant:?}");
        // Determinism across a fresh build.
        let world2 = world_at(&planet, 0, Vec3::new(-27000.0, 0.0, -38000.0));
        assert_eq!(fine, world2.ops_in(min, max, 12.8));
    }
}
