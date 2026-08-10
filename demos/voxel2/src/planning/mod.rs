//! This demo's planning: JSON-authored LayerProcGen layers.
//!
//! `voxel-layers` is the framework — dependency management, threaded
//! chunk generation, spatial organisation — and the concrete layers are
//! the game's, so they live here. [`layers`] holds the layer
//! implementations, [`structure`] the grammar one of them builds with,
//! and [`schema`] the JSON vocabulary that composes them. The engine sees
//! only [`RegionPlanning`] through [`HostPlanning`].

pub mod layers;
pub mod nodes;
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

use schema::{validate_level, EmitDef};

/// This demo's planning: build layers from the level's region nodes.
pub struct RegionPlanning(pub bevy::reflect::TypeRegistryArc);

/// Vertical band xz-facade queries cover: enough for any current world
/// (the deepest LOD tree spans ~±2.5 km), small enough that volumetric
/// emit layers don't enumerate thousands of 132 m y-rows per query.
const FACADE_Y_M: f32 = 2_560.0;

/// The level's region nodes as a [`WorldPlanner`]: one `LayerManager`
/// holding every layer they describe, plus the bookkeeping needed to
/// answer a query without generating layers that cannot contribute to it.
///
/// This is *a* host's planner, not the engine's: it turns the
/// region-domain half of a level's node list into layers. A game with
/// hand-written layers implements [`WorldPlanner`] itself and never
/// touches this.
#[derive(Clone, Default)]
pub struct RegionPlanner {
    /// One graph for every layer the level's region nodes describe, plus
    /// the thread keeping its top dependencies satisfied.
    graph: Option<Arc<LayerRuntime>>,
    /// Handles for the top dependencies that follow the camera.
    tops: Vec<TopHandle>,
    /// What this game's layers share, so its systems can read what they
    /// published.
    ctx: Option<Arc<world::WorldCtx>>,
    /// Emit instances and what each one can produce.
    emitters: Vec<Emitter>,
    /// The level's populations, compiled: what a level wrote as nodes,
    /// in the form the placer and the cover painter read.
    pub populations: Vec<voxel_engine::level::ScatterDef>,
    /// Biome layers: (instance name, ordered biome names).
    /// Region dictionaries: (instance, [(region name, its material)]).
    /// A region IS a generator band; this only names them.
    biome_tables: Vec<(String, Vec<(String, u32)>)>,
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

impl WorldPlanner for RegionPlanner {
    /// Gated emitters drop out wholesale for coarse chunks — the gate is
    /// per chunk, never per op.
    fn ops_in(&self, min: Vec3, max: Vec3, chunk_edge_m: f32) -> Vec<CsgOp> {
        let mut out = Vec::new();
        if let Some(rt) = &self.graph {
            let mgr = rt.graph();
            for e in &self.emitters {
                if e.gate.is_none_or(|g| chunk_edge_m <= g) {
                    out.extend(layers::patches_in(mgr, &e.name, min, max).ops);
                }
            }
        }
        out
    }

    fn as_any(&self) -> &(dyn std::any::Any + Send + Sync) {
        self
    }

    /// What this game thinks is worth looking at, as lines.
    ///
    /// The overlay budgets and orders them; deciding that a marker is a
    /// 30 m stake, a clearance bed is a ground-following segment and a
    /// weight field is a grid of stakes coloured by dominant member is
    /// entirely this host's business.
    fn debug_lines(
        &self,
        min: bevy::math::Vec2,
        max: bevy::math::Vec2,
    ) -> Vec<voxel_engine::planning::DebugLine> {
        use voxel_engine::planning::DebugLine;
        let mut out = Vec::new();
        let line = |a: Vec3, b: Vec3, color: [f32; 3]| DebugLine { a, b, color };

        for m in self.markers_in(min, max, None) {
            out.push(line(m.pos, m.pos + Vec3::Y * 30.0, kind_color(&m.kind)));
        }
        let gen = self.ctx.as_ref().map(|c| c.generator.clone());
        let Some(gen) = gen else { return out };
        let h = |p: bevy::math::Vec2| gen.height(p, 1.0) + 1.0;
        for seg in self.clearance_in(min, max) {
            out.push(line(
                Vec3::new(seg[0].x, h(seg[0]), seg[0].y),
                Vec3::new(seg[1].x, h(seg[1]), seg[1].y),
                [1.0, 0.8, 0.2],
            ));
        }
        for w in self.ribbons_in(min, max) {
            out.push(line(
                Vec3::new(w.a.x, w.levels[0] + 0.5, w.a.y),
                Vec3::new(w.b.x, w.levels[1] + 0.5, w.b.y),
                [0.2, 0.6, 1.0],
            ));
        }

        // Weight fields as a 17x17 readout of the near range: a field
        // sample, not a feature set, so it does not follow the overlay
        // out to where one stake per 5 km would alias it into noise.
        let c = (min + max) * 0.5;
        let step = (max.x - min.x) / 16.0;
        for name in self.weight_fields() {
            for gz in -8..=8 {
                for gx in -8..=8 {
                    let p = c + bevy::math::Vec2::new(gx as f32, gz as f32) * step;
                    let weights = self.weights_at(&name, p);
                    let Some((i, (_, w))) = weights
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
                    else {
                        continue;
                    };
                    let y = gen.height(p, 8.0) + 2.0;
                    let hue = (i as f32 * 137.5) % 360.0;
                    out.push(line(
                        Vec3::new(p.x, y, p.y),
                        Vec3::new(p.x, y + 4.0 + 12.0 * w, p.y),
                        hsl_rgb(hue, 0.8, 0.5),
                    ));
                }
            }
        }
        out
    }

    /// Answer `voxctl`. The engine forwards the query without reading it.
    fn inspect(&self, query: &serde_json::Value) -> serde_json::Value {
        use serde_json::json;
        let f = |k: &str, i: usize| -> f32 {
            query
                .get(k)
                .and_then(|v| v.get(i))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0) as f32
        };
        let r = query
            .get("radius")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(512.0) as f32;
        let c = bevy::math::Vec2::new(f("center", 0), f("center", 1));
        let (min, max) = (
            c - bevy::math::Vec2::splat(r),
            c + bevy::math::Vec2::splat(r),
        );
        match query.get("kind").and_then(serde_json::Value::as_str) {
            Some("ribbons") => {
                let segs: Vec<_> = self
                    .ribbons_in(min, max)
                    .iter()
                    .map(|s| {
                        json!({
                            "a": [s.a.x, s.a.y],
                            "b": [s.b.x, s.b.y],
                            "half_w": s.half_w,
                            "levels": s.levels,
                        })
                    })
                    .collect();
                json!({"count": segs.len(), "segments": segs})
            }
            Some("markers") => {
                let kind = query.get("of").and_then(serde_json::Value::as_str);
                let found: Vec<_> = self
                    .markers_in(min, max, kind)
                    .iter()
                    .map(|m| json!({"pos": [m.pos.x, m.pos.y, m.pos.z], "kind": m.kind}))
                    .collect();
                json!({"count": found.len(), "markers": found})
            }
            // Region weights on a grid: what the ground is, where, and
            // how firmly. Used to place cameras and demo starts by
            // evidence instead of guessing at a noise field.
            Some("regions") => {
                let step = query
                    .get("step")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(f64::from(r) / 4.0) as f32;
                let n = ((2.0 * r / step.max(1.0)) as i32).clamp(1, 64);
                let mut totals: std::collections::BTreeMap<String, f32> =
                    std::collections::BTreeMap::new();
                let mut best: Option<(f32, [f32; 3], String)> = None;
                let want = query.get("of").and_then(serde_json::Value::as_str);
                // A region search that ignores altitude finds the sea
                // floor: most of a region's area can be under water.
                let alt = query
                    .get("altitude")
                    .and_then(|v| Some([v.get(0)?.as_f64()? as f32, v.get(1)?.as_f64()? as f32]));
                let gen = self.ctx.as_ref().map(|c| c.generator.clone());
                for gz in 0..n {
                    for gx in 0..n {
                        let p = bevy::math::Vec2::new(
                            min.x + (max.x - min.x) * gx as f32 / (n - 1).max(1) as f32,
                            min.y + (max.y - min.y) * gz as f32 / (n - 1).max(1) as f32,
                        );
                        let w = self.weights_at("biomes", p);
                        for (name, v) in &w {
                            *totals.entry(name.clone()).or_default() += v;
                        }
                        let in_band = alt.is_none_or(|[lo, hi]| {
                            gen.as_ref().is_some_and(|g| {
                                let h = g.height(p, 8.0);
                                h >= lo && h <= hi
                            })
                        });
                        if !in_band {
                            continue;
                        }
                        if let Some((name, v)) = w
                            .iter()
                            .filter(|(name, _)| want.is_none_or(|k| k == name))
                            .max_by(|a, b| a.1.total_cmp(&b.1))
                        {
                            if best.as_ref().is_none_or(|(bv, _, _)| v > bv) {
                                let h = gen.as_ref().map_or(0.0, |g| g.height(p, 8.0));
                                best = Some((*v, [p.x, h, p.y], name.clone()));
                            }
                        }
                    }
                }
                let samples = (n * n) as f32;
                let share: serde_json::Map<String, serde_json::Value> = totals
                    .into_iter()
                    .map(|(k, v)| (k, json!((v / samples * 1000.0).round() / 10.0)))
                    .collect();
                let (bw, bp, bn) = best.unwrap_or((0.0, [0.0, 0.0, 0.0], String::new()));
                json!({
                    "samples": n * n,
                    "share_pct": share,
                    "strongest": {"region": bn, "weight": bw, "pos": bp},
                })
            }
            // Live placements per class: what "push the density up"
            // actually produced.
            Some("scatter") => {
                let counts = self
                    .ctx
                    .as_ref()
                    .map(|c| c.placements.lock().unwrap().clone())
                    .unwrap_or_default();
                let total: usize = counts.values().sum();
                json!({"total": total, "by_class": counts})
            }
            other => json!({"error": format!("unknown inspect kind {other:?}")}),
        }
    }

    fn set_focus(&self, focus: bevy::math::IVec3) {
        for top in &self.tops {
            top.set_focus(focus);
        }
        self.focused
            .store(true, std::sync::atomic::Ordering::Release);
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
        let Some(rt) = &self.graph else { return };
        while !self.focused.load(std::sync::atomic::Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        rt.wait_idle();
    }

    fn is_idle(&self) -> bool {
        self.graph.as_ref().is_none_or(|rt| {
            self.focused.load(std::sync::atomic::Ordering::Acquire) && rt.is_idle()
        })
    }

    fn reads_missed(&self) -> usize {
        self.graph
            .as_ref()
            .map_or(0, |rt| rt.graph().reads_missed())
    }

    fn stats(&self) -> PlanningStats {
        self.graph
            .as_ref()
            .map_or_else(PlanningStats::default, |rt| PlanningStats {
                resident_chunks: rt.graph().resident_chunks(),
                reads_missed: rt.graph().reads_missed(),
                generating: rt.is_generating(),
                layers: rt.graph().layer_stats(),
            })
    }
}

/// Stable color per marker kind (hash to hue) — this game's palette.
fn kind_color(kind: &str) -> [f32; 3] {
    let mut h = 0u32;
    for b in kind.bytes() {
        h = h.wrapping_mul(31).wrapping_add(b as u32);
    }
    hsl_rgb((h % 360) as f32, 0.9, 0.6)
}

fn hsl_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c * 0.5;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

/// What this HOST asks its own planner. None of it is engine vocabulary:
/// a ribbon, a marker, a clearance bed and a weight field are this
/// game's nouns, reached through `WorldQuery::planner_as`.
impl RegionPlanner {
    pub fn clearance_in(
        &self,
        min: bevy::math::Vec2,
        max: bevy::math::Vec2,
    ) -> Vec<[bevy::math::Vec2; 2]> {
        let (min3, max3) = (
            Vec3::new(min.x, -FACADE_Y_M, min.y),
            Vec3::new(max.x, FACADE_Y_M, max.y),
        );
        let mut out = Vec::new();
        if let Some(rt) = &self.graph {
            let mgr = rt.graph();
            for e in self.emitters.iter().filter(|e| e.clearance) {
                out.extend(layers::patches_in(mgr, &e.name, min3, max3).clearance);
            }
        }
        out
    }

    pub fn ribbons_in(&self, min: bevy::math::Vec2, max: bevy::math::Vec2) -> Vec<RibbonSeg> {
        let (min3, max3) = (
            Vec3::new(min.x, -FACADE_Y_M, min.y),
            Vec3::new(max.x, FACADE_Y_M, max.y),
        );
        let mut out = Vec::new();
        if let Some(rt) = &self.graph {
            let mgr = rt.graph();
            for e in self.emitters.iter().filter(|e| e.ribbons) {
                out.extend(layers::patches_in(mgr, &e.name, min3, max3).ribbons);
            }
        }
        out
    }

    /// The material of the region this population is gated on. See
    /// [`crate::scatter::gate_material`].
    pub fn gate_material(&self, def: &voxel_engine::level::ScatterDef) -> Option<u32> {
        crate::scatter::gate_material(def, &self.biome_tables)
    }

    pub fn weight_fields(&self) -> Vec<String> {
        self.biome_tables.iter().map(|(n, _)| n.clone()).collect()
    }

    pub fn weights_at(&self, instance: &str, p: bevy::math::Vec2) -> Vec<(String, f32)> {
        let (Some(ctx), Some(table)) = (
            self.ctx.as_ref(),
            self.biome_tables
                .iter()
                .find_map(|(n, t)| (n == instance).then_some(t)),
        ) else {
            return Vec::new();
        };
        table
            .iter()
            .map(|(name, material)| {
                (
                    name.clone(),
                    ctx.generator.surface_material_weight(p, 8.0, *material),
                )
            })
            .collect()
    }

    pub fn markers_in(
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
        if let Some(rt) = &self.graph {
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

impl RegionPlanner {
    /// Build the planner the level's region nodes describe, or `None` if
    /// it declares none.
    ///
    /// The nodes come from the same list as the generator's — one document,
    /// one order, one way of naming things. What each layer needs from the
    /// others arrives through a port the graph compiler already checked, so
    /// nothing here scans the list looking for a name.
    pub fn new(
        level: &LevelDef,
        seed: u64,
        generator: &Arc<voxel_worldgen::Generator>,
        registry: &bevy::reflect::TypeRegistryArc,
    ) -> Option<Self> {
        let region: Vec<&voxel_engine::graph::NodeDef> = level
            .nodes
            .iter()
            .filter(|n| n.node.0.domain() == voxel_engine::graph::Domain::Region)
            .collect();
        if region.is_empty() {
            return None;
        }
        let by_name: bevy::platform::collections::HashMap<String, &dyn voxel_engine::graph::Node> =
            region
                .iter()
                .filter_map(|n| Some((n.name.clone()?, &*n.node.0)))
                .collect();

        // Every layer into ONE graph, in author order.
        let ctx = Arc::new(world::WorldCtx::new(generator.clone()));
        let mut graph = LayerGraph::with_context(seed, ctx.clone());
        let mut emitters = Vec::new();
        for node in &region {
            let Some(name) = node.name.as_deref() else {
                continue;
            };
            let rctx = nodes::RegionCtx {
                name,
                wires: &node.wires,
                by_name: &by_name,
            };
            // Dispatched through the registry, not a match: the kinds are
            // open, and a kind that forgot `#[reflect(RegionLayer)]` is a
            // named warning rather than a layer that silently is not there.
            match registry
                .read()
                .get_type_data::<nodes::ReflectRegionLayer>(node.node.0.as_any().type_id())
                .and_then(|d| d.get(node.node.0.as_reflect()))
            {
                Some(layer) => layer.register(&rctx, &mut graph),
                None => warn!("planning: '{name}' is a region node but implements no RegionLayer"),
            }
            if let Some(emit) = node.node.0.as_any().downcast_ref::<nodes::Emit>() {
                emitters.push(Emitter::new(
                    name.to_string(),
                    emit.max_chunk_edge_m,
                    emit.keep_m,
                    &emit.emit,
                ));
            }
        }
        let biome_tables: Vec<(String, Vec<(String, u32)>)> = region
            .iter()
            .filter_map(|n| {
                let b = n.node.0.as_any().downcast_ref::<nodes::Biomes>()?;
                Some((n.name.clone()?, b.table.clone()))
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
            deps.push(TopDep::new(&e.name, size));
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
        // Field slots come from the compiler, which is also what checked
        // this level before it got here: a population names the field node
        // it reads instead of writing the slot number the generator wrote.
        let fields = voxel_engine::graph::compile(&level.nodes)
            .expect("the level compiled when it was validated")
            .fields;
        let defs = nodes::populations(&level.nodes, &fields);
        let emit_names: Vec<String> = emitters.iter().map(|e| e.name.clone()).collect();
        let (scatter_tops, populations) =
            crate::scatter::register(&mut graph, &defs, emit_names, &biome_tables);
        deps.extend(scatter_tops);
        *ctx.populations.lock().unwrap() = Some(populations);

        let runtime = Arc::new(LayerRuntime::start(Arc::new(graph), deps));
        let tops = (0..runtime.tops()).map(|i| runtime.top(i)).collect();
        Some(Self {
            graph: Some(runtime),
            ctx: Some(ctx),
            tops,
            emitters,
            populations: defs,
            biome_tables,
            focused: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }
}

impl HostPlanning for RegionPlanning {
    fn validate(&self, level: &LevelDef) -> Result<(), String> {
        validate_level(level)
    }

    fn build(
        &self,
        level: &LevelDef,
        seed: u64,
        generator: &Arc<voxel_worldgen::Generator>,
    ) -> Option<Arc<dyn WorldPlanner>> {
        RegionPlanner::new(level, seed, generator, &self.0)
            .map(|p| Arc::new(p) as Arc<dyn WorldPlanner>)
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

    use super::schema::validate_level;
    use super::{RegionPlanner, RegionPlanning};

    /// Engine kinds plus this game's, which is what a level names.
    fn shipped(name: &str) -> LevelDef {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../levels/");
        LevelDef::from_json(
            &std::fs::read_to_string(format!("{path}{name}")).unwrap(),
            &super::nodes::kinds(),
        )
        .unwrap()
    }

    fn world_for(def: &LevelDef) -> WorldQuery {
        seeded_world_for(def, 0)
    }

    fn seeded_world_for(def: &LevelDef, seed: u64) -> WorldQuery {
        world_at(def, seed, Vec3::ZERO)
    }

    /// A world whose residency is focused where the test reads. Reads no
    /// longer generate, so a test declares its working set like the game.
    pub(super) fn world_at(def: &LevelDef, seed: u64, focus: Vec3) -> WorldQuery {
        let generator = Arc::new(def.generator(seed));
        let world = WorldQuery::new(generator.clone());
        let world = match RegionPlanning(super::nodes::kinds()).build(def, seed, &generator) {
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
        let planner = world.planner_as::<RegionPlanner>().expect("region planner");
        let min = Vec3::new(-1500.0, -260.0, -1500.0);
        let max = Vec3::new(1500.0, 260.0, 1500.0);
        let ops = world.ops_in(min, max, 12.8);
        assert!(!ops.is_empty(), "the megastructure served no ops");
        // Room shells and tube shells: adds and cuts both present.
        assert!(ops.iter().any(|op| op.kind & 1 == 0), "no shell adds");
        assert!(ops.iter().any(|op| op.kind & 1 == 1), "no bore/room cuts");
        // Markers reach the facade.
        let min2 = bevy::math::Vec2::new(min.x, min.z);
        let max2 = bevy::math::Vec2::new(max.x, max.z);
        assert!(
            !planner.markers_in(min2, max2, Some("pocket")).is_empty(),
            "no pocket markers"
        );
        // A specific marker's room shell exists near the site.
        let m = &planner.markers_in(min2, max2, Some("pocket"))[0];
        let near = world.ops_in(m.pos - Vec3::splat(40.0), m.pos + Vec3::splat(40.0), 12.8);
        assert!(
            !near.is_empty(),
            "no ops within 40 m of pocket marker at {:?}",
            m.pos
        );
    }

    /// The graph compiler refuses what a hand-written validator used to.
    ///
    /// These are the same authoring mistakes, checked one layer down: a
    /// source naming nothing, naming the wrong KIND of thing, or naming
    /// something written later are all port errors now, so the hundred
    /// lines that used to look for them are gone rather than duplicated.
    #[test]
    fn the_compiler_refuses_a_badly_wired_stack() {
        let compile = |json: &str| -> String {
            let nodes: Vec<voxel_engine::graph::NodeDef> =
                voxel_engine::graph::with_registry(&super::nodes::kinds(), || serde_json::from_str(json))
                    .expect("parses");
            voxel_engine::graph::compile(&nodes)
                .expect_err("should not compile")
                .to_string()
        };

        let cases: &[(&str, &str)] = &[
            // A structure nothing defines.
            (
                r#"[{"kind":"scatter","name":"s","chance":1.0},
                    {"kind":"emit","name":"e","in":{"source":"s","structure":"castle"},
                     "pad_m":0.0,"emit":{"type":"site_structure"}}]"#,
                "not a node",
            ),
            // A biome layer nothing declares.
            (
                r#"[{"kind":"scatter","name":"s","chance":1.0,"biome":"forest",
                     "in":{"biome":"nope"}}]"#,
                "not a node",
            ),
            // A source written later than its consumer.
            (
                r#"[{"kind":"connect","name":"c","in":{"source":"s"}},
                    {"kind":"scatter","name":"s","chance":1.0}]"#,
                "written later",
            ),
            // A source of the wrong kind: worm cuts want burrows, not sites.
            (
                r#"[{"kind":"scatter","name":"s","chance":1.0},
                    {"kind":"emit","name":"e","in":{"source":"s"},"pad_m":0.0,
                     "emit":{"type":"worm_cuts"}}]"#,
                "wired to a",
            ),
            // Two nodes with one name.
            (
                r#"[{"kind":"scatter","name":"s","chance":1.0},
                    {"kind":"scatter","name":"s","chance":0.5}]"#,
                "two nodes are called",
            ),
        ];
        for (json, want) in cases {
            let err = compile(json);
            assert!(err.contains(want), "expected {want:?} in {err:?}");
        }
    }

    /// And what a port cannot say, `validate_level` still does.
    #[test]
    fn validation_keeps_what_the_compiler_cannot_check() {
        let planet = shipped("planet.json");
        validate_level(&planet).unwrap();
        validate_level(&shipped("megastructure.json")).unwrap();

        // A biome MEMBER is a parameter, not a wire, so nothing upstream
        // checks it names a region the wired table actually has.
        // The megastructure's habitation pockets are the one gated layer
        // the shipped levels have.
        let mut bad = shipped("megastructure.json");
        let mut found = false;
        for node in &mut bad.nodes {
            if let Some(s) = node
                .node
                .0
                .as_any_mut()
                .downcast_mut::<super::nodes::Scatter3>()
            {
                if s.biome.is_some() {
                    s.biome = Some("nosuchregion".into());
                    found = true;
                    break;
                }
            }
        }
        assert!(found, "megastructure gates its pockets on a district");
        let err = validate_level(&bad).unwrap_err();
        assert!(err.contains("nosuchregion"), "{err}");

        // How far a structure reaches from its site is geometry, not
        // wiring: a port says a structure IS named, and only measuring the
        // variants says the emit index can still find it.
        let mut far = shipped("planet.json");
        let mut found = false;
        for node in &mut far.nodes {
            if let Some(s) = node
                .node
                .0
                .as_any_mut()
                .downcast_mut::<super::nodes::Structure>()
            {
                s.size = [s.size[0] * 100.0, s.size[1] * 100.0];
                found = true;
                break;
            }
        }
        assert!(found, "planet builds structures");
        let err = validate_level(&far).unwrap_err();
        assert!(err.contains("element padding"), "{err}");
    }

    /// Tuning a population must not restream the world.
    ///
    /// The engine's rule proved on a real level: planet's populations are
    /// the only nodes in it that reach no voxel, and every other kind it
    /// ships has to keep the conservative answer. Written against the
    /// SHIPPED nodes rather than a fixture, because what this protects is
    /// a live edit to this level.
    #[test]
    fn tuning_a_population_does_not_restream_the_world() {
        use voxel_engine::graph::{changed, node::Invalidates};
        let planet = shipped("planet.json");
        assert_eq!(changed(&planet.nodes, &planet.nodes), None);

        let mut edited = planet.clone();
        let population = edited
            .nodes
            .iter_mut()
            .find_map(|n| n.node.0.as_any_mut().downcast_mut::<super::nodes::Population>())
            .expect("planet ships populations");
        population.0.per_tile += 1;
        assert_eq!(
            changed(&edited.nodes, &planet.nodes),
            Some(Invalidates::Plan),
            "props are entities — the voxels would come back identical"
        );

        // And every other kind keeps the conservative answer, so a new one
        // is only cheap when somebody says so in its own impl.
        for node in &planet.nodes {
            let is_population = node
                .node
                .0
                .as_any()
                .downcast_ref::<super::nodes::Population>()
                .is_some();
            assert_eq!(
                node.node.0.invalidates() == Invalidates::Plan,
                is_population,
                "{:?} of kind '{}'",
                node.name,
                node.node.0.kind()
            );
        }
    }

    /// A population is placement-only: classes and variants, no models.
    ///
    /// Asserted as PROPERTIES, not as a list — which classes a level ships
    /// is content and changes whenever the world is dressed. What must
    /// hold is that the compiler fills in the three things a level stopped
    /// writing.
    #[test]
    fn planet_compiles_the_populations_it_ships() {
        let planet = shipped("planet.json");
        let program = voxel_engine::graph::compile(&planet.nodes).unwrap();
        let defs = super::nodes::populations(&planet.nodes, &program.fields);

        let classes: Vec<&str> = defs.iter().map(|d| d.class.as_str()).collect();
        for want in ["tree", "boulder", "groundcover"] {
            assert!(classes.contains(&want), "planet lost its {want} population");
        }
        // A class is the NODE's name, so the compiler rejects a duplicate
        // before it can become two layers registered under one name.
        let mut unique = classes.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), classes.len(), "duplicate class: {classes:?}");

        // Ground cover is just another population that outputs points
        // instead of entities.
        use voxel_engine::level::ScatterOutput;
        assert!(defs.iter().any(|d| d.output == ScatterOutput::Points));

        // The wiring, in the form the placer reads it.
        let tree = defs.iter().find(|d| d.class == "tree").unwrap();
        assert_eq!(tree.gate.as_deref(), Some("biomes:forest"));
        assert_eq!(
            tree.density.as_ref().map(|d| d.field),
            Some(0),
            "the tree density wire must resolve to the field node's slot"
        );

        assert!(
            super::nodes::populations(&shipped("megastructure.json").nodes, &program.fields)
                .is_empty(),
            "the megastructure scatters nothing"
        );
    }

    /// A population's gate is half wire, half parameter, and only the
    /// wire half is the compiler's. The member has to be checked here or
    /// a typo is a population that silently grows nowhere.
    #[test]
    fn spawner_biome_refs_are_validated() {
        let mut planet = shipped("planet.json");
        validate_level(&planet).unwrap();
        let found = planet.nodes.iter_mut().any(|n| {
            match n
                .node
                .0
                .as_any_mut()
                .downcast_mut::<super::nodes::Population>()
            {
                Some(p) if p.0.region.is_some() => {
                    p.0.region = Some("forrest".into());
                    true
                }
                _ => false,
            }
        });
        assert!(found, "planet gates a population on a region");
        let err = validate_level(&planet).unwrap_err();
        assert!(err.contains("forrest"), "typo not caught: {err}");
    }

    #[test]
    fn planet_stack_serves_gated_ops_through_world_query() {
        let planet = shipped("planet.json");
        // A land region large enough to hold every feature kind.
        let world = world_at(&planet, 0, Vec3::new(-27000.0, 0.0, -38000.0));
        let planner = world.planner_as::<RegionPlanner>().expect("region planner");
        let min = Vec3::new(-31096.0, -200.0, -42096.0);
        let max = Vec3::new(-22904.0, 500.0, -33904.0);
        let fine = world.ops_in(min, max, 12.8);
        assert!(!fine.is_empty(), "the planner served no ops");
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
        assert!(!planner.clearance_in(min2, max2).is_empty(), "no clearance");
        assert!(
            !planner.ribbons_in(min2, max2).is_empty(),
            "no ribbon segments"
        );
        assert!(
            !planner.markers_in(min2, max2, Some("ruin")).is_empty(),
            "no ruin markers"
        );
        assert!(
            !planner.markers_in(min2, max2, Some("dungeon")).is_empty(),
            "no dungeon markers"
        );
        // Regions blend through the facade: partition of unity, and
        // every one of them dominant somewhere across a wide sweep.
        //
        // Densely, because the mountain range is a NARROW band — an
        // iso-strip of the noise field, a tenth of the world — and a
        // coarse grid can step right over it.
        const N: usize = 48;
        let mut dominant = [false; 5];
        for gz in 0..N {
            for gx in 0..N {
                let t =
                    bevy::math::Vec2::new(gx as f32 / (N - 1) as f32, gz as f32 / (N - 1) as f32);
                let p = min2 + (max2 - min2) * t;
                let w = planner.weights_at("biomes", p);
                assert_eq!(w.len(), 5, "planet declares five regions");
                let sum: f32 = w.iter().map(|(_, v)| v).sum();
                assert!((sum - 1.0).abs() < 1e-4);
                for (b, (_, v)) in w.iter().enumerate() {
                    if *v > 0.7 {
                        dominant[b] = true;
                    }
                }
            }
        }
        assert!(
            dominant.iter().all(|&d| d),
            "every region must dominate somewhere: {dominant:?}"
        );
        // Determinism across a fresh build.
        let world2 = world_at(&planet, 0, Vec3::new(-27000.0, 0.0, -38000.0));
        assert_eq!(fine, world2.ops_in(min, max, 12.8));
    }
}

#[cfg(test)]
mod output_is_unchanged {
    use bevy::math::{Vec2, Vec3};
    use voxel_engine::level::LevelDef;

    /// Exactly what the planner serves in a fixed window of each level.
    ///
    /// The generator has a bit-identity net and planning has none, so this
    /// is it: the numbers were taken before planning moved into the node
    /// list, and they have to survive it. A change here is either a bug or
    /// a decision somebody should be making on purpose.
    #[test]
    fn planning_serves_exactly_what_it_did() {
        for (name, focus, half, want) in [
            (
                "planet.json",
                Vec3::new(-27000.0, 0.0, -38000.0),
                4096.0,
                (50200, 5809, 5809, 146),
            ),
            ("megastructure.json", Vec3::ZERO, 2048.0, (18778, 0, 0, 772)),
            (
                "purgatory.json",
                Vec3::new(-5604.0, 0.0, 5660.0),
                4096.0,
                (5012, 1269, 1277, 51),
            ),
        ] {
            let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../levels/");
            let level = LevelDef::from_json(
                &std::fs::read_to_string(format!("{path}{name}")).unwrap(),
                &crate::planning::nodes::kinds(),
            )
            .unwrap();
            let world = super::tests::world_at(&level, 0, focus);
            let min = focus - Vec3::splat(half);
            let max = focus + Vec3::splat(half);
            let (min2, max2) = (Vec2::new(min.x, min.z), Vec2::new(max.x, max.z));
            let planner = world.planner_as::<super::RegionPlanner>().unwrap();
            let got = (
                world.ops_in(min, max, 12.8).len(),
                planner.clearance_in(min2, max2).len(),
                planner.ribbons_in(min2, max2).len(),
                planner.markers_in(min2, max2, None).len(),
            );
            assert_eq!(got, want, "{name}: (ops, clearance, ribbons, markers)");
        }
    }
}
