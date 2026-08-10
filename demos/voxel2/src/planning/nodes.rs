//! This game's planning vocabulary, as nodes.
//!
//! The engine owns [`Node`] and nothing else about what a kind is. These
//! are the game's, in the game's crate, and the engine never learns that a
//! road or a river exists — it sees a region-domain node with ports and
//! hands the whole set to [`super::StackPlanning`] to turn into layers.
//!
//! What each replaces: a variant of a `StackLayerDef` enum, wired by
//! `"source": "sites:ruins"` strings that nothing checked. A source naming
//! a layer that does not exist used to parse, build, and quietly produce
//! nothing.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use voxel_core::opgen::{Port, Value};
use voxel_engine::graph::node::{Domain, Invalidates, Node, Ports, ReflectNode};
use voxel_engine::graph::{Wire, Wires};
use voxel_layers::LayerGraph;

use super::layers::*;
use super::schema::{
    d_altitude, d_burial, d_cell, d_cell3, d_cell3_y, d_corridor, d_flow_cell, d_flow_steps,
    d_margin, d_margin3, d_path_step, d_reach, d_reach3, d_slope_penalty, d_spill, d_up_interval,
    d_worm_radius, d_worm_steps,
};
use super::schema::{EmitDef, RelaxDef, StructureDef, VariantDef};
use voxel_engine::level::ScatterDef;

/// A named structure the stack can build at a site.
///
/// A node like everything else, so `site_structure` REFERS to one by name
/// through a port the compiler checks, rather than by a string looked up in
/// a side table that could be missing.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, RegionLayer, Serialize, Deserialize, Default)]
pub struct Structure {
    /// Sampled once per site; arrangements scale their radius by it, so a
    /// structure's parts agree with each other.
    pub size: [f32; 2],
    pub variants: Vec<VariantDef>,
}

impl RegionLayer for Structure {
    /// A definition, not a layer: what builds it is the emit wired to it.
    fn register(&self, _ctx: &RegionCtx, _mgr: &mut LayerGraph) {}
}

impl Node for Structure {
    fn kind(&self) -> &'static str {
        "structure"
    }
    fn domain(&self) -> Domain {
        Domain::Region
    }
    fn ports(&self) -> Ports {
        (&[], &[("structure", Value::Host("structure"))])
    }
}

/// Hash-gated candidate sites per cell, filtered by terrain.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, RegionLayer, Serialize, Deserialize, Default)]
pub struct Scatter {
    #[serde(default = "d_cell")]
    pub cell_m: i32,
    pub chance: f32,
    #[serde(default = "d_margin")]
    pub margin_m: f32,
    #[serde(default = "d_altitude")]
    pub altitude: [f32; 2],
    #[serde(default = "d_up_interval")]
    pub up: [f32; 2],
    /// WHICH member of the wired biomes table gates this layer. The
    /// instance is the wire; this is the name inside it.
    #[serde(default)]
    pub biome: Option<String>,
    /// Push sites apart from their neighbours after scattering.
    #[serde(default)]
    pub relax: Option<RelaxDef>,
}

impl Node for Scatter {
    fn kind(&self) -> &'static str {
        "scatter"
    }
    fn domain(&self) -> Domain {
        Domain::Region
    }
    fn ports(&self) -> Ports {
        // The gate is a port only when this layer has one to name.
        let ins: &'static [Port] = match self.biome {
            Some(_) => &[("biome", Value::Host("biomes"))],
            None => &[],
        };
        (ins, &[("sites", Value::Host("sites"))])
    }
}

/// Pathfound links between sites of a scatter instance.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, RegionLayer, Serialize, Deserialize, Default)]
pub struct Connect {
    #[serde(default = "d_cell")]
    pub cell_m: i32,
    #[serde(default = "d_reach")]
    pub reach_m: f32,
    #[serde(default = "d_corridor")]
    pub corridor_m: f32,
    #[serde(default = "d_slope_penalty")]
    pub slope_penalty: f32,
    /// Pathfinding lattice step (meters). The cost of a route is
    /// quadratic in this, so a long-range corridor is planned on a
    /// coarse lattice and a local track on a fine one — which is what
    /// makes the same layer kind serve both scales.
    #[serde(default = "d_path_step")]
    pub step_m: f32,
}

impl Node for Connect {
    fn kind(&self) -> &'static str {
        "connect"
    }
    fn domain(&self) -> Domain {
        Domain::Region
    }
    fn ports(&self) -> Ports {
        (
            &[("source", Value::Host("sites"))],
            &[("paths", Value::Host("paths"))],
        )
    }
}

/// Descent courses (pond-and-spill hydrology) from sites.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, RegionLayer, Serialize, Deserialize, Default)]
pub struct Flow {
    #[serde(default = "d_flow_cell")]
    pub cell_m: i32,
    #[serde(default = "d_flow_steps")]
    pub max_steps: usize,
    #[serde(default = "d_spill")]
    pub max_spill_rise: f32,
}

impl Node for Flow {
    fn kind(&self) -> &'static str {
        "flow"
    }
    fn domain(&self) -> Domain {
        Domain::Region
    }
    fn ports(&self) -> Ports {
        (
            &[("source", Value::Host("sites"))],
            &[("courses", Value::Host("courses"))],
        )
    }
}

/// Noise-steered burrows from sites.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, RegionLayer, Serialize, Deserialize, Default)]
pub struct Worm {
    #[serde(default = "d_cell")]
    pub cell_m: i32,
    #[serde(default = "d_worm_steps")]
    pub steps: u32,
    #[serde(default = "d_worm_radius")]
    pub radius: [f32; 2],
    #[serde(default = "d_burial")]
    pub burial_radii: f32,
}

impl Node for Worm {
    fn kind(&self) -> &'static str {
        "worm"
    }
    fn domain(&self) -> Domain {
        Domain::Region
    }
    fn ports(&self) -> Ports {
        (
            &[("source", Value::Host("sites"))],
            &[("burrows", Value::Host("burrows"))],
        )
    }
}

/// A coarse blended-region field; other layers and spawners gate on
/// its named biomes.
/// Names for the regions the GENERATOR paints, so the rest of the
/// stack can gate on them.
///
/// It defines nothing itself: a region is a `material_band` op and
/// the ground colour is the definition. This is the dictionary from
/// a name to the material that region paints, which is what stops
/// "where trees grow" and "what colour the ground is" from being two
/// descriptions that can disagree.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, RegionLayer, Serialize, Deserialize, Default)]
pub struct Biomes {
    /// (region name, the material its band paints).
    pub table: Vec<(String, u32)>,
}

impl Node for Biomes {
    fn kind(&self) -> &'static str {
        "biomes"
    }
    fn domain(&self) -> Domain {
        Domain::Region
    }
    fn ports(&self) -> Ports {
        (&[], &[("biomes", Value::Host("biomes"))])
    }
}

/// Volumetric sites for interior worlds (no terrain filters).
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, RegionLayer, Serialize, Deserialize, Default)]
pub struct Scatter3 {
    #[serde(default = "d_cell3")]
    pub cell_m: i32,
    #[serde(default = "d_cell3_y")]
    pub cell_y_m: i32,
    pub chance: f32,
    #[serde(default = "d_margin3")]
    pub margin_m: f32,
    /// Snap site y to the structural floor lattice (0 = none).
    #[serde(default)]
    pub snap_y_m: f32,
    /// "instance:biome" gate (planar districts).
    #[serde(default)]
    pub biome: Option<String>,
}

impl Node for Scatter3 {
    fn kind(&self) -> &'static str {
        "scatter3"
    }
    fn domain(&self) -> Domain {
        Domain::Region
    }
    fn ports(&self) -> Ports {
        // The gate is a port only when this layer has one to name.
        let ins: &'static [Port] = match self.biome {
            Some(_) => &[("biome", Value::Host("biomes"))],
            None => &[],
        };
        (ins, &[("sites3", Value::Host("sites3"))])
    }
}

/// Orthogonal links between volumetric sites (walkway tubes).
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, RegionLayer, Serialize, Deserialize, Default)]
pub struct Connect3 {
    #[serde(default = "d_cell3")]
    pub cell_m: i32,
    #[serde(default = "d_cell3_y")]
    pub cell_y_m: i32,
    #[serde(default = "d_reach3")]
    pub reach_m: f32,
}

impl Node for Connect3 {
    fn kind(&self) -> &'static str {
        "connect3"
    }
    fn domain(&self) -> Domain {
        Domain::Region
    }
    fn ports(&self) -> Ports {
        (
            &[("source", Value::Host("sites3"))],
            &[("paths3", Value::Host("paths3"))],
        )
    }
}

/// Turn a source layer's data into world patches (the only kind that
/// produces geometry; also the index that keeps queries local).
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, RegionLayer, Serialize, Deserialize, Default)]
pub struct Emit {
    #[serde(default = "d_cell")]
    pub cell_m: i32,
    /// Cell height for volumetric sources (0 = planar).
    #[serde(default)]
    pub cell_y_m: i32,
    /// Source reach beyond its owning cells (dependency padding, m).
    pub pad_m: f32,
    /// Carve-horizon gate: serve ops only to chunks at least this
    /// fine (edge meters). Uniform per chunk, never per op.
    #[serde(default)]
    pub max_chunk_edge_m: Option<f32>,
    /// Keep this layer's data resident out to this radius (meters),
    /// whether or not its ops are served there.
    ///
    /// The gate above is about CARVING, and carving stops being worth
    /// anything once a chunk's voxels are bigger than the feature. The
    /// data is a different question: a map, an overlay, or a coarse
    /// representation reads where a road IS without wanting geometry
    /// cut for it. Sizing residency from the carve gate conflates the
    /// two and makes "visible on the map at 40 km" cost a carve op per
    /// chunk out to 40 km.
    #[serde(default)]
    pub keep_m: Option<f32>,
    pub emit: EmitDef,
}

impl Node for Emit {
    fn kind(&self) -> &'static str {
        "emit"
    }
    fn domain(&self) -> Domain {
        Domain::Region
    }
    fn ports(&self) -> Ports {
        // What an emit consumes depends on what it emits: a road wants a
        // path, a river a course, a ruin a site. Declaring it here is what
        // makes an emit wired to the wrong producer a refusal with both
        // names in it, rather than a layer that quietly emits nothing.
        let source: &'static [Port] = match self.emit {
            EmitDef::PathSlabs { .. } | EmitDef::PathRibbon { .. } => {
                &[("source", Value::Host("paths"))]
            }
            EmitDef::Ribbon { .. } => &[("source", Value::Host("courses"))],
            EmitDef::WormCuts => &[("source", Value::Host("burrows"))],
            EmitDef::SiteStructure { .. } => &[
                ("source", Value::Host("sites")),
                ("structure", Value::Host("structure")),
            ],
            EmitDef::SiteStructure3 { .. } => &[
                ("source", Value::Host("sites3")),
                ("structure", Value::Host("structure")),
            ],
            EmitDef::Tubes { .. } => &[("source", Value::Host("paths3"))],
        };
        (source, &[])
    }
}

/// A population of props: where a class of thing grows.
///
/// The engine owns everything about one — the placer, the tiles, the
/// coverage, the entities — and this owns the two references that are
/// THIS game's: which weight source gates it, and which field node drives
/// its density. That is why the node is here and the schema is there; an
/// engine node cannot wire to a host value without making a level's engine
/// half unreadable on its own.
///
/// Transparent, so a level writes the population's fields directly rather
/// than nesting them under a wrapper that carries nothing.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, RegionLayer, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct Population(pub ScatterDef);

impl Node for Population {
    fn kind(&self) -> &'static str {
        "population"
    }
    fn domain(&self) -> Domain {
        Domain::Region
    }
    fn ports(&self) -> Ports {
        // A port only where there is something to name: a population with
        // no region reads no biomes, and one with no density reads no
        // field.
        let ins: &'static [Port] = match (self.0.region.is_some(), self.0.density.is_some()) {
            (false, false) => &[],
            (true, false) => &[("gate", Value::Host("biomes"))],
            (false, true) => &[("density", Value::Field)],
            (true, true) => &[("density", Value::Field), ("gate", Value::Host("biomes"))],
        };
        (ins, &[])
    }
    /// Props are entities. A population decides where they go and carves
    /// nothing, so editing one must not restream a world whose voxels
    /// would come back identical.
    fn invalidates(&self) -> Invalidates {
        Invalidates::Plan
    }
}

impl RegionLayer for Population {
    /// Registered in bulk by [`crate::scatter::register`], which needs
    /// every emit name and the whole set at once to size the dependencies
    /// they share. Nothing to do per node.
    fn register(&self, _ctx: &RegionCtx, _mgr: &mut LayerGraph) {}
}

/// Every population a level declares, in the form the placer reads.
///
/// The three things a level stopped writing — the class, the
/// `"instance:member"` gate, the field slot — come from the node's name
/// and its wires. `fields` is the compiler's allocation, which is what
/// removed the slot number a level used to write twice.
pub fn populations(
    nodes: &[voxel_engine::graph::NodeDef],
    fields: &bevy::platform::collections::HashMap<String, u32>,
) -> Vec<ScatterDef> {
    let mut out = Vec::new();
    for node in nodes {
        let Some(p) = node.node.0.as_any().downcast_ref::<Population>() else {
            continue;
        };
        let Some(name) = node.name.as_deref() else {
            continue;
        };
        let source = |port: &str| {
            node.wires
                .get(port)
                .and_then(|w| w.sources().first())
                .cloned()
        };
        let mut def = p.0.clone();
        def.class = name.to_string();
        def.gate = def
            .region
            .as_ref()
            .zip(source("gate"))
            .map(|(member, src)| format!("{src}:{member}"));
        if let Some(density) = &mut def.density {
            density.field = source("density")
                .and_then(|n| fields.get(&n).copied())
                .unwrap_or_default();
        }
        out.push(def);
    }
    out
}

/// What a region node needs from the rest of the graph to build itself.
///
/// Everything it used to reach for by scanning the whole stack — the
/// biomes table behind a `"instance:member"` string, the structure behind
/// a name in a side map — arrives through a PORT the compiler already
/// checked. So the lookups here cannot fail on a level that compiled, and
/// they say so.
pub struct RegionCtx<'a> {
    pub name: &'a str,
    pub wires: &'a Wires,
    /// Every region node by name, for the tables a layer reads.
    pub by_name: &'a HashMap<String, &'a dyn Node>,
}

impl RegionCtx<'_> {
    /// What a port is wired to.
    pub fn wired(&self, port: &str) -> Option<&str> {
        match self.wires.get(port)? {
            Wire::One(name) => Some(name),
            Wire::Many(names) => names.first().map(String::as_str),
        }
    }

    fn node(&self, port: &str) -> Option<&dyn Node> {
        self.by_name.get(self.wired(port)?).copied()
    }

    /// The material id `member` has in the biomes table this node's
    /// `biome` port is wired to.
    pub fn biome_gate(&self, member: &str) -> Option<BiomeGate> {
        let biomes = self.node("biome")?.as_any().downcast_ref::<Biomes>()?;
        let (_, material) = biomes.table.iter().find(|(n, _)| n == member)?;
        Some(BiomeGate {
            material: *material,
        })
    }

    /// The structure this node's `structure` port is wired to.
    pub fn structure(&self) -> Option<StructureDef> {
        let s = self
            .node("structure")?
            .as_any()
            .downcast_ref::<Structure>()?;
        Some(StructureDef {
            size: s.size,
            variants: s.variants.clone(),
        })
    }
}

/// How a region node becomes a layer.
///
/// The game's own trait, on the game's own kinds. Reached through the type
/// registry like [`Node`] is, so the set stays open and the eight-arm
/// match this replaces cannot come back.
#[bevy::reflect::reflect_trait]
pub trait RegionLayer {
    fn register(&self, ctx: &RegionCtx, mgr: &mut LayerGraph);
}

impl RegionLayer for Biomes {
    // Names only: there is no layer to register. A region is a
    // `material_band` op in the generator, and this node is the dictionary
    // the stack's gates resolve through.
    fn register(&self, _ctx: &RegionCtx, _mgr: &mut LayerGraph) {}
}

impl RegionLayer for Scatter {
    fn register(&self, ctx: &RegionCtx, mgr: &mut LayerGraph) {
        let Self {
            cell_m,
            chance,
            margin_m,
            altitude,
            up,
            biome,
            relax,
        } = self.clone();
        let name = ctx.name.to_string();
        let base = ScatterCfg {
            cell_m,
            chance,
            margin_m,
            altitude,
            up,
            biome: biome.as_deref().map(|r| {
                ctx.biome_gate(r)
                    .expect("the biome port is wired and checked")
            }),
            relax_from: None,
        };
        let Some(relax) = relax.filter(|r| r.iterations > 0 && r.strength > 0.0) else {
            return mgr.register_as(&name, ScatterSites { cfg: base });
        };
        // A chain of instances, each reading the one before. The
        // LAST one takes the public name, so consumers are
        // untouched — they declared a dependency on `name` and
        // still get sites from `name`, just better spaced.
        let raw = format!("{name}:scattered");
        mgr.register_as(&raw, ScatterSites { cfg: base.clone() });
        let mut source = raw;
        for i in 1..=relax.iterations {
            let stage = if i == relax.iterations {
                name.clone()
            } else {
                format!("{name}:relax{i}")
            };
            mgr.register_as(
                &stage,
                ScatterSites {
                    cfg: ScatterCfg {
                        relax_from: Some(RelaxFrom {
                            instance: source,
                            strength: relax.strength,
                        }),
                        ..base.clone()
                    },
                },
            );
            source = stage;
        }
    }
}

impl RegionLayer for Connect {
    fn register(&self, ctx: &RegionCtx, mgr: &mut LayerGraph) {
        let Self {
            cell_m,
            reach_m,
            corridor_m,
            slope_penalty,
            step_m,
        } = self.clone();
        let source = ctx
            .wired("source")
            .expect("the source port is checked")
            .to_string();
        let name = ctx.name.to_string();
        mgr.register_as(
            &name,
            ConnectPaths {
                cfg: ConnectCfg {
                    source,
                    reach_m,
                    corridor_m,
                    slope_penalty,
                    step_m,
                },
                cell_m,
            },
        )
    }
}

impl RegionLayer for Flow {
    fn register(&self, ctx: &RegionCtx, mgr: &mut LayerGraph) {
        let Self {
            cell_m,
            max_steps,
            max_spill_rise,
        } = self.clone();
        let source = ctx
            .wired("source")
            .expect("the source port is checked")
            .to_string();
        let name = ctx.name.to_string();
        mgr.register_as(
            &name,
            FlowCourses {
                cfg: FlowCfg {
                    source,
                    max_steps,
                    max_spill_rise,
                },
                cell_m,
            },
        )
    }
}

impl RegionLayer for Worm {
    fn register(&self, ctx: &RegionCtx, mgr: &mut LayerGraph) {
        let Self {
            cell_m,
            steps,
            radius,
            burial_radii,
        } = self.clone();
        let source = ctx
            .wired("source")
            .expect("the source port is checked")
            .to_string();
        let name = ctx.name.to_string();
        mgr.register_as(
            &name,
            WormBurrows {
                cfg: WormCfg {
                    source,
                    steps,
                    radius,
                    burial_radii,
                },
                cell_m,
            },
        )
    }
}

impl RegionLayer for Scatter3 {
    fn register(&self, ctx: &RegionCtx, mgr: &mut LayerGraph) {
        let Self {
            cell_m,
            cell_y_m,
            chance,
            margin_m,
            snap_y_m,
            biome,
        } = self.clone();
        let name = ctx.name.to_string();
        mgr.register_as(
            &name,
            Scatter3Sites {
                cfg: Scatter3Cfg {
                    cell_m,
                    cell_y_m,
                    chance,
                    margin_m,
                    snap_y_m,
                    biome: biome.as_deref().map(|r| {
                        ctx.biome_gate(r)
                            .expect("the biome port is wired and checked")
                    }),
                },
            },
        )
    }
}

impl RegionLayer for Connect3 {
    fn register(&self, ctx: &RegionCtx, mgr: &mut LayerGraph) {
        let Self {
            cell_m,
            cell_y_m,
            reach_m,
        } = self.clone();
        let source = ctx
            .wired("source")
            .expect("the source port is checked")
            .to_string();
        let name = ctx.name.to_string();
        mgr.register_as(
            &name,
            Connect3Paths {
                cfg: Connect3Cfg { source, reach_m },
                cell_m,
                cell_y_m,
            },
        )
    }
}

impl RegionLayer for Emit {
    fn register(&self, ctx: &RegionCtx, mgr: &mut LayerGraph) {
        // `max_chunk_edge_m` and `keep_m` are residency sizes, read by
        // `StackPlanner::new` when it builds this node's top dependency —
        // the layer itself never sees them.
        let Self {
            cell_m,
            cell_y_m,
            pad_m,
            emit,
            ..
        } = self.clone();
        let source = ctx
            .wired("source")
            .expect("the source port is checked")
            .to_string();
        let name = ctx.name.to_string();
        mgr.register_as(
            &name,
            EmitPatches {
                cfg: EmitCfg {
                    source,
                    kind: emit.to_kind(ctx.structure().as_ref()),
                    pad_m,
                },
                cell_m,
                cell_y_m,
            },
        )
    }
}

/// Every kind a level of THIS game can name: the engine's and its own.
pub fn kinds() -> bevy::reflect::TypeRegistryArc {
    let reg = voxel_engine::graph::registry::engine_kinds();
    register(&mut reg.write());
    reg
}

/// Put this game's kinds in the registry.
///
/// Called before a level is parsed: a kind that is not registered is a
/// `"kind"` no document can name.
pub fn register(registry: &mut bevy::reflect::TypeRegistry) {
    registry.register::<Structure>();
    registry.register::<Scatter>();
    registry.register::<Connect>();
    registry.register::<Flow>();
    registry.register::<Worm>();
    registry.register::<Biomes>();
    registry.register::<Scatter3>();
    registry.register::<Connect3>();
    registry.register::<Emit>();
    registry.register::<Population>();
}
