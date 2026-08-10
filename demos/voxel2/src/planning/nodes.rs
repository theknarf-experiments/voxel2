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

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use voxel_core::opgen::{Port, Value};
use voxel_engine::graph::node::{Domain, Node, Ports, ReflectNode};

use super::schema::{EmitDef, RelaxDef, VariantDef};
use super::schema::{
    d_altitude, d_burial, d_cell, d_cell3, d_cell3_y, d_corridor, d_flow_cell, d_flow_steps,
    d_margin, d_margin3, d_path_step, d_reach, d_reach3, d_slope_penalty, d_spill, d_up_interval,
    d_worm_radius, d_worm_steps,
};

/// A named structure the stack can build at a site.
///
/// A node like everything else, so `site_structure` REFERS to one by name
/// through a port the compiler checks, rather than by a string looked up in
/// a side table that could be missing.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct Structure {
    /// Sampled once per site; arrangements scale their radius by it, so a
    /// structure's parts agree with each other.
    pub size: [f32; 2],
    pub variants: Vec<VariantDef>,
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
#[reflect(Node, Serialize, Deserialize, Default)]
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
#[reflect(Node, Serialize, Deserialize, Default)]
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
        (&[("source", Value::Host("sites"))], &[("paths", Value::Host("paths"))])
    }
}

/// Descent courses (pond-and-spill hydrology) from sites.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
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
        (&[("source", Value::Host("sites"))], &[("courses", Value::Host("courses"))])
    }
}

/// Noise-steered burrows from sites.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
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
        (&[("source", Value::Host("sites"))], &[("burrows", Value::Host("burrows"))])
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
#[reflect(Node, Serialize, Deserialize, Default)]
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
#[reflect(Node, Serialize, Deserialize, Default)]
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
#[reflect(Node, Serialize, Deserialize, Default)]
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
        (&[("source", Value::Host("sites3"))], &[("paths3", Value::Host("paths3"))])
    }
}

/// Turn a source layer's data into world patches (the only kind that
/// produces geometry; also the index that keeps queries local).
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
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
}
