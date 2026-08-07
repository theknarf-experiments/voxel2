//! The JSON planning vocabulary this demo authors its world in.
//!
//! Nothing here is engine schema. The engine carries the level's
//! `planning` block verbatim and never looks inside it; these types are
//! how *this* host chooses to describe its layers, and a game with
//! hand-written layers would delete the whole module.

use std::collections::HashMap;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use voxel_engine::level::LevelDef;

use voxel_engine::level::{d_op_material, default_one};

use super::layers;
use super::structure;

/// The `planning` block of a level file.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct PlanningDef {
    /// The planning stack: generic layers composed into one LayerManager.
    #[serde(default)]
    pub stack: Vec<StackLayerDef>,
    /// Named structures the stack's `site_structure` emits build.
    #[serde(default)]
    pub structures: HashMap<String, StructureDef>,
}

impl PlanningDef {
    /// Read a level's planning block, or the empty stack if it has none.
    pub fn of(level: &LevelDef) -> Result<Self, String> {
        if level.planning.is_null() {
            return Ok(Self::default());
        }
        serde_json::from_value(level.planning.clone())
            .map_err(|e| format!("planning block: {e}"))
    }
}

/// A structure: what `site_structure` emits at each site. Authored as
/// data — weighted variants of parts, each placing one shape at every
/// position of an arrangement. See `structure`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct StructureDef {
    /// Sampled once per site; arrangements scale their radius by it, so
    /// a structure's parts agree with each other.
    pub size: [f32; 2],
    pub variants: Vec<VariantDef>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct VariantDef {
    #[serde(default = "default_one")]
    pub weight: f32,
    pub parts: Vec<PartDef>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PartDef {
    pub arrange: ArrangeDef,
    pub shape: ShapeDef,
    #[serde(default = "d_op_material")]
    pub material: u32,
    #[serde(default)]
    pub cut: bool,
    /// Inner cut inset on every axis (hollow shells).
    #[serde(default)]
    pub hollow: Option<f32>,
    /// Per-instance chance to emit nothing (collapsed pieces).
    #[serde(default)]
    pub skip: f32,
    #[serde(default)]
    pub seat: SeatDef,
    #[serde(default)]
    pub anchor: AnchorDef,
    #[serde(default)]
    pub y_offset: [f32; 2],
    #[serde(default)]
    pub yaw: YawDef,
    /// Sweep a tunnel between consecutive instances.
    #[serde(default)]
    pub link: Option<LinkDef>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ArrangeDef {
    Single,
    Ring {
        count: [u32; 2],
        #[serde(default = "d_full_frac")]
        radius_frac: [f32; 2],
    },
    Scatter {
        count: [u32; 2],
        #[serde(default = "d_full_frac")]
        radius_frac: [f32; 2],
    },
    Chain {
        count: [u32; 2],
        step: [f32; 2],
        #[serde(default = "d_turn")]
        turn_deg: f32,
        #[serde(default)]
        descend: [f32; 2],
        #[serde(default)]
        orthogonal: bool,
        #[serde(default = "d_full_frac")]
        radius_frac: [f32; 2],
        /// Start above ground so `link` carves an entrance.
        #[serde(default)]
        from_surface: bool,
    },
}

/// A box half-extent: a range, or `"arc"` for the ring's tangential
/// half-length (wall segments that meet without authored trigonometry).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(untagged)]
pub enum ExtentDef {
    Range([f32; 2]),
    Keyword(ExtentKeyword),
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExtentKeyword {
    Arc,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShapeDef {
    Boxy {
        half: [ExtentDef; 3],
    },
    Cylinder {
        radius: [f32; 2],
        half_height: [f32; 2],
    },
    Sphere {
        radius: [f32; 2],
    },
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SeatDef {
    /// The generator's heightfield (surface structures).
    #[default]
    Terrain,
    /// The site's own y (interiors seated on a structural floor).
    Site,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AnchorDef {
    /// The shape's base rests on the seat.
    #[default]
    Base,
    /// The shape's center sits at the seat.
    Center,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum YawDef {
    #[default]
    Zero,
    Random,
    /// Face along the arrangement (ring tangent, chain heading).
    Tangent,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct LinkDef {
    pub half_w: f32,
    pub half_h: f32,
    #[serde(default = "d_link_step")]
    pub step_m: f32,
    #[serde(default)]
    pub material: u32,
    #[serde(default = "d_true")]
    pub cut: bool,
}

fn d_full_frac() -> [f32; 2] {
    [1.0, 1.0]
}
fn d_turn() -> f32 {
    45.0
}
fn d_link_step() -> f32 {
    3.0
}

impl StructureDef {
    /// Pack into the runtime form the planning stack builds from.
    pub fn pack(&self) -> structure::Structure {
        use structure as rt;
        rt::Structure {
            size: self.size,
            variants: self
                .variants
                .iter()
                .map(|v| rt::Variant {
                    weight: v.weight,
                    parts: v.parts.iter().map(PartDef::pack).collect(),
                })
                .collect(),
        }
    }
}

impl PartDef {
    fn pack(&self) -> structure::Part {
        use structure as rt;
        let extent = |e: &ExtentDef| match e {
            ExtentDef::Range(r) => rt::Extent::Range(*r),
            ExtentDef::Keyword(ExtentKeyword::Arc) => rt::Extent::Arc,
        };
        rt::Part {
            arrange: match self.arrange.clone() {
                ArrangeDef::Single => rt::Arrange::Single,
                ArrangeDef::Ring { count, radius_frac } => rt::Arrange::Ring { count, radius_frac },
                ArrangeDef::Scatter { count, radius_frac } => {
                    rt::Arrange::Scatter { count, radius_frac }
                }
                ArrangeDef::Chain {
                    count,
                    step,
                    turn_deg,
                    descend,
                    orthogonal,
                    radius_frac,
                    from_surface,
                } => rt::Arrange::Chain {
                    count,
                    step,
                    turn_deg,
                    descend,
                    orthogonal,
                    radius_frac,
                    from_surface,
                },
            },
            shape: match &self.shape {
                ShapeDef::Boxy { half } => rt::Shape::Boxy {
                    half: [extent(&half[0]), extent(&half[1]), extent(&half[2])],
                },
                ShapeDef::Cylinder {
                    radius,
                    half_height,
                } => rt::Shape::Cylinder {
                    radius: *radius,
                    half_height: *half_height,
                },
                ShapeDef::Sphere { radius } => rt::Shape::Sphere { radius: *radius },
            },
            material: self.material,
            cut: self.cut,
            hollow: self.hollow,
            skip: self.skip,
            seat: match self.seat {
                SeatDef::Terrain => rt::Seat::Terrain,
                SeatDef::Site => rt::Seat::Site,
            },
            anchor: match self.anchor {
                AnchorDef::Base => rt::Anchor::Base,
                AnchorDef::Center => rt::Anchor::Center,
            },
            y_offset: self.y_offset,
            yaw: match self.yaw {
                YawDef::Zero => rt::Yaw::Zero,
                YawDef::Random => rt::Yaw::Random,
                YawDef::Tangent => rt::Yaw::Tangent,
            },
            link: self.link.map(|l| rt::Link {
                half_w: l.half_w,
                half_h: l.half_h,
                step_m: l.step_m,
                material: l.material,
                cut: l.cut,
            }),
        }
    }
}

/// One layer of the level's planning stack — the generic vocabulary
/// (scatter/connect/flow/worm/emit) every planned feature is expressed
/// in. Layers register in author order into ONE LayerManager per level;
/// `source` references an earlier layer by instance name.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StackLayerDef {
    /// Hash-gated candidate sites per cell, filtered by terrain.
    Scatter {
        name: String,
        #[serde(default = "d_cell")]
        cell_m: i32,
        chance: f32,
        #[serde(default = "d_margin")]
        margin_m: f32,
        #[serde(default = "d_altitude")]
        altitude: [f32; 2],
        #[serde(default = "d_up_interval")]
        up: [f32; 2],
        /// "instance:biome" — accept sites with probability = that
        /// biome's blended weight.
        #[serde(default)]
        biome: Option<String>,
    },
    /// Pathfound links between sites of a scatter instance.
    Connect {
        name: String,
        source: String,
        #[serde(default = "d_cell")]
        cell_m: i32,
        #[serde(default = "d_reach")]
        reach_m: f32,
        #[serde(default = "d_corridor")]
        corridor_m: f32,
        #[serde(default = "d_slope_penalty")]
        slope_penalty: f32,
        /// Pathfinding lattice step (meters). The cost of a route is
        /// quadratic in this, so a long-range corridor is planned on a
        /// coarse lattice and a local track on a fine one — which is what
        /// makes the same layer kind serve both scales.
        #[serde(default = "d_path_step")]
        step_m: f32,
    },
    /// Descent courses (pond-and-spill hydrology) from sites.
    Flow {
        name: String,
        source: String,
        #[serde(default = "d_flow_cell")]
        cell_m: i32,
        #[serde(default = "d_flow_steps")]
        max_steps: usize,
        #[serde(default = "d_spill")]
        max_spill_rise: f32,
    },
    /// Noise-steered burrows from sites.
    Worm {
        name: String,
        source: String,
        #[serde(default = "d_cell")]
        cell_m: i32,
        #[serde(default = "d_worm_steps")]
        steps: u32,
        #[serde(default = "d_worm_radius")]
        radius: [f32; 2],
        #[serde(default = "d_burial")]
        burial_radii: f32,
    },
    /// A coarse blended-region field; other layers and spawners gate on
    /// its named biomes.
    Biomes {
        name: String,
        #[serde(default = "d_biome_cell")]
        cell_m: i32,
        /// (biome name, selection weight) — order defines indices.
        table: Vec<(String, f32)>,
    },
    /// Volumetric sites for interior worlds (no terrain filters).
    Scatter3 {
        name: String,
        #[serde(default = "d_cell3")]
        cell_m: i32,
        #[serde(default = "d_cell3_y")]
        cell_y_m: i32,
        chance: f32,
        #[serde(default = "d_margin3")]
        margin_m: f32,
        /// Snap site y to the structural floor lattice (0 = none).
        #[serde(default)]
        snap_y_m: f32,
        /// "instance:biome" gate (planar districts).
        #[serde(default)]
        biome: Option<String>,
    },
    /// Orthogonal links between volumetric sites (walkway tubes).
    Connect3 {
        name: String,
        source: String,
        #[serde(default = "d_cell3")]
        cell_m: i32,
        #[serde(default = "d_cell3_y")]
        cell_y_m: i32,
        #[serde(default = "d_reach3")]
        reach_m: f32,
    },
    /// Turn a source layer's data into world patches (the only kind that
    /// produces geometry; also the index that keeps queries local).
    Emit {
        name: String,
        source: String,
        #[serde(default = "d_cell")]
        cell_m: i32,
        /// Cell height for volumetric sources (0 = planar).
        #[serde(default)]
        cell_y_m: i32,
        /// Source reach beyond its owning cells (dependency padding, m).
        pad_m: f32,
        /// Carve-horizon gate: serve ops only to chunks at least this
        /// fine (edge meters). Uniform per chunk, never per op.
        #[serde(default)]
        max_chunk_edge_m: Option<f32>,
        emit: EmitDef,
    },
}

/// The emission shape of an `emit` stack layer.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmitDef {
    /// Terrain-seated slabs along a `connect` source (roads).
    PathSlabs {
        #[serde(default = "d_half_w")]
        half_w: f32,
        #[serde(default = "d_thickness")]
        thickness: f32,
        #[serde(default = "d_op_material")]
        material: u32,
        #[serde(default = "d_true")]
        clearance: bool,
    },
    /// Bed notch plus a flat ribbon surface along a `flow` source. The
    /// engine only places it; a river, canal or lava flow is the same
    /// primitive with a different material.
    Ribbon {
        material: u32,
        #[serde(default = "d_course_width")]
        width: [f32; 2],
    },
    /// Sphere-cut chains from a `worm` source (caves).
    WormCuts,
    /// Build a named structure (from the level's `structures` table) at
    /// each site, with an optional marker.
    SiteStructure {
        structure: String,
        #[serde(default)]
        marker: Option<String>,
    },
    /// The same at each `scatter3` site (interiors).
    SiteStructure3 {
        structure: String,
        #[serde(default)]
        marker: Option<String>,
    },
    /// Shell tubes with bored interiors along a `connect3` source.
    Tubes {
        #[serde(default = "d_tube_material")]
        material: u32,
        #[serde(default = "d_tube_bore")]
        bore: f32,
        /// Lift above the site lattice plane (bore floor on the slab top).
        #[serde(default = "d_tube_lift")]
        lift_m: f32,
    },
}

fn d_cell() -> i32 {
    256
}
fn d_flow_cell() -> i32 {
    512
}
fn d_margin() -> f32 {
    32.0
}
fn d_altitude() -> [f32; 2] {
    [f32::MIN, f32::MAX]
}
fn d_up_interval() -> [f32; 2] {
    [0.0, 1.0]
}
fn d_reach() -> f32 {
    700.0
}
fn d_corridor() -> f32 {
    192.0
}
fn d_slope_penalty() -> f32 {
    60.0
}
fn d_path_step() -> f32 {
    8.0
}
fn d_flow_steps() -> usize {
    400
}
fn d_spill() -> f32 {
    7.0
}
fn d_worm_steps() -> u32 {
    70
}
fn d_worm_radius() -> [f32; 2] {
    [2.2, 3.6]
}
fn d_burial() -> f32 {
    2.4
}
fn d_half_w() -> f32 {
    2.4
}
fn d_thickness() -> f32 {
    0.5
}
fn d_true() -> bool {
    true
}
fn d_course_width() -> [f32; 2] {
    [2.0, 7.0]
}
fn d_cell3() -> i32 {
    128
}
fn d_cell3_y() -> i32 {
    132
}
fn d_margin3() -> f32 {
    24.0
}
fn d_reach3() -> f32 {
    400.0
}
fn d_tube_material() -> u32 {
    2
}
fn d_tube_bore() -> f32 {
    1.5
}
fn d_tube_lift() -> f32 {
    3.0
}
fn d_biome_cell() -> i32 {
    2048
}

impl EmitDef {
    fn to_kind(
        &self,
        structures: &std::collections::HashMap<String, StructureDef>,
    ) -> layers::EmitKind {
        use layers::EmitKind;
        // Structures are validated at load, so a miss here is a bug.
        let build = |name: &str| {
            std::sync::Arc::new(
                structures
                    .get(name)
                    .expect("structure validated before registration")
                    .pack(),
            )
        };
        match self.clone() {
            EmitDef::PathSlabs {
                half_w,
                thickness,
                material,
                clearance,
            } => EmitKind::PathSlabs {
                half_w,
                thickness,
                material,
                clearance,
            },
            EmitDef::Ribbon { material, width } => EmitKind::Ribbon { material, width },
            EmitDef::WormCuts => EmitKind::WormCuts,
            EmitDef::SiteStructure { structure, marker } => EmitKind::SiteStructure {
                structure: build(&structure),
                marker,
            },
            EmitDef::SiteStructure3 { structure, marker } => EmitKind::SiteStructure3 {
                structure: build(&structure),
                marker,
            },
            EmitDef::Tubes {
                material,
                bore,
                lift_m,
            } => EmitKind::Tubes {
                material,
                bore,
                lift_m,
            },
        }
    }
}

/// The kind tag of a stack layer, for source-compatibility checks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StackKind {
    Biomes,
    Scatter,
    Scatter3,
    Connect,
    Connect3,
    Flow,
    Worm,
    Emit,
}

/// Validate a level's planning stack before anything registers: every
/// authoring error (bad reference, unknown recipe, kind mismatch,
/// misordered declaration) fails HERE with a message, never as a panic
/// at registration or mid-generation. Boot fails loudly on an invalid
/// shipped level; hot reload warns and keeps the running world.
/// Validate the whole level's data-driven references: the stack itself,
/// plus spawner biome refs (which would otherwise degrade silently to
/// full density on a typo).
pub fn validate_level(level: &LevelDef) -> Result<(), String> {
    let planning = PlanningDef::of(level)?;
    validate_stack(&planning.stack, &planning.structures)?;
    let biome_ref = |owner: &str, reference: &str| -> Result<(), String> {
        let Some((instance, biome)) = reference.rsplit_once(':') else {
            return Err(format!(
                "spawner {owner}: biome ref {reference:?} is not \"instance:biome\""
            ));
        };
        for def in &planning.stack {
            if let StackLayerDef::Biomes { name, table, .. } = def {
                if name == instance {
                    if table.iter().any(|(n, _)| n == biome) {
                        return Ok(());
                    }
                    return Err(format!(
                        "spawner {owner}: biome {biome:?} not in layer {instance:?}"
                    ));
                }
            }
        }
        Err(format!(
            "spawner {owner}: biome layer {instance:?} not found in stack"
        ))
    };
    for def in &level.scatter {
        if let Some(reference) = &def.biome {
            biome_ref(&def.class, reference)?;
        }
        if def.output == voxel_engine::level::ScatterOutput::Entities && def.variants.is_empty() {
            return Err(format!("scatter class {:?} has no variants", def.class));
        }
    }
    Ok(())
}

/// A referenced structure must exist and stay inside the element-padding
/// contract the emit index rests on.
fn check_structure(
    structures: &std::collections::HashMap<String, StructureDef>,
    owner: &str,
    name: &str,
) -> Result<(), String> {
    let Some(def) = structures.get(name) else {
        let known: Vec<&str> = structures.keys().map(String::as_str).collect();
        return Err(format!(
            "layer {owner:?}: unknown structure {name:?} (declared: {known:?})"
        ));
    };
    if def.variants.is_empty() {
        return Err(format!("structure {name:?} has no variants"));
    }
    let reach = def.pack().max_reach();
    let limit = layers::ELEM_PAD_M;
    if reach > limit {
        return Err(format!(
            "structure {name:?} reaches {reach:.0} m from its site, past the {limit:.0} m \
             element padding — queries farther than that would miss its geometry"
        ));
    }
    Ok(())
}

pub fn validate_stack(
    stack: &[StackLayerDef],
    structures: &std::collections::HashMap<String, StructureDef>,
) -> Result<(), String> {
    let mut declared: Vec<(&str, StackKind)> = Vec::new();
    let kind_of = |declared: &[(&str, StackKind)], source: &str| -> Option<StackKind> {
        declared
            .iter()
            .find_map(|(n, k)| (*n == source).then_some(*k))
    };
    // A `source` must name an EARLIER layer of the expected kind (the
    // registration order the layer manager requires).
    let check_source = |declared: &[(&str, StackKind)],
                        owner: &str,
                        source: &str,
                        expect: StackKind|
     -> Result<(), String> {
        match kind_of(declared, source) {
            Some(k) if k == expect => Ok(()),
            Some(k) => Err(format!(
                "layer {owner:?}: source {source:?} is a {k:?} layer, expected {expect:?}"
            )),
            None => Err(format!(
                "layer {owner:?}: source {source:?} is not declared earlier in the stack"
            )),
        }
    };
    for def in stack {
        let (name, kind) = match def {
            StackLayerDef::Biomes { name, table, .. } => {
                if table.is_empty() {
                    return Err(format!("biome layer {name:?} has an empty table"));
                }
                (name.as_str(), StackKind::Biomes)
            }
            StackLayerDef::Scatter { name, biome, .. }
            | StackLayerDef::Scatter3 { name, biome, .. } => {
                if let Some(reference) = biome {
                    StackLayerDef::biome_gate(&declared, stack, name, reference)?;
                }
                let kind = if matches!(def, StackLayerDef::Scatter { .. }) {
                    StackKind::Scatter
                } else {
                    StackKind::Scatter3
                };
                (name.as_str(), kind)
            }
            StackLayerDef::Connect { name, source, .. } => {
                check_source(&declared, name, source, StackKind::Scatter)?;
                (name.as_str(), StackKind::Connect)
            }
            StackLayerDef::Connect3 { name, source, .. } => {
                check_source(&declared, name, source, StackKind::Scatter3)?;
                (name.as_str(), StackKind::Connect3)
            }
            StackLayerDef::Flow { name, source, .. } => {
                check_source(&declared, name, source, StackKind::Scatter)?;
                (name.as_str(), StackKind::Flow)
            }
            StackLayerDef::Worm { name, source, .. } => {
                check_source(&declared, name, source, StackKind::Scatter)?;
                (name.as_str(), StackKind::Worm)
            }
            StackLayerDef::Emit {
                name,
                source,
                emit,
                cell_y_m,
                ..
            } => {
                let expect = match emit {
                    EmitDef::PathSlabs { .. } => StackKind::Connect,
                    EmitDef::Ribbon { .. } => StackKind::Flow,
                    EmitDef::WormCuts => StackKind::Worm,
                    EmitDef::SiteStructure { structure, .. } => {
                        check_structure(structures, name, structure)?;
                        StackKind::Scatter
                    }
                    EmitDef::SiteStructure3 { structure, .. } => {
                        check_structure(structures, name, structure)?;
                        StackKind::Scatter3
                    }
                    EmitDef::Tubes { .. } => StackKind::Connect3,
                };
                check_source(&declared, name, source, expect)?;
                if matches!(expect, StackKind::Scatter3 | StackKind::Connect3) && *cell_y_m <= 0 {
                    return Err(format!(
                        "layer {name:?}: emit over a volumetric source needs cell_y_m > 0 \
                         (a collapsed y axis spans unbounded rows)"
                    ));
                }
                (name.as_str(), StackKind::Emit)
            }
        };
        if declared.iter().any(|(n, _)| *n == name) {
            return Err(format!("duplicate stack layer name {name:?}"));
        }
        declared.push((name, kind));
    }
    Ok(())
}

impl StackLayerDef {
    /// Resolve an "instance:biome" reference against earlier stack layers.
    fn biome_gate(
        declared: &[(&str, StackKind)],
        stack: &[StackLayerDef],
        owner: &str,
        reference: &str,
    ) -> Result<layers::BiomeGate, String> {
        let Some((instance, biome_name)) = reference.rsplit_once(':') else {
            return Err(format!(
                "layer {owner:?}: biome ref {reference:?} is not \"instance:biome\""
            ));
        };
        if !declared.is_empty() && !declared.iter().any(|(n, _)| *n == instance) {
            return Err(format!(
                "layer {owner:?}: biome layer {instance:?} is not declared earlier in the stack"
            ));
        }
        for def in stack {
            if let StackLayerDef::Biomes { name, table, .. } = def {
                if name == instance {
                    let Some(biome) = table.iter().position(|(n, _)| n == biome_name) else {
                        return Err(format!(
                            "layer {owner:?}: biome {biome_name:?} not in layer {instance:?}"
                        ));
                    };
                    return Ok(layers::BiomeGate {
                        instance: instance.to_string(),
                        biome: biome as u32,
                        n_biomes: table.len(),
                    });
                }
            }
        }
        Err(format!(
            "layer {owner:?}: biome layer {instance:?} not found in stack"
        ))
    }

    pub(super) fn register(
        &self,
        stack: &[StackLayerDef],
        structures: &std::collections::HashMap<String, StructureDef>,
        mgr: &mut voxel_layers::LayerGraph,
    ) {
        use layers::*;
        match self.clone() {
            StackLayerDef::Biomes {
                name,
                cell_m,
                table,
            } => mgr.register_as(&name, BiomeField {
                cfg: BiomeCfg { cell_m, table },
            }),
            StackLayerDef::Scatter {
                name,
                cell_m,
                chance,
                margin_m,
                altitude,
                up,
                biome,
            } => mgr.register_as(
                &name,
                ScatterSites {
                    cfg: ScatterCfg {
                        cell_m,
                        chance,
                        margin_m,
                        altitude,
                        up,
                        biome: biome.map(|r| {
                            Self::biome_gate(&[], stack, &name, &r)
                                .expect("stack validated before registration")
                        }),
                    },
                },
            ),
            StackLayerDef::Connect {
                name,
                source,
                cell_m,
                reach_m,
                corridor_m,
                slope_penalty,
                step_m,
            } => mgr.register_as(
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
            ),
            StackLayerDef::Flow {
                name,
                source,
                cell_m,
                max_steps,
                max_spill_rise,
            } => mgr.register_as(
                &name,
                FlowCourses {
                    cfg: FlowCfg {
                        source,
                        max_steps,
                        max_spill_rise,
                    },
                    cell_m,
                },
            ),
            StackLayerDef::Worm {
                name,
                source,
                cell_m,
                steps,
                radius,
                burial_radii,
            } => mgr.register_as(
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
            ),
            StackLayerDef::Scatter3 {
                name,
                cell_m,
                cell_y_m,
                chance,
                margin_m,
                snap_y_m,
                biome,
            } => mgr.register_as(
                &name,
                Scatter3Sites {
                    cfg: Scatter3Cfg {
                        cell_m,
                        cell_y_m,
                        chance,
                        margin_m,
                        snap_y_m,
                        biome: biome.map(|r| {
                            Self::biome_gate(&[], stack, &name, &r)
                                .expect("stack validated before registration")
                        }),
                    },
                },
            ),
            StackLayerDef::Connect3 {
                name,
                source,
                cell_m,
                cell_y_m,
                reach_m,
            } => mgr.register_as(
                &name,
                Connect3Paths {
                    cfg: Connect3Cfg { source, reach_m },
                    cell_m,
                    cell_y_m,
                },
            ),
            StackLayerDef::Emit {
                name,
                source,
                cell_m,
                cell_y_m,
                pad_m,
                // The carve-horizon gate is applied per chunk by the
                // planner facade, never inside the layer.
                max_chunk_edge_m: _,
                emit,
            } => mgr.register_as(
                &name,
                EmitPatches {
                    cfg: EmitCfg {
                        source,
                        kind: emit.to_kind(structures),
                        pad_m,
                    },
                    cell_m,
                    cell_y_m,
                },
            ),
        }
    }
}

