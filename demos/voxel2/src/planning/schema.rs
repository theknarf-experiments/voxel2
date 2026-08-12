//! The value types this game's [`nodes`] take their parameters in, and
//! the checks a compiled graph still cannot make.
//!
//! Nothing here is engine schema: the engine knows a node kind is a
//! registered type and nothing about what a road or a ruin is. These are
//! how *this* host chooses to describe its layers, and a game with
//! hand-written layers would delete the whole module.
//!
//! [`nodes`]: super::nodes

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use voxel_engine::level::LevelDef;

use voxel_engine::level::{d_op_material, default_one};

use super::layers;
use super::structure;

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct StructureDef {
    /// Sampled once per site; arrangements scale their radius by it, so
    /// a structure's parts agree with each other.
    pub size: [f32; 2],
    pub variants: Vec<VariantDef>,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct VariantDef {
    #[serde(default = "default_one")]
    pub weight: f32,
    pub parts: Vec<PartDef>,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
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

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
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

#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(untagged)]
pub enum ExtentDef {
    Range([f32; 2]),
    Keyword(ExtentKeyword),
}

#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExtentKeyword {
    Arc,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
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

#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SeatDef {
    /// The generator's heightfield (surface structures).
    #[default]
    Terrain,
    /// The site's own y (interiors seated on a structural floor).
    Site,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AnchorDef {
    /// The shape's base rests on the seat.
    #[default]
    Base,
    /// The shape's center sits at the seat.
    Center,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum YawDef {
    #[default]
    Zero,
    Random,
    /// Face along the arrangement (ring tangent, chain heading).
    Tangent,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
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

voxel_core::defaults! {
    pub d_full_frac: [f32; 2] = [1.0, 1.0];
    pub d_turn: f32 = 45.0;
    pub d_link_step: f32 = 3.0;
}

impl StructureDef {
    /// Pack into the runtime form the layers build from.
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

/// Push sites apart from their neighbours, in a second pass.
///
/// A partial state is its own INSTANCE, so a graph that would otherwise be
/// circular stays a DAG: place sites in one, relax them against their
/// neighbours in a second that depends on it, and let each consumer name
/// the stage it wants.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RelaxDef {
    /// Fraction of the wanted correction applied per iteration. Above
    /// ~0.5 a site tends to overshoot into the neighbour that pushed it.
    #[serde(default = "d_relax_strength")]
    pub strength: f32,
    #[serde(default = "d_relax_iterations")]
    pub iterations: u32,
}

/// The emission shape of an `emit` node: what it turns its source into.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
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
    /// A ground-seated ribbon along a `connect` source: a road as a
    /// surface rather than as a cut. Carves nothing, so it costs no ops
    /// and stays visible at any distance the data reaches.
    PathRibbon {
        material: u32,
        #[serde(default = "d_course_width")]
        width: [f32; 2],
    },
    /// Sphere-cut chains from a `worm` source (caves).
    ///
    /// The default because it is the only emit with nothing to configure:
    /// a node created empty in an editor should not also be half-filled.
    #[default]
    WormCuts,
    /// Build a named structure (from the level's `structures` table) at
    /// each site, with an optional marker.
    SiteStructure {
        #[serde(default)]
        marker: Option<String>,
    },
    /// The same at each `scatter3` site (interiors).
    SiteStructure3 {
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

voxel_core::defaults! {
    pub d_cell: i32 = 256;
    pub d_flow_cell: i32 = 512;
    pub d_margin: f32 = 32.0;
    pub d_altitude: [f32; 2] = [f32::MIN, f32::MAX];
    pub d_up_interval: [f32; 2] = [0.0, 1.0];
    pub d_reach: f32 = 700.0;
    pub d_corridor: f32 = 192.0;
    pub d_slope_penalty: f32 = 60.0;
    pub d_path_step: f32 = 8.0;
    pub d_flow_steps: usize = 400;
    pub d_spill: f32 = 7.0;
    pub d_worm_steps: u32 = 70;
    pub d_worm_radius: [f32; 2] = [2.2, 3.6];
    pub d_burial: f32 = 2.4;
    pub d_half_w: f32 = 2.4;
    pub d_thickness: f32 = 0.5;
    pub d_true: bool = true;
    pub d_course_width: [f32; 2] = [2.0, 7.0];
    pub d_cell3: i32 = 128;
    pub d_cell3_y: i32 = 132;
    pub d_margin3: f32 = 24.0;
    pub d_reach3: f32 = 400.0;
    pub d_tube_material: u32 = 2;
    pub d_tube_bore: f32 = 1.5;
    pub d_tube_lift: f32 = 3.0;
}
impl EmitDef {
    /// `structure` is whatever the emit's `structure` port is wired to.
    /// The compiler checked that port, so a miss here is a bug rather than
    /// an authoring mistake.
    pub(super) fn to_kind(&self, structure: Option<&StructureDef>) -> layers::EmitKind {
        use layers::EmitKind;
        let build = |_name: &str| {
            std::sync::Arc::new(
                structure
                    .expect("the structure port is wired and checked")
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
            EmitDef::PathRibbon { material, width } => EmitKind::PathRibbon { material, width },
            EmitDef::WormCuts => EmitKind::WormCuts,
            EmitDef::SiteStructure { marker } => EmitKind::SiteStructure {
                structure: build(""),
                marker,
            },
            EmitDef::SiteStructure3 { marker } => EmitKind::SiteStructure3 {
                structure: build(""),
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

/// What the graph compiler cannot check.
///
/// Most of what this used to do is gone: that a `source` names a layer
/// that exists, that it is the RIGHT kind of layer, that a structure is
/// defined — all of it is a port now, checked when the level compiles. What
/// is left is the spawner side, which still refers to a biome by a
/// `"instance:member"` string rather than by a wire.
pub fn validate_level(level: &LevelDef) -> Result<(), String> {
    let biomes: Vec<(&str, &Vec<(String, u32)>)> = level
        .nodes
        .iter()
        .filter_map(|n| {
            let b = n.node.0.as_any().downcast_ref::<super::nodes::Biomes>()?;
            Some((n.name.as_deref()?, &b.table))
        })
        .collect();

    // A biomes table may not be empty: every gate through it would be a
    // reference to nothing.
    for (name, table) in &biomes {
        if table.is_empty() {
            return Err(format!("biomes layer {name:?} has an empty table"));
        }
    }

    // The PORT says which biomes layer a node reads; this says the member
    // it names is in that layer's table. The compiler cannot check the
    // second, because it is a parameter rather than a wire.
    for node in &level.nodes {
        let any = node.node.0.as_any();
        // The port a kind reads its table through, and the member it
        // names. A population calls the port `gate` because what it gates
        // on is a weight; a layer calls it `biome` because it IS one.
        let named = any
            .downcast_ref::<super::nodes::Scatter>()
            .map(|n| ("biome", n.biome.clone()))
            .or_else(|| {
                any.downcast_ref::<super::nodes::Scatter3>()
                    .map(|n| ("biome", n.biome.clone()))
            })
            .or_else(|| {
                any.downcast_ref::<super::nodes::Population>()
                    .map(|n| ("gate", n.0.region.clone()))
            });
        let (Some((port, Some(member))), Some(name)) = (named, node.name.as_deref()) else {
            continue;
        };
        let instance = node
            .wires
            .get(port)
            .and_then(|w| w.sources().first())
            .ok_or_else(|| format!("layer {name:?} names a region but wires no {port} port"))?;
        let (_, table) = biomes
            .iter()
            .find(|(n, _)| n == instance)
            .ok_or_else(|| format!("layer {name:?}: {instance:?} is not a biomes layer"))?;
        if !table.iter().any(|(n, _)| *n == member) {
            return Err(format!(
                "layer {name:?}: region {member:?} not in layer {instance:?}"
            ));
        }
    }

    // A population that marches for floors must march the FINE world.
    //
    // The generator gates its structural ops at `WOP_COARSE_VOXEL_M`: at
    // or above that voxel size an interior is the solid mass its coarse
    // LODs draw, with no slabs in it and so no floors. A population left
    // on the default `detail_vs` reads that world, finds nothing, and
    // places nothing — silently, and looking exactly like a level whose
    // gates are wrong. It cost an afternoon once.
    for node in &level.nodes {
        let Some(p) = node
            .node
            .0
            .as_any()
            .downcast_ref::<super::nodes::Population>()
        else {
            continue;
        };
        if p.0.surface == voxel_engine::level::SurfaceMode::Floors
            && p.0.detail_vs >= voxel_core::worldop::WOP_COARSE_VOXEL_M
        {
            return Err(format!(
                "population {:?} looks for floors at detail_vs {} — at {} and above the \
                 generator draws its coarse world, which has no floors in it",
                node.name.as_deref().unwrap_or("?"),
                p.0.detail_vs,
                voxel_core::worldop::WOP_COARSE_VOXEL_M
            ));
        }
    }

    // A volumetric source needs a volumetric emit: with `cell_y_m` at zero
    // the emit's cells are planar and its sites land in one y-row.
    for node in &level.nodes {
        let Some(emit) = node.node.0.as_any().downcast_ref::<super::nodes::Emit>() else {
            continue;
        };
        let volumetric = matches!(
            emit.emit,
            EmitDef::SiteStructure3 { .. } | EmitDef::Tubes { .. }
        );
        if volumetric && emit.cell_y_m == 0 {
            return Err(format!(
                "emit {:?} has a volumetric source and cell_y_m 0",
                node.name.as_deref().unwrap_or("?")
            ));
        }
    }

    // A structure has to stay inside the element padding the emit index
    // rests on. The compiler checks that a `structure` port is WIRED to
    // one; how far the thing it names reaches is geometry, and only this
    // can measure it.
    for node in &level.nodes {
        let Some(s) = node
            .node
            .0
            .as_any()
            .downcast_ref::<super::nodes::Structure>()
        else {
            continue;
        };
        let name = node.name.as_deref().unwrap_or("?");
        if s.variants.is_empty() {
            return Err(format!("structure {name:?} has no variants"));
        }
        let def = StructureDef {
            size: s.size,
            variants: s.variants.clone(),
        };
        let reach = def.pack().max_reach();
        let limit = layers::ELEM_PAD_M;
        if reach > limit {
            return Err(format!(
                "structure {name:?} reaches {reach:.0} m from its site, past the {limit:.0} m \
                 element padding — queries farther than that would miss its geometry"
            ));
        }
    }

    // A population that spawns entities has to say what they are; one
    // that emits points does not, because a point is a position and the
    // host decides what it draws there.
    for node in &level.nodes {
        let Some(p) = node
            .node
            .0
            .as_any()
            .downcast_ref::<super::nodes::Population>()
        else {
            continue;
        };
        if p.0.output == voxel_engine::level::ScatterOutput::Entities && p.0.variants.is_empty() {
            return Err(format!(
                "population {:?} has no variants",
                node.name.as_deref().unwrap_or("?")
            ));
        }
    }
    Ok(())
}

voxel_core::defaults! {
    pub d_relax_strength: f32 = 0.35;
    pub d_relax_iterations: u32 = 1;
}
