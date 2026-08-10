//! Every node kind the ENGINE ships: the point domain, the scope that
//! gates it, and the origins its chains start from.
//!
//! Each is an ordinary struct. Its fields are its schema — a level writes
//! them, the compiler type-checks the wiring against the ports it declares,
//! and an editor renders them by reflection. Nothing here is enumerated
//! anywhere: `#[reflect(Node)]` is what puts a kind in the registry, and
//! the registry is the vocabulary.
//!
//! A host adds kinds the same way, in its own crate, without touching this
//! file — which is the point.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use voxel_core::opgen;
use voxel_core::worldop::*;

use super::node::{Node, Ports, ReflectNode};
use super::NodeDef;
use crate::level::gate_flags;
use crate::level::{d_band_octaves, d_full_band, default_one, mat_concrete, mat_grass};
use crate::level::{DoorDef, LodGateDef, NoiseModeDef};
use crate::schema;

/// A region gate and the nodes it applies to.
///
/// Nesting intersects: a box inside a box is a box, so any depth still
/// compiles to the single packed gate a `WorldOp` carries. What this
/// replaces is the same four numbers repeated on sixty rows, with the
/// district structure they describe left implicit.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct Region {
    /// `[a0, a1, b0, b1]` in the axes `region_axes` samples.
    pub axes: [f32; 4],
    #[serde(default)]
    pub nodes: Vec<NodeDef>,
}

impl Node for Region {
    fn kind(&self) -> &'static str {
        "region"
    }
    fn ports(&self) -> Ports {
        (&[], &[])
    }
    fn children(&self) -> &[NodeDef] {
        &self.nodes
    }
    fn gate(&self) -> Option<[f32; 4]> {
        Some(self.axes)
    }
}

/// An origin: the register file's initial state, which emits no op.
///
/// They exist so "every input is named" has no exception at the start of a
/// chain, and so an editor can show where one begins.
macro_rules! origin {
    ($name:ident, $kind:literal, $port:literal, $value:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
        #[reflect(Node, Serialize, Deserialize, Default)]
        pub struct $name;

        impl Node for $name {
            fn kind(&self) -> &'static str {
                $kind
            }
            fn ports(&self) -> Ports {
                (&[], &[($port, $value)])
            }
        }
    };
}

origin!(
    HeightZero,
    "height_zero",
    "height",
    opgen::Value::Height,
    "Sea level: the height register before anything adds to it."
);
origin!(
    SdfVoid,
    "sdf_void",
    "sdf",
    opgen::Value::Sdf,
    "Empty space: the SDF register before anything merges into it."
);
origin!(
    WarpNone,
    "warp_none",
    "warp",
    opgen::Value::Warp,
    "No domain warp: the offset height and field ops sample through before\nany `warp_xz` bends it."
);

/// A band-limited FBM heightfield band added to the height register.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct HeightFbm {
    #[serde(default)]
    pub offset: [f32; 2],
    /// Cycles per meter of the first octave.
    pub scale: f32,
    /// Amplitude in meters.
    pub amp: f32,
    pub octaves: u32,
    /// Octave shaping: plain fbm, ridged crests, or rounded billows.
    #[serde(default)]
    pub mode: NoiseModeDef,
}

impl Node for HeightFbm {
    fn kind(&self) -> &'static str {
        "height_fbm"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_HEIGHT_FBM).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            offset,
            scale,
            amp,
            octaves,
            mode,
        } = *self;
        Some(
            WorldOp::new(WOP_HEIGHT_FBM)
                .p0([offset[0], offset[1], scale, amp])
                .p1([octaves as f32, mode as u32 as f32, 0.0, 0.0]),
        )
    }
}

/// Domain-warp the XZ coordinate later height ops sample (swirled
/// coastlines, eroded-looking ridges).
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct WarpXz {
    #[serde(default)]
    pub offset: [f32; 2],
    pub scale: f32,
    /// Warp amplitude in meters.
    pub amp: f32,
    pub octaves: u32,
}

impl Node for WarpXz {
    fn kind(&self) -> &'static str {
        "warp_xz"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_WARP_XZ).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            offset,
            scale,
            amp,
            octaves,
        } = *self;
        Some(
            WorldOp::new(WOP_WARP_XZ)
                .p0([scale, amp, offset[0], offset[1]])
                .p1([octaves as f32, 0.0, 0.0, 0.0]),
        )
    }
}

/// Anisotropic 3D noise solid: union it in (floating islands, mesas)
/// or carve it out (caves, overhangs).
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct Fbm3 {
    /// Cycles per meter horizontally.
    pub scale: f32,
    /// Vertical-to-horizontal frequency ratio (>1 squashes flat).
    #[serde(default = "default_one")]
    pub y_ratio: f32,
    pub octaves: u32,
    /// Noise iso level the surface sits at (~[-0.5, 0.5]).
    pub threshold: f32,
    /// Meters per unit of noise beyond the threshold.
    pub width: f32,
    #[serde(default)]
    pub offset: [f32; 3],
    #[serde(default)]
    pub carve: bool,
    #[serde(default = "mat_grass")]
    #[reflect(@schema::OneOf("materials[].id"))]
    pub material: u32,
}

impl Node for Fbm3 {
    fn kind(&self) -> &'static str {
        "fbm3"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_FBM3).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            scale,
            y_ratio,
            octaves,
            threshold,
            width,
            offset,
            carve,
            material,
        } = *self;
        Some(
            WorldOp::new(WOP_FBM3)
                .material(material)
                .p0([scale, scale * y_ratio, threshold, width])
                .p1([
                    offset[0],
                    offset[1],
                    offset[2],
                    if carve { 1.0 } else { 0.0 },
                ])
                .p2([octaves as f32, 0.0, 0.0, 0.0]),
        )
    }
}

/// Constant meters added to the height register.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct HeightOffset {
    value: f32,
}

impl Node for HeightOffset {
    fn kind(&self) -> &'static str {
        "height_offset"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_HEIGHT_OFFSET).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self { value } = *self;
        Some(WorldOp::new(WOP_HEIGHT_OFFSET).p0([value, 0.0, 0.0, 0.0]))
    }
}

/// Cliff step: terrain crossing the `[start, end]` altitude band grows
/// an `amp`-meter wall (iq's Rainforest cliff term).
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct HeightStep {
    start: f32,
    end: f32,
    amp: f32,
}

impl Node for HeightStep {
    fn kind(&self) -> &'static str {
        "height_step"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_HEIGHT_STEP).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self { start, end, amp } = *self;
        Some(WorldOp::new(WOP_HEIGHT_STEP).p0([start, end, amp, 0.0]))
    }
}

/// Accumulate an FBM band into a field register: named world data for
/// spawner densities and gameplay queries (never the SDF itself).
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct Field {
    #[serde(default)]
    pub offset: [f32; 2],
    pub scale: f32,
    pub amp: f32,
    pub octaves: u32,
    #[serde(default)]
    pub mode: NoiseModeDef,
    #[serde(default)]
    pub bias: f32,
}

impl Node for Field {
    fn kind(&self) -> &'static str {
        "field"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_FIELD).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let Self {
            offset,
            scale,
            amp,
            octaves,
            mode,
            bias,
        } = *self;
        Some(
            WorldOp::new(WOP_FIELD)
                .p0([offset[0], offset[1], scale, amp])
                .p1([octaves as f32, mode as u32 as f32, field_slot as f32, bias]),
        )
    }
}

/// Sample the two region axes every band op in this program tests.
/// Must come before them; in practice, first.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct RegionAxes {
    /// Cycles per meter of each axis. 1e-4 is a ten-kilometre region.
    pub scale: [f32; 2],
    /// Sample offsets `(a_x, a_z, b_x, b_z)`, which is what makes the
    /// two axes independent of each other.
    pub offset: [f32; 4],
    #[serde(default = "d_band_octaves")]
    pub octaves: u32,
}

impl Node for RegionAxes {
    fn kind(&self) -> &'static str {
        "region_axes"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_REGION_AXES).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            scale,
            offset,
            octaves,
        } = *self;
        Some(
            WorldOp::new(WOP_REGION_AXES)
                .p0([offset[0], offset[1], scale[0], scale[1]])
                .p1([offset[2], offset[3], octaves as f32, 0.0]),
        )
    }
}

/// Add terrain shaped by a region: dunes in one, ridges in another.
///
/// Faded in smoothly by how firmly the point is inside the region,
/// because two heights must blend where two materials cannot.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct HeightBandFbm {
    /// The region, in the axes `region_axes` sampled.
    pub a: [f32; 2],
    #[serde(default = "d_full_band")]
    pub b: [f32; 2],
    #[serde(default)]
    pub offset: [f32; 2],
    pub scale: f32,
    pub amp: f32,
    pub octaves: u32,
    #[serde(default)]
    pub mode: NoiseModeDef,
    /// Constant metres this region sits above (or below) the ground
    /// around it. The FBM is zero-mean, so without a lift a region
    /// digs as much as it raises.
    #[serde(default)]
    pub lift: f32,
    /// Half-width of the fade at the region edge, in band units.
    /// Derived from the band by default, and that is almost always
    /// what you want: a feather wider than HALF the band leaves both
    /// edges still fading at its centre, so the region can never
    /// reach full weight and its terrain sits at partial amplitude
    /// everywhere.
    #[serde(default)]
    pub feather: Option<f32>,
}

impl Node for HeightBandFbm {
    fn kind(&self) -> &'static str {
        "height_band_fbm"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_HEIGHT_BAND_FBM).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            a,
            b,
            offset,
            scale,
            amp,
            octaves,
            mode,
            lift,
            feather,
        } = *self;
        Some(
            WorldOp::new(WOP_HEIGHT_BAND_FBM)
                .p0([offset[0], offset[1], scale, amp])
                .p1([
                    octaves as f32,
                    mode as u32 as f32,
                    feather.unwrap_or_else(|| voxel_worldgen::program::band_feather(a)),
                    lift,
                ])
                .p2([a[0], a[1], b[0], b[1]]),
        )
    }
}

/// Repaint the surface material inside a band of two noise axes.
///
/// The engine has no idea what a region IS — it compares two numbers
/// against a box and swaps a material id. A level composes several of
/// these over one `height_surface` to divide a plane into regions,
/// and names them whatever it likes in its own file.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct MaterialBand {
    /// Material to repaint FROM: only ground the earlier ops left as
    /// this id is affected, so roads and water are never touched.
    pub from: u32,
    /// Material to repaint TO.
    #[reflect(@schema::OneOf("materials[].id"))]
    pub material: u32,
    /// Half-open band on the first axis, in 0..1.
    pub a: [f32; 2],
    /// Half-open band on the second axis, in 0..1.
    #[serde(default = "d_full_band")]
    pub b: [f32; 2],
}

impl Node for MaterialBand {
    fn kind(&self) -> &'static str {
        "material_band"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_MATERIAL_BAND).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            from,
            material,
            a,
            b,
        } = *self;
        Some(
            WorldOp::new(WOP_MATERIAL_BAND)
                .material(material)
                .p0([a[0], a[1], b[0], b[1]])
                .p1([0.0, 0.0, from as f32, 0.0]),
        )
    }
}

/// Turn the accumulated height into ground.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct HeightSurface {
    #[serde(default = "mat_grass")]
    #[reflect(@schema::OneOf("materials[].id"))]
    pub material: u32,
}

impl Node for HeightSurface {
    fn kind(&self) -> &'static str {
        "height_surface"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_HEIGHT_SURFACE).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self { material } = *self;
        Some(WorldOp::new(WOP_HEIGHT_SURFACE).material(material))
    }
}

/// Solid mass at coarse LODs (the structure reads as filled from afar).
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct CoarseSolid {
    #[serde(default = "mat_concrete")]
    #[reflect(@schema::OneOf("materials[].id"))]
    pub material: u32,
}

impl Node for CoarseSolid {
    fn kind(&self) -> &'static str {
        "coarse_solid"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_COARSE_SOLID).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self { material } = *self;
        Some(
            WorldOp::new(WOP_COARSE_SOLID)
                .flags(WOP_FLAG_COARSE_ONLY)
                .material(material),
        )
    }
}

/// Establish the structural Y lattice used by slabs/holes/walls/beams.
///
/// Carries the same `lod` gate as everything that reads it: a
/// district whose storeys are hundreds of metres apart can afford to
/// exist at coarse LOD, and its slabs cannot do that without the
/// lattice that puts them at a height. Without it `fy` stays at `p.y`
/// and every floor in the world collapses onto y = 0.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct LatticeY {
    pub spacing: f32,
    #[serde(default)]
    pub lod: LodGateDef,
}

impl Node for LatticeY {
    fn kind(&self) -> &'static str {
        "lattice_y"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_LATTICE_Y).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self { spacing, lod } = *self;
        Some(
            WorldOp::new(WOP_LATTICE_Y)
                .flags(gate_flags(lod, WOP_FLAG_FINE_ONLY))
                .p0([spacing, 0.0, 0.0, 0.0]),
        )
    }
}

/// Floor slabs on the lattice.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct SlabsY {
    pub half_thickness: f32,
    #[serde(default = "mat_concrete")]
    #[reflect(@schema::OneOf("materials[].id"))]
    pub material: u32,
    #[serde(default)]
    pub lod: LodGateDef,
}

impl Node for SlabsY {
    fn kind(&self) -> &'static str {
        "slabs_y"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_SLABS_Y).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            half_thickness,
            material,
            lod,
        } = *self;
        Some(
            WorldOp::new(WOP_SLABS_Y)
                .flags(gate_flags(lod, WOP_FLAG_FINE_ONLY))
                .material(material)
                .p0([half_thickness, 0.0, 0.0, 0.0]),
        )
    }
}

/// Hash-gated holes cut through the slabs on an XZ grid.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct GridHoles {
    pub cell: f32,
    pub chance: f32,
    pub half: [f32; 3],
    #[serde(default)]
    pub lod: LodGateDef,
}

impl Node for GridHoles {
    fn kind(&self) -> &'static str {
        "grid_holes"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_GRID_HOLES).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            cell,
            chance,
            half,
            lod,
        } = *self;
        Some(
            WorldOp::new(WOP_GRID_HOLES)
                .flags(gate_flags(lod, WOP_FLAG_FINE_ONLY))
                .p0([cell, chance, 0.0, 0.0])
                .p1([half[0], half[1], half[2], 0.0]),
        )
    }
}

/// Square columns on a jittered XZ grid.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct PillarsXz {
    pub spacing: f32,
    pub jitter: f32,
    /// Base and hash-scaled extra half-width.
    pub girth: [f32; 2],
    #[serde(default = "mat_concrete")]
    #[reflect(@schema::OneOf("materials[].id"))]
    pub material: u32,
    #[serde(default)]
    pub lod: LodGateDef,
}

impl Node for PillarsXz {
    fn kind(&self) -> &'static str {
        "pillars_xz"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_PILLARS_XZ).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            spacing,
            jitter,
            girth,
            material,
            lod,
        } = *self;
        Some(
            WorldOp::new(WOP_PILLARS_XZ)
                .flags(gate_flags(lod, WOP_FLAG_FINE_ONLY))
                .material(material)
                .p0([spacing, jitter, girth[0], girth[1]]),
        )
    }
}

/// Hash-gated axis-aligned walls with optional doorways.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct Walls {
    /// "x" walls are x-normal (vary along x), "z" walls z-normal.
    pub axis: String,
    pub spacing: f32,
    pub half_thickness: f32,
    pub chance: f32,
    /// Decorrelation salt added to the wall index in the gate hash.
    #[serde(default)]
    pub salt: i32,
    #[serde(default)]
    pub door: Option<DoorDef>,
    #[serde(default = "mat_concrete")]
    #[reflect(@schema::OneOf("materials[].id"))]
    pub material: u32,
    #[serde(default)]
    pub lod: LodGateDef,
}

impl Node for Walls {
    fn kind(&self) -> &'static str {
        "walls"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_WALLS).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            ref axis,
            spacing,
            half_thickness,
            chance,
            salt,
            ref door,
            material,
            lod,
        } = *self;
        Some({
            let axis_flag = if axis == "z" { 1.0 } else { 0.0 };
            let mut op = WorldOp::new(WOP_WALLS)
                .flags(gate_flags(lod, WOP_FLAG_FINE_ONLY))
                .material(material)
                .p0([spacing, half_thickness, chance, axis_flag]);
            if let Some(d) = door {
                op = op
                    .p1([salt as f32, d.cell, d.chance, d.salt as f32])
                    .p2([d.half[0], d.half[1], d.half[2], d.y]);
            } else {
                // No doorways: chance 0 never passes the hash gate.
                op = op.p1([salt as f32, 1.0, 0.0, 0.0]);
            }
            op
        })
    }
}

/// Vertical shaft registers on a jittered XZ grid (cut them with
/// `shafts_cut`; catwalks with `beams`).
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct ShaftsXz {
    pub spacing: f32,
    pub jitter: f32,
    /// Base and hash-scaled extra radius.
    pub radius: [f32; 2],
}

impl Node for ShaftsXz {
    fn kind(&self) -> &'static str {
        "shafts_xz"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_SHAFTS_XZ).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            spacing,
            jitter,
            radius,
        } = *self;
        Some(WorldOp::new(WOP_SHAFTS_XZ).p0([spacing, jitter, radius[0], radius[1]]))
    }
}

/// Carve the shafts out of everything merged so far.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct ShaftsCut;

impl Node for ShaftsCut {
    fn kind(&self) -> &'static str {
        "shafts_cut"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_SHAFTS_CUT).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        Some(WorldOp::new(WOP_SHAFTS_CUT))
    }
}

/// Meta: the world has a water surface at this sea level (drives the
/// ocean draw and shoreline; no SDF effect).
/// Catwalk beams bridging the shafts on every Nth lattice level.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct Beams {
    pub every: u32,
    pub half_width: f32,
    #[serde(default)]
    pub y: f32,
    pub half_height: f32,
    pub reach: f32,
    #[serde(default = "mat_concrete")]
    #[reflect(@schema::OneOf("materials[].id"))]
    pub material: u32,
    #[serde(default)]
    pub lod: LodGateDef,
}

impl Node for Beams {
    fn kind(&self) -> &'static str {
        "beams"
    }
    fn ports(&self) -> Ports {
        opgen::ports(WOP_BEAMS).unwrap_or((&[], &[]))
    }
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        let Self {
            every,
            half_width,
            y,
            half_height,
            reach,
            material,
            lod,
        } = *self;
        Some(
            WorldOp::new(WOP_BEAMS)
                .flags(gate_flags(lod, WOP_FLAG_FINE_ONLY))
                .material(material)
                .p0([every as f32, half_width, y, half_height])
                .p1([reach, 0.0, 0.0, 0.0]),
        )
    }
}

/// Put every kind this crate ships in the registry.
///
/// A list, but not a list anything can silently fall off: a kind missing
/// here fails to load with "no node kind called X is registered", named,
/// the first time a level uses it.
pub fn register(registry: &mut bevy::reflect::TypeRegistry) {
    registry.register::<Beams>();
    registry.register::<CoarseSolid>();
    registry.register::<Fbm3>();
    registry.register::<Field>();
    registry.register::<GridHoles>();
    registry.register::<HeightBandFbm>();
    registry.register::<HeightFbm>();
    registry.register::<HeightOffset>();
    registry.register::<HeightStep>();
    registry.register::<HeightSurface>();
    registry.register::<HeightZero>();
    registry.register::<LatticeY>();
    registry.register::<MaterialBand>();
    registry.register::<PillarsXz>();
    registry.register::<Region>();
    registry.register::<RegionAxes>();
    registry.register::<SdfVoid>();
    registry.register::<ShaftsCut>();
    registry.register::<ShaftsXz>();
    registry.register::<SlabsY>();
    registry.register::<Walls>();
    registry.register::<WarpNone>();
    registry.register::<WarpXz>();
}
