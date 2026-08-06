//! Data-driven levels: a JSON `LevelDef` describes the world itself and
//! nothing else — the *generator program* that is its geometry
//! the material table those ops
//! reference, the lighting/haze environment, LOD configuration, and the
//! planning stack. Presentation belongs to the host, and the seed is a
//! runtime input ([`LevelPlugin::seed`]), so a level editor edits
//! exactly this file. The engine has no hardcoded worlds — a lush planet
//! and a concrete megacity are the same interpreter fed different data.

use std::sync::Arc;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use voxel_core::csg::CsgOp;
use voxel_core::worldop::*;

use crate::planning::{ops_provider, HostPlanning, OpsSource, WorldQuery};
use crate::streaming::StreamingRebuild;
use crate::{LodConfig, VoxelEnginePlugin};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LodDef {
    pub max_level: u8,
    pub top_radius: i32,
    pub top_y: (i32, i32),
    pub split_k: f64,
    pub merge_k: f64,
}


/// Lighting + atmosphere for the chunk draw. Every field has the sun-lit
/// outdoor default, so levels only state what differs.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct EnvDef {
    /// Direction the sun comes FROM (not normalized; twins normalize).
    /// The only lighting value the engine owns: the mesh shader bakes
    /// horizon shadows along it, so it must match the app's sun. Colors,
    /// strengths, ambient and haze are the app's — voxel surfaces shade
    /// through Bevy's PBR, so they come from its lights and `DistanceFog`.
    #[serde(default = "d_sun_direction")]
    pub sun_direction: [f32; 3],
}

fn d_sun_direction() -> [f32; 3] {
    voxel_worldgen::program::DEFAULT_SUN_DIR.to_array()
}

impl Default for EnvDef {
    fn default() -> Self {
        Self {
            sun_direction: d_sun_direction(),
        }
    }
}

/// How a population's placements reach the host.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScatterOutput {
    /// One entity per placement, carrying [`crate::scatter::ScatterInstance`].
    #[default]
    Entities,
    /// Positions + hashes in a shared buffer, for populations too dense
    /// for entities (ground cover, pebbles, sparks).
    Points,
}

/// A scatter population: WHERE props go. What they look like is the
/// host's business — the engine spawns entities carrying
/// [`crate::scatter::ScatterInstance`] and the host dresses them.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ScatterDef {
    /// Host-facing name for this population. Any string: the engine never
    /// interprets it, it only tags the placements so the host can decide
    /// what a member of this population is.
    pub class: String,
    /// What a placement becomes: an entity the host dresses, or a point
    /// in a bulk buffer the host draws. Both are just placements.
    #[serde(default)]
    pub output: ScatterOutput,
    #[serde(default = "d_scatter_tile")]
    pub tile_m: f32,
    /// Streaming radius in tiles.
    #[serde(default = "d_scatter_radius")]
    pub radius_tiles: i32,
    /// Placement attempts per tile at full patch density.
    pub per_tile: u32,
    /// Chance each surviving candidate is kept.
    #[serde(default = "default_one")]
    pub chance: f32,
    /// Altitude band the class lives in.
    pub altitude: [f32; 2],
    /// Minimum surface up-ness (1 = flat).
    #[serde(default)]
    pub min_up: f32,
    /// Voxel size the surface is sampled at: props seat on the LOD the
    /// terrain actually shows across the streaming radius.
    #[serde(default = "d_detail_vs")]
    pub detail_vs: f32,
    /// Coherent patch noise (stands with clearings).
    #[serde(default)]
    pub patch: Option<PatchDef>,
    /// Density from a generator field register.
    #[serde(default)]
    pub density: Option<FieldDensityDef>,
    /// `"instance:biome"` gate.
    #[serde(default)]
    pub biome: Option<String>,
    /// Respect planning clearance (roadbeds, riverbeds).
    #[serde(default = "d_true")]
    pub clearance: bool,
    /// Orientation and banding rules.
    #[serde(default)]
    pub placement: PlacementRulesDef,
    /// Embed depth below the surface, absolute and scale-proportional.
    #[serde(default)]
    pub sink_m: f32,
    #[serde(default)]
    pub sink_scaled: f32,
    /// Exponent on the scale sample: 1 uniform, >1 biases small.
    #[serde(default = "default_one")]
    pub scale_bias: f32,
    /// Weighted variants — species, size tiers, whatever the host maps
    /// them to. The engine only knows their index.
    /// Entity-output variants. Point populations have none: a point is a
    /// position and a hash, and the host decides what it draws there.
    #[serde(default)]
    pub variants: Vec<ScatterVariantDef>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ScatterVariantDef {
    #[serde(default = "default_one")]
    pub weight: f32,
    /// Altitude band this variant is eligible in.
    #[serde(default = "d_any_altitude")]
    pub altitude: [f32; 2],
    /// Uniform scale range.
    #[serde(default = "d_unit_scale")]
    pub scale: [f32; 2],
}

fn d_true() -> bool {
    true
}

fn d_scatter_tile() -> f32 {
    64.0
}
fn d_scatter_radius() -> i32 {
    6
}
fn d_detail_vs() -> f32 {
    4.0
}
fn d_any_altitude() -> [f32; 2] {
    [f32::MIN, f32::MAX]
}
fn d_unit_scale() -> [f32; 2] {
    [1.0, 1.0]
}

/// One authored CSG primitive in a prefab's local space.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CsgOpDef {
    /// "box" or "cylinder".
    pub shape: String,
    /// "add" (default) or "cut".
    #[serde(default = "d_op_add")]
    pub op: String,
    #[serde(default)]
    pub center: [f32; 3],
    /// Box half extents (box shape).
    #[serde(default)]
    pub half: [f32; 3],
    /// Cylinder radius / half height (cylinder shape).
    #[serde(default)]
    pub radius: f32,
    #[serde(default)]
    pub half_height: f32,
    #[serde(default)]
    pub yaw_deg: f32,
    #[serde(default = "d_op_material")]
    pub material: u32,
    #[serde(default)]
    pub blend: f32,
}

fn d_op_add() -> String {
    "add".into()
}
pub fn d_op_material() -> u32 {
    3
}

impl CsgOpDef {
    pub fn to_op(&self) -> CsgOp {
        let cut = self.op == "cut";
        let mut op = if self.shape == "cylinder" {
            CsgOp::cylinder(
                bevy::math::Vec3::from(self.center),
                self.radius,
                self.half_height,
                self.material,
                cut,
            )
        } else {
            CsgOp::boxy(
                bevy::math::Vec3::from(self.center),
                bevy::math::Vec3::from(self.half),
                self.yaw_deg.to_radians(),
                self.material,
                cut,
            )
        };
        op.blend = self.blend;
        op
    }
}

/// A hand-authored instance of a prefab (or inline ops) in the world —
/// VoxelPlugin's placeable asset items as level data. Applied after the
/// procedural op providers, ordered by `priority`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PlacementDef {
    /// Name into the level's `prefabs` table...
    #[serde(default)]
    pub prefab: Option<String>,
    /// ...or inline local-space ops.
    #[serde(default)]
    pub ops: Vec<CsgOpDef>,
    pub position: [f32; 3],
    #[serde(default)]
    pub yaw_deg: f32,
    #[serde(default = "default_one")]
    pub scale: f32,
    /// Seat on the terrain surface: `position[1]` becomes an offset from
    /// the heightfield at (x, z).
    #[serde(default)]
    pub snap_to_terrain: bool,
    #[serde(default)]
    pub priority: i32,
}

/// Spawn density driven by a generator field register (`field` op):
/// gate = clamp(field[slot] * scale + offset, 0, 1). Shared world data —
/// several spawners (and future consumers) can reference one field.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct FieldDensityDef {
    pub field: u32,
    #[serde(default = "default_one")]
    pub scale: f32,
    #[serde(default)]
    pub offset: f32,
}

/// Orientation + banding rules shared by prop spawners (VoxelPlugin's
/// BasicSpawner placement block as data).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct PlacementRulesDef {
    /// Up-ness (surface normal Y) interval: 1 = flat ground. The lower
    /// bound is the spawner's legacy `min_up`; an upper bound below 1
    /// restricts to slopes (scree, wall moss).
    pub max_up: f32,
    /// Soft edge width (meters) on the altitude band: spawn probability
    /// fades linearly across it instead of a hard cut.
    pub altitude_falloff: f32,
    /// "up" (default) or "normal": align instances to the world up axis
    /// or the terrain surface normal.
    pub align: String,
    /// Random tilt (degrees) from the alignment axis.
    pub tilt_deg: f32,
    /// Embed depth (meters) below the seated surface point
    /// (None = the spawner's legacy default).
    pub sink: Option<f32>,
}

impl Default for PlacementRulesDef {
    fn default() -> Self {
        Self {
            max_up: 1.0,
            altitude_falloff: 0.0,
            align: "up".into(),
            tilt_deg: 0.0,
            sink: None,
        }
    }
}

/// Coherent-patch noise for spawn density (clearings in a forest).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PatchDef {
    pub scale: f32,
    #[serde(default)]
    pub offset: [f32; 2],
    pub contrast: f32,
    pub bias: f32,
}



/// One material recipe, referenced by the material ids generator ops emit.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MaterialDef {
    /// Uniform base with grain, optional bands/grime/streaks/moss/emissive.
    Surface {
        id: u32,
        base: [f32; 3],
        #[serde(default = "default_grain")]
        grain: f32,
        #[serde(default)]
        band: Option<BandDef>,
        #[serde(default)]
        grime: Option<GrimeDef>,
        #[serde(default)]
        streaks: f32,
        #[serde(default)]
        moss: Option<MossDef>,
        #[serde(default)]
        emissive: Option<EmissiveDef>,
        #[serde(default = "default_fade")]
        detail_fade: f32,
    },
    /// Forested zoned terrain: crown-noise canopy between the low and
    /// rock zones (perturbed normals + crown AO), strata-bumped rock
    /// above with an implicit snowcap. After iq's Rainforest shading.
    Canopy {
        id: u32,
        low: [f32; 3],
        /// (dark, sun-lit) canopy greens mixed by crown noise.
        canopy: [[f32; 3]; 2],
        rock: [f32; 3],
        /// Dry/brown patch color on gentle ground.
        patch: [f32; 3],
        zones: CanopyZonesDef,
        /// Crown noise (scale 1/m, normal relief).
        #[serde(default = "default_crowns")]
        crowns: [f32; 2],
        /// Rock strata bumps (scale 1/m, normal relief).
        #[serde(default = "default_strata")]
        strata: [f32; 2],
        #[serde(default = "default_steep")]
        steep: [f32; 2],
        #[serde(default = "default_patch_amount")]
        patch_amount: f32,
        #[serde(default = "default_zoned_fade")]
        detail_fade: f32,
    },
    /// Altitude-zoned natural terrain (low/mid/high/peak with noisy
    /// borders and a slope override to the high color).
    Zoned {
        id: u32,
        low: [f32; 3],
        /// Two hues mixed by large-scale noise.
        mid: [[f32; 3]; 2],
        /// Two hues banded by altitude.
        high: [[f32; 3]; 2],
        peak: [f32; 3],
        /// (start altitude, blend width) for mid/high/peak transitions.
        zones: ZonesDef,
        /// Slope override: up-ness (hi, lo) mapping to the high color.
        #[serde(default = "default_steep")]
        steep: [f32; 2],
        #[serde(default = "default_zoned_fade")]
        detail_fade: f32,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BandDef {
    pub freq: f32,
    pub amp: f32,
    pub lo: f32,
    pub hi: f32,
    #[serde(default)]
    pub warp: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GrimeDef {
    pub tint: [f32; 3],
    pub amount: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MossDef {
    pub color: [f32; 3],
    pub amount: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EmissiveDef {
    pub color: [f32; 3],
    #[serde(default = "default_one")]
    pub intensity: f32,
    /// Strip spacing along z / vertical level spacing (meters).
    pub spacing: f32,
    pub level_spacing: f32,
    /// Chance a strip is lit.
    pub chance: f32,
    /// Up-glow intensity on floors below.
    #[serde(default)]
    pub glow: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CanopyZonesDef {
    /// (start altitude, blend width) where canopy replaces the low color.
    pub canopy: [f32; 2],
    /// (start altitude, blend width) where rock replaces canopy.
    pub rock: [f32; 2],
    #[serde(default = "default_border")]
    pub border: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ZonesDef {
    pub mid: [f32; 2],
    pub high: [f32; 2],
    pub peak: [f32; 2],
    #[serde(default = "default_border")]
    pub border: f32,
}

fn default_grain() -> f32 {
    0.35
}
fn default_fade() -> f32 {
    0.004
}
fn default_zoned_fade() -> f32 {
    0.002
}
fn default_steep() -> [f32; 2] {
    [0.72, 0.45]
}
fn default_border() -> f32 {
    60.0
}
pub fn default_one() -> f32 {
    1.0
}
fn default_crowns() -> [f32; 2] {
    [0.35, 0.9]
}
fn default_strata() -> [f32; 2] {
    [0.15, 1.2]
}
fn default_patch_amount() -> f32 {
    0.5
}

impl MaterialDef {
    pub fn id(&self) -> u32 {
        match *self {
            MaterialDef::Surface { id, .. }
            | MaterialDef::Zoned { id, .. }
            | MaterialDef::Canopy { id, .. } => id,
        }
    }

    /// Pack into the GPU recipe form.
    pub fn pack(&self) -> voxel_render::WorldMaterial {
        use bevy::math::{UVec4, Vec4};
        let v3 = |c: [f32; 3], w: f32| Vec4::new(c[0], c[1], c[2], w);
        match *self {
            MaterialDef::Surface {
                base,
                grain,
                ref band,
                ref grime,
                streaks,
                ref moss,
                ref emissive,
                detail_fade,
                ..
            } => {
                let b = band.clone();
                let e = emissive.clone();
                voxel_render::WorldMaterial {
                    head: UVec4::new(voxel_render::MAT_KIND_SURFACE, 0, 0, 0),
                    c0: v3(base, grain),
                    c1: grime.as_ref().map_or(Vec4::ZERO, |g| v3(g.tint, g.amount)),
                    c2: moss.as_ref().map_or(Vec4::ZERO, |m| v3(m.color, m.amount)),
                    c3: e.as_ref().map_or(Vec4::ZERO, |e| v3(e.color, e.intensity)),
                    p0: b
                        .as_ref()
                        .map_or(Vec4::ZERO, |b| Vec4::new(b.freq, b.amp, b.lo, b.hi)),
                    p1: Vec4::new(
                        b.as_ref().map_or(0.0, |b| b.warp),
                        streaks,
                        e.as_ref().map_or(1.0, |e| e.spacing),
                        e.as_ref().map_or(1.0, |e| e.level_spacing),
                    ),
                    p2: Vec4::new(
                        e.as_ref().map_or(0.0, |e| e.chance),
                        e.as_ref().map_or(0.0, |e| e.glow),
                        detail_fade,
                        0.0,
                    ),
                }
            }
            MaterialDef::Canopy {
                low,
                canopy,
                rock,
                patch,
                ref zones,
                crowns,
                strata,
                steep,
                patch_amount,
                detail_fade,
                ..
            } => voxel_render::WorldMaterial {
                head: UVec4::new(voxel_render::MAT_KIND_CANOPY, 0, 0, 0),
                c0: v3(canopy[0], zones.canopy[0]),
                c1: v3(canopy[1], zones.rock[0]),
                c2: v3(rock, zones.rock[1]),
                c3: v3(patch, zones.border),
                p0: v3(low, zones.canopy[1]),
                p1: Vec4::new(crowns[0], crowns[1], strata[0], strata[1]),
                p2: Vec4::new(steep[0], steep[1], detail_fade, patch_amount),
            },
            MaterialDef::Zoned {
                low,
                mid,
                high,
                peak,
                ref zones,
                steep,
                detail_fade,
                ..
            } => voxel_render::WorldMaterial {
                head: UVec4::new(voxel_render::MAT_KIND_ZONED, 0, 0, 0),
                c0: v3(low, zones.mid[0]),
                c1: v3(mid[0], zones.high[0]),
                c2: v3(high[0], zones.peak[0]),
                c3: v3(peak, zones.border),
                p0: v3(mid[1], zones.mid[1]),
                p1: v3(high[1], zones.high[1]),
                p2: Vec4::new(zones.peak[1], steep[0], steep[1], detail_fade),
            },
        }
    }
}

/// A complete level description.
#[derive(Resource, Serialize, Deserialize, Clone, Debug)]
pub struct LevelDef {
    #[serde(default)]
    pub environment: EnvDef,
    pub lod: LodDef,
    /// The world's base geometry (and water/vegetation meta ops),
    /// interpreted in order.
    pub generator: Vec<GenOpDef>,
    /// Material recipes for the ids the generator ops emit.
    #[serde(default)]
    pub materials: Vec<MaterialDef>,
    /// Named prefabs: reusable local-space CSG op groups for placements.
    #[serde(default)]
    pub prefabs: std::collections::HashMap<String, Vec<CsgOpDef>>,
    /// Hand-authored prefab instances in the world.
    #[serde(default)]
    pub placements: Vec<PlacementDef>,
    /// Prop populations: WHERE things go. The host decides what they
    /// look like (see [`crate::scatter::ScatterInstance`]).
    #[serde(default)]
    pub scatter: Vec<ScatterDef>,
    /// How far the host queries planning data for its own prop
    /// rendering (merged impostors and the like). The engine pre-warms
    /// this radius so those queries never generate on the main thread.
    #[serde(default)]
    pub prop_query_reach_m: Option<f32>,
    /// The host's planning data, carried verbatim. Planning layers are
    /// the game's code, so the engine never looks inside this: it hands
    /// the block to the host's [`crate::planning::HostPlanning`] and
    /// takes back a planner.
    #[serde(default)]
    pub planning: serde_json::Value,
}

/// When a generator op applies across the LOD range.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LodGateDef {
    /// The op's natural default: structural detail is fine-only, height
    /// bands and shafts are everywhere, `coarse_solid` is coarse-only.
    #[default]
    Auto,
    All,
    Fine,
    Coarse,
}

/// Doorway cuts punched through walls.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DoorDef {
    /// Spacing of candidate doorways along the wall (meters).
    pub cell: f32,
    /// Chance a candidate becomes a doorway.
    pub chance: f32,
    /// Half extents (along wall normal, up, along wall).
    pub half: [f32; 3],
    /// Doorway center height above the lattice plane (meters).
    #[serde(default)]
    pub y: f32,
    /// Decorrelation salt added to the lattice level in the hash.
    #[serde(default)]
    pub salt: i32,
}

/// One generator op — the JSON authoring form of `voxel_core::WorldOp`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenOpDef {
    /// A band-limited FBM heightfield band added to the height register.
    HeightFbm {
        #[serde(default)]
        offset: [f32; 2],
        /// Cycles per meter of the first octave.
        scale: f32,
        /// Amplitude in meters.
        amp: f32,
        octaves: u32,
        /// Octave shaping: plain fbm, ridged crests, or rounded billows.
        #[serde(default)]
        mode: NoiseModeDef,
    },
    /// Domain-warp the XZ coordinate later height ops sample (swirled
    /// coastlines, eroded-looking ridges).
    WarpXz {
        #[serde(default)]
        offset: [f32; 2],
        scale: f32,
        /// Warp amplitude in meters.
        amp: f32,
        octaves: u32,
    },
    /// Anisotropic 3D noise solid: union it in (floating islands, mesas)
    /// or carve it out (caves, overhangs).
    Fbm3 {
        /// Cycles per meter horizontally.
        scale: f32,
        /// Vertical-to-horizontal frequency ratio (>1 squashes flat).
        #[serde(default = "default_one")]
        y_ratio: f32,
        octaves: u32,
        /// Noise iso level the surface sits at (~[-0.5, 0.5]).
        threshold: f32,
        /// Meters per unit of noise beyond the threshold.
        width: f32,
        #[serde(default)]
        offset: [f32; 3],
        #[serde(default)]
        carve: bool,
        #[serde(default = "mat_grass")]
        material: u32,
    },
    /// Constant meters added to the height register.
    HeightOffset { value: f32 },
    /// Cliff step: terrain crossing the `[start, end]` altitude band grows
    /// an `amp`-meter wall (iq's Rainforest cliff term).
    HeightStep { start: f32, end: f32, amp: f32 },
    /// Accumulate an FBM band into a field register: named world data for
    /// spawner densities and gameplay queries (never the SDF itself).
    Field {
        slot: u32,
        #[serde(default)]
        offset: [f32; 2],
        scale: f32,
        amp: f32,
        octaves: u32,
        #[serde(default)]
        mode: NoiseModeDef,
        #[serde(default)]
        bias: f32,
    },
    /// Turn the accumulated height into ground.
    HeightSurface {
        #[serde(default = "mat_grass")]
        material: u32,
    },
    /// Solid mass at coarse LODs (the structure reads as filled from afar).
    CoarseSolid {
        #[serde(default = "mat_concrete")]
        material: u32,
    },
    /// Establish the structural Y lattice used by slabs/holes/walls/beams.
    LatticeY { spacing: f32 },
    /// Floor slabs on the lattice.
    SlabsY {
        half_thickness: f32,
        #[serde(default = "mat_concrete")]
        material: u32,
        #[serde(default)]
        lod: LodGateDef,
    },
    /// Hash-gated holes cut through the slabs on an XZ grid.
    GridHoles {
        cell: f32,
        chance: f32,
        half: [f32; 3],
        #[serde(default)]
        lod: LodGateDef,
    },
    /// Square columns on a jittered XZ grid.
    PillarsXz {
        spacing: f32,
        jitter: f32,
        /// Base and hash-scaled extra half-width.
        girth: [f32; 2],
        #[serde(default = "mat_concrete")]
        material: u32,
        #[serde(default)]
        lod: LodGateDef,
    },
    /// Hash-gated axis-aligned walls with optional doorways.
    Walls {
        /// "x" walls are x-normal (vary along x), "z" walls z-normal.
        axis: String,
        spacing: f32,
        half_thickness: f32,
        chance: f32,
        /// Decorrelation salt added to the wall index in the gate hash.
        #[serde(default)]
        salt: i32,
        #[serde(default)]
        door: Option<DoorDef>,
        #[serde(default = "mat_concrete")]
        material: u32,
        #[serde(default)]
        lod: LodGateDef,
    },
    /// Vertical shaft registers on a jittered XZ grid (cut them with
    /// `shafts_cut`; catwalks with `beams`).
    ShaftsXz {
        spacing: f32,
        jitter: f32,
        /// Base and hash-scaled extra radius.
        radius: [f32; 2],
    },
    /// Carve the shafts out of everything merged so far.
    ShaftsCut,
    /// Meta: the world has a water surface at this sea level (drives the
    /// ocean draw and shoreline; no SDF effect).
    /// Catwalk beams bridging the shafts on every Nth lattice level.
    Beams {
        every: u32,
        half_width: f32,
        #[serde(default)]
        y: f32,
        half_height: f32,
        reach: f32,
        #[serde(default = "mat_concrete")]
        material: u32,
        #[serde(default)]
        lod: LodGateDef,
    },
}

/// Octave shaping for `height_fbm`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum NoiseModeDef {
    #[default]
    Fbm,
    Ridged,
    Billow,
}

fn mat_grass() -> u32 {
    1
}
fn mat_concrete() -> u32 {
    2
}

fn gate_flags(lod: LodGateDef, auto: u32) -> u32 {
    match lod {
        LodGateDef::Auto => auto,
        LodGateDef::All => 0,
        LodGateDef::Fine => WOP_FLAG_FINE_ONLY,
        LodGateDef::Coarse => WOP_FLAG_COARSE_ONLY,
    }
}

impl GenOpDef {
    /// Pack into the 64-byte interpreter form.
    pub fn pack(&self) -> WorldOp {
        match *self {
            GenOpDef::HeightFbm {
                offset,
                scale,
                amp,
                octaves,
                mode,
            } => WorldOp::new(WOP_HEIGHT_FBM)
                .p0([offset[0], offset[1], scale, amp])
                .p1([octaves as f32, mode as u32 as f32, 0.0, 0.0]),
            GenOpDef::WarpXz {
                offset,
                scale,
                amp,
                octaves,
            } => WorldOp::new(WOP_WARP_XZ)
                .p0([scale, amp, offset[0], offset[1]])
                .p1([octaves as f32, 0.0, 0.0, 0.0]),
            GenOpDef::Fbm3 {
                scale,
                y_ratio,
                octaves,
                threshold,
                width,
                offset,
                carve,
                material,
            } => WorldOp::new(WOP_FBM3)
                .material(material)
                .p0([scale, scale * y_ratio, threshold, width])
                .p1([
                    offset[0],
                    offset[1],
                    offset[2],
                    if carve { 1.0 } else { 0.0 },
                ])
                .p2([octaves as f32, 0.0, 0.0, 0.0]),
            GenOpDef::HeightOffset { value } => {
                WorldOp::new(WOP_HEIGHT_OFFSET).p0([value, 0.0, 0.0, 0.0])
            }
            GenOpDef::HeightStep { start, end, amp } => {
                WorldOp::new(WOP_HEIGHT_STEP).p0([start, end, amp, 0.0])
            }
            GenOpDef::Field {
                slot,
                offset,
                scale,
                amp,
                octaves,
                mode,
                bias,
            } => WorldOp::new(WOP_FIELD)
                .p0([offset[0], offset[1], scale, amp])
                .p1([octaves as f32, mode as u32 as f32, slot as f32, bias]),
            GenOpDef::HeightSurface { material } => {
                WorldOp::new(WOP_HEIGHT_SURFACE).material(material)
            }
            GenOpDef::CoarseSolid { material } => WorldOp::new(WOP_COARSE_SOLID)
                .flags(WOP_FLAG_COARSE_ONLY)
                .material(material),
            GenOpDef::LatticeY { spacing } => WorldOp::new(WOP_LATTICE_Y)
                .flags(WOP_FLAG_FINE_ONLY)
                .p0([spacing, 0.0, 0.0, 0.0]),
            GenOpDef::SlabsY {
                half_thickness,
                material,
                lod,
            } => WorldOp::new(WOP_SLABS_Y)
                .flags(gate_flags(lod, WOP_FLAG_FINE_ONLY))
                .material(material)
                .p0([half_thickness, 0.0, 0.0, 0.0]),
            GenOpDef::GridHoles {
                cell,
                chance,
                half,
                lod,
            } => WorldOp::new(WOP_GRID_HOLES)
                .flags(gate_flags(lod, WOP_FLAG_FINE_ONLY))
                .p0([cell, chance, 0.0, 0.0])
                .p1([half[0], half[1], half[2], 0.0]),
            GenOpDef::PillarsXz {
                spacing,
                jitter,
                girth,
                material,
                lod,
            } => WorldOp::new(WOP_PILLARS_XZ)
                .flags(gate_flags(lod, WOP_FLAG_FINE_ONLY))
                .material(material)
                .p0([spacing, jitter, girth[0], girth[1]]),
            GenOpDef::Walls {
                ref axis,
                spacing,
                half_thickness,
                chance,
                salt,
                ref door,
                material,
                lod,
            } => {
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
            }
            GenOpDef::ShaftsXz {
                spacing,
                jitter,
                radius,
            } => WorldOp::new(WOP_SHAFTS_XZ).p0([spacing, jitter, radius[0], radius[1]]),
            GenOpDef::ShaftsCut => WorldOp::new(WOP_SHAFTS_CUT),
            GenOpDef::Beams {
                every,
                half_width,
                y,
                half_height,
                reach,
                material,
                lod,
            } => WorldOp::new(WOP_BEAMS)
                .flags(gate_flags(lod, WOP_FLAG_FINE_ONLY))
                .material(material)
                .p0([every as f32, half_width, y, half_height])
                .p1([reach, 0.0, 0.0, 0.0]),
        }
    }
}

impl LevelDef {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// This level's generator at `seed`. Layers need one to sample the
    /// world they are planning on top of; at runtime the engine hands the
    /// same value to [`crate::planning::HostPlanning::build`].
    pub fn generator(&self, seed: u64) -> voxel_worldgen::Generator {
        voxel_worldgen::Generator::new(
            self.generator.iter().map(GenOpDef::pack).collect(),
            seed as u32,
            sun_dir(self),
        )
    }
}

/// Pack the level's generator, install it in the CPU interpreter, and
/// produce the resource the GPU interpreter extracts — one program, two
/// twins.
/// The level's sun direction (its `sun` field, or the engine default when
/// sunless — the shadow-bake direction must still be defined).
fn sun_dir(level: &LevelDef) -> Vec3 {
    Vec3::from(level.environment.sun_direction).normalize_or(Vec3::Y)
}

/// Coverage-eval rendering: monotone-white geometry and no water, so a
/// single background-colored pixel below the horizon means missing world
/// coverage. Set from [`LevelPlugin::hole_eval`] — the engine itself
/// reads no environment variables.
#[derive(Resource, Default, Clone, Copy)]
pub struct HoleEval(pub bool);

pub fn eval_holes_mode() -> bool {
    HOLE_EVAL.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) static HOLE_EVAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Build both twins of the level's generator: the GPU upload and the
/// CPU-side [`voxel_worldgen::Generator`] every mirror samples.
fn apply_generator(
    level: &LevelDef,
    seed: u64,
) -> (voxel_render::WorldProgram, Arc<voxel_worldgen::Generator>) {
    let generator = Arc::new(level.generator(seed));
    let ops: Vec<WorldOp> = generator.ops().to_vec();
    let sun = sun_dir(level);
    (
        voxel_render::WorldProgram {
            ops: Arc::new(ops),
            seed: seed as u32,
            sun_dir: sun,
        },
        generator,
    )
}


/// Pack the level's material table, ordered by material id.
fn material_table(level: &LevelDef) -> voxel_render::WorldMaterials {
    let mut table = vec![voxel_render::WorldMaterial::default(); voxel_render::MATERIAL_SLOTS];
    for def in &level.materials {
        let id = def.id() as usize;
        if id < table.len() {
            table[id] = def.pack();
        } else {
            warn!("material id {id} out of range (max {})", table.len() - 1);
        }
    }
    voxel_render::WorldMaterials(table)
}

fn env_params(_level: &LevelDef) -> voxel_render::EnvParams {
    voxel_render::EnvParams {
        flags: Vec4::new(if eval_holes_mode() { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0),
        // Filled in the render world, where the slab slots are known.
        ..default()
    }
}


/// Presents a [`LevelDef`]: generation, streaming, meshing, materials
/// and the planning providers. Presentation the *host* owns — camera,
/// lights, clear color, prop models — is not in here and not in the
/// level file. With a `source` path the file is watched and edits
/// hot-reload: material and environment changes apply instantly;
/// generation parameter changes rebuild the streamed world in place.
pub struct LevelPlugin {
    pub def: LevelDef,
    /// World seed. A runtime input, not level data: the same level
    /// definition generates a different world per seed, so a game picks
    /// it at new-game time and restores it from its save.
    pub seed: u64,
    /// Watch this file and hot-reload the level from it.
    pub source: Option<std::path::PathBuf>,
    /// Coverage-eval rendering (monotone geometry, water off) — a test
    /// affordance the host opts into; the engine reads no environment.
    pub hole_eval: bool,
    /// Start a Bevy Remote Protocol server on this port for tooling.
    pub remote_port: Option<u16>,
    /// How this host builds its planning layers. `None` means the world
    /// is pure generator program plus authored placements — no layers.
    pub planner: Option<Arc<dyn HostPlanning>>,
}

impl LevelPlugin {
    /// Present `def` with default options.
    pub fn new(def: LevelDef) -> Self {
        Self {
            def,
            seed: 0,
            source: None,
            hole_eval: false,
            remote_port: None,
            planner: None,
        }
    }

    /// Hot-reload the level from the file it was loaded from.
    pub fn watching(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.source = Some(path.into());
        self
    }
}

/// The world seed in play, for systems that rebuild generation state.
#[derive(Resource, Clone, Copy, Debug)]
pub struct WorldSeed(pub u64);

/// The host's planner factory, kept so hot reload can rebuild the
/// planning layers against the new level.
#[derive(Resource, Clone, Default)]
pub struct HostPlanner(pub Option<Arc<dyn HostPlanning>>);

/// Watch state for level hot-reload.
#[derive(Resource)]
struct LevelSource {
    path: std::path::PathBuf,
    mtime: Option<std::time::SystemTime>,
    poll: Timer,
}

impl Plugin for LevelPlugin {
    fn build(&self, app: &mut App) {
        let level = self.def.clone();

        app.add_message::<LevelReloaded>()
            .insert_resource(WorldSeed(self.seed))
            .insert_resource(HostPlanner(self.planner.clone()));
        if let Some(path) = &self.source {
            let mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
            app.insert_resource(LevelSource {
                path: path.clone(),
                mtime,
                poll: Timer::from_seconds(0.5, TimerMode::Repeating),
            })
            .add_systems(Update, reload_level);
        }

        let (program, generator) = apply_generator(&level, self.seed);
        if let Some(host) = &self.planner {
            if let Err(e) = host.validate(&level) {
                panic!("level has invalid planning data: {e}");
            }
        }
        let world_query =
            build_ops_provider(&level, self.seed, &generator, self.planner.as_ref());
        app.insert_resource(program)
            .insert_resource(crate::planning::ops_provider(&world_query))
            .insert_resource(material_table(&level))
            .insert_resource(env_params(&level))
            .insert_resource(LodConfig {
                max_level: level.lod.max_level,
                top_radius: level.lod.top_radius,
                top_y: level.lod.top_y,
                split_k: level.lod.split_k,
                merge_k: level.lod.merge_k,
            })
            .insert_resource(world_query)
            .insert_resource(level.clone())
            .add_plugins(VoxelEnginePlugin { vegetation: true })
            .add_systems(Update, crate::planning::follow_stream_source);

        if let Some(port) = self.remote_port {
            let _ = port; // BRP tooling lives in voxel-debug; see VoxelRemotePlugin
        }
    }
}

fn build_ops_provider(
    level: &LevelDef,
    seed: u64,
    generator: &Arc<voxel_worldgen::Generator>,
    planner: Option<&Arc<dyn HostPlanning>>,
) -> WorldQuery {
    let mut sources: Vec<OpsSource> = Vec::new();

    // Authored placements: resolve prefab refs, bake world-space ops once
    // (translate + yaw + uniform scale; optional terrain seating), then
    // serve them AABB-culled like any other source — ordered by priority
    // after the procedural providers.
    let mut placed: Vec<(i32, Vec<CsgOp>)> = Vec::new();
    for p in &level.placements {
        let local: &[CsgOpDef] = match (&p.prefab, &p.ops) {
            (Some(name), _) => match level.prefabs.get(name) {
                Some(ops) => ops,
                None => {
                    warn!("placement references unknown prefab '{name}'");
                    continue;
                }
            },
            (None, ops) => ops,
        };
        let mut pos = bevy::math::Vec3::from(p.position);
        if p.snap_to_terrain {
            pos.y = generator.height(bevy::math::Vec2::new(pos.x, pos.z), 1.0) + p.position[1];
        }
        let (sin, cos) = p.yaw_deg.to_radians().sin_cos();
        let rot = |v: bevy::math::Vec3| {
            bevy::math::Vec3::new(v.x * cos - v.z * sin, v.y, v.x * sin + v.z * cos)
        };
        let ops: Vec<CsgOp> = local
            .iter()
            .map(|def| {
                let mut op = def.to_op();
                op.center = (pos + rot(bevy::math::Vec3::from(op.center) * p.scale)).to_array();
                op.half = (bevy::math::Vec3::from(op.half) * p.scale).to_array();
                op.yaw += p.yaw_deg.to_radians();
                op.blend *= p.scale;
                op
            })
            .collect();
        placed.push((p.priority, ops));
    }
    placed.sort_by_key(|(priority, _)| *priority);
    if !placed.is_empty() {
        let placed = Arc::new(placed);
        sources.push(Arc::new(move |min, max| {
            let mut out = Vec::new();
            for (_, ops) in placed.iter() {
                out.extend(ops.iter().filter(|op| op.touches(min, max)).copied());
            }
            out
        }));
    }

    let mut world = WorldQuery::new(generator.clone());
    if let Some(planner) = planner.and_then(|h| h.build(level, seed, generator)) {
        world = world.with_planner(planner);
    }
    for source in sources {
        world = world.with_source(source);
    }
    world
}

/// Poll the level file; apply edits live. Presentation fields (colors,
/// lights, camera speeds, split/merge tuning, shading) apply directly;
/// changes to the generator/ops/LOD topology rebuild the streamed
/// world in place — including swapping in a completely different world.
#[allow(clippy::too_many_arguments)]
fn reload_level(
    mut commands: Commands,
    time: Res<Time>,
    mut source: ResMut<LevelSource>,
    mut level: ResMut<LevelDef>,
    seed: Res<WorldSeed>,
    planner: Res<HostPlanner>,
    mut lod: ResMut<LodConfig>,
    mut rebuild: ResMut<StreamingRebuild>,
    mut veg_rebuild: Option<ResMut<crate::scatter::ScatterRebuild>>,
    mut reloaded: MessageWriter<LevelReloaded>,
) {
    if !source.poll.tick(time.delta()).just_finished() {
        return;
    }
    let Ok(mtime) = std::fs::metadata(&source.path).and_then(|m| m.modified()) else {
        return;
    };
    if source.mtime == Some(mtime) {
        return;
    }
    source.mtime = Some(mtime);

    let new = match std::fs::read_to_string(&source.path)
        .map_err(|e| e.to_string())
        .and_then(|json| LevelDef::from_json(&json).map_err(|e| e.to_string()))
    {
        Ok(new) => new,
        Err(e) => {
            warn!("level reload: {e}");
            return;
        }
    };
    // Authoring errors must never take down a live session: keep the
    // running world and report, exactly like a parse error.
    if let Some(host) = &planner.0 {
        if let Err(e) = host.validate(&new) {
            warn!("level reload: invalid planning data — {e}");
            return;
        }
    }

    // Engine-owned presentation: the material table and the chunk
    // shader's environment uniform.
    if new.materials != level.materials {
        commands.insert_resource(material_table(&new));
    }
    if new.environment != level.environment {
        commands.insert_resource(env_params(&new));
    }
    lod.split_k = new.lod.split_k;
    lod.merge_k = new.lod.merge_k;

    // Generation-affecting changes rebuild the streamed world.
    let sun_changed = sun_dir(&new) != sun_dir(level.as_ref());
    let generator_changed = new.generator != level.generator || sun_changed;
    // Rebuilt whether or not the program changed: the planning stack and
    // the facade below need one either way.
    let (program, generator) = apply_generator(&new, seed.0);
    if generator_changed {
        commands.insert_resource(program);
    }
    let regen = generator_changed
        || new.planning != level.planning
        || new.placements != level.placements
        || new.prefabs != level.prefabs
        || new.lod.max_level != level.lod.max_level
        || new.lod.top_radius != level.lod.top_radius
        || new.lod.top_y != level.lod.top_y;
    if regen {
        lod.max_level = new.lod.max_level;
        lod.top_radius = new.lod.top_radius;
        lod.top_y = new.lod.top_y;
        let world_query = build_ops_provider(&new, seed.0, &generator, planner.0.as_ref());
        commands.insert_resource(ops_provider(&world_query));
        commands.insert_resource(world_query);
        rebuild.0 = true;
        info!("level reload: generation changed — rebuilding world");
    }
    if regen || new.scatter != level.scatter {
        if let Some(veg) = veg_rebuild.as_mut() {
            veg.0 = true;
        }
    }

    // The host owns the scene: it reads the new definition off this
    // message and applies its own camera, lights and clear color.
    let previous = level.clone();
    *level = new;
    reloaded.write(LevelReloaded { previous });
}

/// Emitted after a hot reload so the host can re-apply whatever it owns
/// (clear color, lights, camera, window title). The engine has already
/// applied everything it owns.
#[derive(Message, Debug, Clone)]
pub struct LevelReloaded {
    /// The definition that was replaced — compare fields to see what
    /// changed. The new one is in the [`LevelDef`] resource.
    pub previous: LevelDef,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped(name: &str) -> String {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../levels/");
        std::fs::read_to_string(format!("{path}{name}")).unwrap()
    }

    #[test]
    fn shipped_levels_parse() {
        let planet = LevelDef::from_json(&shipped("planet.json")).unwrap();
        // Planning is the host's: the engine carries the block
        // without ever parsing it.
        assert!(planet.planning.is_object());
        // Scatter is placement-only: classes and variants, no models.
        assert_eq!(
            planet.scatter.iter().map(|s| s.class.as_str()).collect::<Vec<_>>(),
            vec!["tree", "boulder", "groundcover"]
        );
        assert!(planet.scatter[0].variants.len() == 2);
        // Ground cover is just another scatter population that outputs
        // points instead of entities.
        assert!(planet
            .scatter
            .iter()
            .any(|d| d.output == ScatterOutput::Points));
        // The planet's geometry comes from height ops; sea level is the
        // host's business and no longer part of the program.
        let packed: Vec<_> = planet.generator.iter().map(GenOpDef::pack).collect();
        assert!(packed.iter().any(|op| op.is_height_op()));
        // Materials cover the ids the generator emits.
        assert!(planet.materials.iter().any(|m| m.id() == 1));
        assert!(planet.materials.iter().any(|m| m.id() == 3));

        let mega = LevelDef::from_json(&shipped("megastructure.json")).unwrap();
        assert!(mega.planning.is_object());
        let packed: Vec<_> = mega.generator.iter().map(GenOpDef::pack).collect();
        assert!(mega.scatter.is_empty());
        assert!(mega.materials.iter().any(|m| m.id() == 2));
        // Sunless interior: no height ops, so the horizon-shadow bake
        // self-disables and the sun direction is unused.
        assert!(
            !packed.iter().any(|op| op.is_height_op()),
            "mega should have no height ops"
        );
    }

    #[test]
    fn levels_roundtrip() {
        let planet = LevelDef::from_json(&shipped("planet.json")).unwrap();
        let json = serde_json::to_string(&planet).unwrap();
        let back = LevelDef::from_json(&json).unwrap();
        assert_eq!(back.generator, planet.generator);
        assert_eq!(back.materials, planet.materials);
        assert_eq!(back.environment, planet.environment);
        assert_eq!(back.planning, planet.planning);
        assert_eq!(back.scatter, planet.scatter);
    }

    #[test]
    fn shipped_generators_pack_to_reference_programs() {
        // The shipped JSONs must express exactly the reference programs the
        // CPU interpreter tests verify against the legacy formulas.
        let planet = LevelDef::from_json(&shipped("planet.json")).unwrap();
        let packed: Vec<_> = planet.generator.iter().map(GenOpDef::pack).collect();
        assert_eq!(packed, voxel_worldgen::program::planet_program());

        let mega = LevelDef::from_json(&shipped("megastructure.json")).unwrap();
        let packed: Vec<_> = mega.generator.iter().map(GenOpDef::pack).collect();
        assert_eq!(packed, voxel_worldgen::program::mega_program());
    }
}
