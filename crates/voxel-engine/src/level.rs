//! Data-driven levels: a JSON `LevelDef` describes everything the engine
//! needs to present a world — the *generator program* that is the world's
//! geometry (including water/vegetation meta ops), the material table its
//! ops reference, the lighting/haze environment, seed, LOD configuration,
//! camera, and parameterized planning-op providers. Level editors author
//! these files; the engine has no hardcoded worlds — a lush planet and a
//! concrete megacity are the same interpreter fed different data.

use std::sync::Arc;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use voxel_core::csg::CsgOp;
use voxel_core::worldop::*;
use voxel_core::ChunkKey;

use crate::streaming::{ChunkOpsProvider, StreamingRebuild};
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
    /// Haze color; density is per meter (planet ~6e-5, interior ~3.5e-3).
    pub haze_color: [f32; 3],
    pub haze_density: f32,
    /// Haze tint toward the sun direction (power 0 disables).
    pub haze_sun_tint: [f32; 3],
    pub haze_tint_power: f32,
    /// Direction the sun comes FROM (not normalized; twins normalize).
    /// Engine data: the mesh shader bakes horizon shadows along it.
    #[serde(default = "d_sun_direction")]
    pub sun_direction: [f32; 3],
    pub sun_color: [f32; 3],
    /// 0 = sunless interior (ambient only).
    pub sun_strength: f32,
    pub ambient_sky: [f32; 3],
    pub ambient_ground: [f32; 3],
    pub ambient_strength: f32,
    /// Exponent on up-ness: 1 = hemispheric, 2 = top-lit interior.
    pub ambient_exponent: f32,
}

fn d_sun_direction() -> [f32; 3] {
    voxel_worldgen::program::DEFAULT_SUN_DIR.to_array()
}

impl Default for EnvDef {
    fn default() -> Self {
        Self {
            haze_color: [0.62, 0.72, 0.88],
            haze_density: 0.00006,
            haze_sun_tint: [0.92, 0.85, 0.72],
            haze_tint_power: 4.0,
            sun_direction: d_sun_direction(),
            sun_color: [1.0, 0.96, 0.88],
            sun_strength: 0.85,
            ambient_sky: [0.55, 0.70, 0.95],
            ambient_ground: [0.25, 0.24, 0.20],
            ambient_strength: 0.3,
            ambient_exponent: 1.0,
        }
    }
}

/// A scatter population: WHERE props go. What they look like is the
/// host's business — the engine spawns entities carrying
/// [`crate::scatter::ScatterInstance`] and the host dresses them.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ScatterDef {
    /// Host-facing name for this population ("tree", "boulder", …).
    pub class: String,
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
fn d_op_material() -> u32 {
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

/// The high-density instanced grass population. Blades are generated in
/// the grass shader and colored from this data — hosts that want their
/// own foliage use a [`ScatterDef`] class instead.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GrassDef {
    #[serde(default = "d_grass_tile")]
    pub tile_m: f32,
    #[serde(default = "d_grass_radius")]
    pub radius_tiles: i32,
    #[serde(default = "d_grass_per_tile")]
    pub per_tile: u32,
    pub altitude: [f32; 2],
    #[serde(default = "d_grass_up")]
    pub min_up: f32,
    /// Blade base colors, two hues mixed per instance.
    #[serde(default = "d_grass_base")]
    pub base: [[f32; 3]; 2],
    /// Blade tip colors.
    #[serde(default = "d_grass_tip")]
    pub tip: [[f32; 3]; 2],
    /// View-distance fade (start, end) in meters.
    #[serde(default = "d_grass_fade")]
    pub fade: [f32; 2],
    /// Density from a generator field register.
    #[serde(default)]
    pub density: Option<FieldDensityDef>,
    /// Orientation + banding rules (align/tilt unused for grass).
    #[serde(default)]
    pub placement: PlacementRulesDef,
    /// "instance:biome" — spawn probability scales with the biome's
    /// blended weight (soft borders).
    #[serde(default)]
    pub biome: Option<String>,
}

fn d_grass_tile() -> f32 {
    16.0
}
fn d_grass_radius() -> i32 {
    7
}
fn d_grass_per_tile() -> u32 {
    550
}
fn d_grass_up() -> f32 {
    0.8
}
fn d_grass_base() -> [[f32; 3]; 2] {
    [[0.10, 0.22, 0.06], [0.16, 0.30, 0.09]]
}
fn d_grass_tip() -> [[f32; 3]; 2] {
    [[0.35, 0.52, 0.16], [0.55, 0.62, 0.22]]
}
fn d_grass_fade() -> [f32; 2] {
    [70.0, 110.0]
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
fn default_one() -> f32 {
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
    pub name: String,
    #[serde(default)]
    pub seed: u64,
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
    /// The instanced grass population, if the level has one.
    #[serde(default)]
    pub grass: Option<GrassDef>,
    /// How far the host queries planning data for its own prop
    /// rendering (merged impostors and the like). The engine pre-warms
    /// this radius so those queries never generate on the main thread.
    #[serde(default)]
    pub prop_query_reach_m: Option<f32>,
    /// The planning stack: generic layers composed into one LayerManager.
    #[serde(default)]
    pub stack: Vec<StackLayerDef>,
    /// Named structures the stack's `site_structure` emits build.
    #[serde(default)]
    pub structures: std::collections::HashMap<String, StructureDef>,
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
    Water {
        #[serde(default)]
        level: f32,
    },
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
            GenOpDef::Water { level } => WorldOp::new(WOP_WATER).p0([level, 0.0, 0.0, 0.0]),
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

/// A structure: what `site_structure` emits at each site. Authored as
/// data — weighted variants of parts, each placing one shape at every
/// position of an arrangement. See `voxel_worldgen::structure`.
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
    pub fn pack(&self) -> voxel_worldgen::structure::Structure {
        use voxel_worldgen::structure as rt;
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
    fn pack(&self) -> voxel_worldgen::structure::Part {
        use voxel_worldgen::structure as rt;
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
    /// Bed notch + water ribbon + surface segments along a `flow` source.
    CourseWater {
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
    ) -> voxel_worldgen::stack::EmitKind {
        use voxel_worldgen::stack::EmitKind;
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
            EmitDef::CourseWater { material, width } => EmitKind::CourseWater { material, width },
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
    validate_stack(&level.stack, &level.structures)?;
    let biome_ref = |owner: &str, reference: &str| -> Result<(), String> {
        let Some((instance, biome)) = reference.rsplit_once(':') else {
            return Err(format!(
                "spawner {owner}: biome ref {reference:?} is not \"instance:biome\""
            ));
        };
        for def in &level.stack {
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
        if def.variants.is_empty() {
            return Err(format!("scatter class {:?} has no variants", def.class));
        }
    }
    if let Some(grass) = &level.grass {
        if let Some(reference) = &grass.biome {
            biome_ref("grass", reference)?;
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
    let limit = voxel_worldgen::stack::ELEM_PAD_M;
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
                    EmitDef::CourseWater { .. } => StackKind::Flow,
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
    ) -> Result<voxel_worldgen::stack::BiomeGate, String> {
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
                    return Ok(voxel_worldgen::stack::BiomeGate {
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

    fn register(
        &self,
        stack: &[StackLayerDef],
        structures: &std::collections::HashMap<String, StructureDef>,
        mgr: &mut voxel_layers::LayerManager,
    ) {
        use voxel_worldgen::stack::*;
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
            } => mgr.register_as(
                &name,
                ConnectPaths {
                    cfg: ConnectCfg {
                        source,
                        reach_m,
                        corridor_m,
                        slope_penalty,
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
                        ..Default::default()
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
                max_chunk_edge_m,
                emit,
            } => mgr.register_as(
                &name,
                EmitPatches {
                    cfg: EmitCfg {
                        source,
                        kind: emit.to_kind(structures),
                        pad_m,
                        max_chunk_edge_m,
                    },
                    cell_m,
                    cell_y_m,
                },
            ),
        }
    }
}

impl LevelDef {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
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

fn apply_generator(level: &LevelDef) -> voxel_render::WorldProgram {
    let ops: Vec<WorldOp> = level.generator.iter().map(GenOpDef::pack).collect();
    let sun = sun_dir(level);
    voxel_worldgen::program::set_program(ops.clone());
    voxel_worldgen::program::set_seed(level.seed as u32);
    voxel_worldgen::program::set_sun_direction(sun);
    voxel_render::WorldProgram {
        ops: Arc::new(ops),
        seed: level.seed as u32,
        sun_dir: sun,
    }
}

/// The generator's water surface (its `water` op, if present).
fn water_surface(program: &voxel_render::WorldProgram) -> voxel_render::WaterSurface {
    if eval_holes_mode() {
        return voxel_render::WaterSurface {
            enabled: false,
            level: 0.0,
        };
    }
    match voxel_worldgen::program::water_level(&program.ops) {
        Some(level) => voxel_render::WaterSurface {
            enabled: true,
            level,
        },
        None => voxel_render::WaterSurface {
            enabled: false,
            level: 0.0,
        },
    }
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

fn env_params(level: &LevelDef) -> voxel_render::EnvParams {
    let e = &level.environment;
    let v = |c: [f32; 3], w: f32| Vec4::new(c[0], c[1], c[2], w);
    voxel_render::EnvParams {
        haze: v(e.haze_color, e.haze_density),
        haze_tint: v(e.haze_sun_tint, e.haze_tint_power),
        sun: v(e.sun_color, e.sun_strength),
        sky: v(e.ambient_sky, e.ambient_strength),
        ground: v(e.ambient_ground, e.ambient_exponent),
        sun_dir: sun_dir(level).extend(if eval_holes_mode() { 1.0 } else { 0.0 }),
    }
}

/// The grass shader's style block (colors/fade), from the grass spawner.
fn grass_style(level: &LevelDef) -> voxel_render::GrassStyle {
    let v = |c: [f32; 3]| Vec4::new(c[0], c[1], c[2], 0.0);
    match level.grass.as_ref() {
        Some(g) => voxel_render::GrassStyle {
            base_a: v(g.base[0]),
            base_b: v(g.base[1]),
            tip_a: v(g.tip[0]),
            tip_b: v(g.tip[1]),
            fade: Vec4::new(g.fade[0], g.fade[1], 0.0, 0.0),
        },
        None => voxel_render::GrassStyle::default(),
    }
}

/// Presents a [`LevelDef`]: engine plugins, lighting, camera, planning
/// providers, autopilot/walk controls — everything the old hardcoded demos
/// did, from data. With a `source` path, the file is watched and edits
/// hot-reload: lighting and camera apply instantly; generation parameter
/// changes rebuild the streamed world in place.
pub struct LevelPlugin {
    pub def: LevelDef,
    /// Watch this file and hot-reload the level from it.
    pub source: Option<std::path::PathBuf>,
    /// Coverage-eval rendering (monotone geometry, water off) — a test
    /// affordance the host opts into; the engine reads no environment.
    pub hole_eval: bool,
    /// Start a Bevy Remote Protocol server on this port for tooling.
    pub remote_port: Option<u16>,
}

impl LevelPlugin {
    /// Present `def` with default options.
    pub fn new(def: LevelDef) -> Self {
        Self {
            def,
            source: None,
            hole_eval: false,
            remote_port: None,
        }
    }

    /// Hot-reload the level from the file it was loaded from.
    pub fn watching(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.source = Some(path.into());
        self
    }
}

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

        app.add_message::<LevelReloaded>();
        if let Some(path) = &self.source {
            let mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
            app.insert_resource(LevelSource {
                path: path.clone(),
                mtime,
                poll: Timer::from_seconds(0.5, TimerMode::Repeating),
            })
            .add_systems(Update, reload_level);
        }

        let program = apply_generator(&level);
        let water = water_surface(&program);
        if let Err(e) = validate_level(&level) {
            panic!("level {:?} has invalid planning data: {e}", level.name);
        }
        let (ops_provider, world_query, planning_layers) = build_ops_provider(&level);
        let prepare = ops_prepare(&world_query);
        app.insert_resource(program)
            .insert_resource(prepare)
            .insert_resource(material_table(&level))
            .insert_resource(env_params(&level))
            .insert_resource(LodConfig {
                max_level: level.lod.max_level,
                top_radius: level.lod.top_radius,
                top_y: level.lod.top_y,
                split_k: level.lod.split_k,
                merge_k: level.lod.merge_k,
            })
            .insert_resource(ops_provider)
            .insert_resource(world_query)
            .insert_resource(planning_layers)
            .insert_resource(water)
            .insert_resource(grass_style(&level))
            .insert_resource(level.clone())
            .add_plugins(VoxelEnginePlugin { vegetation: true })
            .add_plugins(voxel_render::WaterPlugin)
            .add_systems(Update, roll_planning_caches)
            .init_resource::<crate::river_water::RiverWaterTiles>()
            .add_systems(Update, crate::river_water::stream_river_water);

        if let Some(port) = self.remote_port {
            let _ = port; // BRP tooling lives in voxel-debug; see VoxelRemotePlugin
        }
    }
}

/// A boxed source of planning ops for a world-space box.
type OpsSource = Arc<dyn Fn(Vec3, Vec3) -> Vec<CsgOp> + Send + Sync>;

/// Layer managers backing the ops providers, exposed so the engine can
/// roll their caches with the camera (they grow unboundedly otherwise).
#[derive(Resource, Default, Clone)]
pub struct PlanningLayers(pub Vec<Arc<voxel_layers::LayerManager>>);

/// The one facade over everything planning produces: CSG ops (with each
/// emitter's carve-horizon gate applied uniformly per chunk), cut ops
/// for spawner ground checks, clearance segments keeping props off
/// roadbeds and riverbeds, water-surface segments, and markers. This
/// replaces the per-feature side channels the engine used to grow
/// (SurfaceCutsQuery, RoadsQuery).
#[derive(Resource, Clone, Default)]
pub struct WorldQuery {
    /// The level's planning stack (one manager for all layers).
    stack: Option<Arc<voxel_layers::LayerManager>>,
    /// Emit instances and what each one can produce.
    emitters: Vec<Emitter>,
    /// Legacy op sources not yet in the stack (pockets, placements).
    sources: Vec<OpsSource>,
    /// Biome layers: (instance name, ordered biome names).
    biome_tables: Vec<(String, Vec<String>)>,
    /// Radius (m) the prop/water streamers query around the camera,
    /// `None` when the level has neither. They ignore the LOD gates, so
    /// they need their own ensure pass — a second top dependency with a
    /// different size, in LayerProcGen terms.
    streamer_radius: Option<f32>,
}

/// The ops horizon: chunks coarser than this never receive planning ops
/// at all (structures are subpixel there and haze covers the hard-cut
/// ring). Shared by the chunk provider and the ensure-load pass so they
/// agree on what the resident planning set must be.
const OPS_HORIZON_EDGE_M: f32 = 1000.0;

/// Vertical band xz-facade queries cover: enough for any current world
/// (the deepest LOD tree spans ~±2.5 km), small enough that volumetric
/// emit layers don't enumerate thousands of 132 m y-rows per query.
const FACADE_Y_M: f32 = 2_560.0;

/// One emit layer as the facade sees it. `produces` keeps a query from
/// touching — and therefore GENERATING — layers that cannot answer it: a
/// level with no water emitter must not pull its structure planning into
/// existence through `water_in`.
#[derive(Clone)]
struct Emitter {
    name: String,
    /// Carve-horizon gate in chunk-edge meters.
    gate: Option<f32>,
    water: bool,
    clearance: bool,
    markers: bool,
}

impl Emitter {
    fn new(name: String, gate: Option<f32>, emit: &EmitDef) -> Self {
        let (water, clearance, markers) = match emit {
            EmitDef::CourseWater { .. } => (true, true, false),
            EmitDef::PathSlabs { clearance, .. } => (false, *clearance, false),
            EmitDef::SiteStructure { marker, .. } | EmitDef::SiteStructure3 { marker, .. } => {
                (false, false, marker.is_some())
            }
            EmitDef::WormCuts | EmitDef::Tubes { .. } => (false, false, false),
        };
        Self {
            name,
            gate,
            water,
            clearance,
            markers,
        }
    }
}

impl WorldQuery {
    /// All ops overlapping the box, as served to a chunk of the given
    /// edge. Gated emitters drop out wholesale for coarse chunks — the
    /// gate is per chunk, never per op (a per-op gate desynchronizes
    /// neighboring LODs and cracks every seam).
    pub fn ops_in(&self, min: Vec3, max: Vec3, chunk_edge_m: f32) -> Vec<CsgOp> {
        let mut out = Vec::new();
        for source in &self.sources {
            out.extend(source(min, max));
        }
        if let Some(mgr) = &self.stack {
            for e in &self.emitters {
                if e.gate.is_none_or(|g| chunk_edge_m <= g) {
                    out.extend(voxel_worldgen::stack::patches_in(mgr, &e.name, min, max).ops);
                }
            }
        }
        out
    }

    /// Ensure the planning data every emitter needs for `keys` exists,
    /// before anything reads it. Each emitter is ensured over the union
    /// of the chunks that will actually query it: chunks past the global
    /// ops horizon or past the emitter's carve-horizon gate are excluded,
    /// so the resident planning set matches what can render.
    pub fn prepare(&self, keys: &[ChunkKey]) {
        let Some(mgr) = &self.stack else {
            return;
        };
        if std::env::var_os("VOXEL_NO_PREPARE").is_some() {
            return; // A/B kill switch: fall back to read-driven generation
        }
        // Per chunk, the exact box its ops query will cover: the chunk,
        // the density apron the provider adds, and the element padding
        // patches_in applies. A hull around all of them would generate
        // planning for the hollow middle of the LOD shell.
        let elem_pad = voxel_worldgen::stack::ELEM_PAD_M;
        let region_of = |key: &ChunkKey| -> voxel_layers::IAabb {
            let edge = key.edge_m() as f32;
            let apron = 4.0 * key.voxel_size_m() as f32;
            let lo = key.min_corner_m().as_vec3() - Vec3::splat(apron + elem_pad);
            let hi = key.min_corner_m().as_vec3() + Vec3::splat(edge + apron + elem_pad);
            voxel_layers::IAabb::new(lo.as_ivec3(), hi.as_ivec3())
        };
        let focus = keys
            .iter()
            .min_by_key(|k| k.edge_m() as i64)
            .map(|k| k.min_corner_m().as_vec3().as_ivec3())
            .unwrap_or(bevy::math::IVec3::ZERO);
        let mut regions: Vec<voxel_layers::IAabb> = Vec::with_capacity(keys.len());
        for e in &self.emitters {
            regions.clear();
            regions.extend(
                keys.iter()
                    .filter(|k| {
                        let edge = k.edge_m() as f32;
                        edge <= OPS_HORIZON_EDGE_M && !e.gate.is_some_and(|g| edge > g)
                    })
                    .map(region_of),
            );
            if regions.is_empty() {
                continue; // nothing in this batch queries this emitter
            }
            let stats = mgr.ensure_loaded_regions(&e.name, &regions, focus);
            if std::env::var_os("VOXEL_LOG_LAYERS").is_some() {
                info!(
                    "prepare {}: {} generated, {} present, {} regions",
                    e.name,
                    stats.generated,
                    stats.present,
                    regions.len()
                );
            }
        }

        // Second top dependency: the streamers query a fixed radius around
        // the camera, ungated — without this their first tiles generate
        // planning (400-step river descents, scatter filters) on the MAIN
        // thread.
        let Some(radius) = self.streamer_radius else {
            return;
        };
        let band = bevy::math::IVec3::new(radius as i32, FACADE_Y_M as i32, radius as i32);
        let near = voxel_layers::IAabb::new(focus - band, focus + band);
        // Every emitter: the prop streamers' carved-ground check
        // (`cuts_in`) queries all of them, not just the ones producing
        // water or clearance.
        for e in &self.emitters {
            mgr.ensure_loaded_regions(&e.name, std::slice::from_ref(&near), focus);
        }
        // Spawner biome gates sample a wide influence window.
        let biome_pad = bevy::math::IVec3::new(
            voxel_worldgen::stack::BIOME_INFLUENCE_CELLS,
            0,
            voxel_worldgen::stack::BIOME_INFLUENCE_CELLS,
        );
        for (name, _) in &self.biome_tables {
            mgr.ensure_loaded_regions(name, &[near.inflate(biome_pad)], focus);
        }
    }

    /// Cut ops (carved voids) overlapping the box: spawners consult this
    /// so props never seat on heightfield ground that a cave mouth or
    /// doorway has carved away.
    pub fn cuts_in(&self, min: Vec3, max: Vec3) -> Vec<CsgOp> {
        let mut ops = self.ops_in(min, max, 0.0);
        ops.retain(|op| op.kind & 1 == 1);
        ops
    }

    /// Clearance segments (roadbeds, riverbeds) overlapping the xz box.
    pub fn clearance_in(&self, min: bevy::math::Vec2, max: bevy::math::Vec2) -> Vec<[bevy::math::Vec2; 2]> {
        let (min3, max3) = (
            Vec3::new(min.x, -FACADE_Y_M, min.y),
            Vec3::new(max.x, FACADE_Y_M, max.y),
        );
        let mut out = Vec::new();
        if let Some(mgr) = &self.stack {
            for e in self.emitters.iter().filter(|e| e.clearance) {
                out.extend(voxel_worldgen::stack::patches_in(mgr, &e.name, min3, max3).clearance);
            }
        }
        out
    }

    /// Water-surface segments overlapping the xz box (river renderer).
    pub fn water_in(
        &self,
        min: bevy::math::Vec2,
        max: bevy::math::Vec2,
    ) -> Vec<voxel_worldgen::stack::WaterSeg> {
        let (min3, max3) = (
            Vec3::new(min.x, -FACADE_Y_M, min.y),
            Vec3::new(max.x, FACADE_Y_M, max.y),
        );
        let mut out = Vec::new();
        if let Some(mgr) = &self.stack {
            for e in self.emitters.iter().filter(|e| e.water) {
                out.extend(voxel_worldgen::stack::patches_in(mgr, &e.name, min3, max3).water);
            }
        }
        out
    }

    /// Blended biome weights at a point: (biome name, weight) for the
    /// named biome layer, from the level's stack. Empty if the layer is
    /// not declared.
    pub fn biomes_at(&self, instance: &str, p: bevy::math::Vec2) -> Vec<(String, f32)> {
        let Some(mgr) = &self.stack else {
            return Vec::new();
        };
        let Some(table) = self.biome_tables.iter().find_map(|(n, t)| {
            (n == instance).then_some(t)
        }) else {
            return Vec::new();
        };
        let w = voxel_worldgen::stack::biome_weights_at(mgr, instance, table.len(), p);
        table.iter().cloned().zip(w).collect()
    }

    /// Markers overlapping the xz box, optionally of one kind (findable
    /// content: dungeon entrances, points of interest).
    pub fn markers_in(
        &self,
        min: bevy::math::Vec2,
        max: bevy::math::Vec2,
        kind: Option<&str>,
    ) -> Vec<voxel_worldgen::stack::Marker> {
        let (min3, max3) = (
            Vec3::new(min.x, -FACADE_Y_M, min.y),
            Vec3::new(max.x, FACADE_Y_M, max.y),
        );
        let mut out = Vec::new();
        if let Some(mgr) = &self.stack {
            for e in self.emitters.iter().filter(|e| e.markers) {
                out.extend(
                    voxel_worldgen::stack::patches_in(mgr, &e.name, min3, max3)
                        .markers
                        .into_iter()
                        .filter(|m| kind.is_none_or(|k| m.kind == k)),
                );
            }
        }
        out
    }
}

/// The engine's top dependency: pre-generate each emitter's planning
/// closure over exactly the region the chunks in `keys` will query —
/// gate-aware, so coarse chunks never drag fine-scale planning into
/// existence. Runs in the async planning task; the per-chunk ops queries
/// that follow are cache reads (`LayerManager::read_generated` reports
/// any that are not).
fn ops_prepare(world: &WorldQuery) -> crate::streaming::ChunkOpsPrepare {
    let world = world.clone();
    crate::streaming::ChunkOpsPrepare(Some(std::sync::Arc::new(move |keys: &[ChunkKey]| {
        world.prepare(keys);
    })))
}

fn build_ops_provider(level: &LevelDef) -> (ChunkOpsProvider, WorldQuery, PlanningLayers) {
    let mut sources: Vec<OpsSource> = Vec::new();
    let mut managers: Vec<Arc<voxel_layers::LayerManager>> = Vec::new();

    // The planning stack: every layer into ONE manager, in author order.
    let (stack, emitters) = if level.stack.is_empty() {
        (None, Vec::new())
    } else {
        let mut mgr = voxel_layers::LayerManager::new(level.seed);
        let mut emitters = Vec::new();
        for def in &level.stack {
            def.register(&level.stack, &level.structures, &mut mgr);
            if let StackLayerDef::Emit {
                name,
                max_chunk_edge_m,
                emit,
                ..
            } = def
            {
                emitters.push(Emitter::new(name.clone(), *max_chunk_edge_m, emit));
            }
        }
        let mgr = Arc::new(mgr);
        managers.push(mgr.clone());
        (Some(mgr), emitters)
    };


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
            pos.y = voxel_worldgen::terrain_height(bevy::math::Vec2::new(pos.x, pos.z), 1.0)
                + p.position[1];
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

    let biome_tables = level
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
    // Streamer working set, derived from the streamers themselves: props
    // (far-forest ring dominates) and river water. Levels with neither
    // skip the pass entirely.
    let mut streamer_radius: Option<f32> = None;
    let mut want = |reach: f32| {
        streamer_radius = Some(streamer_radius.map_or(reach, |r: f32| r.max(reach)));
    };
    for def in &level.scatter {
        want((def.radius_tiles + 2) as f32 * def.tile_m);
    }
    if let Some(grass) = &level.grass {
        want((grass.radius_tiles + 2) as f32 * grass.tile_m);
    }
    if let Some(reach) = level.prop_query_reach_m {
        want(reach);
    }
    if level.stack.iter().any(|l| {
        matches!(
            l,
            StackLayerDef::Emit {
                emit: EmitDef::CourseWater { .. },
                ..
            }
        )
    }) {
        want(crate::river_water::QUERY_REACH_M);
    }
    let world = WorldQuery {
        stack,
        emitters,
        sources,
        biome_tables,
        streamer_radius,
    };
    if world.stack.is_none() && world.sources.is_empty() {
        return (ChunkOpsProvider(None), world, PlanningLayers(managers));
    }
    let wq = world.clone();
    let provider = ChunkOpsProvider(Some(Arc::new(move |key: ChunkKey| {
        // Meter-scale features apply on every level whose chunks can show
        // them at visible size: the ops horizon (where the SDF genuinely
        // loses the ops — a hard-cut seam by doctrine) must sit far enough
        // out that structures are subpixel and haze covers the ring.
        if key.edge_m() as f32 > OPS_HORIZON_EDGE_M {
            return Vec::new();
        }
        // Pad by the density apron: samples extend 2 voxels below and 3
        // above the 32-cell core, so an op grazing only the apron still
        // shapes this chunk's samples — culling it desynchronizes the
        // seam with the neighbor that keeps it (visible slit through
        // structures straddling a chunk boundary).
        let pad = 4.0 * key.voxel_size_m() as f32;
        let min = key.min_corner_m().as_vec3() - Vec3::splat(pad);
        let max = key.min_corner_m().as_vec3() + Vec3::splat(key.edge_m() as f32 + pad);
        wq.ops_in(min, max, key.edge_m() as f32)
    })));
    (provider, world, PlanningLayers(managers))
}

/// Rolling eviction for the planning-layer caches: every few seconds,
/// drop cached layer chunks far outside the region any chunk request can
/// reach (ops horizon + the widest layer padding). Everything is
/// regenerable, so the only cost of evicting too eagerly is regeneration.
fn roll_planning_caches(
    layers: Res<PlanningLayers>,
    time: Res<Time>,
    mut last: Local<f32>,
    sources: crate::StreamSourceQuery,
) {
    if time.elapsed_secs() - *last < 5.0 {
        return;
    }
    *last = time.elapsed_secs();
    let Ok(source) = sources.single() else {
        return; // no streaming source tagged yet
    };
    let camera = source.translation();
    let p = camera;
    const KEEP_M: i32 = 8_000;
    let keep = voxel_layers::IAabb::new(
        bevy::math::IVec3::new(p.x as i32 - KEEP_M, i32::MIN / 2, p.z as i32 - KEEP_M),
        bevy::math::IVec3::new(p.x as i32 + KEEP_M, i32::MAX / 2, p.z as i32 + KEEP_M),
    );
    for mgr in &layers.0 {
        mgr.evict_outside(keep);
    }
}

/// Poll the level file; apply edits live. Presentation fields (colors,
/// lights, camera speeds, split/merge tuning, shading) apply directly;
/// changes to the generator/seed/ops/LOD topology rebuild the streamed
/// world in place — including swapping in a completely different world.
#[allow(clippy::too_many_arguments)]
fn reload_level(
    mut commands: Commands,
    time: Res<Time>,
    mut source: ResMut<LevelSource>,
    mut level: ResMut<LevelDef>,
    mut lod: ResMut<LodConfig>,
    mut rebuild: ResMut<StreamingRebuild>,
    mut water: ResMut<voxel_render::WaterSurface>,
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
    if let Err(e) = validate_level(&new) {
        warn!("level reload: invalid planning data — {e}");
        return;
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
    let generator_changed =
        new.generator != level.generator || new.seed != level.seed || sun_changed;
    if generator_changed {
        let program = apply_generator(&new);
        *water = water_surface(&program);
        commands.insert_resource(program);
    }
    let regen = generator_changed
        || new.stack != level.stack
        || new.structures != level.structures
        || new.placements != level.placements
        || new.prefabs != level.prefabs
        || new.lod.max_level != level.lod.max_level
        || new.lod.top_radius != level.lod.top_radius
        || new.lod.top_y != level.lod.top_y;
    if regen {
        lod.max_level = new.lod.max_level;
        lod.top_radius = new.lod.top_radius;
        lod.top_y = new.lod.top_y;
        let (ops_provider, world_query, planning_layers) = build_ops_provider(&new);
        commands.insert_resource(ops_prepare(&world_query));
        commands.insert_resource(ops_provider);
        commands.insert_resource(world_query);
        commands.insert_resource(planning_layers);
        commands.insert_resource(crate::river_water::RiverWaterTiles::default());
        commands.insert_resource(voxel_render::RiverWater::default());
        rebuild.0 = true;
        info!("level reload: generation changed — rebuilding world");
    }
    if regen || new.scatter != level.scatter || new.grass != level.grass {
        commands.insert_resource(grass_style(&new));
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
        // The planet's features are stack data, not op providers.
        let names: Vec<&str> = planet
            .stack
            .iter()
            .map(|l| match l {
                StackLayerDef::Biomes { name, .. }
                | StackLayerDef::Scatter { name, .. }
                | StackLayerDef::Scatter3 { name, .. }
                | StackLayerDef::Connect { name, .. }
                | StackLayerDef::Connect3 { name, .. }
                | StackLayerDef::Flow { name, .. }
                | StackLayerDef::Worm { name, .. }
                | StackLayerDef::Emit { name, .. } => name.as_str(),
            })
            .collect();
        for expect in [
            "biomes",
            "sites:ruins",
            "ruins",
            "paths:roads",
            "roads",
            "rivers",
            "caves",
            "dungeons",
        ] {
            assert!(names.contains(&expect), "stack missing {expect}");
        }
        // Scatter is placement-only: classes and variants, no models.
        assert_eq!(
            planet.scatter.iter().map(|s| s.class.as_str()).collect::<Vec<_>>(),
            vec!["tree", "boulder"]
        );
        assert!(planet.scatter[0].variants.len() == 2);
        assert!(planet.grass.is_some());
        // Water is a generator op; vegetation is spawner data.
        let packed: Vec<_> = planet.generator.iter().map(GenOpDef::pack).collect();
        assert_eq!(voxel_worldgen::program::water_level(&packed), Some(0.0));
        // Materials cover the ids the generator emits.
        assert!(planet.materials.iter().any(|m| m.id() == 1));
        assert!(planet.materials.iter().any(|m| m.id() == 3));

        let mega = LevelDef::from_json(&shipped("megastructure.json")).unwrap();
        let mega_names: Vec<&str> = mega
            .stack
            .iter()
            .map(|l| match l {
                StackLayerDef::Biomes { name, .. }
                | StackLayerDef::Scatter { name, .. }
                | StackLayerDef::Scatter3 { name, .. }
                | StackLayerDef::Connect { name, .. }
                | StackLayerDef::Connect3 { name, .. }
                | StackLayerDef::Flow { name, .. }
                | StackLayerDef::Worm { name, .. }
                | StackLayerDef::Emit { name, .. } => name.as_str(),
            })
            .collect();
        for expect in ["sites:pockets", "pockets", "links", "tubes"] {
            assert!(mega_names.contains(&expect), "mega stack missing {expect}");
        }
        let packed: Vec<_> = mega.generator.iter().map(GenOpDef::pack).collect();
        assert_eq!(voxel_worldgen::program::water_level(&packed), None);
        assert!(mega.scatter.is_empty() && mega.grass.is_none());
        assert!(mega.materials.iter().any(|m| m.id() == 2));
        assert!(mega.environment.sun_strength == 0.0);
    }

    #[test]
    fn levels_roundtrip() {
        let planet = LevelDef::from_json(&shipped("planet.json")).unwrap();
        let json = serde_json::to_string(&planet).unwrap();
        let back = LevelDef::from_json(&json).unwrap();
        assert_eq!(back.generator, planet.generator);
        assert_eq!(back.materials, planet.materials);
        assert_eq!(back.environment, planet.environment);
        assert_eq!(back.stack, planet.stack);
        assert_eq!(back.scatter, planet.scatter);
        assert_eq!(back.grass, planet.grass);
    }

    use crate::PROGRAM_LOCK;

    #[test]
    fn mega_stack_serves_pockets_and_tubes_through_world_query() {
        let _lock = PROGRAM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mega = LevelDef::from_json(&shipped("megastructure.json")).unwrap();
        let packed: Vec<_> = mega.generator.iter().map(GenOpDef::pack).collect();
        voxel_worldgen::program::set_program(packed);
        voxel_worldgen::program::set_seed(mega.seed as u32);
        let (_, world, _) = build_ops_provider(&mega);
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
        let planet = LevelDef::from_json(&shipped("planet.json")).unwrap();
        validate_stack(&planet.stack, &planet.structures).unwrap();
        let mega = LevelDef::from_json(&shipped("megastructure.json")).unwrap();
        validate_stack(&mega.stack, &mega.structures).unwrap();

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
            let err = validate_stack(&parse(json), &planet.structures).unwrap_err();
            assert!(
                err.contains(expect),
                "error {err:?} missing {expect:?} for {json}"
            );
        }
    }

    #[test]
    fn spawner_biome_refs_are_validated() {
        let mut planet = LevelDef::from_json(&shipped("planet.json")).unwrap();
        validate_level(&planet).unwrap();
        if let Some(def) = planet.scatter.first_mut() {
            def.biome = Some("biomes:forrest".into());
        }
        let err = validate_level(&planet).unwrap_err();
        assert!(err.contains("forrest"), "typo not caught: {err}");
    }

    #[test]
    fn planet_stack_serves_gated_ops_through_world_query() {
        let _lock = PROGRAM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let planet = LevelDef::from_json(&shipped("planet.json")).unwrap();
        let packed: Vec<_> = planet.generator.iter().map(GenOpDef::pack).collect();
        voxel_worldgen::program::set_program(packed);
        voxel_worldgen::program::set_seed(planet.seed as u32);
        let (_, world, _) = build_ops_provider(&planet);
        // A land region large enough to hold every feature kind.
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
        assert!(!world.water_in(min2, max2).is_empty(), "no water segments");
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
        let (_, world2, _) = build_ops_provider(&planet);
        assert_eq!(fine, world2.ops_in(min, max, 12.8));
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
