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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SunDef {
    pub direction: [f32; 3],
    pub illuminance: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AmbientDef {
    pub color: [f32; 3],
    pub brightness: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CameraDef {
    pub start: [f32; 3],
    pub look: [f32; 3],
    pub walk_speed: f32,
    pub run_speed: f32,
}

/// How the walk mode (`VOXEL_WALK=1`) grounds the camera.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum WalkDef {
    /// Glue to the generator's heightfield (open terrain).
    #[default]
    Terrain,
    /// Gravity + capsule collision against the full SDF (interiors).
    Sdf,
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
    pub sun_color: [f32; 3],
    /// 0 = sunless interior (ambient only).
    pub sun_strength: f32,
    pub ambient_sky: [f32; 3],
    pub ambient_ground: [f32; 3],
    pub ambient_strength: f32,
    /// Exponent on up-ness: 1 = hemispheric, 2 = top-lit interior.
    pub ambient_exponent: f32,
}

impl Default for EnvDef {
    fn default() -> Self {
        Self {
            haze_color: [0.62, 0.72, 0.88],
            haze_density: 0.00006,
            haze_sun_tint: [0.92, 0.85, 0.72],
            haze_tint_power: 4.0,
            sun_color: [1.0, 0.96, 0.88],
            sun_strength: 0.85,
            ambient_sky: [0.55, 0.70, 0.95],
            ambient_ground: [0.25, 0.24, 0.20],
            ambient_strength: 0.3,
            ambient_exponent: 1.0,
        }
    }
}

/// A scattered-prop population rule — what grows/lies on the generator's
/// heightfield surface. All rules and looks are level data.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SpawnerDef {
    Trees(TreesDef),
    Grass(GrassDef),
    Boulders(BouldersDef),
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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TreesDef {
    /// Placement attempts per 64 m tile at full patch density.
    #[serde(default = "d_tree_attempts")]
    pub max_per_tile: u32,
    /// Altitude band trees grow in (meters).
    pub altitude: [f32; 2],
    /// Minimum surface up-ness (1 = flat).
    #[serde(default = "d_tree_up")]
    pub min_up: f32,
    /// Density patchiness (None = uniform forests).
    #[serde(default)]
    pub patch: Option<PatchDef>,
    pub species: Vec<SpeciesDef>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SpeciesDef {
    /// Procedural model: "conifer" (trunk + stacked cones) or "broadleaf"
    /// (trunk + blob canopy).
    pub model: String,
    /// Selection weight among species whose altitude band contains the
    /// candidate point.
    #[serde(default = "default_one")]
    pub weight: f32,
    pub altitude: [f32; 2],
    #[serde(default = "d_trunk_color")]
    pub trunk: [f32; 3],
    pub foliage: [f32; 3],
    /// Uniform scale range.
    #[serde(default = "d_tree_scale")]
    pub scale: [f32; 2],
    pub impostor: ImpostorDef,
}

/// Far-forest silhouette: crossed quads colored per species.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ImpostorDef {
    /// "cone" or "diamond".
    pub shape: String,
    pub color: [f32; 3],
    /// (half width, height) before per-tree scale.
    pub size: [f32; 2],
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GrassDef {
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
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BouldersDef {
    #[serde(default = "d_boulder_per_tile")]
    pub per_tile: u32,
    /// Chance each candidate is kept.
    #[serde(default = "d_boulder_chance")]
    pub chance: f32,
    pub altitude: [f32; 2],
    #[serde(default = "d_boulder_up")]
    pub min_up: f32,
    /// Scale range (quadratic bias toward small).
    #[serde(default = "d_boulder_scale")]
    pub scale: [f32; 2],
    #[serde(default = "d_rock_color")]
    pub color: [f32; 3],
}

fn d_tree_attempts() -> u32 {
    18
}
fn d_tree_up() -> f32 {
    0.86
}
fn d_trunk_color() -> [f32; 3] {
    [0.35, 0.24, 0.15]
}
fn d_tree_scale() -> [f32; 2] {
    [0.8, 1.7]
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
fn d_boulder_per_tile() -> u32 {
    4
}
fn d_boulder_chance() -> f32 {
    0.45
}
fn d_boulder_up() -> f32 {
    0.55
}
fn d_boulder_scale() -> [f32; 2] {
    [0.4, 2.6]
}
fn d_rock_color() -> [f32; 3] {
    [0.44, 0.42, 0.40]
}

impl LevelDef {
    /// The first spawner of each kind, if configured.
    pub fn trees(&self) -> Option<&TreesDef> {
        self.spawners.iter().find_map(|s| match s {
            SpawnerDef::Trees(t) => Some(t),
            _ => None,
        })
    }

    pub fn grass(&self) -> Option<&GrassDef> {
        self.spawners.iter().find_map(|s| match s {
            SpawnerDef::Grass(g) => Some(g),
            _ => None,
        })
    }

    pub fn boulders(&self) -> Option<&BouldersDef> {
        self.spawners.iter().find_map(|s| match s {
            SpawnerDef::Boulders(b) => Some(b),
            _ => None,
        })
    }
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

impl MaterialDef {
    pub fn id(&self) -> u32 {
        match *self {
            MaterialDef::Surface { id, .. } | MaterialDef::Zoned { id, .. } => id,
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
    pub clear_color: [f32; 3],
    pub ambient: AmbientDef,
    #[serde(default)]
    pub sun: Option<SunDef>,
    #[serde(default)]
    pub environment: EnvDef,
    pub lod: LodDef,
    pub camera: CameraDef,
    #[serde(default)]
    pub walk: WalkDef,
    /// The world's base geometry (and water/vegetation meta ops),
    /// interpreted in order.
    pub generator: Vec<GenOpDef>,
    /// Material recipes for the ids the generator ops emit.
    #[serde(default)]
    pub materials: Vec<MaterialDef>,
    /// Prop populations on the heightfield surface (trees/grass/boulders).
    #[serde(default)]
    pub spawners: Vec<SpawnerDef>,
    /// Planning-op providers, composed in order.
    #[serde(default)]
    pub ops: Vec<OpsDef>,
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

/// A parameterized planning-op provider.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpsDef {
    /// Scattered ruin sites (per-256 m-cell probability).
    Ruins {
        #[serde(default = "default_ruin_chance")]
        chance: f32,
    },
    /// Roads connecting ruin sites (max connection distance, meters).
    Roads {
        #[serde(default = "default_ruin_chance")]
        site_chance: f32,
        #[serde(default = "default_road_reach")]
        reach: f32,
    },
    /// Megastructure habitation pockets (per-cell probability).
    Pockets {
        #[serde(default = "default_pocket_chance")]
        chance: f32,
    },
}

fn default_ruin_chance() -> f32 {
    0.32
}
fn default_road_reach() -> f32 {
    700.0
}
fn default_pocket_chance() -> f32 {
    0.45
}

impl LevelDef {
    /// The pockets chance, if a pockets provider is configured (used by
    /// walk-mode collision to mirror the GPU world).
    pub fn pocket_chance(&self) -> Option<f32> {
        self.ops.iter().find_map(|o| match o {
            OpsDef::Pockets { chance } => Some(*chance),
            _ => None,
        })
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
    level
        .sun
        .as_ref()
        .map(|s| Vec3::from(s.direction))
        .unwrap_or(Vec3::from(
            voxel_worldgen::program::DEFAULT_SUN_DIR.to_array(),
        ))
        .normalize()
}

/// `VOXEL_EVAL_HOLES=1`: coverage-eval rendering — magenta background,
/// monotone-white geometry, water off. A single background-colored pixel
/// below the horizon means missing world coverage.
pub fn eval_holes_mode() -> bool {
    std::env::var("VOXEL_EVAL_HOLES").is_ok()
}

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
    match level.grass() {
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
    pub source: Option<std::path::PathBuf>,
}

/// Watch state for level hot-reload.
#[derive(Resource)]
struct LevelSource {
    path: std::path::PathBuf,
    mtime: Option<std::time::SystemTime>,
    poll: Timer,
}

/// Marker for the level's sun light (replaced on reload).
#[derive(Component)]
struct LevelSun;

impl Plugin for LevelPlugin {
    fn build(&self, app: &mut App) {
        let level = self.def.clone();
        let c = level.clear_color;

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
        app.insert_resource(program)
            .insert_resource(material_table(&level))
            .insert_resource(env_params(&level))
            .insert_resource(if eval_holes_mode() {
                ClearColor(Color::srgb(1.0, 0.0, 1.0))
            } else {
                ClearColor(Color::srgb(c[0], c[1], c[2]))
            })
            .insert_resource(LodConfig {
                max_level: level.lod.max_level,
                top_radius: level.lod.top_radius,
                top_y: level.lod.top_y,
                split_k: level.lod.split_k,
                merge_k: level.lod.merge_k,
            })
            .insert_resource(build_ops_provider(&level))
            .insert_resource(water)
            .insert_resource(grass_style(&level))
            .insert_resource(level.clone())
            .add_plugins(VoxelEnginePlugin { vegetation: true })
            .add_plugins(voxel_render::WaterPlugin)
            .add_systems(Startup, setup_level)
            .add_systems(Update, (autopilot, walk_mode).chain());
    }
}

/// A boxed source of planning ops for a world-space box.
type OpsSource = Arc<dyn Fn(Vec3, Vec3) -> Vec<CsgOp> + Send + Sync>;

/// Compose the named planning providers into one op source.
fn build_ops_provider(level: &LevelDef) -> ChunkOpsProvider {
    let seed = level.seed;
    let mut sources: Vec<OpsSource> = Vec::new();
    for def in &level.ops {
        match *def {
            OpsDef::Ruins { chance } => sources.push(Arc::new(move |min, max| {
                voxel_worldgen::ruins::ruins_ops(seed, chance, min, max)
            })),
            OpsDef::Roads { site_chance, reach } => {
                let layers = Arc::new(voxel_worldgen::roads::planning_layers(
                    seed,
                    site_chance,
                    reach,
                ));
                sources.push(Arc::new(move |min, max| {
                    voxel_worldgen::roads::road_ops(&layers, min, max)
                }));
            }
            OpsDef::Pockets { chance } => sources.push(Arc::new(move |min, max| {
                voxel_worldgen::mega::pockets_ops(seed, chance, min, max)
            })),
        }
    }
    if sources.is_empty() {
        return ChunkOpsProvider(None);
    }
    ChunkOpsProvider(Some(Arc::new(move |key: ChunkKey| {
        if key.edge_m() > 130.0 {
            return Vec::new(); // meter-scale features: fine LODs only
        }
        // Pad by the density apron: samples extend 2 voxels below and 3
        // above the 32-cell core, so an op grazing only the apron still
        // shapes this chunk's samples — culling it desynchronizes the
        // seam with the neighbor that keeps it (visible slit through
        // structures straddling a chunk boundary).
        let pad = 4.0 * key.voxel_size_m() as f32;
        let min = key.min_corner_m().as_vec3() - Vec3::splat(pad);
        let max = key.min_corner_m().as_vec3() + Vec3::splat(key.edge_m() as f32 + pad);
        let mut ops = Vec::new();
        for source in &sources {
            ops.extend(source(min, max));
        }
        ops
    })))
}

fn setup_level(mut commands: Commands, level: Res<LevelDef>) {
    if let Some(sun) = &level.sun {
        let dir = Vec3::from(sun.direction).normalize();
        commands.spawn((
            LevelSun,
            DirectionalLight {
                illuminance: sun.illuminance,
                ..default()
            },
            Transform::from_translation(Vec3::ZERO).looking_to(-dir, Vec3::Y),
        ));
    }
    let a = &level.ambient;
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(a.color[0], a.color[1], a.color[2]),
        brightness: a.brightness,
        ..default()
    });

    // Camera, with env overrides for repeatable testing.
    let parse3 = |s: String| {
        let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
        (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
    };
    let start = std::env::var("VOXEL_START")
        .ok()
        .and_then(parse3)
        .unwrap_or(Vec3::from(level.camera.start));
    let look = std::env::var("VOXEL_LOOK")
        .ok()
        .and_then(parse3)
        .unwrap_or(Vec3::from(level.camera.look));
    // Up reference must not be parallel to the view direction.
    let up = if look.normalize_or_zero().dot(Vec3::Y).abs() > 0.9 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(start).looking_at(start + look * 1000.0, up),
        voxel_debug::FreeCamera {
            walk_speed: level.camera.walk_speed,
            run_speed: level.camera.run_speed,
            ..default()
        },
    ));
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
    mut clear: ResMut<ClearColor>,
    mut lod: ResMut<LodConfig>,
    mut rebuild: ResMut<StreamingRebuild>,
    mut water: ResMut<voxel_render::WaterSurface>,
    mut veg_rebuild: Option<ResMut<crate::vegetation::VegetationRebuild>>,
    mut cameras: Query<&mut voxel_debug::FreeCamera>,
    mut camera_transforms: Query<&mut Transform, With<Camera3d>>,
    mut windows: Query<&mut Window>,
    suns: Query<Entity, With<LevelSun>>,
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

    // Presentation: apply directly.
    let c = new.clear_color;
    clear.0 = if eval_holes_mode() {
        Color::srgb(1.0, 0.0, 1.0)
    } else {
        Color::srgb(c[0], c[1], c[2])
    };
    let a = &new.ambient;
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(a.color[0], a.color[1], a.color[2]),
        brightness: a.brightness,
        ..default()
    });
    for sun in &suns {
        commands.entity(sun).despawn();
    }
    if let Some(sun) = &new.sun {
        let dir = Vec3::from(sun.direction).normalize();
        commands.spawn((
            LevelSun,
            DirectionalLight {
                illuminance: sun.illuminance,
                ..default()
            },
            Transform::from_translation(Vec3::ZERO).looking_to(-dir, Vec3::Y),
        ));
    }
    for mut cam in &mut cameras {
        cam.walk_speed = new.camera.walk_speed;
        cam.run_speed = new.camera.run_speed;
    }
    lod.split_k = new.lod.split_k;
    lod.merge_k = new.lod.merge_k;
    if new.name != level.name {
        for mut window in &mut windows {
            window.title = format!("voxel2 — {}", new.name);
        }
    }
    if new.materials != level.materials {
        commands.insert_resource(material_table(&new));
    }
    if new.environment != level.environment {
        commands.insert_resource(env_params(&new));
    }
    info!("level reload: presentation applied");

    // Generation-affecting changes: rebuild the streamed world. Water and
    // vegetation are generator ops, so their toggles ride along; the sun
    // direction and seed live in the program header (baked shadows, hashes).
    let sun_changed = sun_dir(&new) != sun_dir(level.as_ref());
    let generator_changed =
        new.generator != level.generator || new.seed != level.seed || sun_changed;
    if generator_changed {
        let program = apply_generator(&new);
        *water = water_surface(&program);
        commands.insert_resource(program);
    }
    let regen = generator_changed
        || new.ops != level.ops
        || new.lod.max_level != level.lod.max_level
        || new.lod.top_radius != level.lod.top_radius
        || new.lod.top_y != level.lod.top_y;
    if regen {
        lod.max_level = new.lod.max_level;
        lod.top_radius = new.lod.top_radius;
        lod.top_y = new.lod.top_y;
        commands.insert_resource(build_ops_provider(&new));
        rebuild.0 = true;
        info!("level reload: generation changed — rebuilding world");
    }
    if generator_changed || new.spawners != level.spawners {
        commands.insert_resource(grass_style(&new));
        if let Some(veg) = veg_rebuild.as_mut() {
            veg.0 = true;
        }
    }
    // A different camera start means a different place (typically a whole
    // different level file): jump there.
    if new.camera.start != level.camera.start {
        let start = Vec3::from(new.camera.start);
        let look = Vec3::from(new.camera.look);
        let up = if look.normalize_or_zero().dot(Vec3::Y).abs() > 0.9 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        for mut t in &mut camera_transforms {
            *t = Transform::from_translation(start).looking_at(start + look * 1000.0, up);
        }
    }
    *level = new;
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
        assert!(matches!(planet.ops[0], OpsDef::Ruins { .. }));
        assert!(matches!(planet.ops[1], OpsDef::Roads { .. }));
        assert!(planet.sun.is_some());
        assert_eq!(planet.walk, WalkDef::Terrain);
        // Water is a generator op; vegetation is spawner data.
        let packed: Vec<_> = planet.generator.iter().map(GenOpDef::pack).collect();
        assert_eq!(voxel_worldgen::program::water_level(&packed), Some(0.0));
        assert!(planet.trees().is_some_and(|t| t.species.len() == 2));
        assert!(planet.grass().is_some() && planet.boulders().is_some());
        // Materials cover the ids the generator emits.
        assert!(planet.materials.iter().any(|m| m.id() == 1));
        assert!(planet.materials.iter().any(|m| m.id() == 3));

        let mega = LevelDef::from_json(&shipped("megastructure.json")).unwrap();
        assert!(matches!(mega.ops[0], OpsDef::Pockets { .. }));
        assert!(mega.sun.is_none());
        assert_eq!(mega.walk, WalkDef::Sdf);
        let packed: Vec<_> = mega.generator.iter().map(GenOpDef::pack).collect();
        assert_eq!(voxel_worldgen::program::water_level(&packed), None);
        assert!(mega.spawners.is_empty());
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
        assert_eq!(back.ops, planet.ops);
        assert_eq!(back.camera.start, planet.camera.start);
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

/// Flies the camera forward when `VOXEL_AUTOPILOT` is set (m/s).
fn autopilot(mut cameras: Query<&mut Transform, With<Camera3d>>, time: Res<Time>) {
    let Ok(speed) = std::env::var("VOXEL_AUTOPILOT") else {
        return;
    };
    let speed: f32 = speed.parse().unwrap_or(50.0);
    let level_flight = std::env::var("VOXEL_AUTOPILOT_LEVEL").is_ok();
    for mut transform in &mut cameras {
        let mut dir = *transform.forward();
        if level_flight {
            dir.y = 0.0;
            dir = dir.normalize_or_zero();
        }
        transform.translation += dir * speed * time.delta_secs();
    }
}

/// `VOXEL_WALK=1`: on-foot mode. `walk: terrain` glues the camera to the
/// generator's heightfield mirror; `walk: sdf` does gravity + capsule
/// collision against the full SDF mirror (including planned ops).
fn walk_mode(
    level: Res<LevelDef>,
    mut cameras: Query<&mut Transform, With<Camera3d>>,
    time: Res<Time>,
    mut fall_speed: Local<f32>,
    mut spawned: Local<bool>,
) {
    if std::env::var("VOXEL_WALK").is_err() {
        return;
    }
    match level.walk {
        WalkDef::Terrain => {
            for mut t in &mut cameras {
                let h = voxel_worldgen::terrain_height(
                    bevy::math::Vec2::new(t.translation.x, t.translation.z),
                    1.0,
                );
                t.translation.y = h + 1.75;
            }
        }
        WalkDef::Sdf => {
            use voxel_worldgen::mega::{mega_sdf, mega_sdf_with_ops, pockets_ops};
            const RADIUS: f32 = 0.5;
            const EYE: f32 = 1.6;
            let seed = level.seed;
            let floor_spacing =
                voxel_worldgen::program::lattice_y_spacing(&voxel_worldgen::program::program())
                    .unwrap_or(44.0);

            let pocket_chance = level.pocket_chance().unwrap_or(0.0);
            for mut t in &mut cameras {
                let local_ops = pockets_ops(
                    seed,
                    pocket_chance,
                    t.translation - Vec3::splat(30.0),
                    t.translation + Vec3::splat(30.0),
                );
                let sdf = |p: Vec3| mega_sdf_with_ops(p, &local_ops);
                let grad = |p: Vec3| {
                    let e = 0.1;
                    Vec3::new(
                        sdf(p + Vec3::X * e) - sdf(p - Vec3::X * e),
                        sdf(p + Vec3::Y * e) - sdf(p - Vec3::Y * e),
                        sdf(p + Vec3::Z * e) - sdf(p - Vec3::Z * e),
                    )
                    .normalize_or_zero()
                };
                // First tick: relocate onto solid floor (spawn cells can be
                // holes).
                if !*spawned {
                    *spawned = true;
                    'probe: for r in 0..40 {
                        for (dx, dz) in [(1.0, 0.3), (-0.7, 1.0), (0.4, -1.0), (-1.0, -0.5)] {
                            let p = t.translation
                                + Vec3::new(dx * r as f32 * 4.0, 0.0, dz * r as f32 * 4.0);
                            let floor = (p.y / floor_spacing).round() * floor_spacing;
                            let foot = Vec3::new(p.x, floor, p.z);
                            if mega_sdf(foot) < -1.0 {
                                t.translation = Vec3::new(foot.x, floor + 1.5 + EYE, foot.z);
                                break 'probe;
                            }
                        }
                    }
                }

                // Clamped dt + substeps so hitches can't tunnel through
                // 1.5 m floor slabs.
                let dt = time.delta_secs().min(0.033);
                *fall_speed = (*fall_speed - 22.0 * dt).max(-30.0);
                let mut body = t.translation - Vec3::Y * EYE;
                let mut remaining = *fall_speed * dt;
                while remaining.abs() > 0.0 {
                    let step = remaining.clamp(-0.4, 0.4);
                    body.y += step;
                    remaining -= step;
                    for _ in 0..4 {
                        let d = sdf(body);
                        if d < RADIUS {
                            let n = grad(body);
                            body += n * (RADIUS - d);
                            if n.y > 0.5 {
                                *fall_speed = 0.0;
                                remaining = 0.0;
                            }
                        }
                    }
                }
                for _ in 0..4 {
                    let d = sdf(body);
                    if d < RADIUS {
                        body += grad(body) * (RADIUS - d);
                    }
                }
                t.translation = body + Vec3::Y * EYE;
            }
        }
    }
}
