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

use crate::graph::{node::Invalidates, NodeDef};
use crate::planning::{HostPlanning, OpsSource, WorldQuery};
use crate::schema;
use crate::streaming::StreamingRebuild;
use crate::{LodConfig, VoxelEnginePlugin};

#[derive(Reflect, Serialize, Deserialize, Clone, Debug)]
pub struct LodDef {
    /// How many times the voxel size doubles above the finest level.
    ///
    /// No [`schema::Range`]: what this can be is a fact about the ENGINE,
    /// and a bound invented for the slider's benefit is a bound that can
    /// exclude a value the level already ships. One did — 12, against a
    /// planet that has always been 14.
    #[reflect(@schema::Rebuilds)]
    pub max_level: u8,
    /// Chunks of the top level kept around the camera.
    #[reflect(@schema::Rebuilds)]
    pub top_radius: i32,
    /// Vertical extent of the top level, in chunks.
    #[reflect(@schema::Rebuilds)]
    pub top_y: (i32, i32),
    /// Refinement thresholds — tuning, not topology, so they apply
    /// without restreaming.
    pub split_k: f64,
    pub merge_k: f64,
}

impl From<&LodDef> for LodConfig {
    fn from(d: &LodDef) -> Self {
        Self {
            max_level: d.max_level,
            top_radius: d.top_radius,
            top_y: d.top_y,
            split_k: d.split_k,
            merge_k: d.merge_k,
        }
    }
}

/// Lighting + atmosphere for the chunk draw. Every field has the sun-lit
/// outdoor default, so levels only state what differs.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct EnvDef {
    /// Direction the sun comes FROM (not normalized; twins normalize).
    /// The only lighting value the engine owns: the mesh shader bakes
    /// horizon shadows along it, so it must match the app's sun. Colors,
    /// strengths, ambient and haze are the app's — voxel surfaces shade
    /// through Bevy's PBR, so they come from its lights and `DistanceFog`.
    ///
    /// Restreams: the shadows are baked into the vertices, so moving the
    /// sun re-meshes rather than re-lights.
    #[serde(default = "d_sun_direction")]
    #[reflect(@schema::Rebuilds)]
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
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
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
///
/// A host wraps this in a node of its own — the gate names something in
/// the HOST's vocabulary, so the node that carries it is the host's — and
/// three of these fields are that node's WIRING rather than a level's
/// text: `class` is the node's name, `gate` is its gate wire plus
/// [`ScatterDef::region`], and the slot behind `density` is the field node
/// its density wire names. Each was a number or a string a level used to
/// write twice and keep agreeing by hand.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Serialize, Deserialize, Default)]
pub struct ScatterDef {
    /// Host-facing name for this population: the node's own name. Any
    /// string — the engine never interprets it, it only tags the
    /// placements so the host can decide what a member of this population
    /// is.
    #[serde(skip)]
    #[reflect(@schema::Hidden)]
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
    #[reflect(@schema::Range(0.0, 1.0))]
    pub chance: f32,
    /// Altitude band the class lives in.
    pub altitude: [f32; 2],
    /// Minimum surface up-ness (1 = flat).
    #[serde(default)]
    #[reflect(@schema::Range(0.0, 1.0))]
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
    /// Which member of the wired weight source gates this population.
    ///
    /// The wire says WHICH source; this says which of its members. The
    /// engine never interprets the name — what a member means is the
    /// host's business (this demo's are biomes; another game's could be
    /// factions or pollution) — it only asks the host for the weight.
    #[serde(default)]
    pub region: Option<String>,
    /// The two halves above, joined: the `"instance:member"` reference the
    /// host resolves. Written by the compiler, never by a level.
    #[serde(skip)]
    #[reflect(@schema::Hidden)]
    pub gate: Option<String>,
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
    /// What this population becomes once it is too far to draw one by one.
    #[serde(default)]
    pub cover: Option<CoverDef>,
}

/// A population painted onto the ground instead of drawn.
///
/// The third thing a scattered population can be, after an entity and a
/// point: past some distance an instance is smaller than a pixel, and a
/// million of them are a colour the ground is. Same trade as a road, which
/// stops being a carve and becomes a material — see the host's surface
/// map. The engine only carries the declaration and
/// [`crate::scatter::coverage`]; where the ground's material comes from is
/// the host's business.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CoverDef {
    /// Material id the covered ground takes.
    #[reflect(@schema::OneOf("materials[].id"))]
    pub material: u32,
    /// Distance the paint takes over at — where the host stopped drawing
    /// the instances. Both numbers are the level's, so the handover is
    /// authored as one decision rather than two that must be kept equal.
    pub from_m: f32,
    /// Coverage at which the ground is solidly this material; below it the
    /// paint thins. A population's edge is a thinning of instances, not a
    /// contour line, and paint that ends on one reads as a painted shape.
    #[serde(default = "d_cover_full_at")]
    pub full_at: f32,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
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
fn d_cover_full_at() -> f32 {
    1.0
}
fn d_any_altitude() -> [f32; 2] {
    [f32::MIN, f32::MAX]
}
fn d_unit_scale() -> [f32; 2] {
    [1.0, 1.0]
}

/// One authored CSG primitive in a prefab's local space.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
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
    #[reflect(@schema::OneOf("materials[].id"))]
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
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
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
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct FieldDensityDef {
    /// The slot the wired `field` node was allocated. The compiler writes
    /// it; a level names the node instead, which is what stopped a slot
    /// number being written once in the generator and once here.
    #[serde(skip)]
    #[reflect(@schema::Hidden)]
    pub field: u32,
    #[serde(default = "default_one")]
    pub scale: f32,
    #[serde(default)]
    pub offset: f32,
}

/// Orientation + banding rules shared by prop spawners (VoxelPlugin's
/// BasicSpawner placement block as data).
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
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
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PatchDef {
    pub scale: f32,
    #[serde(default)]
    pub offset: [f32; 2],
    pub contrast: f32,
    pub bias: f32,
}

/// One material recipe, referenced by the material ids generator ops emit.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MaterialDef {
    /// Uniform base with grain, optional bands/grime/streaks/moss/emissive.
    Surface {
        id: u32,
        #[reflect(@schema::AsColor)]
        base: [f32; 3],
        #[serde(default = "default_grain")]
        #[reflect(@schema::Range(0.0, 1.0))]
        grain: f32,
        #[serde(default)]
        band: Option<BandDef>,
        #[serde(default)]
        grime: Option<GrimeDef>,
        #[serde(default)]
        #[reflect(@schema::Range(0.0, 1.0))]
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
        #[reflect(@schema::AsColor)]
        low: [f32; 3],
        /// (dark, sun-lit) canopy greens mixed by crown noise.
        #[reflect(@schema::AsColor)]
        canopy: [[f32; 3]; 2],
        #[reflect(@schema::AsColor)]
        rock: [f32; 3],
        /// Dry/brown patch color on gentle ground.
        #[reflect(@schema::AsColor)]
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
        #[reflect(@schema::Range(0.0, 1.0))]
        patch_amount: f32,
        #[serde(default = "default_zoned_fade")]
        detail_fade: f32,
    },
    /// Altitude-zoned natural terrain (low/mid/high/peak with noisy
    /// borders and a slope override to the high color).
    Zoned {
        id: u32,
        #[reflect(@schema::AsColor)]
        low: [f32; 3],
        /// Two hues mixed by large-scale noise.
        #[reflect(@schema::AsColor)]
        mid: [[f32; 3]; 2],
        /// Two hues banded by altitude.
        #[reflect(@schema::AsColor)]
        high: [[f32; 3]; 2],
        #[reflect(@schema::AsColor)]
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

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BandDef {
    pub freq: f32,
    pub amp: f32,
    pub lo: f32,
    pub hi: f32,
    #[serde(default)]
    pub warp: f32,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GrimeDef {
    #[reflect(@schema::AsColor)]
    pub tint: [f32; 3],
    #[reflect(@schema::Range(0.0, 1.0))]
    pub amount: f32,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MossDef {
    #[reflect(@schema::AsColor)]
    pub color: [f32; 3],
    #[reflect(@schema::Range(0.0, 1.0))]
    pub amount: f32,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EmissiveDef {
    #[reflect(@schema::AsColor)]
    pub color: [f32; 3],
    #[serde(default = "default_one")]
    pub intensity: f32,
    /// Strip spacing along z / vertical level spacing (meters).
    pub spacing: f32,
    pub level_spacing: f32,
    /// Chance a strip is lit.
    #[reflect(@schema::Range(0.0, 1.0))]
    pub chance: f32,
    /// Up-glow intensity on floors below.
    #[serde(default)]
    pub glow: f32,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CanopyZonesDef {
    /// (start altitude, blend width) where canopy replaces the low color.
    pub canopy: [f32; 2],
    /// (start altitude, blend width) where rock replaces canopy.
    pub rock: [f32; 2],
    #[serde(default = "default_border")]
    pub border: f32,
}

#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
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
    ///
    /// By NAME, through `voxel_core::layout::MATERIALS`, which is also
    /// what generates the shader's accessors. The slot a parameter lives
    /// in used to be written here as a `vec4` position and read there as
    /// `m.p2.y`, agreed by a comment; adding a field meant finding a free
    /// component by counting. Now neither side can move one alone.
    pub fn pack(&self) -> voxel_render::WorldMaterial {
        use voxel_core::layout::MatPack;
        let mut m;
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
                m = MatPack::new("surface");
                m.rgb_w("base", base, "grain", grain)
                    .set("streaks", streaks)
                    .set("detail_fade", detail_fade);
                if let Some(g) = grime {
                    m.rgb_w("grime_tint", g.tint, "grime_amount", g.amount);
                }
                if let Some(moss) = moss {
                    m.rgb_w("moss_color", moss.color, "moss_amount", moss.amount);
                }
                if let Some(b) = band {
                    m.set("band_freq", b.freq)
                        .set("band_amp", b.amp)
                        .set("band_lo", b.lo)
                        .set("band_hi", b.hi)
                        .set("band_warp", b.warp);
                }
                // The strip spacings default to 1 rather than 0 even with
                // no emissive block: the shader divides by them.
                m.set(
                    "strip_spacing",
                    emissive.as_ref().map_or(1.0, |e| e.spacing),
                )
                .set(
                    "strip_level_spacing",
                    emissive.as_ref().map_or(1.0, |e| e.level_spacing),
                );
                if let Some(e) = emissive {
                    m.rgb_w("emissive_color", e.color, "emissive_intensity", e.intensity)
                        .set("strip_chance", e.chance)
                        .set("strip_glow", e.glow);
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
            } => {
                m = MatPack::new("canopy");
                m.rgb_w("canopy_a", canopy[0], "canopy_start", zones.canopy[0])
                    .rgb_w("canopy_b", canopy[1], "rock_start", zones.rock[0])
                    .rgb_w("rock", rock, "rock_width", zones.rock[1])
                    .rgb_w("patch", patch, "border", zones.border)
                    .rgb_w("low", low, "canopy_width", zones.canopy[1])
                    .set("crown_scale", crowns[0])
                    .set("crown_relief", crowns[1])
                    .set("strata_scale", strata[0])
                    .set("strata_relief", strata[1])
                    .set("steep_hi", steep[0])
                    .set("steep_lo", steep[1])
                    .set("detail_fade", detail_fade)
                    .set("patch_amount", patch_amount);
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
            } => {
                m = MatPack::new("zoned");
                m.rgb_w("low", low, "mid_start", zones.mid[0])
                    .rgb_w("mid_a", mid[0], "high_start", zones.high[0])
                    .rgb_w("high_a", high[0], "peak_start", zones.peak[0])
                    .rgb_w("peak", peak, "border", zones.border)
                    .rgb_w("mid_b", mid[1], "mid_width", zones.mid[1])
                    .rgb_w("high_b", high[1], "high_width", zones.high[1])
                    .set("peak_width", zones.peak[1])
                    .set("steep_hi", steep[0])
                    .set("steep_lo", steep[1])
                    .set("detail_fade", detail_fade);
            }
        }
        voxel_render::WorldMaterial::from_packed(m.finish())
    }
}

/// A complete level description.
///
/// Reflected as well as serialized, and those are not the same job. Serde
/// is how a level is READ; reflection is how a running one is REACHED —
/// `world.mutate_resources` addresses a field by path, so a tool can set
/// `materials[7].base` in a live session and the change goes through
/// [`apply_level_change`] exactly as a file edit would. Tuning a colour
/// was a ninety-second relaunch before that.
#[derive(Reflect, Resource, Serialize, Deserialize, Clone, Debug)]
#[reflect(Resource)]
pub struct LevelDef {
    #[serde(default)]
    pub environment: EnvDef,
    pub lod: LodDef,
    /// The world's graph: every node, in an order that is also a valid
    /// topological order. See [`crate::graph`].
    #[reflect(@schema::Rebuilds)]
    pub nodes: Vec<NodeDef>,
    /// Material recipes for the ids the generator ops emit.
    ///
    /// No [`schema::Rebuilds`]: a material is a table upload, which is why
    /// tuning a colour is instant.
    #[serde(default)]
    pub materials: Vec<MaterialDef>,
    /// Named prefabs: reusable local-space CSG op groups for placements.
    #[serde(default)]
    #[reflect(@schema::Rebuilds)]
    pub prefabs: std::collections::HashMap<String, Vec<CsgOpDef>>,
    /// Hand-authored prefab instances in the world.
    #[serde(default)]
    #[reflect(@schema::Rebuilds)]
    pub placements: Vec<PlacementDef>,
}

/// When a generator op applies across the LOD range.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
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
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
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

/// Octave shaping for `height_fbm`.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum NoiseModeDef {
    #[default]
    Fbm,
    Ridged,
    Billow,
}

pub(crate) fn d_full_band() -> [f32; 2] {
    [0.0, 1.0]
}
pub(crate) fn d_band_octaves() -> u32 {
    2
}

pub(crate) fn mat_grass() -> u32 {
    1
}
pub(crate) fn mat_concrete() -> u32 {
    2
}

pub(crate) fn gate_flags(lod: LodGateDef, auto: u32) -> u32 {
    match lod {
        LodGateDef::Auto => auto,
        LodGateDef::All => 0,
        LodGateDef::Fine => WOP_FLAG_FINE_ONLY,
        LodGateDef::Coarse => WOP_FLAG_COARSE_ONLY,
    }
}

impl LevelDef {
    /// Read a level, resolving each node's `"kind"` through `registry`.
    ///
    /// The registry is needed because the node set is OPEN: a kind names a
    /// registered type, and which types exist is decided by whatever the
    /// host registered. Hosts pass the app's registry; tests and tools that
    /// only need the engine's own kinds can use
    /// [`crate::graph::registry::engine_kinds`].
    pub fn from_json(
        json: &str,
        registry: &bevy::reflect::TypeRegistryArc,
    ) -> Result<Self, serde_json::Error> {
        crate::graph::with_registry(registry, || serde_json::from_str(json))
    }

    /// Read a level, dropping every node whose kind `registry` does not
    /// know.
    ///
    /// A level names kinds from two vocabularies — the engine's and the
    /// GAME's — so a crate that can see only one of them is not looking at
    /// a malformed file, it is looking at the half addressed to it. That
    /// is the position of a tool, and of every test in a crate below the
    /// host. Anything that RUNS a level uses [`LevelDef::from_json`],
    /// which rejects an unknown kind, so a typo cannot pass itself off as
    /// a node somebody else implements.
    pub fn from_json_known(
        json: &str,
        registry: &bevy::reflect::TypeRegistryArc,
    ) -> Result<Self, serde_json::Error> {
        let known: Vec<String> = {
            let reg = registry.read();
            crate::graph::registry::kinds(&reg)
                .into_iter()
                .map(|(kind, _)| kind.to_string())
                .collect()
        };
        let mut doc: serde_json::Value = serde_json::from_str(json)?;
        if let Some(nodes) = doc.get_mut("nodes").and_then(|n| n.as_array_mut()) {
            nodes.retain(|n| known.iter().any(|k| n["kind"] == k.as_str()));
        }
        Self::from_json(&doc.to_string(), registry)
    }

    /// Write a level back, with each node's params serialized by its own
    /// type. The inverse of [`LevelDef::from_json`], and it needs the
    /// registry for the same reason.
    pub fn to_json(
        &self,
        registry: &bevy::reflect::TypeRegistryArc,
    ) -> Result<String, serde_json::Error> {
        crate::graph::with_registry(registry, || serde_json::to_string_pretty(self))
    }

    /// This level's generator at `seed`. Layers need one to sample the
    /// world they are planning on top of; at runtime the engine hands the
    /// same value to [`crate::planning::HostPlanning::build`].
    pub fn generator(&self, seed: u64) -> voxel_worldgen::Generator {
        let program =
            crate::graph::compile(&self.nodes).unwrap_or_else(|e| panic!("level graph: {e}"));
        assert_region_axes_first(&program.ops);
        voxel_worldgen::Generator::new(program.ops, seed as u32, sun_dir(self))
    }
}

/// Every reader of the region axes must come after the op that fills
/// them, and this is a hard error rather than a quiet zero.
///
/// The GPU splits the program in two — a column pass that fills `ta`/`tb`
/// and a sample pass that consumes them — so on the GPU a band op reads
/// the FINAL axes wherever it sits in the list, while the CPU twin runs
/// the ops in order and would read zero. An out-of-order program is
/// therefore not merely wrong, it is wrong DIFFERENTLY in each
/// interpreter, which is the expensive kind of bug to find.
fn assert_region_axes_first(ops: &[WorldOp]) {
    let axes_at = ops.iter().position(|op| op.kind == WOP_REGION_AXES);
    let reader_at = ops.iter().position(|op| {
        op.region != 0 || matches!(op.kind, WOP_MATERIAL_BAND | WOP_HEIGHT_BAND_FBM)
    });
    let Some(reader) = reader_at else { return };
    assert!(
        axes_at.is_some_and(|axes| axes < reader),
        "generator op {reader} reads the region axes but no `region_axes` op precedes it \
         (add one, first in the list)"
    );
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
pub(crate) fn build_generator(
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
    /// The mesh slab's budget. Chunk cost is a property of the LEVEL —
    /// terrain is about a page per chunk, dense interior geometry
    /// several — so a host whose worlds are unlike this demo's says so
    /// here instead of the engine guessing on its behalf.
    pub slab: voxel_render::slab::SlabConfig,
}

impl LevelPlugin {
    /// Present `def` with default options.
    pub fn new(def: LevelDef) -> Self {
        Self {
            def,
            seed: 0,
            source: None,
            slab: Default::default(),
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

/// The definition the running world was last built from.
///
/// [`LevelDef`] is what the level SAYS; this is what the world was made
/// of. They differ for exactly one frame, between something writing the
/// resource and [`apply_level_change`] catching up — which is the window
/// the diff lives in. Keeping it separate is what lets any writer at all
/// drive a partial rebuild, rather than only the file watcher that used to
/// own both halves.
#[derive(Resource, Clone)]
struct AppliedLevel(LevelDef);

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
            .add_systems(Update, watch_level_file);
        }

        if let Some(host) = &self.planner {
            if let Err(e) = host.validate(&level) {
                panic!("level has invalid planning data: {e}");
            }
        }
        app.insert_resource(env_params(&level))
            .insert_resource(AppliedLevel(level.clone()))
            .insert_resource(level.clone())
            .register_type::<LevelDef>()
            // Applies whatever wrote the resource, watcher or not, so it
            // runs whether or not this level came from a file.
            .add_systems(Update, apply_level_change)
            .add_plugins(VoxelEnginePlugin { slab: self.slab })
            // World 0 is loaded through the SAME path as any other world.
            // It used to be assembled here, by hand, out of five separate
            // resources — which is why a host adding a second world had to
            // rediscover which five, and got three of them.
            .add_systems(Startup, load_initial_world)
            .add_systems(
                Update,
                crate::planning::follow_stream_source.in_set(crate::WorldFocusSet::Follow),
            );

        if let Some(port) = self.remote_port {
            let _ = port; // BRP tooling lives in voxel-debug; see VoxelRemotePlugin
        }
    }
}

/// Register the level this plugin loaded as world 0.
///
/// In `Startup` rather than at plugin build so it goes through
/// [`crate::WorldLoader`] like every other world. A second world (a
/// portal's far side) arrives long after startup, so the path that loads
/// one has to work at any time — and the only way to be sure it does is
/// for the first world to use it too.
fn load_initial_world(mut loader: crate::WorldLoader, level: Res<LevelDef>, seed: Res<WorldSeed>) {
    let config = LodConfig::from(&level.lod);
    let id = loader.load(level.clone(), seed.0, config);
    assert_eq!(id, 0, "the level plugin's own level must be world 0");
}

pub(crate) fn build_world_query(
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
/// Poll the watched file and publish what it says into [`LevelDef`].
///
/// Writing the resource is the whole job: applying the change is
/// [`apply_level_change`], which watches the resource rather than the
/// file. A level edited by any other route — a remote tool poking a value
/// through reflection, a host's own UI — takes exactly the same path, and
/// gets exactly the same partial rebuild, without knowing this file exists.
fn watch_level_file(
    time: Res<Time>,
    mut source: ResMut<LevelSource>,
    mut level: ResMut<LevelDef>,
    planner: Res<HostPlanner>,
    registry: Res<AppTypeRegistry>,
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
        .and_then(|json| LevelDef::from_json(&json, &registry.0).map_err(|e| e.to_string()))
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
    *level = new;
}

/// What this edit makes stale.
///
/// Everything a world is built from at `build_world_query` time belongs
/// here, and the cost of missing one is silent: the level says one thing,
/// the running planner another, and the edit appears to do nothing at all.
/// `scatter` was missing exactly that way. Named as a function so the list
/// can be tested against the schema rather than read carefully — which is
/// also how [`schema::Rebuilds`] is kept honest.
///
/// The generator and the sun the shadows bake along belong here as much as
/// the planning graph does. They used to be tested at the call site, which
/// gave "does this edit restream" two answers in two places and one of
/// them to an editor that has to ask.
///
/// The node list answers for itself, per node — see [`crate::graph::changed`].
fn staleness(new: &LevelDef, old: &LevelDef) -> Option<Invalidates> {
    let world = sun_dir(new) != sun_dir(old)
        || new.placements != old.placements
        || new.prefabs != old.prefabs
        || new.lod.max_level != old.lod.max_level
        || new.lod.top_radius != old.lod.top_radius
        || new.lod.top_y != old.lod.top_y;
    let nodes = crate::graph::changed(&new.nodes, &old.nodes);
    match (world, nodes) {
        (true, _) => Some(Invalidates::World),
        (false, effect) => effect,
    }
}

/// Apply whatever changed in [`LevelDef`] to the running world.
///
/// Driven by change detection, not by the file watcher, so every writer of
/// the resource is served. `applied` is the copy last acted on: the diff
/// has to be against what the world was BUILT from, and the resource is
/// already the new value by the time this runs.
fn apply_level_change(
    level: Res<LevelDef>,
    mut applied: ResMut<AppliedLevel>,
    seed: Res<WorldSeed>,
    planner: Res<HostPlanner>,
    mut rebuild: ResMut<StreamingRebuild>,
    mut reloaded: MessageWriter<LevelReloaded>,
    // Grouped: the two registries are always touched together, and
    // clippy caps a system's arguments at seven.
    (mut worlds, mut render): (ResMut<crate::Worlds>, ResMut<voxel_render::RenderWorlds>),
) {
    if !level.is_changed() {
        return;
    }
    let new = level.clone();
    let level = &applied.0;
    // Only the world this plugin loaded reloads. A portal's far side was
    // loaded from its own file and is nobody's business here.
    let (Some(world), Some(render)) = (worlds.get_mut(0), render.get_mut(0)) else {
        return;
    };

    if new.materials != level.materials {
        render.materials =
            voxel_render::material_table(new.materials.iter().map(|m| (m.id(), m.pack())));
    }
    world.config.split_k = new.lod.split_k;
    world.config.merge_k = new.lod.merge_k;

    // Generation-affecting changes rebuild the streamed world.
    let sun_changed = sun_dir(&new) != sun_dir(level);
    let generator_changed = new.nodes != level.nodes || sun_changed;
    // Rebuilt whether or not the program changed: the planning graph and
    // the facade below need one either way.
    let (program, generator) = build_generator(&new, seed.0);
    if generator_changed {
        render.program = program;
    }
    // A planning-only edit still replaces the planner — that is where the
    // populations are registered — but leaves the streamed chunks alone.
    // They were carved by ops this edit cannot have changed, so tearing
    // them down would regenerate every one of them into itself.
    if let Some(stale) = staleness(&new, level) {
        if stale == Invalidates::World {
            world.config.max_level = new.lod.max_level;
            world.config.top_radius = new.lod.top_radius;
            world.config.top_y = new.lod.top_y;
            world.generator = generator.clone();
            rebuild.0 = true;
        }
        world.query = build_world_query(&new, seed.0, &generator, planner.0.as_ref());
        world.level = new.clone();
        info!("level reload: {stale:?} is stale — rebuilding it");
    }

    // The host owns the scene: it reads the new definition off this
    // message and applies its own camera, lights and clear color.
    let previous = std::mem::replace(&mut applied.0, new);
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
    use bevy::reflect::TypeInfo;

    /// Does this edit restream the world? What [`schema::Rebuilds`]
    /// promises, and the question every case below asks.
    fn needs_regen(new: &LevelDef, old: &LevelDef) -> bool {
        staleness(new, old) == Some(Invalidates::World)
    }

    fn shipped(name: &str) -> String {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../levels/");
        std::fs::read_to_string(format!("{path}{name}")).unwrap()
    }

    /// A recipe lands in the exact components the shader reads.
    ///
    /// Written out by hand from `planet.json` rather than compared against
    /// `pack()`'s own output, which would only prove it agrees with
    /// itself. This is the assertion that moving the packer onto
    /// `voxel_core::layout` changed no bytes, and it stays as the pin on
    /// the canopy layout: every number here is visible in the level file.
    #[test]
    fn a_recipe_packs_into_the_components_the_shader_reads() {
        let planet = LevelDef::from_json_known(
            &shipped("planet.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();
        let canopy = planet
            .materials
            .iter()
            .find(|m| matches!(m, MaterialDef::Canopy { .. }))
            .expect("planet ships a canopy material");
        let m = canopy.pack();
        let v = |x: f32, y: f32, z: f32, w: f32| bevy::math::Vec4::new(x, y, z, w);
        assert_eq!(m.head.x, voxel_render::MAT_KIND_CANOPY);
        // canopy[0] | zones.canopy[0]
        assert_eq!(m.c0, v(0.014, 0.0261, 0.0084, 1.5));
        // canopy[1] | zones.rock[0]
        assert_eq!(m.c1, v(0.0747, 0.0933, 0.0261, 340.0));
        // rock | zones.rock[1]
        assert_eq!(m.c2, v(0.0635, 0.0541, 0.0467, 140.0));
        // patch | zones.border
        assert_eq!(m.c3, v(0.0597, 0.0448, 0.0205, 60.0));
        // low | zones.canopy[1]
        assert_eq!(m.p0, v(0.112, 0.1027, 0.0784, 3.0));
        // crowns | strata
        assert_eq!(m.p1, v(0.35, 0.9, 0.15, 1.2));
        // steep | detail_fade | patch_amount
        assert_eq!(m.p2, v(0.72, 0.45, 0.0015, 0.55));
    }

    /// Every section the world is BUILT from has to force a rebuild.
    ///
    /// Asserted per section rather than as one list, because the failure
    /// this catches is silent: `scatter` was absent for as long as it took
    /// somebody to edit a population, watch nothing happen, and relaunch
    /// instead of asking why.
    #[test]
    fn an_edit_to_anything_the_world_is_built_from_rebuilds_it() {
        let planet = LevelDef::from_json_known(
            &shipped("planet.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();
        assert!(
            !needs_regen(&planet, &planet),
            "an unchanged level must not rebuild — a repaint would loop"
        );

        let mut placements = planet.clone();
        placements.placements.clear();
        assert!(needs_regen(&placements, &planet), "placements");

        let mut prefabs = planet.clone();
        prefabs.prefabs.clear();
        assert!(needs_regen(&prefabs, &planet), "prefabs");

        let mut lod = planet.clone();
        lod.lod.max_level -= 1;
        assert!(needs_regen(&lod, &planet), "lod.max_level");

        // And the cheap ones must NOT: a material is a table upload, and
        // rebuilding the streamed world to recolour it would make tuning
        // a colour cost what it used to.
        let mut material = planet.clone();
        material.materials[0] = material.materials[0].clone();
        assert!(!needs_regen(&material, &planet), "materials");
        let mut split = planet.clone();
        split.lod.split_k += 1.0;
        assert!(!needs_regen(&split, &planet), "lod.split_k");
    }

    /// Does a field carry `attr`, addressed the way a level is addressed?
    ///
    /// Dotted, through nested structs, because the attribute belongs on
    /// the leaf that decides: `lod.max_level` restreams and `lod.split_k`
    /// does not, and one attribute on `lod` could only be wrong about one
    /// of them.
    #[cfg(test)]
    fn field_has<A: bevy::reflect::Reflect>(root: &'static TypeInfo, path: &str) -> bool {
        let mut here = root;
        let mut steps = path.split('.').peekable();
        while let Some(step) = steps.next() {
            let TypeInfo::Struct(info) = here else {
                panic!("'{path}': '{step}' is not reached through a struct");
            };
            let field = info
                .field(step)
                .unwrap_or_else(|| panic!("'{path}': no reflected field '{step}'"));
            if steps.peek().is_none() {
                return field.has_attribute::<A>();
            }
            here = field.type_info().expect("a nested struct is Typed");
        }
        unreachable!("a path has at least one step")
    }

    /// The rationale written above each field has to survive to runtime.
    ///
    /// `reflect_documentation` is not a bevy default feature, so dropping
    /// it from the workspace `bevy` line compiles fine and silently
    /// unlabels every row of an editor. This is that line's only alarm.
    #[test]
    fn a_fields_documentation_reaches_the_running_program() {
        use bevy::reflect::Typed;

        let TypeInfo::Struct(info) = LevelDef::type_info() else {
            panic!("LevelDef is a struct")
        };
        let docs = info
            .field("nodes")
            .expect("nodes is reflected")
            .docs()
            .expect("reflect_documentation must be on — see Cargo.toml");
        assert!(
            docs.contains("graph"),
            "documentation reached reflection but not this field's: {docs:?}"
        );
    }

    /// [`schema::Rebuilds`] must say exactly what `needs_regen` does.
    ///
    /// Both directions of drift are silent and both are bad: a missing
    /// attribute makes an editor restream the world once per frame of a
    /// slider drag, and a spurious one makes an edit apply only on release
    /// for no reason. So this compares no lists — it EDITS each field and
    /// asks the real function.
    ///
    /// The coverage assertion is the load-bearing half: a section added to
    /// `LevelDef` fails here until somebody decides whether it rebuilds,
    /// which is the question `scatter` went unasked.
    #[test]
    fn the_rebuilds_attribute_says_what_needs_regen_does() {
        use bevy::reflect::Typed;

        let planet = LevelDef::from_json_known(
            &shipped("planet.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();
        let root = LevelDef::type_info();

        /// A field, an edit big enough for `PartialEq` to see, and whether
        /// making it restreams the world.
        type Case = (&'static str, fn(&mut LevelDef), bool);

        let cases: &[Case] = &[
            (
                "environment.sun_direction",
                |l| l.environment.sun_direction[0] += 1.0,
                true,
            ),
            ("lod.max_level", |l| l.lod.max_level -= 1, true),
            ("lod.top_radius", |l| l.lod.top_radius += 1, true),
            ("lod.top_y", |l| l.lod.top_y.1 += 1, true),
            ("lod.split_k", |l| l.lod.split_k += 1.0, false),
            ("lod.merge_k", |l| l.lod.merge_k += 1.0, false),
            // A node that SHAPES the world, chosen deliberately: what an
            // edit to `nodes` costs is per node now, and popping whatever
            // happens to be last would make this case's answer content.
            (
                "nodes",
                |l| {
                    let at = l
                        .nodes
                        .iter()
                        .position(|n| n.node.0.invalidates() == Invalidates::World)
                        .expect("planet is mostly program");
                    l.nodes.remove(at);
                },
                true,
            ),
            (
                "materials",
                |l| {
                    l.materials.pop();
                },
                false,
            ),
            ("prefabs", |l| l.prefabs.clear(), true),
            ("placements", |l| l.placements.clear(), true),
        ];

        for (path, edit, restreams) in cases {
            let mut edited = planet.clone();
            edit(&mut edited);
            assert_ne!(
                format!("{edited:?}"),
                format!("{planet:?}"),
                "the edit for '{path}' changed nothing — every assertion \
                 below it would pass for the wrong reason"
            );
            assert_eq!(
                needs_regen(&edited, &planet),
                *restreams,
                "needs_regen disagrees with this test about '{path}'"
            );
            assert_eq!(
                field_has::<schema::Rebuilds>(root, path),
                *restreams,
                "schema::Rebuilds disagrees with needs_regen about '{path}' \
                 — an editor would apply it at the wrong time"
            );
        }

        // Coverage: every section of a level is decided about above.
        let TypeInfo::Struct(info) = root else {
            panic!("LevelDef is a struct")
        };
        let sections: Vec<&str> = info.iter().map(|f| f.name()).collect();
        for section in sections {
            assert!(
                cases
                    .iter()
                    .any(|(path, ..)| path.split('.').next().is_some_and(|top| top == section)),
                "LevelDef gained '{section}' and nothing here says whether \
                 editing it restreams the world"
            );
        }
    }

    /// A tool has to be able to reach a level's values by field path —
    /// that is what `world.mutate_resources` does over BRP, and it is the
    /// difference between tuning a number and relaunching to see it.
    #[test]
    fn a_level_can_be_reached_by_field_path() {
        use bevy::reflect::GetPath;
        let mut planet = LevelDef::from_json_known(
            &shipped("planet.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();

        // The path BRP would address, on the real shipped schema. Found by
        // VARIANT, not by index: a path only reaches the fields the
        // material actually has, and which recipe sits at which index is
        // content.
        let at = planet
            .materials
            .iter()
            .position(|m| matches!(m, MaterialDef::Surface { .. }))
            .expect("planet ships a surface material");
        let path = format!(".materials[{at}].base[0]");
        let was = *planet
            .path::<f32>(path.as_str())
            .expect("a material colour must be reachable by path");
        *planet.path_mut::<f32>(path.as_str()).unwrap() = was + 0.25;
        assert_eq!(
            *planet.path::<f32>(path.as_str()).unwrap(),
            was + 0.25,
            "a path write must land on the level, not on a copy of it"
        );

        // Registered WITH the resource data, or the remote methods cannot
        // find it however reflectable it is.
        let mut registry = bevy::reflect::TypeRegistry::new();
        registry.register::<LevelDef>();
        assert!(
            registry
                .get_type_data::<bevy::ecs::reflect::ReflectResource>(std::any::TypeId::of::<
                    LevelDef,
                >())
                .is_some(),
            "LevelDef needs #[reflect(Resource)] to be reachable remotely"
        );
    }

    #[test]
    fn shipped_levels_parse() {
        let planet = LevelDef::from_json_known(
            &shipped("planet.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();
        // The planet's geometry comes from height ops; sea level is the
        // host's business and no longer part of the program.
        let packed = crate::graph::compile(&planet.nodes).unwrap().ops;
        assert!(packed.iter().any(|op| op.is_height_op()));
        // Materials cover the ids the generator emits.
        assert!(planet.materials.iter().any(|m| m.id() == 1));
        assert!(planet.materials.iter().any(|m| m.id() == 3));

        let mega = LevelDef::from_json_known(
            &shipped("megastructure.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();
        let packed = crate::graph::compile(&mega.nodes).unwrap().ops;
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
        let planet = LevelDef::from_json_known(
            &shipped("planet.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();
        let reg = crate::graph::registry::engine_kinds();
        let json = planet.to_json(&reg).unwrap();
        let back = LevelDef::from_json(&json, &reg).unwrap();
        assert_eq!(back.nodes, planet.nodes);
        assert_eq!(back.materials, planet.materials);
        assert_eq!(back.environment, planet.environment);
    }

    #[test]
    fn the_planet_packs_to_the_reference_program() {
        // Worth pinning exactly, because an ORACLE stands behind it:
        // `planet_program_matches_legacy_terrain_height` checks the
        // reference against the pre-program terrain formula, so tying the
        // shipped JSON to the reference transitively checks the JSON.
        //
        // The megastructure has no such oracle — see the test below,
        // which asserts what is actually true of it instead.
        let planet = LevelDef::from_json_known(
            &shipped("planet.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();
        let packed = crate::graph::compile(&planet.nodes).unwrap().ops;
        assert_eq!(packed, voxel_worldgen::program::planet_program());
    }

    /// The megastructure is a set of region-gated districts, and these
    /// are the properties that makes it one — not any particular op list.
    ///
    /// Pinning it op-for-op would only assert that nobody edited the
    /// level, which is the one thing a level is FOR. These catch the
    /// authoring mistakes that actually happen: a district with no
    /// architecture, a band nothing paints, a material with no recipe.
    #[test]
    fn every_megastructure_district_is_whole() {
        let mega = LevelDef::from_json_known(
            &shipped("megastructure.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();
        let ops = crate::graph::compile(&mega.nodes).unwrap().ops;

        assert_eq!(
            ops[0].kind, WOP_REGION_AXES,
            "the axes must be filled first"
        );

        // Every band that paints a district must also BUILD one, or the
        // district is a colour swatch with no architecture in it.
        let painted: Vec<_> = ops
            .iter()
            .filter(|op| op.kind == WOP_MATERIAL_BAND)
            .map(|op| {
                (
                    op.material,
                    pack_region([op.p0[0], op.p0[1], op.p0[2], op.p0[3]]),
                )
            })
            .collect();
        assert!(painted.len() >= 8, "only {} districts", painted.len());
        for (mat, band) in &painted {
            let built = ops.iter().filter(|op| op.region == *band).count();
            assert!(built >= 3, "district {mat} has {built} gated ops");
            assert!(
                mega.materials.iter().any(|m| material_id(m) == *mat),
                "district material {mat} has no recipe"
            );
        }

        // Distinct bands, or two districts silently share one architecture.
        let mut bands: Vec<_> = painted.iter().map(|(_, b)| *b).collect();
        bands.sort_unstable();
        bands.dedup();
        assert_eq!(bands.len(), painted.len(), "two districts share a band");

        // One cut at the end serves every district's bores: the shaft
        // registers are per-sample, and only one gate can pass at a point.
        let cut = ops.iter().rposition(|op| op.kind == WOP_SHAFTS_CUT);
        let last_bore = ops.iter().rposition(|op| op.kind == WOP_SHAFTS_XZ);
        assert!(
            cut > last_bore,
            "shafts are cut before the last one is defined"
        );
    }

    /// No district may be a sliver of the world.
    ///
    /// The bands look even written down and are not: the region axes are
    /// sums of noise, so they are bell-shaped around 0.5, and cutting
    /// both at 0.455/0.545 — which reads as thirds — gave the three
    /// middle districts about a twentieth of the world between them. The
    /// shipped cuts are the measured terciles, and this is what keeps
    /// them honest when somebody retunes the axis scale or octaves.
    #[test]
    fn no_megastructure_district_is_a_sliver() {
        let mega = LevelDef::from_json_known(
            &shipped("megastructure.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();
        let gen = mega.generator(0);
        // Through `surface_material_weight`, which is the same query the
        // host gates content with — so this measures the districts as
        // anything placing things in them will see them.
        let mats: Vec<u32> = gen
            .ops()
            .iter()
            .filter(|op| op.kind == WOP_MATERIAL_BAND)
            .map(|op| op.material)
            .collect();

        // By STRONGEST weight, not by weight over a half. In the feather
        // between two districts neither is over a half — at a corner of
        // the axis grid all four are near a quarter — so a threshold
        // would report a sixth of the world as belonging to nobody when
        // the interpreter, which tests the bands hard, leaves no gap.
        const N: i32 = 110;
        let mut hits = vec![0usize; mats.len()];
        for i in 0..N {
            for j in 0..N {
                let xz =
                    bevy::math::Vec2::new(i as f32 * 760.0 - 42_000.0, j as f32 * 760.0 - 42_000.0);
                let strongest = mats
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        let wa = gen.surface_material_weight(xz, 1.0, **a);
                        let wb = gen.surface_material_weight(xz, 1.0, **b);
                        wa.total_cmp(&wb)
                    })
                    .map(|(k, _)| k)
                    .unwrap();
                hits[strongest] += 1;
            }
        }
        let total = f64::from(N * N);
        for (mat, n) in mats.iter().zip(&hits) {
            let share = *n as f64 / total;
            assert!(
                share > 0.03,
                "district {mat} covers {:.1}% of the world",
                share * 100.0
            );
        }
    }

    fn material_id(m: &MaterialDef) -> u32 {
        match m {
            MaterialDef::Surface { id, .. }
            | MaterialDef::Canopy { id, .. }
            | MaterialDef::Zoned { id, .. } => *id,
        }
    }
}
