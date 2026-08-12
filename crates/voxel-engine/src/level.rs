//! Data-driven levels: a JSON `LevelDef` describes the world itself and
//! nothing else — the *generator program* that is its geometry
//! the material table those ops
//! reference, the lighting/haze environment, LOD configuration, and the
//! planning stack. Presentation belongs to the host, and the seed is a
//! runtime input ([`LevelPlugin::seed`]), so a level editor edits
//! exactly this file. The engine has no hardcoded worlds — a lush planet
//! and a concrete megacity are the same interpreter fed different data.

pub mod prefab;

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

/// Where a placement looks for the surface it stands on.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceMode {
    /// The height chain: one surface per column, and only the top one.
    /// Nothing in a cave, under an overhang or on an interior floor — and
    /// in a world with no height ops at all, no surface anywhere.
    #[default]
    Heightfield,
    /// Every floor the FULL program puts inside `altitude`, one of them
    /// picked at random. Costs a march down the column; see
    /// [`voxel_worldgen::Generator::floors`].
    Floors,
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
    /// Altitude band the class lives in. Also the span `surface: floors`
    /// searches, so an interior population says which storeys it wants by
    /// saying where it lives.
    pub altitude: [f32; 2],
    /// Where to look for the surface.
    #[serde(default)]
    pub surface: SurfaceMode,
    /// March step for `surface: floors`, in meters. Must be finer than the
    /// thinnest floor to be found: a slab thinner than one step can fall
    /// between two samples and not exist. Unused by `heightfield`.
    #[serde(default = "d_floor_step")]
    #[reflect(@schema::Range(0.05, 4.0))]
    pub floor_step: f32,
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
fn d_floor_step() -> f32 {
    0.5
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

/// A reusable authored object: local-space CSG ops that a
/// [`PlacementDef`] stamps into the world wherever it likes.
///
/// Normally one per file — `{"use": "prefabs/monolith_circle.json"}` in a
/// level, and the file holds the name and the ops. That is what makes a
/// prefab shareable: two levels naming the same file get the same object,
/// and editing the file changes it in both. Written inline it is still a
/// prefab, just one only its own level can reach.
#[derive(Reflect, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct PrefabDef {
    /// What a placement calls it. Unique within a level.
    #[reflect(@schema::Title)]
    pub name: String,
    /// The object, in its own local space, around the origin.
    pub ops: Vec<CsgOpDef>,
    /// The file this prefab's text lives in, relative to the level that
    /// used it — `None` for one written inline.
    ///
    /// Read from the `use` key the loader leaves behind, and the reason
    /// saving can put the prefab back where it came from instead of
    /// swallowing it into the level.
    #[serde(rename = "use", default)]
    #[reflect(@schema::Hidden)]
    pub from: Option<String>,
}

impl PrefabDef {
    /// This prefab without its link to the file it came from — what gets
    /// written INTO that file.
    pub fn detached(&self) -> Self {
        Self {
            from: None,
            ..self.clone()
        }
    }
}

impl Serialize for PrefabDef {
    /// In a level, a prefab that lives in a file is one line naming it;
    /// everything else about it belongs to that file, and
    /// [`prefab::write`] has already put it there.
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        if let Some(path) = &self.from {
            let mut out = s.serialize_struct("PrefabDef", 1)?;
            out.serialize_field("use", path)?;
            return out.end();
        }
        let mut out = s.serialize_struct("PrefabDef", 2)?;
        out.serialize_field("name", &self.name)?;
        out.serialize_field("ops", &self.ops)?;
        out.end()
    }
}

/// Every authored placement of a level, in the world, grouped by
/// priority — the ONE list of what a level's `placements` put where.
///
/// Read twice and built once: `build_world_query` turns it into an op
/// source, and `fingerprint` hashes it to decide which chunks an edit to
/// it changed. Two loops over `placements` would be two answers to that.
pub fn authored_ops(
    level: &LevelDef,
    ground: impl Fn(bevy::math::Vec2) -> f32 + Copy,
) -> Vec<(i32, Vec<CsgOp>)> {
    let mut placed: Vec<(i32, Vec<CsgOp>)> = level
        .placements
        .iter()
        .filter_map(|p| {
            let Some(local) = level.local_ops(p) else {
                warn!(
                    "placement references unknown prefab '{}'",
                    p.prefab.as_deref().unwrap_or("?")
                );
                return None;
            };
            Some((p.priority, place(p, local, ground)))
        })
        .collect();
    placed.sort_by_key(|(priority, _)| *priority);
    placed
}

/// Those ops as a chunk-culled source, or `None` if a level authored none.
pub fn authored_source(placed: Vec<(i32, Vec<CsgOp>)>) -> Option<OpsSource> {
    if placed.is_empty() {
        return None;
    }
    let placed = Arc::new(placed);
    Some(Arc::new(move |min, max| {
        let mut out = Vec::new();
        for (_, ops) in placed.iter() {
            out.extend(ops.iter().filter(|op| op.touches(min, max)).copied());
        }
        out
    }))
}

/// One placement's local ops, in the world.
///
/// Translate, yaw, uniform scale, and optional terrain seating. Public and
/// separate because it is the ONE description of where an authored object
/// actually is: the world is carved from what this returns, and an editor
/// that drew handles from its own copy of this arithmetic would draw them
/// next to the thing rather than on it.
///
/// `ground` answers the heightfield at an xz — a closure rather than a
/// generator, because a tool drawing a placement it is not standing in
/// does not have that world's generator to hand.
pub fn place(
    placement: &PlacementDef,
    local: &[CsgOpDef],
    ground: impl Fn(bevy::math::Vec2) -> f32,
) -> Vec<CsgOp> {
    let mut pos = bevy::math::Vec3::from(placement.position);
    if placement.snap_to_terrain {
        pos.y = ground(bevy::math::Vec2::new(pos.x, pos.z)) + placement.position[1];
    }
    let (sin, cos) = placement.yaw_deg.to_radians().sin_cos();
    let rot = |v: bevy::math::Vec3| {
        bevy::math::Vec3::new(v.x * cos - v.z * sin, v.y, v.x * sin + v.z * cos)
    };
    local
        .iter()
        .map(|def| {
            let mut op = def.to_op();
            op.center = (pos + rot(bevy::math::Vec3::from(op.center) * placement.scale)).to_array();
            op.half = (bevy::math::Vec3::from(op.half) * placement.scale).to_array();
            op.yaw += placement.yaw_deg.to_radians();
            op.blend *= placement.scale;
            op
        })
        .collect()
}

/// A hand-authored instance of a prefab (or inline ops) in the world —
/// VoxelPlugin's placeable asset items as level data. Applied after the
/// procedural op providers, ordered by `priority`.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PlacementDef {
    /// Name into the level's `prefabs` table...
    #[serde(default)]
    #[reflect(@schema::OneOf("prefabs[].name"))]
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
    /// Reusable authored objects, each usually in a file of its own.
    ///
    /// A list rather than a map keyed by name, because a prefab that lives
    /// in its own file carries its own name — a level that named it again
    /// would be a second copy of the name to keep in step, and two levels
    /// could give one file two names.
    #[serde(default)]
    #[reflect(@schema::Rebuilds)]
    pub prefabs: Vec<PrefabDef>,
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
        Self::read(json, None, registry, false)
    }

    /// Read a level from a file, resolving its prefabs relative to it.
    ///
    /// The path is what makes `{"use": "prefabs/monolith_circle.json"}`
    /// mean something, so a level with prefabs in it comes through here
    /// (or [`LevelDef::from_path_known`]) rather than through a string.
    pub fn from_path(
        path: &std::path::Path,
        registry: &bevy::reflect::TypeRegistryArc,
    ) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::read(&text, path.parent(), registry, false).map_err(|e| e.to_string())
    }

    /// [`LevelDef::from_path`] for a crate that can see only some of the
    /// kinds — see [`LevelDef::from_json_known`].
    pub fn from_path_known(
        path: &std::path::Path,
        registry: &bevy::reflect::TypeRegistryArc,
    ) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::read(&text, path.parent(), registry, true).map_err(|e| e.to_string())
    }

    /// [`LevelDef::from_json_known`] for text a caller built from a level
    /// it has the path of — a fixture with one field patched, a tool
    /// rewriting a document. The text is not on disk, but the prefabs it
    /// names still are, and `base` says where.
    pub fn from_json_known_in(
        json: &str,
        base: &std::path::Path,
        registry: &bevy::reflect::TypeRegistryArc,
    ) -> Result<Self, serde_json::Error> {
        Self::read(json, Some(base), registry, true)
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
        Self::read(json, None, registry, true)
    }

    /// The local-space ops a placement stamps: its prefab's, or its own
    /// inline ones. `None` when it names a prefab this level has not got.
    pub fn local_ops<'a>(&'a self, placement: &'a PlacementDef) -> Option<&'a [CsgOpDef]> {
        match &placement.prefab {
            Some(name) => self
                .prefabs
                .iter()
                .find(|p| p.name == *name)
                .map(|p| p.ops.as_slice()),
            None => Some(&placement.ops),
        }
    }

    /// Splice the prefabs, then read what that leaves.
    ///
    /// One place, because the two things that could disagree — what the
    /// document says and where its prefabs are — have to be decided
    /// together.
    fn read(
        json: &str,
        base: Option<&std::path::Path>,
        registry: &bevy::reflect::TypeRegistryArc,
        drop_unknown: bool,
    ) -> Result<Self, serde_json::Error> {
        use serde::de::Error as _;
        let mut doc: serde_json::Value = serde_json::from_str(json)?;
        prefab::resolve(&mut doc, base).map_err(serde_json::Error::custom)?;
        if drop_unknown {
            let known: Vec<String> = {
                let reg = registry.read();
                crate::graph::registry::kinds(&reg)
                    .into_iter()
                    .map(|(kind, _)| kind.to_string())
                    .collect()
            };
            if let Some(nodes) = doc.get_mut("nodes").and_then(|n| n.as_array_mut()) {
                nodes.retain(|n| known.iter().any(|k| n["kind"] == k.as_str()));
            }
        }
        crate::graph::with_registry(registry, || serde_json::from_value(doc))
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
    ///
    /// Panics on a level whose graph does not compile. Anything that
    /// accepts a level from OUTSIDE — a watched file, a tool writing the
    /// resource, an editor — must ask [`LevelDef::try_generator`] first: a
    /// mistyped wire is an authoring error, and an authoring error must
    /// never take down a live session.
    pub fn generator(&self, seed: u64) -> voxel_worldgen::Generator {
        self.try_generator(seed)
            .unwrap_or_else(|e| panic!("level graph: {e}"))
    }

    /// This level's generator, or the compiler's complaint about why there
    /// is none.
    pub fn try_generator(
        &self,
        seed: u64,
    ) -> Result<voxel_worldgen::Generator, crate::graph::Error> {
        let program = crate::graph::compile(&self.nodes)?;
        assert_region_axes_first(&program.ops);
        Ok(voxel_worldgen::Generator::new(
            program.ops,
            seed as u32,
            sun_dir(self),
        ))
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

/// The file the open level came from.
///
/// Public where [`LevelSource`] is not: a tool needs to know where the
/// document lives — to find its prefabs' other users, say — without being
/// handed the watcher's polling state as well.
#[derive(Resource, Clone, Debug)]
pub struct LevelPath(pub std::path::PathBuf);

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
    /// The level file and every prefab file it uses, each with the mtime
    /// it was last seen at.
    ///
    /// A prefab is a level's text as much as the level file is, so editing
    /// one has to reload the level that uses it — otherwise hot reload
    /// would work on levels and quietly not on the things levels are made
    /// of. Re-derived on every reload, because a reload can add a `use` or
    /// take one away.
    watched: Vec<(std::path::PathBuf, Option<std::time::SystemTime>)>,
    poll: Timer,
}

/// Every file a level's text lives in, with its mtime right now.
fn watch_list(
    level: &LevelDef,
    path: &std::path::Path,
) -> Vec<(std::path::PathBuf, Option<std::time::SystemTime>)> {
    let base = path.parent().unwrap_or(std::path::Path::new("."));
    std::iter::once(path.to_path_buf())
        .chain(prefab::sources(&level.prefabs, base))
        .map(|p| {
            let mtime = std::fs::metadata(&p).and_then(|m| m.modified()).ok();
            (p, mtime)
        })
        .collect()
}

impl Plugin for LevelPlugin {
    fn build(&self, app: &mut App) {
        let level = self.def.clone();

        app.add_message::<LevelReloaded>()
            .add_message::<SaveLevel>()
            .insert_resource(WorldSeed(self.seed))
            .insert_resource(HostPlanner(self.planner.clone()));
        if let Some(path) = &self.source {
            app.insert_resource(LevelPath(path.clone()));
            app.insert_resource(LevelSource {
                path: path.clone(),
                watched: watch_list(&level, path),
                poll: Timer::from_seconds(0.5, TimerMode::Repeating),
            })
            .add_systems(Update, (watch_level_file, save_level));
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
    let mut world = WorldQuery::new(generator.clone());
    if let Some(planner) = planner.and_then(|h| h.build(level, seed, generator)) {
        world = world.with_planner(planner);
    }
    // Authored placements are served after the planner's ops, ordered by
    // priority among themselves.
    if let Some(source) = authored_source(authored_ops(level, |xz| generator.height(xz, 1.0))) {
        world = world.with_source(source);
    }
    world
}

/// Poll the level file; apply edits live. Presentation fields (colors,
/// lights, camera speeds, split/merge tuning, shading) apply directly;
/// changes to the generator/ops/LOD topology rebuild the streamed
/// world in place — including swapping in a completely different world.
#[allow(clippy::too_many_arguments)]
/// Write the live level back to the file it was loaded from.
///
/// The watcher is told the new mtime rather than left to notice it: our
/// own write is not an edit, and reloading it would re-diff and re-apply a
/// level that is already live.
fn save_level(
    mut asked: MessageReader<SaveLevel>,
    level: Res<LevelDef>,
    registry: Res<AppTypeRegistry>,
    source: Option<ResMut<LevelSource>>,
) {
    if asked.read().count() == 0 {
        return;
    }
    let Some(mut source) = source else {
        warn_once!("level save: this level was not loaded from a file");
        return;
    };
    match save_to(&level, &source.path, &registry.0) {
        Ok(()) => {
            // Every file we just wrote, prefabs included — the watcher
            // must not read its own writing back as an edit.
            source.watched = watch_list(&level, &source.path.clone());
            info!("level saved: {}", source.path.display());
        }
        Err(e) => warn!("level save: {e}"),
    }
}

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
    // Any of them: a prefab edit is a level edit.
    let now = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    if !source.watched.iter().any(|(p, seen)| now(p) != *seen) {
        return;
    }
    // Re-stamped before the parse, not after, so a prefab that does not
    // parse is complained about once rather than every poll.
    source.watched = source
        .watched
        .iter()
        .map(|(p, _)| (p.clone(), now(p)))
        .collect();

    let new = match LevelDef::from_path(std::path::Path::new(&source.path), &registry.0) {
        Ok(new) => new,
        Err(e) => {
            warn!("level reload: {e}");
            return;
        }
    };
    // Authoring errors must never take down a live session: keep the
    // running world and report, exactly like a parse error.
    if let Err(e) = crate::graph::compile(&new.nodes) {
        warn!("level reload: {e}");
        return;
    }
    if let Some(host) = &planner.0 {
        if let Err(e) = host.validate(&new) {
            warn!("level reload: invalid planning data — {e}");
            return;
        }
    }
    // The reload may have added a `use` or dropped one, so what to watch
    // is re-derived from what actually loaded rather than kept.
    source.watched = watch_list(&new, &source.path.clone());
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
/// Did only the AUTHORED geometry move?
///
/// The narrow case worth separating: placements and prefabs put ops in a
/// bounded part of the world, so the chunks that care can be found rather
/// than assumed. Everything else about a level — its nodes, its sun, its
/// LOD topology — changes what every chunk is.
fn only_authored_moved(new: &LevelDef, old: &LevelDef) -> bool {
    (new.placements != old.placements || new.prefabs != old.prefabs)
        && sun_dir(new) == sun_dir(old)
        && new.lod.max_level == old.lod.max_level
        && new.lod.top_radius == old.lod.top_radius
        && new.lod.top_y == old.lod.top_y
        && new.nodes == old.nodes
}

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
    // Grouped: the two are the two halves of one answer — everything, or
    // just these chunks — and clippy caps a system at seven arguments.
    (mut rebuild, lod, chunks): (
        ResMut<StreamingRebuild>,
        Res<crate::lod_layers::LodLayers>,
        Res<crate::chunkgen::ChunkGen>,
    ),
    mut reloaded: MessageWriter<LevelReloaded>,
    // Grouped: the two registries are always touched together, and
    // clippy caps a system's arguments at seven.
    (mut worlds, mut render): (ResMut<crate::Worlds>, ResMut<voxel_render::RenderWorlds>),
) {
    if !level.is_changed() {
        return;
    }
    let new = level.clone();
    // Every writer of the resource arrives here — the watched file, a tool
    // poking a value over BRP, the editor. A level that does not compile
    // is an authoring error: report it, keep the running world, and leave
    // `applied` alone so the next good edit diffs against what is live.
    if let Err(e) = crate::graph::compile(&new.nodes) {
        warn!("level: {e} — keeping the running world");
        return;
    }
    let level = &applied.0;
    // Filled in below if this edit only moved authored geometry; acted on
    // after the borrow of `worlds` ends.
    let mut narrow: Option<(crate::lod_layers::Restale, Option<OpsSource>)> = None;
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
    let stale = staleness(&new, level);
    // Only where something needs one. This runs on EVERY write to the
    // resource, and a colour dragged in the editor writes one a frame:
    // compiling the graph and rebuilding the CPU twin to hand both to
    // nobody is the whole cost of tuning a material.
    if generator_changed || stale.is_some() {
        let (program, generator) = build_generator(&new, seed.0);
        if generator_changed {
            render.program = program;
        }
        // A planning-only edit still replaces the planner — that is where
        // the populations are registered — but leaves the streamed chunks
        // alone. They were carved by ops this edit cannot have changed, so
        // tearing them down would regenerate every one of them into itself.
        if let Some(stale) = stale {
            if stale == Invalidates::World {
                world.config.max_level = new.lod.max_level;
                world.config.top_radius = new.lod.top_radius;
                world.config.top_y = new.lod.top_y;
                world.generator = generator.clone();
                // Somebody moved an authored object: rebuild the chunks
                // that care instead of assuming all of them do. Handed
                // out below rather than acted on here, because the
                // rebuild reads the world's ops through `ChunkGen` and
                // that has to be told about this edit FIRST.
                if only_authored_moved(&new, level) {
                    let ground = |xz: bevy::math::Vec2| generator.height(xz, 1.0);
                    let was = authored_ops(level, ground);
                    let now = authored_ops(&new, ground);
                    let flat = |g: &[(i32, Vec<CsgOp>)]| -> Vec<CsgOp> {
                        g.iter().flat_map(|(_, ops)| ops.iter().copied()).collect()
                    };
                    narrow = Some((
                        crate::lod_layers::Restale {
                            seed: seed.0 as u32,
                            ops: std::sync::Arc::new(generator.ops().to_vec()),
                            was_placed: flat(&was),
                            now_placed: flat(&now),
                        },
                        authored_source(now),
                    ));
                } else {
                    rebuild.0 = true;
                }
            }
            // The narrow path keeps the planner it already has and swaps
            // only the authored source, below. Building a fresh one here
            // would throw away a resident planning stack and make the
            // next ops query wait for it to come back — which is where
            // 1.5 s of a two-metre nudge used to go.
            if narrow.is_none() {
                world.query = build_world_query(&new, seed.0, &generator, planner.0.as_ref());
            }
            world.level = new.clone();
            if rebuild.0 {
                info!("level reload: {stale:?} is stale — rebuilding it");
            }
        }
    }

    // A rebuilt chunk reads its ops through `ChunkGen`, and the sync that
    // normally refreshes it runs in PreUpdate — a frame LATER than this.
    // Forcing a rebuild before then regenerates the chunk from the
    // previous edit's ops, which reads as a world one edit behind the
    // handles. So the provider is pushed here, before anything is asked
    // to rebuild, rather than left to a system that runs afterwards.
    if let Some((edit, source)) = narrow {
        // Planning is untouched by moving a rock, so the planner is kept.
        // Rebuilding it would make the next ops query wait for a whole
        // planning stack to regenerate from cold — 1.5 s on the planet.
        if let Some(world) = worlds.get_mut(0) {
            world.query = world.query.replacing_sources(source.into_iter().collect());
        }
        chunks.set_ops_providers(worlds.ops_providers());
        if !lod.restale(0, edit) {
            // Nothing streaming yet, so nothing built to fix.
            rebuild.0 = true;
        }
    }

    // The host owns the scene: it reads the new definition off this
    // message and applies its own camera, lights and clear color.
    let previous = std::mem::replace(&mut applied.0, new);
    reloaded.write(LevelReloaded { previous });
}

/// Ask for the live level to be written back to the file it came from.
///
/// The engine owns the file, so it owns saving; WHAT asks is somebody
/// else's business — a menu item, a keystroke in a panel, a tool over the
/// wire. A level with no source file ignores it and says so once.
#[derive(Message, Debug, Default, Clone, Copy)]
pub struct SaveLevel;

/// Write a level to `path`, as a level file.
///
/// Separate from the system so it can be tested without an `App`: the
/// system is the five lines that decide WHEN, and this is the part that
/// can fail.
pub fn save_to(
    level: &LevelDef,
    path: &std::path::Path,
    registry: &bevy::reflect::TypeRegistryArc,
) -> Result<(), String> {
    // The prefabs first, and only then the level: a level pointing at a
    // file that is not there yet is a level that does not load, and a
    // crash between the two writes should leave the readable half.
    let base = path.parent().unwrap_or(std::path::Path::new("."));
    prefab::write(&level.prefabs, base)?;
    let json = level.to_json(registry).map_err(|e| e.to_string())?;
    // A trailing newline, because a level is a text file under version
    // control and every other tool that writes one leaves it.
    std::fs::write(path, format!("{json}\n")).map_err(|e| e.to_string())
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

    /// A shipped level's path. A level's prefabs live beside it, so it
    /// loads through its path rather than its text.
    fn shipped(name: &str) -> std::path::PathBuf {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../levels/");
        std::path::PathBuf::from(format!("{path}{name}"))
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
        let planet = LevelDef::from_path_known(
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
        let planet = LevelDef::from_path_known(
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

        let planet = LevelDef::from_path_known(
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
        let mut planet = LevelDef::from_path_known(
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

    /// An authoring mistake must be a message, not a dead session.
    ///
    /// The editor makes this reachable by CLICKING — rewiring a port to a
    /// node declared later is one menu pick — and the watched file always
    /// could. Before this, either one panicked the running app from
    /// `generator()`, three calls below where the level was accepted.
    #[test]
    fn a_level_that_does_not_compile_is_refused_not_fatal() {
        let mut planet = LevelDef::from_path_known(
            &shipped("planet.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();
        assert!(
            planet.try_generator(0).is_ok(),
            "the shipped level compiles"
        );

        // Rewire a port to something that is not a node.
        let wired = planet
            .nodes
            .iter_mut()
            .find(|n| !n.wires.is_empty())
            .expect("planet wires its nodes");
        let port = wired.wires.iter().next().map(|(p, _)| p.clone()).unwrap();
        wired
            .wires
            .0
            .insert(port.clone(), crate::graph::Wire::One("nope".into()));

        let Err(err) = planet.try_generator(0) else {
            panic!("a dangling wire has to be refused");
        };
        let said = err.to_string();
        assert!(said.contains("nope") && said.contains(&port), "{said}");
    }

    #[test]
    fn shipped_levels_parse() {
        let planet = LevelDef::from_path_known(
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

        let mega = LevelDef::from_path_known(
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

    /// A saved level is a level: it parses back, and it is the one that
    /// was in memory rather than the one on disk when it loaded.
    #[test]
    fn saving_writes_a_level_that_loads() {
        let reg = crate::graph::registry::engine_kinds();
        let mut planet = LevelDef::from_path_known(&shipped("planet.json"), &reg).unwrap();
        planet.lod.max_level -= 1;

        // Its own directory: saving writes the level AND its prefabs,
        // and what has to load is the pair.
        let dir = std::env::temp_dir().join("voxel2-save-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("level.json");
        save_to(&planet, &path, &reg).expect("writes");
        let back = LevelDef::from_path_known(&path, &reg).unwrap();
        assert_eq!(
            back.prefabs, planet.prefabs,
            "the prefab came back through its own file"
        );
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(
            back.lod.max_level, planet.lod.max_level,
            "the EDIT is there"
        );
        assert_eq!(back.nodes, planet.nodes);
        assert_eq!(back.materials, planet.materials);
    }

    /// A level can be SNAPSHOTTED and put back through reflection, which
    /// is what an undo stack is: a copy taken by something that does not
    /// know what a level is.
    ///
    /// Through the DYNAMIC form rather than `reflect_clone`, because bevy
    /// implements that for `[f32; 3]` no more than for a boxed trait
    /// object, and a level is full of both.
    #[test]
    fn a_level_can_be_snapshotted_and_restored() {
        use bevy::reflect::PartialReflect;
        let reg = crate::graph::registry::engine_kinds();
        let planet = LevelDef::from_path_known(&shipped("planet.json"), &reg).unwrap();

        let snapshot = planet.to_dynamic();
        let mut edited = planet.clone();
        edited.lod.max_level -= 1;
        // Whatever recipe it was, make it a different one.
        edited.materials[0] = serde_json::from_str::<MaterialDef>(
            r#"{"type":"surface","id":99,"base":[1.0,0.0,1.0]}"#,
        )
        .unwrap();
        assert_ne!(edited.materials, planet.materials);

        edited.try_apply(snapshot.as_ref()).expect("restores");
        assert_eq!(edited.lod.max_level, planet.lod.max_level);
        assert_eq!(edited.materials, planet.materials, "a recipe comes back");
        assert_eq!(edited.nodes, planet.nodes, "and so do the nodes");
    }

    #[test]
    fn levels_roundtrip() {
        let planet = LevelDef::from_path_known(
            &shipped("planet.json"),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap();
        let reg = crate::graph::registry::engine_kinds();
        let json = planet.to_json(&reg).unwrap();
        // Back through the same directory: the round trip includes the
        // `use` lines, which mean nothing without it.
        let back = LevelDef::from_json_known_in(&json, shipped("").as_path(), &reg).unwrap();
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
        let planet = LevelDef::from_path_known(
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
        let mega = LevelDef::from_path_known(
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
        let mega = LevelDef::from_path_known(
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
