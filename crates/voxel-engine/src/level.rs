//! Data-driven levels: a JSON `LevelDef` describes everything the engine
//! needs to present a world — the *generator program* that is the world's
//! geometry, seed, LOD configuration, lighting, camera, feature toggles,
//! and parameterized planning-op providers. Level editors author these
//! files; the engine has no hardcoded worlds — a lush planet and a concrete
//! megacity are the same interpreter fed different data.

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

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct FeaturesDef {
    #[serde(default)]
    pub water: bool,
    #[serde(default)]
    pub vegetation: bool,
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

/// Fragment shading family (a preset until materials are palette data).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShadingDef {
    /// Procedural nature zones: grass tops, rock faces, snow, worked stone.
    #[default]
    Zones,
    /// Poured concrete: banded gray with grime and streaks.
    Concrete,
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
    pub lod: LodDef,
    pub camera: CameraDef,
    #[serde(default)]
    pub features: FeaturesDef,
    #[serde(default)]
    pub walk: WalkDef,
    #[serde(default)]
    pub shading: ShadingDef,
    /// The world's base geometry: generator ops, interpreted in order.
    pub generator: Vec<GenOpDef>,
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
            } => WorldOp::new(WOP_HEIGHT_FBM)
                .p0([offset[0], offset[1], scale, amp])
                .p1([octaves as f32, 0.0, 0.0, 0.0]),
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
fn apply_generator(level: &LevelDef) -> voxel_render::WorldProgram {
    let ops: Vec<WorldOp> = level.generator.iter().map(GenOpDef::pack).collect();
    voxel_worldgen::program::set_program(ops.clone());
    voxel_render::WorldProgram(Arc::new(ops))
}

fn shading_mode(level: &LevelDef) -> voxel_render::ShadingMode {
    match level.shading {
        ShadingDef::Zones => voxel_render::ShadingMode::Zones,
        ShadingDef::Concrete => voxel_render::ShadingMode::Concrete,
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
        app.insert_resource(program)
            .insert_resource(shading_mode(&level))
            .insert_resource(ClearColor(Color::srgb(c[0], c[1], c[2])))
            .insert_resource(LodConfig {
                max_level: level.lod.max_level,
                top_radius: level.lod.top_radius,
                top_y: level.lod.top_y,
                split_k: level.lod.split_k,
                merge_k: level.lod.merge_k,
            })
            .insert_resource(build_ops_provider(&level))
            .insert_resource(voxel_render::water::WaterEnabled(level.features.water))
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
        if key.level > 2 {
            return Vec::new(); // meter-scale features: fine LODs only
        }
        let min = key.min_corner_m().as_vec3();
        let max = min + Vec3::splat(key.edge_m() as f32);
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
    mut water: ResMut<voxel_render::WaterEnabled>,
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
    clear.0 = Color::srgb(c[0], c[1], c[2]);
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
    if new.shading != level.shading {
        commands.insert_resource(shading_mode(&new));
    }
    if new.features.water != level.features.water {
        water.0 = new.features.water;
    }
    info!("level reload: presentation applied");

    // Generation-affecting changes: rebuild the streamed world.
    let generator_changed = new.generator != level.generator;
    if generator_changed {
        commands.insert_resource(apply_generator(&new));
    }
    let regen = generator_changed
        || new.seed != level.seed
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
    if generator_changed || new.features.vegetation != level.features.vegetation {
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
        assert!(planet.features.water && planet.features.vegetation);
        assert!(matches!(planet.ops[0], OpsDef::Ruins { .. }));
        assert!(matches!(planet.ops[1], OpsDef::Roads { .. }));
        assert!(planet.sun.is_some());
        assert_eq!(planet.walk, WalkDef::Terrain);
        assert_eq!(planet.shading, ShadingDef::Zones);

        let mega = LevelDef::from_json(&shipped("megastructure.json")).unwrap();
        assert!(!mega.features.water);
        assert!(matches!(mega.ops[0], OpsDef::Pockets { .. }));
        assert!(mega.sun.is_none());
        assert_eq!(mega.walk, WalkDef::Sdf);
        assert_eq!(mega.shading, ShadingDef::Concrete);
    }

    #[test]
    fn levels_roundtrip() {
        let planet = LevelDef::from_json(&shipped("planet.json")).unwrap();
        let json = serde_json::to_string(&planet).unwrap();
        let back = LevelDef::from_json(&json).unwrap();
        assert_eq!(back.generator, planet.generator);
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
    for mut transform in &mut cameras {
        let dir = transform.forward();
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
