//! Data-driven levels: a JSON `LevelDef` describes everything the engine
//! needs to present a world — world kind, seed, LOD configuration, lighting,
//! camera, feature toggles, and *named* planning-op providers. Level editors
//! author these files; the engine has no hardcoded worlds.

use std::sync::Arc;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use voxel_core::csg::CsgOp;
use voxel_core::ChunkKey;
use voxel_render::WorldKind;

use crate::streaming::{ChunkOpsProvider, StreamingRebuild};
use crate::{LodConfig, VoxelEnginePlugin};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorldDef {
    Planet,
    Megastructure,
}

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

/// A complete level description.
#[derive(Resource, Serialize, Deserialize, Clone, Debug)]
pub struct LevelDef {
    pub name: String,
    pub world: WorldDef,
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
    /// Planning-op providers, composed in order.
    #[serde(default)]
    pub ops: Vec<OpsDef>,
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

    fn world_kind(&self) -> WorldKind {
        match self.world {
            WorldDef::Planet => WorldKind::Planet,
            WorldDef::Megastructure => WorldKind::Megastructure,
        }
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

        app.insert_resource(ClearColor(Color::srgb(c[0], c[1], c[2])))
            .insert_resource(LodConfig {
                max_level: level.lod.max_level,
                top_radius: level.lod.top_radius,
                top_y: level.lod.top_y,
                split_k: level.lod.split_k,
                merge_k: level.lod.merge_k,
            })
            .insert_resource(build_ops_provider(&level))
            .insert_resource(level.clone())
            .add_plugins(VoxelEnginePlugin {
                world: level.world_kind(),
                vegetation: level.features.vegetation,
            })
            .add_systems(Startup, setup_level)
            .add_systems(Update, (autopilot, walk_mode).chain());

        if level.features.water {
            app.add_plugins(voxel_render::WaterPlugin);
        }
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
/// lights, camera speeds, split/merge tuning) apply directly; changes to
/// seed/ops/LOD topology rebuild the streamed world; world kind and
/// feature toggles need a restart.
#[allow(clippy::too_many_arguments)]
fn reload_level(
    mut commands: Commands,
    time: Res<Time>,
    mut source: ResMut<LevelSource>,
    mut level: ResMut<LevelDef>,
    mut clear: ResMut<ClearColor>,
    mut lod: ResMut<LodConfig>,
    mut rebuild: ResMut<StreamingRebuild>,
    mut cameras: Query<&mut voxel_debug::FreeCamera>,
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
    info!("level reload: presentation applied");

    // Generation-affecting changes: rebuild the streamed world.
    let regen = new.seed != level.seed
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
    if new.world != level.world
        || new.features.water != level.features.water
        || new.features.vegetation != level.features.vegetation
    {
        warn!("level reload: world/features changed — restart required");
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
        assert_eq!(planet.world, WorldDef::Planet);
        assert!(planet.features.water && planet.features.vegetation);
        assert!(matches!(planet.ops[0], OpsDef::Ruins { .. }));
        assert!(matches!(planet.ops[1], OpsDef::Roads { .. }));
        assert!(planet.sun.is_some());

        let mega = LevelDef::from_json(&shipped("megastructure.json")).unwrap();
        assert_eq!(mega.world, WorldDef::Megastructure);
        assert!(!mega.features.water);
        assert!(matches!(mega.ops[0], OpsDef::Pockets { .. }));
        assert!(mega.sun.is_none());
    }

    #[test]
    fn levels_roundtrip() {
        let planet = LevelDef::from_json(&shipped("planet.json")).unwrap();
        let json = serde_json::to_string(&planet).unwrap();
        let back = LevelDef::from_json(&json).unwrap();
        assert_eq!(back.world, planet.world);
        assert_eq!(back.ops, planet.ops);
        assert_eq!(back.camera.start, planet.camera.start);
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

/// `VOXEL_WALK=1`: on-foot mode. Planet worlds glue the camera to the
/// terrain height mirror; megastructures do gravity + capsule collision
/// against the SDF mirror (including planned ops).
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
    match level.world {
        WorldDef::Planet => {
            for mut t in &mut cameras {
                let h = voxel_worldgen::terrain_height(
                    bevy::math::Vec2::new(t.translation.x, t.translation.z),
                    1.0,
                );
                t.translation.y = h + 1.75;
            }
        }
        WorldDef::Megastructure => {
            use voxel_worldgen::mega::{mega_sdf, mega_sdf_with_ops, pockets_ops};
            const RADIUS: f32 = 0.5;
            const EYE: f32 = 1.6;
            let seed = level.seed;

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
                            let floor = (p.y / 44.0).round() * 44.0;
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
