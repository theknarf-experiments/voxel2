//! voxel2: a demo host for the voxel engine.
//!
//!     cargo run -p voxel2 -- levels/planet.json
//!     cargo run -p voxel2 -- levels/megastructure.json
//!
//! Everything scene-shaped lives here, not in the engine: the camera and
//! its controller, lights, clear color, the environment-variable
//! affordances used by the eval scripts and tooling. A game embedding
//! `voxel_engine` supplies its own — the engine only asks that some
//! entity carries [`VoxelStreamSource`].

use bevy::prelude::*;
use bevy::winit::{UpdateMode, WinitSettings};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voxel_debug::prelude::*;
use voxel_debug::{remote::VoxelRemotePlugin, viz::VoxelVizPlugin};
use voxel_engine::level::LevelReloaded;
use voxel_engine::{LevelDef, LevelPlugin, VoxelStreamSource};

mod props;
use props::{PropTable, PropsPlugin};

/// What the HOST owns in a level file. The engine's [`LevelDef`] is
/// flattened in, so one file still describes a whole level — but the
/// engine's schema has no idea what a camera or a clear color is.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct HostLevel {
    #[serde(flatten)]
    world: LevelDef,
    #[serde(default = "default_clear")]
    clear_color: [f32; 3],
    #[serde(default)]
    camera: CameraDef,
    /// Directional light, if the level has a sun. Its *direction* is
    /// engine data (baked shadows) and lives in `environment`.
    #[serde(default)]
    sun: Option<SunDef>,
    #[serde(default = "default_ambient")]
    ambient: AmbientDef,
    /// Appearance of each scatter class the engine streams.
    #[serde(default)]
    props: HashMap<String, props::PropClassDef>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CameraDef {
    #[serde(default)]
    start: [f32; 3],
    #[serde(default = "default_look")]
    look: [f32; 3],
    #[serde(default = "default_walk")]
    walk_speed: f32,
    #[serde(default = "default_run")]
    run_speed: f32,
}

impl Default for CameraDef {
    fn default() -> Self {
        Self {
            start: [0.0, 100.0, 0.0],
            look: default_look(),
            walk_speed: default_walk(),
            run_speed: default_run(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SunDef {
    illuminance: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AmbientDef {
    color: [f32; 3],
    brightness: f32,
}

fn default_clear() -> [f32; 3] {
    [0.62, 0.72, 0.88]
}
fn default_look() -> [f32; 3] {
    [0.0, 0.0, 1.0]
}
fn default_walk() -> f32 {
    12.0
}
fn default_run() -> f32 {
    60.0
}
fn default_ambient() -> AmbientDef {
    AmbientDef {
        color: [0.6, 0.7, 0.9],
        brightness: 220.0,
    }
}

/// The sun direction for the host's light comes from the engine's
/// environment block, which owns it (the shadow bake uses it too).
fn sun_direction(world: &LevelDef) -> Vec3 {
    Vec3::from(world.environment.sun_direction).normalize_or(Vec3::Y)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "levels/planet.json".to_string());
    let json = match std::fs::read_to_string(&path) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("failed to read level '{path}': {e}");
            std::process::exit(1);
        }
    };
    let host: HostLevel = match serde_json::from_str(&json) {
        Ok(level) => level,
        Err(e) => {
            eprintln!("failed to parse level '{path}': {e}");
            std::process::exit(1);
        }
    };
    let level = host.world.clone();

    let hole_eval = std::env::var_os("VOXEL_EVAL_HOLES").is_some();
    let clear = if hole_eval {
        // Coverage eval paints the background magenta: any background
        // pixel below the horizon is a hole in the world.
        Color::srgb(1.0, 0.0, 1.0)
    } else {
        let c = host.clear_color;
        Color::srgb(c[0], c[1], c[2])
    };

    let mut app = App::new();
    app
        // Keep running at full speed when the window loses focus. Bevy's
        // default (`WinitSettings::game()`) drops to reactive_low_power
        // at 60 Hz unfocused, which silently caps every measurement taken
        // while the window is backgrounded.
        .insert_resource(WinitSettings {
            focused_mode: UpdateMode::Continuous,
            unfocused_mode: UpdateMode::Continuous,
        })
        .insert_resource(ClearColor(clear))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: format!("voxel2 — {}", level.name),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(PropTable(host.props.clone()))
        .insert_resource(HostScene(host.clone()))
        .insert_resource(LevelPath(path.clone().into()))
        .add_plugins((
            VoxelDebugPlugin,
            VoxelVizPlugin,
            PropsPlugin,
            LevelPlugin {
                def: level,
                source: Some(path.into()),
                hole_eval,
                remote_port: None,
            },
        ))
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (autopilot, apply_reloaded_scene));

    // Live tooling: VOXEL_REMOTE=1 (or a port) starts the BRP server the
    // voxctl CLI drives.
    if let Ok(v) = std::env::var("VOXEL_REMOTE") {
        let port = match v.parse::<u16>() {
            Ok(p) if p > 1024 => p,
            _ => 15702,
        };
        app.add_plugins(VoxelRemotePlugin { port });
    }
    app.run();
}

fn parse3(s: String) -> Option<Vec3> {
    let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
    (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
}

/// A look-at up axis that is never parallel to the view direction.
fn up_for(look: Vec3) -> Vec3 {
    if look.normalize_or_zero().dot(Vec3::Y).abs() > 0.9 {
        Vec3::Z
    } else {
        Vec3::Y
    }
}

/// The host's scene: camera (tagged as the streaming source), sun, and
/// ambient light. `VOXEL_START` / `VOXEL_LOOK` override the level's
/// camera for repeatable runs.
fn setup_scene(mut commands: Commands, scene: Res<HostScene>) {
    let host = &scene.0;
    if let Some(sun) = &host.sun {
        let dir = sun_direction(&host.world);
        commands.spawn((
            LevelSun,
            DirectionalLight {
                illuminance: sun.illuminance,
                ..default()
            },
            Transform::from_translation(Vec3::ZERO).looking_to(-dir, Vec3::Y),
        ));
    }
    let a = &host.ambient;
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(a.color[0], a.color[1], a.color[2]),
        brightness: a.brightness,
        ..default()
    });

    let start = std::env::var("VOXEL_START")
        .ok()
        .and_then(parse3)
        .unwrap_or(Vec3::from(host.camera.start));
    let look = std::env::var("VOXEL_LOOK")
        .ok()
        .and_then(parse3)
        .unwrap_or(Vec3::from(host.camera.look));
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(start).looking_at(start + look * 1000.0, up_for(look)),
        // The engine streams around whatever carries this.
        VoxelStreamSource,
        FreeCamera {
            walk_speed: host.camera.walk_speed,
            run_speed: host.camera.run_speed,
            ..default()
        },
    ));
}

#[derive(Component)]
struct LevelSun;

/// The host-owned half of the level file, kept for reloads.
#[derive(Resource, Clone)]
struct HostScene(HostLevel);

/// Re-apply the host-owned parts of a hot-reloaded level: clear color,
/// lights, camera speeds, window title, and a jump if the level moved
/// its camera. The engine has already applied everything it owns.
#[allow(clippy::too_many_arguments)]
fn apply_reloaded_scene(
    mut commands: Commands,
    mut reloaded: MessageReader<LevelReloaded>,
    mut scene: ResMut<HostScene>,
    mut props: ResMut<PropTable>,
    source: Res<LevelPath>,
    mut clear: ResMut<ClearColor>,
    mut cameras: Query<&mut FreeCamera>,
    mut transforms: Query<&mut Transform, With<VoxelStreamSource>>,
    mut windows: Query<&mut Window>,
    suns: Query<Entity, With<LevelSun>>,
) {
    if reloaded.read().count() == 0 {
        return;
    }
    // The engine reloaded its half; re-read ours from the same file.
    let Ok(json) = std::fs::read_to_string(&source.0) else {
        return;
    };
    let Ok(host) = serde_json::from_str::<HostLevel>(&json) else {
        return;
    };
    let previous = std::mem::replace(&mut scene.0, host.clone());
    props.0 = host.props.clone();

    let c = host.clear_color;
    clear.0 = Color::srgb(c[0], c[1], c[2]);
    let a = &host.ambient;
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(a.color[0], a.color[1], a.color[2]),
        brightness: a.brightness,
        ..default()
    });
    for sun in &suns {
        commands.entity(sun).despawn();
    }
    if let Some(sun) = &host.sun {
        let dir = sun_direction(&host.world);
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
        cam.walk_speed = host.camera.walk_speed;
        cam.run_speed = host.camera.run_speed;
    }
    if host.world.name != previous.world.name {
        for mut window in &mut windows {
            window.title = format!("voxel2 — {}", host.world.name);
        }
    }
    // A different camera start means a different place (typically a
    // whole different level file): jump there.
    if host.camera.start != previous.camera.start || host.camera.look != previous.camera.look {
        let start = Vec3::from(host.camera.start);
        let look = Vec3::from(host.camera.look);
        for mut t in &mut transforms {
            *t = Transform::from_translation(start).looking_at(start + look * 1000.0, up_for(look));
        }
    }
}

/// Where the level file lives, so the host can re-read its own half.
#[derive(Resource)]
struct LevelPath(std::path::PathBuf);

/// `VOXEL_AUTOPILOT=<m/s>` flies the camera forward — the smoke-test and
/// coverage-eval driver.
fn autopilot(mut cameras: Query<&mut Transform, With<VoxelStreamSource>>, time: Res<Time>) {
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
