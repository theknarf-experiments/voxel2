//! voxel2: a demo host for the voxel engine.
//!
//!     cargo run -p voxel2 -- levels/planet.json
//!     cargo run -p voxel2 -- levels/megastructure.json
//!
//! Everything scene-shaped is hardcoded HERE, in Rust, not in the level
//! file: the camera and its controller, lights, clear color, prop
//! appearance, the environment-variable affordances the eval scripts
//! use. The JSON holds only what the reusable crates own — the world an
//! editor would edit. A game embedding `voxel_engine` writes its scene
//! the same way, in code; the engine only asks that some entity carries
//! [`VoxelStreamSource`].

use bevy::light::CascadeShadowConfigBuilder;
use bevy::prelude::*;
use bevy::winit::{UpdateMode, WinitSettings};
use voxel_debug::prelude::*;
use voxel_debug::{remote::VoxelRemotePlugin, viz::VoxelVizPlugin};
use voxel_engine::level::LevelReloaded;
use voxel_engine::{LevelDef, LevelPlugin, VoxelStreamSource};

mod props;
mod ribbons;
mod water;
use props::PropsPlugin;
use ribbons::RibbonsPlugin;
use water::WaterPlugin;

/// The demo's presentation for one world. A game would inline these
/// values wherever it spawns its camera and lights.
#[derive(Clone, Debug)]
struct Scene {
    clear_color: Color,
    /// Camera start and look direction, and the flycam's speeds.
    start: Vec3,
    look: Vec3,
    walk_speed: f32,
    run_speed: f32,
    /// Directional light strength in lux, if the world has a visible sun.
    /// Its *direction* is engine data (the shadow bake needs it) and comes
    /// from the level's `environment`.
    sun_illuminance: Option<f32>,
    ambient_color: Color,
    ambient_brightness: f32,
    /// Atmospheric haze. Voxel surfaces shade through Bevy's PBR, so this
    /// is an ordinary `DistanceFog` on the camera.
    fog: Option<DistanceFog>,
}

/// Which scene to dress a world with. Keying off the level file's name
/// keeps both shipped demos in one binary; a real game has exactly one
/// scene and no match at all.
fn scene_for(level_path: &std::path::Path) -> Scene {
    match level_path.file_stem().and_then(|s| s.to_str()) {
        Some("megastructure") => Scene {
            clear_color: Color::srgb(0.035, 0.045, 0.06),
            start: Vec3::new(11.0, 12.0, 7.0),
            look: Vec3::new(0.7, 0.05, 0.7),
            walk_speed: 8.0,
            run_speed: 60.0,
            // Interiors are lit by the level's own emissive materials.
            sun_illuminance: None,
            ambient_color: Color::srgb(0.6, 0.7, 0.9),
            ambient_brightness: 3_800.0,
            fog: None,
        },
        _ => Scene {
            clear_color: Color::srgb(0.65, 0.77, 0.94),
            start: Vec3::new(-27570.0, 80.0, -36770.0),
            look: Vec3::new(0.4, -0.35, 0.4),
            walk_speed: 60.0,
            run_speed: 600.0,
            sun_illuminance: Some(light_consts::lux::FULL_DAYLIGHT),
            ambient_color: Color::srgb(0.7, 0.8, 1.0),
            ambient_brightness: 1_100.0,
            fog: Some(DistanceFog {
                color: Color::srgb(0.62, 0.72, 0.88),
                directional_light_color: Color::srgb(0.92, 0.85, 0.72),
                directional_light_exponent: 4.0,
                falloff: FogFalloff::Exponential { density: 6.0e-5 },
            }),
        },
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
    let level = match LevelDef::from_json(&json) {
        Ok(level) => level,
        Err(e) => {
            eprintln!("failed to parse level '{path}': {e}");
            std::process::exit(1);
        }
    };
    let scene = scene_for(std::path::Path::new(&path));
    let title = format!(
        "voxel2 — {}",
        std::path::Path::new(&path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("world")
    );

    let hole_eval = std::env::var_os("VOXEL_EVAL_HOLES").is_some();
    let clear = if hole_eval {
        // Coverage eval paints the background magenta: any background
        // pixel below the horizon is a hole in the world.
        Color::srgb(1.0, 0.0, 1.0)
    } else {
        scene.clear_color
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
                title,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(HostScene(scene))
        .add_plugins((
            VoxelDebugPlugin,
            VoxelVizPlugin,
            PropsPlugin,
            // Water is the demo's: generic ribbon data in, water look out.
            RibbonsPlugin,
            WaterPlugin,
            LevelPlugin {
                def: level,
                // A game picks this at new-game time and restores it
                // from its save; the demo just wants a stable world.
                seed: 0,
                source: Some(path.into()),
                hole_eval,
                remote_port: None,
            },
        ))
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (autopilot, follow_reloaded_sun));

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
/// ambient light. `VOXEL_START` / `VOXEL_LOOK` override the scene's
/// camera for repeatable runs.
fn setup_scene(mut commands: Commands, scene: Res<HostScene>, level: Res<LevelDef>) {
    let host = &scene.0;
    if let Some(illuminance) = host.sun_illuminance {
        commands.spawn((
            LevelSun,
            DirectionalLight {
                illuminance,
                shadow_maps_enabled: true,
                ..default()
            },
            // Explicit cascades: the default bounds come from the camera's
            // far plane, which is useless in a world this size.
            CascadeShadowConfigBuilder {
                num_cascades: 4,
                minimum_distance: 0.5,
                first_cascade_far_bound: 24.0,
                maximum_distance: 420.0,
                overlap_proportion: 0.2,
            }
            .build(),
            Transform::from_translation(Vec3::ZERO).looking_to(-sun_direction(&level), Vec3::Y),
        ));
    }
    commands.insert_resource(GlobalAmbientLight {
        color: host.ambient_color,
        brightness: host.ambient_brightness,
        ..default()
    });

    let start = std::env::var("VOXEL_START")
        .ok()
        .and_then(parse3)
        .unwrap_or(host.start);
    let look = std::env::var("VOXEL_LOOK")
        .ok()
        .and_then(parse3)
        .unwrap_or(host.look);
    let mut camera = commands.spawn((
        Camera3d::default(),
        Transform::from_translation(start).looking_at(start + look * 1000.0, up_for(look)),
        // The engine streams around whatever carries this.
        VoxelStreamSource,
        FreeCamera {
            walk_speed: host.walk_speed,
            run_speed: host.run_speed,
            ..default()
        },
    ));
    if let Some(fog) = host.fog.clone() {
        camera.insert(fog);
    }
}

#[derive(Component)]
struct LevelSun;

/// The demo's presentation, kept for reloads.
#[derive(Resource, Clone)]
struct HostScene(Scene);

/// The one host-visible thing a level reload can change: the sun
/// direction, which is engine data (the shadow bake uses it) but which
/// the host's own light has to follow.
fn follow_reloaded_sun(
    mut reloaded: MessageReader<LevelReloaded>,
    level: Res<LevelDef>,
    mut suns: Query<&mut Transform, With<LevelSun>>,
) {
    if reloaded.read().count() == 0 {
        return;
    }
    for mut t in &mut suns {
        *t = Transform::from_translation(Vec3::ZERO).looking_to(-sun_direction(&level), Vec3::Y);
    }
}

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
