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

/// Which world a piece of the demo's scene content belongs to.
///
/// Scene content is not global: a portal shows another level's trees and
/// rocks, so each spawned thing carries the world it decorates and the
/// render layer that goes with it.
#[derive(Component, Clone, Copy)]
pub struct OfWorld(pub voxel_engine::WorldId);

impl OfWorld {
    /// Spawn scene content with this, never with `OfWorld` alone.
    ///
    /// Bevy does not propagate `RenderLayers` down `Children` — visibility
    /// reads the layer off each entity and falls back to layer 0 — so a
    /// child mesh spawned under world N's content draws on layer 0, which
    /// is WORLD 0's layer. That is the whole reason another level's trees
    /// and rocks appeared only in the launch level and never through the
    /// opening onto their own.
    pub fn scene(
        world: voxel_engine::WorldId,
    ) -> (Self, bevy::camera::visibility::RenderLayers) {
        (Self(world), voxel_render::world_layer(world))
    }
}

/// The props each loaded world scatters, keyed by world id — the
/// companion to [`WorldScenes`]. A tree is dressed by the level it grows
/// in, which is not necessarily the level you are standing in.
#[derive(Resource, Default)]
pub struct WorldProps(pub bevy::platform::collections::HashMap<voxel_engine::WorldId, props::PropTable>);

/// How each loaded world is dressed: background, sun, ambient, haze.
///
/// One place that answers "how does world W look", because five systems
/// need it and each one that reinvents the lookup is a chance for the
/// far side of a portal to be lit, fogged or coloured like the near one.
#[derive(Resource, Default)]
pub struct WorldScenes(pub bevy::platform::collections::HashMap<voxel_engine::WorldId, Scene>);

mod grass;
mod impostors;
mod planning;
mod portal;
mod scatter;
mod props;
mod ribbons;
mod surface_paint;
mod water;
use portal::{ExtraLevel, PortalPlugin};

/// The levels this binary ships. A real game has one world, or its own
/// list; this demo exists to show several loaded at once.
const SHIPPED_LEVELS: &[&str] = &[
    "levels/planet.json",
    "levels/megastructure.json",
    "levels/purgatory.json",
];
use props::PropsPlugin;
use grass::GrassPlugin;
use impostors::ImpostorPlugin;
use ribbons::RibbonsPlugin;
use surface_paint::SurfacePaintPlugin;
use water::WaterPlugin;

/// The demo's presentation for one world. A game would inline these
/// values wherever it spawns its camera and lights.
#[derive(Clone, Debug)]
pub struct Scene {
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
    /// Whether that light casts. A world lit from OUTSIDE wants shadows;
    /// a sunless interior standing a directional term in for the glow off
    /// its own lit ceilings does not — its cascades would end mid-hall
    /// and leave the near field dark against a bright far one.
    sun_shadows: bool,
    /// Host override for the light's direction, for worlds where the
    /// level's `sun_direction` is bake data about a sun nobody can see.
    ///
    /// An interior has no sun, but it cannot have NO directional term
    /// either: ambient light is the same from every angle, so a world lit
    /// only by ambient has no shading, no form, and no way to tell one
    /// grey from another. Its lamps are on the ceilings, so the one
    /// direction that is true here is down.
    sun_from: Option<Vec3>,
    ambient_color: Color,
    ambient_brightness: f32,
    /// Atmospheric haze. Voxel surfaces shade through Bevy's PBR, so this
    /// is an ordinary `DistanceFog` on the camera.
    fog: Option<DistanceFog>,
    /// Sea level, if this world has an ocean. Host data: the engine
    /// generates nothing against it.
    sea_level: Option<f32>,
}

/// Which scene to dress a world with. Keying off the level file's name
/// keeps both shipped demos in one binary; a real game has exactly one
/// scene and no match at all.
fn scene_for(level_path: &std::path::Path) -> Scene {
    match level_path.file_stem().and_then(|s| s.to_str()) {
        Some("purgatory") => Scene {
            // Ash haze, not sky. The sun is low and the air is thick with
            // whatever is still burning.
            clear_color: Color::srgb(0.115, 0.082, 0.079),
            start: Vec3::new(-5604.0, 112.0, 5660.0),
            look: Vec3::new(0.55, -0.20, -0.81),
            walk_speed: 60.0,
            run_speed: 900.0,
            // An overcast, ash-choked sky: a fifth of full daylight, and
            // a warm ambient doing most of the work, so the ground reads
            // without ever looking sunlit.
            sun_illuminance: Some(light_consts::lux::OVERCAST_DAY),
            sun_shadows: true,
            sun_from: None,
            ambient_color: Color::srgb(0.78, 0.52, 0.42),
            ambient_brightness: 2_600.0,
            fog: Some(DistanceFog {
                color: Color::srgb(0.135, 0.092, 0.086),
                directional_light_color: Color::srgb(1.0, 0.55, 0.30),
                directional_light_exponent: 26.0,
                falloff: FogFalloff::from_visibility_colors(
                    5_200.0,
                    Color::srgb(0.42, 0.20, 0.12),
                    Color::srgb(0.20, 0.13, 0.11),
                ),
            }),
            // Nothing here has an ocean. What pools in the low ground is
            // the lava the planning stack routes, and that is a ribbon.
            sea_level: None,
        },
        Some("megastructure") => Scene {
            clear_color: Color::srgb(0.035, 0.045, 0.06),
            start: Vec3::new(11.0, 12.0, 7.0),
            look: Vec3::new(0.7, 0.05, 0.7),
            walk_speed: 8.0,
            run_speed: 60.0,
            // The glow off its own lit ceilings, standing in for a sun it
            // does not have. Shadowless and from almost straight above:
            // floors catch it, ceilings fall away, walls land between, and
            // that gradient is the only thing giving a concrete interior
            // any form at all.
            //
            // It replaces an ambient-only setup at 3,800 that had no
            // directional term whatsoever. Nine deliberately different
            // concretes came out as one flat blue-grey wash, because a
            // bright uniform ambient both removes the shading AND lifts
            // every albedo toward the same tonemapped value.
            sun_illuminance: Some(2_100.0),
            sun_shadows: false,
            sun_from: Some(Vec3::new(0.18, 1.0, 0.3)),
            // Ambient is now only bounce: dim, and desaturated, so the
            // materials supply the colour instead of the light doing it.
            // Enough ambient to keep the VERTICALS readable: an overhead
            // fill only grazes a column, and a hall of columns lit by
            // nothing else is a black frame with a bright floor.
            ambient_color: Color::srgb(0.52, 0.57, 0.68),
            ambient_brightness: 1_050.0,
            // Haze, and load-bearing rather than decorative. The engine
            // hard-cuts disagreeing SDFs across LOD on purpose — blending
            // them makes phantom surfaces — and the design's answer to the
            // seam that leaves is to let fog cover it. This level had
            // `fog: None`, so every LOD transition showed as a torn edge
            // and the whole interior read as ripped paper rather than
            // architecture. An interior this size would be hazy anyway.
            fog: Some(DistanceFog {
                color: Color::srgb(0.052, 0.060, 0.075),
                directional_light_color: Color::srgb(0.55, 0.60, 0.72),
                directional_light_exponent: 12.0,
                falloff: FogFalloff::Exponential { density: 9.0e-4 },
            }),
            // A sunless interior: no sea.
            sea_level: None,
        },
        _ => Scene {
            clear_color: Color::srgb(0.65, 0.77, 0.94),
            // In the forest, looking along the mountain range. Found
            // with `voxctl inspect '{"kind":"regions", ...}'` rather than
            // guessed: the old spawn ended up INSIDE the range when the
            // range was added, which reads as a black screen.
            start: Vec3::new(-2332.0, 152.0, -38199.0),
            look: Vec3::new(0.7, -0.14, 0.7),
            walk_speed: 60.0,
            run_speed: 600.0,
            sun_illuminance: Some(light_consts::lux::FULL_DAYLIGHT),
            sun_shadows: true,
            sun_from: None,
            ambient_color: Color::srgb(0.7, 0.8, 1.0),
            ambient_brightness: 1_100.0,
            fog: Some(DistanceFog {
                color: Color::srgb(0.62, 0.72, 0.88),
                directional_light_color: Color::srgb(0.92, 0.85, 0.72),
                directional_light_exponent: 4.0,
                falloff: FogFalloff::Exponential { density: 6.0e-6 },
            }),
            sea_level: Some(0.0),
        },
    }
}

/// How a world is dressed for THIS run. Coverage eval renders geometry
/// only, so no world gets an ocean — a launch mode of the demo, not a
/// property of any level.
fn scene_for_run(level_path: &std::path::Path) -> Scene {
    let mut scene = scene_for(level_path);
    if std::env::var_os("VOXEL_EVAL_HOLES").is_some() {
        scene.sea_level = None;
    }
    scene
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
                // Frame time is the only usable signal for render work:
                // the display is 120 Hz, so with vsync on, fps quantizes
                // to 120/60/40 and a frame sitting near the 8.3 ms
                // boundary reads as either. Measure with this set.
                present_mode: if std::env::var_os("VOXEL_NO_VSYNC").is_some() {
                    bevy::window::PresentMode::AutoNoVsync
                } else {
                    bevy::window::PresentMode::default()
                },
                ..default()
            }),
            ..default()
        }))
        .insert_resource(HostScene(scene))
        .insert_resource(WorldProps(
            [(0, props::PropTable::for_level(std::path::Path::new(&path)))]
                .into_iter()
                .collect(),
        ))
        .insert_resource(WorldScenes(
            [(0, scene_for_run(std::path::Path::new(&path)))]
                .into_iter()
                .collect(),
        ))
        .add_plugins((
            VoxelDebugPlugin,
            VoxelVizPlugin,
            PropsPlugin,
            scatter::ScatterPlugin,
            // Water is the demo's: generic ribbon data in, water look out.
            RibbonsPlugin,
            SurfacePaintPlugin,
            WaterPlugin,
            // Ground cover: the engine scatters points, this draws blades.
            GrassPlugin,
            ImpostorPlugin,
            LevelPlugin {
                def: level,
                // A game picks this at new-game time and restores it
                // from its save; the demo just wants a stable world.
                seed: 0,
                source: Some(path.clone().into()),
                hole_eval,
                remote_port: None,
                // This demo authors its layers as JSON; a game with
                // hand-written layers passes its own factory here.
                planner: Some(std::sync::Arc::new(planning::StackPlanning)),
                // The mesh budget is the HOST's, because what a chunk
                // costs is a property of the levels it ships: this
                // demo's terrain is about a page each and its interior
                // level several. The default suits all three; a game
                // with denser geometry raises `total_pages` here rather
                // than editing the engine.
                slab: voxel_render::slab::SlabConfig::default(),
            },
        ));
    // What the portal keys open onto: every other shipped level, in key
    // order. A level is loaded the first time something asks for it and
    // stays loaded after — see `ExtraLevels`.
    let others: Vec<&str> = SHIPPED_LEVELS
        .iter()
        .copied()
        .filter(|p| !path.ends_with(p.trim_start_matches("levels/")))
        .collect();
    app.insert_resource(portal::ExtraLevels(
        others
            .iter()
            .map(|p| ExtraLevel {
                path: (*p).to_string(),
                loaded: None,
                scene: scene_for_run(std::path::Path::new(p)),
                world: None,
            })
            .collect(),
    ))
    .add_plugins(PortalPlugin);
    app
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (autopilot, sync_world_suns));

    // Live tooling for the voxctl CLI: always on in a dev build, never in
    // a release one. It used to need `VOXEL_REMOTE=1`, which meant every
    // check ran with a flag no ordinary launch had — and a system that
    // required a resource the remote plugin creates panicked on every
    // plain `cargo run` while passing all of them. A dev build should
    // behave the way it is actually driven.
    #[cfg(debug_assertions)]
    app.add_plugins(VoxelRemotePlugin {
        port: std::env::var("VOXCTL_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .filter(|p| *p > 1024)
            .unwrap_or(15702),
    });
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
fn setup_scene(mut commands: Commands, scene: Res<HostScene>) {
    let host = &scene.0;
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
        // World 0 to begin with; `follow_camera_world` moves it.
        voxel_render::world_layer(0),
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

/// A world's sun. One per world that has one — the megastructure is a
/// sunless interior and must not be lit by the planet's sun, nor take its
/// tree shadows on the concrete.
#[derive(Component)]
struct LevelSun(voxel_engine::WorldId);

/// The demo's presentation, kept for reloads.
#[derive(Resource, Clone)]
struct HostScene(Scene);

/// Give every loaded world its own sun, and keep each aimed where its
/// own level says.
///
/// A light belongs to a world exactly like grass does. One sun on every
/// world's layer meant the planet's sun lit the megastructure's interior
/// and — because the shadow map is built from whatever the light can see,
/// and worlds share coordinates — laid the planet's tree shadows across
/// its concrete.
///
/// Reload is the same operation: the level's sun direction is engine data
/// (the shadow bake uses it) and the host's light follows it.
fn sync_world_suns(
    mut commands: Commands,
    worlds: Res<voxel_engine::Worlds>,
    scenes: Res<WorldScenes>,
    mut reloaded: MessageReader<LevelReloaded>,
    mut suns: Query<(Entity, &LevelSun, &mut Transform)>,
) {
    let changed = worlds.is_changed() || scenes.is_changed() || reloaded.read().count() > 0;
    if !changed {
        return;
    }
    for world in worlds.iter() {
        let Some(host) = scenes.0.get(&world.id) else {
            continue;
        };
        let from = host
            .sun_from
            .map_or_else(|| sun_direction(&world.level), Vec3::normalize);
        let aim = Transform::from_translation(Vec3::ZERO).looking_to(-from, up_for(-from));
        match suns.iter_mut().find(|(_, sun, _)| sun.0 == world.id) {
            Some((entity, _, mut transform)) => {
                if host.sun_illuminance.is_none() {
                    commands.entity(entity).despawn();
                } else {
                    *transform = aim;
                }
            }
            None => {
                let Some(illuminance) = host.sun_illuminance else {
                    continue;
                };
                commands.spawn((
                    LevelSun(world.id),
                    // Its world's layer. A far view of this world is on
                    // that layer too, so it is lit by this sun; casters
                    // are on it and no other world's, so this sun's
                    // shadow map holds only its own world's trees.
                    voxel_render::world_layer(world.id),
                    DirectionalLight {
                        illuminance,
                        shadow_maps_enabled: host.sun_shadows,
                        ..default()
                    },
                    // Explicit cascades: the default bounds come from the
                    // camera's far plane, which is useless at this scale.
                    CascadeShadowConfigBuilder {
                        num_cascades: 4,
                        minimum_distance: 0.5,
                        first_cascade_far_bound: 24.0,
                        maximum_distance: 420.0,
                        overlap_proportion: 0.2,
                    }
                    .build(),
                    aim,
                ));
            }
        }
    }
}

/// `VOXEL_AUTOPILOT=<m/s>` flies the camera forward — the smoke-test and
/// coverage-eval driver.
///
/// `VOXEL_AUTOPILOT_WEAVE=<deg/s>` makes it turn as it goes, and that is
/// not a cosmetic difference. Flying straight moves a MONOTONE front
/// through the world: tiles are created ahead of you and released behind
/// you, and the two never happen to the same tile close together. Only
/// turning back across ground you just left creates and releases a tile
/// within a frame or two of each other, which is the window where a
/// system holding an entity from `Added<..>` finds it despawned by the
/// time its commands apply. A straight-line flight cannot catch that
/// class of bug; this is what a player wandering around does.
fn autopilot(mut cameras: Query<&mut Transform, With<VoxelStreamSource>>, time: Res<Time>) {
    let Ok(speed) = std::env::var("VOXEL_AUTOPILOT") else {
        return;
    };
    let speed: f32 = speed.parse().unwrap_or(50.0);
    let level_flight = std::env::var("VOXEL_AUTOPILOT_LEVEL").is_ok();
    let weave: f32 = std::env::var("VOXEL_AUTOPILOT_WEAVE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    for mut transform in &mut cameras {
        if weave != 0.0 {
            // Sinusoidal, so it doubles back rather than circling: a
            // steady turn is still a monotone front, just a curved one.
            let t = time.elapsed_secs();
            let yaw = weave.to_radians() * (t * 0.9).sin() * time.delta_secs() * 60.0;
            transform.rotate_y(yaw);
        }
        let mut dir = *transform.forward();
        if level_flight {
            dir.y = 0.0;
            dir = dir.normalize_or_zero();
        }
        transform.translation += dir * speed * time.delta_secs();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::camera::visibility::RenderLayers;

    /// The trap that hid another level's trees and rocks.
    ///
    /// Bevy reads `RenderLayers` off each entity and does NOT inherit it
    /// down `Children`, and its default is layer 0 — which is world 0's
    /// layer. So a child mesh spawned under world N's content is visible
    /// from the LAUNCH level and from nowhere else, including the world it
    /// belongs to. Nothing about that is visible at the spawn site, which
    /// is why it survived a working `OfWorld` on every parent.
    #[test]
    fn a_child_without_its_own_layer_belongs_to_world_zero() {
        let inherited_by_default = RenderLayers::default();
        assert!(inherited_by_default.intersects(&voxel_render::world_layer(0)));
        assert!(!inherited_by_default.intersects(&voxel_render::world_layer(1)));
    }

    /// `OfWorld::scene` is the fix: it hands out the marker and the layer
    /// together, so scene content cannot be spawned with one and not the
    /// other.
    #[test]
    fn scene_content_carries_its_own_worlds_layer() {
        for world in 0..3u8 {
            let (of_world, layers) = OfWorld::scene(world);
            assert_eq!(of_world.0, world);
            assert!(layers.intersects(&voxel_render::world_layer(world)));
            for other in (0..3u8).filter(|w| *w != world) {
                assert!(
                    !layers.intersects(&voxel_render::world_layer(other)),
                    "world {world}'s content must not be visible from world {other}"
                );
            }
        }
    }

    /// Coverage eval renders geometry only; a magenta background pixel
    /// below the horizon is a hole, and an ocean would cover them all.
    #[test]
    fn hole_eval_gives_no_world_an_ocean() {
        // SAFETY: single-threaded test, and the var is only read by
        // `scene_for_run` below.
        unsafe { std::env::set_var("VOXEL_EVAL_HOLES", "1") };
        let planet = scene_for_run(std::path::Path::new("levels/planet.json"));
        unsafe { std::env::remove_var("VOXEL_EVAL_HOLES") };
        assert!(
            scene_for(std::path::Path::new("levels/planet.json"))
                .sea_level
                .is_some(),
            "planet has a sea to suppress"
        );
        assert!(planet.sea_level.is_none());
    }
}
