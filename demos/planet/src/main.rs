//! Planet demo: planet-scale terrain with mountains, forests, and oceans.
//!
//! Currently M4: infinite FBM terrain streamed around a flycam, generated
//! and meshed entirely on the GPU.

use bevy::prelude::*;
use voxel_debug::prelude::*;
use voxel_engine::streaming::ChunkOpsProvider;
use voxel_engine::VoxelEnginePlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "voxel2 — planet".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.65, 0.77, 0.94)))
        // Planning-layer CSG: ruin sites + connecting roads, resolved
        // through the LayerProcGen stack and merged into fine-LOD chunks.
        .insert_resource(ChunkOpsProvider(Some({
            let layers = std::sync::Arc::new(voxel_worldgen::roads::planning_layers(0));
            std::sync::Arc::new(move |key: voxel_engine::ChunkKey| {
                if key.level > 2 {
                    return Vec::new(); // meter-scale features: fine LODs only
                }
                let min = key.min_corner_m().as_vec3();
                let max = min + Vec3::splat(key.edge_m() as f32);
                let mut ops = voxel_worldgen::ruins::ruins_ops(min, max);
                ops.extend(voxel_worldgen::roads::road_ops(&layers, min, max));
                ops
            })
        })))
        .add_plugins((
            VoxelDebugPlugin,
            VoxelEnginePlugin::default(),
            voxel_render::WaterPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (autopilot, walk_mode).chain())
        .run();
}

/// `VOXEL_WALK=1` glues the camera to the terrain surface at eye height —
/// on-foot exploration with the flycam's look/WASD controls.
fn walk_mode(mut cameras: Query<&mut Transform, With<Camera3d>>) {
    if std::env::var("VOXEL_WALK").is_err() {
        return;
    }
    for mut t in &mut cameras {
        let h = voxel_worldgen::terrain_height(
            bevy::math::Vec2::new(t.translation.x, t.translation.z),
            1.0,
        );
        t.translation.y = h + 1.75;
    }
}

/// Flies the camera forward at a constant speed when `VOXEL_AUTOPILOT` is
/// set (m/s, default 100) — used for streaming smoke tests and profiling.
fn autopilot(mut cameras: Query<&mut Transform, With<Camera3d>>, time: Res<Time>) {
    let Ok(speed) = std::env::var("VOXEL_AUTOPILOT") else {
        return;
    };
    let speed: f32 = speed.parse().unwrap_or(100.0);
    for mut transform in &mut cameras {
        let dir = transform.forward();
        transform.translation += dir * speed * time.delta_secs();
    }
}

fn setup(mut commands: Commands) {
    // Sun matching the terrain shader's sun_dir, plus sky ambient so
    // shadowed sides of props aren't black.
    commands.spawn((
        DirectionalLight {
            illuminance: 25_000.0,
            ..default()
        },
        Transform::from_translation(Vec3::ZERO)
            .looking_to(-Vec3::new(0.55, 0.5, 0.32).normalize(), Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.7, 0.8, 1.0),
        brightness: 400.0,
        ..default()
    });

    // Start position/orientation overridable for repeatable tests.
    let start = std::env::var("VOXEL_START")
        .ok()
        .and_then(|s| {
            let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
        })
        // Default: a scouted scenic forest valley (mountains and sea nearby).
        .unwrap_or(Vec3::new(-27570.0, 80.0, -36770.0));
    let look = std::env::var("VOXEL_LOOK")
        .ok()
        .and_then(|s| {
            let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
        })
        .unwrap_or(Vec3::new(0.4, -0.35, 0.4));
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(start).looking_at(start + look * 1000.0, Vec3::Y),
        FreeCamera {
            walk_speed: 60.0,
            run_speed: 600.0,
            ..default()
        },
    ));
}
