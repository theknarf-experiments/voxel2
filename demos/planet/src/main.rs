//! Planet demo: planet-scale terrain with mountains, forests, and oceans.
//!
//! Currently M4: infinite FBM terrain streamed around a flycam, generated
//! and meshed entirely on the GPU.

use bevy::prelude::*;
use voxel_debug::prelude::*;
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
        .add_plugins((VoxelDebugPlugin, VoxelEnginePlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, autopilot)
        .run();
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
    // Start position/orientation overridable for repeatable tests.
    let start = std::env::var("VOXEL_START")
        .ok()
        .and_then(|s| {
            let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
        })
        .unwrap_or(Vec3::new(0.0, 9000.0, 0.0));
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(start)
            .looking_at(start + Vec3::new(0.4, -0.35, 0.4) * 1000.0, Vec3::Y),
        FreeCamera {
            walk_speed: 60.0,
            run_speed: 600.0,
            ..default()
        },
    ));
}
