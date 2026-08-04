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
        // Level flight: follow the camera's heading but stay at altitude.
        let dir = transform.forward().with_y(0.0).normalize_or_zero();
        transform.translation += dir * speed * time.delta_secs();
    }
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 40.0, 0.0).looking_at(Vec3::new(60.0, 10.0, 60.0), Vec3::Y),
        FreeCamera {
            walk_speed: 15.0,
            run_speed: 80.0,
            ..default()
        },
    ));
}
