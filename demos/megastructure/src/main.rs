//! Megastructure demo: an endless Blame!-style concrete interior — floors,
//! pillars, walls, and vast vertical shafts, generated and meshed entirely
//! on the GPU by the same engine that renders the planet.

use bevy::prelude::*;
use voxel_debug::prelude::*;
use voxel_engine::{LodConfig, VoxelEnginePlugin, WorldKind};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "voxel2 — megastructure".into(),
                ..default()
            }),
            ..default()
        }))
        // Match the interior gloom in the concrete shader.
        .insert_resource(ClearColor(Color::srgb(0.035, 0.045, 0.06)))
        .insert_resource(LodConfig {
            // Interior scale, matched to the fog: past ~1 km nothing is
            // visible anyway, so keep the streamed world tight and refine
            // less eagerly than open terrain (interiors self-occlude).
            max_level: 4,
            top_radius: 2,
            top_y: (-3, 3),
            split_k: 1.6,
            merge_k: 2.1,
        })
        .add_plugins((
            VoxelDebugPlugin,
            VoxelEnginePlugin {
                world: WorldKind::Megastructure,
            },
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, autopilot)
        .run();
}

fn setup(mut commands: Commands) {
    let start = std::env::var("VOXEL_START")
        .ok()
        .and_then(|s| {
            let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
        })
        .unwrap_or(Vec3::new(11.0, 12.0, 7.0));
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(start)
            .looking_at(start + Vec3::new(0.7, 0.05, 0.7) * 100.0, Vec3::Y),
        FreeCamera {
            walk_speed: 8.0,
            run_speed: 60.0,
            ..default()
        },
    ));

    // Faint cold ambient so interiors aren't pitch black up close.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.6, 0.7, 0.9),
        brightness: 120.0,
        ..default()
    });
}

/// Flies the camera forward when `VOXEL_AUTOPILOT` is set (m/s).
fn autopilot(mut cameras: Query<&mut Transform, With<Camera3d>>, time: Res<Time>) {
    let Ok(speed) = std::env::var("VOXEL_AUTOPILOT") else {
        return;
    };
    let speed: f32 = speed.parse().unwrap_or(30.0);
    for mut transform in &mut cameras {
        let dir = transform.forward();
        transform.translation += dir * speed * time.delta_secs();
    }
}
