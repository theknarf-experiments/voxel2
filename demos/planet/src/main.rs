//! Planet demo: planet-scale terrain with mountains, forests, and oceans.
//!
//! Currently an M0 scaffold: window, flycam, HUD, placeholder scene.

use bevy::prelude::*;
use voxel_debug::prelude::*;
use voxel_render::VoxelPrototypePlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "voxel2 — planet".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((VoxelDebugPlugin, VoxelPrototypePlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        // Aimed at the M2 prototype chunk (sphere centered at 16,16,16).
        Transform::from_xyz(16.0, 22.0, 65.0).looking_at(Vec3::splat(16.0), Vec3::Y),
        FreeCamera {
            walk_speed: 10.0,
            run_speed: 50.0,
            ..default()
        },
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.4, 0.0)),
    ));

    // Placeholder scene until real chunks exist: ground plane + reference cubes.
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(200.0)));
    let cube = meshes.add(Cuboid::new(2.0, 2.0, 2.0));
    let ground_mat = materials.add(Color::srgb(0.3, 0.5, 0.3));
    let cube_mat = materials.add(Color::srgb(0.6, 0.6, 0.65));

    commands.spawn((Mesh3d(ground), MeshMaterial3d(ground_mat)));
    for i in -3..=3 {
        for j in -3..=3 {
            commands.spawn((
                Mesh3d(cube.clone()),
                MeshMaterial3d(cube_mat.clone()),
                Transform::from_xyz(i as f32 * 20.0, 1.0, j as f32 * 20.0),
            ));
        }
    }
}
