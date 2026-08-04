//! Megastructure demo: Blame!-style concrete interior world.
//!
//! Currently an M0 scaffold: window, flycam, HUD.

use bevy::prelude::*;
use voxel_debug::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "voxel2 — megastructure".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(VoxelDebugPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        FreeCamera::default(),
    ));

    commands.spawn((
        DirectionalLight::default(),
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.4, 0.0)),
    ));
}
