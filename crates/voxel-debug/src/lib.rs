//! Debug tooling: flycam (re-exported Bevy `FreeCamera`) and an on-screen HUD
//! showing fps and camera position. Chunk/pool/slab stats get added here as
//! those systems come online.

use bevy::camera_controller::free_camera::FreeCameraPlugin;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

pub use bevy::camera_controller::free_camera::FreeCamera;

pub mod prelude {
    pub use crate::{DebugHudText, FreeCamera, VoxelDebugPlugin};
}

/// Adds the flycam controller and the debug HUD.
///
/// Demos spawn their own camera entity with a [`FreeCamera`] component; this
/// plugin provides the controller systems and the HUD overlay.
pub struct VoxelDebugPlugin;

impl Plugin for VoxelDebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((FreeCameraPlugin, FrameTimeDiagnosticsPlugin::default()))
            .add_systems(Startup, spawn_hud)
            .add_systems(Update, update_hud);
    }
}

/// Marker for the HUD text node. External systems can append their own lines
/// by pushing onto [`DebugHudExtra`].
#[derive(Component)]
pub struct DebugHudText;

/// Extra HUD lines contributed by other systems (cleared every frame after
/// display). Push formatted lines from anywhere.
#[derive(Resource, Default)]
pub struct DebugHudExtra(pub Vec<String>);

fn spawn_hud(mut commands: Commands) {
    commands.init_resource::<DebugHudExtra>();
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: px(8),
            left: px(8),
            ..default()
        },
        children![(DebugHudText, Text::new(""))],
    ));
}

fn update_hud(
    mut text_query: Query<&mut Text, With<DebugHudText>>,
    camera_query: Query<&Transform, With<Camera3d>>,
    diagnostics: Res<DiagnosticsStore>,
    mut extra: ResMut<DebugHudExtra>,
) {
    let Ok(mut text) = text_query.single_mut() else {
        return;
    };
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let pos = camera_query
        .single()
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);

    let mut out = format!("fps: {fps:.0}\npos: {:.1} {:.1} {:.1}", pos.x, pos.y, pos.z);
    for line in extra.0.drain(..) {
        out.push('\n');
        out.push_str(&line);
    }
    text.0 = out;
}
