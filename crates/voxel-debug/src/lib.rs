//! Debug tooling: flycam (re-exported Bevy `FreeCamera`) and an on-screen HUD
//! showing fps and camera position. Chunk/pool/slab stats get added here as
//! those systems come online.

use bevy::camera_controller::free_camera::FreeCameraPlugin;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

pub use bevy::camera_controller::free_camera::FreeCamera;

pub mod remote;
pub mod viz;

/// HUD lines built from the engine's own telemetry (`StreamProbe`,
/// `SharedRenderStats`) — the engine measures, this crate displays.
fn engine_hud(
    probe: Option<Res<voxel_engine::streaming::StreamProbe>>,
    stats: Option<Res<voxel_render::SharedRenderStats>>,
    mut hud: ResMut<DebugHudExtra>,
) {
    let Some(probe) = probe else { return };
    if let Some(stats) = stats {
        if let Ok(s) = stats.0.lock() {
            hud.0.push(format!(
                "chunks: {} tracked | {} meshed | {} drawn | {} culled | {} pending",
                s.tracked, s.meshed, s.drawn, s.culled, s.awaiting
            ));
            let occ: Vec<String> = s
                .slab_occupancy
                .iter()
                .map(|(free, total)| format!("{}/{}", total - free, total))
                .collect();
            hud.0.push(format!(
                "arena free: {} | slab used: [{}]",
                s.arena_free,
                occ.join(", ")
            ));
        }
    }
    hud.0.push(format!(
        "leaves: {} | planning cached reads: {}",
        probe.leaves, probe.read_generated
    ));
}

pub use remote::VoxelRemotePlugin;
pub use viz::{DebugViz, VoxelVizPlugin};

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
            .init_resource::<ScreenshotRequest>()
            .add_systems(Update, (engine_hud, update_hud, auto_screenshot, dump_camera).chain());
    }
}

/// The offscreen target and mirror camera for `VOXEL_SCREENSHOT`.
#[derive(Resource)]
struct ScreenshotTarget {
    image: Handle<Image>,
    camera: Entity,
}

/// One-shot screenshot requests (paths), served by the same offscreen
/// mirror as `VOXEL_SCREENSHOT` — remote tooling pushes here.
#[derive(Resource, Default)]
pub struct ScreenshotRequest(pub Vec<String>);

/// `VOXEL_SCREENSHOT=path[,interval_secs]`: periodically dump the rendered
/// frame to `path` (default every 10 s, overwriting). Renders through a
/// mirror camera into an offscreen image, so it works even when the window
/// is occluded (macOS gives occluded windows no drawable, which makes
/// window screenshots capture black).
#[allow(clippy::too_many_arguments)]
fn auto_screenshot(
    mut commands: Commands,
    time: Res<Time>,
    mut images: ResMut<Assets<Image>>,
    target: Option<Res<ScreenshotTarget>>,
    main_cam: Query<&Transform, (With<FreeCamera>, With<Camera3d>)>,
    mut mirror_cam: Query<&mut Transform, (Without<FreeCamera>, With<Camera3d>)>,
    mut requests: ResMut<ScreenshotRequest>,
    mut next_at: Local<Option<f32>>,
) {
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
    use bevy::render::view::window::screenshot::{save_to_disk, Screenshot};
    let spec = std::env::var("VOXEL_SCREENSHOT").ok();
    if spec.is_none() && requests.0.is_empty() && target.is_none() {
        return;
    }

    // Lazily create the offscreen target + mirror camera.
    let Some(target) = target else {
        let size = Extent3d {
            width: 1280,
            height: 720,
            depth_or_array_layers: 1,
        };
        let mut image = Image::new_fill(
            size,
            TextureDimension::D2,
            &[0, 0, 0, 255],
            TextureFormat::Rgba8UnormSrgb,
            bevy::asset::RenderAssetUsages::default(),
        );
        image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC
            | TextureUsages::RENDER_ATTACHMENT;
        let handle = images.add(image);
        let camera = commands
            .spawn((
                Camera3d::default(),
                bevy::camera::RenderTarget::Image(handle.clone().into()),
                voxel_render::HelperCamera,
                Transform::default(),
            ))
            .id();
        commands.insert_resource(ScreenshotTarget {
            image: handle,
            camera,
        });
        return;
    };

    // Mirror the flycam every frame.
    if let (Ok(main), Ok(mut mirror)) = (main_cam.single(), mirror_cam.get_mut(target.camera)) {
        *mirror = *main;
    }

    // One-shot requests from remote tooling.
    for path in requests.0.drain(..) {
        commands
            .spawn(Screenshot::image(target.image.clone()))
            .observe(save_to_disk(path));
    }

    // Periodic env-driven dump.
    let Some(spec) = spec else {
        return;
    };
    let (path, interval): (String, f32) = match spec.split_once(',') {
        Some((p, secs)) => (p.to_string(), secs.trim().parse().unwrap_or(10.0)),
        None => (spec, 10.0),
    };
    let now = time.elapsed_secs();
    let due = next_at.get_or_insert(interval.min(5.0));
    if now < *due {
        return;
    }
    *due = now + interval;
    commands
        .spawn(Screenshot::image(target.image.clone()))
        .observe(save_to_disk(path));
}

/// `P`: print the camera position and look direction to the log, as a
/// ready-to-paste `VOXEL_START`/`VOXEL_LOOK` pair.
fn dump_camera(
    keys: Res<ButtonInput<KeyCode>>,
    cameras: Query<&Transform, (With<Camera3d>, With<FreeCamera>)>,
) {
    if !keys.just_pressed(KeyCode::KeyP) {
        return;
    }
    for t in &cameras {
        let p = t.translation;
        let f = t.forward();
        info!(
            "camera: VOXEL_START=\"{:.1},{:.1},{:.1}\" VOXEL_LOOK=\"{:.3},{:.3},{:.3}\"",
            p.x, p.y, p.z, f.x, f.y, f.z
        );
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
