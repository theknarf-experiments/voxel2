//! Debug tooling: flycam (re-exported Bevy `FreeCamera`), Bevy's own fps
//! overlay, and an on-screen HUD of the engine's telemetry.

use bevy::camera_controller::free_camera::FreeCameraPlugin;
use bevy::dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig};
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;

/// Leaves room for Bevy's fps overlay, which spawns at the top-left
/// corner and takes no position config. One text line plus a graph the
/// overlay hardcodes at 40 px tall.
const HUD_TOP_PX: i32 = 84;

pub use bevy::camera_controller::free_camera::FreeCamera;

pub mod remote;
pub mod viz;

/// HUD lines built from the engine's own telemetry (`StreamProbe`,
/// `SharedRenderStats`) — the engine measures, this crate displays.
fn engine_hud(
    probe: Option<Res<voxel_engine::streaming::StreamProbe>>,
    stats: Option<Res<voxel_render::SharedRenderStats>>,
    viz: Option<Res<viz::DebugViz>>,
    mut hud: ResMut<DebugHudExtra>,
) {
    let Some(probe) = probe else { return };
    if let Some(stats) = stats {
        if let Ok(s) = stats.0.lock() {
            hud.0.push(format!(
                "chunks: {} tracked | {} meshed | {} drawn | {} culled | {} pending",
                s.tracked, s.meshed, s.drawn, s.culled, s.awaiting
            ));
            // Pages, and the shape of what is holding them. A run
            // length is how many pages one chunk needed, so `runs`
            // reads "this many chunks wanted 1 page, this many 2".
            // Peaks accumulate as you fly: a reading taken standing
            // still is one sample of a process that depends on where
            // you are.
            let runs: Vec<String> = s
                .slab_peak_runs
                .iter()
                .enumerate()
                .filter(|(_, &n)| n > 0)
                .map(|(i, n)| format!("{}p x{n}", i + 1))
                .collect();
            hud.0.push(format!(
                "arena free: {} | slab: {}/{} pages (peak {}) | longest free run {}",
                s.arena_free,
                s.slab_used_pages,
                s.slab_total_pages,
                s.slab_peak_pages,
                s.slab_longest_free_run,
            ));
            hud.0.push(format!("slab peak shape: [{}]", runs.join(", ")));
        }
    }
    hud.0.push(format!(
        "resident: {}{} | planning reads missed: {}",
        probe.resident,
        if probe.stalled > 0 {
            format!(" | STALLED {}", probe.stalled)
        } else {
            String::new()
        },
        probe.reads_missed
    ));
    hud.0.push(format!(
        "settle: {} | last {:.1}s | worst {:.1}s",
        if probe.settled {
            "yes".to_string()
        } else {
            format!("{:.1}s...", probe.settling_s)
        },
        probe.last_settle_s,
        probe.worst_settle_s,
    ));
    if let Some(viz) = viz.filter(|v| v.layers()) {
        hud.0.push(format!(
            "planning viz: {} m | {} of {} lines{}",
            viz.layer_radius_m as i32,
            viz.drawn,
            viz.wanted,
            if viz.drawn < viz.wanted {
                " (TRUNCATED)"
            } else {
                ""
            },
        ));
    }
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
        app.add_plugins((
            FreeCameraPlugin,
            FrameTimeDiagnosticsPlugin::default(),
            // Bevy's own, for the frame-time GRAPH more than the number:
            // a single fps reading hides exactly what matters here, which
            // is whether a frame spiked. Settle transients, a cover sweep
            // landing, a slab refill — all of them are one tall bar and
            // an unchanged average.
            FpsOverlayPlugin {
                config: FpsOverlayConfig {
                    text_config: TextFont {
                        font_size: bevy::text::FontSize::Px(24.0),
                        ..default()
                    },
                    // The scale has to BRACKET what the demo actually
                    // runs at, 55-95 fps. `target_fps` is the FAST end and
                    // it bites hard: a frame quicker than
                    // `1/(target * 1.2)` clamps to zero height and simply
                    // is not drawn, so `target_fps: 60` renders an empty
                    // box whenever the demo manages 90 — which reads
                    // exactly like a broken graph. 120 is the machine's
                    // native refresh rate, which is what vsync caps at, so
                    // it is the fastest a frame can honestly be; 30 at the
                    // slow end leaves a spike somewhere to go.
                    frame_time_graph_config: FrameTimeGraphConfig {
                        enabled: true,
                        min_fps: 30.0,
                        target_fps: 120.0,
                    },
                    ..default()
                },
            },
        ))
            .add_systems(Startup, spawn_hud)
            .init_resource::<ScreenshotRequest>()
            .add_systems(Update, (engine_hud, update_hud, auto_screenshot, dump_camera).chain())
            .add_systems(Update, log_slow_frames);
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
pub struct ScreenshotRequest(pub Vec<ScreenshotWant>);

/// A queued capture: where to write it, and whether to grab the WINDOW
/// rather than the offscreen mirror.
///
/// The two are different render paths — the mirror is its own camera —
/// so a fault that only affects what the player sees is invisible in a
/// mirror capture. Worth being able to ask for the real thing, even
/// though it comes back black while the window is backgrounded.
pub struct ScreenshotWant {
    pub path: String,
    pub window: bool,
}

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
    for want in requests.0.drain(..) {
        let shot = if want.window {
            Screenshot::primary_window()
        } else {
            Screenshot::image(target.image.clone())
        };
        commands.spawn(shot).observe(save_to_disk(want.path));
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
            top: px(HUD_TOP_PX),
            left: px(8),
            ..default()
        },
        children![(DebugHudText, Text::new(""))],
    ));
}

fn update_hud(
    mut text_query: Query<&mut Text, With<DebugHudText>>,
    // The player's camera specifically: with a portal open there are
    // several, and `With<Camera3d>` reported whichever came first.
    camera_query: Query<&Transform, (With<Camera3d>, With<FreeCamera>)>,
    mut extra: ResMut<DebugHudExtra>,
) {
    let Ok(mut text) = text_query.single_mut() else {
        return;
    };
    let pos = camera_query
        .single()
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);

    // No fps line: Bevy's overlay is showing it directly above, larger,
    // with the graph. `voxctl status` reads the diagnostic itself.
    let mut out = format!("pos: {:.1} {:.1} {:.1}", pos.x, pos.y, pos.z);
    for line in extra.0.drain(..) {
        out.push('\n');
        out.push_str(&line);
    }
    text.0 = out;
}

/// Log any frame that misses the budget badly, with the streaming state
/// at the time.
///
/// A spike is invisible in an average and invisible in a screenshot; the
/// only way to attribute one is to print it beside what the engine was
/// doing that frame and correlate against the other logs by timestamp.
/// Off unless `VOXEL_LOG_SLOW` is set; its value is the threshold in
/// milliseconds, default 25.
fn log_slow_frames(
    time: Res<Time>,
    probe: Option<Res<voxel_engine::streaming::StreamProbe>>,
    stats: Option<Res<voxel_render::SharedRenderStats>>,
    mut threshold: Local<Option<f32>>,
) {
    let threshold = *threshold.get_or_insert_with(|| {
        std::env::var("VOXEL_LOG_SLOW")
            .ok()
            .map_or(f32::INFINITY, |v| v.parse().unwrap_or(25.0))
    });
    let ms = time.delta_secs() * 1000.0;
    if ms < threshold {
        return;
    }
    let (resident, generating) = probe.map_or((0, false), |p| (p.resident, p.generating));
    let (meshed, awaiting) = stats
        .and_then(|s| s.0.lock().ok().map(|s| (s.meshed, s.awaiting)))
        .unwrap_or((0, 0));
    warn!(
        "SLOW FRAME {ms:.1} ms | resident {resident} generating {generating} \
         meshed {meshed} awaiting {awaiting}"
    );
}
