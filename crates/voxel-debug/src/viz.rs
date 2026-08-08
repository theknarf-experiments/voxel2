//! Debug overlays: procgen layer visualization and chunk boundaries,
//! drawn as gizmo lines over the live world.
//!
//! - F8 toggles chunk boundaries (drawn LOD leaves near the camera,
//!   colored by level).
//! - F9 cycles planning layers: off -> near -> far. Markers (by kind),
//!   clearance segments, ribbon segments, and a sample grid of each
//!   planner weight field, colored by its dominant member. "Far" is the
//!   whole streamed world, which is how
//!   you see what a coarse planning layer reaches; it costs a lot of
//!   gizmo lines, so the near range stays for close work.

use bevy::gizmos::config::{GizmoConfigGroup, GizmoConfigStore};
use bevy::prelude::*;


/// The planning overlay's own gizmo group, so it can draw IN FRONT of the
/// world while chunk boundaries stay depth-tested.
///
/// A planned feature is drawn at its full-detail ground height, but the
/// terrain 20 km away is a level-9 mesh whose surface differs from that
/// height by tens of meters — so depth-tested, the far half of the
/// network is buried inside the hill it runs over. Depth-testing an
/// X-ray view of the plan was never what was wanted anyway.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct PlanningGizmos;

#[derive(Resource, Default)]
pub struct DebugViz {
    pub chunks: bool,
    /// Half-extent of the planning query, in meters; 0 = off.
    pub layer_radius_m: f32,
    /// Lines drawn and lines wanted, last frame — a truncated overlay
    /// looks exactly like a world that stops, so it says which it is.
    pub drawn: usize,
    pub wanted: usize,
}

impl DebugViz {
    pub fn layers(&self) -> bool {
        self.layer_radius_m > 0.0
    }
}

/// The two ranges F9 cycles through: the near field, and the whole
/// streamed world.
pub const LAYER_VIZ_NEAR_M: f32 = 5_000.0;
const LAYER_VIZ_FAR_M: f32 = 40_000.0;

pub fn toggle_debug_viz(keys: Res<ButtonInput<KeyCode>>, mut viz: ResMut<DebugViz>) {
    if keys.just_pressed(KeyCode::F8) {
        viz.chunks = !viz.chunks;
        info!("debug viz: chunk boundaries {}", viz.chunks);
    }
    if keys.just_pressed(KeyCode::F9) {
        viz.layer_radius_m = match viz.layer_radius_m {
            r if r <= 0.0 => LAYER_VIZ_NEAR_M,
            r if r <= LAYER_VIZ_NEAR_M => LAYER_VIZ_FAR_M,
            _ => 0.0,
        };
        info!("debug viz: planning layers {} m", viz.layer_radius_m);
    }
}

const CHUNK_VIZ_RADIUS_M: f32 = 400.0;

/// Gizmo lines the planning overlay may draw in one frame.
///
/// Measured, at 4 km up over the planet: the whole resident planning set
/// is ~20k lines and saturates — 20 km and 40 km draw the same thing,
/// because that is all there is. So this is a runaway guard, not a
/// working limit, and it is far above what any measured vantage wants.
/// What it cannot draw is reported rather than silently missing: a
/// truncated overlay looks exactly like a world that stops, which is the
/// one thing this view exists to tell apart.
const LAYER_VIZ_MAX_LINES: usize = 200_000;

#[allow(clippy::too_many_arguments)]
pub fn draw_debug_viz(
    mut viz: ResMut<DebugViz>,
    worlds: Res<voxel_engine::Worlds>,
    camera_world: Res<voxel_render::CameraWorld>,
    stats: Option<Res<voxel_render::SharedRenderStats>>,
    sources: voxel_engine::StreamSourceQuery,
    mut gizmos: Gizmos,
    mut planning: Gizmos<PlanningGizmos>,
) {
    // The overlay describes the world you are standing in.
    let Some(world) = worlds.query(camera_world.0) else {
        return;
    };
    let Ok(source) = sources.single() else {
        return;
    };
    let eye = source.translation();

    if viz.chunks {
        if let Some(stats) = stats {
            let drawn: Vec<(voxel_core::ChunkKey, u32)> =
                stats.0.lock().unwrap().drawn_masks.clone();
            for (key, _) in drawn {
                let min = key.min_corner_m().as_vec3();
                let edge = key.edge_m() as f32;
                let center = min + Vec3::splat(edge * 0.5);
                if center.distance(eye) > CHUNK_VIZ_RADIUS_M + edge {
                    continue;
                }
                // Level → hue: fine chunks green, coarser toward red.
                let color = Color::hsl((110.0 - 20.0 * key.level as f32).max(0.0), 0.9, 0.55);
                gizmos.cube(
                    Transform::from_translation(center).with_scale(Vec3::splat(edge)),
                    color,
                );
            }
        }
    }

    if viz.layers() {
        // Introspection, not a working set: everything beyond what the
        // graph happens to hold is absent by design out here, and must
        // not count against `reads_missed`.
        let _peek = world.peek();
        let radius = viz.layer_radius_m;
        let c2 = Vec2::new(eye.x, eye.z);
        let (min, max) = (c2 - Vec2::splat(radius), c2 + Vec2::splat(radius));
        // Nearest-first, so a truncated overlay is the near field rather
        // than an arbitrary slice.
        let mut lines: Vec<(Vec3, Vec3, Color)> = Vec::new();

        // Whatever the host wants shown. The overlay owns the budget
        // and the ordering; what is worth a line is the planner's call.
        for l in world.debug_lines(min, max) {
            lines.push((l.a, l.b, Color::srgb(l.color[0], l.color[1], l.color[2])));
        }
        viz.wanted = lines.len();
        if lines.len() > LAYER_VIZ_MAX_LINES {
            lines.sort_by_cached_key(|(a, _, _)| (a.distance_squared(eye) * 0.01) as i64);
            lines.truncate(LAYER_VIZ_MAX_LINES);
        }
        viz.drawn = lines.len();
        for (a, b, color) in lines {
            planning.line(a, b, color);
        }
    }
}

/// Debug overlays (F8 chunk boundaries, F9 planning layers). Separate
/// from the engine: a game adds this only when it wants them.
pub struct VoxelVizPlugin;

impl Plugin for VoxelVizPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DebugViz>()
            .init_gizmo_group::<PlanningGizmos>()
            .add_systems(Startup, configure_planning_gizmos)
            .add_systems(Update, (toggle_debug_viz, draw_debug_viz));
    }
}

fn configure_planning_gizmos(mut store: ResMut<GizmoConfigStore>) {
    let (config, _) = store.config_mut::<PlanningGizmos>();
    config.depth_bias = -1.0;
}
