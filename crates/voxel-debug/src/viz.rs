//! Debug overlays: procgen layer visualization and chunk boundaries,
//! drawn as gizmo lines over the live world.
//!
//! - F8 toggles chunk boundaries (drawn LOD leaves near the camera,
//!   colored by level).
//! - F9 cycles planning layers: off -> near -> far. Markers (by kind),
//!   clearance segments, ribbon segments, and a biome sample grid colored
//!   by dominant biome. "Far" is the whole streamed world, which is how
//!   you see what a coarse planning layer reaches; it costs a lot of
//!   gizmo lines, so the near range stays for close work.

use bevy::gizmos::config::{GizmoConfigGroup, GizmoConfigStore};
use bevy::prelude::*;

use voxel_engine::WorldQuery;

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

/// The two ranges F9 cycles through: close work, and the whole streamed
/// world.
const LAYER_VIZ_NEAR_M: f32 = 512.0;
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

/// Stable color per marker kind (hash → hue).
fn kind_color(kind: &str) -> Color {
    let mut h = 0u32;
    for b in kind.bytes() {
        h = h.wrapping_mul(31).wrapping_add(b as u32);
    }
    Color::hsl((h % 360) as f32, 0.9, 0.6)
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

pub fn draw_debug_viz(
    mut viz: ResMut<DebugViz>,
    world: Res<WorldQuery>,
    stats: Option<Res<voxel_render::SharedRenderStats>>,
    sources: voxel_engine::StreamSourceQuery,
    mut gizmos: Gizmos,
    mut planning: Gizmos<PlanningGizmos>,
) {
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

        for m in world.markers_in(min, max, None) {
            let color = kind_color(&m.kind);
            lines.push((m.pos, m.pos + Vec3::Y * 30.0, color));
        }
        let h = |p: Vec2| world.generator().height(p, 1.0) + 1.0;
        for seg in world.clearance_in(min, max) {
            lines.push((
                Vec3::new(seg[0].x, h(seg[0]), seg[0].y),
                Vec3::new(seg[1].x, h(seg[1]), seg[1].y),
                Color::srgb(1.0, 0.8, 0.2),
            ));
        }
        for w in world.ribbons_in(min, max) {
            lines.push((
                Vec3::new(w.a.x, w.levels[0] + 0.5, w.a.y),
                Vec3::new(w.b.x, w.levels[1] + 0.5, w.b.y),
                Color::srgb(0.2, 0.6, 1.0),
            ));
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

        // Biome sample grid: a stake per cell colored by dominant biome.
        for name in world.biome_fields() {
            // The biome grid stays a fixed 17x17 sample regardless of
            // range: it is a field readout, not a feature set, and one
            // stake per 20 km/8 would say nothing.
            let step = LAYER_VIZ_NEAR_M / 8.0;
            for gz in -8..=8 {
                for gx in -8..=8 {
                    let p = c2 + Vec2::new(gx as f32, gz as f32) * step;
                    let weights = world.biomes_at(&name, p);
                    let Some((i, (biome, w))) = weights
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
                    else {
                        continue;
                    };
                    let _ = biome;
                    let y = world.generator().height(p, 8.0) + 2.0;
                    let color = Color::hsl(i as f32 * 137.5 % 360.0, 0.8, 0.5);
                    planning.line(
                        Vec3::new(p.x, y, p.y),
                        Vec3::new(p.x, y + 4.0 + 12.0 * w, p.y),
                        color,
                    );
                }
            }
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
