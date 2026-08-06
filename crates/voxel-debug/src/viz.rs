//! Debug overlays: procgen layer visualization and chunk boundaries,
//! drawn as gizmo lines over the live world.
//!
//! - F8 toggles chunk boundaries (drawn LOD leaves near the camera,
//!   colored by level).
//! - F9 toggles planning layers: markers (by kind), clearance segments,
//!   ribbon segments, and a biome sample grid colored by dominant biome.

use bevy::prelude::*;

use voxel_engine::WorldQuery;

#[derive(Resource, Default)]
pub struct DebugViz {
    pub chunks: bool,
    pub layers: bool,
}

pub fn toggle_debug_viz(keys: Res<ButtonInput<KeyCode>>, mut viz: ResMut<DebugViz>) {
    if keys.just_pressed(KeyCode::F8) {
        viz.chunks = !viz.chunks;
        info!("debug viz: chunk boundaries {}", viz.chunks);
    }
    if keys.just_pressed(KeyCode::F9) {
        viz.layers = !viz.layers;
        info!("debug viz: planning layers {}", viz.layers);
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
const LAYER_VIZ_RADIUS_M: f32 = 512.0;

pub fn draw_debug_viz(
    viz: Res<DebugViz>,
    world: Res<WorldQuery>,
    level: Res<voxel_engine::level::LevelDef>,
    stats: Option<Res<voxel_render::SharedRenderStats>>,
    sources: voxel_engine::StreamSourceQuery,
    mut gizmos: Gizmos,
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

    if viz.layers {
        let c2 = Vec2::new(eye.x, eye.z);
        let (min, max) = (c2 - Vec2::splat(LAYER_VIZ_RADIUS_M), c2 + Vec2::splat(LAYER_VIZ_RADIUS_M));

        for m in world.markers_in(min, max, None) {
            let color = kind_color(&m.kind);
            gizmos.line(m.pos, m.pos + Vec3::Y * 30.0, color);
            gizmos.sphere(m.pos + Vec3::Y * 30.0, 2.0, color);
        }
        for seg in world.clearance_in(min, max) {
            let h = |p: Vec2| world.generator().height(p, 1.0) + 1.0;
            gizmos.line(
                Vec3::new(seg[0].x, h(seg[0]), seg[0].y),
                Vec3::new(seg[1].x, h(seg[1]), seg[1].y),
                Color::srgb(1.0, 0.8, 0.2),
            );
        }
        for w in world.ribbons_in(min, max) {
            gizmos.line(
                Vec3::new(w.a.x, w.levels[0] + 0.5, w.a.y),
                Vec3::new(w.b.x, w.levels[1] + 0.5, w.b.y),
                Color::srgb(0.2, 0.6, 1.0),
            );
        }

        // Biome sample grid: a stake per cell colored by dominant biome.
        for def in &level.stack {
            let voxel_engine::level::StackLayerDef::Biomes { name, table, .. } = def else {
                continue;
            };
            let step = LAYER_VIZ_RADIUS_M / 8.0;
            for gz in -8..=8 {
                for gx in -8..=8 {
                    let p = c2 + Vec2::new(gx as f32, gz as f32) * step;
                    let weights = world.biomes_at(name, p);
                    let Some((i, (biome, w))) = weights
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
                    else {
                        continue;
                    };
                    let _ = (biome, table);
                    let y = world.generator().height(p, 8.0) + 2.0;
                    let color = Color::hsl(i as f32 * 137.5 % 360.0, 0.8, 0.5);
                    gizmos.line(
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
            .add_systems(Update, (toggle_debug_viz, draw_debug_viz));
    }
}
