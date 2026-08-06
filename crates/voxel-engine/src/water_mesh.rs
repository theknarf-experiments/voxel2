//! River water streaming: collects the planning stack's water segments
//! (`WorldQuery::water_in`) around the camera into the render world's
//! [`voxel_render::RiverWater`] buffer. The water pipeline draws them
//! with the ocean's shading — one water look per level; the SDF carries
//! only the carved bed.

use std::collections::HashMap;

use bevy::prelude::*;
use voxel_render::RiverSegGpu;

use crate::level::{eval_holes_mode, LevelDef, MaterialDef, WorldQuery};

const WATER_TILE_M: f32 = 256.0;
/// ~1.5 km of visible river surface around the camera.
const WATER_RADIUS: i32 = 6;

#[derive(Resource, Default)]
pub struct WaterMeshTiles {
    tiles: HashMap<IVec2, Vec<RiverSegGpu>>,
}

/// The level's color for a water material id (worlds are data — no
/// hardcoded river look).
fn water_color(level: &LevelDef, id: u32) -> [f32; 4] {
    for m in &level.materials {
        if let MaterialDef::Surface { id: mid, base, .. } = m {
            if *mid == id {
                return [base[0], base[1], base[2], 0.0];
            }
        }
    }
    [0.16, 0.34, 0.44, 0.0]
}

pub fn stream_water_meshes(
    probe: Res<crate::streaming::StreamProbe>,
    level: Res<LevelDef>,
    world: Res<WorldQuery>,
    mut tiles: ResMut<WaterMeshTiles>,
    mut rivers: ResMut<voxel_render::RiverWater>,
    cameras: Query<&Transform, (With<Camera3d>, Without<voxel_render::HelperCamera>)>,
) {
    if eval_holes_mode() || !probe.world_ready {
        return;
    }
    let Ok(camera) = cameras.single() else {
        return;
    };
    let center = IVec2::new(
        (camera.translation.x / WATER_TILE_M).floor() as i32,
        (camera.translation.z / WATER_TILE_M).floor() as i32,
    );
    let mut changed = false;
    // A couple of tiles per tick keeps the query cost off any one frame.
    let mut budget = 2;
    'outer: for dz in -WATER_RADIUS..=WATER_RADIUS {
        for dx in -WATER_RADIUS..=WATER_RADIUS {
            let tile = center + IVec2::new(dx, dz);
            if tiles.tiles.contains_key(&tile) {
                continue;
            }
            let origin = tile.as_vec2() * WATER_TILE_M;
            let segs: Vec<RiverSegGpu> = world
                .water_in(origin, origin + Vec2::splat(WATER_TILE_M))
                .iter()
                .map(|s| RiverSegGpu {
                    ab: [s.a.x, s.a.y, s.b.x, s.b.y],
                    geo: [s.half_w, s.levels[0], s.levels[1], 0.0],
                    color: water_color(&level, s.material),
                })
                .collect();
            changed |= !segs.is_empty();
            tiles.tiles.insert(tile, segs);
            budget -= 1;
            if budget == 0 {
                break 'outer;
            }
        }
    }
    let keep = WATER_RADIUS + 1;
    let stale: Vec<IVec2> = tiles
        .tiles
        .keys()
        .filter(|t| (**t - center).abs().max_element() > keep)
        .copied()
        .collect();
    for t in stale {
        if let Some(segs) = tiles.tiles.remove(&t) {
            changed |= !segs.is_empty();
        }
    }
    if changed {
        rivers.segments = tiles.tiles.values().flatten().copied().collect();
        rivers.generation += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_color_reads_the_material_table_with_fallback() {
        let mut level = LevelDef::from_json(include_str!("../../../levels/planet.json")).unwrap();
        // planet material 4 is the river surface color.
        let c = water_color(&level, 4);
        assert!(c[0] > 0.0 && c[2] > c[0], "unexpected river tint {c:?}");
        level.materials.clear();
        assert_eq!(water_color(&level, 4), [0.16, 0.34, 0.44, 0.0]);
    }
}
