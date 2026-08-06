//! Ribbon streaming: collects the planning stack's ribbon surfaces
//! (`WorldQuery::ribbons_in`) around the camera into this demo's
//! [`RiverWater`] buffer, which its water pipeline draws. The engine says
//! only where the ribbons are and which material they carry; that they
//! are rivers, and that rivers look like water, is this game's choice.

use std::collections::HashMap;

use bevy::prelude::*;
use crate::water::{RiverSegGpu, RiverWater};

use voxel_engine::{level::{LevelDef, MaterialDef}, WorldQuery};

/// Used when a ribbon's material id is not in the level table.
const FALLBACK_TINT: [f32; 4] = [0.16, 0.34, 0.44, 0.0];

const RIBBON_TILE_M: f32 = 256.0;
/// ~1.5 km of visible river surface around the camera.
const RIBBON_RADIUS: i32 = 6;

// The engine ensures planning data over `RIBBON_QUERY_REACH_M`; keep this
// streamer inside that radius or it will read ungenerated regions.
const _: () = assert!(
    ((RIBBON_RADIUS + 1) as f32 * RIBBON_TILE_M) <= voxel_engine::level::RIBBON_QUERY_REACH_M,
);

#[derive(Resource, Default)]
pub struct RibbonTiles {
    tiles: HashMap<IVec2, Vec<RiverSegGpu>>,
}

/// The level's color for a ribbon material id.
fn ribbon_color(level: &LevelDef, id: u32) -> [f32; 4] {
    for m in &level.materials {
        if let MaterialDef::Surface { id: mid, base, .. } = m {
            if *mid == id {
                return [base[0], base[1], base[2], 0.0];
            }
        }
    }
    FALLBACK_TINT
}

fn stream_ribbons(
    probe: Res<voxel_engine::streaming::StreamProbe>,
    level: Res<LevelDef>,
    world: Res<WorldQuery>,
    mut tiles: ResMut<RibbonTiles>,
    mut rivers: ResMut<RiverWater>,
    sources: voxel_engine::StreamSourceQuery,
) {
    if !probe.world_ready {
        return;
    }
    let Ok(source) = sources.single() else {
        return; // no streaming source tagged yet
    };
    let camera = source.translation();
    let center = IVec2::new(
        (camera.x / RIBBON_TILE_M).floor() as i32,
        (camera.z / RIBBON_TILE_M).floor() as i32,
    );
    let mut changed = false;
    // A couple of tiles per tick keeps the query cost off any one frame.
    let mut budget = 2;
    'outer: for dz in -RIBBON_RADIUS..=RIBBON_RADIUS {
        for dx in -RIBBON_RADIUS..=RIBBON_RADIUS {
            let tile = center + IVec2::new(dx, dz);
            if tiles.tiles.contains_key(&tile) {
                continue;
            }
            let origin = tile.as_vec2() * RIBBON_TILE_M;
            let segs: Vec<RiverSegGpu> = world
                .ribbons_in(origin, origin + Vec2::splat(RIBBON_TILE_M))
                .iter()
                .map(|s| RiverSegGpu {
                    ab: [s.a.x, s.a.y, s.b.x, s.b.y],
                    geo: [s.half_w, s.levels[0], s.levels[1], 0.0],
                    color: ribbon_color(&level, s.material),
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
    let keep = RIBBON_RADIUS + 1;
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

/// Streams the level's ribbon surfaces into the demo's water buffer.
pub struct RibbonsPlugin;

impl Plugin for RibbonsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RibbonTiles>()
            .init_resource::<crate::water::RiverWater>()
            .add_systems(Update, (stream_ribbons, reset_on_reload));
    }
}

/// A reload can swap the whole world, so cached tiles must go with it.
fn reset_on_reload(
    mut reloaded: MessageReader<voxel_engine::level::LevelReloaded>,
    mut tiles: ResMut<RibbonTiles>,
    mut rivers: ResMut<crate::water::RiverWater>,
) {
    if reloaded.read().count() == 0 {
        return;
    }
    *tiles = RibbonTiles::default();
    rivers.segments.clear();
    rivers.generation += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_color_reads_the_material_table_with_fallback() {
        let mut level = LevelDef::from_json(include_str!("../../../levels/planet.json")).unwrap();
        // planet material 4 is the river surface color.
        let c = ribbon_color(&level, 4);
        assert!(c[0] > 0.0 && c[2] > c[0], "unexpected river tint {c:?}");
        level.materials.clear();
        assert_eq!(ribbon_color(&level, 4), FALLBACK_TINT);
    }
}
