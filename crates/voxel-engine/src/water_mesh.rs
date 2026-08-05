//! River surfaces: streamed strip meshes built from the planning stack's
//! water segments (`WorldQuery::water_in`). The SDF carries only the bed
//! notch; the visible water plane lives here, colored by the level's
//! material table — one data path from `flow` layer to pixels.

use std::collections::HashMap;

use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use voxel_worldgen::stack::WaterSeg;

use crate::level::{eval_holes_mode, LevelDef, MaterialDef, WorldQuery};

const WATER_TILE_M: f32 = 256.0;
/// ~1.5 km of visible river surface around the camera.
const WATER_RADIUS: i32 = 6;
/// Drawn a hair under the emitted level so the strip never peeks through
/// the notch rim the cut leaves.
const SURFACE_DROP_M: f32 = 0.05;

#[derive(Resource, Default)]
pub struct WaterMeshTiles {
    tiles: HashMap<IVec2, Entity>,
    materials: HashMap<u32, Handle<StandardMaterial>>,
}

/// One quad per segment, endpoints widened slightly along the flow so
/// joints between consecutive segments never open a sliver.
fn water_strip_mesh(segs: &[WaterSeg]) -> Option<Mesh> {
    if segs.is_empty() {
        return None;
    }
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(segs.len() * 4);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(segs.len() * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(segs.len() * 6);
    for seg in segs {
        let len = seg.a.distance(seg.b);
        if len < 0.01 {
            continue;
        }
        let dir = (seg.b - seg.a) / len;
        let perp = Vec2::new(-dir.y, dir.x) * seg.half_w;
        let a = seg.a - dir * 0.3;
        let b = seg.b + dir * 0.3;
        let base = positions.len() as u32;
        for (p, level) in [(a, seg.levels[0]), (b, seg.levels[1])] {
            let y = level - SURFACE_DROP_M;
            positions.push([p.x - perp.x, y, p.y - perp.y]);
            positions.push([p.x + perp.x, y, p.y + perp.y]);
            normals.push([0.0, 1.0, 0.0]);
            normals.push([0.0, 1.0, 0.0]);
        }
        indices.extend([base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
    if positions.is_empty() {
        return None;
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    Some(mesh)
}

/// The level's color for a water material id (worlds are data — no
/// hardcoded river look).
fn water_color(level: &LevelDef, id: u32) -> Color {
    for m in &level.materials {
        if let MaterialDef::Surface { id: mid, base, .. } = m {
            if *mid == id {
                return Color::srgb(base[0], base[1], base[2]);
            }
        }
    }
    Color::srgb(0.16, 0.34, 0.44)
}

pub fn stream_water_meshes(
    probe: Res<crate::streaming::StreamProbe>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    level: Res<LevelDef>,
    world: Res<WorldQuery>,
    mut tiles: ResMut<WaterMeshTiles>,
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
    // A couple of tiles per tick keeps the query cost off any one frame.
    let mut budget = 2;
    'outer: for dz in -WATER_RADIUS..=WATER_RADIUS {
        for dx in -WATER_RADIUS..=WATER_RADIUS {
            let tile = center + IVec2::new(dx, dz);
            if tiles.tiles.contains_key(&tile) {
                continue;
            }
            let origin = tile.as_vec2() * WATER_TILE_M;
            let segs = world.water_in(origin, origin + Vec2::splat(WATER_TILE_M));
            let entity = match water_strip_mesh(&segs) {
                Some(mesh) => {
                    let mat_id = segs[0].material;
                    let handle = tiles
                        .materials
                        .entry(mat_id)
                        .or_insert_with(|| {
                            // Unlit: the strip sits inside the carved
                            // notch where PBR lighting reads near-black;
                            // flat color matches the stylized world.
                            materials.add(StandardMaterial {
                                base_color: water_color(&level, mat_id),
                                unlit: true,
                                ..default()
                            })
                        })
                        .clone();
                    commands
                        .spawn((Mesh3d(meshes.add(mesh)), MeshMaterial3d(handle)))
                        .id()
                }
                // Empty tiles still get an entry so the query runs once.
                None => commands.spawn(Transform::default()).id(),
            };
            tiles.tiles.insert(tile, entity);
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
        if let Some(e) = tiles.tiles.remove(&t) {
            commands.entity(e).despawn();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_mesh_matches_segment_levels() {
        let segs = vec![
            WaterSeg {
                a: Vec2::new(0.0, 0.0),
                b: Vec2::new(8.0, 0.0),
                half_w: 3.0,
                levels: [10.0, 9.0],
                material: 4,
            },
            WaterSeg {
                a: Vec2::new(8.0, 0.0),
                b: Vec2::new(16.0, 4.0),
                half_w: 3.5,
                levels: [9.0, 9.0],
                material: 4,
            },
        ];
        let mesh = water_strip_mesh(&segs).unwrap();
        let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
        let positions = positions.as_float3().unwrap();
        assert_eq!(positions.len(), 8);
        // Each vertex sits just under its endpoint's water level.
        for (i, p) in positions.iter().enumerate() {
            let seg = &segs[i / 4];
            let level = seg.levels[(i / 2) % 2];
            assert!((p[1] - (level - SURFACE_DROP_M)).abs() < 1e-4);
        }
        // First quad spans the half width across the flow.
        assert!((positions[0][2] + 3.0).abs() < 1e-4);
        assert!((positions[1][2] - 3.0).abs() < 1e-4);
        // Triangles wind upward (a downward winding gets backface-culled
        // and the water silently vanishes).
        let idx: Vec<u32> = match mesh.indices().unwrap() {
            Indices::U32(v) => v.clone(),
            Indices::U16(v) => v.iter().map(|&i| i as u32).collect(),
        };
        for tri in idx.chunks(3) {
            let p = |i: u32| Vec3::from(positions[i as usize]);
            let n = (p(tri[1]) - p(tri[0])).cross(p(tri[2]) - p(tri[0]));
            assert!(n.y > 0.0, "downward-facing water triangle");
        }

        assert!(water_strip_mesh(&[]).is_none());
    }
}
