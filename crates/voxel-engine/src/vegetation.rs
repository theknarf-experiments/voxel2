//! Vegetation scatter: deterministic per-tile tree placement on the CPU
//! terrain-height mirror, drawn as auto-instanced procedural conifer meshes.
//!
//! Tiles are 64 m squares; each tile's trees derive from `chunk_seed`, so
//! the same forest always grows in the same place. (This migrates to a
//! LayerProcGen planning layer + GPU scatter later; the placement contract
//! stays the same.)

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use voxel_core::seed::{chunk_seed, Rng};
use voxel_worldgen::{terrain_height, terrain_up};

const TILE_M: f32 = 64.0;
/// Radius (in tiles) around the camera to populate.
const TILE_RADIUS: i32 = 6;
const TREE_ATTEMPTS: u32 = 18;
const WORLD_SEED: u64 = 0xF0857;
const VEG_LAYER_ID: u64 = 0x7EE5;

const GRASS_TILE_M: f32 = 16.0;
const GRASS_TILE_RADIUS: i32 = 7; // ~112 m of dense grass
const GRASS_PER_TILE: u32 = 550;

pub struct VegetationPlugin;

impl Plugin for VegetationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VegTiles>()
            .init_resource::<GrassTiles>()
            .add_plugins(voxel_render::GrassPlugin)
            .add_systems(Startup, build_tree_assets)
            .add_systems(Update, (stream_vegetation, stream_grass));
    }
}

#[derive(Resource, Default)]
struct GrassTiles {
    tiles: HashMap<IVec2, Vec<voxel_render::GrassInstance>>,
}

fn stream_grass(
    mut tiles: ResMut<GrassTiles>,
    instances: Res<voxel_render::GrassInstances>,
    cameras: Query<&Transform, With<Camera3d>>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let center = IVec2::new(
        (camera.translation.x / GRASS_TILE_M).floor() as i32,
        (camera.translation.z / GRASS_TILE_M).floor() as i32,
    );

    let mut changed = false;
    for dz in -GRASS_TILE_RADIUS..=GRASS_TILE_RADIUS {
        for dx in -GRASS_TILE_RADIUS..=GRASS_TILE_RADIUS {
            let tile = center + IVec2::new(dx, dz);
            if tiles.tiles.contains_key(&tile) {
                continue;
            }
            tiles.tiles.insert(tile, grass_tile(tile));
            changed = true;
        }
    }
    let keep = GRASS_TILE_RADIUS + 1;
    let before = tiles.tiles.len();
    tiles
        .tiles
        .retain(|t, _| (t.x - center.x).abs() <= keep && (t.y - center.y).abs() <= keep);
    changed |= tiles.tiles.len() != before;

    if changed {
        let merged: Vec<voxel_render::GrassInstance> =
            tiles.tiles.values().flatten().copied().collect();
        instances.set(merged);
    }
}

fn grass_tile(tile: IVec2) -> Vec<voxel_render::GrassInstance> {
    let mut rng = Rng::new(chunk_seed(
        WORLD_SEED,
        VEG_LAYER_ID ^ 0x6A55,
        IVec3::new(tile.x, 1, tile.y),
    ));
    let origin = Vec2::new(tile.x as f32, tile.y as f32) * GRASS_TILE_M;
    let mut out = Vec::new();
    for _ in 0..GRASS_PER_TILE {
        let x = origin.x + rng.next_f32() * GRASS_TILE_M;
        let z = origin.y + rng.next_f32() * GRASS_TILE_M;
        let xz = Vec2::new(x, z);
        let y = terrain_height(xz, 1.0);
        // Grass grows where the terrain shader paints grass.
        if !(2.5..300.0).contains(&y) || terrain_up(xz, 1.0) < 0.8 {
            continue;
        }
        // Top byte of the hash carries the baked sun-shadow factor.
        let shadow = voxel_worldgen::sun_shadow(Vec3::new(x, y, z));
        let hash = (rng.next_u64() as u32 & 0x00FF_FFFF) | (((shadow * 255.0) as u32) << 24);
        out.push(voxel_render::GrassInstance {
            pos: [x, y - 0.03, z],
            hash,
        });
    }
    out
}

#[derive(Resource, Default)]
struct VegTiles {
    tiles: HashMap<IVec2, Vec<Entity>>,
}

#[derive(Resource)]
struct TreeAssets {
    foliage_mesh: Handle<Mesh>,
    trunk_mesh: Handle<Mesh>,
    canopy_mesh: Handle<Mesh>,
    rock_mesh: Handle<Mesh>,
    foliage_mat: Handle<StandardMaterial>,
    trunk_mat: Handle<StandardMaterial>,
    canopy_mat: Handle<StandardMaterial>,
    rock_mat: Handle<StandardMaterial>,
}

/// Procedural low-poly conifer: an 8-sided trunk cylinder and three stacked
/// foliage cones, generated as two meshes so the two materials batch.
fn build_tree_assets(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let trunk = cylinder_mesh(0.14, 1.6, 8);
    let mut foliage = MeshBuilder::default();
    foliage.cone(Vec3::new(0.0, 1.0, 0.0), 1.5, 2.3, 9);
    foliage.cone(Vec3::new(0.0, 2.2, 0.0), 1.2, 2.0, 9);
    foliage.cone(Vec3::new(0.0, 3.3, 0.0), 0.85, 1.7, 8);
    foliage.cone(Vec3::new(0.0, 4.3, 0.0), 0.5, 1.2, 7);

    // Broadleaf canopy: a cluster of jittered blobs above the trunk.
    let mut canopy = MeshBuilder::default();
    canopy.blob(Vec3::new(0.0, 3.2, 0.0), 1.6, 0.12, 11);
    canopy.blob(Vec3::new(0.9, 2.7, 0.4), 1.1, 0.14, 23);
    canopy.blob(Vec3::new(-0.8, 2.8, -0.3), 1.0, 0.14, 47);

    // Boulder: heavily jittered squashed blob.
    let mut rock = MeshBuilder::default();
    rock.blob(Vec3::new(0.0, 0.25, 0.0), 1.0, 0.35, 5);

    commands.insert_resource(TreeAssets {
        trunk_mesh: meshes.add(trunk),
        foliage_mesh: meshes.add(foliage.build()),
        canopy_mesh: meshes.add(canopy.build()),
        rock_mesh: meshes.add(rock.build()),
        trunk_mat: materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.24, 0.15),
            perceptual_roughness: 0.95,
            ..default()
        }),
        foliage_mat: materials.add(StandardMaterial {
            base_color: Color::srgb(0.18, 0.38, 0.16),
            perceptual_roughness: 0.9,
            ..default()
        }),
        canopy_mat: materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.44, 0.16),
            perceptual_roughness: 0.9,
            ..default()
        }),
        rock_mat: materials.add(StandardMaterial {
            base_color: Color::srgb(0.44, 0.42, 0.40),
            perceptual_roughness: 0.95,
            ..default()
        }),
    });
}

#[derive(Default)]
struct MeshBuilder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    fn cone(&mut self, base: Vec3, radius: f32, height: f32, sides: u32) {
        let apex = base + Vec3::Y * height;
        let base_idx = self.positions.len() as u32;
        // Flat-ish shading: one ring of base vertices + apex per side pair.
        for i in 0..sides {
            let a0 = std::f32::consts::TAU * i as f32 / sides as f32;
            let a1 = std::f32::consts::TAU * (i + 1) as f32 / sides as f32;
            let p0 = base + Vec3::new(a0.cos() * radius, 0.0, a0.sin() * radius);
            let p1 = base + Vec3::new(a1.cos() * radius, 0.0, a1.sin() * radius);
            let n = (p1 - p0).cross(apex - p0).normalize();
            let n = [-n.x, -n.y, -n.z];
            let s = self.positions.len() as u32;
            self.positions.extend([p0.to_array(), p1.to_array(), apex.to_array()]);
            self.normals.extend([n, n, n]);
            self.indices.extend([s, s + 2, s + 1]);
        }
        let _ = base_idx;
    }

    /// Low-poly UV-sphere blob with per-vertex radial jitter (flat facets).
    fn blob(&mut self, center: Vec3, radius: f32, jitter: f32, seed: u32) {
        let segs = 9u32;
        let rings = 6u32;
        let mut ring_verts: Vec<Vec<Vec3>> = Vec::new();
        for r in 0..=rings {
            let phi = std::f32::consts::PI * r as f32 / rings as f32;
            let mut row = Vec::new();
            for s in 0..segs {
                let theta = std::f32::consts::TAU * s as f32 / segs as f32;
                let dir = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
                let h = {
                    let mut x = seed
                        .wrapping_mul(374_761_393)
                        .wrapping_add(r.wrapping_mul(668_265_263))
                        .wrapping_add(s.wrapping_mul(2_246_822_519));
                    x = (x ^ (x >> 13)).wrapping_mul(1_274_126_177);
                    ((x ^ (x >> 16)) & 0xFFFF) as f32 / 65535.0
                };
                let rr = radius * (1.0 + (h - 0.5) * 2.0 * jitter);
                row.push(center + dir * rr);
            }
            ring_verts.push(row);
        }
        for r in 0..rings {
            for s in 0..segs {
                let s1 = (s + 1) % segs;
                let quad = [
                    ring_verts[r as usize][s as usize],
                    ring_verts[r as usize][s1 as usize],
                    ring_verts[(r + 1) as usize][s1 as usize],
                    ring_verts[(r + 1) as usize][s as usize],
                ];
                let n = (quad[1] - quad[0]).cross(quad[3] - quad[0]).normalize_or_zero();
                let n = if n == Vec3::ZERO { Vec3::Y } else { n };
                let base = self.positions.len() as u32;
                for p in quad {
                    self.positions.push(p.to_array());
                    self.normals.push((-n).to_array());
                }
                self.indices
                    .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
    }

    fn build(self) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        mesh.insert_indices(Indices::U32(self.indices));
        mesh
    }
}

fn cylinder_mesh(radius: f32, height: f32, sides: u32) -> Mesh {
    let mut b = MeshBuilder::default();
    for i in 0..sides {
        let a0 = std::f32::consts::TAU * i as f32 / sides as f32;
        let a1 = std::f32::consts::TAU * (i + 1) as f32 / sides as f32;
        let n0 = Vec3::new(a0.cos(), 0.0, a0.sin());
        let n1 = Vec3::new(a1.cos(), 0.0, a1.sin());
        let s = b.positions.len() as u32;
        b.positions.extend([
            (n0 * radius).to_array(),
            (n1 * radius).to_array(),
            (n0 * radius + Vec3::Y * height).to_array(),
            (n1 * radius + Vec3::Y * height).to_array(),
        ]);
        b.normals.extend([n0.to_array(), n1.to_array(), n0.to_array(), n1.to_array()]);
        b.indices.extend([s, s + 2, s + 1, s + 1, s + 2, s + 3]);
    }
    b.build()
}

fn stream_vegetation(
    mut commands: Commands,
    mut tiles: ResMut<VegTiles>,
    assets: Option<Res<TreeAssets>>,
    cameras: Query<&Transform, With<Camera3d>>,
) {
    let (Some(assets), Ok(camera)) = (assets, cameras.single()) else {
        return;
    };
    let center = IVec2::new(
        (camera.translation.x / TILE_M).floor() as i32,
        (camera.translation.z / TILE_M).floor() as i32,
    );

    // Spawn new tiles in range.
    for dz in -TILE_RADIUS..=TILE_RADIUS {
        for dx in -TILE_RADIUS..=TILE_RADIUS {
            let tile = center + IVec2::new(dx, dz);
            if tiles.tiles.contains_key(&tile) {
                continue;
            }
            let entities = spawn_tile(&mut commands, &assets, tile);
            tiles.tiles.insert(tile, entities);
        }
    }

    // Despawn tiles out of range (hysteresis of 2 tiles).
    let keep = TILE_RADIUS + 2;
    let stale: Vec<IVec2> = tiles
        .tiles
        .keys()
        .filter(|t| (t.x - center.x).abs() > keep || (t.y - center.y).abs() > keep)
        .copied()
        .collect();
    for tile in stale {
        if let Some(entities) = tiles.tiles.remove(&tile) {
            for e in entities {
                commands.entity(e).despawn();
            }
        }
    }
}

fn spawn_tile(commands: &mut Commands, assets: &TreeAssets, tile: IVec2) -> Vec<Entity> {
    let mut rng = Rng::new(chunk_seed(
        WORLD_SEED,
        VEG_LAYER_ID,
        IVec3::new(tile.x, 0, tile.y),
    ));
    let origin = Vec2::new(tile.x as f32, tile.y as f32) * TILE_M;
    let mut entities = Vec::new();

    // Forest density gated by slow noise so woods come in coherent patches.
    let density = voxel_worldgen::forest_density(origin + Vec2::splat(TILE_M * 0.5));
    let attempts = (TREE_ATTEMPTS as f32 * density) as u32;

    for _ in 0..attempts {
        let x = origin.x + rng.next_f32() * TILE_M;
        let z = origin.y + rng.next_f32() * TILE_M;
        let xz = Vec2::new(x, z);
        let y = terrain_height(xz, 1.0);
        // Trees grow on gentle grassland: above the beach, below the rocks.
        if !(3.0..340.0).contains(&y) || terrain_up(xz, 1.0) < 0.86 {
            continue;
        }
        let yaw = rng.next_f32() * std::f32::consts::TAU;
        let scale = 0.8 + rng.next_f32() * 0.9;
        let transform = Transform::from_xyz(x, y - 0.15, z)
            .with_rotation(Quat::from_rotation_y(yaw))
            .with_scale(Vec3::splat(scale));
        // Pines dominate the highlands, broadleaves the lowlands.
        let pine = y > 140.0 || rng.next_f32() < 0.45;
        let (top_mesh, top_mat) = if pine {
            (&assets.foliage_mesh, &assets.foliage_mat)
        } else {
            (&assets.canopy_mesh, &assets.canopy_mat)
        };
        entities.push(
            commands
                .spawn((
                    Mesh3d(assets.trunk_mesh.clone()),
                    MeshMaterial3d(assets.trunk_mat.clone()),
                    transform,
                ))
                .id(),
        );
        entities.push(
            commands
                .spawn((
                    Mesh3d(top_mesh.clone()),
                    MeshMaterial3d(top_mat.clone()),
                    transform,
                ))
                .id(),
        );
    }

    // A few boulders per tile, preferring rougher ground; any altitude
    // below the snow line.
    for _ in 0..4 {
        let x = origin.x + rng.next_f32() * TILE_M;
        let z = origin.y + rng.next_f32() * TILE_M;
        let xz = Vec2::new(x, z);
        let y = terrain_height(xz, 1.0);
        let up = terrain_up(xz, 1.0);
        if !(2.0..800.0).contains(&y) || up < 0.55 || rng.next_f32() < 0.55 {
            continue;
        }
        let scale = 0.4 + rng.next_f32() * rng.next_f32() * 2.2;
        entities.push(
            commands
                .spawn((
                    Mesh3d(assets.rock_mesh.clone()),
                    MeshMaterial3d(assets.rock_mat.clone()),
                    Transform::from_xyz(x, y - 0.2 * scale, z)
                        .with_rotation(Quat::from_rotation_y(rng.next_f32() * 6.28))
                        .with_scale(Vec3::new(scale, scale * 0.75, scale)),
                ))
                .id(),
        );
    }
    entities
}
