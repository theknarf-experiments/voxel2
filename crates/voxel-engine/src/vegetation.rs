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

pub struct VegetationPlugin;

impl Plugin for VegetationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VegTiles>()
            .add_systems(Startup, build_tree_assets)
            .add_systems(Update, stream_vegetation);
    }
}

#[derive(Resource, Default)]
struct VegTiles {
    tiles: HashMap<IVec2, Vec<Entity>>,
}

#[derive(Resource)]
struct TreeAssets {
    foliage_mesh: Handle<Mesh>,
    trunk_mesh: Handle<Mesh>,
    foliage_mat: Handle<StandardMaterial>,
    trunk_mat: Handle<StandardMaterial>,
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
    foliage.cone(Vec3::new(0.0, 1.2, 0.0), 1.35, 2.2, 8);
    foliage.cone(Vec3::new(0.0, 2.4, 0.0), 1.05, 1.9, 8);
    foliage.cone(Vec3::new(0.0, 3.5, 0.0), 0.7, 1.6, 8);

    commands.insert_resource(TreeAssets {
        trunk_mesh: meshes.add(trunk),
        foliage_mesh: meshes.add(foliage.build()),
        trunk_mat: materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.24, 0.15),
            perceptual_roughness: 0.95,
            ..default()
        }),
        foliage_mat: materials.add(StandardMaterial {
            base_color: Color::srgb(0.20, 0.42, 0.18),
            perceptual_roughness: 0.9,
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
                    Mesh3d(assets.foliage_mesh.clone()),
                    MeshMaterial3d(assets.foliage_mat.clone()),
                    transform,
                ))
                .id(),
        );
    }
    entities
}
