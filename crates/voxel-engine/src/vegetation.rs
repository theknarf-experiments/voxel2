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
const VEG_SEED_SALT: u64 = 0xF0857;

/// Vegetation seed: the engine salt mixed with the level seed (via the
/// installed generator program), so different levels grow different woods.
fn world_seed() -> u64 {
    VEG_SEED_SALT ^ (voxel_worldgen::program::seed() as u64)
}
const VEG_LAYER_ID: u64 = 0x7EE5;

const GRASS_TILE_M: f32 = 16.0;
const GRASS_TILE_RADIUS: i32 = 7; // ~112 m of dense grass

/// Far-forest impostors: merged silhouette meshes per 128 m super-tile.
const SUPER_M: f32 = 128.0;
const SUPER_RADIUS: i32 = 24; // ~3 km of visible forest
/// Super-tiles closer than this hide (detailed tree meshes take over).
const SUPER_HIDE_M: f32 = 320.0;
/// Super-tile builds per frame (amortize the height/shadow evaluation).
const SUPER_BUDGET: usize = 8;

/// Set to despawn and regrow all vegetation (terrain tuning changed).
#[derive(Resource, Default)]
pub struct VegetationRebuild(pub bool);

pub struct VegetationPlugin;

impl Plugin for VegetationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VegTiles>()
            .init_resource::<GrassTiles>()
            .init_resource::<FarForest>()
            .init_resource::<VegetationRebuild>()
            .add_plugins(voxel_render::GrassPlugin)
            .add_systems(
                Update,
                (
                    sync_spawner_assets,
                    rebuild_vegetation,
                    (
                        stream_vegetation,
                        stream_grass,
                        stream_far_forest,
                        far_forest_visibility,
                    )
                        .run_if(vegetation_enabled),
                )
                    .chain(),
            );
    }
}

/// Prop populations run only when the level declares spawners — a runtime
/// gate, so a hot-reload can switch worlds (rebuild still runs, so
/// removing the spawners clears what grew).
fn vegetation_enabled(level: Option<Res<crate::LevelDef>>) -> bool {
    !crate::level::eval_holes_mode() && level.is_some_and(|l| !l.spawners.is_empty())
}

/// Despawn everything grown; the streaming systems regrow it on the (new)
/// terrain over the following frames.
fn rebuild_vegetation(
    mut commands: Commands,
    mut flag: ResMut<VegetationRebuild>,
    mut veg: ResMut<VegTiles>,
    mut grass: ResMut<GrassTiles>,
    mut far: ResMut<FarForest>,
    instances: Res<voxel_render::GrassInstances>,
) {
    if !flag.0 {
        return;
    }
    flag.0 = false;
    for (_, entities) in veg.tiles.drain() {
        for e in entities {
            commands.entity(e).despawn();
        }
    }
    grass.tiles.clear();
    instances.set(Vec::new());
    for (_, entity) in far.tiles.drain() {
        if let Some(e) = entity {
            commands.entity(e).despawn();
        }
    }
}

// --- far-forest impostors ----------------------------------------------------

#[derive(Component)]
struct FarForestTile;

#[derive(Resource, Default)]
struct FarForest {
    /// Super-tile → impostor entity (None if the tile has no trees).
    tiles: HashMap<IVec2, Option<Entity>>,
}

fn stream_far_forest(
    mut commands: Commands,
    mut far: ResMut<FarForest>,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Option<Res<TreeAssets>>,
    level: Res<crate::LevelDef>,
    cuts: Res<crate::level::SurfaceCutsQuery>,
    cameras: Query<&Transform, (With<Camera3d>, Without<voxel_render::HelperCamera>)>,
) {
    let (Some(assets), Ok(camera)) = (assets, cameras.single()) else {
        return;
    };
    let Some(trees) = level.trees() else {
        return;
    };
    let center = IVec2::new(
        (camera.translation.x / SUPER_M).floor() as i32,
        (camera.translation.z / SUPER_M).floor() as i32,
    );

    // Build missing super-tiles nearest-first, a few per frame.
    let mut missing: Vec<(i32, IVec2)> = Vec::new();
    for dz in -SUPER_RADIUS..=SUPER_RADIUS {
        for dx in -SUPER_RADIUS..=SUPER_RADIUS {
            let tile = center + IVec2::new(dx, dz);
            if !far.tiles.contains_key(&tile) {
                missing.push((dx * dx + dz * dz, tile));
            }
        }
    }
    missing.sort_by_key(|(d, _)| *d);
    for (_, tile) in missing.into_iter().take(SUPER_BUDGET) {
        let entity = build_super_tile(&mut commands, &mut meshes, &assets, trees, &cuts, tile);
        far.tiles.insert(tile, entity);
    }

    // Despawn far out of range.
    let keep = SUPER_RADIUS + 2;
    let stale: Vec<IVec2> = far
        .tiles
        .keys()
        .filter(|t| (t.x - center.x).abs() > keep || (t.y - center.y).abs() > keep)
        .copied()
        .collect();
    for tile in stale {
        if let Some(Some(entity)) = far.tiles.remove(&tile) {
            commands.entity(entity).despawn();
        }
    }
}

/// Merge crossed-quad silhouettes for every tree in the super-tile's 2×2
/// detail tiles, colored by species and the baked sun shadow.
fn build_super_tile(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    assets: &TreeAssets,
    trees: &crate::level::TreesDef,
    cuts: &crate::level::SurfaceCutsQuery,
    tile: IVec2,
) -> Option<Entity> {
    let sub = (SUPER_M / TILE_M) as i32;
    let mut b = MeshBuilder::default();
    for dz in 0..sub {
        for dx in 0..sub {
            let detail = IVec2::new(tile.x * sub + dx, tile.y * sub + dz);
            for mut tree in tile_trees(detail, trees, cuts) {
                // Seat impostors on the band-limited height that coarse-LOD
                // terrain actually shows at their distance, not the full-
                // detail surface — otherwise they float over smoothed hills.
                tree.pos.y = terrain_height(Vec2::new(tree.pos.x, tree.pos.z), 16.0) - 0.15;
                let shade = 0.45 + 0.55 * voxel_worldgen::sun_shadow(tree.pos);
                let Some(sp) = assets.species.get(tree.species) else {
                    continue;
                };
                let ic = sp.impostor_color;
                let c = [ic[0] * shade, ic[1] * shade, ic[2] * shade, 1.0];
                let (hw, h) = (
                    sp.impostor_size[0] * tree.scale,
                    sp.impostor_size[1] * tree.scale,
                );
                if sp.impostor_cone {
                    b.cross_cone(tree.pos, hw, h, c);
                } else {
                    b.cross_diamond(tree.pos, hw, h, c);
                }
            }
        }
    }
    if b.positions.is_empty() {
        return None;
    }
    Some(
        commands
            .spawn((
                FarForestTile,
                Mesh3d(meshes.add(b.build())),
                MeshMaterial3d(assets.impostor_mat.clone()),
                Transform::default(),
            ))
            .id(),
    )
}

/// Hide super-tiles inside the detailed-tree radius so silhouettes don't
/// poke through the real canopies.
fn far_forest_visibility(
    mut tiles: Query<(&mut Visibility, &Mesh3d), With<FarForestTile>>,
    meshes: Res<Assets<Mesh>>,
    cameras: Query<&Transform, (With<Camera3d>, Without<voxel_render::HelperCamera>)>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let cam = Vec2::new(camera.translation.x, camera.translation.z);
    for (mut vis, mesh) in &mut tiles {
        // Cheap center estimate from the mesh's first vertex tile.
        let Some(mesh) = meshes.get(&mesh.0) else {
            continue;
        };
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(pos)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            continue;
        };
        let Some(first) = pos.first() else {
            continue;
        };
        let tile = Vec2::new(
            (first[0] / SUPER_M).floor() * SUPER_M + SUPER_M * 0.5,
            (first[2] / SUPER_M).floor() * SUPER_M + SUPER_M * 0.5,
        );
        let target = if cam.distance(tile) < SUPER_HIDE_M {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if *vis != target {
            *vis = target;
        }
    }
}

#[derive(Resource, Default)]
struct GrassTiles {
    tiles: HashMap<IVec2, Vec<voxel_render::GrassInstance>>,
    /// Tiles changed since the last merged-buffer rebuild.
    dirty: bool,
    last_merge: f32,
}

fn stream_grass(
    mut tiles: ResMut<GrassTiles>,
    instances: Res<voxel_render::GrassInstances>,
    level: Res<crate::LevelDef>,
    cuts: Res<crate::level::SurfaceCutsQuery>,
    time: Res<Time>,
    cameras: Query<&Transform, (With<Camera3d>, Without<voxel_render::HelperCamera>)>,
) {
    let (Some(grass), Ok(camera)) = (level.grass(), cameras.single()) else {
        return;
    };
    let t0 = std::time::Instant::now();
    let center = IVec2::new(
        (camera.translation.x / GRASS_TILE_M).floor() as i32,
        (camera.translation.z / GRASS_TILE_M).floor() as i32,
    );

    // One tile per frame: a tile costs hundreds of height evaluations and
    // a sun-shadow march per blade; at speed many tiles enter the radius
    // at once and bursting them is a spike frame.
    let mut budget = 1;
    let mut changed = false;
    'outer: for dz in -GRASS_TILE_RADIUS..=GRASS_TILE_RADIUS {
        for dx in -GRASS_TILE_RADIUS..=GRASS_TILE_RADIUS {
            let tile = center + IVec2::new(dx, dz);
            if tiles.tiles.contains_key(&tile) {
                continue;
            }
            if budget == 0 {
                break 'outer;
            }
            budget -= 1;
            tiles.tiles.insert(tile, grass_tile(tile, grass, &cuts));
            changed = true;
        }
    }
    let keep = GRASS_TILE_RADIUS + 1;
    let before = tiles.tiles.len();
    tiles
        .tiles
        .retain(|t, _| (t.x - center.x).abs() <= keep && (t.y - center.y).abs() <= keep);
    changed |= tiles.tiles.len() != before;

    // Rebuilding + re-uploading the merged instance buffer is the other
    // spike (hundreds of tiles copied); batch it to a few Hz — grass
    // popping in a few hundred ms late is invisible, a hitch is not.
    if changed {
        tiles.dirty = true;
    }
    if tiles.dirty && time.elapsed_secs() - tiles.last_merge > 0.25 {
        tiles.dirty = false;
        tiles.last_merge = time.elapsed_secs();
        let merged: Vec<voxel_render::GrassInstance> =
            tiles.tiles.values().flatten().copied().collect();
        instances.set(merged);
    }
    if std::env::var("VOXEL_LOG_FPS").is_ok() {
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        if ms > 4.0 {
            info!("stream_grass {ms:.1} ms");
        }
    }
}

/// Cut ops overlapping a square tile (huge y span: worm tunnels and
/// doorways at any depth). One provider query per tile; candidates then
/// test against the (typically tiny) list.
fn tile_cut_ops(
    cuts: &crate::level::SurfaceCutsQuery,
    origin: Vec2,
    size: f32,
) -> Vec<voxel_core::csg::CsgOp> {
    cuts.0.as_ref().map_or_else(Vec::new, |q| {
        q(
            Vec3::new(origin.x - 4.0, -10_000.0, origin.y - 4.0),
            Vec3::new(origin.x + size + 4.0, 10_000.0, origin.y + size + 4.0),
        )
    })
}

/// Was the heightfield surface at `p` carved away (cave mouth, doorway)?
/// Props must not seat on ground that is not actually there.
fn carved(cut_ops: &[voxel_core::csg::CsgOp], p: Vec3) -> bool {
    cut_ops.iter().any(|op| op.sdf(p) < 0.6)
}

/// Soft altitude-band gate: 1 inside, fading linearly to 0 across
/// `falloff` meters at each edge (0 falloff = hard band).
fn altitude_gate(alt: [f32; 2], falloff: f32, y: f32) -> f32 {
    if falloff <= 0.0 {
        return if (alt[0]..alt[1]).contains(&y) { 1.0 } else { 0.0 };
    }
    (((y - alt[0]) / falloff).clamp(0.0, 1.0)).min(((alt[1] - y) / falloff).clamp(0.0, 1.0))
}

/// Instance rotation from the spawner's placement rules: optional
/// align-to-surface-normal, random tilt cone, then yaw. Draws from `rng`
/// only when tilt is enabled, so legacy data keeps its exact layouts.
fn placement_rotation(
    rules: &crate::level::PlacementRulesDef,
    xz: Vec2,
    yaw: f32,
    rng: &mut Rng,
) -> Quat {
    let base = if rules.align == "normal" {
        Quat::from_rotation_arc(Vec3::Y, voxel_worldgen::terrain_normal(xz, 4.0))
    } else {
        Quat::IDENTITY
    };
    let tilt = if rules.tilt_deg > 0.0 {
        let dir = rng.next_f32() * std::f32::consts::TAU;
        let angle = rng.next_f32() * rules.tilt_deg.to_radians();
        Quat::from_axis_angle(Vec3::new(dir.cos(), 0.0, dir.sin()), angle)
    } else {
        Quat::IDENTITY
    };
    base * tilt * Quat::from_rotation_y(yaw)
}

/// Field-register density gate for spawner candidates (see `WOP_FIELD`).
fn field_gate(density: &Option<crate::level::FieldDensityDef>, xz: Vec2) -> f32 {
    density.as_ref().map_or(1.0, |d| {
        let f = voxel_worldgen::world_fields(xz)[(d.field as usize).min(3)];
        (f * d.scale + d.offset).clamp(0.0, 1.0)
    })
}

fn grass_tile(
    tile: IVec2,
    grass: &crate::level::GrassDef,
    cuts: &crate::level::SurfaceCutsQuery,
) -> Vec<voxel_render::GrassInstance> {
    let mut rng = Rng::new(chunk_seed(
        world_seed(),
        VEG_LAYER_ID ^ 0x6A55,
        IVec3::new(tile.x, 1, tile.y),
    ));
    let origin = Vec2::new(tile.x as f32, tile.y as f32) * GRASS_TILE_M;
    let cut_ops = tile_cut_ops(cuts, origin, GRASS_TILE_M);
    let mut out = Vec::new();
    for _ in 0..grass.per_tile {
        let x = origin.x + rng.next_f32() * GRASS_TILE_M;
        let z = origin.y + rng.next_f32() * GRASS_TILE_M;
        let xz = Vec2::new(x, z);
        if rng.next_f32() > field_gate(&grass.density, xz) {
            continue;
        }
        let y = terrain_height(xz, 1.0);
        let gate = altitude_gate(grass.altitude, grass.placement.altitude_falloff, y);
        if gate <= 0.0 || (gate < 1.0 && rng.next_f32() > gate) {
            continue;
        }
        let up = terrain_up(xz, 1.0);
        if up < grass.min_up || up > grass.placement.max_up {
            continue;
        }
        if carved(&cut_ops, Vec3::new(x, y, z)) {
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

/// Per-species meshes/materials + impostor style, built from the level's
/// tree spawner.
struct SpeciesAssets {
    parts: Vec<(Handle<Mesh>, Handle<StandardMaterial>)>,
    impostor_cone: bool,
    impostor_color: [f32; 3],
    impostor_size: [f32; 2],
}

#[derive(Resource)]
struct TreeAssets {
    species: Vec<SpeciesAssets>,
    rock_mesh: Handle<Mesh>,
    rock_mat: Handle<StandardMaterial>,
    impostor_mat: Handle<StandardMaterial>,
    blob_shadow_mesh: Handle<Mesh>,
    blob_shadow_mat: Handle<StandardMaterial>,
}

/// The spawner list the current [`TreeAssets`] were built from.
#[derive(Resource, Default)]
struct SpawnerStamp(Option<Vec<crate::level::SpawnerDef>>);

/// (Re)build prop assets whenever the level's spawners change; also
/// triggers a vegetation rebuild so everything regrows with the new looks.
fn sync_spawner_assets(
    mut commands: Commands,
    level: Option<Res<crate::LevelDef>>,
    mut stamp: Local<SpawnerStamp>,
    mut rebuild: ResMut<VegetationRebuild>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(level) = level else {
        return;
    };
    if stamp.0.as_deref() == Some(&level.spawners[..]) {
        return;
    }
    stamp.0 = Some(level.spawners.clone());
    rebuild.0 = true;

    let srgb = |c: [f32; 3], rough: f32| StandardMaterial {
        base_color: Color::srgb(c[0], c[1], c[2]),
        perceptual_roughness: rough,
        ..default()
    };
    let mut species = Vec::new();
    if let Some(trees) = level.trees() {
        for def in &trees.species {
            let trunk_mat = materials.add(srgb(def.trunk, 0.95));
            let foliage_mat = materials.add(srgb(def.foliage, 0.9));
            let trunk_mesh = meshes.add(cylinder_mesh(0.14, 1.6, 8));
            let mut top = MeshBuilder::default();
            if def.model == "broadleaf" {
                top.blob(Vec3::new(0.0, 3.2, 0.0), 1.6, 0.12, 11);
                top.blob(Vec3::new(0.9, 2.7, 0.4), 1.1, 0.14, 23);
                top.blob(Vec3::new(-0.8, 2.8, -0.3), 1.0, 0.14, 47);
            } else {
                top.cone(Vec3::new(0.0, 1.0, 0.0), 1.5, 2.3, 9);
                top.cone(Vec3::new(0.0, 2.2, 0.0), 1.2, 2.0, 9);
                top.cone(Vec3::new(0.0, 3.3, 0.0), 0.85, 1.7, 8);
                top.cone(Vec3::new(0.0, 4.3, 0.0), 0.5, 1.2, 7);
            }
            species.push(SpeciesAssets {
                parts: vec![
                    (trunk_mesh, trunk_mat),
                    (meshes.add(top.build()), foliage_mat),
                ],
                impostor_cone: def.impostor.shape != "diamond",
                impostor_color: def.impostor.color,
                impostor_size: def.impostor.size,
            });
        }
    }

    let mut rock = MeshBuilder::default();
    rock.blob(Vec3::new(0.0, 0.25, 0.0), 1.0, 0.35, 5);
    let rock_color = level.boulders().map(|b| b.color).unwrap_or(d_rock());

    commands.insert_resource(TreeAssets {
        species,
        rock_mesh: meshes.add(rock.build()),
        rock_mat: materials.add(srgb(rock_color, 0.95)),
        // Silhouette impostors: unlit + vertex colors (shadow baked in),
        // double-sided so the crossed quads read from every direction.
        impostor_mat: materials.add(StandardMaterial {
            base_color: Color::WHITE,
            unlit: true,
            cull_mode: None,
            ..default()
        }),
        blob_shadow_mesh: meshes.add(bevy::math::primitives::Circle::new(1.0)),
        blob_shadow_mat: materials.add(StandardMaterial {
            base_color: Color::srgba(0.05, 0.07, 0.04, 0.42),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            ..default()
        }),
    });
}

fn d_rock() -> [f32; 3] {
    [0.44, 0.42, 0.40]
}

#[derive(Default)]
struct MeshBuilder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
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
            self.positions
                .extend([p0.to_array(), p1.to_array(), apex.to_array()]);
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
                let n = (quad[1] - quad[0])
                    .cross(quad[3] - quad[0])
                    .normalize_or_zero();
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

    /// Two crossed triangles: a conifer silhouette.
    fn cross_cone(&mut self, at: Vec3, half_w: f32, height: f32, color: [f32; 4]) {
        for axis in 0..2 {
            let side = if axis == 0 {
                Vec3::new(half_w, 0.0, 0.0)
            } else {
                Vec3::new(0.0, 0.0, half_w)
            };
            let n = if axis == 0 { Vec3::Z } else { Vec3::X };
            let base = self.positions.len() as u32;
            self.positions.extend([
                (at - side).to_array(),
                (at + side).to_array(),
                (at + Vec3::Y * height).to_array(),
            ]);
            self.normals.extend([n.to_array(); 3]);
            self.colors.extend([color; 3]);
            self.indices.extend([base, base + 1, base + 2]);
        }
    }

    /// Two crossed diamonds: a broadleaf silhouette.
    fn cross_diamond(&mut self, at: Vec3, half_w: f32, height: f32, color: [f32; 4]) {
        let mid = at + Vec3::Y * (height * 0.55);
        for axis in 0..2 {
            let side = if axis == 0 {
                Vec3::new(half_w, 0.0, 0.0)
            } else {
                Vec3::new(0.0, 0.0, half_w)
            };
            let n = if axis == 0 { Vec3::Z } else { Vec3::X };
            let base = self.positions.len() as u32;
            self.positions.extend([
                at.to_array(),
                (mid - side).to_array(),
                (at + Vec3::Y * height).to_array(),
                (mid + side).to_array(),
            ]);
            self.normals.extend([n.to_array(); 4]);
            self.colors.extend([color; 4]);
            self.indices
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    fn build(self) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        if !self.colors.is_empty() {
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, self.colors);
        }
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
        b.normals
            .extend([n0.to_array(), n1.to_array(), n0.to_array(), n1.to_array()]);
        b.indices.extend([s, s + 2, s + 1, s + 1, s + 2, s + 3]);
    }
    b.build()
}

fn stream_vegetation(
    mut commands: Commands,
    mut tiles: ResMut<VegTiles>,
    assets: Option<Res<TreeAssets>>,
    level: Res<crate::LevelDef>,
    cuts: Res<crate::level::SurfaceCutsQuery>,
    cameras: Query<&Transform, (With<Camera3d>, Without<voxel_render::HelperCamera>)>,
) {
    let (Some(assets), Ok(camera)) = (assets, cameras.single()) else {
        return;
    };
    let t0 = std::time::Instant::now();
    let center = IVec2::new(
        (camera.translation.x / TILE_M).floor() as i32,
        (camera.translation.z / TILE_M).floor() as i32,
    );

    // Spawn new tiles in range, a few per frame (placement runs many
    // height/shadow evaluations per tile).
    let mut budget = 2;
    'outer: for dz in -TILE_RADIUS..=TILE_RADIUS {
        for dx in -TILE_RADIUS..=TILE_RADIUS {
            let tile = center + IVec2::new(dx, dz);
            if tiles.tiles.contains_key(&tile) {
                continue;
            }
            if budget == 0 {
                break 'outer;
            }
            budget -= 1;
            let entities = spawn_tile(&mut commands, &assets, &level, &cuts, tile);
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
    if std::env::var("VOXEL_LOG_FPS").is_ok() {
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        if ms > 4.0 {
            info!("stream_vegetation {ms:.1} ms");
        }
    }
}

/// One deterministic tree placement (shared by near meshes and far
/// impostors so trees never teleport across the detail boundary).
struct TreeInstance {
    pos: Vec3,
    /// Full placement rotation (align/tilt/yaw) for near meshes.
    rot: Quat,
    /// Yaw alone for the crossed-quad impostors.
    yaw: f32,
    scale: f32,
    species: usize,
}

fn tile_trees(
    tile: IVec2,
    trees: &crate::level::TreesDef,
    cuts: &crate::level::SurfaceCutsQuery,
) -> Vec<TreeInstance> {
    let mut rng = Rng::new(chunk_seed(
        world_seed(),
        VEG_LAYER_ID,
        IVec3::new(tile.x, 0, tile.y),
    ));
    let origin = Vec2::new(tile.x as f32, tile.y as f32) * TILE_M;
    let cut_ops = tile_cut_ops(cuts, origin, TILE_M);
    // Density gated by the spawner's patch noise so woods come in coherent
    // patches with clearings.
    let density = trees
        .patch
        .as_ref()
        .map(|p| {
            voxel_worldgen::patch_density(
                origin + Vec2::splat(TILE_M * 0.5),
                p.scale,
                Vec2::from(p.offset),
                p.contrast,
                p.bias,
            )
        })
        .unwrap_or(1.0);
    let attempts = (trees.max_per_tile as f32 * density) as u32;

    let mut out = Vec::new();
    for _ in 0..attempts {
        let x = origin.x + rng.next_f32() * TILE_M;
        let z = origin.y + rng.next_f32() * TILE_M;
        let xz = Vec2::new(x, z);
        // Seat on the band-limited surface mid-LOD terrain shows across the
        // detail radius (tiles spawn at the rim, where the ground is ~L2).
        if rng.next_f32() > field_gate(&trees.density, xz) {
            continue;
        }
        let y = terrain_height(xz, 4.0);
        if carved(&cut_ops, Vec3::new(x, y, z)) {
            continue;
        }
        let gate = altitude_gate(trees.altitude, trees.placement.altitude_falloff, y);
        if gate <= 0.0 || (gate < 1.0 && rng.next_f32() > gate) {
            continue;
        }
        let up = terrain_up(xz, 4.0);
        if up < trees.min_up || up > trees.placement.max_up {
            continue;
        }
        let yaw = rng.next_f32() * std::f32::consts::TAU;
        let roll = rng.next_f32();
        // Weighted pick among species whose altitude band contains y.
        let total: f32 = trees
            .species
            .iter()
            .filter(|sp| (sp.altitude[0]..sp.altitude[1]).contains(&y))
            .map(|sp| sp.weight)
            .sum();
        if total <= 0.0 {
            continue;
        }
        let mut pick = roll * total;
        let mut species = usize::MAX;
        for (i, sp) in trees.species.iter().enumerate() {
            if !(sp.altitude[0]..sp.altitude[1]).contains(&y) {
                continue;
            }
            if pick < sp.weight {
                species = i;
                break;
            }
            pick -= sp.weight;
        }
        if species == usize::MAX {
            continue;
        }
        let sr = trees.species[species].scale;
        let scale = sr[0] + rng.next_f32() * (sr[1] - sr[0]);
        let sink = trees.placement.sink.unwrap_or(0.45);
        out.push(TreeInstance {
            pos: Vec3::new(x, y - sink, z),
            rot: placement_rotation(&trees.placement, xz, yaw, &mut rng),
            yaw,
            scale,
            species,
        });
    }
    out
}

fn spawn_tile(
    commands: &mut Commands,
    assets: &TreeAssets,
    level: &crate::LevelDef,
    cuts: &crate::level::SurfaceCutsQuery,
    tile: IVec2,
) -> Vec<Entity> {
    let mut entities = Vec::new();
    let trees = level.trees();
    for tree in trees.map(|t| tile_trees(tile, t, cuts)).unwrap_or_default() {
        let transform = Transform::from_translation(tree.pos)
            .with_rotation(tree.rot)
            .with_scale(Vec3::splat(tree.scale));
        let Some(sp) = assets.species.get(tree.species) else {
            continue;
        };
        for (mesh, mat) in &sp.parts {
            entities.push(
                commands
                    .spawn((Mesh3d(mesh.clone()), MeshMaterial3d(mat.clone()), transform))
                    .id(),
            );
        }
        // Grounding blob shadow, stretched along the sun direction and
        // offset opposite it. Terrain-conforming shadows are future work;
        // on gentle tree-bearing slopes a flat disc reads fine.
        let sun = voxel_worldgen::program::sun_direction();
        let sun_xz = Vec2::new(sun.x, sun.z).normalize_or(Vec2::X);
        entities.push(
            commands
                .spawn((
                    Mesh3d(assets.blob_shadow_mesh.clone()),
                    MeshMaterial3d(assets.blob_shadow_mat.clone()),
                    Transform::from_translation(
                        tree.pos
                            + Vec3::new(-sun_xz.x, 0.0, -sun_xz.y) * 0.9 * tree.scale
                            + Vec3::Y * 0.22,
                    )
                    .with_rotation(
                        Quat::from_rotation_y(sun_xz.x.atan2(sun_xz.y))
                            * Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                    )
                    .with_scale(Vec3::new(1.5, 2.3, 1.0) * tree.scale),
                ))
                .id(),
        );
    }

    // Boulders per the level's boulder spawner.
    let Some(b) = level.boulders() else {
        return entities;
    };
    let boulder_origin = Vec2::new(tile.x as f32, tile.y as f32) * TILE_M;
    let boulder_cuts = tile_cut_ops(cuts, boulder_origin, TILE_M);
    let mut rng = Rng::new(chunk_seed(
        world_seed(),
        VEG_LAYER_ID ^ 0x0C4,
        IVec3::new(tile.x, 2, tile.y),
    ));
    let origin = Vec2::new(tile.x as f32, tile.y as f32) * TILE_M;
    for _ in 0..b.per_tile {
        let x = origin.x + rng.next_f32() * TILE_M;
        let z = origin.y + rng.next_f32() * TILE_M;
        let xz = Vec2::new(x, z);
        if rng.next_f32() > field_gate(&b.density, xz) {
            continue;
        }
        let y = terrain_height(xz, 4.0);
        if carved(&boulder_cuts, Vec3::new(x, y, z)) {
            continue;
        }
        let gate = altitude_gate(b.altitude, b.placement.altitude_falloff, y);
        if gate <= 0.0 || (gate < 1.0 && rng.next_f32() > gate) {
            continue;
        }
        let up = terrain_up(xz, 4.0);
        if up < b.min_up || up > b.placement.max_up || rng.next_f32() >= b.chance {
            continue;
        }
        let scale = b.scale[0] + rng.next_f32() * rng.next_f32() * (b.scale[1] - b.scale[0]);
        let sink = b.placement.sink.unwrap_or(0.2 * scale);
        let yaw = rng.next_f32() * std::f32::consts::TAU;
        entities.push(
            commands
                .spawn((
                    Mesh3d(assets.rock_mesh.clone()),
                    MeshMaterial3d(assets.rock_mat.clone()),
                    Transform::from_xyz(x, y - sink, z)
                        .with_rotation(placement_rotation(&b.placement, xz, yaw, &mut rng))
                        .with_scale(Vec3::new(scale, scale * 0.75, scale)),
                ))
                .id(),
        );
    }
    entities
}
