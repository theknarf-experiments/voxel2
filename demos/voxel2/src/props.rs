//! Prop appearance — the HOST's half of scatter.
//!
//! The engine says *where* props go ([`ScatterInstance`] on streamed
//! entities); this file decides what they look like. Nothing here is in
//! a reusable crate: the models, the species names, the impostor
//! silhouettes and their colors are this demo's content, written in
//! Rust right here. A game would put its own GLTFs, materials and
//! gameplay components here instead — none of it belongs in the level
//! file, which describes only the world the engine generates.

use std::collections::HashMap;

use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::light::NotShadowCaster;
use bevy::math::DVec3;
use bevy::prelude::*;
use voxel_layers::{ChunkCtx, Dep, Layer, LayerChunk, LayerGraph, TopDep};

use crate::planning::world::WorldCtx;
use voxel_engine::scatter::ScatterInstance;
use voxel_engine::VoxelStreamSource;

/// Host-side appearance for one scatter class, indexed by the engine's
/// variant number.
#[derive(Clone, Debug, Default)]
pub struct PropClass {
    pub variants: Vec<PropVariant>,
    /// Draw a grounding blob shadow under each instance.
    pub blob_shadow: bool,
    /// Non-uniform squash applied on top of the engine's uniform scale.
    pub squash: Vec3,
}

#[derive(Clone, Debug)]
pub struct PropVariant {
    pub model: Model,
    pub trunk: Color,
    pub foliage: Color,
    /// Far-forest silhouette, when this class has one.
    pub impostor: Option<Impostor>,
}

/// The procedural models this demo builds. A real game names an asset
/// path (or a scene handle) instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Conifer,
    Broadleaf,
    Rock,
}

#[derive(Clone, Debug)]
pub struct Impostor {
    pub shape: ImpostorShape,
    pub color: [f32; 3],
    /// (half width, height) before the instance scale.
    pub size: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImpostorShape {
    Cone,
    Diamond,
}

/// The demo's props, keyed by the scatter class names in the level file.
/// Classes the level scatters but this table has no entry for simply
/// stream as bare entities.
#[derive(Resource, Default, Clone, Debug)]
pub struct PropTable(pub HashMap<String, PropClass>);

impl PropTable {
    /// Which props dress a level, keyed off its file name — the same way
    /// [`crate::scene_for`] picks its background. A real game has one
    /// table; this binary ships several demos.
    pub fn for_level(level_path: &std::path::Path) -> Self {
        match level_path.file_stem().and_then(|s| s.to_str()) {
            Some("purgatory") => Self::purgatory(),
            // The megastructure scatters nothing; an empty table is right.
            Some("megastructure") => Self::default(),
            _ => Self::planet(),
        }
    }

    /// Purgatory's litter: bone piles and scorched boulders.
    ///
    /// `bones` is `Model::Rock` in pale grey, squashed flat and scattered
    /// — an impression of a heap rather than modelled bones. At this art
    /// level a low pale clump reads as one, and a real bone model is a
    /// modelling job, not an engine one.
    fn purgatory() -> Self {
        let mut classes = HashMap::new();
        classes.insert(
            "bones".to_string(),
            PropClass {
                variants: vec![PropVariant {
                    model: Model::Rock,
                    trunk: Color::srgb(0.30, 0.28, 0.24),
                    foliage: Color::srgb(0.4520, 0.4310, 0.3800),
                    impostor: None,
                }],
                blob_shadow: false,
                squash: Vec3::new(1.5, 0.42, 1.5),
            },
        );
        classes.insert(
            "boulder".to_string(),
            PropClass {
                variants: vec![PropVariant {
                    model: Model::Rock,
                    trunk: Color::srgb(0.06, 0.05, 0.05),
                    foliage: Color::srgb(0.0605, 0.0512, 0.0470),
                    impostor: None,
                }],
                blob_shadow: false,
                squash: Vec3::new(1.0, 0.85, 1.0),
            },
        );
        Self(classes)
    }

    /// The planet demo's forest and boulders.
    fn planet() -> Self {
        let bark = Color::srgb(0.1462, 0.0916, 0.0469);
        let mut classes = HashMap::new();
        classes.insert(
            "tree".to_string(),
            PropClass {
                variants: vec![
                    PropVariant {
                        model: Model::Broadleaf,
                        trunk: bark,
                        foliage: Color::srgb(0.1214, 0.1910, 0.0518),
                        impostor: Some(Impostor {
                            shape: ImpostorShape::Diamond,
                            color: [0.16, 0.26, 0.08],
                            size: [2.3, 5.5],
                        }),
                    },
                    PropVariant {
                        model: Model::Conifer,
                        trunk: bark,
                        foliage: Color::srgb(0.0618, 0.1612, 0.0518),
                        impostor: Some(Impostor {
                            shape: ImpostorShape::Cone,
                            color: [0.1, 0.22, 0.09],
                            size: [1.7, 6.5],
                        }),
                    },
                ],
                // Real cascaded shadows now; no fake disc needed.
                blob_shadow: false,
                squash: Vec3::ONE,
            },
        );
        classes.insert(
            "boulder".to_string(),
            PropClass {
                variants: vec![PropVariant {
                    model: Model::Rock,
                    trunk: bark,
                    foliage: Color::srgb(0.1910, 0.1810, 0.1711),
                    impostor: None,
                }],
                blob_shadow: false,
                squash: Vec3::new(1.0, 0.75, 1.0),
            },
        );
        Self(classes)
    }
}

/// Far-forest tuning — the host's rendering choice, not the engine's.
const SUPER_M: f32 = 128.0;
/// How far the merged forest is visible. Formerly a tile radius plus a
/// keep-radius plus a per-frame budget; now the size of one top
/// dependency, and the tree placements underneath come with it.
const SUPER_VIEW_M: f32 = 3072.0;
const SUPER_HIDE_M: f32 = 320.0;
/// The scatter class the far forest renders silhouettes for.
const FOREST_CLASS: &str = "tree";

pub struct PropsPlugin;

impl Plugin for PropsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PropTable>()
            .init_resource::<PropAssets>()
            .add_systems(Startup, build_prop_assets)
            .add_systems(
                Update,
                (
                    dress_scatter,
                    reconcile_far_forest,
                    far_forest_visibility,
                ),
            );
    }
}

/// The mesh/material parts one variant is drawn from.
type VariantParts = Vec<(Handle<Mesh>, Handle<StandardMaterial>)>;

/// Meshes and materials per class variant, built once from [`PropTable`].
#[derive(Resource, Default)]
struct PropAssets {
    /// class -> variant -> parts.
    classes: HashMap<String, Vec<VariantParts>>,
    impostors: HashMap<String, Vec<Option<Impostor>>>,
    impostor_mat: Handle<StandardMaterial>,
    blob_mesh: Handle<Mesh>,
    blob_mat: Handle<StandardMaterial>,
}

fn build_prop_assets(
    table: Res<PropTable>,
    mut assets: ResMut<PropAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mat = |base_color: Color, rough: f32| StandardMaterial {
        base_color,
        perceptual_roughness: rough,
        ..default()
    };
    for (class, def) in &table.0 {
        let mut variants = Vec::new();
        let mut impostors = Vec::new();
        for variant in &def.variants {
            let trunk_mat = materials.add(mat(variant.trunk, 0.95));
            let foliage_mat = materials.add(mat(variant.foliage, 0.9));
            let mut parts = Vec::new();
            match variant.model {
                Model::Rock => {
                    let mut rock = MeshBuilder::default();
                    rock.blob(Vec3::ZERO, 1.0, 0.35, 7);
                    parts.push((
                        meshes.add(rock.build()),
                        materials.add(mat(variant.foliage, 0.85)),
                    ));
                }
                model => {
                    parts.push((meshes.add(cylinder_mesh(0.14, 1.6, 8)), trunk_mat));
                    let mut top = MeshBuilder::default();
                    if model == Model::Broadleaf {
                        top.blob(Vec3::new(0.0, 3.2, 0.0), 1.6, 0.12, 11);
                        top.blob(Vec3::new(0.9, 2.7, 0.4), 1.1, 0.14, 23);
                        top.blob(Vec3::new(-0.8, 2.8, -0.3), 1.0, 0.14, 47);
                    } else {
                        top.cone(Vec3::new(0.0, 1.0, 0.0), 1.5, 2.3, 9);
                        top.cone(Vec3::new(0.0, 2.2, 0.0), 1.2, 2.0, 9);
                        top.cone(Vec3::new(0.0, 3.3, 0.0), 0.85, 1.7, 8);
                        top.cone(Vec3::new(0.0, 4.3, 0.0), 0.5, 1.2, 7);
                    }
                    parts.push((meshes.add(top.build()), foliage_mat));
                }
            }
            variants.push(parts);
            impostors.push(variant.impostor.clone());
        }
        assets.classes.insert(class.clone(), variants);
        assets.impostors.insert(class.clone(), impostors);
    }
    assets.impostor_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        // Silhouettes bake their own sun shade into vertex colors.
        unlit: true,
        cull_mode: None,
        ..default()
    });
    assets.blob_mesh = meshes.add(bevy::math::primitives::Circle::new(1.0));
    assets.blob_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.05, 0.07, 0.04, 0.42),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
}

/// Give every newly streamed placement its look. This is the whole
/// host-side contract: react to `Added<ScatterInstance>`.
fn dress_scatter(
    mut commands: Commands,
    assets: Res<PropAssets>,
    table: Res<PropTable>,
    worlds: Res<voxel_engine::Worlds>,
    host: Res<crate::HostWorld>,
    new: Query<(Entity, &ScatterInstance, &Transform), Added<ScatterInstance>>,
) {
    let Some(query) = worlds.query(host.0) else {
        return;
    };
    // EVERY command here is fallible, because the entity may be gone by
    // the time they apply. `Added<ScatterInstance>` hands out entities
    // the engine's scatter layer owns, and a tile created and released
    // within a frame — which is what turning back across ground you just
    // crossed does — despawns them between the query and the flush.
    // `commands.entity(..).insert(..)` panics on that; `try_insert` and
    // `queue_silenced` drop the work for an instance that no longer
    // exists, which is exactly right: there is nothing left to dress.
    for (entity, instance, transform) in &new {
        let Some(variants) = assets.classes.get(&*instance.class) else {
            continue;
        };
        let Some(parts) = variants.get(instance.variant as usize) else {
            continue;
        };
        let class_def = table.0.get(&*instance.class).cloned().unwrap_or_default();
        let squash = class_def.squash.max(Vec3::splat(f32::EPSILON));
        commands.entity(entity).try_insert(Transform {
            scale: transform.scale * squash,
            ..*transform
        });
        for (mesh, material) in parts {
            let (mesh, material) = (mesh.clone(), material.clone());
            commands
                .entity(entity)
                .queue_silenced(move |mut instance: EntityWorldMut| {
                    instance.with_children(|parent| {
                        parent.spawn((Mesh3d(mesh), MeshMaterial3d(material)));
                    });
                });
        }
        if class_def.blob_shadow {
            // Grounding shadow, stretched along the sun and offset away
            // from it. A flat disc reads fine on gentle slopes.
            //
            // It hangs off the instance so the engine's tile eviction
            // takes it with the tree, which means expressing a
            // world-space pose in the parent's LOCAL space.
            let sun = query.generator().sun_direction();
            let sun_xz = Vec2::new(sun.x, sun.z).normalize_or(Vec2::X);
            let scale = instance.scale;
            let world = Transform::from_translation(
                transform.translation
                    + Vec3::new(-sun_xz.x, 0.0, -sun_xz.y) * 0.9 * scale
                    + Vec3::Y * 0.22,
            )
            .with_rotation(
                Quat::from_rotation_y(sun_xz.x.atan2(sun_xz.y))
                    * Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            )
            .with_scale(Vec3::new(1.5, 2.3, 1.0) * scale);
            let parent = Transform {
                scale: transform.scale * squash,
                ..*transform
            };
            let local = Transform::from_matrix(
                parent.to_matrix().inverse() * world.to_matrix(),
            );
            let (blob_mesh, blob_mat) = (assets.blob_mesh.clone(), assets.blob_mat.clone());
            commands
                .entity(entity)
                .queue_silenced(move |mut instance: EntityWorldMut| {
                    instance.with_children(|children| {
                        children.spawn((
                            Mesh3d(blob_mesh),
                            MeshMaterial3d(blob_mat),
                            local,
                        ));
                    });
                });
        }
    }
}

// --- far forest -------------------------------------------------------------
//
// Merged silhouette impostors out to ~3 km. The engine hands over the
// same placements the near meshes use (`tile_placements`), so trees
// never teleport across the detail boundary.

#[derive(Component)]
struct FarForestTile;

/// One impostor the far ring will draw: where it sits on the coarse
/// surface, how big, and how the horizon shadow falls on it.
#[derive(Clone)]
pub struct FarProp {
    pos: Vec3,
    scale: f32,
    variant: u32,
    shade: f32,
}

/// Seats the forest's placements on the surface coarse LOD actually shows
/// at that distance and samples the horizon shadow — the expensive part,
/// which belongs off the main thread. Assembling the merged mesh from the
/// result needs the impostor assets, so that stays in a system.
pub struct FarForest {
    source: String,
}

#[derive(Default)]
pub struct FarForestChunk;

impl Layer for FarForest {
    type Chunk = FarForestChunk;
    const NAME: &'static str = "far-forest";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(SUPER_M as f64, 0.0, SUPER_M as f64)
    }

    fn dependencies(&self, _level: u32) -> Vec<Dep> {
        vec![Dep::named(&self.source, IVec3::ZERO)]
    }
}

impl LayerChunk for FarForestChunk {
    type Layer = FarForest;

    fn create(&mut self, ctx: &ChunkCtx<'_, FarForest>, _level: u32) {
        let layer = ctx.layer();
        let generator = &ctx.context::<WorldCtx>().generator;
        let mut props = Vec::new();
        ctx.get_named::<crate::scatter::ScatterPopulation>(&layer.source, ctx.chunk_bounds())
            .for_each(|_, chunk| {
                for placement in &chunk.placements {
                    // Seat on the band-limited height coarse LOD shows at
                    // that distance, or they float.
                    let mut pos = placement.position;
                    pos.y = generator.height(Vec2::new(pos.x, pos.z), 16.0) - 0.15;
                    props.push(FarProp {
                        pos,
                        scale: placement.scale,
                        variant: placement.variant,
                        shade: 0.45 + 0.55 * generator.sun_shadow(pos),
                    });
                }
            });
        ctx.context::<WorldCtx>()
            .far_props
            .put(ctx.instance_key(), ctx.coord(), props);
    }

    fn destroy(&mut self, ctx: &ChunkCtx<'_, FarForest>, _level: u32) {
        ctx.context::<WorldCtx>()
            .far_props
            .take(ctx.instance_key(), ctx.coord());
    }
}

/// Register the far ring. Its top dependency is how far the forest is
/// visible; the tree placements underneath it come along because this
/// layer declares them, which is why nothing has to agree about reaches.
pub fn register_far_forest(graph: &mut LayerGraph) -> Option<TopDep> {
    if !graph.instances().iter().any(|n| n == FOREST_CLASS) {
        return None; // a level with no forest gets no far ring
    }
    graph.register(FarForest {
        source: FOREST_CLASS.to_string(),
    });
    Some(TopDep::at_level(
        FarForest::NAME,
        0,
        IVec3::new((2.0 * SUPER_VIEW_M) as i32, 0, (2.0 * SUPER_VIEW_M) as i32),
    ))
}

/// Build a merged mesh for every super-tile the layer published, and drop
/// the ones it withdrew.
fn reconcile_far_forest(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<PropAssets>,
    worlds: Res<voxel_engine::Worlds>,
    host: Res<crate::HostWorld>,
    mut spawned: Local<HashMap<crate::planning::world::PartKey, Option<Entity>>>,
    mut seen: Local<u64>,
) {
    let Some(sink) = worlds
        .query(host.0)
        .and_then(|w| w.host_ctx::<WorldCtx>())
        .map(|c| c.far_props.clone())
    else {
        return;
    };
    let sink = &sink;
    let generation = sink.generation();
    if generation == *seen {
        return;
    }
    *seen = generation;
    let live = sink.keys();
    spawned.retain(|part, entity| {
        if live.contains(part) {
            return true;
        }
        if let Some(entity) = entity.take() {
            commands.entity(entity).despawn();
        }
        false
    });
    for part in live {
        if spawned.contains_key(&part) {
            continue;
        }
        let Some(props) = sink.get(part) else { continue };
        spawned.insert(part, build_super_tile(&mut commands, &mut meshes, &assets, &props));
    }
}

/// Merge crossed-quad silhouettes for every placement in the super-tile.
fn build_super_tile(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    assets: &PropAssets,
    props: &[FarProp],
) -> Option<Entity> {
    let impostors = assets.impostors.get(FOREST_CLASS)?;
    let mut b = MeshBuilder::default();
    for prop in props {
        let Some(Some(imp)) = impostors.get(prop.variant as usize) else {
            continue;
        };
        let c = [
            imp.color[0] * prop.shade,
            imp.color[1] * prop.shade,
            imp.color[2] * prop.shade,
            1.0,
        ];
        let (hw, h) = (imp.size[0] * prop.scale, imp.size[1] * prop.scale);
        if imp.shape == ImpostorShape::Cone {
            b.cross_cone(prop.pos, hw, h, c);
        } else {
            b.cross_diamond(prop.pos, hw, h, c);
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
                // Silhouettes stand in for trees that are too far to see;
                // letting them into the shadow map would blanket the world.
                NotShadowCaster,
            ))
            .id(),
    )
}

/// Hide super-tiles inside the detailed radius so silhouettes don't poke
/// through the real canopies.
fn far_forest_visibility(
    mut tiles: Query<(&mut Visibility, &Mesh3d), With<FarForestTile>>,
    meshes: Res<Assets<Mesh>>,
    sources: Query<&GlobalTransform, With<VoxelStreamSource>>,
) {
    let Ok(source) = sources.single() else {
        return;
    };
    let camera = source.translation();
    let cam = Vec2::new(camera.x, camera.z);
    for (mut vis, mesh) in &mut tiles {
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
            bevy::asset::RenderAssetUsages::default(),
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
