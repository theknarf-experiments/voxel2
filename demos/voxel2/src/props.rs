//! Prop appearance — the HOST's half of scatter.
//!
//! The engine says *where* props go ([`ScatterInstance`] on streamed
//! entities); this file decides what they look like. Nothing here is in
//! a reusable crate: the models, the species names and their colors are
//! this demo's content, written in
//! Rust right here. A game would put its own GLTFs, materials and
//! gameplay components here instead — none of it belongs in the level
//! file, which describes only the world the engine generates.

use std::collections::HashMap;

use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use voxel_engine::scatter::ScatterInstance;

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
}

/// The procedural models this demo builds. A real game names an asset
/// path (or a scene handle) instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Conifer,
    Broadleaf,
    /// A third canopy: narrow, pale-trunked, taller than the others.
    Birch,
    Rock,
    /// Low woody clump — no trunk worth drawing at this scale.
    Bush,
    /// Stem plus a wide squashed cap, in two colours.
    Mushroom,
    /// Ribbed column with a pair of raised arms.
    Cactus,
    /// A fallen trunk lying along the ground.
    Log,
    /// A tuft of tall thin blades, for standing water.
    Reed,
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
            Some("megastructure") => Self::megastructure(),
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
                }],
                blob_shadow: false,
                squash: Vec3::new(1.0, 0.85, 1.0),
            },
        );
        Self(classes)
    }

    /// The megastructure's floor litter.
    ///
    /// Nothing grew here: these sit on the storeys the level's slab
    /// lattice builds, found by marching the program rather than by
    /// asking a heightfield that an interior does not have. The models are
    /// the ones the demo already has, read differently — a flattened
    /// `Rock` is a heap of spalled concrete, and a `Log` on a floor is a
    /// dropped conduit.
    fn megastructure() -> Self {
        let mut classes = HashMap::new();
        classes.insert(
            "rubble".to_string(),
            PropClass {
                variants: vec![PropVariant {
                    model: Model::Rock,
                    trunk: Color::srgb(0.1100, 0.1080, 0.1050),
                    foliage: Color::srgb(0.1500, 0.1470, 0.1420),
                }],
                blob_shadow: false,
                squash: Vec3::new(1.4, 0.38, 1.4),
            },
        );
        classes.insert(
            "conduit".to_string(),
            PropClass {
                variants: vec![PropVariant {
                    model: Model::Log,
                    trunk: Color::srgb(0.1900, 0.1780, 0.1550),
                    foliage: Color::srgb(0.2100, 0.1950, 0.1700),
                }],
                blob_shadow: false,
                squash: Vec3::new(1.0, 0.8, 1.0),
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
                    },
                    PropVariant {
                        model: Model::Conifer,
                        trunk: bark,
                        foliage: Color::srgb(0.0618, 0.1612, 0.0518),
                    },
                    PropVariant {
                        model: Model::Birch,
                        trunk: Color::srgb(0.5216, 0.5059, 0.4510),
                        foliage: Color::srgb(0.1600, 0.2210, 0.0700),
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
                }],
                blob_shadow: false,
                squash: Vec3::new(1.0, 0.75, 1.0),
            },
        );

        // Undergrowth. Small, dense and cheap: these are what make a
        // forest floor read as ground you could walk through rather than
        // a lawn with trunks in it.
        let plain = |model, trunk, foliage| PropClass {
            variants: vec![PropVariant {
                model,
                trunk,
                foliage,
            }],
            blob_shadow: false,
            squash: Vec3::ONE,
        };
        classes.insert(
            "bush".to_string(),
            plain(Model::Bush, bark, Color::srgb(0.0512, 0.0912, 0.0299)),
        );
        classes.insert(
            "fern".to_string(),
            PropClass {
                squash: Vec3::new(1.25, 0.55, 1.25),
                ..plain(Model::Bush, bark, Color::srgb(0.0448, 0.1002, 0.0331))
            },
        );
        classes.insert(
            "mushroom".to_string(),
            plain(
                Model::Mushroom,
                Color::srgb(0.5647, 0.5216, 0.4392),
                Color::srgb(0.4510, 0.1294, 0.0863),
            ),
        );
        classes.insert(
            "deadwood".to_string(),
            plain(Model::Log, Color::srgb(0.0912, 0.0699, 0.0448), bark),
        );
        classes.insert(
            "cactus".to_string(),
            plain(Model::Cactus, bark, Color::srgb(0.0699, 0.1502, 0.0562)),
        );
        classes.insert(
            "drybrush".to_string(),
            PropClass {
                squash: Vec3::new(1.1, 0.7, 1.1),
                ..plain(Model::Bush, bark, Color::srgb(0.1502, 0.1305, 0.0699))
            },
        );
        classes.insert(
            "marshcap".to_string(),
            plain(
                Model::Mushroom,
                Color::srgb(0.4510, 0.4275, 0.3765),
                Color::srgb(0.3255, 0.2510, 0.0980),
            ),
        );
        classes.insert(
            "screebush".to_string(),
            PropClass {
                squash: Vec3::new(1.15, 0.4, 1.15),
                ..plain(Model::Bush, bark, Color::srgb(0.0805, 0.0699, 0.0448))
            },
        );
        classes.insert(
            "reed".to_string(),
            plain(Model::Reed, bark, Color::srgb(0.1002, 0.1305, 0.0562)),
        );
        classes.insert(
            "scree".to_string(),
            PropClass {
                squash: Vec3::new(1.3, 0.5, 1.3),
                ..plain(Model::Rock, bark, Color::srgb(0.2100, 0.2050, 0.2000))
            },
        );
        classes.insert(
            "alpinebush".to_string(),
            PropClass {
                squash: Vec3::new(1.2, 0.45, 1.2),
                ..plain(Model::Bush, bark, Color::srgb(0.0699, 0.0805, 0.0505))
            },
        );
        Self(classes)
    }
}

pub struct PropsPlugin;

impl Plugin for PropsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PropAssets>()
            .add_systems(Update, build_prop_assets)
            .add_systems(Update, (dress_scatter,));
    }
}

/// The mesh/material parts one variant is drawn from.
type VariantParts = Vec<(Handle<Mesh>, Handle<StandardMaterial>)>;

/// Meshes and materials per class variant, built once from [`PropTable`].
#[derive(Resource, Default)]
struct PropAssets {
    /// (world, class) -> variant -> parts.
    ///
    /// Keyed by WORLD too, because a class name is level data: the
    /// planet's "boulder" and purgatory's "boulder" are different rocks,
    /// and one map keyed by name alone would hand whichever built first
    /// to both.
    classes: HashMap<(voxel_engine::WorldId, String), Vec<VariantParts>>,
    /// Worlds whose props have been built already.
    built: std::collections::HashSet<voxel_engine::WorldId>,
    blob_mesh: Handle<Mesh>,
    blob_mat: Handle<StandardMaterial>,
}

/// Build each world's prop meshes once, as its table appears.
///
/// Not a one-shot at startup: a world can arrive when a portal opens, and
/// its props have to be built then.
fn build_prop_assets(
    tables: Res<crate::WorldProps>,
    mut assets: ResMut<PropAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !tables.is_changed() {
        return;
    }
    let pending: Vec<(voxel_engine::WorldId, PropTable)> = tables
        .0
        .iter()
        .filter(|(world, _)| !assets.built.contains(world))
        .map(|(world, table)| (*world, table.clone()))
        .collect();
    for (world, table) in pending {
        assets.built.insert(world);
        // Nothing a prop is made of is a mirror — bark, leaves, rock. The
        // default F0 of 0.04 under this demo's sun (FULL_DAYLIGHT against an
        // ambient a hundredth of it) put a white highlight on every skyward
        // face, which turned a dark green conifer into pale mint while the
        // impostor standing behind it stayed green. `voxel_impostor.wgsl`
        // already zeroes reflectance for exactly this reason; the props were
        // the half that never got the memo.
        let mat = |base_color: Color, rough: f32| StandardMaterial {
            base_color,
            perceptual_roughness: rough,
            reflectance: 0.0,
            ..default()
        };
        for (class, def) in &table.0 {
            let mut variants = Vec::new();
            for (slot, variant) in def.variants.iter().enumerate() {
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
                    Model::Bush => {
                        let mut b = MeshBuilder::default();
                        b.blob(Vec3::new(0.0, 0.34, 0.0), 0.46, 0.30, 5);
                        b.blob(Vec3::new(0.30, 0.24, 0.16), 0.30, 0.34, 61);
                        b.blob(Vec3::new(-0.24, 0.22, -0.20), 0.27, 0.34, 131);
                        parts.push((meshes.add(b.build()), foliage_mat));
                    }
                    Model::Mushroom => {
                        parts.push((meshes.add(cylinder_mesh(0.045, 0.22, 6)), trunk_mat));
                        let mut cap = MeshBuilder::default();
                        cap.blob(Vec3::new(0.0, 0.235, 0.0), 0.16, 0.12, 17);
                        parts.push((meshes.add(cap.build()), foliage_mat));
                    }
                    Model::Cactus => {
                        let mut c = MeshBuilder::default();
                        c.limb(Vec3::ZERO, Vec3::Y, 0.16, 1.5, 7);
                        // Arms: up the trunk, out, then up again.
                        c.limb(Vec3::new(0.0, 0.75, 0.0), Vec3::X, 0.09, 0.42, 6);
                        c.limb(Vec3::new(0.42, 0.75, 0.0), Vec3::Y, 0.09, 0.52, 6);
                        c.limb(Vec3::new(0.0, 0.95, 0.0), -Vec3::X, 0.08, 0.34, 6);
                        c.limb(Vec3::new(-0.34, 0.95, 0.0), Vec3::Y, 0.08, 0.40, 6);
                        parts.push((meshes.add(c.build()), foliage_mat));
                    }
                    Model::Log => {
                        let mut l = MeshBuilder::default();
                        l.limb(Vec3::new(-0.9, 0.17, 0.0), Vec3::X, 0.17, 1.8, 7);
                        parts.push((meshes.add(l.build()), trunk_mat));
                    }
                    Model::Reed => {
                        let mut r = MeshBuilder::default();
                        for i in 0..7u32 {
                            let a = std::f32::consts::TAU * i as f32 / 7.0 + i as f32 * 0.9;
                            let (sn, cs) = a.sin_cos();
                            let lean = Vec3::new(cs, 0.0, sn) * 0.16;
                            let base = Vec3::new(cs, 0.0, sn) * 0.09;
                            let h = 0.75 + ((i * 37) % 11) as f32 * 0.06;
                            r.limb(base, (Vec3::Y + lean).normalize(), 0.018, h, 4);
                        }
                        parts.push((meshes.add(r.build()), foliage_mat));
                    }
                    model @ (Model::Broadleaf | Model::Conifer | Model::Birch) => {
                        let (wood, foliage) = grow_tree(&tree_shape(model), tree_seed(model, slot));
                        parts.push((meshes.add(wood), trunk_mat));
                        parts.push((meshes.add(foliage), foliage_mat));
                    }
                }
                variants.push(parts);
            }
            assets.classes.insert((world, class.clone()), variants);
        }
    }
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
    worlds: Res<voxel_engine::Worlds>,
    tables: Res<crate::WorldProps>,
    new: Query<(Entity, &ScatterInstance, &Transform, &crate::OfWorld), Added<ScatterInstance>>,
) {
    // EVERY command here is fallible, because the entity may be gone by
    // the time they apply. `Added<ScatterInstance>` hands out entities
    // the engine's scatter layer owns, and a tile created and released
    // within a frame — which is what turning back across ground you just
    // crossed does — despawns them between the query and the flush.
    // `commands.entity(..).insert(..)` panics on that; `try_insert` and
    // `queue_silenced` drop the work for an instance that no longer
    // exists, which is exactly right: there is nothing left to dress.
    for (entity, instance, transform, of_world) in &new {
        // ITS world's table and ITS world's sun: a tree through a portal
        // is dressed by the level it grows in, not by the one you are
        // standing in.
        let Some(query) = worlds.query(of_world.0) else {
            continue;
        };
        let table = tables.0.get(&of_world.0).cloned().unwrap_or_default();
        let Some(variants) = assets
            .classes
            .get(&(of_world.0, instance.class.to_string()))
        else {
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
        // Every child carries its world's layer of its own — see
        // [`crate::OfWorld::scene`]. Parenting does not confer it.
        let layers = voxel_render::world_layer(of_world.0);
        for (mesh, material) in parts {
            let (mesh, material) = (mesh.clone(), material.clone());
            let layers = layers.clone();
            commands
                .entity(entity)
                .queue_silenced(move |mut instance: EntityWorldMut| {
                    instance.with_children(|parent| {
                        parent.spawn((Mesh3d(mesh), MeshMaterial3d(material), layers));
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
            let local = Transform::from_matrix(parent.to_matrix().inverse() * world.to_matrix());
            let (blob_mesh, blob_mat) = (assets.blob_mesh.clone(), assets.blob_mat.clone());
            commands
                .entity(entity)
                .queue_silenced(move |mut instance: EntityWorldMut| {
                    instance.with_children(|children| {
                        children.spawn((
                            Mesh3d(blob_mesh),
                            MeshMaterial3d(blob_mat),
                            local,
                            layers,
                        ));
                    });
                });
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

    /// A tapered column from `base` along `dir`. The one primitive the
    /// stems, arms, logs and blades are all made of.
    fn limb(&mut self, base: Vec3, dir: Vec3, radius: f32, len: f32, sides: u32) {
        let dir = dir.normalize_or(Vec3::Y);
        let side = dir.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
        let up = side.cross(dir);
        let tip = base + dir * len;
        for i in 0..sides {
            let a0 = std::f32::consts::TAU * i as f32 / sides as f32;
            let a1 = std::f32::consts::TAU * (i + 1) as f32 / sides as f32;
            let r0 = side * a0.cos() + up * a0.sin();
            let r1 = side * a1.cos() + up * a1.sin();
            // Tapered, so a stem reads as organic and a log as a log.
            let t = 0.75;
            let s = self.positions.len() as u32;
            self.positions.extend([
                (base + r0 * radius).to_array(),
                (base + r1 * radius).to_array(),
                (tip + r0 * radius * t).to_array(),
                (tip + r1 * radius * t).to_array(),
            ]);
            self.normals
                .extend([r0.to_array(), r1.to_array(), r0.to_array(), r1.to_array()]);
            self.indices.extend([s, s + 2, s + 1, s + 1, s + 2, s + 3]);
        }
    }

    /// A truncated cone between two points — what one limb of a grown
    /// skeleton is drawn as. Unlike [`Self::limb`] both ends carry their
    /// own radius, because a skeleton decides its own taper.
    fn wood(&mut self, a: Vec3, b: Vec3, r_a: f32, r_b: f32, sides: u32) {
        let dir = (b - a).normalize_or(Vec3::Y);
        let side = dir.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
        let up = side.cross(dir);
        for i in 0..sides {
            let a0 = std::f32::consts::TAU * i as f32 / sides as f32;
            let a1 = std::f32::consts::TAU * (i + 1) as f32 / sides as f32;
            let r0 = side * a0.cos() + up * a0.sin();
            let r1 = side * a1.cos() + up * a1.sin();
            let s = self.positions.len() as u32;
            self.positions.extend([
                (a + r0 * r_a).to_array(),
                (a + r1 * r_a).to_array(),
                (b + r0 * r_b).to_array(),
                (b + r1 * r_b).to_array(),
            ]);
            self.normals
                .extend([r0.to_array(), r1.to_array(), r0.to_array(), r1.to_array()]);
            self.indices.extend([s, s + 2, s + 1, s + 1, s + 2, s + 3]);
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

/// What a grown tree looks like, as the few numbers that actually differ
/// between an oak, a pine and a birch.
///
/// The SHAPE is not authored — no trunk height, no branch angles, no
/// forks. A crown is a volume, the wood is whatever reaches it, and the
/// silhouette falls out. That is the point of growing them: the old
/// broadleaf was a cylinder and three blobs, and every one of them was
/// the same cylinder and the same three blobs.
struct TreeShape {
    /// Clear trunk below the crown, and the crown's own half-extents.
    trunk_m: f32,
    crown: Vec3,
    /// Taper the crown to a point (a conifer) rather than filling it.
    conical: bool,
    /// Attractor spacing — the crown's detail, and most of the cost.
    spacing: f32,
    growth: voxel_core::branch::Growth,
    /// Foliage blob radius at a tip, and how deep a limb must be before
    /// it carries one. Leaves near the trunk read as a bush.
    leaf_r: f32,
    leaf_from: u16,
}

/// The shape each tree model grows into.
fn tree_shape(model: Model) -> TreeShape {
    use voxel_core::branch::Growth;
    // Reach has to span the gap from the root to the crown, or nothing
    // is ever pulled and the tree is a single stump. It is what makes
    // the trunk emerge: from the ground the whole crown averages to
    // straight up, and only inside it do the pulls disagree enough to
    // fork.
    let base = Growth {
        step_m: 0.42,
        kill_m: 0.85,
        max_nodes: 150,
        taper: 2.5,
        tip_r_m: 0.022,
        wobble: 0.14,
        ..Default::default()
    };
    match model {
        Model::Conifer => TreeShape {
            trunk_m: 1.1,
            crown: Vec3::new(1.35, 2.2, 1.35),
            conical: true,
            spacing: 0.62,
            growth: Growth {
                attraction_m: 5.0,
                droop: 0.22, // swept-down firs
                ..base
            },
            leaf_r: 0.42,
            leaf_from: 2,
        },
        Model::Birch => TreeShape {
            trunk_m: 2.6,
            crown: Vec3::new(1.0, 1.25, 1.0),
            conical: false,
            spacing: 0.5,
            growth: Growth {
                attraction_m: 6.5,
                droop: 0.06,
                tip_r_m: 0.016, // whippier than an oak
                ..base
            },
            leaf_r: 0.4,
            leaf_from: 3,
        },
        // Broadleaf: a wide, high crown on a short trunk.
        _ => TreeShape {
            trunk_m: 1.5,
            crown: Vec3::new(1.75, 1.5, 1.75),
            conical: false,
            spacing: 0.6,
            growth: Growth {
                attraction_m: 6.0,
                droop: 0.02,
                ..base
            },
            leaf_r: 0.62,
            leaf_from: 3,
        },
    }
}

/// A seed per (model, variant slot), so two levels asking for the same
/// model get the same tree and two variants of one model do not.
fn tree_seed(model: Model, variant: usize) -> u64 {
    let kind = match model {
        Model::Conifer => 1u64,
        Model::Birch => 2,
        _ => 0,
    };
    voxel_core::seed::splitmix64(0x7BEE_0000 ^ (kind << 32) ^ variant as u64)
}

/// Grow one tree and return (wood, foliage) meshes.
///
/// One mesh per VARIANT, not per instance: the forest draws a million
/// trees by instancing a handful of meshes, so variety is bought by
/// growing several variants from different seeds — never by growing one
/// per tree.
fn grow_tree(shape: &TreeShape, seed: u64) -> (Mesh, Mesh) {
    use voxel_core::branch::{colonize, ellipsoid_cloud};

    let mut rng = voxel_core::seed::Rng::new(seed);
    let center = Vec3::new(0.0, shape.trunk_m + shape.crown.y, 0.0);
    let mut cloud = ellipsoid_cloud(center, shape.crown, shape.spacing, 0.85, &mut rng);
    // The kill radius MUST stay under the attractor spacing, or a single
    // stem consumes the whole cloud as it passes and never forks: the
    // conifer and the birch grew as bare poles with no branching at all
    // because `kill_m` (0.85) was larger than their spacing (0.62, 0.5).
    // Derived here rather than authored so the two cannot drift apart.
    let growth = voxel_core::branch::Growth {
        kill_m: shape.spacing * 0.7,
        ..shape.growth
    };
    if shape.conical {
        // Keep the cone inscribed in the same ellipsoid: full width at
        // the crown's base, a point at its top.
        let base = center.y - shape.crown.y;
        cloud.retain(|p| {
            let t = ((p.y - base) / (2.0 * shape.crown.y)).clamp(0.0, 1.0);
            let r = Vec2::new(p.x - center.x, p.z - center.z).length();
            r <= shape.crown.x.max(shape.crown.z) * (1.0 - t)
        });
    }
    let limbs = colonize(Vec3::ZERO, Vec3::Y, &cloud, &growth, &mut rng);

    let mut wood = MeshBuilder::default();
    for l in &limbs {
        // EVERY limb is drawn. Culling thin ones was a false economy: a
        // skeleton is only a few dozen limbs, and the thin ones are
        // exactly the twigs the foliage hangs on — cull them and the
        // blobs float over bare ground with nothing holding them up.
        // Cheap where it is cheap instead: four sides on a twig.
        let sides = if l.r_a > 0.05 { 6 } else { 4 };
        wood.wood(l.a, l.b, l.r_a, l.r_b, sides);
    }

    // Foliage hangs on tips: a limb no other limb grows out of.
    let mut foliage = MeshBuilder::default();
    for (i, l) in limbs.iter().enumerate() {
        let is_tip = !limbs.iter().any(|o| o.a == l.b);
        if !is_tip || l.depth < shape.leaf_from {
            continue;
        }
        let jitter = 0.18 + 0.12 * ((i * 37 % 13) as f32 / 13.0);
        foliage.blob(
            l.b,
            shape.leaf_r,
            jitter,
            (seed as u32).wrapping_add(i as u32),
        );
    }
    (wood.build(), foliage.build())
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
