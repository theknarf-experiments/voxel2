//! A second level, live alongside the first.
//!
//! The point of a portal is that both worlds exist at once — you can see
//! into one from the other and walk between them — so the far side cannot
//! be a level swap. It is registered as its own world: its own generator,
//! its own LOD field, its own anchor, streaming continuously whether or
//! not you are looking at it.
//!
//! Worlds share coordinates. World 1 is not "over there"; it is the same
//! space, and which one you see is `CameraWorld`. That is why only the
//! camera's world is drawn — two levels rendered together would interleave
//! two solids rather than show two places.

use bevy::prelude::*;
use bevy::camera::visibility::RenderLayers;
use voxel_engine::{LevelDef, StreamedWorlds};
use voxel_render::WorldPrograms;

/// Coarsest level the far world streams while it has no portal to be
/// seen through. See `register_far_world`.
const FAR_MAX_LEVEL: u8 = 5;

/// The far side: which level file to open onto, and — once opened — the
/// level itself with its own background, which the opening shows wherever
/// the far world is empty.
///
/// Loaded on demand rather than at startup: a portal is something you
/// open, and a second world is not free (it roughly doubles the meshed
/// working set), so nobody should pay for one until they ask.
#[derive(Resource)]
pub struct FarLevel {
    pub path: String,
    pub loaded: Option<LevelDef>,
    pub clear_color: Color,
    pub world: Option<u8>,
}

pub struct PortalPlugin;

impl Plugin for PortalPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
                Update,
                (open_portal, traverse_portal, follow_camera_world, drive_portal).chain(),
            );
    }
}

/// Load the far level and register it as its own world, once.
///
/// Returns the world id, or `None` if it could not be read — a portal to
/// a level that will not parse should say so rather than open onto
/// nothing.
fn ensure_far_world(
    far: &mut FarLevel,
    worlds: &mut StreamedWorlds,
    programs: &mut WorldPrograms,
    materials: &mut voxel_render::WorldMaterials,
    near: &LevelDef,
) -> Option<u8> {
    if let Some(id) = far.world {
        return Some(id);
    }
    let json = std::fs::read_to_string(&far.path)
        .inspect_err(|e| error!("portal: cannot read '{}': {e}", far.path))
        .ok()?;
    let level = LevelDef::from_json(&json)
        .inspect_err(|e| error!("portal: cannot parse '{}': {e}", far.path))
        .ok()?;

    let generator = std::sync::Arc::new(level.generator(0));
    // INTERIM: a second world roughly doubles the meshed working set and
    // the slab is one fixed GPU allocation sized for one, so at the far
    // level's own config every class hit 100% and chunks wedged in
    // AwaitingAlloc. Now that the far world streams around the PORTAL
    // rather than the camera its working set is much smaller, so this cap
    // is probably droppable — but on evidence, not on assumption.
    let mut config = voxel_engine::LodConfig::from(&level.lod);
    config.max_level = config.max_level.min(FAR_MAX_LEVEL);
    config.top_radius = 1;
    let id = worlds.add(config, generator.clone());
    programs.0.push(voxel_render::WorldProgram {
        ops: std::sync::Arc::new(generator.ops().to_vec()),
        seed: generator.seed(),
        sun_dir: generator.sun_direction(),
    });
    // One material table serves every world. It is INDEXED by the id a
    // chunk's vertex carries, and ids are level data, so two levels that
    // coexist simply must not reuse one — planet is 1/3/4, the
    // megastructure is 2. A collision would silently repaint one world
    // with the other's recipe.
    for def in &level.materials {
        let mid = def.id() as usize;
        assert!(mid < materials.0.len(), "far material id {mid} out of range");
        assert!(
            !near.materials.iter().any(|n| n.id() == def.id()),
            "far level reuses material id {mid}, which the near level defines",
        );
        materials.0[mid] = def.pack();
    }
    far.loaded = Some(level);
    far.world = Some(id);
    info!("portal: '{}' opened as world {id}", far.path);
    Some(id)
}

/// Scene content belongs to a world, and Bevy already has the mechanism:
/// render layers. Grass, trees, water and props are ordinary entities
/// queued against the view's visible set, so putting the CAMERA on the
/// layer of the world it is in filters every one of them at once, with no
/// change to any of their pipelines.
///
/// Chunks are not entities and cannot use this — they are filtered by
/// `key.world` in the draw loop instead. Two mechanisms, because there are
/// two kinds of thing being drawn.
fn follow_camera_world(
    mut commands: Commands,
    camera_world: Res<voxel_render::CameraWorld>,
    far: Option<Res<FarLevel>>,
    scene: Res<crate::HostScene>,
    mut clear: ResMut<ClearColor>,
    // `Option`, because not every camera carries the component: the
    // offscreen mirror camera `voxctl shot` renders through is spawned
    // without one, so a query that REQUIRED it silently skipped the very
    // camera the screenshots come from — the switch looked broken while
    // working perfectly in the window.
    mut cameras: Query<(Entity, Option<&mut RenderLayers>), With<Camera3d>>,
) {
    // Every frame, not just when the world changes: cameras appear late.
    // The offscreen mirror camera `voxctl shot` renders through is created
    // lazily on the first request, so gating on `is_changed` left it on
    // layer 0 forever — the window showed the right world while every
    // screenshot showed the wrong one.
    // The background belongs to the world you are in: step into an
    // interior and the sky should stop being sky.
    let want_clear = if camera_world.0 == 0 {
        scene.0.clear_color
    } else {
        far.as_ref().map_or(scene.0.clear_color, |f| f.clear_color)
    };
    if clear.0 != want_clear {
        clear.0 = want_clear;
    }
    let want = RenderLayers::layer(usize::from(camera_world.0));
    for (entity, layers) in &mut cameras {
        // The layer filters scene ENTITIES; `ViewWorld` tells the chunk
        // draw which world's list to render. Both, because chunks are not
        // entities.
        commands
            .entity(entity)
            .insert(voxel_render::ViewWorld(camera_world.0));
        match layers {
            Some(mut layers) => {
                if *layers != want {
                    *layers = want.clone();
                }
            }
            None => {
                commands.entity(entity).insert(want.clone());
            }
        }
    }
}

/// A rectangular opening between two worlds.
///
/// `near`/`far` place the SAME opening in each world. Looking through it
/// from the near side shows the far world as if the two spaces were
/// joined along the rectangle, which is what "seamless" means here: the
/// far view is the near view moved by `far * near⁻¹`.
#[derive(Component, Clone, Copy)]
pub struct Portal {
    pub near: Transform,
    pub far: Transform,
    pub near_world: u8,
    pub far_world: u8,
    /// Half width and half height of the opening, in metres.
    pub half: Vec2,
}

impl Portal {
    /// The opening's four corners, in the given placement.
    fn corners(at: &Transform, half: Vec2) -> [Vec3; 4] {
        let x = at.rotation * Vec3::X * half.x;
        let y = at.rotation * Vec3::Y * half.y;
        [
            at.translation - x - y,
            at.translation + x - y,
            at.translation + x + y,
            at.translation - x + y,
        ]
    }
}

/// Near-side views: every camera that is not itself a portal camera.
type NearViews<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static GlobalTransform,
        &'static Camera,
        Option<&'static bevy::camera::RenderTarget>,
    ),
    (With<Camera3d>, Without<PortalCamera>),
>;

/// The quad painted with the far world's background, behind the opening.
#[derive(Component)]
pub struct PortalBackdrop;

/// Marks a camera that renders the far side FOR a particular near-side
/// camera.
///
/// One per view, not one per portal: the window and the offscreen mirror
/// `voxctl shot` renders through are different views of the same near
/// world, and each needs its own paired far view aimed through the same
/// opening into the same target. A single portal camera pointed at the
/// window is invisible to every screenshot, which is a memorable way to
/// spend an afternoon.
#[derive(Component)]
pub struct PortalCamera(pub Entity);

/// Place the opening in front of the camera, once, and give it a camera
/// of its own.
///
/// Positioned from where the camera ACTUALLY is rather than from the
/// level's declared start, so `VOXEL_START` still puts you in front of it.
/// Open the portal in front of the camera, on F7 or on
/// `voxctl portal`.
///
/// Not at startup and not from an env var: a portal is something you
/// open, where you are looking. Opening it again moves it, so the
/// interesting cases — walking in at an angle, standing something in
/// front of it — can all be tried without a restart.
#[allow(clippy::too_many_arguments)]
fn open_portal(
    mut commands: Commands,
    mut far: ResMut<FarLevel>,
    near: Res<LevelDef>,
    camera: Query<&GlobalTransform, With<crate::FreeCamera>>,
    mut portal: Query<&mut Portal>,
    mut backdrop: Query<&mut Transform, With<PortalBackdrop>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut host: ResMut<voxel_debug::remote::HostCommands>,
    (mut worlds, mut programs, mut materials): (
        ResMut<StreamedWorlds>,
        ResMut<WorldPrograms>,
        ResMut<voxel_render::WorldMaterials>,
    ),
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials_assets: ResMut<Assets<StandardMaterial>>,
) {
    let asked = keys.just_pressed(KeyCode::F7) || {
        let n = host.0.len();
        host.0
            .retain(|c| c.get("cmd").and_then(|c| c.as_str()) != Some("portal"));
        host.0.len() != n
    };
    if !asked {
        return;
    }
    let Ok(eye) = camera.single() else {
        return;
    };
    let Some(far_world) = ensure_far_world(
        &mut far,
        &mut worlds,
        &mut programs,
        &mut materials,
        &near,
    ) else {
        return;
    };

    let forward = eye.forward().as_vec3();
    let at = eye.translation() + forward * 14.0;
    let placement = Transform::from_translation(at).looking_to(-forward, Vec3::Y);

    // Already open: move it rather than stacking a second one, which
    // would make `portals.single()` fail and stop the far view dead.
    if let (Ok(mut portal), Ok(mut quad)) = (portal.single_mut(), backdrop.single_mut()) {
        portal.near = placement;
        *quad = placement;
        info!("portal moved to {at:?}");
        return;
    }

    let quad = meshes.add(Rectangle::new(8.0, 6.0));
    let quad_mat = materials_assets.add(StandardMaterial {
        base_color: far.clear_color,
        unlit: true,
        // Both faces: an opening is approachable from either side, and a
        // one-sided quad silently vanishes from whichever side it is not
        // facing — which reads as "the backdrop does not work".
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    commands.spawn(Portal {
        near: placement,
        // Room to stand inside the far level.
        far: Transform::from_translation(Vec3::new(0.0, 22.0, 0.0)),
        near_world: 0,
        far_world,
        half: Vec2::new(4.0, 3.0),
    });
    // The opening's backdrop, in the NEAR world at the opening: the far
    // camera cannot clear only the opening, so without this you see the
    // near world wherever the far world is empty. Being in the near world
    // is the point — anything in FRONT of the opening still occludes it.
    commands.spawn((
        PortalBackdrop,
        Mesh3d(quad),
        MeshMaterial3d(quad_mat),
        placement,
        RenderLayers::layer(0),
    ));
    info!("portal opened at {at:?}");
}

/// Pair every near-side view with a far-side one, aim it, and hand the
/// far world its mask.
///
/// The far view is the near view moved by the pairing, so the two line up
/// at the opening by construction rather than by tuning.
fn drive_portal(
    mut commands: Commands,
    portals: Query<&Portal>,
    sources: NearViews,
    mut portal_cams: Query<(Entity, &PortalCamera, &mut Transform, &mut Camera)>,
    mut clips: ResMut<voxel_render::WorldClips>,
    mut focus: ResMut<voxel_engine::WorldFocus>,
    camera_world: Res<voxel_render::CameraWorld>,
) {
    clips.0.clear();
    clips.0.resize(voxel_render::MAX_WORLDS, Vec::new());
    let Ok(portal) = portals.single() else {
        return;
    };
    let showing = if camera_world.0 == portal.near_world {
        portal.far_world
    } else {
        portal.near_world
    };
    let (from, to) = if camera_world.0 == portal.near_world {
        (portal.near, portal.far)
    } else {
        (portal.far, portal.near)
    };
    let motion =
        Transform::from_matrix(Mat4::from(to.compute_affine() * from.compute_affine().inverse()));

    for (source, eye, camera, target) in &sources {
        let existing = portal_cams
            .iter_mut()
            .find(|(_, paired, _, _)| paired.0 == source);
        let placement = Transform::from_matrix(motion.to_matrix() * eye.to_matrix());
        match existing {
            Some((entity, _, mut transform, mut cam)) => {
                *transform = placement;
                // Just after the view it pairs, into the same target.
                cam.order = camera.order + 1;
                let mut cmd = commands.entity(entity);
                cmd.insert(voxel_render::ViewWorld(showing))
                    .insert(RenderLayers::layer(usize::from(showing)));
                if let Some(target) = target {
                    cmd.insert(target.clone());
                }
            }
            None => {
                let mut spawned = commands.spawn((
                    PortalCamera(source),
                    Camera3d::default(),
                    Camera {
                        order: camera.order + 1,
                        // Keep what the near view already drew: the far
                        // world appears INTO it.
                        clear_color: bevy::camera::ClearColorConfig::None,
                        ..default()
                    },
                    placement,
                    voxel_render::ViewWorld(showing),
                    RenderLayers::layer(usize::from(showing)),
                ));
                if let Some(target) = target {
                    spawned.insert(target.clone());
                }
            }
        }
    }

    // The mask, in the FAR world's coordinates: the pyramid from the
    // (moved) eye through the (far) opening, plus the opening's own plane
    // so nothing between it and the eye leaks in.
    let Some((_, eye, _, _)) = sources.iter().next() else {
        return;
    };
    let far_eye = motion.transform_point(eye.translation());
    // Stream the far world around where the portal looks INTO it. Left on
    // the camera's position it resides chunks in the far world at the near
    // world's coordinates, and the opening looks out onto empty space.
    focus.0.clear();
    focus.0.resize(voxel_render::MAX_WORLDS, None);
    focus.0[usize::from(showing)] = Some(far_eye.as_dvec3());
    let corners = Portal::corners(&to, portal.half);
    let mut planes = Vec::with_capacity(voxel_render::MAX_CLIP_PLANES);
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let mut n = (a - far_eye).cross(b - far_eye).normalize_or_zero();
        if n.dot(corners[(i + 2) % 4] - a) < 0.0 {
            n = -n;
        }
        planes.push(n.extend(-n.dot(a)));
    }
    let mut ahead = to.rotation * Vec3::Z;
    if ahead.dot(to.translation - far_eye) < 0.0 {
        ahead = -ahead;
    }
    planes.push(ahead.extend(-ahead.dot(to.translation)));
    if std::env::var_os("PORTAL_NOCLIP").is_none() {
        clips.0[usize::from(showing)] = planes;
    }
}

/// Step through: crossing the opening moves you by the pairing and swaps
/// which world you are in.
///
/// Tested against the SEGMENT the camera travelled this frame, not against
/// which side it is on now. At walking speed a frame covers centimetres,
/// but a fast flight covers tens of metres and would step straight over a
/// 3 m opening between two samples — the portal would work until you
/// approached it quickly, which is the worst way for it to fail.
fn traverse_portal(
    portals: Query<&Portal>,
    mut camera: Query<&mut Transform, With<crate::FreeCamera>>,
    mut camera_world: ResMut<voxel_render::CameraWorld>,
    mut was_at: Local<Option<Vec3>>,
) {
    let (Ok(portal), Ok(mut transform)) = (portals.single(), camera.single_mut()) else {
        return;
    };
    let now = transform.translation;
    let Some(before) = was_at.replace(now) else {
        return;
    };
    let entering = camera_world.0 == portal.near_world;
    let (from, to) = if entering {
        (portal.near, portal.far)
    } else {
        (portal.far, portal.near)
    };

    // Signed distance to the opening's plane, before and after.
    let normal = from.rotation * Vec3::Z;
    let plane_d = -normal.dot(from.translation);
    let (d0, d1) = (
        normal.dot(before) + plane_d,
        normal.dot(now) + plane_d,
    );
    if (d0 < 0.0) == (d1 < 0.0) {
        return; // did not cross
    }
    // Where it crossed, and whether that is inside the opening.
    let t = d0 / (d0 - d1);
    let hit = before.lerp(now, t.clamp(0.0, 1.0));
    let local = from.rotation.inverse() * (hit - from.translation);
    if local.x.abs() > portal.half.x || local.y.abs() > portal.half.y {
        return; // through the wall beside it, not the opening
    }

    let motion =
        Transform::from_matrix(Mat4::from(to.compute_affine() * from.compute_affine().inverse()));
    transform.translation = motion.transform_point(now);
    transform.rotation = motion.rotation * transform.rotation;
    camera_world.0 = if entering {
        portal.far_world
    } else {
        portal.near_world
    };
    *was_at = Some(transform.translation);
    info!("stepped into world {}", camera_world.0);
}
