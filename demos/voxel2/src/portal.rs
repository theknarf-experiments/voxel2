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

/// Level file for the far side, if the host asked for one.
#[derive(Resource)]
pub struct FarLevel(pub LevelDef);

pub struct PortalPlugin;

impl Plugin for PortalPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, register_far_world)
            .add_systems(Update, (spawn_portal, follow_camera_world, drive_portal).chain());
    }
}

/// Register the far level as world 1: generator into the shared program
/// buffer, LOD field into the streamer.
fn register_far_world(
    far: Option<Res<FarLevel>>,
    mut worlds: ResMut<StreamedWorlds>,
    mut programs: ResMut<WorldPrograms>,
    mut materials: ResMut<voxel_render::WorldMaterials>,
    near: Res<LevelDef>,
) {
    let Some(far) = far else {
        return;
    };
    let level = &far.0;
    let generator = std::sync::Arc::new(level.generator(0));
    // INTERIM, and the number that matters most here: a second world
    // roughly doubles the meshed working set, and the slab is one fixed
    // GPU allocation sized for one world — at the far level's own config
    // every class hit 100% and 427 chunks wedged in AwaitingAlloc.
    //
    // The real fix is the portal: the far side only needs to be resident
    // where you can SEE it, which is a cone through a surface, not a
    // sphere around the camera. Until that exists, hold it to a small
    // field so the thing streams and settles.
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
    // with the other's recipe, so it is worth the assert.
    for def in &level.materials {
        let id = def.id() as usize;
        assert!(
            id < materials.0.len(),
            "far level material id {id} is out of range",
        );
        assert!(
            !near.materials.iter().any(|n| n.id() == def.id()),
            "far level reuses material id {id}, which the near level already defines",
        );
        materials.0[id] = def.pack();
    }
    info!("far level registered as world {id}");
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
fn spawn_portal(
    mut commands: Commands,
    far: Option<Res<FarLevel>>,
    camera: Query<&GlobalTransform, With<crate::FreeCamera>>,
    existing: Query<(), With<Portal>>,
) {
    if far.is_none() || !existing.is_empty() {
        return;
    }
    let Ok(eye) = camera.single() else {
        return;
    };
    let forward = eye.forward().as_vec3();
    let at = eye.translation() + forward * 14.0;
    let near = Transform::from_translation(at).looking_to(-forward, Vec3::Y);
    commands.spawn(Portal {
        near,
        // Room to stand inside the megastructure.
        far: Transform::from_translation(Vec3::new(0.0, 22.0, 0.0)),
        near_world: 0,
        far_world: 1,
        half: Vec2::new(4.0, 3.0),
    });
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
