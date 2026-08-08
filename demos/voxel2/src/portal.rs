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
use voxel_engine::{LevelDef, LodConfig, WorldLoader};

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
    /// How the far side is dressed — its background, its sun, its
    /// ambient. A world's presentation belongs to the world, not to
    /// whichever level the app happened to launch with.
    pub scene: crate::Scene,
    pub world: Option<u8>,
}

pub struct PortalPlugin;

impl Plugin for PortalPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
                Update,
                (
                    open_portal,
                    traverse_portal,
                    follow_camera_world,
                    sync_backdrops,
                    drive_portal,
                )
                    .chain()
                    // `drive_portal` publishes where each world is seen
                    // from; the engine's graphs follow it the same frame.
                    .in_set(voxel_engine::WorldFocusSet::Publish),
            );
    }
}

/// Load the far level and register it as its own world, once.
///
/// Returns the world id, or `None` if it could not be read — a portal to
/// a level that will not parse should say so rather than open onto
/// nothing.
///
/// One call. Everything a world needs — generator, program, materials,
/// planning, ops provider, LOD config — is registered together by
/// [`WorldLoader`], so a second world is not a reduced version of the
/// first. It gets its own material ids (planet's 1 and the
/// megastructure's 1 no longer collide), its own planning graph, and its
/// own painted surface map.
fn ensure_far_world(far: &mut FarLevel, loader: &mut WorldLoader) -> Option<u8> {
    if let Some(id) = far.world {
        return Some(id);
    }
    let json = std::fs::read_to_string(&far.path)
        .inspect_err(|e| error!("portal: cannot read '{}': {e}", far.path))
        .ok()?;
    let level = LevelDef::from_json(&json)
        .inspect_err(|e| error!("portal: cannot parse '{}': {e}", far.path))
        .ok()?;

    // At its authored detail. There used to be a hand-picked cap here,
    // because a second world roughly doubles the meshed working set and
    // the slab is one fixed allocation: every class hit 100% and chunks
    // wedged waiting for space. `WorldLoader::load` now admits a world
    // against the slab slots the loaded ones left, and caps it only as
    // far as it has to — which is the mechanism that cap was guessing at.
    let id = loader.load(level.clone(), 0, LodConfig::from(&level.lod));
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
    mut ambient: ResMut<GlobalAmbientLight>,
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
    // The background AND the ambient belong to the world you are in: step
    // into an interior and the sky should stop being sky. Ambient is a
    // single global in Bevy, so unlike the sun it cannot be given to a
    // world and left there — it has to follow the camera. A sunless
    // interior lit at the planet's ambient is nearly black.
    let here = match far.as_ref().filter(|f| f.world == Some(camera_world.0)) {
        Some(far) => &far.scene,
        None => &scene.0,
    };
    if clear.0 != here.clear_color {
        clear.0 = here.clear_color;
    }
    if ambient.brightness != here.ambient_brightness || ambient.color != here.ambient_color {
        ambient.color = here.ambient_color;
        ambient.brightness = here.ambient_brightness;
    }
    let want = voxel_render::world_layer(camera_world.0);
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
    /// The rigid motion carrying the `from` side onto the `to` side.
    ///
    /// With a HALF TURN in the middle, and that is the whole subtlety.
    /// The two openings of a doorway FACE EACH OTHER: you go in through
    /// one and come out of the other with it behind you. Mapping one
    /// frame directly onto the other — `to * from⁻¹` — instead lines
    /// their fronts up the same way, so you emerge travelling back INTO
    /// the far opening, having apparently turned around. Nothing is
    /// flipped: the pairing was simply missing the turn a doorway has.
    ///
    /// ONE definition, used by both the view and the traversal. They must
    /// agree exactly or you see one thing through the opening and arrive
    /// somewhere else.
    fn motion(from: &Transform, to: &Transform) -> Transform {
        let flip = Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::PI));
        Transform::from_matrix(Mat4::from(
            to.compute_affine() * flip.compute_affine() * from.compute_affine().inverse(),
        ))
    }

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

/// Whether to render the far world through the opening. ON — without it
/// a portal is a coloured rectangle you can walk through.
///
/// It was off while the near world's terrain vanished whenever a second
/// world was registered. That is fixed and understood: a material asset
/// touched every frame never settles, so its bind group was rebuilt under
/// the draw and `SetMaterialBindGroup` skipped the whole terrain. Nothing
/// about it was the portal's doing — the reproducer needed no portal
/// camera at all.
///
/// `VOXEL_PORTAL_VIEW=0` turns it off again, because the far view is the
/// expensive half (a second pass over a second world) and it is worth
/// being able to take it off the frame when measuring something else.
fn far_view_enabled() -> bool {
    std::env::var("VOXEL_PORTAL_VIEW").as_deref() != Ok("0")
}

/// The quad filling the opening with the OTHER world's background.
///
/// ONE PER SIDE. An opening is a hole in both worlds, so each side needs
/// its own quad, in its own world, on its own layer — a single backdrop
/// on the near world's layer is simply not there when you look back from
/// the far side.
#[derive(Component)]
pub struct PortalBackdrop {
    /// The world this quad lives in.
    world: voxel_engine::WorldId,
}

/// Shared handles for the backdrop quads, made once.
#[derive(Resource)]
struct PortalAssets {
    quad: Handle<Mesh>,
    /// Indexed by the world the quad lives IN; it is painted with the
    /// background of the world you are looking THROUGH to.
    material: [Handle<StandardMaterial>; 2],
}

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
    camera: Query<&GlobalTransform, With<crate::FreeCamera>>,
    mut portal: Query<(Entity, &mut Portal)>,
    keys: Res<ButtonInput<KeyCode>>,
    // OPTIONAL: the queue only exists when the remote server is running,
    // and the keybind must work without it. Requiring it panicked the
    // whole schedule on every plain `cargo run` — the remote was on in
    // every test I did, so nothing caught it.
    host: Option<ResMut<voxel_debug::remote::HostCommands>>,
    camera_world: Res<voxel_render::CameraWorld>,
    mut loader: WorldLoader,
    scene: Res<crate::HostScene>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials_assets: ResMut<Assets<StandardMaterial>>,
) {
    let asked = keys.just_pressed(KeyCode::F7)
        || host.is_some_and(|mut host| {
            let n = host.0.len();
            host.0
                .retain(|c| c.get("cmd").and_then(|c| c.as_str()) != Some("portal"));
            host.0.len() != n
        });
    if !asked {
        return;
    }
    let Ok(eye) = camera.single() else {
        return;
    };
    let Some(far_world) = ensure_far_world(&mut far, &mut loader) else {
        return;
    };

    let forward = eye.forward().as_vec3();
    let at = eye.translation() + forward * 14.0;
    let placement = Transform::from_translation(at).looking_to(-forward, Vec3::Y);

    // Already open: MOVE the opening ON THE SIDE YOU ARE STANDING ON.
    //
    // A portal has a placement per world. Moving `near` whatever world
    // the camera is in put world 0's opening at a coordinate in world 1
    // — which is why pressing F7 after stepping through logged "portal
    // moved" and produced nothing you could see.
    //
    // Never stack a second one: two portals make `portals.single()` fail,
    // which silently stops the far view and traversal dead rather than
    // erroring. Despawning strays is self-healing, where `single_mut()`
    // would itself fail once a duplicate existed and leave it wedged.
    let mut open = portal.iter_mut();
    if let Some((_, mut existing)) = open.next() {
        if camera_world.0 == existing.far_world {
            existing.far = placement;
        } else {
            existing.near = placement;
            existing.near_world = camera_world.0;
        }
        for (stray, _) in open {
            warn!("portal: despawning a duplicate opening");
            commands.entity(stray).despawn();
        }
        info!("portal moved to {at:?} in world {}", camera_world.0);
        return;
    }

    // The opening is cut in the world the camera is standing in.
    let near_world = camera_world.0;
    commands.spawn(Portal {
        near: placement,
        // Room to stand inside the far level.
        far: Transform::from_translation(Vec3::new(0.0, 22.0, 0.0)),
        near_world,
        far_world,
        half: Vec2::new(4.0, 3.0),
    });
    let mut backdrop = |color: Color| {
        materials_assets.add(StandardMaterial {
            base_color: color,
            unlit: true,
            // Both faces: an opening is approachable from either side,
            // and a one-sided quad silently vanishes from whichever side
            // it is not facing — which reads as "the backdrop is gone".
            double_sided: true,
            cull_mode: None,
            ..default()
        })
    };
    // Each side is painted with the background of the world it looks INTO.
    commands.insert_resource(PortalAssets {
        quad: meshes.add(Rectangle::new(8.0, 6.0)),
        material: [backdrop(far.scene.clear_color), backdrop(scene.0.clear_color)],
    });
    info!("portal opened at {at:?} in world {near_world}");
}

/// Keep one backdrop quad per side, in that side's world.
///
/// Synced every frame rather than moved when the portal moves: the quads
/// then follow whatever the portal says, including the side that did not
/// exist yet, and a stray or missing one heals itself instead of leaving
/// an opening you cannot see.
fn sync_backdrops(
    mut commands: Commands,
    portals: Query<&Portal>,
    assets: Option<Res<PortalAssets>>,
    mut quads: Query<(Entity, &PortalBackdrop, &mut Transform)>,
) {
    let (Ok(portal), Some(assets)) = (portals.single(), assets) else {
        return;
    };
    let sides = [
        (portal.near_world, portal.near, 0usize),
        (portal.far_world, portal.far, 1usize),
    ];
    for (world, placement, material) in sides {
        match quads.iter_mut().find(|(_, b, _)| b.world == world) {
            Some((_, _, mut transform)) => {
                if *transform != placement {
                    *transform = placement;
                }
            }
            None => {
                commands.spawn((
                    PortalBackdrop { world },
                    Mesh3d(assets.quad.clone()),
                    MeshMaterial3d(assets.material[material].clone()),
                    placement,
                    voxel_render::world_layer(world),
                ));
            }
        }
    }
    // A side that moved to another world leaves its old quad behind.
    for (entity, backdrop, _) in &quads {
        if backdrop.world != portal.near_world && backdrop.world != portal.far_world {
            commands.entity(entity).despawn();
        }
    }
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
    mut render_worlds: ResMut<voxel_render::RenderWorlds>,
    mut focus: ResMut<voxel_engine::WorldFocus>,
    camera_world: Res<voxel_render::CameraWorld>,
) {
    // Republished when it changes, NOT every frame: taking `RenderWorlds`
    // mutably marks it changed, and everything downstream that reacts to
    // "a world changed" then reacts every frame. That cost the terrain its
    // material bind group and the ground with it.
    if render_worlds.iter().any(|w| !w.clip.is_empty()) {
        render_worlds.clear_clips();
    }
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
    let motion = Portal::motion(&from, &to);

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
                cam.is_active = far_view_enabled();
                let mut cmd = commands.entity(entity);
                cmd.insert(voxel_render::ViewWorld(showing))
                    .insert(voxel_render::far_view_layers(showing));
                if let Some(target) = target {
                    cmd.insert(target.clone());
                }
            }
            None => {
                if !far_view_enabled() {
                    continue;
                }
                let mut spawned = commands.spawn((
                    PortalCamera(source),
                    // "Not the player camera." Without it the streamer
                    // can pick THIS one as the eye, and generation
                    // priority is then computed from a point in the other
                    // world — the chunks under your feet go to the back
                    // of the queue and the ground appears to break.
                    voxel_render::HelperCamera,
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
                    // Terrain and the shown world's lights: the clip
                    // planes mask chunks, and nothing masks entities.
                    voxel_render::far_view_layers(showing),
                    // No second tonemap. Cameras sharing a target each run
                    // their own post-processing over the WHOLE image, so a
                    // portal view re-tonemapped the near world that was
                    // already composited — the ground washed out to pale
                    // grey the moment a portal opened.
                    bevy::core_pipeline::tonemapping::Tonemapping::None,
                    bevy::core_pipeline::tonemapping::DebandDither::Disabled,
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
    if render_worlds.clip(showing) != planes {
        if let Some(world) = render_worlds.get_mut(showing) {
            world.clip = planes;
        }
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

    let motion = Portal::motion(&from, &to);
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
