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
//!
//! So a portal is one rectangle at one place, open in two worlds at once.
//! Crossing it changes which world you are in and NOTHING else: same
//! position, same heading, same velocity. There is no pairing transform
//! because there is nothing to pair — both openings are the same opening.

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
                    dress_views,
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
fn ensure_far_world(
    far: &mut FarLevel,
    loader: &mut WorldLoader,
    scenes: &mut crate::WorldScenes,
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

    // At its authored detail. There used to be a hand-picked cap here,
    // because a second world roughly doubles the meshed working set and
    // the slab is one fixed allocation: every class hit 100% and chunks
    // wedged waiting for space. `WorldLoader::load` now admits a world
    // against the slab slots the loaded ones left, and caps it only as
    // far as it has to — which is the mechanism that cap was guessing at.
    let id = loader.load(level.clone(), 0, LodConfig::from(&level.lod));
    scenes.0.insert(id, far.scene.clone());
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
    mut clear: ResMut<ClearColor>,
    scenes: Res<crate::WorldScenes>,
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
    if let Some(here) = scenes.0.get(&camera_world.0) {
        if clear.0 != here.clear_color {
            clear.0 = here.clear_color;
        }
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
/// Give every view the ambient and the haze of the world IT looks at.
///
/// Per camera, not global. Ambient and fog are properties of the world
/// being looked at, and a portal puts two worlds in one frame: the near
/// view of a sunless interior wants its own bright ambient and no haze,
/// while the far view of a planet through the same opening wants the
/// planet's. Sharing one value lit the planet's ground at the interior's
/// ambient — which is why it looked washed out through the opening — and
/// dropped its atmospheric haze entirely.
///
/// Bevy's `AmbientLight` is a camera component that overrides
/// `GlobalAmbientLight`, so this needs no engine support.
fn dress_views(
    mut commands: Commands,
    scenes: Res<crate::WorldScenes>,
    views: Query<(Entity, Option<&voxel_render::ViewWorld>), With<Camera3d>>,
) {
    for (entity, world) in &views {
        let Some(scene) = scenes.0.get(&world.map_or(0, |w| w.0)) else {
            continue;
        };
        let mut view = commands.entity(entity);
        view.insert(AmbientLight {
            color: scene.ambient_color,
            brightness: scene.ambient_brightness,
            ..default()
        });
        match &scene.fog {
            Some(fog) => {
                view.insert(fog.clone());
            }
            None => {
                view.remove::<DistanceFog>();
            }
        }
    }
}

/// A rectangular opening between two dimensions of the SAME space.
///
/// Worlds share coordinates — that is what lets one chunk service and one
/// GPU arena serve all of them, with the world riding in `ChunkKey`. So a
/// portal is one rectangle at one place, open in two worlds at once, and
/// crossing it changes WHICH world you are in and nothing else. No
/// displacement, no rotation, no pairing transform.
///
/// There used to be a placement per side, with the far one hardcoded to
/// the world origin. Since the planet's demo starts 46 km out, stepping
/// through teleported you 46 km — which then had to be undone by a
/// pairing matrix, a half turn to come out the right side, and a separate
/// focus point per world so the far side streamed somewhere other than
/// where you were. All of that was machinery for a coordinate jump that
/// should never have existed.
#[derive(Component, Clone, Copy)]
pub struct Portal {
    /// Where the opening is, in the coordinates both worlds share.
    pub at: Transform,
    /// The two worlds it joins.
    pub worlds: (u8, u8),
    /// Half width and half height of the opening, in metres.
    pub half: Vec2,
}

impl Portal {
    /// The world on the other side from `world`, or `None` if this portal
    /// does not touch it.
    fn other(&self, world: u8) -> Option<u8> {
        match world {
            w if w == self.worlds.0 => Some(self.worlds.1),
            w if w == self.worlds.1 => Some(self.worlds.0),
            _ => None,
        }
    }

    /// The opening's four corners.
    fn corners(&self) -> [Vec3; 4] {
        let x = self.at.rotation * Vec3::X * self.half.x;
        let y = self.at.rotation * Vec3::Y * self.half.y;
        [
            self.at.translation - x - y,
            self.at.translation + x - y,
            self.at.translation + x + y,
            self.at.translation - x + y,
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
    mut scenes: ResMut<crate::WorldScenes>,
    mut meshes: ResMut<Assets<Mesh>>,
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
    let Some(far_world) = ensure_far_world(&mut far, &mut loader, &mut scenes) else {
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
        existing.at = placement;
        existing.worlds = (camera_world.0, far_world);
        for (stray, _) in open {
            warn!("portal: despawning a duplicate opening");
            commands.entity(stray).despawn();
        }
        info!("portal moved to {at:?} in world {}", camera_world.0);
        return;
    }

    let near_world = camera_world.0;
    commands.spawn(Portal {
        at: placement,
        worlds: (near_world, far_world),
        half: Vec2::new(4.0, 3.0),
    });
    commands.insert_resource(PortalAssets {
        quad: meshes.add(Rectangle::new(8.0, 6.0)),
    });
    info!("portal opened at {at:?} in world {near_world}");
}

/// Keep one backdrop quad per side, in that side's world.
///
/// Synced every frame rather than moved when the portal moves: the quads
/// then follow whatever the portal says, including the side that did not
/// exist yet, and a stray or missing one heals itself instead of leaving
/// an opening you cannot see.
#[allow(clippy::too_many_arguments)]
fn sync_backdrops(
    mut commands: Commands,
    portals: Query<&Portal>,
    assets: Option<Res<PortalAssets>>,
    scenes: Res<crate::WorldScenes>,
    mut materials_assets: ResMut<Assets<StandardMaterial>>,
    mut quads: Query<(Entity, &PortalBackdrop, &mut Transform)>,
) {
    let (Ok(portal), Some(assets)) = (portals.single(), assets) else {
        return;
    };
    let placement = portal.at;
    for world in [portal.worlds.0, portal.worlds.1] {
        match quads.iter_mut().find(|(_, b, _)| b.world == world) {
            Some((_, _, mut transform)) => {
                if *transform != placement {
                    *transform = placement;
                }
            }
            None => {
                // Painted with the background of the world it looks INTO,
                // looked up by that world — not by which side of the pair
                // it happens to be, which is only right when the portal
                // was opened from world 0.
                let into = portal.other(world).and_then(|w| scenes.0.get(&w));
                let material = materials_assets.add(StandardMaterial {
                    base_color: into.map_or(Color::BLACK, |s| s.clear_color),
                    unlit: true,
                    // Both faces: an opening is approachable from either
                    // side, and a one-sided quad silently vanishes from
                    // whichever side it is not facing.
                    double_sided: true,
                    cull_mode: None,
                    ..default()
                });
                commands.spawn((
                    PortalBackdrop { world },
                    Mesh3d(assets.quad.clone()),
                    MeshMaterial3d(material),
                    placement,
                    voxel_render::world_layer(world),
                ));
            }
        }
    }
    // A side that moved to another world leaves its old quad behind.
    for (entity, backdrop, _) in &quads {
        if portal.other(backdrop.world).is_none() {
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
    let Some(showing) = portal.other(camera_world.0) else {
        return; // this portal does not touch the world you are in
    };

    for (source, eye, camera, target) in &sources {
        let existing = portal_cams
            .iter_mut()
            .find(|(_, paired, _, _)| paired.0 == source);
        // The SAME eye. Both worlds occupy the same coordinates, so the
        // view of the other one through the opening is this view — there
        // is nothing to transform and so nothing that can fail to line up
        // at the edges.
        let placement = Transform::from_matrix(eye.to_matrix());
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
                    // priority is computed from the wrong view.
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

    // The mask: the pyramid from the eye through the opening, plus the
    // opening's own plane so nothing between it and the eye leaks in.
    // One set of coordinates, so these are the eye and the opening as they
    // already are.
    let Some((_, eye, _, _)) = sources.iter().next() else {
        return;
    };
    let eye = eye.translation();
    // The far world streams around the SAME point the camera is at: it is
    // the same place, in another dimension. Nothing to relocate.
    focus.0.clear();
    focus.0.resize(voxel_render::MAX_WORLDS, None);
    let corners = portal.corners();
    let mut planes = Vec::with_capacity(voxel_render::MAX_CLIP_PLANES);
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let mut n = (a - eye).cross(b - eye).normalize_or_zero();
        if n.dot(corners[(i + 2) % 4] - a) < 0.0 {
            n = -n;
        }
        planes.push(n.extend(-n.dot(a)));
    }
    let mut ahead = portal.at.rotation * Vec3::Z;
    if ahead.dot(portal.at.translation - eye) < 0.0 {
        ahead = -ahead;
    }
    planes.push(ahead.extend(-ahead.dot(portal.at.translation)));
    if render_worlds.clip(showing) != planes {
        if let Some(world) = render_worlds.get_mut(showing) {
            world.clip = planes;
        }
    }
}

/// Step through: crossing the opening changes which world you are in,
/// and nothing else.
///
/// No displacement and no rotation, because there is nothing to displace
/// to — the two worlds occupy the same coordinates and the opening is one
/// rectangle open in both. You keep your position, your velocity and your
/// heading; only the dimension you are in changes.
///
/// Tested against the SEGMENT the camera travelled this frame, not
/// against which side it is on now. At walking speed a frame covers
/// centimetres, but a fast flight covers tens of metres and would step
/// straight over a 3 m opening between two samples — the portal would
/// work until you approached it quickly, which is the worst way for it to
/// fail.
fn traverse_portal(
    portals: Query<&Portal>,
    camera: Query<&Transform, With<crate::FreeCamera>>,
    mut camera_world: ResMut<voxel_render::CameraWorld>,
    mut was_at: Local<Option<Vec3>>,
) {
    let (Ok(portal), Ok(transform)) = (portals.single(), camera.single()) else {
        return;
    };
    let now = transform.translation;
    let Some(before) = was_at.replace(now) else {
        return;
    };
    let Some(other) = portal.other(camera_world.0) else {
        return;
    };

    // Signed distance to the opening's plane, before and after.
    let normal = portal.at.rotation * Vec3::Z;
    let plane_d = -normal.dot(portal.at.translation);
    let (d0, d1) = (normal.dot(before) + plane_d, normal.dot(now) + plane_d);
    if (d0 < 0.0) == (d1 < 0.0) {
        return; // did not cross
    }
    // Where it crossed, and whether that is inside the opening.
    let t = d0 / (d0 - d1);
    let hit = before.lerp(now, t.clamp(0.0, 1.0));
    let local = portal.at.rotation.inverse() * (hit - portal.at.translation);
    if local.x.abs() > portal.half.x || local.y.abs() > portal.half.y {
        return; // through the wall beside it, not the opening
    }

    camera_world.0 = other;
    info!("stepped into world {other}");
}
