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

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::camera::visibility::RenderLayers;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
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
        embedded_asset!(app, "voxel_portal.wgsl");
        app.add_plugins(MaterialPlugin::<PortalViewMaterial>::default())
            .add_systems(Update, size_portal_target);
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

}

/// The view the far world is rendered FOR: the player's.
///
/// One, not every near-side camera. The far view is an image, and an
/// image is a viewpoint — pairing one with each camera meant several
/// cameras rendering into the same target at the same order, which Bevy
/// reports as an order ambiguity and resolves arbitrarily.
///
/// Other views of the same viewpoint share it correctly: the offscreen
/// mirror `voxctl shot` renders through copies the player camera's
/// transform, and the opening samples by NORMALIZED screen position, so
/// its 1280x720 reads the same pixels out of a 2560x1440 far image. A
/// view from somewhere else would sample the player's far view and be
/// wrong — there is only one portal viewpoint, and this is it.
type NearViews<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static GlobalTransform, &'static Camera),
    (With<Camera3d>, Without<PortalCamera>, Without<voxel_render::HelperCamera>),
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

/// The opening's surface: the far view, sampled in screen space.
#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct PortalViewMaterial {
    #[texture(0)]
    #[sampler(1)]
    far_view: Handle<Image>,
}

impl Material for PortalViewMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://voxel2/voxel_portal.wgsl".into()
    }

    /// BOTH FACES. An opening is approachable from either side, and a
    /// quad culled from behind silently vanishes from whichever side it
    /// is not facing — which reads as "the portal does not render". The
    /// backdrop this replaced set `cull_mode: None` on its
    /// `StandardMaterial` for the same reason; a custom material gets
    /// Bevy's default of back-face culling unless it says otherwise.
    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Shared handles for the opening's quads, made once.
#[derive(Resource)]
struct PortalAssets {
    quad: Handle<Mesh>,
    /// What the far camera renders into, and what the opening samples.
    /// One image, because only one view of the far world exists at a
    /// time; a second portal onto the same world would need its own.
    target: Handle<Image>,
    material: Handle<PortalViewMaterial>,
}

/// Size the far view's target to the window, and keep it there.
///
/// Sampling is by fragment position, so the two images must be the same
/// size or the far world slides against the opening as the window
/// changes.
fn size_portal_target(
    assets: Option<Res<PortalAssets>>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
) {
    let (Some(assets), Ok(window)) = (assets, windows.single()) else {
        return;
    };
    let want = bevy::render::render_resource::Extent3d {
        width: window.physical_width().max(1),
        height: window.physical_height().max(1),
        depth_or_array_layers: 1,
    };
    let Some(mut image) = images.get_mut(&assets.target) else {
        return;
    };
    if image.texture_descriptor.size != want {
        image.resize(want);
    }
}

fn far_view_image(width: u32, height: u32) -> Image {
    use bevy::render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureUsages,
    };
    let mut image = Image::new_fill(
        Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::RENDER_ATTACHMENT;
    image
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
    mut images: ResMut<Assets<Image>>,
    mut portal_materials: ResMut<Assets<PortalViewMaterial>>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
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
    let size = windows.single().map_or((1280, 720), |w| {
        (w.physical_width(), w.physical_height())
    });
    let target = images.add(far_view_image(size.0, size.1));
    commands.insert_resource(PortalAssets {
        quad: meshes.add(Rectangle::new(8.0, 6.0)),
        material: portal_materials.add(PortalViewMaterial {
            far_view: target.clone(),
        }),
        target,
    });
    info!("portal opened at {at:?} in world {near_world}");
}

/// Keep one surface per side, in that side's world.
///
/// Synced every frame rather than moved when the portal moves, so the
/// side that did not exist yet gets one and a stray heals itself.
///
/// Both sides share one material, and so one image: only one far view
/// exists at a time — whichever world the camera is NOT in. A second
/// portal onto a third world would need a target of its own.
fn sync_backdrops(
    mut commands: Commands,
    portals: Query<&Portal>,
    assets: Option<Res<PortalAssets>>,
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
                commands.spawn((
                    PortalBackdrop { world },
                    Mesh3d(assets.quad.clone()),
                    MeshMaterial3d(assets.material.clone()),
                    placement,
                    voxel_render::world_layer(world),
                ));
            }
        }
    }
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
#[allow(clippy::too_many_arguments)]
fn drive_portal(
    mut commands: Commands,
    portals: Query<&Portal>,
    sources: NearViews,
    mut portal_cams: Query<(Entity, &PortalCamera, &mut Transform, &mut Camera)>,
    mut render_worlds: ResMut<voxel_render::RenderWorlds>,
    mut focus: ResMut<voxel_engine::WorldFocus>,
    camera_world: Res<voxel_render::CameraWorld>,
    assets: Option<Res<PortalAssets>>,
    scenes: Res<crate::WorldScenes>,
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

    let Some(assets) = assets else {
        return;
    };
    let sky = scenes
        .0
        .get(&showing)
        .map_or(Color::BLACK, |s| s.clear_color);

    for (source, eye, camera) in &sources {
        let existing = portal_cams
            .iter_mut()
            .find(|(_, paired, _, _)| paired.0 == source);
        // The SAME eye. Both worlds occupy the same coordinates, so the
        // view of the other one through the opening IS this view — which
        // is what lets the opening sample it by fragment position and
        // line up pixel for pixel, with nothing to tune.
        let placement = Transform::from_matrix(eye.to_matrix());
        match existing {
            Some((entity, _, mut transform, mut cam)) => {
                *transform = placement;
                // BEFORE the view it pairs, so the image is ready when
                // the opening samples it.
                cam.order = camera.order - 1;
                cam.is_active = far_view_enabled();
                cam.clear_color = bevy::camera::ClearColorConfig::Custom(sky);
                commands
                    .entity(entity)
                    .insert(voxel_render::ViewWorld(showing))
                    // The WHOLE world, scene content included. The image
                    // is confined to the opening by the quad that samples
                    // it, so nothing here has to be clipped.
                    .insert(voxel_render::world_layer(showing));
            }
            None => {
                if !far_view_enabled() {
                    continue;
                }
                commands.spawn((
                    PortalCamera(source),
                    // "Not the player camera." Without it the streamer
                    // can pick THIS one as the eye, and generation
                    // priority is computed from the wrong view.
                    voxel_render::HelperCamera,
                    Camera3d::default(),
                    Camera {
                        order: camera.order - 1,
                        // The far world's own sky, wherever it is empty.
                        clear_color: bevy::camera::ClearColorConfig::Custom(sky),
                        ..default()
                    },
                    bevy::camera::RenderTarget::Image(assets.target.clone().into()),
                    placement,
                    voxel_render::ViewWorld(showing),
                    voxel_render::world_layer(showing),
                ));
            }
        }
    }

    // The only mask left is the opening's own PLANE: the far world must
    // appear beyond the opening, not between it and the eye. The four
    // pyramid planes are gone — the quad does that, exactly, for
    // everything the far camera drew rather than for chunks alone.
    let Some((_, eye, _)) = sources.iter().next() else {
        return;
    };
    let eye = eye.translation();
    focus.0.clear();
    focus.0.resize(voxel_render::MAX_WORLDS, None);
    let mut ahead = portal.at.rotation * Vec3::Z;
    if ahead.dot(portal.at.translation - eye) < 0.0 {
        ahead = -ahead;
    }
    let planes = vec![ahead.extend(-ahead.dot(portal.at.translation))];
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
