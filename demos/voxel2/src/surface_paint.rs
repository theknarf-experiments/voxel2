//! Painting planned features onto the ground.
//!
//! A road is 4.8 m wide and 0.5 m THICK, so cutting it into the SDF stops
//! doing anything about 100 m from the camera — the voxel is bigger than
//! the cut. The first answer we tried was to draw the road as its own
//! surface strip, which is wrong in a way worth writing down: a road is
//! not an object standing on the ground, it IS the ground. Separate
//! geometry has to guess which LOD surface it is lying on, fight it for
//! the depth buffer, and be drawn by somebody — three problems that all
//! exist only because the ground was not drawing it.
//!
//! So the ground draws it. This rasterizes the ribbons the planning
//! layers produced into [`SurfaceMap`], and the mesh pass paints up-facing
//! vertices from it: one texture fetch per vertex, independent of how
//! many roads there are, where serving them as CSG ops costs a loop per
//! density SAMPLE.
//!
//! The raster follows the camera and is rebuilt when it leaves the middle
//! of it — on the order of once per kilometre, never per frame.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::planning::world::WorldCtx;

/// Texels per side. 2048 at [`TEXEL_M`] covers 32.7 km, which is 4 MB of
/// material ids — the whole visible near and middle field for the price
/// of one mid-sized texture.
const MAP_SIZE: u32 = 2048;

/// Meters per texel. A 52 m highway is three texels across, which is
/// enough to read as a line at range and is the scale the feature exists
/// at anyway: it was never going to be sharp at 20 km.
const TEXEL_M: f32 = 16.0;

/// Rebuild once the camera is this far from the raster's middle. The map
/// covers far more than it needs to so this can be rare.
const REPAINT_M: f32 = 4096.0;

/// How far above a water surface the ground may still count as covered by
/// it — how wide the course spreads, not whether it is drawn at all.
///
/// It cannot be zero. A course's level is set at `ground - 0.35` as the
/// descent walks, so the ground along the centre line is always slightly
/// ABOVE its own water surface, and the bed that would sink it is carved
/// by the very ops this map exists because they stopped being served.
///
/// Beyond that it stays tight. Widening it to cover the texel's sampling
/// error instead was the wrong repair: it does not reconnect a dashed
/// course (the gaps are where the bank is steep, which no tolerance
/// reaches without also flooding), and it smears gentle ground into
/// lakes. Connectivity comes from the centre-line walk.
const BANK_M: f32 = 1.0;

/// Detail level for the ground samples. Matches the `flow` layer, so a
/// course's centre line compares against the same heightfield that
/// produced its levels. (The generator is unbanded, so this changes
/// nothing today; it stays as the statement of intent.)
const GROUND_SAMPLE_M: f32 = 8.0;

/// Paint only chunks at least this coarse.
///
/// A road is 4.8 m wide, so a chunk whose voxels are finer than about
/// half that still carries the carved road as real geometry, at a detail
/// the 16 m texels of this map cannot approach. 3.2 m is level 5, which
/// the LOD field puts 256 m out — near enough that the carve is doing the
/// work, far enough that it has stopped resolving.
const PAINT_FROM_VOXEL_M: f32 = 3.2;

/// Rasterizes planning ribbons into the ground's material.
pub struct SurfacePaintPlugin;

impl Plugin for SurfacePaintPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, repaint);
    }
}

/// The voxel size at which the paint takes over from a levelled ribbon's
/// own surface, derived so the two cannot drift apart.
///
/// The LOD field shows level L over `[split_k·E_L, 2·split_k·E_L)`, so the
/// level that begins at the ribbon layer's view distance is the first one
/// that layer no longer covers — exactly where the paint must start.
fn levelled_handover_voxel_m(lod: &voxel_engine::streaming::LodConfig) -> f32 {
    let view = f64::from(crate::ribbons::RIBBON_NEAR_VIEW_M);
    (0..=lod.max_level)
        .map(|level| voxel_core::ChunkKey::new(level, IVec3::ZERO))
        .find(|key| lod.split_k * key.edge_m() >= view)
        .map_or(PAINT_FROM_VOXEL_M, |key| key.voxel_size_m() as f32)
}

/// Repaint every loaded world's ground map.
///
/// Every world, because the map is world content: it is indexed by
/// world-space xz and says nothing about which world it belongs to, so it
/// is held BY the world it paints. Painting only the launched one left
/// every other level's roads and river beds unpainted past the distance
/// their own geometry stops resolving.
fn repaint(
    worlds: Res<voxel_engine::Worlds>,
    sources: voxel_engine::StreamSourceQuery,
    mut render: ResMut<voxel_render::RenderWorlds>,
    mut painted: Local<HashMap<voxel_engine::WorldId, (Vec2, u64)>>,
) {
    let Ok(source) = sources.single() else {
        return;
    };
    for world in worlds.iter() {
        repaint_world(world, source.translation(), &mut render, &mut painted);
    }
}

fn repaint_world(
    world: &voxel_engine::World,
    eye: Vec3,
    render: &mut voxel_render::RenderWorlds,
    painted: &mut HashMap<voxel_engine::WorldId, (Vec2, u64)>,
) {
    let Some(map) = render.get_mut(world.id).map(|w| &mut w.surface_map) else {
        return;
    };
    let lod = &world.config;
    let Some(ctx) = world.query.host_ctx::<WorldCtx>() else {
        return;
    };
    let eye = Vec2::new(eye.x, eye.z);
    // Repaint when the camera leaves the middle of the raster OR when the
    // plan under it changed. Camera movement alone is not enough: at
    // startup nothing is resident yet, so the first paint would find an
    // empty world and never look again.
    let generation = ctx.ribbons.generation();
    let last = painted.get(&world.id).copied();
    let moved = last.is_none_or(|(at, _)| at.distance(eye) > REPAINT_M);
    if !moved && last.is_some_and(|(_, seen)| seen == generation) {
        return;
    }
    painted.insert(world.id, (eye, generation));

    let span = MAP_SIZE as f32 * TEXEL_M;
    let origin = eye - Vec2::splat(span * 0.5);
    let mut texels = vec![0u32; (MAP_SIZE * MAP_SIZE / 4) as usize];
    let mut painted = 0usize;
    let mut coarse_from: Vec<(u32, f32)> = Vec::new();
    // Read the plan, not a render buffer: these carry the material the
    // level asked for and whether they are ground at all. Introspection,
    // so the empty distance is not charged to `reads_missed`.
    let _peek = world.query.peek();
    for seg in world.query.ribbons_in(origin, origin + Vec2::splat(span)) {
        // A seated ribbon IS the ground, so its footprint is the whole
        // capsule. A levelled one is a water surface at a height the plan
        // decided: it covers its own course, and then only as much of the
        // surrounding ground as sits under that height — otherwise the
        // capsule smears the river across the hillside it flows past.
        let under = (!seg.seated).then_some(seg.levels);
        // A levelled material is drawn as a surface by the ribbon layer
        // too, so its paint must wait until that layer has stopped. Taken
        // from the plan rather than named here: the level decides which
        // materials are water, this only notices which ones arrive levelled.
        if under.is_some() && !coarse_from.iter().any(|&(id, _)| id == seg.material) {
            coarse_from.push((seg.material, levelled_handover_voxel_m(lod)));
        }
        painted += stroke(
            &mut texels,
            origin,
            seg.a,
            seg.b,
            seg.half_w,
            seg.material,
            under.map(|levels| (levels, &*ctx.generator)),
        );
    }

    map.texels = std::sync::Arc::new(texels);
    map.origin = origin;
    map.texel_m = TEXEL_M;
    map.size = MAP_SIZE;
    map.min_voxel_m = PAINT_FROM_VOXEL_M;
    map.coarse_from = coarse_from;
    map.generation = map.generation.wrapping_add(1);
    if std::env::var_os("VOXEL_LOG_LAYERS").is_some() {
        info!(
            "surface paint: world {} {painted} texels around {eye:?}",
            world.id
        );
    }
}

/// Stamp one segment's footprint. Returns the texels written.
///
/// A capsule rather than a rectangle: consecutive segments of a path meet
/// at an angle, and rectangles leave a wedge of unpainted ground at every
/// corner of a road that turns.
///
/// `under` makes the stamp conditional on the ground lying beneath a water
/// surface running from one end level to the other — what a levelled
/// ribbon needs and a seated one must not have.
fn stroke(
    texels: &mut [u32],
    origin: Vec2,
    a: Vec2,
    b: Vec2,
    half_w: f32,
    material: u32,
    under: Option<([f32; 2], &voxel_worldgen::Generator)>,
) -> usize {
    let half_w = half_w.max(TEXEL_M * 0.5);
    let lo = a.min(b) - Vec2::splat(half_w);
    let hi = a.max(b) + Vec2::splat(half_w);
    let to_texel = |p: Vec2| (p - origin) / TEXEL_M;
    let (t0, t1) = (to_texel(lo), to_texel(hi));
    let x0 = (t0.x.floor().max(0.0)) as u32;
    let z0 = (t0.y.floor().max(0.0)) as u32;
    let x1 = (t1.x.ceil().min(MAP_SIZE as f32 - 1.0)) as u32;
    let z1 = (t1.y.ceil().min(MAP_SIZE as f32 - 1.0)) as u32;
    if x1 < x0 || z1 < z0 {
        return 0;
    }
    let ab = b - a;
    let len2 = ab.length_squared().max(1e-6);
    let mut n = 0;
    for z in z0..=z1 {
        for x in x0..=x1 {
            let p = origin + Vec2::new(x as f32 + 0.5, z as f32 + 0.5) * TEXEL_M;
            let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
            if p.distance_squared(a + ab * t) > half_w * half_w {
                continue;
            }
            // Only where the water reaches. Sampled per texel, not per
            // segment: a course runs down a valley whose width varies, and
            // that variation is the whole difference between a river and a
            // blue stripe ruled across the landscape.
            if let Some((levels, generator)) = under {
                let level = levels[0] + (levels[1] - levels[0]) * t;
                if generator.height(p, GROUND_SAMPLE_M) > level + BANK_M {
                    continue;
                }
            }
            put(texels, x, z, material);
            n += 1;
        }
    }
    // A course must stay CONNECTED, and the width pass cannot promise
    // that: it samples texel CENTRES, and a river a few metres wide runs
    // through 16 m texels whose centres sit up the bank and fail the
    // depth test. The result is a river drawn as a dashed line. So the
    // centre line is walked and painted for what it is — the texels the
    // water demonstrably runs through — and the width pass only adds the
    // ground around it that the water also covers.
    if under.is_some() {
        let steps = (a.distance(b) / (TEXEL_M * 0.5)).ceil().max(1.0) as usize;
        for i in 0..=steps {
            let p = a.lerp(b, i as f32 / steps as f32);
            let t = (p - origin) / TEXEL_M;
            if t.x < 0.0 || t.y < 0.0 {
                continue;
            }
            let (x, z) = (t.x as u32, t.y as u32);
            if x < MAP_SIZE && z < MAP_SIZE {
                put(texels, x, z, material);
                n += 1;
            }
        }
    }
    n
}

/// Write one texel's material id into the packed 4-per-word raster.
fn put(texels: &mut [u32], x: u32, z: u32, material: u32) {
    let idx = (z * MAP_SIZE + x) as usize;
    let word = &mut texels[idx / 4];
    let shift = (idx % 4) * 8;
    *word = (*word & !(0xFFu32 << shift)) | ((material & 0xFF) << shift);
}
