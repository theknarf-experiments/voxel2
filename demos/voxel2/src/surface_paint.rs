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
use voxel_render::SurfaceMap;

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

/// Rasterizes planning ribbons into the ground's material.
pub struct SurfacePaintPlugin;

impl Plugin for SurfacePaintPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, repaint);
    }
}

fn repaint(
    world: Res<voxel_engine::WorldQuery>,
    sources: voxel_engine::StreamSourceQuery,
    mut map: ResMut<SurfaceMap>,
    mut painted_at: Local<Option<Vec2>>,
    mut seen: Local<u64>,
) {
    let Ok(source) = sources.single() else {
        return;
    };
    let Some(ctx) = world.host_ctx::<WorldCtx>() else {
        return;
    };
    let eye = source.translation();
    let eye = Vec2::new(eye.x, eye.z);
    // Repaint when the camera leaves the middle of the raster OR when the
    // plan under it changed. Camera movement alone is not enough: at
    // startup nothing is resident yet, so the first paint would find an
    // empty world and never look again.
    let generation = ctx.ribbons.generation();
    let moved = painted_at.is_none_or(|at| at.distance(eye) > REPAINT_M);
    if !moved && generation == *seen {
        return;
    }
    *painted_at = Some(eye);
    *seen = generation;

    let span = MAP_SIZE as f32 * TEXEL_M;
    let origin = eye - Vec2::splat(span * 0.5);
    let mut texels = vec![0u32; (MAP_SIZE * MAP_SIZE / 4) as usize];
    let mut painted = 0usize;
    // Read the plan, not a render buffer: these carry the material the
    // level asked for and whether they are ground at all. Introspection,
    // so the empty distance is not charged to `reads_missed`.
    let _peek = world.peek();
    for seg in world.ribbons_in(origin, origin + Vec2::splat(span)) {
        // Only seated ribbons ARE the ground. A water course has its own
        // level and its own surface; painting it here would smear a river
        // across the hillside it flows past.
        if !seg.seated {
            continue;
        }
        painted += stroke(&mut texels, origin, seg.a, seg.b, seg.half_w, seg.material);
    }

    map.texels = std::sync::Arc::new(texels);
    map.origin = origin;
    map.texel_m = TEXEL_M;
    map.size = MAP_SIZE;
    map.generation = map.generation.wrapping_add(1);
    if std::env::var_os("VOXEL_LOG_LAYERS").is_some() {
        info!("surface paint: {painted} texels around {eye:?}");
    }
}

/// Stamp one segment's footprint. Returns the texels written.
///
/// A capsule rather than a rectangle: consecutive segments of a path meet
/// at an angle, and rectangles leave a wedge of unpainted ground at every
/// corner of a road that turns.
fn stroke(
    texels: &mut [u32],
    origin: Vec2,
    a: Vec2,
    b: Vec2,
    half_w: f32,
    material: u32,
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
            let idx = (z * MAP_SIZE + x) as usize;
            let word = &mut texels[idx / 4];
            let shift = (idx % 4) * 8;
            *word = (*word & !(0xFFu32 << shift)) | ((material & 0xFF) << shift);
            n += 1;
        }
    }
    n
}
