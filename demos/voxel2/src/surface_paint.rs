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
//!
//! The same trade applies to a scattered population, one step further out.
//! A tree is drawn as a mesh, then as an impostor, and past the range where
//! even an impostor is a couple of pixels it is not a thing standing on the
//! ground either — it is what colour the ground IS. So the third tier is
//! painted here too, from [`voxel_engine::scatter::coverage`] rather than
//! from ribbons: an area, not a line.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::planning::world::WorldCtx;

/// Texels per side. 4096 at [`TEXEL_M`] covers 32.7 km, which is 16 MB of
/// material ids — the whole visible near and middle field for the price of
/// one large texture.
///
/// Only worlds that actually paint something carry a section this big (see
/// the end of `repaint_world`), so the cost is per PAINTING world, not per
/// loaded one. At `MAX_WORLDS` painting worlds it would be 128 MB, which
/// is exactly wgpu's default `max_storage_buffer_binding_size` — a ceiling
/// worth knowing about before a level ships with eight forested worlds
/// open at once.
const MAP_SIZE: u32 = 4096;

/// Meters per texel.
///
/// 16 m was chosen when the map held only ribbons, where it is generous: a
/// 52 m highway was three texels across. An AREA is a harder customer than
/// a line — a painted region has an EDGE, and a dithered edge on 16 m
/// texels reads as chunky noise at the distance the paint takes over
/// (4.8 px at 4 km). Halving it puts that edge at 2.4 px, which is the
/// scale the impostors it stands in for occupy.
const TEXEL_M: f32 = 8.0;

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

/// Narrowest a stroke is painted, whatever the plan says it is.
///
/// In METRES, not in texels. It used to be half a texel — enough that a
/// stroke could not miss — and that quietly meant "however wide the raster
/// happens to be", so halving [`TEXEL_M`] halved every river. What the
/// floor is really for is legibility: a 5 m stream painted at 5 m is
/// invisible at the kilometres this map is read at, and a river you cannot
/// see is not more accurate than one you can.
const MIN_STROKE_HALF_W_M: f32 = 8.0;

/// Paint only chunks at least this coarse.
///
/// A road is 4.8 m wide, so a chunk whose voxels are finer than about
/// half that still carries the carved road as real geometry, at a detail
/// the texels of this map cannot approach. 3.2 m is level 5, which
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

/// The first LOD level whose chunks are never shown nearer than `view_m` —
/// where paint has to take over from whoever was drawing the real thing
/// out to there.
///
/// The LOD field shows level L over `[split_k·E_L, 2·split_k·E_L)`, so this
/// is the first level that drawer no longer covers. The shader tests a
/// VOXEL SIZE and the host that keeps drawing has to know a DISTANCE; both
/// come from this one level, so the handover cannot be in two places.
fn handover_level(
    lod: &voxel_engine::streaming::LodConfig,
    view_m: f32,
) -> Option<voxel_core::ChunkKey> {
    (0..=lod.max_level)
        .map(|level| voxel_core::ChunkKey::new(level, IVec3::ZERO))
        .find(|key| lod.split_k * key.edge_m() >= f64::from(view_m))
}

fn handover_voxel_m(lod: &voxel_engine::streaming::LodConfig, view_m: f32) -> f32 {
    handover_level(lod, view_m).map_or(PAINT_FROM_VOXEL_M, |key| key.voxel_size_m() as f32)
}

/// The distance the paint actually begins at, which is further out than
/// the level asked for: paint is a per-CHUNK decision, so it can only
/// start where a whole LOD level does. Whoever draws the real thing has to
/// reach here, or a ring of ground shows neither.
pub fn cover_starts_m(lod: &voxel_engine::streaming::LodConfig, view_m: f32) -> f32 {
    handover_level(lod, view_m).map_or(view_m, |key| (lod.split_k * key.edge_m()) as f32)
}

/// Texels between coverage samples in the area pass.
///
/// 64 m, held there as [`TEXEL_M`] changes, because it is not a property
/// of the raster: it is the size of a VOXEL in the chunks that read the
/// paint. Paint only applies past `cover.from_m`, where the LOD field is
/// already showing 51–102 m voxels, so a coverage edge resolved finer than
/// that is drawn onto geometry that cannot express it.
///
/// **Measured, do not re-try.** Refining to 16 m inside a coarse mask (the
/// obvious optimisation, since most of a world holds no population) was
/// tried on the theory that coverage's fast terms — the altitude band and
/// the slope gate — follow the terrain rather than the noise, and so carry
/// detail a treeline would show. They do, and it does not matter: the
/// sweep went from 0.29 s to 3.3 s for a mean difference of 1.7/255 across
/// the frame, invisible at 5x zoom, and most of that was dither reshuffling
/// rather than a sharper edge. A third of the map is inside the mask
/// anyway, because the region gate's feather leaves a long tail of tiny
/// nonzero coverage, so the mask saves much less than it looks like it
/// should. What limits the far field now is the terrain's own LOD, not
/// this.
const COVER_STRIDE: u32 = (64.0 / TEXEL_M) as u32;

/// One world's last raster, and whatever is being built to replace it.
struct Painted {
    /// Where the raster is anchored — NOT the camera. A repaint driven by
    /// the plan changing does not re-sweep the area pass, and a raster
    /// whose origin moved out from under a kept cover layer would put the
    /// forest somewhere the forest is not.
    origin: Vec2,
    generation: u64,
    /// The area pass's output, kept so a ribbon arriving does not re-sweep
    /// it: coverage is a function of position and the camera, and ribbons
    /// arriving change neither.
    cover: Vec<u32>,
    coarse_from: Vec<(u32, f32)>,
    /// A raster being built, and the anchor it is for.
    building: Option<(Vec2, bevy::tasks::Task<PaintResult>)>,
}

/// What one build produces.
struct PaintResult {
    /// The area pass's layer, handed back so the next ribbon change can
    /// start from it instead of sweeping again.
    cover: Vec<u32>,
    texels: Vec<u32>,
    coarse_from: Vec<(u32, f32)>,
    /// Whether anything was painted at all. Counted where the painting
    /// happens: asking the finished raster costs a 16 MB scan, and asking
    /// it on the main thread costs one every time a ribbon arrives.
    any: bool,
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
    mut reloaded: MessageReader<voxel_engine::level::LevelReloaded>,
    mut render: ResMut<voxel_render::RenderWorlds>,
    mut painted: Local<HashMap<voxel_engine::WorldId, Painted>>,
) {
    // The area pass is cached against the camera's position, and a level
    // edit does not move the camera. Nothing else here would notice that
    // the population it swept no longer exists.
    if !reloaded.is_empty() {
        reloaded.clear();
        painted.clear();
    }
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
    painted: &mut HashMap<voxel_engine::WorldId, Painted>,
) {
    let Some(map) = render.get_mut(world.id).map(|w| &mut w.surface_map) else {
        return;
    };
    let lod = &world.config;
    let Some(ctx) = world.query.host_ctx::<WorldCtx>() else {
        return;
    };
    let eye = Vec2::new(eye.x, eye.z);
    let span = MAP_SIZE as f32 * TEXEL_M;
    let generation = ctx.ribbons.generation();
    // Where the raster would be anchored if it were rebuilt now.
    let want = eye - Vec2::splat(span * 0.5);

    // Read the plan, not a render buffer: these carry the material the
    // level asked for and whether they are ground at all. Introspection,
    // so the empty distance is not charged to `reads_missed`.
    let _peek = world.query.peek();
    // Its OWN planner, downcast — the engine has no ribbons to hand out.
    let Some(planner) = world.query.planner_as::<crate::planning::RegionPlanner>() else {
        return;
    };

    // `u64::MAX` is "never painted": a real ribbon generation counts up
    // from zero, so it cannot collide, and the first frame is then a move
    // without having to special-case one.
    let state = painted.entry(world.id).or_insert_with(|| Painted {
        origin: want,
        generation: u64::MAX,
        cover: vec![0; (MAP_SIZE * MAP_SIZE / 4) as usize],
        coarse_from: Vec::new(),
        building: None,
    });

    // Everything is built off the main thread, area pass and ribbons
    // alike. The ribbon pass looks cheap — stamp a few capsules — and is
    // not: a levelled ribbon tests the heightfield PER TEXEL to find how
    // far its water spreads, and at 8 m texels over a 32.7 km raster that
    // measured 49 ms on average and 75 ms at worst, every few seconds,
    // which is precisely the stutter you see while flying.
    //
    // Nothing waits for it. The raster decides ground a quarter of a
    // kilometre away at the very nearest, and until the new one lands the
    // last one is still right about almost all of it.
    if let Some((anchor, task)) = &mut state.building {
        let anchor = *anchor;
        if let Some(done) = bevy::tasks::block_on(bevy::tasks::poll_once(task)) {
            state.origin = anchor;
            state.cover = done.cover;
            state.coarse_from = done.coarse_from.clone();
            state.building = None;
            publish(map, done.texels, anchor, done.coarse_from, done.any);
        }
        return;
    }

    let moved = state.origin.distance(want) > REPAINT_M || state.generation == u64::MAX;
    if !moved && state.generation == generation {
        return;
    }
    state.generation = generation;
    let anchor = if moved { want } else { state.origin };

    // Snapshotting the ribbons is the one part that has to happen here:
    // they live in the planner's layer graph, which is not ours to send.
    let segments = planner.ribbons_in(anchor, anchor + Vec2::splat(span));
    let job = PaintJob {
        anchor,
        // A move invalidates the area pass; anything else reuses it.
        base: (!moved).then(|| state.cover.clone()),
        cover: CoverJob::of(world, planner, anchor),
        segments,
        generator: world.generator.clone(),
        levelled_from_voxel_m: handover_voxel_m(lod, crate::ribbons::RIBBON_NEAR_VIEW_M as f32),
    };
    state.building = Some((
        anchor,
        bevy::tasks::AsyncComputeTaskPool::get().spawn(async move { job.run() }),
    ));
}

/// Hand a finished raster to the renderer.
fn publish(
    map: &mut voxel_render::SurfaceMap,
    texels: Vec<u32>,
    origin: Vec2,
    coarse_from: Vec<(u32, f32)>,
    any: bool,
) {
    // A world that paints nothing publishes NOTHING, rather than a blank
    // section: at this resolution one is 16 MB, and a level with no roads
    // and no cover was carrying that so the shader could read zero out of
    // it. Size 0 is already how the map says "leave the terrain alone".
    let empty = !any && coarse_from.is_empty();
    map.texels = std::sync::Arc::new(texels);
    map.origin = origin;
    map.texel_m = TEXEL_M;
    map.size = if empty { 0 } else { MAP_SIZE };
    map.min_voxel_m = PAINT_FROM_VOXEL_M;
    map.coarse_from = coarse_from;
    map.generation = map.generation.wrapping_add(1);
}

/// Everything a raster is built from, owned, so it can run on a worker.
struct PaintJob {
    anchor: Vec2,
    /// The cover layer to start from, or `None` to sweep a fresh one.
    base: Option<Vec<u32>>,
    cover: CoverJob,
    segments: Vec<voxel_core::patch::RibbonSeg>,
    generator: std::sync::Arc<voxel_worldgen::Generator>,
    levelled_from_voxel_m: f32,
}

impl PaintJob {
    fn run(self) -> PaintResult {
        let (cover, mut coarse_from) = match self.base {
            Some(base) => (base, self.cover.thresholds()),
            None => self.cover.run(),
        };
        let mut texels = cover.clone();
        let mut painted = 0usize;
        for seg in &self.segments {
            // A seated ribbon IS the ground, so its footprint is the whole
            // capsule. A levelled one is a water surface at a height the
            // plan decided: it covers its own course, and then only as
            // much of the surrounding ground as sits under that height —
            // otherwise the capsule smears the river across the hillside
            // it flows past.
            let under = (!seg.seated).then_some(seg.levels);
            // A levelled material is drawn as a surface by the ribbon
            // layer too, so its paint must wait until that layer has
            // stopped. Taken from the plan rather than named here: the
            // level decides which materials are water, this only notices
            // which ones arrive levelled.
            if under.is_some() && !coarse_from.iter().any(|&(id, _)| id == seg.material) {
                coarse_from.push((seg.material, self.levelled_from_voxel_m));
            }
            painted += stroke(
                &mut texels,
                self.anchor,
                seg.a,
                seg.b,
                seg.half_w,
                seg.material,
                under.map(|levels| (levels, &*self.generator)),
            );
        }
        let any = painted > 0 || !coarse_from.is_empty();
        PaintResult {
            cover,
            texels,
            coarse_from,
            any,
        }
    }
}

/// Everything the area pass reads, owned, so it can run on a worker.
///
/// Gathered on the main thread because that is where the level and the
/// planner are; it is a handful of clones, against a fifth of a second of
/// sweeping.
struct CoverJob {
    origin: Vec2,
    generator: std::sync::Arc<voxel_worldgen::Generator>,
    /// Each population that paints, with the region material its placer
    /// gates on and the voxel size its paint takes over at.
    populations: Vec<(voxel_engine::level::ScatterDef, Option<u32>, f32)>,
}

impl CoverJob {
    fn of(
        world: &voxel_engine::World,
        planner: &crate::planning::RegionPlanner,
        origin: Vec2,
    ) -> Self {
        let populations = planner
            .populations
            .iter()
            .filter_map(|def| {
                let cover = def.cover.as_ref()?;
                // The population's own gate, resolved the way its placer
                // resolves it — what grows somewhere and what is painted
                // there are then the same question asked twice, not two
                // answers that must be kept equal by hand.
                let gate = planner.gate_material(def);
                Some((
                    def.clone(),
                    gate,
                    handover_voxel_m(&world.config, cover.from_m),
                ))
            })
            .collect();
        Self {
            origin,
            generator: world.generator.clone(),
            populations,
        }
    }

    /// The handover threshold each cover material takes, without sweeping.
    ///
    /// A raster rebuilt for a ribbon change reuses the cached cover layer
    /// but still has to publish its thresholds, which are level data and
    /// cost nothing to restate.
    fn thresholds(&self) -> Vec<(u32, f32)> {
        self.populations
            .iter()
            .filter_map(|(def, _, from_voxel_m)| {
                Some((def.cover.as_ref()?.material, *from_voxel_m))
            })
            .collect()
    }

    /// Paint every population the level says becomes ground at a distance.
    ///
    /// Returns the raster the ribbon pass then strokes over — a road
    /// through a wood is a road — and the handover threshold each cover
    /// material takes.
    fn run(self) -> (Vec<u32>, Vec<(u32, f32)>) {
        let mut texels = vec![0u32; (MAP_SIZE * MAP_SIZE / 4) as usize];
        let mut coarse_from = Vec::new();
        let origin = self.origin;
        for (def, gate, from_voxel_m) in &self.populations {
            let cover = def.cover.as_ref().expect("filtered on having one");
            coarse_from.push((cover.material, *from_voxel_m));
            let generator = self.generator.clone();
            let gate = *gate;
            let inputs = voxel_engine::scatter::PlacementInputs {
                generator: &self.generator,
                clearance: Vec::new(),
                cut_ops: Vec::new(),
                gate_weight: Box::new(move |xz| {
                    gate.map_or(1.0, |m| generator.surface_material_weight(xz, 8.0, m))
                }),
            };
            let full_at = cover.full_at.max(1.0e-6);
            let started = std::time::Instant::now();
            // Sampled at cell CORNERS, one row longer than there are cells,
            // so every texel sits inside a cell with four known corners and
            // no cell boundary becomes an edge.
            let cells = MAP_SIZE / COVER_STRIDE;
            let step = TEXEL_M * COVER_STRIDE as f32;
            let grid: Vec<f32> = (0..=cells)
                .flat_map(|gz| (0..=cells).map(move |gx| (gx, gz)))
                .map(|(gx, gz)| {
                    let at = origin + Vec2::new(gx as f32, gz as f32) * step;
                    voxel_engine::scatter::coverage(def, &inputs, at)
                })
                .collect();
            // Walked a cell at a time rather than a texel at a time, so the
            // four corners are fetched once per cell instead of once per
            // texel — and, far more than that, so a cell with no population
            // at any corner skips its texels entirely. Most of a world is
            // not forest, and that is what keeps the cost of this pass tied
            // to the size of the woods rather than to the resolution of the
            // map.
            let row = (cells + 1) as usize;
            let inv = 1.0 / COVER_STRIDE as f32;
            for cz in 0..cells {
                for cx in 0..cells {
                    let at = |ix: u32, iz: u32| grid[iz as usize * row + ix as usize];
                    let (c00, c10) = (at(cx, cz), at(cx + 1, cz));
                    let (c01, c11) = (at(cx, cz + 1), at(cx + 1, cz + 1));
                    if c00 <= 0.0 && c10 <= 0.0 && c01 <= 0.0 && c11 <= 0.0 {
                        continue;
                    }
                    for iz in 0..COVER_STRIDE {
                        let fz = iz as f32 * inv;
                        let (top, bot) = (c00 * (1.0 - fz) + c01 * fz, c10 * (1.0 - fz) + c11 * fz);
                        for ix in 0..COVER_STRIDE {
                            let fx = ix as f32 * inv;
                            let c = top * (1.0 - fx) + bot * fx;
                            // Crowns land independently, so what closes
                            // over the ground is Poisson, not linear:
                            // twice the density is not twice the cover,
                            // because the second crown mostly lands on the
                            // first. Painting `c / full_at` instead showed
                            // a wood that reads as unbroken canopy as a
                            // few scattered texels — the attempts that
                            // survive every gate are a small fraction of a
                            // very large number.
                            let share = 1.0 - (-c / full_at).exp();
                            let (x, z) = (cx * COVER_STRIDE + ix, cz * COVER_STRIDE + iz);
                            if share > dither(origin, x, z) {
                                put(&mut texels, x, z, cover.material);
                            }
                        }
                    }
                }
            }
            if std::env::var_os("VOXEL_LOG_LAYERS").is_some() {
                info!("cover sweep '{}' in {:?}", def.class, started.elapsed());
            }
        }
        (texels, coarse_from)
    }
}

/// A fixed 0..1 per square of WORLD ground, so the thinning edge of a
/// population holds still. Hashing the texel's raster index instead would
/// reshuffle every wood in sight each time the map re-anchors.
fn dither(origin: Vec2, x: u32, z: u32) -> f32 {
    let w = (origin / TEXEL_M).floor() + Vec2::new(x as f32, z as f32);
    let mut h = (w.x as i32 as u32).wrapping_mul(0x9E37_79B9)
        ^ (w.y as i32 as u32).wrapping_mul(0x85EB_CA6B);
    h ^= h >> 15;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    (h >> 8) as f32 / 16_777_216.0
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
    let half_w = half_w.max(MIN_STROKE_HALF_W_M);
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
    let idx = voxel_core::layout::texel_index(MAP_SIZE, x, z) as usize;
    let word = &mut texels[idx / 4];
    let shift = (idx % 4) * 8;
    *word = (*word & !(0xFFu32 << shift)) | ((material & 0xFF) << shift);
}
