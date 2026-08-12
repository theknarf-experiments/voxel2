//! The layers behind this game's region nodes: a small set of
//! parameterized layer kinds — scatter, connect, flow, worm, emit — that
//! level JSON composes into ONE LayerManager per level. Features (ruins,
//! roads, rivers, caves, dungeons, districts) are configurations of these
//! kinds, not engine code.

use glam::{DVec3, IVec3, Vec2, Vec3};
use voxel_core::csg::CsgOp;
use voxel_layers::{ChunkCtx, Dep, IAabb, Layer, LayerChunk, LayerGraph};

use voxel_worldgen::Generator;

pub use voxel_core::patch::{Marker, PatchSet, RibbonSeg};

/// Configuration of a `scatter` node: hash-gated candidate sites per
/// cell, filtered by terrain.
#[derive(Clone, Debug)]
pub struct ScatterCfg {
    pub cell_m: i32,
    /// Chance a cell hosts a site.
    pub chance: f32,
    /// Margin (meters) keeping sites away from cell borders.
    pub margin_m: f32,
    /// Altitude band sites may occupy.
    pub altitude: [f32; 2],
    /// Up-ness interval (1 = flat).
    pub up: [f32; 2],
    /// Accept sites with probability = the biome's blended weight.
    pub biome: Option<BiomeGate>,
    /// Relax this cell's site away from its neighbours' instead of
    /// generating one. See [`RelaxFrom`].
    pub relax_from: Option<RelaxFrom>,
}

/// Push a site away from the ones around it, reading a SOURCE instance.
///
/// One cell hosts at most one site, so a plain scatter can put two of
/// them a few metres apart across a cell border — two ossuaries in each
/// other's laps while the cell interiors are empty. This is the classic
/// relaxation pass: read the source's 3x3 neighbourhood and move own
/// site away from the others.
///
/// **A separate instance, not an internal layer level.** LayerProcGen
/// offers levels for exactly this shape (its `LocationLayer` places at
/// level 0 and relaxes at level 1), and a level would work — but a level
/// shares one chunk struct across stages, so the framework cannot check
/// which stage may read what. As an instance the read goes through a
/// declared `Dep` and the padding assert applies, and iterating twice is
/// a second instance rather than a third level.
#[derive(Clone, Debug)]
pub struct RelaxFrom {
    /// Instance to read unrelaxed sites from. Must have the same
    /// `cell_m`, so cell N of the source is cell N here.
    pub instance: String,
    /// How far to move, as a fraction of the distance still wanted
    /// between two sites. 0 does nothing; 1 is a full correction and
    /// tends to overshoot into a neighbour.
    pub strength: f32,
}

impl Default for ScatterCfg {
    fn default() -> Self {
        Self {
            cell_m: 256,
            chance: 0.3,
            margin_m: 32.0,
            altitude: [f32::MIN, f32::MAX],
            up: [0.0, 1.0],
            biome: None,
            relax_from: None,
        }
    }
}

/// Generic site scatter: the sites layer every other kind consumes.
/// Register one instance per feature ("sites:ruins", "sites:springs"...).
#[derive(Clone)]
pub struct ScatterSites {
    pub cfg: ScatterCfg,
}

#[derive(Default)]
pub struct SitesChunk {
    pub sites: Vec<Vec2>,
}

impl Layer for ScatterSites {
    type Chunk = SitesChunk;
    const NAME: &'static str = "stack/scatter";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(self.cfg.cell_m as f64, 0.0, self.cfg.cell_m as f64)
    }

    fn dependencies(&self) -> Vec<Dep> {
        // Relaxing reads the source's 3x3, which is one cell of padding.
        // A relaxed instance does no terrain or biome gating of its own —
        // the source already did it — so the two are exclusive.
        if let Some(relax) = &self.cfg.relax_from {
            let cell = self.cfg.cell_m;
            return vec![Dep::named(&relax.instance, IVec3::new(cell, 0, cell))];
        }
        // A region gate reads no layer at all: the bands live in the
        // generator program, so the weight is a pure function of the
        // point and costs no residency.
        Vec::new()
    }
}

impl ScatterSites {
    /// Move this cell's site away from the ones in the 3x3 around it.
    ///
    /// A site stays inside its OWN cell, clamped to the same margin the
    /// scatter used. That is not cosmetic: consumers read sites by
    /// iterating the cells overlapping their bounds, several of them with
    /// no padding at all, so a site that wandered into the next cell
    /// would simply stop being found. Relaxation improves spacing within
    /// that constraint rather than breaking every reader.
    fn relax(&self, ctx: &ChunkCtx<'_, Self>, relax: &RelaxFrom) -> SitesChunk {
        let cell = self.cfg.cell_m as f32;
        let pad = IVec3::new(self.cfg.cell_m, 0, self.cfg.cell_m);
        let own = ctx.chunk_bounds();
        let view = ctx.get_named::<Self>(&relax.instance, own.inflate(pad));

        // Own cell's site, and everyone else's.
        let mut mine: Option<Vec2> = None;
        let mut others: Vec<Vec2> = Vec::new();
        view.for_each(|coord, chunk| {
            if coord == ctx.coord() {
                mine = chunk.sites.first().copied();
            } else {
                others.extend(chunk.sites.iter().copied());
            }
        });
        let Some(mut p) = mine else {
            return SitesChunk { sites: Vec::new() };
        };

        // Wanted spacing: one cell. Closer than that and a neighbour
        // pushes, proportionally to how much closer.
        let mut push = Vec2::ZERO;
        for other in &others {
            let delta = p - *other;
            let d = delta.length();
            if d >= cell || d <= 1e-3 {
                continue;
            }
            push += delta / d * (cell - d);
        }
        p += push * relax.strength;

        // Back inside its own cell.
        let m = self.cfg.margin_m.clamp(0.0, cell * 0.45);
        let lo = Vec2::new(own.min.x as f32 + m, own.min.z as f32 + m);
        let hi = Vec2::new(own.max.x as f32 - m, own.max.z as f32 - m);
        SitesChunk {
            sites: vec![p.clamp(lo, hi.max(lo))],
        }
    }

    fn build(&self, ctx: &ChunkCtx<'_, Self>) -> SitesChunk {
        if let Some(relax) = &self.cfg.relax_from {
            return self.relax(ctx, relax);
        }
        let generator = &ctx.context::<crate::planning::world::WorldCtx>().generator;
        let mut rng = ctx.rng();
        if rng.next_f32() > self.cfg.chance {
            return SitesChunk { sites: Vec::new() };
        }
        let b = ctx.chunk_bounds();
        let cell = self.cfg.cell_m as f32;
        let m = self.cfg.margin_m.clamp(0.0, cell * 0.45);
        let p = Vec2::new(
            b.min.x as f32 + m + rng.next_f32() * (cell - 2.0 * m),
            b.min.z as f32 + m + rng.next_f32() * (cell - 2.0 * m),
        );
        let h = generator.height(p, 8.0);
        let up = generator.up(p, 8.0);
        if !(self.cfg.altitude[0]..self.cfg.altitude[1]).contains(&h)
            || !(self.cfg.up[0]..=self.cfg.up[1]).contains(&up)
        {
            return SitesChunk { sites: Vec::new() };
        }
        if let Some(gate) = &self.cfg.biome {
            if rng.next_f32() > generator.surface_material_weight(p, 8.0, gate.material) {
                return SitesChunk { sites: Vec::new() };
            }
        }
        SitesChunk { sites: vec![p] }
    }
}

/// A region gate on a scatter layer: sites are accepted with probability
/// equal to how firmly the generator paints that region here, so a
/// population thins out across a border instead of stopping on a line.
#[derive(Clone, Debug)]
pub struct BiomeGate {
    /// The material the gated region paints. Resolved through the wired
    /// `biomes` node at build time; the weight itself comes from the
    /// generator, which owns the bands.
    pub material: u32,
}

/// Configuration of a `scatter3` node: volumetric sites for
/// interior worlds (habitation pockets in a megastructure). No terrain
/// filters — interiors have no heightfield.
#[derive(Clone, Debug)]
pub struct Scatter3Cfg {
    /// Cell extent: (xz, y) meters.
    pub cell_m: i32,
    pub cell_y_m: i32,
    pub chance: f32,
    pub margin_m: f32,
    /// Snap site y to multiples of this (structural floor lattice);
    /// 0 = no snapping.
    pub snap_y_m: f32,
    /// Accept sites with probability = the biome's blended weight (the
    /// biome field is planar; districts are xz regions).
    pub biome: Option<BiomeGate>,
}

impl Default for Scatter3Cfg {
    fn default() -> Self {
        Self {
            cell_m: 128,
            cell_y_m: 132,
            chance: 0.45,
            margin_m: 24.0,
            snap_y_m: 0.0,
            biome: None,
        }
    }
}

#[derive(Clone)]
pub struct Scatter3Sites {
    pub cfg: Scatter3Cfg,
}

#[derive(Default)]
pub struct Sites3Chunk {
    pub sites: Vec<Vec3>,
}

impl Layer for Scatter3Sites {
    type Chunk = Sites3Chunk;
    const NAME: &'static str = "stack/scatter3";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(
            self.cfg.cell_m as f64,
            self.cfg.cell_y_m as f64,
            self.cfg.cell_m as f64,
        )
    }

    fn dependencies(&self) -> Vec<Dep> {
        // A region gate reads no layer at all: the bands live in the
        // generator program, so the weight is a pure function of the
        // point and costs no residency.
        Vec::new()
    }
}

impl Scatter3Sites {
    fn build(&self, ctx: &ChunkCtx<'_, Self>) -> Sites3Chunk {
        let mut rng = ctx.rng();
        if rng.next_f32() > self.cfg.chance {
            return Sites3Chunk { sites: Vec::new() };
        }
        let b = ctx.chunk_bounds();
        let m = self.cfg.margin_m.clamp(0.0, self.cfg.cell_m as f32 * 0.45);
        let x = b.min.x as f32 + m + rng.next_f32() * (self.cfg.cell_m as f32 - 2.0 * m);
        let z = b.min.z as f32 + m + rng.next_f32() * (self.cfg.cell_m as f32 - 2.0 * m);
        let mut y = b.min.y as f32 + rng.next_f32() * self.cfg.cell_y_m as f32;
        if self.cfg.snap_y_m > 0.0 {
            // Snap, then clamp INTO the half-open cell: rounding up to
            // exactly max.y would strand the site outside its owner —
            // emitted by nobody while connect layers still link to it.
            y = (y / self.cfg.snap_y_m).round() * self.cfg.snap_y_m;
            let max_snapped =
                ((b.max.y as f32 - 0.5) / self.cfg.snap_y_m).floor() * self.cfg.snap_y_m;
            y = y.clamp(b.min.y as f32, max_snapped.max(b.min.y as f32));
        }
        if let Some(gate) = &self.cfg.biome {
            let generator = &ctx.context::<crate::planning::world::WorldCtx>().generator;
            let w = generator.surface_material_weight(Vec2::new(x, z), 8.0, gate.material);
            if rng.next_f32() > w {
                return Sites3Chunk { sites: Vec::new() };
            }
        }
        Sites3Chunk {
            sites: vec![Vec3::new(x, y, z)],
        }
    }
}

/// Configuration of a `connect3` node: orthogonal (axis-aligned)
/// links between volumetric sites — walkway tubes in a megastructure.
#[derive(Clone, Debug)]
pub struct Connect3Cfg {
    pub source: String,
    pub reach_m: f32,
}

impl Default for Connect3Cfg {
    fn default() -> Self {
        Self {
            source: String::new(),
            reach_m: 400.0,
        }
    }
}

#[derive(Clone)]
pub struct Connect3Paths {
    pub cfg: Connect3Cfg,
    pub cell_m: i32,
    pub cell_y_m: i32,
}

#[derive(Default)]
pub struct Paths3Chunk {
    /// Orthogonal waypoint chains (every segment varies along one axis).
    pub paths: Vec<Vec<Vec3>>,
}

impl Layer for Connect3Paths {
    type Chunk = Paths3Chunk;
    const NAME: &'static str = "stack/connect3";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(self.cell_m as f64, self.cell_y_m as f64, self.cell_m as f64)
    }

    fn dependencies(&self) -> Vec<Dep> {
        let pad = self.cfg.reach_m as i32;
        vec![Dep::named(&self.cfg.source, IVec3::splat(pad))]
    }
}

impl Connect3Paths {
    fn build(&self, ctx: &ChunkCtx<'_, Self>) -> Paths3Chunk {
        let own = ctx.chunk_bounds();
        let pad = self.cfg.reach_m as i32;
        let view = ctx.get_named::<Scatter3Sites>(&self.cfg.source, own.inflate(IVec3::splat(pad)));
        let mut sites: Vec<Vec3> = Vec::new();
        view.for_each(|_, c| sites.extend(c.sites.iter().copied()));
        let in_own = |p: Vec3| {
            p.x >= own.min.x as f32
                && p.x < own.max.x as f32
                && p.y >= own.min.y as f32
                && p.y < own.max.y as f32
                && p.z >= own.min.z as f32
                && p.z < own.max.z as f32
        };
        let mut paths = Vec::new();
        for &a in &sites {
            let Some(&b) = sites
                .iter()
                .filter(|&&b| b != a && a.distance(b) < self.cfg.reach_m)
                .min_by(|x, y| a.distance_squared(**x).total_cmp(&a.distance_squared(**y)))
            else {
                continue;
            };
            let (lo, hi) = if (a.x, a.y, a.z) <= (b.x, b.y, b.z) {
                (a, b)
            } else {
                (b, a)
            };
            if !in_own((lo + hi) * 0.5) {
                continue;
            }
            // Canonical L-route at right angles: run x at lo's level,
            // then z, then rise to hi. Degenerate legs are dropped.
            let mut waypoints = vec![lo];
            let mut cur = lo;
            for next in [Vec3::new(hi.x, lo.y, lo.z), Vec3::new(hi.x, lo.y, hi.z), hi] {
                if next.distance(cur) > 0.01 {
                    waypoints.push(next);
                    cur = next;
                }
            }
            if waypoints.len() >= 2 && !paths.contains(&waypoints) {
                paths.push(waypoints);
            }
        }
        Paths3Chunk { paths }
    }
}

/// Configuration of a `connect` node: pathfound links between
/// sites of a scatter instance (roads, patrol routes, power lines...).
#[derive(Clone, Debug)]
pub struct ConnectCfg {
    /// Scatter instance supplying the sites.
    pub source: String,
    /// Max link distance (m); each site links to its nearest neighbor.
    pub reach_m: f32,
    /// Pathfinding corridor half-width around the endpoint box.
    pub corridor_m: f32,
    pub slope_penalty: f32,
    /// Pathfinding lattice step (meters). Search cost is quadratic in the
    /// corridor's size measured in these, so a corridor spanning
    /// kilometres is only affordable on a coarse lattice.
    pub step_m: f32,
}

impl Default for ConnectCfg {
    fn default() -> Self {
        Self {
            source: String::new(),
            reach_m: 700.0,
            corridor_m: 192.0,
            slope_penalty: 60.0,
            step_m: 8.0,
        }
    }
}

/// Generic pathfound connections (owned by the link midpoint's cell).
#[derive(Clone)]
pub struct ConnectPaths {
    pub cfg: ConnectCfg,
    pub cell_m: i32,
}

#[derive(Default)]
pub struct PathsChunk {
    pub paths: Vec<Vec<Vec2>>,
}

impl Layer for ConnectPaths {
    type Chunk = PathsChunk;
    const NAME: &'static str = "stack/connect";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(self.cell_m as f64, 0.0, self.cell_m as f64)
    }

    fn dependencies(&self) -> Vec<Dep> {
        let pad = (self.cfg.reach_m + self.cfg.corridor_m) as i32;
        vec![Dep::named(&self.cfg.source, IVec3::new(pad, 0, pad))]
    }
}

impl ConnectPaths {
    fn build(&self, ctx: &ChunkCtx<'_, Self>) -> PathsChunk {
        let generator = &ctx.context::<crate::planning::world::WorldCtx>().generator;
        let own = ctx.chunk_bounds();
        let pad = (self.cfg.reach_m + self.cfg.corridor_m) as i32;
        let view =
            ctx.get_named::<ScatterSites>(&self.cfg.source, own.inflate(IVec3::new(pad, 0, pad)));
        let mut sites: Vec<Vec2> = Vec::new();
        view.for_each(|_, c| sites.extend(c.sites.iter().copied()));
        let in_own = |p: Vec2| {
            p.x >= own.min.x as f32
                && p.x < own.max.x as f32
                && p.y >= own.min.z as f32
                && p.y < own.max.z as f32
        };
        let mut paths = Vec::new();
        for &a in &sites {
            let Some(&b) = sites
                .iter()
                .filter(|&&b| b != a && a.distance(b) < self.cfg.reach_m)
                .min_by(|x, y| a.distance_squared(**x).total_cmp(&a.distance_squared(**y)))
            else {
                continue;
            };
            let (lo, hi) = if (a.x, a.y) <= (b.x, b.y) {
                (a, b)
            } else {
                (b, a)
            };
            if !in_own((lo + hi) * 0.5) {
                continue;
            }
            let clo = lo.min(hi) - Vec2::splat(self.cfg.corridor_m);
            let chi = lo.max(hi) + Vec2::splat(self.cfg.corridor_m);
            let params = voxel_worldgen::path::PathParams {
                slope_penalty: self.cfg.slope_penalty,
                step_m: self.cfg.step_m,
                ..Default::default()
            };
            let waypoints = voxel_worldgen::path::find_path(
                &|p| generator.height(p, 8.0),
                lo,
                hi,
                clo,
                chi,
                &params,
            )
            .unwrap_or_else(|| {
                // Straight fallback, SUBDIVIDED: a single reach-length
                // segment would put slabs ~350 m from its midpoint cell
                // and break the ELEM_PAD_M query contract.
                let n = (lo.distance(hi) / (2.0 * self.cfg.step_m)).ceil().max(1.0) as usize;
                (0..=n).map(|i| lo.lerp(hi, i as f32 / n as f32)).collect()
            });
            if !paths.contains(&waypoints) {
                paths.push(waypoints);
            }
        }
        PathsChunk { paths }
    }
}

/// Configuration of a `flow` node: descent courses from sites
/// (rivers, lava, mudslides).
#[derive(Clone, Debug)]
pub struct FlowCfg {
    pub source: String,
    pub max_steps: usize,
    pub max_spill_rise: f32,
}

impl Default for FlowCfg {
    fn default() -> Self {
        Self {
            source: String::new(),
            max_steps: 400,
            max_spill_rise: 7.0,
        }
    }
}

#[derive(Clone)]
pub struct FlowCourses {
    pub cfg: FlowCfg,
    pub cell_m: i32,
}

#[derive(Default)]
pub struct CoursesChunk {
    pub courses: Vec<(Vec<Vec2>, Vec<f32>)>,
}

impl Layer for FlowCourses {
    type Chunk = CoursesChunk;
    const NAME: &'static str = "stack/flow";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(self.cell_m as f64, 0.0, self.cell_m as f64)
    }

    fn dependencies(&self) -> Vec<Dep> {
        vec![Dep::named(&self.cfg.source, IVec3::ZERO)]
    }
}

impl FlowCourses {
    fn build(&self, ctx: &ChunkCtx<'_, Self>) -> CoursesChunk {
        let generator = &ctx.context::<crate::planning::world::WorldCtx>().generator;
        let own = ctx.chunk_bounds();
        let view = ctx.get_named::<ScatterSites>(&self.cfg.source, own);
        // Own the spring, not the whole source chunk: with mismatched
        // cell sizes a straddled site would otherwise fork per consumer.
        let in_own = |p: Vec2| {
            p.x >= own.min.x as f32
                && p.x < own.max.x as f32
                && p.y >= own.min.z as f32
                && p.y < own.max.z as f32
        };
        let mut courses = Vec::new();
        for (_, chunk) in view.iter() {
            for &start in &chunk.sites {
                if !in_own(start) {
                    continue;
                }
                let params = voxel_worldgen::flow::FlowParams {
                    max_steps: self.cfg.max_steps,
                    max_spill_rise: self.cfg.max_spill_rise,
                    ..Default::default()
                };
                let waypoints =
                    voxel_worldgen::flow::flow_path(&|p| generator.height(p, 8.0), start, &params);
                if waypoints.len() < 6 {
                    continue;
                }
                let mut level = f32::MAX;
                let levels: Vec<f32> = waypoints
                    .iter()
                    .map(|p| {
                        level = level.min(generator.height(*p, 8.0) - 0.35);
                        level
                    })
                    .collect();
                courses.push((waypoints, levels));
            }
        }
        CoursesChunk { courses }
    }
}

/// Configuration of a `worm` node: noise-steered burrows from
/// sites (caves, lava tubes).
#[derive(Clone, Debug)]
pub struct WormCfg {
    pub source: String,
    pub steps: u32,
    pub radius: [f32; 2],
    /// Keep tunnels this many radii under the surface.
    pub burial_radii: f32,
}

impl Default for WormCfg {
    fn default() -> Self {
        Self {
            source: String::new(),
            steps: 70,
            radius: [2.2, 3.6],
            burial_radii: 2.4,
        }
    }
}

#[derive(Clone)]
pub struct WormBurrows {
    pub cfg: WormCfg,
    pub cell_m: i32,
}

#[derive(Default)]
pub struct WormsChunk {
    /// Each worm: sphere centers with radii.
    pub worms: Vec<Vec<(Vec3, f32)>>,
}

impl Layer for WormBurrows {
    type Chunk = WormsChunk;
    const NAME: &'static str = "stack/worm";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(self.cell_m as f64, 0.0, self.cell_m as f64)
    }

    fn dependencies(&self) -> Vec<Dep> {
        vec![Dep::named(&self.cfg.source, IVec3::ZERO)]
    }
}

impl WormBurrows {
    fn build(&self, ctx: &ChunkCtx<'_, Self>) -> WormsChunk {
        let generator = &ctx.context::<crate::planning::world::WorldCtx>().generator;
        let own = ctx.chunk_bounds();
        let view = ctx.get_named::<ScatterSites>(&self.cfg.source, own);
        let in_own = |p: Vec2| {
            p.x >= own.min.x as f32
                && p.x < own.max.x as f32
                && p.y >= own.min.z as f32
                && p.y < own.max.z as f32
        };
        let mut worms = Vec::new();
        for (_, chunk) in view.iter() {
            for &mouth_xz in &chunk.sites {
                if !in_own(mouth_xz) {
                    continue;
                }
                // Per-site stream (iteration-order independent; two
                // mouths in one cell must not dig identical burrows).
                let mut rng = voxel_core::seed::Rng::new(voxel_core::seed::splitmix64(
                    ctx.seed()
                        ^ ((mouth_xz.x.to_bits() as u64) << 32 | mouth_xz.y.to_bits() as u64),
                ));
                let ground = generator.height(mouth_xz, 8.0);
                let base_r =
                    self.cfg.radius[0] + rng.next_f32() * (self.cfg.radius[1] - self.cfg.radius[0]);
                let mut yaw = rng.next_f32() * std::f32::consts::TAU;
                let mut pitch = -0.4 - rng.next_f32() * 0.2;
                let mut pos = Vec3::new(mouth_xz.x, ground + base_r * 0.6, mouth_xz.y);
                let mut worm = Vec::new();
                for _ in 0..self.cfg.steps {
                    let r = base_r * (0.8 + 0.45 * rng.next_f32());
                    worm.push((pos, r));
                    yaw += (rng.next_f32() - 0.5) * 0.55;
                    pitch += (rng.next_f32() - 0.5) * 0.3 - pitch * 0.15;
                    pitch = pitch.clamp(-0.55, 0.25);
                    let ceiling =
                        generator.height(Vec2::new(pos.x, pos.z), 8.0) - r * self.cfg.burial_radii;
                    if pos.y > ceiling {
                        pitch = (pitch - 0.2).min(-0.35);
                    }
                    let dir = Vec3::new(
                        yaw.cos() * pitch.cos(),
                        pitch.sin(),
                        yaw.sin() * pitch.cos(),
                    );
                    pos += dir * (r * 0.9);
                }
                worms.push(worm);
            }
        }
        WormsChunk { worms }
    }
}

/// What an emit layer turns its source's data into.
#[derive(Clone, Debug)]
pub enum EmitKind {
    /// Terrain-seated slab chains along a `connect` source (roads,
    /// walkways). Optionally reserves spawner clearance along the way.
    PathSlabs {
        half_w: f32,
        thickness: f32,
        material: u32,
        clearance: bool,
    },
    /// Bed notch + ribbon surface segments along a `flow`
    /// source. Half width grows from `width[0]` to `width[1]` downstream.
    Ribbon { material: u32, width: [f32; 2] },
    /// A ribbon laid on the ground along a `connect` source: the same
    /// surface primitive, seated instead of levelled, and carving
    /// nothing. What a road IS at a distance where cutting a 0.5 m notch
    /// into 25 m voxels does nothing at all.
    PathRibbon { material: u32, width: [f32; 2] },
    /// Sphere-cut chains from a `worm` source (caves).
    WormCuts,
    /// Build a structure (level data — see [`crate::structure`]) at each
    /// site of a `scatter` source, optionally dropping a marker.
    SiteStructure {
        structure: std::sync::Arc<super::structure::Structure>,
        marker: Option<String>,
    },
    /// The same at each site of a `scatter3` source (interiors), seated
    /// on the site's own y rather than the terrain.
    SiteStructure3 {
        structure: std::sync::Arc<super::structure::Structure>,
        marker: Option<String>,
    },
    /// Shell tubes with bored interiors along a `connect3` source —
    /// walkway corridors. The bore extends past segment ends so tubes
    /// open into the rooms and shafts they meet. `lift_m` raises the
    /// route above the site lattice plane so the bore floor lands on the
    /// structural slab top instead of inside the slab.
    Tubes {
        material: u32,
        bore: f32,
        lift_m: f32,
    },
}

/// Configuration of an `emit` stack layer: the only kind that produces
/// [`PatchSet`]s. It is its own index — every element is bucketed by the
/// cell owning its midpoint (the perf pattern that keeps world queries
/// local no matter how far a source wanders from its owning cell).
#[derive(Clone, Debug)]
pub struct EmitCfg {
    /// Source instance (a scatter/connect/flow/worm layer).
    pub source: String,
    pub kind: EmitKind,
    /// How far source geometry reaches beyond its owning cells (meters);
    /// becomes the dependency padding. Author-declared, like every
    /// LayerProcGen padding.
    pub pad_m: f32,
}

#[derive(Clone)]
pub struct EmitPatches {
    pub cfg: EmitCfg,
    pub cell_m: i32,
    /// Cell height for volumetric sources (`scatter3`/`connect3`);
    /// 0 = planar (y collapsed). MUST be non-zero when the source is
    /// volumetric or the dependency view spans unbounded y.
    pub cell_y_m: i32,
}

#[derive(Default)]
pub struct PatchChunk {
    pub patches: PatchSet,
}

/// Largest reach of one emitted element beyond the cell that owns its
/// midpoint: path/course sub-segments are ≤ 16 m plus their width, worm
/// spheres a few meters, site structures ≤ ~25 m. Queries pad by this.
pub const ELEM_PAD_M: f32 = 64.0;

impl Layer for EmitPatches {
    type Chunk = PatchChunk;
    const NAME: &'static str = "stack/emit";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(self.cell_m as f64, self.cell_y_m as f64, self.cell_m as f64)
    }

    fn dependencies(&self) -> Vec<Dep> {
        let pad = self.cfg.pad_m as i32;
        // Volumetric sources reach vertically too (a link's vertical leg
        // spans rows far from the owning cell); planar emits keep y
        // collapsed.
        let pad_y = if self.cell_y_m > 0 { pad } else { 0 };
        vec![Dep::named(&self.cfg.source, IVec3::new(pad, pad_y, pad))]
    }
}

impl EmitPatches {
    fn build(&self, ctx: &ChunkCtx<'_, Self>) -> PatchChunk {
        let generator = &ctx.context::<crate::planning::world::WorldCtx>().generator;
        let own = ctx.chunk_bounds();
        let pad = self.cfg.pad_m as i32;
        let pad_y = if self.cell_y_m > 0 { pad } else { 0 };
        let padded = own.inflate(IVec3::new(pad, pad_y, pad));
        let in_own = |p: Vec2| {
            p.x >= own.min.x as f32
                && p.x < own.max.x as f32
                && p.y >= own.min.z as f32
                && p.y < own.max.z as f32
        };
        let mut out = PatchSet::default();
        match &self.cfg.kind {
            EmitKind::PathSlabs {
                half_w,
                thickness,
                material,
                clearance,
            } => {
                for (_, c) in ctx
                    .get_named::<ConnectPaths>(&self.cfg.source, padded)
                    .iter()
                {
                    for path in &c.paths {
                        for seg in path.windows(2) {
                            if !in_own((seg[0] + seg[1]) * 0.5) {
                                continue;
                            }
                            slab_segment_ops(
                                seg[0],
                                seg[1],
                                *half_w,
                                *thickness,
                                *material,
                                generator,
                                &mut out.ops,
                            );
                            if *clearance {
                                out.clearance.push([seg[0], seg[1]]);
                            }
                        }
                    }
                }
            }
            EmitKind::Ribbon { material, width } => {
                for (_, c) in ctx
                    .get_named::<FlowCourses>(&self.cfg.source, padded)
                    .iter()
                {
                    for (waypoints, levels) in &c.courses {
                        let n = waypoints.len();
                        for (i, seg) in waypoints.windows(2).enumerate() {
                            if !in_own((seg[0] + seg[1]) * 0.5) {
                                continue;
                            }
                            let t = i as f32 / n as f32;
                            let half_w = width[0] + (width[1] - width[0]) * t;
                            let seg_levels = [levels[i], levels[i + 1]];
                            ribbon_bed_ops(seg[0], seg[1], half_w, seg_levels, &mut out.ops);
                            out.ribbons.push(RibbonSeg {
                                a: seg[0],
                                b: seg[1],
                                half_w,
                                levels: seg_levels,
                                material: *material,
                                // A water course carries its own level.
                                seated: false,
                            });
                            out.clearance.push([seg[0], seg[1]]);
                        }
                    }
                }
            }
            EmitKind::PathRibbon { material, width } => {
                for (_, c) in ctx
                    .get_named::<ConnectPaths>(&self.cfg.source, padded)
                    .iter()
                {
                    for waypoints in &c.paths {
                        let n = waypoints.len();
                        for (i, seg) in waypoints.windows(2).enumerate() {
                            if !in_own((seg[0] + seg[1]) * 0.5) {
                                continue;
                            }
                            let t = i as f32 / n as f32;
                            let half_w = width[0] + (width[1] - width[0]) * t;
                            out.ribbons.push(RibbonSeg {
                                a: seg[0],
                                b: seg[1],
                                half_w,
                                // Seated: whoever draws it decides the
                                // height, against the surface they are
                                // drawing at that distance.
                                levels: [0.0; 2],
                                material: *material,
                                seated: true,
                            });
                        }
                    }
                }
            }
            EmitKind::WormCuts => {
                for (_, c) in ctx
                    .get_named::<WormBurrows>(&self.cfg.source, padded)
                    .iter()
                {
                    for worm in &c.worms {
                        for &(p, r) in worm {
                            if in_own(Vec2::new(p.x, p.z)) {
                                out.ops.push(CsgOp::sphere(p, r, 0, true));
                            }
                        }
                    }
                }
            }
            EmitKind::SiteStructure3 { structure, marker } => {
                let in_own_y = |y: f32| y >= own.min.y as f32 && y < own.max.y as f32;
                let view = ctx.get_named::<Scatter3Sites>(&self.cfg.source, padded);
                for (_, c) in view.iter() {
                    for &site in &c.sites {
                        let flat = Vec2::new(site.x, site.z);
                        if !in_own(flat) || !in_own_y(site.y) {
                            continue;
                        }
                        let mut rng = voxel_core::seed::Rng::new(voxel_core::seed::splitmix64(
                            ctx.seed()
                                ^ ((site.x.to_bits() as u64) << 32 | site.z.to_bits() as u64)
                                ^ (site.y.to_bits() as u64) << 16,
                        ));
                        super::structure::build(structure, site, generator, &mut rng, &mut out.ops);
                        if let Some(kind) = marker {
                            out.markers.push(Marker {
                                pos: site,
                                kind: kind.clone(),
                            });
                        }
                    }
                }
            }
            EmitKind::Tubes {
                material,
                bore,
                lift_m,
            } => {
                let in_own_y = |y: f32| y >= own.min.y as f32 && y < own.max.y as f32;
                let lift = Vec3::Y * *lift_m;
                let view = ctx.get_named::<Connect3Paths>(&self.cfg.source, padded);
                for (_, c) in view.iter() {
                    for path in &c.paths {
                        for seg in path.windows(2) {
                            let seg = [seg[0] + lift, seg[1] + lift];
                            // Legs can span the whole link reach; bucket
                            // short sub-segments so queries stay local.
                            let len = seg[0].distance(seg[1]);
                            let subs = (len / 24.0).ceil().max(1.0) as i32;
                            for i in 0..subs {
                                let t0 = i as f32 / subs as f32;
                                let t1 = (i + 1) as f32 / subs as f32;
                                let a = seg[0].lerp(seg[1], t0);
                                let b = seg[0].lerp(seg[1], t1);
                                let mid = (a + b) * 0.5;
                                if !in_own(Vec2::new(mid.x, mid.z)) || !in_own_y(mid.y) {
                                    continue;
                                }
                                // Overshoot the bore only at the real leg
                                // ends (open into what the corridor meets).
                                tube_segment_ops(a, b, *material, *bore, &mut out.ops);
                            }
                        }
                    }
                }
            }
            EmitKind::SiteStructure { structure, marker } => {
                for (_, c) in ctx
                    .get_named::<ScatterSites>(&self.cfg.source, padded)
                    .iter()
                {
                    for &site in &c.sites {
                        if !in_own(site) {
                            continue;
                        }
                        // Per-site stream independent of cell iteration
                        // order, derived from the emit instance's seed.
                        let mut rng = voxel_core::seed::Rng::new(voxel_core::seed::splitmix64(
                            ctx.seed()
                                ^ ((site.x.to_bits() as u64) << 32 | site.y.to_bits() as u64),
                        ));
                        super::structure::build(
                            structure,
                            Vec3::new(site.x, 0.0, site.y),
                            generator,
                            &mut rng,
                            &mut out.ops,
                        );
                        if let Some(kind) = marker {
                            let y = generator.height(site, 1.0);
                            out.markers.push(Marker {
                                pos: Vec3::new(site.x, y, site.y),
                                kind: kind.clone(),
                            });
                        }
                    }
                }
            }
        }
        PatchChunk { patches: out }
    }
}

/// One orthogonal tube segment: a shell box around a bored interior.
/// The bore overshoots the ends so consecutive tubes, rooms, and other
/// voids the corridor meets open into each other.
fn tube_segment_ops(a: Vec3, b: Vec3, material: u32, bore: f32, out: &mut Vec<CsgOp>) {
    let d = b - a;
    let len = d.length();
    if len < 0.01 {
        return;
    }
    let mid = (a + b) * 0.5;
    let shell = bore + 0.6;
    let half = |along: f32, r: f32| {
        if d.x.abs() > d.y.abs() && d.x.abs() > d.z.abs() {
            Vec3::new(along, r, r)
        } else if d.y.abs() > d.z.abs() {
            Vec3::new(r, along, r)
        } else {
            Vec3::new(r, r, along)
        }
    };
    out.push(CsgOp::boxy(
        mid,
        half(len * 0.5 + shell, shell),
        0.0,
        material,
        false,
    ));
    out.push(CsgOp::boxy(
        mid,
        half(len * 0.5 + bore + 1.2, bore),
        0.0,
        0,
        true,
    ));
}

/// Terrain-seated slab chain along one path segment (roads).
#[allow(clippy::too_many_arguments)]
fn slab_segment_ops(
    a: Vec2,
    b: Vec2,
    half_w: f32,
    thickness: f32,
    material: u32,
    generator: &Generator,
    out: &mut Vec<CsgOp>,
) {
    let len = a.distance(b);
    if len < 0.01 {
        return;
    }
    let dir = (b - a) / len;
    let steps = (len / 3.2).ceil() as i32;
    for i in 0..steps {
        let t = (i as f32 + 0.5) / steps as f32;
        let p = a + dir * (t * len);
        let y = generator.height(p, 1.0);
        out.push(CsgOp::boxy(
            Vec3::new(p.x, y - 0.15, p.y),
            Vec3::new(half_w, thickness, half_w),
            0.0,
            material,
            false,
        ));
    }
}

/// Bed notch along one course segment, flow-aligned with interpolated
/// (monotone) surface heights. The ribbon surface itself is NOT baked into
/// the SDF — the renderer draws it from the emitted [`RibbonSeg`]s.
fn ribbon_bed_ops(a: Vec2, b: Vec2, half_w: f32, levels: [f32; 2], out: &mut Vec<CsgOp>) {
    let len = a.distance(b);
    if len < 0.01 {
        return;
    }
    let dir = (b - a) / len;
    let yaw = dir.to_angle();
    let steps = (len / 3.0).ceil().max(1.0) as i32;
    let sub = len / steps as f32;
    for i in 0..steps {
        let t = (i as f32 + 0.5) / steps as f32;
        let p = a + dir * (t * len);
        let level = levels[0] + (levels[1] - levels[0]) * t;
        out.push(CsgOp::boxy(
            Vec3::new(p.x, level + 0.9, p.y),
            Vec3::new(sub * 0.7 + 0.8, 2.4, half_w + 1.4),
            -yaw,
            0,
            true,
        ));
    }
}

/// Patches of one emit instance overlapping the world box `[min, max]`,
/// filtered to elements that touch it. Local: reads only the index cells
/// within `ELEM_PAD_M` of the box.
pub fn patches_in(mgr: &LayerGraph, instance: &str, min: Vec3, max: Vec3) -> PatchSet {
    let pad = ELEM_PAD_M as i32;
    // Real y bounds (volumetric emits bucket per y-row; a planar emit's
    // collapsed axis ignores them), padded like xz and clamped so a
    // facade query with sentinel y (±1e9) cannot enumerate millions of
    // rows — no world exceeds ±16 km vertically.
    const Y_CLAMP_M: f32 = 16_000.0;
    let y0 = min.y.max(-Y_CLAMP_M) as i32 - pad;
    let y1 = max.y.min(Y_CLAMP_M) as i32 + pad;
    let bounds = IAabb::new(
        IVec3::new(min.x as i32 - pad, y0, min.z as i32 - pad),
        IVec3::new(max.x as i32 + pad, y1.max(y0 + 1), max.z as i32 + pad),
    );
    let seg_touches = |a: Vec2, b: Vec2, half_w: f32| {
        let lo = a.min(b) - Vec2::splat(half_w);
        let hi = a.max(b) + Vec2::splat(half_w);
        lo.x <= max.x && hi.x >= min.x && lo.y <= max.z && hi.y >= min.z
    };
    let mut out = PatchSet::default();
    for (_, c) in mgr.view::<EmitPatches>(instance, bounds).iter() {
        for op in &c.patches.ops {
            if op.touches(voxel_core::csg::Aabb::new(min, max)) {
                out.ops.push(*op);
            }
        }
        for w in &c.patches.ribbons {
            if seg_touches(w.a, w.b, w.half_w) {
                out.ribbons.push(*w);
            }
        }
        for seg in &c.patches.clearance {
            if seg_touches(seg[0], seg[1], 0.0) {
                out.clearance.push(*seg);
            }
        }
        for m in &c.patches.markers {
            if m.pos.x >= min.x
                && m.pos.x <= max.x
                && m.pos.y >= min.y
                && m.pos.y <= max.y
                && m.pos.z >= min.z
                && m.pos.z <= max.z
            {
                out.markers.push(m.clone());
            }
        }
    }
    out
}

impl LayerChunk for SitesChunk {
    type Layer = ScatterSites;

    fn create(&mut self, ctx: &ChunkCtx<'_, ScatterSites>) {
        *self = ctx.layer().build(ctx);
    }

    fn destroy(&mut self, _ctx: &ChunkCtx<'_, ScatterSites>) {
        self.sites.clear();
    }
}

impl LayerChunk for Sites3Chunk {
    type Layer = Scatter3Sites;

    fn create(&mut self, ctx: &ChunkCtx<'_, Scatter3Sites>) {
        *self = ctx.layer().build(ctx);
    }

    fn destroy(&mut self, _ctx: &ChunkCtx<'_, Scatter3Sites>) {
        self.sites.clear();
    }
}

impl LayerChunk for Paths3Chunk {
    type Layer = Connect3Paths;

    fn create(&mut self, ctx: &ChunkCtx<'_, Connect3Paths>) {
        *self = ctx.layer().build(ctx);
    }

    fn destroy(&mut self, _ctx: &ChunkCtx<'_, Connect3Paths>) {
        self.paths.clear();
    }
}

impl LayerChunk for PathsChunk {
    type Layer = ConnectPaths;

    fn create(&mut self, ctx: &ChunkCtx<'_, ConnectPaths>) {
        *self = ctx.layer().build(ctx);
    }

    fn destroy(&mut self, _ctx: &ChunkCtx<'_, ConnectPaths>) {
        self.paths.clear();
    }
}

impl LayerChunk for CoursesChunk {
    type Layer = FlowCourses;

    fn create(&mut self, ctx: &ChunkCtx<'_, FlowCourses>) {
        *self = ctx.layer().build(ctx);
    }

    fn destroy(&mut self, _ctx: &ChunkCtx<'_, FlowCourses>) {
        self.courses.clear();
    }
}

impl LayerChunk for WormsChunk {
    type Layer = WormBurrows;

    fn create(&mut self, ctx: &ChunkCtx<'_, WormBurrows>) {
        *self = ctx.layer().build(ctx);
    }

    fn destroy(&mut self, _ctx: &ChunkCtx<'_, WormBurrows>) {
        self.worms.clear();
    }
}

impl LayerChunk for PatchChunk {
    type Layer = EmitPatches;

    fn create(&mut self, ctx: &ChunkCtx<'_, EmitPatches>) {
        *self = ctx.layer().build(ctx);
    }

    fn destroy(&mut self, _ctx: &ChunkCtx<'_, EmitPatches>) {
        self.patches.ops.clear();
        self.patches.ribbons.clear();
        self.patches.clearance.clear();
        self.patches.markers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sites of a named scatter instance within bounds.
    fn sites_in(mgr: &LayerGraph, instance: &str, bounds: IAabb) -> Vec<Vec2> {
        let mut out = Vec::new();
        mgr.view::<ScatterSites>(instance, bounds)
            .for_each(|_, c| out.extend(c.sites.iter().copied()));
        out
    }

    /// A test world: layers registered up front, then residency started
    /// on the first read. Reads no longer generate, so a test has to hold
    /// a top dependency exactly like the game does.
    struct TestWorld {
        pending: std::sync::Mutex<Option<LayerGraph>>,
        started: std::sync::OnceLock<std::sync::Arc<LayerRuntime>>,
        focus: IVec3,
        size: IVec3,
    }

    impl TestWorld {
        /// Carries its generator in the graph context, exactly as the
        /// engine does — no globals involved.
        fn new(seed: u64) -> Self {
            Self {
                pending: std::sync::Mutex::new(Some(LayerGraph::with_context(
                    seed,
                    std::sync::Arc::new(crate::planning::world::WorldCtx::new(
                        std::sync::Arc::new(Generator::new(
                            voxel_worldgen::program::planet_program(),
                            seed as u32,
                            voxel_worldgen::program::DEFAULT_SUN_DIR,
                        )),
                    )),
                ))),
                started: std::sync::OnceLock::new(),
                focus: IVec3::ZERO,
                size: IVec3::splat(8192),
            }
        }

        fn around(mut self, focus: IVec3, size: IVec3) -> Self {
            self.focus = focus;
            self.size = size;
            self
        }

        fn register_as<L: Layer>(&mut self, instance: &str, layer: L) {
            self.pending
                .lock()
                .unwrap()
                .as_mut()
                .expect("register before the first read")
                .register_as(instance, layer);
        }

        /// One top dependency per registered instance — a test reads
        /// whatever it likes, so everything stays resident for it.
        fn graph(&self) -> &LayerGraph {
            self.started
                .get_or_init(|| {
                    let graph = self.pending.lock().unwrap().take().expect("started twice");
                    let tops = graph
                        .instances()
                        .iter()
                        .map(|name| TopDep::new(name, self.size))
                        .collect();
                    let runtime =
                        std::sync::Arc::new(LayerRuntime::start(std::sync::Arc::new(graph), tops));
                    for i in 0..runtime.tops() {
                        runtime.top(i).set_focus(self.focus);
                    }
                    runtime.wait_idle();
                    runtime
                })
                .graph()
        }
    }

    fn test_manager(seed: u64) -> TestWorld {
        TestWorld::new(seed)
    }

    /// The same world a `test_manager(seed)` generates for — assertions
    /// must sample the world under test, not a differently-seeded one.
    fn generator(seed: u32) -> Generator {
        Generator::new(
            voxel_worldgen::program::planet_program(),
            seed,
            voxel_worldgen::program::DEFAULT_SUN_DIR,
        )
    }
    use crate::planning::structure::{
        Anchor, Arrange, Extent, Part, Seat, Shape, Structure, Variant, Yaw,
    };
    use voxel_layers::{LayerRuntime, TopDep};

    /// A minimal structure for emit tests: one seated block per site.
    fn test_structure(material: u32, cut: bool, seat: Seat) -> std::sync::Arc<Structure> {
        std::sync::Arc::new(Structure {
            size: [4.0, 6.0],
            variants: vec![Variant {
                weight: 1.0,
                parts: vec![Part {
                    arrange: Arrange::Scatter {
                        count: [2, 4],
                        radius_frac: [0.0, 1.0],
                    },
                    shape: Shape::Boxy {
                        half: [
                            Extent::Range([1.0, 2.0]),
                            Extent::Range([1.0, 2.0]),
                            Extent::Range([1.0, 2.0]),
                        ],
                    },
                    material,
                    cut,
                    hollow: None,
                    skip: 0.0,
                    seat,
                    anchor: Anchor::Base,
                    y_offset: [-0.5, -0.5],
                    yaw: Yaw::Random,
                    link: None,
                }],
            }],
        })
    }

    fn bounds(r: i32) -> IAabb {
        IAabb::new(IVec3::new(-r, 0, -r), IVec3::new(r, 1, r))
    }

    /// A region of the reference planet known to be land (the test area
    /// around the shipped level's start) — world origin is open ocean and
    /// altitude-filtered scatters would be vacuously empty there. A test
    /// reading here has to focus its residency here too.
    const LAND: IVec3 = IVec3::new(-27000, 0, -38000);

    fn land_bounds(r: i32) -> IAabb {
        let c = LAND;
        IAabb::new(
            IVec3::new(c.x - r, 0, c.z - r),
            IVec3::new(c.x + r, 1, c.z + r),
        )
    }

    #[test]
    fn scatter_instances_are_independent_and_deterministic() {
        let mut mgr = test_manager(3);
        mgr.register_as(
            "sites:common",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.9,
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "sites:rare",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.05,
                    ..Default::default()
                },
            },
        );
        let common = sites_in(mgr.graph(), "sites:common", bounds(4096));
        let rare = sites_in(mgr.graph(), "sites:rare", bounds(4096));
        assert!(
            common.len() > rare.len() * 3,
            "chance config ignored: {} vs {}",
            common.len(),
            rare.len()
        );

        let mut mgr2 = test_manager(3);
        mgr2.register_as(
            "sites:common",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.9,
                    ..Default::default()
                },
            },
        );
        assert_eq!(common, sites_in(mgr2.graph(), "sites:common", bounds(4096)));
    }

    /// Relaxation is the pattern LayerProcGen offers internal levels
    /// for. Done as a second INSTANCE it has to actually improve
    /// spacing, stay deterministic, and — the constraint that keeps
    /// every existing consumer working — leave each site in its own cell.
    #[test]
    fn relaxing_sites_spreads_them_without_leaving_their_cells() {
        let cell = 256;
        let scattered = |mgr: &mut TestWorld| {
            mgr.register_as(
                "sites:raw",
                ScatterSites {
                    cfg: ScatterCfg {
                        cell_m: cell,
                        chance: 1.0,
                        margin_m: 8.0,
                        ..Default::default()
                    },
                },
            );
        };

        let mut plain = test_manager(3);
        scattered(&mut plain);
        let before = sites_in(plain.graph(), "sites:raw", bounds(4096));

        let mut relaxed = test_manager(3);
        scattered(&mut relaxed);
        relaxed.register_as(
            "sites",
            ScatterSites {
                cfg: ScatterCfg {
                    cell_m: cell,
                    margin_m: 8.0,
                    relax_from: Some(RelaxFrom {
                        instance: "sites:raw".into(),
                        strength: 0.35,
                    }),
                    ..Default::default()
                },
            },
        );
        let after = sites_in(relaxed.graph(), "sites", bounds(4096));

        assert_eq!(
            before.len(),
            after.len(),
            "relaxing must not add or drop sites"
        );
        assert!(!after.is_empty(), "the fixture must produce sites");

        // Every site stays in the cell that owns it, or consumers that
        // read without padding would stop finding it.
        for p in &after {
            let cx = (p.x / cell as f32).floor() as i32;
            let cz = (p.y / cell as f32).floor() as i32;
            let owner = before
                .iter()
                .filter(|q| {
                    (q.x / cell as f32).floor() as i32 == cx
                        && (q.y / cell as f32).floor() as i32 == cz
                })
                .count();
            assert_eq!(owner, 1, "site {p:?} is not alone in cell ({cx},{cz})");
        }

        // The point of the exercise: the crowded pairs get less crowded.
        let closest = |sites: &[Vec2]| {
            let mut d: Vec<f32> = sites
                .iter()
                .map(|a| {
                    sites
                        .iter()
                        .filter(|b| *b != a)
                        .map(|b| a.distance(*b))
                        .fold(f32::MAX, f32::min)
                })
                .collect();
            d.sort_by(f32::total_cmp);
            d
        };
        let (d0, d1) = (closest(&before), closest(&after));
        assert!(
            d1[0] > d0[0],
            "worst pair got no better: {} -> {}",
            d0[0],
            d1[0]
        );
        let mean = |d: &[f32]| d.iter().sum::<f32>() / d.len() as f32;
        assert!(
            mean(&d1) > mean(&d0),
            "mean spacing got no better: {} -> {}",
            mean(&d0),
            mean(&d1)
        );

        // Same inputs, same output.
        let mut again = test_manager(3);
        scattered(&mut again);
        again.register_as(
            "sites",
            ScatterSites {
                cfg: ScatterCfg {
                    cell_m: cell,
                    margin_m: 8.0,
                    relax_from: Some(RelaxFrom {
                        instance: "sites:raw".into(),
                        strength: 0.35,
                    }),
                    ..Default::default()
                },
            },
        );
        assert_eq!(after, sites_in(again.graph(), "sites", bounds(4096)));
    }

    #[test]
    fn connect_paths_join_sites_within_reach() {
        let mut mgr = test_manager(5).around(LAND, IVec3::new(9216, 0, 9216));
        mgr.register_as(
            "sites:towns",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.6,
                    altitude: [3.0, 400.0],
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "roads",
            ConnectPaths {
                cfg: ConnectCfg {
                    source: "sites:towns".into(),
                    ..Default::default()
                },
                cell_m: 256,
            },
        );
        let b = land_bounds(4096);
        let mut total = 0;
        for (_, c) in mgr.graph().view::<ConnectPaths>("roads", b).iter() {
            for path in &c.paths {
                total += 1;
                assert!(path.len() >= 2);
                let (a, z) = (path[0], *path.last().unwrap());
                assert!(a.distance(z) < 700.0 + 1.0);
                // Corridor containment: the emitter's padding contract.
                let lo = a.min(z) - Vec2::splat(192.0);
                let hi = a.max(z) + Vec2::splat(192.0);
                for w in path {
                    assert!(w.x >= lo.x && w.x <= hi.x && w.y >= lo.y && w.y <= hi.y);
                }
            }
        }
        assert!(total > 0, "no connections in 8 km x 8 km");
    }

    #[test]
    fn flow_and_worm_kinds_generate_from_scatter_instances() {
        let mut mgr = test_manager(5).around(LAND, IVec3::new(9216, 0, 9216));
        mgr.register_as(
            "sites:springs",
            ScatterSites {
                cfg: ScatterCfg {
                    cell_m: 512,
                    chance: 0.6,
                    altitude: [60.0, 400.0],
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "rivers",
            FlowCourses {
                cfg: FlowCfg {
                    source: "sites:springs".into(),
                    ..Default::default()
                },
                cell_m: 512,
            },
        );
        mgr.register_as(
            "sites:mouths",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.6,
                    altitude: [6.0, 500.0],
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "caves",
            WormBurrows {
                cfg: WormCfg {
                    source: "sites:mouths".into(),
                    ..Default::default()
                },
                cell_m: 256,
            },
        );
        let b = land_bounds(4096);
        let mut courses = 0;
        for (_, c) in mgr.graph().view::<FlowCourses>("rivers", b).iter() {
            for (wp, levels) in &c.courses {
                courses += 1;
                assert_eq!(wp.len(), levels.len());
                // Surface line is monotone non-increasing (it flows downhill).
                for w in levels.windows(2) {
                    assert!(w[1] <= w[0] + 1e-4);
                }
            }
        }
        assert!(courses > 0, "no rivers");
        let mut worms = 0;
        for (_, c) in mgr.graph().view::<WormBurrows>("caves", b).iter() {
            for worm in &c.worms {
                worms += 1;
                assert!(worm.len() as u32 == WormCfg::default().steps);
            }
        }
        assert!(worms > 0, "no worms");
    }

    #[test]
    fn emit_site_recipe_produces_ops_and_markers_at_sites() {
        let build = || {
            let mut mgr = test_manager(7).around(LAND, IVec3::new(9216, 0, 9216));
            mgr.register_as(
                "sites:ruins",
                ScatterSites {
                    cfg: ScatterCfg {
                        chance: 0.32,
                        altitude: [8.0, 280.0],
                        up: [0.88, 1.0],
                        ..Default::default()
                    },
                },
            );
            mgr.register_as(
                "ruins",
                EmitPatches {
                    cell_y_m: 0,
                    cfg: EmitCfg {
                        source: "sites:ruins".into(),
                        kind: EmitKind::SiteStructure {
                            structure: test_structure(3, false, Seat::Terrain),
                            marker: Some("ruin".into()),
                        },
                        pad_m: 0.0,
                    },
                    cell_m: 256,
                },
            );
            mgr
        };
        let mgr = build();
        let b = land_bounds(4096);
        let (min, max) = (
            Vec3::new(b.min.x as f32, -100.0, b.min.z as f32),
            Vec3::new(b.max.x as f32, 500.0, b.max.z as f32),
        );
        let patches = build_patches(mgr.graph(), "ruins", min, max);
        // sites_in returns whole overlapping cells; markers are filtered
        // to the exact box — compare against the filtered set.
        let all_sites = sites_in(mgr.graph(), "sites:ruins", b.inflate(IVec3::splat(512)));
        let sites: Vec<Vec2> = all_sites
            .iter()
            .copied()
            .filter(|s| s.x >= min.x && s.x <= max.x && s.y >= min.z && s.y <= max.z)
            .collect();
        assert!(!sites.is_empty(), "no ruin sites on land");
        assert!(!patches.ops.is_empty(), "recipe emitted no geometry");
        assert_eq!(patches.markers.len(), sites.len());
        for op in &patches.ops {
            let p = Vec2::new(op.center[0], op.center[2]);
            let near = all_sites.iter().any(|s| s.distance(p) < ELEM_PAD_M);
            assert!(near, "op at {p:?} far from every site");
        }
        assert_eq!(patches, build_patches(build().graph(), "ruins", min, max));
    }

    fn build_patches(mgr: &LayerGraph, instance: &str, min: Vec3, max: Vec3) -> PatchSet {
        patches_in(mgr, instance, min, max)
    }

    #[test]
    fn emit_path_slabs_seat_on_terrain_with_clearance() {
        let mut mgr = test_manager(5).around(LAND, IVec3::new(9216, 0, 9216));
        mgr.register_as(
            "sites:towns",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.6,
                    altitude: [3.0, 400.0],
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "paths:roads",
            ConnectPaths {
                cfg: ConnectCfg {
                    source: "sites:towns".into(),
                    ..Default::default()
                },
                cell_m: 256,
            },
        );
        mgr.register_as(
            "roads",
            EmitPatches {
                cell_y_m: 0,
                cfg: EmitCfg {
                    source: "paths:roads".into(),
                    kind: EmitKind::PathSlabs {
                        half_w: 2.4,
                        thickness: 0.5,
                        material: 3,
                        clearance: true,
                    },
                    // Endpoints within reach/2 of the midpoint cell, plus
                    // the pathfinding corridor.
                    pad_m: 700.0 * 0.5 + 192.0 + 64.0,
                },
                cell_m: 256,
            },
        );
        let b = land_bounds(4096);
        let (min, max) = (
            Vec3::new(b.min.x as f32, -100.0, b.min.z as f32),
            Vec3::new(b.max.x as f32, 500.0, b.max.z as f32),
        );
        let patches = patches_in(mgr.graph(), "roads", min, max);
        assert!(!patches.ops.is_empty(), "no road slabs");
        assert!(!patches.clearance.is_empty(), "no clearance segments");
        for op in &patches.ops {
            let ground = generator(5).height(Vec2::new(op.center[0], op.center[2]), 1.0);
            assert!((op.center[1] - ground).abs() < 2.0, "slab far from ground");
        }
        // Sub-box query = filtered superset (locality contract).
        let (smin, smax) = (
            min + Vec3::new(2048.0, 0.0, 2048.0),
            max - Vec3::new(2048.0, 0.0, 2048.0),
        );
        let sub = patches_in(mgr.graph(), "roads", smin, smax);
        let expect: Vec<_> = patches
            .ops
            .iter()
            .filter(|op| op.touches(voxel_core::csg::Aabb::new(smin, smax)))
            .copied()
            .collect();
        assert_eq!(sub.ops, expect);
    }

    #[test]
    fn emit_ribbon_and_worm_cuts() {
        let mut mgr = test_manager(5).around(LAND, IVec3::new(9216, 0, 9216));
        mgr.register_as(
            "sites:springs",
            ScatterSites {
                cfg: ScatterCfg {
                    cell_m: 512,
                    chance: 0.6,
                    altitude: [60.0, 400.0],
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "flow:rivers",
            FlowCourses {
                cfg: FlowCfg {
                    source: "sites:springs".into(),
                    ..Default::default()
                },
                cell_m: 512,
            },
        );
        mgr.register_as(
            "rivers",
            EmitPatches {
                cell_y_m: 0,
                cfg: EmitCfg {
                    source: "flow:rivers".into(),
                    kind: EmitKind::Ribbon {
                        material: 4,
                        width: [2.0, 7.0],
                    },
                    // Courses run up to max_steps * step from their spring.
                    pad_m: 400.0 * 8.0 + 64.0,
                },
                cell_m: 512,
            },
        );
        mgr.register_as(
            "sites:mouths",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.6,
                    altitude: [6.0, 500.0],
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "worm:caves",
            WormBurrows {
                cfg: WormCfg {
                    source: "sites:mouths".into(),
                    ..Default::default()
                },
                cell_m: 256,
            },
        );
        mgr.register_as(
            "caves",
            EmitPatches {
                cell_y_m: 0,
                cfg: EmitCfg {
                    source: "worm:caves".into(),
                    kind: EmitKind::WormCuts,
                    pad_m: 340.0,
                },
                cell_m: 256,
            },
        );
        let b = land_bounds(4096);
        let (min, max) = (
            Vec3::new(b.min.x as f32, -200.0, b.min.z as f32),
            Vec3::new(b.max.x as f32, 500.0, b.max.z as f32),
        );
        let rivers = patches_in(mgr.graph(), "rivers", min, max);
        assert!(!rivers.ribbons.is_empty(), "no ribbon segments");
        assert!(!rivers.ops.is_empty(), "no river bed ops");
        assert!(!rivers.clearance.is_empty(), "no river clearance");
        // The surface is drawn from segments, never baked into the SDF:
        // every bed op is a cut.
        for op in &rivers.ops {
            assert_eq!(op.kind & 1, 1, "river emitted an additive op");
        }
        for w in &rivers.ribbons {
            assert!(w.half_w >= 2.0 && w.half_w <= 7.0);
            assert!(
                w.levels[1] <= w.levels[0] + 1e-4,
                "ribbon surface flows uphill"
            );
            assert_eq!(w.material, 4);
        }
        let caves = patches_in(mgr.graph(), "caves", min, max);
        assert!(!caves.ops.is_empty(), "no cave cuts");
        for op in &caves.ops {
            assert_eq!(op.kind, voxel_core::csg::CSG_KIND_SPHERE_CUT);
        }
    }

    #[test]
    fn interior_stack_links_pockets_with_orthogonal_tubes() {
        let mut mgr = test_manager(9).around(IVec3::ZERO, IVec3::new(3072, 768, 3072));
        mgr.register_as(
            "sites:pockets",
            Scatter3Sites {
                cfg: Scatter3Cfg {
                    snap_y_m: 44.0,
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "links",
            Connect3Paths {
                cfg: Connect3Cfg {
                    source: "sites:pockets".into(),
                    ..Default::default()
                },
                cell_m: 128,
                cell_y_m: 132,
            },
        );
        mgr.register_as(
            "pockets",
            EmitPatches {
                cell_y_m: 132,
                cfg: EmitCfg {
                    source: "sites:pockets".into(),
                    kind: EmitKind::SiteStructure3 {
                        structure: test_structure(2, false, Seat::Site),
                        marker: Some("pocket".into()),
                    },
                    pad_m: 0.0,
                },
                cell_m: 128,
            },
        );
        mgr.register_as(
            "tubes",
            EmitPatches {
                cell_y_m: 132,
                cfg: EmitCfg {
                    source: "links".into(),
                    kind: EmitKind::Tubes {
                        material: 2,
                        bore: 1.5,
                        lift_m: 3.0,
                    },
                    pad_m: 400.0 + 64.0,
                },
                cell_m: 128,
            },
        );
        let b = IAabb::new(IVec3::new(-1024, -264, -1024), IVec3::new(1024, 264, 1024));

        // Sites snap to the floor lattice.
        let mut sites = Vec::new();
        for (_, c) in mgr.graph().view::<Scatter3Sites>("sites:pockets", b).iter() {
            for s in &c.sites {
                assert!((s.y / 44.0 - (s.y / 44.0).round()).abs() < 1e-3);
                sites.push(*s);
            }
        }
        assert!(sites.len() > 4, "too few pocket sites: {}", sites.len());

        // Links exist, are orthogonal, and join real sites within reach.
        let mut links = 0;
        for (_, c) in mgr.graph().view::<Connect3Paths>("links", b).iter() {
            for path in &c.paths {
                links += 1;
                assert!(path[0].distance(*path.last().unwrap()) < 400.0);
                for seg in path.windows(2) {
                    let d = seg[1] - seg[0];
                    let moving = (d.x.abs() > 0.01) as u8
                        + (d.y.abs() > 0.01) as u8
                        + (d.z.abs() > 0.01) as u8;
                    assert_eq!(moving, 1, "diagonal corridor segment: {d:?}");
                }
            }
        }
        assert!(links > 0, "no links");

        // Emissions: pocket shells + markers, tube shells enclosing bores.
        let (min, max) = (
            Vec3::new(-1024.0, -264.0, -1024.0),
            Vec3::new(1024.0, 264.0, 1024.0),
        );
        let pockets = patches_in(mgr.graph(), "pockets", min, max);
        assert!(!pockets.ops.is_empty());
        assert!(!pockets.markers.is_empty());
        let tubes = patches_in(mgr.graph(), "tubes", min, max);
        assert!(!tubes.ops.is_empty(), "no tube geometry");
        assert!(
            tubes.ops.iter().any(|op| op.kind & 1 == 0)
                && tubes.ops.iter().any(|op| op.kind & 1 == 1),
            "tubes need both shell adds and bore cuts"
        );
        // Determinism.
        let mut mgr2 = test_manager(9).around(IVec3::ZERO, IVec3::new(3072, 768, 3072));
        mgr2.register_as(
            "sites:pockets",
            Scatter3Sites {
                cfg: Scatter3Cfg {
                    snap_y_m: 44.0,
                    ..Default::default()
                },
            },
        );
        let mut sites2 = Vec::new();
        for (_, c) in mgr2
            .graph()
            .view::<Scatter3Sites>("sites:pockets", b)
            .iter()
        {
            sites2.extend(c.sites.iter().copied());
        }
        assert_eq!(sites, sites2);
    }

    /// Region weights come from the generator's own bands, so they
    /// partition the plane and every declared region is reachable.
    #[test]
    fn region_weights_partition_and_every_region_is_reachable() {
        let generator = generator(3);
        let mats = [1u32, 2, 5, 6, 7];
        let mut dominant = [false; 5];
        for i in 0..400 {
            let p = Vec2::new(
                LAND.x as f32 + (i % 20) as f32 * 1700.0,
                LAND.z as f32 + (i / 20) as f32 * 1700.0,
            );
            let w: Vec<f32> = mats
                .iter()
                .map(|&m| generator.surface_material_weight(p, 8.0, m))
                .collect();
            let sum: f32 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1.0e-3,
                "weights must partition: {w:?} sums to {sum}"
            );
            let best = w
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap();
            dominant[best] = true;
        }
        assert!(
            dominant.iter().all(|&d| d),
            "some region is never dominant: {dominant:?}"
        );
    }

    #[test]
    fn biome_gated_scatter_concentrates_in_its_biome() {
        let mut mgr = test_manager(11).around(LAND, IVec3::new(17408, 0, 17408));
        // Gate on the forest region — material 1, the one the planet's
        // `height_surface` lays down and no band repaints.
        mgr.register_as(
            "sites:gated",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 1.0,
                    biome: Some(BiomeGate { material: 1 }),
                    ..Default::default()
                },
            },
        );
        let b = land_bounds(8192);
        let sites = sites_in(mgr.graph(), "sites:gated", b);
        assert!(!sites.is_empty(), "gate rejected everything");
        // Accepted sites average a high weight of their region; the
        // probabilistic gate keeps some low-weight border sites (blending).
        let generator = generator(11);
        let mean: f32 = sites
            .iter()
            .map(|&p| generator.surface_material_weight(p, 8.0, 1))
            .sum::<f32>()
            / sites.len() as f32;
        // Acceptance probability IS the weight, so the accepted mean is
        // E[w^2]/E[w], which exceeds the unconditioned mean by the
        // weight's variance over it. Compared against the unconditioned
        // mean measured on the same ground rather than a constant, which
        // would only re-encode how much of the world is forest today.
        let b = land_bounds(8192);
        let mut unconditioned = 0.0f32;
        let n = 40;
        for gz in 0..n {
            for gx in 0..n {
                let p = Vec2::new(
                    b.min.x as f32 + (b.max.x - b.min.x) as f32 * gx as f32 / (n - 1) as f32,
                    b.min.z as f32 + (b.max.z - b.min.z) as f32 * gz as f32 / (n - 1) as f32,
                );
                unconditioned += generator.surface_material_weight(p, 8.0, 1);
            }
        }
        unconditioned /= (n * n) as f32;
        assert!(
            mean > unconditioned * 1.15,
            "gated sites not concentrated: accepted mean {mean} vs unconditioned {unconditioned}"
        );
    }

    /// Regression for the audit's C1/C2/M2: volumetric emits must serve
    /// every y row, vertical link legs must be emitted by the rows they
    /// cross, and floor-snapped sites must stay inside their owning cell.
    #[test]
    fn volumetric_emits_cover_all_y_rows_and_vertical_legs() {
        let mut mgr = test_manager(9).around(IVec3::ZERO, IVec3::new(3072, 1536, 3072));
        mgr.register_as(
            "sites:pockets",
            Scatter3Sites {
                cfg: Scatter3Cfg {
                    snap_y_m: 44.0,
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "links",
            Connect3Paths {
                cfg: Connect3Cfg {
                    source: "sites:pockets".into(),
                    ..Default::default()
                },
                cell_m: 128,
                cell_y_m: 132,
            },
        );
        mgr.register_as(
            "pockets",
            EmitPatches {
                cell_y_m: 132,
                cfg: EmitCfg {
                    source: "sites:pockets".into(),
                    kind: EmitKind::SiteStructure3 {
                        structure: test_structure(2, false, Seat::Site),
                        marker: Some("pocket".into()),
                    },
                    pad_m: 0.0,
                },
                cell_m: 128,
            },
        );
        mgr.register_as(
            "tubes",
            EmitPatches {
                cell_y_m: 132,
                cfg: EmitCfg {
                    source: "links".into(),
                    kind: EmitKind::Tubes {
                        material: 2,
                        bore: 1.5,
                        lift_m: 3.0,
                    },
                    pad_m: 464.0,
                },
                cell_m: 128,
            },
        );

        // Sites stay strictly inside their owning cell after snapping.
        let b = IAabb::new(IVec3::new(-1024, -528, -1024), IVec3::new(1024, 528, 1024));
        for (coord, c) in mgr.graph().view::<Scatter3Sites>("sites:pockets", b).iter() {
            for site in &c.sites {
                let lo = coord.y * 132;
                let hi = lo + 132;
                assert!(
                    site.y >= lo as f32 && site.y < hi as f32,
                    "snapped site y {} outside its cell [{lo}, {hi})",
                    site.y
                );
            }
        }

        // Markers surface from EVERY y row that has sites — including a
        // facade-style sentinel-y query.
        let all = patches_in(
            mgr.graph(),
            "pockets",
            Vec3::new(-1024.0, -1.0e9, -1024.0),
            Vec3::new(1024.0, 1.0e9, 1024.0),
        );
        let mut rows: Vec<i32> = all
            .markers
            .iter()
            .map(|m| (m.pos.y / 132.0).floor() as i32)
            .collect();
        rows.sort_unstable();
        rows.dedup();
        assert!(
            rows.len() >= 3,
            "markers confined to y rows {rows:?} — volumetric rows not served"
        );
        // A bounded-row query returns exactly that row's markers.
        let row1 = patches_in(
            mgr.graph(),
            "pockets",
            Vec3::new(-1024.0, 132.0, -1024.0),
            Vec3::new(1024.0, 264.0, 1024.0),
        );
        assert!(row1
            .markers
            .iter()
            .all(|m| (132.0..=264.0).contains(&m.pos.y)));
        assert!(
            !row1.markers.is_empty(),
            "no markers in y row 1 — either bad luck (seed change?) or rows unserved"
        );

        // Every vertical link leg is emitted by the rows it crosses: ops
        // exist near the leg midpoint even when it is far from the link
        // midpoint's owning row.
        let mut vertical_checked = 0;
        for (_, c) in mgr.graph().view::<Connect3Paths>("links", b).iter() {
            for path in &c.paths {
                for seg in path.windows(2) {
                    let d = seg[1] - seg[0];
                    if d.y.abs() < 100.0 {
                        continue;
                    }
                    let mid = (seg[0] + seg[1]) * 0.5 + Vec3::Y * 3.0;
                    let near = patches_in(
                        mgr.graph(),
                        "tubes",
                        mid - Vec3::splat(20.0),
                        mid + Vec3::splat(20.0),
                    );
                    assert!(
                        !near.ops.is_empty(),
                        "no tube ops near vertical leg midpoint {mid:?}"
                    );
                    vertical_checked += 1;
                }
            }
        }
        assert!(vertical_checked > 0, "no vertical legs found to check");
    }

    #[test]
    fn scatter_respects_terrain_filters() {
        let mut mgr = test_manager(3).around(LAND, IVec3::new(13312, 0, 13312));
        mgr.register_as(
            "sites:highland",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 1.0,
                    altitude: [120.0, 10_000.0],
                    up: [0.8, 1.0],
                    ..Default::default()
                },
            },
        );
        let found = sites_in(mgr.graph(), "sites:highland", land_bounds(6000));
        assert!(!found.is_empty(), "no highland sites on the land region");
        let generator = generator(3);
        for p in found {
            let h = generator.height(p, 8.0);
            let up = generator.up(p, 8.0);
            assert!(h >= 120.0, "lowland site at {p:?} h={h}");
            assert!(up >= 0.8, "steep site at {p:?} up={up}");
        }
    }
}
