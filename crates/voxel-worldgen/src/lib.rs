//! World-generation primitives: the CPU twin of the GPU generator-program
//! interpreter ([`program`]), plus the building blocks planning layers are
//! written from — a grid pathfinder ([`path`]) and a descent walk
//! ([`flow`]).
//!
//! There are no layers here. Concrete layers are the game's code (see
//! `voxel_engine::planning::WorldPlanner`); this crate only supplies the
//! pieces they compute with.
//!
//! [`program::eval`] MUST stay bit-compatible with
//! `voxel-render/src/shaders/voxel_world_density.wgsl` — vegetation and
//! gameplay place things on the surface the GPU generates.

pub mod flow;
pub mod path;
pub mod program;

use glam::{Vec2, Vec3};

/// Mirrors the WGSL `hash2` (level seed mixed in; 0 = identity).
pub(crate) fn hash2(seed: u32, p: glam::IVec2) -> f32 {
    let mut h: u32 = (p.x as u32)
        .wrapping_mul(374_761_393)
        .wrapping_add((p.y as u32).wrapping_mul(668_265_263))
        .wrapping_add(seed.wrapping_mul(2_654_435_769));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0xFF_FFFF) as f32 / 16_777_216.0
}

/// Mirrors the WGSL `value_noise` (quintic smoothstep).
fn value_noise(seed: u32, p: Vec2) -> f32 {
    let i = p.floor();
    let f = p - i;
    let i = glam::IVec2::new(i.x as i32, i.y as i32);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = hash2(seed, i);
    let b = hash2(seed, i + glam::IVec2::new(1, 0));
    let c = hash2(seed, i + glam::IVec2::new(0, 1));
    let d = hash2(seed, i + glam::IVec2::new(1, 1));
    let ab = a + (b - a) * u.x;
    let cd = c + (d - c) * u.x;
    ab + (cd - ab) * u.y
}

/// The generator is unbanded: a pure function of position, so all LODs
/// sample identical values and seams cannot disagree. (Kept as a hook —
/// per-LOD band-limiting must never return without a seam-exactness
/// story.)
pub(crate) fn band_fade(_wavelength: f32, _voxel_size: f32) -> f32 {
    1.0
}

pub(crate) fn fbm(seed: u32, p: Vec2, base_scale: f32, octaves: i32, voxel_size: f32) -> f32 {
    fbm_mode(seed, p, base_scale, octaves, voxel_size, 0)
}

/// The range of `value_noise` over an xz BOX, from the lattice itself.
///
/// The noise interpolates four corner hashes with weights that are
/// convex — a quintic of a value in [0,1], so each weight is in [0,1] and
/// they sum to one. A convex combination cannot leave the span of the
/// values it combines, so the min and max of every corner the box touches
/// IS a bound, and a tight one.
///
/// A box spanning many cells touches so many corners that the bound
/// approaches the noise's full [0,1] anyway, so past a few cells this
/// stops enumerating and says so.
fn value_noise_range(seed: u32, lo: Vec2, hi: Vec2) -> voxel_core::interval::Interval {
    use voxel_core::interval::Interval;
    /// Cells per axis past which enumeration stops paying.
    const MAX_CELLS: i32 = 4;
    let (x0, y0) = (lo.x.floor() as i32, lo.y.floor() as i32);
    let (x1, y1) = (hi.x.floor() as i32, hi.y.floor() as i32);
    if x1.saturating_sub(x0) > MAX_CELLS || y1.saturating_sub(y0) > MAX_CELLS {
        return Interval::new(0.0, 1.0);
    }
    // Plain floats, not an empty Interval: `Interval::new` orders its
    // ends, so an "empty" accumulator built from ±inf comes back as all
    // of the number line and never narrows.
    let (mut lo_v, mut hi_v) = (f32::INFINITY, f32::NEG_INFINITY);
    for y in y0..=y1 + 1 {
        for x in x0..=x1 + 1 {
            let v = hash2(seed, glam::IVec2::new(x, y));
            lo_v = lo_v.min(v);
            hi_v = hi_v.max(v);
        }
    }
    Interval::new(lo_v, hi_v)
}

/// Bounds an FBM over an xz BOX, rather than at a point.
///
/// The amplitude alone bounds it everywhere — `±0.5 * amp` — but that is
/// the whole world's range, so it decides nothing about a chunk near the
/// ground. Two independent bounds are available per octave, tight in
/// opposite regimes, and since both are valid the INTERSECTION is too:
///
/// - a gradient bound around the box's middle. Value noise interpolates
///   corner hashes in [0,1) with a quintic whose slope peaks at 1.875 per
///   lattice unit; ridged and billow fold that through `|2n-1|`, doubling
///   it. Tight when the box is small against the wavelength.
/// - the span of the lattice corners the box touches. Interpolation
///   weights are convex, so the value cannot leave that span. Tight once
///   the box is comparable to a cell, where the gradient bound has
///   already exceeded the full swing.
///
/// Taking only the corner span is much worse: a box a thousandth of a
/// cell across still touches corners spanning the whole cell, which on
/// the shipped planet turned a 33 m bound into a 900 m one.
///
/// Conservative by construction and checked by sampling: see
/// `program::range_tests`.
pub(crate) fn fbm_range(
    seed: u32,
    lo: Vec2,
    hi: Vec2,
    base_scale: f32,
    octaves: i32,
    voxel_size: f32,
    mode: u32,
) -> voxel_core::interval::Interval {
    use voxel_core::interval::Interval;
    let slope = if mode == 0 { 1.875 } else { 3.75 };
    let mid = (lo + hi) * 0.5;
    let reach = (hi - lo) * 0.5;
    let mut sum = Interval::point(0.0);
    let mut amp = 0.5;
    let mut freq = base_scale;
    for _ in 0..octaves {
        let fade = band_fade(1.0 / freq, voxel_size);
        // This octave's own contribution at the middle, and how far it can
        // move away from that across the box.
        let n = value_noise(seed, mid * freq);
        let centre = match mode {
            1 => 0.5 - (2.0 * n - 1.0).abs(),
            2 => (2.0 * n - 1.0).abs() - 0.5,
            _ => n - 0.5,
        };
        let span = value_noise_range(seed, lo * freq, hi * freq);
        // The slope constant is per unit of noise VALUE change, and the
        // corners the box touches bound how much value is available to
        // change over: 1.875 assumes adjacent corners differ by the whole
        // of [0,1), where a typical pair differs by about a third.
        let full = fade * 0.5;
        let slid = slope * span.width() * freq * (reach.x + reach.y);
        let gradient = Interval::new(centre - slid.min(full), centre + slid.min(full));
        // The same shapings, applied to the corner span.
        let corners = match mode {
            1 => Interval::point(0.5) - (span * 2.0 - 1.0).abs(),
            2 => (span * 2.0 - 1.0).abs() - 0.5,
            _ => span - 0.5,
        } * fade;
        // Both hold, so the tighter ends of each do.
        let octave = Interval {
            lo: gradient.lo.max(corners.lo),
            hi: gradient.hi.min(corners.hi),
        };
        sum = sum + octave * amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum
}

/// FBM with a per-octave shaping mode: 0 plain, 1 ridged (sharp crests),
/// 2 billow (rounded mounds). Mirrors the WGSL exactly.
pub(crate) fn fbm_mode(
    seed: u32,
    p: Vec2,
    base_scale: f32,
    octaves: i32,
    voxel_size: f32,
    mode: u32,
) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = base_scale;
    for _ in 0..octaves {
        let fade = band_fade(1.0 / freq, voxel_size);
        let n = value_noise(seed, p * freq);
        let v = match mode {
            1 => 0.5 - (2.0 * n - 1.0).abs(),
            2 => (2.0 * n - 1.0).abs() - 0.5,
            _ => n - 0.5,
        };
        sum += amp * fade * v;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum
}

/// A world's generator: its op program, seed and sun direction — the
/// whole CPU-side world in one value.
///
/// Everything that samples the world takes one of these, so an app can
/// host several worlds at once (two planets, a planet and an interior,
/// a preview world beside the live one). Clone is cheap: the ops are
/// shared.
#[derive(Clone)]
pub struct Generator {
    ops: std::sync::Arc<Vec<voxel_core::worldop::WorldOp>>,
    seed: u32,
    sun: glam::Vec3,
}

impl Generator {
    pub fn new(ops: Vec<voxel_core::worldop::WorldOp>, seed: u32, sun: glam::Vec3) -> Self {
        Self {
            ops: std::sync::Arc::new(ops),
            seed,
            sun,
        }
    }

    pub fn ops(&self) -> &[voxel_core::worldop::WorldOp] {
        &self.ops
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Direction the sun comes from (normalized).
    pub fn sun_direction(&self) -> glam::Vec3 {
        self.sun.normalize_or(glam::Vec3::Y)
    }

    /// Signed distance and material at a point — the CPU twin of the
    /// density shader.
    pub fn sample(&self, p: glam::Vec3, voxel_size: f32) -> (f32, u32) {
        program::eval(&self.ops, self.seed, p, voxel_size)
    }

    /// How firmly the program paints `material` on the ground here: 1
    /// well inside the region that paints it, 0 outside, soft across the
    /// edge. See [`program::surface_material_weight`].
    pub fn surface_material_weight(&self, xz: Vec2, voxel_size: f32, material: u32) -> f32 {
        program::surface_material_weight(&self.ops, self.seed, xz, voxel_size, material)
    }

    /// Heightfield (meters) at a world XZ, evaluated at `voxel_size`
    /// (1.0 = full detail). Mirrors the GPU exactly.
    pub fn height(&self, xz: Vec2, voxel_size: f32) -> f32 {
        program::eval_height(&self.ops, self.seed, xz, voxel_size)
    }

    /// Field registers at a column (prop densities, gameplay queries).
    pub fn fields(&self, xz: Vec2) -> [f32; voxel_core::worldop::FIELD_SLOTS] {
        program::eval_fields(&self.ops, self.seed, xz, 4.0)
    }

    /// Bounds on this world's SDF over a box, or `None` if the program
    /// has an op nobody has taught to bound itself. See
    /// [`program::eval_range`].
    pub fn range(
        &self,
        min: glam::Vec3,
        max: glam::Vec3,
        voxel_size: f32,
    ) -> Option<voxel_core::interval::Interval> {
        program::eval_range(&self.ops, self.seed, min, max, voxel_size)
    }

    /// The structural Y-lattice spacing, if the program has one.
    pub fn lattice_y_spacing(&self) -> Option<f32> {
        program::lattice_y_spacing(&self.ops)
    }

    /// Patch density in [0, 1]: slow spatial noise so scattered props
    /// come in coherent patches with clearings.
    pub fn patch_density(
        &self,
        xz: Vec2,
        scale: f32,
        offset: Vec2,
        contrast: f32,
        bias: f32,
    ) -> f32 {
        let n = fbm(self.seed, xz + offset, scale, 3, 1.0) + 0.5;
        (n * contrast + bias).clamp(0.0, 1.0)
    }

    /// Every floor in a vertical span at one column, deepest last: a `y`
    /// where the FULL program has air above and solid below.
    ///
    /// The heightfield answers "where is the ground" with one number per
    /// column, which is the whole world for a planet and none of it for an
    /// interior — a lattice of slabs has a floor every few metres and the
    /// height chain does not know any of them exist. This walks the actual
    /// SDF instead, so a cave floor, the underside of an overhang and a
    /// megastructure's twentieth storey are all just floors.
    ///
    /// SAMPLED, not solved: `eval_range` declines on any program with a
    /// lattice in it (nothing bounds those ops yet), so there is no
    /// interval to prune with and this walks at `step` and bisects each
    /// sign change. **`step` must be finer than the thinnest floor to be
    /// found** — a slab thinner than one step can fall between two samples
    /// and not exist. The shipped interior's thinnest is 1.2 m thick.
    ///
    /// Appends to `out` rather than returning, because the caller is a
    /// placement loop running this per attempt and a Vec per attempt is
    /// the allocation that costs more than the march.
    pub fn floors(&self, xz: Vec2, span: [f32; 2], step: f32, voxel_size: f32, out: &mut Vec<f32>) {
        let step = step.max(0.05);
        let solid = |y: f32| self.sample(Vec3::new(xz.x, y, xz.y), voxel_size).0 <= 0.0;
        let mut y = span[1];
        let mut above = solid(y);
        while y > span[0] {
            let next = (y - step).max(span[0]);
            let below = solid(next);
            // Air over solid, walking down: a surface you could stand on.
            // Solid over air is a ceiling, and nothing is placed on those.
            if !above && below {
                let (mut lo, mut hi) = (next, y);
                for _ in 0..8 {
                    let mid = 0.5 * (lo + hi);
                    if solid(mid) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                out.push(hi);
            }
            above = below;
            if next <= span[0] {
                break;
            }
            y = next;
        }
    }

    /// Surface normal from the SDF gradient — the volumetric twin of
    /// [`Generator::normal`], which can only ever describe a heightfield.
    pub fn normal_at(&self, p: Vec3, voxel_size: f32) -> Vec3 {
        let e = 0.5;
        let d = |o: Vec3| self.sample(p + o, voxel_size).0;
        let g = Vec3::new(
            d(Vec3::new(e, 0.0, 0.0)) - d(Vec3::new(-e, 0.0, 0.0)),
            d(Vec3::new(0.0, e, 0.0)) - d(Vec3::new(0.0, -e, 0.0)),
            d(Vec3::new(0.0, 0.0, e)) - d(Vec3::new(0.0, 0.0, -e)),
        );
        g.normalize_or(Vec3::Y)
    }

    /// Approximate surface up-ness (normal Y) via central differences.
    pub fn up(&self, xz: Vec2, voxel_size: f32) -> f32 {
        self.normal(xz, voxel_size).y
    }

    /// Approximate surface normal via central differences.
    pub fn normal(&self, xz: Vec2, voxel_size: f32) -> glam::Vec3 {
        let e = 2.0;
        let hx = self.height(xz + Vec2::new(e, 0.0), voxel_size)
            - self.height(xz - Vec2::new(e, 0.0), voxel_size);
        let hz = self.height(xz + Vec2::new(0.0, e), voxel_size)
            - self.height(xz - Vec2::new(0.0, e), voxel_size);
        glam::Vec3::new(-hx, 2.0 * e, -hz).normalize()
    }
}

#[cfg(test)]
mod floor_tests {
    use super::*;

    /// The interior has floors, and the heightfield cannot see one of
    /// them. This is the whole reason the march exists.
    #[test]
    fn the_interior_has_floors_the_heightfield_does_not_know_about() {
        let g = Generator::new(program::mega_program(), 0, program::DEFAULT_SUN_DIR);
        assert_eq!(
            g.height(Vec2::new(120.0, -80.0), 1.0),
            0.0,
            "an interior has no height chain, so every column reads 0"
        );
        let mut found = 0;
        let mut floors = Vec::new();
        for i in 0..40 {
            let xz = Vec2::new(i as f32 * 37.0 - 700.0, i as f32 * -53.0 + 400.0);
            floors.clear();
            g.floors(xz, [-200.0, 200.0], 0.5, 1.0, &mut floors);
            for &y in &floors {
                // Standing ON it: solid just below, air just above.
                assert!(
                    g.sample(Vec3::new(xz.x, y - 0.3, xz.y), 1.0).0 <= 0.0,
                    "no solid under the floor at {y}"
                );
                assert!(
                    g.sample(Vec3::new(xz.x, y + 0.3, xz.y), 1.0).0 > 0.0,
                    "no air over the floor at {y}"
                );
            }
            found += floors.len();
        }
        assert!(found > 40, "400 m of interior column found {found} floors");
    }

    /// And on a planet it agrees with the heightfield it replaces: the
    /// topmost floor over open ground IS the terrain height.
    #[test]
    fn the_topmost_floor_is_the_heightfield_where_there_is_one() {
        let g = Generator::new(program::planet_program(), 0, program::DEFAULT_SUN_DIR);
        let mut floors = Vec::new();
        let mut checked = 0;
        for i in 0..60 {
            let xz = Vec2::new(i as f32 * 211.0 - 6000.0, i as f32 * -137.0 + 2000.0);
            let h = g.height(xz, 1.0);
            floors.clear();
            g.floors(xz, [h - 40.0, h + 40.0], 0.25, 1.0, &mut floors);
            let Some(&top) = floors.first() else { continue };
            assert!(
                (top - h).abs() < 0.5,
                "topmost floor {top} but the heightfield says {h}"
            );
            checked += 1;
        }
        assert!(checked > 40, "only {checked} columns had a surface at all");
    }

    /// A floor is flat-side-up: the gradient normal points at the sky.
    #[test]
    fn a_floor_normal_points_up() {
        let g = Generator::new(program::mega_program(), 0, program::DEFAULT_SUN_DIR);
        let mut floors = Vec::new();
        let mut checked = 0;
        for i in 0..30 {
            let xz = Vec2::new(i as f32 * 71.0 - 500.0, i as f32 * 43.0);
            floors.clear();
            g.floors(xz, [-150.0, 150.0], 0.5, 1.0, &mut floors);
            for &y in floors.iter() {
                let n = g.normal_at(Vec3::new(xz.x, y + 0.1, xz.y), 1.0);
                assert!(n.y > 0.5, "floor normal {n:?} at {y} is not up");
                checked += 1;
            }
        }
        assert!(checked > 20);
    }
}
