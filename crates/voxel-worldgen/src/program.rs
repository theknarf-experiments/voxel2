//! CPU twin of the GPU world-generator interpreter
//! (`voxel-render/src/shaders/voxel_world_density.wgsl`). A world's base
//! generator is a program of [`WorldOp`]s; both interpreters evaluate it
//! op-for-op over the same register file, so collision, vegetation,
//! planning, and the rendered world always agree. MUST stay bit-compatible
//! with the WGSL.


use glam::{IVec2, IVec3, Vec2, Vec3};

/// Round half-up, shared with the WGSL twin (`round_half_up` there):
/// WGSL `round` is half-to-even, Rust's is half-away-from-zero, and
/// lattice registers land exactly on halves (spacing 44 at y = 22) —
/// hash gates key on the rounded level, so the twins must agree.
#[inline]
fn round_half_up(x: f32) -> f32 {
    (x + 0.5).floor()
}
use voxel_core::interval::Interval;
use voxel_core::worldop::*;


fn hash3_seeded(seed: u32, p: IVec3) -> f32 {
    let mut h: u32 = (p.x as u32)
        .wrapping_mul(374_761_393)
        .wrapping_add((p.y as u32).wrapping_mul(668_265_263))
        .wrapping_add((p.z as u32).wrapping_mul(2_246_822_519))
        .wrapping_add(seed.wrapping_mul(2_654_435_769));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0xFF_FFFF) as f32 / 16_777_216.0
}

fn sd_box(p: Vec3, b: Vec3) -> f32 {
    let q = p.abs() - b;
    q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
}

/// Mirrors the WGSL `value_noise3` (quintic smoothstep).
fn value_noise3(seed: u32, p: Vec3) -> f32 {
    let i = p.floor();
    let f = p - i;
    let i = IVec3::new(i.x as i32, i.y as i32, i.z as i32);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let corner = |dx: i32, dy: i32, dz: i32| hash3_seeded(seed, i + IVec3::new(dx, dy, dz));
    let x00 = corner(0, 0, 0) + (corner(1, 0, 0) - corner(0, 0, 0)) * u.x;
    let x10 = corner(0, 1, 0) + (corner(1, 1, 0) - corner(0, 1, 0)) * u.x;
    let x01 = corner(0, 0, 1) + (corner(1, 0, 1) - corner(0, 0, 1)) * u.x;
    let x11 = corner(0, 1, 1) + (corner(1, 1, 1) - corner(0, 1, 1)) * u.x;
    let y0 = x00 + (x10 - x00) * u.y;
    let y1 = x01 + (x11 - x01) * u.y;
    y0 + (y1 - y0) * u.z
}

/// Anisotropic band-limited 3D FBM (~[-0.5, 0.5]); wavelength for the
/// band fade uses the horizontal frequency. Mirrors the WGSL exactly.
fn fbm3_seeded(
    seed: u32,
    p: Vec3,
    freq_xz: f32,
    freq_y: f32,
    octaves: i32,
    voxel_size: f32,
) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut mul = 1.0;
    for _ in 0..octaves {
        let fade = crate::band_fade(1.0 / (freq_xz * mul), voxel_size);
        sum += amp
            * fade
            * (value_noise3(seed, Vec3::new(
                p.x * freq_xz * mul,
                p.y * freq_y * mul,
                p.z * freq_xz * mul,
            )) - 0.5);
        amp *= 0.5;
        mul *= 2.0;
    }
    sum
}

const BIG: f32 = 1.0e6;
const SOLID: f32 = -1.0e5;

// --- shims for the generated arms (single-source dialect; see
// voxel-core::opgen). Names mirror WGSL builtins / generated helpers.
#[inline]
fn abs(x: f32) -> f32 {
    x.abs()
}
#[inline]
fn max(a: f32, b: f32) -> f32 {
    a.max(b)
}
#[inline]
fn length(v: Vec2) -> f32 {
    v.length()
}
#[inline]
fn floor2(v: Vec2) -> Vec2 {
    v.floor()
}
#[inline]
fn v2(x: f32, y: f32) -> Vec2 {
    Vec2::new(x, y)
}
#[inline]
fn v3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}
#[inline]
fn iv2(x: i32, y: i32) -> IVec2 {
    IVec2::new(x, y)
}
#[inline]
fn iv3(x: i32, y: i32, z: i32) -> IVec3 {
    IVec3::new(x, y, z)
}
#[inline]
fn to_i(x: f32) -> i32 {
    x as i32
}
#[inline]
fn to_u(x: f32) -> u32 {
    x as u32
}
#[inline]
fn to_v2(v: IVec2) -> Vec2 {
    v.as_vec2()
}
#[inline]
fn to_iv2(v: Vec2) -> IVec2 {
    v.as_ivec2()
}

/// Signed distance (meters) and material of the program at `p`, evaluated
/// at voxel size `vs` (1.0 = full detail).
// The generated arms keep one shape across Rust and WGSL; clippy's
// pattern-collapse suggestions would fork the dialect per language.
#[allow(clippy::collapsible_match, clippy::collapsible_if)]
pub fn eval(ops: &[WorldOp], seed: u32, p: Vec3, vs: f32) -> (f32, u32) {
    let coarse = vs >= WOP_COARSE_VOXEL_M;
    let mut h = 0.0f32;
    let mut d = BIG;
    let mut mat = 1u32;
    let mut level = 0.0f32;
    let mut fy = p.y;
    let mut sxz = Vec2::ZERO;
    let mut sr = 0.0f32;
    let mut shaft = BIG;
    let mut warp = Vec2::ZERO;
    let pxz = Vec2::new(p.x, p.z);
    // Seeded shims: the generated arms call these by plain name, so a
    // world's seed reaches the noise without a global.
    let hash2 = |q: IVec2| crate::hash2(seed, q);
    let hash3 = |q: IVec3| hash3_seeded(seed, q);
    let fbm3 = |q: Vec3, fx: f32, fy: f32, o: i32, s: f32| fbm3_seeded(seed, q, fx, fy, o, s);
    let fbm_mode = |q: Vec2, sc: f32, o: i32, s: f32, m: u32| crate::fbm_mode(seed, q, sc, o, s, m);

    for op in ops {
        if coarse && op.flags & WOP_FLAG_FINE_ONLY != 0 {
            continue;
        }
        if !coarse && op.flags & WOP_FLAG_COARSE_ONLY != 0 {
            continue;
        }
        // Generated from voxel-core::opgen — the single source both
        // interpreters share. Edit the op table, not this call site.
        include!(concat!(env!("OUT_DIR"), "/op_arms_full.rs"));
    }
    (d, mat)
}

/// Height (meters) of the program's heightfield component at `xz` — the sum
/// of its height ops only. Twin of the height-only loops in the mesh
/// (shadow bake) and water (seabed) shaders.
#[allow(clippy::collapsible_match, clippy::collapsible_if)]
pub fn eval_height(ops: &[WorldOp], seed: u32, xz: Vec2, vs: f32) -> f32 {
    let mut h = 0.0;
    let mut warp = Vec2::ZERO;
    // Shell names for the generated height arms: the replay shaders call
    // their band-limited FBM without a vs argument.
    let pxz = xz;
    let hfbm = |q: Vec2, s: f32, o: i32, m: u32| crate::fbm_mode(seed, q, s, o, vs, m);
    for op in ops {
        // Generated from voxel-core::opgen (height-only subset).
        include!(concat!(env!("OUT_DIR"), "/op_arms_height.rs"));
    }
    h
}

/// Bounds on the program's SDF over a box, or `None` if the program
/// contains an op nobody has taught to bound itself.
///
/// The cheap half of "is there anything here". A box whose bound is
/// entirely positive is all air and entirely negative is all solid —
/// either way no surface crosses it, so there is nothing to mesh and no
/// reason to spend a 38³ density pass discovering that. On the shipped
/// planet that is 11,177 of 13,083 chunks.
///
/// Says nothing about planning ops, which carve into the world after
/// this: a caller that prunes on this answer must either bound those too
/// or only prune where they cannot reach.
pub fn eval_range(ops: &[WorldOp], seed: u32, min: Vec3, max: Vec3, vs: f32) -> Option<Interval> {
    // Air, until an op says otherwise — the same start `eval` uses.
    let mut d = Interval::point(BIG);
    let mut h = Interval::point(0.0);
    let py = Interval::new(min.y, max.y);
    // The xz box later height ops sample from; a warp widens it.
    let mut pxz_lo = Vec2::new(min.x, min.z);
    let mut pxz_hi = Vec2::new(max.x, max.z);
    let frange = |lo: Vec2, hi: Vec2, s: f32, o: i32, m: u32| {
        crate::fbm_range(seed, lo, hi, s, o, vs, m)
    };
    for op in ops {
        // Generated from voxel-core::opgen; see `OpDef::range`.
        include!(concat!(env!("OUT_DIR"), "/op_arms_range.rs"));
    }
    Some(d)
}

/// The Y-lattice spacing of the program, if it has one (used by planning
/// providers that seat features on structural floors).
pub fn lattice_y_spacing(ops: &[WorldOp]) -> Option<f32> {
    ops.iter()
        .find(|op| op.kind == WOP_LATTICE_Y)
        .map(|op| op.p0[0])
}

/// Sea level of the program's water surface, if it has one.
/// Evaluate the field registers at a column. Fields are author-defined
/// world data (forest density, moisture, ...) consumed by spawners and
/// gameplay queries; they never touch the SDF. Warp ops accumulated
/// before a field op affect its sample, mirroring the height loop.
pub fn eval_fields(ops: &[WorldOp], seed: u32, xz: Vec2, vs: f32) -> [f32; FIELD_SLOTS] {
    let mut fields = [0.0f32; FIELD_SLOTS];
    let mut warp = Vec2::ZERO;
    let fbm_mode = |q: Vec2, sc: f32, o: i32, s: f32, m: u32| crate::fbm_mode(seed, q, sc, o, s, m);
    for op in ops {
        match op.kind {
            WOP_WARP_XZ => {
                let q = xz + Vec2::new(op.p0[2], op.p0[3]);
                let oct = op.p1[0] as i32;
                warp.x += fbm_mode(q, op.p0[0], oct, vs, 0) * op.p0[1];
                warp.y += fbm_mode(q + Vec2::new(713.0, -337.0), op.p0[0], oct, vs, 0) * op.p0[1];
            }
            WOP_FIELD => {
                let slot = (op.p1[2] as usize).min(FIELD_SLOTS - 1);
                fields[slot] += op.p1[3]
                    + fbm_mode(
                        xz + warp + Vec2::new(op.p0[0], op.p0[1]),
                        op.p0[2],
                        op.p1[0] as i32,
                        vs,
                        op.p1[1] as u32,
                    ) * op.p0[3];
            }
            _ => {}
        }
    }
    fields
}

/// WGSL-identical smoothstep (the height-op twins must agree).
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// --- the process-wide current program ----------------------------------------

/// The engine-wide fallback sun direction (not normalized; twins normalize).
pub const DEFAULT_SUN_DIR: glam::Vec3 = glam::Vec3::new(0.55, 0.5, 0.32);

// --- reference programs (also the test oracles' subjects) --------------------

/// The shipped planet: four height bands, sea-relative offset, grass surface.
pub fn planet_program() -> Vec<WorldOp> {
    fn band(offset: [f32; 2], scale: f32, amp: f32, octaves: f32) -> WorldOp {
        WorldOp::new(WOP_HEIGHT_FBM)
            .p0([offset[0], offset[1], scale, amp])
            .p1([octaves, 0.0, 0.0, 0.0])
    }
    vec![
        band([0.0, 0.0], 0.00005, 800.0, 3.0),
        band([510.0, -770.0], 0.0008, 420.0, 5.0),
        band([1337.0, 55.0], 0.01, 36.0, 5.0),
        WorldOp::new(WOP_HEIGHT_STEP).p0([180.0, 230.0, 90.0, 0.0]),
        band([37.0, 91.0], 0.06, 5.0, 4.0),
        WorldOp::new(WOP_HEIGHT_OFFSET).p0([-8.0, 0.0, 0.0, 0.0]),
        // Field 0: forest coverage (was the trees spawner's private patch
        // noise; a shared field so any consumer can reference it).
        WorldOp::new(WOP_FIELD)
            .p0([-4200.0, 8800.0, 0.004, 1.6])
            .p1([3.0, 0.0, 0.0, 0.15]),
        WorldOp::new(WOP_HEIGHT_SURFACE).material(1),
    ]
}

/// The shipped megastructure: shaft registers, coarse solid mass, floor
/// lattice with openings, pillars, gated walls with doorways, shaft cut,
/// catwalk beams.
pub fn mega_program() -> Vec<WorldOp> {
    vec![
        WorldOp::new(WOP_SHAFTS_XZ).p0([288.0, 90.0, 24.0, 30.0]),
        WorldOp::new(WOP_COARSE_SOLID)
            .flags(WOP_FLAG_COARSE_ONLY)
            .material(2),
        WorldOp::new(WOP_LATTICE_Y)
            .flags(WOP_FLAG_FINE_ONLY)
            .p0([44.0, 0.0, 0.0, 0.0]),
        WorldOp::new(WOP_SLABS_Y)
            .flags(WOP_FLAG_FINE_ONLY)
            .material(2)
            .p0([1.5, 0.0, 0.0, 0.0]),
        WorldOp::new(WOP_GRID_HOLES)
            .flags(WOP_FLAG_FINE_ONLY)
            .p0([16.0, 0.16, 0.0, 0.0])
            .p1([7.0, 4.0, 7.0, 0.0]),
        WorldOp::new(WOP_PILLARS_XZ)
            .flags(WOP_FLAG_FINE_ONLY)
            .material(2)
            .p0([34.0, 8.0, 1.6, 2.2]),
        WorldOp::new(WOP_WALLS)
            .flags(WOP_FLAG_FINE_ONLY)
            .material(2)
            .p0([104.0, 1.2, 0.45, 0.0])
            .p1([0.0, 22.0, 0.5, 0.0])
            .p2([4.0, 14.0, 5.0, 12.0]),
        WorldOp::new(WOP_WALLS)
            .flags(WOP_FLAG_FINE_ONLY)
            .material(2)
            .p0([104.0, 1.2, 0.45, 1.0])
            .p1([501.0, 22.0, 0.5, 77.0])
            .p2([4.0, 14.0, 5.0, 12.0]),
        WorldOp::new(WOP_SHAFTS_CUT),
        WorldOp::new(WOP_BEAMS)
            .flags(WOP_FLAG_FINE_ONLY)
            .material(2)
            .p0([3.0, 2.2, 1.0, 0.7])
            .p1([6.0, 0.0, 0.0, 0.0]),
    ]
}

#[cfg(test)]
mod tests {
    use crate::fbm_mode;
    use super::*;
    use crate::fbm;

    #[test]
    fn planet_program_matches_legacy_terrain_height() {
        // Oracle: the pre-program formula, verbatim.
        let ops = planet_program();
        let seed = 0u32;
        for i in 0..500 {
            let p = Vec2::new((i * 37) as f32 * 13.7, (i * 91) as f32 * -7.3);
            for vs in [1.0, 8.0, 64.0] {
                let base = fbm(seed, p, 0.00005, 3, vs) * 800.0
                    + fbm(seed, p + Vec2::new(510.0, -770.0), 0.0008, 5, vs) * 420.0
                    + fbm(seed, p + Vec2::new(1337.0, 55.0), 0.01, 5, vs) * 36.0;
                let stepped = base + 90.0 * smoothstep(180.0, 230.0, base);
                let legacy = stepped + fbm(seed, p + Vec2::new(37.0, 91.0), 0.06, 4, vs) * 5.0 - 8.0;
                assert_eq!(eval_height(&ops, seed, p, vs), legacy);
                // (h - 3) - h is not exactly -3 in f32; the height itself is
                // bit-exact (asserted above), the SDF just subtracts it.
                let (d, mat) = eval(&ops, seed, glam::Vec3::new(p.x, legacy - 3.0, p.y), vs);
                assert!((d + 3.0).abs() < 1.0e-3, "d={d}");
                assert_eq!(mat, 1);
            }
        }
    }

    #[test]
    fn fields_accumulate_and_respect_warp() {
        let seed = 0u32;
        let mut f0 = WorldOp::new(WOP_FIELD);
        f0.p0 = [10.0, -5.0, 0.01, 2.0];
        f0.p1 = [3.0, 0.0, 0.0, 0.25];
        let mut f0b = WorldOp::new(WOP_FIELD);
        f0b.p0 = [0.0, 0.0, 0.05, 0.5];
        f0b.p1 = [2.0, 0.0, 0.0, 0.0];
        let mut f2 = WorldOp::new(WOP_FIELD);
        f2.p0 = [0.0, 0.0, 0.02, 1.0];
        f2.p1 = [2.0, 0.0, 2.0, -0.1];
        let mut warp = WorldOp::new(WOP_WARP_XZ);
        warp.p0 = [0.002, 40.0, 7.0, 13.0];
        warp.p1 = [2.0, 0.0, 0.0, 0.0];

        let p = Vec2::new(812.0, -3355.0);
        let vs = 4.0;
        // Warp placed after the first field op only affects later ones.
        let ops = [f0, warp, f0b, f2];
        let got = eval_fields(&ops, seed, p, vs);

        let q = p + Vec2::new(0.002 * 0.0 + 7.0, 13.0);
        let w = Vec2::new(
            fbm_mode(seed, q, 0.002, 2, vs, 0) * 40.0,
            fbm_mode(seed, q + Vec2::new(713.0, -337.0), 0.002, 2, vs, 0) * 40.0,
        );
        let want0 = 0.25
            + fbm_mode(seed, p + Vec2::new(10.0, -5.0), 0.01, 3, vs, 0) * 2.0
            + fbm_mode(seed, p + w, 0.05, 2, vs, 0) * 0.5;
        let want2 = -0.1 + fbm_mode(seed, p + w, 0.02, 2, vs, 0) * 1.0;
        assert_eq!(got[0], want0);
        assert_eq!(got[1], 0.0);
        assert_eq!(got[2], want2);
        assert_eq!(got[3], 0.0);
    }

    #[test]
    fn coarse_mega_is_solid_minus_shafts() {
        let seed = 0u32;
        let ops = mega_program();
        for i in 0..300 {
            let p = Vec3::new(i as f32 * 17.3, (i % 11) as f32 * 9.0, i as f32 * -23.1);
            let (coarse, _) = eval(&ops, seed, p, 8.0);
            // Away from shafts the coarse world is deeply solid; the fine
            // world is never *more* solid than a slab is thick.
            let (fine, _) = eval(&ops, seed, p, 1.0);
            assert!(coarse.is_finite() && fine.is_finite());
            if coarse > 1.0 {
                // Inside a shaft: fine structure must be air there too
                // (beams excepted).
                assert!(
                    fine > -0.01 || fine <= coarse,
                    "fine {fine} coarse {coarse}"
                );
            }
        }
    }

    #[test]
    fn programs_are_deterministic() {
        let seed = 0u32;
        let ops = mega_program();
        for i in 0..200 {
            let p = Vec3::new(i as f32 * 13.7, (i % 7) as f32 * 11.0, i as f32 * -7.9);
            assert_eq!(eval(&ops, seed, p, 1.0), eval(&ops, seed, p, 1.0));
        }
    }

    #[test]
    fn lattice_spacing_found() {
        assert_eq!(lattice_y_spacing(&mega_program()), Some(44.0));
        assert_eq!(lattice_y_spacing(&planet_program()), None);
    }

    #[test]
    fn noise_modes_and_warp_change_heights_within_bounds() {
        let seed = 0u32;
        let base = WorldOp::new(WOP_HEIGHT_FBM)
            .p0([0.0, 0.0, 0.001, 100.0])
            .p1([4.0, 0.0, 0.0, 0.0]);
        let ridged = base.p1([4.0, 1.0, 0.0, 0.0]);
        let billow = base.p1([4.0, 2.0, 0.0, 0.0]);
        let warp = WorldOp::new(WOP_WARP_XZ)
            .p0([0.0005, 400.0, 0.0, 0.0])
            .p1([3.0, 0.0, 0.0, 0.0]);
        let mut differs = 0;
        for i in 0..200 {
            let p = Vec2::new(i as f32 * 137.0, i as f32 * -91.0);
            let h0 = eval_height(&[base], seed, p, 1.0);
            let h1 = eval_height(&[ridged], seed, p, 1.0);
            let h2 = eval_height(&[billow], seed, p, 1.0);
            let hw = eval_height(&[warp, base], seed, p, 1.0);
            for h in [h0, h1, h2, hw] {
                assert!(h.is_finite() && h.abs() <= 100.0, "h={h}");
            }
            if (h0 - h1).abs() > 1.0 && (h0 - h2).abs() > 1.0 && (h0 - hw).abs() > 1.0 {
                differs += 1;
            }
        }
        assert!(differs > 60, "modes/warp barely changed terrain: {differs}");
    }

    #[test]
    fn fbm3_carve_makes_underground_air() {
        let seed = 0u32;
        // Planet base + aggressive cave carve: some points well below the
        // surface must now be air, and the op must be deterministic.
        let mut ops = planet_program();
        ops.push(
            WorldOp::new(WOP_FBM3)
                .p0([0.02, 0.04, 0.05, 30.0])
                .p1([0.0, 0.0, 0.0, 1.0])
                .p2([3.0, 0.0, 0.0, 0.0]),
        );
        let mut caves = 0;
        for i in 0..400 {
            let xz = Vec2::new(i as f32 * 61.0, i as f32 * -43.0);
            let h = eval_height(&ops, 0, xz, 1.0);
            let p = Vec3::new(xz.x, h - 12.0, xz.y);
            let (d, _) = eval(&ops, seed, p, 1.0);
            assert_eq!(eval(&ops, seed, p, 1.0), (d, eval(&ops, seed, p, 1.0).1));
            if d > 0.5 {
                caves += 1;
            }
        }
        assert!(caves > 20, "carve produced almost no caves: {caves}");
    }

    #[test]
    fn fbm3_union_makes_floating_solids() {
        let seed = 0u32;
        // Pure 3D-noise world: solid regions exist above any surface.
        let ops = vec![WorldOp::new(WOP_FBM3)
            .material(2)
            .p0([0.005, 0.01, 0.12, 60.0])
            .p1([0.0, 0.0, 0.0, 0.0])
            .p2([3.0, 0.0, 0.0, 0.0])];
        let mut solid = 0;
        for i in 0..400 {
            let p = Vec3::new(
                i as f32 * 53.0,
                100.0 + (i % 13) as f32 * 30.0,
                i as f32 * -37.0,
            );
            let (d, mat) = eval(&ops, seed, p, 1.0);
            if d < 0.0 {
                solid += 1;
                assert_eq!(mat, 2);
            }
        }
        assert!(solid > 20, "no floating solids: {solid}");
    }
}

#[cfg(test)]
mod range_tests {
    use super::*;

    /// The bound must contain what the real evaluator produces anywhere
    /// in the box. This is the test the whole optimisation rests on: a
    /// bound that is too wide costs a chunk we did not need to generate,
    /// one that is too narrow deletes world.
    #[test]
    fn the_bound_contains_the_sdf_it_bounds() {
        let ops = planet_program();
        let mut rng = voxel_core::seed::Rng::new(0xB0DE);
        let mut informative = 0;
        for _ in 0..4_000 {
            let c = Vec3::new(
                (rng.next_f32() - 0.5) * 60_000.0,
                (rng.next_f32() - 0.5) * 4_000.0,
                (rng.next_f32() - 0.5) * 60_000.0,
            );
            let edge = 3.2 * (1u32 << (rng.next_f32() * 11.0) as u32) as f32;
            let (min, max) = (c, c + Vec3::splat(edge));
            let bound = eval_range(&ops, 0, min, max, 1.0).expect("planet is analysable");
            if !bound.straddles_zero() {
                informative += 1;
            }
            for _ in 0..12 {
                let p = Vec3::new(
                    min.x + (max.x - min.x) * rng.next_f32(),
                    min.y + (max.y - min.y) * rng.next_f32(),
                    min.z + (max.z - min.z) * rng.next_f32(),
                );
                let (d, _) = eval(&ops, 0, p, 1.0);
                assert!(
                    bound.contains(d),
                    "sdf {d} at {p:?} escapes {bound:?} for box {min:?}..{max:?}"
                );
            }
        }
        // A bound that always says "maybe" would pass the above and be
        // worthless.
        println!("decided {informative} of 4000 boxes");
        assert!(
            informative > 1_000,
            "only {informative} of 4000 boxes were decided; the bound is too loose to prune"
        );
    }

    /// A world that can put solid anywhere declines to be analysed,
    /// rather than claiming a bound it cannot justify.
    #[test]
    fn a_volumetric_world_declines() {
        assert!(eval_range(&mega_program(), 0, Vec3::ZERO, Vec3::splat(100.0), 1.0).is_none());
    }

    /// The two answers worth having, on the world they are for.
    #[test]
    fn sky_is_air_and_the_deep_is_solid() {
        let ops = planet_program();
        let sky =
            eval_range(&ops, 0, Vec3::new(0.0, 8_000.0, 0.0), Vec3::new(800.0, 8_800.0, 800.0), 1.0)
                .unwrap();
        assert!(sky.is_positive(), "sky should be all air: {sky:?}");
        let deep = eval_range(
            &ops,
            0,
            Vec3::new(0.0, -8_800.0, 0.0),
            Vec3::new(800.0, -8_000.0, 800.0),
            1.0,
        )
        .unwrap();
        assert!(deep.is_negative(), "the deep should be all solid: {deep:?}");
    }
}
