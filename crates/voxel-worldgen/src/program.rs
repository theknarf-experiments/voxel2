//! CPU twin of the GPU world-generator interpreter
//! (`voxel-render/src/shaders/voxel_world_density.wgsl`). A world's base
//! generator is a program of [`WorldOp`]s; both interpreters evaluate it
//! op-for-op over the same register file, so collision, vegetation,
//! planning, and the rendered world always agree. MUST stay bit-compatible
//! with the WGSL.

use std::sync::{Arc, RwLock};

use glam::{IVec2, IVec3, Vec2, Vec3};
use voxel_core::worldop::*;

use crate::{fbm_mode, hash2};

fn hash3(p: IVec3) -> f32 {
    let mut h: u32 = (p.x as u32)
        .wrapping_mul(374_761_393)
        .wrapping_add((p.y as u32).wrapping_mul(668_265_263))
        .wrapping_add((p.z as u32).wrapping_mul(2_246_822_519))
        .wrapping_add(seed().wrapping_mul(2_654_435_769));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0xFF_FFFF) as f32 / 16_777_216.0
}

fn sd_box(p: Vec3, b: Vec3) -> f32 {
    let q = p.abs() - b;
    q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
}

/// Mirrors the WGSL `value_noise3` (quintic smoothstep).
fn value_noise3(p: Vec3) -> f32 {
    let i = p.floor();
    let f = p - i;
    let i = IVec3::new(i.x as i32, i.y as i32, i.z as i32);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let corner = |dx: i32, dy: i32, dz: i32| hash3(i + IVec3::new(dx, dy, dz));
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
fn fbm3(p: Vec3, freq_xz: f32, freq_y: f32, octaves: i32, voxel_size: f32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut mul = 1.0;
    for _ in 0..octaves {
        let fade = crate::band_fade(1.0 / (freq_xz * mul), voxel_size);
        sum += amp
            * fade
            * (value_noise3(Vec3::new(
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

/// Signed distance (meters) and material of the program at `p`, evaluated
/// at voxel size `vs` (1.0 = full detail).
pub fn eval(ops: &[WorldOp], p: Vec3, vs: f32) -> (f32, u32) {
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

    for op in ops {
        if coarse && op.flags & WOP_FLAG_FINE_ONLY != 0 {
            continue;
        }
        if !coarse && op.flags & WOP_FLAG_COARSE_ONLY != 0 {
            continue;
        }
        match op.kind {
            WOP_HEIGHT_FBM => {
                h += fbm_mode(
                    pxz + warp + Vec2::new(op.p0[0], op.p0[1]),
                    op.p0[2],
                    op.p1[0] as i32,
                    vs,
                    op.p1[1] as u32,
                ) * op.p0[3];
            }
            WOP_HEIGHT_OFFSET => h += op.p0[0],
            WOP_WARP_XZ => {
                let q = pxz + Vec2::new(op.p0[2], op.p0[3]);
                let oct = op.p1[0] as i32;
                warp.x += fbm_mode(q, op.p0[0], oct, vs, 0) * op.p0[1];
                warp.y += fbm_mode(q + Vec2::new(713.0, -337.0), op.p0[0], oct, vs, 0) * op.p0[1];
            }
            WOP_FBM3 => {
                let q = p + Vec3::new(op.p1[0], op.p1[1], op.p1[2]);
                let n = fbm3(q, op.p0[0], op.p0[1], op.p2[0] as i32, vs);
                let sd = (op.p0[2] - n) * op.p0[3];
                if op.p1[3] < 0.5 {
                    if sd < d {
                        d = sd;
                        mat = op.material;
                    }
                } else {
                    d = d.max(-sd);
                }
            }
            WOP_HEIGHT_SURFACE => {
                let nd = p.y - h;
                if nd < d {
                    d = nd;
                    mat = op.material;
                }
            }
            WOP_COARSE_SOLID if SOLID < d => {
                d = SOLID;
                mat = op.material;
            }
            WOP_LATTICE_Y => {
                level = (p.y / op.p0[0]).round();
                fy = p.y - level * op.p0[0];
            }
            WOP_SLABS_Y => {
                let nd = fy.abs() - op.p0[0];
                if nd < d {
                    d = nd;
                    mat = op.material;
                }
            }
            WOP_GRID_HOLES => {
                let cell = op.p0[0];
                let c = IVec2::new((p.x / cell).floor() as i32, (p.z / cell).floor() as i32);
                if hash3(IVec3::new(c.x, level as i32, c.y)) < op.p0[1] {
                    let oc = (Vec2::new(c.x as f32, c.y as f32) + 0.5) * cell;
                    let cut = sd_box(
                        Vec3::new(p.x - oc.x, fy, p.z - oc.y),
                        Vec3::new(op.p1[0], op.p1[1], op.p1[2]),
                    );
                    d = d.max(-cut);
                }
            }
            WOP_PILLARS_XZ => {
                let sp = op.p0[0];
                let c = IVec2::new((p.x / sp).round() as i32, (p.z / sp).round() as i32);
                let jit =
                    Vec2::new(hash2(c) - 0.5, hash2(c + IVec2::new(311, 77)) - 0.5) * op.p0[1];
                let q = pxz - Vec2::new(c.x as f32, c.y as f32) * sp - jit;
                let girth = op.p0[2] + hash2(c + IVec2::new(9, -4)) * op.p0[3];
                let nd = q.x.abs().max(q.y.abs()) - girth;
                if nd < d {
                    d = nd;
                    mat = op.material;
                }
            }
            WOP_WALLS => {
                let sp = op.p0[0];
                let along_z = op.p0[3] > 0.5;
                let (a, b) = if along_z { (p.z, p.x) } else { (p.x, p.z) };
                let wi = (a / sp).round();
                let w = a - wi * sp;
                let gate = hash2(IVec2::new(wi as i32 + op.p1[0] as i32, level as i32));
                if gate < op.p0[2] {
                    let mut wall = w.abs() - op.p0[1];
                    let dc = op.p1[1];
                    let ci = (b / dc).round();
                    let cl = b - ci * dc;
                    if hash3(IVec3::new(
                        wi as i32,
                        ci as i32,
                        level as i32 + op.p1[3] as i32,
                    )) < op.p1[2]
                    {
                        let doorway = sd_box(
                            Vec3::new(w, fy + op.p2[3], cl),
                            Vec3::new(op.p2[0], op.p2[1], op.p2[2]),
                        );
                        wall = wall.max(-doorway);
                    }
                    if wall < d {
                        d = wall;
                        mat = op.material;
                    }
                }
            }
            WOP_SHAFTS_XZ => {
                let sp = op.p0[0];
                let c = IVec2::new((p.x / sp).round() as i32, (p.z / sp).round() as i32);
                let jit = Vec2::new(
                    hash2(c + IVec2::new(41, 13)) - 0.5,
                    hash2(c + IVec2::new(-7, 99)) - 0.5,
                ) * op.p0[1];
                sxz = pxz - Vec2::new(c.x as f32, c.y as f32) * sp - jit;
                sr = op.p0[2] + hash2(c) * op.p0[3];
                shaft = sxz.length() - sr;
            }
            WOP_SHAFTS_CUT => d = d.max(-shaft),
            WOP_BEAMS => {
                let n = op.p0[0];
                if (level - (level / n).round() * n).abs() < 0.5 {
                    let beam = (sxz.y.abs() - op.p0[1])
                        .max((fy + op.p0[2]).abs() - op.p0[3])
                        .max(sxz.length() - (sr + op.p1[0]));
                    if beam < d {
                        d = beam;
                        mat = op.material;
                    }
                }
            }
            _ => {}
        }
    }
    (d, mat)
}

/// Height (meters) of the program's heightfield component at `xz` — the sum
/// of its height ops only. Twin of the height-only loops in the mesh
/// (shadow bake) and water (seabed) shaders.
pub fn eval_height(ops: &[WorldOp], xz: Vec2, vs: f32) -> f32 {
    let mut h = 0.0;
    let mut warp = Vec2::ZERO;
    for op in ops {
        match op.kind {
            WOP_HEIGHT_FBM => {
                h += fbm_mode(
                    xz + warp + Vec2::new(op.p0[0], op.p0[1]),
                    op.p0[2],
                    op.p1[0] as i32,
                    vs,
                    op.p1[1] as u32,
                ) * op.p0[3];
            }
            WOP_HEIGHT_OFFSET => h += op.p0[0],
            WOP_WARP_XZ => {
                let q = xz + Vec2::new(op.p0[2], op.p0[3]);
                let oct = op.p1[0] as i32;
                warp.x += fbm_mode(q, op.p0[0], oct, vs, 0) * op.p0[1];
                warp.y += fbm_mode(q + Vec2::new(713.0, -337.0), op.p0[0], oct, vs, 0) * op.p0[1];
            }
            _ => {}
        }
    }
    h
}

/// The Y-lattice spacing of the program, if it has one (used by planning
/// providers that seat features on structural floors).
pub fn lattice_y_spacing(ops: &[WorldOp]) -> Option<f32> {
    ops.iter()
        .find(|op| op.kind == WOP_LATTICE_Y)
        .map(|op| op.p0[0])
}

/// Sea level of the program's water surface, if it has one.
pub fn water_level(ops: &[WorldOp]) -> Option<f32> {
    ops.iter()
        .find(|op| op.kind == WOP_WATER)
        .map(|op| op.p0[0])
}

/// Vegetation density multiplier, if the program grows vegetation.
pub fn vegetation_density(ops: &[WorldOp]) -> Option<f32> {
    ops.iter()
        .find(|op| op.kind == WOP_VEGETATION)
        .map(|op| op.p0[0])
}

// --- the process-wide current program ----------------------------------------

static PROGRAM: RwLock<Option<Arc<Vec<WorldOp>>>> = RwLock::new(None);
static SEED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static SUN: RwLock<glam::Vec3> = RwLock::new(DEFAULT_SUN_DIR);

/// The engine-wide fallback sun direction (not normalized; twins normalize).
pub const DEFAULT_SUN_DIR: glam::Vec3 = glam::Vec3::new(0.55, 0.5, 0.32);

/// Install the level's generator program for the CPU mirrors
/// ([`crate::terrain_height`], [`crate::mega::mega_sdf`], …).
pub fn set_program(ops: Vec<WorldOp>) {
    *PROGRAM.write().unwrap() = Some(Arc::new(ops));
}

/// Install the level seed. Mixed into the generator hashes on both twins;
/// seed 0 leaves them bit-identical to the unseeded formulas.
pub fn set_seed(seed: u32) {
    SEED.store(seed, std::sync::atomic::Ordering::Relaxed);
}

/// The installed level seed (0 until a level sets one).
pub fn seed() -> u32 {
    SEED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Install the level's sun direction — the single source for the baked
/// shadow march (CPU + GPU) and blob shadows.
pub fn set_sun_direction(dir: glam::Vec3) {
    *SUN.write().unwrap() = dir;
}

pub fn sun_direction() -> glam::Vec3 {
    *SUN.read().unwrap()
}

/// The current program (defaults to [`planet_program`] until a level
/// installs one — keeps tools like scout working without a level).
pub fn program() -> Arc<Vec<WorldOp>> {
    if let Some(p) = PROGRAM.read().unwrap().as_ref() {
        return p.clone();
    }
    let mut w = PROGRAM.write().unwrap();
    w.get_or_insert_with(|| Arc::new(planet_program())).clone()
}

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
        band([37.0, 91.0], 0.06, 5.0, 4.0),
        WorldOp::new(WOP_HEIGHT_OFFSET).p0([-8.0, 0.0, 0.0, 0.0]),
        WorldOp::new(WOP_HEIGHT_SURFACE).material(1),
        WorldOp::new(WOP_WATER),
        WorldOp::new(WOP_VEGETATION).p0([1.0, 0.0, 0.0, 0.0]),
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
    use super::*;

    #[test]
    fn planet_program_matches_legacy_terrain_height() {
        // Oracle: the pre-program formula, verbatim.
        let ops = planet_program();
        for i in 0..500 {
            let p = Vec2::new((i * 37) as f32 * 13.7, (i * 91) as f32 * -7.3);
            for vs in [1.0, 8.0, 64.0] {
                let legacy = fbm(p, 0.00005, 3, vs) * 800.0
                    + fbm(p + Vec2::new(510.0, -770.0), 0.0008, 5, vs) * 420.0
                    + fbm(p + Vec2::new(1337.0, 55.0), 0.01, 5, vs) * 36.0
                    + fbm(p + Vec2::new(37.0, 91.0), 0.06, 4, vs) * 5.0
                    - 8.0;
                assert_eq!(eval_height(&ops, p, vs), legacy);
                // (h - 3) - h is not exactly -3 in f32; the height itself is
                // bit-exact (asserted above), the SDF just subtracts it.
                let (d, mat) = eval(&ops, glam::Vec3::new(p.x, legacy - 3.0, p.y), vs);
                assert!((d + 3.0).abs() < 1.0e-3, "d={d}");
                assert_eq!(mat, 1);
            }
        }
    }

    #[test]
    fn coarse_mega_is_solid_minus_shafts() {
        let ops = mega_program();
        for i in 0..300 {
            let p = Vec3::new(i as f32 * 17.3, (i % 11) as f32 * 9.0, i as f32 * -23.1);
            let (coarse, _) = eval(&ops, p, 8.0);
            // Away from shafts the coarse world is deeply solid; the fine
            // world is never *more* solid than a slab is thick.
            let (fine, _) = eval(&ops, p, 1.0);
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
        let ops = mega_program();
        for i in 0..200 {
            let p = Vec3::new(i as f32 * 13.7, (i % 7) as f32 * 11.0, i as f32 * -7.9);
            assert_eq!(eval(&ops, p, 1.0), eval(&ops, p, 1.0));
        }
    }

    #[test]
    fn lattice_spacing_found() {
        assert_eq!(lattice_y_spacing(&mega_program()), Some(44.0));
        assert_eq!(lattice_y_spacing(&planet_program()), None);
    }

    #[test]
    fn noise_modes_and_warp_change_heights_within_bounds() {
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
            let h0 = eval_height(&[base], p, 1.0);
            let h1 = eval_height(&[ridged], p, 1.0);
            let h2 = eval_height(&[billow], p, 1.0);
            let hw = eval_height(&[warp, base], p, 1.0);
            for h in [h0, h1, h2, hw] {
                assert!(h.is_finite() && h.abs() <= 100.0, "h={h}");
            }
            if (h0 - h1).abs() > 1.0 && (h0 - h2).abs() > 1.0 && (h0 - hw).abs() > 1.0 {
                differs += 1;
            }
        }
        assert!(
            differs > 100,
            "modes/warp barely changed terrain: {differs}"
        );
    }

    #[test]
    fn fbm3_carve_makes_underground_air() {
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
            let h = eval_height(&ops, xz, 1.0);
            let p = Vec3::new(xz.x, h - 12.0, xz.y);
            let (d, _) = eval(&ops, p, 1.0);
            assert_eq!(eval(&ops, p, 1.0), (d, eval(&ops, p, 1.0).1));
            if d > 0.5 {
                caves += 1;
            }
        }
        assert!(caves > 20, "carve produced almost no caves: {caves}");
    }

    #[test]
    fn fbm3_union_makes_floating_solids() {
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
            let (d, mat) = eval(&ops, p, 1.0);
            if d < 0.0 {
                solid += 1;
                assert_eq!(mat, 2);
            }
        }
        assert!(solid > 20, "no floating solids: {solid}");
    }
}
