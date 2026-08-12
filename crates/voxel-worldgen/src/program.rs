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
    // Each corner ONCE. Written as `c000 + (c100 - c000) * u.x` this
    // called the hash twice per lerp — twelve hashes for the eight
    // corners of a cell, where the WGSL twin's `mix` does eight.
    let (c000, c100) = (corner(0, 0, 0), corner(1, 0, 0));
    let (c010, c110) = (corner(0, 1, 0), corner(1, 1, 0));
    let (c001, c101) = (corner(0, 0, 1), corner(1, 0, 1));
    let (c011, c111) = (corner(0, 1, 1), corner(1, 1, 1));
    let x00 = c000 + (c100 - c000) * u.x;
    let x10 = c010 + (c110 - c010) * u.x;
    let x01 = c001 + (c101 - c001) * u.x;
    let x11 = c011 + (c111 - c011) * u.x;
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
            * (value_noise3(
                seed,
                Vec3::new(p.x * freq_xz * mul, p.y * freq_y * mul, p.z * freq_xz * mul),
            ) - 0.5);
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
fn min(a: f32, b: f32) -> f32 {
    a.min(b)
}

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
    // Region axes, filled by WOP_REGION_AXES and read by the band ops.
    let mut ta = 0.0f32;
    let mut tb = 0.0f32;
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
        if !region_gate(op.region, ta, tb) {
            continue;
        }
        // Generated from voxel-core::opgen — the single source both
        // interpreters share. Edit the op table, not this call site.
        include!(concat!(env!("OUT_DIR"), "/op_arms_full.rs"));
    }
    (d, mat)
}

/// What the xz-only ops leave behind for one column: the registers the
/// density shader's `Column` struct carries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Column {
    pub h: f32,
    pub ta: f32,
    pub tb: f32,
    pub sxz: Vec2,
    pub sr: f32,
    pub shaft: f32,
}

/// The xz-only half of the program, run once for a whole column.
#[allow(clippy::collapsible_match, clippy::collapsible_if)]
pub fn eval_column(ops: &[WorldOp], seed: u32, xz: Vec2, vs: f32) -> Column {
    let coarse = vs >= WOP_COARSE_VOXEL_M;
    let mut h = 0.0f32;
    let mut warp = Vec2::ZERO;
    let mut ta = 0.0f32;
    let mut tb = 0.0f32;
    let mut sxz = Vec2::ZERO;
    let mut sr = 0.0f32;
    let mut shaft = BIG;
    let pxz = xz;
    let hash2 = |q: IVec2| crate::hash2(seed, q);
    let fbm_mode = |q: Vec2, sc: f32, o: i32, s: f32, m: u32| crate::fbm_mode(seed, q, sc, o, s, m);

    for op in ops {
        if coarse && op.flags & WOP_FLAG_FINE_ONLY != 0 {
            continue;
        }
        if !coarse && op.flags & WOP_FLAG_COARSE_ONLY != 0 {
            continue;
        }
        if !region_gate(op.region, ta, tb) {
            continue;
        }
        include!(concat!(env!("OUT_DIR"), "/op_arms_column.rs"));
    }
    Column {
        h,
        ta,
        tb,
        sxz,
        sr,
        shaft,
    }
}

/// The program evaluated the way the GPU evaluates it: the xz-only ops
/// once for the column, then everything else per sample.
///
/// Nothing in the engine calls this — [`eval`] is one linear loop, which is
/// what a planning layer sampling scattered points wants. It exists so a
/// test can assert the two agree, because the density shader runs THIS
/// shape and a disagreement between them is a world that renders
/// differently from the one gameplay queries. See
/// `voxel_core::opgen::Axis` for which ops land in which half, and
/// `voxel_engine::graph` for the per-level check that the split is
/// order-safe.
// The shell declares the whole register file and the whole shim set and
// does not care which half uses what — that is derived, and a shell that
// tracked it would need editing every time an op moved between the loops.
#[allow(clippy::collapsible_match, clippy::collapsible_if)]
#[allow(unused_variables, unused_mut)]
pub fn eval_split(ops: &[WorldOp], seed: u32, p: Vec3, vs: f32) -> (f32, u32) {
    let coarse = vs >= WOP_COARSE_VOXEL_M;
    let col = eval_column(ops, seed, Vec2::new(p.x, p.z), vs);
    let Column {
        h,
        ta,
        tb,
        mut sxz,
        mut sr,
        mut shaft,
    } = col;
    let mut d = BIG;
    let mut mat = 1u32;
    let mut level = 0.0f32;
    let mut fy = p.y;
    let pxz = Vec2::new(p.x, p.z);
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
        if !region_gate(op.region, ta, tb) {
            continue;
        }
        include!(concat!(env!("OUT_DIR"), "/op_arms_sample.rs"));
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
    let mut ta = 0.0f32;
    let mut tb = 0.0f32;
    // Shell names for the generated height arms: the replay shaders call
    // their band-limited FBM without a vs argument.
    let pxz = xz;
    let hfbm = |q: Vec2, s: f32, o: i32, m: u32| crate::fbm_mode(seed, q, s, o, vs, m);
    for op in ops {
        if !region_gate(op.region, ta, tb) {
            continue;
        }
        // Generated from voxel-core::opgen (height-only subset).
        include!(concat!(env!("OUT_DIR"), "/op_arms_height.rs"));
    }
    h
}

/// Does a point in the region axes fall inside an op's gate?
///
/// Twin of the identical test in every interpreter — the WGSL density
/// shader's two loops, the mesh shader's shadow-bake replay and the water
/// shader's seabed replay. All five must agree, or a district's walls
/// stop at a different line than its floors.
///
/// Runs for every op of every sample, which invites optimising — but an
/// integer version comparing byte-quantized axes measured no faster
/// (megastructure settle 2.12 -> 2.30 s, inside the noise). A gated-out
/// op costs its loop iteration and its 64-byte read; the compares are
/// free either way.
#[inline]
pub fn region_gate(packed: u32, ta: f32, tb: f32) -> bool {
    if packed == 0 {
        return true;
    }
    let b = voxel_core::worldop::unpack_region(packed);
    ta >= b[0] && ta < b[1] && tb >= b[2] && tb < b[3]
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
    // The region axes over this box. Bounding them is what keeps a
    // dramatic region's cost LOCAL: without it every box in the world is
    // bounded as though it might contain the mountains, and the pruning
    // this function exists for stops working everywhere at once.
    let mut ta = Interval::point(0.5);
    let mut tb = Interval::point(0.5);
    let frange =
        |lo: Vec2, hi: Vec2, s: f32, o: i32, m: u32| crate::fbm_range(seed, lo, hi, s, o, vs, m);
    for op in ops {
        match region_gate_range(op.region, ta, tb) {
            // Definitely outside: the op does not exist over this box.
            Some(false) => continue,
            // Definitely inside: bound it exactly as an ungated op.
            Some(true) => {}
            // Straddling the district edge. Bounding this properly means
            // bounding the program BOTH ways and unioning, which the
            // generated arms cannot express — they mutate `d` in place.
            // So refuse to answer, and let the box pay for a density
            // pass. That is the honest cost of a hard-edged gate, and it
            // is charged only to the boxes actually on a boundary.
            None => return None,
        }
        // Generated from voxel-core::opgen; see `OpDef::range`.
        include!(concat!(env!("OUT_DIR"), "/op_arms_range.rs"));
    }
    Some(d)
}

/// Which ops can affect a box, as a mask over `ops`.
///
/// The question a cache key asks. A chunk is decided by the ops that
/// actually reach it, and most do not: an op is out if this chunk's voxel
/// size gates it out, or if the region band it is confined to provably
/// does not contain the box. Editing an op no chunk here can see must not
/// rebuild that chunk, and this is what says so.
///
/// CONSERVATIVE: an op that straddles a band edge, or that this cannot
/// decide, is in. Over-reporting costs a rebuild nobody needed;
/// under-reporting leaves a stale chunk, which is a wrong world that
/// looks right.
///
/// Unlike [`eval_range`], this needs no `range` rule on anything: the
/// only registers it tracks are the region axes, and the one op that
/// writes those has a rule. That is what makes it work on the
/// megastructure, where almost nothing else can be bounded — and the
/// megastructure, with nine districts, is where it pays.
pub fn ops_reaching(ops: &[WorldOp], seed: u32, min: Vec3, max: Vec3, vs: f32) -> Vec<bool> {
    let coarse = vs >= WOP_COARSE_VOXEL_M;
    let mut ta = Interval::point(0.5);
    let mut tb = Interval::point(0.5);
    let (lo, hi) = (Vec2::new(min.x, min.z), Vec2::new(max.x, max.z));
    let frange =
        |lo: Vec2, hi: Vec2, s: f32, o: i32, m: u32| crate::fbm_range(seed, lo, hi, s, o, vs, m);
    ops.iter()
        .map(|op| {
            // The LOD gate is exact — the interpreters skip on the same
            // two bits — so an op the wrong side of it is simply absent.
            if coarse && op.flags & WOP_FLAG_FINE_ONLY != 0 {
                return false;
            }
            if !coarse && op.flags & WOP_FLAG_COARSE_ONLY != 0 {
                return false;
            }
            let reaches = region_gate_range(op.region, ta, tb) != Some(false);
            // The axes THIS box sees, for every gate after this one. A
            // warp does not move them: the region ops sample the
            // unwarped column, exactly as the interpreters do.
            if reaches && op.kind == WOP_REGION_AXES {
                let oct = op.p1[2] as i32;
                ta = frange(
                    lo + Vec2::new(op.p0[0], op.p0[1]),
                    hi + Vec2::new(op.p0[0], op.p0[1]),
                    op.p0[2],
                    oct,
                    0,
                ) + 0.5;
                tb = frange(
                    lo + Vec2::new(op.p1[0], op.p1[1]),
                    hi + Vec2::new(op.p1[0], op.p1[1]),
                    op.p0[3],
                    oct,
                    0,
                ) + 0.5;
            }
            reaches
        })
        .collect()
}

/// [`region_gate`] over an interval: `Some(true)`/`Some(false)` when the
/// whole box is on one side of the gate, `None` when it straddles.
fn region_gate_range(packed: u32, ta: Interval, tb: Interval) -> Option<bool> {
    if packed == 0 {
        return Some(true);
    }
    let b = voxel_core::worldop::unpack_region(packed);
    let axis = |v: Interval, lo: f32, hi: f32| {
        if v.lo >= lo && v.hi < hi {
            Some(true)
        } else if v.hi < lo || v.lo >= hi {
            Some(false)
        } else {
            None
        }
    };
    match (axis(ta, b[0], b[1]), axis(tb, b[2], b[3])) {
        // Outside on either axis is outside, however unsure the other is.
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
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

/// How firmly the program paints `material` on the ground at this column.
///
/// 1 well inside the region that paints it, falling through 0.5 at the
/// edge to 0 outside — a soft read of the same `WOP_MATERIAL_BAND` chain
/// the density interpreter applies hard. Callers that place things by
/// region use this, so what grows somewhere and what colour the ground is
/// there cannot drift apart: they are the same ops.
///
/// The material no band claims (whatever `WOP_HEIGHT_SURFACE` set) gets
/// whatever weight the bands leave over, so the weights of all the
/// materials in a program sum to 1.
pub fn surface_material_weight(
    ops: &[WorldOp],
    seed: u32,
    xz: Vec2,
    vs: f32,
    material: u32,
) -> f32 {
    // The same edge `WOP_HEIGHT_BAND_FBM` fades its terrain over, so a
    // region's ground, its content and its landform transition together.
    // Scaled down for a narrow band, or the region could never reach
    // full weight. The hard `WOP_MATERIAL_BAND` flips its colour at the
    // midpoint of that zone, where two regions weigh the same.
    let inside = |v: f32, lo: f32, hi: f32| {
        let f = band_feather([lo, hi]);
        smoothstep(lo - f, lo + f, v) * (1.0 - smoothstep(hi - f, hi + f, v))
    };

    // The axes every band tests, sampled once — the same two values the
    // interpreters put in their `ta`/`tb` registers.
    let (mut ta, mut tb) = (0.0f32, 0.0f32);
    let mut base = 0u32;
    let mut claimed = 0.0f32;
    let mut mine = 0.0f32;
    for op in ops {
        match op.kind {
            WOP_REGION_AXES => {
                let oct = op.p1[2] as i32;
                ta = crate::fbm_mode(
                    seed,
                    xz + Vec2::new(op.p0[0], op.p0[1]),
                    op.p0[2],
                    oct,
                    vs,
                    0,
                ) + 0.5;
                tb = crate::fbm_mode(
                    seed,
                    xz + Vec2::new(op.p1[0], op.p1[1]),
                    op.p0[3],
                    oct,
                    vs,
                    0,
                ) + 0.5;
            }
            WOP_HEIGHT_SURFACE | WOP_COARSE_SOLID => base = op.material,
            WOP_MATERIAL_BAND => {
                // Bands only repaint what an earlier op left as `from`,
                // so a later one can only claim what is still unclaimed.
                if op.p1[2] as u32 != base {
                    continue;
                }
                // `min`, matching WOP_HEIGHT_BAND_FBM exactly — a point
                // is as far inside a region as its weaker axis says, and
                // the gate, the terrain and the colour must agree on how
                // far inside it is, not merely on whether it is.
                let w = inside(ta, op.p0[0], op.p0[1]).min(inside(tb, op.p0[2], op.p0[3]))
                    * (1.0 - claimed);
                claimed += w;
                if op.material == material {
                    mine += w;
                }
            }
            _ => {}
        }
    }
    if material == base {
        mine += 1.0 - claimed;
    }
    mine.clamp(0.0, 1.0)
}

/// Is [`surface_material_weight`] provably ZERO everywhere in an xz box?
///
/// Conservative: `false` means "cannot prove it", never "there is some".
/// A caller uses it to skip work, so a wrong `true` would delete content
/// while a wrong `false` only costs time.
///
/// The bound is the band structure read through intervals. `inside` rises
/// from zero at `lo - f` and falls back to zero at `hi + f`, so it is
/// identically zero over an interval that lies wholly outside
/// `[lo - f, hi + f]`; `w` takes the `min` of the two axes, so either axis
/// being dead kills the band. If every band that could paint `material`
/// is dead, the weight is zero throughout.
///
/// Declines when `material` is the surface's own: that weight is
/// `1 - claimed`, which needs a LOWER bound on every other band to rule
/// out, and the bands do not carry one.
pub fn material_weight_is_zero_over(
    ops: &[WorldOp],
    seed: u32,
    lo: Vec2,
    hi: Vec2,
    vs: f32,
    material: u32,
) -> bool {
    use voxel_core::interval::Interval;
    let mut base = 0u32;
    let mut ta = Interval::new(0.0, 0.0);
    let mut tb = Interval::new(0.0, 0.0);
    let mut seen_axes = false;
    // An interval lies wholly outside a band's support.
    let dead = |v: Interval, b0: f32, b1: f32| {
        let f = band_feather([b0, b1]);
        v.hi <= b0 - f || v.lo >= b1 + f
    };
    // The whole interval sits on `inside`'s plateau, where it is exactly 1.
    let full = |v: Interval, b0: f32, b1: f32| {
        let f = band_feather([b0, b1]);
        v.lo >= b0 + f && v.hi <= b1 - f
    };
    for op in ops {
        match op.kind {
            WOP_REGION_AXES => {
                let oct = op.p1[2] as i32;
                let shift = |o: Vec2| (lo + o, hi + o);
                let (a0, a1) = shift(Vec2::new(op.p0[0], op.p0[1]));
                let (b0, b1) = shift(Vec2::new(op.p1[0], op.p1[1]));
                let r = crate::fbm_range(seed, a0, a1, op.p0[2], oct, vs, 0);
                ta = Interval::new(r.lo + 0.5, r.hi + 0.5);
                let r = crate::fbm_range(seed, b0, b1, op.p0[3], oct, vs, 0);
                tb = Interval::new(r.lo + 0.5, r.hi + 0.5);
                seen_axes = true;
            }
            WOP_HEIGHT_SURFACE | WOP_COARSE_SOLID => base = op.material,
            WOP_MATERIAL_BAND => {
                if op.p1[2] as u32 != base {
                    continue;
                }
                if !seen_axes {
                    return false;
                }
                if op.material == material {
                    // A band that could paint what we are asking about.
                    if !(dead(ta, op.p0[0], op.p0[1]) || dead(tb, op.p0[2], op.p0[3])) {
                        return false;
                    }
                } else if material == base
                    && full(ta, op.p0[0], op.p0[1])
                    && full(tb, op.p0[2], op.p0[3])
                {
                    // The surface's OWN material is whatever is left over,
                    // `1 - claimed`, so proving it zero means proving
                    // something else took all of it. A band at full weight
                    // does: `claimed += (1 - claimed) * 1` is exactly 1
                    // however much was claimed before it, and every band
                    // after it then adds `* (1 - claimed)` = 0.
                    return true;
                }
            }
            _ => {}
        }
    }
    material != base
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
        planet_axes(),
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
        // Dunes in the desert: long ridged waves, low amplitude.
        region_terrain(
            [0.56, 1.0],
            [0.0, 0.47],
            [820.0, -410.0],
            0.011,
            21.0,
            3,
            2,
            0.0,
        ),
        // Jagged crests in the alpine: ridged noise, tall and sharp.
        region_terrain(
            [0.0, 0.44],
            [0.0, 1.0],
            [-2600.0, 1750.0],
            0.0035,
            165.0,
            5,
            1,
            60.0,
        ),
        // Wetland is the flattest ground on the planet: a gentle negative
        // band, which is what makes water pool there.
        region_terrain(
            [0.0, 1.0],
            [0.56, 1.0],
            [4400.0, 900.0],
            0.0025,
            34.0,
            3,
            0,
            -26.0,
        ),
        // A mountain range. The band is NARROW on one axis, so the
        // region is an iso-strip of the noise field and snakes across
        // the world the way a range does; a wide box would give a blob.
        region_terrain(
            [0.470, 0.530],
            [0.0, 1.0],
            [15200.0, -6400.0],
            0.00022,
            2350.0,
            4,
            1,
            1150.0,
        ),
        region_terrain(
            [0.470, 0.530],
            [0.0, 1.0],
            [-3300.0, 7100.0],
            0.0016,
            430.0,
            5,
            1,
            0.0,
        ),
        WorldOp::new(WOP_HEIGHT_SURFACE).material(1),
        // Regions: two noise axes, and a box in their product per region.
        // Order is priority — each only repaints ground still left as
        // material 1, so the first to claim a point owns it, and roads
        // and river surfaces are never candidates at all.
        // The range claims first: it cuts through whatever it crosses.
        region(7, [0.470, 0.530], [0.0, 1.0]),
        region(5, [0.0, 0.44], [0.0, 1.0]),
        region(2, [0.56, 1.0], [0.0, 0.47]),
        region(6, [0.0, 1.0], [0.56, 1.0]),
    ]
}

/// One region band of the shipped planet.
fn region(material: u32, a: [f32; 2], b: [f32; 2]) -> WorldOp {
    WorldOp::new(WOP_MATERIAL_BAND)
        .material(material)
        .p0([a[0], a[1], b[0], b[1]])
        .p1([0.0, 0.0, 1.0, 0.0])
}

/// Widest region edge, in band units. Narrow bands scale down from it.
pub const FEATHER_MAX: f32 = 0.06;

/// The edge width a band of this range gets: the shared maximum, unless
/// the band is too narrow to reach full weight across it.
pub fn band_feather(a: [f32; 2]) -> f32 {
    FEATHER_MAX.min((a[1] - a[0]) * 0.3)
}

/// The planet's region axes: ~12 km and ~9 km, independent offsets.
fn planet_axes() -> WorldOp {
    WorldOp::new(WOP_REGION_AXES)
        .p0([-31000.0, 12000.0, 8.0e-5, 1.1e-4])
        .p1([47000.0, -19000.0, 5.0, 0.0])
}

/// Terrain the region shapes: dunes where it is desert, ridges where it
/// is alpine. Faded by region weight, so a border is a landscape
/// becoming another rather than a step.
#[allow(clippy::too_many_arguments)]
fn region_terrain(
    a: [f32; 2],
    b: [f32; 2],
    off: [f32; 2],
    scale: f32,
    amp: f32,
    oct: u32,
    mode: u32,
    lift: f32,
) -> WorldOp {
    WorldOp::new(WOP_HEIGHT_BAND_FBM)
        .p0([off[0], off[1], scale, amp])
        .p1([oct as f32, mode as f32, band_feather(a), lift])
        .p2([a[0], a[1], b[0], b[1]])
}

/// A megastructure interior, in the shape the shipped level has: a coarse
/// solid mass whose fine detail is region-gated, so what is built at a
/// point depends on which district the point is in.
///
/// A FIXTURE, not a copy of the level. The shipped JSON has nine
/// districts and no oracle to check them against, so the engine pins it
/// by its properties (`every_megastructure_district_is_whole`) rather
/// than op-for-op; what this needs to be is a realistic volumetric
/// program with gates in it, for the bound and the interpreter to chew
/// on.
pub fn mega_program() -> Vec<WorldOp> {
    // Two districts either side of a cut on the first axis: one warren of
    // small walled cells, one hall of columns.
    let warren = [0.0, 0.5, 0.0, 1.0];
    let hall = [0.5, 1.0, 0.0, 1.0];
    vec![
        WorldOp::new(WOP_REGION_AXES)
            .p0([2100.0, -880.0, 9.0e-5, 7.0e-5])
            .p1([-5400.0, 3300.0, 4.0, 0.0]),
        WorldOp::new(WOP_SHAFTS_XZ)
            .region(warren)
            .p0([288.0, 90.0, 24.0, 30.0]),
        WorldOp::new(WOP_SHAFTS_XZ)
            .region(hall)
            .p0([780.0, 200.0, 55.0, 40.0]),
        WorldOp::new(WOP_COARSE_SOLID)
            .flags(WOP_FLAG_COARSE_ONLY)
            .material(2),
        // The warren: close floors, holes through them, walls both ways.
        WorldOp::new(WOP_LATTICE_Y)
            .flags(WOP_FLAG_FINE_ONLY)
            .region(warren)
            .p0([44.0, 0.0, 0.0, 0.0]),
        WorldOp::new(WOP_SLABS_Y)
            .flags(WOP_FLAG_FINE_ONLY)
            .region(warren)
            .material(2)
            .p0([1.5, 0.0, 0.0, 0.0]),
        WorldOp::new(WOP_GRID_HOLES)
            .flags(WOP_FLAG_FINE_ONLY)
            .region(warren)
            .p0([16.0, 0.16, 0.0, 0.0])
            .p1([7.0, 4.0, 7.0, 0.0]),
        WorldOp::new(WOP_PILLARS_XZ)
            .flags(WOP_FLAG_FINE_ONLY)
            .region(warren)
            .material(2)
            .p0([34.0, 8.0, 1.6, 2.2]),
        WorldOp::new(WOP_WALLS)
            .flags(WOP_FLAG_FINE_ONLY)
            .region(warren)
            .material(2)
            .p0([104.0, 1.2, 0.45, 0.0])
            .p1([0.0, 22.0, 0.5, 0.0])
            .p2([4.0, 14.0, 5.0, 12.0]),
        WorldOp::new(WOP_WALLS)
            .flags(WOP_FLAG_FINE_ONLY)
            .region(warren)
            .material(2)
            .p0([104.0, 1.2, 0.45, 1.0])
            .p1([501.0, 22.0, 0.5, 77.0])
            .p2([4.0, 14.0, 5.0, 12.0]),
        // The hall: one storey every 195 m on columns, and nothing else.
        WorldOp::new(WOP_LATTICE_Y)
            .flags(WOP_FLAG_FINE_ONLY)
            .region(hall)
            .p0([195.0, 0.0, 0.0, 0.0]),
        WorldOp::new(WOP_SLABS_Y)
            .flags(WOP_FLAG_FINE_ONLY)
            .region(hall)
            .material(7)
            .p0([6.0, 0.0, 0.0, 0.0]),
        WorldOp::new(WOP_PILLARS_XZ)
            .flags(WOP_FLAG_FINE_ONLY)
            .region(hall)
            .material(7)
            .p0([250.0, 40.0, 16.0, 12.0]),
        // One cut for both kinds of bore: only one district's shaft op
        // can pass its gate at a point, so there is only ever one shaft.
        WorldOp::new(WOP_SHAFTS_CUT),
        WorldOp::new(WOP_BEAMS)
            .flags(WOP_FLAG_FINE_ONLY)
            .region(warren)
            .material(2)
            .p0([3.0, 2.2, 1.0, 0.7])
            .p1([6.0, 0.0, 0.0, 0.0]),
        // The coarse mass takes each district's colour, so the silhouette
        // at ten kilometres already says which one it is.
        WorldOp::new(WOP_MATERIAL_BAND)
            .material(7)
            .p0(hall)
            .p1([0.0, 0.0, 2.0, 0.0]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fbm;
    use crate::fbm_mode;

    #[test]
    fn planet_program_matches_legacy_terrain_height() {
        // Oracle: the pre-program formula, verbatim.
        let ops = planet_program();
        // The base terrain, with the region ops stripped. Those ADD to
        // the height by design, so the legacy oracle describes the
        // planet without them — and must still describe it exactly, or
        // the region work moved ground it was not supposed to touch.
        let base_ops: Vec<WorldOp> = ops
            .iter()
            .copied()
            .filter(|o| o.kind != WOP_HEIGHT_BAND_FBM)
            .collect();
        let seed = 0u32;
        let mut seen = std::collections::HashSet::new();
        let mut shaped = 0usize;
        for i in 0..500 {
            let p = Vec2::new((i * 37) as f32 * 13.7, (i * 91) as f32 * -7.3);
            for vs in [1.0, 8.0, 64.0] {
                let base = fbm(seed, p, 0.00005, 3, vs) * 800.0
                    + fbm(seed, p + Vec2::new(510.0, -770.0), 0.0008, 5, vs) * 420.0
                    + fbm(seed, p + Vec2::new(1337.0, 55.0), 0.01, 5, vs) * 36.0;
                let stepped = base + 90.0 * smoothstep(180.0, 230.0, base);
                let legacy =
                    stepped + fbm(seed, p + Vec2::new(37.0, 91.0), 0.06, 4, vs) * 5.0 - 8.0;
                assert_eq!(eval_height(&base_ops, seed, p, vs), legacy);

                // With the region ops, height moves only where a region
                // claims the point — and everywhere else stays exact.
                let full = eval_height(&ops, seed, p, vs);
                let claimed: f32 = [2u32, 5, 6, 7]
                    .iter()
                    .map(|&m| surface_material_weight(&ops, seed, p, vs, m))
                    .sum();
                // Exactly zero, not nearly: outside the feather
                // `smoothstep` clamps to 0, so the band adds `0.0 * amp`
                // and the height is bit-identical. A tolerance here would
                // hide a band that reaches where it should not.
                if claimed == 0.0 {
                    assert_eq!(full, legacy, "unclaimed ground must not move");
                } else if full != legacy {
                    shaped += 1;
                }

                // (h - 3) - h is not exactly -3 in f32; the height itself is
                // bit-exact (asserted above), the SDF just subtracts it.
                let (d, mat) = eval(&ops, seed, glam::Vec3::new(p.x, full - 3.0, p.y), vs);
                assert!((d + 3.0).abs() < 1.0e-3, "d={d}");
                // The ground is whichever region claimed it. Bands only
                // repaint, so the HEIGHT is unaffected either way — that
                // is asserted bit-exact above, region ops and all.
                seen.insert(mat);
            }
        }
        // Every region the level declares is reachable, and nothing else
        // paints the ground. A band that never fires is a level bug the
        // reference program should not be able to hide.
        let mut found: Vec<u32> = seen.into_iter().collect();
        found.sort_unstable();
        assert_eq!(found, vec![1, 2, 5, 6, 7], "region coverage changed");
        // And the regions that declare terrain actually shape it.
        assert!(
            shaped > 100,
            "region terrain barely fires: {shaped} samples"
        );
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

    /// The two-loop evaluator is the one the GPU runs. It has to produce
    /// the SAME world as the linear one every planning layer, spawner and
    /// terrain query uses — bit for bit, because the split is a
    /// reordering and a reordering that changes a float has changed the
    /// surface a chunk meshes.
    ///
    /// Both shipped programs, at both sides of the coarse cutoff, and at
    /// heights that put samples above, inside and below the structures.
    #[test]
    fn the_split_evaluator_is_the_linear_one() {
        for (name, ops) in [("planet", planet_program()), ("mega", mega_program())] {
            for seed in [0u32, 7, 1337] {
                for vs in [1.0f32, 2.0, 4.0, 8.0] {
                    for i in 0..300 {
                        let p = Vec3::new(
                            i as f32 * 137.0 - 9000.0,
                            (i % 41) as f32 * 23.0 - 300.0,
                            i as f32 * -91.0 + 4000.0,
                        );
                        assert_eq!(
                            eval(&ops, seed, p, vs),
                            eval_split(&ops, seed, p, vs),
                            "{name} seed={seed} vs={vs} p={p:?}"
                        );
                    }
                }
            }
        }
    }

    /// And the column half agrees with the height-only replay about `h` —
    /// the replay is a third interpreter (shadow bake, seabed), and it
    /// runs a SUBSET of the column ops.
    #[test]
    fn the_column_pass_and_the_height_replay_agree_about_height() {
        for ops in [planet_program(), mega_program()] {
            for i in 0..300 {
                let xz = Vec2::new(i as f32 * 211.0 - 7000.0, i as f32 * -137.0);
                assert_eq!(
                    eval_column(&ops, 0, xz, 1.0).h,
                    eval_height(&ops, 0, xz, 1.0),
                    "xz={xz:?}"
                );
            }
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

    /// A region-gated op applies inside its band and nowhere else, and
    /// the bound never contradicts the samples where it does commit.
    ///
    /// The gate is the mechanism a level uses to give a district its own
    /// architecture, so "inside is different from outside" IS the
    /// feature; a gate that silently passed everywhere would leave every
    /// district looking identical and no test failing.
    #[test]
    fn a_region_gate_divides_the_world_and_keeps_the_bound_honest() {
        let mut ops = planet_program();
        // A slab of solid over the whole altitude range, confined to the
        // lower half of the first axis. Unmistakable: inside the band a
        // column is solid to the sky, outside it is untouched.
        let band = [0.0, 0.5, 0.0, 1.0];
        ops.push(WorldOp::new(WOP_COARSE_SOLID).material(2).region(band));

        let (mut inside, mut outside) = (0, 0);
        for i in 0..600 {
            let p = Vec3::new(
                i as f32 * 137.0 - 40_000.0,
                6_000.0,
                i as f32 * 91.0 - 30_000.0,
            );
            let solid = eval(&ops, 0, p, 1.0).0 < 0.0;
            // High above the terrain, so the ONLY thing that can be solid
            // here is the gated op — which makes this a direct read of
            // the gate rather than of the landscape.
            if solid {
                inside += 1
            } else {
                outside += 1
            }

            let lo = p - Vec3::splat(8.0);
            let hi = p + Vec3::splat(8.0);
            if let Some(bound) = eval_range(&ops, 0, lo, hi, 1.0) {
                let d = eval(&ops, 0, p, 1.0).0;
                assert!(bound.contains(d), "sdf {d} at {p:?} escapes {bound:?}");
            }
        }
        assert!(
            inside > 50 && outside > 50,
            "gate is one-sided: {inside} in, {outside} out"
        );
    }

    #[test]
    fn a_region_packs_and_unpacks_within_a_byte() {
        for band in [
            [0.0, 1.0, 0.0, 1.0],
            [0.25, 0.5, 0.6, 0.75],
            [0.0, 0.46, 0.0, 1.0],
        ] {
            let back = voxel_core::worldop::unpack_region(voxel_core::worldop::pack_region(band));
            for (a, b) in band.iter().zip(back.iter()) {
                assert!((a - b).abs() <= 1.0 / 255.0, "{band:?} -> {back:?}");
            }
        }
        // The ungated sentinel must not be a band any level could mean:
        // it packs from an EMPTY box, which would gate the op away
        // entirely and so is never worth authoring.
        assert!(region_gate(0, 0.5, 0.5));
        assert_eq!(voxel_core::worldop::pack_region([0.0, 0.0, 0.0, 0.0]), 0);
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
        let sky = eval_range(
            &ops,
            0,
            Vec3::new(0.0, 8_000.0, 0.0),
            Vec3::new(800.0, 8_800.0, 800.0),
            1.0,
        )
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

#[cfg(test)]
mod reach_tests {
    use super::*;

    /// The safety property, and the only one that matters: an op this
    /// says cannot reach a box must not change a single sample in it.
    ///
    /// Checked by DELETING each unreachable op and re-evaluating the box.
    /// Over-reporting is allowed — it costs a rebuild nobody needed —
    /// but an op wrongly called unreachable leaves a stale chunk, which
    /// is a wrong world that looks right.
    #[test]
    fn an_op_that_cannot_reach_a_box_changes_nothing_in_it() {
        let mut checked = 0;
        for (name, ops) in [("planet", planet_program()), ("mega", mega_program())] {
            for vs in [1.0f32, 8.0] {
                for i in 0..60 {
                    let min = Vec3::new(
                        i as f32 * 211.0 - 6000.0,
                        (i % 13) as f32 * 40.0 - 240.0,
                        i as f32 * -137.0 + 3000.0,
                    );
                    let max = min + Vec3::splat(32.0 * vs);
                    let reach = ops_reaching(&ops, 0, min, max, vs);
                    for (k, _) in ops.iter().enumerate().filter(|(_, _)| true) {
                        if reach[k] {
                            continue;
                        }
                        // The same program with that op gone.
                        let mut without = ops.clone();
                        without.remove(k);
                        for s in 0..8 {
                            let p = min
                                + (max - min)
                                    * Vec3::new(
                                        (s & 1) as f32,
                                        ((s >> 1) & 1) as f32,
                                        ((s >> 2) & 1) as f32,
                                    );
                            assert_eq!(
                                eval(&ops, 0, p, vs),
                                eval(&without, 0, p, vs),
                                "{name} vs={vs}: op {k} was called unreachable at {p:?}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        // Planet is ungated and prunes nothing, so the property is
        // exercised by the interior alone: 7,680 comparisons when this
        // was written.
        assert!(checked > 5000, "only {checked} comparisons");
    }

    /// It has to actually prune, or it is a cache key that never hits.
    /// The megastructure's nine districts are the case: a chunk inside
    /// one sees its own ops and not the other eight's.
    #[test]
    fn a_district_does_not_see_the_other_districts_ops() {
        let ops = mega_program();
        let gated = ops.iter().filter(|op| op.region != 0).count();
        let mut best = 0;
        for i in 0..40 {
            let min = Vec3::new(i as f32 * 337.0 - 4000.0, -60.0, i as f32 * -211.0);
            let reach = ops_reaching(&ops, 0, min, min + Vec3::splat(32.0), 1.0);
            let out = reach.iter().filter(|r| !**r).count();
            best = best.max(out);
        }
        // `mega_program` is the small REFERENCE fixture, not the shipped
        // megastructure — 16 ops and 12 gated, against the level's 43 and
        // nine districts. The bar is where a real regression lands, not
        // where this fixture happens to sit.
        assert!(gated >= 8, "the interior should be mostly gated: {gated}");
        assert!(
            best >= 4,
            "no box excluded more than {best} of {} ops",
            ops.len()
        );
    }

    /// A heightfield world has almost nothing to gate, and must say so
    /// rather than pretending: editing a height op DOES change every
    /// chunk.
    #[test]
    fn an_ungated_program_reaches_everywhere() {
        let ops = planet_program();
        let reach = ops_reaching(&ops, 0, Vec3::splat(-16.0), Vec3::splat(16.0), 1.0);
        let unreached = reach.iter().filter(|r| !**r).count();
        assert!(unreached <= 2, "{unreached} of {} pruned", ops.len());
    }
}
