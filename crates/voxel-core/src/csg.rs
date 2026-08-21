//! CSG operations: the compact IR that CPU planning layers (LayerProcGen)
//! hand to the GPU density shaders. Layout is shared bit-for-bit with the
//! WGSL `CsgOp` struct.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::interval::Interval;

pub const CSG_KIND_BOX_ADD: u32 = 0;
pub const CSG_KIND_BOX_CUT: u32 = 1;
pub const CSG_KIND_CYLINDER_ADD: u32 = 2;
pub const CSG_KIND_CYLINDER_CUT: u32 = 3;
pub const CSG_KIND_SPHERE_ADD: u32 = 4;
pub const CSG_KIND_SPHERE_CUT: u32 = 5;
pub const CSG_KIND_CAPSULE_ADD: u32 = 6;
pub const CSG_KIND_CAPSULE_CUT: u32 = 7;

/// One CSG operation, 48 bytes, `#[repr(C)]` — uploaded verbatim.
///
/// Boxes: `center` + `half` extents, rotated `yaw` radians about Y.
/// Cylinders: `center` (mid-height), `half.x` = radius, `half.y` = half
/// height. Capsules: `center` is the base, `half` the AXIS to the tip,
/// `yaw` the base radius and `aux.x` the tip radius — see
/// [`CsgOp::capsule`]. `blend` is the smooth-min radius in meters
/// (0 = hard).
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct CsgOp {
    pub center: [f32; 3],
    pub kind: u32,
    pub half: [f32; 3],
    pub material: u32,
    pub yaw: f32,
    pub blend: f32,
    /// Kind-specific extra, zero for the shapes that need none. It was
    /// padding to 48 bytes and still is for every other kind; the
    /// capsule spends `aux.x` on its tip radius rather than growing the
    /// struct, because 48 bytes is a layout twin in three places.
    pub aux: [f32; 2],
}

impl CsgOp {
    pub fn boxy(center: Vec3, half: Vec3, yaw: f32, material: u32, cut: bool) -> Self {
        Self::of(CSG_KIND_BOX_ADD, center, half, material, cut).yaw(yaw)
    }

    pub fn cylinder(center: Vec3, radius: f32, half_height: f32, material: u32, cut: bool) -> Self {
        let half = Vec3::new(radius, half_height, radius);
        Self::of(CSG_KIND_CYLINDER_ADD, center, half, material, cut)
    }

    /// Sphere: `half.x` = radius (spheres ignore yaw).
    pub fn sphere(center: Vec3, radius: f32, material: u32, cut: bool) -> Self {
        Self::of(
            CSG_KIND_SPHERE_ADD,
            center,
            Vec3::splat(radius),
            material,
            cut,
        )
    }

    /// A tapered capsule from `a` (radius `r_a`) to `b` (radius `r_b`).
    ///
    /// The one primitive that points anywhere. Every other kind is
    /// yaw-only, which is fine for a wall or a shaft and useless for a
    /// branch, a root or a tendril — those go where they grow. So the
    /// axis is stored as a VECTOR rather than an orientation, and the
    /// capsule is the one kind `op_sdf` answers before rotating anything.
    ///
    /// One limb, one op: a skeleton of a few dozen limbs costs a few
    /// dozen ops, which is what makes organic shapes affordable in a
    /// field that is evaluated per sample.
    pub fn capsule(a: Vec3, b: Vec3, r_a: f32, r_b: f32, material: u32, cut: bool) -> Self {
        Self {
            center: a.to_array(),
            kind: CSG_KIND_CAPSULE_ADD + u32::from(cut),
            half: (b - a).to_array(),
            material,
            yaw: r_a,
            blend: 0.0,
            aux: [r_b, 0.0],
        }
    }

    /// The shared body of the three constructors. `add` is the ADD kind;
    /// every cut kind is its successor, which is also what `apply` and the
    /// WGSL twin rely on when they test `kind & 1`.
    fn of(add: u32, center: Vec3, half: Vec3, material: u32, cut: bool) -> Self {
        Self {
            center: center.to_array(),
            kind: add + u32::from(cut),
            half: half.to_array(),
            material,
            yaw: 0.0,
            blend: 0.0,
            aux: [0.0; 2],
        }
    }

    fn yaw(mut self, yaw: f32) -> Self {
        self.yaw = yaw;
        self
    }

    /// Signed distance to this op's primitive (mirrors the WGSL `op_sdf`).
    pub fn sdf(&self, p: Vec3) -> f32 {
        // BEFORE the yaw rotation: a capsule has no yaw — that field is
        // its base radius, and rotating by it would bend every limb.
        if self.kind >= CSG_KIND_CAPSULE_ADD {
            let pa = p - Vec3::from(self.center);
            let ba = Vec3::from(self.half);
            let t = (pa.dot(ba) / ba.dot(ba).max(1.0e-8)).clamp(0.0, 1.0);
            return (pa - ba * t).length() - (self.yaw + (self.aux[0] - self.yaw) * t);
        }
        let mut q = p - Vec3::from(self.center);
        let (s, c) = (-self.yaw).sin_cos();
        q = Vec3::new(q.x * c - q.z * s, q.y, q.x * s + q.z * c);
        let h = Vec3::from(self.half);
        if self.kind >= 4 {
            q.length() - h.x
        } else if self.kind < 2 {
            let a = q.abs() - h;
            a.max(Vec3::ZERO).length() + a.x.max(a.y.max(a.z)).min(0.0)
        } else {
            let dr = (q.x * q.x + q.z * q.z).sqrt() - h.x;
            let dy = q.y.abs() - h.y;
            glam::Vec2::new(dr.max(0.0), dy.max(0.0)).length() + dr.max(dy).min(0.0)
        }
    }

    /// Bound of [`Self::sdf`] over an axis-aligned box.
    ///
    /// Every primitive here is a TRUE distance field, so its gradient
    /// has magnitude at most one and the whole interval follows from a
    /// single evaluation: within `r` metres of the centre the value can
    /// have moved by at most `r`. That is what makes pruning cheap
    /// enough to do per sub-cell — one `sdf` call, not a per-kind
    /// interval arithmetic twin that would have to stay in sync.
    ///
    /// The tapered capsule is the exception and is handled: its
    /// round-cone field changes by `|r_a - r_b|` more than distance does
    /// over its own length, so its Lipschitz bound is inflated by
    /// exactly that ratio. Understating it would drop an op that matters,
    /// which is a hole in the world.
    pub fn sdf_range(&self, min: Vec3, max: Vec3) -> Interval {
        let c = (min + max) * 0.5;
        let r = (max - min).length() * 0.5;
        let lip = if self.kind >= CSG_KIND_CAPSULE_ADD {
            let len = Vec3::from(self.half).length().max(1.0e-6);
            1.0 + (self.aux[0] - self.yaw).abs() / len
        } else {
            1.0
        };
        let d = self.sdf(c);
        Interval::new(d - lip * r, d + lip * r)
    }

    /// Fold this op into a scene distance (ignores smooth blend — CPU
    /// collision does not need it).
    pub fn apply(&self, d: f32, p: Vec3) -> f32 {
        let od = self.sdf(p);
        if self.kind & 1 == 0 {
            d.min(od)
        } else {
            d.max(-od)
        }
    }

    /// Conservative world-space AABB (yaw-safe: uses the diagonal, and
    /// a smooth blend reaches past the shape by its own radius).
    pub fn aabb(&self) -> Aabb {
        let h = Vec3::from(self.half);
        // A capsule's `half` is an axis, not an extent: the box is the
        // union of a ball at each end. Reading it as an extent would
        // give a box centred on the BASE that misses most of the limb.
        if self.kind >= CSG_KIND_CAPSULE_ADD {
            let a = Vec3::from(self.center);
            let b = a + h;
            let (ra, rb) = (self.yaw + self.blend, self.aux[0] + self.blend);
            return Aabb::new(
                (a - Vec3::splat(ra)).min(b - Vec3::splat(rb)),
                (a + Vec3::splat(ra)).max(b + Vec3::splat(rb)),
            );
        }
        let r = (h.x * h.x + h.z * h.z).sqrt().max(h.x.max(h.z));
        Aabb::around(
            Vec3::from(self.center),
            Vec3::new(r, h.y, r) + Vec3::splat(self.blend),
        )
    }

    /// Does this op affect `box`?
    pub fn touches(&self, r#box: Aabb) -> bool {
        self.aabb().touches(r#box)
    }
}

/// What interval evaluation proved about one op over a box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    /// Cannot change the result anywhere in the box — drop it.
    Skip,
    /// Decides the result everywhere in the box — every op BEFORE it is
    /// dead, whatever they were.
    Replaces,
    /// Undecided: it matters somewhere in here.
    Both,
}

/// The ops in a union/cut chain that can change the result over a box.
///
/// This is Keeter's tape pruning (MPR, Algorithms 1 and 2) specialised to
/// the chain `apply_csg` actually runs: a forward interval pass records a
/// choice per op, and a backward pass keeps only what is live. A chain is
/// linear rather than a DAG, so "live" collapses to "after the last op
/// that decides the result", and no register liveness is needed.
///
/// `start` is the interval of the distance the chain STARTS from — the
/// generator's own field over this box. It is what makes pruning work at
/// all: without a finite bound on what is already there, no op can be
/// proved irrelevant.
///
/// Sound, never exact: it can keep an op that turns out not to matter,
/// and must never drop one that does. Every bound below is conservative
/// in that direction.
pub fn prune_chain(ops: &[CsgOp], start: Interval, min: Vec3, max: Vec3) -> Vec<u32> {
    let mut choices: Vec<Choice> = Vec::with_capacity(ops.len());
    let mut d = start;
    for op in ops {
        let od = op.sdf_range(min, max);
        let choice = if op.kind & 1 == 0 {
            // Union: `d = min(d, od)`.
            if od.lo > d.hi {
                Choice::Skip // always further than what we have
            } else if od.hi < d.lo {
                Choice::Replaces // always nearer: it IS the result
            } else {
                Choice::Both
            }
        } else {
            // Cut: `d = max(d, -od)`. Irrelevant when `-od <= d` over the
            // whole box, i.e. when `-od.lo <= d.lo`.
            if -od.lo <= d.lo {
                Choice::Skip
            } else if -od.hi >= d.hi {
                Choice::Replaces
            } else {
                Choice::Both
            }
        };
        d = match choice {
            Choice::Skip => d,
            Choice::Replaces if op.kind & 1 == 0 => od,
            Choice::Replaces => Interval::new(-od.hi, -od.lo),
            Choice::Both if op.kind & 1 == 0 => d.min(od),
            Choice::Both => d.max(Interval::new(-od.hi, -od.lo)),
        };
        choices.push(choice);
    }
    // Backward: everything before the last decider is dead.
    let from = choices
        .iter()
        .rposition(|c| *c == Choice::Replaces)
        .unwrap_or(0);
    (from..ops.len())
        .filter(|i| choices[*i] != Choice::Skip)
        .map(|i| i as u32)
        .collect()
}

/// An axis-aligned box in world meters.
///
/// One name for a question asked all over: does this thing reach that
/// place. It was written out six times before this — in the op cull, the
/// chunk fingerprint, the edit sweep and three tests — and six copies of
/// an inequality chain are six chances to get one `<=` backwards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// A box of half-extents `half` about `center`.
    pub fn around(center: Vec3, half: Vec3) -> Self {
        Self::new(center - half, center + half)
    }

    /// Do the two overlap? Touching at a face counts: a chunk reads the
    /// samples on its own boundary.
    pub fn touches(self, other: Self) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }

    /// The smallest box holding both.
    pub fn union(self, other: Self) -> Self {
        Self::new(self.min.min(other.min), self.max.max(other.max))
    }

    /// Grown by `by` meters on every side.
    pub fn inflate(self, by: f32) -> Self {
        Self::new(self.min - Vec3::splat(by), self.max + Vec3::splat(by))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_is_48_bytes_pod() {
        assert_eq!(std::mem::size_of::<CsgOp>(), 48);
        let op = CsgOp::boxy(Vec3::new(1.0, 2.0, 3.0), Vec3::ONE, 0.5, 3, false);
        let bytes: &[u8] = bytemuck::bytes_of(&op);
        let back: &CsgOp = bytemuck::from_bytes(bytes);
        assert_eq!(*back, op);
    }

    /// THE property the whole optimisation rests on: a pruned chain
    /// answers exactly what the full chain answers, everywhere inside
    /// the box it was pruned for.
    ///
    /// Dropping an op that mattered is a hole in the world that appears
    /// only where some camera happens to look, so this is checked
    /// against the real evaluator over randomised scenes rather than
    /// argued about. Randomised because the failure mode is a bound that
    /// is tight for the shapes you thought of.
    #[test]
    fn a_pruned_chain_answers_what_the_full_chain_answers() {
        use crate::seed::Rng;
        let mut rng = Rng::new(0xC5_6A);
        let mut total_kept = 0usize;
        let mut total_ops = 0usize;
        for case in 0..200 {
            let mut f = || rng.next_f32();
            // A scene of mixed kinds scattered over a wide area, so most
            // ops are far from any one box and prunable — the situation
            // a chunk full of trees is actually in.
            let ops: Vec<CsgOp> = (0..24)
                .map(|i| {
                    let c = Vec3::new(f() * 40.0 - 20.0, f() * 40.0 - 20.0, f() * 40.0 - 20.0);
                    let cut = i % 4 == 3;
                    match i % 4 {
                        0 => CsgOp::sphere(c, 0.5 + f() * 3.0, 1, cut),
                        1 => CsgOp::boxy(c, Vec3::splat(0.5 + f() * 2.0), f() * 3.0, 1, cut),
                        2 => CsgOp::capsule(
                            c,
                            c + Vec3::new(f() * 6.0 - 3.0, f() * 6.0 - 3.0, f() * 6.0 - 3.0),
                            0.2 + f() * 1.0,
                            0.2 + f() * 1.0,
                            1,
                            cut,
                        ),
                        _ => CsgOp::cylinder(c, 0.5 + f() * 2.0, 0.5 + f() * 3.0, 1, cut),
                    }
                })
                .collect();

            // A box somewhere in the scene, at a chunk-ish scale.
            let lo = Vec3::new(f() * 30.0 - 15.0, f() * 30.0 - 15.0, f() * 30.0 - 15.0);
            let size = 0.5 + f() * 6.0;
            let hi = lo + Vec3::splat(size);

            // The chain starts from a terrain-like field. Its interval
            // has to BOUND it or pruning is unsound, so use a plane and
            // its exact bound.
            let plane_h = f() * 10.0 - 5.0;
            let start = Interval::new(lo.y - plane_h, hi.y - plane_h);

            let kept = prune_chain(&ops, start, lo, hi);
            total_kept += kept.len();
            total_ops += ops.len();

            for _ in 0..40 {
                let p = lo + Vec3::new(f(), f(), f()) * size;
                let d0 = p.y - plane_h;
                let full = ops.iter().fold(d0, |d, op| op.apply(d, p));
                let pruned = kept.iter().fold(d0, |d, i| ops[*i as usize].apply(d, p));
                assert!(
                    (full - pruned).abs() < 1.0e-4,
                    "case {case}: pruned chain disagrees at {p:?}: {full} vs {pruned} \
                     (kept {} of {})",
                    kept.len(),
                    ops.len(),
                );
            }
        }
        // And it has to actually prune, or the test above passes for the
        // most boring possible reason.
        assert!(
            total_kept * 3 < total_ops,
            "pruning kept {total_kept} of {total_ops} — not pruning"
        );
        println!("pruned to {total_kept} of {total_ops}");
    }

    /// A capsule is the one kind that points anywhere, so the thing to
    /// pin is that its distance field is right OFF the axis and in every
    /// direction — not just along it.
    #[test]
    fn a_capsule_measures_distance_to_its_axis() {
        // A diagonal limb, deliberately not axis-aligned: an
        // implementation that fell back to yaw rotation or read `half`
        // as an extent passes an axis-aligned test and fails this one.
        let a = Vec3::new(1.0, 2.0, -1.0);
        let b = Vec3::new(3.0, 5.0, 2.0);
        let op = CsgOp::capsule(a, b, 0.5, 0.5, 0, false);

        // On the axis: inside by the radius.
        let mid = (a + b) * 0.5;
        assert!((op.sdf(mid) + 0.5).abs() < 1.0e-5, "{}", op.sdf(mid));
        // Exactly on the surface, measured perpendicular to the axis.
        let axis = (b - a).normalize();
        let perp = axis.cross(Vec3::Y).normalize();
        assert!((op.sdf(mid + perp * 0.5)).abs() < 1.0e-5);
        assert!((op.sdf(mid + perp * 1.5) - 1.0).abs() < 1.0e-5);
        // Past an END it is a ball, not an infinite cylinder.
        assert!((op.sdf(a - axis * 2.0) - 1.5).abs() < 1.0e-5);
        assert!((op.sdf(b + axis * 2.0) - 1.5).abs() < 1.0e-5);

        // Taper: the radius follows the parameter along the axis.
        let cone = CsgOp::capsule(Vec3::ZERO, Vec3::Y * 4.0, 1.0, 0.0, 0, false);
        assert!((cone.sdf(Vec3::new(1.0, 0.0, 0.0))).abs() < 1.0e-5);
        assert!((cone.sdf(Vec3::new(0.5, 2.0, 0.0))).abs() < 1.0e-5);
        assert!(cone.sdf(Vec3::new(0.9, 3.6, 0.0)) > 0.0, "tip is thin");
    }

    /// The conservative AABB has to cover a capsule wherever it points —
    /// a box read off `half` as an extent misses most of a limb, and a
    /// missed op is geometry that vanishes when a chunk culls it.
    #[test]
    fn aabb_covers_a_capsule_in_any_direction() {
        for dir in [
            Vec3::new(3.0, 5.0, 2.0),
            Vec3::new(-4.0, -1.0, 2.5),
            Vec3::new(0.0, -6.0, 0.0),
        ] {
            let a = Vec3::new(1.0, 2.0, -1.0);
            let op = CsgOp::capsule(a, a + dir, 0.5, 0.9, 0, false);
            let bb = op.aabb();
            // Sample along the limb; every solid point must be inside.
            for i in 0..=20 {
                let t = i as f32 / 20.0;
                let p = a + dir * t;
                let r = 0.5 + (0.9 - 0.5) * t;
                for off in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z] {
                    let s = p + off * r;
                    assert!(
                        bb.min.cmple(s).all() && s.cmple(bb.max).all(),
                        "capsule surface {s:?} outside {bb:?} for dir {dir:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn aabb_covers_rotated_box() {
        // A yawed box's corners stay inside the conservative AABB.
        let op = CsgOp::boxy(Vec3::ZERO, Vec3::new(4.0, 1.0, 1.0), 0.7, 0, false);
        let b = op.aabb();
        for sx in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let corner = Vec3::new(4.0 * sx, 0.0, 1.0 * sz);
                let (s, c) = (0.7f32.sin(), 0.7f32.cos());
                let world = Vec3::new(
                    corner.x * c - corner.z * s,
                    0.0,
                    corner.x * s + corner.z * c,
                );
                assert!(world.x >= b.min.x && world.x <= b.max.x);
                assert!(world.z >= b.min.z && world.z <= b.max.z);
            }
        }
    }

    #[test]
    fn sdf_matches_primitives() {
        let b = CsgOp::boxy(Vec3::ZERO, Vec3::new(2.0, 1.0, 3.0), 0.0, 0, false);
        assert!(b.sdf(Vec3::ZERO) < 0.0);
        assert!((b.sdf(Vec3::new(4.0, 0.0, 0.0)) - 2.0).abs() < 1e-5);
        let cyl = CsgOp::cylinder(Vec3::ZERO, 1.5, 2.0, 0, false);
        assert!((cyl.sdf(Vec3::new(3.0, 0.0, 0.0)) - 1.5).abs() < 1e-5);
        assert!((cyl.sdf(Vec3::new(0.0, 5.0, 0.0)) - 3.0).abs() < 1e-5);
        // Cut ops carve: applying a cut around a point makes it air.
        let cut = CsgOp::boxy(Vec3::ZERO, Vec3::ONE, 0.0, 0, true);
        assert!(cut.apply(-10.0, Vec3::ZERO) > 0.0);
    }

    #[test]
    fn touches_is_conservative() {
        let op = CsgOp::cylinder(Vec3::new(100.0, 0.0, 0.0), 5.0, 10.0, 0, false);
        assert!(op.touches(Aabb::new(
            Vec3::new(90.0, -5.0, -5.0),
            Vec3::new(110.0, 5.0, 5.0)
        )));
        assert!(!op.touches(Aabb::new(
            Vec3::new(200.0, 0.0, 0.0),
            Vec3::new(210.0, 10.0, 10.0)
        )));
    }
}
