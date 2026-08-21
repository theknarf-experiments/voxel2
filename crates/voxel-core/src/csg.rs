//! CSG operations: the compact IR that CPU planning layers (LayerProcGen)
//! hand to the GPU density shaders. Layout is shared bit-for-bit with the
//! WGSL `CsgOp` struct.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

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
