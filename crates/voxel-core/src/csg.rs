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

/// One CSG operation, 48 bytes, `#[repr(C)]` — uploaded verbatim.
///
/// Boxes: `center` + `half` extents, rotated `yaw` radians about Y.
/// Cylinders: `center` (mid-height), `half.x` = radius, `half.y` = half
/// height. `blend` is the smooth-min radius in meters (0 = hard).
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct CsgOp {
    pub center: [f32; 3],
    pub kind: u32,
    pub half: [f32; 3],
    pub material: u32,
    pub yaw: f32,
    pub blend: f32,
    pub _pad: [u32; 2],
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
            _pad: [0; 2],
        }
    }

    fn yaw(mut self, yaw: f32) -> Self {
        self.yaw = yaw;
        self
    }

    /// Signed distance to this op's primitive (mirrors the WGSL `op_sdf`).
    pub fn sdf(&self, p: Vec3) -> f32 {
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
