//! CSG operations: the compact IR that CPU planning layers (LayerProcGen)
//! hand to the GPU density shaders. Layout is shared bit-for-bit with the
//! WGSL `CsgOp` struct.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

pub const CSG_KIND_BOX_ADD: u32 = 0;
pub const CSG_KIND_BOX_CUT: u32 = 1;
pub const CSG_KIND_CYLINDER_ADD: u32 = 2;
pub const CSG_KIND_CYLINDER_CUT: u32 = 3;

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
        Self {
            center: center.to_array(),
            kind: if cut { CSG_KIND_BOX_CUT } else { CSG_KIND_BOX_ADD },
            half: half.to_array(),
            material,
            yaw,
            blend: 0.0,
            _pad: [0; 2],
        }
    }

    pub fn cylinder(center: Vec3, radius: f32, half_height: f32, material: u32, cut: bool) -> Self {
        Self {
            center: center.to_array(),
            kind: if cut { CSG_KIND_CYLINDER_CUT } else { CSG_KIND_CYLINDER_ADD },
            half: [radius, half_height, radius],
            material,
            yaw: 0.0,
            blend: 0.0,
            _pad: [0; 2],
        }
    }

    /// Conservative world-space AABB (yaw-safe: uses the diagonal).
    pub fn aabb(&self) -> (Vec3, Vec3) {
        let c = Vec3::from(self.center);
        let h = Vec3::from(self.half);
        let r = (h.x * h.x + h.z * h.z).sqrt().max(h.x.max(h.z));
        let e = Vec3::new(r, h.y, r) + Vec3::splat(self.blend);
        (c - e, c + e)
    }

    /// Does this op affect the chunk box `[min, max]` (meters)?
    pub fn touches(&self, min: Vec3, max: Vec3) -> bool {
        let (lo, hi) = self.aabb();
        lo.x <= max.x && hi.x >= min.x && lo.y <= max.y && hi.y >= min.y && lo.z <= max.z && hi.z >= min.z
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
        let (lo, hi) = op.aabb();
        for sx in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let corner = Vec3::new(4.0 * sx, 0.0, 1.0 * sz);
                let (s, c) = (0.7f32.sin(), 0.7f32.cos());
                let world = Vec3::new(corner.x * c - corner.z * s, 0.0, corner.x * s + corner.z * c);
                assert!(world.x >= lo.x && world.x <= hi.x);
                assert!(world.z >= lo.z && world.z <= hi.z);
            }
        }
    }

    #[test]
    fn touches_is_conservative() {
        let op = CsgOp::cylinder(Vec3::new(100.0, 0.0, 0.0), 5.0, 10.0, 0, false);
        assert!(op.touches(Vec3::new(90.0, -5.0, -5.0), Vec3::new(110.0, 5.0, 5.0)));
        assert!(!op.touches(Vec3::new(200.0, 0.0, 0.0), Vec3::new(210.0, 10.0, 10.0)));
    }
}
