//! The "pocket" volumetric recipe for the megastructure's `scatter3`
//! sites, placed by the stack's `site_recipe3` emit.

use glam::Vec3;
use voxel_core::csg::CsgOp;
use voxel_core::seed::Rng;

/// The "pocket" volumetric recipe (stack `scatter3` sites): a light
/// well, or 2-4 hollow orthogonal room shells with doorways, seated on
/// the structural floor at the site (`site.y` is the lattice level).
pub fn pocket_recipe_ops(site: Vec3, rng: &mut Rng, out: &mut Vec<CsgOp>) {
    let floor_top = site.y + 1.5;
    let (x, z) = (site.x, site.z);
    let fs = crate::program::lattice_y_spacing(&crate::program::program()).unwrap_or(44.0);
    let roll = rng.next_f32();

    if roll < 0.12 {
        // Light well: a square shaft cut through two levels. Its whole
        // AABB must stay within stack::ELEM_PAD_M of the site — queries
        // farther than that never see site-bucketed ops.
        out.push(CsgOp::boxy(
            Vec3::new(x, floor_top + fs * 0.65, z),
            Vec3::new(2.4, fs * 0.7, 2.4),
            0.0,
            0,
            true,
        ));
        return;
    }

    // Habitation pocket (Blame! is right angles).
    let rooms = 2 + rng.next_range(3);
    let mut px = x;
    let mut pz = z;
    for _ in 0..rooms {
        let hx = 4.0 + rng.next_f32() * 4.0;
        let hy = 2.6 + rng.next_f32() * 2.0;
        let hz = 4.0 + rng.next_f32() * 4.0;
        let center = Vec3::new(px, floor_top + hy - 0.4, pz);
        out.push(CsgOp::boxy(center, Vec3::new(hx, hy, hz), 0.0, 2, false));
        out.push(CsgOp::boxy(
            center,
            Vec3::new(hx - 0.7, hy - 0.7, hz - 0.7),
            0.0,
            0,
            true,
        ));
        let side = rng.next_range(4);
        let (dx, dz) = match side {
            0 => (hx, 0.0),
            1 => (-hx, 0.0),
            2 => (0.0, hz),
            _ => (0.0, -hz),
        };
        out.push(CsgOp::boxy(
            Vec3::new(px + dx, floor_top + 1.4, pz + dz),
            Vec3::new(
                if dz == 0.0 { 1.0 } else { 1.3 },
                1.4,
                if dz == 0.0 { 1.3 } else { 1.0 },
            ),
            0.0,
            0,
            true,
        ));
        if rng.next_f32() < 0.5 {
            px += (hx + 5.0) * if rng.next_f32() < 0.5 { 1.0 } else { -1.0 };
        } else {
            pz += (hz + 5.0) * if rng.next_f32() < 0.5 { 1.0 } else { -1.0 };
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::seed::{chunk_seed, Rng};

    #[test]
    fn pocket_recipe_is_deterministic_and_seated_on_its_floor() {
        let site = Vec3::new(40.0, 88.0, -70.0);
        let ops = |salt: u64| {
            let mut rng = Rng::new(chunk_seed(salt, 0x0c, glam::IVec3::new(3, 1, 4)));
            let mut out = Vec::new();
            pocket_recipe_ops(site, &mut rng, &mut out);
            out
        };
        let a = ops(5);
        assert_eq!(a, ops(5));
        assert_ne!(a, ops(6));
        assert!(!a.is_empty());
        // Rooms sit on the site's floor plane; light wells stay within
        // the element-padding reach.
        for op in &a {
            assert!(op.center[1] > site.y - 2.0 && op.center[1] < site.y + 120.0);
        }
    }
}
