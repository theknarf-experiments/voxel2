//! Structure-world gameplay queries: signed distance against the current
//! level's generator program (see [`crate::program`]) plus the planned
//! habitation-pocket variation ops.

use glam::Vec3;

/// Signed distance (meters) to the current level's generator program at
/// full detail — used for collision and gameplay queries.
pub fn mega_sdf(p: Vec3) -> f32 {
    crate::program::eval(&crate::program::program(), p, 1.0).0
}

/// Numerical SDF gradient (central differences, 0.1 m).
pub fn mega_gradient(p: Vec3) -> Vec3 {
    let e = 0.1;
    Vec3::new(
        mega_sdf(p + Vec3::X * e) - mega_sdf(p - Vec3::X * e),
        mega_sdf(p + Vec3::Y * e) - mega_sdf(p - Vec3::Y * e),
        mega_sdf(p + Vec3::Z * e) - mega_sdf(p - Vec3::Z * e),
    )
    .normalize_or_zero()
}

// --- planned variation: habitation pockets & light wells ---------------------

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


/// Full-world SDF for collision: base structure plus planned ops near `p`.
pub fn mega_sdf_with_ops(p: Vec3, ops: &[CsgOp]) -> f32 {
    let mut d = mega_sdf(p);
    for op in ops {
        d = op.apply(d, p);
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    // Evaluate the reference mega program explicitly: the process-global
    // program is shared test state and defaults to the planet.
    fn msdf(p: Vec3) -> f32 {
        crate::program::eval(&crate::program::mega_program(), p, 1.0).0
    }

    /// A floor point that is solid slab (not inside an opening/shaft).
    fn solid_floor_point() -> Vec3 {
        for i in 0..200 {
            let p = Vec3::new(37.0 + i as f32 * 17.0, 0.0, 91.0 + i as f32 * 5.0);
            if msdf(p) < -1.0 {
                return p;
            }
        }
        panic!("no solid floor found on scan line");
    }

    #[test]
    fn floors_are_solid_and_rooms_are_air() {
        let floor = solid_floor_point();
        assert!(msdf(floor) < -1.0);
        // Mid-room air exists somewhere above the slab plane.
        let mut found_air = false;
        for i in 0..80 {
            let p = Vec3::new(3.0 + i as f32 * 5.0, 20.0, 9.0);
            if msdf(p) > 1.0 {
                found_air = true;
                break;
            }
        }
        assert!(found_air, "no open room space found");
    }


    #[test]
    fn gradient_points_out_of_floor() {
        let floor = solid_floor_point();
        let p = floor + Vec3::Y * 1.4;
        let e = 0.1;
        let g = Vec3::new(
            msdf(p + Vec3::X * e) - msdf(p - Vec3::X * e),
            msdf(p + Vec3::Y * e) - msdf(p - Vec3::Y * e),
            msdf(p + Vec3::Z * e) - msdf(p - Vec3::Z * e),
        )
        .normalize_or_zero();
        assert!(g.y > 0.6, "gradient near floor top should point up: {g}");
    }
}
