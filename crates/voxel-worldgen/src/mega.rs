//! Structure-world gameplay queries: signed distance against the current
//! level's generator program (see [`crate::program`]) plus the planned
//! habitation-pocket variation ops.

use glam::{IVec3, Vec3};

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
use voxel_core::seed::{chunk_seed, Rng};

const POCKET_CELL_XZ: f32 = 128.0;
const POCKET_CELL_Y: f32 = 44.0 * 3.0;
const POCKET_SEED: u64 = 0xB10C;

/// Planned features overlapping `[min, max]`: hollow room shells with
/// doorways on floor tops, and vertical light wells cut through several
/// levels. Shared by the GPU density pass and CPU collision.
pub fn pockets_ops(seed: u64, chance: f32, min: Vec3, max: Vec3) -> Vec<CsgOp> {
    let lo = |v: f32, c: f32| ((v - 20.0) / c).floor() as i32;
    let hi = |v: f32, c: f32| ((v + 20.0) / c).floor() as i32;
    let mut out = Vec::new();
    for cy in lo(min.y, POCKET_CELL_Y)..=hi(max.y, POCKET_CELL_Y) {
        for cz in lo(min.z, POCKET_CELL_XZ)..=hi(max.z, POCKET_CELL_XZ) {
            for cx in lo(min.x, POCKET_CELL_XZ)..=hi(max.x, POCKET_CELL_XZ) {
                pocket_cell_ops(seed, chance, cx, cy, cz, &mut out);
            }
        }
    }
    out.retain(|op| op.touches(min, max));
    out
}

fn pocket_cell_ops(seed: u64, chance: f32, cx: i32, cy: i32, cz: i32, out: &mut Vec<CsgOp>) {
    let mut rng = Rng::new(chunk_seed(POCKET_SEED ^ seed, 0x1, IVec3::new(cx, cy, cz)));
    let roll = rng.next_f32();
    if roll > chance {
        return;
    }
    // Sub-features keyed to the original scale so chance only adds/removes
    // pockets rather than reshaping survivors.
    let roll = roll * 0.45 / chance.max(1.0e-6);

    let x = cx as f32 * POCKET_CELL_XZ + 24.0 + rng.next_f32() * (POCKET_CELL_XZ - 48.0);
    let z = cz as f32 * POCKET_CELL_XZ + 24.0 + rng.next_f32() * (POCKET_CELL_XZ - 48.0);
    // Snap to the nearest structural floor level inside the cell.
    let fs = crate::program::lattice_y_spacing(&crate::program::program()).unwrap_or(44.0);
    let level = ((cy as f32 + 0.5) * POCKET_CELL_Y / fs).round();
    let floor_top = level * fs + 1.5;

    if roll < 0.12 {
        // Light well: a square shaft cut through three levels.
        out.push(CsgOp::boxy(
            Vec3::new(x, floor_top + fs, z),
            Vec3::new(2.4, fs * 1.6, 2.4),
            0.0,
            0,
            true,
        ));
        return;
    }

    // Habitation pocket: 2-4 hollow room shells on the floor, orthogonal
    // (Blame! is right angles), with doorway cuts.
    let rooms = 2 + rng.next_range(3);
    let mut px = x;
    let mut pz = z;
    for _ in 0..rooms {
        let hx = 4.0 + rng.next_f32() * 4.0;
        let hy = 2.6 + rng.next_f32() * 2.0;
        let hz = 4.0 + rng.next_f32() * 4.0;
        let center = Vec3::new(px, floor_top + hy - 0.4, pz);
        // Shell: solid box minus interior.
        out.push(CsgOp::boxy(center, Vec3::new(hx, hy, hz), 0.0, 2, false));
        out.push(CsgOp::boxy(
            center,
            Vec3::new(hx - 0.7, hy - 0.7, hz - 0.7),
            0.0,
            0,
            true,
        ));
        // Doorway on a random side.
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
        // Next room steps orthogonally.
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
    fn pockets_are_deterministic_and_culled() {
        let min = Vec3::new(-500.0, -100.0, -500.0);
        let max = Vec3::new(500.0, 150.0, 500.0);
        let a = pockets_ops(0, 0.45, min, max);
        let b = pockets_ops(0, 0.45, min, max);
        assert_eq!(a, b);
        assert!(!a.is_empty(), "no pockets in 1 km cube");
        for op in &a {
            assert!(op.touches(min, max));
        }
        // Room shells: at least one add op paired with a cut.
        assert!(a.iter().any(|o| o.kind & 1 == 0));
        assert!(a.iter().any(|o| o.kind & 1 == 1));
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
