//! CPU mirror of the megastructure SDF
//! (`voxel-render/src/shaders/voxel_mega_density.wgsl`) — used for
//! collision and gameplay queries inside the Blame! world. Must stay
//! function-identical to the WGSL (same hashes, constants, and op order).

use glam::{IVec2, IVec3, Vec2, Vec3};

const FLOOR_SPACING: f32 = 44.0;
const PILLAR_SPACING: f32 = 34.0;
const WALL_SPACING: f32 = 104.0;
const SHAFT_SPACING: f32 = 288.0;

fn hash2(p: IVec2) -> f32 {
    let mut h: u32 = (p.x as u32)
        .wrapping_mul(374_761_393)
        .wrapping_add((p.y as u32).wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0xFF_FFFF) as f32 / 16_777_216.0
}

fn hash3(p: IVec3) -> f32 {
    let mut h: u32 = (p.x as u32)
        .wrapping_mul(374_761_393)
        .wrapping_add((p.y as u32).wrapping_mul(668_265_263))
        .wrapping_add((p.z as u32).wrapping_mul(2_246_822_519));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0xFF_FFFF) as f32 / 16_777_216.0
}

fn sd_box(p: Vec3, b: Vec3) -> f32 {
    let q = p.abs() - b;
    q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
}

/// Signed distance (meters) to the megastructure at full detail.
pub fn mega_sdf(p: Vec3) -> f32 {
    let pxz = Vec2::new(p.x, p.z);

    // Megashafts.
    let sc = IVec2::new(
        (p.x / SHAFT_SPACING).round() as i32,
        (p.z / SHAFT_SPACING).round() as i32,
    );
    let sjit = Vec2::new(
        hash2(sc + IVec2::new(41, 13)) - 0.5,
        hash2(sc + IVec2::new(-7, 99)) - 0.5,
    ) * 90.0;
    let sxz = pxz - Vec2::new(sc.x as f32, sc.y as f32) * SHAFT_SPACING - sjit;
    let sr = 24.0 + hash2(sc) * 30.0;
    let shaft = sxz.length() - sr;

    // Floors.
    let level = (p.y / FLOOR_SPACING).round();
    let fy = p.y - level * FLOOR_SPACING;
    let mut d = fy.abs() - 1.5;

    // Floor openings.
    let op_cell = IVec2::new((p.x / 16.0).floor() as i32, (p.z / 16.0).floor() as i32);
    let op = hash3(IVec3::new(op_cell.x, level as i32, op_cell.y));
    if op < 0.16 {
        let oc = (Vec2::new(op_cell.x as f32, op_cell.y as f32) + 0.5) * 16.0;
        let cut = sd_box(
            Vec3::new(p.x - oc.x, fy, p.z - oc.y),
            Vec3::new(7.0, 4.0, 7.0),
        );
        d = d.max(-cut);
    }

    // Pillars.
    let pc = IVec2::new(
        (p.x / PILLAR_SPACING).round() as i32,
        (p.z / PILLAR_SPACING).round() as i32,
    );
    let jit = Vec2::new(hash2(pc) - 0.5, hash2(pc + IVec2::new(311, 77)) - 0.5) * 8.0;
    let pp = pxz - Vec2::new(pc.x as f32, pc.y as f32) * PILLAR_SPACING - jit;
    let girth = 1.6 + hash2(pc + IVec2::new(9, -4)) * 2.2;
    let pillar = pp.x.abs().max(pp.y.abs()) - girth;
    d = d.min(pillar);

    // Walls with doorways (x-normal walls).
    let wxi = (p.x / WALL_SPACING).round();
    let wx = p.x - wxi * WALL_SPACING;
    if hash2(IVec2::new(wxi as i32, level as i32)) < 0.45 {
        let mut wall = wx.abs() - 1.2;
        let cz = (p.z / 22.0).round();
        let czl = p.z - cz * 22.0;
        if hash3(IVec3::new(wxi as i32, cz as i32, level as i32)) < 0.5 {
            let doorway = sd_box(Vec3::new(wx, fy + 12.0, czl), Vec3::new(4.0, 14.0, 5.0));
            wall = wall.max(-doorway);
        }
        d = d.min(wall);
    }
    // z-normal walls.
    let wzi = (p.z / WALL_SPACING).round();
    let wz = p.z - wzi * WALL_SPACING;
    if hash2(IVec2::new(wzi as i32 + 501, level as i32)) < 0.45 {
        let mut wall = wz.abs() - 1.2;
        let cx = (p.x / 22.0).round();
        let cxl = p.x - cx * 22.0;
        if hash3(IVec3::new(wzi as i32, cx as i32, level as i32 + 77)) < 0.5 {
            let doorway = sd_box(Vec3::new(wz, fy + 12.0, cxl), Vec3::new(4.0, 14.0, 5.0));
            wall = wall.max(-doorway);
        }
        d = d.min(wall);
    }

    // Carve shafts.
    d = d.max(-shaft);

    // Catwalk beams every third level.
    if (level - (level / 3.0).round() * 3.0).abs() < 0.5 {
        let beam = (sxz.y.abs() - 2.2)
            .max((fy + 1.0).abs() - 0.7)
            .max(sxz.length() - (sr + 6.0));
        d = d.min(beam);
    }

    d
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
const POCKET_CELL_Y: f32 = FLOOR_SPACING * 3.0;
const POCKET_SEED: u64 = 0xB10C;

/// Planned features overlapping `[min, max]`: hollow room shells with
/// doorways on floor tops, and vertical light wells cut through several
/// levels. Shared by the GPU density pass and CPU collision.
pub fn pockets_ops(seed: u64, min: Vec3, max: Vec3) -> Vec<CsgOp> {
    let lo = |v: f32, c: f32| ((v - 20.0) / c).floor() as i32;
    let hi = |v: f32, c: f32| ((v + 20.0) / c).floor() as i32;
    let mut out = Vec::new();
    for cy in lo(min.y, POCKET_CELL_Y)..=hi(max.y, POCKET_CELL_Y) {
        for cz in lo(min.z, POCKET_CELL_XZ)..=hi(max.z, POCKET_CELL_XZ) {
            for cx in lo(min.x, POCKET_CELL_XZ)..=hi(max.x, POCKET_CELL_XZ) {
                pocket_cell_ops(seed, cx, cy, cz, &mut out);
            }
        }
    }
    out.retain(|op| op.touches(min, max));
    out
}

fn pocket_cell_ops(seed: u64, cx: i32, cy: i32, cz: i32, out: &mut Vec<CsgOp>) {
    let mut rng = Rng::new(chunk_seed(POCKET_SEED ^ seed, 0x1, IVec3::new(cx, cy, cz)));
    let roll = rng.next_f32();
    if roll > 0.45 {
        return;
    }

    let x = cx as f32 * POCKET_CELL_XZ + 24.0 + rng.next_f32() * (POCKET_CELL_XZ - 48.0);
    let z = cz as f32 * POCKET_CELL_XZ + 24.0 + rng.next_f32() * (POCKET_CELL_XZ - 48.0);
    // Snap to the nearest floor level inside the cell.
    let level = ((cy as f32 + 0.5) * POCKET_CELL_Y / FLOOR_SPACING).round();
    let floor_top = level * FLOOR_SPACING + 1.5;

    if roll < 0.12 {
        // Light well: a square shaft cut through three levels.
        out.push(CsgOp::boxy(
            Vec3::new(x, floor_top + FLOOR_SPACING, z),
            Vec3::new(2.4, FLOOR_SPACING * 1.6, 2.4),
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

    /// A floor point that is solid slab (not inside an opening/shaft).
    fn solid_floor_point() -> Vec3 {
        for i in 0..200 {
            let p = Vec3::new(37.0 + i as f32 * 17.0, 0.0, 91.0 + i as f32 * 5.0);
            if mega_sdf(p) < -1.0 {
                return p;
            }
        }
        panic!("no solid floor found on scan line");
    }

    #[test]
    fn floors_are_solid_and_rooms_are_air() {
        let floor = solid_floor_point();
        assert!(mega_sdf(floor) < -1.0);
        // Mid-room air exists somewhere above the slab plane.
        let mut found_air = false;
        for i in 0..80 {
            let p = Vec3::new(3.0 + i as f32 * 5.0, 20.0, 9.0);
            if mega_sdf(p) > 1.0 {
                found_air = true;
                break;
            }
        }
        assert!(found_air, "no open room space found");
    }

    #[test]
    fn deterministic_and_finite() {
        for i in 0..500 {
            let p = Vec3::new(i as f32 * 13.7, (i % 7) as f32 * 11.0, i as f32 * -7.9);
            let a = mega_sdf(p);
            assert!(a.is_finite());
            assert_eq!(a, mega_sdf(p));
        }
    }

    #[test]
    fn pockets_are_deterministic_and_culled() {
        let min = Vec3::new(-500.0, -100.0, -500.0);
        let max = Vec3::new(500.0, 150.0, 500.0);
        let a = pockets_ops(0, min, max);
        let b = pockets_ops(0, min, max);
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
        let g = mega_gradient(floor + Vec3::Y * 1.4);
        assert!(g.y > 0.6, "gradient near floor top should point up: {g}");
    }
}
