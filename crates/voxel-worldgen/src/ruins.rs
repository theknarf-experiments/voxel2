//! Ruins planning: deterministic scattered ruin sites (broken wall rings,
//! tower stubs, fallen blocks) emitted as CSG ops for the GPU density pass.
//!
//! Structured as a self-contained 256 m planning cell for now; this slots
//! into a `voxel-layers` Layer the moment ruins need cross-cell context
//! (roads between sites, etc.). Placement conforms to the terrain via the
//! CPU height mirror, so walls sit on slopes correctly.

use glam::{IVec3, Vec2, Vec3};
use voxel_core::csg::CsgOp;
use voxel_core::seed::{chunk_seed, Rng};

use crate::{terrain_height, terrain_up};

const CELL_M: f32 = 256.0;
const RUIN_SEED: u64 = 0x8115;
/// Stone material id for carved/added geometry.
pub const MAT_STONE: u32 = 3;

/// All ops from ruin cells overlapping the box `[min, max]` (world meters),
/// filtered to ops that actually touch it.
pub fn ruins_ops(min: Vec3, max: Vec3) -> Vec<CsgOp> {
    let lo_x = ((min.x - 40.0) / CELL_M).floor() as i32;
    let hi_x = ((max.x + 40.0) / CELL_M).floor() as i32;
    let lo_z = ((min.z - 40.0) / CELL_M).floor() as i32;
    let hi_z = ((max.z + 40.0) / CELL_M).floor() as i32;
    let mut out = Vec::new();
    for cz in lo_z..=hi_z {
        for cx in lo_x..=hi_x {
            cell_ops(cx, cz, &mut out);
        }
    }
    out.retain(|op| op.touches(min, max));
    out
}

fn cell_ops(cx: i32, cz: i32, out: &mut Vec<CsgOp>) {
    let mut rng = Rng::new(chunk_seed(RUIN_SEED, 0x101, IVec3::new(cx, 0, cz)));
    if rng.next_f32() > 0.32 {
        return; // most cells have no ruin
    }
    let center_xz = Vec2::new(
        cx as f32 * CELL_M + 32.0 + rng.next_f32() * (CELL_M - 64.0),
        cz as f32 * CELL_M + 32.0 + rng.next_f32() * (CELL_M - 64.0),
    );
    let ground = terrain_height(center_xz, 1.0);
    // Ruins stand on gentle inland ground.
    if !(8.0..280.0).contains(&ground) || terrain_up(center_xz, 1.0) < 0.88 {
        return;
    }

    let radius = 8.0 + rng.next_f32() * 9.0;
    let segments = 6 + rng.next_range(4);
    let base_angle = rng.next_f32() * std::f32::consts::TAU;

    // Broken ring wall: some segments missing, heights varied.
    for i in 0..segments {
        if rng.next_f32() < 0.28 {
            continue; // collapsed gap
        }
        let angle = base_angle + std::f32::consts::TAU * i as f32 / segments as f32;
        let pos_xz = center_xz + Vec2::new(angle.cos(), angle.sin()) * radius;
        let y = terrain_height(pos_xz, 1.0);
        let height = 1.2 + rng.next_f32() * 2.6;
        let seg_len = radius * std::f32::consts::PI / segments as f32 * 0.95;
        out.push(CsgOp::boxy(
            Vec3::new(pos_xz.x, y + height * 0.5 - 0.6, pos_xz.y),
            Vec3::new(seg_len, height * 0.5 + 0.6, 0.55),
            angle + std::f32::consts::FRAC_PI_2,
            MAT_STONE,
            false,
        ));
    }

    // Tower stubs on the ring.
    for _ in 0..(1 + rng.next_range(2)) {
        let angle = rng.next_f32() * std::f32::consts::TAU;
        let pos_xz = center_xz + Vec2::new(angle.cos(), angle.sin()) * radius;
        let y = terrain_height(pos_xz, 1.0);
        let tower_r = 1.8 + rng.next_f32() * 1.4;
        let half_h = 2.5 + rng.next_f32() * 4.5;
        out.push(CsgOp::cylinder(
            Vec3::new(pos_xz.x, y + half_h - 1.0, pos_xz.y),
            tower_r,
            half_h,
            MAT_STONE,
            false,
        ));
        // Hollow the stub so tall ones read as broken shells.
        if half_h > 4.5 {
            out.push(CsgOp::cylinder(
                Vec3::new(pos_xz.x, y + half_h + 1.5, pos_xz.y),
                tower_r - 0.8,
                half_h,
                MAT_STONE,
                true,
            ));
        }
    }

    // Fallen blocks scattered inside.
    for _ in 0..(2 + rng.next_range(3)) {
        let a = rng.next_f32() * std::f32::consts::TAU;
        let r = rng.next_f32() * radius * 0.8;
        let pos_xz = center_xz + Vec2::new(a.cos(), a.sin()) * r;
        let y = terrain_height(pos_xz, 1.0);
        let s = 0.5 + rng.next_f32() * 0.9;
        out.push(CsgOp::boxy(
            Vec3::new(pos_xz.x, y + s * 0.4, pos_xz.y),
            Vec3::splat(s),
            rng.next_f32() * std::f32::consts::TAU,
            MAT_STONE,
            false,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_culled() {
        let min = Vec3::new(-2048.0, -100.0, -2048.0);
        let max = Vec3::new(2048.0, 400.0, 2048.0);
        let a = ruins_ops(min, max);
        let b = ruins_ops(min, max);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x, y);
        }
        // Everything returned touches the query box.
        for op in &a {
            assert!(op.touches(min, max));
        }
    }

    #[test]
    fn some_region_has_ruins() {
        // Over a large area at least one ruin site must exist.
        let ops = ruins_ops(
            Vec3::new(-8192.0, -500.0, -8192.0),
            Vec3::new(8192.0, 600.0, 8192.0),
        );
        assert!(!ops.is_empty(), "no ruins in 16 km x 16 km");
    }
}
