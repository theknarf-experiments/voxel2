//! The "ruin" structure recipe (broken wall ring, tower stubs, fallen
//! blocks), placed by the stack's `site_recipe` emit. Geometry conforms
//! to the terrain via the CPU height mirror, so walls sit on slopes.

use glam::{Vec2, Vec3};
use voxel_core::csg::CsgOp;
use voxel_core::seed::Rng;

use crate::terrain_height;

/// Stone material id for carved/added geometry.
pub const MAT_STONE: u32 = 3;

/// The ruin structure recipe: geometry for one site, from any rng stream.
/// Largest reach from the site: ring radius (17) + tower radius — well
/// under the stack's element-padding contract.
pub fn ruin_recipe_ops(center_xz: Vec2, rng: &mut Rng, out: &mut Vec<CsgOp>) {
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
    use voxel_core::seed::{chunk_seed, Rng};

    #[test]
    fn recipe_is_deterministic_and_terrain_seated() {
        let site = Vec2::new(-26800.0, -37900.0);
        let ops = |salt: u64| {
            let mut rng = Rng::new(chunk_seed(salt, 0x77, glam::IVec3::new(4, 0, 2)));
            let mut out = Vec::new();
            ruin_recipe_ops(site, &mut rng, &mut out);
            out
        };
        let a = ops(1);
        assert_eq!(a, ops(1));
        assert_ne!(a, ops(2));
        assert!(!a.is_empty());
        // Walls seat near the terrain around the site.
        for op in &a {
            let ground = terrain_height(Vec2::new(op.center[0], op.center[2]), 1.0);
            assert!((op.center[1] - ground).abs() < 14.0, "op far from ground");
        }
    }
}
