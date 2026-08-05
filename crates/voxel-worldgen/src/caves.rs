//! Perlin-worm caves: deterministic noise-steered tunnels emitted as
//! sphere-cut CSG chains (VoxelPlugin's worm utility as a planning
//! source). One 256 m planning cell may host a worm; the walk is a
//! cell-seeded RNG stream, so every chunk that queries the cell sees the
//! identical tunnel.

use glam::{IVec3, Vec2, Vec3};
use voxel_core::csg::CsgOp;
use voxel_core::seed::{chunk_seed, Rng};

use crate::terrain_height;

const CELL_M: f32 = 256.0;
const CAVE_SEED: u64 = 0xCAFE5;
/// Steps per worm; with ~0.9 radius spacing this bounds tunnel length.
const STEPS: u32 = 70;
/// Conservative worm reach for the cell scan. Must dominate the true
/// worst case (STEPS × max radius × 0.9 step + mouth offset + max
/// radius ≈ 290 m for the default radius range) — a chunk just past an
/// undersized reach misses a worm's tail while its neighbor carves it:
/// an asymmetric-culling crack along their shared face.
const REACH_M: f32 = 340.0;

/// All cave ops from worms whose cells could reach the box `[min, max]`.
pub fn caves_ops(seed: u64, chance: f32, radius: [f32; 2], min: Vec3, max: Vec3) -> Vec<CsgOp> {
    let lo_x = ((min.x - REACH_M) / CELL_M).floor() as i32;
    let hi_x = ((max.x + REACH_M) / CELL_M).floor() as i32;
    let lo_z = ((min.z - REACH_M) / CELL_M).floor() as i32;
    let hi_z = ((max.z + REACH_M) / CELL_M).floor() as i32;
    let mut out = Vec::new();
    for cz in lo_z..=hi_z {
        for cx in lo_x..=hi_x {
            worm_ops(seed, chance, radius, cx, cz, &mut out);
        }
    }
    out.retain(|op| op.touches(min, max));
    out
}

/// The mouth of a cell's worm, if any (scouting/debug).
pub fn cave_mouth(seed: u64, chance: f32, cx: i32, cz: i32) -> Option<Vec3> {
    let mut rng = Rng::new(chunk_seed(CAVE_SEED ^ seed, 0x77, IVec3::new(cx, 0, cz)));
    if rng.next_f32() > chance {
        return None;
    }
    let x = cx as f32 * CELL_M + 24.0 + rng.next_f32() * (CELL_M - 48.0);
    let z = cz as f32 * CELL_M + 24.0 + rng.next_f32() * (CELL_M - 48.0);
    let ground = terrain_height(Vec2::new(x, z), 1.0);
    // Mouths on dry, reasonably flat-to-hilly ground only.
    if !(6.0..500.0).contains(&ground) {
        return None;
    }
    Some(Vec3::new(x, ground, z))
}

fn worm_ops(seed: u64, chance: f32, radius: [f32; 2], cx: i32, cz: i32, out: &mut Vec<CsgOp>) {
    let Some(mouth) = cave_mouth(seed, chance, cx, cz) else {
        return;
    };
    // Fresh stream for the walk so mouth selection stays stable if the
    // walk parameters evolve.
    let mut rng = Rng::new(chunk_seed(CAVE_SEED ^ seed, 0x78, IVec3::new(cx, 0, cz)));
    let base_r = radius[0] + rng.next_f32() * (radius[1] - radius[0]);
    let mut yaw = rng.next_f32() * std::f32::consts::TAU;
    let mut pitch = -0.4 - rng.next_f32() * 0.2; // dive from the mouth
    let mut pos = mouth + Vec3::new(0.0, base_r * 0.6, 0.0);
    for _ in 0..STEPS {
        let r = base_r * (0.8 + 0.45 * rng.next_f32());
        out.push(CsgOp::sphere(pos, r, 0, true));
        // Noise-steered heading; pitch relaxes toward horizontal so worms
        // level out into galleries after the entrance dive.
        yaw += (rng.next_f32() - 0.5) * 0.55;
        pitch += (rng.next_f32() - 0.5) * 0.3 - pitch * 0.15;
        pitch = pitch.clamp(-0.55, 0.25);
        // Stay under the surface (worms that graze it open skylights,
        // which is fine occasionally — but not for whole galleries).
        let ceiling = terrain_height(Vec2::new(pos.x, pos.z), 1.0) - r * 2.4;
        if pos.y > ceiling {
            pitch = (pitch - 0.2).min(-0.35);
        }
        let dir = Vec3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        );
        pos += dir * (r * 0.9);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_culled() {
        let min = Vec3::new(-2048.0, -200.0, -2048.0);
        let max = Vec3::new(2048.0, 400.0, 2048.0);
        let a = caves_ops(7, 0.6, [2.0, 3.5], min, max);
        let b = caves_ops(7, 0.6, [2.0, 3.5], min, max);
        assert_eq!(a, b);
        for op in &a {
            assert!(op.touches(min, max));
            assert_eq!(op.kind, voxel_core::csg::CSG_KIND_SPHERE_CUT);
        }
        // A sub-box query returns exactly the sub-filtered set: chunks
        // agree on the worm regardless of which cell scan found it.
        let smin = Vec3::new(-512.0, -200.0, -512.0);
        let smax = Vec3::new(512.0, 400.0, 512.0);
        let sub = caves_ops(7, 0.6, [2.0, 3.5], smin, smax);
        let expect: Vec<_> = a
            .iter()
            .filter(|op| op.touches(smin, smax))
            .copied()
            .collect();
        assert_eq!(sub, expect);
    }
}
