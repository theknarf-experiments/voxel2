//! The "dungeon" structure recipe: a descending chain of carved rooms
//! under a site, reached through a sloped entrance tunnel from the
//! surface. Purely a recipe — levels place dungeons by pairing a
//! `scatter` layer with a `site_recipe` emit (plus a marker so the
//! content is findable through the world-query facade).

use glam::{Vec2, Vec3};
use voxel_core::csg::CsgOp;
use voxel_core::seed::Rng;

use crate::terrain_height;

/// Everything stays within this radius of the site so the stack's
/// element-padding contract (`stack::ELEM_PAD_M`) holds.
pub const MAX_REACH_M: f32 = 48.0;

/// Carve rooms, corridors, and the entrance for one dungeon site.
/// Deterministic from `rng`; geometry conforms to the terrain via the
/// CPU height mirror (the entrance always breaks the surface).
pub fn dungeon_recipe_ops(site: Vec2, rng: &mut Rng, out: &mut Vec<CsgOp>) {
    let surface = terrain_height(site, 1.0);
    let rooms = 3 + rng.next_range(3) as usize;

    // Room chain: each further from the entrance and deeper.
    let mut centers: Vec<Vec3> = Vec::with_capacity(rooms);
    let mut halves: Vec<Vec3> = Vec::with_capacity(rooms);
    let mut angle = rng.next_f32() * std::f32::consts::TAU;
    let mut pos = Vec3::new(site.x, surface - 9.0, site.y);
    for i in 0..rooms {
        let half = Vec3::new(
            3.5 + rng.next_f32() * 3.0,
            2.2 + rng.next_f32() * 1.0,
            3.5 + rng.next_f32() * 3.0,
        );
        centers.push(pos);
        halves.push(half);
        out.push(CsgOp::boxy(pos, half, rng.next_f32() * 0.6, 0, true));
        if i + 1 == rooms {
            break;
        }
        // Next room: bounded walk so the chain stays inside MAX_REACH_M.
        angle += (rng.next_f32() - 0.5) * 1.8;
        let step = 12.0 + rng.next_f32() * 8.0;
        let mut next = pos + Vec3::new(angle.cos() * step, -(3.0 + rng.next_f32() * 4.0), angle.sin() * step);
        let flat = Vec2::new(next.x - site.x, next.z - site.y);
        let max_r = MAX_REACH_M - 8.0;
        if flat.length() > max_r {
            let clamped = flat.normalize() * max_r;
            next.x = site.x + clamped.x;
            next.z = site.y + clamped.y;
        }
        pos = next;
    }

    // Corridors between consecutive rooms: a yawed box spanning the gap.
    for w in centers.windows(2) {
        corridor(w[0], w[1], out);
    }

    // Entrance: a sloped tunnel from just above the surface down into
    // the first room, as overlapping descending cuts (readable stairs).
    let first = centers[0];
    let mouth = Vec3::new(site.x, surface + 1.2, site.y);
    let steps = 6;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let p = mouth.lerp(first, t);
        out.push(CsgOp::boxy(p, Vec3::new(2.2, 2.0, 2.2), 0.0, 0, true));
    }

    // The query contract: every op's AABB must stay within
    // stack::ELEM_PAD_M (64 m) of the site. Worst case today is ~62 m —
    // catch any future size tweak that silently breaks it.
    #[cfg(debug_assertions)]
    for op in out.iter() {
        let (lo, hi) = op.aabb();
        let reach = (site.x - lo.x)
            .max(hi.x - site.x)
            .max(site.y - lo.z)
            .max(hi.z - site.y);
        debug_assert!(reach <= 64.0, "dungeon op reaches {reach:.1} m from site");
    }
}

fn corridor(a: Vec3, b: Vec3, out: &mut Vec<CsgOp>) {
    let flat = Vec2::new(b.x - a.x, b.z - a.z);
    let len = flat.length().max(0.01);
    let yaw = flat.to_angle();
    let mid = (a + b) * 0.5;
    out.push(CsgOp::boxy(
        mid,
        Vec3::new(len * 0.5 + 1.5, (b.y - a.y).abs() * 0.5 + 1.8, 1.6),
        -yaw,
        0,
        true,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::seed::{chunk_seed, Rng};

    fn ops_for(site: Vec2, salt: u64) -> Vec<CsgOp> {
        let mut rng = Rng::new(chunk_seed(salt, 0x0d, glam::IVec3::new(1, 2, 3)));
        let mut out = Vec::new();
        dungeon_recipe_ops(site, &mut rng, &mut out);
        out
    }

    /// AABB overlap with a little slack — "rooms you can walk between".
    fn touches(a: &CsgOp, b: &CsgOp) -> bool {
        let (amin, amax) = a.aabb();
        let (bmin, bmax) = b.aabb();
        amin.cmple(bmax + Vec3::splat(0.5)).all() && bmin.cmple(amax + Vec3::splat(0.5)).all()
    }

    #[test]
    fn dungeon_is_carved_connected_and_reaches_the_surface() {
        // A land site in the reference world.
        let site = Vec2::new(-26800.0, -37900.0);
        let ops = ops_for(site, 7);
        assert!(ops.len() > 8, "too little geometry: {}", ops.len());
        // Everything is a cut (a dungeon adds no matter)...
        for op in &ops {
            assert_eq!(op.kind & 1, 1, "dungeon emitted an additive op");
            // ...within the element-padding contract...
            let p = Vec2::new(op.center[0], op.center[2]);
            assert!(
                p.distance(site) <= MAX_REACH_M + 4.0,
                "op {:.0} m from site",
                p.distance(site)
            );
        }
        // ...forming ONE connected void (flood fill over AABB overlap)...
        let mut joined = vec![false; ops.len()];
        joined[0] = true;
        let mut grew = true;
        while grew {
            grew = false;
            for i in 0..ops.len() {
                if joined[i] {
                    continue;
                }
                if (0..ops.len()).any(|j| joined[j] && touches(&ops[i], &ops[j])) {
                    joined[i] = true;
                    grew = true;
                }
            }
        }
        assert!(
            joined.iter().all(|&j| j),
            "disconnected dungeon: {:?}",
            joined
        );
        // ...that breaks the surface at the entrance...
        let surface = terrain_height(site, 1.0);
        assert!(
            ops.iter().any(|op| {
                let (min, max) = op.aabb();
                min.y < surface && max.y > surface
            }),
            "no cut crosses the surface — dungeon is sealed"
        );
        // ...with rooms genuinely underground.
        assert!(
            ops.iter()
                .any(|op| op.center[1] < surface - 15.0),
            "no deep rooms"
        );
    }

    #[test]
    fn dungeon_is_deterministic_and_seed_sensitive() {
        let site = Vec2::new(-26800.0, -37900.0);
        assert_eq!(ops_for(site, 7), ops_for(site, 7));
        assert_ne!(ops_for(site, 7), ops_for(site, 8));
    }
}
