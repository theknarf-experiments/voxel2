//! What a chunk was built from, in one number.
//!
//! An edit to a level makes SOME chunks stale. Which ones was, until now,
//! answered "all of them": the streamed world was dropped and rebuilt, 4.7
//! seconds on the planet, for moving one authored rock. This is the
//! narrower answer — and it is a pull rather than a push, because the
//! interesting edits have no bounding box to push. A generator op
//! confined to a region is confined to a NOISE BAND: an editor cannot say
//! where district three is, but a chunk can be asked whether district
//! three reaches it, with the same predicate the density pass uses.
//!
//! So: fingerprint every resident chunk against the level as it was and
//! as it is, and rebuild the ones whose number moved. A prefab overlapping
//! six chunks rebuilds six chunks, and nobody had to write down that a
//! prefab has an extent.
//!
//! **Conservative in one direction only.** Reporting a change that did not
//! happen costs a rebuild nobody needed. MISSING one leaves a stale chunk
//! — a wrong world that looks right until something else happens to
//! rebuild it — so everything here rounds towards including more.

use std::hash::{Hash, Hasher};
use voxel_core::csg::{Aabb, CsgOp};
use voxel_core::worldop::WorldOp;
use voxel_core::ChunkKey;

/// How far outside its own box a chunk reads.
///
/// The density pass samples 38 points across 32 cells, with sample `i`
/// holding cell corner `i - 2`: two voxels of apron on every side. A
/// fingerprint that ignored it would miss an edit landing in the skirt,
/// and the seam would show at the chunk boundary.
const APRON_VOXELS: f64 = 2.0;

/// The box a chunk's generation actually reads.
pub fn read_box(key: ChunkKey) -> Aabb {
    let pad = key.voxel_size_m() * APRON_VOXELS;
    let lo = key.min_corner_m();
    Aabb::new(lo.as_vec3(), (lo + key.edge_m()).as_vec3()).inflate(pad as f32)
}

/// Everything that decides this chunk's voxels, hashed.
///
/// The seed and the chunk's own identity, the generator ops that can
/// reach it, and the authored ops that touch it. Planning's carved ops are
/// NOT in here: those are invalidated by the layer graph that produced
/// them, and a level edit is not how they change.
pub fn of(key: ChunkKey, seed: u32, ops: &[WorldOp], placed: &[CsgOp]) -> u64 {
    let read = read_box(key);
    let vs = key.voxel_size_m() as f32;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut h);
    key.world.hash(&mut h);
    key.level.hash(&mut h);
    key.pos.to_array().hash(&mut h);

    let reaching = voxel_worldgen::program::ops_reaching(ops, seed, read.min, read.max, vs);
    for (op, reaches) in ops.iter().zip(reaching) {
        if reaches {
            hash_world_op(op, &mut h);
        }
    }
    for op in placed.iter().filter(|op| op.touches(read)) {
        hash_csg_op(op, &mut h);
    }
    h.finish()
}

/// The box holding every op the two lists disagree about, or `None` when
/// they agree.
///
/// A cheap pre-filter for [`of`]: a chunk this box misses sees the same
/// ops in both lists, so its two prints are equal and asking is wasted.
/// The union of BOTH sides, because an op that moved has to rebuild where
/// it was as well as where it went.
pub fn touched(was: &[CsgOp], now: &[CsgOp]) -> Option<Aabb> {
    // Quadratic in the number of authored ops, which is dozens. The
    // alternative is hashing them into a set, and a set of floats is a
    // set of things that compare equal and hash apart.
    let mut out: Option<Aabb> = None;
    for (a, b) in [(was, now), (now, was)] {
        for op in a.iter().filter(|op| !b.contains(op)) {
            out = Some(match out {
                None => op.aabb(),
                Some(all) => all.union(op.aabb()),
            });
        }
    }
    out
}

/// Ops are plain data with floats in them; hash the bits.
///
/// By BITS rather than by value, so two ops that differ only in a sign of
/// zero still hash apart. An edit that changes nothing observable costs a
/// rebuild; one that changes something must never hash the same.
fn hash_world_op(op: &WorldOp, h: &mut impl Hasher) {
    op.kind.hash(h);
    op.flags.hash(h);
    op.material.hash(h);
    op.region.hash(h);
    for p in [op.p0, op.p1, op.p2] {
        for v in p {
            v.to_bits().hash(h);
        }
    }
}

fn hash_csg_op(op: &CsgOp, h: &mut impl Hasher) {
    op.kind.hash(h);
    op.material.hash(h);
    for v in op.center.iter().chain(&op.half) {
        v.to_bits().hash(h);
    }
    op.yaw.to_bits().hash(h);
    op.blend.to_bits().hash(h);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec3;
    use voxel_core::worldop::{WorldOp, WOP_HEIGHT_OFFSET};

    fn key(level: u8, pos: [i32; 3]) -> ChunkKey {
        ChunkKey::in_world(0, level, bevy::math::IVec3::from(pos))
    }

    /// The whole point: an edit somewhere else does not move this number.
    #[test]
    fn an_op_that_cannot_reach_a_chunk_does_not_change_its_print() {
        let ops = voxel_worldgen::program::mega_program();
        let k = key(0, [0, 0, 0]);
        let before = of(k, 0, &ops, &[]);

        // Nudge every op in turn; the ones this chunk cannot see must
        // leave the print alone, and at least one must.
        let read = read_box(k);
        let reaching = voxel_worldgen::program::ops_reaching(
            &ops,
            0,
            read.min,
            read.max,
            k.voxel_size_m() as f32,
        );
        let mut unreached = 0;
        for (i, reaches) in reaching.iter().enumerate() {
            let mut edited = ops.clone();
            edited[i].p0[0] += 13.0;
            let after = of(k, 0, &edited, &[]);
            if *reaches {
                continue;
            }
            unreached += 1;
            assert_eq!(before, after, "op {i} cannot reach this chunk but moved it");
        }
        assert!(unreached > 0, "nothing was out of reach to test");
    }

    /// And an edit it CAN see does move it.
    #[test]
    fn an_op_the_chunk_reads_changes_its_print() {
        let ops = vec![WorldOp::new(WOP_HEIGHT_OFFSET).p0([4.0, 0.0, 0.0, 0.0])];
        let k = key(0, [0, 0, 0]);
        let mut edited = ops.clone();
        edited[0].p0[0] = 5.0;
        assert_ne!(of(k, 0, &ops, &[]), of(k, 0, &edited, &[]));
    }

    /// A placement moves only the chunks it overlaps — the case that
    /// started this.
    #[test]
    fn a_placement_moves_only_the_chunks_it_touches() {
        let ops = Vec::new();
        let rock = CsgOp::boxy(Vec3::new(4.0, 4.0, 4.0), Vec3::splat(2.0), 0.0, 3, false);
        let moved = CsgOp::boxy(Vec3::new(6.0, 4.0, 4.0), Vec3::splat(2.0), 0.0, 3, false);

        let near = key(0, [0, 0, 0]);
        assert_ne!(
            of(near, 0, &ops, std::slice::from_ref(&rock)),
            of(near, 0, &ops, std::slice::from_ref(&moved)),
            "the chunk it sits in must rebuild"
        );

        let far = key(0, [40, 0, 40]);
        assert_eq!(
            of(far, 0, &ops, std::slice::from_ref(&rock)),
            of(far, 0, &ops, std::slice::from_ref(&moved)),
            "a chunk a kilometre away must not"
        );
    }

    /// The pre-filter must cover BOTH ends of a move: where the op went
    /// and where it was, or the chunk it left keeps a copy of it.
    #[test]
    fn the_touched_box_covers_where_an_op_was_and_where_it_went() {
        let was = CsgOp::boxy(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(1.0), 0.0, 3, false);
        let now = CsgOp::boxy(Vec3::new(50.0, 0.0, 0.0), Vec3::splat(1.0), 0.0, 3, false);
        let all = touched(std::slice::from_ref(&was), std::slice::from_ref(&now)).unwrap();
        assert!(all.min.x <= -1.0 && all.max.x >= 51.0, "{all:?}");

        // Unchanged lists touch nothing, whatever is in them.
        let same = [was, now];
        assert!(touched(&same, &same).is_none());
    }

    /// Anything the pre-filter skips must have had equal prints anyway —
    /// the filter is an accelerator, never a second opinion.
    #[test]
    fn the_pre_filter_only_skips_chunks_whose_prints_agree() {
        let ops = Vec::new();
        let was = [CsgOp::boxy(Vec3::ZERO, Vec3::splat(2.0), 0.0, 3, false)];
        let now = [CsgOp::boxy(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::splat(2.0),
            0.0,
            3,
            false,
        )];
        let moved = touched(&was, &now).unwrap();
        for i in -6..6 {
            for j in -6..6 {
                let k = key(0, [i, 0, j]);
                if !read_box(k).touches(moved) {
                    assert_eq!(
                        of(k, 0, &ops, &was),
                        of(k, 0, &ops, &now),
                        "chunk {i},{j} was skipped but its print moved"
                    );
                }
            }
        }
    }

    /// The apron is part of the chunk. An op just outside its box is
    /// still read by its outermost samples, so it belongs to the print.
    #[test]
    fn the_apron_is_inside_the_fingerprint() {
        let k = key(0, [0, 0, 0]);
        let lo = read_box(k).min;
        // Inside the apron on x, in the middle of the chunk on y and z —
        // a level-0 chunk is only a few metres tall, so a probe picked
        // out of the air lands above it.
        let mid = (k.min_corner_m() + k.edge_m() * 0.5).as_vec3();
        let just_outside = Vec3::new(lo.x + 0.01, mid.y, mid.z);
        let op = CsgOp::boxy(just_outside, Vec3::splat(0.05), 0.0, 3, false);
        let mut moved = op;
        moved.material = 9;
        assert!(
            k.min_corner_m().x as f32 > just_outside.x,
            "the probe must sit outside the chunk proper"
        );
        assert_ne!(
            of(k, 0, &[], std::slice::from_ref(&op)),
            of(k, 0, &[], std::slice::from_ref(&moved))
        );
    }
}
