//! CSG operations: the compact IR that CPU planning layers (LayerProcGen)
//! hand to the GPU density shaders. Layout is shared bit-for-bit with the
//! WGSL `CsgOp` struct.

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};

use crate::interval::Interval;

pub const CSG_KIND_BOX_ADD: u32 = 0;
pub const CSG_KIND_BOX_CUT: u32 = 1;
pub const CSG_KIND_CYLINDER_ADD: u32 = 2;
pub const CSG_KIND_CYLINDER_CUT: u32 = 3;
pub const CSG_KIND_SPHERE_ADD: u32 = 4;
pub const CSG_KIND_SPHERE_CUT: u32 = 5;
pub const CSG_KIND_CAPSULE_ADD: u32 = 6;
pub const CSG_KIND_CAPSULE_CUT: u32 = 7;

/// Candidate ops the per-cell index examined, summed over cells — what
/// explains [`ChunkOps::build`]'s cost, since every candidate costs an
/// interval evaluation whether it survives or not.
pub static CELL_TESTED: crate::Stage = crate::Stage::new();
/// Candidates that survived, summed over cells. The ratio against
/// `CELL_TESTED` is what the pruning is worth.
pub static CELL_KEPT: crate::Stage = crate::Stage::new();

/// Residency passes run, and chunks committed by them. A cold start with
/// a stationary camera should need ONE: the LOD field is a pure function
/// of (config, anchor, key) and neither moves.
pub static LOD_PASSES: crate::Stage = crate::Stage::new();
pub static LOD_COMMITTED: crate::Stage = crate::Stage::new();

/// Blocking inside the ops path: waiting for this chunk's planning to be
/// resident, against actually answering the query. They are charged apart
/// because a stage that is mostly SLEEP wants a different fix from one
/// that is mostly work, and summed thread time cannot tell them apart.
pub static OPS_WAIT: crate::Stage = crate::Stage::new();
pub static OPS_ANSWER: crate::Stage = crate::Stage::new();
/// Chunks that reached the wait at all.
pub static OPS_WAITED: crate::Stage = crate::Stage::new();
/// The two halves of the readiness CHECK that guards the wait.
pub static OPS_IS_IDLE: crate::Stage = crate::Stage::new();
pub static OPS_COVERED: crate::Stage = crate::Stage::new();

/// Chunks whose ops were asked for, and how many of those a LANDMARK had
/// refined. If the second is zero the landmark gate is not firing at all.
pub static OPS_CHUNKS: crate::Stage = crate::Stage::new();
pub static OPS_BIASED: crate::Stage = crate::Stage::new();
/// Times a gated emitter was skipped because the chunk was fine only
/// because of a landmark.
pub static GATE_SKIPPED: crate::Stage = crate::Stage::new();

/// `patches_in` split: taking the layer view (locks, slot lookups, an
/// `Arc` clone each) against walking the ops it found.
pub static QUERY_VIEW: crate::Stage = crate::Stage::new();
pub static QUERY_SCAN: crate::Stage = crate::Stage::new();
/// Queries that returned no ops at all.
pub static QUERY_EMPTY: crate::Stage = crate::Stage::new();

/// [`ChunkOps::build`] split in two: bounding the terrain and pruning the
/// coarse boxes, against binning and pruning the 512 fine cells.
pub static INDEX_COARSE: crate::Stage = crate::Stage::new();
pub static INDEX_FINE: crate::Stage = crate::Stage::new();

/// Ops the emit index walked to answer a chunk's query, and ops it
/// returned. Their ratio is what the index is worth.
pub static QUERY_WALKED: crate::Stage = crate::Stage::new();
pub static QUERY_RETURNED: crate::Stage = crate::Stage::new();

/// One CSG operation, 48 bytes, `#[repr(C)]` — uploaded verbatim.
///
/// Boxes: `center` + `half` extents, rotated `yaw` radians about Y.
/// Cylinders: `center` (mid-height), `half.x` = radius, `half.y` = half
/// height. Capsules: `center` is the base, `half` the AXIS to the tip,
/// `yaw` the base radius and `aux.x` the tip radius — see
/// [`CsgOp::capsule`]. `blend` is the smooth-min radius in meters
/// (0 = hard).
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct CsgOp {
    pub center: [f32; 3],
    pub kind: u32,
    pub half: [f32; 3],
    pub material: u32,
    pub yaw: f32,
    pub blend: f32,
    /// Kind-specific extra, zero for the shapes that need none. It was
    /// padding to 48 bytes and still is for every other kind; the
    /// capsule spends `aux.x` on its tip radius rather than growing the
    /// struct, because 48 bytes is a layout twin in three places.
    pub aux: [f32; 2],
}

impl CsgOp {
    pub fn boxy(center: Vec3, half: Vec3, yaw: f32, material: u32, cut: bool) -> Self {
        Self::of(CSG_KIND_BOX_ADD, center, half, material, cut).yaw(yaw)
    }

    pub fn cylinder(center: Vec3, radius: f32, half_height: f32, material: u32, cut: bool) -> Self {
        let half = Vec3::new(radius, half_height, radius);
        Self::of(CSG_KIND_CYLINDER_ADD, center, half, material, cut)
    }

    /// Sphere: `half.x` = radius (spheres ignore yaw).
    pub fn sphere(center: Vec3, radius: f32, material: u32, cut: bool) -> Self {
        Self::of(
            CSG_KIND_SPHERE_ADD,
            center,
            Vec3::splat(radius),
            material,
            cut,
        )
    }

    /// A tapered capsule from `a` (radius `r_a`) to `b` (radius `r_b`).
    ///
    /// The one primitive that points anywhere. Every other kind is
    /// yaw-only, which is fine for a wall or a shaft and useless for a
    /// branch, a root or a tendril — those go where they grow. So the
    /// axis is stored as a VECTOR rather than an orientation, and the
    /// capsule is the one kind `op_sdf` answers before rotating anything.
    ///
    /// One limb, one op: a skeleton of a few dozen limbs costs a few
    /// dozen ops, which is what makes organic shapes affordable in a
    /// field that is evaluated per sample.
    pub fn capsule(a: Vec3, b: Vec3, r_a: f32, r_b: f32, material: u32, cut: bool) -> Self {
        Self {
            center: a.to_array(),
            kind: CSG_KIND_CAPSULE_ADD + u32::from(cut),
            half: (b - a).to_array(),
            material,
            yaw: r_a,
            blend: 0.0,
            aux: [r_b, 0.0],
        }
    }

    /// The shared body of the three constructors. `add` is the ADD kind;
    /// every cut kind is its successor, which is also what `apply` and the
    /// WGSL twin rely on when they test `kind & 1`.
    fn of(add: u32, center: Vec3, half: Vec3, material: u32, cut: bool) -> Self {
        Self {
            center: center.to_array(),
            kind: add + u32::from(cut),
            half: half.to_array(),
            material,
            yaw: 0.0,
            blend: 0.0,
            aux: [0.0; 2],
        }
    }

    fn yaw(mut self, yaw: f32) -> Self {
        self.yaw = yaw;
        self
    }

    /// This op moved onto a placement: scaled about the origin, rotated,
    /// then translated.
    ///
    /// What makes a grown shape affordable in bulk. Space colonization is
    /// O(nodes x attractors) and a forest places hundreds of trees in a
    /// chunk, so the skeleton is grown ONCE in unit space and every
    /// instance is this — a handful of flops per op instead of a growth.
    /// The prop meshes have always done it this way (one mesh per variant,
    /// instanced a million times); this is the same trick for the field.
    ///
    /// Capsules and spheres transform exactly: an axis is a vector, so an
    /// arbitrary rotation is just the vector rotated. Boxes and cylinders
    /// carry a yaw and nothing else, so a rotation with any tilt in it
    /// keeps them upright and takes only the yaw — the shape has no way to
    /// express the rest. Documented rather than asserted because the tilt
    /// a population applies is a few degrees, and refusing it would refuse
    /// the whole placement over a difference nothing can see.
    pub fn transformed(&self, translation: Vec3, rotation: glam::Quat, scale: f32) -> Self {
        let mut out = *self;
        let center = Vec3::from(self.center);
        out.center = (translation + rotation * (center * scale)).to_array();
        if self.kind >= CSG_KIND_CAPSULE_ADD {
            // `half` is the axis to the tip and `yaw`/`aux.x` the two
            // radii — every one of them a length or a vector, so the
            // whole limb scales and rotates without a special case.
            out.half = (rotation * (Vec3::from(self.half) * scale)).to_array();
            out.yaw = self.yaw * scale;
            out.aux[0] = self.aux[0] * scale;
        } else {
            out.half = (Vec3::from(self.half) * scale).to_array();
            out.yaw = self.yaw + Self::yaw_of(rotation);
        }
        out.blend = self.blend * scale;
        out
    }

    /// The yaw a rotation carries, for the kinds that can hold nothing
    /// else. Degenerate when the rotation points Z straight up or down,
    /// which no placement tilt of a few degrees can reach.
    fn yaw_of(q: glam::Quat) -> f32 {
        let forward = q * Vec3::Z;
        if forward.x.abs() + forward.z.abs() < 1.0e-6 {
            return 0.0;
        }
        forward.x.atan2(forward.z)
    }

    /// Signed distance to this op's primitive (mirrors the WGSL `op_sdf`).
    pub fn sdf(&self, p: Vec3) -> f32 {
        // BEFORE the yaw rotation: a capsule has no yaw — that field is
        // its base radius, and rotating by it would bend every limb.
        if self.kind >= CSG_KIND_CAPSULE_ADD {
            let pa = p - Vec3::from(self.center);
            let ba = Vec3::from(self.half);
            let t = (pa.dot(ba) / ba.dot(ba).max(1.0e-8)).clamp(0.0, 1.0);
            return (pa - ba * t).length() - (self.yaw + (self.aux[0] - self.yaw) * t);
        }
        let mut q = p - Vec3::from(self.center);
        let (s, c) = (-self.yaw).sin_cos();
        q = Vec3::new(q.x * c - q.z * s, q.y, q.x * s + q.z * c);
        let h = Vec3::from(self.half);
        if self.kind >= 4 {
            q.length() - h.x
        } else if self.kind < 2 {
            let a = q.abs() - h;
            a.max(Vec3::ZERO).length() + a.x.max(a.y.max(a.z)).min(0.0)
        } else {
            let dr = (q.x * q.x + q.z * q.z).sqrt() - h.x;
            let dy = q.y.abs() - h.y;
            glam::Vec2::new(dr.max(0.0), dy.max(0.0)).length() + dr.max(dy).min(0.0)
        }
    }

    /// Bound of [`Self::sdf`] over an axis-aligned box.
    ///
    /// Every primitive here is a TRUE distance field, so its gradient
    /// has magnitude at most one and the whole interval follows from a
    /// single evaluation: within `r` metres of the centre the value can
    /// have moved by at most `r`. That is what makes pruning cheap
    /// enough to do per sub-cell — one `sdf` call, not a per-kind
    /// interval arithmetic twin that would have to stay in sync.
    ///
    /// The tapered capsule is the exception and is handled: its
    /// round-cone field changes by `|r_a - r_b|` more than distance does
    /// over its own length, so its Lipschitz bound is inflated by
    /// exactly that ratio. Understating it would drop an op that matters,
    /// which is a hole in the world.
    pub fn sdf_range(&self, min: Vec3, max: Vec3) -> Interval {
        let c = (min + max) * 0.5;
        let r = (max - min).length() * 0.5;
        let lip = if self.kind >= CSG_KIND_CAPSULE_ADD {
            let len = Vec3::from(self.half).length().max(1.0e-6);
            1.0 + (self.aux[0] - self.yaw).abs() / len
        } else {
            1.0
        };
        let d = self.sdf(c);
        Interval::new(d - lip * r, d + lip * r)
    }

    /// Fold this op into a scene distance (ignores smooth blend — CPU
    /// collision does not need it).
    pub fn apply(&self, d: f32, p: Vec3) -> f32 {
        let od = self.sdf(p);
        if self.kind & 1 == 0 {
            d.min(od)
        } else {
            d.max(-od)
        }
    }

    /// Conservative world-space AABB (yaw-safe: uses the diagonal, and
    /// a smooth blend reaches past the shape by its own radius).
    pub fn aabb(&self) -> Aabb {
        let h = Vec3::from(self.half);
        // A capsule's `half` is an axis, not an extent: the box is the
        // union of a ball at each end. Reading it as an extent would
        // give a box centred on the BASE that misses most of the limb.
        if self.kind >= CSG_KIND_CAPSULE_ADD {
            let a = Vec3::from(self.center);
            let b = a + h;
            let (ra, rb) = (self.yaw + self.blend, self.aux[0] + self.blend);
            return Aabb::new(
                (a - Vec3::splat(ra)).min(b - Vec3::splat(rb)),
                (a + Vec3::splat(ra)).max(b + Vec3::splat(rb)),
            );
        }
        let r = (h.x * h.x + h.z * h.z).sqrt().max(h.x.max(h.z));
        Aabb::around(
            Vec3::from(self.center),
            Vec3::new(r, h.y, r) + Vec3::splat(self.blend),
        )
    }

    /// Does this op affect `box`?
    pub fn touches(&self, r#box: Aabb) -> bool {
        self.aabb().touches(r#box)
    }
}

/// What interval evaluation proved about one op over a box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    /// Cannot change the result anywhere in the box — drop it.
    Skip,
    /// Decides the result everywhere in the box — every op BEFORE it is
    /// dead, whatever they were.
    Replaces,
    /// Undecided: it matters somewhere in here.
    Both,
}

/// The ops in a union/cut chain that can change the result over a box.
///
/// This is Keeter's tape pruning (MPR, Algorithms 1 and 2) specialised to
/// the chain `apply_csg` actually runs: a forward interval pass records a
/// choice per op, and a backward pass keeps only what is live. A chain is
/// linear rather than a DAG, so "live" collapses to "after the last op
/// that decides the result", and no register liveness is needed.
///
/// `start` is the interval of the distance the chain STARTS from — the
/// generator's own field over this box. It is what makes pruning work at
/// all: without a finite bound on what is already there, no op can be
/// proved irrelevant.
///
/// Sound, never exact: it can keep an op that turns out not to matter,
/// and must never drop one that does. Every bound below is conservative
/// in that direction.
pub fn prune_chain(ops: &[CsgOp], start: Interval, min: Vec3, max: Vec3) -> Vec<u32> {
    let mut choices: Vec<Choice> = Vec::with_capacity(ops.len());
    let mut d = start;
    for op in ops {
        let od = op.sdf_range(min, max);
        let choice = if op.kind & 1 == 0 {
            // Union: `d = min(d, od)`.
            if od.lo > d.hi {
                Choice::Skip // always further than what we have
            } else if od.hi < d.lo {
                Choice::Replaces // always nearer: it IS the result
            } else {
                Choice::Both
            }
        } else {
            // Cut: `d = max(d, -od)`. Irrelevant when `-od <= d` over the
            // whole box, i.e. when `-od.lo <= d.lo`.
            if -od.lo <= d.lo {
                Choice::Skip
            } else if -od.hi >= d.hi {
                Choice::Replaces
            } else {
                Choice::Both
            }
        };
        d = match choice {
            Choice::Skip => d,
            Choice::Replaces if op.kind & 1 == 0 => od,
            Choice::Replaces => Interval::new(-od.hi, -od.lo),
            Choice::Both if op.kind & 1 == 0 => d.min(od),
            Choice::Both => d.max(Interval::new(-od.hi, -od.lo)),
        };
        choices.push(choice);
    }
    // Backward: everything before the last decider is dead.
    let from = choices
        .iter()
        .rposition(|c| *c == Choice::Replaces)
        .unwrap_or(0);
    (from..ops.len())
        .filter(|i| choices[*i] != Choice::Skip)
        .map(|i| i as u32)
        .collect()
}

/// Prune a chain without knowing what it starts from.
///
/// [`prune_chain`] needs a bound on the incoming distance — the
/// generator's own field — and only the side that owns the generator has
/// one. The renderer does not, and plumbing terrain intervals down to it
/// would be three layers of new payload.
///
/// DOMINATION needs no such bound. If one add is always nearer than
/// another over this box, the farther one cannot change `min` whatever
/// else is in the chain, terrain included. That is the case that matters:
/// a cell inside one tree is dominated by that tree's own limbs, and
/// every other tree in the chunk falls out.
///
/// What it deliberately cannot do, and why:
/// - No `Replaces`. Deciding an op overrides everything before it would
///   override the TERRAIN, which is exactly the thing not known here.
/// - Cuts are always kept. A cut is irrelevant when the terrain is
///   already further out than it reaches, and how deep the terrain sits
///   is the unknown. Cheap to keep: limbs are adds.
pub fn prune_dominated(ops: &[CsgOp], min: Vec3, max: Vec3) -> Vec<u32> {
    // The running minimum over ADDS ONLY. It over-estimates the true
    // distance (the terrain can only pull it down), and over-estimating
    // is the safe direction for a skip test: `od.lo > d.hi` still
    // implies `od.lo` is above the true distance.
    let mut d = Interval::point(f32::MAX);
    let mut keep = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        if op.kind & 1 != 0 {
            keep.push(i as u32);
            continue;
        }
        let od = op.sdf_range(min, max);
        if od.lo > d.hi {
            continue; // another add is always nearer
        }
        d = d.min(od);
        keep.push(i as u32);
    }
    keep
}

/// Cells per axis a chunk's ops are indexed by.
///
/// MEASURED, on 200 limbs in a 12.8 m chunk — a forest chunk's real
/// load — as ops kept per sample: 4 -> 36% (2.8x), 8 -> 14% (7.2x).
/// Cell size is the whole lever, because what survives is whatever
/// reaches within the terrain's own spread across the cell; halving the
/// cell halves that reach.
///
/// 8 is not free: it is 512 interval evaluations per chunk instead of
/// 64, and the index it writes is larger than the ops it indexes
/// (~57 KB against 9.6 KB for a chunk this dense). Both land on a
/// worker thread and a once-per-generation upload, against a saving
/// paid back on every one of a chunk's ~55k density samples, which is
/// why the lopsided trade is worth making twice.
pub const CSG_CELLS: usize = 8;

/// Fewer ops than this and the index is not built. Walking eight ops is
/// cheaper than the indirection to find out you have to walk six.
const INDEX_FROM: usize = 24;

/// One chunk's ops, and a per-cell index into them.
///
/// `cells` is `[table][runs]`: `2 · CSG_CELLS³` u32 of `(offset, count)`
/// addressed by cell, followed by the index runs they point at, all
/// indices being into `ops`. Empty means "no index" — the reader walks
/// every op, which is what a chunk with a handful of them should do.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkOps {
    pub ops: Vec<CsgOp>,
    pub cells: Vec<u32>,
}

impl ChunkOps {
    /// Index `ops` over `min..max`, pruning each cell against the field
    /// the chain starts from.
    ///
    /// `terrain` bounds the generator over a box, or returns `None` where
    /// it cannot — an unbounded answer keeps every op for that cell,
    /// which is the safe direction. This is the whole reason indexing
    /// lives on the engine side: without that bound only domination is
    /// provable, and domination alone measured 2x where this measures
    /// far better.
    ///
    /// `apron` widens every cell by what the density pass can sample
    /// outside it. Understate it and a sample reads a list that was
    /// pruned for somewhere it is not.
    ///
    /// `band` is the half-width the stored field is clamped to, in the
    /// same meters — [`crate::SDF_BAND`] voxels. It is what makes the
    /// index worth building at forest density: an op whose box is
    /// further than `band` from a cell cannot change ONE stored value in
    /// it, so the question becomes a box test and every op can be pushed
    /// into the few cells it reaches instead of being asked about by all
    /// 512. The answer this index gives is therefore equal to the flat
    /// list's AFTER CLAMPING, not before, which is the only sense in
    /// which the density pass ever reads it.
    pub fn build(
        ops: Vec<CsgOp>,
        min: Vec3,
        max: Vec3,
        apron: f32,
        band: f32,
        terrain: impl Fn(Vec3, Vec3) -> Option<Interval>,
    ) -> Self {
        if ops.len() < INDEX_FROM {
            return Self {
                ops,
                cells: Vec::new(),
            };
        }
        const N: usize = CSG_CELLS;
        let table_len = 2 * N * N * N;
        let mut cells = vec![0u32; table_len];
        let step = (max - min) / N as f32;
        // Boxes computed ONCE per op rather than once per op per cell.
        // The fine loop below asks 512 times whether each surviving op
        // reaches its cell, and `aabb()` is a square root and a dozen
        // min/max — in a forest chunk carrying a few thousand ops that
        // was millions of calls per chunk, and the largest single cost
        // in a cold start.
        let boxes: Vec<Aabb> = ops.iter().map(CsgOp::aabb).collect();
        // Terrain intervals on a COARSER grid than the op cells.
        //
        // `terrain` walks the whole generator program; at one call per op
        // cell that is 512 program evaluations per chunk on top of the
        // ones admission control already pays, and it stalled the
        // pipeline outright — 2484 chunks past their create timeout.
        // A coarse box's interval BOUNDS every fine box inside it, so
        // reusing it is sound and merely looser, and 8 calls buy most of
        // what 512 did.
        const T: usize = 4;
        let tstep = (max - min) / T as f32;
        let mut coarse: Vec<Option<(Option<Interval>, Vec<u32>)>> = vec![None; T * T * T];
        let coarse_started = std::time::Instant::now();
        for (i, slot) in coarse.iter_mut().enumerate() {
            let c = Vec3::new((i % T) as f32, ((i / T) % T) as f32, (i / (T * T)) as f32);
            let lo = min + tstep * c - Vec3::splat(apron);
            let hi = min + tstep * (c + Vec3::ONE) + Vec3::splat(apron);
            let start = terrain(lo, hi);
            // Prune the coarse box FIRST and refine only what survives —
            // the recursion the paper is built on. A fine cell can only
            // need ops its enclosing box needed, so this is exact, and it
            // turns 512xN work into 8xN + 512x(survivors).
            let keep: Vec<u32> = match start {
                Some(i) => prune_chain(&ops, i, lo, hi),
                None => (0..ops.len() as u32).collect(),
            };
            *slot = Some((start, keep));
        }
        // Fine cells filled by BINNING, not by asking every cell about
        // every op.
        //
        // Why a box test is enough: a stored voxel is clamped to
        // +/-`band`. An ADD whose box is further than that can only ever
        // be further than the band, and `min` keeps whatever was already
        // out there — both clamp to +band. A CUT that far leaves `max`
        // at what it had, or drags it somewhere between that and -band —
        // both clamp to -band. So neither can change a value, and which
        // cells an op can reach is a question about boxes.
        //
        // The field test stays on top of it for adds, because the
        // terrain's own upper bound over the cell is often tighter than
        // the band: an add further than the terrain can be cannot win
        // `min` at all.
        //
        // Measured on the planet's forest before this: 146071 candidates
        // evaluated per chunk of which 85.8% survived — 10.7 ms a chunk
        // to remove a seventh of the work, because at that density the
        // intervals all overlap and `prune_chain` proves nothing.
        INDEX_COARSE.add_nanos(coarse_started.elapsed().as_nanos() as u64);
        let fine_started = std::time::Instant::now();
        const SUB: usize = N / T;
        let mut bucket: Vec<Vec<u32>> = vec![Vec::new(); SUB * SUB * SUB];
        let mut subset: Vec<CsgOp> = Vec::new();
        for tz in 0..T {
            for ty in 0..T {
                for tx in 0..T {
                    let t = tx + ty * T + tz * T * T;
                    let (start, outer) = coarse[t].as_ref().expect("filled above");
                    let base = IVec3::new((tx * SUB) as i32, (ty * SUB) as i32, (tz * SUB) as i32);
                    for b in bucket.iter_mut() {
                        b.clear();
                    }
                    for &i in outer {
                        let op = ops[i as usize];
                        // Adds also lose to the terrain's own bound; a
                        // negative one means the cell is solid throughout
                        // and no add can matter in it at all.
                        let inflate = match (op.kind & 1 == 0, start) {
                            (true, Some(s)) => band.min(s.hi),
                            _ => band,
                        };
                        if inflate < 0.0 {
                            continue;
                        }
                        let bb = boxes[i as usize];
                        let pad = Vec3::splat(inflate + apron);
                        // Cell c spans `min + step*c - apron ..
                        // min + step*(c+1) + apron`, so the low end is a
                        // cell earlier than the division says.
                        let lo = ((bb.min - pad - min) / step - Vec3::ONE).floor();
                        let hi = ((bb.max + pad - min) / step).floor();
                        let lo = IVec3::new(lo.x as i32, lo.y as i32, lo.z as i32).max(base);
                        let hi = IVec3::new(hi.x as i32, hi.y as i32, hi.z as i32)
                            .min(base + IVec3::splat(SUB as i32 - 1));
                        for z in lo.z..=hi.z {
                            for y in lo.y..=hi.y {
                                for x in lo.x..=hi.x {
                                    let s = (x - base.x) as usize
                                        + (y - base.y) as usize * SUB
                                        + (z - base.z) as usize * SUB * SUB;
                                    bucket[s].push(i);
                                }
                            }
                        }
                    }
                    for (s, near) in bucket.iter().enumerate() {
                        let c = base
                            + IVec3::new(
                                (s % SUB) as i32,
                                ((s / SUB) % SUB) as i32,
                                (s / (SUB * SUB)) as i32,
                            );
                        let cf = c.as_vec3();
                        let lo = min + step * cf - Vec3::splat(apron);
                        let hi = min + step * (cf + Vec3::ONE) + Vec3::splat(apron);
                        let keep: Vec<u32> = match start {
                            Some(start) => {
                                subset.clear();
                                subset.extend(near.iter().map(|i| ops[*i as usize]));
                                prune_chain(&subset, *start, lo, hi)
                                    .into_iter()
                                    .map(|j| near[j as usize])
                                    .collect()
                            }
                            None => near.clone(),
                        };
                        CELL_TESTED.count(near.len() as u64);
                        CELL_KEPT.count(keep.len() as u64);
                        let cell = c.x as usize + c.y as usize * N + c.z as usize * N * N;
                        cells[cell * 2] = cells.len() as u32;
                        cells[cell * 2 + 1] = keep.len() as u32;
                        cells.extend_from_slice(&keep);
                    }
                }
            }
        }
        INDEX_FINE.add_nanos(fine_started.elapsed().as_nanos() as u64);
        Self { ops, cells }
    }

    /// The ops a point in this chunk actually needs, as indices. `rel` is
    /// the point's position in `0..1` across the chunk; out-of-range
    /// clamps, which is what makes the apron sound — a sample in the
    /// apron reads the border cell, and that cell was pruned with the
    /// apron included.
    pub fn cell_run(&self, rel: Vec3) -> (usize, usize) {
        if self.cells.is_empty() {
            return (0, self.ops.len());
        }
        let n = CSG_CELLS as f32;
        let i = (rel * n).floor();
        let idx = |v: f32| (v as isize).clamp(0, CSG_CELLS as isize - 1) as usize;
        let cell = idx(i.x) + idx(i.y) * CSG_CELLS + idx(i.z) * CSG_CELLS * CSG_CELLS;
        (
            self.cells[cell * 2] as usize,
            self.cells[cell * 2 + 1] as usize,
        )
    }
}

/// A uniform grid over one set of ops, so a box query walks the bins it
/// overlaps instead of every op in the set.
///
/// The other half of the op index, and a different question from
/// [`ChunkOps`]. That one asks "of the ops this CHUNK was given, which
/// can each of its cells need" — a few hundred ops, pruned against the
/// field. This one asks "of the ops in this emit CELL, which reach that
/// box at all" — thousands of ops, and the answer is pure geometry.
///
/// It exists because the emit index bottoms out in a linear scan: a
/// chunk's ops query walked 125 million ops to return 988 thousand on
/// the planet's forest, a 127x over-scan, and it could not be fixed by
/// making the emit cells smaller — sixteen times as many layer chunks
/// cost back exactly what the shorter scan saved. The cell has to stay
/// big and get an index INSIDE it.
///
/// Each op is bucketed by its box's MIDPOINT and a query is widened by
/// the largest half-extent in the grid — the same rule the emit cells
/// themselves use, one level down. The alternative, storing an op in
/// every bin it overlaps, needs the reader to dedupe: an op handed over
/// twice is applied twice, and for a smooth blend that is not the same
/// field. One bin per op means there is nothing to dedupe.
///
/// Bins hold op INDICES, not ops: the caller still has the op array and
/// still applies the exact test, so this only ever narrows the
/// candidates and can never change an answer.
#[derive(Clone, Debug, Default)]
pub struct OpGrid {
    min: Vec3,
    max: Vec3,
    inv_step: Vec3,
    dim: IVec3,
    /// Largest half-extent of any op in here, per axis — what a query
    /// must widen by to be sure it reaches every midpoint that matters.
    reach: Vec3,
    /// Prefix sums, `dim.x*dim.y*dim.z + 1` long.
    starts: Vec<u32>,
    /// Op indices, one entry each, bucketed by midpoint.
    items: Vec<u32>,
}

/// Bin edge, in meters. Comfortably larger than one emitted element and
/// comfortably smaller than an emit cell, which is the whole requirement:
/// bigger and a bin holds ops a query does not want, smaller and the
/// query's widening spans so many bins that nothing is narrowed.
const BIN_M: f32 = 8.0;

/// Bins per axis, capped. A cell far taller than it is wide (a
/// megastructure column) would otherwise ask for a grid with more bins
/// than it has ops to put in them.
const BIN_MAX: i32 = 24;

/// Ops below this stay a flat scan: the grid costs two passes and an
/// allocation to build, and under a bin's worth of ops there is nothing
/// for it to narrow.
const GRID_FROM: usize = 64;

impl OpGrid {
    /// Index `ops` by their boxes. Cheap enough to do at cell creation:
    /// two passes and one allocation, against a scan per query for the
    /// life of the cell.
    pub fn build(ops: &[CsgOp]) -> Self {
        if ops.len() < GRID_FROM {
            return Self::default();
        }
        let boxes: Vec<Aabb> = ops.iter().map(CsgOp::aabb).collect();
        let mut lo = boxes[0].min;
        let mut hi = boxes[0].max;
        let mut reach = Vec3::ZERO;
        for b in &boxes {
            lo = lo.min(b.min);
            hi = hi.max(b.max);
            reach = reach.max((b.max - b.min) * 0.5);
        }
        let span = (hi - lo).max(Vec3::splat(1.0e-3));
        let dim = IVec3::new(
            ((span.x / BIN_M).ceil() as i32).clamp(1, BIN_MAX),
            ((span.y / BIN_M).ceil() as i32).clamp(1, BIN_MAX),
            ((span.z / BIN_M).ceil() as i32).clamp(1, BIN_MAX),
        );
        let inv_step = dim.as_vec3() / span;
        let bins = (dim.x * dim.y * dim.z) as usize;
        let bin_of = |p: Vec3| -> IVec3 {
            let c = ((p - lo) * inv_step).floor();
            IVec3::new(c.x as i32, c.y as i32, c.z as i32).clamp(IVec3::ZERO, dim - IVec3::ONE)
        };
        // Counting sort: count, prefix-sum, place. No per-bin `Vec` — a
        // cell holds thousands of ops and `bins` allocations would cost
        // more than the scan this replaces.
        let mut starts = vec![0u32; bins + 1];
        let mids: Vec<usize> = boxes
            .iter()
            .map(|b| {
                let c = bin_of((b.min + b.max) * 0.5);
                Self::bin(dim, c)
            })
            .collect();
        for &b in &mids {
            starts[b + 1] += 1;
        }
        for i in 1..=bins {
            starts[i] += starts[i - 1];
        }
        let mut items = vec![0u32; ops.len()];
        let mut fill = starts.clone();
        for (i, &b) in mids.iter().enumerate() {
            items[fill[b] as usize] = i as u32;
            fill[b] += 1;
        }
        Self {
            min: lo,
            max: hi,
            inv_step,
            dim,
            reach,
            starts,
            items,
        }
    }

    fn bin(dim: IVec3, c: IVec3) -> usize {
        (c.x + c.y * dim.x + c.z * dim.x * dim.y) as usize
    }

    /// Every op index that could touch `query`, each exactly once, or
    /// `None` if this grid was never built and the caller must scan.
    pub fn candidates(&self, query: Aabb, mut f: impl FnMut(u32)) -> bool {
        if self.items.is_empty() {
            return false;
        }
        // Widened by the largest op in here, because ops are bucketed by
        // midpoint: one whose midpoint sits a bin away can still reach in.
        let want = Aabb::new(query.min - self.reach, query.max + self.reach);
        // A query outside the indexed span clamps INTO it, so reject it
        // before the clamp turns "outside" into "the border bin".
        if !Aabb::new(self.min, self.max).touches(want) {
            return true;
        }
        let bin_of = |p: Vec3| -> IVec3 {
            let c = ((p - self.min) * self.inv_step).floor();
            IVec3::new(c.x as i32, c.y as i32, c.z as i32).clamp(IVec3::ZERO, self.dim - IVec3::ONE)
        };
        let (a, z) = (bin_of(want.min), bin_of(want.max));
        for cz in a.z..=z.z {
            for cy in a.y..=z.y {
                for cx in a.x..=z.x {
                    let b = Self::bin(self.dim, IVec3::new(cx, cy, cz));
                    for &i in &self.items[self.starts[b] as usize..self.starts[b + 1] as usize] {
                        f(i);
                    }
                }
            }
        }
        true
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// An axis-aligned box in world meters.
///
/// One name for a question asked all over: does this thing reach that
/// place. It was written out six times before this — in the op cull, the
/// chunk fingerprint, the edit sweep and three tests — and six copies of
/// an inequality chain are six chances to get one `<=` backwards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// A box of half-extents `half` about `center`.
    pub fn around(center: Vec3, half: Vec3) -> Self {
        Self::new(center - half, center + half)
    }

    /// Do the two overlap? Touching at a face counts: a chunk reads the
    /// samples on its own boundary.
    pub fn touches(self, other: Self) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }

    /// The smallest box holding both.
    pub fn union(self, other: Self) -> Self {
        Self::new(self.min.min(other.min), self.max.max(other.max))
    }

    /// Grown by `by` meters on every side.
    pub fn inflate(self, by: f32) -> Self {
        Self::new(self.min - Vec3::splat(by), self.max + Vec3::splat(by))
    }
}

#[cfg(test)]
mod tests {

    /// The grid narrows candidates and never loses one.
    ///
    /// The whole contract: whatever the emit index hands a chunk must be
    /// what a full scan would have handed it. A grid that dropped an op
    /// on a bin edge is a hole in the world that only shows from some
    /// camera positions, so this checks EVERY op against every query
    /// rather than sampling.
    #[test]
    fn a_grid_narrows_candidates_without_losing_any() {
        use crate::seed::Rng;
        let mut rng = Rng::new(0x0_611D);
        let mut f = || rng.next_f32();
        // A cell's worth of forest: clustered limbs, not a uniform
        // scatter, which is what the bins have to cope with.
        let trunks: Vec<Vec3> = (0..60)
            .map(|_| Vec3::new(f() * 64.0, f() * 40.0, f() * 64.0))
            .collect();
        let ops: Vec<CsgOp> = (0..1200)
            .map(|i| {
                let t = trunks[i % trunks.len()];
                let c = t + Vec3::new(f() * 2.0 - 1.0, f() * 5.0, f() * 2.0 - 1.0);
                if i % 5 == 0 {
                    CsgOp::sphere(c, 0.3 + f() * 1.2, 8, false)
                } else {
                    CsgOp::capsule(
                        c,
                        c + Vec3::new(f() * 2.0 - 1.0, f() * 2.0, f() * 2.0 - 1.0),
                        0.05 + f() * 0.3,
                        0.05 + f() * 0.3,
                        9,
                        false,
                    )
                }
            })
            .collect();
        let grid = OpGrid::build(&ops);
        assert!(!grid.is_empty(), "1200 ops should be worth indexing");
        let (mut candidates, mut hits) = (0usize, 0usize);
        for q in 0..200u32 {
            let lo = Vec3::new(f() * 70.0 - 3.0, f() * 45.0 - 3.0, f() * 70.0 - 3.0);
            let query = Aabb::new(lo, lo + Vec3::splat(3.2 + f() * 22.4));
            // What a full scan would return.
            let want: Vec<u32> = (0..ops.len() as u32)
                .filter(|i| ops[*i as usize].touches(query))
                .collect();
            let mut got: Vec<u32> = Vec::new();
            assert!(grid.candidates(query, |i| {
                candidates += 1;
                if ops[i as usize].touches(query) {
                    got.push(i);
                }
            }));
            let _ = q;
            got.sort_unstable();
            assert_eq!(got, want, "query {q} {query:?} lost or duplicated ops");
            hits += want.len();
        }
        println!("grid walked {candidates} candidates for {hits} hits");
        // The point of it: a query looks at a small multiple of what it
        // returns, not at the whole cell. A flat scan would be 1200 per
        // query, 240000 in all.
        assert!(
            candidates * 4 < 200 * ops.len(),
            "walked {candidates} — barely narrowing"
        );
    }

    /// An op far longer than a bin is still found, from either end.
    ///
    /// Bucketing by midpoint is only sound because the query is widened
    /// by the largest half-extent in the grid; a long diagonal limb is
    /// what that rule is for.
    #[test]
    fn a_grid_finds_an_op_far_bigger_than_a_bin() {
        let long = CsgOp::capsule(
            Vec3::new(-90.0, 0.0, -90.0),
            Vec3::new(90.0, 5.0, 90.0),
            1.0,
            1.0,
            3,
            false,
        );
        let mut ops = vec![long];
        // Enough small ops around it to be worth indexing at all.
        for i in 0..GRID_FROM {
            let t = i as f32;
            ops.push(CsgOp::sphere(
                Vec3::new(t * 2.0 - 60.0, 0.0, t * 2.0 - 60.0),
                0.5,
                4,
                false,
            ));
        }
        let grid = OpGrid::build(&ops);
        assert!(!grid.is_empty());
        for end in [Vec3::new(-88.0, 0.0, -88.0), Vec3::new(88.0, 5.0, 88.0)] {
            let query = Aabb::new(end - Vec3::splat(2.0), end + Vec3::splat(2.0));
            let mut found = false;
            assert!(grid.candidates(query, |i| found |= i == 0));
            assert!(found, "the long limb was lost at {end:?}");
        }
    }

    /// A query that misses the indexed span entirely walks nothing —
    /// the clamp into the grid must not turn "outside" into "the border
    /// bin".
    #[test]
    fn a_grid_rejects_a_query_that_misses_it() {
        let ops: Vec<CsgOp> = (0..GRID_FROM)
            .map(|i| CsgOp::sphere(Vec3::splat(i as f32 * 0.5), 1.0, 3, false))
            .collect();
        let grid = OpGrid::build(&ops);
        let mut n = 0;
        assert!(
            grid.candidates(Aabb::new(Vec3::splat(500.0), Vec3::splat(510.0)), |_| n +=
                1)
        );
        assert_eq!(n, 0);
    }

    /// Under the threshold there is no grid, and the caller is told to
    /// scan — a silent empty answer there would be a world with no ops
    /// in it.
    #[test]
    fn a_small_set_is_not_worth_indexing() {
        let ops = vec![CsgOp::sphere(Vec3::ZERO, 1.0, 3, false)];
        let grid = OpGrid::build(&ops);
        assert!(grid.is_empty());
        assert!(
            !grid.candidates(Aabb::new(Vec3::splat(-9.0), Vec3::splat(9.0)), |_| {
                panic!("no grid should yield nothing")
            })
        );
    }

    /// A transformed op is the SAME field, moved.
    ///
    /// The whole memoization rests on this: a skeleton grown once in unit
    /// space and placed by [`CsgOp::transformed`] has to be the field the
    /// growth would have produced there. Checked where it matters — the
    /// distance at a point, against the distance at the point pulled back
    /// through the same transform, which for a scaled field is scaled too.
    #[test]
    fn a_transformed_op_is_the_same_field_moved() {
        let q = glam::Quat::from_euler(glam::EulerRot::YXZ, 0.7, 0.12, -0.05);
        let t = Vec3::new(-31.5, 88.25, 7.0);
        for scale in [0.5f32, 1.0, 1.7] {
            for op in [
                CsgOp::capsule(
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.4, 2.6, -0.9),
                    0.31,
                    0.08,
                    3,
                    false,
                ),
                CsgOp::sphere(Vec3::new(0.2, 3.1, -0.4), 0.75, 4, false),
            ] {
                let moved = op.transformed(t, q, scale);
                for p in [
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::new(1.3, 2.2, -0.7),
                    Vec3::new(-2.0, 0.2, 3.4),
                    Vec3::new(0.05, 4.0, 0.05),
                ] {
                    let want = op.sdf(p) * scale;
                    let got = moved.sdf(t + q * (p * scale));
                    assert!(
                        (want - got).abs() < 1.0e-3 * scale.max(1.0),
                        "kind {} scale {scale}: {want} vs {got}",
                        op.kind
                    );
                }
                assert_eq!(moved.kind, op.kind);
                assert_eq!(moved.material, op.material);
            }
        }
    }

    /// The bounds of a moved op still contain it — what the per-cell index
    /// prunes against, so an op that escaped its own box would vanish from
    /// chunks that need it.
    #[test]
    fn a_transformed_op_stays_inside_its_bounds() {
        let q = glam::Quat::from_euler(glam::EulerRot::YXZ, 2.1, 0.06, 0.0);
        let op = CsgOp::capsule(Vec3::ZERO, Vec3::new(-0.7, 3.4, 0.25), 0.28, 0.06, 3, false)
            .transformed(Vec3::new(11.0, -4.0, 60.0), q, 1.4);
        let b = op.aabb();
        let (min, max) = (b.min, b.max);
        for i in 0..=8 {
            let t = i as f32 / 8.0;
            let a = Vec3::from(op.center);
            let p = a + Vec3::from(op.half) * t;
            let r = op.yaw + (op.aux[0] - op.yaw) * t;
            assert!(
                (p - Vec3::splat(r)).cmpge(min - Vec3::splat(1.0e-4)).all()
                    && (p + Vec3::splat(r)).cmple(max + Vec3::splat(1.0e-4)).all(),
                "t {t}: {p:?} r {r} outside {min:?}..{max:?}"
            );
        }
    }
    use super::*;

    #[test]
    fn op_is_48_bytes_pod() {
        assert_eq!(std::mem::size_of::<CsgOp>(), 48);
        let op = CsgOp::boxy(Vec3::new(1.0, 2.0, 3.0), Vec3::ONE, 0.5, 3, false);
        let bytes: &[u8] = bytemuck::bytes_of(&op);
        let back: &CsgOp = bytemuck::from_bytes(bytes);
        assert_eq!(*back, op);
    }

    /// THE property the whole optimisation rests on: a pruned chain
    /// answers exactly what the full chain answers, everywhere inside
    /// the box it was pruned for.
    ///
    /// Dropping an op that mattered is a hole in the world that appears
    /// only where some camera happens to look, so this is checked
    /// against the real evaluator over randomised scenes rather than
    /// argued about. Randomised because the failure mode is a bound that
    /// is tight for the shapes you thought of.
    #[test]
    fn a_pruned_chain_answers_what_the_full_chain_answers() {
        use crate::seed::Rng;
        let mut rng = Rng::new(0x0000_C56A);
        let mut total_kept = 0usize;
        let mut total_ops = 0usize;
        for case in 0..200 {
            let mut f = || rng.next_f32();
            // A scene of mixed kinds scattered over a wide area, so most
            // ops are far from any one box and prunable — the situation
            // a chunk full of trees is actually in.
            let ops: Vec<CsgOp> = (0..24)
                .map(|i| {
                    let c = Vec3::new(f() * 40.0 - 20.0, f() * 40.0 - 20.0, f() * 40.0 - 20.0);
                    let cut = i % 4 == 3;
                    match i % 4 {
                        0 => CsgOp::sphere(c, 0.5 + f() * 3.0, 1, cut),
                        1 => CsgOp::boxy(c, Vec3::splat(0.5 + f() * 2.0), f() * 3.0, 1, cut),
                        2 => CsgOp::capsule(
                            c,
                            c + Vec3::new(f() * 6.0 - 3.0, f() * 6.0 - 3.0, f() * 6.0 - 3.0),
                            0.2 + f() * 1.0,
                            0.2 + f() * 1.0,
                            1,
                            cut,
                        ),
                        _ => CsgOp::cylinder(c, 0.5 + f() * 2.0, 0.5 + f() * 3.0, 1, cut),
                    }
                })
                .collect();

            // A box somewhere in the scene, at a chunk-ish scale.
            let lo = Vec3::new(f() * 30.0 - 15.0, f() * 30.0 - 15.0, f() * 30.0 - 15.0);
            let size = 0.5 + f() * 6.0;
            let hi = lo + Vec3::splat(size);

            // The chain starts from a terrain-like field. Its interval
            // has to BOUND it or pruning is unsound, so use a plane and
            // its exact bound.
            let plane_h = f() * 10.0 - 5.0;
            let start = Interval::new(lo.y - plane_h, hi.y - plane_h);

            let kept = prune_chain(&ops, start, lo, hi);
            total_kept += kept.len();
            total_ops += ops.len();

            for _ in 0..40 {
                let p = lo + Vec3::new(f(), f(), f()) * size;
                let d0 = p.y - plane_h;
                let full = ops.iter().fold(d0, |d, op| op.apply(d, p));
                let pruned = kept.iter().fold(d0, |d, i| ops[*i as usize].apply(d, p));
                assert!(
                    (full - pruned).abs() < 1.0e-4,
                    "case {case}: pruned chain disagrees at {p:?}: {full} vs {pruned} \
                     (kept {} of {})",
                    kept.len(),
                    ops.len(),
                );
            }
        }
        // And it has to actually prune, or the test above passes for the
        // most boring possible reason.
        assert!(
            total_kept * 3 < total_ops,
            "pruning kept {total_kept} of {total_ops} — not pruning"
        );
        println!("pruned to {total_kept} of {total_ops}");
    }

    /// `prune_dominated` claims to be right for ANY terrain, since it is
    /// used where the terrain is not known — so it is checked against
    /// several, including ones that put the surface far above and far
    /// below the ops. A pruning that only works for the terrain you
    /// happened to test with is a hole that appears in one biome.
    #[test]
    fn domination_pruning_holds_for_any_terrain() {
        use crate::seed::Rng;
        let mut rng = Rng::new(0x0000_D011);
        let (mut kept, mut total) = (0usize, 0usize);
        for _ in 0..150 {
            let mut f = || rng.next_f32();
            // Clustered, like limbs of a few trees: domination only has
            // anything to prune when some ops are much nearer than others.
            // Eight "trees" of ten limbs, spread over a chunk-sized area
            // — the shape the pruning exists for.
            let ops: Vec<CsgOp> = (0..80)
                .map(|i| {
                    let t = (i / 10) as f32;
                    let cluster = Vec3::new(t * 9.0, (t * 1.7) % 4.0, ((t * 5.0) % 11.0) * 1.1);
                    let c = cluster + Vec3::new(f() * 4.0, f() * 4.0, f() * 4.0);
                    if i % 7 == 6 {
                        CsgOp::sphere(c, 0.3 + f(), 1, true)
                    } else {
                        CsgOp::capsule(
                            c,
                            c + Vec3::new(f() * 2.0 - 1.0, f() * 2.0, f() * 2.0 - 1.0),
                            0.1 + f() * 0.3,
                            0.1 + f() * 0.3,
                            1,
                            false,
                        )
                    }
                })
                .collect();
            // A sub-cell, not a whole chunk: this is what the shader
            // will look ops up per, so it is what must prune well.
            let lo = Vec3::new(f() * 70.0, f() * 6.0, f() * 12.0);
            let hi = lo + Vec3::splat(0.4 + f() * 1.2);
            let idx = prune_dominated(&ops, lo, hi);
            kept += idx.len();
            total += ops.len();

            for _ in 0..25 {
                let p = lo + (hi - lo) * Vec3::new(f(), f(), f());
                // Every terrain from "far above everything" to "solid
                // rock", including exactly grazing.
                for d0 in [-1000.0, -5.0, -0.001, 0.0, 0.001, 5.0, 1000.0, f32::MAX] {
                    let full = ops.iter().fold(d0, |d, op| op.apply(d, p));
                    let pruned = idx.iter().fold(d0, |d, i| ops[*i as usize].apply(d, p));
                    assert!(
                        (full - pruned).abs() < 1.0e-4,
                        "terrain {d0}: {full} vs {pruned} at {p:?}"
                    );
                }
            }
        }
        assert!(kept * 2 < total, "kept {kept} of {total} — not pruning");
        println!("domination kept {kept} of {total}");
    }

    /// The index a chunk hands the GPU has to answer what the flat op
    /// list answers, at every point of the chunk INCLUDING the apron —
    /// which is the part a per-cell scheme gets wrong, because a sample
    /// in the apron is outside the cell grid entirely.
    #[test]
    fn a_cell_index_answers_what_the_flat_list_answers() {
        use crate::seed::Rng;
        let mut rng = Rng::new(0x0000_DE11);
        let (mut kept, mut total) = (0usize, 0usize);
        for _ in 0..40 {
            let mut f = || rng.next_f32();
            // A chunk-sized box with a forest's worth of limbs in it.
            let min = Vec3::new(f() * 100.0, f() * 20.0, f() * 100.0);
            let edge = 12.8;
            let max = min + Vec3::splat(edge);
            let apron = 0.4;
            // Twenty trees of ten limbs, not a uniform scatter: limbs
            // cluster into trunks and trunks stand apart, which is the
            // structure the pruning exploits and the one it will meet.
            let trunks: Vec<Vec3> = (0..20)
                .map(|_| min + Vec3::new(f() * edge, f() * edge * 0.4, f() * edge))
                .collect();
            let ops: Vec<CsgOp> = (0..200)
                .map(|i| {
                    let t = trunks[i % trunks.len()];
                    let c = t + Vec3::new(f() * 1.2 - 0.6, f() * 2.4, f() * 1.2 - 0.6);
                    CsgOp::capsule(
                        c,
                        c + Vec3::new(f() * 1.6 - 0.8, f() * 1.6, f() * 1.6 - 0.8),
                        0.05 + f() * 0.25,
                        0.05 + f() * 0.25,
                        3,
                        false,
                    )
                })
                .collect();
            // A sloped plane as the terrain, bounded exactly.
            let plane = min.y + edge * 0.5;
            let terrain = |lo: Vec3, hi: Vec3| Some(Interval::new(lo.y - plane, hi.y - plane));
            // A band a chunk of this edge would really have: 12.8 m
            // over 32 cells is 0.4 m voxels, four of them.
            let band = crate::SDF_BAND * (edge / crate::CHUNK_CELLS as f32);
            let indexed = ChunkOps::build(ops.clone(), min, max, apron, band, terrain);
            let ncells = CSG_CELLS * CSG_CELLS * CSG_CELLS;
            kept += indexed.cells.len().saturating_sub(2 * ncells);
            total += ops.len() * ncells;

            for _ in 0..300 {
                // Sample the apron too: `-apron .. edge + apron`.
                let p = min + Vec3::new(f(), f(), f()) * (edge + 2.0 * apron) - Vec3::splat(apron);
                let d0 = p.y - plane;
                let full = ops.iter().fold(d0, |d, op| op.apply(d, p));
                let rel = (p - min) / edge;
                let (off, count) = indexed.cell_run(rel);
                let via = if indexed.cells.is_empty() {
                    ops.iter().fold(d0, |d, op| op.apply(d, p))
                } else {
                    (0..count).fold(d0, |d, i| {
                        indexed.ops[indexed.cells[off + i] as usize].apply(d, p)
                    })
                };
                // CLAMPED, which is the only form the density pass ever
                // stores: the index drops ops that can only ever put the
                // value outside the band, and outside the band every
                // value is the same value. Comparing raw distances here
                // would be asserting something the field does not keep.
                let clamp = |d: f32| d.clamp(-band, band);
                assert!(
                    (clamp(full) - clamp(via)).abs() < 1.0e-4,
                    "cell index disagrees at {p:?} (rel {rel:?}): {full} vs {via}"
                );
            }
        }
        println!("cell index kept {kept} of {total}");
        // A floor, not the measured number: what it prunes depends on
        // how clustered the scene is, and encoding one scene's ratio
        // here would just be a number to update.
        // Was `kept * 4 < total` when the index pruned by field alone
        // and this scene was the generous case for it. The band reject
        // is a different order of thing: 4.7% here against 25%.
        //
        // The field prune stays ON TOP of the band reject even though it
        // removes only a further 0.1% in THIS scene — dropping it
        // measured as no settle change at all on the planet, and the
        // megastructure is a solid interior where the box test is the
        // one that proves nothing.
        assert!(kept * 20 < total, "kept {kept} of {total} — not pruning");
    }

    /// A capsule is the one kind that points anywhere, so the thing to
    /// pin is that its distance field is right OFF the axis and in every
    /// direction — not just along it.
    #[test]
    fn a_capsule_measures_distance_to_its_axis() {
        // A diagonal limb, deliberately not axis-aligned: an
        // implementation that fell back to yaw rotation or read `half`
        // as an extent passes an axis-aligned test and fails this one.
        let a = Vec3::new(1.0, 2.0, -1.0);
        let b = Vec3::new(3.0, 5.0, 2.0);
        let op = CsgOp::capsule(a, b, 0.5, 0.5, 0, false);

        // On the axis: inside by the radius.
        let mid = (a + b) * 0.5;
        assert!((op.sdf(mid) + 0.5).abs() < 1.0e-5, "{}", op.sdf(mid));
        // Exactly on the surface, measured perpendicular to the axis.
        let axis = (b - a).normalize();
        let perp = axis.cross(Vec3::Y).normalize();
        assert!((op.sdf(mid + perp * 0.5)).abs() < 1.0e-5);
        assert!((op.sdf(mid + perp * 1.5) - 1.0).abs() < 1.0e-5);
        // Past an END it is a ball, not an infinite cylinder.
        assert!((op.sdf(a - axis * 2.0) - 1.5).abs() < 1.0e-5);
        assert!((op.sdf(b + axis * 2.0) - 1.5).abs() < 1.0e-5);

        // Taper: the radius follows the parameter along the axis.
        let cone = CsgOp::capsule(Vec3::ZERO, Vec3::Y * 4.0, 1.0, 0.0, 0, false);
        assert!((cone.sdf(Vec3::new(1.0, 0.0, 0.0))).abs() < 1.0e-5);
        assert!((cone.sdf(Vec3::new(0.5, 2.0, 0.0))).abs() < 1.0e-5);
        assert!(cone.sdf(Vec3::new(0.9, 3.6, 0.0)) > 0.0, "tip is thin");
    }

    /// The conservative AABB has to cover a capsule wherever it points —
    /// a box read off `half` as an extent misses most of a limb, and a
    /// missed op is geometry that vanishes when a chunk culls it.
    #[test]
    fn aabb_covers_a_capsule_in_any_direction() {
        for dir in [
            Vec3::new(3.0, 5.0, 2.0),
            Vec3::new(-4.0, -1.0, 2.5),
            Vec3::new(0.0, -6.0, 0.0),
        ] {
            let a = Vec3::new(1.0, 2.0, -1.0);
            let op = CsgOp::capsule(a, a + dir, 0.5, 0.9, 0, false);
            let bb = op.aabb();
            // Sample along the limb; every solid point must be inside.
            for i in 0..=20 {
                let t = i as f32 / 20.0;
                let p = a + dir * t;
                let r = 0.5 + (0.9 - 0.5) * t;
                for off in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z] {
                    let s = p + off * r;
                    assert!(
                        bb.min.cmple(s).all() && s.cmple(bb.max).all(),
                        "capsule surface {s:?} outside {bb:?} for dir {dir:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn aabb_covers_rotated_box() {
        // A yawed box's corners stay inside the conservative AABB.
        let op = CsgOp::boxy(Vec3::ZERO, Vec3::new(4.0, 1.0, 1.0), 0.7, 0, false);
        let b = op.aabb();
        for sx in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let corner = Vec3::new(4.0 * sx, 0.0, 1.0 * sz);
                let (s, c) = (0.7f32.sin(), 0.7f32.cos());
                let world = Vec3::new(
                    corner.x * c - corner.z * s,
                    0.0,
                    corner.x * s + corner.z * c,
                );
                assert!(world.x >= b.min.x && world.x <= b.max.x);
                assert!(world.z >= b.min.z && world.z <= b.max.z);
            }
        }
    }

    #[test]
    fn sdf_matches_primitives() {
        let b = CsgOp::boxy(Vec3::ZERO, Vec3::new(2.0, 1.0, 3.0), 0.0, 0, false);
        assert!(b.sdf(Vec3::ZERO) < 0.0);
        assert!((b.sdf(Vec3::new(4.0, 0.0, 0.0)) - 2.0).abs() < 1e-5);
        let cyl = CsgOp::cylinder(Vec3::ZERO, 1.5, 2.0, 0, false);
        assert!((cyl.sdf(Vec3::new(3.0, 0.0, 0.0)) - 1.5).abs() < 1e-5);
        assert!((cyl.sdf(Vec3::new(0.0, 5.0, 0.0)) - 3.0).abs() < 1e-5);
        // Cut ops carve: applying a cut around a point makes it air.
        let cut = CsgOp::boxy(Vec3::ZERO, Vec3::ONE, 0.0, 0, true);
        assert!(cut.apply(-10.0, Vec3::ZERO) > 0.0);
    }

    #[test]
    fn touches_is_conservative() {
        let op = CsgOp::cylinder(Vec3::new(100.0, 0.0, 0.0), 5.0, 10.0, 0, false);
        assert!(op.touches(Aabb::new(
            Vec3::new(90.0, -5.0, -5.0),
            Vec3::new(110.0, 5.0, 5.0)
        )));
        assert!(!op.touches(Aabb::new(
            Vec3::new(200.0, 0.0, 0.0),
            Vec3::new(210.0, 10.0, 10.0)
        )));
    }
}
