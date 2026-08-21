//! Where a settle's CPU goes, by stage.
//!
//! A settle is dozens of worker threads and hundreds of frames, so the
//! per-frame [`voxel_core::timed`] cannot see it and a frame-time graph
//! says nothing about it. A sampling profiler answers "which symbol",
//! which moves every time something is fixed; these answer "which stage",
//! which is the question when deciding what to CUT.
//!
//! The numbers are thread-summed wall time, so they overrun the settle
//! itself on a parallel machine — a stage at 8000 ms inside a 10 s settle
//! is not 80% of it. What they compare honestly is stage against stage,
//! and one run against another.
//!
//! Per-LAYER planning cost is not here: `voxctl status` -> `planning`
//! already reports `create_ms` per layer instance, which is the same
//! thing at a finer grain.
//!
//! Read with `voxctl status` -> `stages`.

use voxel_core::Stage;

/// The host's ops provider answering "which ops reach this chunk" —
/// a spatial query over the emit index, per chunk.
pub static OPS_QUERY: Stage = Stage::new();

/// Building a chunk's per-cell op index (`ChunkOps::build`): the interval
/// pruning that decides which ops each of the 512 cells can need.
pub static OPS_INDEX: Stage = Stage::new();

/// Ops handed to one chunk, summed — the size that explains both of the
/// stages above. Counted, not timed.
pub static OPS_PER_CHUNK: Stage = Stage::new();

/// Deciding whether a chunk can hold a surface at all (admission control).
pub static CAN_HOLD: Stage = Stage::new();

/// Every stage by name, for reporting.
pub fn all() -> Vec<(&'static str, f64, u64)> {
    [
        ("ops_query", &OPS_QUERY),
        ("ops_index", &OPS_INDEX),
        ("ops_per_chunk", &OPS_PER_CHUNK),
        ("can_hold", &CAN_HOLD),
        ("cell_tested", &voxel_core::csg::CELL_TESTED),
        ("cell_kept", &voxel_core::csg::CELL_KEPT),
        ("query_walked", &voxel_core::csg::QUERY_WALKED),
    ]
    .into_iter()
    .map(|(name, s)| {
        let (ms, calls) = s.read();
        (name, ms, calls)
    })
    .collect()
}

pub fn reset() {
    for s in [
        &OPS_QUERY,
        &OPS_INDEX,
        &OPS_PER_CHUNK,
        &CAN_HOLD,
        &voxel_core::csg::CELL_TESTED,
        &voxel_core::csg::CELL_KEPT,
        &voxel_core::csg::QUERY_WALKED,
    ] {
        s.reset();
    }
}
