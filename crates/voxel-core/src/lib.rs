//! Core voxel types shared by every other crate: chunk keys, voxel formats,
//! quantization, morton indexing, and deterministic seeding.
//!
//! This crate deliberately has no Bevy dependency.

pub mod csg;
pub mod interval;
pub mod key;
pub mod layout;
pub mod morton;
pub mod opgen;
pub mod patch;
pub mod pos;
pub mod seed;
pub mod voxel;
pub mod worldop;

pub use key::{ChunkKey, WorldId};
pub use pos::GlobalPos;
pub use voxel::Voxel;

/// Cells per chunk axis, at every LOD level.
pub const CHUNK_CELLS: u32 = 32;

/// Density samples per chunk axis: 33 cell corners plus an apron
/// (one sample below, two above) for gradients and skirts.
pub const CHUNK_SAMPLES: u32 = 36;

/// Offset of the sample grid relative to the cell-corner grid: sample index
/// `i` holds the value at corner `i + SAMPLE_OFFSET`, so samples cover
/// corners `-1..=34` for cells `0..32`.
pub const SAMPLE_OFFSET: i32 = -1;

/// Voxel edge length in meters at LOD 0.
pub const BASE_VOXEL_M: f64 = 0.1;

/// SDF narrow band half-width, in units of the voxel size at the chunk's LOD.
/// Stored SDF values are clamped to `±SDF_BAND`.
pub const SDF_BAND: f32 = 4.0;

/// Warn when a block of work overruns a frame budget, naming it.
///
/// Attribution for a stutter has to be per-SYSTEM: a frame-time graph
/// says a frame was slow and nothing about which of forty systems did it,
/// and a chrome trace needs a dependency this workspace cannot fetch.
/// Costs a `Instant::now()` pair on paths that already do far more.
#[macro_export]
macro_rules! timed {
    ($name:literal, $budget_ms:expr, $body:expr) => {{
        let __started = std::time::Instant::now();
        let __out = $body;
        let __ms = __started.elapsed().as_secs_f32() * 1000.0;
        if __ms > $budget_ms {
            bevy::log::warn!("COST {} {:.1}ms", $name, __ms);
        }
        __out
    }};
}
