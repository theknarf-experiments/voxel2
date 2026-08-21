//! Core voxel types shared by every other crate: chunk keys, voxel formats,
//! quantization, morton indexing, and deterministic seeding.
//!
//! This crate deliberately has no Bevy dependency.

pub mod branch;
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

/// Serde `default` stubs: a name, a type and a value.
///
/// `#[serde(default = "...")]` names a FUNCTION, so every authored default
/// needs one to exist. Fifty-three of them were three lines each to say a
/// single number, and their names are the documentation — `d_scatter_tile`
/// and `d_floor_step` are both `0.5` and must stay two names, so this
/// collapses the syntax and nothing else.
#[macro_export]
macro_rules! defaults {
    ($($vis:vis $name:ident: $ty:ty = $value:expr;)*) => {
        $($vis fn $name() -> $ty { $value })*
    };
}
