//! Core voxel types shared by every other crate: chunk keys, voxel formats,
//! quantization, morton indexing, and deterministic seeding.
//!
//! This crate deliberately has no Bevy dependency.

pub mod key;
pub mod morton;
pub mod pos;
pub mod seed;
pub mod voxel;

pub use key::ChunkKey;
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
pub const BASE_VOXEL_M: f64 = 1.0;

/// SDF narrow band half-width, in units of the voxel size at the chunk's LOD.
/// Stored SDF values are clamped to `±SDF_BAND`.
pub const SDF_BAND: f32 = 4.0;
