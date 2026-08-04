//! Rendering: surface-nets/dual-contouring compute meshing into slab buffers,
//! the LOD chunk octree, custom phase item with per-chunk draws, and chunk
//! material shading.

pub mod chunks;
pub mod slab;

pub use chunks::{ChunkCommandQueue, SharedRenderStats, VoxelChunksPlugin};
