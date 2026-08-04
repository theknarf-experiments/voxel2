//! Rendering: surface-nets/dual-contouring compute meshing into slab buffers,
//! the LOD chunk octree, custom phase item with per-chunk draws, and chunk
//! material shading.

pub mod chunks;
pub mod grass;
pub mod slab;

pub use chunks::{
    ChunkCommand, ChunkCommandQueue, ChunkReadyChannel, SharedRenderStats, VoxelChunksPlugin,
};
pub use grass::{GrassInstance, GrassInstances, GrassPlugin};
