//! Rendering: surface-nets/dual-contouring compute meshing into slab buffers,
//! the LOD chunk octree, custom phase item with per-chunk draws, and chunk
//! material shading.

pub mod chunks;
pub mod grass;
pub mod slab;
pub mod water;

pub use chunks::{
    ChunkCommand, ChunkCommandQueue, ChunkReadyChannel, ShadingMode, SharedRenderStats,
    VoxelChunksPlugin, WorldProgram,
};
pub use grass::{GrassInstance, GrassInstances, GrassPlugin};
pub use water::{WaterEnabled, WaterPlugin};
