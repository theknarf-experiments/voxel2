//! Rendering: surface-nets/dual-contouring compute meshing into slab buffers,
//! the LOD chunk octree, custom phase item with per-chunk draws, and chunk
//! material shading.

pub mod chunks;
pub mod scatter_points;
pub mod material;
/// Shared plumbing so a HOST can build a pipeline that shades like the
/// terrain does (Bevy's view bind group, matching keys and shader defs).
pub mod pbr_view;
pub mod slab;

pub use chunks::{
    ChunkGpuResources, GpuWorldProgram,
    ChunkCommand, ChunkCommandQueue, ChunkReadyChannel, ChunkWaiters, EnvParams, FieldParams,
    SharedRenderStats,
    SurfaceMap, VoxelChunksPlugin, WorldMaterial, WorldMaterials, WorldProgram, MATERIAL_SLOTS,
    MAX_WORLDS,
    MAT_KIND_CANOPY, MAT_KIND_SURFACE, MAT_KIND_ZONED,
};
pub use scatter_points::{ScatterPoint, ScatterPoints};
pub use material::VoxelSurfaceMaterial;

/// Marker for helper cameras (offscreen screenshot mirrors, etc.) that
/// gameplay/streaming systems must ignore when looking for "the player
/// camera".
#[derive(bevy::prelude::Component)]
pub struct HelperCamera;

/// Query filter for "the player camera" (see [`HelperCamera`]).
pub type PlayerCameraFilter = (
    bevy::prelude::With<bevy::prelude::Camera3d>,
    bevy::prelude::Without<HelperCamera>,
);
