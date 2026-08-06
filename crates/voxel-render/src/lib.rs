//! Rendering: surface-nets/dual-contouring compute meshing into slab buffers,
//! the LOD chunk octree, custom phase item with per-chunk draws, and chunk
//! material shading.

pub mod chunks;
pub mod grass;
pub mod material;
pub(crate) mod pbr_view;
pub mod slab;
pub mod water;

pub use chunks::{
    ChunkCommand, ChunkCommandQueue, ChunkReadyChannel, EnvParams, FieldParams, SharedRenderStats,
    VoxelChunksPlugin, WorldMaterial, WorldMaterials, WorldProgram, MATERIAL_SLOTS,
    MAT_KIND_CANOPY, MAT_KIND_SURFACE, MAT_KIND_ZONED,
};
pub use grass::{GrassInstance, GrassInstances, GrassPlugin, GrassStyle};
pub use material::VoxelSurfaceMaterial;
pub use water::{RiverSegGpu, RiverWater, WaterPlugin, WaterSurface};

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
