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
    material_table, material_slot_index,
    ChunkGpuResources, GpuWorldProgram,
    ChunkCommand, ChunkCommandQueue, ChunkReadyChannel, ChunkWaiters, EnvParams,
    SharedRenderStats,
    SurfaceMap, VoxelChunksPlugin, WorldMaterial, WorldProgram, MATERIAL_SLOTS,
    MAX_WORLDS, RenderWorld, RenderWorlds, CameraWorld, ViewWorld, MAX_CLIP_PLANES,
    MAT_KIND_CANOPY, MAT_KIND_SURFACE, MAT_KIND_ZONED,
};
pub use slab::SlabAllocator;
pub use scatter_points::{ScatterPoint, ScatterPoints};
pub use material::VoxelSurfaceMaterial;

/// Render layer for a world's scene content. Layer N is world N.
///
/// Chunks are not entities and are filtered by `key.world` in the draw
/// loop. Everything else a world contains — grass, props, water, the
/// backdrop — IS an entity, queued against the view's visible set, so
/// putting the camera on its world's layer filters all of them at once
/// with no change to any of their pipelines. Two mechanisms, because
/// there are two kinds of thing being drawn.
///
/// **Layers `0..MAX_WORLDS` are spent.** A host wanting its own (a
/// first-person view model, a 3D UI pass) must start at
/// [`FIRST_HOST_LAYER`] AND keep its world's layer in the set: Bevy hides
/// an entity when its layers do not intersect the view's, so a camera on
/// `{0, 5}` and an entity on `{1, 5}` still intersect at 5 — the entity
/// would be visible from the wrong world.
pub fn world_layer(world: voxel_core::WorldId) -> bevy::camera::visibility::RenderLayers {
    bevy::camera::visibility::RenderLayers::layer(usize::from(world))
}

/// Visible from every world. For content that is not IN a world: the
/// terrain draw's anchor entity, and lights, which have to reach whichever
/// world is being viewed.
pub fn all_world_layers() -> bevy::camera::visibility::RenderLayers {
    (0..=TERRAIN_LAYER).fold(
        bevy::camera::visibility::RenderLayers::none(),
        |layers, world| layers.with(world),
    )
}

/// The terrain anchor's layer. It is not IN a world — which world's
/// chunks a view draws is `ViewWorld`, not a layer — so it is visible to
/// every view.
pub const TERRAIN_LAYER: usize = MAX_WORLDS;

/// A world's PORTAL SURFACES — the quads an opening is drawn on.
///
/// They cannot live on the world's own layer. A portal surface samples
/// the image the far camera renders, and that camera draws the world it
/// is showing; if the surface shared that world's layer the far camera
/// would draw the quad that samples its own target. A texture read while
/// it is being written is undefined, and it presents as an image that
/// updates sometimes — which is a great deal harder to recognise than
/// one that never updates at all.
pub fn portal_layer(world: voxel_core::WorldId) -> bevy::camera::visibility::RenderLayers {
    bevy::camera::visibility::RenderLayers::layer(FIRST_PORTAL_LAYER + usize::from(world))
}

/// What a camera standing IN `world` sees: its content, its lights, and
/// the portal surfaces cut into it.
pub fn near_view_layers(world: voxel_core::WorldId) -> bevy::camera::visibility::RenderLayers {
    world_layer(world).union(&portal_layer(world))
}

pub const FIRST_PORTAL_LAYER: usize = TERRAIN_LAYER + 1;

/// First render layer a host may use for its own purposes. See
/// [`world_layer`].
pub const FIRST_HOST_LAYER: usize = FIRST_PORTAL_LAYER + MAX_WORLDS;

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
