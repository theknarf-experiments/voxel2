//! Rendering: surface-nets/dual-contouring compute meshing into slab buffers,
//! the LOD chunk octree, custom phase item with per-chunk draws, and chunk
//! material shading.

pub mod chunks;
pub mod material;
/// Shared plumbing so a HOST can build a pipeline that shades like the
/// terrain does (Bevy's view bind group, matching keys and shader defs).
/// Re-exported so everything that draws through Bevy's PBR views —
/// terrain here, a host's own pipelines — names one copy of the
/// machinery. The module moved to its own crate because nothing in it is
/// voxel: it is how ANY hand-rolled pipeline borrows Bevy's view bind
/// groups.
pub use bevy_pbr_view as pbr_view;
pub mod scatter_points;
pub mod slab;

pub use chunks::{
    material_slot_index, material_table, CameraWorld, ChunkCommand, ChunkCommandQueue,
    ChunkGpuResources, ChunkReadyChannel, ChunkWaiters, EnvParams, GpuWorldProgram, RenderWorld,
    RenderWorlds, SharedRenderStats, SurfaceMap, ViewWorld, VoxelChunksPlugin, WorldMaterial,
    WorldProgram, MATERIAL_SLOTS, MAT_KIND_CANOPY, MAT_KIND_SURFACE, MAT_KIND_ZONED,
    MAX_CLIP_PLANES, MAX_WORLDS,
};
pub use material::VoxelSurfaceMaterial;
pub use scatter_points::{ScatterPoint, ScatterPoints};
pub use slab::SlabAllocator;

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

/// First render layer a host may use for its own purposes.
///
/// The engine claims `0..=TERRAIN_LAYER` and nothing above it. A host
/// wanting per-world surfaces of its own — a view model, a 3D UI, the
/// quad an opening between two worlds is drawn on — allocates its own
/// band from here, and must keep its world's layer in the set too: Bevy
/// hides an entity when its layers do not intersect the view's, so a
/// camera on `{0, N}` and an entity on `{1, N}` still intersect at N and
/// the entity would be visible from the wrong world.
pub const FIRST_HOST_LAYER: usize = TERRAIN_LAYER + 1;

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
