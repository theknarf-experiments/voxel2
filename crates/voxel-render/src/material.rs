//! The terrain material, as an ordinary Bevy [`Material`] asset.
//!
//! The voxel draw is not a `Mesh3d`, but a material is just an
//! `AsBindGroup` plus an asset, and neither needs a mesh: Bevy extracts
//! `MeshMaterial3d<M>` from any visible entity, so the terrain marker
//! carries one and the draw binds it with Bevy's own
//! `SetMaterialBindGroup<3>` — the same command, at the same group index,
//! that every `StandardMaterial` uses.
//!
//! Each surface recipe is its own asset, and they are BINDLESS: Bevy
//! packs them into one slab that a single bind group covers, so the
//! shader can select a recipe per vertex. A per-vertex material id
//! cannot choose a per-draw bind group, so the engine uploads the
//! id → slab-slot mapping and the shader indirects through it.

use bevy::{prelude::*, render::render_resource::AsBindGroup};

use crate::chunks::WorldMaterial;

/// One surface recipe, as its own asset. All of a world's recipes live in
/// a single bindless slab, so the draw binds one bind group and the
/// shader picks a recipe per vertex — which is what lets a per-vertex
/// material id select a material at all.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
#[bindless]
#[data(0, WorldMaterial, binding_array(1))]
pub struct VoxelSurfaceMaterial {
    pub recipe: WorldMaterial,
}

impl From<&VoxelSurfaceMaterial> for WorldMaterial {
    fn from(material: &VoxelSurfaceMaterial) -> Self {
        material.recipe
    }
}

// The voxel pipeline supplies its own shaders; the trait's defaults only
// matter if someone puts this material on an actual mesh, which the custom
// vertex layout would not fit anyway.
impl Material for VoxelSurfaceMaterial {}
