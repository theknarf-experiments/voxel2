//! The terrain material, as an ordinary Bevy [`Material`] asset.
//!
//! The voxel draw is not a `Mesh3d`, but a material is just an
//! `AsBindGroup` plus an asset, and neither needs a mesh: Bevy extracts
//! `MeshMaterial3d<M>` from any visible entity, so the terrain marker
//! carries one and the draw binds it with Bevy's own
//! `SetMaterialBindGroup<3>` — the same command, at the same group index,
//! that every `StandardMaterial` uses.
//!
//! One material covers the whole world. A voxel's per-vertex material id
//! selects a *recipe within* it, the way a terrain layer index selects
//! within an array texture — per-vertex ids cannot map to per-draw bind
//! groups, so the id is an input to the material rather than a choice of
//! material. Bindless will let those recipes become separately-authored
//! assets living in one slab.

use bevy::{asset::uuid_handle, prelude::*, render::render_resource::AsBindGroup};

use crate::chunks::{GpuMaterialTable, WorldMaterial, MATERIAL_SLOTS};

/// The handle the terrain marker carries. Fixed so the engine can update
/// the recipes in place on a level reload without respawning anything.
pub const VOXEL_TERRAIN_MATERIAL: Handle<VoxelTerrainMaterial> =
    uuid_handle!("9f2f6a2c-3a5e-4d4c-9c1f-7c9a2b0f5d31");

/// Surface recipes for the material ids a level's generator ops emit.
///
/// Authored by the level (they describe the world, not its presentation),
/// but an app is free to replace the asset wholesale.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct VoxelTerrainMaterial {
    #[uniform(0)]
    pub(crate) table: GpuMaterialTable,
}

impl VoxelTerrainMaterial {
    /// Build from the level's recipes, padding to the fixed slot count.
    pub fn from_recipes(recipes: &[WorldMaterial]) -> Self {
        Self {
            table: GpuMaterialTable::from_slice(recipes),
        }
    }

    /// How many recipes one material can hold.
    pub const SLOTS: usize = MATERIAL_SLOTS;
}

// The voxel pipeline supplies its own shaders; the trait's defaults only
// matter if someone puts this material on an actual mesh, which the custom
// vertex layout would not fit anyway.
impl Material for VoxelTerrainMaterial {}
