//! What the engine knows about large flat surfaces: nothing but where
//! they are, and only for the generator's sea-level meta op.
//!
//! Ribbon surfaces (rivers, canals, lava) are plain planning-stack data —
//! see `voxel_worldgen::stack::RibbonSeg` — and the host draws them. This
//! file holds only the world-level surface the generator itself declares.

use bevy::prelude::*;

/// The generator's water surface (from its `water` op): presence and sea
/// level. Runtime (not build-time) so a hot-reload can switch worlds.
#[derive(Resource, Clone, Copy, Default)]
pub struct WaterSurface {
    pub enabled: bool,
    pub level: f32,
}
