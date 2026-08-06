//! Water DATA: where the water is. How it *looks* is the host's business —
//! the pipeline and shader that draw it live in the app (see the demo's
//! `water.rs`), so a game can replace the whole water look without
//! touching the engine.

use bevy::prelude::*;

/// One river water segment for the GPU (layout twins the WGSL RiverSeg).
#[derive(Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RiverSegGpu {
    /// a.xz | b.xz (world meters).
    pub ab: [f32; 4],
    /// half width | level at a | level at b | unused.
    pub geo: [f32; 4],
    /// river tint rgb | unused.
    pub color: [f32; 4],
}

/// River segments near the camera, maintained by the engine's streamer.
/// `generation` bumps on change so a renderer can re-upload.
#[derive(Resource, Clone, Default)]
pub struct RiverWater {
    pub segments: Vec<RiverSegGpu>,
    pub generation: u64,
}

/// The generator's water surface (from its `water` op): presence and sea
/// level. Runtime (not build-time) so a hot-reload can switch worlds.
#[derive(Resource, Clone, Copy, Default)]
pub struct WaterSurface {
    pub enabled: bool,
    pub level: f32,
}
