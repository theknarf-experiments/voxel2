//! World tuning parameters, settable from level definitions.
//!
//! Stored as process globals so the dozens of mirror call sites
//! (vegetation, planning, collision, scouting) stay signature-free; the
//! level presenter sets them once at startup and again on hot-reload, and
//! the GPU receives the same numbers through the `WorldTuning` uniform —
//! keeping the CPU/GPU twins in sync by construction.

use std::sync::RwLock;

/// Terrain heightfield bands (scale = cycles/m, amp = meters).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainParams {
    pub continents_scale: f32,
    pub continents_amp: f32,
    pub mountains_scale: f32,
    pub mountains_amp: f32,
    pub rolling_scale: f32,
    pub rolling_amp: f32,
    pub detail_scale: f32,
    pub detail_amp: f32,
    pub offset: f32,
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            continents_scale: 0.00005,
            continents_amp: 800.0,
            mountains_scale: 0.0008,
            mountains_amp: 420.0,
            rolling_scale: 0.01,
            rolling_amp: 36.0,
            detail_scale: 0.06,
            detail_amp: 5.0,
            offset: -8.0,
        }
    }
}

/// Megastructure lattice dimensions and probabilities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MegaParams {
    pub floor_spacing: f32,
    pub pillar_spacing: f32,
    pub wall_spacing: f32,
    pub shaft_spacing: f32,
    pub wall_chance: f32,
    pub opening_chance: f32,
}

impl Default for MegaParams {
    fn default() -> Self {
        Self {
            floor_spacing: 44.0,
            pillar_spacing: 34.0,
            wall_spacing: 104.0,
            shaft_spacing: 288.0,
            wall_chance: 0.45,
            opening_chance: 0.16,
        }
    }
}

static TERRAIN: RwLock<TerrainParams> = RwLock::new(TerrainParams {
    continents_scale: 0.00005,
    continents_amp: 800.0,
    mountains_scale: 0.0008,
    mountains_amp: 420.0,
    rolling_scale: 0.01,
    rolling_amp: 36.0,
    detail_scale: 0.06,
    detail_amp: 5.0,
    offset: -8.0,
});

static MEGA: RwLock<MegaParams> = RwLock::new(MegaParams {
    floor_spacing: 44.0,
    pillar_spacing: 34.0,
    wall_spacing: 104.0,
    shaft_spacing: 288.0,
    wall_chance: 0.45,
    opening_chance: 0.16,
});

pub fn set_terrain_params(p: TerrainParams) {
    *TERRAIN.write().unwrap() = p;
}

pub fn terrain_params() -> TerrainParams {
    *TERRAIN.read().unwrap()
}

pub fn set_mega_params(p: MegaParams) {
    *MEGA.write().unwrap() = p;
}

pub fn mega_params() -> MegaParams {
    *MEGA.read().unwrap()
}
