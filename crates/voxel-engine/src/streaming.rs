//! Main-world chunk streaming: maintains the desired set of chunks around
//! the camera and feeds requests/frees to the render-world pipeline.

use std::collections::HashSet;

use bevy::prelude::*;
use voxel_render::{ChunkCommandQueue, SharedRenderStats};

/// Streaming configuration for the fixed-LOD (M4) terrain.
#[derive(Resource)]
pub struct StreamingConfig {
    /// Horizontal radius, in chunks, around the camera.
    pub radius_xz: i32,
    /// Extra hysteresis ring before a chunk is released.
    pub release_margin: i32,
    /// Vertical chunk range (inclusive) to keep loaded.
    pub y_range: (i32, i32),
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            radius_xz: 12,
            release_margin: 2,
            y_range: (-2, 2),
        }
    }
}

#[derive(Resource, Default)]
struct TrackedChunks(HashSet<IVec3>);

pub struct VoxelStreamingPlugin;

impl Plugin for VoxelStreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StreamingConfig>()
            .init_resource::<TrackedChunks>()
            .add_systems(Update, (stream_chunks, hud_stats));
    }
}

const CHUNK_M: f32 = 32.0;

fn stream_chunks(
    config: Res<StreamingConfig>,
    mut tracked: ResMut<TrackedChunks>,
    queue: Res<ChunkCommandQueue>,
    cameras: Query<&Transform, With<Camera3d>>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let center = IVec3::new(
        (camera.translation.x / CHUNK_M).floor() as i32,
        0,
        (camera.translation.z / CHUNK_M).floor() as i32,
    );

    // Request everything inside the radius.
    let r = config.radius_xz;
    for dz in -r..=r {
        for dx in -r..=r {
            if dx * dx + dz * dz > r * r {
                continue;
            }
            for y in config.y_range.0..=config.y_range.1 {
                let coord = IVec3::new(center.x + dx, y, center.z + dz);
                if tracked.0.insert(coord) {
                    queue.request(coord);
                }
            }
        }
    }

    // Release everything outside radius + margin.
    let keep = r + config.release_margin;
    let keep_sq = keep * keep;
    tracked.0.retain(|coord| {
        let dx = coord.x - center.x;
        let dz = coord.z - center.z;
        let inside = dx * dx + dz * dz <= keep_sq
            && coord.y >= config.y_range.0
            && coord.y <= config.y_range.1;
        if !inside {
            queue.free(*coord);
        }
        inside
    });
}

fn hud_stats(
    stats: Res<SharedRenderStats>,
    hud: Option<ResMut<voxel_debug::DebugHudExtra>>,
) {
    let Some(mut hud) = hud else {
        return;
    };
    let Ok(s) = stats.0.lock() else {
        return;
    };
    hud.0.push(format!(
        "chunks: {} tracked | {} meshed | {} empty | {} pending",
        s.tracked, s.meshed, s.empty_classified, s.awaiting
    ));
    let occ: Vec<String> = s
        .slab_occupancy
        .iter()
        .map(|(free, total)| format!("{}/{}", total - free, total))
        .collect();
    hud.0.push(format!(
        "arena free: {} | slab used: [{}]",
        s.arena_free,
        occ.join(", ")
    ));
}
