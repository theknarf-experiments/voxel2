//! Main-world LOD controller: a retained chunk octree around the camera.
//!
//! Leaves are the currently-shown chunks. A leaf closer than
//! `split_k × edge` refines into its 8 children; 8 sibling leaves farther
//! than `merge_k × parent_edge` coarsen back (the gap between the constants
//! is the hysteresis band that prevents flicker at thresholds).
//!
//! Swaps are ready-before-swap: the replacement chunks are requested hidden,
//! and only when *all* of them are drawable (meshed or classified empty)
//! does one command batch show them and free the replaced chunk — so the
//! terrain never has holes and never shows two LODs of the same region.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::math::DVec3;
use bevy::prelude::*;
use voxel_core::csg::CsgOp;
use voxel_core::ChunkKey;
use voxel_render::{ChunkCommand, ChunkCommandQueue, ChunkReadyChannel, SharedRenderStats};

/// Optional hook supplying planning-layer CSG ops for a requested chunk
/// (already AABB-culled to the chunk). Installed by the app/worldgen.
#[derive(Resource, Default)]
pub struct ChunkOpsProvider(pub Option<Arc<dyn Fn(ChunkKey) -> Vec<CsgOp> + Send + Sync>>);

fn request(
    queue: &ChunkCommandQueue,
    provider: &ChunkOpsProvider,
    key: ChunkKey,
    show_on_ready: bool,
) {
    let ops = provider
        .0
        .as_ref()
        .map(|f| f(key))
        .filter(|v| !v.is_empty())
        .map(Arc::new);
    queue.push(ChunkCommand::Request { key, show_on_ready, ops });
}

/// LOD configuration.
#[derive(Resource)]
pub struct LodConfig {
    /// Coarsest chunk level. Edge = 32 · 2^max_level meters.
    pub max_level: u8,
    /// Radius, in top-level chunks, kept loaded around the camera.
    pub top_radius: i32,
    /// Vertical range of top-level chunks (inclusive).
    pub top_y: (i32, i32),
    /// Split when camera distance < split_k × edge.
    pub split_k: f64,
    /// Merge when camera distance > merge_k × parent edge. Must exceed
    /// `split_k` for hysteresis.
    pub merge_k: f64,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            // Top chunks of 8.2 km; ring radius 3 → ~25 km view distance.
            max_level: 8,
            top_radius: 3,
            top_y: (-1, 0),
            split_k: 2.5,
            merge_k: 3.0,
        }
    }
}

#[derive(Resource, Default)]
struct LodTree {
    /// Currently-shown chunks (some may still be generating right after
    /// being requested with show-on-ready).
    leaves: HashSet<ChunkKey>,
    /// Parent → its 8 requested children, awaiting readiness.
    splitting: HashMap<ChunkKey, [ChunkKey; 8]>,
    /// Parent (requested, hidden) → the 8 active children it will replace.
    merging: HashMap<ChunkKey, [ChunkKey; 8]>,
    /// Chunks the render world reported drawable.
    ready: HashSet<ChunkKey>,
    /// Top-level cells whose subtree is live.
    top_cells: HashSet<IVec3>,
}

pub struct VoxelStreamingPlugin;

impl Plugin for VoxelStreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LodConfig>()
            .init_resource::<ChunkOpsProvider>()
            .init_resource::<LodTree>()
            .add_systems(Update, (lod_tick, hud_stats));
    }
}

fn aabb_distance(camera: DVec3, key: ChunkKey) -> f64 {
    let min = key.min_corner_m();
    let max = min + DVec3::splat(key.edge_m());
    camera.distance(camera.clamp(min, max))
}

/// The top-level ancestor cell of a key.
fn top_ancestor(key: ChunkKey, max_level: u8) -> IVec3 {
    let mut k = key;
    while k.level < max_level {
        k = k.parent();
    }
    k.pos
}

#[allow(clippy::too_many_arguments)]
fn lod_tick(
    config: Res<LodConfig>,
    mut tree: ResMut<LodTree>,
    queue: Res<ChunkCommandQueue>,
    ops_provider: Res<ChunkOpsProvider>,
    ready_rx: Res<ChunkReadyChannel>,
    cameras: Query<&Transform, With<Camera3d>>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let camera = camera.translation.as_dvec3();
    let tree = &mut *tree;

    // 1. Absorb readiness notifications.
    for key in ready_rx.rx.try_iter() {
        tree.ready.insert(key);
    }

    // 2. Complete splits whose 8 children are all drawable: atomic swap.
    let done_splits: Vec<(ChunkKey, [ChunkKey; 8])> = tree
        .splitting
        .iter()
        .filter(|(_, children)| children.iter().all(|c| tree.ready.contains(c)))
        .map(|(p, c)| (*p, *c))
        .collect();
    for (parent, children) in done_splits {
        tree.splitting.remove(&parent);
        for child in children {
            queue.push(ChunkCommand::Show(child));
            tree.leaves.insert(child);
        }
        tree.leaves.remove(&parent);
        tree.ready.remove(&parent);
        queue.push(ChunkCommand::Free(parent));
    }

    // 3. Complete merges whose parent is drawable: atomic swap.
    let done_merges: Vec<(ChunkKey, [ChunkKey; 8])> = tree
        .merging
        .iter()
        .filter(|(parent, _)| tree.ready.contains(parent))
        .map(|(p, c)| (*p, *c))
        .collect();
    for (parent, children) in done_merges {
        tree.merging.remove(&parent);
        queue.push(ChunkCommand::Show(parent));
        tree.leaves.insert(parent);
        for child in children {
            tree.leaves.remove(&child);
            tree.ready.remove(&child);
            queue.push(ChunkCommand::Free(child));
        }
    }

    // 4. Top-level ring maintenance.
    let top_edge = ChunkKey::new(config.max_level, IVec3::ZERO).edge_m();
    let center_x = (camera.x / top_edge).floor() as i32;
    let center_z = (camera.z / top_edge).floor() as i32;
    let r = config.top_radius;
    for dz in -r..=r {
        for dx in -r..=r {
            for y in config.top_y.0..=config.top_y.1 {
                let cell = IVec3::new(center_x + dx, y, center_z + dz);
                if tree.top_cells.insert(cell) {
                    let key = ChunkKey::new(config.max_level, cell);
                    request(&queue, &ops_provider, key, true);
                    tree.leaves.insert(key);
                }
            }
        }
    }
    let keep = r + 1;
    let stale: Vec<IVec3> = tree
        .top_cells
        .iter()
        .filter(|c| (c.x - center_x).abs() > keep || (c.z - center_z).abs() > keep)
        .copied()
        .collect();
    for cell in stale {
        tree.top_cells.remove(&cell);
        free_subtree(tree, &queue, cell, config.max_level);
    }

    // 5. Splits: leaves too close for their level refine.
    let candidates: Vec<ChunkKey> = tree
        .leaves
        .iter()
        .filter(|leaf| {
            leaf.level > 0
                && !tree.splitting.contains_key(leaf)
                && !in_merge(tree, **leaf)
                && aabb_distance(camera, **leaf) < config.split_k * leaf.edge_m()
        })
        .copied()
        .collect();
    for leaf in candidates {
        let children = leaf.children();
        for child in children {
            request(&queue, &ops_provider, child, false);
        }
        tree.splitting.insert(leaf, children);
    }

    // 6. Merges: complete sibling sets far enough away coarsen.
    let mut sibling_count: HashMap<ChunkKey, u8> = HashMap::new();
    for leaf in &tree.leaves {
        if leaf.level >= config.max_level {
            continue;
        }
        *sibling_count.entry(leaf.parent()).or_default() += 1;
    }
    for (parent, count) in sibling_count {
        if count != 8
            || tree.merging.contains_key(&parent)
            || aabb_distance(camera, parent) <= config.merge_k * parent.edge_m()
        {
            continue;
        }
        let children = parent.children();
        // Children mid-split cannot merge.
        if children.iter().any(|c| tree.splitting.contains_key(c)) {
            continue;
        }
        request(&queue, &ops_provider, parent, false);
        tree.merging.insert(parent, children);
    }
}

fn in_merge(tree: &LodTree, key: ChunkKey) -> bool {
    tree.merging.contains_key(&key)
        || (key.level < 30 && tree.merging.contains_key(&key.parent()))
}

/// Free every requested chunk whose subtree hangs under `cell`.
fn free_subtree(tree: &mut LodTree, queue: &ChunkCommandQueue, cell: IVec3, max_level: u8) {
    let in_subtree = |key: &ChunkKey| top_ancestor(*key, max_level) == cell;

    let mut to_free: HashSet<ChunkKey> = HashSet::new();
    tree.leaves.retain(|k| {
        let stale = in_subtree(k);
        if stale {
            to_free.insert(*k);
        }
        !stale
    });
    let stale_splits: Vec<ChunkKey> = tree.splitting.keys().filter(|k| in_subtree(k)).copied().collect();
    for parent in stale_splits {
        if let Some(children) = tree.splitting.remove(&parent) {
            to_free.extend(children);
        }
    }
    let stale_merges: Vec<ChunkKey> = tree.merging.keys().filter(|k| in_subtree(k)).copied().collect();
    for parent in stale_merges {
        tree.merging.remove(&parent);
        to_free.insert(parent);
    }
    for key in to_free {
        tree.ready.remove(&key);
        queue.push(ChunkCommand::Free(key));
    }
    tree.ready.retain(|k| !in_subtree(k));
}

fn hud_stats(
    stats: Res<SharedRenderStats>,
    tree: Res<LodTree>,
    hud: Option<ResMut<voxel_debug::DebugHudExtra>>,
) {
    let Some(mut hud) = hud else {
        return;
    };
    let Ok(s) = stats.0.lock() else {
        return;
    };
    hud.0.push(format!(
        "chunks: {} tracked | {} meshed | {} drawn | {} pending",
        s.tracked, s.meshed, s.drawn, s.awaiting
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

    // Leaf histogram by level (finest first).
    let mut histo: HashMap<u8, usize> = HashMap::new();
    for leaf in &tree.leaves {
        *histo.entry(leaf.level).or_default() += 1;
    }
    let mut levels: Vec<u8> = histo.keys().copied().collect();
    levels.sort_unstable();
    let parts: Vec<String> = levels.iter().map(|l| format!("L{l}:{}", histo[l])).collect();
    hud.0.push(format!(
        "leaves: {} [{}] | splits: {} merges: {}",
        tree.leaves.len(),
        parts.join(" "),
        tree.splitting.len(),
        tree.merging.len(),
    ));
}
