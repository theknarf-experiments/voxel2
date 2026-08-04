//! The LayerProcGen acid test: contextual generation (connections reach
//! across chunk boundaries into neighbor chunks' points) must be
//! byte-identical regardless of thread count, request order, or cache state.

use std::sync::Arc;

use glam::{IVec3, Vec2};
use voxel_core::seed::Rng;
use voxel_layers::{Dep, IAabb, Layer, LayerCtx, LayerManager};

const CHUNK_M: i32 = 256;
const POINTS_PER_CHUNK: usize = 8;

/// Lower layer: deterministic scatter of points per 256 m planar chunk.
struct PointsLayer;

struct PointsChunk {
    points: Vec<Vec2>, // world-space xz, meters
}

impl Layer for PointsLayer {
    type Chunk = PointsChunk;
    const NAME: &'static str = "test/points";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CHUNK_M, 0, CHUNK_M)
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> PointsChunk {
        let bounds = ctx.chunk_bounds();
        let mut rng = ctx.rng();
        let origin = Vec2::new(bounds.min.x as f32, bounds.min.z as f32);
        let points = (0..POINTS_PER_CHUNK)
            .map(|_| origin + Vec2::new(rng.next_f32(), rng.next_f32()) * CHUNK_M as f32)
            .collect();
        PointsChunk { points }
    }
}

/// Upper layer: each own point connects to the nearest point in the padded
/// neighborhood — contextual generation across chunk boundaries.
struct ConnectionsLayer;

struct ConnectionsChunk {
    connections: Vec<(Vec2, Vec2)>,
}

impl Layer for ConnectionsLayer {
    type Chunk = ConnectionsChunk;
    const NAME: &'static str = "test/connections";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CHUNK_M, 0, CHUNK_M)
    }

    fn dependencies(&self) -> Vec<Dep> {
        vec![Dep::of::<PointsLayer>(IVec3::new(CHUNK_M, 0, CHUNK_M))]
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> ConnectionsChunk {
        let own = ctx.chunk_bounds();
        let padded = own.inflate(IVec3::new(CHUNK_M, 0, CHUNK_M));
        let view = ctx.get::<PointsLayer>(padded);

        let mut all_points: Vec<Vec2> = Vec::new();
        let mut own_points: Vec<Vec2> = Vec::new();
        for (coord, chunk) in view.iter() {
            all_points.extend_from_slice(&chunk.points);
            let is_own = coord.x * CHUNK_M == own.min.x && coord.z * CHUNK_M == own.min.z;
            if is_own {
                own_points.extend_from_slice(&chunk.points);
            }
        }

        let connections = own_points
            .iter()
            .map(|&p| {
                let nearest = all_points
                    .iter()
                    .copied()
                    .filter(|&q| q != p)
                    .min_by(|a, b| {
                        a.distance_squared(p)
                            .partial_cmp(&b.distance_squared(p))
                            .unwrap()
                    })
                    .unwrap();
                (p, nearest)
            })
            .collect();
        ConnectionsChunk { connections }
    }
}

fn build_manager() -> LayerManager {
    let mut mgr = LayerManager::new(0xDEADBEEF);
    mgr.register(PointsLayer);
    mgr.register(ConnectionsLayer);
    mgr
}

/// Flatten a region's connections into comparable bytes.
fn snapshot(mgr: &LayerManager, coords: &[IVec3]) -> Vec<u8> {
    let mut out = Vec::new();
    for &coord in coords {
        let chunk = mgr.get_chunk::<ConnectionsLayer>(coord);
        for (a, b) in &chunk.connections {
            out.extend_from_slice(&a.x.to_le_bytes());
            out.extend_from_slice(&a.y.to_le_bytes());
            out.extend_from_slice(&b.x.to_le_bytes());
            out.extend_from_slice(&b.y.to_le_bytes());
        }
    }
    out
}

fn region_coords() -> Vec<IVec3> {
    let mut coords = Vec::new();
    for z in -3..3 {
        for x in -3..3 {
            coords.push(IVec3::new(x, 0, z));
        }
    }
    coords
}

#[test]
fn dependencies_generate_recursively_on_demand() {
    let mgr = build_manager();
    assert_eq!(mgr.cached_chunks(), 0);
    let chunk = mgr.get_chunk::<ConnectionsLayer>(IVec3::ZERO);
    assert_eq!(chunk.connections.len(), POINTS_PER_CHUNK);
    // 1 connections chunk + its 3x3 padded points neighborhood.
    assert_eq!(mgr.cached_chunks(), 1 + 9);
}

#[test]
fn connections_cross_chunk_boundaries() {
    let mgr = build_manager();
    // Some connection in a 4x4 region must span a chunk boundary, otherwise
    // the padded read is not actually contextual.
    let mut crossing = 0;
    for z in 0..4 {
        for x in 0..4 {
            let chunk = mgr.get_chunk::<ConnectionsLayer>(IVec3::new(x, 0, z));
            for (a, b) in &chunk.connections {
                let ca = (a / CHUNK_M as f32).floor();
                let cb = (b / CHUNK_M as f32).floor();
                if ca != cb {
                    crossing += 1;
                }
            }
        }
    }
    assert!(crossing > 0, "no connection crossed a chunk boundary");
}

#[test]
fn identical_across_thread_counts_and_orders() {
    let coords = region_coords();

    // Reference: single thread, ascending order.
    let reference = {
        let mgr = build_manager();
        snapshot(&mgr, &coords)
    };

    for threads in [2usize, 4, 8] {
        let mgr = Arc::new(build_manager());

        // Each thread generates the whole region in its own shuffled order,
        // racing the others chunk by chunk.
        std::thread::scope(|scope| {
            for t in 0..threads {
                let mgr = Arc::clone(&mgr);
                let mut order = coords.clone();
                let mut rng = Rng::new(t as u64 + 1);
                for i in (1..order.len()).rev() {
                    order.swap(i, rng.next_range(i as u32 + 1) as usize);
                }
                scope.spawn(move || {
                    for &coord in &order {
                        let _ = mgr.get_chunk::<ConnectionsLayer>(coord);
                    }
                });
            }
        });

        let result = snapshot(&mgr, &coords);
        assert_eq!(
            result, reference,
            "generation differed with {threads} racing threads"
        );

        // And regenerating from a cold cache must also match.
        mgr.evict_all();
        assert_eq!(snapshot(&mgr, &coords), reference, "cold regen differed");
    }
}
