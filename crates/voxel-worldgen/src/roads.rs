//! Roads between ruin sites — the first production `voxel-layers` stack.
//!
//! `SitesLayer` (256 m planar chunks) exposes ruin site positions.
//! `RoadsLayer` depends on it with 768 m padding: each site connects to its
//! nearest neighbor within reach, and a road is *owned* by the chunk
//! containing its midpoint (no duplicates, deterministic under any
//! generation order — the LayerProcGen contextual-generation pattern).
//!
//! Road geometry: chains of shallow stone slabs following the terrain,
//! with a gentle S-wiggle so paths don't read as ruler lines.

use glam::{IVec3, Vec2, Vec3};
use voxel_core::csg::CsgOp;
use voxel_layers::{Dep, IAabb, Layer, LayerCtx, LayerManager};

use crate::ruins::{site_center, MAT_STONE};
use crate::terrain_height;

const CELL_M: i32 = 256;
const ROAD_REACH_M: f32 = 700.0;

pub struct SitesLayer;

pub struct SitesChunk {
    pub site: Option<Vec2>,
}

impl Layer for SitesLayer {
    type Chunk = SitesChunk;
    const NAME: &'static str = "worldgen/sites";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CELL_M, 0, CELL_M)
    }

    fn generate(&self, _ctx: &LayerCtx<'_, Self>, coord: IVec3) -> SitesChunk {
        SitesChunk {
            site: site_center(coord.x, coord.z),
        }
    }
}

pub struct RoadsLayer;

pub struct RoadsChunk {
    /// Site-to-site connections owned by this chunk (midpoint rule).
    pub roads: Vec<(Vec2, Vec2)>,
}

impl Layer for RoadsLayer {
    type Chunk = RoadsChunk;
    const NAME: &'static str = "worldgen/roads";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CELL_M, 0, CELL_M)
    }

    fn dependencies(&self) -> Vec<Dep> {
        // A road midpoint in this chunk implies endpoints within
        // ROAD_REACH_M/2 + site jitter; pad generously.
        vec![Dep::of::<SitesLayer>(IVec3::new(768, 0, 768))]
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> RoadsChunk {
        let own = ctx.chunk_bounds();
        let padded = own.inflate(IVec3::new(768, 0, 768));
        let view = ctx.get::<SitesLayer>(padded);

        let sites: Vec<Vec2> = view.iter().filter_map(|(_, c)| c.site).collect();
        let in_own = |p: Vec2| {
            p.x >= own.min.x as f32
                && p.x < own.max.x as f32
                && p.y >= own.min.z as f32
                && p.y < own.max.z as f32
        };

        let mut roads = Vec::new();
        for &a in &sites {
            let Some(&b) = sites
                .iter()
                .filter(|&&b| b != a && a.distance(b) < ROAD_REACH_M)
                .min_by(|x, y| a.distance_squared(**x).total_cmp(&a.distance_squared(**y)))
            else {
                continue;
            };
            // Each road appears once: owned by its midpoint's chunk, with a
            // canonical endpoint order.
            let (lo, hi) = if (a.x, a.y) <= (b.x, b.y) { (a, b) } else { (b, a) };
            if in_own((lo + hi) * 0.5) && !roads.contains(&(lo, hi)) {
                roads.push((lo, hi));
            }
        }
        RoadsChunk { roads }
    }
}

/// Build the standard planning stack.
pub fn planning_layers(world_seed: u64) -> LayerManager {
    let mut mgr = LayerManager::new(world_seed);
    mgr.register(SitesLayer);
    mgr.register(RoadsLayer);
    mgr
}

/// Emit slab ops for every road overlapping `[min, max]`.
pub fn road_ops(mgr: &LayerManager, min: Vec3, max: Vec3) -> Vec<CsgOp> {
    // Roads owned by chunks whose midpoint is within reach of the box.
    let pad = (ROAD_REACH_M * 0.5 + 64.0) as i32;
    let bounds = IAabb::new(
        IVec3::new(min.x as i32 - pad, 0, min.z as i32 - pad),
        IVec3::new(max.x as i32 + pad, 1, max.z as i32 + pad),
    );
    let mut out = Vec::new();
    for (_, chunk) in mgr.get::<RoadsLayer>(bounds).iter() {
        for &(a, b) in &chunk.roads {
            road_segment_ops(a, b, &mut out);
        }
    }
    out.retain(|op| op.touches(min, max));
    out
}

fn road_segment_ops(a: Vec2, b: Vec2, out: &mut Vec<CsgOp>) {
    let len = a.distance(b);
    let dir = (b - a) / len;
    let perp = Vec2::new(-dir.y, dir.x);
    let steps = (len / 3.2).ceil() as i32;
    for i in 0..steps {
        let t = (i as f32 + 0.5) / steps as f32;
        // Gentle S-curve so paths meander.
        let wiggle = (t * std::f32::consts::TAU).sin() * len * 0.03;
        let p = a + dir * (t * len) + perp * wiggle;
        let y = terrain_height(p, 1.0);
        let yaw = (dir + perp * (t * std::f32::consts::TAU).cos() * 0.19)
            .to_angle();
        out.push(CsgOp::boxy(
            Vec3::new(p.x, y - 0.15, p.y),
            Vec3::new(2.2, 0.5, 1.2),
            -yaw,
            MAT_STONE,
            false,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roads_connect_real_sites_deterministically() {
        let mgr = planning_layers(0);
        let bounds = IAabb::new(IVec3::new(-6000, 0, -6000), IVec3::new(6000, 1, 6000));
        let mut total = 0;
        let mut pairs_a = Vec::new();
        for (coord, chunk) in mgr.get::<RoadsLayer>(bounds).iter() {
            for &(a, b) in &chunk.roads {
                total += 1;
                pairs_a.push((coord, a, b));
                // Both endpoints are genuine sites.
                let ca = (a / CELL_M as f32).floor();
                let cb = (b / CELL_M as f32).floor();
                assert_eq!(site_center(ca.x as i32, ca.y as i32), Some(a));
                assert_eq!(site_center(cb.x as i32, cb.y as i32), Some(b));
                assert!(a.distance(b) < ROAD_REACH_M);
            }
        }
        assert!(total > 0, "no roads in 12 km x 12 km");

        // Regenerating from scratch matches.
        let mgr2 = planning_layers(0);
        let mut pairs_b = Vec::new();
        for (coord, chunk) in mgr2.get::<RoadsLayer>(bounds).iter() {
            for &(a, b) in &chunk.roads {
                pairs_b.push((coord, a, b));
            }
        }
        assert_eq!(pairs_a, pairs_b);
    }

    #[test]
    fn road_ops_follow_terrain() {
        let mgr = planning_layers(0);
        let ops = road_ops(
            &mgr,
            Vec3::new(-6000.0, -100.0, -6000.0),
            Vec3::new(6000.0, 500.0, 6000.0),
        );
        for op in &ops {
            let ground = terrain_height(Vec2::new(op.center[0], op.center[2]), 1.0);
            assert!((op.center[1] - ground).abs() < 2.0, "slab far from ground");
        }
    }
}
