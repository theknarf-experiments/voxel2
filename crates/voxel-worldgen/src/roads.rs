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

pub struct SitesLayer {
    pub seed: u64,
    pub site_chance: f32,
}

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
            site: site_center(self.seed, self.site_chance, coord.x, coord.z),
        }
    }
}

pub struct RoadsLayer {
    pub reach: f32,
}

/// Pathfinding corridor half-width around a road's endpoint box: the
/// A* search (and thus every waypoint) is confined to it, which is what
/// the road_ops query padding and the ownership rule rely on.
pub const CORRIDOR_PAD_M: f32 = 192.0;
/// Coarse height sampling for road pathfinding (GeoGridLayer analog).
pub const PATH_HEIGHT_VS: f32 = 8.0;

/// One planned road: a terrain-aware path between two sites.
#[derive(Clone, Debug, PartialEq)]
pub struct Road {
    pub a: Vec2,
    pub b: Vec2,
    /// Waypoints from `a` to `b` (spacing ~ the path grid step).
    pub waypoints: Vec<Vec2>,
}

pub struct RoadsChunk {
    /// Site-to-site roads owned by this chunk (midpoint rule).
    pub roads: Vec<Road>,
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
                .filter(|&&b| b != a && a.distance(b) < self.reach)
                .min_by(|x, y| a.distance_squared(**x).total_cmp(&a.distance_squared(**y)))
            else {
                continue;
            };
            // Each road appears once: owned by its midpoint's chunk, with a
            // canonical endpoint order.
            let (lo, hi) = if (a.x, a.y) <= (b.x, b.y) {
                (a, b)
            } else {
                (b, a)
            };
            if in_own((lo + hi) * 0.5) && !roads.iter().any(|r: &Road| r.a == lo && r.b == hi) {
                // Terrain-aware path over the coarse height field: slope
                // and water penalties make switchbacks and passes emerge.
                let clo = lo.min(hi) - Vec2::splat(CORRIDOR_PAD_M);
                let chi = lo.max(hi) + Vec2::splat(CORRIDOR_PAD_M);
                let waypoints = crate::path::find_path(
                    &|p| terrain_height(p, PATH_HEIGHT_VS),
                    lo,
                    hi,
                    clo,
                    chi,
                    &crate::path::PathParams::default(),
                )
                .unwrap_or_else(|| vec![lo, hi]);
                roads.push(Road {
                    a: lo,
                    b: hi,
                    waypoints,
                });
            }
        }
        RoadsChunk { roads }
    }
}

/// Build the standard planning stack.
/// Road segments overlapping the box `[min, max]` (world xz meters) —
/// the overlapping-bounds query spawners use for clearance (props must
/// not grow on the roadbed).
pub fn roads_near(mgr: &LayerManager, min: Vec2, max: Vec2) -> Vec<[Vec2; 2]> {
    let pad = (ROAD_REACH_M * 0.5 + CORRIDOR_PAD_M + 64.0) as i32;
    let bounds = IAabb::new(
        IVec3::new(min.x as i32 - pad, 0, min.y as i32 - pad),
        IVec3::new(max.x as i32 + pad, 1, max.y as i32 + pad),
    );
    let mut out = Vec::new();
    for (_, chunk) in mgr.get::<RoadsLayer>(bounds).iter() {
        for road in &chunk.roads {
            for seg in road.waypoints.windows(2) {
                let lo = seg[0].min(seg[1]);
                let hi = seg[0].max(seg[1]);
                if lo.x <= max.x && hi.x >= min.x && lo.y <= max.y && hi.y >= min.y {
                    out.push([seg[0], seg[1]]);
                }
            }
        }
    }
    out
}

/// Distance from `p` to the segment `a`-`b`.
pub fn dist_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

pub fn planning_layers(world_seed: u64, site_chance: f32, road_reach: f32) -> LayerManager {
    let mut mgr = LayerManager::new(world_seed);
    mgr.register(SitesLayer {
        seed: world_seed,
        site_chance,
    });
    mgr.register(RoadsLayer { reach: road_reach });
    mgr
}

/// Emit slab ops for every road overlapping `[min, max]`.
pub fn road_ops(mgr: &LayerManager, min: Vec3, max: Vec3) -> Vec<CsgOp> {
    // Roads owned by chunks whose midpoint is within reach of the box,
    // plus the pathfinding corridor the waypoints may wander into.
    let pad = (ROAD_REACH_M * 0.5 + CORRIDOR_PAD_M + 64.0) as i32;
    let bounds = IAabb::new(
        IVec3::new(min.x as i32 - pad, 0, min.z as i32 - pad),
        IVec3::new(max.x as i32 + pad, 1, max.z as i32 + pad),
    );
    let mut out = Vec::new();
    for (_, chunk) in mgr.get::<RoadsLayer>(bounds).iter() {
        for road in &chunk.roads {
            for seg in road.waypoints.windows(2) {
                road_segment_ops(seg[0], seg[1], &mut out);
            }
        }
    }
    out.retain(|op| op.touches(min, max));
    out
}

fn road_segment_ops(a: Vec2, b: Vec2, out: &mut Vec<CsgOp>) {
    let len = a.distance(b);
    if len < 0.01 {
        return;
    }
    let dir = (b - a) / len;
    let steps = (len / 3.2).ceil() as i32;
    for i in 0..steps {
        let t = (i as f32 + 0.5) / steps as f32;
        let p = a + dir * (t * len);
        let y = terrain_height(p, 1.0);
        out.push(CsgOp::boxy(
            Vec3::new(p.x, y - 0.15, p.y),
            Vec3::new(2.4, 0.5, 2.4),
            0.0,
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
        let mgr = planning_layers(0, 0.32, 700.0);
        let bounds = IAabb::new(IVec3::new(-6000, 0, -6000), IVec3::new(6000, 1, 6000));
        let mut total = 0;
        let mut pairs_a = Vec::new();
        for (coord, chunk) in mgr.get::<RoadsLayer>(bounds).iter() {
            for road in &chunk.roads {
                let (a, b) = (road.a, road.b);
                total += 1;
                pairs_a.push((coord, road.waypoints.clone()));
                // Both endpoints are genuine sites, joined by the path.
                let ca = (a / CELL_M as f32).floor();
                let cb = (b / CELL_M as f32).floor();
                assert_eq!(site_center(0, 0.32, ca.x as i32, ca.y as i32), Some(a));
                assert_eq!(site_center(0, 0.32, cb.x as i32, cb.y as i32), Some(b));
                assert!(a.distance(b) < ROAD_REACH_M);
                assert_eq!(road.waypoints.first(), Some(&a));
                assert_eq!(road.waypoints.last(), Some(&b));
                // Waypoints stay inside the declared pathfinding corridor
                // (the padding math road_ops depends on).
                let lo = a.min(b) - Vec2::splat(CORRIDOR_PAD_M);
                let hi = a.max(b) + Vec2::splat(CORRIDOR_PAD_M);
                for w in &road.waypoints {
                    assert!(
                        w.x >= lo.x && w.x <= hi.x && w.y >= lo.y && w.y <= hi.y,
                        "waypoint {w:?} escapes corridor {lo:?}..{hi:?}"
                    );
                }
                // Terrain-aware: bounded per-segment climb over the coarse
                // height field the pathfinder used.
                for seg in road.waypoints.windows(2) {
                    let rise = (terrain_height(seg[1], PATH_HEIGHT_VS)
                        - terrain_height(seg[0], PATH_HEIGHT_VS))
                    .abs();
                    let run = seg[0].distance(seg[1]).max(0.001);
                    assert!(
                        rise / run < 1.2,
                        "road climbs a cliff: rise {rise:.1} over run {run:.1}"
                    );
                }
            }
        }
        assert!(total > 0, "no roads in 12 km x 12 km");

        // Regenerating from scratch matches.
        let mgr2 = planning_layers(0, 0.32, 700.0);
        let mut pairs_b = Vec::new();
        for (coord, chunk) in mgr2.get::<RoadsLayer>(bounds).iter() {
            for road in &chunk.roads {
                pairs_b.push((coord, road.waypoints.clone()));
            }
        }
        assert_eq!(pairs_a, pairs_b);
    }

    #[test]
    fn roads_near_returns_overlapping_segments_only() {
        let mgr = planning_layers(0, 0.32, 700.0);
        // A wide sweep to find any road, then query a small box around
        // one of its interior waypoints.
        let bounds = IAabb::new(IVec3::new(-6000, 0, -6000), IVec3::new(6000, 1, 6000));
        let mut some_mid = None;
        for (_, chunk) in mgr.get::<RoadsLayer>(bounds).iter() {
            for road in &chunk.roads {
                if road.waypoints.len() > 4 {
                    some_mid = Some(road.waypoints[road.waypoints.len() / 2]);
                }
            }
        }
        let mid = some_mid.expect("no road with interior waypoints");
        let lo = mid - Vec2::splat(30.0);
        let hi = mid + Vec2::splat(30.0);
        let segs = roads_near(&mgr, lo, hi);
        assert!(
            !segs.is_empty(),
            "query box around a road waypoint returns no segments"
        );
        // Every returned segment actually overlaps the (padded) box.
        for [a, b] in &segs {
            let smin = a.min(*b);
            let smax = a.max(*b);
            assert!(
                smin.x <= hi.x && smax.x >= lo.x && smin.y <= hi.y && smax.y >= lo.y,
                "segment {a:?}-{b:?} does not overlap box"
            );
        }
        // A far-away empty box returns nothing near the road.
        let far = roads_near(&mgr, mid + Vec2::splat(50_000.0), mid + Vec2::splat(50_060.0));
        for [a, b] in &far {
            assert!(a.distance(mid) > 1000.0 && b.distance(mid) > 1000.0);
        }
    }

    #[test]
    fn dist_to_segment_basics() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 0.0);
        assert!((dist_to_segment(Vec2::new(5.0, 3.0), a, b) - 3.0).abs() < 1e-5);
        assert!((dist_to_segment(Vec2::new(-4.0, 0.0), a, b) - 4.0).abs() < 1e-5);
        assert!((dist_to_segment(Vec2::new(13.0, 4.0), a, b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn road_ops_follow_terrain() {
        let mgr = planning_layers(0, 0.32, 700.0);
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
