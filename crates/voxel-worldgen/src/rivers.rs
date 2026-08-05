//! Rivers: layered planning (LayerProcGen style). Springs are rare
//! high-ground sites; each river is a deterministic steepest-descent walk
//! over the coarse height field down to the sea (or a pit, where it ends
//! in a pond). The course carves a bed and fills a water-material slab.

use glam::{IVec3, Vec2, Vec3};
use voxel_core::csg::CsgOp;
use voxel_layers::{Dep, IAabb, Layer, LayerCtx, LayerManager};
use voxel_core::seed::{chunk_seed, Rng};

use crate::terrain_height;

const CELL_M: i32 = 512;
const SPRING_SEED: u64 = 0x51_BE_5;
/// Coarse height sampling for the descent (matches roads' GeoGrid vs).
pub const FLOW_HEIGHT_VS: f32 = 8.0;
/// Descent step (meters).
pub const FLOW_STEP_M: f32 = 8.0;
const MAX_STEPS: usize = 400;
/// Conservative river reach for the cell scan.
pub const REACH_M: f32 = (MAX_STEPS as f32 * FLOW_STEP_M) + 64.0;
/// Water material id in the level's material table.
pub const MAT_RIVER: u32 = 4;

/// Parameters of the descent walk.
pub struct FlowParams {
    pub step_m: f32,
    /// Stop once the height drops to this level (sea).
    pub stop_level: f32,
    pub max_steps: usize,
    /// Maximum lip height above the pond floor the walk may spill over.
    pub max_spill_rise: f32,
}

impl Default for FlowParams {
    fn default() -> Self {
        Self {
            step_m: FLOW_STEP_M,
            stop_level: 0.4,
            max_steps: MAX_STEPS,
            max_spill_rise: 7.0,
        }
    }
}

/// Deterministic lattice descent from `start` with pond-and-spill: while
/// downhill neighbors exist the walk takes the steepest one; at a local
/// minimum a bounded Dijkstra looks for the nearest escape route whose
/// lip stays within `max_spill_rise` of the pond floor and which ends
/// LOWER than the pond entry (a shallow pond overflows; a deep basin
/// ends the river as a lake). Ends at the sea (`stop_level`), in a deep
/// pit, or at `max_steps`.
pub fn flow_path(height: &dyn Fn(Vec2) -> f32, start: Vec2, params: &FlowParams) -> Vec<Vec2> {
    let step = params.step_m;
    let node = |p: Vec2| ((p.x / step).round() as i32, (p.y / step).round() as i32);
    let posf = |n: (i32, i32)| Vec2::new(n.0 as f32 * step, n.1 as f32 * step);
    let hn = |n: (i32, i32)| height(posf(n));

    let mut path = vec![start];
    let mut cur = node(start);
    let mut steps = 0usize;
    while steps < params.max_steps {
        let h_here = hn(cur);
        if h_here <= params.stop_level {
            break;
        }
        // Steepest strictly-downhill 8-neighbor.
        let mut best: Option<((i32, i32), f32)> = None;
        for dz in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dz == 0 {
                    continue;
                }
                let n = (cur.0 + dx, cur.1 + dz);
                let h = hn(n);
                if h < h_here && best.is_none_or(|(bn, bh)| h < bh || (h == bh && n < bn)) {
                    best = Some((n, h));
                }
            }
        }
        if let Some((n, _)) = best {
            cur = n;
            path.push(posf(cur));
            steps += 1;
            continue;
        }
        // Pond: bounded Dijkstra for the nearest node lower than the pond
        // entry, over lattice nodes no higher than floor + spill rise.
        let ceiling = h_here + params.max_spill_rise;
        let mut open: std::collections::BinaryHeap<(
            std::cmp::Reverse<(u32, (i32, i32))>,
        )> = Default::default();
        let mut came: std::collections::HashMap<(i32, i32), (i32, i32)> = Default::default();
        came.insert(cur, cur);
        open.push((std::cmp::Reverse((0, cur)),));
        let mut escape = None;
        let mut expanded = 0u32;
        while let Some((std::cmp::Reverse((dist, n)),)) = open.pop() {
            if hn(n) < h_here - 1e-3 {
                escape = Some(n);
                break;
            }
            expanded += 1;
            if expanded > 4_000 {
                break;
            }
            for dz in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dz == 0 {
                        continue;
                    }
                    let m = (n.0 + dx, n.1 + dz);
                    if came.contains_key(&m) || hn(m) > ceiling {
                        continue;
                    }
                    came.insert(m, n);
                    open.push((std::cmp::Reverse((dist + 1, m)),));
                }
            }
        }
        let Some(mut e) = escape else {
            break; // deep basin: the river ends in a lake
        };
        // Splice the escape route (pond crossing) into the path.
        let mut route = vec![e];
        while came[&e] != e {
            e = came[&e];
            route.push(e);
        }
        route.pop(); // current node already in the path
        route.reverse();
        for n in route {
            path.push(posf(n));
            steps += 1;
            cur = n;
        }
    }
    path
}

/// The spring of a cell, if any: rare, high, reasonably flat ground.
pub fn spring(seed: u64, chance: f32, cx: i32, cz: i32) -> Option<Vec2> {
    let mut rng = Rng::new(chunk_seed(SPRING_SEED ^ seed, 0x11, IVec3::new(cx, 0, cz)));
    if rng.next_f32() > chance {
        return None;
    }
    let x = cx as f32 * CELL_M as f32 + 48.0 + rng.next_f32() * (CELL_M as f32 - 96.0);
    let z = cz as f32 * CELL_M as f32 + 48.0 + rng.next_f32() * (CELL_M as f32 - 96.0);
    let p = Vec2::new(x, z);
    let h = terrain_height(p, FLOW_HEIGHT_VS);
    if !(60.0..400.0).contains(&h) {
        return None;
    }
    Some(p)
}

/// Springs layer: one candidate site per 512 m cell.
pub struct SpringsLayer {
    pub seed: u64,
    pub chance: f32,
}

pub struct SpringsChunk {
    pub spring: Option<Vec2>,
}

impl Layer for SpringsLayer {
    type Chunk = SpringsChunk;
    const NAME: &'static str = "worldgen/springs";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CELL_M, 0, CELL_M)
    }

    fn generate(&self, _ctx: &LayerCtx<'_, Self>, coord: IVec3) -> SpringsChunk {
        SpringsChunk {
            spring: spring(self.seed, self.chance, coord.x, coord.z),
        }
    }
}

/// One river: waypoints from spring to mouth, widening downstream, with
/// a monotonically non-increasing water line (running minimum of the
/// terrain along the course — per-segment levels stair-step on slopes
/// and read as floating slabs).
#[derive(Clone, Debug, PartialEq)]
pub struct River {
    pub waypoints: Vec<Vec2>,
    pub levels: Vec<f32>,
}

pub struct RiversChunk {
    pub rivers: Vec<River>,
}

/// Rivers layer: each cell's spring (if any) produces its full course,
/// owned by the spring's cell.
pub struct RiversLayer;

impl Layer for RiversLayer {
    type Chunk = RiversChunk;
    const NAME: &'static str = "worldgen/rivers";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CELL_M, 0, CELL_M)
    }

    fn dependencies(&self) -> Vec<Dep> {
        vec![Dep::of::<SpringsLayer>(IVec3::ZERO)]
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> RiversChunk {
        let own = ctx.chunk_bounds();
        let view = ctx.get::<SpringsLayer>(own);
        let mut rivers = Vec::new();
        for (_, chunk) in view.iter() {
            if let Some(start) = chunk.spring {
                let waypoints = flow_path(
                    &|p| terrain_height(p, FLOW_HEIGHT_VS),
                    start,
                    &FlowParams::default(),
                );
                if waypoints.len() >= 6 {
                    let mut level = f32::MAX;
                    let levels = waypoints
                        .iter()
                        .map(|p| {
                            level = level.min(terrain_height(*p, FLOW_HEIGHT_VS) - 0.35);
                            level
                        })
                        .collect();
                    rivers.push(River { waypoints, levels });
                }
            }
        }
        RiversChunk { rivers }
    }
}

/// Segment index: river segments bucketed by the cell that owns their
/// midpoint. Rivers travel far from their spring cell (REACH_M), so
/// querying RiversLayer directly forces a huge scan per request; the
/// index pays that scan ONCE per cell at generation time (off the hot
/// path) and makes every ops/clearance query local.
pub struct RiverIndexLayer;

pub struct RiverIndexChunk {
    /// (segment, half width, water level at each end) owned by this cell
    /// (midpoint rule).
    pub segments: Vec<([Vec2; 2], f32, [f32; 2])>,
}

impl Layer for RiverIndexLayer {
    type Chunk = RiverIndexChunk;
    const NAME: &'static str = "worldgen/river-index";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CELL_M, 0, CELL_M)
    }

    fn dependencies(&self) -> Vec<Dep> {
        vec![Dep::of::<RiversLayer>(IVec3::new(
            REACH_M as i32,
            0,
            REACH_M as i32,
        ))]
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> RiverIndexChunk {
        let own = ctx.chunk_bounds();
        let padded = own.inflate(IVec3::new(REACH_M as i32, 0, REACH_M as i32));
        let view = ctx.get::<RiversLayer>(padded);
        let in_own = |p: Vec2| {
            p.x >= own.min.x as f32
                && p.x < own.max.x as f32
                && p.y >= own.min.z as f32
                && p.y < own.max.z as f32
        };
        let mut segments = Vec::new();
        for (_, chunk) in view.iter() {
            for river in &chunk.rivers {
                let n = river.waypoints.len();
                for (i, seg) in river.waypoints.windows(2).enumerate() {
                    if in_own((seg[0] + seg[1]) * 0.5) {
                        let t = i as f32 / n as f32;
                        segments.push((
                            [seg[0], seg[1]],
                            2.0 + 5.0 * t,
                            [river.levels[i], river.levels[i + 1]],
                        ));
                    }
                }
            }
        }
        RiverIndexChunk { segments }
    }
}

/// Longest possible indexed segment (spill splices step diagonally).
const MAX_SEG_M: f32 = 16.0;

pub fn planning_layers(world_seed: u64, chance: f32) -> LayerManager {
    let mut mgr = LayerManager::new(world_seed);
    mgr.register(SpringsLayer {
        seed: world_seed,
        chance,
    });
    mgr.register(RiversLayer);
    mgr.register(RiverIndexLayer);
    mgr
}

/// Bed + water ops for rivers overlapping `[min, max]` — served from the
/// segment index, so the query is local.
pub fn river_ops(mgr: &LayerManager, min: Vec3, max: Vec3) -> Vec<CsgOp> {
    let pad = (MAX_SEG_M + 16.0) as i32;
    let bounds = IAabb::new(
        IVec3::new(min.x as i32 - pad, 0, min.z as i32 - pad),
        IVec3::new(max.x as i32 + pad, 1, max.z as i32 + pad),
    );
    let mut out = Vec::new();
    for (_, chunk) in mgr.get::<RiverIndexLayer>(bounds).iter() {
        for ([a, b], half_w, levels) in &chunk.segments {
            segment_ops(*a, *b, *half_w, *levels, &mut out);
        }
    }
    out.retain(|op| op.touches(min, max));
    out
}

fn segment_ops(a: Vec2, b: Vec2, half_w: f32, levels: [f32; 2], out: &mut Vec<CsgOp>) {
    let len = a.distance(b);
    if len < 0.01 {
        return;
    }
    let dir = (b - a) / len;
    let yaw = dir.to_angle();
    // Short flow-aligned sub-boxes with interpolated (monotone) water
    // levels: gentle slopes read as a continuous ribbon, steep ones as
    // small rapids steps instead of floating slabs.
    let steps = (len / 3.0).ceil().max(1.0) as i32;
    let sub = len / steps as f32;
    for i in 0..steps {
        let t = (i as f32 + 0.5) / steps as f32;
        let p = a + dir * (t * len);
        let level = levels[0] + (levels[1] - levels[0]) * t;
        // Carve the valley notch...
        out.push(CsgOp::boxy(
            Vec3::new(p.x, level + 0.9, p.y),
            Vec3::new(sub * 0.7 + 0.8, 2.4, half_w + 1.4),
            -yaw,
            0,
            true,
        ));
        // ...and lay the water ribbon just below its rim.
        out.push(CsgOp::boxy(
            Vec3::new(p.x, level - 0.8, p.y),
            Vec3::new(sub * 0.7 + 0.6, 1.0, half_w),
            -yaw,
            MAT_RIVER,
            false,
        ));
    }
}

/// River segments overlapping the box — spawner clearance, served from
/// the local segment index.
pub fn rivers_near(mgr: &LayerManager, min: Vec2, max: Vec2) -> Vec<[Vec2; 2]> {
    let pad = (MAX_SEG_M + 16.0) as i32;
    let bounds = IAabb::new(
        IVec3::new(min.x as i32 - pad, 0, min.y as i32 - pad),
        IVec3::new(max.x as i32 + pad, 1, max.y as i32 + pad),
    );
    let mut out = Vec::new();
    for (_, chunk) in mgr.get::<RiverIndexLayer>(bounds).iter() {
        for ([a, b], _, _) in &chunk.segments {
            let lo = a.min(*b);
            let hi = a.max(*b);
            if lo.x <= max.x && hi.x >= min.x && lo.y <= max.y && hi.y >= min.y {
                out.push([*a, *b]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_descends_monotonically_and_reaches_the_sea() {
        // A uniform slope down toward +x reaching sea level at x = 800.
        let slope = |p: Vec2| (800.0 - p.x).max(0.0) * 0.1;
        let path = flow_path(&slope, Vec2::new(0.0, 0.0), &FlowParams::default());
        assert!(path.len() > 10);
        // Pure slope with no pits: strictly non-increasing (spill unused).
        for w in path.windows(2) {
            assert!(
                slope(w[1]) <= slope(w[0]) + 1e-3,
                "river flows uphill on a pure slope: {} -> {}",
                slope(w[0]),
                slope(w[1])
            );
        }
        let end = *path.last().unwrap();
        assert!(slope(end) <= 0.5, "river never reached the sea: h={}", slope(end));
    }

    #[test]
    fn flow_spills_over_a_shallow_lip() {
        // Downhill toward +x, interrupted by a 3 m ridge across x=400:
        // the walk must pond, spill over, and continue to the sea.
        let terrain = |p: Vec2| {
            let base = (900.0 - p.x).max(0.0) * 0.08;
            let ridge = if (390.0..410.0).contains(&p.x) { 4.0 } else { 0.0 };
            base + ridge
        };
        let path = flow_path(&terrain, Vec2::new(0.0, 0.0), &FlowParams::default());
        let end = *path.last().unwrap();
        assert!(
            end.x > 850.0,
            "river failed to spill past the ridge: ended at {end:?} h={}",
            terrain(end)
        );
    }

    #[test]
    fn flow_stops_in_a_pit() {
        // A bowl: minimum at the center, well above sea level.
        let bowl = |p: Vec2| 50.0 + p.length() * 0.05;
        let path = flow_path(&bowl, Vec2::new(300.0, 0.0), &FlowParams::default());
        let end = *path.last().unwrap();
        assert!(end.length() < 40.0, "river did not settle in the pit: {end:?}");
        assert!(path.len() < 100, "walk wandered instead of ending: {}", path.len());
    }

    #[test]
    fn deterministic_and_culled() {
        let mgr = planning_layers(0, 0.6);
        let min = Vec3::new(-4096.0, -100.0, -4096.0);
        let max = Vec3::new(4096.0, 600.0, 4096.0);
        let a = river_ops(&mgr, min, max);
        let mgr2 = planning_layers(0, 0.6);
        let b = river_ops(&mgr2, min, max);
        assert_eq!(a, b);
        for op in &a {
            assert!(op.touches(min, max));
        }
        // Sub-box query = filtered superset (chunks agree on each river).
        let smin = Vec3::new(-1024.0, -100.0, -1024.0);
        let smax = Vec3::new(1024.0, 600.0, 1024.0);
        let sub = river_ops(&mgr, smin, smax);
        let expect: Vec<_> = a.iter().filter(|op| op.touches(smin, smax)).copied().collect();
        assert_eq!(sub, expect);
    }
}
