//! Descent walk with pond-and-spill: from a start point, follow a
//! heightfield downhill, escaping basins by spilling over the lowest lip
//! within reach, and stop at a given level.
//!
//! Nothing here is hydrology. It is a generic descent over a scalar
//! field — what a level uses it for (a river, a lava channel, a drainage
//! ditch, a retreating path) is data.

use glam::Vec2;

/// Descent step (meters).
pub const FLOW_STEP_M: f32 = 8.0;

/// Parameters of the descent walk.
pub struct FlowParams {
    pub step_m: f32,
    /// Stop once the height drops to this level.
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
            max_steps: 400,
            max_spill_rise: 7.0,
        }
    }
}

/// Deterministic lattice descent from `start` with pond-and-spill: while
/// downhill neighbors exist the walk takes the steepest one; at a local
/// minimum a bounded Dijkstra looks for the nearest escape route whose
/// lip stays within `max_spill_rise` of the pond floor and which ends
/// LOWER than the pond entry (a shallow basin overflows; a deep one
/// ends the walk). Ends at `stop_level`, in a deep
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
        type Frontier = std::collections::BinaryHeap<(std::cmp::Reverse<(u32, (i32, i32))>,)>;
        let mut open: Frontier = Default::default();
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
            break; // deep basin: the walk terminates here
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_descends_monotonically_and_reaches_the_stop_level() {
        // A uniform slope down toward +x reaching the stop level at x = 800.
        let slope = |p: Vec2| (800.0 - p.x).max(0.0) * 0.1;
        let path = flow_path(&slope, Vec2::new(0.0, 0.0), &FlowParams::default());
        assert!(path.len() > 10);
        // Pure slope with no pits: strictly non-increasing (spill unused).
        for w in path.windows(2) {
            assert!(
                slope(w[1]) <= slope(w[0]) + 1e-3,
                "flow ascends on a pure slope: {} -> {}",
                slope(w[0]),
                slope(w[1])
            );
        }
        let end = *path.last().unwrap();
        assert!(
            slope(end) <= 0.5,
            "flow never reached the stop level: h={}",
            slope(end)
        );
    }

    #[test]
    fn flow_spills_over_a_shallow_lip() {
        // Downhill toward +x, interrupted by a 3 m ridge across x=400:
        // the walk must pond, spill over, and continue to the stop level.
        let terrain = |p: Vec2| {
            let base = (900.0 - p.x).max(0.0) * 0.08;
            let ridge = if (390.0..410.0).contains(&p.x) {
                4.0
            } else {
                0.0
            };
            base + ridge
        };
        let path = flow_path(&terrain, Vec2::new(0.0, 0.0), &FlowParams::default());
        let end = *path.last().unwrap();
        assert!(
            end.x > 850.0,
            "flow failed to spill past the ridge: ended at {end:?} h={}",
            terrain(end)
        );
    }

    #[test]
    fn flow_stops_in_a_pit() {
        // A bowl: minimum at the center, well above the stop level.
        let bowl = |p: Vec2| 50.0 + p.length() * 0.05;
        let path = flow_path(&bowl, Vec2::new(300.0, 0.0), &FlowParams::default());
        let end = *path.last().unwrap();
        assert!(
            end.length() < 40.0,
            "flow did not settle in the pit: {end:?}"
        );
        assert!(
            path.len() < 100,
            "walk wandered instead of ending: {}",
            path.len()
        );
    }
}
