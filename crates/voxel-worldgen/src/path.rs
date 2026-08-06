//! Deterministic terrain-aware pathfinding for planning layers
//! (LayerProcGen's "natural paths" technique): A* over a world-aligned
//! grid with a cost that penalizes steepness, so switchbacks and passes
//! emerge. Pure — same inputs, same path, on any machine.

use glam::Vec2;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Search parameters.
pub struct PathParams {
    /// Grid step (meters). Nodes snap to a world-aligned lattice, so the
    /// same endpoints always search the same graph.
    pub step_m: f32,
    /// Cost multiplier on squared slope (rise/run): higher = flatter
    /// paths with more detours.
    pub slope_penalty: f32,
    /// Heights at or below this are heavily penalized, so routes avoid
    /// low ground (a seabed, a flood plain). Not a water concept: the
    /// caller decides what "low" means for its world.
    pub low_ground_m: f32,
}

impl Default for PathParams {
    fn default() -> Self {
        Self {
            step_m: 8.0,
            slope_penalty: 60.0,
            low_ground_m: 0.5,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Cost(f32);
impl Eq for Cost {}
impl Ord for Cost {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A* from `a` to `b`, constrained to the closed box `[lo, hi]`.
/// Returns world-space waypoints from `a` to `b`, or None when no path
/// exists within the corridor.
pub fn find_path(
    height: &dyn Fn(Vec2) -> f32,
    a: Vec2,
    b: Vec2,
    lo: Vec2,
    hi: Vec2,
    params: &PathParams,
) -> Option<Vec<Vec2>> {
    let step = params.step_m;
    let node = |p: Vec2| -> (i32, i32) { ((p.x / step).round() as i32, (p.y / step).round() as i32) };
    let pos = |n: (i32, i32)| -> Vec2 { Vec2::new(n.0 as f32 * step, n.1 as f32 * step) };
    let (start, goal) = (node(a), node(b));
    let in_box = |n: (i32, i32)| {
        let p = pos(n);
        p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y
    };
    if !in_box(start) || !in_box(goal) {
        return None;
    }

    let move_cost = |from: (i32, i32), to: (i32, i32)| -> f32 {
        let (pf, pt) = (pos(from), pos(to));
        let run = pf.distance(pt);
        let rise = height(pt) - height(pf);
        let slope = rise / run;
        let mut c = run * (1.0 + params.slope_penalty * slope * slope);
        if height(pt) <= params.low_ground_m {
            c *= 25.0;
        }
        c
    };
    let heuristic = |n: (i32, i32)| pos(n).distance(pos(goal));

    let mut open: BinaryHeap<(std::cmp::Reverse<Cost>, (i32, i32))> = BinaryHeap::new();
    let mut best: std::collections::HashMap<(i32, i32), (f32, (i32, i32))> =
        std::collections::HashMap::new();
    best.insert(start, (0.0, start));
    open.push((std::cmp::Reverse(Cost(heuristic(start))), start));
    let mut visited = 0u32;
    while let Some((_, current)) = open.pop() {
        if current == goal {
            let mut path = vec![pos(goal)];
            let mut n = goal;
            while n != start {
                n = best[&n].1;
                path.push(pos(n));
            }
            path.reverse();
            // Exact endpoints replace the snapped lattice ends.
            *path.first_mut().unwrap() = a;
            *path.last_mut().unwrap() = b;
            return Some(path);
        }
        visited += 1;
        if visited > 200_000 {
            return None; // corridor too hard — caller falls back
        }
        let (g_cur, _) = best[&current];
        for dz in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dz == 0 {
                    continue;
                }
                let next = (current.0 + dx, current.1 + dz);
                if !in_box(next) {
                    continue;
                }
                let g = g_cur + move_cost(current, next);
                let better = match best.get(&next) {
                    None => true,
                    Some((old, _)) => {
                        g < *old - 1e-6
                            || (g < *old + 1e-6 && (current.0, current.1) < (best[&next].1 .0, best[&next].1 .1))
                    }
                };
                if better {
                    best.insert(next, (g, current));
                    open.push((std::cmp::Reverse(Cost(g + heuristic(next))), next));
                }
            }
        }
    }
    None
}

/// Distance from `p` to the segment `a`-`b` (prop clearance checks).
pub fn dist_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist_to_segment_basics() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 0.0);
        assert!((dist_to_segment(Vec2::new(5.0, 3.0), a, b) - 3.0).abs() < 1e-5);
        assert!((dist_to_segment(Vec2::new(-4.0, 0.0), a, b) - 4.0).abs() < 1e-5);
        assert!((dist_to_segment(Vec2::new(13.0, 4.0), a, b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn flat_ground_is_nearly_straight() {
        let flat = |_: Vec2| 10.0;
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(200.0, 80.0);
        let path = find_path(&flat, a, b, Vec2::splat(-100.0), Vec2::splat(400.0), &PathParams::default())
            .expect("path on flat ground");
        assert_eq!(path[0], a);
        assert_eq!(*path.last().unwrap(), b);
        let len: f32 = path.windows(2).map(|w| w[0].distance(w[1])).sum();
        assert!(len < a.distance(b) * 1.15, "flat path detours: {len}");
    }

    #[test]
    fn detours_through_the_pass() {
        // A 100 m ridge wall along x in [-16, 16], with a flat pass around
        // z in [72, 104]. The straight line from (-80,0) to (80,0) climbs
        // the wall; the cheap route goes through the pass.
        let terrain = |p: Vec2| -> f32 {
            if p.x.abs() < 16.0 && !(72.0..104.0).contains(&p.y) {
                100.0
            } else {
                10.0
            }
        };
        let a = Vec2::new(-80.0, 0.0);
        let b = Vec2::new(80.0, 0.0);
        let path = find_path(
            &terrain,
            a,
            b,
            Vec2::new(-200.0, -200.0),
            Vec2::new(200.0, 200.0),
            &PathParams::default(),
        )
        .expect("path exists");
        // Where the path crosses x = 0 it must be inside the pass.
        let crossing = path
            .windows(2)
            .find(|w| (w[0].x <= 0.0) != (w[1].x <= 0.0))
            .expect("path crosses the wall line");
        let z = (crossing[0].y + crossing[1].y) * 0.5;
        assert!(
            (60.0..116.0).contains(&z),
            "path climbed the wall instead of using the pass (crossed at z={z})"
        );
        // And it never climbs the ridge.
        for w in path.windows(2) {
            let rise = (terrain(w[1]) - terrain(w[0])).abs();
            assert!(rise < 50.0, "path steps up the cliff: {rise}");
        }
    }

    #[test]
    fn avoids_water() {
        // A lake strip across the direct route with a dry causeway far to
        // one side.
        let terrain = |p: Vec2| -> f32 {
            if (40.0..70.0).contains(&p.x) && p.y < 150.0 {
                0.0 // water
            } else {
                5.0
            }
        };
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(120.0, 0.0);
        let path = find_path(
            &terrain,
            a,
            b,
            Vec2::new(-50.0, -50.0),
            Vec2::new(250.0, 250.0),
            &PathParams::default(),
        )
        .expect("path exists");
        let wet = path.iter().filter(|p| terrain(**p) <= 0.5).count();
        assert_eq!(wet, 0, "path wades through the lake");
    }

    #[test]
    fn deterministic() {
        let bumpy = |p: Vec2| (p.x * 0.05).sin() * 20.0 + (p.y * 0.031).cos() * 17.0;
        let a = Vec2::new(-130.0, -40.0);
        let b = Vec2::new(170.0, 90.0);
        let lo = Vec2::splat(-300.0);
        let hi = Vec2::splat(300.0);
        let p1 = find_path(&bumpy, a, b, lo, hi, &PathParams::default()).unwrap();
        let p2 = find_path(&bumpy, a, b, lo, hi, &PathParams::default()).unwrap();
        assert_eq!(p1, p2);
        // Diagonal grid steps are step*sqrt(2); the exact-endpoint splice
        // adds up to half a step at each end.
        for w in p1.windows(2) {
            assert!(w[0].distance(w[1]) <= PathParams::default().step_m * 2.0);
        }
        for p in &p1 {
            assert!(p.x >= lo.x - 1.0 && p.x <= hi.x + 1.0);
            assert!(p.y >= lo.y - 1.0 && p.y <= hi.y + 1.0);
        }
    }
}
