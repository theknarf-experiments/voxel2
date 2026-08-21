//! Space colonization: a branching skeleton grown toward a cloud of
//! attractor points.
//!
//! A PRIMITIVE, like the descent walk or the A* path. It knows about
//! points, segments and radii and nothing about trees, roots or tendrils
//! — what a skeleton MEANS is the host's, and so is the cloud it grows
//! into. That split is the whole reason this can serve both a prop mesh
//! and a set of CSG ops without either learning about the other.
//!
//! The algorithm is Runions et al.: every attractor pulls the nearest
//! node within `attraction_m`, each pulled node steps `step_m` along the
//! average of its pulls, and any attractor a node reaches within
//! `kill_m` is consumed. Branching is emergent — a node pulled in two
//! directions at once does not split, but its children diverge because
//! each inherits a different subset of the cloud.

use glam::Vec3;

use crate::seed::Rng;

/// One grown segment: a truncated cone from `a` to `b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limb {
    pub a: Vec3,
    pub b: Vec3,
    /// Radius at `a` — never smaller than `r_b`, so a limb always tapers
    /// outward-thin and a consumer can trust the wider end is the base.
    pub r_a: f32,
    pub r_b: f32,
    /// Steps from the root. 0 is the first segment out of it.
    pub depth: u16,
}

impl Limb {
    pub fn length(&self) -> f32 {
        self.a.distance(self.b)
    }
}

/// How a skeleton grows. Distances in the same units as the attractors.
#[derive(Clone, Copy, Debug)]
pub struct Growth {
    /// How far a node reaches for attractors. Below `step_m` nothing can
    /// ever be reached and the skeleton is a single segment.
    pub attraction_m: f32,
    /// An attractor this close to any node is consumed. Must be under
    /// `attraction_m` or an attractor pulls forever without being spent
    /// and the skeleton grows until `max_nodes`.
    pub kill_m: f32,
    /// Internode length — the resolution of the whole skeleton.
    pub step_m: f32,
    /// Hard bound on the node count, so a bad parameter set costs a
    /// bounded amount of work instead of hanging.
    pub max_nodes: usize,
    /// Radius of a tip, and the exponent thicknesses combine under.
    ///
    /// da Vinci's rule: a parent's cross-section is the sum of its
    /// children's, i.e. `r_parent^taper = Σ r_child^taper`. 2.0 is the
    /// literal reading and looks spindly; 2.2–2.8 reads as wood.
    pub tip_r_m: f32,
    pub taper: f32,
    /// Downward bias added to every step, in units of the step direction.
    /// Positive droops (branches), negative lifts.
    pub droop: f32,
    /// How much of each step is random, 0..1. Breaks the lattice look a
    /// regular attractor cloud otherwise prints onto the skeleton.
    pub wobble: f32,
}

impl Default for Growth {
    fn default() -> Self {
        Self {
            attraction_m: 3.2,
            kill_m: 0.9,
            step_m: 0.45,
            max_nodes: 900,
            tip_r_m: 0.02,
            taper: 2.4,
            droop: 0.0,
            wobble: 0.12,
        }
    }
}

struct Node {
    pos: Vec3,
    parent: Option<u32>,
    depth: u16,
    r: f32,
}

/// Grow a skeleton from `root`, initially heading `dir`, into `cloud`.
///
/// Returns one limb per node except the root. Empty only if the cloud is
/// empty or nothing in it is reachable — a caller that must have
/// geometry should check.
///
/// Cost is O(cloud × nodes) per round, which is why `max_nodes` exists.
/// Fine for a few hundred attractors built once; a caller growing
/// thousands per planning tile wants a smaller cloud, not a bigger cap.
pub fn colonize(root: Vec3, dir: Vec3, cloud: &[Vec3], g: &Growth, rng: &mut Rng) -> Vec<Limb> {
    let mut nodes = vec![Node {
        pos: root,
        parent: None,
        depth: 0,
        r: g.tip_r_m,
    }];
    let mut live: Vec<Vec3> = cloud.to_vec();
    let step = g.step_m.max(1.0e-4);
    let seed_dir = dir.normalize_or(Vec3::Y);

    // Pulls accumulated per node this round, reused across rounds.
    let mut pull: Vec<Vec3> = Vec::new();
    let mut pulled: Vec<u32> = Vec::new();
    while !live.is_empty() && nodes.len() < g.max_nodes {
        pull.clear();
        pull.resize(nodes.len(), Vec3::ZERO);
        pulled.clear();
        for a in &live {
            // Nearest node in reach. Ties break toward the earlier node,
            // which is deterministic under a stable node order.
            let mut best = None;
            let mut best_d2 = g.attraction_m * g.attraction_m;
            for (i, n) in nodes.iter().enumerate() {
                let d2 = n.pos.distance_squared(*a);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = Some(i);
                }
            }
            if let Some(i) = best {
                if pull[i] == Vec3::ZERO {
                    pulled.push(i as u32);
                }
                pull[i] += (*a - nodes[i].pos).normalize_or_zero();
            }
        }
        if pulled.is_empty() {
            break; // cloud is out of reach; nothing further can grow
        }
        // Sorted so the node order — and therefore every tie above — does
        // not depend on the attractor order.
        pulled.sort_unstable();
        for &i in &pulled {
            if nodes.len() >= g.max_nodes {
                break;
            }
            let i = i as usize;
            let jitter = Vec3::new(
                rng.next_f32() - 0.5,
                rng.next_f32() - 0.5,
                rng.next_f32() - 0.5,
            ) * g.wobble;
            let bias = Vec3::NEG_Y * g.droop;
            let dir = (pull[i].normalize_or(seed_dir) + jitter + bias).normalize_or(seed_dir);
            let pos = nodes[i].pos + dir * step;
            let depth = nodes[i].depth.saturating_add(1);
            nodes.push(Node {
                pos,
                parent: Some(i as u32),
                depth,
                r: g.tip_r_m,
            });
        }
        // Consume what the new nodes reached. Done after the whole round
        // so two nodes cannot race for the same attractor.
        let kill2 = g.kill_m * g.kill_m;
        let grown = &nodes;
        live.retain(|a| !grown.iter().any(|n| n.pos.distance_squared(*a) < kill2));
    }

    thicken(&mut nodes, g);
    nodes
        .iter()
        .filter_map(|n| {
            let p = n.parent? as usize;
            Some(Limb {
                a: nodes[p].pos,
                b: n.pos,
                r_a: nodes[p].r.max(n.r),
                r_b: n.r,
                depth: n.depth.saturating_sub(1),
            })
        })
        .collect()
}

/// Radii from the tips down, by da Vinci's rule.
///
/// Nodes are appended in growth order, so a parent always precedes its
/// children — walking backwards visits every child before its parent and
/// needs no tree traversal at all.
fn thicken(nodes: &mut [Node], g: &Growth) {
    // Wood a node carries, counted as nodes in its own subtree.
    //
    // `tip_r · nˆ(1/taper)` IS da Vinci's rule where the tree forks —
    // two subtrees of n₁ and n₂ meet at n₁+n₂, and
    // `(r₁ᵗ + r₂ᵗ)^(1/t)` is the same number — but it also thickens a
    // limb that does NOT fork, which the literal rule does not.
    //
    // That mattered: read literally, one child means "the same wood
    // continuing", so an unbranched run keeps the TIP radius all the
    // way down. A conifer whose crown the stem eats on the way up never
    // forks, so its trunk came out 2 cm thick from root to tip and the
    // mesh dropped every limb of it — zero wood, foliage floating over
    // bare ground. A longer cantilever needs more wood whether or not
    // it happens to branch.
    let mut n = vec![1u32; nodes.len()];
    let taper = g.taper.max(1.0);
    for i in (0..nodes.len()).rev() {
        nodes[i].r = g.tip_r_m * (n[i] as f32).powf(1.0 / taper);
        if let Some(p) = nodes[i].parent {
            n[p as usize] += n[i];
        }
    }
}

/// A jittered lattice of points filling an ellipsoid — the cloud shape a
/// crown, a root ball or a blob of tendrils all want.
///
/// Jitter is a fraction of the spacing, so 1.0 lets neighbours just touch
/// and 0.0 is a hard lattice (which prints its own grid into the
/// skeleton — see [`Growth::wobble`]).
pub fn ellipsoid_cloud(
    center: Vec3,
    radii: Vec3,
    spacing: f32,
    jitter: f32,
    rng: &mut Rng,
) -> Vec<Vec3> {
    let spacing = spacing.max(1.0e-3);
    let steps = (radii / spacing).ceil().as_ivec3();
    let mut out = Vec::new();
    for iz in -steps.z..=steps.z {
        for iy in -steps.y..=steps.y {
            for ix in -steps.x..=steps.x {
                let cell = Vec3::new(ix as f32, iy as f32, iz as f32) * spacing;
                let j = Vec3::new(
                    rng.next_f32() - 0.5,
                    rng.next_f32() - 0.5,
                    rng.next_f32() - 0.5,
                ) * (spacing * jitter);
                let p = cell + j;
                // Inside the ellipsoid, tested on the jittered point so
                // the surface is ragged rather than a clean shell.
                if (p / radii.max(Vec3::splat(1.0e-3))).length_squared() <= 1.0 {
                    out.push(center + p);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crown(seed: u64) -> Vec<Limb> {
        let mut rng = Rng::new(seed);
        let cloud = ellipsoid_cloud(
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(2.0, 2.4, 2.0),
            0.7,
            0.8,
            &mut rng,
        );
        colonize(Vec3::ZERO, Vec3::Y, &cloud, &Growth::default(), &mut rng)
    }

    /// Same seed, same skeleton — the property every consumer of this
    /// rests on, since a prop mesh and a set of CSG ops grown from one
    /// seed must agree about where the wood is.
    #[test]
    fn growth_is_deterministic() {
        assert_eq!(crown(7), crown(7));
        assert_ne!(crown(7), crown(8), "the seed must actually reach the shape");
    }

    /// It grows a connected, branching skeleton — not a single stem, and
    /// not a cloud of orphans.
    #[test]
    fn it_branches_and_stays_connected() {
        let limbs = crown(3);
        assert!(limbs.len() > 30, "barely grew: {} limbs", limbs.len());
        // Connected: every limb's base is some other limb's tip, or the
        // root itself.
        for l in &limbs {
            let joined = l.a == Vec3::ZERO || limbs.iter().any(|o| o.b == l.a);
            assert!(joined, "orphan limb {l:?}");
        }
        // Branching: at least one point is the base of two limbs.
        let forks = limbs
            .iter()
            .filter(|l| limbs.iter().filter(|o| o.a == l.a).count() > 1)
            .count();
        assert!(forks > 0, "grew a stick, not a tree");
    }

    /// Wood thickens toward the root, and every limb is finite. A NaN
    /// here becomes a degenerate triangle or a CSG op with no bounds.
    #[test]
    fn limbs_taper_toward_the_tips() {
        let limbs = crown(11);
        for l in &limbs {
            assert!(l.r_a >= l.r_b, "limb widens toward its tip: {l:?}");
            assert!(l.r_b > 0.0 && l.r_a.is_finite(), "bad radius: {l:?}");
            assert!(l.a.is_finite() && l.b.is_finite(), "bad position: {l:?}");
            assert!(l.length() > 0.0, "zero-length limb: {l:?}");
        }
        // The base is the thickest thing in the skeleton.
        let base = limbs.iter().find(|l| l.a == Vec3::ZERO).expect("a root");
        let widest = limbs.iter().map(|l| l.r_a).fold(0.0, f32::max);
        assert_eq!(base.r_a, widest, "the trunk is not the thickest limb");
    }

    /// `max_nodes` is a real bound, not a suggestion: a cloud that can
    /// never be consumed (kill radius of zero) must still terminate.
    #[test]
    fn an_unconsumable_cloud_still_terminates() {
        let mut rng = Rng::new(5);
        let cloud = ellipsoid_cloud(Vec3::Y * 3.0, Vec3::splat(2.0), 0.5, 0.5, &mut rng);
        let g = Growth {
            kill_m: 0.0,
            max_nodes: 120,
            ..Default::default()
        };
        let limbs = colonize(Vec3::ZERO, Vec3::Y, &cloud, &g, &mut rng);
        assert!(limbs.len() < 120, "grew past the cap: {}", limbs.len());
    }

    /// An empty or unreachable cloud grows nothing rather than panicking
    /// or looping — a level can ask for a shape that cannot exist.
    #[test]
    fn nothing_to_grow_toward_grows_nothing() {
        let mut rng = Rng::new(1);
        let g = Growth::default();
        assert!(colonize(Vec3::ZERO, Vec3::Y, &[], &g, &mut rng).is_empty());
        let far = [Vec3::new(0.0, 1000.0, 0.0)];
        assert!(colonize(Vec3::ZERO, Vec3::Y, &far, &g, &mut rng).is_empty());
    }
}
