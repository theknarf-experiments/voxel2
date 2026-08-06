//! Structures as data: the small grammar that replaced the hand-written
//! ruin / dungeon / pocket recipes.
//!
//! A structure is a set of weighted variants; a variant is a list of
//! parts; a part places one primitive shape at every position of an
//! arrangement (ring, scatter, chain, or a single point), optionally
//! hollowed into a shell and optionally linked to the next position by a
//! swept tunnel. One `size` is sampled per site so a structure's parts
//! agree with each other (the wall ring, its towers, and the rubble
//! inside it share one radius).
//!
//! Everything is sampled from the caller's [`Rng`], so structures stay
//! deterministic per site, and every emitted op is bounded by
//! [`Structure::max_reach`] — the contract `stack::ELEM_PAD_M` rests on.

use glam::{Vec2, Vec3};
use voxel_core::csg::CsgOp;
use voxel_core::seed::Rng;

use voxel_worldgen::Generator;

/// Inclusive value range, sampled uniformly.
pub type Range = [f32; 2];

/// A box half-extent: a range, or the ring's tangential arc half-length
/// (so wall segments meet without the author computing trigonometry).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Extent {
    Range(Range),
    Arc,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Boxy { half: [Extent; 3] },
    Cylinder { radius: Range, half_height: Range },
    Sphere { radius: Range },
}

/// Where a part's vertical anchor comes from.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Seat {
    /// The generator's heightfield at the instance's xz (surface props).
    #[default]
    Terrain,
    /// The site's own y (interiors: the structural floor it sits on).
    Site,
}

/// What the seat positions: the shape's base, or its center.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Anchor {
    /// Bottom of the shape rests on the seat (walls, towers, rooms).
    #[default]
    Base,
    /// Center of the shape sits at the seat (buried rooms, shafts).
    Center,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Yaw {
    #[default]
    Zero,
    Random,
    /// Face along the arrangement (ring tangent, chain heading).
    Tangent,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Arrange {
    /// One instance at the site.
    Single,
    /// `count` instances evenly spaced on a circle of `radius_frac` ×
    /// the structure size, from a random start angle.
    Ring { count: [u32; 2], radius_frac: Range },
    /// `count` instances at random angles within `radius_frac` × size.
    Scatter { count: [u32; 2], radius_frac: Range },
    /// A walk from the site: each step turns by up to `turn_deg`,
    /// advances `step`, and descends `descend`. Positions are clamped
    /// into `radius_frac` × size so the structure stays bounded.
    Chain {
        count: [u32; 2],
        step: Range,
        turn_deg: f32,
        descend: Range,
        /// Snap headings to the cardinal axes (right-angled interiors).
        orthogonal: bool,
        radius_frac: Range,
        /// Prepend a position at the terrain surface above the first
        /// instance, so `link` carves an entrance from outside.
        from_surface: bool,
    },
}

/// Sweeps a box between consecutive instances: corridors, doorways, the
/// entrance tunnel. Emitted as overlapping steps because CSG boxes have
/// no pitch — a sloped tunnel is a stack of yawed boxes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Link {
    pub half_w: f32,
    pub half_h: f32,
    pub step_m: f32,
    pub material: u32,
    pub cut: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub arrange: Arrange,
    pub shape: Shape,
    pub material: u32,
    pub cut: bool,
    /// Emit an inner cut inset by this much on every axis (shells).
    pub hollow: Option<f32>,
    /// Per-instance chance to emit nothing (collapsed, ruined).
    pub skip: f32,
    pub seat: Seat,
    pub anchor: Anchor,
    pub y_offset: Range,
    pub yaw: Yaw,
    pub link: Option<Link>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Variant {
    pub weight: f32,
    pub parts: Vec<Part>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Structure {
    /// Sampled once per site; arrangements scale their radius by it, so
    /// a structure's parts stay coherent with each other.
    pub size: Range,
    pub variants: Vec<Variant>,
}

impl Structure {
    /// Worst-case horizontal reach of any op from the site — what a
    /// level must keep under `stack::ELEM_PAD_M`.
    pub fn max_reach(&self) -> f32 {
        let size = self.size[1];
        let mut reach: f32 = 0.0;
        for variant in &self.variants {
            for part in &variant.parts {
                let radius = match &part.arrange {
                    Arrange::Single => 0.0,
                    Arrange::Ring { radius_frac, .. }
                    | Arrange::Scatter { radius_frac, .. }
                    | Arrange::Chain { radius_frac, .. } => radius_frac[1] * size,
                };
                let extent = match &part.shape {
                    Shape::Boxy { half } => half
                        .iter()
                        .map(|e| match e {
                            Extent::Range(r) => r[1],
                            Extent::Arc => size,
                        })
                        .fold(0.0f32, f32::max),
                    Shape::Cylinder { radius, .. } | Shape::Sphere { radius } => radius[1],
                };
                let link = part.link.map_or(0.0, |l| l.half_w.max(l.half_h));
                // Yawed boxes reach their diagonal.
                reach = reach.max(radius + extent * std::f32::consts::SQRT_2 + link);
            }
        }
        reach
    }
}

fn sample(rng: &mut Rng, range: Range) -> f32 {
    range[0] + rng.next_f32() * (range[1] - range[0]).max(0.0)
}

fn sample_count(rng: &mut Rng, range: [u32; 2]) -> u32 {
    range[0] + rng.next_range(range[1].saturating_sub(range[0]).max(1))
}

/// One placed instance of a part.
struct Instance {
    pos: Vec3,
    /// Heading of the arrangement at this instance (ring tangent / chain).
    heading: f32,
    /// Ring arc half-length, for `Extent::Arc`.
    arc: f32,
}

/// Emit `structure` at `site` (planar callers pass the terrain site with
/// any y; `Seat::Site` uses `site.y`, which interiors set to their floor).
pub fn build(
    structure: &Structure,
    site: Vec3,
    generator: &Generator,
    rng: &mut Rng,
    out: &mut Vec<CsgOp>,
) {
    let Some(variant) = pick_variant(structure, rng) else {
        return;
    };
    let size = sample(rng, structure.size);
    for part in &variant.parts {
        let instances = arrange(part, site, size, generator, rng);
        emit_part(part, &instances, generator, rng, out);
    }
}

fn pick_variant<'a>(structure: &'a Structure, rng: &mut Rng) -> Option<&'a Variant> {
    let total: f32 = structure.variants.iter().map(|v| v.weight.max(0.0)).sum();
    if total <= 0.0 {
        return structure.variants.first();
    }
    let mut roll = rng.next_f32() * total;
    for variant in &structure.variants {
        roll -= variant.weight.max(0.0);
        if roll <= 0.0 {
            return Some(variant);
        }
    }
    structure.variants.last()
}

fn seat_y(part: &Part, site: Vec3, xz: Vec2, generator: &Generator) -> f32 {
    match part.seat {
        Seat::Terrain => generator.height(xz, 1.0),
        Seat::Site => site.y,
    }
}

fn arrange(
    part: &Part,
    site: Vec3,
    size: f32,
    generator: &Generator,
    rng: &mut Rng,
) -> Vec<Instance> {
    let site_xz = Vec2::new(site.x, site.z);
    let mut out = Vec::new();
    match &part.arrange {
        Arrange::Single => {
            out.push(Instance {
                pos: Vec3::new(site.x, 0.0, site.z),
                heading: 0.0,
                arc: 0.0,
            });
        }
        Arrange::Ring {
            count,
            radius_frac,
        } => {
            let n = sample_count(rng, *count).max(1);
            let radius = size * sample(rng, *radius_frac);
            let base = rng.next_f32() * std::f32::consts::TAU;
            // Segments meet end to end: half the arc between neighbors.
            let arc = radius * std::f32::consts::PI / n as f32 * 0.95;
            for i in 0..n {
                let angle = base + std::f32::consts::TAU * i as f32 / n as f32;
                let xz = site_xz + Vec2::new(angle.cos(), angle.sin()) * radius;
                out.push(Instance {
                    pos: Vec3::new(xz.x, 0.0, xz.y),
                    heading: angle + std::f32::consts::FRAC_PI_2,
                    arc,
                });
            }
        }
        Arrange::Scatter {
            count,
            radius_frac,
        } => {
            let n = sample_count(rng, *count).max(1);
            for _ in 0..n {
                let angle = rng.next_f32() * std::f32::consts::TAU;
                let radius = size * sample(rng, *radius_frac);
                let xz = site_xz + Vec2::new(angle.cos(), angle.sin()) * radius;
                out.push(Instance {
                    pos: Vec3::new(xz.x, 0.0, xz.y),
                    heading: angle,
                    arc: 0.0,
                });
            }
        }
        Arrange::Chain {
            count,
            step,
            turn_deg,
            descend,
            orthogonal,
            radius_frac,
            from_surface,
        } => {
            let n = sample_count(rng, *count).max(1);
            let max_r = size * radius_frac[1];
            let mut heading = rng.next_f32() * std::f32::consts::TAU;
            if *orthogonal {
                heading = quantize_heading(heading);
            }
            let mut xz = site_xz;
            let mut drop = 0.0f32;
            for i in 0..n {
                out.push(Instance {
                    pos: Vec3::new(xz.x, drop, xz.y),
                    heading,
                    arc: 0.0,
                });
                if i + 1 == n {
                    break;
                }
                heading += (rng.next_f32() - 0.5) * turn_deg.to_radians() * 2.0;
                if *orthogonal {
                    heading = quantize_heading(heading);
                }
                let advance = sample(rng, *step);
                let mut next = xz + Vec2::new(heading.cos(), heading.sin()) * advance;
                let from_site = next - site_xz;
                if from_site.length() > max_r {
                    next = site_xz + from_site.normalize_or_zero() * max_r;
                }
                xz = next;
                drop -= sample(rng, *descend);
            }
            if *from_surface {
                // An entry point above the first instance; `link` carves
                // the tunnel down to it.
                let first = out[0].pos;
                let surface = generator.height(Vec2::new(first.x, first.z), 1.0);
                let above = surface + 1.2 - (site.y + first.y);
                out.insert(
                    0,
                    Instance {
                        pos: Vec3::new(first.x, first.y + above, first.z),
                        heading: out[0].heading,
                        arc: 0.0,
                    },
                );
            }
        }
    }
    out
}

fn quantize_heading(heading: f32) -> f32 {
    let quarter = std::f32::consts::FRAC_PI_2;
    (heading / quarter).round() * quarter
}

fn emit_part(
    part: &Part,
    instances: &[Instance],
    generator: &Generator,
    rng: &mut Rng,
    out: &mut Vec<CsgOp>,
) {
    let mut placed: Vec<Vec3> = Vec::with_capacity(instances.len());
    for instance in instances {
        // The link needs every position, including skipped ones, or a
        // collapsed room would break the corridor chain.
        let xz = Vec2::new(instance.pos.x, instance.pos.z);
        let site_y = seat_y(part, Vec3::new(xz.x, 0.0, xz.y), xz, generator);
        let yaw = match part.yaw {
            Yaw::Zero => 0.0,
            Yaw::Random => rng.next_f32() * std::f32::consts::TAU,
            Yaw::Tangent => instance.heading,
        };
        let (center, half) = shape_at(part, instance, site_y, rng);
        placed.push(center);
        if rng.next_f32() < part.skip {
            continue;
        }
        push_shape(part, center, half, yaw, out);
    }
    if let Some(link) = part.link {
        for pair in placed.windows(2) {
            push_link(&link, pair[0], pair[1], out);
        }
    }
}

/// Center and half extents of one instance, resolving seat and anchor.
fn shape_at(part: &Part, instance: &Instance, seat: f32, rng: &mut Rng) -> (Vec3, Vec3) {
    let half = match &part.shape {
        Shape::Boxy { half } => Vec3::new(
            extent(half[0], instance, rng),
            extent(half[1], instance, rng),
            extent(half[2], instance, rng),
        ),
        Shape::Cylinder {
            radius,
            half_height,
        } => {
            let r = sample(rng, *radius);
            Vec3::new(r, sample(rng, *half_height), r)
        }
        Shape::Sphere { radius } => Vec3::splat(sample(rng, *radius)),
    };
    let base = seat + instance.pos.y + sample(rng, part.y_offset);
    let y = match part.anchor {
        Anchor::Base => base + half.y,
        Anchor::Center => base,
    };
    (Vec3::new(instance.pos.x, y, instance.pos.z), half)
}

fn extent(extent: Extent, instance: &Instance, rng: &mut Rng) -> f32 {
    match extent {
        Extent::Range(r) => sample(rng, r),
        Extent::Arc => instance.arc.max(0.1),
    }
}

fn push_shape(part: &Part, center: Vec3, half: Vec3, yaw: f32, out: &mut Vec<CsgOp>) {
    match part.shape {
        Shape::Boxy { .. } => {
            out.push(CsgOp::boxy(center, half, yaw, part.material, part.cut));
            if let Some(inset) = part.hollow {
                let inner = (half - Vec3::splat(inset)).max(Vec3::splat(0.05));
                out.push(CsgOp::boxy(center, inner, yaw, 0, true));
            }
        }
        Shape::Cylinder { .. } => {
            out.push(CsgOp::cylinder(
                center,
                half.x,
                half.y,
                part.material,
                part.cut,
            ));
            if let Some(inset) = part.hollow {
                out.push(CsgOp::cylinder(
                    center + Vec3::Y * inset,
                    (half.x - inset).max(0.05),
                    half.y,
                    0,
                    true,
                ));
            }
        }
        Shape::Sphere { .. } => {
            out.push(CsgOp::sphere(center, half.x, part.material, part.cut));
            if let Some(inset) = part.hollow {
                out.push(CsgOp::sphere(center, (half.x - inset).max(0.05), 0, true));
            }
        }
    }
}

fn push_link(link: &Link, a: Vec3, b: Vec3, out: &mut Vec<CsgOp>) {
    let span = b - a;
    let flat = Vec2::new(span.x, span.z);
    let len = span.length();
    if len < 0.01 {
        return;
    }
    let yaw = if flat.length() > 0.01 {
        flat.to_angle()
    } else {
        0.0
    };
    // Overlapping steps: CSG boxes have no pitch, so a sloped tunnel is
    // a stack of them.
    let steps = (len / link.step_m.max(0.5)).ceil().max(1.0) as i32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let p = a + span * t;
        out.push(CsgOp::boxy(
            p,
            Vec3::new(link.half_w, link.half_h, link.half_w),
            -yaw,
            link.material,
            link.cut,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::seed::chunk_seed;

    fn generator() -> Generator {
        Generator::new(
            voxel_worldgen::program::planet_program(),
            0,
            voxel_worldgen::program::DEFAULT_SUN_DIR,
        )
    }

    fn rng(salt: u64) -> Rng {
        Rng::new(chunk_seed(salt, 0x57, glam::IVec3::new(2, 0, 5)))
    }

    /// A ring wall with towers and rubble — the shape the ruin recipe had.
    fn ring_structure() -> Structure {
        Structure {
            size: [8.0, 17.0],
            variants: vec![Variant {
                weight: 1.0,
                parts: vec![
                    Part {
                        arrange: Arrange::Ring {
                            count: [6, 10],
                            radius_frac: [1.0, 1.0],
                        },
                        shape: Shape::Boxy {
                            half: [
                                Extent::Arc,
                                Extent::Range([1.2, 2.6]),
                                Extent::Range([0.55, 0.55]),
                            ],
                        },
                        material: 3,
                        cut: false,
                        hollow: None,
                        skip: 0.28,
                        seat: Seat::Terrain,
                        anchor: Anchor::Base,
                        y_offset: [-0.6, -0.6],
                        yaw: Yaw::Tangent,
                        link: None,
                    },
                    Part {
                        arrange: Arrange::Scatter {
                            count: [2, 5],
                            radius_frac: [0.0, 0.8],
                        },
                        shape: Shape::Boxy {
                            half: [
                                Extent::Range([0.5, 1.4]),
                                Extent::Range([0.5, 1.4]),
                                Extent::Range([0.5, 1.4]),
                            ],
                        },
                        material: 3,
                        cut: false,
                        hollow: None,
                        skip: 0.0,
                        seat: Seat::Terrain,
                        anchor: Anchor::Base,
                        y_offset: [-0.3, -0.3],
                        yaw: Yaw::Random,
                        link: None,
                    },
                ],
            }],
        }
    }

    #[test]
    fn structures_are_deterministic_and_bounded() {
        let s = ring_structure();
        let site = Vec3::new(-26800.0, 0.0, -37900.0);
        let build_once = |salt| {
            let mut r = rng(salt);
            let mut out = Vec::new();
            build(&s, site, &generator(), &mut r, &mut out);
            out
        };
        let a = build_once(1);
        assert_eq!(a, build_once(1));
        assert_ne!(a, build_once(2));
        assert!(!a.is_empty());

        // Every op is inside the declared reach, and seated on terrain.
        let reach = s.max_reach();
        assert!(reach < 64.0, "reach {reach} exceeds the element padding");
        for op in &a {
            let p = Vec2::new(op.center[0], op.center[2]);
            assert!(
                p.distance(Vec2::new(site.x, site.z)) <= reach,
                "op {p:?} beyond reach {reach}"
            );
            let ground = generator().height(p, 1.0);
            assert!((op.center[1] - ground).abs() < 12.0, "op far from ground");
        }
    }

    #[test]
    fn ring_segments_meet_and_skip_thins_them() {
        let mut s = ring_structure();
        s.variants[0].parts.truncate(1);
        s.variants[0].parts[0].skip = 0.0;
        let mut r = rng(3);
        let mut out = Vec::new();
        build(&s, Vec3::ZERO, &generator(), &mut r, &mut out);
        // Consecutive wall segments touch: the gap between neighboring
        // centers is at most the sum of their arc half-lengths.
        let centers: Vec<Vec2> = out
            .iter()
            .map(|op| Vec2::new(op.center[0], op.center[2]))
            .collect();
        let arc = out[0].half[0];
        for pair in centers.windows(2) {
            assert!(
                pair[0].distance(pair[1]) <= arc * 2.2,
                "ring segments leave a gap"
            );
        }
        // Skipping removes segments.
        s.variants[0].parts[0].skip = 0.9;
        let mut r = rng(3);
        let mut thinned = Vec::new();
        build(&s, Vec3::ZERO, &generator(), &mut r, &mut thinned);
        assert!(thinned.len() < out.len());
    }

    #[test]
    fn chain_links_form_one_connected_void() {
        // The dungeon shape: a descending chain of rooms, linked, with an
        // entrance carved from the surface.
        let s = Structure {
            size: [40.0, 40.0],
            variants: vec![Variant {
                weight: 1.0,
                parts: vec![Part {
                    arrange: Arrange::Chain {
                        count: [4, 6],
                        step: [12.0, 20.0],
                        turn_deg: 52.0,
                        descend: [3.0, 7.0],
                        orthogonal: false,
                        radius_frac: [0.0, 1.0],
                        from_surface: true,
                    },
                    shape: Shape::Boxy {
                        half: [
                            Extent::Range([3.5, 6.5]),
                            Extent::Range([2.2, 3.2]),
                            Extent::Range([3.5, 6.5]),
                        ],
                    },
                    material: 0,
                    cut: true,
                    hollow: None,
                    skip: 0.0,
                    seat: Seat::Terrain,
                    anchor: Anchor::Center,
                    y_offset: [-9.0, -9.0],
                    yaw: Yaw::Random,
                    link: Some(Link {
                        half_w: 1.6,
                        half_h: 1.8,
                        step_m: 3.0,
                        material: 0,
                        cut: true,
                    }),
                }],
            }],
        };
        let site = Vec3::new(-26800.0, 0.0, -37900.0);
        let mut r = rng(11);
        let mut ops = Vec::new();
        build(&s, site, &generator(), &mut r, &mut ops);
        assert!(ops.len() > 8);
        for op in &ops {
            assert_eq!(op.kind & 1, 1, "dungeon part emitted solid");
        }
        // One connected void (flood fill over overlapping AABBs).
        let touches = |a: &CsgOp, b: &CsgOp| {
            let (amin, amax) = a.aabb();
            let (bmin, bmax) = b.aabb();
            amin.cmple(bmax + Vec3::splat(0.5)).all() && bmin.cmple(amax + Vec3::splat(0.5)).all()
        };
        let mut joined = vec![false; ops.len()];
        joined[0] = true;
        let mut grew = true;
        while grew {
            grew = false;
            for i in 0..ops.len() {
                if !joined[i] && (0..ops.len()).any(|j| joined[j] && touches(&ops[i], &ops[j])) {
                    joined[i] = true;
                    grew = true;
                }
            }
        }
        assert!(joined.iter().all(|&j| j), "dungeon is disconnected");
        // The entrance breaks the surface.
        let surface = generator().height(Vec2::new(site.x, site.z), 1.0);
        assert!(
            ops.iter().any(|op| {
                let (min, max) = op.aabb();
                min.y < surface && max.y > surface
            }),
            "no cut crosses the surface"
        );
    }

    #[test]
    fn weighted_variants_are_selected() {
        let part = |material: u32| Part {
            arrange: Arrange::Single,
            shape: Shape::Boxy {
                half: [
                    Extent::Range([1.0, 1.0]),
                    Extent::Range([1.0, 1.0]),
                    Extent::Range([1.0, 1.0]),
                ],
            },
            material,
            cut: false,
            hollow: None,
            skip: 0.0,
            seat: Seat::Site,
            anchor: Anchor::Center,
            y_offset: [0.0, 0.0],
            yaw: Yaw::Zero,
            link: None,
        };
        let s = Structure {
            size: [1.0, 1.0],
            variants: vec![
                Variant {
                    weight: 9.0,
                    parts: vec![part(7)],
                },
                Variant {
                    weight: 1.0,
                    parts: vec![part(8)],
                },
            ],
        };
        let mut common = 0;
        let mut rare = 0;
        for salt in 0..200 {
            let mut r = rng(salt);
            let mut out = Vec::new();
            build(&s, Vec3::ZERO, &generator(), &mut r, &mut out);
            match out[0].material {
                7 => common += 1,
                8 => rare += 1,
                m => panic!("unexpected material {m}"),
            }
        }
        assert!(rare > 0, "rare variant never chosen");
        assert!(common > rare * 3, "weights ignored: {common} vs {rare}");
    }
}
