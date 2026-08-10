//! Scatter: deterministic placement of props on the generated surface.
//!
//! This module decides **where** things go and nothing about what they
//! look like, and nothing about when they exist. [`tile_placements`] is a
//! pure function of a tile and its [`PlacementInputs`]: a game calls it
//! from a layer, which is what decides residency, and dresses the results
//! with its own models, materials, colliders and gameplay components.
//!
//! Placement is gated by the world the engine already knows: terrain
//! height and slope, altitude bands with soft falloff, generator field
//! registers, blended host gate weights, coherent patch noise, planning
//! clearance (paths, ribbon beds) and carved ground (cave mouths).

use std::sync::Arc;

use bevy::prelude::*;
use voxel_core::seed::{chunk_seed, Rng};

use crate::level::ScatterDef;

/// What the engine attaches to every placed entity. The host reads it to
/// decide what to attach in turn.
#[derive(Component, Debug, Clone)]
pub struct ScatterInstance {
    /// The level-declared class this placement belongs to ("tree", …).
    pub class: Arc<str>,
    /// Index into the class's `variants` — species, size tier, whatever
    /// the level author meant by it.
    pub variant: u32,
    /// Uniform scale already applied to the entity's `Transform`, handed
    /// over so hosts can size non-uniform models or effects from it.
    pub scale: f32,
    /// Per-placement seed for host-side variation (mesh choice, tint).
    pub seed: u64,
}

/// One deterministic placement. Hosts that render their own batches get
/// these from [`tile_placements`].
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: f32,
    pub variant: u32,
    pub seed: u64,
}

const SCATTER_SALT: u64 = 0x5CA7;

// --- gates ------------------------------------------------------------------

/// Soft altitude-band gate: 1 inside, fading linearly to 0 across
/// `falloff` meters at each edge (0 falloff = hard band).
fn altitude_gate(alt: [f32; 2], falloff: f32, y: f32) -> f32 {
    if falloff <= 0.0 {
        return if (alt[0]..alt[1]).contains(&y) { 1.0 } else { 0.0 };
    }
    (((y - alt[0]) / falloff).clamp(0.0, 1.0)).min(((alt[1] - y) / falloff).clamp(0.0, 1.0))
}

/// Generator field-register density (see `WOP_FIELD`).
fn field_gate(
    generator: &voxel_worldgen::Generator,
    density: &Option<crate::level::FieldDensityDef>,
    xz: Vec2,
) -> f32 {
    density.as_ref().map_or(1.0, |d| {
        let f = generator.fields(xz)[(d.field as usize).min(3)];
        (f * d.scale + d.offset).clamp(0.0, 1.0)
    })
}

/// Everything placement reads from outside itself, gathered once per tile.
///
/// Explicit rather than a `WorldQuery` handle, because a layer must read
/// through the dependencies it declared — it has no facade to consult, by
/// design. The same inputs serve a caller that does have one.
pub struct PlacementInputs<'a> {
    pub generator: &'a voxel_worldgen::Generator,
    /// Path and ribbon beds props must stay off.
    pub clearance: Vec<[Vec2; 2]>,
    /// Carved voids props must not float over.
    pub cut_ops: Vec<voxel_core::csg::CsgOp>,
    /// Blended weight of this population's host gate at a point. What
    /// the gate classifies is the host's business; this is only a number.
    pub gate_weight: Box<dyn Fn(Vec2) -> f32 + 'a>,
}

/// Clearance the planning stack reserved (path and ribbon beds).
const CLEAR_M: f32 = 4.5;

fn on_clearance(segments: &[[Vec2; 2]], p: Vec2) -> bool {
    segments
        .iter()
        .any(|[a, b]| voxel_worldgen::path::dist_to_segment(p, *a, *b) < CLEAR_M)
}

/// Was the surface here carved away? Props must not float over a void.
fn carved(cut_ops: &[voxel_core::csg::CsgOp], p: Vec3) -> bool {
    cut_ops.iter().any(|op| op.sdf(p) < 0.6)
}

/// Align-to-normal, random tilt cone, then yaw.
fn placement_rotation(
    generator: &voxel_worldgen::Generator,
    rules: &crate::level::PlacementRulesDef,
    xz: Vec2,
    yaw: f32,
    rng: &mut Rng,
) -> Quat {
    let base = if rules.align == "normal" {
        Quat::from_rotation_arc(Vec3::Y, generator.normal(xz, 4.0))
    } else {
        Quat::IDENTITY
    };
    let tilt = if rules.tilt_deg > 0.0 {
        let dir = rng.next_f32() * std::f32::consts::TAU;
        let angle = rng.next_f32() * rules.tilt_deg.to_radians();
        Quat::from_axis_angle(Vec3::new(dir.cos(), 0.0, dir.sin()), angle)
    } else {
        Quat::IDENTITY
    };
    base * tilt * Quat::from_rotation_y(yaw)
}

// --- placement --------------------------------------------------------------

/// Every placement of `def` in one tile — deterministic from the world
/// seed and the tile coordinate, so any consumer (the entity streamer,
/// a host's impostor batcher) sees exactly the same props.
pub fn tile_placements(def: &ScatterDef, inputs: &PlacementInputs<'_>, tile: IVec2) -> Vec<Placement> {
    let generator = inputs.generator;
    let size = def.tile_m;
    let mut rng = Rng::new(chunk_seed(
        generator.seed() as u64,
        SCATTER_SALT ^ class_salt(&def.class),
        IVec3::new(tile.x, 0, tile.y),
    ));
    let origin = tile.as_vec2() * size;
    let cut_ops = &inputs.cut_ops;
    let clearance = &inputs.clearance;

    // Coherent patch noise: woods come in stands with clearings between.
    let density = def
        .patch
        .as_ref()
        .map(|p| {
            generator.patch_density(
                origin + Vec2::splat(size * 0.5),
                p.scale,
                Vec2::from(p.offset),
                p.contrast,
                p.bias,
            )
        })
        .unwrap_or(1.0);
    let attempts = (def.per_tile as f32 * density) as u32;

    let mut out = Vec::new();
    for _ in 0..attempts {
        let xz = origin + Vec2::new(rng.next_f32(), rng.next_f32()) * size;
        if rng.next_f32() > field_gate(generator, &def.density, xz) * (inputs.gate_weight)(xz) {
            continue;
        }
        if def.clearance && on_clearance(clearance, xz) {
            continue;
        }
        // Seat on the band-limited surface the mid-LOD terrain shows
        // across the streaming radius (tiles appear at the rim).
        let y = generator.height(xz, def.detail_vs);
        if carved(cut_ops, Vec3::new(xz.x, y, xz.y)) {
            continue;
        }
        let gate = altitude_gate(def.altitude, def.placement.altitude_falloff, y);
        if gate <= 0.0 || (gate < 1.0 && rng.next_f32() > gate) {
            continue;
        }
        let up = generator.up(xz, def.detail_vs);
        if up < def.min_up || up > def.placement.max_up || rng.next_f32() >= def.chance {
            continue;
        }
        let yaw = rng.next_f32() * std::f32::consts::TAU;
        let roll = rng.next_f32();
        // Point populations have no variants: a point is a position and a
        // hash, so there is nothing to pick and nothing to scale.
        let (variant, range) = if def.variants.is_empty() {
            (0usize, [1.0, 1.0])
        } else {
            let Some(variant) = pick_variant(def, y, roll) else {
                continue;
            };
            (variant, def.variants[variant].scale)
        };
        let t = rng.next_f32().powf(def.scale_bias.max(0.01));
        let scale = range[0] + t * (range[1] - range[0]);
        let sink = def
            .placement
            .sink
            .unwrap_or(def.sink_m + def.sink_scaled * scale);
        out.push(Placement {
            position: Vec3::new(xz.x, y - sink, xz.y),
            rotation: placement_rotation(generator, &def.placement, xz, yaw, &mut rng),
            scale,
            variant: variant as u32,
            seed: rng.next_u64(),
        });
    }
    out
}

/// The share of attempts at `xz` that a placement would survive — how
/// much of this ground the population covers, without placing anything.
///
/// For consumers that must draw a population where its instances do not
/// exist: past its streaming radius, or past the range where drawing them
/// one by one stops being worth it. Built from the same gate helpers
/// [`tile_placements`] applies per attempt, so the two cannot drift; it
/// deliberately does not reuse the placement LOOP, whose early-outs
/// consume the rng in an order every world's props already depend on.
///
/// The gates it leaves out are the ones that are not functions of position
/// at the scale this is read: clearance and carved ground are metres wide,
/// and a caller asking "is there forest here" is asking about kilometres.
pub fn coverage(def: &ScatterDef, inputs: &PlacementInputs<'_>, xz: Vec2) -> f32 {
    let generator = inputs.generator;
    let patch = def.patch.as_ref().map_or(1.0, |p| {
        generator.patch_density(xz, p.scale, Vec2::from(p.offset), p.contrast, p.bias)
    });
    // Cheapest gates first: each one that zeroes here saves a heightfield
    // evaluation, and this is called once per texel of a raster.
    let w = patch * def.chance * field_gate(generator, &def.density, xz) * (inputs.gate_weight)(xz);
    if w <= 0.0 {
        return 0.0;
    }
    let y = generator.height(xz, def.detail_vs);
    let w = w * altitude_gate(def.altitude, def.placement.altitude_falloff, y);
    if w <= 0.0 {
        return 0.0;
    }
    let up = generator.up(xz, def.detail_vs);
    if up < def.min_up || up > def.placement.max_up {
        return 0.0;
    }
    w
}

/// Weighted pick among the variants whose altitude band contains `y`.
fn pick_variant(def: &ScatterDef, y: f32, roll: f32) -> Option<usize> {
    let eligible = |v: &crate::level::ScatterVariantDef| (v.altitude[0]..v.altitude[1]).contains(&y);
    let total: f32 = def
        .variants
        .iter()
        .filter(|v| eligible(v))
        .map(|v| v.weight)
        .sum();
    if total <= 0.0 {
        return None;
    }
    let mut pick = roll * total;
    for (i, v) in def.variants.iter().enumerate() {
        if !eligible(v) {
            continue;
        }
        if pick < v.weight {
            return Some(i);
        }
        pick -= v.weight;
    }
    None
}

/// Distinct seed stream per class, so adding a class never reshuffles
/// the others.
fn class_salt(class: &str) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for b in class.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x1000_0000_01B3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped_planet() -> crate::LevelDef {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../levels/planet.json");
        crate::LevelDef::from_json(
            &std::fs::read_to_string(path).unwrap(),
            &crate::graph::registry::engine_kinds(),
        )
        .unwrap()
    }

    /// The whole point of [`coverage`]: a population painted where its
    /// instances are not must be painted where they WOULD be.
    ///
    /// Compared as a correlation over tiles rather than tile by tile —
    /// coverage is the expected share of surviving attempts and placement
    /// is one draw of it, so a single 64 m tile is noise. What must hold
    /// is that the tiles coverage calls empty are empty and the ones it
    /// calls full are full.
    #[test]
    fn coverage_predicts_where_the_placer_actually_places() {
        let level = shipped_planet();
        let generator = level.generator(0);
        let def = level
            .scatter
            .iter()
            .find(|s| s.cover.is_some())
            .expect("planet has a population that paints");
        let gen = &generator;
        let inputs = PlacementInputs {
            generator: &generator,
            clearance: Vec::new(),
            cut_ops: Vec::new(),
            // The planet gates this population on its forest region.
            gate_weight: Box::new(move |xz| gen.surface_material_weight(xz, 8.0, 1)),
        };

        // Over a patch that has forest AND its edge. The origin is ocean,
        // so a window there would compare zero against zero.
        let centre = IVec2::new(-16384, -31744) / def.tile_m as i32;
        let mut pairs = Vec::new();
        for tz in -30..30 {
            for tx in -30..30 {
                let tile = centre + IVec2::new(tx, tz);
                let placed = tile_placements(def, &inputs, tile).len() as f32;
                let mid = (tile.as_vec2() + Vec2::splat(0.5)) * def.tile_m;
                pairs.push((coverage(def, &inputs, mid), placed));
            }
        }
        assert!(
            pairs.iter().any(|&(c, _)| c > 0.0),
            "the probe found no forest at all — pick a gate that exists"
        );

        // Most of the population must live where coverage saw it. Not ALL
        // of it: coverage is one point and a tile is 900, so a midpoint
        // that lands on a slope steeper than `min_up` reads zero for a
        // tile whose flatter corners are wooded. That is a point sample's
        // error, not drift — the raster samples on a grid and interpolates
        // for exactly this reason.
        let total: f32 = pairs.iter().map(|p| p.1).sum();
        let missed: f32 = pairs.iter().filter(|p| p.0 <= 0.0).map(|p| p.1).sum();
        let missed = missed / total;
        assert!(
            missed < 0.25,
            "coverage saw none of {:.0}% of the population",
            100.0 * missed
        );

        // And they rise together. Measured at 0.89 and 14% when this was
        // written; the bar is where a real drift would land, not where the
        // sampling noise does.
        let n = pairs.len() as f32;
        let (mx, my) = (
            pairs.iter().map(|p| p.0).sum::<f32>() / n,
            pairs.iter().map(|p| p.1).sum::<f32>() / n,
        );
        let cov = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum::<f32>();
        let sx = pairs.iter().map(|(x, _)| (x - mx).powi(2)).sum::<f32>().sqrt();
        let sy = pairs.iter().map(|(_, y)| (y - my).powi(2)).sum::<f32>().sqrt();
        let r = cov / (sx * sy);
        assert!(
            r > 0.8,
            "coverage and placement have drifted apart: correlation {r:.3}"
        );
    }
}
