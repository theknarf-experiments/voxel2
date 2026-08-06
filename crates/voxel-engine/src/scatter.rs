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
//! registers, blended biome weights, coherent patch noise, planning
//! clearance (paths, ribbon beds) and carved ground (cave mouths).

use std::sync::Arc;

use bevy::prelude::*;
use voxel_core::seed::{chunk_seed, Rng};

use crate::level::ScatterDef;
use crate::planning::WorldQuery;

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
    /// Blended weight of this population's biome reference at a point.
    pub biome_weight: Box<dyn Fn(Vec2) -> f32 + 'a>,
}

impl<'a> PlacementInputs<'a> {
    /// Gather from a world facade, for callers that have one.
    pub fn from_world(world: &'a WorldQuery, def: &'a ScatterDef, tile: IVec2) -> Self {
        let origin = tile.as_vec2() * def.tile_m;
        Self {
            generator: world.generator(),
            clearance: tile_clearance(world, origin, def.tile_m),
            cut_ops: tile_cut_ops(world, origin, def.tile_m),
            biome_weight: Box::new(move |xz| biome_gate(world, &def.biome, xz)),
        }
    }
}

/// Blended weight of an `"instance:biome"` reference (1 when unset).
fn biome_gate(world: &WorldQuery, reference: &Option<String>, xz: Vec2) -> f32 {
    let Some(reference) = reference else {
        return 1.0;
    };
    let Some((instance, biome)) = reference.rsplit_once(':') else {
        return 1.0;
    };
    world
        .biomes_at(instance, xz)
        .iter()
        .find_map(|(n, w)| (n == biome).then_some(*w))
        .unwrap_or(1.0)
}

/// Clearance the planning stack reserved (path and ribbon beds).
const CLEAR_M: f32 = 4.5;

fn tile_clearance(world: &WorldQuery, origin: Vec2, size: f32) -> Vec<[Vec2; 2]> {
    world.clearance_in(
        origin - Vec2::splat(CLEAR_M),
        origin + Vec2::splat(size + CLEAR_M),
    )
}

fn on_clearance(segments: &[[Vec2; 2]], p: Vec2) -> bool {
    segments
        .iter()
        .any(|[a, b]| voxel_worldgen::path::dist_to_segment(p, *a, *b) < CLEAR_M)
}

/// Cut ops overlapping a tile (huge y span: cave mouths at any depth).
fn tile_cut_ops(world: &WorldQuery, origin: Vec2, size: f32) -> Vec<voxel_core::csg::CsgOp> {
    world.cuts_in(
        Vec3::new(origin.x - 4.0, -10_000.0, origin.y - 4.0),
        Vec3::new(origin.x + size + 4.0, 10_000.0, origin.y + size + 4.0),
    )
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
        if rng.next_f32() > field_gate(generator, &def.density, xz) * (inputs.biome_weight)(xz) {
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
