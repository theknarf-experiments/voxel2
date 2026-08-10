//! Ribbon surfaces, as a layer.
//!
//! The planning stack says where ribbons are and which material they
//! carry; that they are rivers, and that rivers look like water, is this
//! game's choice — so turning ribbon segments into water geometry is a
//! layer of this game, sitting on top of the emit layers that produce
//! them.
//!
//! It used to be a hand-rolled tile streamer with its own radius, its own
//! staleness scan, a per-frame budget and a compile-time assert that its
//! radius stayed inside whatever the engine happened to pre-generate. All
//! of that was a worse re-implementation of the dependency graph. What is
//! left is a chunk that declares its padding, fills itself in, and gives
//! the geometry back when nothing needs it.

use bevy::math::DVec3;
use bevy::prelude::*;
use voxel_layers::{ChunkCtx, Dep, Layer, LayerChunk, LayerGraph, TopDep};

use crate::planning::world::WorldCtx;
use crate::water::{RiverSegGpu, RiverWater};
use voxel_engine::level::{LevelDef, MaterialDef};

/// Used when a ribbon's material id is not in the level table.
const FALLBACK_TINT: [f32; 4] = [0.16, 0.34, 0.44, 0.0];

/// The levelled (water) scale: dense courses close to the camera.
///
/// The view distance is where the water surface hands over to the painted
/// ground, and it is not free to be any number. A course's BED is carved
/// by ops the level gates at 140 m of chunk edge — level 5, which the LOD
/// field shows out to `2·split_k·E₅` = 512 m. Drawing the surface further
/// than that lays it over ground with no channel cut in it, and painting
/// nearer draws the same river twice. So both ends meet here, and
/// [`crate::surface_paint`] derives its handover from this.
pub const RIBBON_NEAR_TILE_M: i32 = 256;
pub const RIBBON_NEAR_VIEW_M: i32 = 512;

/// Ribbon segments reach beyond the chunk that owns them, so a tile has to
/// see its neighbours' emitters to catch the ones crossing into it.
const RIBBON_PAD_M: i32 = 512;

/// Vertical band a ribbon tile reads; ribbons sit on the surface, but the
/// emit layers that carry them can be volumetric.
const RIBBON_Y_M: i32 = 4096;

/// Turns the emit layers' ribbon segments into drawable surface strips.
///
/// Registered once per SCALE. A levelled ribbon (a water course) is dense
/// and near, so it tiles finely over a few kilometres; a seated ribbon (a
/// road) is sparse and wants to reach the horizon, so it tiles coarsely
/// over tens. Same layer, same geometry, different grid — which is the
/// whole reason instances are named rather than typed.
pub struct RibbonSurface {
    /// Emit instances that produce ribbons.
    sources: Vec<String>,
    /// Material id → color, resolved from the level once.
    palette: Vec<(u32, [f32; 4])>,
    tile_m: i32,
    pad_m: i32,
}

#[derive(Default)]
pub struct RibbonSurfaceChunk;

impl Layer for RibbonSurface {
    type Chunk = RibbonSurfaceChunk;
    const NAME: &'static str = "ribbon-surface";

    fn chunk_extent(&self) -> DVec3 {
        DVec3::new(self.tile_m as f64, 0.0, self.tile_m as f64)
    }

    fn dependencies(&self) -> Vec<Dep> {
        self.sources
            .iter()
            .map(|name| Dep::named(name, IVec3::new(self.pad_m, RIBBON_Y_M, self.pad_m)))
            .collect()
    }
}

impl RibbonSurface {
    fn color(&self, id: u32) -> [f32; 4] {
        self.palette
            .iter()
            .find_map(|(mid, c)| (*mid == id).then_some(*c))
            .unwrap_or(FALLBACK_TINT)
    }
}

impl LayerChunk for RibbonSurfaceChunk {
    type Layer = RibbonSurface;

    fn create(&mut self, ctx: &ChunkCtx<'_, RibbonSurface>) {
        let layer = ctx.layer();
        let own = ctx.chunk_bounds();
        let pad = IVec3::new(layer.pad_m, RIBBON_Y_M, layer.pad_m);
        let mut segs = Vec::new();
        for source in &layer.sources {
            ctx.get_named::<crate::planning::layers::EmitPatches>(
                source,
                voxel_layers::dep_bounds(own, pad),
            )
            .for_each(|_, chunk| {
                for s in &chunk.patches.ribbons {
                    segs.push(RiverSegGpu {
                        ab: [s.a.x, s.a.y, s.b.x, s.b.y],
                        geo: [s.half_w, s.levels[0], s.levels[1], 0.0],
                        color: layer.color(s.material),
                    });
                }
            });
        }
        ctx.context::<WorldCtx>()
            .ribbons
            .put(ctx.instance_key(), ctx.coord(), segs);
    }

    fn destroy(&mut self, ctx: &ChunkCtx<'_, RibbonSurface>) {
        ctx.context::<WorldCtx>()
            .ribbons
            .take(ctx.instance_key(), ctx.coord());
    }
}

/// Register the layer and its top dependency. Called while the world's
/// graph is being built, after the emit layers it reads.
pub fn register(
    graph: &mut LayerGraph,
    level: &LevelDef,
    instance: &str,
    ribbon_sources: Vec<String>,
    tile_m: i32,
    view_m: i32,
) -> Option<TopDep> {
    if ribbon_sources.is_empty() {
        return None; // a level with no ribbons registers no ribbon layer
    }
    let palette = level
        .materials
        .iter()
        .filter_map(|m| match m {
            MaterialDef::Surface { id, base, .. } => Some((*id, [base[0], base[1], base[2], 0.0])),
            _ => None,
        })
        .collect();
    graph.register_as(
        instance,
        RibbonSurface {
            sources: ribbon_sources,
            palette,
            tile_m,
            // A segment reaches beyond the cell that owns it, and a
            // coarse cell's segments reach proportionally further.
            pad_m: (tile_m * 2).max(RIBBON_PAD_M),
        },
    );
    Some(TopDep::new(instance, IVec3::new(2 * view_m, 0, 2 * view_m)))
}

/// Rebuild each world's water buffer when its resident set changed.
///
/// Every loaded world, not the launched one. A course is world content
/// and worlds share coordinates, so publishing only world 0's put the
/// launch level's rivers over whatever level you were standing in and
/// left that level's own courses undrawn.
fn publish_ribbons(worlds: Res<voxel_engine::Worlds>, mut rivers: ResMut<RiverWater>) {
    for world in worlds.iter() {
        let Some(sink) = world
            .query
            .host_ctx::<WorldCtx>()
            .map(|c| c.ribbons.clone())
        else {
            continue;
        };
        let generation = sink.generation();
        // Bypassed, then set explicitly: reading the map through `ResMut`
        // marks it changed every frame, and the render world re-uploads
        // every world's segment buffer when it is.
        let entry = rivers
            .bypass_change_detection()
            .0
            .entry(world.id)
            .or_default();
        if generation == entry.generation {
            continue;
        }
        entry.segments = sink.collect();
        entry.generation = generation;
        rivers.set_changed();
    }
}

/// Draws the level's ribbon surfaces as this game's water.
pub struct RibbonsPlugin;

impl Plugin for RibbonsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RiverWater>()
            .add_systems(Update, publish_ribbons);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface_for(level: &LevelDef) -> RibbonSurface {
        RibbonSurface {
            tile_m: RIBBON_NEAR_TILE_M,
            pad_m: RIBBON_PAD_M,
            sources: vec!["rivers".into()],
            palette: level
                .materials
                .iter()
                .filter_map(|m| match m {
                    MaterialDef::Surface { id, base, .. } => {
                        Some((*id, [base[0], base[1], base[2], 0.0]))
                    }
                    _ => None,
                })
                .collect(),
        }
    }

    #[test]
    fn water_color_reads_the_material_table_with_fallback() {
        let mut level = LevelDef::from_json(
            include_str!("../../../levels/planet.json"),
            &crate::planning::nodes::kinds(),
        )
        .unwrap();
        // planet material 4 is the river surface color.
        let c = surface_for(&level).color(4);
        assert!(c[0] > 0.0 && c[2] > c[0], "unexpected river tint {c:?}");
        level.materials.clear();
        assert_eq!(surface_for(&level).color(4), FALLBACK_TINT);
    }
}
