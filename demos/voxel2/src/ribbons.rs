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

use bevy::prelude::*;
use voxel_layers::{ChunkCtx, Dep, Layer, LayerChunk, LayerGraph, TopDep};

use crate::planning::world::WorldCtx;
use crate::water::{RiverSegGpu, RiverWater};
use voxel_engine::level::{LevelDef, MaterialDef};

/// Used when a ribbon's material id is not in the level table.
const FALLBACK_TINT: [f32; 4] = [0.16, 0.34, 0.44, 0.0];

const RIBBON_TILE_M: i32 = 256;

/// How much visible river surface to keep around the camera. Formerly a
/// tile radius policed by an assert; now simply the size of this layer's
/// top dependency, and the graph guarantees the emit layers underneath it
/// reach far enough because this layer declares that as padding.
const RIBBON_VIEW_M: i32 = 3072;

/// Ribbon segments reach beyond the chunk that owns them, so a tile has to
/// see its neighbours' emitters to catch the ones crossing into it.
const RIBBON_PAD_M: i32 = 512;

/// Vertical band a ribbon tile reads; ribbons sit on the surface, but the
/// emit layers that carry them can be volumetric.
const RIBBON_Y_M: i32 = 4096;

/// Turns the emit layers' ribbon segments into water geometry.
pub struct RibbonSurface {
    /// Emit instances that produce ribbons.
    sources: Vec<String>,
    /// Material id → color, resolved from the level once.
    palette: Vec<(u32, [f32; 4])>,
}

#[derive(Default)]
pub struct RibbonSurfaceChunk;

impl Layer for RibbonSurface {
    type Chunk = RibbonSurfaceChunk;
    const NAME: &'static str = "ribbon-surface";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(RIBBON_TILE_M, 0, RIBBON_TILE_M)
    }

    fn dependencies(&self, _level: u32) -> Vec<Dep> {
        self.sources
            .iter()
            .map(|name| Dep::named(name, IVec3::new(RIBBON_PAD_M, RIBBON_Y_M, RIBBON_PAD_M)))
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

    fn create(&mut self, ctx: &ChunkCtx<'_, RibbonSurface>, _level: u32) {
        let layer = ctx.layer();
        let own = ctx.chunk_bounds();
        let pad = IVec3::new(RIBBON_PAD_M, RIBBON_Y_M, RIBBON_PAD_M);
        let mut segs = Vec::new();
        for source in &layer.sources {
            ctx.get_named::<crate::planning::layers::EmitPatches>(source, voxel_layers::dep_bounds(own, pad))
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
        ctx.context::<WorldCtx>().ribbons.put(ctx.coord(), segs);
    }

    fn destroy(&mut self, ctx: &ChunkCtx<'_, RibbonSurface>, _level: u32) {
        ctx.context::<WorldCtx>().ribbons.take(ctx.coord());
    }
}

/// Register the layer and its top dependency. Called while the world's
/// graph is being built, after the emit layers it reads.
pub fn register(graph: &mut LayerGraph, level: &LevelDef, ribbon_sources: Vec<String>) -> Option<TopDep> {
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
    graph.register(RibbonSurface {
        sources: ribbon_sources,
        palette,
    });
    Some(TopDep::at_level(
        RibbonSurface::NAME,
        0,
        IVec3::new(2 * RIBBON_VIEW_M, 0, 2 * RIBBON_VIEW_M),
    ))
}

/// Rebuild the water pipeline's buffer when the resident set changed.
fn publish_ribbons(world: Res<voxel_engine::WorldQuery>, mut rivers: ResMut<RiverWater>) {
    let Some(sink) = world.host_ctx::<WorldCtx>().map(|c| c.ribbons.clone()) else {
        return;
    };
    let generation = sink.generation();
    if generation == rivers.generation {
        return;
    }
    rivers.segments = sink.collect();
    rivers.generation = generation;
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
        let mut level = LevelDef::from_json(include_str!("../../../levels/planet.json")).unwrap();
        // planet material 4 is the river surface color.
        let c = surface_for(&level).color(4);
        assert!(c[0] > 0.0 && c[2] > c[0], "unexpected river tint {c:?}");
        level.materials.clear();
        assert_eq!(surface_for(&level).color(4), FALLBACK_TINT);
    }
}
