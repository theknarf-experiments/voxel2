//! Dependency-driven generation (LayerProcGen's EnsureLoadedInBounds):
//! ensure-load resolves the whole closure up front, in parallel, so that
//! subsequent reads never generate.

use glam::IVec3;
use voxel_layers::{Dep, IAabb, Layer, LayerCtx, LayerManager};

struct Base;
struct BaseChunk {
    value: i32,
}

impl Layer for Base {
    type Chunk = BaseChunk;
    const NAME: &'static str = "ensure/base";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(64, 0, 64)
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, coord: IVec3) -> BaseChunk {
        let mut rng = ctx.rng();
        BaseChunk {
            value: coord.x * 31 + coord.z + (rng.next_f32() * 1000.0) as i32,
        }
    }
}

/// Reads its dependency across a declared 64 m padding, so ensure-load
/// must expand the region before generating the lower layer.
struct Top;
struct TopChunk {
    sum: i32,
}

impl Layer for Top {
    type Chunk = TopChunk;
    const NAME: &'static str = "ensure/top";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(64, 0, 64)
    }

    fn dependencies(&self) -> Vec<Dep> {
        vec![Dep::of::<Base>(IVec3::new(64, 0, 64))]
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> TopChunk {
        let padded = ctx.chunk_bounds().inflate(IVec3::new(64, 0, 64));
        let sum = ctx
            .get::<Base>(padded)
            .iter()
            .map(|(_, c)| c.value)
            .sum::<i32>();
        TopChunk { sum }
    }
}

fn build() -> LayerManager {
    let mut mgr = LayerManager::new(7);
    mgr.register(Base);
    mgr.register(Top);
    mgr
}

fn bounds(r: i32) -> IAabb {
    IAabb::new(IVec3::new(-r, 0, -r), IVec3::new(r, 1, r))
}

#[test]
fn ensure_load_covers_the_closure_so_reads_never_generate() {
    let mgr = build();
    let region = bounds(512);
    let stats = mgr.ensure_loaded("ensure/top", region);
    // Exact closure: 16x16 top chunks, plus 18x18 base chunks (the region
    // expanded by the declared 64 m padding). ensure_loaded only counts
    // what IT scheduled, so this equality proves the dependency was
    // resolved up front rather than lazily inside top's generate.
    assert_eq!(stats.generated, 16 * 16 + 18 * 18);
    assert_eq!(
        mgr.read_generated(),
        0,
        "ensure-load pass leaked read-driven generation"
    );

    // Reading the ensured region is pure cache hits.
    let view = mgr.get::<Top>(region);
    assert_eq!(view.len(), 16 * 16);
    assert_eq!(
        mgr.read_generated(),
        0,
        "reads inside the ensured region generated"
    );

    // A second pass finds everything present.
    let again = mgr.ensure_loaded("ensure/top", region);
    assert_eq!(again.generated, 0);
    assert!(again.present > 0);

    // Reading OUTSIDE it still works (nothing silently missing) but is
    // counted as read-driven.
    let outside = IAabb::new(IVec3::new(4096, 0, 4096), IVec3::new(4160, 1, 4160));
    assert_eq!(mgr.get::<Top>(outside).len(), 1);
    assert!(mgr.read_generated() > 0, "read-driven generation not counted");
}

#[test]
fn parallel_ensure_load_matches_lazy_generation() {
    let region = bounds(256);
    let eager = build();
    eager.ensure_loaded("ensure/top", region);
    let eager: Vec<(IVec3, i32)> = eager
        .get::<Top>(region)
        .iter()
        .map(|(c, chunk)| (c, chunk.sum))
        .collect();

    let lazy = build();
    let lazy: Vec<(IVec3, i32)> = lazy
        .get::<Top>(region)
        .iter()
        .map(|(c, chunk)| (c, chunk.sum))
        .collect();

    assert_eq!(eager, lazy, "parallel ensure-load changed results");
}
