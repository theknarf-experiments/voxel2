//! Named layer instances: a data-driven stack registers N differently-
//! parameterized instances of one layer TYPE. Each instance has its own
//! cache, its own seed stream, and is addressable as a dependency by
//! instance name.

use glam::{IVec3, Vec2};
use voxel_layers::{Dep, IAabb, Layer, LayerCtx, LayerManager};

const CHUNK_M: i32 = 100;

#[derive(Clone)]
struct ScatterLayer {
    per_chunk: usize,
}

struct ScatterChunk {
    points: Vec<Vec2>,
}

impl Layer for ScatterLayer {
    type Chunk = ScatterChunk;
    const NAME: &'static str = "test/scatter";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CHUNK_M, 0, CHUNK_M)
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> ScatterChunk {
        let mut rng = ctx.rng();
        let b = ctx.chunk_bounds();
        let origin = Vec2::new(b.min.x as f32, b.min.z as f32);
        ScatterChunk {
            points: (0..self.per_chunk)
                .map(|_| origin + Vec2::new(rng.next_f32(), rng.next_f32()) * CHUNK_M as f32)
                .collect(),
        }
    }
}

/// Counts points from a NAMED scatter dependency.
struct CountLayer {
    source: String,
}

struct CountChunk {
    count: usize,
}

impl Layer for CountLayer {
    type Chunk = CountChunk;
    const NAME: &'static str = "test/count";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CHUNK_M, 0, CHUNK_M)
    }

    fn dependencies(&self) -> Vec<Dep> {
        vec![Dep::named(&self.source, IVec3::ZERO)]
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> CountChunk {
        let view = ctx.get_named::<ScatterLayer>(&self.source, ctx.chunk_bounds());
        CountChunk {
            count: view.iter().map(|(_, c)| c.points.len()).sum(),
        }
    }
}

#[test]
fn instances_have_independent_configs_seeds_and_caches() {
    let mut mgr = LayerManager::new(9);
    mgr.register_as("sparse", ScatterLayer { per_chunk: 2 });
    mgr.register_as("dense", ScatterLayer { per_chunk: 9 });
    let bounds = IAabb::new(IVec3::ZERO, IVec3::new(1, 1, 1));

    let sparse = mgr.get_named::<ScatterLayer>("sparse", bounds);
    let dense = mgr.get_named::<ScatterLayer>("dense", bounds);
    let (_, sc) = sparse.iter().next().unwrap();
    let (_, dc) = dense.iter().next().unwrap();
    assert_eq!(sc.points.len(), 2);
    assert_eq!(dc.points.len(), 9);
    // Different instances draw from different seed streams.
    assert_ne!(sc.points[0], dc.points[0]);
}

#[test]
fn dependencies_resolve_by_instance_name() {
    let mut mgr = LayerManager::new(9);
    mgr.register_as("sparse", ScatterLayer { per_chunk: 2 });
    mgr.register_as("dense", ScatterLayer { per_chunk: 9 });
    mgr.register_as(
        "count_dense",
        CountLayer {
            source: "dense".into(),
        },
    );
    let bounds = IAabb::new(IVec3::ZERO, IVec3::new(1, 1, 1));
    let view = mgr.get_named::<CountLayer>("count_dense", bounds);
    let (_, c) = view.iter().next().unwrap();
    assert_eq!(c.count, 9);
}

#[test]
#[should_panic(expected = "registered twice")]
fn duplicate_instance_name_panics() {
    let mut mgr = LayerManager::new(9);
    mgr.register_as("a", ScatterLayer { per_chunk: 1 });
    mgr.register_as("a", ScatterLayer { per_chunk: 2 });
}

#[test]
fn typed_api_remains_the_default_instance() {
    let mut mgr = LayerManager::new(9);
    mgr.register(ScatterLayer { per_chunk: 3 });
    let bounds = IAabb::new(IVec3::ZERO, IVec3::new(1, 1, 1));
    let view = mgr.get::<ScatterLayer>(bounds);
    let (_, c) = view.iter().next().unwrap();
    assert_eq!(c.points.len(), 3);
}
