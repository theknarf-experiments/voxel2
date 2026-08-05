//! Rolling eviction: chunks intersecting the keep region survive (no
//! regeneration on re-request), chunks outside are dropped and regenerate.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use glam::IVec3;
use voxel_layers::{IAabb, Layer, LayerCtx, LayerManager};

const CHUNK_M: i32 = 100;

struct CountingLayer {
    generated: Arc<AtomicUsize>,
}

struct CountingChunk;

impl Layer for CountingLayer {
    type Chunk = CountingChunk;
    const NAME: &'static str = "test/counting";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CHUNK_M, 0, CHUNK_M)
    }

    fn generate(&self, _ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> CountingChunk {
        self.generated.fetch_add(1, Ordering::SeqCst);
        CountingChunk
    }
}

#[test]
fn evict_outside_keeps_near_chunks_and_drops_far_ones() {
    let generated = Arc::new(AtomicUsize::new(0));
    let mut mgr = LayerManager::new(7);
    mgr.register(CountingLayer {
        generated: generated.clone(),
    });

    // Generate a near chunk (around origin) and a far one.
    let near = IAabb::new(IVec3::new(0, 0, 0), IVec3::new(1, 1, 1));
    let far = IAabb::new(IVec3::new(10_000, 0, 10_000), IVec3::new(10_001, 1, 10_001));
    mgr.get::<CountingLayer>(near);
    mgr.get::<CountingLayer>(far);
    let after_first = generated.load(Ordering::SeqCst);
    assert!(after_first >= 2);

    // Keep only the neighborhood of the origin.
    let keep = IAabb::new(IVec3::new(-500, -500, -500), IVec3::new(500, 500, 500));
    mgr.evict_outside(keep);

    // Near chunk survives: re-request must not regenerate.
    mgr.get::<CountingLayer>(near);
    assert_eq!(generated.load(Ordering::SeqCst), after_first, "near chunk was evicted");

    // Far chunk was dropped: re-request regenerates.
    mgr.get::<CountingLayer>(far);
    assert_eq!(
        generated.load(Ordering::SeqCst),
        after_first + 1,
        "far chunk was not evicted"
    );
}

#[test]
fn evict_outside_reports_dropped_count() {
    let generated = Arc::new(AtomicUsize::new(0));
    let mut mgr = LayerManager::new(7);
    mgr.register(CountingLayer { generated });
    let wide = IAabb::new(IVec3::new(0, 0, 0), IVec3::new(1000, 1, 1000));
    mgr.get::<CountingLayer>(wide);
    let cached = mgr.cached_chunks();
    assert!(cached >= 100); // 11x11 chunks of 100 m
    let keep = IAabb::new(IVec3::new(0, -10, 0), IVec3::new(199, 10, 199));
    let dropped = mgr.evict_outside(keep);
    assert!(dropped > 0);
    assert_eq!(mgr.cached_chunks(), cached - dropped);
}
