//! The properties that separate a dependency graph from a cache.
//!
//! Every test here fails against the old `manager`, which is the point:
//! residency is exact rather than heuristic, destruction is deterministic
//! rather than absent, reads never generate, and a dependency can name a
//! level.

use std::sync::Mutex;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use glam::IVec3;
use voxel_layers::v2::{ChunkCtx, Dep, IAabb, Layer, LayerChunk, LayerGraph, TopDep};

const CELL: i32 = 256;

/// Shared per-world context: a create/destroy ledger the tests assert on.
#[derive(Default)]
struct Ledger {
    created: AtomicUsize,
    destroyed: AtomicUsize,
    /// (layer, level, coord) in the order they were created.
    events: Mutex<Vec<(&'static str, u32, IVec3)>>,
}

impl Ledger {
    fn created_at(&self, layer: &str, level: u32) -> usize {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|(l, lv, _)| *l == layer && *lv == level)
            .count()
    }
}

// ---------------------------------------------------------------- base layer

/// Two levels, so a dependent can name one of them.
struct Base;

#[derive(Default)]
struct BaseChunk {
    /// Level 0 output: a deterministic value per chunk.
    seeded: u64,
    /// Level 1 output: level 0 blended with the neighbourhood.
    blended: u64,
}

impl Layer for Base {
    type Chunk = BaseChunk;
    const NAME: &'static str = "base";
    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CELL, 0, CELL)
    }
    fn levels(&self) -> u32 {
        2
    }
    fn level_padding(&self, _level: u32) -> IVec3 {
        IVec3::new(CELL, 0, CELL)
    }
}

impl LayerChunk for BaseChunk {
    type Layer = Base;

    fn create(&mut self, ctx: &ChunkCtx<'_, Base>, level: u32) {
        let ledger = ctx.context::<Ledger>();
        ledger.created.fetch_add(1, Ordering::Relaxed);
        ledger
            .events
            .lock()
            .unwrap()
            .push(("base", level, ctx.coord()));
        match level {
            0 => self.seeded = ctx.seed(),
            _ => {
                // Reads its own level 0 through `self`, and its neighbours'
                // through the framework — the internal-levels pattern.
                let mut sum = self.seeded;
                ctx.get_self(ctx.chunk_bounds().inflate(IVec3::new(CELL, 0, CELL)))
                    .for_each(|_, chunk| sum ^= chunk.seeded);
                self.blended = sum;
            }
        }
    }

    fn destroy(&mut self, ctx: &ChunkCtx<'_, Base>, level: u32) {
        ctx.context::<Ledger>()
            .destroyed
            .fetch_add(1, Ordering::Relaxed);
        match level {
            0 => self.seeded = 0,
            _ => self.blended = 0,
        }
    }
}

// ----------------------------------------------------------------- top layer

/// Holds no data of its own — it exists to declare dependencies, exactly
/// like LayerProcGen's `PlayLayer`.
struct Play {
    /// Which level of `base` this instance wants.
    base_level: u32,
    pad: i32,
}

#[derive(Default)]
struct PlayChunk {
    sum: u64,
}

impl Layer for Play {
    type Chunk = PlayChunk;
    const NAME: &'static str = "play";
    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CELL, 0, CELL)
    }
    fn dependencies(&self, _level: u32) -> Vec<Dep> {
        vec![Dep::named_at(
            "base",
            self.base_level,
            IVec3::new(self.pad, 0, self.pad),
        )]
    }
}

impl LayerChunk for PlayChunk {
    type Layer = Play;

    fn create(&mut self, ctx: &ChunkCtx<'_, Play>, level: u32) {
        let ledger = ctx.context::<Ledger>();
        ledger.created.fetch_add(1, Ordering::Relaxed);
        ledger
            .events
            .lock()
            .unwrap()
            .push(("play", level, ctx.coord()));
        let pad = IVec3::new(ctx.layer().pad, 0, ctx.layer().pad);
        let mut sum = 0u64;
        ctx.get_named::<Base>("base", ctx.chunk_bounds().inflate(pad))
            .for_each(|_, chunk| sum = sum.wrapping_add(chunk.seeded ^ chunk.blended));
        self.sum = sum;
    }

    fn destroy(&mut self, ctx: &ChunkCtx<'_, Play>, _level: u32) {
        ctx.context::<Ledger>()
            .destroyed
            .fetch_add(1, Ordering::Relaxed);
        self.sum = 0;
    }
}

fn graph(ledger: Arc<Ledger>, base_level: u32, pad: i32, threads: usize) -> LayerGraph {
    let mut graph = LayerGraph::with_context(0xBEEF, ledger).with_threads(threads);
    graph.register(Base);
    graph.register(Play { base_level, pad });
    graph
}

// --------------------------------------------------------------------- tests

/// The invariant the whole design exists for: what is resident is exactly
/// the transitive dependency closure of the active top dependencies —
/// before, after, and between focus moves. No timer, no distance
/// heuristic, no drift.
#[test]
fn residency_equals_dependency_closure() {
    let ledger = Arc::new(Ledger::default());
    let graph = graph(ledger.clone(), 1, CELL, 4);

    // One play chunk, whose 256 m padding pulls a 3x3 of base.
    let mut top = TopDep::new(&graph, "play", IVec3::new(1, 0, 1));
    top.set_focus(&graph, IVec3::ZERO);
    graph.process_top(&mut top);

    assert_eq!(graph.resident_in("play"), 1);
    assert_eq!(
        graph.resident_in("base"),
        25,
        "3x3 of base at level 1, each pulling a 3x3 of level 0",
    );

    // Base level 1 reads a 3x3 of base level 0, so the level-0 footprint
    // is one ring wider than the level-1 footprint.
    assert_eq!(ledger.created_at("base", 1), 9);
    assert_eq!(ledger.created_at("base", 0), 25, "5x5 for the level-1 ring");

    // Move far enough that nothing overlaps: the old closure must be gone,
    // not lingering until an eviction pass notices.
    top.set_focus(&graph, IVec3::new(CELL * 20, 0, 0));
    graph.process_top(&mut top);
    assert_eq!(graph.resident_in("play"), 1);
    assert_eq!(graph.resident_in("base"), 25);

    // Releasing the root releases everything.
    top.set_active(false);
    graph.process_top(&mut top);
    assert_eq!(graph.resident_chunks(), 0, "nothing outlives its last user");
}

/// A chunk that owns a resource must get it back. Every create is paired
/// with exactly one destroy, including across focus moves.
#[test]
fn every_create_is_paired_with_a_destroy() {
    let ledger = Arc::new(Ledger::default());
    let graph = graph(ledger.clone(), 1, CELL, 4);

    let mut top = TopDep::new(&graph, "play", IVec3::new(CELL * 2, 0, CELL * 2));
    for step in 0..6 {
        top.set_focus(&graph, IVec3::new(step * CELL, 0, step * CELL / 2));
        graph.process_top(&mut top);
    }
    top.set_active(false);
    graph.process_top(&mut top);

    assert_eq!(graph.resident_chunks(), 0);
    assert_eq!(
        ledger.created.load(Ordering::Relaxed),
        ledger.destroyed.load(Ordering::Relaxed),
        "create/destroy must balance",
    );
    assert!(ledger.created.load(Ordering::Relaxed) > 0);
}

/// Overlapping focus positions must reuse chunks rather than regenerate
/// them — the refcount is what makes a moving camera cheap.
#[test]
fn overlapping_moves_reuse_chunks() {
    let ledger = Arc::new(Ledger::default());
    let graph = graph(ledger.clone(), 0, 0, 1);

    let mut top = TopDep::new(&graph, "play", IVec3::new(CELL * 8, 0, CELL * 8));
    top.set_focus(&graph, IVec3::new(CELL / 2, 0, CELL / 2));
    graph.process_top(&mut top);
    let first = ledger.created.load(Ordering::Relaxed);

    // One cell to the side: the overwhelming majority of the window is the
    // same, so only the new column generates.
    top.set_focus(&graph, IVec3::new(CELL + CELL / 2, 0, CELL / 2));
    graph.process_top(&mut top);
    let second = ledger.created.load(Ordering::Relaxed) - first;
    assert!(
        second * 4 < first,
        "shifting by one cell regenerated {second} of {first} chunks",
    );

    // Moving back within the same chunk index is not a change at all.
    let before = ledger.created.load(Ordering::Relaxed);
    top.set_focus(&graph, IVec3::new(CELL + CELL / 2 + 1, 0, CELL / 2 + 1));
    graph.process_top(&mut top);
    assert_eq!(ledger.created.load(Ordering::Relaxed), before);
}

/// Reads report what is resident and never generate. The old manager
/// silently generated here, on the reading thread.
#[test]
fn reads_never_generate() {
    let ledger = Arc::new(Ledger::default());
    let graph = graph(ledger.clone(), 0, 0, 1);

    let far = IAabb::new(
        IVec3::new(CELL * 100, 0, CELL * 100),
        IVec3::new(CELL * 101, 0, CELL * 101),
    );
    let view = graph.view::<Base>("base", far);
    assert!(view.is_empty());
    assert!(!view.is_complete());
    assert_eq!(view.missing(), 1);
    assert_eq!(graph.reads_missed(), 1);
    assert_eq!(graph.resident_chunks(), 0, "a read must not have generated");
    assert_eq!(ledger.created.load(Ordering::Relaxed), 0);
}

/// A dependency names a level, so a consumer can depend on a partial state
/// — the mechanism that keeps an otherwise-circular graph a DAG.
#[test]
fn dependency_on_a_non_final_level_stops_there() {
    let ledger = Arc::new(Ledger::default());
    let graph = graph(ledger.clone(), 0, 0, 1);

    let mut top = TopDep::new(&graph, "play", IVec3::new(1, 0, 1));
    top.set_focus(&graph, IVec3::ZERO);
    graph.process_top(&mut top);

    assert_eq!(ledger.created_at("base", 0), 1);
    assert_eq!(
        ledger.created_at("base", 1),
        0,
        "depending on level 0 must not drag level 1 into existence",
    );
}

/// Reaching level N always walks 0..N: a chunk can never skip a pass.
#[test]
fn levels_are_walked_in_order() {
    let ledger = Arc::new(Ledger::default());
    let graph = graph(ledger.clone(), 1, 0, 1);

    let mut top = TopDep::new(&graph, "play", IVec3::new(1, 0, 1));
    top.set_focus(&graph, IVec3::ZERO);
    graph.process_top(&mut top);

    let events = ledger.events.lock().unwrap();
    let origin_levels: Vec<u32> = events
        .iter()
        .filter(|(l, _, c)| *l == "base" && *c == IVec3::ZERO)
        .map(|(_, level, _)| *level)
        .collect();
    assert_eq!(origin_levels, vec![0, 1]);
}

/// Generation is a pure function of coordinates and dependencies, so the
/// thread count cannot change the result.
#[test]
fn results_are_identical_across_thread_counts() {
    let sample = |threads: usize| -> Vec<((i32, i32, i32), u64)> {
        let graph = graph(Arc::new(Ledger::default()), 1, CELL, threads);
        let mut top = TopDep::new(&graph, "play", IVec3::new(CELL * 6, 0, CELL * 6));
        top.set_focus(&graph, IVec3::new(CELL * 3, 0, -CELL * 2));
        graph.process_top(&mut top);
        let bounds = top.bounds();
        let mut out = Vec::new();
        graph
            .view::<Play>("play", bounds)
            .for_each(|coord, chunk| out.push(((coord.x, coord.y, coord.z), chunk.sum)));
        out.sort();
        out
    };
    let serial = sample(1);
    assert!(!serial.is_empty());
    assert_eq!(serial, sample(4));
    assert_eq!(serial, sample(8));
}

/// Two top dependencies over the same region share chunks; dropping one
/// must not take the other's data with it.
#[test]
fn shared_chunks_survive_one_holder_leaving() {
    let ledger = Arc::new(Ledger::default());
    let graph = graph(ledger.clone(), 0, 0, 1);

    let mut a = TopDep::new(&graph, "play", IVec3::new(CELL * 4, 0, CELL * 4));
    let mut b = TopDep::new(&graph, "play", IVec3::new(CELL * 4, 0, CELL * 4));
    a.set_focus(&graph, IVec3::ZERO);
    b.set_focus(&graph, IVec3::ZERO);
    graph.process_top(&mut a);
    graph.process_top(&mut b);
    let resident = graph.resident_chunks();
    assert!(resident > 0);

    a.set_active(false);
    graph.process_top(&mut a);
    assert_eq!(
        graph.resident_chunks(),
        resident,
        "b still needs every one of them",
    );

    b.set_active(false);
    graph.process_top(&mut b);
    assert_eq!(graph.resident_chunks(), 0);
}

// ------------------------------------------------------------------ runtime

/// The generation thread picks up a focus published by another thread and
/// brings residency in line with it — the app never blocks on generation.
#[test]
fn runtime_follows_a_published_focus() {
    use voxel_layers::v2::LayerRuntime;

    let ledger = Arc::new(Ledger::default());
    let graph = Arc::new(graph(ledger.clone(), 0, 0, 2));
    let top = TopDep::new(&graph, "play", IVec3::new(CELL * 4, 0, CELL * 4));
    let runtime = LayerRuntime::start(graph.clone(), vec![top]);
    let handle = runtime.top(0);

    handle.set_focus(IVec3::new(CELL / 2, 0, CELL / 2));
    runtime.wait_idle();
    let resident = graph.resident_in("play");
    assert!(resident > 0, "focus published but nothing generated");

    // Somewhere else entirely: the old closure goes, the new one arrives,
    // and the resident count does not grow.
    handle.set_focus(IVec3::new(CELL * 500, 0, -CELL * 500));
    runtime.wait_idle();
    assert_eq!(graph.resident_in("play"), resident);
}

/// Dropping a world runs every chunk's destroy. Without that, a layer that
/// owned entities or GPU slots would leak them on teardown.
#[test]
fn dropping_the_runtime_releases_everything() {
    use voxel_layers::v2::LayerRuntime;

    let ledger = Arc::new(Ledger::default());
    let graph = Arc::new(graph(ledger.clone(), 1, CELL, 2));
    {
        let top = TopDep::new(&graph, "play", IVec3::new(CELL * 3, 0, CELL * 3));
        let runtime = LayerRuntime::start(graph.clone(), vec![top]);
        runtime.top(0).set_focus(IVec3::new(CELL / 2, 0, CELL / 2));
        runtime.wait_idle();
        assert!(graph.resident_chunks() > 0);
    }
    assert_eq!(graph.resident_chunks(), 0);
    assert_eq!(
        ledger.created.load(Ordering::Relaxed),
        ledger.destroyed.load(Ordering::Relaxed),
    );
}
