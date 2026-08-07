//! The generation thread.
//!
//! Top dependencies live on a background thread that owns them; the app
//! only ever writes a requested focus into atomics. That split is what
//! keeps a moving camera from ever blocking on generation — the frame
//! loop publishes where it is, and the thread catches up when it can.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use glam::IVec3;

use crate::graph::{LayerGraph, TopDep};

/// How long the thread idles when no top dependency has moved. Short
/// enough that a camera crossing a chunk boundary is picked up within a
/// frame, long enough not to spin a core.
const IDLE_SLEEP: Duration = Duration::from_millis(5);

struct TopSlot {
    /// Written by the app, read by the generation thread. The three axes
    /// are not read as one atomic; a torn read yields a point between two
    /// camera positions, which the next iteration corrects.
    focus: [AtomicI32; 3],
    size: [AtomicI32; 3],
    active: AtomicBool,
}

impl TopSlot {
    fn new(size: IVec3) -> Self {
        Self {
            focus: [AtomicI32::new(0), AtomicI32::new(0), AtomicI32::new(0)],
            size: [
                AtomicI32::new(size.x),
                AtomicI32::new(size.y),
                AtomicI32::new(size.z),
            ],
            // Inactive until the app says where to look. Otherwise the
            // first pass generates a whole world at the origin and the
            // first published focus throws all of it away — which for the
            // shipped planet was half of everything ever generated.
            active: AtomicBool::new(false),
        }
    }

    fn load(triple: &[AtomicI32; 3]) -> IVec3 {
        IVec3::new(
            triple[0].load(Ordering::Relaxed),
            triple[1].load(Ordering::Relaxed),
            triple[2].load(Ordering::Relaxed),
        )
    }

    fn store(triple: &[AtomicI32; 3], v: IVec3) {
        triple[0].store(v.x, Ordering::Relaxed);
        triple[1].store(v.y, Ordering::Relaxed);
        triple[2].store(v.z, Ordering::Relaxed);
    }
}

struct Shared {
    tops: Vec<TopSlot>,
    stop: AtomicBool,
    /// True while the thread is inside a `process_top`. A HUD signal, not
    /// a synchronization primitive.
    generating: AtomicBool,
    /// Bumped by every setter. A pass may only declare itself idle if
    /// this has not moved while it ran — otherwise a request that landed
    /// mid-pass would be reported as already satisfied.
    requests: AtomicUsize,
    /// Every top dependency is satisfied and nothing new has been asked
    /// for.
    idle: AtomicBool,
}

/// A running layer graph: the graph itself plus the thread that keeps its
/// top dependencies satisfied.
///
/// Dropping it stops the thread and releases every resident chunk, which
/// runs each chunk's `destroy` — so a world can be torn down without
/// leaking whatever its layers owned.
pub struct LayerRuntime {
    graph: Arc<LayerGraph>,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

/// Work that must land between ensuring the new configuration and
/// releasing the old one. See [`LayerGraph::process_tops_with`].
pub type BetweenPasses = Arc<dyn Fn(&LayerGraph) + Send + Sync>;

impl LayerRuntime {
    /// Start generating for `tops`. Their order fixes the handle indices.
    pub fn start(graph: Arc<LayerGraph>, tops: Vec<TopDep>) -> Self {
        Self::start_with(graph, tops, None)
    }

    /// Start, with work to run between ensuring and releasing on every
    /// pass.
    pub fn start_with(
        graph: Arc<LayerGraph>,
        tops: Vec<TopDep>,
        between: Option<BetweenPasses>,
    ) -> Self {
        Self::start_hooked(graph, tops, None, between)
    }

    /// Start, with work at the head of every pass as well.
    ///
    /// `before` runs before the focuses are read, which is the only place
    /// a caller can freeze whatever its layers derive from: the focuses
    /// are read one dependency at a time, so anything sampled per-chunk
    /// during a pass has to be snapshotted here or two levels can end up
    /// working from camera positions a frame apart.
    pub fn start_hooked(
        graph: Arc<LayerGraph>,
        tops: Vec<TopDep>,
        before: Option<BetweenPasses>,
        between: Option<BetweenPasses>,
    ) -> Self {
        let shared = Arc::new(Shared {
            tops: tops.iter().map(|t| TopSlot::new(t.size())).collect(),
            stop: AtomicBool::new(false),
            generating: AtomicBool::new(false),
            requests: AtomicUsize::new(0),
            idle: AtomicBool::new(false),
        });
        let thread = std::thread::Builder::new()
            .name("voxel-layers".into())
            .spawn({
                let graph = graph.clone();
                let shared = shared.clone();
                move || run(graph, shared, tops, before, between)
            })
            .expect("spawn layer thread");
        Self {
            graph,
            shared,
            thread: Some(thread),
        }
    }

    pub fn graph(&self) -> &Arc<LayerGraph> {
        &self.graph
    }

    /// Handle to the `index`-th top dependency, in the order given to
    /// [`Self::start`].
    pub fn top(&self, index: usize) -> TopHandle {
        assert!(index < self.shared.tops.len(), "no such top dependency");
        TopHandle {
            shared: self.shared.clone(),
            index,
        }
    }

    pub fn tops(&self) -> usize {
        self.shared.tops.len()
    }

    /// True while a generation pass is running.
    pub fn is_generating(&self) -> bool {
        self.shared.generating.load(Ordering::Relaxed)
    }

    /// True when every top dependency is satisfied and nothing further
    /// has been requested.
    pub fn is_idle(&self) -> bool {
        self.shared.idle.load(Ordering::Acquire)
    }

    /// Block until every top dependency is satisfied. For tests and
    /// loading screens — never call it from a frame.
    ///
    /// Panics if the generation thread has died, which means a layer's
    /// `create` panicked. Without this the wait spins forever and a plain
    /// bug in one layer reads as a hang with no output — which is exactly
    /// how it presented the first time.
    pub fn wait_idle(&self) {
        while !self.is_idle() {
            assert!(
                !self
                    .thread
                    .as_ref()
                    .is_some_and(std::thread::JoinHandle::is_finished),
                "layer generation thread stopped; a layer's create panicked",
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for LayerRuntime {
    fn drop(&mut self) {
        // Abort first: a pass mid-way through a large resident set would
        // otherwise have to finish before the join.
        self.graph.abort();
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run(
    graph: Arc<LayerGraph>,
    shared: Arc<Shared>,
    mut tops: Vec<TopDep>,
    before: Option<BetweenPasses>,
    between: Option<BetweenPasses>,
) {
    while !shared.stop.load(Ordering::Relaxed) {
        // Sampled before the pass: a request arriving while it runs must
        // not be swallowed by the idle flag it sets afterwards.
        let requests = shared.requests.load(Ordering::Acquire);
        if let Some(before) = &before {
            before(&graph);
        }
        for (slot, top) in shared.tops.iter().zip(tops.iter_mut()) {
            top.set_size(TopSlot::load(&slot.size));
            top.set_focus(&graph, TopSlot::load(&slot.focus));
            top.set_active(slot.active.load(Ordering::Relaxed));
        }
        // One pass over all of them, not one pass each: every ensure runs
        // before any release, so a region one dependency gives up and
        // another takes is never held by neither. Consecutive LOD levels
        // do exactly that at every ring boundary.
        let worked = tops.iter().any(TopDep::changed);
        if worked {
            shared.generating.store(true, Ordering::Relaxed);
            graph.process_tops_with(&mut tops, |g| {
                if let Some(between) = &between {
                    between(g);
                }
            });
            shared.generating.store(false, Ordering::Relaxed);
        }
        let quiet = !worked && shared.requests.load(Ordering::Acquire) == requests;
        shared.idle.store(quiet, Ordering::Release);
        if quiet {
            std::thread::sleep(IDLE_SLEEP);
        }
    }
    // Release everything on the way out, so each chunk's destroy runs.
    for top in &mut tops {
        top.set_active(false);
    }
    graph.process_tops(&mut tops);
}

/// App-side control of one top dependency. Every setter is a plain atomic
/// store: the frame loop never waits on generation.
#[derive(Clone)]
pub struct TopHandle {
    shared: Arc<Shared>,
    index: usize,
}

impl TopHandle {
    /// Publish where to look. The first call is also what starts this
    /// dependency generating.
    pub fn set_focus(&self, focus: IVec3) {
        self.request();
        TopSlot::store(&self.shared.tops[self.index].focus, focus);
        self.shared.tops[self.index]
            .active
            .store(true, Ordering::Relaxed);
    }

    pub fn set_size(&self, size: IVec3) {
        self.request();
        TopSlot::store(&self.shared.tops[self.index].size, size);
    }

    pub fn set_active(&self, active: bool) {
        self.request();
        self.shared.tops[self.index]
            .active
            .store(active, Ordering::Relaxed);
    }

    /// Announce a change before making it, so a pass already running
    /// cannot conclude it satisfied this one.
    fn request(&self) {
        self.shared.idle.store(false, Ordering::Release);
        self.shared.requests.fetch_add(1, Ordering::Release);
    }
}
