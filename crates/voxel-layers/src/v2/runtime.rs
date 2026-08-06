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

use crate::v2::graph::{LayerGraph, TopDep};

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
            active: AtomicBool::new(true),
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
    /// Completed top-dependency passes, so a test or a loading screen can
    /// wait for the world to catch up with the camera.
    passes: AtomicUsize,
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

impl LayerRuntime {
    /// Start generating for `tops`. Their order fixes the handle indices.
    pub fn start(graph: Arc<LayerGraph>, tops: Vec<TopDep>) -> Self {
        let shared = Arc::new(Shared {
            tops: tops.iter().map(|t| TopSlot::new(t.size())).collect(),
            stop: AtomicBool::new(false),
            generating: AtomicBool::new(false),
            passes: AtomicUsize::new(0),
        });
        let thread = std::thread::Builder::new()
            .name("voxel-layers".into())
            .spawn({
                let graph = graph.clone();
                let shared = shared.clone();
                move || run(graph, shared, tops)
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

    /// Completed passes. Waiting for this to advance twice after moving a
    /// focus guarantees the move has been acted on.
    pub fn passes(&self) -> usize {
        self.shared.passes.load(Ordering::Relaxed)
    }

    /// Block until every top dependency is satisfied. For tests and
    /// loading screens — never call it from a frame.
    pub fn wait_idle(&self) {
        let mut seen = self.passes();
        loop {
            std::thread::sleep(Duration::from_millis(1));
            let now = self.passes();
            if now == seen && !self.is_generating() {
                return;
            }
            seen = now;
        }
    }
}

impl Drop for LayerRuntime {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run(graph: Arc<LayerGraph>, shared: Arc<Shared>, mut tops: Vec<TopDep>) {
    while !shared.stop.load(Ordering::Relaxed) {
        let mut worked = false;
        for (slot, top) in shared.tops.iter().zip(tops.iter_mut()) {
            top.set_size(TopSlot::load(&slot.size));
            top.set_focus(&graph, TopSlot::load(&slot.focus));
            top.set_active(slot.active.load(Ordering::Relaxed));
            if !top.changed() {
                continue;
            }
            worked = true;
            shared.generating.store(true, Ordering::Relaxed);
            graph.process_top(top);
            shared.generating.store(false, Ordering::Relaxed);
            if shared.stop.load(Ordering::Relaxed) {
                break;
            }
        }
        shared.passes.fetch_add(1, Ordering::Relaxed);
        if !worked {
            std::thread::sleep(IDLE_SLEEP);
        }
    }
    // Release everything on the way out, so each chunk's destroy runs.
    for top in &mut tops {
        top.set_active(false);
        graph.process_top(top);
    }
}

/// App-side control of one top dependency. Every setter is a plain atomic
/// store: the frame loop never waits on generation.
#[derive(Clone)]
pub struct TopHandle {
    shared: Arc<Shared>,
    index: usize,
}

impl TopHandle {
    pub fn set_focus(&self, focus: IVec3) {
        TopSlot::store(&self.shared.tops[self.index].focus, focus);
    }

    pub fn set_size(&self, size: IVec3) {
        TopSlot::store(&self.shared.tops[self.index].size, size);
    }

    pub fn set_active(&self, active: bool) {
        self.shared.tops[self.index]
            .active
            .store(active, Ordering::Relaxed);
    }
}
