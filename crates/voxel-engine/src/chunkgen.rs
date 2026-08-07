//! The chunk generation service: the one owner of request / ready / free.
//!
//! Everything that drives a voxel chunk's lifecycle goes through here —
//! resolving its planning ops, asking the render world to generate it,
//! learning that it became drawable, committing it, freeing it. The epoch
//! machine in [`crate::streaming`] is the only caller today; per-level
//! `VoxelLod` layers are the next one, and the two cannot both drain the
//! readiness channel or both decide when a slab is released. One owner,
//! two callers.
//!
//! The service is a cloneable handle rather than a system param because a
//! layer's `create` runs on a generation thread: it needs to request a
//! chunk and block on [`ChunkGen::wait_for`] until it exists.

use std::sync::{Arc, Mutex};

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use voxel_core::csg::CsgOp;
use voxel_core::ChunkKey;
use voxel_render::{ChunkCommand, ChunkCommandQueue, ChunkReadyChannel, ChunkWaiters};

/// Planning-layer CSG ops for one chunk, already AABB-culled to it.
pub type OpsFn = Arc<dyn Fn(ChunkKey) -> Vec<CsgOp> + Send + Sync>;

/// Optional hook supplying planning-layer CSG ops for a requested chunk.
/// Installed by the app/worldgen; the service picks it up when it changes.
#[derive(Resource, Default)]
pub struct ChunkOpsProvider(pub Option<OpsFn>);

/// Handle to the chunk generation service. See the module docs.
#[derive(Resource, Clone)]
pub struct ChunkGen(Arc<Service>);

struct Service {
    queue: ChunkCommandQueue,
    ready_rx: crossbeam_channel::Receiver<(ChunkKey, u32)>,
    waiters: ChunkWaiters,
    /// Latest drawable mesh per requested chunk, with the seam mask it was
    /// built with (`u32::MAX` = classified empty, satisfies any mask).
    ready: Mutex<HashMap<ChunkKey, u32>>,
    ops: Mutex<Option<OpsFn>>,
}

impl ChunkGen {
    pub(crate) fn new(
        queue: ChunkCommandQueue,
        ready_rx: crossbeam_channel::Receiver<(ChunkKey, u32)>,
        waiters: ChunkWaiters,
    ) -> Self {
        Self(Arc::new(Service {
            queue,
            ready_rx,
            waiters,
            ready: Mutex::new(HashMap::new()),
            ops: Mutex::new(None),
        }))
    }

    // --- ops ---------------------------------------------------------------

    pub fn set_ops_provider(&self, ops: Option<OpsFn>) {
        *self.0.ops.lock().unwrap() = ops;
    }

    /// The provider itself, for callers that resolve ops off the main
    /// thread (planning does, so provider cost never lands on a frame).
    pub fn ops_fn(&self) -> Option<OpsFn> {
        self.0.ops.lock().unwrap().clone()
    }

    /// Resolve `key`'s ops now. Empty is `None`: the density pass binds a
    /// dummy op buffer rather than an empty one.
    pub fn ops_for(&self, key: ChunkKey) -> Option<Arc<Vec<CsgOp>>> {
        resolve_ops(self.ops_fn().as_deref(), key)
    }

    // --- readiness ---------------------------------------------------------

    /// Drain the render world's readiness reports and fan them out: the
    /// batch a planner polls, and a wake for anything blocked on one chunk.
    /// Exactly one drain exists, and this is it.
    pub fn pump(&self) {
        let mut ready = self.0.ready.lock().unwrap();
        for (key, mask) in self.0.ready_rx.try_iter() {
            ready.insert(key, mask);
            self.0.waiters.notify(key, mask);
        }
    }

    /// Is `key` drawable with a mesh that satisfies `want`? An
    /// empty-classified chunk (`u32::MAX`) satisfies any seam mask.
    pub fn is_ready(&self, key: ChunkKey, want: u32) -> bool {
        matches!(self.ready_mask(key), Some(r) if r == want || r == u32::MAX)
    }

    pub fn ready_mask(&self, key: ChunkKey) -> Option<u32> {
        self.0.ready.lock().unwrap().get(&key).copied()
    }

    /// Drop readiness for every key matching `pred` without freeing
    /// anything — for keys the caller has stopped tracking.
    pub fn forget_ready_matching(&self, pred: impl Fn(ChunkKey) -> bool) {
        self.0.ready.lock().unwrap().retain(|k, _| !pred(*k));
    }

    /// A receiver that fires when `key` next becomes drawable, carrying the
    /// seam mask of the mesh. Disconnects if the chunk is freed or its
    /// request cancelled, so a blocked `create` cannot wait forever.
    pub fn wait_for(&self, key: ChunkKey) -> crossbeam_channel::Receiver<u32> {
        self.0.waiters.wait_for(key)
    }

    // --- lifecycle ---------------------------------------------------------

    /// Generate `key` with `face_mask`. `show_on_ready` draws it the moment
    /// it is drawable; otherwise it waits for [`Self::commit`]. `hold`
    /// marks an in-place remesh of an already-shown chunk: the old mesh
    /// keeps drawing until the commit swaps them.
    ///
    /// Requesting always forgets earlier readiness: a wait may only be
    /// satisfied by a report that ARRIVES after the request, or a stale
    /// entry (a held mesh cancelled by an abort, an old empty
    /// classification) lets a caller act on a mesh that no longer exists.
    pub fn request(
        &self,
        key: ChunkKey,
        face_mask: u32,
        show_on_ready: bool,
        hold: bool,
        ops: Option<Arc<Vec<CsgOp>>>,
    ) {
        self.0.ready.lock().unwrap().remove(&key);
        self.0.queue.push(ChunkCommand::Request {
            key,
            show_on_ready,
            hold,
            ops,
            face_mask,
        });
    }

    /// Make `key` visible, swapping in any held remesh.
    pub fn commit(&self, key: ChunkKey) {
        self.0.queue.push(ChunkCommand::Commit(key));
    }

    /// Drop a held remesh without swapping: the old mesh keeps drawing.
    pub fn cancel_hold(&self, key: ChunkKey) {
        self.forget(key);
        self.0.queue.push(ChunkCommand::CancelHold(key));
    }

    /// Release the chunk and its slab allocation.
    pub fn free(&self, key: ChunkKey) {
        self.forget(key);
        self.0.queue.push(ChunkCommand::Free(key));
    }

    /// Everything is gone (a full streaming rebuild): the caller has freed
    /// what it tracked, and no report from before is worth keeping.
    pub fn reset(&self) {
        self.0.ready.lock().unwrap().clear();
    }

    fn forget(&self, key: ChunkKey) {
        self.0.ready.lock().unwrap().remove(&key);
        self.0.waiters.abandon(key);
    }
}

/// Shared by the service and the planning tasks that resolve ops off-thread.
pub(crate) fn resolve_ops(
    provider: Option<&(dyn Fn(ChunkKey) -> Vec<CsgOp> + Send + Sync)>,
    key: ChunkKey,
) -> Option<Arc<Vec<CsgOp>>> {
    provider
        .map(|f| f(key))
        .filter(|v| !v.is_empty())
        .map(Arc::new)
}

/// Installs the service. Requires `voxel_render::VoxelChunksPlugin` — the
/// pipeline it drives — to have been added first.
pub struct ChunkGenPlugin;

impl Plugin for ChunkGenPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<voxel_render::VoxelChunksPlugin>(),
            "ChunkGenPlugin drives the chunk pipeline; add VoxelChunksPlugin first"
        );
        let queue = app.world().resource::<ChunkCommandQueue>().clone();
        let ready_rx = app.world().resource::<ChunkReadyChannel>().rx.clone();
        let waiters = app.world().resource::<ChunkWaiters>().clone();
        app.insert_resource(ChunkGen::new(queue, ready_rx, waiters))
            .init_resource::<ChunkOpsProvider>()
            .add_systems(PreUpdate, (sync_ops_provider, pump_ready).chain());
    }
}

/// The host declares its provider as a resource; the service holds it, so
/// a generation thread can resolve ops without touching the world.
fn sync_ops_provider(provider: Res<ChunkOpsProvider>, chunks: Res<ChunkGen>) {
    if provider.is_changed() {
        chunks.set_ops_provider(provider.0.clone());
    }
}

fn pump_ready(chunks: Res<ChunkGen>) {
    chunks.pump();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service() -> (
        ChunkGen,
        crossbeam_channel::Sender<(ChunkKey, u32)>,
        ChunkCommandQueue,
    ) {
        let (tx, rx) = crossbeam_channel::unbounded();
        let queue = ChunkCommandQueue::default();
        (
            ChunkGen::new(queue.clone(), rx, ChunkWaiters::default()),
            tx,
            queue,
        )
    }

    fn key() -> ChunkKey {
        ChunkKey::new(2, IVec3::new(3, 0, -1))
    }

    /// One drain, two consumers: a planner polling the batch and a create
    /// blocked on the single chunk it owns both learn from the same report.
    #[test]
    fn readiness_reaches_the_batch_and_the_waiter() {
        let (chunks, tx, _queue) = service();
        let waiting = chunks.wait_for(key());
        tx.send((key(), 0x5)).unwrap();
        chunks.pump();
        assert_eq!(chunks.ready_mask(key()), Some(0x5));
        assert_eq!(waiting.recv(), Ok(0x5));
        assert!(chunks.is_ready(key(), 0x5));
        assert!(!chunks.is_ready(key(), 0x6));
    }

    /// An empty-classified chunk has no seams to get wrong.
    #[test]
    fn an_empty_chunk_satisfies_any_mask() {
        let (chunks, tx, _queue) = service();
        tx.send((key(), u32::MAX)).unwrap();
        chunks.pump();
        assert!(chunks.is_ready(key(), 0x0));
        assert!(chunks.is_ready(key(), 0x3ff));
    }

    /// Requesting invalidates the previous report — otherwise a caller can
    /// commit against a mesh the new request is replacing.
    #[test]
    fn requesting_forgets_earlier_readiness() {
        let (chunks, tx, queue) = service();
        tx.send((key(), 0x5)).unwrap();
        chunks.pump();
        chunks.request(key(), 0x7, false, true, None);
        assert_eq!(chunks.ready_mask(key()), None);
        assert!(matches!(
            queue.take().as_slice(),
            [ChunkCommand::Request {
                key: k,
                show_on_ready: false,
                hold: true,
                face_mask: 0x7,
                ..
            }] if *k == key()
        ));
    }

    /// A chunk that will never arrive must not leave a create blocked.
    #[test]
    fn freeing_forgets_readiness_and_releases_the_waiter() {
        let (chunks, tx, queue) = service();
        tx.send((key(), 0x5)).unwrap();
        chunks.pump();
        let waiting = chunks.wait_for(key());
        chunks.free(key());
        assert_eq!(chunks.ready_mask(key()), None);
        assert!(waiting.recv().is_err(), "waiter left hanging on a freed chunk");
        assert!(matches!(queue.take().as_slice(), [ChunkCommand::Free(k)] if *k == key()));
    }

    /// A cancelled hold is the same promise broken a different way.
    #[test]
    fn cancelling_a_hold_forgets_readiness_and_releases_the_waiter() {
        let (chunks, tx, queue) = service();
        tx.send((key(), 0x5)).unwrap();
        chunks.pump();
        let waiting = chunks.wait_for(key());
        chunks.cancel_hold(key());
        assert_eq!(chunks.ready_mask(key()), None);
        assert!(waiting.recv().is_err());
        assert!(matches!(queue.take().as_slice(), [ChunkCommand::CancelHold(k)] if *k == key()));
    }

    /// Ops resolve through the installed provider; an empty result is
    /// `None`, not an empty buffer.
    #[test]
    fn ops_resolve_through_the_provider() {
        let (chunks, _tx, _queue) = service();
        assert!(chunks.ops_for(key()).is_none());
        chunks.set_ops_provider(Some(Arc::new(|k: ChunkKey| {
            if k.level == 2 {
                vec![CsgOp::boxy(Vec3::ZERO, Vec3::ONE, 0.0, 0, false)]
            } else {
                Vec::new()
            }
        })));
        assert_eq!(chunks.ops_for(key()).map(|o| o.len()), Some(1));
        assert!(chunks.ops_for(ChunkKey::new(3, IVec3::ZERO)).is_none());
    }
}
