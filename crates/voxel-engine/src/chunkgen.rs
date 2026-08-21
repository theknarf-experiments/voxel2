//! The chunk generation service: the one owner of request / ready / free.
//!
//! Everything that drives a voxel chunk's lifecycle goes through here —
//! resolving its planning ops, asking the render world to generate it,
//! learning that it became drawable, committing it, freeing it. Nothing
//! else may: two owners of one readiness channel, or two opinions about
//! when a slab is released, is not a configuration that has a meaning.
//!
//! The service is a cloneable handle rather than a system param because
//! its caller is a layer's `create`, which runs on a generation thread:
//! it requests a chunk and blocks on [`ChunkGen::wait_for`] until the
//! chunk exists.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use voxel_core::csg::CsgOp;
use voxel_core::ChunkKey;
use voxel_render::{ChunkCommand, ChunkCommandQueue, ChunkReadyChannel, ChunkWaiters};

/// Planning-layer CSG ops for one chunk, already AABB-culled to it.
pub type OpsFn = Arc<dyn Fn(ChunkKey) -> Vec<CsgOp> + Send + Sync>;

/// Handle to the chunk generation service. See the module docs.
#[derive(Resource, Clone)]
pub struct ChunkGen(Arc<Service>);

struct Service {
    queue: ChunkCommandQueue,
    ready_rx: crossbeam_channel::Receiver<(ChunkKey, u32)>,
    waiters: ChunkWaiters,
    /// One provider per world, indexed by `ChunkKey::world`. A world with
    /// nothing to plan has `None`; it does NOT fall back to world 0's,
    /// which would answer about coordinates in a world it knows nothing
    /// about (worlds share coordinates, so it always answers something).
    ops: Mutex<Vec<Option<OpsFn>>>,
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
            ops: Mutex::new(Vec::new()),
        }))
    }

    // --- ops ---------------------------------------------------------------

    /// Install one provider per world, indexed by world id.
    pub fn set_ops_providers(&self, ops: Vec<Option<OpsFn>>) {
        *self.0.ops.lock().unwrap() = ops;
    }

    /// Resolve `key`'s ops now, from ITS world's provider. Empty is
    /// `None`: the density pass binds a dummy op buffer rather than an
    /// empty one.
    pub fn ops_for(&self, key: ChunkKey) -> Option<Arc<Vec<CsgOp>>> {
        // Clone the provider OUT of the lock before calling it. Calling
        // it inside serialized every LOD chunk create in the world on one
        // mutex: the provider is a spatial query over the planning graph,
        // not a lookup, and every worker thread wanted it for every chunk.
        // The `Arc` is here precisely so this is a refcount bump.
        let provider = self
            .0
            .ops
            .lock()
            .unwrap()
            .get(usize::from(key.world))
            .and_then(Option::as_ref)
            .cloned();
        provider
            .map(|f| f(key))
            .filter(|v| !v.is_empty())
            .map(Arc::new)
    }

    // --- readiness ---------------------------------------------------------

    /// Drain the render world's readiness reports and wake whoever is
    /// waiting on each chunk. Exactly one drain exists, and this is it.
    pub fn pump(&self) {
        for (key, mask) in self.0.ready_rx.try_iter() {
            self.0.waiters.notify(key, mask);
        }
    }

    /// A receiver that fires when `key` next becomes drawable, carrying the
    /// seam mask of the mesh. Disconnects if the chunk is freed, so a
    /// blocked caller cannot wait forever.
    fn wait_for(&self, key: ChunkKey) -> crossbeam_channel::Receiver<u32> {
        self.0.waiters.wait_for(key)
    }

    // --- lifecycle ---------------------------------------------------------

    /// Generate `key` with `face_mask`. `show_on_ready` draws it the moment
    /// it is drawable; otherwise it waits for [`Self::commit`]. `hold`
    /// marks an in-place remesh of an already-shown chunk: the old mesh
    /// keeps drawing until the commit swaps them.
    fn request(
        &self,
        key: ChunkKey,
        face_mask: u32,
        show_on_ready: bool,
        hold: bool,
        ops: Option<Arc<voxel_core::csg::ChunkOps>>,
    ) {
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

    /// Release the chunk and its slab allocation. Anything waiting on it
    /// is released too — a chunk that will never arrive must not leave a
    /// `create` blocked.
    pub fn free(&self, key: ChunkKey) {
        self.0.waiters.abandon(key);
        self.0.queue.push(ChunkCommand::Free(key));
    }
}

/// Chunks asked for together, so they can be revealed together.
///
/// Half a swapped set is a crack, so a caller that changes several chunks
/// at once has to wait for all of them before showing any. Batching that
/// here also makes the one ordering rule impossible to get wrong: the
/// wait is registered BEFORE the request, or a chunk that meshes in
/// between reports to nobody and the wait times out.
/// Owned rather than borrowing the service, so a batch can outlive the
/// call that started it: the LOD pass registers builds from every level
/// and waits for the lot ONCE, instead of each level waiting out its own
/// GPU round trip while the pipeline drains.
#[derive(Default)]
pub struct ChunkBatch {
    waits: Vec<(ChunkKey, crossbeam_channel::Receiver<u32>)>,
}

/// What a batch managed to build. A chunk that stalled has NOT been
/// built, and a caller must not record it as though it had — the mask it
/// would have carried is not the mask on screen.
pub struct Built {
    pub built: Vec<ChunkKey>,
    pub stalled: Vec<ChunkKey>,
}

impl ChunkBatch {
    /// Ask for one chunk, hidden until it is committed. `hold` keeps the
    /// old mesh drawing meanwhile, for a chunk that is already shown.
    ///
    /// Ops are passed in rather than resolved here: a chunk being rebuilt
    /// for its seam alone is the same chunk at the same coordinate, so
    /// re-querying planning for it would be a planning-graph read per
    /// chunk for an answer the caller already holds.
    pub fn add(
        &mut self,
        chunks: &ChunkGen,
        key: ChunkKey,
        face_mask: u32,
        hold: bool,
        ops: Option<Arc<voxel_core::csg::ChunkOps>>,
    ) {
        let ready = chunks.wait_for(key);
        chunks.request(key, face_mask, false, hold, ops);
        self.waits.push((key, ready));
    }

    pub fn is_empty(&self) -> bool {
        self.waits.is_empty()
    }

    /// Block until every chunk has reported, or `timeout` elapses for the
    /// BATCH.
    ///
    /// A deadline, not a timeout per chunk: the waits are walked in order,
    /// so a per-chunk timeout would let a wedged pipeline hold the caller
    /// for `timeout` times the number of chunks — with a whole pass in one
    /// batch, hours.
    ///
    /// Running out means the pipeline could not place the chunk — slab
    /// exhaustion — and the caller decides what a hole is worth; blocking
    /// forever instead would wedge the generation thread and present as a
    /// frozen world.
    pub fn wait(&mut self, timeout: std::time::Duration) -> Built {
        let deadline = std::time::Instant::now() + timeout;
        let mut out = Built {
            built: Vec::with_capacity(self.waits.len()),
            stalled: Vec::new(),
        };
        for (key, ready) in self.waits.drain(..) {
            if ready.recv_deadline(deadline).is_ok() {
                out.built.push(key);
            } else {
                out.stalled.push(key);
            }
        }
        out
    }
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
            .add_systems(PreUpdate, pump_ready);
    }
}

fn pump_ready(chunks: Res<ChunkGen>) {
    chunks.pump();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

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

    /// A batch registers its wait before requesting, so a chunk that
    /// meshes in between still reports to the caller.
    #[test]
    fn a_batch_hears_a_report_that_lands_immediately() {
        let (chunks, tx, queue) = service();
        let mut batch = ChunkBatch::default();
        batch.add(&chunks, key(), 0x7, false, None);
        assert!(matches!(
            queue.take().as_slice(),
            [ChunkCommand::Request {
                key: k,
                show_on_ready: false,
                hold: false,
                face_mask: 0x7,
                ..
            }] if *k == key()
        ));
        tx.send((key(), 0x7)).unwrap();
        chunks.pump();
        let out = batch.wait(Duration::from_secs(1));
        assert_eq!(out.built, vec![key()]);
        assert!(out.stalled.is_empty());
    }

    /// A chunk the pipeline cannot place is reported as stalled rather
    /// than waited on forever — and NOT as built, because a caller that
    /// records it as built records a mask that is not on screen.
    #[test]
    fn a_chunk_that_never_arrives_is_stalled_not_built() {
        let (chunks, _tx, _queue) = service();
        let mut batch = ChunkBatch::default();
        batch.add(&chunks, key(), 0x7, false, None);
        let out = batch.wait(Duration::from_millis(20));
        assert!(out.built.is_empty());
        assert_eq!(out.stalled, vec![key()]);
    }

    /// A batch waits for every member: half a swapped set is a crack.
    #[test]
    fn a_batch_waits_for_all_of_its_members() {
        let (chunks, tx, _queue) = service();
        let other = ChunkKey::new(2, IVec3::new(4, 0, -1));
        let mut batch = ChunkBatch::default();
        batch.add(&chunks, key(), 0x1, true, None);
        batch.add(&chunks, other, 0x2, true, None);
        tx.send((key(), 0x1)).unwrap();
        chunks.pump();
        let out = batch.wait(Duration::from_millis(20));
        assert_eq!(out.built, vec![key()]);
        assert_eq!(out.stalled, vec![other]);
    }

    /// A chunk that will never arrive must not leave a caller blocked.
    #[test]
    fn freeing_releases_the_waiter() {
        let (chunks, _tx, queue) = service();
        let mut batch = ChunkBatch::default();
        batch.add(&chunks, key(), 0x1, false, None);
        let _ = queue.take();
        chunks.free(key());
        // The waiter is disconnected, so the wait ends at once rather
        // than burning the timeout.
        let out = batch.wait(Duration::from_secs(30));
        assert_eq!(out.stalled, vec![key()]);
        assert!(matches!(queue.take().as_slice(), [ChunkCommand::Free(k)] if *k == key()));
    }

    /// Ops resolve through the installed provider; an empty result is
    /// `None`, not an empty buffer.
    #[test]
    fn ops_resolve_through_the_provider() {
        let (chunks, _tx, _queue) = service();
        assert!(chunks.ops_for(key()).is_none());
        chunks.set_ops_providers(vec![Some(Arc::new(|k: ChunkKey| {
            if k.level == 2 {
                vec![CsgOp::boxy(Vec3::ZERO, Vec3::ONE, 0.0, 0, false)]
            } else {
                Vec::new()
            }
        }))]);
        assert_eq!(chunks.ops_for(key()).map(|o| o.len()), Some(1));
        assert!(chunks.ops_for(ChunkKey::new(3, IVec3::ZERO)).is_none());
    }

    /// A provider serves ITS world and no other. Worlds share
    /// coordinates, so a provider reached by anything but the key's world
    /// answers confidently about a place it has never heard of.
    #[test]
    fn a_provider_serves_only_its_own_world() {
        let (chunks, _tx, _queue) = service();
        let op = |material: u32| {
            let ops = vec![CsgOp::boxy(Vec3::ZERO, Vec3::ONE, 0.0, material, false)];
            Some(Arc::new(move |_: ChunkKey| ops.clone()) as OpsFn)
        };
        // World 1 has no planning at all; world 2 has its own.
        chunks.set_ops_providers(vec![op(7), None, op(9)]);
        let at = |world| ChunkKey::in_world(world, 2, IVec3::ZERO);
        assert_eq!(chunks.ops_for(at(0)).unwrap()[0].material, 7);
        assert!(
            chunks.ops_for(at(1)).is_none(),
            "world 1 plans nothing and must not inherit world 0's planner"
        );
        assert_eq!(chunks.ops_for(at(2)).unwrap()[0].material, 9);
        // A world past the end is unregistered, not world 0.
        assert!(chunks.ops_for(at(3)).is_none());
    }
}
