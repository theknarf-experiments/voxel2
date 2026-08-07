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
            ops: Mutex::new(None),
        }))
    }

    // --- ops ---------------------------------------------------------------

    pub fn set_ops_provider(&self, ops: Option<OpsFn>) {
        *self.0.ops.lock().unwrap() = ops;
    }

    fn ops_fn(&self) -> Option<OpsFn> {
        self.0.ops.lock().unwrap().clone()
    }

    /// Resolve `key`'s ops now. Empty is `None`: the density pass binds a
    /// dummy op buffer rather than an empty one.
    pub fn ops_for(&self, key: ChunkKey) -> Option<Arc<Vec<CsgOp>>> {
        resolve_ops(self.ops_fn().as_deref(), key)
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
    /// A wait may only be satisfied by a report that ARRIVES after the
    /// request, which is why [`Self::wait_for`] is registered first and
    /// readiness is a notification rather than a state anyone polls.
    pub fn request(
        &self,
        key: ChunkKey,
        face_mask: u32,
        show_on_ready: bool,
        hold: bool,
        ops: Option<Arc<Vec<CsgOp>>>,
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
        self.forget(key);
        self.0.queue.push(ChunkCommand::Free(key));
    }

    fn forget(&self, key: ChunkKey) {
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

    /// The drain wakes whoever is waiting on that chunk, carrying the
    /// seam mask its mesh was built with. This is the whole readiness
    /// surface: a `create` registers, requests, and blocks.
    #[test]
    fn readiness_reaches_the_waiter() {
        let (chunks, tx, _queue) = service();
        let waiting = chunks.wait_for(key());
        tx.send((key(), 0x5)).unwrap();
        chunks.pump();
        assert_eq!(waiting.recv(), Ok(0x5));
    }

    /// A chunk that will never arrive must not leave a create blocked.
    #[test]
    fn freeing_releases_the_waiter() {
        let (chunks, _tx, queue) = service();
        let waiting = chunks.wait_for(key());
        chunks.free(key());
        assert!(waiting.recv().is_err(), "waiter left hanging on a freed chunk");
        assert!(matches!(queue.take().as_slice(), [ChunkCommand::Free(k)] if *k == key()));
    }

    /// A request carries what the pipeline needs to build the chunk, and
    /// nothing decides visibility but the caller.
    #[test]
    fn a_request_carries_its_mask_and_visibility() {
        let (chunks, _tx, queue) = service();
        chunks.request(key(), 0x7, false, true, None);
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
