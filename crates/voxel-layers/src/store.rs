//! Chunk slots: identity, lifetime bookkeeping, and the data.
//!
//! Every chunk records the chunks it was generated from (its *providers*)
//! and how many things currently depend on it (its *user count*). Those
//! two facts are the whole lifetime model: a chunk lives exactly as long
//! as something needs it, and releasing it releases what it needed,
//! recursively.

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use glam::IVec3;

use crate::layer::LayerKey;

pub(crate) type ErasedChunk = Box<dyn Any + Send + Sync>;

/// A chunk that some other chunk (or a top dependency) was generated from
/// and therefore keeps alive.
pub(crate) type Provider = Arc<ChunkSlot>;

/// What one create — or one top dependency's whole closure — touched.
///
/// Holding a `Usage` is what keeps chunks resident; dropping it through
/// [`super::graph::LayerGraph::release`] is what lets them die. It is
/// deliberately not `Drop`: releasing needs the graph, and a silent
/// release on unwind would hide a bug rather than fix one.
#[derive(Default)]
pub struct Usage {
    pub(crate) providers: Vec<Provider>,
}

impl Usage {
    pub(crate) fn from_providers(providers: Vec<Provider>) -> Self {
        Self { providers }
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

#[derive(Default)]
pub(crate) struct SlotState {
    /// Chunks this one was generated from.
    pub providers: Vec<Provider>,
    /// How many things depend on this chunk.
    pub users: i32,
}

pub(crate) struct ChunkSlot {
    pub layer: LayerKey,
    pub coord: IVec3,
    /// Has `create` finished? Written only under `lock`.
    pub generated: AtomicBool,
    /// Held across create or destroy so two workers can never both
    /// generate this chunk. Locks are taken in dependency order (a
    /// chunk's lock, then its providers'), and the graph is a DAG, so
    /// they cannot cycle.
    pub lock: Mutex<()>,
    /// Providers and user count.
    pub state: Mutex<SlotState>,
    /// The chunk. Write-locked across create and destroy, read-locked by
    /// readers — who, by the residency rules, only ever read chunks that
    /// something is holding, and so are never racing a destroy.
    pub data: RwLock<ErasedChunk>,
}

impl ChunkSlot {
    pub fn new(layer: LayerKey, coord: IVec3, data: ErasedChunk) -> Self {
        Self {
            layer,
            coord,
            generated: AtomicBool::new(false),
            lock: Mutex::new(()),
            state: Mutex::new(SlotState::default()),
            data: RwLock::new(data),
        }
    }

    pub fn is_generated(&self) -> bool {
        self.generated.load(Ordering::Acquire)
    }

    pub fn add_user(&self) {
        self.state.lock().unwrap().users += 1;
    }

    /// Drop one user. Returns the providers to release when that was the
    /// last one — the caller performs the cascade, so this never recurses
    /// while holding a lock.
    pub fn drop_user(&self) -> Option<Vec<Provider>> {
        let mut state = self.state.lock().unwrap();
        state.users -= 1;
        debug_assert!(state.users >= 0, "user count went negative");
        if state.users > 0 {
            return None;
        }
        Some(std::mem::take(&mut state.providers))
    }
}
