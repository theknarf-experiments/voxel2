//! Chunk slots: identity, per-level lifetime bookkeeping, and the data.
//!
//! Every chunk level records the chunk levels it was generated from (its
//! *providers*) and how many things currently depend on it (its *user
//! count*). Those two facts are the whole lifetime model: a level lives
//! exactly as long as something needs it, and releasing it releases what
//! it needed, recursively.

use std::any::Any;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use glam::IVec3;

use crate::layer::LayerKey;

pub(crate) type ErasedChunk = Box<dyn Any + Send + Sync>;

/// A chunk level that some other chunk level (or a top dependency) was
/// generated from and therefore keeps alive.
pub(crate) type Provider = (Arc<ChunkSlot>, u32);

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
pub(crate) struct LevelState {
    /// Chunk levels this level was generated from.
    pub providers: Vec<Provider>,
    /// How many things depend on this level.
    pub users: i32,
}

pub(crate) struct ChunkSlot {
    pub layer: LayerKey,
    pub coord: IVec3,
    /// Highest level generated so far; -1 = nothing yet. Written only
    /// under the corresponding level lock.
    pub level: AtomicI32,
    /// One lock per level, held across that level's create or destroy so
    /// two workers can never both generate it. Locks are taken in
    /// dependency order (a chunk's lock, then its providers'), and the
    /// graph is a DAG, so they cannot cycle.
    pub level_locks: Vec<Mutex<()>>,
    /// Providers and user count per level.
    pub levels: Mutex<Vec<LevelState>>,
    /// The chunk. Write-locked across create and destroy, read-locked by
    /// readers — who, by the residency rules, only ever read levels that
    /// something is holding, and so are never racing a destroy.
    pub data: RwLock<ErasedChunk>,
}

impl ChunkSlot {
    pub fn new(layer: LayerKey, coord: IVec3, levels: usize, data: ErasedChunk) -> Self {
        Self {
            layer,
            coord,
            level: AtomicI32::new(-1),
            level_locks: (0..levels).map(|_| Mutex::new(())).collect(),
            levels: Mutex::new((0..levels).map(|_| LevelState::default()).collect()),
            data: RwLock::new(data),
        }
    }

    /// Has this chunk been generated to at least `level`?
    pub fn has_level(&self, level: u32) -> bool {
        self.level.load(Ordering::Acquire) >= level as i32
    }

    pub fn add_user(&self, level: u32) {
        self.levels.lock().unwrap()[level as usize].users += 1;
    }

    /// Drop one user of `level`. Returns the providers to release when
    /// that was the last one — the caller performs the cascade, so this
    /// never recurses while holding a lock.
    pub fn drop_user(&self, level: u32) -> Option<Vec<Provider>> {
        let mut levels = self.levels.lock().unwrap();
        let state = &mut levels[level as usize];
        state.users -= 1;
        debug_assert!(state.users >= 0, "user count went negative");
        if state.users > 0 {
            return None;
        }
        Some(std::mem::take(&mut state.providers))
    }
}
