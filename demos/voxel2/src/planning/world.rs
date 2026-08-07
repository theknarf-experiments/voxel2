//! What this game's layers share.
//!
//! The graph carries one opaque per-world value, handed to every chunk's
//! create. For planning layers that was just the generator; presentation
//! layers also need somewhere to put what they produce, because a chunk
//! that owns a resource has to hand it back in `destroy` and the framework
//! is deliberately ignorant of what a resource is.
//!
//! A sink is that somewhere: a chunk publishes its contribution under its
//! own coordinate on create and withdraws it on destroy, and a Bevy system
//! rebuilds whatever buffer draws it whenever the set changes. Residency
//! decides what exists — there is no radius, no eviction scan and no
//! "is it still near the camera" test anywhere in this file.
//!
//! A contribution is keyed by INSTANCE and coordinate, not coordinate
//! alone. Two instances of one layer at different scales — the same
//! ribbons tiled at 256 m near and 4096 m far — both have a chunk (0,0,0),
//! and keyed by coordinate the coarse one silently erases the fine one.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use glam::IVec3;

use crate::water::RiverSegGpu;

/// Per-world state every layer in this game can reach.
pub struct WorldCtx {
    pub generator: Arc<voxel_worldgen::Generator>,
    /// Ribbon surface geometry, contributed per chunk.
    pub ribbons: Sink<RiverSegGpu>,
    /// Scatter population handles, taken once by the app.
    pub populations: Mutex<Option<crate::scatter::Populations>>,
    /// Merged far-forest impostors, contributed per super-tile.
    pub far_props: Sink<crate::props::FarProp>,
}

impl WorldCtx {
    pub fn new(generator: Arc<voxel_worldgen::Generator>) -> Self {
        Self {
            generator,
            ribbons: Sink::default(),
            populations: Mutex::new(None),
            far_props: Sink::default(),
        }
    }
}

/// Per-chunk contributions to one shared buffer.
///
/// Cheap to clone into a chunk's create; the generation counter is what a
/// renderer watches, so it rebuilds only when the resident set actually
/// changed rather than every frame.
pub struct Sink<T> {
    inner: Arc<Mutex<SinkInner<T>>>,
}

/// Instance key and chunk coordinate: what identifies a contribution.
pub type PartKey = (u64, IVec3);

struct SinkInner<T> {
    parts: HashMap<PartKey, Vec<T>>,
    generation: u64,
}

impl<T> Default for Sink<T> {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SinkInner {
                parts: HashMap::new(),
                generation: 0,
            })),
        }
    }
}

impl<T: Clone> Sink<T> {
    /// Publish a chunk's contribution. Empty contributions are not stored:
    /// most tiles of a sparse feature have nothing in them, and an empty
    /// entry would still cost a rebuild.
    pub fn put(&self, instance: u64, coord: IVec3, items: Vec<T>) {
        let mut inner = self.inner.lock().unwrap();
        if items.is_empty() {
            if inner.parts.remove(&(instance, coord)).is_some() {
                inner.generation += 1;
            }
            return;
        }
        inner.parts.insert((instance, coord), items);
        inner.generation += 1;
    }

    /// Withdraw a chunk's contribution, from its `destroy`.
    pub fn take(&self, instance: u64, coord: IVec3) {
        let mut inner = self.inner.lock().unwrap();
        if inner.parts.remove(&(instance, coord)).is_some() {
            inner.generation += 1;
        }
    }

    pub fn generation(&self) -> u64 {
        self.inner.lock().unwrap().generation
    }

    /// Contributions present — what an entity population diffs its
    /// spawned set against.
    pub fn keys(&self) -> std::collections::HashSet<PartKey> {
        self.inner.lock().unwrap().parts.keys().copied().collect()
    }

    /// One chunk's contribution.
    pub fn get(&self, key: PartKey) -> Option<Vec<T>> {
        self.inner.lock().unwrap().parts.get(&key).cloned()
    }

    /// Everything currently published, flattened.
    pub fn collect(&self) -> Vec<T> {
        let inner = self.inner.lock().unwrap();
        inner.parts.values().flatten().cloned().collect()
    }
}

impl<T> Clone for Sink<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
