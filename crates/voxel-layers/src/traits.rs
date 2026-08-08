//! The layer and chunk traits.
//!
//! A layer is a singleton describing a grid; a chunk is a mutable,
//! reusable object that holds one cell's data and owns whatever resources
//! that data needs. The pairing, and the symmetry of
//! [`LayerChunk::create`] / [`LayerChunk::destroy`], is what lets a layer
//! own entities, GPU slots or pooled buffers — the framework guarantees
//! destroy runs exactly once for every create, in dependency order.

use glam::{DVec3, IVec3};

use crate::layer::{layer_key, LayerKey};

/// A dependency on another layer instance.
///
/// A layer that wants to expose partial states publishes each as its own
/// INSTANCE, and different consumers depend on different ones — which is
/// how a graph that would otherwise be circular (locations need paths,
/// paths need locations) is expressed as a DAG. There used to be a second
/// mechanism for this, internal levels within one layer; see
/// [`crate`] docs for why it is gone.
pub struct Dep {
    /// Instance key (hashed instance name) of the depended-on layer.
    pub key: LayerKey,
    /// How far outside its own chunk bounds (meters, per axis) the
    /// dependent may read. Declared, not inferred: a wider padding costs
    /// residency, so it should be a decision.
    pub padding: IVec3,
}

impl Dep {
    /// Depend on `L`'s default instance.
    pub fn of<L: Layer>(padding: IVec3) -> Self {
        Self::named(L::NAME, padding)
    }

    /// Depend on a named instance.
    pub fn named(name: &str, padding: IVec3) -> Self {
        Self {
            key: layer_key(name),
            padding,
        }
    }
}

/// One data layer of the procedural world.
///
/// Chunks of different layers can differ in scale by orders of magnitude;
/// an axis with `chunk_extent` 0 is *collapsed* — the layer has a single
/// chunk along it (coordinate 0, unbounded extent), which is how planar 2D
/// layers are expressed in a 3D world.
pub trait Layer: Send + Sync + 'static {
    type Chunk: LayerChunk<Layer = Self>;

    /// Stable identifier; hashed into every chunk seed of this layer.
    /// Renaming reshuffles randomness — treat like a save-format change.
    const NAME: &'static str;

    /// Chunk extent in meters per axis (0 = collapsed axis). Fractional,
    /// because a voxel LOD level's chunk edge is `3.2 · 2^level` and a
    /// layer that owns one voxel chunk per cell has to sit on that grid.
    fn chunk_extent(&self) -> DVec3;

    /// What this layer reads. Dependencies must already be registered,
    /// which makes the graph a DAG by construction.
    fn dependencies(&self) -> Vec<Dep> {
        Vec::new()
    }
}

/// One cell of a layer.
///
/// Chunks are pooled and reused, so `create` must not assume it starts
/// from a fresh value and `destroy` must leave the chunk reusable —
/// clearing collections rather than dropping them is the point of the
/// pairing.
pub trait LayerChunk: Default + Send + Sync + 'static {
    type Layer: Layer<Chunk = Self>;

    /// Generate this chunk. Every declared dependency is already
    /// resident, so reads through `ctx` cannot fail.
    fn create(&mut self, ctx: &ChunkCtx<'_, Self::Layer>);

    /// Release this chunk: clear data, despawn entities, free GPU slots.
    /// Called exactly once per `create`, before the chunk's own providers
    /// are released.
    fn destroy(&mut self, _ctx: &ChunkCtx<'_, Self::Layer>) {}
}

pub use crate::graph::ChunkCtx;

