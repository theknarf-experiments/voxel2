//! The layer and chunk traits.
//!
//! A layer is a singleton describing a grid; a chunk is a mutable,
//! reusable object that holds one cell's data and owns whatever resources
//! that data needs. The pairing, and the symmetry of
//! [`LayerChunk::create`] / [`LayerChunk::destroy`], is what lets a layer
//! own entities, GPU slots or pooled buffers — the framework guarantees
//! destroy runs exactly once for every create, in dependency order.

use glam::IVec3;

use crate::layer::{layer_key, IAabb, LayerKey};

/// A dependency on a level of another layer.
///
/// The `level` is the point of the whole mechanism: a layer exposes
/// partial states, and different consumers depend on different ones. That
/// is how a graph that would otherwise be circular — locations need paths,
/// paths need locations — is expressed as a DAG.
pub struct Dep {
    /// Instance key (hashed instance name) of the depended-on layer.
    pub key: LayerKey,
    /// Which level of that layer is required. Layers with one level always
    /// use 0.
    pub level: u32,
    /// How far outside its own chunk bounds (meters, per axis) the
    /// dependent may read. Declared, not inferred: a wider padding costs
    /// residency, so it should be a decision.
    pub padding: IVec3,
}

impl Dep {
    /// Depend on the final level of `L`'s default instance.
    pub fn of<L: Layer>(padding: IVec3) -> Self {
        Self::named(L::NAME, padding)
    }

    /// Depend on the final level of a named instance. Resolved to the
    /// instance's top level at registration, when the level count is known.
    pub fn named(name: &str, padding: IVec3) -> Self {
        Self {
            key: layer_key(name),
            level: FINAL_LEVEL,
            padding,
        }
    }

    /// Depend on a specific level of a named instance.
    pub fn named_at(name: &str, level: u32, padding: IVec3) -> Self {
        Self {
            key: layer_key(name),
            level,
            padding,
        }
    }
}

/// `Dep::level` sentinel meaning "whatever the instance's top level is",
/// resolved at registration.
pub const FINAL_LEVEL: u32 = u32::MAX;

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

    /// Chunk extent in meters per axis (0 = collapsed axis).
    fn chunk_extent(&self) -> IVec3;

    /// Generation levels within this layer. Each is a separate create /
    /// destroy pass over the same chunk, and other layers can depend on
    /// any of them.
    fn levels(&self) -> u32 {
        1
    }

    /// What `level` of this layer reads. Dependencies must already be
    /// registered, which makes the graph a DAG by construction.
    fn dependencies(&self, _level: u32) -> Vec<Dep> {
        Vec::new()
    }

    /// Padded reach of level `level`'s reads into level `level - 1` of
    /// this same layer.
    fn level_padding(&self, _level: u32) -> IVec3 {
        IVec3::ZERO
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

    /// Generate `level` of this chunk. Every dependency declared for this
    /// level is already resident, so reads through `ctx` cannot fail.
    fn create(&mut self, ctx: &ChunkCtx<'_, Self::Layer>, level: u32);

    /// Release `level` of this chunk: clear data, despawn entities, free
    /// GPU slots. Called exactly once per `create`, before the chunk's own
    /// providers are released.
    fn destroy(&mut self, _ctx: &ChunkCtx<'_, Self::Layer>, _level: u32) {}
}

pub use crate::v2::graph::ChunkCtx;

/// World-space bounds of a chunk, honoring collapsed axes.
pub(crate) fn bounds_of(extent: IVec3, coord: IVec3) -> IAabb {
    crate::layer::chunk_bounds(extent, coord)
}

/// Inclusive range of chunk coordinates covering `bounds`.
pub(crate) fn range_of(extent: IVec3, bounds: IAabb) -> (IVec3, IVec3) {
    crate::layer::chunk_range(extent, bounds)
}
