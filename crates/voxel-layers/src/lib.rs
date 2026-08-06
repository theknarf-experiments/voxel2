//! Data layers chunked at per-layer scales, with declared padded
//! dependencies on lower layers.
//!
//! [`v2`] is the framework, after runevision/LayerProcGen: lifetime
//! reference-counted through the dependency graph, top dependencies as the
//! only thing that generates, symmetric create/destroy, dependencies that
//! name a level. [`manager`] is the on-demand cache it replaces, kept only
//! until its callers are ported.
//!
//! The contract both share, and that determinism rests on:
//! - A layer writes only its own chunk and reads only *lower* layers, within
//!   its own bounds inflated by the padding it declared for that dependency.
//! - All randomness derives from `(world_seed, layer_name, chunk_coord)`.
//! - Therefore any chunk can be generated at any time, on any thread, in any
//!   order, and the result is byte-identical.

pub mod layer;
pub mod manager;
pub mod v2;

pub use layer::{Dep, IAabb, Layer};
pub use manager::{LayerCtx, LayerManager, LayerView};
