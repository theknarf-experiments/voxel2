//! LayerProcGen-style framework: data layers chunked at per-layer scales,
//! with declared padded dependencies on lower layers, generated recursively
//! on demand with strict input/output separation for determinism.
//!
//! The core contract (after runevision/LayerProcGen):
//! - A layer writes only its own chunk and reads only *lower* layers, within
//!   its own bounds inflated by the padding it declared for that dependency.
//! - All randomness derives from `(world_seed, layer_name, chunk_coord)`.
//! - Therefore any chunk can be generated at any time, on any thread, in any
//!   order, and the result is byte-identical.

pub mod layer;
pub mod manager;

pub use layer::{Dep, IAabb, Layer};
pub use manager::{LayerCtx, LayerManager, LayerView};
