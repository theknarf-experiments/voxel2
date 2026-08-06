//! LayerProcGen: dependency management between data layers, threaded chunk
//! generation, and spatial organisation — and no procedural generation
//! algorithms whatsoever. Concrete layers are the game's code.
//!
//! - **Lifetime is reference-counted through the dependency graph.** Every
//!   chunk level records what it was generated from and how many things
//!   need it. The resident set is exactly the transitive closure of the
//!   active [`TopDep`]s, not an approximation kept by an eviction pass.
//! - **Top dependencies are the only thing that generates**, and they
//!   process ensure-new-then-release-old, so a moving focus never drops
//!   what a consumer still holds. Reads return what is resident; a miss
//!   reports the padding that would have covered it.
//! - **`create` and `destroy` are symmetric**, so a chunk can own entities,
//!   GPU slots or pooled buffers and give them back at a defined moment.
//! - **Dependencies name a level**, so a layer publishes partial states and
//!   a graph that would otherwise be circular stays a DAG.
//!
//! Determinism rests on one rule: a layer writes only its own chunk and
//! reads only what it declared, within its own bounds inflated by that
//! dependency's padding, deriving all randomness from
//! `(world_seed, instance, chunk_coord)`. Any chunk can then be generated
//! at any time, on any thread, in any order, byte-identically.

pub mod graph;
pub mod layer;
pub mod runtime;
pub mod store;
pub mod traits;

pub use graph::{dep_bounds, ChunkCtx, ChunkRef, LayerGraph, TopDep, View};
pub use layer::{layer_key, IAabb, LayerKey};
pub use runtime::{LayerRuntime, TopHandle};
pub use store::Usage;
pub use traits::{Dep, Layer, LayerChunk, FINAL_LEVEL};
