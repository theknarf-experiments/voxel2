//! LayerProcGen, as it is actually built.
//!
//! The framework provides dependency management between layers, threaded
//! chunk generation and spatial organisation — and no procedural
//! generation algorithms whatsoever. Concrete layers are the game's code.
//!
//! What distinguishes this from a cache keyed by chunk coordinate, which
//! is what [`crate::manager`] is:
//!
//! - **Lifetime is reference-counted through the dependency graph.** Every
//!   chunk level records what it was generated from and how many things
//!   need it. The resident set is exactly the transitive closure of the
//!   active [`TopDep`]s — not an approximation maintained by an eviction
//!   heuristic.
//! - **Top dependencies are the only thing that generates.** Reads return
//!   what is resident. A read that misses is a bug in someone's declared
//!   padding or in the top dependency's size, and it says so.
//! - **`create` and `destroy` are symmetric**, so a chunk can own
//!   entities, GPU slots or pooled buffers and give them back at a defined
//!   moment.
//! - **Dependencies name a level**, so a layer can publish partial states
//!   and a graph that would otherwise be circular stays a DAG.
//!
//! This module will become the crate root once the old manager's callers
//! are ported; it is separate only so the tree keeps compiling meanwhile.

mod graph;
mod layer;
mod runtime;
mod store;

pub use graph::{ChunkCtx, ChunkRef, LayerGraph, TopDep, View};
pub use runtime::{LayerRuntime, TopHandle};
pub use layer::{Dep, Layer, LayerChunk, FINAL_LEVEL};
pub use store::Usage;

pub use crate::layer::{layer_key, IAabb, LayerKey};
