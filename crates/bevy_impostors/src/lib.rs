//! Instanced point-scatter props, and the impostor that is their reason
//! to exist.
//!
//! A prop population is a small static mesh drawn once per published
//! point — a tuft, a pebble, a crossed silhouette — with one uniform of
//! look parameters at group 2 and Bevy's PBR view groups at 0 and 1, so
//! every instance is lit, fogged and tonemapped like the rest of the
//! frame. An instance is 16 bytes, which is what lets a population run to
//! millions: the bound is how many points the host cares to publish, not
//! the renderer.
//!
//! [`Prop`] is one population: its mesh, its shader, its uniform, all in
//! the implementing type — which IS the marker component a host spawns
//! one of per instance SET (a world, a map, a layer; the crate does not
//! care what a set is). [`PropPlugin`] is everything else: buffers,
//! pipeline, bind group, extract, prepare, queue and draw. Points arrive
//! through [`PropPoints`], which the host fills from wherever its
//! placements come from.
//!
//! [`Impostors`] is the shipped population: a crossed silhouette per
//! point, shaped and shaded per instance from a 32-bit hash. See
//! [`impostor`] for the contract.

pub mod impostor;
pub mod prop;

pub use impostor::{ImpostorStyle, Impostors, IMPOSTOR_FADE_FROM};
pub use prop::{Prop, PropFlags, PropInstance, PropPlugin, PropPoints};
