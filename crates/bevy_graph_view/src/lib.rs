//! A node-graph view for Bevy UI.
//!
//! Two halves, split where testability splits. [`layout`] is pure
//! arithmetic: it takes a list of [`GraphNode`]s — an opaque `id`, a
//! `name` other nodes wire to, ports, and children for scopes — and
//! returns every box, frame and wire segment as plain geometry, testable
//! at a terminal. [`canvas::scene`] turns that geometry into a `bsn!`
//! scene: absolutely-positioned boxes under one pannable, zoomable
//! [`GraphCanvas`], themed with Feathers tokens.
//!
//! This crate has no opinion about what a node IS or where the graph came
//! from. The host describes its document as [`GraphNode`]s, spawns the
//! scene, and hears back through [`SelectsNode`] when a box is clicked —
//! what selection MEANS is the host's business. Likewise the camera:
//! [`GraphCamera`] owns the zoom-about-a-point arithmetic, but the host
//! stores it wherever survives its own respawns, and applies it to the
//! canvas's `UiTransform` however it schedules such things.
//!
//! Two systems ship here — [`hover`], and [`zoom_label`] for the readout
//! in the viewport's lower corner — because they touch nothing but this
//! crate's own components. They are plain systems, not registered by
//! [`GraphViewPlugin`], so a host that gates its UI work can put them
//! under its own run condition.

pub mod canvas;
pub mod layout;

pub use canvas::{
    hover, scene, zoom_label, GraphCamera, GraphCanvas, GraphViewport, SelectsNode, ZoomLabel,
    PLAIN_BORDER,
};
pub use layout::{layout, Edge, Frame, GraphNode, GraphStyle, Layout, Placed, Seg};

use bevy::prelude::*;

/// Registers [`GraphStyle`] as a reflected resource, so tooling can restyle
/// a running graph, and [`GraphCamera`] for hosts that reflect their own
/// state. Nothing is scheduled: drawing and interaction stay in the host's
/// hands — see [`hover`].
#[derive(Default)]
pub struct GraphViewPlugin;

impl Plugin for GraphViewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GraphStyle>()
            .register_type::<GraphStyle>()
            .register_type::<GraphCamera>();
    }
}
