//! Dense point scatter: the engine's placement output for things a host
//! draws in bulk (ground cover, pebbles, sparks).
//!
//! The engine decides WHERE the points are; what they look like — blade
//! geometry, colors, wind, fade — is the host's, so the pipeline that
//! draws them lives in the app (see the demo's `grass.rs`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

/// One scattered point: a world position plus a per-point hash the host
/// can use for variation (tint, phase, size).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ScatterPoint {
    pub pos: [f32; 3],
    pub hash: u32,
}

/// Main-world resource: the current point set plus a dirty flag.
/// Interior mutability so extraction (read-only) can clear the flag.
#[derive(Resource, Default)]
pub struct ScatterPoints {
    inner: Mutex<ScatterShared>,
}

#[derive(Default)]
struct ScatterShared {
    /// Keyed by the population's class name, which the level chose. The
    /// engine never interprets it; the host draws the classes it knows.
    classes: HashMap<Arc<str>, Vec<ScatterPoint>>,
    dirty: bool,
}

impl ScatterPoints {
    /// Replace one class's points (the streamer rebuilds wholesale).
    pub fn set_class(&self, class: &str, points: Vec<ScatterPoint>) {
        let mut inner = self.inner.lock().unwrap();
        inner.classes.insert(Arc::from(class), points);
        inner.dirty = true;
    }

    /// Drop every class.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.classes.clear();
        inner.dirty = true;
    }

    /// Take one class's points if anything changed since the last call.
    /// The host asks for the classes it knows how to draw.
    pub fn take_class_if_dirty(&self, class: &str) -> Option<Vec<ScatterPoint>> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.dirty {
            return None;
        }
        inner.dirty = false;
        Some(inner.classes.get(class).cloned().unwrap_or_default())
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().classes.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
