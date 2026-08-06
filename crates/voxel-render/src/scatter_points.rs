//! Dense point scatter: the engine's placement output for things a host
//! draws in bulk (ground cover, pebbles, sparks).
//!
//! The engine decides WHERE the points are; what they look like — blade
//! geometry, colors, wind, fade — is the host's, so the pipeline that
//! draws them lives in the app (see the demo's `grass.rs`).

use std::sync::Mutex;

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
    points: Vec<ScatterPoint>,
    dirty: bool,
}

impl ScatterPoints {
    /// Replace the whole set (the streamer rebuilds it wholesale).
    pub fn set(&self, points: Vec<ScatterPoint>) {
        let mut inner = self.inner.lock().unwrap();
        inner.points = points;
        inner.dirty = true;
    }

    /// Take the set if it changed since the last call.
    pub fn take_if_dirty(&self) -> Option<Vec<ScatterPoint>> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.dirty {
            return None;
        }
        inner.dirty = false;
        Some(inner.points.clone())
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
