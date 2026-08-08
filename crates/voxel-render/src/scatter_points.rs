//! Dense point scatter: the engine's placement output for things a host
//! draws in bulk (ground cover, pebbles, sparks).
//!
//! The engine decides WHERE the points are; what they look like — blade
//! geometry, colors, wind, fade — is the host's, so the pipeline that
//! draws them lives in the app (see the demo's `grass.rs`).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};
use voxel_core::WorldId;

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
    /// Keyed by the WORLD that scattered them and the population's class
    /// name, which the level chose. The engine never interprets the name;
    /// the host draws the classes it knows.
    ///
    /// By world because points are world content and worlds share
    /// coordinates: one bucket per class meant the second world to publish
    /// its ground cover replaced the first world's, and whichever level
    /// you were standing in got the other one's grass — seated on a
    /// heightfield that is not the ground under your feet.
    classes: HashMap<(WorldId, Arc<str>), Vec<ScatterPoint>>,
    /// Class names changed since each was last taken.
    ///
    /// Per class, not one flag: a host draws each class with its own
    /// pipeline and takes them independently, so a single flag let
    /// whichever extracted first clear it and the rest never see the
    /// change at all.
    dirty: HashSet<Arc<str>>,
}

impl ScatterPoints {
    /// Replace one world's points for a class (the streamer rebuilds
    /// wholesale).
    pub fn set_class(&self, world: WorldId, class: &str, points: Vec<ScatterPoint>) {
        let mut inner = self.inner.lock().unwrap();
        let class: Arc<str> = Arc::from(class);
        inner.classes.insert((world, class.clone()), points);
        inner.dirty.insert(class);
    }

    /// Drop every class of every world.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        let names: Vec<Arc<str>> = inner.classes.keys().map(|(_, name)| name.clone()).collect();
        inner.classes.clear();
        inner.dirty.extend(names);
    }

    /// Take every world's points for one class if anything changed since
    /// the last call. The host asks for the classes it knows how to draw,
    /// and gets one buffer's worth per world because it has to draw them
    /// under different views.
    pub fn take_class_if_dirty(&self, class: &str) -> Option<Vec<(WorldId, Vec<ScatterPoint>)>> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.dirty.remove(class) {
            return None;
        }
        Some(
            inner
                .classes
                .iter()
                .filter(|((_, name), _)| &**name == class)
                .map(|((world, _), points)| (*world, points.clone()))
                .collect(),
        )
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().classes.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32) -> ScatterPoint {
        ScatterPoint {
            pos: [x, 0.0, 0.0],
            hash: 0,
        }
    }

    #[test]
    fn two_worlds_scattering_one_class_do_not_overwrite_each_other() {
        let points = ScatterPoints::default();
        points.set_class(0, "groundcover", vec![point(1.0)]);
        points.set_class(1, "groundcover", vec![point(2.0), point(3.0)]);

        let mut taken = points.take_class_if_dirty("groundcover").unwrap();
        taken.sort_by_key(|(world, _)| *world);
        assert_eq!(taken.len(), 2, "one entry per world");
        assert_eq!(taken[0].1.len(), 1);
        assert_eq!(taken[1].1.len(), 2);
        assert_eq!(points.len(), 3);
    }

    #[test]
    fn taking_one_class_does_not_consume_anothers_change() {
        let points = ScatterPoints::default();
        points.set_class(0, "groundcover", vec![point(1.0)]);
        points.set_class(0, "pebbles", vec![point(2.0)]);
        assert_eq!(points.take_class_if_dirty("pebbles").unwrap().len(), 1);
        assert_eq!(
            points.take_class_if_dirty("groundcover").unwrap().len(),
            1,
            "each class is taken on its own"
        );
        assert!(points.take_class_if_dirty("groundcover").is_none());
    }
}
