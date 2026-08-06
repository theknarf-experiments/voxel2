//! What planning produces, as plain data.
//!
//! Planning layers are host code, so the engine and the host need a shared
//! vocabulary for their results. It lives here, below both: CSG ops shape
//! the density field, ribbons and clearance describe flat strips on the
//! ground, markers name places. None of these say what they are *for* —
//! a ribbon is a river, a canal or a conveyor depending on its material.

use glam::{Vec2, Vec3};

use crate::csg::CsgOp;

/// A flat ribbon segment along a path: endpoints, half width, and the
/// surface height at each end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RibbonSeg {
    pub a: Vec2,
    pub b: Vec2,
    pub half_w: f32,
    pub levels: [f32; 2],
    /// Level material id. The host decides what a ribbon of this material
    /// looks like; the engine only says where it is.
    pub material: u32,
}

/// A point of interest emitted by planning (dungeon entrance, bridge,
/// spawn anchor...). `kind` is a host-defined tag.
#[derive(Clone, Debug, PartialEq)]
pub struct Marker {
    pub pos: Vec3,
    pub kind: String,
}

/// Everything one planning chunk emits, bucketed together so a single
/// spatial query answers every consumer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PatchSet {
    pub ops: Vec<CsgOp>,
    pub ribbons: Vec<RibbonSeg>,
    /// Segments props must keep off: roadbeds, ribbon beds, thresholds.
    pub clearance: Vec<[Vec2; 2]>,
    pub markers: Vec<Marker>,
}

impl PatchSet {
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
            && self.ribbons.is_empty()
            && self.clearance.is_empty()
            && self.markers.is_empty()
    }

    pub fn extend(&mut self, other: PatchSet) {
        self.ops.extend(other.ops);
        self.ribbons.extend(other.ribbons);
        self.clearance.extend(other.clearance);
        self.markers.extend(other.markers);
    }
}
