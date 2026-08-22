//! What a node kind adds to the EDITOR beyond its fields.
//!
//! The editor is built out of reflection: a node's fields become rows
//! because `TypeInfo` says what is in them, and a kind this crate has
//! never heard of is editable the moment it is annotated. That covers
//! everything a value can be typed into — and nothing a value has to be
//! POINTED at.
//!
//! Geometry is the case. A node that authors a shape wants an outline in
//! the world and a handle you can drag, and the editor already owns every
//! generic part of that: drawing, picking under the pointer, turning a
//! drag into the same [`crate::level::LevelDef`] edit a row would have
//! queued, undo, and the partial rebuild that follows. What it cannot
//! know is which fields of which kind ARE geometry, and where in the
//! world the thing they describe ends up.
//!
//! So that is all this is. A kind answers with its shapes in world space
//! and the reflect path of the field each one came from; the editor does
//! the rest and never names a kind. Found through the type registry like
//! [`Node`] itself, so a kind never names the editor either.
//!
//! [`Node`]: super::Node

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use voxel_core::csg::CsgOp;

use super::Node;

/// One draggable piece of a node's authored geometry.
pub struct Shape {
    /// Where it is, in WORLD space — what gets drawn and picked. Built
    /// by the kind, because only the kind knows what its fields mean.
    pub op: CsgOp,
    /// Reflect path of the op that defines it, from the document root —
    /// `".nodes[7].ops[2]"`. The same path `world.mutate_resources`
    /// takes, which is what makes a drag and a script the same write.
    ///
    /// ABSOLUTE rather than relative to the node that answered, because
    /// the shape a node shows is often not a shape it OWNS: a placement
    /// puts an object somewhere, and what a handle on it moves is the
    /// prefab. [`ShapeCtx::path_of`] is how a kind names the other one.
    pub path: String,
    /// World to LOCAL for this shape. A drag is measured in world meters
    /// against a field authored in local ones, and a placement that
    /// yaws its object is exactly where those two disagree.
    pub to_local: Quat,
    /// Local meters per world meter, for a placement that scales.
    pub scale: f32,
}

impl Shape {
    /// A shape whose local space IS world space — nothing placed around
    /// it, so a world drag is a local edit unchanged.
    pub fn upright(op: CsgOp, path: impl Into<String>) -> Self {
        Self {
            op,
            path: path.into(),
            to_local: Quat::IDENTITY,
            scale: 1.0,
        }
    }
}

/// What a kind needs to answer with: the rest of the document, and the
/// ground.
///
/// `by_name` rather than the whole level, for the same reason
/// `RegionCtx` takes it: a kind resolves what its ports NAME, and handing
/// it the document would let it read anything and make the wiring a
/// suggestion.
pub struct ShapeCtx<'a> {
    pub by_name: &'a HashMap<String, &'a dyn Node>,
    /// This node's own wires, so a kind reads what its PORT names rather
    /// than a field it would have had to duplicate the name into.
    pub wires: &'a super::Wires,
    /// Reflect path of a node, by name. What a kind needs to point an
    /// edit at geometry another node owns.
    pub path_of: &'a dyn Fn(&str) -> Option<String>,
    /// This node's own reflect path, for the shapes it does own.
    pub here: &'a str,
    /// Terrain height, for a node that seats itself on it.
    pub ground: &'a dyn Fn(Vec2) -> f32,
}

impl ShapeCtx<'_> {
    /// The node a port names, if the document has one.
    pub fn named(&self, name: &str) -> Option<&dyn Node> {
        self.by_name.get(name).copied()
    }

    /// What this node's `port` is wired to.
    pub fn wired(&self, port: &str) -> Option<&str> {
        self.wires.get(port)?.sources().first().map(String::as_str)
    }

    /// Reflect path of a named node, or of this one.
    pub fn path(&self, name: Option<&str>) -> Option<String> {
        match name {
            Some(name) => (self.path_of)(name),
            None => Some(self.here.to_string()),
        }
    }
}

/// A node kind with geometry the editor can show and drag.
///
/// Optional: a kind that implements nothing here has fields and no
/// handles, which is every kind that describes a field rather than a
/// place.
#[bevy::reflect::reflect_trait]
pub trait NodeShapes {
    fn shapes(&self, ctx: &ShapeCtx) -> Vec<Shape>;
}
