//! Authored objects, and where a level puts them — as nodes.
//!
//! These were two lists beside `nodes[]`: a `prefabs` table and a
//! `placements` list that named into it. Everything else a level says is
//! a node, and being the exception cost them all of it — no ports for the
//! compiler to check, so a placement naming a prefab that was not there
//! was a runtime warning; no place in the graph view; and an editor that
//! had to know the two field names by heart to draw a handle on one.
//!
//! As nodes they are the pair the rest of the vocabulary already uses:
//! [`Prefab`] is a definition and nothing else, exactly as a `structure`
//! is, and [`Place`] refers to one through a port. What the compiler
//! already does for every other wire it now does for these.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use voxel_core::csg::CsgOp;
use voxel_core::opgen::Value;

use crate::graph::node::Invalidates;
use crate::graph::shapes::{NodeShapes, ReflectNodeShapes, Shape, ShapeCtx};
use crate::graph::{Domain, Node, Ports, ReflectNode};
use crate::level::CsgOpDef;
use crate::schema;

/// A reusable authored object: local-space ops around the origin.
///
/// A definition, not a thing in the world — [`Place`] is what puts one
/// somewhere, and several can put the same one in several places.
///
/// Normally one per file: `{"kind": "prefab", "use": "prefabs/hut.json"}`
/// in a level, and the file holds the rest. That is what makes a prefab
/// shareable — two levels naming the same file get the same object, and
/// editing it changes both.
#[derive(Reflect, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Node, Serialize, Deserialize, Default)]
pub struct Prefab {
    /// The object, in its own local space, around the origin.
    #[serde(default)]
    pub ops: Vec<CsgOpDef>,
    /// The file this prefab's text lives in, relative to the level that
    /// used it — absent for one written inline.
    ///
    /// Read from the `use` key the loader leaves behind, and the reason
    /// saving can put the prefab back where it came from instead of
    /// swallowing it into the level.
    #[serde(rename = "use", default, skip_serializing_if = "Option::is_none")]
    #[reflect(@schema::Hidden)]
    pub from: Option<String>,
}

impl Node for Prefab {
    fn kind(&self) -> &'static str {
        "prefab"
    }
    /// Not a point node: it compiles to no op and reaches the world only
    /// through whatever places it.
    fn domain(&self) -> Domain {
        Domain::Region
    }
    fn ports(&self) -> Ports {
        (&[], &[("prefab", Value::Host("prefab"))])
    }
    /// Editing an authored object moves voxels, but only the ones it
    /// touched — which is knowable from the ops, so the world is rebuilt
    /// where it moved rather than everywhere.
    fn invalidates(&self) -> Invalidates {
        Invalidates::Authored
    }
}

impl Prefab {
    /// This prefab without its link to the file it came from — what gets
    /// written INTO that file.
    pub fn detached(&self) -> Self {
        Self {
            from: None,
            ..self.clone()
        }
    }
}

impl Serialize for Prefab {
    /// In a level, a prefab that lives in a file is its `use` line and
    /// nothing else: everything about it belongs to that file, and
    /// [`super::prefab::write`] has already put it there. Writing the ops
    /// as well would make the level a second copy that drifts — and the
    /// splice refuses to load one, so it would not even survive a
    /// round trip.
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        if let Some(path) = &self.from {
            let mut out = s.serialize_struct("Prefab", 1)?;
            out.serialize_field("use", path)?;
            return out.end();
        }
        let mut out = s.serialize_struct("Prefab", 1)?;
        out.serialize_field("ops", &self.ops)?;
        out.end()
    }
}

/// One authored instance in the world: a [`Prefab`] put somewhere, or
/// ops written inline for something that is only ever in one place.
///
/// Applied after the procedural op sources, ordered among themselves by
/// `priority`.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Node, NodeShapes, Serialize, Deserialize, Default)]
pub struct Place {
    pub position: [f32; 3],
    #[serde(default)]
    pub yaw_deg: f32,
    #[serde(default = "d_scale")]
    pub scale: f32,
    /// Seat on the terrain: `position[1]` becomes an offset from the
    /// heightfield at (x, z) rather than an absolute height.
    #[serde(default)]
    pub snap_to_terrain: bool,
    #[serde(default)]
    pub priority: i32,
    /// Local-space ops, for a one-off with no prefab to name. A place
    /// with these declares no `prefab` port and needs no wire.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ops: Vec<CsgOpDef>,
}

voxel_core::defaults! {
    d_scale: f32 = 1.0;
}

impl Default for Place {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            yaw_deg: 0.0,
            scale: 1.0,
            snap_to_terrain: false,
            priority: 0,
            ops: Vec::new(),
        }
    }
}

impl Node for Place {
    fn kind(&self) -> &'static str {
        "place"
    }
    fn domain(&self) -> Domain {
        Domain::Region
    }
    /// A port only when there is something to name. Ops written inline
    /// ARE the object, so demanding a wire as well would make every
    /// one-off declare a prefab it does not have.
    fn ports(&self) -> Ports {
        if self.ops.is_empty() {
            (&[("prefab", Value::Host("prefab"))], &[])
        } else {
            (&[], &[])
        }
    }
    fn invalidates(&self) -> Invalidates {
        Invalidates::Authored
    }
}

/// The world-space ops one placement stamps, given the object it places.
///
/// The ONE definition of what a placement means, read by the op source
/// that carves it, the fingerprint that decides which chunks an edit
/// touched, and the handles that draw it. Three answers to "where is it"
/// would be three chances to draw a manipulator somewhere the geometry
/// is not.
pub fn stamp(place: &Place, local: &[CsgOpDef], ground: impl Fn(Vec2) -> f32) -> Vec<CsgOp> {
    let mut pos = Vec3::from(place.position);
    if place.snap_to_terrain {
        pos.y = ground(Vec2::new(pos.x, pos.z)) + place.position[1];
    }
    let (sin, cos) = place.yaw_deg.to_radians().sin_cos();
    let rot = |v: Vec3| Vec3::new(v.x * cos - v.z * sin, v.y, v.x * sin + v.z * cos);
    local
        .iter()
        .map(|def| {
            let mut op = def.to_op();
            op.center = (pos + rot(Vec3::from(op.center) * place.scale)).to_array();
            op.half = (Vec3::from(op.half) * place.scale).to_array();
            op.yaw += place.yaw_deg.to_radians();
            op.blend *= place.scale;
            op
        })
        .collect()
}

impl NodeShapes for Place {
    /// What the handles draw and drag.
    ///
    /// The path points at the PREFAB when there is one: a placement puts
    /// an object somewhere and the handles change the object, so dragging
    /// a face of a hut that stands in three villages changes the hut. The
    /// editor says which other levels share the file before it lets that
    /// happen; what it must not do is quietly edit the placement instead
    /// and leave the other two behind.
    fn shapes(&self, ctx: &ShapeCtx) -> Vec<Shape> {
        let (local, owner): (&[CsgOpDef], Option<&str>) = if self.ops.is_empty() {
            let Some(name) = ctx.wired("prefab") else {
                return Vec::new();
            };
            let Some(prefab) = ctx
                .named(name)
                .and_then(|n| n.as_any().downcast_ref::<Prefab>())
            else {
                return Vec::new();
            };
            (&prefab.ops, Some(name))
        } else {
            (&self.ops, None)
        };
        let Some(at) = ctx.path(owner) else {
            return Vec::new();
        };
        // The placement's own yaw and scale are what a world drag has to
        // be undone by; the op's yaw is the shape's own and the editor
        // already accounts for it.
        let to_local = Quat::from_rotation_y(-self.yaw_deg.to_radians());
        let scale = if self.scale.abs() < 1.0e-6 {
            1.0
        } else {
            1.0 / self.scale
        };
        stamp(self, local, ctx.ground)
            .into_iter()
            .enumerate()
            .map(|(i, op)| Shape {
                op,
                path: format!("{at}.ops[{i}]"),
                to_local,
                scale,
            })
            .collect()
    }
}

/// Every prefab a level declares, with the name its node carries.
///
/// The name is the NODE's — a prefab node is named like any other and
/// referred to by that name — so anything that needs both has to walk
/// them together rather than reading a field off the kind.
pub fn prefabs(level: &crate::LevelDef) -> Vec<(&str, &Prefab)> {
    let mut out = Vec::new();
    crate::graph::walk(&level.nodes, "", &mut |node, _| {
        if let (Some(name), Some(prefab)) = (
            node.name.as_deref(),
            node.node.0.as_any().downcast_ref::<Prefab>(),
        ) {
            out.push((name, prefab));
        }
    });
    out
}
