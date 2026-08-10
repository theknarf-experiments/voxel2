//! What a node IS, generically.
//!
//! The engine owns this trait and nothing else about the vocabulary. A node
//! kind is an ordinary Rust struct that derives `Reflect` and implements
//! [`Node`]; its FIELDS are its schema, read back by reflection, and the
//! type registry is the kind registry. Adding a kind is registering a type
//! — there is no central enum to extend, and nothing in the engine needs to
//! know that the demo's planning layers exist.
//!
//! Taken from `space-wizard-horror`'s `#[reflect_trait] ProcNode`, which
//! does the same thing for its editor. What it did not need and this does
//! is LOADING an open set from a file: SWH builds its graph in Rust, so its
//! kinds are known at the call site. Here `"kind"` is a string in a level,
//! resolved through the registry — see [`Registry`].

use core::any::Any;
use core::fmt::Formatter;

use bevy::prelude::*;
use bevy::reflect::{
    ApplyError, PartialReflect, ReflectKind, ReflectMut, ReflectOwned, ReflectRef, TypeInfo,
};
use voxel_core::opgen::Port;
use voxel_core::worldop::WorldOp;

/// What a node consumes and produces.
pub type Ports = (&'static [Port], &'static [Port]);

/// Where a node runs.
///
/// The one irreducible split: a point node is a pure function of position
/// and compiles into the program both interpreters run; a region node has a
/// lifetime and becomes a layer. Both are declared, named, referenced and
/// edited identically — this decides the backend, not the language.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Domain {
    #[default]
    Point,
    Region,
}

/// Cloning a node behind a box, which `Reflect` alone cannot do.
///
/// Blanket-implemented for every cloneable node, so no kind writes it.
/// `space-wizard-horror` spells this `ProcNodeClone`.
pub trait CloneNode {
    fn clone_node(&self) -> Box<dyn Node>;
}

impl<T: Node + Clone> CloneNode for T {
    fn clone_node(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

/// A node of a level's graph.
#[bevy::reflect::reflect_trait]
pub trait Node: Reflect + CloneNode {
    /// What a level writes in `"kind"`. Snake case by convention, and the
    /// only name this type answers to in a document.
    fn kind(&self) -> &'static str;

    /// The values this node consumes and produces.
    fn ports(&self) -> Ports;

    fn domain(&self) -> Domain {
        Domain::Point
    }

    /// Lower to the interpreter form. `None` for anything that emits no op:
    /// a region node, a scope, or an origin, which IS the register file's
    /// initial state rather than something that sets it.
    fn op(&self, field_slot: u32) -> Option<WorldOp> {
        let _ = field_slot;
        None
    }

    /// A scope's children. Empty for everything else.
    fn children(&self) -> &[super::NodeDef] {
        &[]
    }

    /// The gate a scope applies to its children.
    fn gate(&self) -> Option<[f32; 4]> {
        None
    }
}

/// A node, stored in a reflected document.
///
/// The delegation below exists because `Box<dyn Node>` is not itself
/// reflectable and bevy has no blanket impl for one — so a struct holding
/// a node could not derive `Reflect`, and an editor walking a level would
/// stop at exactly the interesting part.
///
/// Every method forwards to the node INSIDE, which is what makes the walk
/// see `HeightFbm` and its fields rather than a wrapper: `reflect_ref`
/// returns the concrete struct, and `get_represented_type_info` its type.
pub struct AnyNode(pub Box<dyn Node>);

impl AnyNode {
    pub fn kind(&self) -> &'static str {
        self.0.kind()
    }
}

impl Clone for AnyNode {
    fn clone(&self) -> Self {
        Self(self.0.clone_node())
    }
}

impl core::fmt::Debug for AnyNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.0.as_partial_reflect().debug(f)
    }
}

impl PartialEq for AnyNode {
    fn eq(&self, other: &Self) -> bool {
        // Different kinds are never equal, whatever their fields say.
        self.0.kind() == other.0.kind()
            && self
                .0
                .as_partial_reflect()
                .reflect_partial_eq(other.0.as_partial_reflect())
                .unwrap_or(false)
    }
}

impl Default for AnyNode {
    fn default() -> Self {
        Self(Box::new(super::nodes::SdfVoid))
    }
}

impl PartialReflect for AnyNode {
    fn get_represented_type_info(&self) -> Option<&'static TypeInfo> {
        self.0.as_partial_reflect().get_represented_type_info()
    }
    fn into_partial_reflect(self: Box<Self>) -> Box<dyn PartialReflect> {
        self.0.into_partial_reflect()
    }
    fn as_partial_reflect(&self) -> &dyn PartialReflect {
        self.0.as_partial_reflect()
    }
    fn as_partial_reflect_mut(&mut self) -> &mut dyn PartialReflect {
        self.0.as_partial_reflect_mut()
    }
    // Identity stays SELF while inspection delegates. A walk asks
    // `reflect_ref` and must see `HeightFbm`'s fields; a container asks
    // `as_any` and must get an `AnyNode` back, or a `Vec<AnyNode>` cannot
    // be reflected at all.
    fn try_into_reflect(self: Box<Self>) -> Result<Box<dyn Reflect>, Box<dyn PartialReflect>> {
        Ok(self)
    }
    fn try_as_reflect(&self) -> Option<&dyn Reflect> {
        Some(self)
    }
    fn try_as_reflect_mut(&mut self) -> Option<&mut dyn Reflect> {
        Some(self)
    }
    fn try_apply(&mut self, value: &dyn PartialReflect) -> Result<(), ApplyError> {
        self.0.as_partial_reflect_mut().try_apply(value)
    }
    fn reflect_kind(&self) -> ReflectKind {
        self.0.as_partial_reflect().reflect_kind()
    }
    fn reflect_ref(&self) -> ReflectRef<'_> {
        self.0.as_partial_reflect().reflect_ref()
    }
    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        self.0.as_partial_reflect_mut().reflect_mut()
    }
    fn reflect_owned(self: Box<Self>) -> ReflectOwned {
        self.0.into_partial_reflect().reflect_owned()
    }
    fn reflect_partial_eq(&self, value: &dyn PartialReflect) -> Option<bool> {
        self.0.as_partial_reflect().reflect_partial_eq(value)
    }
    fn debug(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.0.as_partial_reflect().debug(f)
    }
}

impl Reflect for AnyNode {
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn into_reflect(self: Box<Self>) -> Box<dyn Reflect> {
        self
    }
    fn as_reflect(&self) -> &dyn Reflect {
        self
    }
    fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
        self
    }
    fn set(&mut self, value: Box<dyn Reflect>) -> Result<(), Box<dyn Reflect>> {
        match value.downcast::<Self>() {
            Ok(v) => {
                *self = *v;
                Ok(())
            }
            Err(v) => Err(v),
        }
    }
}

/// Only from another node: a node's real type is named by a document and
/// resolved through the registry, so there is nothing to rebuild one from
/// out of loose reflected fields.
impl bevy::reflect::FromReflect for AnyNode {
    fn from_reflect(value: &dyn PartialReflect) -> Option<Self> {
        value.try_downcast_ref::<Self>().cloned()
    }
}

impl bevy::reflect::TypePath for AnyNode {
    fn type_path() -> &'static str {
        "voxel_engine::graph::node::AnyNode"
    }
    fn short_type_path() -> &'static str {
        "AnyNode"
    }
}

/// Opaque STATICALLY, concrete dynamically.
///
/// `Reflect` requires static type info and a node's real type is not known
/// until a level names it — so this reports opaque here, while
/// `get_represented_type_info` and `reflect_ref` hand back the node
/// inside. Those are different questions and they get different answers.
impl bevy::reflect::Typed for AnyNode {
    fn type_info() -> &'static TypeInfo {
        static CELL: bevy::reflect::utility::NonGenericTypeInfoCell =
            bevy::reflect::utility::NonGenericTypeInfoCell::new();
        CELL.get_or_set(|| TypeInfo::Opaque(bevy::reflect::OpaqueInfo::new::<Self>()))
    }
}

impl bevy::reflect::GetTypeRegistration for AnyNode {
    fn get_type_registration() -> bevy::reflect::TypeRegistration {
        bevy::reflect::TypeRegistration::of::<Self>()
    }
}
