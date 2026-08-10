//! Loading and saving an open set of node kinds.
//!
//! `"kind"` is a string in a document and the type it names is not known
//! until runtime, so both directions go through the type registry: the kind
//! table is built by walking every registration that carries `ReflectNode`,
//! and the params are read by the type's OWN serde impl through
//! `ReflectDeserialize`. That last part is why every `#[serde(default)]` a
//! node declares keeps working — a reflection-only deserializer would have
//! thrown those away and made levels spell out every field.
//!
//! Serde cannot carry a registry through `Deserialize`, so one is put in
//! scope for the call. The same shape as `voxel_worldgen::program`'s
//! thread-local snapshot, and for the same reason.

use std::cell::RefCell;

use bevy::prelude::*;
use bevy::reflect::{TypeRegistration, TypeRegistry, TypeRegistryArc};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::node::{AnyNode, ReflectNode};
use super::NodeDef;

thread_local! {
    /// The registry in scope for the current load or save.
    static SCOPED: RefCell<Option<TypeRegistryArc>> = const { RefCell::new(None) };
}

/// Put `registry` in scope for `f`, so nodes inside it can be resolved.
pub fn with_registry<R>(registry: &TypeRegistryArc, f: impl FnOnce() -> R) -> R {
    SCOPED.with(|slot| *slot.borrow_mut() = Some(registry.clone()));
    let out = f();
    SCOPED.with(|slot| *slot.borrow_mut() = None);
    out
}

fn scoped<R>(f: impl FnOnce(&TypeRegistry) -> R) -> Option<R> {
    SCOPED.with(|slot| {
        let slot = slot.borrow();
        let arc = slot.as_ref()?;
        let guard = arc.read();
        Some(f(&guard))
    })
}

/// Every registered node kind, by the name a level writes.
///
/// Built by walking the registry rather than from a list, so a host adds a
/// kind by registering a type and a forgotten line cannot silently remove
/// one — the failure `space-wizard-horror` invites with its 22-line
/// registration block.
pub fn kinds(registry: &TypeRegistry) -> Vec<(&'static str, &TypeRegistration)> {
    registry
        .iter()
        .filter_map(|reg| {
            let node = reg.data::<ReflectNode>()?;
            let default = reg.data::<bevy::reflect::std_traits::ReflectDefault>()?;
            let value = default.default();
            let kind = node.get(value.as_ref())?.kind();
            Some((kind, reg))
        })
        .collect()
}

/// A registry holding every node kind this crate ships.
///
/// For tests and for a host that wants to read a level before it has an
/// `App`. A host's own kinds are registered on top of it.
pub fn engine_kinds() -> TypeRegistryArc {
    let arc = TypeRegistryArc::default();
    {
        let mut reg = arc.write();
        super::nodes::register(&mut reg);
        reg.register::<NodeDef>();
    }
    arc
}

impl<'de> Deserialize<'de> for NodeDef {
    /// Read `{kind, name, in, ...params}`.
    ///
    /// Self-describing formats only: the params cannot be read until
    /// `"kind"` has been seen, and serde offers no way to look ahead. A
    /// level is JSON, so this is not a limitation it can run into.
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let mut map = serde_json::Map::deserialize(d)?;

        let kind = match map.remove("kind") {
            Some(serde_json::Value::String(k)) => k,
            _ => return Err(D::Error::custom("a node needs a \"kind\"")),
        };
        let name = match map.remove("name") {
            Some(serde_json::Value::String(n)) => Some(n),
            Some(other) => return Err(D::Error::custom(format!("name must be a string: {other}"))),
            None => None,
        };
        let wires = match map.remove("in") {
            Some(v) => serde_json::from_value(v).map_err(D::Error::custom)?,
            None => Default::default(),
        };

        let params = serde_json::Value::Object(map);
        let node = scoped(|registry| {
            let (_, reg) = kinds(registry)
                .into_iter()
                .find(|(k, _)| *k == kind)
                .ok_or_else(|| format!("no node kind called '{kind}' is registered"))?;
            let de = reg
                .data::<bevy::reflect::ReflectDeserialize>()
                .ok_or_else(|| format!("'{kind}' is a node but not deserializable"))?;
            // A kind with no parameters is a unit struct, and serde wants
            // a unit for one rather than an empty object. Both spellings
            // reach the same place, so a level never has to know which its
            // node happens to be.
            let value = de.deserialize(&params).or_else(|e| {
                if params.as_object().is_some_and(serde_json::Map::is_empty) {
                    de.deserialize(&serde_json::Value::Null)
                        .map_err(|e| format!("'{kind}': {e}"))
                } else {
                    Err(format!("'{kind}': {e}"))
                }
            })?;
            let node = registry
                .get_type_data::<ReflectNode>(reg.type_id())
                .and_then(|n| n.get_boxed(value).ok())
                .ok_or_else(|| format!("'{kind}' did not resolve to a node"))?;
            Ok::<_, String>(AnyNode(node))
        })
        .ok_or_else(|| {
            D::Error::custom(
                "no type registry in scope — load a level through \
                 LevelDef::from_json, which puts one there",
            )
        })?
        .map_err(D::Error::custom)?;

        Ok(NodeDef { name, wires, node })
    }
}

impl Serialize for NodeDef {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error as _;
        let params = scoped(|registry| {
            let ty = self
                .node
                .0
                .as_partial_reflect()
                .get_represented_type_info()
                .ok_or("a node has no type")?;
            let reg = registry
                .get(ty.type_id())
                .ok_or_else(|| format!("'{}' is not registered", ty.type_path()))?;
            let ser = reg
                .data::<bevy::reflect::ReflectSerialize>()
                .ok_or_else(|| format!("'{}' is a node but not serializable", ty.type_path()))?;
            serde_json::to_value(&*ser.get_serializable(self.node.0.as_reflect()))
                .map_err(|e| e.to_string())
        })
        .ok_or_else(|| S::Error::custom("no type registry in scope"))?
        .map_err(S::Error::custom)?;

        let mut map = match params {
            serde_json::Value::Object(map) => map,
            // A kind with no parameters is a unit struct and writes a unit;
            // it has nothing to merge, only its kind and its wiring.
            serde_json::Value::Null => serde_json::Map::new(),
            other => {
                return Err(S::Error::custom(format!(
                    "a node must serialize to an object or a unit, got {other}"
                )))
            }
        };
        // Written first, so a level reads as "what this is, then what it is
        // wired to, then its numbers".
        let mut out = serde_json::Map::new();
        out.insert("kind".into(), self.node.kind().into());
        if let Some(name) = &self.name {
            out.insert("name".into(), name.clone().into());
        }
        if !self.wires.is_empty() {
            out.insert(
                "in".into(),
                serde_json::to_value(&self.wires).map_err(S::Error::custom)?,
            );
        }
        out.append(&mut map);
        out.serialize(s)
    }
}
