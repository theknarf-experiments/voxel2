//! The editor contract: how the level's own types say what an editor
//! should make of them.
//!
//! Everything here is a `bevy_reflect` custom attribute, read back with
//! [`NamedField::get_attribute`]. The declaration lives on the type, so
//! there is no second list to forget to update — the same reason the GPU
//! layouts are generated from one table rather than restated per shader.
//!
//! Field NAMES and TOOLTIPS are not in here: `reflect_documentation` is
//! on, so `NamedField::docs()` already returns the rationale written
//! above each field. An editor that restated those would be a second
//! place for them to disagree.
//!
//! [`NamedField::get_attribute`]: bevy::reflect::NamedField::get_attribute

use bevy::prelude::*;
use bevy::reflect::{PartialReflect, ReflectRef};

/// Bounds a number to a slider instead of an unbounded drag field.
///
/// Only meaningful where the bounds are a fact about the WORLD rather
/// than about this level: a chance is in `[0, 1]` everywhere, an
/// amplitude in meters is bounded by nothing.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct Range(pub f32, pub f32);

/// Units per pixel of drag, for numbers [`Range`] cannot bound.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct Speed(pub f32);

/// The `[f32; 3]` / `[f32; 4]` leaves under this field are colours, not
/// three or four numbers.
///
/// Stated of the leaves rather than the field because several material
/// recipes hold a PAIR of colours (`[[f32; 3]; 2]` — two hues mixed by
/// noise), and those are two swatches, not a six-number row.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct AsColor;

/// Carried, not edited. Distinct from `#[reflect(ignore)]`, which hides a
/// field from reflection entirely and therefore from BRP too.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct Hidden;

/// This field REFERS to something else in the document: its value must be
/// one of those the given pattern enumerates (see [`resolve_options`]).
///
/// The point of the whole contract. A stack layer's `source` naming a
/// layer that does not exist is a silent nothing today — the level parses,
/// the world builds, and the thing you asked for is absent. Enumerated
/// from `"stack[].name"` it cannot be mistyped.
///
/// Names and ids are the same relationship, so this covers both: a
/// material field is `"materials[].id"`, and that it is spelled as a
/// number rather than a word is a fact about the id, not about the
/// reference. There is deliberately no separate "index into" attribute —
/// nothing in a level refers to another thing by its POSITION in a list,
/// and an attribute that said otherwise would invite it.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct OneOf(pub &'static str);

/// A reference to another NODE, by the name it declared.
///
/// A [`OneOf`] pattern cannot say this: a name may sit at any depth,
/// because a `region` scope holds nodes too and the compiler resolves
/// through it. Everything it reaches — a map's values, an enum's payload —
/// inherits it, so the attribute goes on the field that HOLDS references
/// rather than on every string one can be spelled as.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct NodeRef;

/// This field is what the row it sits in is CALLED.
///
/// Fifty-five nodes labelled by their index is a list you open one at a
/// time to find anything. Labelled by the name and the kind the level
/// wrote, it reads like the document it is — which is the whole claim of
/// building the panel out of the document.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct Title;

/// Editing this MAY restream the world.
///
/// The set of fields carrying it must agree with what `level::staleness`
/// reads; a test asserts it, because both directions of drift are silent.
/// Missing it means a slider drag restreams the world once per frame;
/// having it wrongly means an edit applies only on release, for no reason.
///
/// The worst case, not the only one: `nodes` carries it because most nodes
/// are the program, while a population inside the same list invalidates
/// only the plan. A widget cannot be half committed-on-release, so the
/// section answers for the most expensive thing in it.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct Rebuilds;

/// Enumerate what a [`OneOf`] pattern can refer to.
///
/// The pattern is dotted field steps relative to the document root, where
/// a step may end in `[]` (a list) or `{}` (a map). A trailing iteration
/// yields the container's own keys — indices for a list, keys for a map —
/// and anything after it is read from each element:
///
/// - `"materials[].id"` — every material id the level defines.
/// - `"structures{}"` — every structure's name.
/// - `"stack[].name"` — the `name` of every stack layer.
///
/// This is deliberately a pattern rather than a callback: a host adds a
/// reference space by annotating a field, not by implementing a trait and
/// remembering to register it.
/// Every name a [`NodeRef`] could hold: the level's own nodes, at any
/// depth. The compiler's list, so the menu and the checker agree.
pub fn node_names(root: &dyn PartialReflect) -> Vec<String> {
    root.try_downcast_ref::<crate::level::LevelDef>()
        .map(|level| crate::graph::names(&level.nodes))
        .unwrap_or_default()
}

pub fn resolve_options(root: &dyn PartialReflect, pattern: &str) -> Vec<String> {
    let steps: Vec<&str> = pattern.split('.').collect();
    let mut out = Vec::new();
    walk(root, &steps, pattern, &mut out);
    out
}

fn walk(value: &dyn PartialReflect, steps: &[&str], pattern: &str, out: &mut Vec<String>) {
    let Some((step, rest)) = steps.split_first() else {
        // Terminal: the value itself is the option.
        if let Some(s) = as_option_string(value) {
            out.push(s);
        }
        return;
    };

    if let Some(name) = step.strip_suffix("[]") {
        let Some(field) = descend(value, name, pattern) else {
            return;
        };
        let Ok(list) = field.reflect_ref().as_list() else {
            warn!("editor ref pattern '{pattern}': '{name}' is not a list");
            return;
        };
        // An empty list is a level with none of that thing, not a bad
        // pattern — say nothing.
        for i in 0..list.len() {
            if rest.is_empty() {
                out.push(i.to_string());
            } else if let Some(item) = list.get(i) {
                walk(item, rest, pattern, out);
            }
        }
    } else if let Some(name) = step.strip_suffix("{}") {
        let Some(field) = descend(value, name, pattern) else {
            return;
        };
        let Ok(map) = field.reflect_ref().as_map() else {
            warn!("editor ref pattern '{pattern}': '{name}' is not a map");
            return;
        };
        for (key, val) in map.iter() {
            if rest.is_empty() {
                if let Some(s) = as_option_string(key) {
                    out.push(s);
                }
            } else {
                walk(val, rest, pattern, out);
            }
        }
    } else if let Some(field) = descend(value, step, pattern) {
        walk(field, rest, pattern, out);
    }
}

/// One field step, through a struct OR an enum variant.
///
/// Enums matter as much as structs here: the stack layer vocabulary is an
/// enum whose every variant carries a `name`, so a pattern that only knew
/// how to read structs would resolve to nothing on exactly the type the
/// contract exists for.
fn descend<'a>(
    value: &'a dyn PartialReflect,
    name: &str,
    pattern: &str,
) -> Option<&'a dyn PartialReflect> {
    let field = match value.reflect_ref() {
        ReflectRef::Struct(s) => s.field(name),
        ReflectRef::Enum(e) => e.field(name),
        _ => None,
    };
    if field.is_none() {
        // A missing field is a wrong ANNOTATION, not a sparse document:
        // loud, or the dropdown is simply empty and reads as "there are
        // none of those yet".
        warn!("editor ref pattern '{pattern}': no field '{name}' here");
    }
    field
}

/// The text form of a terminal value. Names are strings and ids are
/// integers; anything else is not something one field can refer to.
fn as_option_string(value: &dyn PartialReflect) -> Option<String> {
    if let Some(s) = value.try_downcast_ref::<String>() {
        return Some(s.clone());
    }
    if let Some(n) = value.try_downcast_ref::<u32>() {
        return Some(n.to_string());
    }
    if let Some(n) = value.try_downcast_ref::<i32>() {
        return Some(n.to_string());
    }
    if let Some(n) = value.try_downcast_ref::<usize>() {
        return Some(n.to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::platform::collections::HashMap;

    #[derive(Reflect)]
    struct Doc {
        materials: Vec<Mat>,
        structures: HashMap<String, Mat>,
        stack: Vec<Layer>,
    }

    #[derive(Reflect)]
    struct Mat {
        base: f32,
    }

    #[derive(Reflect)]
    enum Layer {
        Scatter { name: String },
        Emit { name: String, source: String },
    }

    fn doc() -> Doc {
        Doc {
            materials: vec![Mat { base: 0.0 }, Mat { base: 1.0 }, Mat { base: 2.0 }],
            structures: HashMap::from_iter([("ruin".to_string(), Mat { base: 0.0 })]),
            stack: vec![
                Layer::Scatter {
                    name: "sites".into(),
                },
                Layer::Emit {
                    name: "walls".into(),
                    source: "sites".into(),
                },
            ],
        }
    }

    #[test]
    fn list_indices() {
        assert_eq!(resolve_options(&doc(), "materials[]"), ["0", "1", "2"]);
    }

    #[test]
    fn map_keys() {
        assert_eq!(resolve_options(&doc(), "structures{}"), ["ruin"]);
    }

    /// The case the contract exists for: every variant of the layer enum
    /// carries `name`, and a struct-only walk would find none of them.
    #[test]
    fn field_of_every_enum_variant() {
        assert_eq!(resolve_options(&doc(), "stack[].name"), ["sites", "walls"]);
    }

    #[test]
    fn empty_list_is_no_options_not_an_error() {
        let mut d = doc();
        d.stack.clear();
        assert!(resolve_options(&d, "stack[].name").is_empty());
    }
}
