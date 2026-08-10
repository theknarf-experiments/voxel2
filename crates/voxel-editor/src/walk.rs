//! Turning a reflected document into a flat list of rows.
//!
//! Flat with a depth, not a tree of scenes: the rows are respawned
//! whenever the document or the expansion set changes, and a flat list is
//! the shape that survives being rebuilt. Only expanded paths are
//! descended into, so the cost is the rows you can see rather than the
//! several thousand a level contains.

use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use bevy::reflect::{NamedField, PartialReflect, ReflectRef, TypeInfo};
use bevy::reflect::enums::{VariantInfo, VariantType};
use voxel_engine::schema;

/// One line of the panel.
pub struct Row {
    /// Reflect path from the document root. `.field` and `[index]` are
    /// BRP's own syntax; `{key}` extends it to maps, which BRP cannot
    /// address at all.
    pub path: String,
    pub label: String,
    /// The rationale written above the field, via `reflect_documentation`.
    pub docs: Option<&'static str>,
    pub depth: usize,
    /// Editing this restreams the world ([`schema::Rebuilds`]), so its
    /// widget commits on release rather than continuously.
    pub rebuilds: bool,
    pub kind: RowKind,
}

/// What a row shows. One variant per widget, and one for "no widget yet".
pub enum RowKind {
    /// A struct, tuple, list, map or array: a disclosure and a summary of
    /// what is inside, so a collapsed row still says something.
    Group { expanded: bool, summary: String },
    /// An enum: a disclosure AND a variant choice, because those are the
    /// same row — picking a variant changes what is inside it.
    Variant {
        expanded: bool,
        current: String,
        options: Vec<String>,
    },
    Number {
        value: f64,
        num: Num,
        range: Option<schema::Range>,
    },
    Bool(bool),
    Text(String),
    /// Linear RGB, from a field marked [`schema::AsColor`].
    Color([f32; 3]),
    /// A [`schema::OneOf`] reference: the value must be one of `options`.
    Choice {
        current: String,
        options: Vec<String>,
        num: Option<Num>,
    },
    /// A type with no widget. Shown, never skipped: a field that silently
    /// vanishes reads as a field that does not exist, and this crate would
    /// rather admit the gap than hide it.
    Unsupported(&'static str),
}

/// The numeric types a level is written in. Anything else is
/// [`RowKind::Unsupported`] rather than quietly truncated.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Num {
    #[default]
    F32,
    F64,
    U8,
    U32,
    I32,
    Usize,
}

impl Num {
    fn of(value: &dyn PartialReflect) -> Option<(Self, f64)> {
        if let Some(v) = value.try_downcast_ref::<f32>() {
            return Some((Num::F32, *v as f64));
        }
        if let Some(v) = value.try_downcast_ref::<f64>() {
            return Some((Num::F64, *v));
        }
        if let Some(v) = value.try_downcast_ref::<u8>() {
            return Some((Num::U8, *v as f64));
        }
        if let Some(v) = value.try_downcast_ref::<u32>() {
            return Some((Num::U32, *v as f64));
        }
        if let Some(v) = value.try_downcast_ref::<i32>() {
            return Some((Num::I32, *v as f64));
        }
        if let Some(v) = value.try_downcast_ref::<usize>() {
            return Some((Num::Usize, *v as f64));
        }
        None
    }
}

/// How a field asked to be shown — the [`schema`] attributes on it,
/// gathered once at the field and carried down into what it contains.
#[derive(Clone, Copy, Default)]
struct Hints {
    /// [`schema::AsColor`] applies to the `[f32; 3]` LEAVES, so it
    /// survives the step through an array of colour pairs.
    color: bool,
    range: Option<schema::Range>,
    one_of: Option<schema::OneOf>,
    /// Inherited all the way down: every number inside a section that
    /// restreams the world restreams it too.
    rebuilds: bool,
}

/// Everything the walk needs that is not the value in hand.
struct Cx<'a> {
    root: &'a dyn PartialReflect,
    expanded: &'a HashSet<String>,
    out: Vec<Row>,
}

/// Build the visible rows of `root`.
pub fn rows(root: &dyn PartialReflect, expanded: &HashSet<String>) -> Vec<Row> {
    let mut cx = Cx {
        root,
        expanded,
        out: Vec::new(),
    };
    // The root's own container row is the panel header, so descend
    // straight into its children.
    children(&mut cx, root, "", 0, Hints::default());
    cx.out
}

/// Emit one row for `value`, and its children if it is open.
fn emit(
    cx: &mut Cx,
    value: &dyn PartialReflect,
    path: &str,
    label: String,
    docs: Option<&'static str>,
    depth: usize,
    hints: Hints,
) {
    let expanded = cx.expanded.contains(path);

    // A reference beats the value's own type: a material id is a choice
    // among the ids this level defines, not a number to be typed.
    if let Some(schema::OneOf(pattern)) = hints.one_of {
        let options = schema::resolve_options(cx.root, pattern);
        let (num, current) = match Num::of(value) {
            Some((n, v)) => (Some(n), format_num(v, n)),
            None => (
                None,
                value
                    .try_downcast_ref::<String>()
                    .cloned()
                    .unwrap_or_default(),
            ),
        };
        cx.out.push(Row {
            path: path.to_string(),
            label,
            docs,
            depth,
            rebuilds: hints.rebuilds,
            kind: RowKind::Choice {
                current,
                options,
                num,
            },
        });
        return;
    }

    if hints.color && is_rgb(value) {
        let mut rgb = [0.0; 3];
        let ReflectRef::Array(a) = value.reflect_ref() else {
            unreachable!("is_rgb checked it")
        };
        for (i, slot) in rgb.iter_mut().enumerate() {
            *slot = *a
                .get(i)
                .and_then(|c| c.try_downcast_ref::<f32>())
                .unwrap_or(&0.0);
        }
        cx.out.push(Row {
            path: path.to_string(),
            label,
            docs,
            depth,
            rebuilds: hints.rebuilds,
            kind: RowKind::Color(rgb),
        });
        return;
    }

    if let Some((num, value_f)) = Num::of(value) {
        cx.out.push(Row {
            path: path.to_string(),
            label,
            docs,
            depth,
            rebuilds: hints.rebuilds,
            kind: RowKind::Number {
                value: value_f,
                num,
                range: hints.range,
            },
        });
        return;
    }
    if let Some(b) = value.try_downcast_ref::<bool>() {
        cx.out.push(Row {
            path: path.to_string(),
            label,
            docs,
            depth,
            rebuilds: hints.rebuilds,
            kind: RowKind::Bool(*b),
        });
        return;
    }
    if let Some(s) = value.try_downcast_ref::<String>() {
        cx.out.push(Row {
            path: path.to_string(),
            label,
            docs,
            depth,
            rebuilds: hints.rebuilds,
            kind: RowKind::Text(s.clone()),
        });
        return;
    }

    let kind = match value.reflect_ref() {
        ReflectRef::Struct(s) => RowKind::Group {
            expanded,
            summary: format!("{} fields", s.field_len()),
        },
        ReflectRef::TupleStruct(t) => RowKind::Group {
            expanded,
            summary: format!("{} fields", t.field_len()),
        },
        ReflectRef::Tuple(t) => RowKind::Group {
            expanded,
            summary: format!("{} fields", t.field_len()),
        },
        ReflectRef::List(l) => RowKind::Group {
            expanded,
            summary: format!("{} items", l.len()),
        },
        ReflectRef::Array(a) => RowKind::Group {
            expanded,
            summary: format!("{} items", a.len()),
        },
        ReflectRef::Map(m) => RowKind::Group {
            expanded,
            summary: format!("{} entries", m.len()),
        },
        ReflectRef::Enum(e) => RowKind::Variant {
            expanded,
            current: e.variant_name().to_string(),
            options: variant_names(value),
        },
        _ => RowKind::Unsupported(type_name(value)),
    };

    let container = !matches!(kind, RowKind::Unsupported(_));
    cx.out.push(Row {
        path: path.to_string(),
        label,
        docs,
        depth,
        rebuilds: hints.rebuilds,
        kind,
    });
    if container && expanded {
        children(cx, value, path, depth + 1, hints);
    }
}

/// One row per member of a container.
fn children(cx: &mut Cx, value: &dyn PartialReflect, path: &str, depth: usize, hints: Hints) {
    match value.reflect_ref() {
        ReflectRef::Struct(s) => {
            for i in 0..s.field_len() {
                let Some(field) = s.field_at(i) else { continue };
                // `NamedField::name` is already `&'static str`; reading the
                // name off the VALUE instead gives a borrowed one and
                // tempts a `Box::leak` to get out of it.
                let Some(named) = struct_field_info(value, i) else {
                    continue;
                };
                if named.has_attribute::<schema::Hidden>() {
                    continue;
                }
                emit(
                    cx,
                    field,
                    &format!("{path}.{}", named.name()),
                    named.name().to_string(),
                    named.docs(),
                    depth,
                    field_hints(named, hints),
                );
            }
        }
        ReflectRef::Enum(e) => match e.variant_type() {
            VariantType::Struct => {
                for i in 0..e.field_len() {
                    let (Some(name), Some(field)) = (e.name_at(i), e.field_at(i)) else {
                        continue;
                    };
                    let named = variant_field_info(value, e.variant_name(), i);
                    if named.is_some_and(|f| f.has_attribute::<schema::Hidden>()) {
                        continue;
                    }
                    emit(
                        cx,
                        field,
                        &format!("{path}.{name}"),
                        name.to_string(),
                        named.and_then(|f| f.docs()),
                        depth,
                        named.map_or(hints, |f| field_hints(f, hints)),
                    );
                }
            }
            VariantType::Tuple => {
                for i in 0..e.field_len() {
                    let Some(field) = e.field_at(i) else { continue };
                    // An `Option<T>`'s payload is the field, not a numbered
                    // slot of it — labelling it "0" would be true and
                    // useless.
                    let label = if e.field_len() == 1 {
                        e.variant_name().to_string()
                    } else {
                        i.to_string()
                    };
                    emit(cx, field, &format!("{path}.{i}"), label, None, depth, hints);
                }
            }
            VariantType::Unit => {}
        },
        ReflectRef::TupleStruct(t) => {
            for i in 0..t.field_len() {
                let Some(field) = t.field(i) else { continue };
                emit(
                    cx,
                    field,
                    &format!("{path}.{i}"),
                    i.to_string(),
                    None,
                    depth,
                    hints,
                );
            }
        }
        ReflectRef::Tuple(t) => {
            for i in 0..t.field_len() {
                let Some(field) = t.field(i) else { continue };
                emit(
                    cx,
                    field,
                    &format!("{path}.{i}"),
                    i.to_string(),
                    None,
                    depth,
                    hints,
                );
            }
        }
        ReflectRef::List(l) => {
            for i in 0..l.len() {
                let Some(item) = l.get(i) else { continue };
                emit(
                    cx,
                    item,
                    &format!("{path}[{i}]"),
                    item_label(item, i),
                    None,
                    depth,
                    hints,
                );
            }
        }
        ReflectRef::Array(a) => {
            for i in 0..a.len() {
                let Some(item) = a.get(i) else { continue };
                emit(
                    cx,
                    item,
                    &format!("{path}[{i}]"),
                    i.to_string(),
                    None,
                    depth,
                    hints,
                );
            }
        }
        ReflectRef::Map(m) => {
            // Sorted, because a map's iteration order is not stable and
            // rows that reshuffle every rebuild cannot be clicked.
            let mut entries: Vec<(String, &dyn PartialReflect)> = m
                .iter()
                .filter_map(|(k, v)| Some((k.try_downcast_ref::<String>()?.clone(), v)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            for (key, val) in entries {
                emit(
                    cx,
                    val,
                    &format!("{path}{{{key}}}"),
                    key.clone(),
                    None,
                    depth,
                    hints,
                );
            }
        }
        _ => {}
    }
}

/// A list item names itself when it can — an op's variant, a material's
/// recipe — because "3" is not what you are looking for in a list of forty.
fn item_label(item: &dyn PartialReflect, index: usize) -> String {
    if let ReflectRef::Enum(e) = item.reflect_ref() {
        return format!("{index}  {}", e.variant_name());
    }
    // A struct that flattens an enum (a generator entry holding an op)
    // shows the inner variant instead of its own field count.
    if let ReflectRef::Struct(s) = item.reflect_ref() {
        for i in 0..s.field_len() {
            if let Some(ReflectRef::Enum(e)) = s.field_at(i).map(|f| f.reflect_ref()) {
                return format!("{index}  {}", e.variant_name());
            }
        }
    }
    index.to_string()
}

/// A named field's own annotations replace whatever it was reached
/// through. Inheritance happens on the way through anonymous containers —
/// an array element has no annotations of its own, so a colour pair stays
/// a colour pair — not on the way into a new field.
fn field_hints(field: &NamedField, inherited: Hints) -> Hints {
    Hints {
        color: field.has_attribute::<schema::AsColor>(),
        range: field.get_attribute::<schema::Range>().copied(),
        one_of: field.get_attribute::<schema::OneOf>().copied(),
        // Not replaced: a section marked as restreaming contains only
        // fields that restream, however deep.
        rebuilds: inherited.rebuilds || field.has_attribute::<schema::Rebuilds>(),
    }
}

fn struct_field_info(
    value: &dyn PartialReflect,
    index: usize,
) -> Option<&'static NamedField> {
    match value.get_represented_type_info()? {
        TypeInfo::Struct(info) => info.field_at(index),
        _ => None,
    }
}

fn variant_field_info(
    value: &dyn PartialReflect,
    variant: &str,
    index: usize,
) -> Option<&'static NamedField> {
    let TypeInfo::Enum(info) = value.get_represented_type_info()? else {
        return None;
    };
    match info.variant(variant)? {
        VariantInfo::Struct(v) => v.field_at(index),
        _ => None,
    }
}

fn variant_names(value: &dyn PartialReflect) -> Vec<String> {
    match value.get_represented_type_info() {
        Some(TypeInfo::Enum(info)) => info.iter().map(|v| v.name().to_string()).collect(),
        // A dynamic enum knows its current variant but not its siblings.
        // Better an empty menu than a fabricated one.
        _ => Vec::new(),
    }
}

fn is_rgb(value: &dyn PartialReflect) -> bool {
    match value.reflect_ref() {
        ReflectRef::Array(a) => {
            (a.len() == 3 || a.len() == 4)
                && a.get(0).is_some_and(|c| c.try_downcast_ref::<f32>().is_some())
        }
        _ => false,
    }
}

fn type_name(value: &dyn PartialReflect) -> &'static str {
    value
        .get_represented_type_info()
        .map_or("<dynamic>", |i| i.type_path())
}

/// A number as the level would write it.
///
/// Rust's float `Display` is the shortest form that round-trips, which is
/// what a level file already holds — so this shows `0.00005` and `800`
/// rather than padding or truncating either. A FIXED number of decimals
/// cannot do both: at four places a generator's `scale` of `5e-5` renders
/// as `0`, and the panel is then confidently wrong about the value it was
/// opened to check.
///
/// Widened to `f64` on the way in, so an `f32` is narrowed back before
/// printing or `0.3f32` reads as `0.30000001192092896`.
pub fn format_num(value: f64, num: Num) -> String {
    match num {
        Num::F32 => format!("{}", value as f32),
        Num::F64 => format!("{value}"),
        _ => format!("{}", value as i64),
    }
}
