//! Turning a reflected document into a flat list of rows.
//!
//! Flat with a depth, not a tree of scenes: the rows are respawned
//! whenever the document or the expansion set changes, and a flat list is
//! the shape that survives being rebuilt. Only expanded paths are
//! descended into, so the cost is the rows you can see rather than the
//! several thousand a level contains.

use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use bevy::reflect::enums::{VariantInfo, VariantType};
use bevy::reflect::{NamedField, PartialReflect, ReflectRef, TypeInfo};
use voxel_engine::schema;

use crate::Sections;

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
    Group {
        expanded: bool,
        summary: String,
    },
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
        /// Units per pixel of drag, for a number no [`schema::Range`]
        /// bounds. Declared by the field where it knows better, and
        /// otherwise proportional to the value: a scale of `5e-5` and an
        /// amplitude of `800` cannot share a step, and one that suited
        /// either would make the other untouchable.
        speed: f32,
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
    speed: Option<schema::Speed>,
    one_of: Option<schema::OneOf>,
    /// Inherited all the way down: every number inside a section that
    /// restreams the world restreams it too.
    rebuilds: bool,
    /// [`schema::NodeRef`], inherited the same way `color` is: the strings
    /// it applies to sit inside a map and an enum, neither of which is a
    /// named field that could carry it.
    node_ref: bool,
}

/// Everything the walk needs that is not the value in hand.
struct Cx<'a> {
    root: &'a dyn PartialReflect,
    expanded: &'a HashSet<String>,
    /// Which top-level sections this tab shows. Applied at the root only:
    /// below it, a field named `nodes` is a field, not a section.
    sections: &'a Sections,
    out: Vec<Row>,
}

/// Build the visible rows of the whole document.
pub fn rows(root: &dyn PartialReflect, expanded: &HashSet<String>) -> Vec<Row> {
    rows_in(root, expanded, &Sections::All)
}

/// Build the visible rows of one tab's view of `root`.
pub fn rows_in(
    root: &dyn PartialReflect,
    expanded: &HashSet<String>,
    sections: &Sections,
) -> Vec<Row> {
    let mut cx = Cx {
        root,
        expanded,
        sections,
        out: Vec::new(),
    };
    // A tab that names exactly ONE section shows that section's contents.
    // A tab called "Nodes" whose whole body is a collapsed row called
    // `nodes` has spent a tab to say a word. The paths stay absolute, so
    // what a row edits does not depend on which tab it was reached from.
    if let Sections::Only(names) = sections {
        if let [only] = names.as_slice() {
            if let Some((field, info)) = named_field(root, only) {
                let hints = field_hints(info, Hints::default());
                children(&mut cx, field, &format!(".{only}"), 0, hints);
                return cx.out;
            }
        }
    }
    // The root's own container row is the panel header, so descend
    // straight into its children.
    children(&mut cx, root, "", 0, Hints::default());
    cx.out
}

/// The rows of ONE value, reached at `path`.
///
/// The paths the rows carry stay absolute, so a widget built from them
/// edits the document and not a copy — and `root` is still the whole
/// document, because a reference row's options are enumerated from it.
///
/// `section` names the top-level field the value lives under, so the
/// attributes it carries — a section that restreams the world, above all —
/// reach the rows the same way they would on the way down.
pub fn rows_at(
    root: &dyn PartialReflect,
    value: &dyn PartialReflect,
    path: &str,
    section: &str,
    expanded: &HashSet<String>,
) -> Vec<Row> {
    let mut cx = Cx {
        root,
        expanded,
        sections: &Sections::All,
        out: Vec::new(),
    };
    let hints = named_field(root, section)
        .map(|(_, info)| field_hints(info, Hints::default()))
        .unwrap_or_default();
    children(&mut cx, value, path, 0, hints);
    cx.out
}

/// A struct field by name, with the type info that carries its attributes.
fn named_field<'a>(
    value: &'a dyn PartialReflect,
    name: &str,
) -> Option<(&'a dyn PartialReflect, &'static NamedField)> {
    let TypeInfo::Struct(info) = value.get_represented_type_info()? else {
        return None;
    };
    let index = info.index_of(name)?;
    let ReflectRef::Struct(s) = value.reflect_ref() else {
        return None;
    };
    Some((s.field_at(index)?, info.field_at(index)?))
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
    // A newtype is its contents. `Wires(BTreeMap<..>)` and the demo's
    // `Population(ScatterDef)` are spellings Rust needs and the document
    // does not; shown as themselves they cost a disclosure and a row
    // saying "1 fields" before the level resumes. The path keeps the `.0`,
    // so reflection still resolves what the row addresses.
    let mut path = path.to_string();
    let mut value = value;
    while let Some(inner) = newtype(value) {
        path.push_str(".0");
        value = inner;
    }
    let path = path.as_str();

    let expanded = cx.expanded.contains(path);

    // A reference beats the value's own type: a material id is a choice
    // among the ids this level defines, not a number to be typed.
    if hints.node_ref && value.try_downcast_ref::<String>().is_some() {
        cx.out.push(Row {
            path: path.to_string(),
            label,
            docs,
            depth,
            rebuilds: hints.rebuilds,
            kind: RowKind::Choice {
                current: value
                    .try_downcast_ref::<String>()
                    .cloned()
                    .unwrap_or_default(),
                options: schema::node_names(cx.root),
                num: None,
            },
        });
        return;
    }
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
                speed: hints
                    .speed
                    .map_or_else(|| drag_speed(value_f, num), |schema::Speed(units)| units),
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
                // The tab's own filter, at the document's top level.
                if path.is_empty() && !cx.sections.shows(named.name()) {
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

/// A list item names itself when it can — a node's name and kind, an op's
/// variant, a material's recipe — because "3" is not what you are looking
/// for in a list of fifty-five.
fn item_label(item: &dyn PartialReflect, index: usize) -> String {
    if let ReflectRef::Enum(e) = item.reflect_ref() {
        return format!("{index}  {}", e.variant_name());
    }
    if let ReflectRef::Struct(s) = item.reflect_ref() {
        // What the type asked to be called by, if it asked.
        let titled: Vec<String> = (0..s.field_len())
            .filter(|&i| {
                struct_field_info(item, i).is_some_and(|f| f.has_attribute::<schema::Title>())
            })
            .filter_map(|i| s.field_at(i).and_then(title_of))
            .collect();
        if !titled.is_empty() {
            return format!("{index}  {}", titled.join("  "));
        }
        // A struct that flattens an enum shows the inner variant instead
        // of its own field count.
        for i in 0..s.field_len() {
            if let Some(ReflectRef::Enum(e)) = s.field_at(i).map(|f| f.reflect_ref()) {
                return format!("{index}  {}", e.variant_name());
            }
        }
    }
    index.to_string()
}

/// What a [`schema::Title`] field contributes to its row's label: the text
/// if it holds text, and otherwise WHAT IT IS — which for a node is the
/// concrete kind behind the box, because a dynamic value reports the type
/// it actually holds.
fn title_of(value: &dyn PartialReflect) -> Option<String> {
    if let Some(s) = value.try_downcast_ref::<String>() {
        return Some(s.clone());
    }
    match value.reflect_ref() {
        // An absent `Option` contributes nothing rather than "None": an
        // unnamed node is one the level did not need to name.
        ReflectRef::Enum(e) if e.field_len() == 1 => e.field_at(0).and_then(title_of),
        ReflectRef::Enum(_) => None,
        _ => Some(
            value
                .get_represented_type_info()?
                .type_path_table()
                .short_path()
                .to_string(),
        ),
    }
}

/// The single unnamed field a value WRAPS, or `None` for anything else.
///
/// A one-field tuple struct (`Wires`, the demo's `Population`) and a
/// one-field tuple variant (`Some(name)`, `Wire::One(source)`) are the same
/// thing to a reader: a spelling Rust needs around the value they came to
/// see. Struct variants and multi-field ones are not — those carry a
/// choice, and collapsing them would hide it.
fn newtype(value: &dyn PartialReflect) -> Option<&dyn PartialReflect> {
    match value.reflect_ref() {
        ReflectRef::TupleStruct(t) if t.field_len() == 1 => t.field(0),
        ReflectRef::Enum(e) if e.variant_type() == VariantType::Tuple && e.field_len() == 1 => {
            e.field_at(0)
        }
        _ => None,
    }
}

/// A named field's own annotations replace whatever it was reached
/// through. Inheritance happens on the way through anonymous containers —
/// an array element has no annotations of its own, so a colour pair stays
/// a colour pair — not on the way into a new field.
fn field_hints(field: &NamedField, inherited: Hints) -> Hints {
    Hints {
        color: field.has_attribute::<schema::AsColor>(),
        range: field.get_attribute::<schema::Range>().copied(),
        speed: field.get_attribute::<schema::Speed>().copied(),
        one_of: field.get_attribute::<schema::OneOf>().copied(),
        // Not replaced: a section marked as restreaming contains only
        // fields that restream, however deep.
        rebuilds: inherited.rebuilds || field.has_attribute::<schema::Rebuilds>(),
        node_ref: field.has_attribute::<schema::NodeRef>(),
    }
}

fn struct_field_info(value: &dyn PartialReflect, index: usize) -> Option<&'static NamedField> {
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
                && a.get(0)
                    .is_some_and(|c| c.try_downcast_ref::<f32>().is_some())
        }
        _ => false,
    }
}

fn type_name(value: &dyn PartialReflect) -> &'static str {
    value
        .get_represented_type_info()
        .map_or("<dynamic>", |i| i.type_path())
}

/// How fast an undeclared number moves under the pointer.
///
/// One percent of itself per pixel, so a hundred pixels roughly doubles
/// whatever it is: a level holds `5e-5` beside `800`, and any fixed step
/// that suited one would leave the other unusable. Integers move at least
/// one per hundred pixels, and a value of zero has to start somewhere.
fn drag_speed(value: f64, num: Num) -> f32 {
    let proportional = (value.abs() * 0.01) as f32;
    match num {
        Num::F32 | Num::F64 => proportional.max(1e-6),
        _ => proportional.max(0.01),
    }
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
