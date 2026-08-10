//! One [`Row`] as one Feathers scene.
//!
//! Every row carries [`FieldPath`], which is what makes editing one
//! observer rather than one per field.

use bevy::feathers::constants::size;
use bevy::feathers::controls::{
    ColorSwatchValue, FeathersCheckbox, FeathersColorSwatch, FeathersDisclosureToggle,
    FeathersMenuButton, FeathersSlider,
};
use bevy::feathers::theme::ThemedText;
use bevy::prelude::*;
use bevy::text::{FontSize, LineBreak, TextLayout};
use bevy::ui::{px, AlignItems, Checked, Display, FlexDirection, Node, Overflow, UiRect};
use bevy::ui_widgets::{SliderPrecision, SliderStep};

use crate::walk::{format_num, Num, Row, RowKind};

/// The reflect path this row edits, from its document's root.
///
/// `.field` and `[index]` are BRP's own syntax, so most of these are
/// exactly what `world.mutate_resources` would take; `{key}` extends it to
/// map entries, which BRP cannot address.
#[derive(Component, Clone, Debug, Default)]
pub struct FieldPath(pub String);

/// A container row whose disclosure opens and closes `path`.
#[derive(Component, Clone, Debug, Default)]
pub struct TogglesPath(pub String);

/// How a written-back value has to be typed.
///
/// Kept beside the widget rather than inferred from it: every numeric
/// widget deals in `f32`, and a `u32` field assigned an `f32` is refused
/// by `try_apply` rather than rounded.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct WritesNum(pub Num);

/// This widget's field restreams the world, so only its FINAL value is
/// applied.
///
/// Dragging a slider emits a value per frame; rebuilding a streamed world
/// at that rate makes the drag useless and the change invisible. Which
/// fields those are is the level's own declaration
/// (`voxel_engine::schema::Rebuilds`), never a guess made here.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct CommitOnRelease;


/// Indent per nesting level. The rows are a flat list, so this is the only
/// thing that says what contains what.
const INDENT: f32 = 14.0;

/// Docs are the field's whole rationale, several sentences of it. A row
/// has space for the first clause, and only if it does not wrap: a level
/// is a long list of short rows, and three-line rows turn it into a page
/// of prose with the values hidden in it.
const TIP_CHARS: usize = 48;

pub fn scene(row: &Row) -> impl Scene {
    let path = row.path.clone();
    let indent = px(INDENT * row.depth as f32 + 4.0);
    let label = row.label.clone();
    let tip = tip(row.docs);

    bsn! {
        FieldPath({path})
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(8),
            padding: UiRect::left(indent),
            min_height: size::ROW_HEIGHT,
            // Rows must not shrink. A column of forty in a fixed-height
            // pane shrinks every one of them below its text and they
            // overlap into an unreadable stack — flexbox doing exactly
            // what it is told, which looks like a rendering bug.
            flex_shrink: 0.0,
        }
        Children [
            (Node { width: px(150), flex_shrink: 0.0 } {text(label)}),
            {value_widget(row)},
            (
                Node { flex_grow: 1.0, flex_basis: px(0), overflow: Overflow::clip() }
                {text(tip)}
                TextLayout { linebreak: LineBreak::NoWrap }
            ),
        ]
    }
}

/// The first clause of the rationale written above the field.
///
/// Straight off `NamedField::docs()` — the editor never restates a label
/// or an explanation that the type already carries.
fn tip(docs: Option<&'static str>) -> String {
    let Some(docs) = docs else {
        return String::new();
    };
    let flat = docs
        .split('\n')
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    if flat.chars().count() <= TIP_CHARS {
        return flat;
    }
    let cut: String = flat.chars().take(TIP_CHARS).collect();
    match cut.rsplit_once(' ') {
        Some((head, _)) => format!("{head}…"),
        None => format!("{cut}…"),
    }
}

/// Panel text at the panel's size.
///
/// Stated here rather than inherited: Feathers propagates `TextFont` from
/// an ancestor `InheritableFont`, and a row is spawned as its own scene
/// under a container this crate does not own the font of.
fn text(s: String) -> impl Scene {
    bsn! {
        Text({s})
        TextFont { font_size: {FontSize::Px(13.0)} }
        ThemedText
    }
}

/// `bsn!` builds a fixed shape; the value side of a row is a different
/// widget per kind, so it comes back boxed. `Vec<Box<dyn SceneList>>` is
/// what makes a reflection-driven panel expressible in a static macro.
fn value_widget(row: &Row) -> Box<dyn SceneList> {
    // `FieldPath` goes on the WIDGET as well as the row: a `ValueChange`
    // fires on the checkbox or slider, which is a child, and an observer
    // that looked only at the row would never find the path.
    let p = row.path.clone();
    let hold = row.rebuilds;
    match &row.kind {
        RowKind::Group { expanded, summary } => {
            disclosure(&row.path, *expanded, summary.clone())
        }
        RowKind::Variant {
            expanded, current, ..
        } => disclosure(&row.path, *expanded, current.clone()),

        RowKind::Number { value, num, range } => match range {
            Some(r) => {
                let (v, lo, hi) = (*value as f32, r.0, r.1);
                let writes = *num;
                // `SliderPrecision` and `SliderStep` are NOT part of the
                // Feathers slider scene, and `update_slider_pos` queries
                // for the former — without it the fill never moves and the
                // label keeps the literal placeholder the scene ships,
                // which is the string "10.0". Four different fields all
                // reading 10.0 is what that looks like.
                let (step, digits) = match writes {
                    Num::F32 | Num::F64 => ((hi - lo) / 100.0, 3),
                    _ => (1.0, 0),
                };
                if hold {
                    Box::new(bsn_list!(
                        @FeathersSlider { @value: {v}, @min: {lo}, @max: {hi} }
                        Node { width: px(150) }
                        SliderStep({step}) SliderPrecision({digits})
                        WritesNum({writes}) FieldPath({p}) CommitOnRelease
                    ))
                } else {
                    Box::new(bsn_list!(
                        @FeathersSlider { @value: {v}, @min: {lo}, @max: {hi} }
                        Node { width: px(150) }
                        SliderStep({step}) SliderPrecision({digits})
                        WritesNum({writes}) FieldPath({p})
                    ))
                }
            }
            // Always `CommitOnRelease`. A number input emits a
            // non-final `ValueChange` for every text change — including
            // the one that gives it its initial value, and including each
            // keystroke of a number being typed. Taking those would write
            // `1` on the way to `14` and rebuild the panel out from under
            // the cursor. Enter and focus-loss are the final ones.
            // Shown, not yet edited in the panel.
            //
            // `FeathersNumberInput` displayed `14` as `4` and `2.5` as
            // `5` — it keeps its text in a child it manages, and only the
            // last character survived. Rendered beside our own text the
            // two disagreed in the same row. A tool opened to check a
            // number must not be the thing that gets it wrong, so the
            // control is gone and the value stands on its own until there
            // is one that can be trusted. These fields are still reachable
            // by file edit and by `world.mutate_resources`.
            None => {
                let shown = format_num(*value, *num);
                Box::new(bsn_list!(
                    Node { width: px(150) }
                    {self::text(shown)}
                ))
            }
        },

        RowKind::Bool(v) => match (*v, hold) {
            (true, true) => Box::new(bsn_list!(@FeathersCheckbox Checked FieldPath({p}) CommitOnRelease)),
            (true, false) => Box::new(bsn_list!(@FeathersCheckbox Checked FieldPath({p}))),
            (false, true) => Box::new(bsn_list!(@FeathersCheckbox FieldPath({p}) CommitOnRelease)),
            (false, false) => Box::new(bsn_list!(@FeathersCheckbox FieldPath({p}))),
        },

        RowKind::Text(s) => {
            let s = s.clone();
            Box::new(bsn_list!(Node { width: px(150) } {self::text(s)}))
        }

        RowKind::Color(rgb) => {
            let value = Color::linear_rgb(rgb[0], rgb[1], rgb[2]);
            Box::new(bsn_list!(
                @FeathersColorSwatch
                Node { width: px(150) }
                ColorSwatchValue({value})
            ))
        }

        RowKind::Choice {
            current, options, ..
        } => {
            // An empty option list is a level that defines none of that
            // thing, or a pattern that points nowhere. Both are worth
            // seeing in the row rather than as an inert menu.
            let text = if options.is_empty() {
                format!("{current}  (nothing to refer to)")
            } else {
                format!("{current}  ({} available)", options.len())
            };
            Box::new(bsn_list!(
                @FeathersMenuButton {
                    @caption: {Box::new(vec![self::text(text)]) as Box<dyn SceneList>},
                }
                Node { width: px(150) }
            ))
        }

        RowKind::Unsupported(type_path) => {
            // Shown greyed rather than omitted. A panel admitting it has no
            // widget for a type is a bug report; a row that silently
            // vanishes reads as a field that does not exist.
            let text = format!("no widget for {type_path}");
            Box::new(bsn_list!(Node { width: px(150) } {self::text(text)}))
        }
    }
}

fn disclosure(path: &str, expanded: bool, summary: String) -> Box<dyn SceneList> {
    let toggles = path.to_string();
    if expanded {
        Box::new(bsn_list!(
            (@FeathersDisclosureToggle Checked TogglesPath({toggles})),
            (Node { width: px(130) } {text(summary)}),
        ))
    } else {
        Box::new(bsn_list!(
            (@FeathersDisclosureToggle TogglesPath({toggles})),
            (Node { width: px(130) } {text(summary)}),
        ))
    }
}
