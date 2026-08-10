//! One [`Row`] as one Feathers scene.
//!
//! A row is two lines, not three columns: the value line, and under it the
//! field's own documentation, dimmed and smaller. As a third column the
//! docs squeezed the values into a strip and still showed only a clause;
//! under the value they have the panel's whole width, and the panel is
//! resizable now, so widening it reveals more of the sentence.
//!
//! Every row carries [`FieldPath`], which is what makes editing one
//! observer rather than one per field.

use bevy::feathers::constants::{fonts, size};
use bevy::feathers::controls::{
    ButtonVariant, ColorSwatchValue, FeathersButton, FeathersCheckbox, FeathersColorSwatch,
    FeathersDisclosureToggle, FeathersMenu, FeathersMenuButton, FeathersMenuItem,
    FeathersMenuPopup, FeathersSlider,
};
use bevy::feathers::cursor::EntityCursor;
use bevy::feathers::font_styles::InheritableFont;
use bevy::feathers::theme::{InheritableThemeTextColor, ThemedText};
use bevy::feathers::tokens;
use bevy::prelude::*;
use bevy::text::{FontSize, FontWeight, LineBreak, TextLayout};
use bevy::ui::{px, AlignItems, Checked, Display, FlexDirection, Node, Overflow, UiRect};
use bevy::ui_widgets::{SliderPrecision, SliderStep};

use crate::style::PanelStyle;
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

/// A number the pointer can drag.
///
/// `from` is the value the row was BUILT with, and the drag reports its
/// total distance, so the new value is one multiplication rather than an
/// accumulation that drifts — and the panel is frozen while a drag is in
/// flight, so `from` cannot go stale under it.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct DragsNum {
    pub from: f32,
    pub speed: f32,
}

/// A menu item that writes one of a reference's options.
///
/// The option is carried on the ITEM rather than looked up by index when
/// it is picked: the panel is respawned whenever the document changes, so
/// an index into a list that has since been rebuilt is the one thing a
/// menu must not hold.
#[derive(Component, Clone, Debug, Default)]
pub struct PicksOption {
    pub path: String,
    pub value: String,
    /// The field's numeric type, when the reference is spelled as a number
    /// — a material id is a `u32` and a prefab name is a `String`, and the
    /// menu that offers them is the same menu.
    pub num: Option<Num>,
    /// The value NAMES a variant of the enum at `path` rather than being
    /// one it can hold. Picking it changes what the row contains.
    pub variant: bool,
}

pub fn scene(row: &Row, style: &PanelStyle) -> impl Scene {
    let path = row.path.clone();
    let indent = px(style.indent * row.depth as f32);
    let label = row.label.clone();
    let (gap, label_w) = (px(style.gap), px(style.label));

    bsn! {
        FieldPath({path})
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            // Rows must not shrink. A column of forty in a fixed-height
            // pane shrinks every one below its text and they overlap into
            // an unreadable stack — flexbox doing exactly what it is told,
            // which looks like a rendering bug.
            flex_shrink: 0.0,
        }
        Children [
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: {gap},
                    // Feathers' own controls are `ROW_HEIGHT`; a shorter
                    // line clips a slider.
                    height: {size::ROW_HEIGHT},
                    flex_shrink: 0.0,
                }
                Children [
                    // The NAME column holds the nesting: the indent and the
                    // disclosure are inside it, and it is a fixed width. So
                    // the value column starts at the same x on every row
                    // however deep it sits — an inspector reads down its
                    // values, and a value column that stepped right with
                    // depth cannot be read down at all.
                    (
                        Node {
                            width: {label_w},
                            height: {size::ROW_HEIGHT},
                            flex_shrink: 0.0,
                            display: Display::Flex,
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            column_gap: {gap},
                            padding: UiRect::left(indent),
                            overflow: Overflow::clip(),
                        }
                        Children [
                            {gutter(row, style)},
                            {vec![text(label, style.font)]},
                        ]
                    ),
                    {value_widget(row, style)},
                ]
            ),
            {doc_line(row.docs, style.indent * row.depth as f32, style)},
        ]
    }
}

/// A dropdown of what a field may hold: the level's own answers, or an
/// enum's own variants.
fn menu(
    current: &str,
    options: &[String],
    path: &str,
    num: Option<Num>,
    variant: bool,
    style: &PanelStyle,
) -> Box<dyn SceneList> {
    let items: Vec<_> = options
        .iter()
        .map(|option| {
            let (path, value) = (path.to_string(), option.clone());
            let caption = Box::new(vec![text(option.clone(), style.font)]) as Box<dyn SceneList>;
            bsn! {
                @FeathersMenuItem { @caption: {caption} }
                PicksOption { path: {path}, value: {value}, num: {num}, variant: {variant} }
            }
        })
        .collect();
    let caption = Box::new(vec![text(current.to_string(), style.font)]) as Box<dyn SceneList>;
    Box::new(bsn_list!(
        @FeathersMenu
        Node { width: {px(style.value)}, flex_shrink: 0.0 }
        Children [
            (
                @FeathersMenuButton { @caption: {caption} }
                Node { flex_grow: 1.0, min_width: px(0), height: px(style.tab) }
            ),
            (@FeathersMenuPopup Children [ {items} ]),
        ]
    ))
}

/// The disclosure column: a chevron for a container, empty for a leaf.
///
/// Always present and always the same width, so the label and value
/// columns start at the same x on every row.
fn gutter(row: &Row, style: &PanelStyle) -> Box<dyn SceneList> {
    let expanded = match row.kind {
        RowKind::Group { expanded, .. } | RowKind::Variant { expanded, .. } => expanded,
        _ => {
            let w = px(style.gutter);
            return Box::new(bsn_list!(Node {
                width: { w },
                flex_shrink: 0.0
            }));
        }
    };
    let toggles = row.path.clone();
    let w = px(style.gutter);
    if expanded {
        Box::new(bsn_list!(
            Node {
                width: {w},
                height: {size::ROW_HEIGHT},
                flex_shrink: 0.0,
                display: Display::Flex,
                align_items: AlignItems::Center,
            }
            Children [(@FeathersDisclosureToggle Checked TogglesPath({toggles}))]
        ))
    } else {
        Box::new(bsn_list!(
            Node {
                width: {w},
                height: {size::ROW_HEIGHT},
                flex_shrink: 0.0,
                display: Display::Flex,
                align_items: AlignItems::Center,
            }
            Children [(@FeathersDisclosureToggle TogglesPath({toggles}))]
        ))
    }
}

/// A fixed-width cell of the value line, with its text centred.
///
/// Explicit height and centring on every cell rather than auto-sizing:
/// a disclosure toggle is 12px and a line of 13px text is about 15, so
/// leaving them to `align_items` alone let the value column sit half a row
/// below its own label.
fn cell(width: f32, inner: impl Scene) -> impl Scene {
    bsn! {
        Node {
            width: px(width),
            height: {size::ROW_HEIGHT},
            flex_shrink: 0.0,
            display: Display::Flex,
            align_items: AlignItems::Center,
            overflow: Overflow::clip(),
        }
        Children [ {vec![inner]} ]
    }
}

/// Panel text, at the size [`PanelStyle`] asks for.
///
/// Set on the span rather than inherited from the body's
/// `InheritableFont`. The cascade IS there — it is what dims and shrinks
/// the documentation line, whose `InheritableFont` sits directly above its
/// own span — but it did not reach through the row / line / cell nodes
/// between the body and these spans, and text that silently stayed at
/// Bevy's 20px default is worse than a size stated where it applies. Still
/// one number, still in `PanelStyle`.
fn text(s: String, size: f32) -> impl Scene {
    bsn! {
        Text({s})
        TextFont { font_size: {FontSize::Px(size)} }
        // Never wrapped: a cell is one line of a grid, and a long label
        // that wrapped pushed its own row out of alignment with every
        // other one. The cell clips; the panel resizes.
        TextLayout { linebreak: LineBreak::NoWrap }
        ThemedText
    }
}

/// The rationale written above the field, under the value it explains.
///
/// Not truncated and wrapped: the whole sentence, on as many lines as it
/// needs, indented to start under the label rather than under the
/// disclosure gutter.
fn doc_line(docs: Option<&'static str>, indent: f32, style: &PanelStyle) -> Box<dyn SceneList> {
    let Some(docs) = docs else {
        return Box::new(bsn_list!());
    };
    let flat = docs
        .split('\n')
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    if flat.is_empty() {
        return Box::new(bsn_list!());
    }
    let font = FontSize::Px(style.doc_font);
    let doc_indent = px(indent + style.gutter + style.gap);
    Box::new(bsn_list!(
        // No height and no clipping: the sentence wraps to as many lines
        // as it needs, and the row grows to hold them. The value line
        // above it stays a fixed-height grid either way.
        Node { flex_shrink: 0.0, padding: UiRect::left(doc_indent) }
        // Smaller and dimmer than the value it explains, said once here
        // and inherited by the span — not set on the text itself.
        InheritableFont {
            font: fonts::MONO,
            font_size: {font},
            weight: FontWeight::NORMAL,
        }
        InheritableThemeTextColor(tokens::TEXT_DIM)
        Children [( Text({flat}) ThemedText )]
    ))
}

/// `bsn!` builds a fixed shape; the value side of a row is a different
/// widget per kind, so it comes back boxed. `Vec<Box<dyn SceneList>>` is
/// what makes a reflection-driven panel expressible in a static macro.
fn value_widget(row: &Row, style: &PanelStyle) -> Box<dyn SceneList> {
    // `FieldPath` goes on the WIDGET as well as the row: a `ValueChange`
    // fires on the checkbox or slider, which is a child, and an observer
    // that looked only at the row would never find the path.
    let p = row.path.clone();
    let hold = row.rebuilds;
    match &row.kind {
        RowKind::Group { summary, .. } => {
            Box::new(vec![cell(style.value, text(summary.clone(), style.font))])
        }
        RowKind::Variant {
            current, options, ..
        } => {
            // A dynamic value knows the variant it holds but not its
            // siblings; better an inert row than a menu of nothing.
            if options.is_empty() {
                return Box::new(vec![cell(style.value, text(current.clone(), style.font))]);
            }
            menu(current, options, &row.path, None, true, style)
        }

        RowKind::Number {
            value,
            num,
            range,
            speed,
        } => match range {
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
                        Node { width: px(style.value), flex_shrink: 0.0 }
                        SliderStep({step}) SliderPrecision({digits})
                        WritesNum({writes}) FieldPath({p}) CommitOnRelease
                    ))
                } else {
                    Box::new(bsn_list!(
                        @FeathersSlider { @value: {v}, @min: {lo}, @max: {hi} }
                        Node { width: px(style.value), flex_shrink: 0.0 }
                        SliderStep({step}) SliderPrecision({digits})
                        WritesNum({writes}) FieldPath({p})
                    ))
                }
            }
            // No bounds to slide between, so it DRAGS: press and move
            // sideways. The number is still shown as the level writes it.
            //
            // Not `FeathersNumberInput`, which displayed `14` as `4` and
            // `2.5` as `5` — it keeps its text in a child it manages, and
            // only the last character survived. A tool opened to check a
            // number must not be the thing that gets it wrong.
            None => {
                let shown = format_num(*value, *num);
                let (writes, from, speed) = (*num, *value as f32, *speed);
                if hold {
                    Box::new(bsn_list!(
                        Node {
                            width: {px(style.value)},
                            flex_shrink: 0.0,
                            justify_content: bevy::ui::JustifyContent::FlexEnd,
                            align_items: AlignItems::Center,
                            overflow: Overflow::clip(),
                        }
                        EntityCursor::System(bevy::window::SystemCursorIcon::EwResize)
                        DragsNum { from: {from}, speed: {speed} } WritesNum({writes}) FieldPath({p}) CommitOnRelease
                        Children [ {vec![text(shown, style.font)]} ]
                    ))
                } else {
                    Box::new(bsn_list!(
                        Node {
                            width: {px(style.value)},
                            flex_shrink: 0.0,
                            justify_content: bevy::ui::JustifyContent::FlexEnd,
                            align_items: AlignItems::Center,
                            overflow: Overflow::clip(),
                        }
                        EntityCursor::System(bevy::window::SystemCursorIcon::EwResize)
                        DragsNum { from: {from}, speed: {speed} } WritesNum({writes}) FieldPath({p})
                        Children [ {vec![text(shown, style.font)]} ]
                    ))
                }
            }
        },

        RowKind::Bool(v) => match (*v, hold) {
            (true, true) => {
                Box::new(bsn_list!(@FeathersCheckbox Checked FieldPath({p}) CommitOnRelease))
            }
            (true, false) => Box::new(bsn_list!(@FeathersCheckbox Checked FieldPath({p}))),
            (false, true) => Box::new(bsn_list!(@FeathersCheckbox FieldPath({p}) CommitOnRelease)),
            (false, false) => Box::new(bsn_list!(@FeathersCheckbox FieldPath({p}))),
        },

        RowKind::Text(s) => {
            let s = s.clone();
            Box::new(vec![cell(style.value, self::text(s, style.font))])
        }

        RowKind::Color(rgb) => {
            let value = Color::linear_rgb(rgb[0], rgb[1], rgb[2]);
            Box::new(bsn_list!(
                @FeathersColorSwatch
                Node { width: px(style.value), flex_shrink: 0.0 }
                ColorSwatchValue({value})
            ))
        }

        RowKind::Choice {
            current,
            options,
            num,
        } => {
            // An empty option list is a level that defines none of that
            // thing, or a pattern that points nowhere. Both are worth
            // seeing in the row rather than as an inert menu.
            if options.is_empty() {
                let text = format!("{current}  (nothing to refer to)");
                return Box::new(vec![cell(style.value, self::text(text, style.font))]);
            }
            let items: Vec<_> = options
                .iter()
                .map(|option| {
                    let (path, value, num) = (row.path.clone(), option.clone(), *num);
                    let caption = Box::new(vec![self::text(option.clone(), style.font)])
                        as Box<dyn SceneList>;
                    bsn! {
                        @FeathersMenuItem { @caption: {caption} }
                        PicksOption { path: {path}, value: {value}, num: {num} }
                    }
                })
                .collect();
            let caption =
                Box::new(vec![self::text(current.clone(), style.font)]) as Box<dyn SceneList>;
            Box::new(bsn_list!(
                @FeathersMenu
                Node { width: {px(style.value)}, flex_shrink: 0.0 }
                Children [
                    (
                        @FeathersMenuButton { @caption: {caption} }
                        Node { flex_grow: 1.0, min_width: px(0), height: px(style.tab) }
                    ),
                    (@FeathersMenuPopup Children [ {items} ]),
                ]
            ))
        }

        RowKind::Unsupported(type_path) => {
            // Shown greyed rather than omitted. A panel admitting it has no
            // widget for a type is a bug report; a row that silently
            // vanishes reads as a field that does not exist.
            let text = format!("no widget for {type_path}");
            Box::new(vec![cell(style.value, self::text(text, style.font))])
        }
    }
}

/// Which tab a button selects. On the button, not the row: the tab strip
/// is not part of the document.
#[derive(Component, Clone, Debug, Default)]
pub struct SelectsRoot(pub usize);

/// One tab. The active one is `Primary`; the rest are `Plain`, so the
/// strip reads as a strip rather than as a row of buttons.
pub fn tab(index: usize, label: &str, active: bool, style: &PanelStyle) -> impl Scene {
    let caption = Box::new(vec![text(label.to_string(), style.font)]) as Box<dyn SceneList>;
    let variant = if active {
        ButtonVariant::Primary
    } else {
        ButtonVariant::Plain
    };
    bsn! {
        @FeathersButton {
            @caption: {caption},
            @variant: {variant},
        }
        SelectsRoot({index})
        Node { flex_grow: 1.0, min_width: px(0), height: px(style.tab) }
    }
}
