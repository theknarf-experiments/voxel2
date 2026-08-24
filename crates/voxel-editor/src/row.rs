//! One [`Row`] as one Feathers scene.
//!
//! A row is two lines, not three columns: the value line, and under it the
//! field's own documentation, dimmed and smaller. As a third column the
//! docs squeezed the values into a strip and still showed only a clause;
//! under the value they have the panel's whole width, and the panel is
//! resizable now, so widening it reveals more of the sentence.
//!
//! Every row carries [`FieldPath`], which is what makes editing one
//! observer rather than one per field. What a row can be told to become
//! is the widget's business: a box to type into for a name and for a
//! number alike, a slider beside it where the field declares bounds, a
//! scrub on the LABEL where it does not, a menu of the level's own answers
//! for a reference, a menu of an enum's own variants. A kind with no
//! widget renders as a greyed line saying so — a row that silently
//! vanished would read as a field that does not exist.

use bevy::feathers::constants::{fonts, size};
use bevy::feathers::controls::{
    ButtonVariant, ColorSwatchValue, FeathersButton, FeathersCheckbox, FeathersColorSwatch,
    FeathersDisclosureToggle, FeathersMenu, FeathersMenuButton, FeathersMenuItem,
    FeathersMenuPopup, FeathersSlider, FeathersTextInput, FeathersTextInputContainer,
};
use bevy::feathers::cursor::EntityCursor;
use bevy::feathers::font_styles::InheritableFont;
use bevy::feathers::theme::{InheritableThemeTextColor, ThemedText};
use bevy::feathers::tokens;
use bevy::prelude::*;
use bevy::text::EditableText;
use bevy::text::{FontSize, FontWeight, LineBreak, TextLayout};
use bevy::ui::{px, AlignItems, Checked, Display, FlexDirection, Node, Overflow, UiRect};
use bevy::ui_widgets::{SliderPrecision, SliderStep};

use crate::style::PanelStyle;
use crate::walk::{format_num, Num, Row, RowKind};

/// The number a row's LABEL scrubs, if it scrubs one.
///
/// The label rather than the value cell, because the value cell is now the
/// box you type in and a pointer cannot mean both at once: dragging inside
/// a text field selects text. Only numbers with no declared bounds — a
/// bounded one has a slider, which is what a sweep is for.
fn scrub(row: &Row) -> Option<(f32, f32, Num)> {
    match row.kind {
        RowKind::Number {
            value,
            num,
            range: None,
            speed,
        } => Some((value as f32, speed, num)),
        _ => None,
    }
}

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
    let gap = px(style.gap);

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
                    {label_column(row, style)},
                    {value_widget(row, style)},
                ]
            ),
            {doc_line(row.docs, style.indent * row.depth as f32, style)},
        ]
    }
}

/// A box the value is TYPED into — the one answer to typed input, for a
/// name and a number alike.
///
/// Seeded with what the document holds, and read back when the edit is
/// COMMITTED — never per keystroke: a name is a reference and half a name
/// refers to nothing, and half a number is a value the world would be
/// rebuilt for.
///
/// `num` is the whole difference between the two: it is what makes
/// `edit::on_typed` parse the text at the field's type instead of writing
/// the text itself, so a box without one can only ever hold a string.
fn typed_box(
    value: String,
    path: String,
    num: Option<Num>,
    width: f32,
    style: &PanelStyle,
) -> Box<dyn SceneList> {
    let (width, height) = (px(width), px(style.tab));
    match num {
        Some(num) => Box::new(bsn_list!(
            @FeathersTextInputContainer
            Node { width: {width}, flex_shrink: 0.0, height: {height} }
            FieldPath({path}) WritesNum({num})
            Children [( @FeathersTextInput EditableText::new(value) )]
        )),
        None => Box::new(bsn_list!(
            @FeathersTextInputContainer
            Node { width: {width}, flex_shrink: 0.0, height: {height} }
            FieldPath({path})
            Children [( @FeathersTextInput EditableText::new(value) )]
        )),
    }
}

/// How a bounded number splits its control column: the slider's track,
/// and the box beside it if there is room for one.
///
/// Both widgets have a floor. Below [`PanelStyle::track_min`] a slider is
/// not something a pointer can sweep, and below
/// [`PanelStyle::number_min`] the box shows a prefix of its own number.
/// The graph inspector's column is a third of a 300px panel, which does
/// not hold `number` and a usable track at once — an unclamped
/// `value - number - gap` left it 39px there, and any narrower column
/// asks for a NEGATIVE width, which no layout can give back.
///
/// The box is what goes when only one fits: a Feathers slider prints the
/// value on its own track, so the row keeps its number either way and
/// loses only exact entry.
fn number_columns(style: &PanelStyle) -> (f32, Option<f32>) {
    let value = style.value.max(0.0);
    let room = value - style.track_min - style.gap;
    if room < style.number_min {
        return (value, None);
    }
    let box_w = style.number.min(room);
    (value - box_w - style.gap, Some(box_w))
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

/// The name column, and on a plain number the surface that SCRUBS it.
///
/// The drag components go on a wrapper rather than on the cell, so the
/// cell stays written once: a pointer event bubbles, so a drag anywhere in
/// the name reaches the wrapper whichever of the two it started on.
fn label_column(row: &Row, style: &PanelStyle) -> Box<dyn SceneList> {
    let Some((from, speed, writes)) = scrub(row) else {
        return Box::new(vec![label_cell(row, style)]);
    };
    let p = row.path.clone();
    let inner = vec![label_cell(row, style)];
    // A field that restreams the world waits for the pointer to be let go
    // — see `CommitOnRelease`.
    if row.rebuilds {
        Box::new(bsn_list!(
            Node {
                height: {size::ROW_HEIGHT},
                flex_shrink: 0.0,
                display: Display::Flex,
                align_items: AlignItems::Center,
            }
            EntityCursor::System(bevy::window::SystemCursorIcon::EwResize)
            DragsNum { from: {from}, speed: {speed} } WritesNum({writes}) FieldPath({p}) CommitOnRelease
            Children [ {inner} ]
        ))
    } else {
        Box::new(bsn_list!(
            Node {
                height: {size::ROW_HEIGHT},
                flex_shrink: 0.0,
                display: Display::Flex,
                align_items: AlignItems::Center,
            }
            EntityCursor::System(bevy::window::SystemCursorIcon::EwResize)
            DragsNum { from: {from}, speed: {speed} } WritesNum({writes}) FieldPath({p})
            Children [ {inner} ]
        ))
    }
}

/// The name itself: the indent and the disclosure inside a FIXED width, so
/// the value column starts at the same x on every row however deep it sits
/// — an inspector reads down its values, and a value column that stepped
/// right with depth cannot be read down at all.
fn label_cell(row: &Row, style: &PanelStyle) -> impl Scene {
    let indent = px(style.indent * row.depth as f32);
    let (gap, label_w) = (px(style.gap), px(style.label));
    let label = row.label.clone();
    bsn! {
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
    }
}

/// The disclosure column: a chevron for a container, empty for a leaf.
///
/// Always present and always the same width, so the label and value
/// columns start at the same x on every row.
fn gutter(row: &Row, style: &PanelStyle) -> Box<dyn SceneList> {
    let expanded = match row.kind {
        RowKind::Group { expanded, .. }
        | RowKind::Variant { expanded, .. }
        | RowKind::Color { expanded, .. } => expanded,
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

        // Typed, always: a drag moves by a percent of the value it starts
        // from, so a field left at its default of 0 is hundreds of
        // millions of pixels away from 800 and every node created from
        // scratch would be unreachable. The label scrubs it — see
        // `label_column` — and where the field declares bounds a slider
        // sweeps them.
        //
        // Not `FeathersNumberInput`, which displayed `14` as `4` and `2.5`
        // as `5` — it keeps its text in a child it manages, and only the
        // last character survived. A tool opened to check a number must
        // not be the thing that gets it wrong.
        RowKind::Number {
            value, num, range, ..
        } => {
            let shown = format_num(*value, *num);
            let writes = *num;
            let Some(r) = range else {
                return typed_box(shown, p, Some(writes), style.value, style);
            };
            let (v, lo, hi) = (*value as f32, r.0, r.1);
            // `SliderPrecision` and `SliderStep` are NOT part of the
            // Feathers slider scene, and `update_slider_pos` queries for
            // the former — without it the fill never moves and the label
            // keeps the literal placeholder the scene ships, which is the
            // string "10.0". Four different fields all reading 10.0 is
            // what that looks like.
            let (step, digits) = match writes {
                Num::F32 | Num::F64 => ((hi - lo) / 100.0, 3),
                _ => (1.0, 0),
            };
            // A slider quantises to its hundred steps, so a bounded field
            // needs the box too: 0.437 is not reachable by sweeping. The
            // column is not always wide enough for both, and where it is
            // not the BOX goes: a Feathers slider prints its own value, so
            // dropping the slider would cost the row its number as well as
            // its sweep.
            let (track, box_w) = number_columns(style);
            let (gap, boxed) = match box_w {
                Some(box_w) => (
                    px(style.gap),
                    typed_box(shown, p.clone(), Some(writes), box_w, style),
                ),
                None => (px(0.0), Box::new(bsn_list!()) as Box<dyn SceneList>),
            };
            let slider: Box<dyn SceneList> = if hold {
                Box::new(bsn_list!(
                    @FeathersSlider { @value: {v}, @min: {lo}, @max: {hi} }
                    Node { width: px(track), flex_shrink: 0.0 }
                    SliderStep({step}) SliderPrecision({digits})
                    WritesNum({writes}) FieldPath({p}) CommitOnRelease
                ))
            } else {
                Box::new(bsn_list!(
                    @FeathersSlider { @value: {v}, @min: {lo}, @max: {hi} }
                    Node { width: px(track), flex_shrink: 0.0 }
                    SliderStep({step}) SliderPrecision({digits})
                    WritesNum({writes}) FieldPath({p})
                ))
            };
            Box::new(bsn_list!(
                Node {
                    width: {px(style.value)},
                    flex_shrink: 0.0,
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: {gap},
                }
                Children [ {slider}, {boxed} ]
            ))
        }

        RowKind::Bool(v) => match (*v, hold) {
            (true, true) => {
                Box::new(bsn_list!(@FeathersCheckbox Checked FieldPath({p}) CommitOnRelease))
            }
            (true, false) => Box::new(bsn_list!(@FeathersCheckbox Checked FieldPath({p}))),
            (false, true) => Box::new(bsn_list!(@FeathersCheckbox FieldPath({p}) CommitOnRelease)),
            (false, false) => Box::new(bsn_list!(@FeathersCheckbox FieldPath({p}))),
        },

        RowKind::Text(s) => typed_box(s.clone(), p, None, style.value, style),

        RowKind::Color { rgb, .. } => {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The control column is split between a track and a box, and neither
    /// may be squeezed below the width that makes it a widget at all —
    /// least of all below ZERO, which is not a width.
    ///
    /// The graph inspector's column is the case that made this: a third of
    /// a 300px panel, which an unclamped `value - number - gap` left a
    /// 39px track.
    #[test]
    fn a_narrow_control_column_never_makes_a_useless_widget() {
        let default = PanelStyle::default();
        let split = |value: f32| {
            number_columns(&PanelStyle {
                value,
                ..default.clone()
            })
        };

        // Wide enough for both: the box is the width it asks for.
        let (track, boxed) = split(default.value);
        assert_eq!(boxed, Some(default.number));
        assert_eq!(track + default.number + default.gap, default.value);

        // The graph inspector, where the box is squeezed rather than the
        // track being made unusable.
        let inspector = default.inspector * 0.35;
        let (track, boxed) = split(inspector);
        let boxed = boxed.expect("a 105px column still holds both");
        assert!(track >= default.track_min, "a track that can be swept");
        assert!(boxed >= default.number_min, "beside a readable number");
        assert_eq!(track + boxed + default.gap, inspector);

        // Narrower than the two floors together: the box goes and the
        // slider — which prints its own value — takes the column.
        for value in [90.0, 40.0, 0.0] {
            let (track, boxed) = split(value);
            assert_eq!(boxed, None, "no room for a box at {value}px");
            assert_eq!(track, value);
        }
    }
}
