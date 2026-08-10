//! Panel metrics, in one declarative place.
//!
//! Feathers' theme layer — `UiTheme` plus `ThemeToken`, which behave like
//! CSS custom properties, and `InheritableFont` /
//! `InheritableThemeTextColor`, which behave like inheritance — covers
//! COLOUR and TEXT only. `ThemeProps` has one field and says as much:
//!
//! ```ignore
//! pub struct ThemeProps {
//!     pub color: HashMap<ThemeToken, Color>,
//!     // Other style property types to be added later.
//! }
//! ```
//!
//! So colours and fonts are themed (see `panel` and `row`, which name
//! tokens and never a literal colour), and the SIZES that have no token
//! yet live here rather than as constants scattered through the widgets.
//! Reflected, so a host can restyle the panel and `voxctl` can poke it,
//! which is the nearest thing to a stylesheet 0.19 offers for metrics.
//!
//! Anything Feathers already defines is taken from `constants::size`
//! rather than restated: a row is `ROW_HEIGHT`, body text is
//! `COMPACT_FONT`, documentation is `EXTRA_SMALL_FONT`.

use bevy::prelude::*;

#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource)]
pub struct PanelStyle {
    /// Left offset per level of nesting. The rows are a flat list, so this
    /// is the only thing that says what contains what.
    pub indent: f32,
    /// The name column: indent, disclosure and field name together. Fixed,
    /// so the value column does not step right with nesting depth.
    pub label: f32,
    /// Control column: slider, swatch, value or menu.
    pub value: f32,
    /// The disclosure column, ahead of the label. Constant whether or not
    /// a row has a chevron, so the columns after it always align.
    pub gutter: f32,
    /// Between the columns of a row.
    pub gap: f32,
    /// Between rows.
    pub row_gap: f32,
    /// Body text size. A font size is no more themable than a margin in
    /// 0.19 — Feathers keeps its own as plain constants — so it sits here
    /// with the rest of the metrics, and can be tuned on a running panel.
    pub font: f32,
    /// Documentation text size, under the value it explains.
    pub doc_font: f32,
    /// The strip along the panel's inner edge that resizes it.
    pub grip: f32,
    /// How far the panel may be dragged.
    pub width: std::ops::Range<f32>,
    /// A wheel "line" in pixels. Bevy reports lines on some platforms and
    /// pixels on others; both have to move the list comparably.
    pub wheel_line: f32,
    /// Padding inside the scrolling body.
    pub pad: f32,
    /// The properties column beside the graph.
    pub inspector: f32,
}

impl Default for PanelStyle {
    fn default() -> Self {
        Self {
            indent: 14.0,
            label: 230.0,
            value: 150.0,
            gutter: 16.0,
            gap: 8.0,
            row_gap: 2.0,
            font: 10.0,
            doc_font: 9.0,
            grip: 5.0,
            width: 320.0..1400.0,
            wheel_line: 24.0,
            pad: 6.0,
            inspector: 300.0,
        }
    }
}
