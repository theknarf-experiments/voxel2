//! Drawing a [`Layout`] with Bevy UI, and moving the camera over it.
//!
//! Everything here is placed ABSOLUTELY from geometry `graph` already
//! decided, so no part of the picture depends on flexbox measuring
//! anything: a node knows where its ports are before it is spawned, which
//! is what makes an edge a straight line between two known points instead
//! of a value read back a frame later.
//!
//! Pan and zoom are one `UiTransform` on the canvas. Bevy's picking
//! backend inverse-transforms the cursor, so a box under a zoomed canvas
//! is still hit where it is drawn — nothing here does that arithmetic.

use bevy::feathers::font_styles::InheritableFont;
use bevy::feathers::theme::{InheritableThemeTextColor, ThemeBackgroundColor, ThemedText};
use bevy::feathers::{constants::fonts, tokens};
use bevy::prelude::*;
use bevy::text::{FontSize, FontWeight, LineBreak, TextLayout};
use bevy::ui::{px, Display, FlexDirection, Node, PositionType, UiRect, UiTransform, Val2};

use crate::graph::{Edge, Frame, GraphStyle, Layout, Placed};
use crate::style::PanelStyle;

/// The pannable, zoomable layer every box sits on.
#[derive(Component, Clone, Default)]
pub struct GraphCanvas;

/// The clipping window the canvas moves inside.
#[derive(Component, Clone, Default)]
pub struct GraphViewport;

/// Where the graph is looked at from.
///
/// In [`crate::EditorState`] rather than on the canvas entity, for the
/// same reason the panel's width is: the panel is respawned whenever the
/// document changes, and a camera kept on the node would snap back to the
/// start of the graph every time a value moved.
#[derive(Clone, Copy, Debug, Reflect, PartialEq)]
pub struct GraphCamera {
    pub pan: Vec2,
    pub zoom: f32,
}

impl Default for GraphCamera {
    fn default() -> Self {
        Self {
            pan: Vec2::splat(16.0),
            zoom: 1.0,
        }
    }
}

/// How far the wheel zooms per notch, and the range it stays in.
const ZOOM_STEP: f32 = 1.12;
const ZOOM_RANGE: std::ops::Range<f32> = 0.2..2.0;

impl GraphCamera {
    /// Zoom by `notches`, holding `at` — a point in the viewport — still.
    ///
    /// Without a fixed point, zooming moves the graph out from under
    /// whatever it was aimed at: the corner stays put and everything the
    /// gesture was about slides away.
    pub fn zoomed(self, notches: f32, at: Option<Vec2>) -> Self {
        // Snapped to whole steps. Zoom changes the layout, so every
        // distinct value respawns the graph; a continuous pinch would do
        // that once a frame for three hundred boxes.
        let step = (self.zoom.log(ZOOM_STEP) + notches).round();
        let zoom = ZOOM_STEP.powf(step).clamp(ZOOM_RANGE.start, ZOOM_RANGE.end);
        let Some(at) = at else {
            return Self { zoom, ..self };
        };
        // The graph point under `at` before and after must be the same:
        //   (at - pan) / zoom == (at - pan') / zoom'
        let pan = at - (at - self.pan) * (zoom / self.zoom);
        Self { pan, zoom }
    }
}

/// The whole graph, as one scene under a clipping viewport.
pub fn scene(
    layout: &Layout,
    graph: &GraphStyle,
    style: &PanelStyle,
    camera: GraphCamera,
) -> impl Scene {
    // Text is sized with the layout, so a zoomed box holds a zoomed label
    // rather than a clipped one.
    let style = &PanelStyle {
        font: style.font * camera.zoom,
        ..style.clone()
    };
    // Frames first, then edges, then boxes: a wire passes behind the node
    // it lands on, and a frame behind everything it gates.
    let frames: Vec<_> = layout
        .frames
        .iter()
        .map(|f| frame(f, graph, style))
        .collect();
    let edges: Vec<_> = layout.edges.iter().map(|e| edge(e, style)).collect();
    let nodes: Vec<_> = layout.nodes.iter().map(|n| node(n, graph, style)).collect();
    let font = FontSize::Px(style.font);
    let pan = camera.pan;

    bsn! {
        GraphViewport
        Node {
            flex_grow: 1.0,
            min_height: px(0),
            // Clipped, not scrolled: the canvas moves under this window.
            overflow: bevy::ui::Overflow::clip(),
        }
        Children [(
            GraphCanvas
            Node {
                position_type: PositionType::Absolute,
                left: px(0),
                top: px(0),
                width: px(0),
                height: px(0),
            }
            // Pan only: the zoom is already in the geometry.
            UiTransform { translation: Val2::px(pan.x, pan.y) }
            InheritableFont {
                font: fonts::MONO,
                font_size: {font},
                weight: FontWeight::NORMAL,
            }
            Children [ {frames}, {edges}, {nodes} ]
        )]
    }
}

/// One node: a title bar, then one row per port.
fn node(placed: &Placed, graph: &GraphStyle, style: &PanelStyle) -> impl Scene {
    let title = match &placed.name {
        Some(name) => format!("{name}  {}", placed.kind),
        None => placed.kind.to_string(),
    };
    let ports: Vec<_> = placed
        .ins
        .iter()
        .map(|p| port(p, true, graph, style))
        .chain(placed.outs.iter().map(|p| port(p, false, graph, style)))
        .collect();
    let (at, size) = (placed.at, placed.size);
    let header = px(graph.header);
    let font = FontSize::Px(style.font * 0.9);

    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(at.x), top: px(at.y),
            width: px(size.x), height: px(size.y),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            border: UiRect::all(px(1)),
        }
        BorderColor::all(Color::srgba(0.0, 0.0, 0.0, 0.6))
        ThemeBackgroundColor(tokens::SUBPANE_BODY_BG)
        Children [
            (
                Node {
                    height: {header},
                    flex_shrink: 0.0,
                    padding: UiRect::horizontal(px(4)),
                    align_items: bevy::ui::AlignItems::Center,
                    overflow: bevy::ui::Overflow::clip(),
                }
                ThemeBackgroundColor(tokens::SUBPANE_HEADER_BG)
                Children [(
                    Text({title})
                    TextFont { font_size: {font} }
                    TextLayout { linebreak: LineBreak::NoWrap }
                    ThemedText
                )]
            ),
            {ports},
        ]
    }
}

/// One port row, its name pushed to the edge the wire arrives at.
fn port(name: &str, input: bool, graph: &GraphStyle, style: &PanelStyle) -> impl Scene {
    let label = name.to_string();
    let height = px(graph.port);
    let font = FontSize::Px(style.font * 0.8);
    let justify = if input {
        bevy::ui::JustifyContent::FlexStart
    } else {
        bevy::ui::JustifyContent::FlexEnd
    };
    bsn! {
        Node {
            height: {height},
            flex_shrink: 0.0,
            padding: UiRect::horizontal(px(4)),
            align_items: bevy::ui::AlignItems::Center,
            justify_content: {justify},
            overflow: bevy::ui::Overflow::clip(),
        }
        Children [(
            Text({label})
            TextFont { font_size: {font} }
            TextLayout { linebreak: LineBreak::NoWrap }
            InheritableThemeTextColor(tokens::TEXT_DIM)
            ThemedText
        )]
    }
}

/// A wire, as a thin bar rotated onto the line between its two ends.
///
/// Bevy's UI has no line primitive and no curves. A straight bar is what
/// `space-wizard-horror` draws through egui's painter too, so nothing is
/// lost but the flourish — and one node per wire is ninety-seven entities
/// on the biggest level this game ships.
fn edge(edge: &Edge, _style: &PanelStyle) -> impl Scene {
    let span = edge.to - edge.from;
    let length = span.length().max(1.0);
    let mid = (edge.from + edge.to) * 0.5;
    // Rot2 turns clockwise and UI y grows downward, so the screen-space
    // angle needs no flip.
    let angle = Rot2::radians(span.y.atan2(span.x));
    let thickness = 1.5;
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(mid.x - length * 0.5),
            top: px(mid.y - thickness * 0.5),
            width: px(length),
            height: px(thickness),
        }
        UiTransform { rotation: {angle} }
        BackgroundColor({Color::srgba(0.45, 0.62, 0.85, 0.75)})
    }
}

/// A scope, as a titled box behind what it gates.
fn frame(frame: &Frame, graph: &GraphStyle, style: &PanelStyle) -> impl Scene {
    let title = match &frame.name {
        Some(name) => format!("{name}  {}", frame.kind),
        None => frame.kind.to_string(),
    };
    let (at, size) = (frame.at, frame.size);
    let header = px(graph.frame_header);
    let font = FontSize::Px(style.font * 0.9);
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(at.x), top: px(at.y),
            width: px(size.x), height: px(size.y),
            flex_direction: FlexDirection::Column,
            border: UiRect::all(px(1)),
        }
        BorderColor::all(Color::srgba(0.55, 0.5, 0.35, 0.7))
        BackgroundColor({Color::srgba(0.5, 0.45, 0.25, 0.07)})
        Children [(
            Node {
                height: {header},
                padding: UiRect::horizontal(px(5)),
                align_items: bevy::ui::AlignItems::Center,
                overflow: bevy::ui::Overflow::clip(),
            }
            Children [(
                Text({title})
                TextFont { font_size: {font} }
                TextLayout { linebreak: LineBreak::NoWrap }
                InheritableThemeTextColor(tokens::TEXT_DIM)
                ThemedText
            )]
        )]
    }
}
