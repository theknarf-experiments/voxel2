//! Drawing a [`Layout`] with Bevy UI, and moving the camera over it.
//!
//! Everything here is placed ABSOLUTELY from geometry `layout` already
//! decided, so no part of the picture depends on flexbox measuring
//! anything: a node knows where its ports are before it is spawned, which
//! is what makes an edge a straight line between two known points instead
//! of a value read back a frame later.
//!
//! Pan and zoom are one `UiTransform` on the canvas. Bevy's picking
//! backend inverse-transforms the cursor, so a box under a zoomed canvas
//! is still hit where it is drawn — nothing here does that arithmetic.
//!
//! The graph is drawn through its own camera into a TEXTURE, shown by a
//! `ViewportNode` — Bevy's widget for exactly this — rather than clipped
//! with `Overflow::clip`. The overflow path adjusts quad corners in
//! screen pixels while the fill and glyph coordinates move in unscaled
//! ones (its own source says it "won't work with rotation/scaling"), so
//! on a zoomed canvas any element straddling the clip edge rendered
//! wrong — a box clipped on the right lost its whole background. A
//! texture has no such arithmetic: whatever falls outside is not in the
//! image, exact at any zoom. The camera targets an image and not the
//! window, which is also what keeps it out of Bevy's default-UI-camera
//! resolution — that fallback is "the highest-order camera on the
//! PRIMARY WINDOW", and a window-target graph camera would win it and
//! adopt the host's entire untargeted UI. [`create_cameras`] does the
//! wiring when a canvas spawns; [`cleanup_cameras`] buries everything
//! when its viewport node dies.

use bevy::camera::RenderTarget;
use bevy::feathers::cursor::EntityCursor;
use bevy::feathers::font_styles::InheritableFont;
use bevy::feathers::theme::{InheritableThemeTextColor, ThemeBackgroundColor, ThemedText};
use bevy::feathers::{constants::fonts, tokens};
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy::text::{FontSize, FontWeight, LineBreak, TextLayout};
use bevy::ui::widget::ViewportNode;
use bevy::ui::{
    px, Display, FlexDirection, GlobalZIndex, Node, PositionType, UiRect, UiTargetCamera,
    UiTransform, Val2,
};

use crate::layout::{Frame, GraphStyle, Layout, Placed, Seg};

/// The pannable, zoomable layer every box sits on.
#[derive(Component, Clone, Default)]
pub struct GraphCanvas;

/// Which node a box inspects when clicked, by [`GraphNode::id`].
///
/// [`GraphNode::id`]: crate::layout::GraphNode::id
#[derive(Component, Clone, Debug, Default)]
pub struct SelectsNode(pub String);

/// The window the canvas is seen through.
///
/// [`create_cameras`] turns it into a `ViewportNode`: the graph renders
/// into a texture this node displays, sized to it by Bevy whenever layout
/// moves it. The texture edge is the clip.
#[derive(Component, Clone, Default)]
pub struct GraphViewport;

/// The camera that draws one viewport's canvas, made by [`create_cameras`].
#[derive(Component)]
pub struct GraphViewCamera {
    /// The [`GraphViewport`] node this camera draws for. When it despawns
    /// — a host rebuilding its UI despawns the whole panel — the camera
    /// and everything it draws go too.
    pub viewport: Entity,
}

/// The click-catcher behind a canvas's boxes.
///
/// The graph's content is picked by the viewport's own POINTER, which
/// `viewport_picking` mirrors the real one into; the real pointer only
/// ever hits the `ViewportNode` itself, on every click, box or not. So
/// "the click landed on empty canvas" must be asked INSIDE the texture:
/// this full-viewport surface is what such a click hits, and a host
/// clears its selection when [`SelectsNode`]-less clicks land here.
#[derive(Component, Clone, Default)]
pub struct GraphBackdrop;

/// The zoom readout in the viewport's lower corner.
///
/// A child of the VIEWPORT, not the canvas: the one thing a zoom label
/// must not do is scale with the thing it describes. [`scene`] bakes the
/// spawn-time value in; the [`zoom_label`] system keeps it current.
#[derive(Component, Clone, Default)]
pub struct ZoomLabel;

/// Where the graph is looked at from.
///
/// A plain value rather than a component, so a host that respawns the
/// graph whenever its document changes can keep the camera somewhere
/// durable — kept on the canvas entity it would snap back to the start of
/// the graph every time a value moved.
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
        // Continuous, not snapped to steps. Zoom is one `UiTransform` and
        // nothing respawns per value — the old comment claiming otherwise
        // was wrong — so there is no cost to following a trackpad pinch
        // smoothly, and snapping made it judder. Worse, the clamp lands
        // OFF the step lattice (2.0 is not a power of 1.12), so at the
        // stop the snap rounded the clamped value DOWN a step: zooming in
        // at 200% read 197%.
        let zoom = (self.zoom * ZOOM_STEP.powf(notches)).clamp(ZOOM_RANGE.start, ZOOM_RANGE.end);
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
///
/// `selected` and `blamed` call out one box each: `selected` by
/// [`GraphNode::id`], `blamed` by name — a document that does not compile
/// has one thing worth looking at, and it is not the one you happened to
/// select.
///
/// The geometry is built once, at 1:1, and the camera scales what is
/// DRAWN. Bevy's UI does not go through a camera projection — `bevy_ui`
/// reads a camera's scale factor and viewport size and nothing else — so
/// the transform on this canvas IS the camera.
///
/// [`GraphNode::id`]: crate::layout::GraphNode::id
pub fn scene(
    layout: &Layout,
    style: &GraphStyle,
    camera: GraphCamera,
    selected: Option<&str>,
    blamed: Option<&str>,
) -> impl Scene {
    // The boxes are drawn at the OVERSAMPLED size the layout was built
    // at; the transform divides it back out, so the user's 100% shows the
    // authored metrics and the user's 200% is the native 1:1 picture —
    // crisp text instead of scaled-up text. The chip is NOT under the
    // transform and keeps the authored font.
    let oversample = style.oversample;
    let chip_font = FontSize::Px(style.font);
    let drawn = style.effective();
    let style = &drawn;
    // Frames first, then edges, then boxes: a wire passes behind the node
    // it lands on, and a frame behind everything it gates.
    let frames: Vec<_> = layout.frames.iter().map(|f| frame(f, style)).collect();
    let edges: Vec<_> = layout
        .edges
        .iter()
        .flat_map(|e| e.segments.iter().copied())
        .map(wire)
        .collect();
    let nodes: Vec<_> = layout
        .nodes
        .iter()
        .map(|n| {
            let mark = if blamed.is_some() && blamed == n.name.as_deref() {
                Mark::Blamed
            } else if selected == Some(n.id.as_str()) {
                Mark::Selected
            } else {
                Mark::None
            };
            node(n, style, mark)
        })
        .collect();
    let font = FontSize::Px(style.font);
    let pan = camera.pan;
    let scale = Vec2::splat(camera.zoom / oversample);
    let zoom = percent(camera.zoom);

    bsn! {
        GraphViewport
        Node {
            flex_grow: 1.0,
            min_height: px(0),
        }
        Children [
            (
                GraphCanvas
                Node {
                    position_type: PositionType::Absolute,
                    left: px(0),
                    top: px(0),
                    width: px(0),
                    height: px(0),
                }
                UiTransform {
                    translation: Val2::px(pan.x, pan.y),
                    scale: {scale},
                }
                InheritableFont {
                    font: fonts::MONO,
                    font_size: {font},
                    weight: FontWeight::NORMAL,
                }
                Children [ {frames}, {edges}, {nodes} ]
            ),
            (
                ZoomLabel
                Text({zoom})
                TextFont { font_size: {chip_font} }
                InheritableThemeTextColor(tokens::TEXT_DIM)
                ThemedText
                Node {
                    position_type: PositionType::Absolute,
                    right: px(6),
                    bottom: px(6),
                    padding: UiRect::axes(px(6), px(2)),
                }
                BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5))
            ),
        ]
    }
}

/// What the label says at `scale`.
fn percent(scale: f32) -> String {
    format!("{:.0}%", scale * 100.0)
}

/// Give every newly spawned canvas a camera and a texture of its own.
///
/// [`scene`] spawns the canvas as a CHILD of the viewport, because a
/// scene cannot name an entity that does not exist yet; this system does
/// the wiring on the frame the canvas appears. The viewport node becomes
/// a `ViewportNode` showing the camera's texture — Bevy keeps the texture
/// sized to the node from here on — and the canvas moves under the
/// camera, along with a fresh [`GraphBackdrop`] for clicks that hit no
/// box. Run it right after whatever spawns the scene: wired in the same
/// schedule pass, a canvas is never drawn un-clipped.
pub fn create_cameras(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    canvases: Query<(Entity, &ChildOf), Added<GraphCanvas>>,
) {
    // The texture is displayed at the node's PHYSICAL size, so the camera
    // must lay its UI out at the window's scale factor or everything in
    // the graph renders at half size on a hidpi display.
    let scale_factor = windows.single().map(|w| w.scale_factor()).unwrap_or(1.0);
    for (canvas, seat) in &canvases {
        let viewport = seat.parent();
        // 1x1 is a placeholder: `update_viewport_render_target_size`
        // resizes it to the node the moment layout knows the node.
        let target = images.add(Image::new_target_texture(
            1,
            1,
            TextureFormat::Bgra8UnormSrgb,
            None,
        ));
        let camera = commands
            .spawn((
                Camera2d,
                Camera {
                    // Before the host's cameras: the texture must be
                    // drawn before the frame that samples it.
                    order: -1,
                    // A transparent texture, so the panel behind the
                    // viewport node shows through as the graph's
                    // background, whatever the host's theme says.
                    clear_color: bevy::camera::ClearColorConfig::Custom(Color::NONE),
                    ..Default::default()
                },
                RenderTarget::Image(bevy::camera::ImageRenderTarget {
                    handle: target,
                    scale_factor,
                }),
                bevy::render::view::Msaa::Off,
                GraphViewCamera { viewport },
            ))
            .id();
        commands.entity(viewport).insert(ViewportNode::new(camera));
        commands
            .entity(canvas)
            .remove::<ChildOf>()
            .insert(UiTargetCamera(camera));
        commands.spawn((
            GraphBackdrop,
            Node {
                position_type: PositionType::Absolute,
                left: px(0),
                top: px(0),
                width: bevy::ui::percent(100),
                height: bevy::ui::percent(100),
                ..Default::default()
            },
            UiTargetCamera(camera),
            // Behind the canvas, so a box always wins the pick.
            GlobalZIndex(-1),
        ));
    }
}

/// Bury a canvas whose viewport node has died.
///
/// The canvas stopped being the viewport's descendant when
/// [`create_cameras`] moved it under the camera, so a host despawning its
/// panel no longer takes the graph with it — this does. The camera's
/// texture goes when the camera does: the render target holds the only
/// strong handle.
pub fn cleanup_cameras(
    mut commands: Commands,
    cameras: Query<(Entity, &GraphViewCamera)>,
    viewports: Query<(), With<GraphViewport>>,
    drawn: Query<(Entity, &UiTargetCamera)>,
) {
    for (camera, of) in &cameras {
        if viewports.contains(of.viewport) {
            continue;
        }
        for (orphan, target) in &drawn {
            if target.entity() == camera {
                commands.entity(orphan).despawn();
            }
        }
        commands.entity(camera).despawn();
    }
}

/// Keep each [`ZoomLabel`] agreeing with its own canvas.
///
/// It reads the canvas's `UiTransform` rather than any camera type,
/// because the transform is the one place the zoom is true regardless of
/// where the host keeps its camera — a host that writes the transform
/// directly (the cheap way to pan) updates the label for nothing. Not
/// scheduled by [`crate::GraphViewPlugin`], for the same reason [`hover`]
/// is not: a gated host puts it under its own run condition.
///
/// The label stayed a child of the viewport when the canvas moved under
/// the camera, so the pairing goes canvas → camera → viewport → label.
/// The transform's scale carries the oversample division, so the USER's
/// zoom — what the label owes them — is scale times oversample.
pub fn zoom_label(
    style: Res<GraphStyle>,
    canvases: Query<ZoomedCanvas, ZoomedCanvasFilter>,
    cameras: Query<&GraphViewCamera>,
    mut labels: Query<(&mut Text, &ChildOf), With<ZoomLabel>>,
) {
    for (transform, drawn_by) in &canvases {
        let Ok(camera) = cameras.get(drawn_by.entity()) else {
            continue;
        };
        for (mut text, seat) in &mut labels {
            if seat.parent() == camera.viewport {
                text.0 = percent(transform.scale.x * style.oversample);
            }
        }
    }
}

/// A canvas whose transform has just changed, with the camera drawing it.
type ZoomedCanvas = (&'static UiTransform, &'static UiTargetCamera);
type ZoomedCanvasFilter = (With<GraphCanvas>, Changed<UiTransform>);

/// The border a box wears when it is none of the interesting things, and
/// what [`hover`] puts back.
pub const PLAIN_BORDER: Color = Color::srgba(0.0, 0.0, 0.0, 0.6);

/// Bright enough to find, dim enough not to compete with a selection.
const HOVER_BORDER: Color = Color::srgba(0.55, 0.60, 0.70, 0.9);

/// Light a node box while the pointer is over it.
///
/// Written straight onto the border rather than by respawning: a host
/// rebuilds the graph from its DOCUMENT, and moving a pointer changes no
/// document. A box that is blamed or selected keeps the border it has —
/// those say something a hover does not.
pub fn hover(mut boxes: Query<HoveredBox, HoveredBoxFilter>) {
    for (hovered, mut border) in &mut boxes {
        let plain = border.left == PLAIN_BORDER;
        let lit = border.left == HOVER_BORDER;
        if hovered.get() && plain {
            *border = BorderColor::all(HOVER_BORDER);
        } else if !hovered.get() && lit {
            *border = BorderColor::all(PLAIN_BORDER);
        }
    }
}

/// A node box whose hover state has just changed.
type HoveredBox = (&'static Hovered, &'static mut BorderColor);
type HoveredBoxFilter = (Changed<Hovered>, With<SelectsNode>);

/// How a box is called out, if it is.
#[derive(Clone, Copy, PartialEq)]
enum Mark {
    None,
    Selected,
    /// Named by the host's complaint. Beats selection: a document that
    /// does not compile has one thing worth looking at.
    Blamed,
}

impl Mark {
    pub fn border(self) -> Color {
        match self {
            Mark::Blamed => Color::srgb(0.85, 0.25, 0.25),
            Mark::Selected => Color::srgb(0.30, 0.55, 0.92),
            Mark::None => PLAIN_BORDER,
        }
    }
}

/// One node: a title bar, then one row per port.
fn node(placed: &Placed, style: &GraphStyle, mark: Mark) -> impl Scene {
    let title = match &placed.name {
        Some(name) => format!("{name}  {}", placed.kind),
        None => placed.kind.clone(),
    };
    let ports: Vec<_> = placed
        .ins
        .iter()
        .map(|p| port(p, true, style))
        .chain(placed.outs.iter().map(|p| port(p, false, style)))
        .collect();
    let (at, size) = (placed.at, placed.size);
    let header = px(style.header);
    let font = FontSize::Px(style.font * 0.9);
    let id = placed.id.clone();
    let border = mark.border();

    bsn! {
        SelectsNode({id})
        // A box is a thing you click. The cursor says so before you try,
        // and `Hovered` is what [`hover`] reads to light the border.
        EntityCursor::System(bevy::window::SystemCursorIcon::Pointer)
        Hovered
        Node {
            position_type: PositionType::Absolute,
            left: px(at.x), top: px(at.y),
            width: px(size.x), height: px(size.y),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            border: UiRect::all(px(1)),
        }
        BorderColor::all(border)
        ThemeBackgroundColor(tokens::SUBPANE_BODY_BG)
        Children [
            (
                Node {
                    height: {header},
                    flex_shrink: 0.0,
                    padding: UiRect::horizontal(px(4)),
                    align_items: bevy::ui::AlignItems::Center,
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
fn port(name: &str, input: bool, style: &GraphStyle) -> impl Scene {
    let label = name.to_string();
    let height = px(style.port);
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

/// One run of a wire: an ordinary axis-aligned box.
///
/// Not a rotated bar. Bevy has no line primitive, and a rotated node is
/// clipped against an axis-aligned rect that does not follow the rotation,
/// which broke every near-vertical wire into moving dashes — see
/// [`crate::layout::Edge`].
fn wire(seg: Seg) -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(seg.at.x),
            top: px(seg.at.y),
            width: px(seg.size.x),
            height: px(seg.size.y),
        }
        BackgroundColor(Color::srgba(0.45, 0.62, 0.85, 0.75))
    }
}

/// A scope, as a titled box behind what it gates.
fn frame(frame: &Frame, style: &GraphStyle) -> impl Scene {
    let title = match &frame.name {
        Some(name) => format!("{name}  {}", frame.kind),
        None => frame.kind.clone(),
    };
    let (at, size) = (frame.at, frame.size);
    let header = px(style.frame_header);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The label tracks the canvas whose camera points at its viewport,
    /// live, and leaves another viewport's label alone. It reports the
    /// USER's zoom: the transform's scale times the style's oversample.
    #[test]
    fn the_label_follows_its_own_canvas() {
        let mut app = App::new();
        // The default style's oversample of 2: a transform at 0.7 is the
        // user's 140%.
        app.init_resource::<GraphStyle>();
        app.add_systems(Update, zoom_label);
        let world = app.world_mut();
        let viewport = world.spawn(GraphViewport).id();
        let camera = world.spawn(GraphViewCamera { viewport }).id();
        let canvas = world
            .spawn((
                GraphCanvas,
                UiTransform {
                    scale: Vec2::splat(0.7),
                    ..Default::default()
                },
                UiTargetCamera(camera),
            ))
            .id();
        let label = world
            .spawn((ZoomLabel, Text("100%".to_string()), ChildOf(viewport)))
            .id();
        let other_viewport = world.spawn(GraphViewport).id();
        let other = world
            .spawn((ZoomLabel, Text("100%".to_string()), ChildOf(other_viewport)))
            .id();

        // A fresh spawn counts as changed, so the first frame settles it.
        app.update();
        assert_eq!(app.world().get::<Text>(label).unwrap().0, "140%");
        assert_eq!(
            app.world().get::<Text>(other).unwrap().0,
            "100%",
            "a label without a canvas keeps what it was spawned saying"
        );

        // And a live zoom moves it without a respawn.
        app.world_mut()
            .get_mut::<UiTransform>(canvas)
            .unwrap()
            .scale = Vec2::splat(0.4);
        app.update();
        assert_eq!(app.world().get::<Text>(label).unwrap().0, "80%");
    }

    /// A canvas spawned under a viewport gets a camera and a texture, the
    /// viewport becomes the node that shows it — and when the viewport
    /// dies, the camera and everything it drew die with it, because the
    /// canvas stopped being the viewport's descendant at the handover.
    #[test]
    fn a_canvas_moves_behind_glass_and_is_buried_with_its_viewport() {
        let mut app = App::new();
        app.insert_resource(Assets::<Image>::default());
        app.add_systems(Update, (create_cameras, cleanup_cameras).chain());
        let world = app.world_mut();
        let viewport = world.spawn((GraphViewport, Node::default())).id();
        let canvas = world
            .spawn((GraphCanvas, Node::default(), ChildOf(viewport)))
            .id();

        app.update();
        let world = app.world_mut();
        let camera = world
            .get::<UiTargetCamera>(canvas)
            .expect("the canvas is drawn by the new camera")
            .entity();
        assert!(
            world.get::<ChildOf>(canvas).is_none(),
            "the canvas is a root now, not the viewport's child"
        );
        assert_eq!(
            world.get::<ViewportNode>(viewport).and_then(|v| v.camera),
            Some(camera),
            "the viewport node shows what the camera draws"
        );
        assert_eq!(
            world.get::<GraphViewCamera>(camera).map(|c| c.viewport),
            Some(viewport),
            "the camera knows whose viewport it draws for"
        );
        assert!(
            matches!(
                world.get::<RenderTarget>(camera),
                Some(RenderTarget::Image(_))
            ),
            "the camera draws to a texture, not the window — a window\
             camera would win Bevy's default-UI-camera fallback"
        );
        let backdrop = world
            .query_filtered::<Entity, With<GraphBackdrop>>()
            .single(world)
            .expect("one backdrop per canvas");

        world.entity_mut(viewport).despawn();
        app.update();
        let world = app.world_mut();
        for (gone, name) in [
            (canvas, "canvas"),
            (backdrop, "backdrop"),
            (camera, "camera"),
        ] {
            assert!(
                world.get_entity(gone).is_err(),
                "the {name} outlived its viewport"
            );
        }
    }
}
