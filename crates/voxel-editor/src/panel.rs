//! The panel: open it, and respawn its rows whenever what they show has
//! changed.
//!
//! Bevy's UI is retained and the document is not, so the panel is rebuilt
//! rather than diffed. Rebuilding is affordable because the walk only
//! descends into expanded paths — the rows on screen, not the several
//! thousand a level contains — and it is CORRECT, which diffing a tree
//! whose shape changes under it (a list gains an item, an enum changes
//! variant) is not without more machinery than the rebuild costs.

use bevy::feathers::constants::fonts;
use bevy::feathers::containers::{pane, pane_body};
use bevy::feathers::cursor::EntityCursor;
use bevy::feathers::font_styles::InheritableFont;
use bevy::feathers::theme::{ThemeBackgroundColor, ThemedText};
use bevy::feathers::tokens;
use bevy::prelude::*;
use bevy::text::FontWeight;
use bevy::ui::{
    percent, px, Display, FlexDirection, Node, Overflow, OverflowAxis, PositionType,
    ScrollPosition, UiRect, UiTransform, Val2,
};
use bevy::window::SystemCursorIcon;

use crate::style::PanelStyle;
use crate::{canvas, graph, row, walk};
use crate::{EditorRoots, EditorState, View, TOGGLE_KEY};

/// The panel's root entity. One at a time.
#[derive(Component)]
pub struct EditorPanel;

/// The scrolling list of rows.
#[derive(Component, Clone, Default)]
pub struct EditorBody;

/// The strip along the panel's inner edge that resizes it.
#[derive(Component, Clone, Default)]
pub struct ResizeGrip;

/// Drag the panel's inner edge.
///
/// The width lives in [`EditorState`] rather than on the node, because the
/// panel is respawned whenever the document changes and a width kept only
/// on the node would snap back to the default mid-drag.
pub fn on_grip_drag(
    drag: On<Pointer<Drag>>,
    grips: Query<(), With<ResizeGrip>>,
    style: Res<PanelStyle>,
    mut state: ResMut<EditorState>,
) {
    if !grips.contains(drag.event_target()) {
        return;
    }
    // Pinned right: dragging the grip LEFT makes the panel wider.
    state.width = (state.width - drag.delta.x).clamp(style.width.start, style.width.end);
}

/// Apply the camera to the live canvas.
///
/// Directly, for the same reason the width is: a rebuild per drag frame
/// would throw away every box in the graph to move the picture a pixel.
pub fn apply_camera(
    state: Res<EditorState>,
    mut canvases: Query<&mut UiTransform, With<canvas::GraphCanvas>>,
) {
    if !state.is_changed() {
        return;
    }
    let scale = Vec2::splat(state.camera.zoom);
    for mut transform in &mut canvases {
        transform.translation = Val2::px(state.camera.pan.x, state.camera.pan.y);
        transform.scale = scale;
    }
}

/// Apply the width to the live panel.
///
/// Directly, not by respawning: a rebuild per drag frame would throw away
/// and rebuild every row forty times a second to move an edge.
pub fn apply_width(
    state: Res<EditorState>,
    shown: Option<Res<Shown>>,
    mut nodes: Query<&mut Node>,
) {
    if !state.is_changed() {
        return;
    }
    let Some(shown) = shown else { return };
    if let Ok(mut node) = nodes.get_mut(shown.entity) {
        node.width = px(state.width);
    }
}

/// Two-finger scroll: down the list, or across the graph.
///
/// The graph PANS rather than scrolls, on both axes, because a graph has
/// no reading direction — and because the mouse drag that would otherwise
/// do it is reserved for dragging nodes.
pub fn on_wheel(
    mut wheel: MessageReader<bevy::input::mouse::MouseWheel>,
    windows: Query<&Window>,
    roots: Res<EditorRoots>,
    style: Res<PanelStyle>,
    mut state: ResMut<EditorState>,
    mut bodies: Query<&mut ScrollPosition, With<EditorBody>>,
) {
    let Ok(window) = windows.single() else { return };
    // Only when the pointer is over the panel — the panel is pinned to the
    // right edge, so that is one comparison rather than a hit test.
    let over = window
        .cursor_position()
        .is_some_and(|c| c.x >= window.width() - state.width);
    if !over {
        wheel.clear();
        return;
    }
    let mut delta = Vec2::ZERO;
    for ev in wheel.read() {
        let scale = match ev.unit {
            bevy::input::mouse::MouseScrollUnit::Line => style.wheel_line,
            bevy::input::mouse::MouseScrollUnit::Pixel => 1.0,
        };
        delta += Vec2::new(ev.x, ev.y) * scale;
    }
    if delta == Vec2::ZERO {
        return;
    }
    // Over the properties column, the wheel scrolls it; over the graph it
    // pans. The column is a fixed width at the panel's inner edge, so this
    // is one comparison, like the panel's own.
    let over_props = window
        .cursor_position()
        .is_some_and(|c| c.x >= window.width() - style.inspector);
    if graph_view(&roots, state.root) && !over_props {
        // The gesture is in screen pixels and so is the pan: the zoom is
        // already in the geometry underneath it.
        state.camera.pan += delta;
        return;
    }
    for mut scroll in &mut bodies {
        scroll.0.y -= delta.y;
    }
}

/// Pinch to zoom the graph.
///
/// macOS and iOS only — Bevy reports the gesture nowhere else — so a
/// platform without it keeps whatever zoom it was left at rather than
/// losing the view entirely.
pub fn on_pinch(
    mut pinch: MessageReader<bevy::input::gestures::PinchGesture>,
    windows: Query<&Window>,
    viewports: Query<(&ComputedNode, &UiGlobalTransform), With<canvas::GraphViewport>>,
    roots: Res<EditorRoots>,
    mut state: ResMut<EditorState>,
) {
    let notches: f32 = pinch.read().map(|p| p.0).sum();
    if notches == 0.0 || !graph_view(&roots, state.root) {
        return;
    }
    // About the POINTER, not the corner. Zooming about the origin slides
    // whatever you were looking at off the screen, which reads as the
    // graph running away from the gesture.
    let at = windows
        .single()
        .ok()
        .and_then(|w| w.cursor_position())
        .zip(viewports.single().ok())
        .map(|(cursor, (node, transform))| {
            // `UiGlobalTransform` is the node's CENTRE in physical pixels
            // and the cursor is in logical ones, so the viewport's
            // top-left has to come back through the scale factor.
            let top_left = (transform.translation - node.size() * 0.5) * node.inverse_scale_factor;
            cursor - top_left
        });
    state.camera = state.camera.zoomed(notches * PINCH_NOTCHES, at);
}

/// A pinch reports a fraction of the screen; a zoom notch is coarser than
/// that or a small gesture does nothing at all.
const PINCH_NOTCHES: f32 = 12.0;

/// Is the showing tab a graph?
fn graph_view(roots: &EditorRoots, root: usize) -> bool {
    roots.0.get(root).map(|r| r.view) == Some(View::Graph)
}

/// What the rows on screen were built from.
///
/// The panel respawns when this stops matching, so it has to be everything
/// the rows depend on: which document, what is open in it, and whether the
/// document itself changed.
#[derive(Resource, PartialEq)]
pub struct Shown {
    root: usize,
    expanded: Vec<String>,
    /// Which node the properties column is showing.
    selected: Option<String>,
    /// The change tick of the document resource the rows were read from.
    tick: u32,
    /// The panel entity, so finding it again is not a fresh `QueryState`
    /// built and matched against every archetype in the world — of which a
    /// streaming voxel world has a great many.
    entity: Entity,
}

/// A widget is being dragged, so the panel must hold still.
///
/// Every edit changes the document, and the panel is respawned when the
/// document changes — which would DESPAWN the very widget under the
/// pointer, ending the drag on its first frame. So a drag freezes the
/// rebuild and the panel catches up when the pointer is released. This is
/// why the ranged sliders could not be dragged either.
#[derive(Resource, Default)]
pub struct Dragging(pub bool);

pub fn on_drag_start(_: On<Pointer<DragStart>>, mut dragging: ResMut<Dragging>) {
    dragging.0 = true;
}

pub fn on_drag_end(_: On<Pointer<DragEnd>>, mut dragging: ResMut<Dragging>) {
    dragging.0 = false;
}

/// Is there anything for the panel systems to do?
///
/// A closed dev tool has to cost NOTHING. These systems are exclusive, so
/// running them every frame splits the schedule at two more sync points,
/// and the frames that costs come out of the streamer's budget: `fly`
/// exhausted the mesh slabs on planet with the panel shut, because chunks
/// are released per frame and there were fewer frames to do it in.
pub fn active(state: Res<EditorState>, shown: Option<Res<Shown>>) -> bool {
    state.open || shown.is_some()
}

pub fn toggle(keys: Res<ButtonInput<KeyCode>>, mut state: ResMut<EditorState>) {
    if keys.just_pressed(TOGGLE_KEY) {
        state.open = !state.open;
    }
}

/// The panel's own keys: save, and undo.
///
/// Only while the panel is OPEN: a level editor that wrote the file on a
/// stray keystroke in a game window would be a trap.
pub fn save(
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<EditorState>,
    mut asked: MessageWriter<crate::SaveRequested>,
) {
    let held = |k: [KeyCode; 2]| k.iter().any(|k| keys.pressed(*k));
    let modifier = held([KeyCode::SuperLeft, KeyCode::SuperRight])
        || held([KeyCode::ControlLeft, KeyCode::ControlRight]);
    if !state.open {
        return;
    }
    if keys.just_pressed(KeyCode::KeyZ) && modifier {
        // Shift-Z is the other direction, as everywhere else.
        let shifted = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
        if shifted {
            state.redo = true;
        } else {
            state.undo = true;
        }
    }
    if !(keys.just_pressed(KeyCode::KeyS) && modifier || state.save) {
        return;
    }
    state.save = false;
    asked.write(crate::SaveRequested);
}

/// Inspect the node that was clicked, or clear the selection when the
/// click lands on the canvas behind them.
pub fn on_select(
    click: On<Pointer<Click>>,
    boxes: Query<&canvas::SelectsNode>,
    viewports: Query<(), With<canvas::GraphViewport>>,
    mut state: ResMut<EditorState>,
) {
    // A pointer event BUBBLES, so this observer runs once per ancestor: a
    // click on a node's title bar reaches the title, then the box, then
    // the canvas, then the viewport. Selecting on the way up is right —
    // that is how a click on any part of a box selects it — but clearing
    // has to ask where the click STARTED, or the viewport at the end of
    // the chain undoes what the box just did. It did exactly that.
    if let Ok(canvas::SelectsNode(path)) = boxes.get(click.event_target()) {
        state.selected = Some(path.clone());
    } else if viewports.contains(click.original_event_target()) {
        state.selected = None;
    }
}

/// Switch to the tab that was clicked.
///
/// The panel respawns from [`EditorState`], so nothing else has to happen:
/// the strip is rendered from `root` and cannot disagree with what is
/// shown below it.
pub fn on_tab(
    activate: On<bevy::ui_widgets::Activate>,
    tabs: Query<&row::SelectsRoot>,
    mut state: ResMut<EditorState>,
) {
    if let Ok(row::SelectsRoot(index)) = tabs.get(activate.event_target()) {
        state.root = *index;
    }
}

/// Open or close whatever a clicked disclosure names.
///
/// The widget is not told to update itself: the panel respawns from the
/// expansion set, so the set is the only state and the checkbox's own
/// `Checked` is rendered from it. Retained UI and a rebuilt panel agree
/// exactly when nothing is stored in both.
pub fn on_disclosure(
    change: On<bevy::ui_widgets::ValueChange<bool>>,
    toggles: Query<&row::TogglesPath>,
    mut state: ResMut<EditorState>,
) {
    let Ok(row::TogglesPath(path)) = toggles.get(change.event_target()) else {
        return;
    };
    if change.value {
        state.expanded.insert(path.clone());
    } else {
        state.expanded.remove(path);
    }
}

/// Respawn the panel when what it shows has changed.
///
/// An exclusive system because reading an arbitrary reflected resource
/// needs the whole `World` and the type registry at once — the price of
/// the panel not knowing what it is showing.
pub fn rebuild(world: &mut World) {
    // Not while a widget is being dragged: respawning the panel would
    // despawn the thing the pointer is holding.
    if world.resource::<Dragging>().0 {
        return;
    }
    let open = world.resource::<EditorState>().open;
    if !open {
        if let Some(shown) = world.remove_resource::<Shown>() {
            world.entity_mut(shown.entity).despawn();
        }
        return;
    }

    let Some(Document {
        tick,
        complaint,
        body,
        props,
    }) = read_document(world)
    else {
        return;
    };

    let state = world.resource::<EditorState>();
    let mut expanded: Vec<String> = state.expanded.iter().cloned().collect();
    expanded.sort();
    let mut wanted = Shown {
        root: state.root,
        expanded,
        selected: state.selected.clone(),
        tick,
        entity: Entity::PLACEHOLDER,
    };
    // The entity is an OUTPUT of this rebuild, not part of what decides
    // whether one is needed.
    if let Some(shown) = world.get_resource::<Shown>() {
        wanted.entity = shown.entity;
        if *shown == wanted {
            return;
        }
        world.entity_mut(shown.entity).despawn();
    }
    // Only where there is a choice: a single-document app should not grow
    // a strip with one thing in it.
    let roots = world.resource::<EditorRoots>().0.clone();
    let current = world.resource::<EditorState>().root;
    let width = px(world.resource::<EditorState>().width);
    let style = world.resource::<PanelStyle>().clone();
    let (grip, row_gap) = (px(style.grip), px(style.row_gap));
    // `impl SceneList for Vec<S: Scene>` is what lets a panel whose shape
    // is only known at runtime be expressed in a static macro.
    let blamed = complaint.as_ref().map(|(at, _)| at.clone());
    let body_scene: Box<dyn SceneList> = match &body {
        Body::Rows(rows) => Box::new(rows_body(rows, &style)),
        Body::Graph(layout) => {
            let camera = world.resource::<EditorState>().camera;
            let selected = world.resource::<EditorState>().selected.clone();
            let graph_style = world.resource::<graph::GraphStyle>().clone();
            let canvas = bsn_list! {(
                {canvas::scene(
                    layout,
                    &graph_style,
                    &style,
                    camera,
                    selected.as_deref(),
                    blamed.as_deref(),
                )}
            )};
            let inspector = inspector(&props, &style);
            Box::new(bsn_list! {(
                Node {
                    flex_grow: 1.0,
                    min_height: px(0),
                    flex_direction: FlexDirection::Row,
                }
                Children [ {canvas}, {inspector} ]
            )})
        }
    };
    let complaint_scene = complaint_bar(complaint.as_ref().map(|(_, said)| said.as_str()), &style);
    let tab_scenes: Vec<_> = if roots.len() > 1 {
        roots
            .iter()
            .enumerate()
            .map(|(i, r)| row::tab(i, &r.label, i == current, &style))
            .collect()
    } else {
        Vec::new()
    };
    let tabs = if tab_scenes.is_empty() {
        Display::None
    } else {
        Display::Flex
    };
    let panel = world
        .spawn_scene(bsn! {
            // Docked to the right edge, full height. The debug HUD lives
            // in the top-left corner and the two used to overlap.
            //
            // A ROW, so the resize grip can sit on the panel's inner edge:
            // the panel is pinned right, so its left edge is the only one
            // that can move.
            Node {
                position_type: PositionType::Absolute,
                right: px(0),
                top: px(0),
                bottom: px(0),
                width: {width},
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
            }
            Children [
                (
                    ResizeGrip
                    Node { width: {grip}, height: percent(100), flex_shrink: 0.0 }
                    ThemeBackgroundColor(tokens::PANE_HEADER_BORDER)
                    EntityCursor::System(SystemCursorIcon::EwResize)
                ),
                (
                    pane()
                    Node { flex_grow: 1.0, min_width: px(0), flex_direction: FlexDirection::Column }
                    Children [
                        (
                            Node {
                                display: {tabs},
                                flex_direction: FlexDirection::Row,
                                column_gap: {row_gap},
                                padding: UiRect::all(px(2)),
                                flex_shrink: 0.0,
                            }
                            // Part of the header, not of the document: an
                            // unthemed strip shows whatever is behind it.
                            ThemeBackgroundColor(tokens::PANE_HEADER_BG)
                            Children [ {tab_scenes} ]
                        ),
                        {complaint_scene},
                        (pane_body() Node { flex_grow: 1.0, min_height: px(0), flex_direction: FlexDirection::Column } Children [ {body_scene} ]),
                    ]
                ),
            ]
        })
        .map(|e| e.id());
    match panel {
        Ok(panel) => {
            world.entity_mut(panel).insert(EditorPanel);
            wanted.entity = panel;
            world.insert_resource(wanted);
        }
        // A scene that fails to resolve is a bug in this crate, not in the
        // level; keeping the old `Shown` makes it retry rather than sit
        // silently on an empty panel.
        Err(e) => warn_once!("editor panel did not spawn: {e}"),
    }
}

/// What one tab's document turned into.
struct Document {
    tick: u32,
    /// Why the document does not compile, if it does not. The panel can
    /// now MAKE a level invalid — one menu pick rewires a port to a node
    /// declared later — and the engine's answer to that is to keep the
    /// running world and warn. A warning in a log is not an answer to
    /// somebody looking at the panel that caused it.
    complaint: Option<(String, String)>,
    body: Body,
    /// The selected node's own rows, for the graph view's properties
    /// column. Empty when nothing is selected.
    props: Option<(String, Vec<walk::Row>)>,
}

/// The two things a tab can be.
enum Body {
    Rows(Vec<walk::Row>),
    Graph(graph::Layout),
}

/// What the graph view is inspecting, if anything.
fn state_selected(world: &World) -> Option<String> {
    world.resource::<EditorState>().selected.clone()
}

/// The properties column beside the graph, or NOTHING when no node is
/// selected — a column explaining that it is empty is still a column.
///
/// The same rows the list view builds, at the same paths: a value edited
/// here is edited in the document, not in a copy of the selection.
fn inspector(props: &Option<(String, Vec<walk::Row>)>, style: &PanelStyle) -> impl SceneList {
    let Some((title, rows)) = props else {
        return None;
    };
    let width = px(style.inspector);
    // Its own columns: the list view's are sized for a panel three times
    // this wide, and a value column that does not fit is a value you
    // cannot read.
    let style = &PanelStyle {
        label: style.inspector * 0.45,
        value: style.inspector * 0.35,
        ..style.clone()
    };
    let header = self::rows_header(title.clone(), style);
    let body = rows_body(rows, style);
    Some(bsn_list! {(
        Node {
            width: {width},
            flex_shrink: 0.0,
            flex_direction: FlexDirection::Column,
            border: UiRect::left(px(1)),
        }
        BorderColor::all(Color::srgba(0.0, 0.0, 0.0, 0.5))
        ThemeBackgroundColor(tokens::PANE_HEADER_BG)
        Children [ {header}, {body} ]
    )})
}

/// What the compiler says is wrong, along the top of the panel.
///
/// Absent when there is nothing to say, so the panel does not carry a
/// permanent empty strip for the sake of the rare moment it is needed.
fn complaint_bar(complaint: Option<&str>, style: &PanelStyle) -> impl SceneList {
    let said = complaint?.to_string();
    let font = bevy::text::FontSize::Px(style.font);
    let pad = px(style.pad);
    Some(bsn_list! {(
        Node {
            padding: UiRect::all(pad),
            flex_shrink: 0.0,
        }
        BackgroundColor(Color::srgb(0.42, 0.13, 0.13))
        Children [(
            Text({said})
            TextFont { font_size: {font} }
            ThemedText
        )]
    )})
}

/// The properties column's own title bar.
fn rows_header(title: String, style: &PanelStyle) -> impl SceneList {
    let font = bevy::text::FontSize::Px(style.font);
    let pad = px(style.pad);
    bsn_list! {(
        Node {
            padding: UiRect::all(pad),
            flex_shrink: 0.0,
            overflow: Overflow::clip(),
        }
        Children [(
            Text({title})
            TextFont { font_size: {font} }
            bevy::text::TextLayout { linebreak: bevy::text::LineBreak::NoWrap }
            ThemedText
        )]
    )}
}

/// The scrolling list of rows.
fn rows_body(rows: &[walk::Row], style: &PanelStyle) -> impl SceneList {
    let scenes: Vec<_> = rows.iter().map(|r| row::scene(r, style)).collect();
    let (pad, row_gap) = (px(style.pad), px(style.row_gap));
    let body_font = bevy::text::FontSize::Px(style.font);
    bsn_list! {(
        EditorBody
        ScrollPosition
        // Feathers propagates `TextFont` only from an ancestor carrying
        // this; without it every row rendered at Bevy's default 20px and a
        // level's worth of them was three screens tall.
        InheritableFont {
            font: fonts::MONO,
            font_size: {body_font},
            weight: FontWeight::NORMAL,
        }
        Node {
            // The body takes the rest of the height and scrolls, or a
            // level with an open section runs off the bottom of the screen.
            flex_grow: 1.0,
            min_height: px(0),
            flex_direction: FlexDirection::Column,
            row_gap: {row_gap},
            padding: UiRect::all(pad),
            // Clip across, scroll down: a long doc line must not paint over
            // the world beside the panel.
            overflow: Overflow { x: OverflowAxis::Clip, y: OverflowAxis::Scroll },
        }
        Children [ {scenes} ]
    )}
}

/// Read the selected document and walk it into rows.
///
/// A resource is an entity in Bevy 0.19 and `ReflectResource` is only a
/// marker implying `ReflectComponent`, so this reads the document off that
/// entity — no raw pointer and no `unsafe` to reach a resource whose type
/// is not known until runtime.
///
/// Returns `None` when the root is not a registered reflected resource.
/// Reported, because a tab that is silently empty is exactly the failure
/// this crate exists to stop making.
fn read_document(world: &mut World) -> Option<Document> {
    let index = world.resource::<EditorState>().root;
    let root = world.resource::<EditorRoots>().0.get(index)?.clone();
    let expanded = world.resource::<EditorState>().expanded.clone();

    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let Some(registration) = registry.get_with_type_path(root.type_path) else {
        warn_once!(
            "editor root '{}' is not in the type registry — nothing to show",
            root.type_path
        );
        return None;
    };
    let Some(reflect) = registration.data::<ReflectComponent>() else {
        warn_once!(
            "editor root '{}' is registered but not as a resource — it needs \
             #[reflect(Resource)]",
            root.type_path
        );
        return None;
    };

    let component_id = world.components().get_id(registration.type_id())?;
    let tick = world
        .get_resource_change_ticks_by_id(component_id)?
        .changed
        .get();
    let entity = world.resource_entities().get(component_id)?;
    let value = reflect.reflect(world.entity(entity))?;

    // The compiler is cheap and the panel is the thing making the
    // mistakes, so it asks directly rather than waiting to be told.
    let complaint = value
        .as_partial_reflect()
        .try_downcast_ref::<voxel_engine::LevelDef>()
        .and_then(|level| voxel_engine::graph::compile(&level.nodes).err())
        .map(|e| (e.at().to_string(), e.to_string()));

    let body = match root.view {
        View::Rows => Body::Rows(walk::rows_in(
            value.as_partial_reflect(),
            &expanded,
            &root.sections,
        )),
        // The graph is of the LEVEL's nodes, so this is the one place the
        // panel asks what it is showing. A root that is not a level draws
        // an empty graph rather than pretending otherwise.
        View::Graph => {
            let style = world.resource::<graph::GraphStyle>().clone();
            Body::Graph(
                value
                    .as_partial_reflect()
                    .try_downcast_ref::<voxel_engine::LevelDef>()
                    .map(|level| graph::layout(&level.nodes, &style))
                    .unwrap_or_default(),
            )
        }
    };
    // The selection is a path into the SAME document, so the rows it
    // produces address the level and edit it like any other row.
    let props = state_selected(world).and_then(|path| {
        // The level's own box is the document root, and what it holds is
        // what this tab declared it shows.
        if path == graph::LEVEL {
            let rows = walk::rows_in(value.as_partial_reflect(), &expanded, &root.sections);
            return Some(("level".to_string(), rows));
        }
        let node = value.reflect_path(path.as_str()).ok()?;
        // What the level calls it, if it named it at all.
        let label = value
            .reflect_path(format!("{path}.name.0").as_str())
            .ok()
            .and_then(|n| n.try_downcast_ref::<String>().cloned())
            .unwrap_or_else(|| path.clone());
        Some((
            label,
            walk::rows_at(
                value.as_partial_reflect(),
                node.as_partial_reflect(),
                &path,
                "nodes",
                &expanded,
            ),
        ))
    });
    Some(Document {
        tick,
        complaint,
        body,
        props,
    })
}
