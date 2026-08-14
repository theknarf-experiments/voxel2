//! A window with nothing in it but a graph.
//!
//! This is `bevy_graph_view` exercised on its own, away from the level
//! editor it was built inside: a hand-written document (a pour-over
//! recipe), converted to [`GraphNode`]s, laid out and drawn. It doubles as
//! the reference for what a HOST has to write, because the crate ships
//! geometry, drawing and `hover` and deliberately nothing else: the
//! wheel-pan, pinch-zoom, click-select and rebuild systems here are the
//! host's half of the contract.
//!
//! What to try: scroll pans, pinch or `-`/`=` zooms about the pointer,
//! hovering lights a box, clicking selects one (the canvas behind them
//! clears), and `B` blames the kettle — the red border a host would put
//! on whatever its compiler is complaining about.
//!
//! `GRAPH_SHOT=/path/out.png` takes a screenshot a second in and exits,
//! so a change to the crate can be looked at without a hand on the mouse.

use bevy::feathers::theme::{ThemeBackgroundColor, ThemedText, UiTheme};
use bevy::feathers::{dark_theme::create_dark_theme, tokens, FeathersPlugins};
use bevy::prelude::*;
use bevy::text::FontSize;
use bevy::ui::{percent, px, FlexDirection, Node, UiRect, UiTransform, Val2};
use bevy_graph_view::{
    layout, scene, GraphCamera, GraphCanvas, GraphNode, GraphStyle, GraphViewPlugin, GraphViewport,
    SelectsNode,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_graph_view".to_string(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FeathersPlugins)
        .insert_resource(UiTheme(create_dark_theme()))
        .add_plugins(GraphViewPlugin)
        .init_resource::<ViewState>()
        .add_observer(on_click)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                on_keys,
                on_wheel,
                on_pinch,
                rebuild,
                apply_camera,
                bevy_graph_view::hover,
                bevy_graph_view::zoom_label,
                status,
                auto_shot,
            )
                .chain(),
        )
        .run();
}

/// The document, in the graph view's vocabulary.
///
/// A host would convert its own format; here the recipe IS `GraphNode`s.
/// It is small but hits everything the layout does: two origins sharing a
/// column, a chain walking right, a two-input port, a scope drawn as a
/// frame, wires crossing the scope boundary in and out, and a name
/// (`pour`) declared inside the scope and read by a later sibling.
fn recipe() -> Vec<GraphNode> {
    let node =
        |id: &str, name: &str, kind: &str, ins: &[&str], outs: &[&str], wires: &[(&str, &str)]| {
            GraphNode {
                id: id.to_string(),
                name: Some(name.to_string()),
                kind: kind.to_string(),
                ins: ins.iter().map(|s| s.to_string()).collect(),
                outs: outs.iter().map(|s| s.to_string()).collect(),
                wires: wires
                    .iter()
                    .map(|(port, source)| (port.to_string(), vec![source.to_string()]))
                    .collect(),
                children: Vec::new(),
            }
        };
    vec![
        node("water", "water", "supply", &[], &["water"], &[]),
        node("beans", "beans", "supply", &[], &["beans"], &[]),
        node(
            "kettle",
            "kettle",
            "heat",
            &["water"],
            &["water"],
            &[("water", "water")],
        ),
        node(
            "grinder",
            "grinder",
            "grind",
            &["beans"],
            &["grounds"],
            &[("beans", "beans")],
        ),
        GraphNode {
            id: "brew".to_string(),
            name: Some("brew".to_string()),
            kind: "scope".to_string(),
            children: vec![
                node(
                    "brew/bloom",
                    "bloom",
                    "steep",
                    &["grounds", "water"],
                    &["slurry"],
                    &[("grounds", "grinder"), ("water", "kettle")],
                ),
                node(
                    "brew/pour",
                    "pour",
                    "filter",
                    &["slurry"],
                    &["coffee"],
                    &[("slurry", "bloom")],
                ),
            ],
            ..Default::default()
        },
        node(
            "serve",
            "serve",
            "cup",
            &["coffee"],
            &[],
            &[("coffee", "pour")],
        ),
    ]
}

/// The recipe's own box, above the graph it heads.
fn head() -> GraphNode {
    GraphNode {
        id: String::new(),
        name: Some("pour-over".to_string()),
        kind: "recipe".to_string(),
        ..Default::default()
    }
}

/// Everything the picture depends on besides the document itself.
///
/// One resource rather than state on entities, because the graph is
/// RESPAWNED when the selection changes — anything kept on a node would
/// go down with it. This is the same shape the level editor uses.
#[derive(Resource, Default)]
struct ViewState {
    camera: GraphCamera,
    selected: Option<String>,
    /// `B`: pretend a compiler is complaining about the kettle.
    blame: bool,
}

/// What the spawned graph was built from, so `rebuild` knows when it is
/// stale — and where it is, so it can be despawned.
#[derive(Resource, PartialEq)]
struct Shown {
    selected: Option<String>,
    blame: bool,
    entity: Entity,
}

/// The shell the graph hangs under: a header bar, then the viewport.
#[derive(Resource)]
struct Shell(Entity);

/// The header's one line of text.
#[derive(Component, Clone, Default)]
struct StatusLine;

fn setup(world: &mut World) {
    world.spawn(Camera2d);
    let shell = world
        .spawn_scene(bsn! {
            Node {
                width: percent(100),
                height: percent(100),
                flex_direction: FlexDirection::Column,
            }
            Children [(
                Node { padding: UiRect::all(px(8)), flex_shrink: 0.0 }
                ThemeBackgroundColor(tokens::PANE_HEADER_BG)
                Children [(
                    StatusLine
                    Text("")
                    TextFont { font_size: FontSize::Px(11.0) }
                    ThemedText
                )]
            )]
        })
        .expect("the shell is static and must resolve")
        .id();
    world.insert_resource(Shell(shell));
}

/// Respawn the graph when what it shows has changed.
///
/// Exclusive for `spawn_scene`; cheap because it compares against
/// [`Shown`] first and selection changes are pointer-rate, not
/// frame-rate. The camera is NOT part of `Shown`: panning must not
/// rebuild the picture — see [`apply_camera`].
fn rebuild(world: &mut World) {
    let state = world.resource::<ViewState>();
    let (selected, blame) = (state.selected.clone(), state.blame);
    if let Some(shown) = world.get_resource::<Shown>() {
        if shown.selected == selected && shown.blame == blame {
            return;
        }
        let stale = shown.entity;
        world.entity_mut(stale).despawn();
        world.remove_resource::<Shown>();
    }

    let style = world.resource::<GraphStyle>().clone();
    let camera = world.resource::<ViewState>().camera;
    let placed = layout(&recipe(), Some(&head()), &style);
    let blamed = blame.then_some("kettle");
    let shell = world.resource::<Shell>().0;
    match world.spawn_scene(scene(&placed, &style, camera, selected.as_deref(), blamed)) {
        Ok(graph) => {
            let entity = graph.id();
            world.entity_mut(entity).insert(ChildOf(shell));
            world.insert_resource(Shown {
                selected,
                blame,
                entity,
            });
        }
        Err(e) => warn_once!("graph did not spawn: {e}"),
    }
}

/// Select the box that was clicked, or clear the selection when the click
/// lands on the canvas behind them.
///
/// A pointer event BUBBLES, so this runs once per ancestor: selecting on
/// the way up is how a click on any part of a box selects it, but
/// clearing has to ask where the click STARTED, or the viewport at the
/// end of the chain undoes what the box just did.
fn on_click(
    click: On<Pointer<Click>>,
    boxes: Query<&SelectsNode>,
    viewports: Query<(), With<GraphViewport>>,
    mut state: ResMut<ViewState>,
) {
    if let Ok(SelectsNode(id)) = boxes.get(click.event_target()) {
        state.selected = Some(id.clone());
    } else if viewports.contains(click.original_event_target()) {
        state.selected = None;
    }
}

/// Two-finger scroll pans, on both axes: a graph has no reading direction.
fn on_wheel(
    mut wheel: MessageReader<bevy::input::mouse::MouseWheel>,
    mut state: ResMut<ViewState>,
) {
    let mut delta = Vec2::ZERO;
    for ev in wheel.read() {
        let scale = match ev.unit {
            bevy::input::mouse::MouseScrollUnit::Line => 24.0,
            bevy::input::mouse::MouseScrollUnit::Pixel => 1.0,
        };
        delta += Vec2::new(ev.x, ev.y) * scale;
    }
    if delta != Vec2::ZERO {
        state.camera.pan += delta;
    }
}

/// Pinch to zoom, about the pointer. macOS and iOS only — Bevy reports
/// the gesture nowhere else — which is why [`on_keys`] zooms too.
fn on_pinch(
    mut pinch: MessageReader<bevy::input::gestures::PinchGesture>,
    windows: Query<&Window>,
    viewports: Query<(&ComputedNode, &UiGlobalTransform), With<GraphViewport>>,
    mut state: ResMut<ViewState>,
) {
    let notches: f32 = pinch.read().map(|p| p.0).sum();
    if notches == 0.0 {
        return;
    }
    // About the POINTER, not the corner: zooming about the origin slides
    // whatever you were looking at off the screen.
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
    state.camera = state.camera.zoomed(notches * 12.0, at);
}

/// `=`/`-` zoom a notch at a time, `B` toggles the blamed box.
fn on_keys(keys: Res<ButtonInput<KeyCode>>, mut state: ResMut<ViewState>) {
    if keys.just_pressed(KeyCode::Equal) {
        state.camera = state.camera.zoomed(1.0, None);
    }
    if keys.just_pressed(KeyCode::Minus) {
        state.camera = state.camera.zoomed(-1.0, None);
    }
    if keys.just_pressed(KeyCode::KeyB) {
        state.blame = !state.blame;
    }
}

/// Apply the camera to the live canvas.
///
/// Directly, not by respawning: a rebuild per scroll frame would throw
/// away every box in the graph to move the picture a pixel. A canvas
/// respawned by [`rebuild`] already carries the camera — `scene` bakes it
/// in — so this only has to chase the frames where a gesture moved it.
fn apply_camera(state: Res<ViewState>, mut canvases: Query<&mut UiTransform, With<GraphCanvas>>) {
    if !state.is_changed() {
        return;
    }
    for mut transform in &mut canvases {
        transform.translation = Val2::px(state.camera.pan.x, state.camera.pan.y);
        transform.scale = Vec2::splat(state.camera.zoom);
    }
}

/// Say what is selected, and what there is to try.
fn status(state: Res<ViewState>, mut lines: Query<&mut Text, With<StatusLine>>) {
    if !state.is_changed() {
        return;
    }
    let selected = match &state.selected {
        Some(id) if id.is_empty() => "the recipe",
        Some(id) => id.as_str(),
        None => "nothing",
    };
    for mut text in &mut lines {
        // ASCII separators: the mono font has no glyph for a middle dot,
        // which renders as tofu.
        text.0 = format!(
            "scroll pans | pinch or -/= zooms | click selects | B blames the kettle | selected: {selected}"
        );
    }
}

/// `GRAPH_SHOT=/path/out.png`: capture the window a second in, then exit.
///
/// The window must be frontmost — macOS gives occluded windows no
/// drawable, and the capture comes back black. A fresh launch is
/// frontmost, so `GRAPH_SHOT=x.png cargo run -p graph_view_demo` just
/// works.
fn auto_shot(mut commands: Commands, mut frame: Local<u32>, mut exit: MessageWriter<AppExit>) {
    use bevy::render::view::window::screenshot::{save_to_disk, Screenshot};
    let Ok(path) = std::env::var("GRAPH_SHOT") else {
        return;
    };
    *frame += 1;
    // Late enough for layout and fonts to have settled.
    if *frame == 60 {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
    // Late enough for the capture to have been written.
    if *frame == 120 {
        exit.write(AppExit::Success);
    }
}
