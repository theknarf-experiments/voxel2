//! An edit made in the panel has to reach the document, and the document
//! has to notice.
//!
//! Against the real `LevelDef` and a real shipped level rather than a
//! fixture: what this checks is that a path of the shape the walk produces
//! resolves back to the actual field of the actual schema, and a toy
//! struct would only agree with the walk about a schema neither has.
//!
//! The widget half — a click becoming a queued edit — is six lines of
//! observer and needs a window. The half that can be wrong quietly (the
//! path, the type, the change tick) is all here.

use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use voxel_editor::edit::{apply, Edit, Pending, Value};
use voxel_editor::{EditorRoots, EditorState, Num, Root};
use voxel_engine::graph::{nodes, registry};
use voxel_engine::LevelDef;

/// Set by change detection, exactly as `apply_level_change` is driven.
#[derive(Resource, Default)]
struct Rebuilt(bool);

fn note_rebuild(level: Res<LevelDef>, mut flag: ResMut<Rebuilt>) {
    flag.0 = level.is_changed();
}

fn planet() -> LevelDef {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../levels/planet.json");
    LevelDef::from_path_known(std::path::Path::new(&path), &registry::engine_kinds()).unwrap()
}

fn app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .register_type::<LevelDef>()
        .insert_resource(planet())
        .init_resource::<Rebuilt>()
        .insert_resource(EditorRoots(vec![Root {
            sections: voxel_editor::Sections::All,
            view: voxel_editor::View::Rows,
            label: "Level".into(),
            type_path: <LevelDef as TypePath>::type_path(),
        }]))
        .insert_resource(EditorState {
            open: true,
            root: 0,
            expanded: HashSet::default(),
            ..default()
        })
        .init_resource::<Pending>()
        .init_resource::<voxel_editor::History>()
        .add_systems(Update, (apply, note_rebuild).chain());
    // Settle the insert's own change tick.
    app.update();
    app
}

fn edit(app: &mut App, path: &str, value: f64, num: Num) {
    app.world_mut().resource_mut::<Pending>().0.push(Edit {
        root: 0,
        path: path.to_string(),
        value: Value::Num(value, num),
    });
    app.update();
}

#[test]
fn an_edit_reaches_the_field_its_path_names() {
    let mut app = app();
    let count = app.world().resource::<LevelDef>().materials.len();

    edit(&mut app, ".materials[0].id", 42.0, Num::U32);

    let level = app.world().resource::<LevelDef>();
    assert_eq!(level.materials.len(), count, "it changed the wrong thing");
    assert_eq!(level.materials[0].id(), 42);
}

/// The path shape the whole editor turns on: a number inside a struct
/// variant of an enum inside a list. Every generator node is one, and it
/// is the shape `GetPath` alone would not have reached through the map case
/// this crate's resolver exists for.
#[test]
fn an_edit_reaches_inside_a_generator_node() {
    let mut app = app();
    // By variant, not by index: which node sits where is content, and a
    // test that hardcoded it would break on any re-authoring of the level.
    let at = app
        .world()
        .resource::<LevelDef>()
        .nodes
        .iter()
        .position(|n| n.node.kind() == "height_fbm")
        .expect("planet's terrain is fbm height bands");

    edit(&mut app, &format!(".nodes[{at}].node.amp"), 123.0, Num::F32);

    let level = app.world().resource::<LevelDef>();
    let node = level.nodes[at]
        .node
        .0
        .as_any()
        .downcast_ref::<nodes::HeightFbm>()
        .expect("the edit changed the kind");
    assert_eq!(node.amp, 123.0);
}

/// `LevelDef` changing IS the mechanism: `apply_level_change` runs on
/// change detection, so an edit that mutates the value without marking it
/// applies to nothing and reads as a broken editor.
#[test]
fn an_edit_marks_the_document_changed() {
    let mut app = app();
    app.update();
    assert!(
        !app.world().resource::<Rebuilt>().0,
        "nothing edited it yet"
    );

    edit(&mut app, ".materials[0].id", 7.0, Num::U32);
    assert!(
        app.world().resource::<Rebuilt>().0,
        "the world would never rebuild"
    );
}

/// A number widget deals in floats; the field may not. The conversion
/// happens once, on the way in, and a `u32` field must end up holding a
/// `u32` rather than refusing the edit.
#[test]
fn a_float_widget_can_edit_an_integer_field() {
    let mut app = app();
    edit(&mut app, ".lod.max_level", 6.4, Num::U8);
    assert_eq!(app.world().resource::<LevelDef>().lod.max_level, 6);
}

/// A path that does not resolve leaves the document alone and says so,
/// rather than panicking the session or half-applying.
#[test]
fn a_bad_path_changes_nothing() {
    let mut app = app();
    let before = format!("{:?}", app.world().resource::<LevelDef>().materials[0]);

    edit(&mut app, ".materials[0].no_such_field", 1.0, Num::F32);
    edit(&mut app, ".materials[9999].id", 1.0, Num::U32);
    edit(&mut app, "materials", 1.0, Num::F32);

    let after = format!("{:?}", app.world().resource::<LevelDef>().materials[0]);
    assert_eq!(before, after);
}

/// Clicking a tab switches which view the panel shows.
///
/// The observer half is small but it is the ONE thing standing between a
/// strip that looks right in a screenshot and a strip that does nothing:
/// the button carries the index, the observer writes it, and the panel
/// respawns from it.
#[test]
fn activating_a_tab_selects_its_root() {
    let mut app = app();
    app.world_mut().resource_mut::<EditorRoots>().0.push(Root {
        label: "Nodes".into(),
        type_path: <LevelDef as TypePath>::type_path(),
        sections: voxel_editor::Sections::Only(vec!["nodes".into()]),
        view: voxel_editor::View::Rows,
    });
    app.add_observer(voxel_editor::on_tab);

    let tab = app.world_mut().spawn(voxel_editor::SelectsRoot(1)).id();
    assert_eq!(app.world().resource::<EditorState>().root, 0);

    app.world_mut()
        .trigger(bevy::ui_widgets::Activate { entity: tab });
    app.update();
    assert_eq!(app.world().resource::<EditorState>().root, 1);
}

/// Picking a reference from its menu writes it, at the field's own type.
///
/// A material id is a number and a prefab name is a string; the menu that
/// offers them is the same menu, so the item carries which it is.
#[test]
fn picking_a_reference_writes_it() {
    use voxel_editor::PicksOption;
    let mut app = app();
    app.add_observer(voxel_editor::on_pick);

    // A numeric reference: the id an op paints with.
    let id = app.world().resource::<LevelDef>().materials[1].id();
    let item = app
        .world_mut()
        .spawn(PicksOption {
            path: ".materials[0].id".into(),
            value: id.to_string(),
            num: Some(Num::U32),
            variant: false,
        })
        .id();
    app.world_mut()
        .trigger(bevy::ui_widgets::Activate { entity: item });
    app.update();
    assert_eq!(
        app.world().resource::<LevelDef>().materials[0].id(),
        id,
        "a numeric reference lands as a number"
    );

    // A textual one: the node a port is wired to. The name it is wired to
    // NOW comes with it, because picking that same name back would assert
    // something a write that never landed also satisfies — which is how
    // `.0` failing to resolve on the write side went unnoticed.
    let (at, port, was) = app
        .world()
        .resource::<LevelDef>()
        .nodes
        .iter()
        .enumerate()
        .find_map(|(i, n)| match n.wires.iter().next()? {
            // One source: a `Wire::Many` is a list, and the path its
            // entries carry ends in an index this would have to guess.
            (port, voxel_engine::graph::Wire::One(source)) => {
                Some((i, port.clone(), source.clone()))
            }
            _ => None,
        })
        .expect("planet wires its nodes");
    let name = app
        .world()
        .resource::<LevelDef>()
        .nodes
        .iter()
        .filter_map(|n| n.name.clone())
        .find(|n| *n != was)
        .expect("planet names more than one node");
    let item = app
        .world_mut()
        .spawn(PicksOption {
            path: format!(".nodes[{at}].wires.0{{{port}}}.0"),
            value: name.clone(),
            num: None,
            variant: false,
        })
        .id();
    app.world_mut()
        .trigger(bevy::ui_widgets::Activate { entity: item });
    app.update();
    let wired = app.world().resource::<LevelDef>().nodes[at]
        .wires
        .get(&port)
        .and_then(|w| w.sources().first().cloned());
    assert_eq!(
        wired.as_deref(),
        Some(name.as_str()),
        "a wire is rewired: it said '{was}'"
    );
}

/// Save is asked for, never done here: this crate edits any reflected
/// resource and has no idea where one came from.
#[test]
fn the_panel_asks_to_save_and_only_when_open() {
    use bevy::input::ButtonInput;
    let mut app = app();
    app.add_message::<voxel_editor::SaveRequested>()
        .init_resource::<ButtonInput<KeyCode>>()
        .add_systems(Update, voxel_editor::save);

    let asked = |app: &mut App| -> usize {
        let world = app.world_mut();
        let messages =
            world.resource::<bevy::ecs::message::Messages<voxel_editor::SaveRequested>>();
        messages.iter_current_update_messages().count()
    };

    // Shut: the flag is ignored, or a level editor would write the file
    // from a keystroke in a game window.
    app.world_mut().resource_mut::<EditorState>().open = false;
    app.world_mut().resource_mut::<EditorState>().save = true;
    app.update();
    assert_eq!(asked(&mut app), 0, "a shut panel saves nothing");

    app.world_mut().resource_mut::<EditorState>().open = true;
    app.world_mut().resource_mut::<EditorState>().save = true;
    app.update();
    assert_eq!(asked(&mut app), 1, "open, and asked");
    assert!(
        !app.world().resource::<EditorState>().save,
        "the ask is cleared, or it would save every frame"
    );

    app.update();
    assert_eq!(asked(&mut app), 1, "and only once");
}

/// Picking a variant changes what the field IS, with the new variant's
/// own fields at their defaults.
///
/// A `zoned` material is not a `surface` one with different numbers, so
/// nothing is carried across: switching a recipe is a real edit, and the
/// row that follows it shows what actually has to be filled in.
#[test]
fn picking_a_variant_changes_the_shape_of_the_field() {
    use voxel_editor::PicksOption;
    let mut app = app();
    app.add_observer(voxel_editor::on_pick);

    // A unit variant: the shaping of a noise op.
    let at = app
        .world()
        .resource::<LevelDef>()
        .nodes
        .iter()
        .position(|n| n.node.0.kind() == "height_fbm")
        .expect("planet has fbm height");
    let path = format!(".nodes[{at}].node.mode");
    let item = app
        .world_mut()
        .spawn(PicksOption {
            path: path.clone(),
            value: "Ridged".into(),
            num: None,
            variant: true,
        })
        .id();
    app.world_mut()
        .trigger(bevy::ui_widgets::Activate { entity: item });
    app.update();

    let mode = app.world().resource::<LevelDef>().nodes[at]
        .node
        .0
        .as_reflect()
        .reflect_path(".mode")
        .unwrap();
    let bevy::reflect::ReflectRef::Enum(mode) = mode.reflect_ref() else {
        panic!("mode is an enum")
    };
    assert_eq!(mode.variant_name(), "Ridged");

    // And a struct variant: a whole different material recipe.
    let item = app
        .world_mut()
        .spawn(PicksOption {
            path: ".materials[0]".into(),
            value: "Zoned".into(),
            num: None,
            variant: true,
        })
        .id();
    app.world_mut()
        .trigger(bevy::ui_widgets::Activate { entity: item });
    app.update();
    let materials = &app.world().resource::<LevelDef>().materials;
    assert!(
        matches!(materials[0], voxel_engine::level::MaterialDef::Zoned { .. }),
        "the recipe changed"
    );
}

/// A number with no declared bounds is dragged, and lands where the
/// pointer says: `from + distance * speed`, at the field's own type.
#[test]
fn dragging_a_number_writes_where_the_pointer_went() {
    use voxel_editor::{DragsNum, FieldPath, WritesNum};
    let mut app = app();
    app.add_observer(voxel_editor::on_drag);

    let at = app
        .world()
        .resource::<LevelDef>()
        .nodes
        .iter()
        .position(|n| n.node.0.kind() == "height_fbm")
        .expect("planet has fbm height");
    let path = format!(".nodes[{at}].node.amp");
    let amp = read(&app, &path);

    let widget = app
        .world_mut()
        .spawn((
            FieldPath(path.clone()),
            DragsNum {
                from: amp,
                speed: 2.0,
            },
            WritesNum(Num::F32),
        ))
        .id();
    drag(&mut app, widget, 10.0);
    let after = read(&app, &path);
    assert!(
        (after - (amp + 20.0)).abs() < 1e-3,
        "{amp} + 10px * 2 = {}, got {after}",
        amp + 20.0
    );

    // The SAME drag continued further is absolute, not cumulative: the
    // distance is the drag's own total.
    drag(&mut app, widget, 15.0);
    let after = read(&app, &path);
    assert!((after - (amp + 30.0)).abs() < 1e-3, "got {after}");
}

/// An `f32` the level holds, by path.
fn read(app: &App, path: &str) -> f32 {
    *app.world()
        .resource::<LevelDef>()
        .as_reflect()
        .reflect_path(path)
        .unwrap()
        .try_downcast_ref::<f32>()
        .unwrap()
}

/// Drag `widget` `x` pixels sideways.
///
/// The event is built by hand rather than pumped through picking: what is
/// under test is the arithmetic from a drag to a value, and a real pointer
/// would need a window, a camera and a hit.
fn drag(app: &mut App, widget: Entity, x: f32) {
    use bevy::picking::events::{Pointer, PointerDrag};
    use bevy::picking::pointer::{Location, PointerId};
    let window = app.world_mut().spawn(bevy::window::Window::default()).id();
    let location = Location {
        target: bevy::window::WindowRef::Entity(window)
            .normalize(None)
            .unwrap()
            .into(),
        position: Vec2::ZERO,
    };
    app.world_mut().trigger(PointerDrag {
        entity: widget,
        pointer: Pointer::new(PointerId::Mouse, location),
        button: bevy::picking::pointer::PointerButton::Primary,
        distance: Vec2::new(x, 0.0),
        delta: Vec2::new(x, 0.0),
    });
    app.update();
}

/// Undo puts the document back as it was before the last batch.
///
/// A batch, not an edit: a drag queues one a frame, and undoing a drag a
/// pixel at a time would be its own kind of unusable.
#[test]
fn undo_restores_what_the_last_batch_changed() {
    let mut app = app();
    app.add_systems(Update, voxel_editor::undo);

    let was = app.world().resource::<LevelDef>().materials[0].id();
    edit(&mut app, ".materials[0].id", 77.0, Num::U32);
    assert_eq!(app.world().resource::<LevelDef>().materials[0].id(), 77);

    app.world_mut().resource_mut::<EditorState>().undo = true;
    app.update();
    assert_eq!(
        app.world().resource::<LevelDef>().materials[0].id(),
        was,
        "the id comes back"
    );

    // A variant change has no inverse — the fields it replaced are gone —
    // which is why the step is a whole snapshot.
    let before = format!("{:?}", app.world().resource::<LevelDef>().materials[0]);
    app.world_mut().resource_mut::<Pending>().0.push(Edit {
        root: 0,
        path: ".materials[0]".into(),
        value: Value::Variant("Zoned".into()),
    });
    app.update();
    assert_ne!(
        format!("{:?}", app.world().resource::<LevelDef>().materials[0]),
        before
    );
    app.world_mut().resource_mut::<EditorState>().undo = true;
    app.update();
    assert_eq!(
        format!("{:?}", app.world().resource::<LevelDef>().materials[0]),
        before,
        "a switched recipe comes back whole"
    );

    // And undoing with nothing to undo is a no-op, not a panic.
    app.world_mut().resource_mut::<EditorState>().undo = true;
    app.update();
    app.world_mut().resource_mut::<EditorState>().undo = true;
    app.update();
}

/// Redo puts back what undo took away, and a fresh edit forgets it: the
/// future that was undone is not the future of a document that has since
/// been edited down a different path.
#[test]
fn redo_puts_back_what_undo_took_and_an_edit_forgets_it() {
    let mut app = app();
    app.add_systems(Update, voxel_editor::undo);

    let was = app.world().resource::<LevelDef>().materials[0].id();
    edit(&mut app, ".materials[0].id", 77.0, Num::U32);

    app.world_mut().resource_mut::<EditorState>().undo = true;
    app.update();
    assert_eq!(app.world().resource::<LevelDef>().materials[0].id(), was);

    app.world_mut().resource_mut::<EditorState>().redo = true;
    app.update();
    assert_eq!(
        app.world().resource::<LevelDef>().materials[0].id(),
        77,
        "redo puts it back"
    );

    // Undo, then edit: the redo is gone.
    app.world_mut().resource_mut::<EditorState>().undo = true;
    app.update();
    edit(&mut app, ".materials[0].id", 5.0, Num::U32);
    assert!(!app.world().resource::<voxel_editor::History>().can_redo());
    app.world_mut().resource_mut::<EditorState>().redo = true;
    app.update();
    assert_eq!(
        app.world().resource::<LevelDef>().materials[0].id(),
        5,
        "and nothing comes back"
    );
}

/// A click anywhere in a node box selects that node, and a click on the
/// empty canvas clears the selection.
///
/// The case that matters is a click on a CHILD — a title bar, a port row
/// — because a pointer event bubbles and this observer runs once per
/// ancestor. Selecting on the way up is what makes any part of a box
/// work; the surface at the END of that chain used to clear what the box
/// had just set, so selection never worked by clicking at all.
///
/// The clearing surface is the BACKDROP inside the graph's texture, not
/// the viewport node: the real pointer clicks the viewport on every
/// click, box or not, so clearing there would race every selection.
#[test]
fn clicking_a_node_selects_it_even_through_its_children() {
    use bevy::picking::events::{Pointer, PointerClick};
    use bevy::picking::pointer::{Location, PointerButton, PointerId};
    use bevy_graph_view::GraphBackdrop;
    use voxel_editor::SelectsNode;

    let mut app = app();
    app.add_observer(voxel_editor::on_select);

    let backdrop = app.world_mut().spawn(GraphBackdrop).id();
    let boxed = app.world_mut().spawn(SelectsNode(".nodes[3]".into())).id();
    let title = app.world_mut().spawn(ChildOf(boxed)).id();

    let click = |app: &mut App, at: Entity| {
        let window = app.world_mut().spawn(bevy::window::Window::default()).id();
        let location = Location {
            target: bevy::window::WindowRef::Entity(window)
                .normalize(None)
                .unwrap()
                .into(),
            position: Vec2::ZERO,
        };
        app.world_mut().trigger(PointerClick {
            entity: at,
            pointer: Pointer::new(PointerId::Mouse, location),
            button: PointerButton::Primary,
            hit: bevy::picking::backend::HitData::new(Entity::PLACEHOLDER, 0.0, None, None),
            duration: std::time::Duration::ZERO,
            count: 1,
        });
        app.update();
    };

    click(&mut app, title);
    assert_eq!(
        app.world().resource::<EditorState>().selected.as_deref(),
        Some(".nodes[3]"),
        "a click on a node's title selects the node"
    );

    click(&mut app, backdrop);
    assert_eq!(
        app.world().resource::<EditorState>().selected,
        None,
        "and a click on the backdrop behind them clears it"
    );
}

/// Two-finger scroll pans the graph, on both axes, and the picture
/// follows the fingers the way the row list does.
///
/// Pinch zooms about the POINTER rather than the corner. Neither can be
/// performed by hand in a test harness, but both are messages, so what
/// the systems do with them is not a matter of opinion.
#[test]
fn scrolling_pans_the_graph_and_pinching_zooms_it() {
    use bevy::input::gestures::PinchGesture;
    use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
    use voxel_editor::View;

    let mut app = app();
    app.add_message::<MouseWheel>()
        .add_message::<PinchGesture>()
        .init_resource::<voxel_editor::PanelStyle>()
        .add_systems(Update, (voxel_editor::on_wheel, voxel_editor::on_pinch));
    // A graph tab, and a pointer over the panel.
    app.world_mut().resource_mut::<EditorRoots>().0[0].view = View::Graph;
    let window = app.world_mut().spawn(bevy::window::Window::default()).id();
    let width = app
        .world()
        .entity(window)
        .get::<bevy::window::Window>()
        .unwrap()
        .width();
    let mut win = app.world_mut().entity_mut(window);
    let mut win = win.get_mut::<bevy::window::Window>().unwrap();
    // Over the GRAPH: further right is the properties column, where a
    // scroll deliberately scrolls instead of panning.
    win.set_cursor_position(Some(Vec2::new(width - 400.0, 100.0)));

    let was = app.world().resource::<EditorState>().camera;
    app.world_mut().write_message(MouseWheel {
        unit: MouseScrollUnit::Pixel,
        x: 7.0,
        y: 13.0,
        window,
        phase: bevy::input::touch::TouchPhase::Moved,
    });
    app.update();
    let now = app.world().resource::<EditorState>().camera;
    assert_eq!(
        now.pan,
        was.pan + Vec2::new(7.0, 13.0),
        "the graph follows the fingers, on both axes"
    );

    // Pinching out zooms in. WHERE it zooms about needs a laid-out
    // viewport to read the pointer against, which a bare harness has no
    // reason to have — that arithmetic is `zooming_holds_the_point_it_is_
    // aimed_at`, and without a viewport the documented fallback is to
    // hold the pan and zoom about the corner.
    let before = app.world().resource::<EditorState>().camera;
    app.world_mut().write_message(PinchGesture(0.05));
    app.update();
    let after = app.world().resource::<EditorState>().camera;
    assert!(after.zoom > before.zoom, "a pinch out zooms in");
    assert_eq!(
        after.pan, before.pan,
        "and holds the corner with nothing to aim at"
    );

    // Pinching the other way zooms out again.
    app.world_mut().write_message(PinchGesture(-0.05));
    app.update();
    let back = app.world().resource::<EditorState>().camera;
    assert!(back.zoom < after.zoom, "a pinch in zooms out");
}

/// Typing a new name for a node takes every wire with it.
///
/// A name is the only way anything refers to a node, so writing one as an
/// ordinary field would be the same edit as deleting the node: the level
/// would stop compiling on the first keystroke that lands.
#[test]
fn renaming_a_node_rewires_the_level() {
    let mut app = app();
    let at = app
        .world()
        .resource::<LevelDef>()
        .nodes
        .iter()
        .position(|n| n.name.as_deref() == Some("sea"))
        .expect("planet names its sea level");
    let readers: Vec<String> = app
        .world()
        .resource::<LevelDef>()
        .nodes
        .iter()
        .filter(|n| {
            n.wires
                .iter()
                .any(|(_, w)| w.sources().iter().any(|s| s == "sea"))
        })
        .filter_map(|n| n.name.clone())
        .collect();
    assert!(!readers.is_empty(), "something reads it");

    app.world_mut().resource_mut::<Pending>().0.push(Edit {
        root: 0,
        path: format!(".nodes[{at}].name.0"),
        value: Value::Text("ocean".into()),
    });
    app.update();

    let level = app.world().resource::<LevelDef>();
    assert_eq!(level.nodes[at].name.as_deref(), Some("ocean"));
    for name in &readers {
        let node = level
            .nodes
            .iter()
            .find(|n| n.name.as_deref() == Some(name.as_str()))
            .unwrap();
        assert!(
            node.wires
                .iter()
                .any(|(_, w)| w.sources().iter().any(|s| s == "ocean")),
            "{name} still points at the old name"
        );
    }
    assert!(
        voxel_engine::graph::compile(&level.nodes).is_ok(),
        "and the level still compiles"
    );
}

/// Typing commits on ENTER, not per keystroke.
///
/// A name is a reference, so every intermediate would be a document that
/// does not compile and a step in the undo stack.
#[test]
fn text_commits_on_enter_and_only_then() {
    use bevy::input::ButtonInput;
    use bevy::input_focus::{FocusCause, InputFocus};
    use bevy::text::EditableText;
    use voxel_editor::FieldPath;

    let mut app = app();
    app.init_resource::<ButtonInput<KeyCode>>()
        .init_resource::<InputFocus>()
        .init_resource::<voxel_editor::PressedOut>()
        .init_resource::<voxel_editor::Typing>()
        .add_systems(Update, voxel_editor::on_typed);

    let container = app
        .world_mut()
        .spawn(FieldPath(".nodes[0].name.0".into()))
        .id();
    let input = app
        .world_mut()
        .spawn((EditableText::new("ocean"), ChildOf(container)))
        .id();
    app.world_mut()
        .resource_mut::<InputFocus>()
        .set(input, FocusCause::Pressed);

    // Focused, but nothing pressed: the document is untouched.
    let before = app.world().resource::<LevelDef>().nodes[0].name.clone();
    app.update();
    assert_eq!(app.world().resource::<LevelDef>().nodes[0].name, before);

    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(KeyCode::Enter);
    app.update();
    // Queued AND applied in the same frame — the harness runs `apply` too,
    // which is the whole path this is here to check.
    let level = app.world().resource::<LevelDef>();
    assert_eq!(
        level.nodes[0].name.as_deref(),
        Some("ocean"),
        "Enter commits what was typed"
    );
    assert!(
        voxel_engine::graph::compile(&level.nodes).is_ok(),
        "and the rename took its wires with it"
    );
}

/// A box with `WritesNum` on it, focused, and nothing that drains the
/// queue: what a keystroke QUEUES is the half these pin, and the half that
/// applies it is every other test in this file.
///
/// `num` is what makes a box a NUMBER rather than a name — the same
/// distinction `row::typed_box` draws — so a `None` here is a text row.
fn box_holding(path: &str, num: Option<Num>, text: &str) -> App {
    use bevy::input::ButtonInput;
    use bevy::input_focus::{FocusCause, InputFocus};
    use bevy::text::EditableText;
    use voxel_editor::{FieldPath, WritesNum};

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .register_type::<LevelDef>()
        .insert_resource(planet())
        .init_resource::<Pending>()
        .init_resource::<EditorState>()
        .init_resource::<ButtonInput<KeyCode>>()
        .init_resource::<InputFocus>()
        .init_resource::<voxel_editor::PressedOut>()
        .init_resource::<voxel_editor::Typing>()
        .add_observer(voxel_editor::on_press)
        .add_systems(Update, voxel_editor::on_typed);

    let mut container = app.world_mut().spawn(FieldPath(path.to_string()));
    if let Some(num) = num {
        container.insert(WritesNum(num));
    }
    let container = container.id();
    let input = app
        .world_mut()
        .spawn((EditableText::new(text), ChildOf(container)))
        .id();
    app.world_mut()
        .resource_mut::<InputFocus>()
        .set(input, FocusCause::Pressed);
    app.update();
    app
}

fn enter(app: &mut App) {
    app.world_mut()
        .resource_mut::<bevy::input::ButtonInput<KeyCode>>()
        .press(KeyCode::Enter);
    app.update();
}

/// What is waiting to be applied.
fn queued(app: &App) -> &[Edit] {
    &app.world().resource::<Pending>().0
}

/// What the panel is saying was refused, if anything.
fn refused(app: &App) -> Option<String> {
    app.world().resource::<EditorState>().refused.clone()
}

/// A typed number goes down the SAME path a drag does: one `Value::Num`,
/// at the field's own type, so it gets the same `apply`, the same undo
/// snapshot and the same rebuild decision.
///
/// Typing exists because dragging cannot do this: the step is a percent of
/// the value, so a field left at its default of 0 is hundreds of millions
/// of pixels away from 800 and every node created from scratch would be
/// unreachable.
#[test]
fn a_typed_number_is_queued_as_a_number_at_the_fields_own_type() {
    let mut app = box_holding(".lod.max_level", Some(Num::U8), "9");
    enter(&mut app);
    match queued(&app) {
        [Edit {
            path,
            value: Value::Num(v, num),
            ..
        }] => {
            assert_eq!(path, ".lod.max_level");
            assert_eq!(*v, 9.0);
            assert_eq!(*num, Num::U8, "a u8 field is written as a u8");
        }
        other => panic!("one numeric edit, got {} edits", other.len()),
    }
    assert_eq!(refused(&app), None);

    // And the box with no `WritesNum` is still a name: the text IS the
    // value, not something to parse.
    let mut app = box_holding(".nodes[0].name.0", None, "9");
    enter(&mut app);
    assert!(matches!(queued(&app), [Edit { value: Value::Text(s), .. }] if s == "9"));
}

/// Nonsense writes NOTHING, and the panel says so.
///
/// A refusal that only warned in the log is indistinguishable from a
/// broken editor: nothing in the document moves, so silence would read as
/// "that landed". What was typed stays in the box to be corrected — see
/// `panel::notice`, which is why a refusal does not rebuild the panel.
#[test]
fn an_unparseable_entry_queues_nothing_and_the_panel_says_which_field() {
    let mut app = box_holding(".lod.split_k", Some(Num::F64), "eight hundred");
    enter(&mut app);
    assert!(queued(&app).is_empty(), "nothing is written");
    let said = refused(&app).expect("the panel says it was refused");
    assert!(said.contains(".lod.split_k"), "and which field: {said}");
    assert!(said.contains("f64"), "and what it would have to be: {said}");
}

/// An integer field refuses a fraction rather than rounding one.
///
/// The drag rounds because a pointer has no exact value to mean; text
/// does, and `3.7` silently becoming 3 is the tool getting the level
/// wrong.
#[test]
fn a_fraction_typed_into_an_integer_field_is_refused_not_truncated() {
    for (num, text) in [(Num::U8, "3.7"), (Num::U32, "3.7"), (Num::I32, "-3.7")] {
        let mut app = box_holding(".lod.max_level", Some(num), text);
        enter(&mut app);
        assert!(
            queued(&app).is_empty(),
            "{text} is not a {num:?} and must not be rounded into one"
        );
        assert!(refused(&app).is_some());
    }
}

/// And refuses what its type cannot hold, rather than clamping it.
#[test]
fn a_number_outside_its_types_range_is_refused_not_clamped() {
    for (num, text) in [(Num::U8, "999"), (Num::Usize, "-1"), (Num::U32, "-1")] {
        let mut app = box_holding(".lod.max_level", Some(num), text);
        enter(&mut app);
        assert!(
            queued(&app).is_empty(),
            "{text} is not a {num:?} and must not be clamped into one"
        );
    }
}

/// Leaving the box commits it, exactly once.
///
/// A designer who types a number and clicks the next row has said what
/// they meant. Committing again on the way out of a box Enter already
/// committed would be a second, identical edit — and a step of undo that
/// does nothing.
#[test]
fn leaving_a_box_commits_what_is_in_it_and_enter_does_not_commit_twice() {
    use bevy::input_focus::InputFocus;

    let mut app = box_holding(".lod.split_k", Some(Num::F64), "0.75");
    assert!(queued(&app).is_empty(), "focused is not committed");

    app.world_mut().resource_mut::<InputFocus>().clear();
    app.update();
    assert_eq!(queued(&app).len(), 1, "leaving the box commits it");

    // Enter, then focus moves on: still one edit.
    let mut app = box_holding(".lod.split_k", Some(Num::F64), "0.75");
    enter(&mut app);
    assert_eq!(queued(&app).len(), 1);
    app.world_mut().resource_mut::<InputFocus>().clear();
    app.update();
    assert_eq!(queued(&app).len(), 1, "and not again on the way out");
}

/// End to end: what is typed reaches the document, and undo takes it back
/// — because it is the same `Value::Num` the drag produces and there is
/// only one write path.
#[test]
fn a_typed_number_reaches_the_document_and_can_be_undone() {
    use bevy::input::ButtonInput;
    use bevy::input_focus::{FocusCause, InputFocus};
    use bevy::text::EditableText;
    use voxel_editor::{FieldPath, WritesNum};

    let mut app = app();
    app.init_resource::<ButtonInput<KeyCode>>()
        .init_resource::<InputFocus>()
        .init_resource::<voxel_editor::PressedOut>()
        .init_resource::<voxel_editor::Typing>()
        .add_systems(Update, voxel_editor::on_typed.before(apply));

    let at = app
        .world()
        .resource::<LevelDef>()
        .nodes
        .iter()
        .position(|n| n.node.0.kind() == "height_fbm")
        .expect("planet has fbm height");
    let path = format!(".nodes[{at}].node.amp");
    let was = read(&app, &path);

    let container = app.world_mut().spawn(FieldPath(path.clone())).id();
    app.world_mut()
        .entity_mut(container)
        .insert(WritesNum(Num::F32));
    let input = app
        .world_mut()
        .spawn((EditableText::new("800"), ChildOf(container)))
        .id();
    app.world_mut()
        .resource_mut::<InputFocus>()
        .set(input, FocusCause::Pressed);
    app.update();
    assert_eq!(read(&app, &path), was, "nothing lands before Enter");

    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(KeyCode::Enter);
    app.update();
    assert_eq!(read(&app, &path), 800.0, "a number no drag could reach");

    // No `InputPlugin` here, so the key stays "just pressed" until it is
    // told otherwise — the real one is cleared every frame.
    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .clear();
    app.add_systems(Update, voxel_editor::undo);
    app.world_mut().resource_mut::<EditorState>().undo = true;
    app.update();
    assert_eq!(read(&app, &path), was, "and one snapshot took it back");
}

/// Queue a string at `path`, the way a box on the way out does.
fn text_edit(app: &mut App, path: &str, value: &str) {
    app.world_mut().resource_mut::<Pending>().0.push(Edit {
        root: 0,
        path: path.to_string(),
        value: Value::Text(value.to_string()),
    });
    app.update();
}

/// What the level calls the node at `at`.
fn name_of(app: &App, at: usize) -> Option<String> {
    app.world().resource::<LevelDef>().nodes[at].name.clone()
}

/// A named node, and one no wire refers to.
fn a_named_node(app: &App) -> (usize, String) {
    let level = app.world().resource::<LevelDef>();
    level
        .nodes
        .iter()
        .enumerate()
        .find_map(|(i, n)| Some((i, n.name.clone()?)))
        .expect("planet names its nodes")
}

/// A no-op edit is not a step, and taking a step is what throws the redo
/// stack away.
///
/// Renaming reports how many WIRES followed, and zero is the answer both
/// to "renamed, and nothing referred to it" and to "the name did not
/// change". Counting the second as an edit costs an undo step that undoes
/// A no-op edit must not mark the document changed.
///
/// The tick is what `apply_level_change` re-diffs on and what
/// `panel::rebuild` respawns on, so an edit that writes nothing and stamps
/// it anyway rebuilds the world and throws the panel away for nothing. A
/// value box commits when it loses focus, so clicking from one box to the
/// next commits the first with the value it was seeded with — and the
/// respawn would despawn the box just clicked into, along with the caret
/// the click had put there.
#[test]
fn a_no_op_edit_does_not_mark_the_document_changed() {
    let mut app = app();
    let was = app.world().resource::<LevelDef>().materials[0].id();

    // Something landed: the tick has to move, or this test would pass
    // against a document nothing can write to at all.
    app.world_mut().resource_mut::<Pending>().0.push(Edit {
        root: 0,
        path: ".materials[0].id".into(),
        value: Value::Num(f64::from(was) + 1.0, Num::U32),
    });
    app.update();
    let after_real = app
        .world()
        .get_resource_ref::<LevelDef>()
        .unwrap()
        .last_changed();

    // And now the same value it already holds, which is what a box echoes
    // back on its way out.
    let now = app.world().resource::<LevelDef>().materials[0].id();
    app.world_mut().resource_mut::<Pending>().0.push(Edit {
        root: 0,
        path: ".materials[0].id".into(),
        value: Value::Num(f64::from(now), Num::U32),
    });
    app.update();
    assert_eq!(
        app.world()
            .get_resource_ref::<LevelDef>()
            .unwrap()
            .last_changed(),
        after_real,
        "a no-op edit stamped the change tick, so the world re-diffs and \
         the panel respawns for an edit that wrote nothing"
    );
}

/// nothing and a future that cannot be redone — and the commit on the way
/// out of a box makes it reachable with a click: focus a name box, click
/// away, lose your redo.
#[test]
fn a_no_op_rename_costs_no_step_and_leaves_the_redo_alone() {
    let mut app = app();
    app.add_systems(Update, voxel_editor::undo);
    let (at, name) = a_named_node(&app);
    let path = format!(".nodes[{at}].name.0");

    text_edit(&mut app, &path, "renamed");
    assert_eq!(name_of(&app, at).as_deref(), Some("renamed"));

    app.world_mut().resource_mut::<EditorState>().undo = true;
    app.update();
    assert_eq!(name_of(&app, at), Some(name.clone()), "undo put it back");
    let history = app.world().resource::<voxel_editor::History>();
    assert!(history.can_redo(), "and left a redo to reach for");
    assert!(!history.can_undo(), "with nothing behind it");

    // The blur a click away from an untouched name box produces.
    text_edit(&mut app, &path, &name);
    let history = app.world().resource::<voxel_editor::History>();
    assert!(
        history.can_redo(),
        "a no-op must not throw the redo stack away"
    );
    assert!(!history.can_undo(), "nor be a step of its own");
}

/// And the other half of that distinction: a REAL rename is a step even
/// when no wire happened to say the old name.
///
/// The two are the same wire count, so anything that told them apart by
/// the count alone would take this away with the bug it was fixing.
#[test]
fn a_rename_no_wire_referred_to_is_still_a_step() {
    let mut app = app();
    let (at, name) = {
        let level = app.world().resource::<LevelDef>();
        level
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| Some((i, n.name.clone()?)))
            .find(|(_, name)| {
                !level.nodes.iter().any(|n| {
                    n.wires
                        .iter()
                        .any(|(_, w)| w.sources().iter().any(|s| s == name))
                })
            })
            .expect("planet names a node nothing is wired to")
    };
    assert_ne!(name, "unreferenced");

    text_edit(&mut app, &format!(".nodes[{at}].name.0"), "unreferenced");

    assert_eq!(name_of(&app, at).as_deref(), Some("unreferenced"));
    assert!(
        app.world().resource::<voxel_editor::History>().can_undo(),
        "a rename nothing followed still changed the document"
    );
}

/// A press on something that takes no focus still ends the typing.
///
/// The text input, the checkbox and the menu are the only widgets in this
/// stack that take focus when pressed, so a slider, a scrub label, the tab
/// strip or the resize grip left the box focused and what was typed in it
/// uncommitted. That is the silent discard "leaving the box commits" is
/// there to rule out.
#[test]
fn a_press_outside_the_box_commits_what_is_in_it() {
    use bevy::input_focus::InputFocus;

    let mut app = box_holding(".lod.split_k", Some(Num::F64), "0.75");
    // A slider, as far as focus is concerned: something with no editable
    // text anywhere above it.
    let elsewhere = app.world_mut().spawn_empty().id();
    press(&mut app, elsewhere);
    app.update();
    assert_eq!(queued(&app).len(), 1, "the press committed the box");
    assert!(
        app.world().resource::<InputFocus>().get().is_none(),
        "and let it go, so the next keystroke does not land in it"
    );

    // Pressing your own text moves the caret; it is not leaving the box.
    let mut app = box_holding(".lod.split_k", Some(Num::F64), "0.75");
    let input = app.world().resource::<InputFocus>().get().unwrap();
    press(&mut app, input);
    app.update();
    assert!(queued(&app).is_empty(), "still typing");
    assert_eq!(app.world().resource::<InputFocus>().get(), Some(input));
}

/// Focus follows the FIELD through the rebuild that despawns its box.
///
/// A commit and the click that caused it land in the same frame — focus is
/// set during picking in `PreUpdate`, and the commit, the apply and the
/// rebuild all run in `Update` behind it — so the box the click was FOR is
/// despawned before a keystroke can reach it. The path outlives the
/// rebuild; the entity does not.
#[test]
fn focus_follows_the_field_through_a_rebuild() {
    use bevy::ecs::system::RunSystemOnce;
    use bevy::input_focus::InputFocus;
    use bevy::text::EditableText;
    use voxel_editor::{FieldPath, WritesNum};

    let path = ".lod.split_k";
    let mut app = box_holding(path, Some(Num::F64), "0.75");
    let was = app.world().resource::<InputFocus>().get().unwrap();

    // What a rebuild does: every widget of the panel, gone and made again.
    let container = app.world().get::<ChildOf>(was).unwrap().parent();
    app.world_mut().entity_mut(container).despawn();
    let container = app
        .world_mut()
        .spawn((FieldPath(path.into()), WritesNum(Num::F64)))
        .id();
    let now = app
        .world_mut()
        .spawn((EditableText::new("0.75"), ChildOf(container)))
        .id();

    app.world_mut()
        .run_system_once(voxel_editor::refocus)
        .unwrap();
    assert_eq!(
        app.world().resource::<InputFocus>().get(),
        Some(now),
        "the box at the same path took the focus back"
    );

    // And a path with no box left — a collapsed section, a shut panel —
    // leaves focus on nothing rather than on a despawned entity.
    app.world_mut().entity_mut(container).despawn();
    app.world_mut()
        .run_system_once(voxel_editor::refocus)
        .unwrap();
    assert_eq!(app.world().resource::<InputFocus>().get(), None);
}

/// Press `at`, the way the picking backend does.
fn press(app: &mut App, at: Entity) {
    use bevy::picking::events::{Pointer, PointerPress};
    use bevy::picking::pointer::{Location, PointerButton, PointerId};

    let window = app.world_mut().spawn(bevy::window::Window::default()).id();
    let location = Location {
        target: bevy::window::WindowRef::Entity(window)
            .normalize(None)
            .unwrap()
            .into(),
        position: Vec2::ZERO,
    };
    app.world_mut().trigger(PointerPress {
        entity: at,
        pointer: Pointer::new(PointerId::Mouse, location),
        button: PointerButton::Primary,
        hit: bevy::picking::backend::HitData::new(Entity::PLACEHOLDER, 0.0, None, None),
        count: 1,
    });
}

/// The panel knows whether there is work not yet on disk.
#[test]
fn an_edit_marks_the_document_unsaved_and_a_save_clears_it() {
    use bevy::input::ButtonInput;
    let mut app = app();
    app.add_message::<voxel_editor::SaveRequested>()
        .init_resource::<ButtonInput<KeyCode>>()
        .add_systems(Update, voxel_editor::save);

    assert!(!app.world().resource::<EditorState>().edited);
    edit(&mut app, ".materials[0].id", 12.0, Num::U32);
    assert!(
        app.world().resource::<EditorState>().edited,
        "an edit is unsaved work"
    );

    app.world_mut().resource_mut::<EditorState>().open = true;
    app.world_mut().resource_mut::<EditorState>().save = true;
    app.update();
    assert!(
        !app.world().resource::<EditorState>().edited,
        "and asking to save clears it"
    );

    // An edit that changes nothing is not work.
    edit(&mut app, ".materials[0].id", 12.0, Num::U32);
    assert!(!app.world().resource::<EditorState>().edited);
}

/// Hovering lights a node box, and leaving it puts the border back —
/// without touching a box that is selected or blamed, which are saying
/// something a hover does not.
#[test]
fn hovering_lights_a_box_but_never_overrides_what_it_is_saying() {
    use bevy::picking::hover::Hovered;
    use voxel_editor::{SelectsNode, PLAIN_BORDER};

    let mut app = app();
    app.add_systems(Update, voxel_editor::hover);

    let plain = app
        .world_mut()
        .spawn((
            SelectsNode(".nodes[1]".into()),
            Hovered(false),
            BorderColor::all(PLAIN_BORDER),
        ))
        .id();
    let selected_colour = Color::srgb(0.30, 0.55, 0.92);
    let selected = app
        .world_mut()
        .spawn((
            SelectsNode(".nodes[2]".into()),
            Hovered(false),
            BorderColor::all(selected_colour),
        ))
        .id();

    let border = |app: &App, e: Entity| app.world().entity(e).get::<BorderColor>().unwrap().left;

    app.world_mut().entity_mut(plain).insert(Hovered(true));
    app.world_mut().entity_mut(selected).insert(Hovered(true));
    app.update();
    assert_ne!(border(&app, plain), PLAIN_BORDER, "a plain box lights up");
    assert_eq!(
        border(&app, selected),
        selected_colour,
        "a selected one keeps what it was saying"
    );

    app.world_mut().entity_mut(plain).insert(Hovered(false));
    app.update();
    assert_eq!(border(&app, plain), PLAIN_BORDER, "and it goes back");
}
