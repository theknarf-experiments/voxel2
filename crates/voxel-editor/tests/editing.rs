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
    LevelDef::from_json_known(
        &std::fs::read_to_string(path).unwrap(),
        &registry::engine_kinds(),
    )
    .unwrap()
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

    // A textual one: the node a port is wired to.
    let (at, port) = app
        .world()
        .resource::<LevelDef>()
        .nodes
        .iter()
        .enumerate()
        .find_map(|(i, n)| Some((i, n.wires.iter().next()?.0.clone())))
        .expect("planet wires its nodes");
    let name = app.world().resource::<LevelDef>().nodes[0]
        .name
        .clone()
        .expect("the first node is named");
    let item = app
        .world_mut()
        .spawn(PicksOption {
            path: format!(".nodes[{at}].wires.0{{{port}}}.0"),
            value: name.clone(),
            num: None,
        })
        .id();
    app.world_mut()
        .trigger(bevy::ui_widgets::Activate { entity: item });
    app.update();
    let wired = app.world().resource::<LevelDef>().nodes[at]
        .wires
        .get(&port)
        .and_then(|w| w.sources().first().cloned());
    assert_eq!(wired.as_deref(), Some(name.as_str()), "a wire is rewired");
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
