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
