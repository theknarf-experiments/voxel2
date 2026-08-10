//! What the walk makes of a real level.
//!
//! The panel is hard to assert on and easy to misread — four sliders all
//! showing the same number look like a walk bug and were a widget bug.
//! These pin the half that is testable: the value, the type, the bounds
//! and the reference list each row is built with.

use bevy::platform::collections::HashSet;
use voxel_editor::{rows, Num, RowKind};
use voxel_engine::LevelDef;

fn planet() -> LevelDef {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../levels/planet.json");
    LevelDef::from_json(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn open(paths: &[&str]) -> HashSet<String> {
    paths.iter().map(|p| p.to_string()).collect()
}

fn row_at<'a>(rows: &'a [voxel_editor::Row], path: &str) -> &'a voxel_editor::Row {
    rows.iter()
        .find(|r| r.path == path)
        .unwrap_or_else(|| panic!("no row at '{path}'"))
}

#[test]
fn a_number_row_carries_the_levels_own_value_and_type() {
    let level = planet();
    let rows = rows(&level, &open(&[".lod"]));

    let RowKind::Number { value, num, .. } = row_at(&rows, ".lod.max_level").kind else {
        panic!("max_level is a number")
    };
    assert_eq!(value, level.lod.max_level as f64);
    assert_eq!(num, Num::U8);

    let RowKind::Number { value, num, .. } = row_at(&rows, ".lod.split_k").kind else {
        panic!("split_k is a number")
    };
    assert_eq!(value, level.lod.split_k);
    assert_eq!(num, Num::F64);
}

/// No `schema::Range` may exclude a value a shipped level already holds.
///
/// A slider clamps silently: the row then shows a number the document does
/// not contain, and moving it writes that wrong number back. `max_level`
/// was annotated `1..=12` against a planet that has always been 14, which
/// is why this checks the whole class rather than that one field —
/// bounds are easy to invent and only the levels know if they are true.
#[test]
fn no_declared_range_excludes_a_value_a_level_ships() {
    for name in ["planet", "megastructure", "purgatory"] {
        let path = format!(
            "{}/../../levels/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let level = LevelDef::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for row in rows(&level, &everything(&level)) {
            let RowKind::Number {
                value,
                range: Some(range),
                ..
            } = row.kind
            else {
                continue;
            };
            assert!(
                (range.0 as f64) <= value && value <= (range.1 as f64),
                "{name}: '{}' is {value}, outside its declared {range:?}",
                row.path
            );
        }
    }
}

/// Every path in the level, found by opening what the last pass revealed
/// until nothing new appears.
fn everything(level: &LevelDef) -> HashSet<String> {
    let mut set: HashSet<String> = HashSet::default();
    loop {
        let found: HashSet<String> = rows(level, &set)
            .iter()
            .filter(|r| matches!(r.kind, RowKind::Group { .. } | RowKind::Variant { .. }))
            .map(|r| r.path.clone())
            .collect();
        if found.is_subset(&set) {
            return set;
        }
        set.extend(found);
    }
}

/// Only expanded paths are walked. The panel costs the rows on screen,
/// not the several thousand a level contains.
#[test]
fn nothing_is_walked_that_is_not_open() {
    let level = planet();
    let shut = rows(&level, &HashSet::default());
    assert_eq!(shut.len(), 7, "just the sections");
    assert!(rows(&level, &open(&[".lod"])).len() > shut.len());
}

/// The node vocabulary is an enum, so the palette is `EnumInfo` and needs
/// no registration list to go stale.
#[test]
fn a_node_row_offers_every_kind_in_the_vocabulary() {
    let level = planet();
    let rows = rows(&level, &open(&[".nodes", ".nodes[4]"]));
    let RowKind::Variant { options, current, .. } = &row_at(&rows, ".nodes[4].kind").kind else {
        panic!("a node kind is an enum")
    };
    assert!(options.len() > 10, "the whole op set: {options:?}");
    assert!(options.contains(&"HeightFbm".to_string()));
    assert!(options.contains(current), "the current variant is offered");
}

/// A material id is a choice among the ids this level defines, not a
/// number — the reason the schema has references at all.
#[test]
fn a_material_field_offers_the_levels_own_ids() {
    let level = planet();
    // Whichever op has one: which op sits where is content.
    let rows = rows(&level, &everything(&level));
    let choice = rows.iter().find_map(|r| match &r.kind {
        RowKind::Choice {
            options, current, ..
        } if r.label == "material" => Some((options.clone(), current.clone())),
        _ => None,
    });
    let Some((options, current)) = choice else {
        panic!("no op in planet paints a material")
    };
    let ids: Vec<String> = level.materials.iter().map(|m| m.id().to_string()).collect();
    assert_eq!(options, ids, "the options are the level's material ids");
    assert!(options.contains(&current), "the op paints a defined material");
}

/// `AsColor` reaches the leaves, so a pair of hues is two swatches and not
/// six numbers.
#[test]
fn a_colour_pair_is_two_swatches() {
    let level = planet();
    let at = level
        .materials
        .iter()
        .position(|m| matches!(m, voxel_engine::level::MaterialDef::Zoned { .. }))
        .expect("planet ships a zoned material");
    let rows = rows(
        &level,
        &open(&[
            ".materials",
            &format!(".materials[{at}]"),
            &format!(".materials[{at}].mid"),
        ]),
    );
    assert!(matches!(
        row_at(&rows, &format!(".materials[{at}].low")).kind,
        RowKind::Color(_)
    ));
    assert!(matches!(
        row_at(&rows, &format!(".materials[{at}].mid[0]")).kind,
        RowKind::Color(_)
    ));
}

/// `Rebuilds` is inherited: every number inside a section that restreams
/// the world restreams it too, and its widget must commit on release.
#[test]
fn rebuilds_reaches_the_numbers_inside_a_restreaming_section() {
    let level = planet();
    let rows = rows(&level, &open(&[".nodes", ".nodes[4]", ".nodes[4].kind"]));
    assert!(row_at(&rows, ".nodes[4].kind.amp").rebuilds);
    // And does not leak sideways into the cheap sections.
    let rows = rows_of_materials(&level);
    assert!(!rows.iter().any(|r| r.rebuilds), "a material is a table upload");
}

fn rows_of_materials(level: &LevelDef) -> Vec<voxel_editor::Row> {
    rows(level, &open(&[".materials", ".materials[0]"]))
        .into_iter()
        .filter(|r| r.path.starts_with(".materials"))
        .collect()
}
