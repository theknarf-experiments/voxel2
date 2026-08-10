//! What the walk makes of a real level.
//!
//! The panel is hard to assert on and easy to misread — four sliders all
//! showing the same number look like a walk bug and were a widget bug.
//! These pin the half that is testable: the value, the type, the bounds
//! and the reference list each row is built with.

use bevy::platform::collections::HashSet;
use bevy::reflect::Typed;
use voxel_editor::{rows, Num, RowKind};
use voxel_engine::graph::registry;
use voxel_engine::LevelDef;

fn planet() -> LevelDef {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../levels/planet.json");
    LevelDef::from_json_known(
        &std::fs::read_to_string(path).unwrap(),
        &registry::engine_kinds(),
    )
    .unwrap()
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
        let path = format!("{}/../../levels/{name}.json", env!("CARGO_MANIFEST_DIR"));
        let level = LevelDef::from_json_known(
            &std::fs::read_to_string(&path).unwrap(),
            &registry::engine_kinds(),
        )
        .unwrap();
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
    // Counted off the type rather than written down: what a level's
    // sections ARE is the schema's business, and a hard number here fails
    // the day one is added for no reason a reader could learn from it.
    let sections = match LevelDef::type_info() {
        bevy::reflect::TypeInfo::Struct(s) => s.field_len(),
        _ => unreachable!("LevelDef is a struct"),
    };
    assert_eq!(shut.len(), sections, "just the sections");
    assert!(rows(&level, &open(&[".lod"])).len() > shut.len());
}

/// The vocabulary is the type registry, not an enum.
///
/// So the palette cannot go stale against a list somebody forgot to
/// extend, and a host's kinds appear in it without this crate — or the
/// engine — naming them.
#[test]
fn the_vocabulary_is_whatever_is_registered() {
    let kinds = registry::engine_kinds();
    let names: Vec<&str> = {
        let reg = kinds.read();
        registry::kinds(&reg).into_iter().map(|(k, _)| k).collect()
    };
    assert!(names.len() > 15, "the whole shipped set: {names:?}");
    for expected in ["height_fbm", "region", "sdf_void", "shafts_cut"] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }
}

/// A node walks as the struct it IS: the editor reaches its fields through
/// the boxed trait object, so opening one shows `HeightFbm`'s parameters
/// and not a wrapper.
#[test]
fn a_node_walks_as_its_own_type() {
    let level = planet();
    let rows = rows(&level, &open(&[".nodes", ".nodes[4]", ".nodes[4].node"]));
    let node = row_at(&rows, ".nodes[4].node");
    assert!(
        matches!(node.kind, RowKind::Group { .. }),
        "a node is a struct, not an opaque value"
    );
    assert!(
        rows.iter().any(|r| r.path == ".nodes[4].node.amp"),
        "its own fields are reachable"
    );
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
    assert!(
        options.contains(&current),
        "the op paints a defined material"
    );
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
    let rows = rows(&level, &open(&[".nodes", ".nodes[4]", ".nodes[4].node"]));
    assert!(row_at(&rows, ".nodes[4].node.amp").rebuilds);
    // And does not leak sideways into the cheap sections.
    let rows = rows_of_materials(&level);
    assert!(
        !rows.iter().any(|r| r.rebuilds),
        "a material is a table upload"
    );
}

fn rows_of_materials(level: &LevelDef) -> Vec<voxel_editor::Row> {
    rows(level, &open(&[".materials", ".materials[0]"]))
        .into_iter()
        .filter(|r| r.path.starts_with(".materials"))
        .collect()
}

/// A node names itself in the list, by what a level calls it and what it
/// IS. Fifty-five rows of an index is a document you cannot read.
#[test]
fn a_node_is_labelled_by_its_name_and_its_kind() {
    let level = planet();
    let rows = rows(&level, &open(&[".nodes"]));
    let labels: Vec<&str> = rows
        .iter()
        .filter(|r| r.path.starts_with(".nodes["))
        .map(|r| r.label.as_str())
        .collect();
    assert!(labels.len() > 20, "the whole list: {labels:?}");
    // The kind is the type behind the box, so a row says what the level
    // wrote in `"kind"` and not that it is a node.
    assert!(
        labels.iter().any(|l| l.contains("HeightFbm")),
        "no kind in {labels:?}"
    );
    // And the name it declared, which is what everything else refers to.
    let named = level
        .nodes
        .iter()
        .find_map(|n| n.name.clone())
        .expect("planet names its nodes");
    assert!(
        labels.iter().any(|l| l.contains(&named)),
        "'{named}' missing from {labels:?}"
    );
}

/// A port is a reference, and the panel offers what it can refer to: the
/// same names the compiler resolves, at any depth.
#[test]
fn a_wire_offers_the_levels_own_node_names() {
    let level = planet();
    // Whichever node has wires — which one does is content.
    let at = level
        .nodes
        .iter()
        .position(|n| !n.wires.is_empty())
        .expect("planet wires its nodes");
    let rows = rows(
        &level,
        &open(&[
            ".nodes",
            &format!(".nodes[{at}]"),
            &format!(".nodes[{at}].wires.0"),
        ]),
    );
    let wire = rows
        .iter()
        .find(|r| r.path.starts_with(&format!(".nodes[{at}].wires.0{{")))
        .expect("a row per port");
    let RowKind::Choice {
        current, options, ..
    } = &wire.kind
    else {
        panic!("a wire is a reference, not {:?}", wire.path)
    };
    assert!(
        options.contains(current),
        "'{current}' is wired to something the menu does not offer: {options:?}"
    );
    assert!(options.len() > 10, "every node in the level: {options:?}");
}

/// A newtype costs no row. `Wires` and the demo's `Population` are
/// spellings Rust needs; a reader looking for a port should not have to
/// open "1 fields" to find one.
#[test]
fn a_newtype_does_not_cost_a_row() {
    let level = planet();
    let at = level
        .nodes
        .iter()
        .position(|n| !n.wires.is_empty())
        .expect("planet wires its nodes");
    let rows = rows(&level, &open(&[".nodes", &format!(".nodes[{at}]")]));
    let wires = row_at(&rows, &format!(".nodes[{at}].wires.0"));
    assert!(
        matches!(wires.kind, RowKind::Group { .. }),
        "the wires row IS the map"
    );
    assert_eq!(wires.label, "wires", "and keeps the field's own name");
}

/// Every reference in a shipped level resolves to something the level
/// actually has.
///
/// The guard that was missing: a `OneOf` pattern is a string, and one
/// naming a section that has been renamed or removed degrades to an empty
/// menu and a warning nobody reads. `"stack[].name"` outlived the section
/// it named by three commits.
#[test]
fn every_reference_a_level_makes_resolves() {
    for name in ["planet", "megastructure", "purgatory"] {
        let path = format!("{}/../../levels/{name}.json", env!("CARGO_MANIFEST_DIR"));
        let level = LevelDef::from_json_known(
            &std::fs::read_to_string(&path).unwrap(),
            &registry::engine_kinds(),
        )
        .unwrap();
        let mut seen = 0;
        for row in rows(&level, &everything(&level)) {
            let RowKind::Choice {
                current, options, ..
            } = &row.kind
            else {
                continue;
            };
            seen += 1;
            assert!(
                options.contains(current),
                "{name}: '{}' is {current:?}, which is not among {options:?}",
                row.path
            );
        }
        assert!(seen > 0, "{name} makes no references at all");
    }
}

/// A tab is a VIEW of a document, not a document: both halves of the split
/// read the same resource, the same change tick and the same paths.
#[test]
fn a_tab_shows_the_sections_it_asked_for() {
    use voxel_editor::{rows_in, Sections};
    let level = planet();
    let shut = bevy::platform::collections::HashSet::default();

    let nodes = Sections::Only(vec!["nodes".into()]);
    let rest = Sections::Except(vec!["nodes".into()]);
    let labels = |s: &Sections| -> Vec<String> {
        rows_in(&level, &shut, s)
            .iter()
            .map(|r| r.label.clone())
            .collect()
    };

    // A tab naming one section shows that section's CONTENTS, so this is
    // the node list itself rather than a row called `nodes`.
    assert!(labels(&nodes).len() > 20, "{:?}", labels(&nodes));
    assert!(rows_in(&level, &shut, &nodes)
        .iter()
        .all(|r| r.path.starts_with(".nodes[")));
    let rest_labels: Vec<String> = labels(&rest);
    assert!(!rest_labels.iter().any(|l| l == "nodes"), "{rest_labels:?}");
    assert!(
        rest_labels.iter().any(|l| l == "materials"),
        "{rest_labels:?}"
    );

    // Between them they cover the document: `except` is what makes a
    // section added to the level land in a tab without anyone naming it.
    let all: Vec<String> = rows(&level, &shut)
        .iter()
        .map(|r| r.label.clone())
        .collect();
    for section in &all {
        let shown = section == "nodes" || rest_labels.contains(section);
        assert!(shown, "'{section}' is in no tab");
    }
}

/// The filter is the TAB's, not a rule about the name: a field called
/// `nodes` inside a section is still a field.
#[test]
fn a_section_filter_applies_only_at_the_top() {
    use voxel_editor::{rows_in, Sections};
    let level = planet();
    // A `region` scope holds a `nodes` field of its own.
    let at = level
        .nodes
        .iter()
        .position(|n| !n.node.0.children().is_empty());
    let Some(at) = at else { return };
    let open = open(&[
        ".nodes",
        &format!(".nodes[{at}]"),
        &format!(".nodes[{at}].node"),
    ]);
    let rows = rows_in(&level, &open, &Sections::Except(vec!["nodes".into()]));
    assert!(
        rows.iter().any(|r| r.path.ends_with(".node.nodes")),
        "a scope's own `nodes` must survive a tab that hides the section"
    );
}
