use super::*;
use voxel_engine::graph::{registry, with_registry};
use voxel_engine::LevelDef;

fn parse(json: &str) -> Vec<NodeDef> {
    with_registry(&registry::engine_kinds(), || serde_json::from_str(json)).unwrap()
}

fn shipped(name: &str) -> LevelDef {
    let path = format!("{}/../../levels/{name}.json", env!("CARGO_MANIFEST_DIR"));
    LevelDef::from_json_known(
        &std::fs::read_to_string(path).unwrap(),
        &registry::engine_kinds(),
    )
    .unwrap()
}

/// A column is dependency DEPTH, so a chain reads left to right and two
/// things that depend on nothing start side by side.
#[test]
fn a_chain_walks_right_and_siblings_stack() {
    let nodes = parse(
        r#"[
          {"kind":"height_zero","name":"sea"},
          {"kind":"warp_none","name":"flat"},
          {"kind":"height_fbm","name":"a","in":{"height":"sea","warp":"flat"},
            "scale":5e-5,"amp":800,"octaves":3},
          {"kind":"height_fbm","name":"b","in":{"height":"a","warp":"flat"},
            "scale":5e-5,"amp":80,"octaves":3}
        ]"#,
    );
    let style = GraphStyle::default();
    let out = layout(&nodes, &style);
    let at = |name: &str| {
        out.nodes
            .iter()
            .find(|p| p.name.as_deref() == Some(name))
            .unwrap()
            .at
    };

    // Two origins, no inputs: same column, stacked.
    assert_eq!(at("sea").x, at("flat").x);
    assert!(at("flat").y > at("sea").y, "siblings stack");
    // Each consumer sits one column right of the furthest thing it reads.
    assert!(at("a").x > at("sea").x);
    assert!(at("b").x > at("a").x);
}

/// Every wire is drawn, and it lands on the port it names rather than on
/// the middle of a box.
#[test]
fn a_wire_lands_on_the_port_it_names() {
    let nodes = parse(
        r#"[
          {"kind":"height_zero","name":"sea"},
          {"kind":"warp_none","name":"flat"},
          {"kind":"height_fbm","name":"a","in":{"height":"sea","warp":"flat"},
            "scale":5e-5,"amp":800,"octaves":3}
        ]"#,
    );
    let style = GraphStyle::default();
    let out = layout(&nodes, &style);
    assert_eq!(out.edges.len(), 2, "one per wired source");

    let a = out
        .nodes
        .iter()
        .find(|p| p.name.as_deref() == Some("a"))
        .unwrap();
    let height = out.edges.iter().find(|e| e.port == "height").unwrap();
    let warp = out.edges.iter().find(|e| e.port == "warp").unwrap();
    let sea = out
        .nodes
        .iter()
        .find(|p| p.name.as_deref() == Some("sea"))
        .unwrap();

    // Every run is axis-aligned — a diagonal would have to be a rotated
    // node, which Bevy draws in pieces.
    for edge in &out.edges {
        for seg in &edge.segments {
            assert!(
                seg.size.x.min(seg.size.y) <= 2.0,
                "{:?} is not a line: {seg:?}",
                edge.port
            );
        }
    }
    // The first run leaves the producer's right edge; the last arrives at
    // the consumer's left one, on the port's own row.
    let ends = |e: &super::Edge| {
        let (first, last) = (e.segments.first().unwrap(), e.segments.last().unwrap());
        (first.at, last.at + last.size)
    };
    let (from, to) = ends(height);
    assert!((from.x - (sea.at.x + sea.size.x)).abs() <= 2.0, "{from:?}");
    assert!((to.x - a.at.x).abs() <= 2.0, "{to:?}");
    assert_ne!(to.y, ends(warp).1.y, "a port is a row, not a box");
}

/// A scope is a frame around its children, and the frame contains them.
#[test]
fn a_scope_frames_what_it_gates() {
    let nodes = parse(
        r#"[
          {"kind":"sdf_void","name":"void"},
          {"kind":"region","name":"district","axes":[0.0,0.4,0.0,1.0],"nodes":[
            {"kind":"lattice_y","name":"floors","spacing":4.0},
            {"kind":"coarse_solid","name":"solid","in":{"sdf":"void"},"material":2}
          ]}
        ]"#,
    );
    let style = GraphStyle::default();
    let out = layout(&nodes, &style);

    assert_eq!(out.frames.len(), 1, "one frame per scope");
    let frame = &out.frames[0];
    assert_eq!(frame.name.as_deref(), Some("district"));

    let inside: Vec<&Placed> = out
        .nodes
        .iter()
        .filter(|p| p.path.contains(".node.nodes"))
        .collect();
    assert_eq!(inside.len(), 2, "the district's own nodes");
    for node in inside {
        assert!(
            node.at.x >= frame.at.x
                && node.at.y >= frame.at.y
                && node.at.x + node.size.x <= frame.at.x + frame.size.x
                && node.at.y + node.size.y <= frame.at.y + frame.size.y,
            "'{:?}' escapes its frame: {:?} in {:?}",
            node.name,
            (node.at, node.size),
            (frame.at, frame.size)
        );
    }
    // A scope is not a node box: it is drawn, not placed among them.
    assert!(out.nodes.iter().all(|p| p.kind != "region"));
}

/// Nothing overlaps anything, on every level this game ships.
///
/// The property a hand-checked screenshot cannot give: two boxes drawn on
/// top of each other read as one node with the wrong contents.
#[test]
fn no_two_boxes_overlap_in_any_shipped_level() {
    let style = GraphStyle::default();
    for name in ["planet", "megastructure", "purgatory"] {
        let level = shipped(name);
        let out = layout(&level.nodes, &style);
        assert!(!out.nodes.is_empty(), "{name} placed nothing");

        let overlap = |a: (Vec2, Vec2), b: (Vec2, Vec2)| {
            a.0.x < b.0.x + b.1.x
                && b.0.x < a.0.x + a.1.x
                && a.0.y < b.0.y + b.1.y
                && b.0.y < a.0.y + a.1.y
        };
        for (i, a) in out.nodes.iter().enumerate() {
            for b in &out.nodes[i + 1..] {
                assert!(
                    !overlap((a.at, a.size), (b.at, b.size)),
                    "{name}: {:?} and {:?} overlap",
                    a.name.as_deref().unwrap_or(a.kind),
                    b.name.as_deref().unwrap_or(b.kind),
                );
            }
        }
        // Frames may contain nodes, but never each other's children.
        for (i, a) in out.frames.iter().enumerate() {
            for b in &out.frames[i + 1..] {
                assert!(
                    !overlap((a.at, a.size), (b.at, b.size)),
                    "{name}: frames {:?} and {:?} overlap",
                    a.name,
                    b.name
                );
            }
        }
    }
}

/// Every wire in a shipped level finds both of its ends.
#[test]
fn every_wire_is_drawn() {
    let style = GraphStyle::default();
    for name in ["planet", "megastructure", "purgatory"] {
        let level = shipped(name);
        let out = layout(&level.nodes, &style);
        let wires: usize = super::flat(&level.nodes)
            .iter()
            .map(|(_, n)| {
                n.wires
                    .iter()
                    .map(|(_, w)| w.sources().len())
                    .sum::<usize>()
            })
            .sum();
        assert_eq!(out.edges.len(), wires, "{name}: a wire with no line");
    }
}

/// Zooming holds the point under the pointer still. Without that the
/// corner stays put and whatever the gesture was aimed at slides away.
#[test]
fn zooming_holds_the_point_it_is_aimed_at() {
    use crate::canvas::GraphCamera;
    let camera = GraphCamera {
        pan: Vec2::new(40.0, -10.0),
        zoom: 1.0,
    };
    let at = Vec2::new(300.0, 200.0);
    // The graph point under the pointer, before.
    let before = (at - camera.pan) / camera.zoom;
    for notches in [1.0, -1.0, 4.0, -7.5] {
        let zoomed = camera.zoomed(notches, Some(at));
        let after = (at - zoomed.pan) / zoomed.zoom;
        assert!(
            (after - before).length() < 1e-3,
            "{notches} notches moved the graph under the pointer: {before} -> {after}"
        );
    }
    // And it stays inside its range however hard you pinch.
    assert!(camera.zoomed(100.0, Some(at)).zoom <= 2.0);
    assert!(camera.zoomed(-100.0, Some(at)).zoom >= 0.2);
}
