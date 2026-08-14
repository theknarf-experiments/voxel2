use super::*;
use bevy::prelude::*;
use bevy_graph_view::{layout, GraphStyle, Placed};
use voxel_engine::graph::{registry, with_registry};
use voxel_engine::LevelDef;

fn parse(json: &str) -> Vec<NodeDef> {
    with_registry(&registry::engine_kinds(), || serde_json::from_str(json)).unwrap()
}

fn shipped(name: &str) -> LevelDef {
    let path = format!("{}/../../levels/{name}.json", env!("CARGO_MANIFEST_DIR"));
    LevelDef::from_path_known(std::path::Path::new(&path), &registry::engine_kinds()).unwrap()
}

/// The document is a node too: one box, no ports, above the graph it
/// heads, and addressed by the root path so selecting it inspects the
/// level itself.
#[test]
fn the_level_is_a_node_in_its_own_graph() {
    let nodes = parse(r#"[{"kind":"height_zero","name":"sea"}]"#);
    let style = GraphStyle::default();
    let out = layout(&graph_of(&nodes), Some(&head()), &style);

    let level = out
        .nodes
        .iter()
        .find(|p| p.id == LEVEL)
        .expect("the level has a box");
    assert_eq!(level.kind, "level");
    assert!(level.ins.is_empty() && level.outs.is_empty());
    let sea = out
        .nodes
        .iter()
        .find(|p| p.name.as_deref() == Some("sea"))
        .unwrap();
    assert!(
        level.at.y + level.size.y <= sea.at.y,
        "the level heads the graph: {:?} vs {:?}",
        level.at,
        sea.at
    );
}

/// A scope's children get ids under `.node.nodes`, and the frame contains
/// them — the conversion preserves what the layout needs.
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
    let out = layout(&graph_of(&nodes), None, &style);

    assert_eq!(out.frames.len(), 1, "one frame per scope");
    let frame = &out.frames[0];
    assert_eq!(frame.name.as_deref(), Some("district"));

    let inside: Vec<&Placed> = out
        .nodes
        .iter()
        .filter(|p| p.id.contains(".node.nodes"))
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
        let out = layout(&graph_of(&level.nodes), Some(&head()), &style);
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
                    a.name.as_deref().unwrap_or(&a.kind),
                    b.name.as_deref().unwrap_or(&b.kind),
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
    fn wires(nodes: &[NodeDef]) -> usize {
        nodes
            .iter()
            .map(|n| {
                n.wires
                    .iter()
                    .map(|(_, w)| w.sources().len())
                    .sum::<usize>()
                    + wires(n.node.0.children())
            })
            .sum()
    }
    let style = GraphStyle::default();
    for name in ["planet", "megastructure", "purgatory"] {
        let level = shipped(name);
        let out = layout(&graph_of(&level.nodes), Some(&head()), &style);
        assert_eq!(
            out.edges.len(),
            wires(&level.nodes),
            "{name}: a wire with no line"
        );
    }
}
