use super::*;

/// A leaf node, wired by name.
fn node(id: &str, name: &str, ins: &[&str], outs: &[&str], wires: &[(&str, &str)]) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        name: Some(name.to_string()),
        kind: format!("kind_of_{name}"),
        ins: ins.iter().map(|s| s.to_string()).collect(),
        outs: outs.iter().map(|s| s.to_string()).collect(),
        wires: wires
            .iter()
            .map(|(port, source)| (port.to_string(), vec![source.to_string()]))
            .collect(),
        children: Vec::new(),
    }
}

/// A chain and two origins: sea and flat feed a, a feeds b.
fn chain() -> Vec<GraphNode> {
    vec![
        node("n0", "sea", &[], &["height"], &[]),
        node("n1", "flat", &[], &["warp"], &[]),
        node(
            "n2",
            "a",
            &["height", "warp"],
            &["height"],
            &[("height", "sea"), ("warp", "flat")],
        ),
        node(
            "n3",
            "b",
            &["height", "warp"],
            &["height"],
            &[("height", "a"), ("warp", "flat")],
        ),
    ]
}

/// A column is dependency DEPTH, so a chain reads left to right and two
/// things that depend on nothing start side by side.
#[test]
fn a_chain_walks_right_and_siblings_stack() {
    let style = GraphStyle::default();
    let out = layout(&chain(), None, &style);
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
    let style = GraphStyle::default();
    let nodes = &chain()[..3];
    let out = layout(nodes, None, &style);
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
    // node, which Bevy draws in pieces. A run's thin axis is the wire's
    // drawn thickness, which carries the oversample like everything else.
    let thick = style.effective().wire;
    for edge in &out.edges {
        for seg in &edge.segments {
            assert!(
                seg.size.x.min(seg.size.y) <= thick + 0.1,
                "{:?} is not a line: {seg:?}",
                edge.port
            );
        }
    }
    // The first run leaves the producer's right edge; the last arrives at
    // the consumer's left one, on the port's own row.
    let ends = |e: &Edge| {
        let (first, last) = (e.segments.first().unwrap(), e.segments.last().unwrap());
        (first.at, last.at + last.size)
    };
    let (from, to) = ends(height);
    assert!((from.x - (sea.at.x + sea.size.x)).abs() <= 2.0, "{from:?}");
    assert!((to.x - a.at.x).abs() <= 2.0, "{to:?}");
    assert_ne!(to.y, ends(warp).1.y, "a port is a row, not a box");
}

/// The document is a node too: one box, no ports, above the graph it
/// heads, and addressed by whatever id the host gave it.
#[test]
fn the_head_is_a_node_in_its_own_graph() {
    let style = GraphStyle::default();
    let head = GraphNode {
        id: String::new(),
        kind: "doc".to_string(),
        ..Default::default()
    };
    let out = layout(&chain()[..1], Some(&head), &style);

    let doc = out
        .nodes
        .iter()
        .find(|p| p.id.is_empty())
        .expect("the document has a box");
    assert_eq!(doc.kind, "doc");
    assert!(doc.ins.is_empty() && doc.outs.is_empty());
    let sea = out
        .nodes
        .iter()
        .find(|p| p.name.as_deref() == Some("sea"))
        .unwrap();
    assert!(
        doc.at.y + doc.size.y <= sea.at.y,
        "the head heads the graph: {:?} vs {:?}",
        doc.at,
        sea.at
    );
}

/// A scope is a frame around its children, and the frame contains them.
#[test]
fn a_scope_frames_what_it_gates() {
    let scope = GraphNode {
        id: "n1".to_string(),
        name: Some("district".to_string()),
        kind: "scope".to_string(),
        children: vec![
            node("n1/0", "floors", &[], &[], &[]),
            node("n1/1", "solid", &["sdf"], &[], &[("sdf", "void")]),
        ],
        ..Default::default()
    };
    let nodes = vec![node("n0", "void", &[], &["sdf"], &[]), scope];
    let style = GraphStyle::default();
    let out = layout(&nodes, None, &style);

    assert_eq!(out.frames.len(), 1, "one frame per scope");
    let frame = &out.frames[0];
    assert_eq!(frame.name.as_deref(), Some("district"));

    let inside: Vec<&Placed> = out
        .nodes
        .iter()
        .filter(|p| p.id.starts_with("n1/"))
        .collect();
    assert_eq!(inside.len(), 2, "the scope's own nodes");
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
    assert!(out.nodes.iter().all(|p| p.kind != "scope"));
    // And a wire crossing its boundary still finds both ends.
    assert_eq!(out.edges.len(), 1, "the wire into the scope is drawn");
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

/// The oversample scales the whole picture linearly and invisibly: a
/// layout at oversample 2 is the oversample-1 layout times two, so the
/// canvas transform dividing two back out shows the authored metrics.
#[test]
fn the_oversample_scales_the_picture_linearly() {
    let authored = GraphStyle {
        oversample: 1.0,
        ..Default::default()
    };
    let drawn = GraphStyle {
        oversample: 2.0,
        ..Default::default()
    };
    let flat = layout(&chain(), None, &authored);
    let big = layout(&chain(), None, &drawn);
    assert_eq!(flat.nodes.len(), big.nodes.len());
    for (a, b) in flat.nodes.iter().zip(&big.nodes) {
        assert!(
            (b.at - a.at * 2.0).length() < 1e-3 && (b.size - a.size * 2.0).length() < 1e-3,
            "{:?}: {:?}/{:?} is not twice {:?}/{:?}",
            a.name,
            b.at,
            b.size,
            a.at,
            a.size
        );
    }
    // And folding it twice changes nothing: effective() is idempotent.
    assert_eq!(drawn.effective().effective().font, drawn.effective().font);
}

/// A pinch is a smooth gesture, so the zoom must follow it smoothly —
/// and hold still at the stops.
#[test]
fn zoom_is_continuous_and_steady_at_the_stops() {
    use crate::canvas::GraphCamera;
    let camera = GraphCamera {
        pan: Vec2::ZERO,
        zoom: 1.0,
    };
    // A small gesture moves the zoom a little: not nothing, and not a
    // whole snapped step.
    let nudged = camera.zoomed(0.1, None).zoom;
    assert!(
        nudged > 1.005 && nudged < 1.02,
        "0.1 notches should move ~1%: {nudged}"
    );
    // Riding the gesture to a stop pins it there. Pushing PAST the stop
    // stays there too: the stop is not on the step lattice, and snapping
    // used to round the clamped 2.0 DOWN to 1.12^6 — zooming in at 200%
    // read 197%.
    let maxed = camera.zoomed(100.0, None);
    assert_eq!(maxed.zoom, 2.0);
    assert_eq!(maxed.zoomed(0.3, None).zoom, 2.0);
    let minned = camera.zoomed(-100.0, None);
    assert_eq!(minned.zoom, 0.2);
    assert_eq!(minned.zoomed(-0.3, None).zoom, 0.2);
    // And a small retreat from the stop is as smooth as anywhere else.
    let back = maxed.zoomed(-0.1, None).zoom;
    assert!(back > 1.97 && back < 2.0, "{back}");
}
