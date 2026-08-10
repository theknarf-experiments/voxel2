//! Where the boxes go: laying a node list out as a 2D graph.
//!
//! Pure arithmetic over [`NodeDef`], with no Bevy in it, because the part
//! of a graph view that can be WRONG is the geometry and the part that
//! needs a window is the drawing. Everything here is testable at a
//! terminal.
//!
//! Unlike the row walk, this is not generic over reflected documents: a
//! graph view needs ports, wires and scopes, which belong to
//! `voxel_engine::graph` and to nothing else. The rows remain the view
//! that works on any annotated type.
//!
//! **Boxes are a fixed size.** Layout is therefore decided here rather
//! than by flexbox, which means the edge anchors are known before anything
//! is spawned — no measuring a node, waiting a frame, and placing it on
//! the next one. It also means the picture does not move when a value
//! inside a node changes.

use bevy::prelude::*;
use voxel_engine::graph::NodeDef;

/// Sizes the layout is built from, in logical pixels.
///
/// A resource, and reflected, for the same reason [`crate::PanelStyle`] is:
/// the metrics of a view have no theme token to live in, and being able to
/// widen a node box on a running panel is the nearest thing to a
/// stylesheet.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource)]
pub struct GraphStyle {
    pub node_width: f32,
    /// The title bar of a node box.
    pub header: f32,
    /// One port row.
    pub port: f32,
    /// Between columns, and between rows within a column.
    pub gap: Vec2,
    /// Inside a scope's frame, around its children.
    pub frame_pad: f32,
    /// The title bar of a scope's frame.
    pub frame_header: f32,
}

impl Default for GraphStyle {
    fn default() -> Self {
        Self {
            node_width: 168.0,
            header: 18.0,
            port: 13.0,
            gap: Vec2::new(56.0, 12.0),
            frame_pad: 10.0,
            frame_header: 16.0,
        }
    }
}

/// The box that stands for the DOCUMENT itself, addressed by the empty
/// path — the reflect path of the root.
///
/// Everything in the picture is a node, including the level: selecting it
/// shows the sections that are not nodes, in the same column as any other
/// node's fields.
pub const LEVEL: &str = "";

/// One node, placed.
#[derive(Clone, Debug, PartialEq)]
pub struct Placed {
    /// Reflect path of the node, so a box can be clicked back to its row.
    pub path: String,
    /// What the level calls it, if anything.
    pub name: Option<String>,
    pub kind: &'static str,
    pub ins: Vec<&'static str>,
    pub outs: Vec<&'static str>,
    /// Top-left, in graph space.
    pub at: Vec2,
    pub size: Vec2,
}

impl Placed {
    /// Where an input port's wire lands: the left edge, one row per port
    /// under the header.
    pub fn in_anchor(&self, port: &str, style: &GraphStyle) -> Vec2 {
        self.anchor(self.ins.iter().position(|p| *p == port), 0.0, style)
    }

    /// Where an output port's wire leaves: the right edge.
    pub fn out_anchor(&self, port: &str, style: &GraphStyle) -> Vec2 {
        self.anchor(
            self.outs
                .iter()
                .position(|p| *p == port)
                .map(|i| i + self.ins.len()),
            self.size.x,
            style,
        )
    }

    fn anchor(&self, row: Option<usize>, x: f32, style: &GraphStyle) -> Vec2 {
        let row = row.unwrap_or(0) as f32;
        self.at + Vec2::new(x, style.header + style.port * (row + 0.5))
    }
}

/// A scope, drawn around what it gates.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub path: String,
    pub name: Option<String>,
    pub kind: &'static str,
    pub at: Vec2,
    pub size: Vec2,
}

/// One wire, as axis-aligned segments from one port to the other.
///
/// Right angles rather than a diagonal, for two reasons. A diagonal has to
/// be drawn as a ROTATED node, and Bevy clips a rotated node against an
/// axis-aligned box in the wrong space — a near-vertical wire came out as
/// a dashed line that crawled along itself as the graph panned, while
/// near-horizontal ones were fine. And a right-angled route is what a node
/// editor looks like: sixty crossing diagonals are a haystack.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    /// The port on the consuming end, for a label or a hover.
    pub port: String,
    pub segments: Vec<Seg>,
}

/// One axis-aligned run of a wire.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seg {
    pub at: Vec2,
    pub size: Vec2,
}

impl Edge {
    /// Route from an output port to an input one: out, across, in.
    ///
    /// The vertical run sits midway between the two boxes, so wires
    /// leaving one column share a lane instead of each cutting its own
    /// diagonal across everything between.
    fn route(from: Vec2, to: Vec2, port: String) -> Self {
        let thick = 1.5;
        let half = thick * 0.5;
        let mid = (from.x + to.x) * 0.5;
        let bar = |a: Vec2, b: Vec2| Seg {
            at: Vec2::new(a.x.min(b.x), a.y.min(b.y)) - Vec2::splat(half),
            size: Vec2::new((b.x - a.x).abs(), (b.y - a.y).abs()) + Vec2::splat(thick),
        };
        let mut segments = vec![bar(from, Vec2::new(mid, from.y))];
        if from.y != to.y {
            segments.push(bar(Vec2::new(mid, from.y), Vec2::new(mid, to.y)));
        }
        segments.push(bar(Vec2::new(mid, to.y), to));
        Self { port, segments }
    }
}

/// A whole level, placed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layout {
    pub nodes: Vec<Placed>,
    pub frames: Vec<Frame>,
    pub edges: Vec<Edge>,
    /// Bounding size of everything, for the canvas.
    pub size: Vec2,
}

/// Place every node of `nodes`, and the scopes around them.
///
/// Columns are dependency DEPTH: a node sits one column right of the
/// furthest thing it reads. Source order is already a valid topological
/// order — the compiler refuses a level where it is not — so one forward
/// pass settles every depth, and the picture agrees with the document
/// rather than with a sort nobody asked for.
pub fn layout(nodes: &[NodeDef], style: &GraphStyle) -> Layout {
    let mut out = Layout::default();
    let mut depths: bevy::platform::collections::HashMap<String, usize> = Default::default();
    place(nodes, "", style, &mut depths, &mut out);

    // The level's own box, above the graph it heads. Everything else moves
    // down to make room, which is cheaper than teaching `place` about a
    // node that is not in the list.
    let head = Vec2::new(0.0, style.header + style.port + style.gap.y);
    for node in &mut out.nodes {
        node.at += head;
    }
    for frame in &mut out.frames {
        frame.at += head;
    }
    out.nodes.push(Placed {
        path: LEVEL.to_string(),
        name: None,
        kind: "level",
        ins: Vec::new(),
        outs: Vec::new(),
        at: Vec2::ZERO,
        size: Vec2::new(style.node_width, style.header + style.port),
    });

    // Wires last: every box has to be placed before an anchor can be
    // asked for, and a wire may cross a scope boundary.
    let by_name: bevy::platform::collections::HashMap<String, usize> = out
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, p)| Some((p.name.clone()?, i)))
        .collect();
    let by_path: bevy::platform::collections::HashMap<String, usize> = out
        .nodes
        .iter()
        .enumerate()
        .map(|(i, p)| (p.path.clone(), i))
        .collect();
    let mut edges = Vec::new();
    for (path, node) in flat(nodes) {
        let Some(&to) = by_path.get(&path) else {
            continue;
        };
        for (port, wire) in node.wires.iter() {
            for source in wire.sources() {
                let Some(&from) = by_name.get(source) else {
                    continue;
                };
                let produced = out.nodes[from].outs.first().copied().unwrap_or("");
                edges.push(Edge::route(
                    out.nodes[from].out_anchor(produced, style),
                    out.nodes[to].in_anchor(port, style),
                    port.clone(),
                ));
            }
        }
    }
    out.edges = edges;

    let far = out
        .nodes
        .iter()
        .map(|p| p.at + p.size)
        .chain(out.frames.iter().map(|f| f.at + f.size))
        .fold(Vec2::ZERO, Vec2::max);
    out.size = far;
    out
}

/// Every node in the tree with its reflect path, scopes included.
fn flat(nodes: &[NodeDef]) -> Vec<(String, &NodeDef)> {
    fn walk<'a>(nodes: &'a [NodeDef], path: &str, out: &mut Vec<(String, &'a NodeDef)>) {
        for (i, node) in nodes.iter().enumerate() {
            let here = format!("{path}[{i}]");
            out.push((here.clone(), node));
            walk(node.node.0.children(), &format!("{here}.node.nodes"), out);
        }
    }
    let mut out = Vec::new();
    walk(nodes, ".nodes", &mut out);
    out
}

/// Place one list of siblings, recursing into scopes, and return the
/// column each one landed in.
fn place(
    nodes: &[NodeDef],
    path: &str,
    style: &GraphStyle,
    depths: &mut bevy::platform::collections::HashMap<String, usize>,
    out: &mut Layout,
) {
    let prefix = if path.is_empty() {
        ".nodes".to_string()
    } else {
        path.to_string()
    };
    // Column of every sibling, and the running height of each column.
    let mut placed_here: Vec<(usize, usize)> = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        let depth = node
            .wires
            .iter()
            .flat_map(|(_, w)| w.sources())
            .filter_map(|s| depths.get(s.as_str()).copied())
            .max()
            .map_or(0, |d| d + 1);
        if let Some(name) = &node.name {
            depths.insert(name.clone(), depth);
        }
        placed_here.push((i, depth));
    }

    let mut column_y: bevy::platform::collections::HashMap<usize, f32> = Default::default();
    for (i, depth) in placed_here {
        let node = &nodes[i];
        let here = format!("{prefix}[{i}]");
        let children = node.node.0.children();
        let x = depth as f32 * (style.node_width + style.gap.x);
        let y = *column_y.get(&depth).unwrap_or(&0.0);

        if children.is_empty() {
            let (ins, outs) = node.node.0.ports();
            let rows = (ins.len() + outs.len()).max(1) as f32;
            let size = Vec2::new(style.node_width, style.header + style.port * rows);
            out.nodes.push(Placed {
                path: here,
                name: node.name.clone(),
                kind: node.node.kind(),
                ins: ins.iter().map(|(p, _)| *p).collect(),
                outs: outs.iter().map(|(p, _)| *p).collect(),
                at: Vec2::new(x, y),
                size,
            });
            column_y.insert(depth, y + size.y + style.gap.y);
            continue;
        }

        // A scope: lay its children out on their own, then move them
        // inside the frame. The children keep their own columns, so a
        // district reads like a small level rather than like a list.
        let mut inner = Layout::default();
        let mut inner_depths = depths.clone();
        place(
            children,
            &format!("{here}.node.nodes"),
            style,
            &mut inner_depths,
            &mut inner,
        );
        let inner_size = inner
            .nodes
            .iter()
            .map(|p| p.at + p.size)
            .chain(inner.frames.iter().map(|f| f.at + f.size))
            .fold(Vec2::ZERO, Vec2::max);
        let origin = Vec2::new(
            x + style.frame_pad,
            y + style.frame_header + style.frame_pad,
        );
        for mut p in inner.nodes {
            p.at += origin;
            out.nodes.push(p);
        }
        for mut f in inner.frames {
            f.at += origin;
            out.frames.push(f);
        }
        let size =
            inner_size + Vec2::splat(2.0 * style.frame_pad) + Vec2::new(0.0, style.frame_header);
        out.frames.push(Frame {
            path: here,
            name: node.name.clone(),
            kind: node.node.kind(),
            at: Vec2::new(x, y),
            size,
        });
        // A frame owns every column it spans, or the next sibling would
        // be drawn through it.
        let spanned = ((size.x / (style.node_width + style.gap.x)).ceil() as usize).max(1);
        for d in depth..depth + spanned {
            let bottom = y + size.y + style.gap.y;
            let at = column_y.entry(d).or_insert(0.0);
            *at = at.max(bottom);
        }
        // Names declared inside a scope are visible to later siblings.
        for (name, d) in inner_depths {
            depths.entry(name).or_insert(d);
        }
    }
}

#[cfg(test)]
mod tests;
