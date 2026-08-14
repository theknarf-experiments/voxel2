//! The level's node list, in the graph view's vocabulary.
//!
//! `bevy_graph_view` draws boxes, ports, wires and scopes and knows
//! nothing about levels; this module is the translation. A node's id is
//! its REFLECT PATH — the same path `world.mutate_resources` uses over
//! BRP — so a clicked box comes back as an address straight into the
//! document, and the `.node.nodes` scheme for a scope's children is
//! written down here and nowhere else.

use bevy_graph_view::GraphNode;
use voxel_engine::graph::NodeDef;

/// The box that stands for the DOCUMENT itself, addressed by the empty
/// path — the reflect path of the root.
///
/// Everything in the picture is a node, including the level: selecting it
/// shows the sections that are not nodes, in the same column as any other
/// node's fields.
pub const LEVEL: &str = "";

/// The level's own box, drawn above the graph it heads.
pub fn head() -> GraphNode {
    GraphNode {
        id: LEVEL.to_string(),
        kind: "level".to_string(),
        ..Default::default()
    }
}

/// Every node of the level, ids assigned from source order.
pub fn graph_of(nodes: &[NodeDef]) -> Vec<GraphNode> {
    convert(nodes, ".nodes")
}

fn convert(nodes: &[NodeDef], path: &str) -> Vec<GraphNode> {
    nodes
        .iter()
        .enumerate()
        .map(|(i, def)| {
            let here = format!("{path}[{i}]");
            let (ins, outs) = def.node.0.ports();
            let children = convert(def.node.0.children(), &format!("{here}.node.nodes"));
            GraphNode {
                id: here,
                name: def.name.clone(),
                kind: def.node.kind().to_string(),
                ins: ins.iter().map(|(p, _)| p.to_string()).collect(),
                outs: outs.iter().map(|(p, _)| p.to_string()).collect(),
                wires: def
                    .wires
                    .iter()
                    .map(|(port, wire)| (port.clone(), wire.sources().to_vec()))
                    .collect(),
                children,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
