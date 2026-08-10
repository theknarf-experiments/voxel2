//! The level graph, and the compiler that lowers it.
//!
//! A level is one list of nodes. Each names the nodes it consumes, so the
//! dataflow is written down rather than implied by a shared register file
//! and a position in a list. This module checks that wiring and lowers the
//! point-domain part of it back to the `WorldOp` program both interpreters
//! already run.
//!
//! **The order a level is written IS the program order.** The compiler
//! verifies that order is a valid topological one rather than computing a
//! new one, for three reasons: a level reads top to bottom like the
//! program it is, the error for a forward reference can name the two nodes
//! involved, and a topological sort is not unique — reordering would make
//! "the same graph" compile to a different program, which is exactly what
//! nobody wants to debug.
//!
//! **Registers are the lowering, not the language.** Every value but a
//! field lives in one register, so one thing is live at a time. Reading is
//! not consuming — every height op reads the same warp — so what the
//! compiler tracks is LIVENESS: which writes are still standing, and which
//! of them a reader's own region can see. A gated write displaces only the
//! ones it covers, which is what lets nine districts each define a lattice
//! and one `shafts_cut` read seven shafts.

use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::reflect::ReflectMut;
use serde::{Deserialize, Serialize};
use voxel_core::opgen::Value;
use voxel_core::worldop::{WorldOp, FIELD_SLOTS};

/// A compiled level graph.
#[derive(Debug)]
pub struct Program {
    /// The point-domain program, in emission order.
    pub ops: Vec<WorldOp>,
    /// Which slot each named field node was allocated. Consumers name the
    /// node; this is how a slot number is found without a level ever
    /// writing one.
    pub fields: HashMap<String, u32>,
}

/// Everything that can be wrong with a graph, said in terms of the nodes a
/// level wrote rather than the registers it never mentioned.
#[derive(Debug, PartialEq)]
pub enum Error {
    DuplicateName(String),
    UnknownNode {
        at: String,
        port: String,
        name: String,
    },
    ForwardReference {
        at: String,
        port: String,
        name: String,
    },
    UnknownPort {
        at: String,
        port: String,
    },
    MissingPort {
        at: String,
        port: String,
        value: Value,
    },
    WrongType {
        at: String,
        port: String,
        want: Value,
        got: Value,
    },
    StaleRead {
        at: String,
        port: String,
        wired: Vec<String>,
        live: Vec<String>,
    },
    TooManyFields {
        at: String,
        limit: usize,
    },
    Unnamed {
        at: String,
    },
}

impl Error {
    /// The node the complaint is ABOUT, by the name a level gave it.
    ///
    /// Every diagnostic names one, because a compiler that says "type
    /// error" about a hand-written document is a compiler people guess
    /// against. This is that name, for a tool that can point at it.
    pub fn at(&self) -> &str {
        match self {
            Error::DuplicateName(name) => name,
            Error::UnknownNode { at, .. }
            | Error::ForwardReference { at, .. }
            | Error::UnknownPort { at, .. }
            | Error::MissingPort { at, .. }
            | Error::WrongType { at, .. }
            | Error::StaleRead { at, .. }
            | Error::TooManyFields { at, .. }
            | Error::Unnamed { at } => at,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DuplicateName(n) => write!(f, "two nodes are called '{n}'"),
            Error::UnknownNode { at, port, name } => {
                write!(
                    f,
                    "'{at}' wires port '{port}' to '{name}', which is not a node"
                )
            }
            Error::ForwardReference { at, port, name } => write!(
                f,
                "'{at}' wires port '{port}' to '{name}', which is written later — \
                 a level is its own program order, so '{name}' has to come first"
            ),
            Error::UnknownPort { at, port } => {
                write!(f, "'{at}' has no port called '{port}'")
            }
            Error::MissingPort { at, port, value } => write!(
                f,
                "'{at}' consumes {value:?} through port '{port}' and nothing is wired to it"
            ),
            Error::WrongType {
                at,
                port,
                want,
                got,
            } => write!(
                f,
                "'{at}' port '{port}' wants {want:?} but is wired to a {got:?}"
            ),
            Error::StaleRead {
                at,
                port,
                wired,
                live,
            } => write!(
                f,
                "'{at}' reads {wired:?} through port '{port}', but by then that \
                 value has been replaced by {live:?} — one thing is live at a time, \
                 so whatever reads a value has to come before what overwrites it"
            ),
            Error::TooManyFields { at, limit } => {
                write!(
                    f,
                    "'{at}' is field {} and there are only {limit}",
                    limit + 1
                )
            }
            Error::Unnamed { at } => {
                write!(f, "{at} is referred to by name but has none")
            }
        }
    }
}

/// One node after scopes are flattened.
struct Flat<'a> {
    name: Option<&'a str>,
    /// What to call it in an error when it has no name.
    label: String,
    node: &'a NodeDef,
    /// The intersection of every enclosing scope, or `None` for ungated.
    region: Option<[f32; 4]>,
}

/// Flatten scopes, intersecting their gates.
///
/// A box inside a box is a box, so any depth of nesting still lands in the
/// single packed gate a `WorldOp` carries.
fn flatten<'a>(
    nodes: &'a [NodeDef],
    region: Option<[f32; 4]>,
    path: &str,
    out: &mut Vec<Flat<'a>>,
) {
    for (i, node) in nodes.iter().enumerate() {
        let label = match &node.name {
            Some(n) => n.to_string(),
            None => format!("{path}[{i}] ({})", node.node.kind()),
        };
        let inner = node.node.0.children();
        if let Some(axes) = node.node.0.gate() {
            let region = Some(match region {
                None => axes,
                Some(o) => [
                    o[0].max(axes[0]),
                    o[1].min(axes[1]),
                    o[2].max(axes[2]),
                    o[3].min(axes[3]),
                ],
            });
            flatten(inner, region, &label, out);
        } else {
            out.push(Flat {
                name: node.name.as_deref(),
                label,
                node,
                region,
            });
        }
    }
}

/// Does gate `outer` contain every point of gate `inner`?
///
/// `None` is everywhere: an ungated write covers any region, and nothing
/// but another ungated write covers an ungated one.
fn covers(outer: Option<[f32; 4]>, inner: Option<[f32; 4]>) -> bool {
    let Some(outer) = outer else { return true };
    let Some(inner) = inner else { return false };
    outer[0] <= inner[0] && outer[1] >= inner[1] && outer[2] <= inner[2] && outer[3] >= inner[3]
}

/// Do two region gates cover any of the same points?
///
/// `None` is everywhere, so it overlaps anything.
fn gates_overlap(a: Option<[f32; 4]>, b: Option<[f32; 4]>) -> bool {
    let (Some(a), Some(b)) = (a, b) else {
        return true;
    };
    let axis = |lo_a: f32, hi_a: f32, lo_b: f32, hi_b: f32| lo_a < hi_b && lo_b < hi_a;
    axis(a[0], a[1], b[0], b[1]) && axis(a[2], a[3], b[2], b[3])
}

/// Check a level's wiring and lower its point-domain nodes.
pub fn compile(nodes: &[NodeDef]) -> Result<Program, Error> {
    let mut flat = Vec::new();
    flatten(nodes, None, "", &mut flat);

    // Names, and the value each node produces.
    let mut by_name: HashMap<&str, usize> = HashMap::default();
    for (i, f) in flat.iter().enumerate() {
        if let Some(name) = f.name {
            if by_name.insert(name, i).is_some() {
                return Err(Error::DuplicateName(name.to_string()));
            }
        }
    }

    // Field slots, allocated in declaration order — which is what makes
    // the level stop writing slot numbers and stop keeping them agreeing
    // with the spawners that read them.
    let mut fields: HashMap<String, u32> = HashMap::default();
    for f in &flat {
        if f.node.node.kind() != "field" {
            continue;
        }
        let Some(name) = f.name else {
            return Err(Error::Unnamed {
                at: f.label.clone(),
            });
        };
        if fields.len() >= FIELD_SLOTS {
            return Err(Error::TooManyFields {
                at: f.label.clone(),
                limit: FIELD_SLOTS,
            });
        }
        fields.insert(name.to_string(), fields.len() as u32);
    }

    // Wiring: every port named, every name known and already written,
    // every type agreeing, and every read reading what is actually live.
    //
    // Liveness rather than "consumed once", because reading a value is not
    // consuming it: every height op reads the same warp and every slab op
    // the same lattice. What a single-slot register cannot do is hold two
    // things at once — so this tracks, per value, which nodes' writes are
    // still standing, and a gated write only displaces the ones its region
    // covers.
    let mut live: HashMap<Value, Vec<usize>> = HashMap::default();
    let mut ops = Vec::with_capacity(flat.len());

    for (i, f) in flat.iter().enumerate() {
        let (ins, outs) = f.node.node.0.ports();

        for (port, _) in f.node.wires.iter() {
            if !ins.iter().any(|(p, _)| p == port) {
                return Err(Error::UnknownPort {
                    at: f.label.clone(),
                    port: port.clone(),
                });
            }
        }

        for (port, want) in ins {
            let Some(wire) = f.node.wires.get(port) else {
                return Err(Error::MissingPort {
                    at: f.label.clone(),
                    port: port.to_string(),
                    value: *want,
                });
            };
            let sources = wire.sources();
            let mut resolved = Vec::with_capacity(sources.len());
            for name in sources {
                let Some(&j) = by_name.get(name.as_str()) else {
                    return Err(Error::UnknownNode {
                        at: f.label.clone(),
                        port: port.to_string(),
                        name: name.clone(),
                    });
                };
                if j >= i {
                    return Err(Error::ForwardReference {
                        at: f.label.clone(),
                        port: port.to_string(),
                        name: name.clone(),
                    });
                }
                let (_, produced) = flat[j].node.node.0.ports();
                let got = produced.first().map(|(_, v)| *v);
                if got != Some(*want) {
                    return Err(Error::WrongType {
                        at: f.label.clone(),
                        port: port.to_string(),
                        want: *want,
                        got: got.unwrap_or(Value::Field),
                    });
                }
                resolved.push(j);
            }

            // A host value is addressed by name, and so is a field slot:
            // several stand at once and every consumer reads the one it
            // named, so there is nothing to be stale about.
            if !want.is_single() {
                continue;
            }

            // What this node can actually see: the still-standing writes
            // of this value whose region overlaps its own. A district's
            // `slabs_y` sees its own district's lattice and no other, so
            // it names one source and not nine.
            let standing = live.get(want).map(Vec::as_slice).unwrap_or_default();
            // A write that covers this node's whole region hides every
            // write before it: inside district one, `void` is not visible
            // because the district's own `coarse_solid` replaced all of it.
            let from = standing
                .iter()
                .rposition(|&j| covers(flat[j].region, f.region))
                .unwrap_or(0);
            let mut visible: Vec<usize> = standing[from..]
                .iter()
                .copied()
                .filter(|&j| gates_overlap(flat[j].region, f.region))
                .collect();
            visible.sort_unstable();
            let mut asked = resolved.clone();
            asked.sort_unstable();
            if asked != visible {
                let label = |v: &[usize]| v.iter().map(|&j| flat[j].label.clone()).collect();
                return Err(Error::StaleRead {
                    at: f.label.clone(),
                    port: port.to_string(),
                    wired: label(&asked),
                    live: label(&visible),
                });
            }
        }

        // Writes. An ungated write replaces the value everywhere; a gated
        // one only where it applies, which is what lets nine districts each
        // define a lattice and one `shafts_cut` read seven shafts.
        for (_, produced) in outs {
            if !produced.is_single() {
                continue;
            }
            let standing = live.entry(*produced).or_default();
            standing.retain(|&j| !covers(f.region, flat[j].region));
            standing.push(i);
        }

        let slot = f
            .name
            .and_then(|n| fields.get(n))
            .copied()
            .unwrap_or_default();
        if let Some(op) = f.node.node.0.op(slot) {
            ops.push(match f.region {
                Some(band) => op.region(band),
                None => op,
            });
        }
    }

    Ok(Program { ops, fields })
}

/// What an edit to a node list makes stale, or `None` if nothing changed.
///
/// The blunt answer — "the list differs, restream everything" — charges a
/// population edit the price of the whole world, which is the price of
/// regenerating every chunk to place exactly the voxels it already had.
/// This asks each node that actually differs what IT invalidates and takes
/// the worst.
///
/// Nodes are paired by NAME where they have one, so inserting or deleting
/// a population does not read as "every node after it changed". Unnamed
/// nodes are the long unbranching middles of chains and pair by position,
/// where an insertion genuinely does shift what follows.
pub fn changed(new: &[NodeDef], old: &[NodeDef]) -> Option<node::Invalidates> {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    scan(new, "", &mut a);
    scan(old, "", &mut b);

    let mut worst = None;
    let mut note = |node: &NodeDef| {
        let effect = node.node.0.invalidates();
        if worst.is_none_or(|w| effect > w) {
            worst = Some(effect);
        }
    };
    let by_key: HashMap<&str, &NodeDef> = b.iter().map(|(k, n)| (k.as_str(), *n)).collect();
    for (key, node) in &a {
        match by_key.get(key.as_str()) {
            // A scope's own row is its gate: its children are keys of
            // their own, so comparing the whole node would report every
            // district as changed the moment one op inside it moved.
            Some(was) if same(node, was) => {}
            Some(was) => {
                note(node);
                note(was);
            }
            None => note(node),
        }
    }
    let by_key: HashMap<&str, &NodeDef> = a.iter().map(|(k, n)| (k.as_str(), *n)).collect();
    for (key, node) in &b {
        if !by_key.contains_key(key.as_str()) {
            note(node);
        }
    }
    worst
}

/// Are these the same node, ignoring what a scope CONTAINS?
fn same(a: &NodeDef, b: &NodeDef) -> bool {
    if a.node.0.gate().is_some() || b.node.0.gate().is_some() {
        return a.name == b.name && a.wires == b.wires && a.node.0.gate() == b.node.0.gate();
    }
    a == b
}

/// Every node in the tree, keyed by name where it has one and by position
/// where it does not.
fn scan<'a>(nodes: &'a [NodeDef], path: &str, out: &mut Vec<(String, &'a NodeDef)>) {
    for (i, node) in nodes.iter().enumerate() {
        let key = match &node.name {
            Some(name) => name.clone(),
            None => format!("{path}[{i}]"),
        };
        scan(node.node.0.children(), &key, out);
        out.push((key, node));
    }
}

/// Rename a node, and every wire that named it.
///
/// A name is not a label: it is the only way anything refers to a node, so
/// renaming one WITHOUT its references is the same edit as deleting it.
/// An editor that offered the first and left the second is offering a
/// trap, so this is the operation, and renaming is not a field write.
///
/// Returns how many wires followed.
pub fn rename(nodes: &mut [NodeDef], from: &str, to: &str) -> usize {
    fn walk(nodes: &mut [NodeDef], from: &str, to: &str, moved: &mut usize) {
        for node in nodes {
            if node.name.as_deref() == Some(from) {
                node.name = Some(to.to_string());
            }
            for (_, wire) in node.wires.0.iter_mut() {
                match wire {
                    Wire::One(name) if name == from => {
                        *name = to.to_string();
                        *moved += 1;
                    }
                    Wire::Many(names) => {
                        for name in names.iter_mut().filter(|n| *n == from) {
                            *name = to.to_string();
                            *moved += 1;
                        }
                    }
                    Wire::One(_) => {}
                }
            }
            // A scope's children are nodes, and refer by the same names.
            if let ReflectMut::Struct(s) = node.node.0.as_partial_reflect_mut().reflect_mut() {
                if let Some(inner) = s
                    .field_mut("nodes")
                    .and_then(|f| f.try_downcast_mut::<Vec<NodeDef>>())
                {
                    walk(inner, from, to, moved);
                }
            }
        }
    }
    let mut moved = 0;
    walk(nodes, from, to, &mut moved);
    moved
}

/// Names every node a level defines, for the reference lists an editor
/// offers. Scopes included: they are nodes too.
pub fn names(nodes: &[NodeDef]) -> Vec<String> {
    let mut flat = Vec::new();
    flatten(nodes, None, "", &mut flat);
    let mut seen: HashSet<&str> = HashSet::default();
    flat.iter()
        .filter_map(|f| f.name)
        .filter(|n| seen.insert(*n))
        .map(str::to_string)
        .collect()
}

pub mod node;
pub mod nodes;
pub mod registry;

pub use node::{AnyNode, CloneNode, Domain, Node, Ports, ReflectNode};
pub use registry::with_registry;

/// One node of a level's graph: a name, the inputs it consumes, and what
/// it is.
///
/// Every input is named. There is no ordering rule that supplies one
/// implicitly, because an edge nothing writes down is an edge no level can
/// rewire, no editor can draw and no invalidation can follow — which is
/// what the generator's shared registers were.
#[derive(Reflect, Clone, Debug, PartialEq, Default)]
pub struct NodeDef {
    /// What other nodes call this one. Required only where something
    /// refers to it, so the long unbranching middle of a chain need not
    /// invent names nothing uses.
    #[reflect(@crate::schema::Title)]
    pub name: Option<String>,
    /// Port name to the node feeding it. A port may take several sources
    /// where the domain genuinely merges — the megastructure's seven
    /// region-gated `shafts_xz` into one `shafts_cut` — and the compiler
    /// checks those gates are disjoint.
    #[reflect(@crate::schema::NodeRef)]
    pub wires: Wires,
    /// What this node IS. Its type is the kind, its fields are the schema.
    #[reflect(@crate::schema::Title)]
    pub node: AnyNode,
}

/// A node's inputs, by port name.
///
/// A `BTreeMap` so a port is written once and the file round-trips in a
/// stable order — a level is a document under version control, and a map
/// that reshuffled its keys on every save would make every diff a lie.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(transparent)]
pub struct Wires(pub std::collections::BTreeMap<String, Wire>);

impl Wires {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, port: &str) -> Option<&Wire> {
        self.0.get(port)
    }

    /// Every (port, wire) pair, for the compiler's unknown-port check.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Wire)> {
        self.0.iter()
    }
}

/// What a port is wired to.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum Wire {
    One(String),
    /// Several producers into one port, for a value only one of them
    /// defines at any given point.
    Many(Vec<String>),
}

impl Wire {
    pub fn sources(&self) -> &[String] {
        match self {
            Wire::One(name) => std::slice::from_ref(name),
            Wire::Many(names) => names,
        }
    }
}

#[cfg(test)]
mod tests;
