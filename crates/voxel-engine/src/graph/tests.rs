use super::*;
use crate::graph::registry;
use crate::level::LevelDef;

fn shipped(name: &str) -> LevelDef {
    let path = format!("{}/../../levels/{name}.json", env!("CARGO_MANIFEST_DIR"));
    LevelDef::from_json(&std::fs::read_to_string(path).unwrap(), &registry::engine_kinds()).unwrap()
}

/// The program each level compiled to before it was a graph.
fn golden(name: &str) -> Vec<WorldOp> {
    let path = format!("{}/tests/golden/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    raw.as_array()
        .unwrap()
        .iter()
        .map(|o| {
            let f = |k: &str| -> [f32; 4] {
                let a = o[k].as_array().unwrap();
                std::array::from_fn(|i| a[i].as_f64().unwrap() as f32)
            };
            let u = |k: &str| o[k].as_u64().unwrap() as u32;
            WorldOp {
                kind: u("kind"),
                flags: u("flags"),
                material: u("material"),
                region: u("region"),
                p0: f("p0"),
                p1: f("p1"),
                p2: f("p2"),
            }
        })
        .collect()
}

/// **The load-bearing test of the whole migration.**
///
/// Every shipped level must compile to exactly the program it compiled to
/// when its ops were an ordered list with implicit register edges. Not "an
/// equivalent program" — the same ops, in the same order, with the same
/// bytes. If this passes, no rendered voxel can have moved, and the graph
/// is a different way of SAYING each level rather than a different level.
#[test]
fn every_shipped_level_compiles_to_the_program_it_always_had() {
    for name in ["planet", "megastructure", "purgatory"] {
        let level = shipped(name);
        let program = compile(&level.nodes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = golden(name);
        assert_eq!(program.ops.len(), want.len(), "{name}: op count");
        for (i, (got, want)) in program.ops.iter().zip(&want).enumerate() {
            assert_eq!(got, want, "{name}: op {i} differs");
        }
    }
}

/// Field slots are the compiler's now. Purgatory wrote 0 and 1 by hand on
/// the generator side and again on the spawner side; the names have to land
/// on the same numbers.
#[test]
fn fields_are_allocated_in_declaration_order() {
    let purgatory = compile(&shipped("purgatory").nodes).unwrap();
    let mut slots: Vec<_> = purgatory.fields.values().copied().collect();
    slots.sort_unstable();
    assert_eq!(slots, vec![0, 1], "purgatory declares two fields");
}

// --- diagnostics -----------------------------------------------------------
//
// Each of these is a mistake a level can make, and each has to come back
// naming the nodes involved. A compiler that says "type error" about a
// document somebody hand-wrote is a compiler they will guess against.

fn node(json: &str) -> NodeDef {
    with_registry(&registry::engine_kinds(), || serde_json::from_str(json)).unwrap()
}

/// Parse a list of nodes with the engine's kinds in scope.
fn parse(json: &str) -> Result<Vec<NodeDef>, serde_json::Error> {
    with_registry(&registry::engine_kinds(), || serde_json::from_str(json))
}

fn err(nodes: &[NodeDef]) -> String {
    compile(nodes).expect_err("should not compile").to_string()
}

#[test]
fn a_port_wired_to_nothing_names_the_node_and_the_port() {
    let nodes = vec![
        node(r#"{"kind":"height_zero","name":"sea"}"#),
        node(r#"{"kind":"height_offset","name":"a","in":{"height":"nope"},"value":1.0}"#),
    ];
    let e = err(&nodes);
    assert!(e.contains("'a'") && e.contains("'nope'"), "{e}");
}

#[test]
fn an_unwired_port_says_which_value_it_wanted() {
    let nodes = vec![node(r#"{"kind":"height_offset","name":"a","value":1.0}"#)];
    let e = err(&nodes);
    assert!(e.contains("Height") && e.contains("height"), "{e}");
}

#[test]
fn a_port_wired_to_the_wrong_type_says_both() {
    let nodes = vec![
        node(r#"{"kind":"sdf_void","name":"void"}"#),
        node(r#"{"kind":"height_offset","name":"a","in":{"height":"void"},"value":1.0}"#),
    ];
    let e = err(&nodes);
    assert!(e.contains("wants Height") && e.contains("Sdf"), "{e}");
}

/// The level is its own program order, so a reference forward is an error
/// rather than a reordering.
#[test]
fn a_forward_reference_says_it_has_to_come_first() {
    let nodes = vec![
        node(r#"{"kind":"height_zero","name":"sea"}"#),
        node(r#"{"kind":"height_offset","name":"a","in":{"height":"b"},"value":1.0}"#),
        node(r#"{"kind":"height_offset","name":"b","in":{"height":"sea"},"value":1.0}"#),
    ];
    let e = err(&nodes);
    assert!(e.contains("written later"), "{e}");
}

/// The error a level author actually hits when they try to branch: 'b'
/// wants the value 'a' already replaced, and the message says so in terms
/// of the two nodes rather than of a register the level never named.
#[test]
fn branching_a_single_slot_value_says_what_replaced_it() {
    let nodes = vec![
        node(r#"{"kind":"height_zero","name":"sea"}"#),
        node(r#"{"kind":"height_offset","name":"a","in":{"height":"sea"},"value":1.0}"#),
        node(r#"{"kind":"height_offset","name":"b","in":{"height":"sea"},"value":2.0}"#),
    ];
    let e = err(&nodes);
    assert!(e.contains("'b'") && e.contains("\"a\"") && e.contains("\"sea\""), "{e}");
    assert!(e.contains("one thing is live at a time"), "{e}");
}

#[test]
fn a_duplicate_name_is_refused() {
    let nodes = vec![
        node(r#"{"kind":"height_zero","name":"sea"}"#),
        node(r#"{"kind":"height_zero","name":"sea"}"#),
    ];
    assert!(err(&nodes).contains("two nodes are called 'sea'"));
}

#[test]
fn a_port_that_does_not_exist_is_refused() {
    let nodes = vec![
        node(r#"{"kind":"height_zero","name":"sea"}"#),
        node(r#"{"kind":"height_offset","in":{"nope":"sea"},"value":1.0}"#),
    ];
    let e = err(&nodes);
    assert!(e.contains("no port called 'nope'"), "{e}");
}

/// Nesting intersects, so a district inside a district still lands in the
/// one packed gate a `WorldOp` carries.
#[test]
fn nested_scopes_intersect_into_one_gate() {
    let nodes: Vec<NodeDef> = parse(r#"[
          {"kind":"sdf_void","name":"void"},
          {"kind":"region","axes":[0.0,0.8,0.0,1.0],"nodes":[
            {"kind":"region","axes":[0.2,1.0,0.0,0.5],"nodes":[
              {"kind":"coarse_solid","in":{"sdf":"void"},"material":1}
            ]}
          ]}
        ]"#,
    ).unwrap();
    let ops = compile(&nodes).unwrap().ops;
    assert_eq!(ops.len(), 1, "origins and scopes emit nothing");
    let band = voxel_core::worldop::unpack_region(ops[0].region);
    // The overlap of the two boxes, quantised to the byte a gate holds.
    assert!((band[0] - 0.2).abs() < 0.01, "{band:?}");
    assert!((band[1] - 0.8).abs() < 0.01, "{band:?}");
    assert!((band[3] - 0.5).abs() < 0.01, "{band:?}");
}

/// What a node reads is what its own region can SEE.
///
/// A register holds one thing at a time, but a gated write only replaces
/// it inside its gate — so an ungated reader after two gated writers sees
/// both, and a reader gated to one of those regions sees only that one.
/// This is the rule the megastructure turns on: nine districts each define
/// a lattice, one `shafts_cut` reads seven shafts, and each district's
/// `slabs_y` names its own lattice and not the other eight.
#[test]
fn a_read_sees_the_writes_its_own_region_can_reach() {
    let level = |reader_region: &str| {
        format!(
            r#"[
              {{"kind":"sdf_void","name":"void"}},
              {{"kind":"region","axes":[0.0,0.4,0.0,1.0],"nodes":[
                {{"kind":"shafts_xz","name":"a","spacing":10.0,"jitter":1.0,
                  "radius":[2.0,0.0]}}
              ]}},
              {{"kind":"region","axes":[0.6,1.0,0.0,1.0],"nodes":[
                {{"kind":"shafts_xz","name":"b","spacing":10.0,"jitter":1.0,
                  "radius":[2.0,0.0]}}
              ]}},
              {reader_region}
            ]"#
        )
    };

    // Ungated: both gated writes are still standing, so both are named.
    let both: Vec<NodeDef> = parse(&level(
        r#"{"kind":"shafts_cut","in":{"sdf":"void","shafts":["a","b"]}}"#,
    ))
    .unwrap();
    assert_eq!(compile(&both).unwrap().ops.len(), 3);

    // Naming only one of them from outside is a stale read, and says so.
    let partial: Vec<NodeDef> = parse(&level(
        r#"{"kind":"shafts_cut","in":{"sdf":"void","shafts":"a"}}"#,
    ))
    .unwrap();
    let e = compile(&partial).unwrap_err().to_string();
    assert!(e.contains("\"a\", \"b\"") || e.contains("\"b\""), "{e}");

    // Gated to one region: only that region's write is reachable, so
    // naming just it is right and naming both is not.
    let inside: Vec<NodeDef> = parse(&level(
        r#"{"kind":"region","axes":[0.0,0.4,0.0,1.0],"nodes":[
             {"kind":"shafts_cut","in":{"sdf":"void","shafts":"a"}}
           ]}"#,
    ))
    .unwrap();
    assert_eq!(compile(&inside).unwrap().ops.len(), 3);
}
