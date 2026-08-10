//! Reflect paths, and how to walk one to the value it names.
//!
//! `.field` and `[index]` are BRP's own syntax, so a row's path is
//! usually exactly the string `world.mutate_resources` would take. `{key}`
//! is the one addition: a map entry, which BRP cannot address at all and
//! which a level uses for its prefab and structure tables.
//!
//! Written here rather than borrowed from `GetPath` for that one reason —
//! and it earns its keep twice, because the errors say which STEP failed
//! against which type, and a wrong path in a dev tool should read as a
//! sentence rather than as nothing happening.

use bevy::reflect::{PartialReflect, ReflectMut};

#[derive(Debug, PartialEq, Eq)]
pub enum Step<'a> {
    Field(&'a str),
    Index(usize),
    Key(&'a str),
}

/// Split a path into its steps, or say where it stopped making sense.
pub fn parse(path: &str) -> Result<Vec<Step<'_>>, String> {
    let mut steps = Vec::new();
    let mut rest = path;
    while !rest.is_empty() {
        let (step, tail) = match rest.as_bytes()[0] {
            b'.' => {
                let end = rest[1..]
                    .find(['.', '[', '{'])
                    .map_or(rest.len(), |i| i + 1);
                (Step::Field(&rest[1..end]), &rest[end..])
            }
            b'[' => {
                let end = rest.find(']').ok_or_else(|| format!("'{path}': no ]"))?;
                let n = rest[1..end]
                    .parse()
                    .map_err(|_| format!("'{path}': '{}' is not an index", &rest[1..end]))?;
                (Step::Index(n), &rest[end + 1..])
            }
            b'{' => {
                let end = rest.find('}').ok_or_else(|| format!("'{path}': no }}"))?;
                (Step::Key(&rest[1..end]), &rest[end + 1..])
            }
            c => {
                return Err(format!(
                    "'{path}': expected . [ or {{, found '{}'",
                    c as char
                ))
            }
        };
        steps.push(step);
        rest = tail;
    }
    Ok(steps)
}

/// The value a path names, mutably.
///
/// Enums are stepped through by field name as well as structs: the whole
/// generator vocabulary is an enum of struct variants, so a resolver that
/// only understood structs would reach nothing worth editing.
pub fn resolve_mut<'a>(
    root: &'a mut dyn PartialReflect,
    path: &str,
) -> Result<&'a mut dyn PartialReflect, String> {
    let steps = parse(path)?;
    let mut here = root;
    for (i, step) in steps.iter().enumerate() {
        let so_far = || {
            steps[..i].iter().fold(String::new(), |mut s, st| {
                match st {
                    Step::Field(f) => s.push_str(&format!(".{f}")),
                    Step::Index(n) => s.push_str(&format!("[{n}]")),
                    Step::Key(k) => s.push_str(&format!("{{{k}}}")),
                }
                s
            })
        };
        here = match (step, here.reflect_mut()) {
            (Step::Field(name), ReflectMut::Struct(s)) => s.field_mut(name),
            (Step::Field(name), ReflectMut::Enum(e)) => e.field_mut(name),
            (Step::Index(n), ReflectMut::List(l)) => l.get_mut(*n),
            (Step::Index(n), ReflectMut::Array(a)) => a.get_mut(*n),
            (Step::Index(n), ReflectMut::Tuple(t)) => t.field_mut(*n),
            (Step::Index(n), ReflectMut::TupleStruct(t)) => t.field_mut(*n),
            (Step::Index(n), ReflectMut::Enum(e)) => e.field_at_mut(*n),
            (Step::Key(k), ReflectMut::Map(m)) => m.get_mut(&k.to_string()),
            (step, _) => {
                return Err(format!(
                    "'{path}': {step:?} does not apply at '{}'",
                    so_far()
                ))
            }
        }
        .ok_or_else(|| format!("'{path}': nothing at {step:?} under '{}'", so_far()))?;
    }
    Ok(here)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::platform::collections::HashMap;
    use bevy::prelude::*;

    #[derive(Reflect, Debug, PartialEq)]
    struct Doc {
        lod: Lod,
        ops: Vec<Op>,
        prefabs: HashMap<String, f32>,
        pair: (i32, i32),
    }

    #[derive(Reflect, Debug, PartialEq)]
    struct Lod {
        split_k: f64,
        colour: [f32; 3],
    }

    #[derive(Reflect, Debug, PartialEq)]
    enum Op {
        Fbm { amp: f32 },
        Solid,
    }

    fn doc() -> Doc {
        Doc {
            lod: Lod {
                split_k: 2.0,
                colour: [0.1, 0.2, 0.3],
            },
            ops: vec![Op::Fbm { amp: 800.0 }, Op::Solid],
            prefabs: HashMap::from_iter([("hut".to_string(), 4.0)]),
            pair: (1, 2),
        }
    }

    fn set(d: &mut Doc, path: &str, v: impl PartialReflect) -> Result<(), String> {
        let slot = resolve_mut(d, path)?;
        slot.try_apply(&v).map_err(|e| e.to_string())
    }

    #[test]
    fn nested_field() {
        let mut d = doc();
        set(&mut d, ".lod.split_k", 4.0f64).unwrap();
        assert_eq!(d.lod.split_k, 4.0);
    }

    #[test]
    fn array_element() {
        let mut d = doc();
        set(&mut d, ".lod.colour[1]", 0.9f32).unwrap();
        assert_eq!(d.lod.colour, [0.1, 0.9, 0.3]);
    }

    /// The case the whole editor turns on: a level's op list is a `Vec` of
    /// enum struct-variants, and every number worth editing is inside one.
    #[test]
    fn field_of_a_struct_variant_in_a_list() {
        let mut d = doc();
        set(&mut d, ".ops[0].amp", 42.0f32).unwrap();
        assert_eq!(d.ops[0], Op::Fbm { amp: 42.0 });
    }

    /// BRP cannot address a map entry; a level's prefab table is one.
    #[test]
    fn map_entry_by_key() {
        let mut d = doc();
        set(&mut d, ".prefabs{hut}", 9.0f32).unwrap();
        assert_eq!(d.prefabs.get("hut"), Some(&9.0));
    }

    #[test]
    fn tuple_element() {
        let mut d = doc();
        set(&mut d, ".pair[1]", 7i32).unwrap();
        assert_eq!(d.pair, (1, 7));
    }

    /// A bad path has to say what was wrong with it. A dev tool that
    /// silently does nothing is worse than one that refuses out loud.
    #[test]
    fn a_wrong_path_says_where_it_broke() {
        let mut d = doc();
        let e = set(&mut d, ".lod.nope", 1.0f64).unwrap_err();
        assert!(e.contains("nope"), "{e}");
        assert!(e.contains(".lod"), "{e}");

        let e = parse("lod").unwrap_err();
        assert!(e.contains("expected"), "{e}");
    }

    /// Applying the wrong type must fail rather than corrupt the document.
    #[test]
    fn a_wrong_type_is_refused() {
        let mut d = doc();
        assert!(set(&mut d, ".lod.split_k", 4.0f32).is_err());
        assert_eq!(d.lod.split_k, 2.0);
    }
}
