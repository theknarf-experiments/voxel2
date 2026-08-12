//! Prefabs in files of their own, so several levels can share one.
//!
//! A level's `prefabs` list holds either a prefab written out —
//! `{"name": ..., "ops": [...]}` — or one line naming a file:
//! `{"use": "prefabs/monolith_circle.json"}`, whose contents are exactly
//! that same written-out form. The file carries the NAME, so a prefab is
//! readable on its own and two levels cannot give one object two names.
//!
//! Splicing happens on the DOCUMENT, before anything is deserialized, so
//! nothing downstream learns that files are involved: by the time
//! [`crate::level::LevelDef`] exists, a prefab that came from a file and
//! one written inline are the same value apart from remembering where to
//! be written back to.
//!
//! Paths are relative to the level that names them, so a level moved
//! between directories takes its prefabs with it.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::level::PrefabDef;

/// Splice every `use` in the document's `prefabs` list, in place.
///
/// `base` is the directory of the file `doc` came from. `None` is a level
/// that came from a string with no place on disk — those cannot resolve a
/// prefab, and saying so is better than resolving against a working
/// directory nobody chose.
pub fn resolve(doc: &mut Value, base: Option<&Path>) -> Result<(), String> {
    let Some(list) = doc.get_mut("prefabs").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for entry in list.iter_mut() {
        splice(entry, base)?;
    }
    Ok(())
}

/// Replace one `{"use": path}` with what the file says, keeping the `use`
/// key so the prefab remembers where it lives.
fn splice(entry: &mut Value, base: Option<&Path>) -> Result<(), String> {
    let Some(map) = entry.as_object() else {
        return Err(format!("a prefab must be an object, not {entry}"));
    };
    let Some(rel) = map.get("use") else {
        return Ok(());
    };
    let Some(rel) = rel.as_str() else {
        return Err(format!("use must be a path, not {rel}"));
    };
    if map.len() > 1 {
        let extra: Vec<&str> = map
            .keys()
            .filter(|k| *k != "use")
            .map(String::as_str)
            .collect();
        return Err(format!(
            "the prefab using '{rel}' also sets {extra:?} — the file is the one copy of it, \
             so there is nothing here to override"
        ));
    }
    let Some(base) = base else {
        return Err(format!(
            "this level uses the prefab '{rel}' but was loaded from a string, which has no \
             directory to resolve it against — load it with LevelDef::from_path"
        ));
    };

    let path = base.join(rel);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("prefab '{rel}' ({}): {e}", path.display()))?;
    let inner: Value = serde_json::from_str(&text).map_err(|e| format!("prefab '{rel}': {e}"))?;
    let Value::Object(mut merged) = inner else {
        return Err(format!("prefab '{rel}' is not an object"));
    };
    if merged.contains_key("use") {
        return Err(format!(
            "prefab '{rel}' is itself a reference — a prefab file holds an object, not a \
             pointer to one"
        ));
    }
    merged.insert("use".into(), Value::String(rel.to_string()));
    *entry = Value::Object(merged);
    Ok(())
}

/// Write every prefab that came from a file back to it.
///
/// Serializing a [`PrefabDef`] that has a `from` already writes only its
/// `use` line, so the level half needs nothing here — this is the other
/// half, and it runs FIRST: a level pointing at a file that does not exist
/// yet is a level that does not load.
pub fn write(prefabs: &[PrefabDef], base: &Path) -> Result<(), String> {
    for prefab in prefabs {
        let Some(rel) = &prefab.from else { continue };
        let path = base.join(rel);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("prefab '{rel}' ({}): {e}", dir.display()))?;
        }
        let json = serde_json::to_string_pretty(&prefab.detached())
            .map_err(|e| format!("prefab '{rel}': {e}"))?;
        std::fs::write(&path, format!("{json}\n"))
            .map_err(|e| format!("prefab '{rel}' ({}): {e}", path.display()))?;
    }
    Ok(())
}

/// The other levels beside `level` whose `prefabs` name the same file.
///
/// Read off the documents rather than off anything loaded, because a
/// prefab is shared by whoever names its path and a level in memory knows
/// only that it named it. Parsed shallowly — the `prefabs` list and
/// nothing else — so asking is cheap enough to do when a selection moves.
///
/// Levels are the `.json` files beside this one. A game that keeps them
/// elsewhere gets an empty answer rather than a wrong one.
pub fn users(level: &Path, rel: &str) -> Vec<String> {
    let Some(dir) = level.parent() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p != level && p.extension().is_some_and(|e| e == "json"))
        .filter(|p| names_prefab(p, rel))
        .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(str::to_string))
        .collect();
    out.sort();
    out
}

/// Does this document's `prefabs` list name `rel`?
fn names_prefab(path: &Path, rel: &str) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(doc) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    doc.get("prefabs")
        .and_then(Value::as_array)
        .is_some_and(|list| {
            list.iter()
                .any(|e| e.get("use").and_then(Value::as_str) == Some(rel))
        })
}

/// Every prefab file a level reads, deduplicated and in a stable order.
///
/// The watcher needs these: a prefab edited on disk has to reload the
/// levels that use it, or hot reload would work on levels and quietly not
/// on the things levels are made of.
pub fn sources(prefabs: &[PrefabDef], base: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = prefabs
        .iter()
        .filter_map(|p| p.from.as_ref())
        .map(|rel| base.join(rel))
        .collect();
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use crate::LevelDef;

    fn reg() -> bevy::reflect::TypeRegistryArc {
        crate::graph::registry::engine_kinds()
    }

    /// A scratch directory per test, so two running at once cannot read
    /// each other's files.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("voxel2-prefab-{name}"));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(dir.join("prefabs")).unwrap();
        dir
    }

    const LOD: &str =
        r#""lod":{"max_level":8,"top_radius":3,"top_y":[-1,0],"split_k":2.5,"merge_k":3.0}"#;
    const MONOLITH: &str = r#"{"name":"monolith","ops":[
        {"shape":"cylinder","center":[0.0,4.0,0.0],"radius":1.1,"half_height":4.5,"material":3}
    ]}"#;

    /// A level whose `prefabs` list is `prefabs`, written to `dir/name`.
    fn level_at(dir: &std::path::Path, name: &str, prefabs: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        let placement = r#"{"prefab":"monolith","position":[0.0,0.0,0.0]}"#;
        std::fs::write(
            &path,
            format!(
                r#"{{{LOD},"materials":[],"nodes":[],
                    "prefabs":[{prefabs}],"placements":[{placement}]}}"#
            ),
        )
        .unwrap();
        path
    }

    #[test]
    fn a_prefab_from_a_file_is_an_ordinary_prefab() {
        let dir = scratch("load");
        std::fs::write(dir.join("prefabs/monolith.json"), MONOLITH).unwrap();
        let path = level_at(&dir, "l.json", r#"{"use":"prefabs/monolith.json"}"#);

        let level = LevelDef::from_path(&path, &reg()).unwrap();
        assert_eq!(level.prefabs.len(), 1);
        let p = &level.prefabs[0];
        assert_eq!(p.name, "monolith", "the FILE carries the name");
        assert_eq!(p.ops.len(), 1);
        assert_eq!(p.from.as_deref(), Some("prefabs/monolith.json"));
    }

    /// The point of the whole thing.
    #[test]
    fn two_levels_use_one_prefab_and_get_the_same_object() {
        let dir = scratch("share");
        std::fs::write(dir.join("prefabs/monolith.json"), MONOLITH).unwrap();
        let entry = r#"{"use":"prefabs/monolith.json"}"#;
        let a = LevelDef::from_path(&level_at(&dir, "a.json", entry), &reg()).unwrap();
        let b = LevelDef::from_path(&level_at(&dir, "b.json", entry), &reg()).unwrap();
        assert_eq!(a.prefabs, b.prefabs);

        // Edit the file and BOTH levels have the edit, which is what makes
        // it one prefab rather than two copies that agree today.
        std::fs::write(
            dir.join("prefabs/monolith.json"),
            MONOLITH.replace("4.5", "40.0"),
        )
        .unwrap();
        let a2 = LevelDef::from_path(&level_at(&dir, "a.json", entry), &reg()).unwrap();
        let b2 = LevelDef::from_path(&level_at(&dir, "b.json", entry), &reg()).unwrap();
        assert_eq!(a2.prefabs, b2.prefabs);
        assert_ne!(a2.prefabs, a.prefabs, "the edit reached the level");
    }

    /// Saving puts each half back where it came from: the level keeps its
    /// one line, the prefab keeps the shapes.
    #[test]
    fn saving_writes_the_prefab_to_its_own_file() {
        let dir = scratch("save");
        std::fs::write(dir.join("prefabs/monolith.json"), MONOLITH).unwrap();
        let path = level_at(&dir, "l.json", r#"{"use":"prefabs/monolith.json"}"#);
        let mut level = LevelDef::from_path(&path, &reg()).unwrap();

        level.prefabs[0].ops[0].material = 77;
        crate::level::save_to(&level, &path, &reg()).unwrap();

        let level_text = std::fs::read_to_string(&path).unwrap();
        assert!(level_text.contains("prefabs/monolith.json"), "{level_text}");
        assert!(
            !level_text.contains("cylinder"),
            "the level swallowed the prefab: {level_text}"
        );
        let prefab_text = std::fs::read_to_string(dir.join("prefabs/monolith.json")).unwrap();
        assert!(prefab_text.contains("\"material\": 77"), "{prefab_text}");
        assert!(
            prefab_text.contains("\"name\": \"monolith\""),
            "{prefab_text}"
        );
        assert!(!prefab_text.contains("use"), "{prefab_text}");

        let back = LevelDef::from_path(&path, &reg()).unwrap();
        assert_eq!(back.prefabs, level.prefabs);
    }

    /// An inline prefab is still a prefab, and saving leaves it inline.
    #[test]
    fn an_inline_prefab_stays_in_the_level() {
        let dir = scratch("inline");
        let path = level_at(&dir, "l.json", MONOLITH);
        let level = LevelDef::from_path(&path, &reg()).unwrap();
        assert_eq!(level.prefabs[0].from, None);

        crate::level::save_to(&level, &path, &reg()).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("cylinder"), "{text}");
        assert!(!text.contains("\"use\""), "{text}");
    }

    /// Who else uses this file. The editor asks before letting a handle
    /// move a shape, because an edit that quietly changed a second level
    /// would be worse than one that says it is about to.
    #[test]
    fn a_shared_prefab_names_the_levels_that_share_it() {
        let dir = scratch("users");
        std::fs::write(dir.join("prefabs/monolith.json"), MONOLITH).unwrap();
        let entry = r#"{"use":"prefabs/monolith.json"}"#;
        let a = level_at(&dir, "alpha.json", entry);
        level_at(&dir, "beta.json", entry);
        level_at(&dir, "gamma.json", MONOLITH); // its own copy, inline
        assert_eq!(super::users(&a, "prefabs/monolith.json"), ["beta"]);

        // And a prefab nobody else names is shared with nobody.
        assert!(super::users(&a, "prefabs/nothing.json").is_empty());
    }

    #[test]
    fn a_missing_prefab_names_the_file() {
        let dir = scratch("missing");
        let path = level_at(&dir, "l.json", r#"{"use":"prefabs/nope.json"}"#);
        let e = LevelDef::from_path(&path, &reg()).unwrap_err();
        assert!(e.contains("prefabs/nope.json"), "{e}");
    }

    /// The file is the one copy, so a level cannot keep half of it too.
    #[test]
    fn a_use_that_also_sets_fields_is_refused() {
        let dir = scratch("override");
        std::fs::write(dir.join("prefabs/monolith.json"), MONOLITH).unwrap();
        let path = level_at(
            &dir,
            "l.json",
            r#"{"use":"prefabs/monolith.json","name":"other"}"#,
        );
        let e = LevelDef::from_path(&path, &reg()).unwrap_err();
        assert!(e.contains("name") && e.contains("override"), "{e}");
    }

    /// A prefab file holds an object, not a pointer to another file.
    #[test]
    fn a_prefab_file_that_points_elsewhere_is_refused() {
        let dir = scratch("chain");
        std::fs::write(
            dir.join("prefabs/monolith.json"),
            r#"{"use":"prefabs/other.json"}"#,
        )
        .unwrap();
        let path = level_at(&dir, "l.json", r#"{"use":"prefabs/monolith.json"}"#);
        let e = LevelDef::from_path(&path, &reg()).unwrap_err();
        assert!(e.contains("not a pointer"), "{e}");
    }

    /// A level with nowhere to resolve against says so, rather than
    /// resolving against whatever directory the process happens to be in.
    #[test]
    fn a_level_from_a_string_cannot_reach_a_prefab() {
        let e = LevelDef::from_json(
            &format!(r#"{{{LOD},"nodes":[],"prefabs":[{{"use":"prefabs/x.json"}}]}}"#),
            &reg(),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("from_path"), "{e}");
    }

    #[test]
    fn the_watch_list_covers_every_prefab_file_once() {
        let dir = scratch("watch");
        std::fs::write(dir.join("prefabs/monolith.json"), MONOLITH).unwrap();
        let path = level_at(&dir, "l.json", r#"{"use":"prefabs/monolith.json"}"#);
        let level = LevelDef::from_path(&path, &reg()).unwrap();
        assert_eq!(
            super::sources(&level.prefabs, &dir),
            vec![dir.join("prefabs/monolith.json")]
        );
    }
}
