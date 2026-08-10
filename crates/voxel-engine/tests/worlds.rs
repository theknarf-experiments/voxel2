//! Loading more than one world.
//!
//! Every world-related defect this suite pins was live at once and none of
//! them failed a test, because not one test loaded a second world. The
//! renderer's half is pinned in `voxel_render::chunks::multi_world_tests`;
//! this is the simulation's, plus the seam between them — the two
//! registries have to agree about what a world id means or a chunk's key
//! addresses one world's ops and another world's materials.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use serde_json::json;
use voxel_engine::{level::HostPlanner, LevelDef, LodConfig, WorldLoader, Worlds};
use voxel_render::RenderWorlds;

fn shipped_json(name: &str) -> serde_json::Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../levels/");
    serde_json::from_str(&std::fs::read_to_string(format!("{path}{name}")).unwrap()).unwrap()
}

fn shipped(name: &str) -> LevelDef {
    from_json_value(shipped_json(name))
}

/// Parse with the engine's node kinds in scope. The node set is open, so
/// a kind is only a name until a registry says what type it is.
fn from_json_value(v: serde_json::Value) -> LevelDef {
    voxel_engine::graph::with_registry(&voxel_engine::graph::registry::engine_kinds(), || {
        serde_json::from_value(v)
    })
    .unwrap()
}

/// A shipped level with one field replaced. Patching the JSON keeps these
/// fixtures honest: they go through the same parser a real level does.
fn level_with(name: &str, field: &str, value: serde_json::Value) -> LevelDef {
    let mut level = shipped_json(name);
    level[field] = value;
    from_json_value(level)
}

/// Just enough app to run a system that loads worlds. Deliberately NOT
/// the engine plugin: a world must be registrable without a GPU, or the
/// only place this can be checked is a running window.
fn app() -> App {
    let mut app = App::new();
    app.init_resource::<Worlds>()
        .init_resource::<RenderWorlds>()
        .init_resource::<voxel_engine::streaming::StreamingRebuild>()
        .insert_resource(HostPlanner(None));
    app
}

/// Load levels through [`WorldLoader`], the way a host system does.
/// Going through a real system is the point: the loader is a
/// `SystemParam`, and that is what makes registering half a world
/// impossible to write.
fn load_in(app: &mut App, levels: Vec<(LevelDef, LodConfig)>) -> Vec<u8> {
    let mut levels = levels.into_iter();
    app.world_mut()
        .run_system_once(move |mut loader: WorldLoader| -> Vec<u8> {
            levels
                .by_ref()
                .map(|(level, config)| loader.load(level, 0, config))
                .collect()
        })
        .unwrap()
}

/// A world id means the same thing on both sides. The renderer indexes
/// its per-world state by `ChunkKey::world`, so if the two registries
/// ever disagree a chunk gets one world's ops and another's materials —
/// with no error anywhere, because both indices are in range.
#[test]
fn loading_keeps_both_registries_aligned() {
    let mut app = app();
    let planet = shipped("planet.json");
    let mega = shipped("megastructure.json");
    let ids = load_in(
        &mut app,
        vec![
            (planet.clone(), LodConfig::from(&planet.lod)),
            (mega.clone(), LodConfig::from(&mega.lod)),
        ],
    );

    assert_eq!(ids, vec![0, 1], "ids are handed out in order from 0");
    let worlds = app.world().resource::<Worlds>();
    let render = app.world().resource::<RenderWorlds>();
    assert_eq!(worlds.len(), 2);
    assert_eq!(render.len(), worlds.len(), "the registries stay the same length");
    for world in worlds.iter() {
        assert_eq!(world.id, worlds.get(world.id).unwrap().id);
        assert!(
            render.get(world.id).is_some(),
            "world {} has no render record",
            world.id,
        );
    }
}

/// Each world's GPU program is its OWN level's, not the first one's. The
/// two shipped levels pack to different reference programs, so this fails
/// loudly if a registration path ever reuses world 0's.
#[test]
fn each_world_uploads_its_own_generator() {
    let mut app = app();
    let planet = shipped("planet.json");
    let mega = shipped("megastructure.json");
    load_in(
        &mut app,
        vec![
            (planet.clone(), LodConfig::from(&planet.lod)),
            (mega.clone(), LodConfig::from(&mega.lod)),
        ],
    );

    let render = app.world().resource::<RenderWorlds>();
    assert_eq!(
        render.get(0).unwrap().program.ops.as_slice(),
        voxel_worldgen::program::planet_program(),
    );
    // World 1 against the level it was loaded FROM, not against a
    // fixture: what this test is for is that the second registration
    // does not quietly reuse the first world's slot, and comparing each
    // world to its own JSON says that without pinning either level's
    // contents (the megastructure's are checked by their properties, in
    // `every_megastructure_district_is_whole`).
    assert_eq!(
        render.get(1).unwrap().program.ops.as_slice(),
        mega.generator(0).ops(),
    );
    assert_ne!(
        render.get(0).unwrap().program.ops.as_slice(),
        render.get(1).unwrap().program.ops.as_slice(),
    );
    // And the CPU twin agrees with the GPU one, per world.
    let worlds = app.world().resource::<Worlds>();
    assert_eq!(
        worlds.get(1).unwrap().generator.ops(),
        render.get(1).unwrap().program.ops.as_slice(),
    );
}

/// The shipped levels use overlapping material ids on purpose: planet
/// paints 1/3/4 and the megastructure paints 2. Loading both used to
/// require an `assert!` in the host asking them not to collide, because
/// one global table served every world.
#[test]
fn two_levels_may_use_the_same_material_ids() {
    let mut app = app();
    let planet = shipped("planet.json");
    // Make the collision explicit rather than relying on the shipped ids
    // staying disjoint: give the megastructure planet's id 1 as well, in
    // a colour planet never uses.
    let mega = level_with(
        "megastructure.json",
        "materials",
        json!([{"type": "surface", "id": 1, "base": [1.0, 0.0, 1.0]}]),
    );

    load_in(
        &mut app,
        vec![
            (planet.clone(), LodConfig::from(&planet.lod)),
            (mega.clone(), LodConfig::from(&mega.lod)),
        ],
    );

    let render = app.world().resource::<RenderWorlds>();
    let planet_1 = render.get(0).unwrap().materials[1];
    let mega_1 = render.get(1).unwrap().materials[1];
    assert_ne!(
        planet_1, mega_1,
        "id 1 must mean each level's own recipe, not whichever loaded last",
    );
    // The other level's ids do not leak in as anything but the neutral
    // default: world 1 never defined 3.
    assert_eq!(
        render.get(1).unwrap().materials[3],
        voxel_render::WorldMaterial::default(),
    );
}

/// A world is worth streaming at less than its authored detail when it is
/// only ever seen through a portal, so the config is per world and not
/// read back off the level.
#[test]
fn a_world_keeps_the_config_it_was_loaded_with() {
    let mut app = app();
    let planet = shipped("planet.json");
    let full = LodConfig::from(&planet.lod);
    let mut capped = full.clone();
    capped.max_level = 5;
    capped.top_radius = 1;
    assert_ne!(full.max_level, capped.max_level, "the fixture must differ");

    load_in(
        &mut app,
        vec![(planet.clone(), full.clone()), (planet, capped.clone())],
    );

    let worlds = app.world().resource::<Worlds>();
    assert_eq!(worlds.get(0).unwrap().config.max_level, full.max_level);
    assert_eq!(worlds.get(1).unwrap().config.max_level, capped.max_level);
    assert_eq!(worlds.get(1).unwrap().config.top_radius, 1);
}

/// Ops providers are per world and indexed by id, so a world with nothing
/// to plan gets `None` rather than its neighbour's planner. Serving one
/// world's planning to another asked its graph about coordinates in a
/// world it had never generated — and since worlds share coordinates it
/// answered, with the wrong level's roads and ruins.
#[test]
fn a_world_with_nothing_to_plan_gets_no_provider() {
    let mut app = app();
    let planet = shipped("planet.json");
    // No host planner is installed, so a level's ops come only from its
    // authored placements. Strip them from one world and not the other.
    let mut bare = planet.clone();
    bare.placements.clear();
    bare.prefabs.clear();
    let placed = level_with(
        "planet.json",
        "placements",
        json!([{
            "position": [0.0, 0.0, 0.0],
            "ops": [{"shape": "box", "center": [0, 0, 0], "half": [1, 1, 1]}],
        }]),
    );

    load_in(
        &mut app,
        vec![
            (placed.clone(), LodConfig::from(&placed.lod)),
            (bare.clone(), LodConfig::from(&bare.lod)),
        ],
    );

    let providers = app.world().resource::<Worlds>().ops_providers();
    assert_eq!(providers.len(), 2, "one entry per world, indexed by id");
    assert!(providers[0].is_some(), "world 0 has authored placements");
    assert!(
        providers[1].is_none(),
        "world 1 plans nothing and must not inherit world 0's provider",
    );
}

/// `get` is exact; `query` deliberately falls back to world 0 for callers
/// that mean "whatever the player is looking at" and can be asked before
/// a world exists.
#[test]
fn an_absent_world_is_absent() {
    let mut app = app();
    let planet = shipped("planet.json");
    load_in(&mut app, vec![(planet.clone(), LodConfig::from(&planet.lod))]);

    let worlds = app.world().resource::<Worlds>();
    assert!(worlds.get(0).is_some());
    assert!(worlds.get(3).is_none(), "get must not invent a world");
    assert!(worlds.query(3).is_some(), "query falls back to world 0");

    let empty = Worlds::default();
    assert!(empty.query(0).is_none(), "and to nothing when none exist");
}

/// The slab is ONE allocation shared by every world, so loading a second
/// re-divides it rather than handing over the leftovers. First-come
/// admission gave the launched level everything it asked for and the next
/// one the scraps — with three loaded, the third got 173 of 3656 slots
/// and streamed a horizon and nothing else.
#[test]
fn the_slab_is_shared_between_worlds_not_claimed_in_order() {
    let mut app = app();
    let planet = shipped("planet.json");
    let mega = shipped("megastructure.json");
    let purgatory = shipped("purgatory.json");
    load_in(
        &mut app,
        vec![
            (planet.clone(), LodConfig::from(&planet.lod)),
            (mega.clone(), LodConfig::from(&mega.lod)),
            (purgatory.clone(), LodConfig::from(&purgatory.lod)),
        ],
    );

    let worlds = app.world().resource::<Worlds>();
    // Headless, so no render world has reported what chunks cost yet;
    // admission falls back to the configured page count, which is the
    // same bound it uses on the first frame of a real run.
    let capacity = voxel_render::slab::SlabConfig::default().total_pages as usize;
    // Over-committing a little is the FLOOR, by design: a world dense
    // enough to want more than its share even at `MIN_STREAMED_LEVEL` is
    // admitted anyway rather than refused, and the excess is absorbed by
    // deferral, which no longer wedges. Bounded, though — this is the
    // difference between "slightly oversubscribed" and "pretending".
    let committed: usize = worlds.iter().map(|w| w.slab_demand).sum();
    assert!(
        committed <= capacity * 3 / 2,
        "three worlds committed {committed}, more than half again the {capacity} available",
    );
    // The point: NONE of them is starved down to a token horizon. An
    // equal share of the slab is capacity/3; every world gets within
    // reach of that or its full authored demand, whichever is smaller.
    let floor = capacity / 3 / 4;
    for world in worlds.iter() {
        assert!(
            world.slab_demand >= floor,
            "world {} got {} slots, under a quarter of an equal share ({floor})",
            world.id,
            world.slab_demand,
        );
    }
}

/// Re-fitting is from the AUTHORED config, so a world capped while
/// others were loaded is not permanently ratcheted down by it.
#[test]
fn capping_is_recomputed_from_what_the_host_asked_for() {
    let mut app = app();
    let planet = shipped("planet.json");
    let mega = shipped("megastructure.json");
    load_in(
        &mut app,
        vec![
            (planet.clone(), LodConfig::from(&planet.lod)),
            (mega.clone(), LodConfig::from(&mega.lod)),
        ],
    );
    let worlds = app.world().resource::<Worlds>();
    for world in worlds.iter() {
        assert_eq!(
            world.authored.max_level,
            LodConfig::from(&world.level.lod).max_level,
            "world {} forgot what it was asked for",
            world.id,
        );
        assert!(world.config.max_level <= world.authored.max_level);
    }
}

/// One world alone must not be capped: the shipped level has to look the
/// way it was authored. If this fails the budget is too small for the
/// world it ships with, which is a sizing bug, not a policy one.
#[test]
fn the_first_world_streams_at_its_authored_detail() {
    let mut app = app();
    let planet = shipped("planet.json");
    let full = LodConfig::from(&planet.lod);
    load_in(&mut app, vec![(planet.clone(), full.clone())]);

    let worlds = app.world().resource::<Worlds>();
    assert_eq!(
        worlds.get(0).unwrap().config.max_level,
        full.max_level,
        "the only loaded world was capped — the slab is too small for it",
    );
    assert!(worlds.get(0).unwrap().slab_demand > 0, "demand was measured");
}

/// Capping is monotone and bounded: it only ever reduces detail, it
/// stops at the floor, and it terminates however small the budget is.
///
/// A world nobody can afford is still admitted, at the floor, rather
/// than looping forever or refusing to load — deferral is safe now, so a
/// few chunks with nowhere to go is a hole, not a wedge.
#[test]
fn admission_only_ever_reduces_detail_and_terminates() {
    use voxel_engine::streaming::{fit_to_budget, meshable_count};
    let anchor = bevy::math::DVec3::ZERO;
    for name in ["planet.json", "megastructure.json"] {
        let level = shipped(name);
        let generator = std::sync::Arc::new(level.generator(0));
        let authored = LodConfig::from(&level.lod);
        for available in [0, 1, 64, 512, 100_000] {
            let (fitted, demand) = fit_to_budget(&authored, &generator, anchor, available, 4);
            assert!(
                fitted.max_level <= authored.max_level,
                "{name}: admission raised detail from L{} to L{}",
                authored.max_level,
                fitted.max_level,
            );
            assert!(fitted.max_level >= 4, "{name}: capped below the floor");
            assert_eq!(
                demand,
                meshable_count(&fitted, &generator, anchor),
                "{name}: the reported demand is not the fitted config's",
            );
            // Either it fits, or it is at the floor and admitted anyway.
            assert!(
                demand <= available || fitted.max_level == 4,
                "{name}: {demand} chunks admitted into {available} slots above the floor",
            );
        }
        // A budget that fits everything must not touch the config.
        let (fitted, _) = fit_to_budget(&authored, &generator, anchor, usize::MAX, 4);
        assert_eq!(fitted.max_level, authored.max_level, "{name}");
    }
}
