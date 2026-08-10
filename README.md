# voxel2

A GPU-driven voxel engine in Rust/Bevy: infinite chunked worlds with
planet-scale view distance, where **procedural level design runs in real
time on the GPU**. One engine renders both rolling Shadow-of-the-Colossus
grasslands and endless Blame!-style concrete interiors at 100+ fps.

Inspirations: [LayerProcGen] (layered deterministic contextual generation),
[Voxel Plugin] (SDF voxels + LOD octree), stitched surface nets à la
[GDVoxelTerrain].

[LayerProcGen]: https://github.com/runevision/LayerProcGen
[Voxel Plugin]: https://github.com/VoxelPlugin/VoxelPluginFreeLegacy
[GDVoxelTerrain]: https://github.com/JorisAR/GDVoxelTerrain

## Running levels

Worlds are data: one binary presents a JSON level definition.

```sh
cargo run -p voxel2 -- levels/planet.json         # forests, biomes, rivers, ruins, roads, caves, dungeons
cargo run -p voxel2 -- levels/megastructure.json  # endless interior: pocket districts linked by tubes
```

There are no hardcoded worlds. A level file describes everything as **one
list of nodes**. Each node names its `kind`, its `name`, and the nodes it
consumes (`"in": {"height": "base", "warp": "coast"}`), so the dataflow is
written down rather than implied by position in a list. A node's kind is a
registered Rust type, its fields are its schema, and there is no enum
anywhere: a host adds a kind by registering a type, and the type registry
IS the vocabulary — for the file format, the compiler and the editor at
once.

Nodes come in two **domains**, which is a backend decision and not a
second language. Point nodes (FBM height bands, floor lattices,
pillar/wall grids, shafts, catwalk beams, 3D noise carves, domain warp)
compile to the generator program one GPU interpreter evaluates as the
world's SDF — a lush planet and a concrete megacity are the same engine
fed different data. Region nodes have a lifetime and become LayerProcGen
layers: `biomes` (blended regions), `scatter`/`scatter3` (sites),
`connect`/`connect3` (pathfound or orthogonal links), `flow` (descent
hydrology), `worm` (burrows), `emit` (CSG patches, ribbons, clearance,
markers) and `population` (props). Ruins, roads, rivers, caves, dungeons,
and the megastructure's habitation districts are all node graphs, not
engine features. A `region` node is a scope: it carries a gate and the
nodes it applies to, so nine districts are nine nested scopes rather than
the same four numbers repeated on sixty rows.

Shading follows the same rule: a **material table** (parameterized
`surface`, `zoned`, and `canopy` recipes) indexed by the material ids the
ops emit, an **environment** block for lighting and haze, and
**populations** — props described only by where they go (density,
altitude band, slope, patch noise, region and field gates, clearance,
weighted variants). The engine streams one entity per placement carrying
a `ScatterInstance`; the host dresses those entities with its own models,
materials and gameplay components, so no models live in the reusable
crates. Even the structures an `emit` builds are data: a **structure** is weighted variants of parts, each
placing a shape (box/cylinder/sphere, added or cut, optionally hollowed
into a shell) at every position of an arrangement — `ring`, `scatter`,
`chain`, or a single point — seated on the terrain or an interior floor,
and optionally linked to the next position by a swept tunnel. Ruins
(broken ring wall, tower stubs, rubble), dungeons (a descending room
chain with corridors and a surface entrance), and the megastructure's
habitation pockets are all written that way, with no engine code behind
them. See `levels/*.json` and `voxel_engine::level::LevelDef`.

**In-game editor** (**F10**): a panel built entirely out of the level's own
reflection — no widget code per field and no schema restated. `TypeInfo`
says what is in a level, the field's own doc comment supplies the label's
explanation (`reflect_documentation`), and a small attribute vocabulary
(`voxel_engine::schema`) says what the types cannot: which numbers are
bounded, which `[f32; 3]` is a colour, which `u32` is a reference to a
material id rather than a number, and which fields restream the world when
edited. Every row carries the reflect path it came from, so editing is one
observer rather than one per field — the same path `world.mutate_resources`
takes over BRP. A new node kind is editable the moment it is
registered.

**Live editing**: the level file is watched while running. Materials,
environment and LOD tuning apply instantly; changes to the nodes,
providers, or LOD topology rebuild the streamed world in place — copying
a different level over the watched file swaps the whole world without a
restart. The file holds only what the engine owns: the camera, lights,
clear color and prop models are the host's, written in Rust in
`demos/voxel2/src/`, and the seed is a runtime input a game picks per
save.

Flycam: mouse look (hold right button), WASD + QE, shift to run, scroll for
speed. Env vars:

| Var | Effect |
|---|---|
| `VOXEL_START=x,y,z` | Spawn position |
| `VOXEL_LOOK=dx,dy,dz` | Initial look direction |
| `VOXEL_AUTOPILOT=<m/s>` | Fly forward continuously (smoke tests); `VOXEL_AUTOPILOT_LEVEL=1` keeps it level |
| `VOXEL_SCREENSHOT=path[,secs]` | Periodic offscreen frame dumps (works occluded) |
| `VOXEL_EVAL_HOLES=1` | Coverage-eval rendering (used by `mise run eval`) |
| `VOXEL_LOG_LAYERS=<n>` | Log the ensure-load passes and the first `n` read-driven generations (with a backtrace) |
| `VOXEL_ENSURE_THREADS=<n>` | Override the planning ensure-load worker count |

**Live tooling**: dev builds always serve the BRP; drive the running game:
`cargo run -p voxctl -- status | goto X Y Z [DX DY DZ] | ribbons X Z [R] |
markers X Z [R] [KIND] | scan X Z [R] [STEP] | shot PATH | portal | world N |
raw METHOD [JSON]`
— `scan` ranks
scenic spots (steep, high terrain) from the CPU world mirror. In-game overlays: F8 chunk
boundaries by LOD, F9 planning layers (markers, clearance, ribbons, biome
field).

**Portals**: `portal [N]` (or **F1**, **F2**, … — one key per other shipped
level, avoiding the F8/F9 overlays) opens a window onto that level 14 m ahead of
you, loading it as another world the first time. All of them then stream and
render at once, and walking through swaps which one you are in; pressing a
level's own key again closes the opening, and a different level's key switches
where it looks. `world N` switches the camera's world without moving it.

A LOADED LEVEL AND AN OPENING ARE DIFFERENT THINGS: closing a portal leaves the
world loaded and streaming. Nothing is loaded until you ask, because each world
is admitted against the slab budget and caps what the others can stream.

## How it works

- **Voxels are transient GPU artifacts.** One density compute pass
  interprets the program the level's point nodes compile to (a storage
  buffer of 64-byte ops over a small register file) into a 38³-sample
  arena slot per chunk; an exact-count pass feeds a staging-ring readback
  that drives bucketed slab allocation; surface-nets compute passes then
  mesh straight into shared vertex/index slabs. Nothing but 8-byte counts
  ever crosses the bus. Everything is deterministic and regenerable.
- **LOD levels are layers, with ready-before-swap.** 32³-cell chunks at
  every level (voxel size doubles per level). Which chunks exist is not
  planned: it is a pure function of the chunk and a sticky camera anchor,
  declared to the dependency graph as one top dependency per level, so
  residency IS the drawn set. A chunk's `create` blocks until it is
  drawable and the graph runs every ensure before any release, so a
  replacement always exists before the chunk it replaces goes — no holes,
  no double-LOD, and no epoch machine to arrange it. Seams are closed by
  *stitched surface nets*: boundary-band vertices geomorph onto the
  coarse-parity surface, which is bit-identical across neighbors.
- **Compressed geometry.** 12-byte vertices (unorm16 positions, octahedral
  normals, material + baked sun shadow in the spare u16) and packed u16
  indices; drawn camera-relative through a custom phase item (the view
  matrix is applied with w = 0, so world-space f32 error never grows with
  distance).
- **Planning layers drive the GPU** (the LayerProcGen part): generation is
  *dependency-driven* — nothing generates except to satisfy a declared
  dependency, resolved nearest-first, in parallel and off the main thread,
  and a read returns what is resident rather than generating it.
  `reads_missed` in the debug status reports any consumer whose working
  set a top dependency does not cover. The level's stack builds one
  `LayerGraph` of CPU layers with declared padded dependencies; `emit`
  layers bucket their output by owning cell (a road is owned by the chunk
  containing its midpoint) and a `WorldQuery`
  facade per world serves CSG ops to chunks (with per-emitter carve-horizon gates),
  clearance to spawners, ribbon segments to the host that draws them,
  and markers to gameplay. Ops upload with the generation batch and the density shader
  applies them after the base SDF. Determinism is enforced by construction
  and by tests that race generation across thread counts.
- **Worlds are plural.** A level is loaded as a world through one path
  (`WorldLoader::load`), including the first one, and everything about it
  — generator program, LOD config, material table, painted surface map,
  planning graph, portal clip planes — is indexed by its id in `Worlds`
  and `RenderWorlds`. Several coexist: they share one chunk service, one
  GPU arena and one program buffer, because the world rides in
  `ChunkKey`. They also share COORDINATES, which is why nothing per-world
  may be stored globally — see `crates/voxel-engine/tests/worlds.rs`.
- **CPU mirrors for gameplay.** The generator program's interpreter is
  generated from ONE op table (`voxel-core::opgen`) into both Rust and
  WGSL: vegetation placement, planning layers, and terrain queries all
  consume the same world the GPU renders.
- **Fully procedural, fully data-driven shading.** No texture assets and
  no world-specific shader branches: one fragment path indexes the level's
  material table (noise-grained bands, grime, streaks, moss, emissive
  light strips, altitude-zoned terrain) and its environment uniform
  (sun, hemispheric ambient, tinted haze). Horizon-marched sun shadows
  bake per vertex at mesh time (free per frame). Grass is one instanced
  draw with wind sway and distance shrink; far forests are merged
  silhouette impostors to ~3 km.

## Workspace

| Crate | Role |
|---|---|
| `voxel-core` | Chunk keys, packed voxel format, CSG op IR, morton, seeding (no Bevy) |
| `voxel-layers` | LayerProcGen framework: padded deps, recursive on-demand generation |
| `voxel-worldgen` | CPU twin of the generator interpreter, plus the primitives the game's layers are built from: scatter points, descent walks, A* paths, ribbons |
| `voxel-render` | Density/meshing compute, slab allocator, LOD draw, grass, materials |
| `voxel-engine` | The LOD field and its layers, chunk generation service, level definitions (JSON), remote tooling |
| `voxel-debug` | Flycam + HUD (fps, chunk/arena/slab occupancy, LOD histogram) |
| `voxel-editor` | Level editor: reflection → Feathers rows, `bsn!` scenes, edits by reflect path |

`tools/voxctl` drives a running instance over the Bevy Remote Protocol;
`mise run setup` installs the pre-commit gate (`cargo fmt --check`,
workspace check, clippy `-D warnings`).

`cargo test --workspace` runs the property tests (determinism under racing
threads, format round-trips, slab allocation, planning invariants).
