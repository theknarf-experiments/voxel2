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

There are no hardcoded worlds. A level file describes everything, most
importantly the **generator program**: an ordered list of composable ops
(FBM height bands, floor lattices, pillar/wall grids, shafts, catwalk
beams, 3D noise carves, domain warp) that one GPU interpreter evaluates as
the world's SDF — a lush planet and a concrete megacity are the same
engine fed different data. Planned structure is data too, declared as a
**planning stack**: a small generic layer vocabulary — `biomes` (blended
regions), `scatter`/`scatter3` (sites), `connect`/`connect3` (pathfound
or orthogonal links), `flow` (descent hydrology), `worm` (burrows), and
`emit` (CSG patches, ribbons, clearance, markers) — that levels
compose by instance name: ruins, roads, rivers, caves, dungeons, and the
megastructure's habitation districts are all stack configurations, not
engine features. Shading follows the same rule: a **material table**
(parameterized `surface`, `zoned`, and `canopy` recipes) indexed by the
material ids the ops emit, an **environment** block for lighting and
haze, and **scatter classes** — populations of props described only by
where they go (density, altitude band, slope, patch noise, biome and
field gates, clearance, weighted variants). The engine streams one
entity per placement carrying a `ScatterInstance`; the host dresses
those entities with its own models, materials and gameplay components,
so no models live in the reusable crates. Even the structures the stack
builds are data: a **structure** is weighted variants of parts, each
placing a shape (box/cylinder/sphere, added or cut, optionally hollowed
into a shell) at every position of an arrangement — `ring`, `scatter`,
`chain`, or a single point — seated on the terrain or an interior floor,
and optionally linked to the next position by a swept tunnel. Ruins
(broken ring wall, tower stubs, rubble), dungeons (a descending room
chain with corridors and a surface entrance), and the megastructure's
habitation pockets are all written that way, with no engine code behind
them. See `levels/*.json` and `voxel_engine::level::LevelDef`.

**Live editing**: the level file is watched while running. Materials,
environment and LOD tuning apply instantly; changes to the generator,
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

**Portals**: `portal` (or **F7**) opens a window onto the other shipped level 14 m
ahead of you, loading it as a second world the first time — both then stream and
render at once, and walking through swaps which one you are in. `world N` switches
the camera's world without moving it. Nothing is loaded until you ask: a second
world roughly doubles the meshed working set.

## How it works

- **Voxels are transient GPU artifacts.** One density compute pass
  interprets the level's generator program (a storage buffer of 64-byte
  ops over a small register file) into a 38³-sample
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
  planning stack, portal clip planes — is indexed by its id in `Worlds`
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
| `voxel-worldgen` | CPU twin of the generator interpreter + the stack vocabulary (scatter/connect/flow/worm/biomes/emit), recipes, hydrology, pathfinding |
| `voxel-render` | Density/meshing compute, slab allocator, LOD draw, grass, materials |
| `voxel-engine` | The LOD field and its layers, chunk generation service, level definitions (JSON), remote tooling |
| `voxel-debug` | Flycam + HUD (fps, chunk/arena/slab occupancy, LOD histogram) |

`tools/voxctl` drives a running instance over the Bevy Remote Protocol;
`mise run setup` installs the pre-commit gate (workspace check + clippy
`-D warnings`).

`cargo test --workspace` runs the property tests (determinism under racing
threads, format round-trips, slab allocation, planning invariants).
