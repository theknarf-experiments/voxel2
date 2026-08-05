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
cargo run -p voxel2 -- levels/planet.json         # grasslands, forests, ocean, ruins, roads
cargo run -p voxel2 -- levels/megastructure.json  # endless concrete interior
cargo run -p scout --release                      # offline: scan for scenic locations
```

There are no hardcoded worlds. A level file describes everything, most
importantly the **generator program**: an ordered list of composable ops
(FBM height bands, floor lattices, pillar/wall grids, shafts, catwalk
beams, …) that one GPU interpreter evaluates as the world's SDF — a lush
planet and a concrete megacity are the same engine fed different data. The
rest of the file covers seed, LOD configuration, lighting, camera, feature
toggles (water/vegetation), walk and shading modes, and parameterized
planning-op providers that author structures (ruins site chance, road
reach, pocket density, …). See `levels/*.json` and
`voxel_engine::level::LevelDef`.

**Live editing**: the level file is watched while running. Lighting,
colors, shading, and camera tuning apply instantly; changes to the
generator, seed, providers, or LOD topology rebuild the streamed world in
place — copying a different level over the watched file swaps the whole
world without a restart.

Flycam: mouse look (hold right button), WASD + QE, shift to run, scroll for
speed. Env vars:

| Var | Effect |
|---|---|
| `VOXEL_START=x,y,z` | Spawn position |
| `VOXEL_LOOK=dx,dy,dz` | Initial look direction |
| `VOXEL_AUTOPILOT=<m/s>` | Fly forward continuously (smoke tests) |
| `VOXEL_WALK=1` | On-foot: heightfield glue or SDF collision (level's `walk` mode) |

## How it works

- **Voxels are transient GPU artifacts.** One density compute pass
  interprets the level's generator program (a storage buffer of 64-byte
  ops over a small register file) into a 38³-sample
  arena slot per chunk; an exact-count pass feeds a staging-ring readback
  that drives bucketed slab allocation; surface-nets compute passes then
  mesh straight into shared vertex/index slabs. Nothing but 8-byte counts
  ever crosses the bus. Everything is deterministic and regenerable.
- **LOD octree with ready-before-swap.** 32³-cell chunks at every level
  (voxel size doubles per level); a main-world controller splits/merges
  with hysteresis and only swaps when every replacement chunk is drawable —
  no holes, no double-LOD. Seams are closed by *stitched surface nets*:
  boundary-band vertices geomorph onto the coarse-parity surface, which is
  bit-identical across neighbors.
- **Compressed geometry.** 12-byte vertices (unorm16 positions, octahedral
  normals, material + baked sun shadow in the spare u16) and packed u16
  indices; drawn camera-relative through a custom phase item (the view
  matrix is applied with w = 0, so world-space f32 error never grows with
  distance).
- **Planning layers drive the GPU** (the LayerProcGen part): CPU layers with
  declared padded dependencies emit compact `CsgOp` lists per chunk —
  ruin sites, then roads connecting them (a road is owned by the chunk
  containing its midpoint). Ops upload with the generation batch and the
  density shader applies them after the base SDF. Determinism is enforced
  by construction and by tests that race generation across thread counts.
- **CPU mirrors for gameplay.** The generator program has a bit-compatible
  Rust twin interpreter: vegetation placement, ruins/roads planning,
  walk-mode collision, and scenic-location scouting all consume the same
  world the GPU renders — one op table, two interpreters.
- **Fully procedural shading.** No texture assets: noise-grained material
  zones, worked-stone with mortar bands and moss, concrete with pour bands
  and grime, emissive light strips, hemispheric ambient, sun-tinted haze,
  and horizon-marched sun shadows baked per vertex at mesh time (free per
  frame). Grass is one instanced draw with wind sway and distance shrink;
  far forests are merged silhouette impostors to ~3 km.

## Workspace

| Crate | Role |
|---|---|
| `voxel-core` | Chunk keys, packed voxel format, CSG op IR, morton, seeding (no Bevy) |
| `voxel-layers` | LayerProcGen framework: padded deps, recursive on-demand generation |
| `voxel-worldgen` | CPU twin of the generator interpreter + planning layers (sites, roads, ruins) |
| `voxel-render` | Density/meshing compute, slab allocator, LOD draw, grass, materials |
| `voxel-engine` | LOD controller, level definitions (JSON), vegetation streaming, walk modes |
| `voxel-debug` | Flycam + HUD (fps, chunk/arena/slab occupancy, LOD histogram) |

`cargo test --workspace` runs the property tests (determinism under racing
threads, format round-trips, slab allocation, planning invariants).
