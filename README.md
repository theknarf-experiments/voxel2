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

A level file describes the world kind, seed, LOD configuration, lighting,
camera, feature toggles (water/vegetation), and which named planning-op
providers author structures ("ruins", "roads", "pockets"). See
`levels/*.json` and `voxel_engine::level::LevelDef`.

Flycam: mouse look (hold right button), WASD + QE, shift to run, scroll for
speed. Env vars:

| Var | Effect |
|---|---|
| `VOXEL_START=x,y,z` | Spawn position |
| `VOXEL_LOOK=dx,dy,dz` | Initial look direction |
| `VOXEL_AUTOPILOT=<m/s>` | Fly forward continuously (smoke tests) |
| `VOXEL_WALK=1` | On-foot: terrain glue (planet) / SDF collision (mega) |

## How it works

- **Voxels are transient GPU artifacts.** Density compute passes evaluate a
  world SDF (FBM terrain or architectural CSG lattice) into a 38³-sample
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
- **CPU mirrors for gameplay.** The terrain height and megastructure SDF
  are mirrored bit-compatibly in Rust: vegetation placement, ruins/roads
  planning, walk-mode collision, and scenic-location scouting all consume
  the same world the GPU renders.
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
| `voxel-worldgen` | CPU world mirrors + planning layers (sites, roads, ruins, mega SDF) |
| `voxel-render` | Density/meshing compute, slab allocator, LOD draw, grass, materials |
| `voxel-engine` | LOD controller, level definitions (JSON), vegetation streaming, walk modes |
| `voxel-debug` | Flycam + HUD (fps, chunk/arena/slab occupancy, LOD histogram) |

`cargo test --workspace` runs the property tests (determinism under racing
threads, format round-trips, slab allocation, planning invariants).
