# voxel2 — agent notes

GPU-driven voxel engine (Rust, Bevy 0.19 pinned). Read `README.md` for
architecture. Design plan: `~/.claude/plans/binary-twirling-brooks.md`.

## Build / test / run

- `cargo test --workspace` — property tests; keep green.
- `cargo run -p voxel2 -- levels/<name>.json` — visual verification is
  mandatory for render changes: run ~35 s (LOD refinement needs time),
  screenshot, and look. Use `caffeinate -dis` so the display can't sleep,
  and capture only the app window region: `screencapture -x -R390,160,1245,730 out.png`.
- Env vars for repeatable scenes: `VOXEL_START=x,y,z`, `VOXEL_LOOK=dx,dy,dz`,
  `VOXEL_AUTOPILOT=<m/s>`, `VOXEL_WALK=1`. `cargo run -p scout --release`
  scans for scenic spots (edit its main.rs per need).
- Zero `Validation Error` lines in the log is part of "verified".

## Invariants that bite

- **WGSL/Rust twins must stay in sync**: the generator-program interpreter
  (`voxel_world_density.wgsl` ↔ `voxel_worldgen::program::eval` — same op
  semantics, register file, and bit-exact integer hashing; the height-only
  loops in the mesh/water shaders ↔ `eval_height`), `ChunkParams` struct
  (2 shaders + chunks.rs), `CsgOp` and `WorldOp` layouts (voxel-core ↔
  WGSL structs — note `meta` is a reserved WGSL word, the GPU field is
  `head`), vertex packing (mesh shader ↔ draw shader ↔ slab layouts),
  baked-shadow march (mesh shader ↔ voxel_worldgen::sun_shadow).
- **Worlds are data, not code**: never add a world-kind enum,
  world-specific shader, feature flag, or shading branch; extend the op
  set (voxel-core::worldop + both interpreters + GenOpDef — water and
  vegetation are meta ops) or the material recipe kinds
  (voxel-render WorldMaterial ↔ MaterialDef::pack ↔ the WGSL
  MaterialTable — field-position layout twins) and express the world in
  the level JSON. Lighting/haze are the level's `environment` block.
  `voxel_engine` tests pin the shipped JSONs to the reference programs.
- Count pass and emit passes in `voxel_mesh_chunks.wgsl` must agree
  *exactly* on skip rules — allocation uses counted values.
- `map_async` on the counts staging ring only the frame *after* the copy
  submits (wgpu validation error otherwise).
- Never blend disagreeing SDFs across LOD (phantom surfaces); hard-cut and
  let fog cover it.
- Slab exhaustion wedges generation (AwaitingAlloc holds arena slots): the
  HUD shows `arena free: 0` + full classes. Fix class sizing, not budgets.
- Layer determinism: all randomness from `chunk_seed`; reads only within
  declared padded bounds (debug-assert enforced).

## Style

- Comments state constraints, not narration. Milestone-sized commits with
  a "Verified:" line. Update `~/.claude/.../memory/` after each commit.
