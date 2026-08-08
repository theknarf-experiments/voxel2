# voxel2 — agent notes

GPU-driven voxel engine (Rust, Bevy 0.19 pinned). Read `README.md` for
architecture. Design plan: `~/.claude/plans/binary-twirling-brooks.md`.

## Build / test / run

- `cargo test --workspace` — property tests; keep green.
- `mise run smoke` — boots every shipped level with NO env vars and fails
  on any panic. Every other check sets `VOXEL_REMOTE`/`VOXEL_START`, so a
  system that only breaks on a plain `cargo run` survives all of them.
  Run it before claiming a change works.
- `cargo run -p voxel2 -- levels/<name>.json` — visual verification is
  mandatory for render changes: run ~35 s (LOD refinement needs time),
  screenshot, and look. Use `caffeinate -dis` so the display can't sleep,
  and take shots with `voxctl shot` — NEVER `screencapture -R`, which
  grabs whatever is on screen, not the app.
- Env vars for repeatable scenes: `VOXEL_START=x,y,z`, `VOXEL_LOOK=dx,dy,dz`,
  `VOXEL_AUTOPILOT=<m/s>`.
- PREFER live verification over relaunch cycles: run with `VOXEL_REMOTE=1`
  and drive via `cargo run -p voxctl -q -- status | goto X Y Z [DIR] |
  ribbons/markers/ops X Z [R] | scan X Z [R] [STEP] | shot PATH`
  (offscreen screenshots; wait ~1 s for the file, ~10-15 s after a goto
  for streaming). `scan` ranks scenic spots; `portal`/F7 opens a
  window onto a second level (loaded on demand) and `world N` switches
  which world the camera is in; F8/F9 (or `raw voxel/viz`)
  toggle chunk/layer debug overlays.
- Zero `Validation Error` lines in the log is part of "verified".
- Kill running app processes before spawning another and after each
  capture.
- **fps is only meaningful in the foreground, settled, with nothing else
  compiling.** The display is 120 Hz and vsync caps there, so anything
  above ~120 means the window was backgrounded and no shaders ran — a
  measurement of nothing. Below it, a reading can still be contention
  from a concurrent build (too low) or a pre-settle transient (too high).
  Sample 3-4 times over 60+ s and compare like with like.

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
  set (voxel-core::worldop + both interpreters + GenOpDef — vegetation is
  scatter data), the material recipe kinds (voxel-render WorldMaterial ↔
  MaterialDef::pack ↔ the WGSL MaterialTable — field-position layout
  twins), or the host's planning vocabulary, and express the world in the
  level JSON. `voxel_engine` tests pin the shipped JSONs to the reference
  programs.
- **The crates hold no named nouns and no concrete layers.** A reusable
  crate may contain primitives (ribbon, scatter point, descent walk,
  A* path) but never an instance of a domain noun — water, river, grass,
  tree, ruin are level data the host interprets. Likewise `voxel-layers`
  is the LayerProcGen *framework* only; concrete layers are the game's,
  written against `voxel_engine::planning::WorldPlanner` (this demo's
  live in `demos/voxel2/src/planning/`, driven by the level's opaque
  `planning` block — NEVER add a hand-written structure recipe fn; a new
  structure is level JSON). The engine keeps only what seams depend on:
  the ops horizon, the density apron, and per-chunk (never per-op)
  gating.
- Count pass and emit passes in `voxel_mesh_chunks.wgsl` must agree
  *exactly* on skip rules — allocation uses counted values.
- `map_async` on the counts staging ring only the frame *after* the copy
  submits (wgpu validation error otherwise).
- Never blend disagreeing SDFs across LOD (phantom surfaces); hard-cut and
  let fog cover it.
- Slab exhaustion wedges generation (AwaitingAlloc holds arena slots): the
  HUD shows `arena free: 0` + full classes. Fix class sizing, not budgets.
- Layer determinism: all randomness from `chunk_seed`; reads only within
  declared padded bounds (asserted, with the needed padding in the message).
- Planning generation is **dependency-driven**: consumers `ensure_loaded`
  the region they are about to query (the LOD planner does this per epoch,
  plus a streamer-radius pass), then read. `voxctl status` →
  `stream.read_generated` must stay ~0; anything else means a consumer's
  working set is uncovered and is generating on whatever thread reads.
- CPU mirrors sample the program via `program::with_program` (thread-local
  snapshot). Never call `program::program()` per sample — the `Arc` clone
  writes a shared cache line and makes parallel generation slower than
  serial.

## Style

- Comments state constraints, not narration. Milestone-sized commits with
  a "Verified:" line. Update `~/.claude/.../memory/` after each commit.
