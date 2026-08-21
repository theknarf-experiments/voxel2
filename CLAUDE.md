# voxel2 — agent notes

GPU-driven voxel engine (Rust, Bevy 0.19 pinned). Read `README.md` for
architecture. Design plan: `~/.claude/plans/binary-twirling-brooks.md`.

## Build / test / run

- `cargo test --workspace` — property tests; keep green.
- `mise run smoke` — boots every shipped level with NO env vars and fails
  on any panic. Checks that set `VOXEL_START` and friends can hide a
  system that only breaks on a plain launch, so a
  system that only breaks on a plain `cargo run` survives all of them.
  Run it before claiming a change works.
- `mise run fly` — flies a WEAVING autopilot through every level. Booting
  only exercises the first residency pass; everything that needs a chunk,
  a scatter tile or a planning layer to be RELEASED survives `smoke`. The
  weave is load-bearing: a straight line moves a monotone front (created
  ahead, released behind, never the same tile twice in a frame), so it
  cannot catch the create-then-release races. `dress_scatter` panicked on
  exactly one, straight flight missed it for 100 s, weaving found it in
  under 40. Run this too before claiming a change works.
- Both are for changes that reach GENERATION: the engine, the renderer,
  planning, a level. An editor-only change does not need them — with one
  exception that has actually happened: the panel's exclusive systems ran
  every frame and split the schedule, and `fly` exhausted the slabs with
  the panel SHUT. So run them when the editor's SCHEDULING changes (a new
  system, a changed run condition), not when its drawing does.
- **Build first, then launch the binary directly.** `cargo build
  --workspace && ./target/debug/voxel2 levels/planet.json` is up and
  SETTLED in ~6 s; `cargo run` bundles a rebuild into that wait and makes
  it 80+. Poll for it rather than sleeping blind: loop on `voxctl status`
  until `stream.settled` is true.
- `cargo run -p voxel2 -- levels/<name>.json` — visual verification is
  mandatory for render changes: settle, then let LOD refine,
  screenshot, and look. Use `caffeinate -dis` so the display can't sleep,
  and take shots with `voxctl shot` — NEVER `screencapture -R`, which
  grabs whatever is on screen, not the app.
- Env vars for repeatable scenes: `VOXEL_START=x,y,z`, `VOXEL_LOOK=dx,dy,dz`,
  `VOXEL_AUTOPILOT=<m/s>`.
- PREFER live verification over relaunch cycles: a dev build always
  serves the BRP, so just drive it via `cargo run -p voxctl -q -- status | goto X Y Z [DIR] |
  ribbons/markers X Z [R] | scan X Z [R] [STEP] | shot PATH`
  (offscreen screenshots; wait ~1 s for the file, ~10-15 s after a goto
  for streaming). `scan` ranks scenic spots; `portal [N]`/F1,F2,… (one key
  per other shipped level) opens a window onto it, loaded on demand and
  left loaded when the opening closes, and `world N` switches which world
  the camera is in; F8/F9 (or `raw voxel/viz`) toggle chunk/layer debug
  overlays and F10 the level editor — do not reuse those three.
- **`voxctl shot` renders an offscreen 3D target with NO UI on it.** To see
  the editor or the HUD you need `shot PATH --window`, which is black
  unless the window is in the FOREGROUND — check the file size (a real
  frame is megabytes, a black one ~57 KB) rather than trusting the shot.
  Drive the editor from tooling with
  `raw world.mutate_resources '{"resource":"voxel_editor::EditorState",
  "path":".open","value":true}'`. `EditorState` also carries `.expanded`
  (which rows are open), `.selected` (the node the graph is inspecting, by
  reflect path), `.camera` (`{pan,zoom}`), `.width`, and `.save`/`.undo`/
  `.redo` — flags so the keystrokes are drivable from a script, which is
  the only way any of it is testable on a running panel.
- Zero `Validation Error` lines in the log is part of "verified".
- Kill running app processes before spawning another and after each
  capture.
- **fps is only meaningful in the foreground, settled, with nothing else
  compiling.** The display is 120 Hz and vsync caps there, so anything
  above ~120 means the window was backgrounded and no shaders ran — a
  measurement of nothing. Below it, a reading can still be contention
  from a concurrent build (too low) or a pre-settle transient (too high).
  Sample 3-4 times over 60+ s and compare like with like. Read
  `frame_ms.mean` (a 120-frame average), NOT `fps`, which is one sample and
  swung 39→91 between consecutive reads of an idle scene; the means either
  side of that were 11.7 and 11.8 ms.

## Invariants that bite

- **WGSL/Rust twins must stay in sync**: the generator-program interpreter
  (`voxel_world_density.wgsl` ↔ `voxel_worldgen::program::eval` — same op
  semantics, register file, and bit-exact integer hashing; the height-only
  loops in the mesh/water shaders ↔ `eval_height`), `ChunkParams` struct
  (2 shaders + chunks.rs), `CsgOp` and `WorldOp` layouts (voxel-core ↔
  WGSL structs — note `meta` is a reserved WGSL word, the GPU field is
  `head`), vertex packing (mesh shader ↔ draw shader ↔ slab layouts).
  The baked-shadow march has NO Rust twin any more: the CPU mirror was
  unused and deleted, so the shader is the only definition.
- **Worlds are data, not code**: never add a world-kind enum,
  world-specific shader, feature flag, or shading branch; extend the op
  set (voxel-core::worldop + both interpreters + a node struct — vegetation
  is population data), the material recipe kinds (voxel-render
  WorldMaterial ↔ MaterialDef::pack ↔ the WGSL MaterialTable —
  field-position layout twins), or the host's node vocabulary, and express
  the world in the level JSON. `voxel_engine` tests pin the shipped JSONs
  to the reference programs.
- **The crates hold no named nouns and no concrete layers.** A reusable
  crate may contain primitives (ribbon, scatter point, descent walk,
  A* path) but never an instance of a domain noun — water, river, grass,
  tree, ruin are level data the host interprets. Likewise `voxel-layers`
  is the LayerProcGen *framework* only; concrete layers are the game's,
  written against `voxel_engine::planning::WorldPlanner` (this demo's
  live in `demos/voxel2/src/planning/nodes.rs` as node kinds — NEVER add
  a hand-written structure recipe fn; a new structure is level JSON).
  A node whose ports touch the HOST's vocabulary is the host's, even when
  the engine owns everything else about it: `population` is a demo newtype
  over the engine's `ScatterDef`, because an engine node wired to a host
  value makes a level's engine half unreadable on its own. The engine
  keeps only what seams depend on: the ops horizon, the density apron, and
  per-chunk (never per-op) gating.
- **One node list, one referencing rule.** A level is `nodes[]`; a node is
  `{kind, name, in, ...params}`, and `in` names EVERY input — there is no
  ordering rule that supplies one implicitly. Source order is program
  order and the compiler verifies it rather than sorting (a topological
  sort is not unique). Never reintroduce a hand-written index: field slots
  and gate references come from node names, resolved by
  `voxel_engine::graph::compile`.
- Count pass and emit passes in `voxel_mesh_chunks.wgsl` must agree
  *exactly* on skip rules — allocation uses counted values.
- `map_async` on the counts staging ring only the frame *after* the copy
  submits (wgpu validation error otherwise).
- Never blend disagreeing SDFs across LOD (phantom surfaces); hard-cut and
  let fog cover it.
- Slab exhaustion wedges generation (AwaitingAlloc holds arena slots): the
  HUD shows `arena free: 0`, and `slab: P/Q pages` with a `longest free
  run` far below what a chunk needs. The pool is pages, not size classes
  (a7da99c → aa5c845), so the fix is fragmentation or total pages — never
  budgets.
- Layer determinism: all randomness from `chunk_seed`; reads only within
  declared padded bounds (asserted, with the needed padding in the message).
- **Detail volumes refine beyond distance, and their coverage is pinned
  deps, not bigger boxes.** `lod.detail` biases the split rule; the bias
  fades across space AND scale (both fades load-bearing — see
  `DetailVolume`). Anything that covers residency near a volume — the
  LOD side's pinned per-level deps and the demo's pinned planning deps —
  sizes from `detail_reach_m`, never from `resident_reach`, which
  deliberately excludes volumes. A consumer sized without it silently
  clips the refined island, a hole that shows only from some camera
  positions.
- Planning generation is **dependency-driven**: consumers `ensure_loaded`
  the region they are about to query (the LOD planner does this per epoch,
  plus a streamer-radius pass), then read. `voxctl status` →
  `planning.reads_missed` must stay 0; anything else means a consumer's
  working set is uncovered and is generating on whatever thread reads.
- CPU mirrors sample the program via `program::with_program` (thread-local
  snapshot). Never call `program::program()` per sample — the `Arc` clone
  writes a shared cache line and makes parallel generation slower than
  serial.

## Style

- Comments state constraints, not narration. Milestone-sized commits with
  a "Verified:" line. Update `~/.claude/.../memory/` after each commit.
