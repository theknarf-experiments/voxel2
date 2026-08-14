# bevy_graph_view

A node-graph view for Bevy UI, drawn with `bsn!`.

Two halves, split where testability splits:

- **`layout`** — pure arithmetic. Describe your document as a list of
  `GraphNode`s (an opaque `id`, a `name` other nodes wire to, port lists,
  `wires`, and `children` for scopes) and get back every box, frame and
  axis-aligned wire segment as plain geometry. Columns are dependency
  depth; boxes are fixed-size, so every edge anchor is known before
  anything is spawned. Testable without a window.
- **`canvas`** — the drawing. `scene()` turns a `Layout` into a `bsn!`
  scene: absolutely-positioned boxes under one pannable, zoomable
  `GraphCanvas` inside a clipping `GraphViewport`, themed with Feathers
  tokens. `GraphCamera` owns the zoom-about-a-point arithmetic; clicked
  boxes report back through the `SelectsNode` component. A `ZoomLabel`
  chip in the viewport's lower corner reads the zoom as a percentage,
  kept current by the `zoom_label` system off the canvas transform.

This crate has no opinion about what a node *is* or what selection
*means* — the host converts its own document into `GraphNode`s and reads
`SelectsNode` ids back out. Nothing is scheduled by `GraphViewPlugin`
(it only registers `GraphStyle` and `GraphCamera` for reflection), so a
host that gates its UI work can run the provided `hover` and `zoom_label`
systems — and its own pan/zoom/select handling — under its own run
conditions.

`demos/graph_view_demo` in this workspace is the crate on its own: a
hand-written document converted to `GraphNode`s, plus the host-side
pan/zoom/select/rebuild systems this crate deliberately does not ship.
`cargo run -p graph_view_demo` opens it; `GRAPH_SHOT=/tmp/g.png` makes it
screenshot itself and exit, for eyeballing a change without a mouse.
When extracted to its own repo it becomes `examples/`.

Currently developed inside the voxel2 workspace and inheriting its Bevy
feature set; when extracted, the dependency needs `bevy_feathers` (and
Bevy's default UI features) enabled explicitly.
