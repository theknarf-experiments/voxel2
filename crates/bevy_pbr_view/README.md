# bevy_pbr_view

Draw hand-rolled pipelines through Bevy's PBR view bind groups.

A custom draw — instanced scatter, a GPU-driven terrain pass, a
special-cased water surface — still wants Bevy's per-view data: lights,
cascaded shadow maps, clusters, fog, tonemapping LUTs. Groups 0 and 1 of
a PBR pipeline are that data, and the view *layout*, the shader defs and
the color target format must all agree with the bind group Bevy built
for the view. Bevy derives all three from one `MeshPipelineKey`; this
crate derives yours from the same key, the same way.

- `view_key` / `PbrViewQuery` — build the key for a view exactly as
  `bevy_pbr` does.
- `specialize_for_view` / `ViewSpecializer` / `ViewKey` — point a
  descriptor's groups 0/1 at Bevy's view layouts for that key, with the
  matching shader defs and target format.
- `DrawPipeline<M>` — the specialized pipeline variants plus your group-2
  layout, keyed by your marker component.
- `queue_by_marker<M, D>` — put every `M` a view can see into the opaque
  phase, drawn by `D`.

Grew inside a voxel engine where terrain, water and every instanced prop
population each carried a copy of this; the copies are why it is one
crate now.

Currently developed inside the voxel2 workspace and inheriting its Bevy
feature set; when extracted, the dependency needs `bevy_pbr`,
`bevy_core_pipeline` and `bevy_render` enabled explicitly.
