# bevy_impostors

Instanced point-scatter props for Bevy, and a crossed-silhouette
impostor that draws a forest's worth of them in one call.

An instance is 16 bytes — a position and a hash — so a population is
bounded by how many points you publish, not by the renderer: half a
million impostors is 8 MB of vertex buffer and one draw per view.
Everything shades through Bevy's PBR view bind groups (via
`bevy_pbr_view`), so instances are lit, fogged and tonemapped like every
other surface in the frame.

Two layers:

- **`prop`** — the framework. A population is a `Prop` impl: its static
  mesh, its shader, its group-2 uniform, all in the marker component the
  host spawns one of per instance *set* (a world, a map, a floor — the
  crate does not care). `PropPlugin<P>` supplies buffers, pipeline, bind
  group, extract, prepare, queue and draw. Points arrive through
  `PropPoints<P>`; `replace()` wholesale, re-uploaded only when dirty,
  allocations kept and grown but never shrunk.
- **`impostor`** — the shipped population. Two fixed crossed planes
  (billboards swim as the camera turns; crossed planes read as a solid
  from any angle), shaped per instance from the hash: bits 0–7 yaw, 8–15
  size, 16–23 silhouette pick + sway phase, 24–31 a baked shade factor.
  The fragment path is a tuned sun-and-ambient read off Bevy's own
  lights buffer — measured: the pass costs what its fragment shader
  costs, so that is where the shader spends nothing it can avoid.
  `ImpostorStyle` holds the two silhouette colors and the fade-in /
  cull distances; `IMPOSTOR_FADE_FROM` is where the far fade starts, for
  hosts aligning it with whatever takes over beyond.

The host's half is deliberately small: spawn one marker per set (with
whatever visibility layers your views use), publish points, optionally
drive the style every frame. In the voxel2 workspace the demo maps
worlds onto sets, bridges the engine's scatter classes into
`PropPoints`, and takes the impostor palette from the tree props it
stands in for — the crate never hears the word "tree".

Currently developed inside the voxel2 workspace and inheriting its Bevy
feature set; when extracted, the dependency needs `bevy_pbr`,
`bevy_core_pipeline`, `bevy_render` and embedded assets enabled
explicitly.
