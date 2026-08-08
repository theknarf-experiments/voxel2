// The opening's surface: the far world, sampled in SCREEN SPACE.
//
// The far camera sits at exactly the same transform as the near one, so
// its image is what you would see if you were in the other world at this
// spot. Sampling it by fragment position therefore lines the two up
// pixel for pixel, with nothing to tune: the far world appears through
// the opening as a continuation of the same view rather than as a
// picture hung in front of it.
//
// This is also what confines the far world to the opening. Clip planes
// live in the per-chunk draw uniform and mask CHUNKS; grass, trees and
// water have their own pipelines and nothing clips them, so a far view
// drawn straight into the frame painted the other world's meadow over
// the whole screen. Here the quad is the mask, by construction, for
// everything the far camera drew.
//
// Fragment only: Bevy's own mesh vertex shader already puts the quad on
// screen, and `VertexOutput.position` arrives as the framebuffer
// coordinate, which is exactly what this needs. A hand-written vertex
// stage would have to reproduce the mesh bind group and instance
// indexing for no gain.

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

@group(3) @binding(0) var far_view: texture_2d<f32>;
@group(3) @binding(1) var far_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalized by THIS view's viewport, so a view of a different size
    // than the far image still samples the right pixel — the offscreen
    // screenshot mirror is 1280x720 against a 2560x1440 window, and both
    // show the same viewpoint at the same aspect.
    let uv = in.position.xy / view.viewport.zw;
    return vec4<f32>(textureSample(far_view, far_sampler, uv).rgb, 1.0);
}
