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

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}

@group(3) @binding(0) var far_view: texture_2d<f32>;
@group(3) @binding(1) var far_sampler: sampler;

struct VsIn {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vertex(in: VsIn) -> VsOut {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    let world = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(in.position, 1.0),
    );
    var out: VsOut;
    out.clip = position_world_to_clip(world.xyz);
    return out;
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    // `clip` arrives as framebuffer coordinates; the viewport's size turns
    // them into the UV of the same pixel in the far view's image.
    let uv = in.clip.xy / view.viewport.zw;
    return vec4<f32>(textureSample(far_view, far_sampler, uv).rgb, 1.0);
}
