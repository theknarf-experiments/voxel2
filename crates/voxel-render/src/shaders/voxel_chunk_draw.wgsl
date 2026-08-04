// M2 prototype chunk draw: plain vertex-attribute pipeline over the
// compute-generated vertex/index buffers, with a fixed-direction lambert
// shade so the surface reads clearly.

#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
}

@vertex
fn vertex(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = view.clip_from_world * vec4<f32>(in.pos, 1.0);
    out.normal = in.normal;
    return out;
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.5, 0.8, 0.3));
    let nd = max(dot(normalize(in.normal), light), 0.0);
    let base = vec3<f32>(0.55, 0.55, 0.6);
    return vec4<f32>(base * (0.25 + 0.75 * nd), 1.0);
}
