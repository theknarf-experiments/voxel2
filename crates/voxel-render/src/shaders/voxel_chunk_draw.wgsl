// Chunk draw: camera-relative vertex transform over the slab buffers.
//
// Vertices are chunk-local; the per-chunk uniform carries the chunk origin
// relative to the camera (computed in f64 on CPU). Multiplying the view
// matrix with w = 0 drops its translation, so the camera effectively sits at
// the origin and world-space f32 error never grows with distance.

#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;

struct ChunkDrawUniform {
    // Chunk minimum corner relative to the camera position, meters.
    offset: vec4<f32>,
}
@group(1) @binding(0) var<uniform> chunk: ChunkDrawUniform;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) cam_rel: vec3<f32>,
}

@vertex
fn vertex(in: VsIn) -> VsOut {
    let cam_rel = in.pos + chunk.offset.xyz;
    let view_space = (view.view_from_world * vec4<f32>(cam_rel, 0.0)).xyz;
    var out: VsOut;
    out.clip = view.clip_from_view * vec4<f32>(view_space, 1.0);
    out.normal = in.normal;
    out.cam_rel = cam_rel;
    return out;
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let light = normalize(vec3<f32>(0.4, 0.85, 0.3));
    let nd = max(dot(n, light), 0.0);

    // Slope-based grass/rock split until real materials land (M7).
    let grass = vec3<f32>(0.30, 0.45, 0.22);
    let rock = vec3<f32>(0.42, 0.40, 0.38);
    let base = mix(rock, grass, smoothstep(0.55, 0.8, n.y));

    // Cheap distance haze to make depth readable.
    let dist = length(in.cam_rel);
    let haze = 1.0 - exp(-dist * 0.0006);
    let sky = vec3<f32>(0.65, 0.75, 0.9);

    let lit = base * (0.25 + 0.75 * nd);
    return vec4<f32>(mix(lit, sky, haze), 1.0);
}
