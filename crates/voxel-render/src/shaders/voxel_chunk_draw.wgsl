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
    // Low sun for readable relief (placeholder until real materials, M7).
    let light = normalize(vec3<f32>(0.6, 0.45, 0.35));
    let nd = max(dot(n, light), 0.0);

    let world_y = view.world_position.y + in.cam_rel.y;

    // Height/slope zoned coloring: sand, grass, rock, snow.
    let sand = vec3<f32>(0.66, 0.60, 0.44);
    let grass = vec3<f32>(0.25, 0.40, 0.18);
    let rock = vec3<f32>(0.38, 0.35, 0.32);
    let snow = vec3<f32>(0.85, 0.87, 0.92);

    var base = mix(sand, grass, smoothstep(2.0, 12.0, world_y));
    base = mix(base, rock, smoothstep(280.0, 600.0, world_y));
    base = mix(base, snow, smoothstep(700.0, 1000.0, world_y));
    // Steep slopes are rock regardless of altitude.
    base = mix(rock, base, smoothstep(0.45, 0.7, n.y));

    // Cheap distance haze to make depth readable at planet scales.
    let dist = length(in.cam_rel);
    let haze = 1.0 - exp(-dist * 0.00004);
    let sky = vec3<f32>(0.65, 0.75, 0.9);

    let lit = base * (0.22 + 0.78 * nd);
    return vec4<f32>(mix(lit, sky, haze), 1.0);
}
