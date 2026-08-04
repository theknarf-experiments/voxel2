// Chunk draw: camera-relative vertex transform over the slab buffers, with
// fully procedural terrain shading (no texture assets): noise-based albedo
// per material zone, hemispheric ambient, sun light, planet-scale haze.
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

// --- procedural detail noise -------------------------------------------------

fn hash3(p: vec3<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u + u32(p.z) * 2246822519u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(
            mix(hash3(i), hash3(i + vec3(1, 0, 0)), u.x),
            mix(hash3(i + vec3(0, 1, 0)), hash3(i + vec3(1, 1, 0)), u.x),
            u.y,
        ),
        mix(
            mix(hash3(i + vec3(0, 0, 1)), hash3(i + vec3(1, 0, 1)), u.x),
            mix(hash3(i + vec3(0, 1, 1)), hash3(i + vec3(1, 1, 1)), u.x),
            u.y,
        ),
        u.z,
    );
}

fn fbm3(p: vec3<f32>) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < 3; i++) {
        sum += amp * noise3(q);
        amp *= 0.5;
        q *= 2.17;
    }
    return sum; // ~[0, 1]
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let world = vec3<f32>(
        view.world_position.x + in.cam_rel.x,
        view.world_position.y + in.cam_rel.y,
        view.world_position.z + in.cam_rel.z,
    );
    let dist = length(in.cam_rel);

    // Fade detail octaves out with distance so far terrain doesn't shimmer.
    let detail_fade = exp(-dist * 0.002);
    var detail = 0.5;
    if (detail_fade > 0.02) {
        detail = mix(0.5, fbm3(world * 0.35), detail_fade);
    }
    let macro_var = fbm3(world * 0.012); // large patchiness

    // --- material zones ------------------------------------------------------
    // Grass: hue patchiness + fine detail.
    let grass_a = vec3<f32>(0.21, 0.35, 0.12);
    let grass_b = vec3<f32>(0.34, 0.42, 0.16);
    var grass = mix(grass_a, grass_b, macro_var);
    grass *= 0.8 + 0.45 * detail;

    // Rock: banded by altitude + detail grain.
    let rock_a = vec3<f32>(0.33, 0.30, 0.27);
    let rock_b = vec3<f32>(0.46, 0.42, 0.38);
    let band = fract(world.y * 0.06 + macro_var * 2.0);
    var rock = mix(rock_a, rock_b, smoothstep(0.2, 0.8, band));
    rock *= 0.8 + 0.4 * detail;

    // Sand and snow.
    var sand = vec3<f32>(0.62, 0.56, 0.42) * (0.85 + 0.3 * detail);
    var snow = vec3<f32>(0.82, 0.85, 0.92) * (0.9 + 0.2 * detail);

    // Zone blending by altitude with noisy borders, then slope override.
    let border = (macro_var - 0.5) * 60.0;
    var base = mix(sand, grass, smoothstep(1.5 + border * 0.05, 5.0 + border * 0.05, world.y));
    base = mix(base, rock, smoothstep(340.0 + border, 620.0 + border, world.y));
    base = mix(base, snow, smoothstep(820.0 + border, 1050.0 + border, world.y));
    let steep = smoothstep(0.72, 0.45, n.y); // 1 on cliffs
    base = mix(base, rock, steep);

    // --- lighting ------------------------------------------------------------
    let sun_dir = normalize(vec3<f32>(0.55, 0.5, 0.32));
    let sun_color = vec3<f32>(1.0, 0.96, 0.88);
    let sky_color = vec3<f32>(0.55, 0.70, 0.95);
    let ground_bounce = vec3<f32>(0.25, 0.24, 0.20);

    let nd = max(dot(n, sun_dir), 0.0);
    let hemi = mix(ground_bounce, sky_color, n.y * 0.5 + 0.5);
    var lit = base * (sun_color * nd * 0.85 + hemi * 0.3);

    // --- aerial haze ---------------------------------------------------------
    let haze_amount = 1.0 - exp(-dist * 0.00006);
    // Haze tinted warmer toward the sun direction.
    let view_dir = normalize(in.cam_rel);
    let sun_amount = pow(max(dot(view_dir, sun_dir), 0.0), 4.0);
    let haze_color = mix(vec3<f32>(0.62, 0.72, 0.88), vec3<f32>(0.92, 0.85, 0.72), sun_amount);
    lit = mix(lit, haze_color, haze_amount);

    return vec4<f32>(lit, 1.0);
}
