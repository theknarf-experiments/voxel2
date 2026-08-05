// Procedural ocean: a camera-following grid whose cell size grows with
// distance (power-warped UVs), displaced by a small sum of directional
// waves with analytic normals. Shorelines come from evaluating the coarse
// terrain heightfield at the fragment — shallow tint and a foam band where
// the seabed grazes sea level. Opaque; matches the terrain's sun and haze.

#import bevy_render::view::View
#import bevy_render::globals::Globals

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;

struct WaterParams {
    // xz = grid origin (camera-snapped, world meters), y = sea level, w unused.
    origin: vec4<f32>,
}
@group(0) @binding(2) var<uniform> params: WaterParams;

struct WorldTuning {
    t0: vec4<f32>, // continents scale/amp, mountains scale/amp
    t1: vec4<f32>, // rolling scale/amp, detail scale/amp
    t2: vec4<f32>, // height offset, floor/pillar/wall spacing
    t3: vec4<f32>, // shaft spacing, wall chance, opening chance, unused
}
@group(0) @binding(3) var<uniform> tuning: WorldTuning;

const GRID_N: u32 = 192u;       // vertices per side
const RANGE_M: f32 = 30000.0;   // farthest grid reach from center
const SEA_SNAP: f32 = 64.0;     // origin snap so the grid never swims

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_xz: vec2<f32>,
    @location(1) cam_rel: vec3<f32>,
    @location(2) normal: vec3<f32>,
}

// Three directional waves; returns (height, dh/dx, dh/dz).
fn waves(p: vec2<f32>, t: f32) -> vec3<f32> {
    var h = 0.0;
    var dx = 0.0;
    var dz = 0.0;
    let dirs = array<vec2<f32>, 3>(
        normalize(vec2<f32>(1.0, 0.35)),
        normalize(vec2<f32>(-0.55, 1.0)),
        normalize(vec2<f32>(0.2, -1.0)),
    );
    let amps = array<f32, 3>(0.16, 0.11, 0.06);
    let lens = array<f32, 3>(21.0, 9.0, 4.2);
    let spds = array<f32, 3>(3.1, 2.2, 1.4);
    for (var i = 0; i < 3; i++) {
        let k = 6.2831853 / lens[i];
        let phase = dot(dirs[i], p) * k + t * spds[i] * k;
        let s = sin(phase);
        let c = cos(phase);
        h += amps[i] * s;
        dx += amps[i] * k * dirs[i].x * c;
        dz += amps[i] * k * dirs[i].y * c;
    }
    return vec3<f32>(h, dx, dz);
}

@vertex
fn vertex(@builtin(vertex_index) vid: u32) -> VsOut {
    // Two triangles per cell, generated procedurally from the vertex index.
    let cells = GRID_N - 1u;
    let quad = vid / 6u;
    let corner = vid % 6u;
    var cx = quad % cells;
    var cz = quad / cells;
    // Two triangles: (0,0)(1,0)(1,1) and (0,0)(1,1)(0,1).
    if (corner == 1u || corner == 2u || corner == 4u) { cx += 1u; }
    if (corner == 2u || corner == 4u || corner == 5u) { cz += 1u; }

    // Power-warped offset: dense cells near the center, huge far out.
    let u = (f32(cx) / f32(cells)) * 2.0 - 1.0;
    let v = (f32(cz) / f32(cells)) * 2.0 - 1.0;
    let wx = sign(u) * pow(abs(u), 2.6) * RANGE_M;
    let wz = sign(v) * pow(abs(v), 2.6) * RANGE_M;
    let world_xz = params.origin.xz + vec2<f32>(wx, wz);

    let w = waves(world_xz, globals.time);
    // Fade displacement out where cells are huge (avoids crawling).
    let center_d = length(vec2<f32>(wx, wz));
    let disp_fade = exp(-center_d * 0.002);
    let y = params.origin.y + w.x * disp_fade;

    let world = vec3<f32>(world_xz.x, y, world_xz.y);
    let cam_rel = world - view.world_position;
    let view_space = (view.view_from_world * vec4<f32>(cam_rel, 0.0)).xyz;

    var out: VsOut;
    out.clip = view.clip_from_view * vec4<f32>(view_space, 1.0);
    out.world_xz = world_xz;
    out.cam_rel = cam_rel;
    out.normal = normalize(vec3<f32>(-w.y * disp_fade, 1.0, -w.z * disp_fade));
    return out;
}

// --- coarse terrain height (shoreline) ---------------------------------------

fn hash2(p: vec2<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

fn value_noise2(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = hash2(i);
    let b = hash2(i + vec2<i32>(1, 0));
    let c = hash2(i + vec2<i32>(0, 1));
    let d = hash2(i + vec2<i32>(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2(p: vec2<f32>, base_scale: f32, octaves: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = base_scale;
    for (var i = 0; i < octaves; i++) {
        sum += amp * (value_noise2(p * freq) - 0.5);
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum;
}

fn seabed_height(xz: vec2<f32>) -> f32 {
    let continents = fbm2(xz, tuning.t0.x, 3) * tuning.t0.y;
    let mountains = fbm2(xz + vec2<f32>(510.0, -770.0), tuning.t0.z, 4) * tuning.t0.w;
    let rolling = fbm2(xz + vec2<f32>(1337.0, 55.0), tuning.t1.x, 3) * tuning.t1.y;
    return continents + mountains + rolling + tuning.t2.x;
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let dist = length(in.cam_rel);
    let view_dir = normalize(-in.cam_rel);
    let sun_dir = normalize(vec3<f32>(0.55, 0.5, 0.32));

    // Depth-based color from the seabed heightfield.
    let bed = seabed_height(in.world_xz);
    let depth = max(-bed, 0.0);
    let deep = vec3<f32>(0.04, 0.13, 0.22);
    let shallow = vec3<f32>(0.10, 0.34, 0.36);
    var water = mix(shallow, deep, smoothstep(1.5, 26.0, depth));

    // Foam where the seabed grazes sea level, animated by wave phase.
    let foam_noise = value_noise2(in.world_xz * 0.7 + globals.time * 0.35);
    let foam_band = smoothstep(2.2, 0.3, depth) * (0.55 + 0.45 * foam_noise);
    water = mix(water, vec3<f32>(0.85, 0.9, 0.9), clamp(foam_band, 0.0, 1.0) * 0.8);

    // Fresnel toward the sky, sun glint.
    let fresnel = pow(1.0 - max(dot(n, view_dir), 0.0), 4.0);
    let sky = vec3<f32>(0.55, 0.70, 0.95);
    var col = mix(water, sky, fresnel * 0.65);
    let half_dir = normalize(sun_dir + view_dir);
    let spec = pow(max(dot(n, half_dir), 0.0), 240.0) * 1.4;
    col += vec3<f32>(1.0, 0.95, 0.85) * spec;

    // Sun light + haze, matching the terrain shader.
    let nd = max(dot(n, sun_dir), 0.0);
    col *= 0.35 + 0.75 * nd;
    let haze_amount = 1.0 - exp(-dist * 0.00006);
    let sun_amount = pow(max(dot(-view_dir, sun_dir), 0.0), 4.0);
    let haze_color = mix(vec3<f32>(0.62, 0.72, 0.88), vec3<f32>(0.92, 0.85, 0.72), sun_amount);
    col = mix(col, haze_color, haze_amount);
    return vec4<f32>(col, 1.0);
}
