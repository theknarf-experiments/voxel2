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
    // xyz = chunk minimum corner relative to the camera (m), w = voxel size.
    offset: vec4<f32>,
}
@group(1) @binding(0) var<uniform> chunk: ChunkDrawUniform;

struct VsIn {
    // Quantized position: unorm16 x4 mapping [-8, 40] voxels (w unused).
    @location(0) pos: vec4<f32>,
    // Octahedral-encoded normal, snorm16 x2.
    @location(1) oct: vec2<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) cam_rel: vec3<f32>,
    @location(2) shadow: f32,
    // Flat-interpolated material id from the most-solid corner.
    @location(3) @interpolate(flat) material: u32,
}

const POS_BIAS: f32 = 8.0;
const POS_RANGE: f32 = 48.0;

fn oct_decode(o: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(o, 1.0 - abs(o.x) - abs(o.y));
    if (n.z < 0.0) {
        let sx = select(-1.0, 1.0, n.x >= 0.0);
        let sy = select(-1.0, 1.0, n.y >= 0.0);
        n = vec3<f32>((1.0 - abs(n.y)) * sx, (1.0 - abs(n.x)) * sy, n.z);
    }
    return normalize(n);
}

@vertex
fn vertex(in: VsIn) -> VsOut {
    let pos_local = (in.pos.xyz * POS_RANGE - POS_BIAS) * chunk.offset.w;
    let cam_rel = pos_local + chunk.offset.xyz;
    let view_space = (view.view_from_world * vec4<f32>(cam_rel, 0.0)).xyz;
    var out: VsOut;
    out.clip = view.clip_from_view * vec4<f32>(view_space, 1.0);
    out.normal = oct_decode(in.oct);
    out.cam_rel = cam_rel;
    // Spare u16: material in the high byte, baked shadow in the low byte.
    let extra = u32(round(in.pos.w * 65535.0));
    out.shadow = f32(extra & 0xFFu) / 255.0;
    out.material = (extra >> 8u) & 0xFFu;
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


#ifdef MEGASTRUCTURE
@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let world = vec3<f32>(
        view.world_position.x + in.cam_rel.x,
        view.world_position.y + in.cam_rel.y,
        view.world_position.z + in.cam_rel.z,
    );
    let dist = length(in.cam_rel);

    // Concrete: pale gray with pour-band striations, grime, and streaks.
    let detail_fade = exp(-dist * 0.004);
    var grain = 0.5;
    if (detail_fade > 0.02) {
        grain = mix(0.5, fbm3(world * 0.9), detail_fade);
    }
    let stains = fbm3(world * 0.035);
    let band = fract(world.y * 0.22 + stains * 0.4);
    var base = vec3<f32>(0.42, 0.42, 0.43);
    base *= 0.82 + 0.18 * smoothstep(0.1, 0.9, band); // pour bands
    base *= 0.75 + 0.35 * grain;                       // fine grain
    base = mix(base, base * vec3<f32>(0.55, 0.58, 0.55), smoothstep(0.55, 0.85, stains));

    // Vertical drip streaks on walls.
    let wallness = 1.0 - abs(n.y);
    let streak = fbm3(vec3<f32>(world.x * 0.6, world.y * 0.03, world.z * 0.6));
    base *= 1.0 - wallness * smoothstep(0.6, 0.9, streak) * 0.35;

    // Dim top-light as if from distant shafts above; heavy darkness below.
    let up = n.y * 0.5 + 0.5;
    let key = vec3<f32>(0.75, 0.78, 0.82) * (0.12 + 0.55 * up * up);
    let rim = vec3<f32>(0.20, 0.24, 0.30) * (1.0 - up) * 0.25;
    var lit = base * (key + rim);

    // Emissive service-light strips on ceiling undersides: sparse lines
    // running along x, flickered out on some floors.
    let lf = floor(world.y / 22.0);
    let ry = world.y - lf * 22.0;
    let ceilingness = smoothstep(-0.55, -0.85, n.y);
    let line = 1.0 - smoothstep(0.25, 0.75, abs(fract(world.z / 13.0) - 0.5) * 13.0);
    let works = step(0.35, hash3(vec3<i32>(i32(floor(world.z / 13.0)), i32(lf), 7)));
    let strip = ceilingness * line * works;
    lit += vec3<f32>(1.3, 1.25, 1.05) * strip;

    // Faint up-glow from the strips onto nearby floors.
    let floorness = smoothstep(0.55, 0.85, n.y);
    lit += vec3<f32>(0.10, 0.10, 0.085) * floorness * line * works;

    // Thick interior gloom.
    let haze_amount = 1.0 - exp(-dist * 0.0035);
    let haze_color = vec3<f32>(0.035, 0.045, 0.06);
    lit = mix(lit, haze_color, haze_amount);
    return vec4<f32>(lit, 1.0);
}
#else
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

    // Material 3: worked stone (ruins, roads) — cut-block gray with grime,
    // faint moss only in upward crevices.
    if (in.material == 3u) {
        var stone = vec3<f32>(0.52, 0.50, 0.46);
        let block = fract(world.y * 0.55 + macro_var * 0.7);
        stone *= 0.82 + 0.18 * smoothstep(0.08, 0.25, block); // mortar lines
        stone *= 0.75 + 0.4 * detail;
        let moss = smoothstep(0.6, 0.9, macro_var) * smoothstep(0.5, 0.9, n.y) * 0.5;
        base = mix(stone, grass, moss);
    }

    // --- lighting ------------------------------------------------------------
    let sun_dir = normalize(vec3<f32>(0.55, 0.5, 0.32));
    let sun_color = vec3<f32>(1.0, 0.96, 0.88);
    let sky_color = vec3<f32>(0.55, 0.70, 0.95);
    let ground_bounce = vec3<f32>(0.25, 0.24, 0.20);

    let nd = max(dot(n, sun_dir), 0.0);
    let hemi = mix(ground_bounce, sky_color, n.y * 0.5 + 0.5);
    // Terrain sun shadow baked per vertex at mesh time (spare pos channel).
    let shadow = in.shadow;
    var lit = base * (sun_color * nd * 0.85 * shadow + hemi * 0.3);

    // --- aerial haze ---------------------------------------------------------
    let haze_amount = 1.0 - exp(-dist * 0.00006);
    // Haze tinted warmer toward the sun direction.
    let view_dir = normalize(in.cam_rel);
    let sun_amount = pow(max(dot(view_dir, sun_dir), 0.0), 4.0);
    let haze_color = mix(vec3<f32>(0.62, 0.72, 0.88), vec3<f32>(0.92, 0.85, 0.72), sun_amount);
    lit = mix(lit, haze_color, haze_amount);

    return vec4<f32>(lit, 1.0);
}
#endif
