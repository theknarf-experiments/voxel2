// Chunk draw: camera-relative vertex transform over the slab buffers, with
// fully procedural, fully data-driven shading (no texture assets, no
// world-specific branches): the per-vertex material id indexes the level's
// material table, and lighting/atmosphere come from the level's
// environment uniform.
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

// Material recipes (128 B each, layout mirrors voxel-render WorldMaterial).
// head.x = kind: 0 = surface (base/grain/bands/grime/streaks/moss/emissive),
// 1 = zoned altitude terrain.
struct WorldMaterial {
    head: vec4<u32>,
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    c3: vec4<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
}
struct MaterialTable {
    materials: array<WorldMaterial, 8>,
}
@group(1) @binding(1) var<uniform> mats: MaterialTable;

// Level lighting + atmosphere.
struct EnvParams {
    haze: vec4<f32>,      // rgb | density
    haze_tint: vec4<f32>, // rgb | tint power (0 = untinted)
    sun: vec4<f32>,       // rgb | strength
    sky: vec4<f32>,       // ambient sky rgb | ambient strength
    ground: vec4<f32>,    // ambient ground rgb | up exponent
    sun_dir: vec4<f32>,   // toward the sun | unused
}
@group(1) @binding(2) var<uniform> env: EnvParams;

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


// Uniform-base surface: grain, pour/mortar bands, grime, drip streaks,
// moss in upward crevices. Returns albedo; emissive is added after
// lighting.
fn surface_albedo(m: WorldMaterial, world: vec3<f32>, n: vec3<f32>, dist: f32) -> vec3<f32> {
    let detail_fade = exp(-dist * m.p2.z);
    var grain = 0.5;
    if (detail_fade > 0.02) {
        grain = mix(0.5, fbm3(world * 0.9), detail_fade);
    }
    let stains = fbm3(world * 0.035);
    var base = m.c0.rgb;
    if (m.p0.y > 0.0) {
        let band = fract(world.y * m.p0.x + stains * m.p1.x);
        base *= (1.0 - m.p0.y) + m.p0.y * smoothstep(m.p0.z, m.p0.w, band);
    }
    base *= 0.75 + m.c0.w * grain;
    // Grime patches darken toward the tint.
    base = mix(base, base * m.c1.rgb, smoothstep(0.55, 0.85, stains) * m.c1.w);
    // Vertical drip streaks on walls.
    if (m.p1.y > 0.0) {
        let wallness = 1.0 - abs(n.y);
        let streak = fbm3(vec3<f32>(world.x * 0.6, world.y * 0.03, world.z * 0.6));
        base *= 1.0 - wallness * smoothstep(0.6, 0.9, streak) * m.p1.y;
    }
    // Moss in upward crevices.
    if (m.c2.w > 0.0) {
        let macro_var = fbm3(world * 0.012);
        let moss = smoothstep(0.6, 0.9, macro_var) * smoothstep(0.5, 0.9, n.y) * m.c2.w;
        base = mix(base, m.c2.rgb, moss);
    }
    return base;
}

// Altitude-zoned natural terrain: low/mid/high/peak colors with noisy
// borders, slope override to the high (rock) color.
fn zoned_albedo(m: WorldMaterial, world: vec3<f32>, n: vec3<f32>, dist: f32) -> vec3<f32> {
    let detail_fade = exp(-dist * m.p2.w);
    var detail = 0.5;
    if (detail_fade > 0.02) {
        detail = mix(0.5, fbm3(world * 0.35), detail_fade);
    }
    let macro_var = fbm3(world * 0.012);

    var low = m.c0.rgb * (0.85 + 0.3 * detail);
    var mid = mix(m.c1.rgb, m.p0.rgb, macro_var);
    mid *= 0.8 + 0.45 * detail;
    let band = fract(world.y * 0.06 + macro_var * 2.0);
    var high = mix(m.c2.rgb, m.p1.rgb, smoothstep(0.2, 0.8, band));
    high *= 0.8 + 0.4 * detail;
    var peak = m.c3.rgb * (0.9 + 0.2 * detail);

    let border = (macro_var - 0.5) * m.c3.w;
    var base = mix(low, mid, smoothstep(m.c0.w + border * 0.05, m.c0.w + m.p0.w + border * 0.05, world.y));
    base = mix(base, high, smoothstep(m.c1.w + border, m.c1.w + m.p1.w + border, world.y));
    base = mix(base, peak, smoothstep(m.c2.w + border, m.c2.w + m.p2.x + border, world.y));
    let steep = smoothstep(m.p2.y, m.p2.z, n.y); // 1 on cliffs
    return mix(base, high, steep);
}

// Central-difference gradient of fbm3 (for bump-style normal perturbation).
fn fbm3_grad(p: vec3<f32>, eps: f32) -> vec3<f32> {
    return vec3<f32>(
        fbm3(p + vec3<f32>(eps, 0.0, 0.0)) - fbm3(p - vec3<f32>(eps, 0.0, 0.0)),
        fbm3(p + vec3<f32>(0.0, eps, 0.0)) - fbm3(p - vec3<f32>(0.0, eps, 0.0)),
        fbm3(p + vec3<f32>(0.0, 0.0, eps)) - fbm3(p - vec3<f32>(0.0, 0.0, eps)),
    ) / (2.0 * eps);
}

struct MatSample {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    ao: f32,
}

// Forested zoned terrain (after iq's Rainforest): the canopy is crown
// noise — two-tone green mixed by crown height, normals perturbed by the
// crown gradient so the sun lights individual crowns, AO from crown depth.
// Rock gets steepness-proportional anisotropic bumps (y squashed 5x →
// horizontal strata), moss on flat tops, and an implicit snowcap above
// the rock zone.
// Layout: c0 canopy_dark | canopy start; c1 canopy_lit | rock start;
// c2 rock | rock width; c3 patch (dry/brown) | border; p0 low | canopy
// width; p1 (crown scale, crown relief, strata scale, strata relief);
// p2 (steep hi, steep lo, detail fade, patch amount).
fn canopy_material(m: WorldMaterial, world: vec3<f32>, n: vec3<f32>, dist: f32) -> MatSample {
    let fade = exp(-dist * m.p2.z);
    let macro_var = fbm3(world * 0.012);
    let border = (macro_var - 0.5) * m.c3.w;

    let veg_edge = smoothstep(m.c0.w + border * 0.05, m.c0.w + m.p0.w + border * 0.05, world.y);
    let rockness_alt = smoothstep(m.c1.w + border, m.c1.w + m.c2.w + border, world.y);
    let steep = smoothstep(m.p2.x, m.p2.y, n.y);
    let rockness = max(rockness_alt, steep);
    let veg = veg_edge * (1.0 - rockness);

    // --- canopy: crowns as noise ------------------------------------------
    let crown_p = world * m.p1.x;
    let crown = fbm3(crown_p);
    var nn = n;
    let crelief = m.p1.y * veg * mix(0.4, 1.0, fade);
    if (crelief > 0.01) {
        nn = normalize(n + crelief * fbm3_grad(crown_p, 0.3));
    }
    var ccol = mix(m.c0.rgb, m.c1.rgb, smoothstep(0.3, 0.8, crown));
    // Dry/brown patches on gentle ground, iq-style. (`patch` is a
    // reserved WGSL word, like `meta`.)
    let dry = smoothstep(0.55, 0.8, fbm3(world * 0.015)) * m.p2.w * smoothstep(0.5, 0.85, n.y);
    ccol = mix(ccol, m.c3.rgb, dry);
    // Crown-depth occlusion: canopy hollows swallow light.
    let cao = 0.4 + 0.6 * smoothstep(0.15, 0.8, crown);

    // --- rock: anisotropic strata bumps -----------------------------------
    let strata_p = world * vec3<f32>(m.p1.z, m.p1.z * 0.2, m.p1.z);
    var rn = n;
    let srelief = m.p1.w * rockness * (1.0 - abs(n.y) * 0.6) * mix(0.5, 1.0, fade);
    if (srelief > 0.01) {
        rn = normalize(n + srelief * fbm3_grad(strata_p, 0.3));
    }
    var rcol = m.c2.rgb * (0.75 + 0.5 * mix(0.5, fbm3(world * 0.9), fade));
    // Moss creeps onto flat rock shelves.
    rcol = mix(rcol, m.c0.rgb, 0.45 * smoothstep(0.7, 0.92, rn.y) * (1.0 - rockness_alt));
    // Implicit snowcap well above the rock line.
    let snow = smoothstep(m.c1.w + 2.5 * m.c2.w + border, m.c1.w + 3.5 * m.c2.w + border, world.y)
        * smoothstep(0.35, 0.65, rn.y);
    rcol = mix(rcol, vec3<f32>(0.85, 0.88, 0.95), snow);

    var out: MatSample;
    out.albedo = mix(mix(m.p0.rgb, ccol, veg_edge), rcol, rockness);
    out.normal = normalize(mix(mix(n, nn, veg), rn, rockness));
    out.ao = mix(1.0, cao, veg);
    return out;
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    // Coverage-eval mode: monotone geometry against a magenta background,
    // tinted by LOD level so cracks identify the leaking seam pair.
    if (env.sun_dir.w > 0.5) {
        if (in.material == 255u) {
            // Failed parity snap (thin feature) — debug-highlighted.
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
        let g = 1.0 - clamp(log2(chunk.offset.w), 0.0, 8.0) * 0.1;
        return vec4<f32>(g, g, g, 1.0);
    }
    let n = normalize(in.normal);
    let world = vec3<f32>(
        view.world_position.x + in.cam_rel.x,
        view.world_position.y + in.cam_rel.y,
        view.world_position.z + in.cam_rel.z,
    );
    let dist = length(in.cam_rel);
    let m = mats.materials[min(in.material, 7u)];

    var base: vec3<f32>;
    var nl = n;
    var ao = 1.0;
    if (m.head.x == 2u) {
        let ms = canopy_material(m, world, n, dist);
        base = ms.albedo;
        nl = ms.normal;
        ao = ms.ao;
    } else if (m.head.x == 1u) {
        base = zoned_albedo(m, world, n, dist);
    } else {
        base = surface_albedo(m, world, n, dist);
    }

    // --- lighting: sun (baked-shadowed) + hemispheric ambient ---------------
    let sun_dir = normalize(env.sun_dir.xyz);
    let nd = max(dot(nl, sun_dir), 0.0);
    let up = nl.y * 0.5 + 0.5;
    let ambient = mix(env.ground.rgb, env.sky.rgb, pow(up, env.ground.w)) * env.sky.w;
    var lit = base * (env.sun.rgb * nd * env.sun.w * in.shadow + ambient) * ao;

    // --- emissive ceiling light strips (surface materials) -------------------
    if (m.head.x == 0u && m.c3.w > 0.0) {
        let lf = floor(world.y / m.p1.w);
        let ceilingness = smoothstep(-0.55, -0.85, n.y);
        let line = 1.0 - smoothstep(0.25, 0.75, abs(fract(world.z / m.p1.z) - 0.5) * m.p1.z);
        let works = step(1.0 - m.p2.x, hash3(vec3<i32>(i32(floor(world.z / m.p1.z)), i32(lf), 7)));
        lit += m.c3.rgb * m.c3.w * ceilingness * line * works;
        // Faint up-glow from the strips onto nearby floors.
        let floorness = smoothstep(0.55, 0.85, n.y);
        lit += m.c3.rgb * m.p2.y * floorness * line * works;
    }

    // --- haze, optionally tinted toward the sun ------------------------------
    let haze_amount = 1.0 - exp(-dist * env.haze.w);
    var haze_color = env.haze.rgb;
    if (env.haze_tint.w > 0.0) {
        let view_dir = normalize(in.cam_rel);
        let sun_amount = pow(max(dot(view_dir, sun_dir), 0.0), env.haze_tint.w);
        haze_color = mix(haze_color, env.haze_tint.rgb, sun_amount);
    }
    lit = mix(lit, haze_color, haze_amount);
    return vec4<f32>(lit, 1.0);
}
