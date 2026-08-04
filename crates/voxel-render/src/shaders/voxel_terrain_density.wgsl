// Terrain density pass: fills one arena slot with an FBM heightfield SDF for
// the chunk described by the dynamic-offset params. Runs once per chunk in
// the generation batch.

const SAMPLES: u32 = 36u;
const SLOT_STRIDE: u32 = 46656u; // 36^3

struct ChunkParams {
    // Chunk minimum corner in world meters (w unused).
    origin: vec4<f32>,
    slot: u32,
    base_vertex: u32,
    first_index: u32,
    counts_slot: u32,
}

@group(0) @binding(0) var<storage, read_write> density: array<u32>;
@group(0) @binding(1) var<uniform> params: ChunkParams;

// --- deterministic value noise -----------------------------------------------

fn hash2(p: vec2<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    // Quintic smoothstep for C2-continuous gradients.
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = hash2(i);
    let b = hash2(i + vec2<i32>(1, 0));
    let c = hash2(i + vec2<i32>(0, 1));
    let d = hash2(i + vec2<i32>(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 5; i++) {
        sum += amp * value_noise(p * freq);
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum; // ~[0, 1]
}

fn terrain_height(xz: vec2<f32>) -> f32 {
    let rolling = (fbm(xz * 0.01) - 0.5) * 36.0;
    let detail = (fbm(xz * 0.06 + vec2<f32>(37.0, 91.0)) - 0.5) * 5.0;
    return rolling + detail;
}

@compute @workgroup_size(6, 6, 6)
fn density_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(SAMPLES))) {
        return;
    }
    // Sample i holds cell corner i - 1; voxel size is 1 m at LOD 0.
    let p = params.origin.xyz + vec3<f32>(vec3<i32>(id) - vec3<i32>(1));
    let sdf = clamp(p.y - terrain_height(p.xz), -4.0, 4.0);
    let material = select(0u, 1u, sdf < 0.0);
    let packed = (pack2x16float(vec2<f32>(sdf, 0.0)) & 0xFFFFu) | (material << 16u);
    let base = params.slot * SLOT_STRIDE;
    density[base + id.x + SAMPLES * (id.y + SAMPLES * id.z)] = packed;
}
