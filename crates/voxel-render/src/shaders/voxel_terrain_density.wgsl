// Terrain density pass: fills one arena slot with an FBM heightfield SDF for
// the chunk described by the dynamic-offset params. Runs once per chunk in
// the generation batch.
//
// LOD-aware: params.origin.w carries the voxel size in meters. Noise octaves
// whose wavelength approaches the voxel size are faded out (band-limiting) so
// coarse LODs don't alias and LOD swaps don't pop.

const SAMPLES: u32 = 38u;
const SLOT_STRIDE: u32 = 54872u; // 38^3

struct ChunkParams {
    // xyz = chunk minimum corner in world meters, w = voxel size in meters.
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
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = hash2(i);
    let b = hash2(i + vec2<i32>(1, 0));
    let c = hash2(i + vec2<i32>(0, 1));
    let d = hash2(i + vec2<i32>(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Octave weight: full at wavelength >= 4 voxels, gone below 2 voxels.
fn band_fade(wavelength: f32, voxel_size: f32) -> f32 {
    return smoothstep(2.0 * voxel_size, 4.0 * voxel_size, wavelength);
}

// FBM with per-octave band-limiting. `base_scale` is cycles per meter of the
// first octave (wavelength = 1 / base_scale).
fn fbm(p: vec2<f32>, base_scale: f32, octaves: i32, voxel_size: f32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = base_scale;
    for (var i = 0; i < octaves; i++) {
        let fade = band_fade(1.0 / freq, voxel_size);
        sum += amp * fade * (value_noise(p * freq) - 0.5);
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum; // ~[-0.5, 0.5]
}

fn terrain_height(xz: vec2<f32>, voxel_size: f32) -> f32 {
    // Continent-scale relief (20 km wavelength) survives even 256 m voxels,
    // so orbit views show real terrain; finer bands fade in as LOD refines.
    let continents = fbm(xz, 0.00005, 3, voxel_size) * 800.0;
    let mountains = fbm(xz + vec2<f32>(510.0, -770.0), 0.0008, 5, voxel_size) * 420.0;
    let rolling = fbm(xz + vec2<f32>(1337.0, 55.0), 0.01, 5, voxel_size) * 36.0;
    let detail = fbm(xz + vec2<f32>(37.0, 91.0), 0.06, 4, voxel_size) * 5.0;
    return continents + mountains + rolling + detail - 8.0;
}

@compute @workgroup_size(6, 6, 6)
fn density_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(SAMPLES))) {
        return;
    }
    let vs = params.origin.w;
    // Sample i holds cell corner i - 2 (apron covers coarse-parity cells).
    let p = params.origin.xyz + vec3<f32>(vec3<i32>(id) - vec3<i32>(2)) * vs;
    // SDF stored in voxel-size units, narrow band ±4.
    let sdf = clamp((p.y - terrain_height(p.xz, vs)) / vs, -4.0, 4.0);
    let material = select(0u, 1u, sdf < 0.0);
    let packed = (pack2x16float(vec2<f32>(sdf, 0.0)) & 0xFFFFu) | (material << 16u);
    let base = params.slot * SLOT_STRIDE;
    density[base + id.x + SAMPLES * (id.y + SAMPLES * id.z)] = packed;
}
