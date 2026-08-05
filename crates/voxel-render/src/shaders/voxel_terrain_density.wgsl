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
    csg_offset: u32,
    csg_count: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> density: array<u32>;
@group(0) @binding(1) var<uniform> params: ChunkParams;

// Planning-layer CSG ops (48 B, layout mirrors voxel-core CsgOp).
struct CsgOp {
    center: vec3<f32>,
    kind: u32, // 0 box add, 1 box cut, 2 cylinder add, 3 cylinder cut
    half: vec3<f32>,
    material: u32,
    yaw: f32,
    blend: f32,
    _pad: vec2<u32>,
}
@group(0) @binding(2) var<storage, read_write> csg_ops: array<CsgOp>;

struct WorldTuning {
    t0: vec4<f32>, // continents scale/amp, mountains scale/amp
    t1: vec4<f32>, // rolling scale/amp, detail scale/amp
    t2: vec4<f32>, // height offset, floor/pillar/wall spacing
    t3: vec4<f32>, // shaft spacing, wall chance, opening chance, unused
}
@group(0) @binding(3) var<uniform> tuning: WorldTuning;

fn op_sdf(op: CsgOp, p: vec3<f32>) -> f32 {
    var q = p - op.center;
    let c = cos(-op.yaw);
    let s = sin(-op.yaw);
    q = vec3<f32>(q.x * c - q.z * s, q.y, q.x * s + q.z * c);
    if (op.kind < 2u) {
        let a = abs(q) - op.half;
        return length(max(a, vec3<f32>(0.0))) + min(max(a.x, max(a.y, a.z)), 0.0);
    }
    let dr = length(q.xz) - op.half.x;
    let dy = abs(q.y) - op.half.y;
    return length(vec2<f32>(max(dr, 0.0), max(dy, 0.0))) + min(max(dr, dy), 0.0);
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

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
    // Band scales/amplitudes come from the level's world tuning; finer
    // bands fade in as LOD refines (band-limiting).
    let continents = fbm(xz, tuning.t0.x, 3, voxel_size) * tuning.t0.y;
    let mountains = fbm(xz + vec2<f32>(510.0, -770.0), tuning.t0.z, 5, voxel_size) * tuning.t0.w;
    let rolling = fbm(xz + vec2<f32>(1337.0, 55.0), tuning.t1.x, 5, voxel_size) * tuning.t1.y;
    let detail = fbm(xz + vec2<f32>(37.0, 91.0), tuning.t1.z, 4, voxel_size) * tuning.t1.w;
    return continents + mountains + rolling + detail + tuning.t2.x;
}

@compute @workgroup_size(6, 6, 6)
fn density_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(SAMPLES))) {
        return;
    }
    let vs = params.origin.w;
    // Sample i holds cell corner i - 2 (apron covers coarse-parity cells).
    let p = params.origin.xyz + vec3<f32>(vec3<i32>(id) - vec3<i32>(2)) * vs;

    var d_m = p.y - terrain_height(p.xz, vs); // meters
    var mat = 1u;

    // Planning-layer CSG: additions merge (optionally smoothly) into the
    // terrain, cuts carve it. Ops are meter-scale, so they only apply at
    // fine LODs (they are also only provided there).
    if (params.csg_count > 0u && vs < 4.0) {
        for (var i = 0u; i < params.csg_count; i++) {
            let op = csg_ops[params.csg_offset + i];
            let od = op_sdf(op, p);
            if ((op.kind & 1u) == 0u) {
                if (op.blend > 0.0) {
                    d_m = smin(d_m, od, op.blend);
                } else {
                    d_m = min(d_m, od);
                }
                if (od < 0.3) {
                    mat = op.material;
                }
            } else {
                d_m = max(d_m, -od);
            }
        }
    }

    // SDF stored in voxel-size units, narrow band ±4.
    let sdf = clamp(d_m / vs, -4.0, 4.0);
    let material = select(0u, mat, sdf < 0.0);
    let packed = (pack2x16float(vec2<f32>(sdf, 0.0)) & 0xFFFFu) | (material << 16u);
    let base = params.slot * SLOT_STRIDE;
    density[base + id.x + SAMPLES * (id.y + SAMPLES * id.z)] = packed;
}
