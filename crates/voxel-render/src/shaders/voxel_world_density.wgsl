// Generic world-generator density pass: fills one arena slot by
// interpreting the level's generator program — an ordered list of WorldOps
// (heightfield bands, lattice slabs/pillars/walls, shafts, beams) over a
// small register file. Worlds are data; there is exactly one density
// shader. MUST stay bit-compatible with the CPU twin in
// voxel-worldgen/src/program.rs.
//
// LOD-aware: params.origin.w carries the voxel size in meters. Height noise
// band-limits per octave; structural ops gate on FINE_ONLY/COARSE_ONLY
// (cutoff at 4 m voxels) so coarse LODs read as solid mass with voids.

const SAMPLES: u32 = 38u;
const SLOT_STRIDE: u32 = 54872u; // 38^3
const BIG: f32 = 1.0e6;
const SOLID: f32 = -1.0e5;
const COARSE_VOXEL_M: f32 = 4.0;

struct ChunkParams {
    // xyz = chunk minimum corner in world meters, w = voxel size in meters.
    origin: vec4<f32>,
    slot: u32,
    base_vertex: u32,
    first_index: u32,
    counts_slot: u32,
    csg_offset: u32,
    csg_count: u32,
    // x = seam mask: 2 bits per face (+x,-x,+y,-y,+z,-z); 1 = coarser.
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

// Generator program (64 B ops, layout mirrors voxel-core WorldOp).
// meta = (kind, flags, material, unused); count = (total, height ops, -, -).
struct WorldOp {
    head: vec4<u32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
}
struct WorldProgram {
    count: vec4<u32>,  // total ops, height ops, seed, unused
    sun: vec4<f32>,    // sun direction | unused
    anchor: vec4<f32>, // LOD field anchor | dist_scale
    field: vec4<f32>,  // max_vs | unused
    ops: array<WorldOp>,
}
@group(0) @binding(3) var<storage, read> prog: WorldProgram;

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

// --- deterministic hashes / noise --------------------------------------------

fn hash2(p: vec2<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u
        + prog.count.z * 2654435769u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

fn hash3(p: vec3<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u + u32(p.z) * 2246822519u
        + prog.count.z * 2654435769u;
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

// FBM with a per-octave shaping mode: 0 plain, 1 ridged (sharp crests),
// 2 billow (rounded mounds).
fn fbm(p: vec2<f32>, base_scale: f32, octaves: i32, voxel_size: f32, mode: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = base_scale;
    for (var i = 0; i < octaves; i++) {
        let fade = band_fade(1.0 / freq, voxel_size);
        let n = value_noise(p * freq);
        var v = n - 0.5;
        if (mode == 1u) {
            v = 0.5 - abs(2.0 * n - 1.0);
        } else if (mode == 2u) {
            v = abs(2.0 * n - 1.0) - 0.5;
        }
        sum += amp * fade * v;
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum; // ~[-0.5, 0.5]
}

fn value_noise3(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let x00 = mix(hash3(i), hash3(i + vec3<i32>(1, 0, 0)), u.x);
    let x10 = mix(hash3(i + vec3<i32>(0, 1, 0)), hash3(i + vec3<i32>(1, 1, 0)), u.x);
    let x01 = mix(hash3(i + vec3<i32>(0, 0, 1)), hash3(i + vec3<i32>(1, 0, 1)), u.x);
    let x11 = mix(hash3(i + vec3<i32>(0, 1, 1)), hash3(i + vec3<i32>(1, 1, 1)), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// Anisotropic band-limited 3D FBM (~[-0.5, 0.5]); band fade keys on the
// horizontal frequency.
fn fbm3(p: vec3<f32>, freq_xz: f32, freq_y: f32, octaves: i32, voxel_size: f32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var mul = 1.0;
    for (var i = 0; i < octaves; i++) {
        let fade = band_fade(1.0 / (freq_xz * mul), voxel_size);
        sum += amp * fade
            * (value_noise3(vec3<f32>(p.x * freq_xz * mul, p.y * freq_y * mul, p.z * freq_xz * mul))
                - 0.5);
        amp *= 0.5;
        mul *= 2.0;
    }
    return sum;
}

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// --- the interpreter ---------------------------------------------------------

struct WorldSample {
    d: f32,
    mat: u32,
}

fn eval_program(p: vec3<f32>, vs: f32) -> WorldSample {
    let coarse = vs >= COARSE_VOXEL_M;
    var h = 0.0;
    var d = BIG;
    var mat = 1u;
    var level = 0.0;
    var fy = p.y;
    var sxz = vec2<f32>(0.0);
    var sr = 0.0;
    var shaft = BIG;
    var warp = vec2<f32>(0.0);
    let pxz = p.xz;

    for (var i = 0u; i < prog.count.x; i++) {
        let op = prog.ops[i];
        if (coarse && (op.head.y & 1u) != 0u) { continue; }
        if (!coarse && (op.head.y & 2u) != 0u) { continue; }
        switch op.head.x {
            case 0u: { // height fbm band
                h += fbm(pxz + warp + op.p0.xy, op.p0.z, i32(op.p1.x), vs, u32(op.p1.y)) * op.p0.w;
            }
            case 1u: { // height offset
                h += op.p0.x;
            }
            case 14u: { // domain warp for later height ops
                let q = pxz + op.p0.zw;
                let oct = i32(op.p1.x);
                warp.x += fbm(q, op.p0.x, oct, vs, 0u) * op.p0.y;
                warp.y += fbm(q + vec2<f32>(713.0, -337.0), op.p0.x, oct, vs, 0u) * op.p0.y;
            }
            case 15u: { // 3D fbm solid: union or carve
                let q = p + op.p1.xyz;
                let n = fbm3(q, op.p0.x, op.p0.y, i32(op.p2.x), vs);
                let sd = (op.p0.z - n) * op.p0.w;
                if (op.p1.w < 0.5) {
                    if (sd < d) { d = sd; mat = op.head.z; }
                } else {
                    d = max(d, -sd);
                }
            }
            case 2u: { // height surface
                let nd = p.y - h;
                if (nd < d) { d = nd; mat = op.head.z; }
            }
            case 3u: { // coarse solid mass
                if (SOLID < d) { d = SOLID; mat = op.head.z; }
            }
            case 4u: { // y lattice registers
                level = round(p.y / op.p0.x);
                fy = p.y - level * op.p0.x;
            }
            case 5u: { // slabs on the lattice
                let nd = abs(fy) - op.p0.x;
                if (nd < d) { d = nd; mat = op.head.z; }
            }
            case 6u: { // hash-gated grid holes
                let cell = op.p0.x;
                let c = vec2<i32>(floor(pxz / cell));
                if (hash3(vec3<i32>(c.x, i32(level), c.y)) < op.p0.y) {
                    let oc = (vec2<f32>(c) + 0.5) * cell;
                    let cut = sd_box(vec3<f32>(p.x - oc.x, fy, p.z - oc.y), op.p1.xyz);
                    d = max(d, -cut);
                }
            }
            case 7u: { // pillars
                let sp = op.p0.x;
                let c = vec2<i32>(round(pxz / sp));
                let jit = vec2<f32>(hash2(c) - 0.5, hash2(c + vec2<i32>(311, 77)) - 0.5) * op.p0.y;
                let q = pxz - vec2<f32>(c) * sp - jit;
                let girth = op.p0.z + hash2(c + vec2<i32>(9, -4)) * op.p0.w;
                let nd = max(abs(q.x), abs(q.y)) - girth;
                if (nd < d) { d = nd; mat = op.head.z; }
            }
            case 8u: { // gated walls with doorways
                let sp = op.p0.x;
                let along_z = op.p0.w > 0.5;
                var a = p.x;
                var b = p.z;
                if (along_z) { a = p.z; b = p.x; }
                let wi = round(a / sp);
                let w = a - wi * sp;
                if (hash2(vec2<i32>(i32(wi) + i32(op.p1.x), i32(level))) < op.p0.z) {
                    var wall = abs(w) - op.p0.y;
                    let dc = op.p1.y;
                    let ci = round(b / dc);
                    let cl = b - ci * dc;
                    if (hash3(vec3<i32>(i32(wi), i32(ci), i32(level) + i32(op.p1.w))) < op.p1.z) {
                        let doorway = sd_box(vec3<f32>(w, fy + op.p2.w, cl), op.p2.xyz);
                        wall = max(wall, -doorway);
                    }
                    if (wall < d) { d = wall; mat = op.head.z; }
                }
            }
            case 9u: { // shaft registers
                let sp = op.p0.x;
                let c = vec2<i32>(round(pxz / sp));
                let jit = vec2<f32>(
                    hash2(c + vec2<i32>(41, 13)) - 0.5,
                    hash2(c + vec2<i32>(-7, 99)) - 0.5,
                ) * op.p0.y;
                sxz = pxz - vec2<f32>(c) * sp - jit;
                sr = op.p0.z + hash2(c) * op.p0.w;
                shaft = length(sxz) - sr;
            }
            case 10u: { // carve shafts
                d = max(d, -shaft);
            }
            case 11u: { // catwalk beams
                let n = op.p0.x;
                if (abs(level - round(level / n) * n) < 0.5) {
                    let beam = max(
                        max(abs(sxz.y) - op.p0.y, abs(fy + op.p0.z) - op.p0.w),
                        length(sxz) - (sr + op.p1.x),
                    );
                    if (beam < d) { d = beam; mat = op.head.z; }
                }
            }
            default: {}
        }
    }
    return WorldSample(d, mat);
}

// The continuous LOD field: the band every sample is evaluated at is a
// pure function of world position and the field anchor — identical for
// every chunk that samples this corner, at every LOD. Seam values cannot
// disagree, including at shell corners.
fn field_vs(p: vec3<f32>) -> f32 {
    let d = distance(p, prog.anchor.xyz);
    return clamp(d / prog.anchor.w, 1.0, prog.field.x);
}

fn apply_csg(d_in: f32, mat_in: u32, p: vec3<f32>, bvs: f32) -> vec2<f32> {
    var d_m = d_in;
    var mat = mat_in;
    if (params.csg_count > 0u && bvs < COARSE_VOXEL_M) {
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
    return vec2<f32>(d_m, bitcast<f32>(mat));
}

@compute @workgroup_size(6, 6, 6)
fn density_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(SAMPLES))) {
        return;
    }
    let vs = params.origin.w;
    // Sample i holds cell corner i - 2 (apron covers coarse-parity cells).
    let c = vec3<i32>(id) - vec3<i32>(2);
    let p = params.origin.xyz + vec3<f32>(c) * vs;

    let bvs = field_vs(p);
    let s = eval_program(p, bvs);
    var d_m = s.d;
    var mat = s.mat;
    let fine = apply_csg(d_m, mat, p, bvs);
    d_m = fine.x;
    mat = bitcast<u32>(fine.y);

    // SDF stored in voxel-size units, narrow band ±4.
    let sdf = clamp(d_m / vs, -4.0, 4.0);
    let material = select(0u, mat, sdf < 0.0);
    let packed = (pack2x16float(vec2<f32>(sdf, 0.0)) & 0xFFFFu) | (material << 16u);
    let base = params.slot * SLOT_STRIDE;
    density[base + id.x + SAMPLES * (id.y + SAMPLES * id.z)] = packed;
}
