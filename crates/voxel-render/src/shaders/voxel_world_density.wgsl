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
/// Samples one density invocation walks down its column. Trades the
/// redundant height chain against thread count; see `density_main`.
///
/// Measured on the planet, settle in seconds: 1 -> 2.40, 2 -> 2.11,
/// 4 -> 1.85, 6 -> 1.86, 8 -> 1.95, 12 -> 2.07, whole column -> 3.50.
const Y_PER_THREAD: u32 = 4u;
const SLOT_STRIDE: u32 = 54872u; // 38^3
const BIG: f32 = 1.0e6;
const SOLID: f32 = -1.0e5;
const COARSE_VOXEL_M: f32 = 4.0;

struct ChunkParams {
    // xyz = chunk minimum corner in world meters, w = voxel size in meters.
    origin: vec4<f32>,
    // Minimum corner in integer world-voxel units (pos * 32, this
    // chunk's scale; w unused): sample positions derive from these exact
    // integers so shared samples are bit-identical across chunks at any
    // voxel size (see chunks.rs twin).
    origin_voxels: vec4<i32>,
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
// One world's slice: count = (op offset, op count, height ops, seed).
struct WorldHeader {
    count: vec4<u32>,
    sun: vec4<f32>,
}
struct WorldProgram {
    worlds: array<WorldHeader, 8>,
    ops: array<WorldOp>,
}

/// This chunk's world slice. Every loaded world's ops share one buffer,
/// so a dispatch can mix chunks from different worlds.
fn world_header() -> WorldHeader {
    return prog.worlds[u32(params.origin_voxels.w)];
}
@group(0) @binding(3) var<storage, read> prog: WorldProgram;

fn op_sdf(op: CsgOp, p: vec3<f32>) -> f32 {
    var q = p - op.center;
    let c = cos(-op.yaw);
    let s = sin(-op.yaw);
    q = vec3<f32>(q.x * c - q.z * s, q.y, q.x * s + q.z * c);
    if (op.kind >= 4u) {
        return length(q) - op.half.x;
    }
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

// Round half-up, shared with the CPU twin (`round_half_up` there):
// WGSL round() is half-to-even, Rust's is half-away-from-zero, and
// lattice registers land exactly on halves — the hash gates keyed on
// the rounded level must agree bit-for-bit across the twins.
fn round_half_up(x: f32) -> f32 {
    return floor(x + 0.5);
}

fn hash2(p: vec2<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u
        + world_header().count.w * 2654435769u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

fn hash3(p: vec3<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u + u32(p.z) * 2246822519u
        + world_header().count.w * 2654435769u;
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

// The generator is a pure function of position: no band-limiting, so
// every chunk at every LOD samples bit-identical values at shared
// corners — seams cannot disagree, at any LOD pair, in any epoch.
fn band_fade(wavelength: f32, voxel_size: f32) -> f32 {
    return 1.0;
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

// GENOPS HELPERS BEGIN (generated from voxel-core::opgen — run `mise run genops` after editing the op table)
fn v2(x: f32, y: f32) -> vec2<f32> { return vec2<f32>(x, y); }
fn to_i(x: f32) -> i32 { return i32(x); }
fn to_u(x: f32) -> u32 { return u32(x); }
// Region gate (WorldOp::region). Emitted here rather than written into
// each shader so the three files cannot drift: every interpreter loop
// calls this before its switch, and 0 means the op is ungated.
//
// An integer form that compared the axes quantized to bytes, skipping
// the unpack, measured no faster (megastructure settle 2.12 -> 2.30 s,
// inside the run-to-run noise): what a gated-out op costs is the loop
// iteration and the 64-byte read, not the arithmetic. Left in the
// obvious form.
fn region_gate(packed: u32, ta: f32, tb: f32) -> bool {
    if packed == 0u { return true; }
    let a0 = f32(packed & 0xFFu) / 255.0;
    let a1 = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b0 = f32((packed >> 16u) & 0xFFu) / 255.0;
    let b1 = f32((packed >> 24u) & 0xFFu) / 255.0;
    return ta >= a0 && ta < a1 && tb >= b0 && tb < b1;
}
fn v3(x: f32, y: f32, z: f32) -> vec3<f32> { return vec3<f32>(x, y, z); }
fn iv2(x: i32, y: i32) -> vec2<i32> { return vec2<i32>(x, y); }
fn iv3(x: i32, y: i32, z: i32) -> vec3<i32> { return vec3<i32>(x, y, z); }
fn to_v2(v: vec2<i32>) -> vec2<f32> { return vec2<f32>(v); }
fn to_iv2(v: vec2<f32>) -> vec2<i32> { return vec2<i32>(v); }
fn floor2(v: vec2<f32>) -> vec2<f32> { return floor(v); }
// GENOPS HELPERS END

struct WorldSample {
    d: f32,
    mat: u32,
}

/// What the height chain produces for one column. Everything in it
/// depends on xz alone, which is why it is computed once per column
/// rather than once per sample — on a heightfield world that IS the
/// program, evaluated 38 times over for every column before this split.
struct Column {
    h: f32,
    ta: f32,
    tb: f32,
}

fn eval_column(pxz: vec2<f32>, vs: f32) -> Column {
    let coarse = vs >= COARSE_VOXEL_M;
    var h = 0.0;
    var warp = vec2<f32>(0.0);
    var ta = 0.0;
    var tb = 0.0;
    let w = world_header();
    for (var i = 0u; i < w.count.y; i++) {
        let op = prog.ops[w.count.x + i];
        // The same LOD gating the sample pass applies: a height op can
        // be fine- or coarse-only too, and the two passes must agree
        // about which ops exist or the column is not the column.
        if (coarse && (op.head.y & 1u) != 0u) { continue; }
        if (!coarse && (op.head.y & 2u) != 0u) { continue; }
        if (!region_gate(op.head.w, ta, tb)) { continue; }
        switch op.head.x {
// GENOPS COLUMN ARMS BEGIN (generated from voxel-core::opgen — run `mise run genops` after editing the op table)
            case 0u: { // WOP_HEIGHT_FBM
                h += fbm(pxz + warp + op.p0.xy, op.p0.z, to_i(op.p1.x), vs, to_u(op.p1.y)) * op.p0.w;
            }
            case 1u: { // WOP_HEIGHT_OFFSET
                h += op.p0.x;
            }
            case 16u: { // WOP_HEIGHT_STEP
                h += op.p0.z * smoothstep(op.p0.x, op.p0.y, h);
            }
            case 14u: { // WOP_WARP_XZ
                let q = pxz + op.p0.zw;
                let oct = to_i(op.p1.x);
                warp.x += fbm(q, op.p0.x, oct, vs, 0) * op.p0.y;
                warp.y += fbm(q + v2(713.0, -337.0), op.p0.x, oct, vs, 0) * op.p0.y;
            }
            case 19u: { // WOP_REGION_AXES
                ta = fbm(pxz + op.p0.xy, op.p0.z, to_i(op.p1.z), vs, 0) + 0.5;
                tb = fbm(pxz + op.p1.xy, op.p0.w, to_i(op.p1.z), vs, 0) + 0.5;
            }
            case 20u: { // WOP_HEIGHT_BAND_FBM
                let fa = op.p1.z;
                let wa = smoothstep(op.p2.x - fa, op.p2.x + fa, ta) * (1.0 - smoothstep(op.p2.y - fa, op.p2.y + fa, ta));
                let wb = smoothstep(op.p2.z - fa, op.p2.z + fa, tb) * (1.0 - smoothstep(op.p2.w - fa, op.p2.w + fa, tb));
                h += min(wa, wb) * (op.p1.w + fbm(pxz + warp + op.p0.xy, op.p0.z, to_i(op.p1.x), vs, to_u(op.p1.y)) * op.p0.w);
            }
// GENOPS COLUMN ARMS END
            default: {}
        }
    }
    return Column(h, ta, tb);
}

fn eval_program(p: vec3<f32>, vs: f32, col: Column) -> WorldSample {
    let coarse = vs >= COARSE_VOXEL_M;
    let h = col.h;
    let ta = col.ta;
    let tb = col.tb;
    var d = BIG;
    var mat = 1u;
    var level = 0.0;
    var fy = p.y;
    var sxz = vec2<f32>(0.0);
    var sr = 0.0;
    var shaft = BIG;
    let pxz = p.xz;

    let w = world_header();
    for (var i = 0u; i < w.count.y; i++) {
        let op = prog.ops[w.count.x + i];
        if (coarse && (op.head.y & 1u) != 0u) { continue; }
        if (!coarse && (op.head.y & 2u) != 0u) { continue; }
        if (!region_gate(op.head.w, ta, tb)) { continue; }
        switch op.head.x {
// GENOPS ARMS BEGIN (generated from voxel-core::opgen — run `mise run genops` after editing the op table)
            case 15u: { // WOP_FBM3
                let q = p + op.p1.xyz;
                let n = fbm3(q, op.p0.x, op.p0.y, to_i(op.p2.x), vs);
                let sd = (op.p0.z - n) * op.p0.w;
                if op.p1.w < 0.5 {
                    if sd < d { d = sd; mat = op.head.z; }
                } else {
                    d = max(d, -sd);
                }
            }
            case 2u: { // WOP_HEIGHT_SURFACE
                let nd = p.y - h;
                if nd < d { d = nd; mat = op.head.z; }
            }
            case 18u: { // WOP_MATERIAL_BAND
                if mat == to_u(op.p1.z) && ta >= op.p0.x && ta < op.p0.y && tb >= op.p0.z && tb < op.p0.w { mat = op.head.z; }
            }
            case 3u: { // WOP_COARSE_SOLID
                if SOLID < d { d = SOLID; mat = op.head.z; }
            }
            case 4u: { // WOP_LATTICE_Y
                level = round_half_up(p.y / op.p0.x);
                fy = p.y - level * op.p0.x;
            }
            case 5u: { // WOP_SLABS_Y
                let nd = abs(fy) - op.p0.x;
                if nd < d { d = nd; mat = op.head.z; }
            }
            case 6u: { // WOP_GRID_HOLES
                let cell = op.p0.x;
                let c = to_iv2(floor2(pxz / cell));
                if hash3(iv3(c.x, to_i(level), c.y)) < op.p0.y {
                    let oc = (to_v2(c) + 0.5) * cell;
                    let cut = sd_box(v3(p.x - oc.x, fy, p.z - oc.y), op.p1.xyz);
                    d = max(d, -cut);
                }
            }
            case 7u: { // WOP_PILLARS_XZ
                let sp = op.p0.x;
                let c = iv2(to_i(round_half_up(pxz.x / sp)), to_i(round_half_up(pxz.y / sp)));
                let jit = v2(hash2(c) - 0.5, hash2(c + iv2(311, 77)) - 0.5) * op.p0.y;
                let q = pxz - to_v2(c) * sp - jit;
                let girth = op.p0.z + hash2(c + iv2(9, -4)) * op.p0.w;
                let nd = max(abs(q.x), abs(q.y)) - girth;
                if nd < d { d = nd; mat = op.head.z; }
            }
            case 8u: { // WOP_WALLS
                let sp = op.p0.x;
                var a = p.x;
                var b = p.z;
                if op.p0.w > 0.5 { a = p.z; b = p.x; }
                let wi = round_half_up(a / sp);
                let w = a - wi * sp;
                if hash2(iv2(to_i(wi) + to_i(op.p1.x), to_i(level))) < op.p0.z {
                    var wall = abs(w) - op.p0.y;
                    let dc = op.p1.y;
                    let ci = round_half_up(b / dc);
                    let cl = b - ci * dc;
                    if hash3(iv3(to_i(wi), to_i(ci), to_i(level) + to_i(op.p1.w))) < op.p1.z {
                        let doorway = sd_box(v3(w, fy + op.p2.w, cl), op.p2.xyz);
                        wall = max(wall, -doorway);
                    }
                    if wall < d { d = wall; mat = op.head.z; }
                }
            }
            case 9u: { // WOP_SHAFTS_XZ
                let sp = op.p0.x;
                let c = iv2(to_i(round_half_up(pxz.x / sp)), to_i(round_half_up(pxz.y / sp)));
                let jit = v2(hash2(c + iv2(41, 13)) - 0.5, hash2(c + iv2(-7, 99)) - 0.5) * op.p0.y;
                sxz = pxz - to_v2(c) * sp - jit;
                sr = op.p0.z + hash2(c) * op.p0.w;
                shaft = length(sxz) - sr;
            }
            case 10u: { // WOP_SHAFTS_CUT
                d = max(d, -shaft);
            }
            case 11u: { // WOP_BEAMS
                let n = op.p0.x;
                if abs(level - round_half_up(level / n) * n) < 0.5 {
                    let beam = max(max(abs(sxz.y) - op.p0.y, abs(fy + op.p0.z) - op.p0.w), length(sxz) - (sr + op.p1.x));
                    if beam < d { d = beam; mat = op.head.z; }
                }
            }
// GENOPS ARMS END
            default: {}
        }
    }
    return WorldSample(d, mat);
}

fn apply_csg(d_in: f32, mat_in: u32, p: vec3<f32>) -> vec2<f32> {
    var d_m = d_in;
    var mat = mat_in;
    if (params.csg_count > 0u) {
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

/// One invocation per RUN of Y_PER_THREAD samples down a column.
///
/// The height chain is xz-only, so evaluating it per sample re-did it
/// for all 38 layers of every column — and on a heightfield world it is
/// the entire program. But a whole column per thread is 1444 heavy
/// threads where there were 54872 light ones, and the occupancy loss
/// cost more than the redundancy did (measured: 2.40 s -> 3.50 s to
/// settle the planet). A short run amortises the column over several
/// samples while keeping enough threads in flight to hide latency.
@compute @workgroup_size(8, 8, 1)
fn density_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let y0 = id.z * Y_PER_THREAD;
    if (id.x >= SAMPLES || id.y >= SAMPLES || y0 >= SAMPLES) {
        return;
    }
    let vs = params.origin.w;
    // Sample i holds cell corner i - 2 (apron covers coarse-parity cells).
    let cx = i32(id.x) - 2;
    let cz = i32(id.y) - 2;
    // Exact integer world-voxel index -> one rounding, identical for
    // every chunk (and LOD: (2k)*(vs/2) == k*vs bit-exactly) that
    // evaluates this world sample.
    let ox = params.origin_voxels.xyz;
    let pxz = vec2<f32>(f32(ox.x + cx), f32(ox.z + cz)) * vs;
    let col = eval_column(pxz, vs);

    let base = params.slot * SLOT_STRIDE;
    let y1 = min(y0 + Y_PER_THREAD, SAMPLES);
    for (var iy = y0; iy < y1; iy++) {
        let p = vec3<f32>(pxz.x, f32(ox.y + i32(iy) - 2) * vs, pxz.y);
        let s = eval_program(p, vs, col);
        var d_m = s.d;
        var mat = s.mat;
        let fine = apply_csg(d_m, mat, p);
        d_m = fine.x;
        mat = bitcast<u32>(fine.y);

        // SDF stored in voxel-size units, narrow band ±4.
        let sdf = clamp(d_m / vs, -4.0, 4.0);
        let material = select(0u, mat, sdf < 0.0);
        let packed = (pack2x16float(vec2<f32>(sdf, 0.0)) & 0xFFFFu) | (material << 16u);
        density[base + id.x + SAMPLES * (iy + SAMPLES * id.y)] = packed;
    }
}
