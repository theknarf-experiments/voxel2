// Batched surface-nets meshing over arena density slots.
//
//   sn_count    — exact vertex/quad counts for one chunk, written to the
//                 counts buffer for CPU readback (drives slab allocation).
//   sn_vertices — emit vertices into the slab at params.base_vertex; local
//                 vertex index recorded in the per-slot cell_indices scratch.
//   sn_quads    — emit local indices into the slab at params.first_index.
//
// The count and emit passes must agree exactly on skip rules; allocation
// uses the counted values, so emission can never overflow its slot.
//
// Seam-free chunking at equal LOD: vertices are generated for cells [-1, 32]
// (the apron provides the samples) and a chunk owns exactly the quads whose
// edge origin corner lies in [0, 32)³, so boundary quads reference
// duplicated apron vertices that are bit-identical to the neighbor's.
//
// Cracks at DIFFERENT-LOD boundaries are hidden by skirts: every vertex in a
// boundary cell layer gets a twin displaced into the solid along -normal,
// and every face-crossing quad is emitted a second time using the twins.
// Seen through a crack, the displaced copy reads as ground instead of void.
// (Boundary-vertex snapping replaces this in a later milestone.)

const CELLS: u32 = 32u;
const CELLS_EXT: u32 = 34u;        // cells -1..=32
const SAMPLES: u32 = 38u;          // corners -2..=35
const SLOT_STRIDE: u32 = 54872u;   // 38^3
const CELL_STRIDE: u32 = 39304u;   // 34^3
const NONE16: u32 = 0xFFFFu;
const NONE: u32 = 0xFFFFFFFFu;
// Compressed vertex: 3 u32 words — pos.xy (unorm16), pos.z + pad, oct normal.
const VERTEX_WORDS: u32 = 3u;
// Position quantization: chunk-local voxels mapped from [-8, 40] to [0, 1]
// (covers the apron and the deepest skirt with margin).
const POS_BIAS: f32 = 8.0;
const POS_RANGE: f32 = 48.0;
// Skirt depth in voxel units; must cover the surface deviation of a ±1 LOD
// neighbor (whose voxels are 2x, deviating up to ~2 coarse voxels).
const SKIRT_VOXELS: f32 = 6.0;

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

struct SlotCounts {
    verts: atomic<u32>,
    quads: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> density: array<u32>;
@group(0) @binding(1) var<uniform> params: ChunkParams;
@group(0) @binding(2) var<storage, read_write> cell_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> vertices: array<u32>;
// u16 indices, two per word; quads always write whole words (6 indices).
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> counts: array<SlotCounts>;

fn corner_offset(i: u32) -> vec3<i32> {
    return vec3<i32>(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
}

fn sample_sdf(c: vec3<i32>) -> f32 {
    // Clamp keeps gradient probes at the apron edge in range.
    let cc = clamp(c, vec3<i32>(-2), vec3<i32>(35));
    let i = vec3<u32>(cc + vec3<i32>(2));
    let packed = density[params.slot * SLOT_STRIDE + i.x + SAMPLES * (i.y + SAMPLES * i.z)];
    return unpack2x16float(packed & 0xFFFFu).x;
}

// Cell coordinates run -1..=32; scratch indexing is offset by one.
fn cell_slot_index(c: vec3<i32>) -> u32 {
    let i = vec3<u32>(c + vec3<i32>(1));
    return params.slot * CELL_STRIDE + i.x + CELLS_EXT * (i.y + CELLS_EXT * i.z);
}

// Boundary cells participate in skirts: the two outermost cell layers on
// each face (the crack sits on the face plane between them).
fn is_boundary_cell(c: vec3<i32>) -> bool {
    return any(c <= vec3<i32>(0)) || any(c >= vec3<i32>(31));
}

// Stitched-surface-nets snap factor: 1 at the outermost (apron) cell layer,
// 0.5 one cell in, 0 in the interior. Corner parity is global (chunk
// origins are even in voxel units), so equal-LOD neighbors snap to the same
// coarse solution and a fine chunk's face vertices land exactly on a
// coarser neighbor's vertices — watertight in both cases.
fn snap_factor(c: vec3<i32>) -> f32 {
    var t = 0.0;
    if (any(c <= vec3<i32>(-1)) || any(c >= vec3<i32>(32))) {
        t = 1.0;
    } else if (any(c <= vec3<i32>(0)) || any(c >= vec3<i32>(31))) {
        t = 0.5;
    }
    return t;
}

// Surface-nets vertex of the coarse-parity (2x) cell containing fine cell
// `c`, in fine-voxel units. Returns w = 0 if the coarse cell has no surface
// crossing (thin feature that vanishes at the parent LOD).
fn coarse_vertex(c: vec3<i32>) -> vec4<f32> {
    let big = vec3<i32>(
        i32(floor(f32(c.x) / 2.0)),
        i32(floor(f32(c.y) / 2.0)),
        i32(floor(f32(c.z) / 2.0)),
    );
    let base = big * 2;
    var s: array<f32, 8>;
    var mask = 0u;
    for (var i = 0u; i < 8u; i++) {
        s[i] = sample_sdf(base + corner_offset(i) * 2);
        if (s[i] < 0.0) {
            mask |= 1u << i;
        }
    }
    if (mask == 0u || mask == 255u) {
        return vec4<f32>(0.0);
    }
    var sum = vec3<f32>(0.0);
    var n = 0.0;
    for (var e = 0u; e < 12u; e++) {
        let a = EDGES[e].x;
        let b = EDGES[e].y;
        if ((s[a] < 0.0) != (s[b] < 0.0)) {
            let t = s[a] / (s[a] - s[b]);
            sum += mix(vec3<f32>(corner_offset(a)), vec3<f32>(corner_offset(b)), t);
            n += 1.0;
        }
    }
    return vec4<f32>(vec3<f32>(base) + (sum / n) * 2.0, 1.0);
}

const EDGES = array<vec2<u32>, 12>(
    vec2(0u, 1u), vec2(2u, 3u), vec2(4u, 5u), vec2(6u, 7u),
    vec2(0u, 2u), vec2(1u, 3u), vec2(4u, 6u), vec2(5u, 7u),
    vec2(0u, 4u), vec2(1u, 5u), vec2(2u, 6u), vec2(3u, 7u),
);

fn cell_sign_mask(c: vec3<i32>) -> u32 {
    var mask = 0u;
    for (var i = 0u; i < 8u; i++) {
        if (sample_sdf(c + corner_offset(i)) < 0.0) {
            mask |= 1u << i;
        }
    }
    return mask;
}

fn quad_exists(c: vec3<i32>, axis: u32) -> bool {
    var e = vec3<i32>(0);
    e[axis] = 1;
    let s0 = sample_sdf(c);
    let s1 = sample_sdf(c + e);
    return (s0 < 0.0) != (s1 < 0.0);
}

// A face-crossing quad (uses apron cells) gets a skirt copy.
fn quad_is_boundary(c: vec3<i32>, axis: u32) -> bool {
    let u_axis = (axis + 1u) % 3u;
    let v_axis = (axis + 2u) % 3u;
    return c[u_axis] == 0 || c[v_axis] == 0;
}

@compute @workgroup_size(4, 4, 4)
fn sn_count(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(CELLS_EXT))) {
        return;
    }
    let c = vec3<i32>(id) - vec3<i32>(1); // -1..=32
    let mask = cell_sign_mask(c);
    if (mask != 0u && mask != 255u) {
        var n = 1u;
        if (is_boundary_cell(c)) {
            n = 2u; // twin vertex for the skirt
        }
        atomicAdd(&counts[params.counts_slot].verts, n);
    }
    // Quads are only owned for origin corners inside [0, 32)³.
    if (all(c >= vec3<i32>(0)) && all(c < vec3<i32>(i32(CELLS)))) {
        for (var axis = 0u; axis < 3u; axis++) {
            if (quad_exists(c, axis)) {
                var n = 1u;
                if (quad_is_boundary(c, axis)) {
                    n = 2u; // skirt quad
                }
                atomicAdd(&counts[params.counts_slot].quads, n);
            }
        }
    }
}

@compute @workgroup_size(4, 4, 4)
fn sn_vertices(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(CELLS_EXT))) {
        return;
    }
    let c = vec3<i32>(id) - vec3<i32>(1); // -1..=32

    var s: array<f32, 8>;
    var mask = 0u;
    for (var i = 0u; i < 8u; i++) {
        s[i] = sample_sdf(c + corner_offset(i));
        if (s[i] < 0.0) {
            mask |= 1u << i;
        }
    }
    if (mask == 0u || mask == 255u) {
        cell_indices[cell_slot_index(c)] = NONE;
        return;
    }

    var sum = vec3<f32>(0.0);
    var n = 0.0;
    for (var e = 0u; e < 12u; e++) {
        let a = EDGES[e].x;
        let b = EDGES[e].y;
        if ((s[a] < 0.0) != (s[b] < 0.0)) {
            let t = s[a] / (s[a] - s[b]);
            sum += mix(vec3<f32>(corner_offset(a)), vec3<f32>(corner_offset(b)), t);
            n += 1.0;
        }
    }

    var g = vec3<f32>(0.0);
    for (var i = 0u; i < 8u; i++) {
        let q = c + corner_offset(i);
        g += vec3<f32>(
            sample_sdf(q + vec3(1, 0, 0)) - sample_sdf(q - vec3(1, 0, 0)),
            sample_sdf(q + vec3(0, 1, 0)) - sample_sdf(q - vec3(0, 1, 0)),
            sample_sdf(q + vec3(0, 0, 1)) - sample_sdf(q - vec3(0, 0, 1)),
        );
    }
    // Guard degenerate gradients (flat SDF regions) against NaN vertices.
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    if (dot(g, g) > 1e-12) {
        normal = normalize(g);
    }

    // Position in voxel units (chunk-local); scaling to meters happens at
    // decode using the per-chunk voxel size.
    var pv = vec3<f32>(c) + sum / n;

    // Stitch chunk boundaries: morph boundary-band vertices onto the
    // coarse-parity surface so neighboring chunks (same or ±1 LOD) meet.
    let snap = snap_factor(c);
    if (snap > 0.0) {
        let cv = coarse_vertex(c);
        if (cv.w > 0.5) {
            pv = mix(pv, cv.xyz, snap);
        }
    }

    let boundary = is_boundary_cell(c);
    var count = 1u;
    if (boundary) {
        count = 2u;
    }
    let local = atomicAdd(&counts[params.counts_slot].verts, count);

    write_vertex(params.base_vertex + local, pv, normal);
    var twin = NONE16;
    if (boundary) {
        twin = local + 1u;
        write_vertex(params.base_vertex + twin, pv - normal * SKIRT_VOXELS, normal);
    }
    cell_indices[cell_slot_index(c)] = (local & 0xFFFFu) | (twin << 16u);
}

// --- baked sun shadow (planet worlds) ----------------------------------------
// The sun is static, so a horizon march over the coarse terrain heightfield
// is baked per vertex at mesh time (free at draw time; carried in the spare
// u16 of the packed position).

fn shash2(p: vec2<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

fn svalue_noise2(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = shash2(i);
    let b = shash2(i + vec2<i32>(1, 0));
    let c = shash2(i + vec2<i32>(0, 1));
    let d = shash2(i + vec2<i32>(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn sfbm2(p: vec2<f32>, base_scale: f32, octaves: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = base_scale;
    for (var i = 0; i < octaves; i++) {
        sum += amp * (svalue_noise2(p * freq) - 0.5);
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum;
}

fn height_coarse(xz: vec2<f32>) -> f32 {
    let continents = sfbm2(xz, 0.00005, 3) * 800.0;
    let mountains = sfbm2(xz + vec2<f32>(510.0, -770.0), 0.0008, 4) * 420.0;
    let rolling = sfbm2(xz + vec2<f32>(1337.0, 55.0), 0.01, 3) * 36.0;
    return continents + mountains + rolling - 8.0;
}

fn baked_sun_shadow(world: vec3<f32>) -> f32 {
#ifdef MEGASTRUCTURE
    return 1.0;
#else
    let sun_dir = normalize(vec3<f32>(0.55, 0.5, 0.32));
    var occ = 0.0;
    var t = 8.0;
    for (var i = 0; i < 9; i++) {
        let sp = world + sun_dir * t;
        let dh = height_coarse(sp.xz) - sp.y;
        occ = max(occ, dh / t);
        t *= 1.8;
    }
    return 1.0 - smoothstep(0.0, 0.2, occ);
#endif
}

// Octahedral normal encoding.
fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let l1 = abs(n.x) + abs(n.y) + abs(n.z);
    var v = n.xy / l1;
    if (n.z < 0.0) {
        let sx = select(-1.0, 1.0, v.x >= 0.0);
        let sy = select(-1.0, 1.0, v.y >= 0.0);
        v = vec2<f32>((1.0 - abs(v.y)) * sx, (1.0 - abs(v.x)) * sy);
    }
    return v;
}

fn write_vertex(index: u32, pos_voxels: vec3<f32>, normal: vec3<f32>) {
    let pn = clamp((pos_voxels + POS_BIAS) / POS_RANGE, vec3<f32>(0.0), vec3<f32>(1.0));
    let world = params.origin.xyz + pos_voxels * params.origin.w;
    let shadow = baked_sun_shadow(world);
    let out = index * VERTEX_WORDS;
    vertices[out + 0u] = pack2x16unorm(pn.xy);
    vertices[out + 1u] = pack2x16unorm(vec2<f32>(pn.z, shadow));
    vertices[out + 2u] = pack2x16snorm(oct_encode(normal));
}

@compute @workgroup_size(4, 4, 4)
fn sn_quads(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(CELLS))) {
        return;
    }
    let c = vec3<i32>(id);
    let s0 = sample_sdf(c);

    for (var axis = 0u; axis < 3u; axis++) {
        if (!quad_exists(c, axis)) {
            continue;
        }
        var u = vec3<i32>(0);
        var v = vec3<i32>(0);
        u[(axis + 1u) % 3u] = 1;
        v[(axis + 2u) % 3u] = 1;
        let cells = array<vec3<i32>, 4>(c, c - u, c - u - v, c - v);

        var quad: array<u32, 4>;
        var twins: array<u32, 4>;
        for (var i = 0u; i < 4u; i++) {
            let packed = cell_indices[cell_slot_index(cells[i])];
            quad[i] = packed & 0xFFFFu;
            let t = packed >> 16u;
            // Non-boundary cells have no twin; fall back to the vertex.
            twins[i] = select(t, quad[i], t == NONE16);
        }

        let boundary = quad_is_boundary(c, axis);
        var emit = 1u;
        if (boundary) {
            emit = 2u;
        }
        let q = atomicAdd(&counts[params.counts_slot].quads, emit);
        write_quad(q, quad, s0 < 0.0);
        if (boundary) {
            write_quad(q + 1u, twins, s0 < 0.0);
        }
    }
}

// Six u16 indices per quad, packed two per u32 word. `first_index` is
// always even (quad-aligned), so the three words never straddle quads.
fn write_quad(quad_index: u32, corners: array<u32, 4>, flip: bool) {
    var i: array<u32, 6>;
    if (flip) {
        i = array<u32, 6>(corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]);
    } else {
        i = array<u32, 6>(corners[0], corners[2], corners[1], corners[0], corners[3], corners[2]);
    }
    let word = (params.first_index + quad_index * 6u) >> 1u;
    indices[word + 0u] = (i[0] & 0xFFFFu) | (i[1] << 16u);
    indices[word + 1u] = (i[2] & 0xFFFFu) | (i[3] << 16u);
    indices[word + 2u] = (i[4] & 0xFFFFu) | (i[5] << 16u);
}
