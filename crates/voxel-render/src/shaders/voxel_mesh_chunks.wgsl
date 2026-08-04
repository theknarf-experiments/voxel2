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
const SAMPLES: u32 = 36u;
const SLOT_STRIDE: u32 = 46656u;   // 36^3
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
    // Clamp keeps apron-cell gradient probes in range (corner -2 / 35).
    let cc = clamp(c, vec3<i32>(-1), vec3<i32>(34));
    let i = vec3<u32>(cc + vec3<i32>(1));
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
    let pv = vec3<f32>(c) + sum / n;

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
    let out = index * VERTEX_WORDS;
    vertices[out + 0u] = pack2x16unorm(pn.xy);
    vertices[out + 1u] = pack2x16unorm(vec2<f32>(pn.z, 0.0));
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
