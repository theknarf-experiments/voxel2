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
// Seam-free chunking: vertices are generated for cells [-1, 32] (the apron
// provides the samples), while a chunk owns exactly the quads whose edge
// origin corner lies in [0, 32)³. Boundary quads reference the duplicated
// apron vertices, which are bit-identical to the neighbor's because the
// density function is evaluated at identical coordinates.

const CELLS: u32 = 32u;
const CELLS_EXT: u32 = 34u;        // cells -1..=32
const SAMPLES: u32 = 36u;
const SLOT_STRIDE: u32 = 46656u;   // 36^3
const CELL_STRIDE: u32 = 39304u;   // 34^3
const NONE: u32 = 0xFFFFFFFFu;
const VERTEX_FLOATS: u32 = 6u;

struct ChunkParams {
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
@group(0) @binding(3) var<storage, read_write> vertices: array<f32>;
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

// A quad exists on the edge leaving corner `c` along `axis` when the edge
// crosses the surface. The four sharing cells reach into the apron at most
// one cell, which the extended vertex pass covers.
fn quad_exists(c: vec3<i32>, axis: u32) -> bool {
    var e = vec3<i32>(0);
    e[axis] = 1;
    let s0 = sample_sdf(c);
    let s1 = sample_sdf(c + e);
    return (s0 < 0.0) != (s1 < 0.0);
}

@compute @workgroup_size(4, 4, 4)
fn sn_count(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(CELLS_EXT))) {
        return;
    }
    let c = vec3<i32>(id) - vec3<i32>(1); // -1..=32
    let mask = cell_sign_mask(c);
    if (mask != 0u && mask != 255u) {
        atomicAdd(&counts[params.counts_slot].verts, 1u);
    }
    // Quads are only owned for origin corners inside [0, 32)³.
    if (all(c >= vec3<i32>(0)) && all(c < vec3<i32>(i32(CELLS)))) {
        for (var axis = 0u; axis < 3u; axis++) {
            if (quad_exists(c, axis)) {
                atomicAdd(&counts[params.counts_slot].quads, 1u);
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
    let normal = normalize(g);

    // Local index via the counts slot (zeroed before this pass).
    let local = atomicAdd(&counts[params.counts_slot].verts, 1u);
    let p = vec3<f32>(c) + sum / n; // chunk-local meters
    let out = (params.base_vertex + local) * VERTEX_FLOATS;
    vertices[out + 0u] = p.x;
    vertices[out + 1u] = p.y;
    vertices[out + 2u] = p.z;
    vertices[out + 3u] = normal.x;
    vertices[out + 4u] = normal.y;
    vertices[out + 5u] = normal.z;
    cell_indices[cell_slot_index(c)] = local;
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
        for (var i = 0u; i < 4u; i++) {
            quad[i] = cell_indices[cell_slot_index(cells[i])];
        }

        let q = atomicAdd(&counts[params.counts_slot].quads, 1u);
        let base = params.first_index + q * 6u;
        if (s0 < 0.0) {
            indices[base + 0u] = quad[0];
            indices[base + 1u] = quad[1];
            indices[base + 2u] = quad[2];
            indices[base + 3u] = quad[0];
            indices[base + 4u] = quad[2];
            indices[base + 5u] = quad[3];
        } else {
            indices[base + 0u] = quad[0];
            indices[base + 1u] = quad[2];
            indices[base + 2u] = quad[1];
            indices[base + 3u] = quad[0];
            indices[base + 4u] = quad[3];
            indices[base + 5u] = quad[2];
        }
    }
}
