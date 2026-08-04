// M2 prototype surface-nets meshing: three entry points dispatched in order
// against the same bind group.
//
//   sn_vertices — one thread per cell; sign-changing cells emit one vertex
//                 (mean of edge crossings) and record its index in the
//                 cell_indices volume.
//   sn_quads    — one thread per cell; each cell owns the three grid edges
//                 leaving its origin corner. A sign-changing edge emits a
//                 quad connecting the four cells around it.
//   sn_finalize — writes the DrawIndexedIndirect args.
//
// Vertices are written tightly packed (6 f32: pos.xyz, normal.xyz) so the
// same buffer binds as a vertex buffer for drawing.

const CELLS: u32 = 32u;
const SAMPLES: u32 = 36u;
const MAX_VERTS: u32 = 65536u;
const MAX_INDICES: u32 = 393216u;
const NONE: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<storage, read_write> density: array<u32>;
@group(0) @binding(1) var<storage, read_write> cell_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(3) var<storage, read_write> indices: array<u32>;

struct Counts {
    verts: atomic<u32>,
    idx: atomic<u32>,
}
@group(0) @binding(4) var<storage, read_write> counts: Counts;

// DrawIndexedIndirect: index_count, instance_count, first_index, base_vertex, first_instance
@group(0) @binding(5) var<storage, read_write> indirect: array<u32>;

// Corner `i` of a cell is offset (i&1, i>>1&1, i>>2&1) from its origin corner.
fn corner_offset(i: u32) -> vec3<i32> {
    return vec3<i32>(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
}

// Sample the SDF at a cell-corner coordinate in [-1, 34].
fn sample_sdf(c: vec3<i32>) -> f32 {
    let i = vec3<u32>(c + vec3<i32>(1));
    let packed = density[i.x + SAMPLES * (i.y + SAMPLES * i.z)];
    return unpack2x16float(packed & 0xFFFFu).x;
}

fn cell_index(c: vec3<u32>) -> u32 {
    return c.x + CELLS * (c.y + CELLS * c.z);
}

// The 12 cell edges as pairs of corner indices.
const EDGES = array<vec2<u32>, 12>(
    vec2(0u, 1u), vec2(2u, 3u), vec2(4u, 5u), vec2(6u, 7u), // along x
    vec2(0u, 2u), vec2(1u, 3u), vec2(4u, 6u), vec2(5u, 7u), // along y
    vec2(0u, 4u), vec2(1u, 5u), vec2(2u, 6u), vec2(3u, 7u), // along z
);

@compute @workgroup_size(4, 4, 4)
fn sn_vertices(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(CELLS))) {
        return;
    }
    let c = vec3<i32>(id);

    var s: array<f32, 8>;
    var inside_mask = 0u;
    for (var i = 0u; i < 8u; i++) {
        s[i] = sample_sdf(c + corner_offset(i));
        if (s[i] < 0.0) {
            inside_mask |= 1u << i;
        }
    }
    if (inside_mask == 0u || inside_mask == 255u) {
        cell_indices[cell_index(id)] = NONE;
        return;
    }

    // Surface-nets vertex: mean of the edge/surface crossing points.
    var sum = vec3<f32>(0.0);
    var count = 0.0;
    for (var e = 0u; e < 12u; e++) {
        let a = EDGES[e].x;
        let b = EDGES[e].y;
        if ((s[a] < 0.0) != (s[b] < 0.0)) {
            let t = s[a] / (s[a] - s[b]);
            sum += mix(vec3<f32>(corner_offset(a)), vec3<f32>(corner_offset(b)), t);
            count += 1.0;
        }
    }
    let local = sum / count;

    // Normal: average of central-difference gradients at the 8 corners.
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

    let vi = atomicAdd(&counts.verts, 1u);
    if (vi >= MAX_VERTS) {
        cell_indices[cell_index(id)] = NONE;
        return;
    }
    let p = vec3<f32>(c) + local;
    let base = vi * 6u;
    vertices[base + 0u] = p.x;
    vertices[base + 1u] = p.y;
    vertices[base + 2u] = p.z;
    vertices[base + 3u] = normal.x;
    vertices[base + 4u] = normal.y;
    vertices[base + 5u] = normal.z;
    cell_indices[cell_index(id)] = vi;
}

@compute @workgroup_size(4, 4, 4)
fn sn_quads(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(CELLS))) {
        return;
    }
    let c = vec3<i32>(id);
    let s0 = sample_sdf(c);

    for (var axis = 0u; axis < 3u; axis++) {
        var e = vec3<i32>(0);
        e[axis] = 1;
        let s1 = sample_sdf(c + e);
        if ((s0 < 0.0) == (s1 < 0.0)) {
            continue;
        }

        // The four cells sharing this edge: c, c-u, c-u-v, c-v where u, v are
        // the other two axes. Boundary edges (any cell outside the chunk) are
        // skipped — cross-chunk stitching is a later milestone.
        var u = vec3<i32>(0);
        var v = vec3<i32>(0);
        u[(axis + 1u) % 3u] = 1;
        v[(axis + 2u) % 3u] = 1;
        let cells = array<vec3<i32>, 4>(c, c - u, c - u - v, c - v);

        var quad: array<u32, 4>;
        var valid = true;
        for (var i = 0u; i < 4u; i++) {
            let q = cells[i];
            if (any(q < vec3<i32>(0))) {
                valid = false;
                break;
            }
            let vi = cell_indices[cell_index(vec3<u32>(q))];
            if (vi == NONE) {
                valid = false;
                break;
            }
            quad[i] = vi;
        }
        if (!valid) {
            continue;
        }

        let base = atomicAdd(&counts.idx, 6u);
        if (base + 6u > MAX_INDICES) {
            continue; // overflow: drop the quad (finalize clamps the count)
        }
        // Winding flips with the edge direction (solid at c vs at c + e).
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

@compute @workgroup_size(1)
fn sn_finalize() {
    indirect[0] = min(atomicLoad(&counts.idx), MAX_INDICES);
    indirect[1] = 1u;
    indirect[2] = 0u;
    indirect[3] = 0u;
    indirect[4] = 0u;
}
