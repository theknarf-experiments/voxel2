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
// DIFFERENT-LOD boundaries are exact, not hidden: the density pass blends
// the stored field to the parent band in the shell toward a coarser
// neighbor, apron vertices there snap onto the coarse-parity surface-nets
// vertex (bit-equal to the neighbor's own vertex), and seam-quad ownership
// follows the per-face LOD mask — the finer side owns each seam plane.
// No skirts anywhere.

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
// (covers the apron with margin; kept for vertex-format stability).
const POS_BIAS: f32 = 8.0;
const POS_RANGE: f32 = 48.0;

// Layout twin of `ChunkParams` in voxel-render, generated from
// `voxel_core::layout::CHUNK_PARAMS`. Run `mise run genops` after
// editing it.
// GENMAT CHUNKPARAMS BEGIN
struct ChunkParams {
    // xyz = chunk minimum corner in world meters, w = voxel size in meters.
    origin: vec4<f32>,
    // Minimum corner in integer world-voxel units (pos * 32, this chunk's
    // scale); w = which WORLD's program to interpret. Sample positions
    // derive from these EXACT integers so two chunks sharing a sample
    // compute a bit-identical position at any voxel size — `origin + idx
    // * vs` rounds differently per chunk whenever the voxel size is not
    // an exact binary float (0.1 m is not), and one ULP flips a sign
    // where a surface grazes a sample: deterministic seam cracks.
    origin_voxels: vec4<i32>,
    // Density arena slot this chunk's samples live in.
    slot: u32,
    base_vertex: u32,
    first_index: u32,
    counts_slot: u32,
    // Range into this frame's concatenated CSG op buffer.
    csg_offset: u32,
    csg_count: u32,
    // x = seam mask, 2 bits per face (+x,-x,+y,-y,+z,-z): 1 = neighbour
    // coarser, 2 = neighbour finer. Read by the mesh pass only.
    _pad: vec2<u32>,
}
// GENMAT CHUNKPARAMS END

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

// Generator program (64 B ops, layout mirrors voxel-core WorldOp); the
// shadow bake reads the height ops. count = (total, height ops, -, -).
struct WorldOp {
    head: vec4<u32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
}
// One world's slice: count = (op offset, op count, height ops, seed).
// Layout twin of `GpuWorldProgram` and the density shader.
struct WorldHeader {
    count: vec4<u32>,
    sun: vec4<f32>,
}
struct WorldProgram {
    worlds: array<WorldHeader, 8>,
    ops: array<WorldOp>,
}
@group(0) @binding(6) var<storage, read> prog: WorldProgram;

/// This chunk's world slice — see the density shader.
fn world_header() -> WorldHeader {
    return prog.worlds[u32(params.origin_voxels.w)];
}

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

fn sample_material(c: vec3<i32>) -> u32 {
    let cc = clamp(c, vec3<i32>(-2), vec3<i32>(35));
    let i = vec3<u32>(cc + vec3<i32>(2));
    let packed = density[params.slot * SLOT_STRIDE + i.x + SAMPLES * (i.y + SAMPLES * i.z)];
    return (packed >> 16u) & 0xFFu;
}

// Cell coordinates run -1..=32; scratch indexing is offset by one.
fn cell_slot_index(c: vec3<i32>) -> u32 {
    let i = vec3<u32>(c + vec3<i32>(1));
    return params.slot * CELL_STRIDE + i.x + CELLS_EXT * (i.y + CELLS_EXT * i.z);
}

// Seam mask: one coarser-neighbor bit per direction of the 26-neighborhood,
// in scan order (dz, dy, dx in -1..=1, center skipped) — twin of
// PostState::seam_mask in voxel-engine streaming. Faces alone are not
// enough: a chunk whose *diagonal* neighbor is coarser while both face
// neighbors are equal must still snap its shared edge/corner cells, or a
// pinhole opens at the junction where the face neighbors do snap.
fn dir_coarser(bit: u32) -> bool {
    return ((params._pad.x >> bit) & 1u) == 1u;
}

// Apron cells snap onto the coarse-parity vertex (bit-equal to the coarser
// neighbor's own vertex — the density field is a pure function of
// position) whenever ANY neighbor region the cell touches is coarser.
fn snap_to_parity(c: vec3<i32>) -> bool {
    var idx = 0u;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) { continue; }
                if (dir_coarser(idx)) {
                    let on_x = (dx == 0) || (dx == 1 && c.x == 32) || (dx == -1 && c.x == -1);
                    let on_y = (dy == 0) || (dy == 1 && c.y == 32) || (dy == -1 && c.y == -1);
                    let on_z = (dz == 0) || (dz == 1 && c.z == 32) || (dz == -1 && c.z == -1);
                    if (on_x && on_y && on_z) {
                        return true;
                    }
                }
                idx++;
            }
        }
    }
    return false;
}

// Scan-order bits of the three +axis face directions (+x, +y, +z).
const PLUS_FACE_BITS = vec3<u32>(13u, 15u, 21u);

// Seam-aware quad ownership. Default: edge-origin in [0, 32)³. Toward a
// coarser +face the chunk additionally owns the seam-plane quads (origin
// 32 on that axis). Nothing is ever ceded: coverage is a union of
// unilateral contributions (holes are impossible by construction); where
// a finer neighbor also meshes a seam plane, the parity snap makes the
// two copies exactly coplanar, so overlap is invisible.
fn owns_quad(c: vec3<i32>, axis: u32) -> bool {
    for (var a = 0u; a < 3u; a++) {
        let ca = c[a];
        if (ca < 0 || ca > 32) {
            return false;
        }
        if (ca == 32 && !(axis != a && dir_coarser(PLUS_FACE_BITS[a]))) {
            return false;
        }
    }
    return true;
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
    for (var axis = 0u; axis < 3u; axis++) {
        if (owns_quad(c, axis) && quad_exists(c, axis)) {
            atomicAdd(&counts[params.counts_slot].quads, 1u);
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

    // Seam vertices toward a coarser neighbor land exactly on its own
    // surface-nets vertex: the parity cell's SN solution over parent-band
    // samples (the density pass blended this shell to the parent band).
    var snap_failed = false;
    if (snap_to_parity(c)) {
        let cv = coarse_vertex(c);
        if (cv.w > 0.5) {
            pv = cv.xyz;
        } else {
            // Thin feature: the containing parity cell has no crossing, so
            // there is no coarse vertex to land on and the fine surface
            // would hang as an open rim (visible sliver into the void).
            // Weld onto the nearest neighboring parity vertex instead —
            // the scan order is fixed and both sides of any seam sample
            // identical world positions, so every copy of this cell picks
            // the same weld target on the coarse neighbor's surface.
            snap_failed = true;
            let big = vec3<i32>(
                i32(floor(f32(c.x) / 2.0)),
                i32(floor(f32(c.y) / 2.0)),
                i32(floor(f32(c.z) / 2.0)),
            );
            // Search ONLY along axes where this cell is safely interior
            // (local coord 1..30). Cells at -1/0/31/32 on an axis are the
            // ones two neighboring chunks both own under different local
            // coordinates — searching along such an axis gives the two
            // copies different candidate sets (one side's neighbor falls
            // outside its sample range) and they weld to different
            // targets: a crack exactly on the seam. Restricting to
            // shared-interior axes keeps the weld a pure function of
            // world position.
            let free_axis = vec3<bool>(
                c.x >= 1 && c.x <= 30,
                c.y >= 1 && c.y <= 30,
                c.z >= 1 && c.z <= 30,
            );
            for (var dz = -1; dz <= 1 && snap_failed; dz++) {
                if (dz != 0 && !free_axis.z) { continue; }
                for (var dy = -1; dy <= 1 && snap_failed; dy++) {
                    if (dy != 0 && !free_axis.y) { continue; }
                    for (var dx = -1; dx <= 1 && snap_failed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) { continue; }
                        if (dx != 0 && !free_axis.x) { continue; }
                        let nb = big + vec3<i32>(dx, dy, dz);
                        if (any(nb < vec3<i32>(-1)) || any(nb > vec3<i32>(16))) {
                            continue;
                        }
                        let nv = coarse_vertex(nb * 2);
                        if (nv.w > 0.5) {
                            pv = nv.xyz;
                            snap_failed = false;
                        }
                    }
                }
            }
        }
    }

    let local = atomicAdd(&counts[params.counts_slot].verts, 1u);

    // Vertex material: the material of the most-solid corner.
    var mat = 0u;
    var best = 1.0e9;
    for (var i = 0u; i < 8u; i++) {
        if (s[i] < best) {
            best = s[i];
            mat = sample_material(c + corner_offset(i));
        }
    }
    // The painted surface material is NOT applied here. It used to be, and
    // a vertex is the wrong place for it: paint only applies where the LOD
    // field has gone coarse, which is exactly where vertices are 51-102 m
    // apart, so an 8 m map arrived as flat 100 m quads. The draw shader
    // reads it per fragment instead — see `painted_material` there.
    //
    // Debug (eval): flag failed-snap vertices via a reserved material so
    // the draw shader can paint them for correlation with hole pixels.
    if (snap_failed && params._pad.y == 1u) {
        mat = 255u;
    }

    write_vertex(params.base_vertex + local, pv, normal, mat);
    cell_indices[cell_slot_index(c)] = local & 0xFFFFu;
}

// --- baked sun shadow (planet worlds) ----------------------------------------
// The sun is static, so a horizon march over the coarse terrain heightfield
// is baked per vertex at mesh time (free at draw time; carried in the spare
// u16 of the packed position).

fn shash2(p: vec2<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u
        + world_header().count.w * 2654435769u;
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

// Sum of the generator program's height ops at coarse detail (finer bands
// fade out below ~16 m wavelengths, matching CPU impostor seating).
// Band-limited (~16 m cutoff) FBM with shaping mode, for height-op replay.
fn coarse_fbm(p: vec2<f32>, base_scale: f32, octaves: i32, mode: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = base_scale;
    for (var o = 0; o < octaves; o++) {
        let fade = 1.0;
        let n = svalue_noise2(p * freq);
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
    return sum;
}

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
// GENOPS HELPERS END

// The height replay's band-limited FBM (no vs — inherently coarse).
fn hfbm(q: vec2<f32>, s: f32, o: i32, m: u32) -> f32 {
    return coarse_fbm(q, s, o, m);
}

fn height_coarse(xz: vec2<f32>) -> f32 {
    var h = 0.0;
    var warp = vec2<f32>(0.0);
    // Region axes, filled by WOP_REGION_AXES and read by the band ops.
    var ta = 0.0;
    var tb = 0.0;
    let pxz = xz;
    let w = world_header();
    for (var i = 0u; i < w.count.y; i++) {
        let op = prog.ops[w.count.x + i];
        if (!region_gate(op.head.w, ta, tb)) { continue; }
        switch op.head.x {
// GENOPS ARMS BEGIN (generated from voxel-core::opgen — run `mise run genops` after editing the op table)
            case 0u: { // WOP_HEIGHT_FBM
                h += hfbm(pxz + warp + op.p0.xy, op.p0.z, to_i(op.p1.x), to_u(op.p1.y)) * op.p0.w;
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
                warp.x += hfbm(q, op.p0.x, oct, 0) * op.p0.y;
                warp.y += hfbm(q + v2(713.0, -337.0), op.p0.x, oct, 0) * op.p0.y;
            }
            case 19u: { // WOP_REGION_AXES
                ta = hfbm(pxz + op.p0.xy, op.p0.z, to_i(op.p1.z), 0) + 0.5;
                tb = hfbm(pxz + op.p1.xy, op.p0.w, to_i(op.p1.z), 0) + 0.5;
            }
            case 20u: { // WOP_HEIGHT_BAND_FBM
                let fa = op.p1.z;
                let wa = smoothstep(op.p2.x - fa, op.p2.x + fa, ta) * (1.0 - smoothstep(op.p2.y - fa, op.p2.y + fa, ta));
                let wb = smoothstep(op.p2.z - fa, op.p2.z + fa, tb) * (1.0 - smoothstep(op.p2.w - fa, op.p2.w + fa, tb));
                h += min(wa, wb) * (op.p1.w + hfbm(pxz + warp + op.p0.xy, op.p0.z, to_i(op.p1.x), to_u(op.p1.y)) * op.p0.w);
            }
// GENOPS ARMS END
            default {}
        }
    }
    return h;
}

fn baked_sun_shadow(world: vec3<f32>) -> f32 {
    // Heightfield-free programs (pure structures) have no horizon to march.
    let wh = world_header();
    if (wh.count.z == 0u) {
        return 1.0;
    }
    let sun_dir = normalize(wh.sun.xyz);
    var occ = 0.0;
    var t = 8.0;
    for (var i = 0; i < 9; i++) {
        let sp = world + sun_dir * t;
        let dh = height_coarse(sp.xz) - sp.y;
        occ = max(occ, dh / t);
        t *= 1.8;
    }
    return 1.0 - smoothstep(0.0, 0.2, occ);
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

// The spare u16 packs material (high byte) and baked shadow (low byte).
fn write_vertex(index: u32, pos_voxels: vec3<f32>, normal: vec3<f32>, material: u32) {
    let pn = clamp((pos_voxels + POS_BIAS) / POS_RANGE, vec3<f32>(0.0), vec3<f32>(1.0));
    let world = params.origin.xyz + pos_voxels * params.origin.w;
    let shadow_u = u32(round(baked_sun_shadow(world) * 255.0));
    let extra = f32((material << 8u) | shadow_u) / 65535.0;
    let out = index * VERTEX_WORDS;
    vertices[out + 0u] = pack2x16unorm(pn.xy);
    vertices[out + 1u] = pack2x16unorm(vec2<f32>(pn.z, extra));
    vertices[out + 2u] = pack2x16snorm(oct_encode(normal));
}

@compute @workgroup_size(4, 4, 4)
fn sn_quads(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(CELLS_EXT))) {
        return;
    }
    let c = vec3<i32>(id) - vec3<i32>(1); // -1..=32 (seam ownership extends)
    let s0 = sample_sdf(c);

    for (var axis = 0u; axis < 3u; axis++) {
        if (!owns_quad(c, axis) || !quad_exists(c, axis)) {
            continue;
        }
        var u = vec3<i32>(0);
        var v = vec3<i32>(0);
        u[(axis + 1u) % 3u] = 1;
        v[(axis + 2u) % 3u] = 1;
        let cells = array<vec3<i32>, 4>(c, c - u, c - u - v, c - v);

        var quad: array<u32, 4>;
        for (var i = 0u; i < 4u; i++) {
            quad[i] = cell_indices[cell_slot_index(cells[i])] & 0xFFFFu;
        }

        // Split across the shorter diagonal (better triangles on saddles) —
        // but never across a collapsed one: parity snapping merges seam
        // vertex pairs, and a quad whose 0-2 diagonal has collapsed loses
        // BOTH triangles if split along it (each contains the duplicate
        // pair). Splitting across the surviving diagonal keeps the quad's
        // one real triangle.
        let p0 = read_pos(quad[0]);
        let p1 = read_pos(quad[1]);
        let p2 = read_pos(quad[2]);
        let p3 = read_pos(quad[3]);
        let d02 = p2 - p0;
        let d13 = p3 - p1;
        let l02 = dot(d02, d02);
        let l13 = dot(d13, d13);
        var alt = l13 < l02;
        if (l02 < 1e-10) {
            alt = true;
        } else if (l13 < 1e-10) {
            alt = false;
        }

        let q = atomicAdd(&counts[params.counts_slot].quads, 1u);
        write_quad(q, quad, s0 < 0.0, alt);
    }
}

fn read_pos(local: u32) -> vec3<f32> {
    let base = (params.base_vertex + local) * VERTEX_WORDS;
    let xy = unpack2x16unorm(vertices[base]);
    let zw = unpack2x16unorm(vertices[base + 1u]);
    return vec3<f32>(xy.x, xy.y, zw.x) * POS_RANGE - POS_BIAS;
}

// Six u16 indices per quad, packed two per u32 word. `first_index` is
// always even (quad-aligned), so the three words never straddle quads.
// `alt` splits across the 1-3 diagonal instead of 0-2.
fn write_quad(quad_index: u32, corners: array<u32, 4>, flip: bool, alt: bool) {
    var i: array<u32, 6>;
    if (flip && alt) {
        i = array<u32, 6>(corners[1], corners[2], corners[3], corners[1], corners[3], corners[0]);
    } else if (flip) {
        i = array<u32, 6>(corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]);
    } else if (alt) {
        i = array<u32, 6>(corners[1], corners[3], corners[2], corners[1], corners[0], corners[3]);
    } else {
        i = array<u32, 6>(corners[0], corners[2], corners[1], corners[0], corners[3], corners[2]);
    }
    let word = (params.first_index + quad_index * 6u) >> 1u;
    indices[word + 0u] = (i[0] & 0xFFFFu) | (i[1] << 16u);
    indices[word + 1u] = (i[2] & 0xFFFFu) | (i[3] << 16u);
    indices[word + 2u] = (i[4] & 0xFFFFu) | (i[5] << 16u);
}
