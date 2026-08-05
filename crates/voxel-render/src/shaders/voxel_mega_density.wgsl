// Megastructure density: an endless Blame!-style architectural interior
// built from CSG over repeating lattices — floor slabs, pillar grids, wall
// grids with corridor cuts, and giant vertical shafts — with hash-driven
// variation per structural cell. Same bindings/output as the terrain pass.

const SAMPLES: u32 = 38u;
const SLOT_STRIDE: u32 = 54872u; // 38^3

struct ChunkParams {
    origin: vec4<f32>, // xyz = chunk min corner (m), w = voxel size (m)
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

fn hash2(p: vec2<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

fn hash3(p: vec3<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u + u32(p.z) * 2246822519u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

// Signed distance to a box centered at origin with half-extents `b`.
fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

const FLOOR_SPACING: f32 = 44.0;
const PILLAR_SPACING: f32 = 34.0;
const WALL_SPACING: f32 = 104.0;
const SHAFT_SPACING: f32 = 288.0;

fn mega_sdf(p: vec3<f32>, vs: f32) -> f32 {
    // --- megashafts (needed at every LOD) -----------------------------------
    let sc = vec2<i32>(round(p.xz / SHAFT_SPACING));
    let sjit = vec2<f32>(hash2(sc + vec2<i32>(41, 13)) - 0.5, hash2(sc + vec2<i32>(-7, 99)) - 0.5)
        * 90.0;
    let sxz = p.xz - vec2<f32>(sc) * SHAFT_SPACING - sjit;
    let sr = 24.0 + hash2(sc) * 30.0;
    let shaft = length(sxz) - sr;

    // Structural band-limiting: floors/pillars/walls are meter-scale — below
    // the voxel resolution of coarse LODs. From afar the structure reads as
    // a solid mass with voids. A hard cut (not a blend — mixing disagreeing
    // SDFs manufactures phantom surfaces) keeps coarse meshes tiny; the
    // interior fog hides the transition.
    if (vs >= 4.0) {
        return -shaft;
    }

    // --- floors: horizontal slabs every FLOOR_SPACING, 3 m thick ------------
    let level = round(p.y / FLOOR_SPACING);
    let fy = p.y - level * FLOOR_SPACING;
    var d = abs(fy) - 1.5;

    // Floor openings: some 16 m grid cells of each level are cut away.
    let op_cell = vec2<i32>(floor(p.xz / 16.0));
    let op = hash3(vec3<i32>(op_cell.x, i32(level), op_cell.y));
    if (op < 0.16) {
        let oc = (vec2<f32>(op_cell) + 0.5) * 16.0;
        let cut = sd_box(vec3<f32>(p.x - oc.x, fy, p.z - oc.y), vec3<f32>(7.0, 4.0, 7.0));
        d = max(d, -cut);
    }

    // --- pillars: square columns on a jittered grid -------------------------
    let pc = vec2<i32>(round(p.xz / PILLAR_SPACING));
    let jit = vec2<f32>(hash2(pc) - 0.5, hash2(pc + vec2<i32>(311, 77)) - 0.5) * 8.0;
    let pxz = p.xz - vec2<f32>(pc) * PILLAR_SPACING - jit;
    let girth = 1.6 + hash2(pc + vec2<i32>(9, -4)) * 2.2;
    let pillar = max(abs(pxz.x), abs(pxz.y)) - girth;
    d = min(d, pillar);

    // --- walls: sparse room-scale partitions with corridor cuts -------------
    let wxi = round(p.x / WALL_SPACING);
    let wx = p.x - wxi * WALL_SPACING;
    if (hash2(vec2<i32>(i32(wxi), i32(level))) < 0.45) {
        var wall = abs(wx) - 1.2;
        // Corridor openings punched through on a 22 m grid, 2 floors tall.
        let cz = round(p.z / 22.0);
        let czl = p.z - cz * 22.0;
        if (hash3(vec3<i32>(i32(wxi), i32(cz), i32(level))) < 0.5) {
            let doorway = sd_box(vec3<f32>(wx, fy + 12.0, czl), vec3<f32>(4.0, 14.0, 5.0));
            wall = max(wall, -doorway);
        }
        d = min(d, wall);
    }
    let wzi = round(p.z / WALL_SPACING);
    let wz = p.z - wzi * WALL_SPACING;
    if (hash2(vec2<i32>(i32(wzi) + 501, i32(level))) < 0.45) {
        var wall = abs(wz) - 1.2;
        let cx = round(p.x / 22.0);
        let cxl = p.x - cx * 22.0;
        if (hash3(vec3<i32>(i32(wzi), i32(cx), i32(level) + 77)) < 0.5) {
            let doorway = sd_box(vec3<f32>(wz, fy + 12.0, cxl), vec3<f32>(4.0, 14.0, 5.0));
            wall = max(wall, -doorway);
        }
        d = min(d, wall);
    }

    // Carve the megashafts out of the fine structure.
    d = max(d, -shaft);

    // Catwalk beams bridging the shafts along x, every third level.
    if (abs(level - round(level / 3.0) * 3.0) < 0.5) {
        let beam = max(
            max(abs(sxz.y) - 2.2, abs(fy + 1.0) - 0.7),
            length(sxz) - (sr + 6.0),
        );
        d = min(d, beam);
    }

    return d;
}

@compute @workgroup_size(6, 6, 6)
fn density_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(SAMPLES))) {
        return;
    }
    let vs = params.origin.w;
    // Sample i holds cell corner i - 2 (apron covers coarse-parity cells).
    let p = params.origin.xyz + vec3<f32>(vec3<i32>(id) - vec3<i32>(2)) * vs;
    let sdf = clamp(mega_sdf(p, vs) / vs, -4.0, 4.0);
    let material = select(0u, 2u, sdf < 0.0);
    let packed = (pack2x16float(vec2<f32>(sdf, 0.0)) & 0xFFFFu) | (material << 16u);
    let base = params.slot * SLOT_STRIDE;
    density[base + id.x + SAMPLES * (id.y + SAMPLES * id.z)] = packed;
}
