// Procedural ocean: a camera-following grid whose cell size grows with
// distance (power-warped UVs), displaced by a small sum of directional
// waves with analytic normals. Shorelines come from evaluating the coarse
// terrain heightfield at the fragment — shallow tint and a foam band where
// the seabed grazes sea level. Opaque; matches the terrain's sun and haze.

#import bevy_pbr::{
    mesh_view_bindings::{view, globals},
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    pbr_types,
    pbr_functions,
}

struct WaterParams {
    // xz = grid origin (camera-snapped, world meters), y = sea level, w unused.
    origin: vec4<f32>,
    // x = ocean enabled (0/1), y = river segment count, z = world index,
    // w unused.
    counts: vec4<f32>,
}
@group(2) @binding(0) var<uniform> params: WaterParams;

// River water segments (planning-stack WaterSeg), sharing this pipeline
// so rivers get the exact ocean shading: one water look per level.
struct RiverSeg {
    ab: vec4<f32>,     // a.xz | b.xz (world meters)
    geo: vec4<f32>,    // half width | level at a | level at b | unused
    color: vec4<f32>,  // river tint rgb | unused
}
@group(2) @binding(2) var<storage, read> rivers: array<RiverSeg>;

// Generator program (64 B ops, layout mirrors voxel-core WorldOp); the
// shoreline reads the height ops. count = (total, height ops, -, -).
struct WorldOp {
    head: vec4<u32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
}
// Layout twin of `GpuWorldProgram` — the ARRAY LENGTH is part of the
// twin, because `ops` starts after the table and a short table shifts
// every op index. It must equal `voxel_render::MAX_WORLDS`.
//
// Water belongs to ONE world: the host spawns a surface per world and
// passes its index in `params.counts.z`, so this reads that world's ops.
struct WorldHeader {
    count: vec4<u32>,
    sun: vec4<f32>,
}
struct WorldProgram {
    anchor: vec4<f32>,
    field: vec4<f32>,
    worlds: array<WorldHeader, 8>,
    ops: array<WorldOp>,
}
@group(2) @binding(1) var<storage, read> prog: WorldProgram;

fn world_header() -> WorldHeader {
    return prog.worlds[u32(params.counts.z)];
}

const GRID_N: u32 = 192u;       // vertices per side
const RANGE_M: f32 = 30000.0;   // farthest grid reach from center
const SEA_SNAP: f32 = 64.0;     // origin snap so the grid never swims

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_xz: vec2<f32>,
    @location(1) cam_rel: vec3<f32>,
    @location(2) normal: vec3<f32>,
    // River varyings (zero for the ocean): flow direction xz.
    @location(3) flow: vec2<f32>,
    // River tint rgb | river flag (0 = ocean fragment path).
    @location(4) river_color: vec4<f32>,
    // Across-strip coordinate [-1, 1] | half width | unused.
    @location(5) strip: vec3<f32>,
}

// Three directional waves; returns (height, dh/dx, dh/dz).
fn waves(p: vec2<f32>, t: f32) -> vec3<f32> {
    var h = 0.0;
    var dx = 0.0;
    var dz = 0.0;
    let dirs = array<vec2<f32>, 3>(
        normalize(vec2<f32>(1.0, 0.35)),
        normalize(vec2<f32>(-0.55, 1.0)),
        normalize(vec2<f32>(0.2, -1.0)),
    );
    let amps = array<f32, 3>(0.16, 0.11, 0.06);
    let lens = array<f32, 3>(21.0, 9.0, 4.2);
    let spds = array<f32, 3>(3.1, 2.2, 1.4);
    for (var i = 0; i < 3; i++) {
        let k = 6.2831853 / lens[i];
        let phase = dot(dirs[i], p) * k + t * spds[i] * k;
        let s = sin(phase);
        let c = cos(phase);
        h += amps[i] * s;
        dx += amps[i] * k * dirs[i].x * c;
        dz += amps[i] * k * dirs[i].y * c;
    }
    return vec3<f32>(h, dx, dz);
}

fn project(world: vec3<f32>) -> vec4<f32> {
    let cam_rel = world - view.world_position;
    let view_space = (view.view_from_world * vec4<f32>(cam_rel, 0.0)).xyz;
    return view.clip_from_view * vec4<f32>(view_space, 1.0);
}


// One widened quad per river segment: endpoints stretched 0.3 m along the
// flow so consecutive segments never open a sliver, corners offset by the
// half width across it, seated a hair under the emitted water line.
fn river_vertex(rv: u32) -> VsOut {
    let quad = rv / 6u;
    let corner = rv % 6u;
    let seg = rivers[quad];
    let a = seg.ab.xy;
    let b = seg.ab.zw;
    let len = max(distance(a, b), 0.01);
    let dir = (b - a) / len;
    let perp = vec2<f32>(-dir.y, dir.x) * seg.geo.x;

    // Same two-triangle corner pattern as the ocean grid.
    var along = 0.0;
    if (corner == 1u || corner == 2u || corner == 4u) { along = 1.0; }
    var across = -1.0;
    if (corner == 2u || corner == 4u || corner == 5u) { across = 1.0; }

    let base = mix(a - dir * 0.3, b + dir * 0.3, along);
    let world_xz = base + perp * across;
    let level = mix(seg.geo.y, seg.geo.z, along);
    let world = vec3<f32>(world_xz.x, level - 0.05, world_xz.y);

    var out: VsOut;
    out.clip = project(world);
    out.world_xz = world_xz;
    out.cam_rel = world - view.world_position;
    out.normal = vec3<f32>(0.0, 1.0, 0.0);
    out.flow = dir;
    out.river_color = vec4<f32>(seg.color.rgb, 1.0);
    out.strip = vec3<f32>(across, seg.geo.x, 0.0);
    return out;
}

@vertex
fn vertex(@builtin(vertex_index) vid: u32) -> VsOut {
    // Two triangles per cell, generated procedurally from the vertex index.
    let cells = GRID_N - 1u;
    let ocean_count = cells * cells * 6u;
    if (vid >= ocean_count) {
        return river_vertex(vid - ocean_count);
    }
    let quad = vid / 6u;
    let corner = vid % 6u;
    var cx = quad % cells;
    var cz = quad / cells;
    // Two triangles: (0,0)(1,0)(1,1) and (0,0)(1,1)(0,1).
    if (corner == 1u || corner == 2u || corner == 4u) { cx += 1u; }
    if (corner == 2u || corner == 4u || corner == 5u) { cz += 1u; }

    // Power-warped offset: dense cells near the center, huge far out.
    let u = (f32(cx) / f32(cells)) * 2.0 - 1.0;
    let v = (f32(cz) / f32(cells)) * 2.0 - 1.0;
    let wx = sign(u) * pow(abs(u), 2.6) * RANGE_M;
    let wz = sign(v) * pow(abs(v), 2.6) * RANGE_M;
    let world_xz = params.origin.xz + vec2<f32>(wx, wz);

    let w = waves(world_xz, globals.time);
    // Fade displacement out where cells are huge (avoids crawling).
    let center_d = length(vec2<f32>(wx, wz));
    let disp_fade = exp(-center_d * 0.002);
    let y = params.origin.y + w.x * disp_fade;

    let world = vec3<f32>(world_xz.x, y, world_xz.y);
    let cam_rel = world - view.world_position;
    let view_space = (view.view_from_world * vec4<f32>(cam_rel, 0.0)).xyz;

    var out: VsOut;
    out.clip = view.clip_from_view * vec4<f32>(view_space, 1.0);
    // Ocean disabled: collapse the grid to a point (zero-area triangles).
    if (params.counts.x < 0.5) {
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
    out.world_xz = world_xz;
    out.cam_rel = cam_rel;
    out.normal = normalize(vec3<f32>(-w.y * disp_fade, 1.0, -w.z * disp_fade));
    out.flow = vec2<f32>(0.0);
    out.river_color = vec4<f32>(0.0);
    out.strip = vec3<f32>(0.0);
    return out;
}
fn hash2(p: vec2<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u
        + world_header().count.w * 2654435769u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

fn value_noise2(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = hash2(i);
    let b = hash2(i + vec2<i32>(1, 0));
    let c = hash2(i + vec2<i32>(0, 1));
    let d = hash2(i + vec2<i32>(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2(p: vec2<f32>, base_scale: f32, octaves: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = base_scale;
    for (var i = 0; i < octaves; i++) {
        sum += amp * (value_noise2(p * freq) - 0.5);
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum;
}

// Sum of the generator program's height ops at coarse detail (bands under
// ~16 m wavelength fade out — the shoreline doesn't need them).
// Band-limited (~16 m cutoff) FBM with shaping mode, for height-op replay.
fn coarse_fbm(p: vec2<f32>, base_scale: f32, octaves: i32, mode: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = base_scale;
    for (var o = 0; o < octaves; o++) {
        let fade = 1.0;
        let n = value_noise2(p * freq);
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

fn seabed_height(xz: vec2<f32>) -> f32 {
    var h = 0.0;
    var warp = vec2<f32>(0.0);
    // Region axes, filled by WOP_REGION_AXES and read by the band ops.
    var ta = 0.0;
    var tb = 0.0;
    let pxz = xz;
    let header = world_header();
    for (var i = 0u; i < header.count.y; i++) {
        let op = prog.ops[header.count.x + i];
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


// --- coarse terrain height (shoreline) ---------------------------------------


@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);
    let dist = length(in.cam_rel);
    let view_dir = normalize(-in.cam_rel);

    var water: vec3<f32>;
    if (in.river_color.w > 0.5) {
        // River: depth proxy from the across-strip coordinate (the carved
        // bed is deepest mid-stream); foam hugs the banks, drifting
        // downstream with the flow.
        let edge = abs(in.strip.x);
        let depth = (1.0 - edge) * in.strip.y * 1.4;
        let deep = in.river_color.rgb * 0.5;
        let shallow = mix(in.river_color.rgb, vec3<f32>(0.084, 0.112, 0.112), 0.35);
        water = mix(shallow, deep, smoothstep(0.2, 2.6, depth));

        let drift = in.flow * globals.time * 1.6;
        let foam_noise = value_noise2((in.world_xz - drift) * 1.3);
        let foam_band = smoothstep(0.55, 0.95, edge) * (0.45 + 0.55 * foam_noise);
        water = mix(water, vec3<f32>(0.1587, 0.168, 0.168), clamp(foam_band, 0.0, 1.0) * 0.7);

        // Flow ripples: a scrolled noise gradient perturbs the normal
        // (4 taps — per-pixel noise budgets matter at fragment cost).
        let rp = (in.world_xz - drift) * 0.9;
        let e = 0.35;
        let gx = value_noise2(rp + vec2<f32>(e, 0.0)) - value_noise2(rp - vec2<f32>(e, 0.0));
        let gz = value_noise2(rp + vec2<f32>(0.0, e)) - value_noise2(rp - vec2<f32>(0.0, e));
        n = normalize(vec3<f32>(-gx * 0.35, 1.0, -gz * 0.35));
    } else {
        // Ocean: depth-based color from the seabed heightfield.
        let bed = seabed_height(in.world_xz);
        let depth = max(params.origin.y - bed, 0.0);
        let deep = vec3<f32>(0.0075, 0.0243, 0.0411);
        let shallow = vec3<f32>(0.0187, 0.0635, 0.0672);
        water = mix(shallow, deep, smoothstep(1.5, 26.0, depth));

        // Foam where the seabed grazes sea level, animated by wave phase.
        let foam_noise = value_noise2(in.world_xz * 0.7 + globals.time * 0.35);
        let foam_band = smoothstep(2.2, 0.3, depth) * (0.55 + 0.45 * foam_noise);
        water = mix(water, vec3<f32>(0.1587, 0.168, 0.168), clamp(foam_band, 0.0, 1.0) * 0.8);
    }

    // Water is a smooth dielectric: hand the surface to Bevy's PBR and let
    // it do fresnel, the sun's specular highlight, ambient/environment
    // reflection, fog and tonemapping — the same treatment every other
    // surface in the app gets.
    var pbr_input = pbr_types::pbr_input_new();
    pbr_input.material.base_color = vec4<f32>(water, 1.0);
    pbr_input.material.perceptual_roughness = 0.08;
    pbr_input.material.metallic = 0.0;
    pbr_input.material.reflectance = vec3<f32>(0.35);
    pbr_input.material.flags = pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    pbr_input.frag_coord = in.clip;
    pbr_input.world_position = vec4<f32>(
        view.world_position.x + in.cam_rel.x,
        view.world_position.y + in.cam_rel.y,
        view.world_position.z + in.cam_rel.z,
        1.0,
    );
    pbr_input.world_normal = n;
    pbr_input.N = n;
    pbr_input.V = view_dir;
    pbr_input.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;

    let color = pbr_functions::apply_pbr_lighting(pbr_input);
    return pbr_functions::main_pass_post_lighting_processing(pbr_input, color);
}
