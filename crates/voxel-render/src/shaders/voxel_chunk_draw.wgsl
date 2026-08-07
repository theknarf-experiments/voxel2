// Chunk draw: camera-relative vertex transform over the slab buffers, with
// fully procedural, fully data-driven surfaces (no texture assets, no
// world-specific branches): the per-vertex material id indexes the level's
// material table, which produces the albedo/emissive fed to Bevy's PBR.
//
// Lighting is BEVY'S: groups 0 and 1 are the mesh view bind group, so
// terrain sees the app's directional/point lights, cascaded shadow maps,
// ambient and environment light, distance fog and tonemapping exactly
// like any `Mesh3d` does. Group 2 is where Bevy puts per-mesh data; ours
// sits there so the material group at 3 stays free.
//
// Vertices are chunk-local; the per-chunk uniform carries the chunk origin
// relative to the camera (computed in f64 on CPU). Multiplying the view
// matrix with w = 0 drops its translation, so the camera effectively sits at
// the origin and world-space f32 error never grows with distance.

#import bevy_pbr::{
    mesh_view_bindings::view,
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    pbr_types,
    pbr_functions,
}

struct ChunkDrawUniform {
    // xyz = chunk minimum corner relative to the camera (m), w = voxel size.
    offset: vec4<f32>,
    // x = number of active clip planes.
    clip_count: vec4<f32>,
    // World-space half-spaces; a fragment survives where it is inside all
    // of them. A portal masks the far world with the pyramid from the eye
    // through its opening, plus the opening's own plane.
    clip: array<vec4<f32>, 5>,
}
@group(2) @binding(0) var<uniform> chunk: ChunkDrawUniform;

// The terrain material at group 3 — the same group index Bevy binds a
// `StandardMaterial` at, bound by the same `SetMaterialBindGroup`.
// Material recipes (128 B each, layout mirrors voxel-render WorldMaterial).
// head.x = kind: 0 = surface (base/grain/bands/grime/streaks/moss/emissive),
// 1 = zoned altitude terrain.
struct WorldMaterial {
    head: vec4<u32>,
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    c3: vec4<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
}
// Bindless: every recipe of this world lives in one slab. The index table
// maps a slab slot to the recipe's offset in the data array; `material` is
// the field at bindless index 0, which is our `#[data(0, ...)]`.
struct VoxelMaterialBindings {
    material: u32,
}
@group(3) @binding(0) var<storage> material_indices: array<VoxelMaterialBindings>;
@group(3) @binding(1) var<storage> material_array: array<WorldMaterial>;

/// The recipe a per-vertex material id selects, via the engine's
/// id → slab-slot map.
fn material_for(id: u32) -> WorldMaterial {
    let i = min(id, 7u);
    let slot = env.material_slots[i / 4u][i % 4u];
    return material_array[material_indices[slot].material];
}

// Engine render flags. Lighting is Bevy's.
struct EnvParams {
    flags: vec4<f32>,                    // x = coverage-eval mode
    material_slots: array<vec4<u32>, 2>, // material id -> bindless slab slot
}
@group(2) @binding(1) var<uniform> env: EnvParams;

struct VsIn {
    // Quantized position: unorm16 x4 mapping [-8, 40] voxels (w unused).
    @location(0) pos: vec4<f32>,
    // Octahedral-encoded normal, snorm16 x2.
    @location(1) oct: vec2<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) cam_rel: vec3<f32>,
    @location(2) shadow: f32,
    // Flat-interpolated material id from the most-solid corner.
    @location(3) @interpolate(flat) material: u32,
}

const POS_BIAS: f32 = 8.0;
const POS_RANGE: f32 = 48.0;

// Where the baked sun shadow takes over from Bevy's cascades (m).
const BAKED_SHADOW_NEAR: f32 = 140.0;
const BAKED_SHADOW_FAR: f32 = 320.0;

fn oct_decode(o: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(o, 1.0 - abs(o.x) - abs(o.y));
    if (n.z < 0.0) {
        let sx = select(-1.0, 1.0, n.x >= 0.0);
        let sy = select(-1.0, 1.0, n.y >= 0.0);
        n = vec3<f32>((1.0 - abs(n.y)) * sx, (1.0 - abs(n.x)) * sy, n.z);
    }
    return normalize(n);
}

@vertex
fn vertex(in: VsIn) -> VsOut {
    let pos_local = (in.pos.xyz * POS_RANGE - POS_BIAS) * chunk.offset.w;
    let cam_rel = pos_local + chunk.offset.xyz;
    let view_space = (view.view_from_world * vec4<f32>(cam_rel, 0.0)).xyz;
    var out: VsOut;
    out.clip = view.clip_from_view * vec4<f32>(view_space, 1.0);
    out.normal = oct_decode(in.oct);
    out.cam_rel = cam_rel;
    // Spare u16: material in the high byte, baked shadow in the low byte.
    let extra = u32(round(in.pos.w * 65535.0));
    out.shadow = f32(extra & 0xFFu) / 255.0;
    out.material = (extra >> 8u) & 0xFFu;
    return out;
}

// --- procedural detail noise -------------------------------------------------

fn hash3(p: vec3<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u + u32(p.z) * 2246822519u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

struct MatSample {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    ao: f32,
}

// --- material recipes --------------------------------------------------------
//
// These are evaluated PER PIXEL, so they are flat lookups and nothing
// else. They used to synthesize their detail procedurally — around
// thirteen 3D value-noise evaluations per fragment across the canopy
// path alone (two `fbm3_2`, two `noise3_grad`, one `noise3`, plus the
// grain/stain/streak/moss fields on the surface path). That is a texture
// fetch's worth of information computed from scratch for every pixel of
// every frame, and it cost roughly a third of the frame to produce
// mottling nobody wanted.
//
// Detail belongs in a texture (one fetch, mip-mapped, filtered, and
// cheaper at distance instead of more expensive). Until there is one,
// the recipes select their own colours by the two things that carry real
// information about the surface — how high it is and how steep it is.

fn surface_albedo(m: WorldMaterial, world: vec3<f32>, n: vec3<f32>, dist: f32) -> vec3<f32> {
    return m.c0.rgb;
}

/// Altitude-zoned natural terrain: low/mid/high/peak, slope overriding to
/// the high (rock) colour.
fn zoned_albedo(m: WorldMaterial, world: vec3<f32>, n: vec3<f32>, dist: f32) -> vec3<f32> {
    var base = mix(m.c0.rgb, m.c1.rgb, smoothstep(m.c0.w, m.c0.w + m.p0.w, world.y));
    base = mix(base, m.c2.rgb, smoothstep(m.c1.w, m.c1.w + m.p1.w, world.y));
    base = mix(base, m.c3.rgb, smoothstep(m.c2.w, m.c2.w + m.p2.x, world.y));
    // 1 on cliffs (the edges are inverted on purpose: n.y falls as it steepens).
    return mix(base, m.c2.rgb, smoothstep(m.p2.y, m.p2.z, n.y));
}

fn canopy_material(m: WorldMaterial, world: vec3<f32>, n: vec3<f32>, dist: f32) -> MatSample {
    let veg_edge = smoothstep(m.c0.w, m.c0.w + m.p0.w, world.y);
    let rockness = max(
        smoothstep(m.c1.w, m.c1.w + m.c2.w, world.y),
        smoothstep(m.p2.x, m.p2.y, n.y),
    );
    // The canopy's two greens averaged: the crown noise that used to pick
    // between them per pixel is gone.
    let canopy = mix(m.c0.rgb, m.c1.rgb, 0.5);
    var out: MatSample;
    out.albedo = mix(mix(m.p0.rgb, canopy, veg_edge), m.c2.rgb, rockness);
    out.normal = n;
    out.ao = 1.0;
    return out;
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    // Coverage-eval mode: monotone geometry against a magenta background,
    // tinted by LOD level so cracks identify the leaking seam pair.
    if (env.flags.x > 0.5) {
        if (in.material == 255u) {
            // Failed parity snap (thin feature) — debug-highlighted.
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
        // Tint per level relative to the finest voxel (0.1 m) so fine
        // levels are distinguishable too.
        let g = 1.0 - clamp(log2(chunk.offset.w / 0.1), 0.0, 12.0) * 0.07;
        return vec4<f32>(g, g, g, 1.0);
    }
    let n = normalize(in.normal);
    let world = vec3<f32>(
        view.world_position.x + in.cam_rel.x,
        view.world_position.y + in.cam_rel.y,
        view.world_position.z + in.cam_rel.z,
    );
    // A world seen through a portal exists only inside the opening. The
    // planes are the pyramid from the eye through it, plus the opening's
    // own plane, so this is the stencil we cannot have — exact, because
    // the opening is convex.
    let clips = u32(chunk.clip_count.x);
    for (var i = 0u; i < clips; i++) {
        let plane = chunk.clip[i];
        if (dot(plane.xyz, world) + plane.w < 0.0) {
            discard;
        }
    }
    let dist = length(in.cam_rel);
    let m = material_for(in.material);

    var base: vec3<f32>;
    var nl = n;
    var ao = 1.0;
    if (m.head.x == 2u) {
        let ms = canopy_material(m, world, n, dist);
        base = ms.albedo;
        nl = ms.normal;
        ao = ms.ao;
    } else if (m.head.x == 1u) {
        base = zoned_albedo(m, world, n, dist);
    } else {
        base = surface_albedo(m, world, n, dist);
    }

    // Emissive ceiling light strips (surface materials).
    var emissive = vec3<f32>(0.0);
    if (m.head.x == 0u && m.c3.w > 0.0) {
        let lf = floor(world.y / m.p1.w);
        let ceilingness = smoothstep(-0.55, -0.85, n.y);
        let line = 1.0 - smoothstep(0.25, 0.75, abs(fract(world.z / m.p1.z) - 0.5) * m.p1.z);
        let works = step(1.0 - m.p2.x, hash3(vec3<i32>(i32(floor(world.z / m.p1.z)), i32(lf), 7)));
        emissive += m.c3.rgb * m.c3.w * ceilingness * line * works;
        // Faint up-glow from the strips onto nearby floors.
        let floorness = smoothstep(0.55, 0.85, n.y);
        emissive += m.c3.rgb * m.p2.y * floorness * line * works;
    }

    var pbr_input = pbr_types::pbr_input_new();
    pbr_input.material.base_color = vec4<f32>(base, 1.0);
    // w is the exposure weight, not alpha: 0 keeps the recipe's emissive
    // in display units (Bevy's own StandardMaterial default), so light
    // strips read the same regardless of the camera's exposure.
    pbr_input.material.emissive = vec4<f32>(emissive, 0.0);
    // Rock, soil and concrete: rough dielectrics. (Per-material control
    // arrives with the material-asset slice.)
    pbr_input.material.perceptual_roughness = 0.95;
    pbr_input.material.metallic = 0.0;
    pbr_input.material.flags = pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    pbr_input.frag_coord = in.clip;
    pbr_input.world_position = vec4<f32>(world, 1.0);
    pbr_input.world_normal = nl;
    pbr_input.N = nl;
    pbr_input.V = normalize(-in.cam_rel);
    pbr_input.diffuse_occlusion = vec3<f32>(ao);
    pbr_input.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;

    var color = pbr_functions::apply_pbr_lighting(pbr_input);

    // Baked horizon sun shadow, faded in past the cascade range: Bevy's
    // shadow maps cover the near field, the mesh-time march covers the
    // kilometres they can't reach. Never both at full strength.
    let baked = mix(1.0, in.shadow, smoothstep(BAKED_SHADOW_NEAR, BAKED_SHADOW_FAR, dist));
    color = vec4<f32>(color.rgb * baked, color.a);

    return pbr_functions::main_pass_post_lighting_processing(pbr_input, color);
}
