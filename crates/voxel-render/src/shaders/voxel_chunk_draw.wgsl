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
    // x = number of active clip planes, y = which world this chunk is in.
    head: vec4<u32>,
    // World-space half-spaces; a fragment survives where it is inside
    // all of them. What they are for is the host's business — showing one
    // world through an opening in another is the case they exist for.
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
/// (world, id) → slab-slot map.
///
/// Per world: a material id is level data, so planet's 1 and the
/// megastructure's 1 are different recipes that have to coexist while a
/// both are loaded at once. Twin of `material_slot_index`.
fn material_for(id: u32) -> WorldMaterial {
    let i = chunk.head.y * 32u + min(id, 31u);
    let slot = env.material_slots[i / 4u][i % 4u];
    return material_array[material_indices[slot].material];
}

// Engine render flags. Lighting is Bevy's.
struct EnvParams {
    flags: vec4<f32>,                    // x = coverage-eval mode
    // (world, material id) -> bindless slab slot, world-major.
    material_slots: array<vec4<u32>, 64>,
}
@group(2) @binding(1) var<uniform> env: EnvParams;

// Surface material map: a header, then one byte per texel, four per word,
// row-major. Size 0 = nothing painted. Layout twin of
// `SurfaceMap::to_words`.
//
// Read PER FRAGMENT, not per vertex. The map is 8 m and the chunks that
// read it are 51-102 m ones — paint only applies where the LOD field has
// gone coarse — so choosing the material at a vertex threw away everything
// finer than the vertex spacing and drew a wood as flat 100 m quads. A
// material id cannot be interpolated, so there is no half-way house: the
// lookup either happens here or the map may as well be a sixteenth of the
// size.
const SURFACE_MAP_THRESHOLDS: u32 = 8u;
// Texel order, generated from `voxel_core::layout` so it cannot disagree
// with the writer. Run `mise run genops` after changing it.
// GENMAT TEXELORDER BEGIN
const TEXEL_TILE: u32 = 8u;
fn surface_texel_index(size: u32, x: u32, z: u32) -> u32 {
    let tiles_per_row = size / TEXEL_TILE;
    let tile = (z / TEXEL_TILE) * tiles_per_row + (x / TEXEL_TILE);
    return tile * TEXEL_TILE * TEXEL_TILE + (z % TEXEL_TILE) * TEXEL_TILE
        + (x % TEXEL_TILE);
}
// GENMAT TEXELORDER END
const SURFACE_MAP_HEADER: u32 = 264u;
@group(2) @binding(2) var<storage, read> surface_map: array<u32>;

/// The painted material at a world position, or 0 for "leave the terrain's
/// own material alone".
///
/// Per world because the map is indexed by world-space xz and says nothing
/// about which world it belongs to. Worlds share coordinates by design, so
/// one global raster painted the near level's rivers onto the far level's
/// ground everywhere.
fn painted_material(world_xz: vec2<f32>) -> u32 {
    // Offset 0 means this world paints nothing: the table itself occupies
    // word 0, so no real section can start there.
    let base = surface_map[chunk.head.y];
    if (base == 0u) {
        return 0u;
    }
    let size = surface_map[base];
    if (size == 0u) {
        return 0u;
    }
    // Nothing painted can apply to a chunk finer than the coarsest thing
    // the map holds, and that covers the NEAREST chunks — which is most of
    // the screen. Tested before the texel read, which is a random access
    // into 16 MB and the expensive half: the whole lookup measured 1.1 ms
    // a frame, and near chunks were paying all of it to be told no.
    //
    // `min_voxel_m` is the floor over every material; the per-material
    // threshold is still checked below, once a texel has named one.
    if (chunk.offset.w < bitcast<f32>(surface_map[base + 4u])) {
        return 0u;
    }
    let texel_m = bitcast<f32>(surface_map[base + 1u]);
    let origin = vec2<f32>(bitcast<f32>(surface_map[base + 2u]),
                           bitcast<f32>(surface_map[base + 3u]));
    let t = floor((world_xz - origin) / texel_m);
    if (t.x < 0.0 || t.y < 0.0 || u32(t.x) >= size || u32(t.y) >= size) {
        return 0u;
    }
    let idx = surface_texel_index(size, u32(t.x), u32(t.y));
    let painted = (surface_map[base + SURFACE_MAP_HEADER + idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
    // The scale the paint takes over at is PER MATERIAL, because it is a
    // property of the thing painted, not of the map: a road's carve stops
    // resolving within 100 m, while a water course has a carved bed AND a
    // surface drawn over it out to a distance its own layer sets. One
    // threshold for both drew the river twice — the real surface and a
    // painted band around it — everywhere the two ranges overlapped.
    if (painted == 0u
        || chunk.offset.w < bitcast<f32>(surface_map[base + SURFACE_MAP_THRESHOLDS + painted])) {
        return 0u;
    }
    return painted;
}

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

// Which component of which vec4 each named parameter lives in is
// generated from `voxel_core::layout::MATERIALS` — the same table the
// engine's packer writes through, so a parameter cannot move on one side
// only. Run `mise run genops` after editing it.
// GENMAT ACCESSORS BEGIN
// --- surface (head.x == 0u) ---
const MAT_KIND_SURFACE: u32 = 0u;
fn surface_base(m: WorldMaterial) -> vec3<f32> { return m.c0.rgb; }
fn surface_grain(m: WorldMaterial) -> f32 { return m.c0.w; }
fn surface_grime_tint(m: WorldMaterial) -> vec3<f32> { return m.c1.rgb; }
fn surface_grime_amount(m: WorldMaterial) -> f32 { return m.c1.w; }
fn surface_moss_color(m: WorldMaterial) -> vec3<f32> { return m.c2.rgb; }
fn surface_moss_amount(m: WorldMaterial) -> f32 { return m.c2.w; }
fn surface_emissive_color(m: WorldMaterial) -> vec3<f32> { return m.c3.rgb; }
fn surface_emissive_intensity(m: WorldMaterial) -> f32 { return m.c3.w; }
fn surface_band_freq(m: WorldMaterial) -> f32 { return m.p0.x; }
fn surface_band_amp(m: WorldMaterial) -> f32 { return m.p0.y; }
fn surface_band_lo(m: WorldMaterial) -> f32 { return m.p0.z; }
fn surface_band_hi(m: WorldMaterial) -> f32 { return m.p0.w; }
fn surface_band_warp(m: WorldMaterial) -> f32 { return m.p1.x; }
fn surface_streaks(m: WorldMaterial) -> f32 { return m.p1.y; }
fn surface_strip_spacing(m: WorldMaterial) -> f32 { return m.p1.z; }
fn surface_strip_level_spacing(m: WorldMaterial) -> f32 { return m.p1.w; }
fn surface_strip_chance(m: WorldMaterial) -> f32 { return m.p2.x; }
fn surface_strip_glow(m: WorldMaterial) -> f32 { return m.p2.y; }
fn surface_detail_fade(m: WorldMaterial) -> f32 { return m.p2.z; }
// --- zoned (head.x == 1u) ---
const MAT_KIND_ZONED: u32 = 1u;
fn zoned_low(m: WorldMaterial) -> vec3<f32> { return m.c0.rgb; }
fn zoned_mid_start(m: WorldMaterial) -> f32 { return m.c0.w; }
fn zoned_mid_a(m: WorldMaterial) -> vec3<f32> { return m.c1.rgb; }
fn zoned_high_start(m: WorldMaterial) -> f32 { return m.c1.w; }
fn zoned_high_a(m: WorldMaterial) -> vec3<f32> { return m.c2.rgb; }
fn zoned_peak_start(m: WorldMaterial) -> f32 { return m.c2.w; }
fn zoned_peak(m: WorldMaterial) -> vec3<f32> { return m.c3.rgb; }
fn zoned_border(m: WorldMaterial) -> f32 { return m.c3.w; }
fn zoned_mid_b(m: WorldMaterial) -> vec3<f32> { return m.p0.rgb; }
fn zoned_mid_width(m: WorldMaterial) -> f32 { return m.p0.w; }
fn zoned_high_b(m: WorldMaterial) -> vec3<f32> { return m.p1.rgb; }
fn zoned_high_width(m: WorldMaterial) -> f32 { return m.p1.w; }
fn zoned_peak_width(m: WorldMaterial) -> f32 { return m.p2.x; }
fn zoned_steep_hi(m: WorldMaterial) -> f32 { return m.p2.y; }
fn zoned_steep_lo(m: WorldMaterial) -> f32 { return m.p2.z; }
fn zoned_detail_fade(m: WorldMaterial) -> f32 { return m.p2.w; }
// --- canopy (head.x == 2u) ---
const MAT_KIND_CANOPY: u32 = 2u;
fn canopy_canopy_a(m: WorldMaterial) -> vec3<f32> { return m.c0.rgb; }
fn canopy_canopy_start(m: WorldMaterial) -> f32 { return m.c0.w; }
fn canopy_canopy_b(m: WorldMaterial) -> vec3<f32> { return m.c1.rgb; }
fn canopy_rock_start(m: WorldMaterial) -> f32 { return m.c1.w; }
fn canopy_rock(m: WorldMaterial) -> vec3<f32> { return m.c2.rgb; }
fn canopy_rock_width(m: WorldMaterial) -> f32 { return m.c2.w; }
fn canopy_patch(m: WorldMaterial) -> vec3<f32> { return m.c3.rgb; }
fn canopy_border(m: WorldMaterial) -> f32 { return m.c3.w; }
fn canopy_low(m: WorldMaterial) -> vec3<f32> { return m.p0.rgb; }
fn canopy_canopy_width(m: WorldMaterial) -> f32 { return m.p0.w; }
fn canopy_crown_scale(m: WorldMaterial) -> f32 { return m.p1.x; }
fn canopy_crown_relief(m: WorldMaterial) -> f32 { return m.p1.y; }
fn canopy_strata_scale(m: WorldMaterial) -> f32 { return m.p1.z; }
fn canopy_strata_relief(m: WorldMaterial) -> f32 { return m.p1.w; }
fn canopy_steep_hi(m: WorldMaterial) -> f32 { return m.p2.x; }
fn canopy_steep_lo(m: WorldMaterial) -> f32 { return m.p2.y; }
fn canopy_detail_fade(m: WorldMaterial) -> f32 { return m.p2.z; }
fn canopy_patch_amount(m: WorldMaterial) -> f32 { return m.p2.w; }
// GENMAT ACCESSORS END

fn surface_albedo(m: WorldMaterial, world: vec3<f32>, n: vec3<f32>, dist: f32) -> vec3<f32> {
    return surface_base(m);
}

/// Altitude-zoned natural terrain: low/mid/high/peak, slope overriding to
/// the high (rock) colour.
fn zoned_albedo(m: WorldMaterial, world: vec3<f32>, n: vec3<f32>, dist: f32) -> vec3<f32> {
    let mid_start = zoned_mid_start(m);
    let high_start = zoned_high_start(m);
    let peak_start = zoned_peak_start(m);
    let rock = zoned_high_a(m);
    var base = mix(zoned_low(m), zoned_mid_a(m), smoothstep(mid_start, mid_start + zoned_mid_width(m), world.y));
    base = mix(base, rock, smoothstep(high_start, high_start + zoned_high_width(m), world.y));
    base = mix(base, zoned_peak(m), smoothstep(peak_start, peak_start + zoned_peak_width(m), world.y));
    // 1 on cliffs (the edges are inverted on purpose: n.y falls as it steepens).
    return mix(base, rock, smoothstep(zoned_steep_hi(m), zoned_steep_lo(m), n.y));
}

fn canopy_material(m: WorldMaterial, world: vec3<f32>, n: vec3<f32>, dist: f32) -> MatSample {
    let canopy_start = canopy_canopy_start(m);
    let rock_start = canopy_rock_start(m);
    let veg_edge = smoothstep(canopy_start, canopy_start + canopy_canopy_width(m), world.y);
    let rockness = max(
        smoothstep(rock_start, rock_start + canopy_rock_width(m), world.y),
        smoothstep(canopy_steep_hi(m), canopy_steep_lo(m), n.y),
    );
    // The canopy's two greens averaged: the crown noise that used to pick
    // between them per pixel is gone.
    let canopy = mix(canopy_canopy_a(m), canopy_canopy_b(m), 0.5);
    var out: MatSample;
    out.albedo = mix(mix(canopy_low(m), canopy, veg_edge), canopy_rock(m), rockness);
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
    // The stencil we cannot have (no stencil aspect on the depth
    // texture), and exact for planar boundaries.
    let clips = chunk.head.x;
    for (var i = 0u; i < clips; i++) {
        let plane = chunk.clip[i];
        if (dot(plane.xyz, world) + plane.w < 0.0) {
            discard;
        }
    }
    let dist = length(in.cam_rel);
    // Up-facing only, because the map is a plan view: a road crossing a
    // cliff paints the ledge it runs along, not the rock face beside it.
    var id = in.material;
    if (n.y > 0.5) {
        let painted = painted_material(world.xz);
        if (painted != 0u) {
            id = painted;
        }
    }
    let m = material_for(id);

    var base: vec3<f32>;
    var nl = n;
    var ao = 1.0;
    if (m.head.x == MAT_KIND_CANOPY) {
        let ms = canopy_material(m, world, n, dist);
        base = ms.albedo;
        nl = ms.normal;
        ao = ms.ao;
    } else if (m.head.x == MAT_KIND_ZONED) {
        base = zoned_albedo(m, world, n, dist);
    } else {
        base = surface_albedo(m, world, n, dist);
    }

    // Emissive ceiling light strips (surface materials).
    var emissive = vec3<f32>(0.0);
    if (m.head.x == MAT_KIND_SURFACE && surface_emissive_intensity(m) > 0.0) {
        let spacing = surface_strip_spacing(m);
        let chance = surface_strip_chance(m);
        let lf = floor(world.y / surface_strip_level_spacing(m));
        let ceilingness = smoothstep(-0.55, -0.85, n.y);
        let line = 1.0 - smoothstep(0.25, 0.75, abs(fract(world.z / spacing) - 0.5) * spacing);
        let works = step(1.0 - chance, hash3(vec3<i32>(i32(floor(world.z / spacing)), i32(lf), 7)));
        // Converge to the pattern's MEAN once a period stops spanning a
        // pixel, which is what a mip chain would do if this were a
        // texture. Point-sampling it instead turns a grid of lamps seen
        // across a kilometre of floor into moiré arcs that read as
        // scratches — and since `surface_albedo` became a flat colour,
        // this is the only procedural detail left to alias.
        //
        // Faded to the mean rather than to zero: a lit district has to
        // still look lit from the far side of the hall, or the light is
        // exactly the identity it was giving the district away by.
        // Against the STRIP's width, not the period: a strip is about a
        // metre of every thirteen to sixty, so it stops spanning a pixel
        // long before the pattern's period does, and fading on the
        // period leaves the aliasing untouched at every distance you can
        // actually see it at.
        let resolved = 1.0 - smoothstep(0.35, 1.4, fwidth(world.z));
        // A strip is ~1 m of every `spacing`, and `works` is on with
        // probability `chance` — so those are the two means.
        let lit = mix(0.5 / spacing, line, resolved) * mix(chance, works, resolved);
        emissive += surface_emissive_color(m) * surface_emissive_intensity(m) * ceilingness * lit;
        // Faint up-glow from the strips onto nearby floors.
        let floorness = smoothstep(0.55, 0.85, n.y);
        emissive += surface_emissive_color(m) * surface_strip_glow(m) * floorness * lit;
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
