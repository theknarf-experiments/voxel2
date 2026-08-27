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
    // Layout twin of `write_vertex` in voxel_mesh_chunks.wgsl: octahedral
    // normal at 12 bits a component, baked occlusion in the two low
    // nibbles. Fetched raw (Uint32) because a normalized vertex format
    // would decode the word before the shader could split it.
    @location(1) packed: u32,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) cam_rel: vec3<f32>,
    @location(2) shadow: f32,
    // 1 where that shadow came from the EXACT half of the bake (the
    // massing) rather than the coarse half (the horizon march). It
    // decides whether the near-field fade below applies at all.
    @location(4) shadow_exact: f32,
    // Flat-interpolated material id from the most-solid corner.
    @location(3) @interpolate(flat) material: u32,
    // Baked crease occlusion, exactly 0 on any locally planar field
    // whatever its gradient. Read by `surface_material` and by nothing
    // else — it is a field of that recipe and takes its `detail_fade`.
    @location(5) occlusion: f32,
}

const POS_BIAS: f32 = 8.0;
const POS_RANGE: f32 = 48.0;

// Where the baked sun shadow takes over from Bevy's cascades (m).
//
// It used to start at 140 m because the only thing baked was a march
// over a COARSE heightfield, which disagrees with the geometry it is
// shading by metres and had to be kept out of the near field. The
// massing half of the bake is a closed-form ray test against boxes and
// agrees exactly, and it is the half that matters for a building — so
// the handover moved in to where the cascades actually stop being the
// better answer.
const BAKED_SHADOW_NEAR: f32 = 140.0;
const BAKED_SHADOW_FAR: f32 = 320.0;

// What `baked_occlusion` reports at a right-angled crease, MEASURED
// against the estimator rather than reasoned from what it ought to mean:
// the peak over 300 random sub-voxel placements of a 90-degree fold is
// 0.248 at the low end, 0.280 at the median and 0.341 at p90 (0.435 in
// the degenerate case where the crease line runs exactly through lattice
// corners). It is not "half the hemisphere" and never was — the estimator
// is a one-voxel band, not a visibility test, so this constant has no
// derivation, only a measurement.
//
// Anything tighter or more enclosed clamps, which is deliberate: the
// dihedral sweep runs 0.094 at 150 degrees, 0.185 at 120, 0.273 at 90,
// 0.396 at 60 and 0.541 at 30, and a wall standing on a slab reads 0.578
// at the junction. The gradient that survives the clamp is the SPATIAL
// one — that same wall reads 0.578 / 0.347 / 0.088 / 0 at 0.5, 1.5, 2.5
// and 3.5 voxels out along the floor, which is the contact shadow.
const AO_CORNER: f32 = 0.28;
// How much of its ambient a full crease gives up. A starting point, not a
// measurement — it is one constant for four levels and has to be chosen
// by looking at all of them. Setting it to 0 turns the feature off with
// no remesh, which is why the raw occlusion is what gets baked.
const AO_STRENGTH: f32 = 0.5;

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
    // `(w2 << 16u) & 0xFFF00000u` lifts the low u16's 12 value bits to
    // 31..20 so the signed shift sign-extends them; the high u16's
    // already sit there.
    let w2 = in.packed;
    let qx = bitcast<i32>((w2 << 16u) & 0xFFF00000u) >> 20u;
    let qy = bitcast<i32>(w2 & 0xFFF00000u) >> 20u;
    out.normal = oct_decode(vec2<f32>(max(f32(qx) / 2047.0, -1.0),
                                      max(f32(qy) / 2047.0, -1.0)));
    out.occlusion = f32((w2 & 0xFu) | (((w2 >> 16u) & 0xFu) << 4u)) / 255.0;
    out.cam_rel = cam_rel;
    // Spare u16: material in the high byte, baked shadow in the low —
    // seven bits of it, with the top bit saying which half of the bake
    // the value came from. A seventh of a percent per step is finer than
    // a soft shadow's own gradient, and the bit is worth more than the
    // step.
    let extra = u32(round(in.pos.w * 65535.0));
    out.shadow = f32(extra & 0x7Fu) / 127.0;
    out.shadow_exact = f32((extra >> 7u) & 1u);
    out.material = (extra >> 8u) & 0xFFu;
    return out;
}

// --- procedural detail noise -------------------------------------------------

// Salts in use: 3 and 4 (grain octaves), 7 (emissive strip), 11 and 12
// (pour mottle), 19 (streaks), 31 (course tone). Reusing one correlates
// two fields that are meant to be independent.
fn hash3(p: vec3<i32>) -> f32 {
    var h: u32 = u32(p.x) * 374761393u + u32(p.y) * 668265263u + u32(p.z) * 2246822519u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

/// 1 while a world-space feature of size `size` still spans more than a
/// pixel or two, 0 once it cannot be resolved and must be replaced by its
/// own mean. `fw` is `fwidth` of the coordinate the feature varies along.
///
/// Size a field against its FINEST feature, never against its period: a
/// field gated on its coarse octave goes on perturbing pixels with a fine
/// octave that stopped resolving an octave earlier, and a hard step
/// aliases at any distance whatever its period is.
///
/// The constants are the emissive strip's, which was written against a
/// ~1 m strip; the strip calls this with `size = 1.0` and is unchanged.
fn resolved(size: f32, fw: f32) -> f32 {
    return 1.0 - smoothstep(size * 0.35, size * 1.4, fw);
}

/// 2D value noise, mean 0.5. Safe on plain `floor`/`fract` only while
/// `p` stays small enough that the fraction keeps its precision — see
/// `lattice` for the lattice that does not.
fn vnoise2(p: vec2<f32>, salt: i32) -> f32 {
    let fp = floor(p);
    let cell = vec2<i32>(fp);
    let f = p - fp;
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash3(vec3<i32>(cell.x,     cell.y,     salt));
    let b = hash3(vec3<i32>(cell.x + 1, cell.y,     salt));
    let c = hash3(vec3<i32>(cell.x,     cell.y + 1, salt));
    let d = hash3(vec3<i32>(cell.x + 1, cell.y + 1, salt));
    return a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
}

/// Value and analytic gradient — `(v, dv/df.x, dv/df.y)` — from the SAME
/// four hashes the value needs. A finite difference would have cost three
/// more lookups; the gradient is what stops concrete reading as clay.
fn vnoise2_cell_d(cell: vec2<i32>, f: vec2<f32>, salt: i32) -> vec3<f32> {
    let u = f * f * (3.0 - 2.0 * f);
    let du = 6.0 * f * (1.0 - f);
    let a = hash3(vec3<i32>(cell.x,     cell.y,     salt));
    let b = hash3(vec3<i32>(cell.x + 1, cell.y,     salt));
    let c = hash3(vec3<i32>(cell.x,     cell.y + 1, salt));
    let d = hash3(vec3<i32>(cell.x + 1, cell.y + 1, salt));
    let k1 = b - a;
    let k2 = c - a;
    let k3 = a - b - c + d;
    return vec3<f32>(
        a + k1 * u.x + k2 * u.y + k3 * u.x * u.y,
        du.x * (k1 + k3 * u.y),
        du.y * (k2 + k3 * u.x),
    );
}

/// Exact cell index and fraction for a high-frequency lattice: `(cell,
/// fraction)`.
///
/// Needed because the caller is on a 4 cm lattice and planet reaches
/// 37 km: `floor(world * 24.0)` is 9e5 in f32, where one ulp is a tenth
/// of a cell, so the fraction quantizes to about ten steps and the grain
/// degenerates into flat blocks. The fraction is therefore taken off the
/// CAMERA-RELATIVE position, which stays small, plus the camera's own
/// sub-cell residual — dropping that residual instead would lock the
/// pattern to the camera and slide it a whole cell for every cell of
/// camera travel. What is left over is the f32 ulp of `cam_world * freq`:
/// <= 0.06 of a cell (~2 mm) at 37 km, far below a pixel at any range
/// where grain is still resolved at all.
fn lattice(cam_rel: vec2<f32>, cam_world: vec2<f32>, freq: f32) -> vec4<f32> {
    let c = round(cam_world * freq);
    let q = cam_rel * freq + (cam_world * freq - c);
    let fq = floor(q);
    return vec4<f32>(fq + c, q - fq);
}

/// A symmetric joint on a unit lattice: `(value, tilt)`, 0 in the joint
/// and 1 mid-panel, chamfer running from `lo` to `hi` as a fraction of
/// the half-period. The period is unchanged at 1 and the groove is
/// centred on `fract(x) == 0.5`; the mean over a period is exactly
/// `1 - (lo + hi)/2`, which is what the caller converges to.
///
/// `tilt` is the chamfer-NORMALIZED slope — `w/3 * d value / d x` — odd
/// about the groove, so its mean is 0 and its peak is exactly 1 at the
/// middle of a chamfer. The true slope peaks at `3/w`, which would make
/// a groove's depth the author's chamfer WIDTH in disguise: planet's
/// `w = 0.17` would be shaded 2.4x deeper than brutalist's `w = 0.40`
/// off a lower authored amplitude, and narrowing a chamfer — which says
/// the joint is sharper — would say it is deeper as well. Normalizing
/// here is also what makes `max(hi - lo, 1e-4)` guard the derivative and
/// not only the value: at `lo == hi`, one drag apart in the live editor,
/// the true slope reaches 923 and tips the shading normal 89.9 degrees
/// into the tangent plane on whatever pixels land in the chamfer.
///
/// Symmetric on purpose. The deleted version used a raw `fract`: one edge
/// of every board was a cliff and the other a ramp, so it read as a
/// painted gradient rather than a shutter line.
fn notch(x: f32, lo: f32, hi: f32) -> vec2<f32> {
    let w = max(hi - lo, 1e-4);
    let s = fract(x) - 0.5;
    let d = abs(s) * 2.0;
    let t = clamp((d - lo) / w, 0.0, 1.0);
    return vec2<f32>(t * t * (3.0 - 2.0 * t), 4.0 * t * (1.0 - t) * sign(s));
}

/// A downward wash on a unit lattice: strongest just below a line,
/// decaying to nothing at the line below, returning over `ch` of a period
/// instead of dropping off a cliff.
///
/// C1 everywhere, which is the whole point — `fract` carries a
/// full-amplitude step at every line, and a step aliases at every
/// distance. `ch` is the only feature the caller has to resolve. Its
/// argument is uniform over a period, so the mean is exactly 0.5 and the
/// mean of its square exactly 13/35.
fn wash(x: f32, ch: f32) -> f32 {
    let s = fract(x);
    let t = min(s / (1.0 - ch), (1.0 - s) / ch);
    return t * t * (3.0 - 2.0 * t);
}

struct MatSample {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    ao: f32,
    rough: f32,
}

// --- material recipes --------------------------------------------------------
//
// These are evaluated PER PIXEL. `zoned` and `canopy` are flat lookups
// and nothing else: they used to synthesize detail procedurally — around
// thirteen 3D value-noise evaluations per fragment on the canopy path
// alone — and that cost roughly a third of the frame to produce mottling
// nobody wanted. They still select their colours by the two things that
// carry real information about natural terrain: how high it is and how
// steep it is.
//
// `surface` is the exception, because a cast concrete wall has no
// altitude and no slope to select by, and a wall with neither reads as an
// untextured clay render at eye level. It synthesizes detail again, but
// under the two gates the deleted version had no notion of — see THE RULE
// on `surface_material`.

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

// Rock, soil and concrete: rough dielectrics. The surface recipe moves
// off this by its own detail, nothing else does.
const SURFACE_ROUGHNESS: f32 = 0.95;

// Aggregate and pinhole grain: the scale at which concrete stops being a
// colour, and the only field here that carries a micro-normal. `resolved`
// turns octave 0 off past ~35 m and octave 1 past ~17 m, where a 4.2 cm
// and a 2.0 cm feature stop spanning a pixel (fw ~= 0.0016 * dist at 60
// degrees / 1080p).
const GRAIN_FREQ: f32 = 24.0;      // cycles/m -> 4.2 cm
const GRAIN_OCT: f32 = 2.07;       // off the octave so the two never phase-lock
const GRAIN_W0: f32 = 0.667;
const GRAIN_W1: f32 = 0.333;
const GRAIN_ALBEDO: f32 = 0.10;
const GRAIN_RELIEF: f32 = 0.06;
const GRAIN_ROUGH: f32 = 0.06;

// Pour mottle: the metre-scale unevenness of a cast face. ONE field feeds
// grime, moss AND the formwork warp, because they are the same weathering
// and three fields would cost three times. GAIN normalizes two-octave 2D
// value noise (sd 0.16 about 0.5) to sd 0.49 on [-1, 1].
const MOTTLE_FREQ: f32 = 0.16;     // -> 6.3 m
const MOTTLE_OCT: f32 = 2.13;
const MOTTLE_W0: f32 = 0.667;
const MOTTLE_W1: f32 = 0.333;
const MOTTLE_GAIN: f32 = 3.1;
const GRIME_MAX: f32 = 0.55;
// The mottle's steepest possible slope, per metre: a value-noise cell's
// smoothstep interpolant peaks at 1.5 and its span is at most 1, so this
// is GAIN * 1.5 * FREQ * (W0 + OCT * W1). Worst case and not RMS — the
// field's mean slope is about a seventh of this, and a warp bounded on
// the mean still folds exactly where the field is steep.
const MOTTLE_SLOPE: f32 = 1.03;

// Drip streaks: one octave stretched 20:1, features ~2 m wide and ~40 m
// long. The deleted version spent three 3D octaves on a field that was
// near-constant in y anyway.
const STREAK_FREQ: f32 = 0.5;
const STREAK_ASPECT: f32 = 0.05;
const STREAK_GAIN: f32 = 2.6;
const STREAK_MAX: f32 = 0.5;
const STREAK_ROUGH: f32 = 0.18;
const STREAK_RUN_FREQ: f32 = 0.18;       // storey lines, where no band supplies them
const STREAK_CHAMFER: f32 = 0.25;        // of a run; `wash`'s only resolvable feature
const STREAK_RUN_MEAN2: f32 = 0.3714286; // 13/35, the mean of `wash` squared

// Formwork. A cast deck is jointed on a grid wider than the boards on a
// wall are tall. A material carries no notion of what it was cast as, so
// this is every up-facing face of every banded material — planet's roads
// included; a level that wants no joints on its flat faces authors no
// band.
const FLOOR_JOINT_RATIO: f32 = 0.35;
// Peak tilt of a joint at band amp 1. `notch`'s tilt is normalized to
// peak 1, so a groove's depth is this times the authored amp times
// `band_r` and nothing else: 0.092 at brutalist's amp 0.22, against the
// grain's peak of 0.099, and 0.18 at purgatory's 0.42, which is the
// amplitude that material authors.
const BAND_RELIEF: f32 = 0.42;
// A warp is a PHASE, and a phase whose slope can cancel the lattice's own
// turns a band into a closed contour: the argument stops being monotone,
// the band becomes a blob, and the relief goes on lighting it as though
// it were still a straight groove. Capping the warp's slope at this
// fraction of the band's own keeps the argument strictly increasing, so a
// joint is always a single line. It also bounds the LOCAL period at two
// thirds of the nominal, which is the factor by which `band_r` — sized on
// the nominal — can be late.
const WARP_SLOPE_MAX: f32 = 0.5;
const BOARD_TONE: f32 = 0.06;

/// Cast concrete: base colour, pour weathering, formwork and aggregate
/// grain, from the nine parameters the surface recipe already carries.
///
/// THE RULE, which every term below obeys: each field is normalized so
/// that its own average is exactly 1 (multiplicative) or 0 (additive),
/// and each is multiplied by `fade` and by `resolved` against its OWN
/// finest feature. So a fragment that is far away, or whose fields cannot
/// be resolved, returns exactly `surface_base(m)` — the flat colour this
/// function returned when it had no detail at all. That is what makes the
/// change safe for planet, megastructure and purgatory by construction,
/// and it is the same statement as "unresolvable detail converges to its
/// mean", which is the anti-aliasing guarantee. Moss is the one
/// deliberate exception: it fades out, not to its mean.
///
/// The two gates do different jobs and are not interchangeable. `fade` is
/// the ART budget the level authors and the shader used to ignore;
/// `resolved` is the ALIASING guarantee and is per field. Neither may be
/// applied to a PHASE — see the band's warp.
///
/// Cost, per fragment, on the surface path only (`zoned` and `canopy` do
/// not call this). These are ALU counts read off the source and NOT a
/// measurement — `hash3` is ~12 ALU, so a 4-tap lookup is ~55-60 and one
/// with its analytic gradient ~65. Nothing below has been profiled on a
/// GPU, and fill rate is this project's frame budget, so treat the whole
/// table as an estimate to be checked. The baseline it is against is
/// zero: brutalist and megastructure are 100% surface, so all of this is
/// new cost on every pixel they draw.
///   past the fade horizon, `ln(50) / detail_fade` — 978 m at the 0.004
///     default, 1304 m at brutalist's 0.003, 1957 m at planet's 0.002
///     and 2608 m at its 0.0015 — one `exp` and a compare.
///   inside it: mottle ~120 wherever grime, moss or a warp is authored.
///     All ten of megastructure's materials author grime 1.0, and
///     planet's material 3 authors moss and a warp — and material 3 is
///     the path ribbons and slabs as well as the ruins, so both levels
///     pay it on effectively every surface pixel out to their own
///     horizon.
///   + band and course tone ~35, streaks ~70, where authored.
///   inside ~35 m: + grain ~130 for two gradient octaves, the octave-1
///     half of which stops at ~17 m. `grain` defaults to 0.35, so a
///     material pays this unless it authors 0 — megastructure's ten all
///     take the default.
/// First lever if it measures hot, in order: drop the mottle to one
/// octave, then the grain to one, then GRAIN_FREQ to 16 (which shrinks
/// the grain radius to ~23 m).
fn surface_material(m: WorldMaterial, world: vec3<f32>, cam_rel: vec3<f32>,
                    cam_world: vec3<f32>, n: vec3<f32>, dist: f32,
                    fw: vec3<f32>, occlusion: f32) -> MatSample {
    var out: MatSample;
    out.albedo = surface_base(m);
    out.normal = n;
    out.ao = 1.0;
    out.rough = SURFACE_ROUGHNESS;

    // The author's own distance budget, and the early-out that keeps this
    // whole block off a terrain horizon, where most of a frame's
    // fragments are. It is a compare on `dist`, so it is coherent across
    // a wave. The cutoff is taken to zero rather than stepped off at 2%:
    // a step leaves a ring of changed albedo at a fixed radius from the
    // camera — 978 m at the 0.004 default, 1304 m at brutalist's 0.003 —
    // and a distant view that changes at all is the one thing three
    // shipped levels cannot afford here.
    let budget = exp(-dist * surface_detail_fade(m));
    if (budget < 0.02) {
        return out;
    }
    let fade = budget * smoothstep(0.02, 0.05, budget);

    // Baked crease occlusion, and it is a field of this recipe like any
    // other: multiplied by `fade`, so it converges to 1 — no occlusion,
    // today's output — past the author's own horizon and through the same
    // early-out above.
    //
    // The gate is why it lives here and not in the fragment. This term is
    // baked per VERTEX, so its darkened band is one to two voxels wide
    // and its world width grows with the LOD level: the estimator's own
    // mean rises with voxel size on natural relief (measured on planet's
    // terrain: p50 0.003 at 0.1 m voxels, 0.031 at 1.6 m, 0.055 at 6.4 m).
    // On the `zoned` and `canopy` paths there is no authored distance
    // budget to bound that with, so what those materials would get is a
    // wash on open ground whose strength changes as you walk toward it —
    // the same defect the deleted grime field was removed for. They do
    // not take the term at all, which is also why planet's terrain and
    // forest cannot move. Cast surfaces do: a crease in concrete is what
    // the term was added for, and `detail_fade` is the lever that bounds
    // it.
    //
    // It darkens INDIRECT light only — `apply_pbr_lighting` reads
    // `diffuse_occlusion` for the ambient, environment and irradiance
    // terms and nothing else. A crease in direct sun keeps its sun; a
    // crease in shade, where the ambient IS the light, goes dark.
    // Multiplying the final colour is what the baked sun shadow does, and
    // doing it here as well would flatten every lit face.
    out.ao = 1.0 - AO_STRENGTH * fade * saturate(occlusion / AO_CORNER);

    // ONE projection, and the axis pick is `abs(n).y > 0.5` rather than
    // "whichever axis dominates". The two are the same test for a face
    // that is nearly flat — y can only dominate at an.y >= 0.577 — but
    // they differ for a slope, and there the wider one is the correct
    // one: `painted_material` applies wherever `n.y > 0.5`, and n.y =
    // 0.51 admits |n.x| up to 0.86, so a dominant-axis pick would flip
    // partway down a smooth hillside and put a seam in albedo AND in the
    // perturbed normal along a curve on a continuous surface. Sharing the
    // paint's own threshold means a painted face never flips inside
    // itself, and on formwork the faces are axis-aligned, so the pick
    // flips exactly on a geometric edge where a discontinuity is
    // invisible. It is not a guarantee for every surface: a material_band
    // op paints nothing and applies a recipe to a whole terrain region, so
    // a 60-degree hillside crosses this line with no edge under it. What
    // changes across the flip is the PHASE of the mottle and the grain —
    // invisible, since no field's mean depends on the projection and the
    // grain is 4 cm noise — and, for a banded material, wall courses
    // becoming a deck grid, which is not. No shipped level bands a region
    // rather than a structure.
    //
    // Three projections blended is the general answer and costs three
    // times the noise; a 45-degree cast face is the price paid.
    let an = abs(n);
    let flat_face = an.y > 0.5;
    var uv: vec2<f32>;
    var uv_rel: vec2<f32>;
    var uv_cam: vec2<f32>;
    var t_axis: vec3<f32>;
    var b_axis: vec3<f32>;
    var fw_uv: vec2<f32>;
    if (flat_face) {
        uv = world.xz; uv_rel = cam_rel.xz; uv_cam = cam_world.xz;
        t_axis = vec3<f32>(1.0, 0.0, 0.0); b_axis = vec3<f32>(0.0, 0.0, 1.0);
        fw_uv = vec2<f32>(fw.x, fw.z);
    } else if (an.x >= an.z) {
        uv = world.zy; uv_rel = cam_rel.zy; uv_cam = cam_world.zy;
        t_axis = vec3<f32>(0.0, 0.0, 1.0); b_axis = vec3<f32>(0.0, 1.0, 0.0);
        fw_uv = vec2<f32>(fw.z, fw.y);
    } else {
        uv = world.xy; uv_rel = cam_rel.xy; uv_cam = cam_world.xy;
        t_axis = vec3<f32>(1.0, 0.0, 0.0); b_axis = vec3<f32>(0.0, 1.0, 0.0);
        fw_uv = vec2<f32>(fw.x, fw.y);
    }
    // Exact world axes spanning the projection plane. On an axis-aligned
    // face they are exactly perpendicular to the normal, so a tangent
    // tilt needs no projection; on a slope the small non-orthogonality is
    // absorbed by the `normalize` each tilt goes through.
    let wallness = 1.0 - an.y;
    let fw_max = max(fw_uv.x, fw_uv.y);

    // --- formwork lattice, resolved before anything reads it ------------
    // The warp below has to know whether the joints are still visible
    // before it decides to pay for the field that bends them.
    let band_amp = surface_band_amp(m);
    let band_warp = surface_band_warp(m);
    // Guarded: a material with amp but no freq would divide by zero into
    // `resolved` and NaN across the whole face.
    let base_band_freq = max(surface_band_freq(m), 1e-3);
    let band_freq = select(base_band_freq, base_band_freq * FLOOR_JOINT_RATIO, flat_face);
    let band_r = select(
        0.0,
        resolved(1.0 / band_freq, select(fw.y, fw_max, flat_face)) * fade,
        band_amp > 0.0,
    );

    // --- metre-scale weathering: one field, three consumers (8 hashes) --
    let grime_amt = surface_grime_amount(m);
    let moss_amt = surface_moss_amount(m);
    // `q_raw` is the field; `q` is the field faded. The difference is
    // load-bearing — grime and moss take the faded one, the warp does
    // not. A material that consumes none of the three pays nothing.
    //
    // The condition reads the material's own parameters and nothing
    // distance-dependent. Skipping the field once `band_r` fell below a
    // threshold dropped a PHASE to zero at a fixed radius, which is the
    // ring the whole warp rule exists to prevent (~0.1% of albedo at
    // ~900 m on a material with a warp but no grime and no moss).
    var q_raw = 0.0;
    if (grime_amt > 0.0 || moss_amt > 0.0 || (band_warp > 0.0 && band_amp > 0.0)) {
        let p = uv * MOTTLE_FREQ;
        let v = vnoise2(p, 11) * MOTTLE_W0 + vnoise2(p * MOTTLE_OCT, 12) * MOTTLE_W1;
        q_raw = clamp((v - 0.5) * MOTTLE_GAIN, -1.0, 1.0);
    }
    // Against the FINER octave, so neither octave outlives its own gate.
    // Both outlast the fade horizon at every setting any level ships, so
    // sizing it this way costs nothing today and cannot be wrong later.
    let q = q_raw * resolved(1.0 / (MOTTLE_FREQ * MOTTLE_OCT), fw_max) * fade;

    // Grime, divided by its own average: a material's authored colour is
    // what it IS, not what it is before dirt. The deleted version
    // multiplied the mean down, so terrain got dirtier the closer you
    // stood to it, which is the "mottling nobody wanted".
    if (grime_amt > 0.0) {
        let tint = surface_grime_tint(m);
        let mid = grime_amt * GRIME_MAX * 0.5;
        out.albedo *= mix(vec3<f32>(1.0), tint, mid * (1.0 + q))
                    / mix(vec3<f32>(1.0), tint, mid);
    }

    // Moss, in damp flat crevices. The ONE field that does not converge
    // to the base colour: it fades out with `q`, so a distant view is
    // still unchanged, but its near-field average is a few per cent of
    // green. Coverage is ~11%, so it stays a patch.
    if (moss_amt > 0.0) {
        let moss = smoothstep(0.35, 0.95, q) * smoothstep(0.45, 0.85, n.y) * moss_amt;
        out.albedo = mix(out.albedo, surface_moss_color(m), moss);
    }

    // --- aggregate grain: albedo AND micro-relief (8 hashes, < ~35 m) ---
    // The relief is the half that matters. A 40 m wall with one exact
    // normal is a clay render whatever colour it is.
    let a_grain = surface_grain(m) * fade * resolved(1.0 / GRAIN_FREQ, fw_max);
    if (a_grain > 0.004) {
        let l0 = lattice(uv_rel, uv_cam, GRAIN_FREQ);
        let g0 = vnoise2_cell_d(vec2<i32>(l0.xy), l0.zw, 3);
        var v = g0.x * GRAIN_W0;
        var dv = g0.yz * GRAIN_W0;
        // Octave 1 gets its OWN gate, and not for its colour: the chain
        // rule weights its gradient GRAIN_W1 * GRAIN_OCT = 0.69, MORE
        // than octave 0's 0.667, so most of the normal perturbation comes
        // from the finer field — which stops resolving at half octave 0's
        // distance. Behind one gate sized for the coarse octave it would
        // go on tilting normals a full octave past the point where it is
        // nothing but noise, and that is grazing-angle sparkle.
        let r1 = resolved(1.0 / (GRAIN_FREQ * GRAIN_OCT), fw_max);
        var v1 = 0.5;                       // octave 1's own mean
        if (r1 > 0.004) {
            let l1 = lattice(uv_rel, uv_cam, GRAIN_FREQ * GRAIN_OCT);
            let g1 = vnoise2_cell_d(vec2<i32>(l1.xy), l1.zw, 4);
            v1 = mix(0.5, g1.x, r1);
            dv += g1.yz * (GRAIN_W1 * GRAIN_OCT * r1);
        }
        v += v1 * GRAIN_W1;
        let g = (v - 0.5) * 2.0;
        let dg = dv * 2.0;
        out.albedo *= 1.0 + GRAIN_ALBEDO * a_grain * g;
        // `dg` is a dimensionless tilt, NOT a heightfield slope:
        // deliberately not scaled by frequency, or finer grain would mean
        // deeper bumps. Every relief term stays inside its field's
        // `resolve * fade` guard; hoisting one out to avoid a branch is
        // what makes a shader sparkle at range.
        out.normal = normalize(out.normal
            - GRAIN_RELIEF * a_grain * (dg.x * t_axis + dg.y * b_axis));
        out.rough = out.rough - GRAIN_ROUGH * a_grain * g;
    }

    // --- formwork: shutter joints, course tone, deck grid (1 hash) ------
    if (band_amp > 0.0) {
        let lo = surface_band_lo(m);
        let hi = surface_band_hi(m);
        let mean1 = 1.0 - (lo + hi) * 0.5;
        // The pour wanders — but on `q_raw`, NOT on `q`. This is a PHASE,
        // and the two gates may only ever scale an amplitude: fading a
        // phase slides a physically fixed joint across the wall as the
        // camera approaches (a third of a period between 5 m and 200 m at
        // brutalist's fade), and THE RULE does not catch it because a
        // phase shift preserves the mean.
        //
        // Capped rather than taken as authored: the parameter is in
        // PERIODS, so one value is a lean on a 1.8 m course and a fold on
        // purgatory's 11 m one. The cap binds only where the authored
        // warp would let the mottle out-run the lattice — brutalist's 0.3
        // against freq 0.75 passes untouched, purgatory's 0.85 against
        // 0.09 does not. The contour can still lean up to 45 degrees off
        // the axis the tilt below is emitted along where the field is at
        // its steepest, and a fraction of a degree where it is not; what
        // the cap buys is that the contour cannot close.
        let warp = min(band_warp, WARP_SLOPE_MAX * band_freq / MOTTLE_SLOPE) * q_raw;
        var v_raw: f32;
        var mean: f32;
        var tilt: vec3<f32>;
        if (flat_face) {
            // A cast deck is jointed on a grid. The PRODUCT of the two
            // lattices, not their min: a product's mean is exactly the
            // product of the means, which is what the convergence needs,
            // and it pits the crossings the way a real joint does.
            let a = notch(uv.x * band_freq + warp, lo, hi);
            let b = notch(uv.y * band_freq + warp, lo, hi);
            v_raw = a.x * b.x;
            mean = mean1 * mean1;
            tilt = (a.y * b.x) * t_axis + (a.x * b.y) * b_axis;
        } else {
            // On a wall the boards run horizontally, so the lattice is
            // world Y — the one axis every wall shares, which is why the
            // boards line up around a corner.
            let a = notch(world.y * band_freq + warp, lo, hi);
            v_raw = a.x;
            mean = mean1;
            tilt = a.y * vec3<f32>(0.0, 1.0, 0.0);
        }
        let v = mix(mean, v_raw, band_r);
        // Guarded like `band_freq`: amp 1 with lo = hi = 1 is an all-joint
        // band whose mean is 0, and dividing by it would NaN the face.
        out.albedo *= ((1.0 - band_amp) + band_amp * v)
                    / max((1.0 - band_amp) + band_amp * mean, 1e-3);
        // A joint is a groove. Painting a stripe is what made it read as
        // a decal; tilting the normal is what lets the sun find it.
        // `tilt` is chamfer-normalized and peaks at 1, on the deck's
        // product of two lattices as well (measured: one mid-chamfer
        // against the other mid-panel), so the whole term is bounded by
        // BAND_RELIEF * band_amp * band_r. Depth follows the authored
        // AMPLITUDE and never the chamfer width, and the warp's own
        // gradient is not in it.
        out.normal = normalize(out.normal - BAND_RELIEF * band_amp * band_r * tilt);
        if (!flat_face) {
            // Per-course tone: one hash, constant over a course, so it
            // has no frequency of its own and the band's own gate covers
            // it. The index is the notch's own argument plus a half, so
            // its step lands in the GROOVE, which `notch` centres on
            // fract = 0.5: `floor(world.y * band_freq)` alone would
            // change the tone down the middle of a board, where nothing
            // hides it, and would drift out of the groove wherever the
            // warp moves it.
            //
            // Per course and not per BOARD. A second index along the wall
            // — `floor(dot(world, t_axis) / 4)` — put a 4%-typical,
            // 11.6%-worst albedo step every four metres with no groove
            // under it, a hard one-pixel edge at every distance out to
            // where the band dies, and it re-indexed on the projection
            // flip, landing that step on the smooth face of a curved
            // column. Boards butt-jointed end to end need their own
            // `notch` so the step sits in a groove like every other joint
            // here; until there is one the tone runs the whole course.
            let bi = i32(floor(world.y * band_freq + warp + 0.5));
            out.albedo *= 1.0 + BOARD_TONE * band_r
                        * (hash3(vec3<i32>(bi, 0, 31)) - 0.5) * 2.0;
        }
    }

    // --- drip streaks (4 hashes, walls only) ----------------------------
    let a_streak = surface_streaks(m) * fade * wallness
                 * resolved(1.0 / STREAK_FREQ, fw_uv.x);
    if (a_streak > 0.004) {
        let s = vnoise2(vec2<f32>(uv.x * STREAK_FREQ,
                                  world.y * STREAK_FREQ * STREAK_ASPECT), 19);
        // Odd about the field's own centre, so its mean is exactly zero
        // without a measured constant: streaks darken and the washed
        // strips between them lighten, which is what a stained wall does.
        let d = clamp((s - 0.5) * STREAK_GAIN, -1.0, 1.0);
        // Water leaves a joint and runs DOWN, so the wash is strongest
        // just under a line and decays to the line below. The lines are
        // the material's own band where it has one, a default storey
        // where it does not.
        let run_f = select(STREAK_RUN_FREQ, base_band_freq * 0.25, band_amp > 0.0);
        // Faded to the mean of its own square once the CHAMFER — not the
        // period — stops spanning a pixel. `fract` here would have been a
        // full-amplitude step along a horizontal line every 5.6 m, the
        // same defect the band code was rewritten to stop committing.
        let run = wash(world.y * run_f, STREAK_CHAMFER);
        let run2 = mix(STREAK_RUN_MEAN2, run * run,
                       resolved(STREAK_CHAMFER / run_f, fw.y));
        // `run2` is independent of `d`, so the product's mean stays zero.
        let streak = d * abs(d) * run2;
        out.albedo *= 1.0 - STREAK_MAX * a_streak * streak;
        // Zero-mean, like the grain's. Keeping only the darkening half
        // would give perceptual_roughness a mean that moves with
        // distance, which is the defect grime is divided by its average
        // to avoid.
        out.rough = out.rough - STREAK_ROUGH * a_streak * streak;
    }

    out.rough = clamp(out.rough, 0.45, 1.0);
    return out;
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
    out.rough = SURFACE_ROUGHNESS;
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
    // Screen-space footprint, hoisted: derivatives have to be taken in
    // uniform control flow, and everything below this point either
    // branches on a flat-interpolated material id that neighbouring
    // triangles do not agree on, or sits after a `discard`. Of `cam_rel`
    // rather than `world` — they differ by a uniform, so the derivative
    // is identical, and `cam_rel` stays small where `world` reaches tens
    // of kilometres and loses the difference.
    let fw = fwidth(in.cam_rel);
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
    var rough = SURFACE_ROUGHNESS;
    if (m.head.x == MAT_KIND_CANOPY) {
        let ms = canopy_material(m, world, n, dist);
        base = ms.albedo;
        nl = ms.normal;
        ao = ms.ao;
        rough = ms.rough;
    } else if (m.head.x == MAT_KIND_ZONED) {
        base = zoned_albedo(m, world, n, dist);
    } else {
        let ms = surface_material(m, world, in.cam_rel, view.world_position.xyz,
                                  n, dist, fw, in.occlusion);
        base = ms.albedo;
        nl = ms.normal;
        ao = ms.ao;
        rough = ms.rough;
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
        // scratches.
        //
        // Faded to the mean rather than to zero: a lit district has to
        // still look lit from the far side of the hall, or the light is
        // exactly the identity it was giving the district away by.
        // Against the STRIP's width, not the period: a strip is about a
        // metre of every thirteen to sixty, so it stops spanning a pixel
        // long before the pattern's period does, and fading on the
        // period leaves the aliasing untouched at every distance you can
        // actually see it at. `resolved(1.0, x)` IS this rule — one
        // definition for the whole file, and it takes the last `fwidth`
        // out of non-uniform control flow.
        let strip_res = resolved(1.0, fw.z);
        // A strip is ~1 m of every `spacing`, and `works` is on with
        // probability `chance` — so those are the two means.
        let lit = mix(0.5 / spacing, line, strip_res) * mix(chance, works, strip_res);
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
    // Per fragment, from the recipe's own detail rather than from an
    // asset: grain and drip staining both move it, and both converge back
    // to SURFACE_ROUGHNESS at distance like every other field.
    pbr_input.material.perceptual_roughness = rough;
    pbr_input.material.metallic = 0.0;
    pbr_input.material.flags = pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    pbr_input.frag_coord = in.clip;
    pbr_input.world_position = vec4<f32>(world, 1.0);
    pbr_input.world_normal = nl;
    pbr_input.N = nl;
    pbr_input.V = normalize(-in.cam_rel);
    pbr_input.diffuse_occlusion = vec3<f32>(ao);
    // The Lagarde formula is `pow(NdotV + ao, exp2(-16*rough - 1)) - 1 +
    // ao`, and over the whole roughness range this shader can produce
    // (0.45 to 1.0) the exponent is at most 0.0034, so the `pow` is 1 to
    // within 0.003 and the whole thing collapses to `ao`. Occluding the
    // environment specular alongside the diffuse is the point: a crease
    // that does not see the sky does not reflect it either.
    pbr_input.specular_occlusion = ao;
    pbr_input.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;

    var color = pbr_functions::apply_pbr_lighting(pbr_input);

    // Baked sun shadow. The coarse half fades in past the cascade range
    // — Bevy's shadow maps cover the near field, the mesh-time march
    // covers the kilometres they can't reach, and never both at full
    // strength. The exact half is not faded at all, because nothing else
    // casts it: voxel chunks are not in the shadow map, so a building
    // that gave this up in the near field would stand in its own light.
    let faded = mix(1.0, in.shadow, smoothstep(BAKED_SHADOW_NEAR, BAKED_SHADOW_FAR, dist));
    let baked = mix(faded, in.shadow, in.shadow_exact);
    color = vec4<f32>(color.rgb * baked, color.a);

    return pbr_functions::main_pass_post_lighting_processing(pbr_input, color);
}
