// Instanced tree impostors: crossed quads, one draw call over a point
// population. Per-instance data is a world position plus a hash carrying
// yaw, size, species and the baked terrain sun-shadow.
//
// Crossed quads rather than billboards on purpose: a billboard has to be
// rotated per frame and swims as the camera turns, while two fixed
// crossed planes read as a tree from any angle and cost nothing to place.
//
// These are the FAR half of the forest. The near half is real prop
// meshes, so an impostor fades IN with distance — the reverse of grass,
// which fades out.

#import bevy_pbr::{
    mesh_view_bindings::{view, globals, lights},
    pbr_types,
    pbr_functions,
}

struct ImpostorEnv {
    flags: vec4<f32>,    // x = coverage-eval flag
    canopy_a: vec4<f32>, // species A canopy
    canopy_b: vec4<f32>, // species B canopy
    base: vec4<f32>,     // x = how dark the canopy goes at its base
    // x = fade-in start, y = fade-in end, z = cull distance, w = height
    size: vec4<f32>,
}
@group(2) @binding(0) var<uniform> env: ImpostorEnv;

struct VsIn {
    // Quad vertex: x/z in -1..1 across the plane, y in 0..1 up it.
    @location(0) pos: vec3<f32>,
    // 0 at the base, 1 at the crown. The two waist vertices are at 0.5,
    // which is what a conifer drops to the base.
    @location(1) tip: f32,
    @location(2) inst_pos: vec3<f32>,
    @location(3) inst_hash: u32,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) cam_rel: vec3<f32>,
}

@vertex
fn vertex(in: VsIn) -> VsOut {
    let h01 = f32(in.inst_hash & 0xFFu) / 255.0;
    let h02 = f32((in.inst_hash >> 8u) & 0xFFu) / 255.0;
    let h03 = f32((in.inst_hash >> 16u) & 0xFFu) / 255.0;
    let shadow = f32((in.inst_hash >> 24u) & 0xFFu) / 255.0;

    // The silhouette IS the species: a diamond's waist at half height, or
    // dropped to the base to make a cone. One mesh, no collapsed geometry
    // — the version that carried both outlines shaded fourteen vertices
    // per tree to draw six of them.
    let is_broadleaf = step(0.5, h03);
    let tip = select(select(0.0, in.tip, in.tip != 0.5), in.tip, is_broadleaf > 0.5);
    var local = vec3<f32>(in.pos.x, tip, in.pos.z);

    let yaw = h01 * 6.2831853;
    let c = cos(yaw);
    let s = sin(yaw);
    var p = vec3<f32>(local.x * c - local.z * s, local.y, local.x * s + local.z * c);

    // Size from the hash: crown width scales with height so a big tree is
    // not a stretched small one.
    let height = env.size.w * (0.72 + h02 * 0.62);
    p.x *= height * 0.30;
    p.z *= height * 0.30;
    p.y *= height;

    let cam_rel_root = in.inst_pos - view.world_position;
    let dist = length(cam_rel_root);
    // Grow in as the real meshes stop, and collapse past the cull so the
    // far edge is a shrink rather than a wall of popping quads.
    let grow = smoothstep(env.size.x, env.size.y, dist);
    let cull = 1.0 - smoothstep(env.size.z * 0.82, env.size.z, dist);
    p *= grow * cull;

    // A slow lean, so a forest is not a field of identical stamps.
    let t = globals.time;
    let phase = in.inst_pos.x * 0.05 + in.inst_pos.z * 0.037 + h03 * 6.28;
    p.x += sin(t * 0.5 + phase) * tip * tip * height * 0.02;

    let cam_rel = cam_rel_root + p;
    let view_space = (view.view_from_world * vec4<f32>(cam_rel, 0.0)).xyz;

    // Two species by hash, darker toward the base where a real canopy
    // shades itself.
    //
    // A SHADE of the canopy, not a second colour. It used to blend to a
    // hand-picked brown, which is the wrong thing in two ways: almost none
    // of an impostor's area is trunk, and that brown was as red as the
    // canopy was green, so the lower half of every tree dragged a stand
    // warm. Against the real trees it stands in for it read as a different
    // species — the whole reason the canopy colours are now taken from
    // them rather than authored again.
    let canopy = mix(env.canopy_a.rgb, env.canopy_b.rgb, is_broadleaf);
    let color = canopy * mix(env.base.x, 1.0, tip) * (0.45 + 0.55 * shadow);

    var out: VsOut;
    out.clip = view.clip_from_view * vec4<f32>(view_space, 1.0);
    out.color = color;
    out.cam_rel = cam_rel;
    return out;
}

/// Sun and ambient only, off Bevy's own light buffer.
///
/// The impostor pass costs almost exactly what its fragment shader costs:
/// cutting the mesh to eight vertices changed nothing, and so did thinning
/// the population while holding covered area constant. Returning a
/// constant from the fragment shader was worth 1.9 ms of a 10 ms frame, so
/// the fragment shader is the only place there is anything to win here.
///
/// This takes 0.5 ms of that 1.9 (mean 9.9 -> 9.4 ms, matched settle
/// time). The rest is fog, tonemapping and the deband dither, which are
/// what make an impostor agree with every other surface in the frame and
/// so are not ours to skip.
///
/// What is dropped, and why it is safe here:
///   - clustered point/spot iteration — a forest is lit by the sun;
///   - specular — `reflectance` is already zero, so it contributed zero;
///   - environment map / `EnvBRDFApprox` — there is no environment map,
///     and ambient here is a percent of the sun;
///   - Burley — at `roughness = 1` its two scatter terms are within a few
///     percent of Lambert except at grazing angles, which a canopy a few
///     pixels across cannot resolve;
///   - cascaded shadows — the cascades stop at 420 m and impostors run to
///     4 km, so they covered the near sliver only, and every instance
///     already carries a baked terrain sun-shadow in its hash.
///
/// The light VALUES are still Bevy's, read from the per-view lights
/// buffer, so there is nothing here to drift out of step with the props
/// these hand over to — only the BRDF is simplified, not the lighting.
fn sun_and_ambient(base: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    // Per view, so this is already only the suns that light this world —
    // Bevy filters directional lights by render layer on the CPU.
    var direct = vec3<f32>(0.0);
    for (var i = 0u; i < lights.n_directional_lights; i = i + 1u) {
        let l = lights.directional_lights[i];
        let n_dot_l = saturate(dot(n, l.direction_to_light.xyz));
        direct += l.color.rgb * n_dot_l;
    }
    // Two factors that are easy to drop and both blow the canopy out:
    // the 1/PI lives inside `Fd_Burley`, so dropping Burley drops the
    // normalisation with it, and Bevy applies `view.exposure` once over
    // the summed lighting rather than per light.
    const FRAC_1_PI: f32 = 0.31830987;
    return base * (direct * FRAC_1_PI + lights.ambient_color.rgb) * view.exposure;
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    if (env.flags.x > 0.5) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    let cam_rel = in.cam_rel;
    let world = view.world_position.xyz + cam_rel;
    // Crossed planes have no meaningful normal of their own, but a TREE
    // does: what a camera sees of a canopy is the hemisphere facing it. So
    // the normal leans from up toward the viewer, horizontally, and a
    // stand is lit from the sun side and unlit from the other.
    //
    // Shading off up alone — even, like grass — is right until you walk
    // round a wood. Then every real tree in front of you has turned its
    // shaded side to you and gone dark, and the impostors behind them have
    // not: measured four times too bright against the props they hand over
    // to. Up is only the correct normal for the one view from directly
    // overhead, which is where this degenerates and falls back to it.
    //
    // A smooth function of where the camera is, so the whole stand turns
    // together as you move with no instant at which anything pops.
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let flat = vec2<f32>(-cam_rel.x, -cam_rel.z);
    let flat_len = length(flat);
    // Scaled by how side-on the tree is seen: from overhead a canopy
    // really does present its top, and that is also where the horizontal
    // direction stops being well conditioned, so the same term fixes the
    // look and the degenerate case.
    let sideness = flat_len / max(length(cam_rel), 1e-3);
    var lean = up;
    if (flat_len > 1e-3) {
        let side = vec3<f32>(flat.x / flat_len, 0.0, flat.y / flat_len);
        lean = mix(up, side, env.base.y * sideness);
    }
    let n = normalize(lean);

    // Fog, tonemapping and dither stay Bevy's — they are cheap, and they
    // are what has to agree with every other surface in the frame. Only
    // the lighting above is ours. This carries just the fields that path
    // reads.
    var pbr_input = pbr_types::pbr_input_new();
    pbr_input.material.flags = pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    pbr_input.frag_coord = in.clip;
    pbr_input.world_position = vec4<f32>(world, 1.0);

    let color = vec4<f32>(sun_and_ambient(in.color, n), 1.0);
    return pbr_functions::main_pass_post_lighting_processing(pbr_input, color);
}
