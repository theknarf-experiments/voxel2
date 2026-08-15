// Instanced crossed-silhouette impostors: one draw call over a point
// population. Per-instance data is a world position plus a hash carrying
// yaw, size, silhouette pick and a baked shade factor.
//
// Crossed planes rather than billboards on purpose: a billboard has to be
// rotated per frame and swims as the camera turns, while two fixed
// crossed planes read as a solid from any angle and cost nothing to
// place.
//
// Impostors are the FAR half of a population — something with real
// geometry covers the near field — so an impostor fades IN with distance.

#import bevy_pbr::{
    mesh_view_bindings::{view, globals, lights},
    pbr_types,
    pbr_functions,
}

struct ImpostorEnv {
    flags: vec4<f32>,   // x = draw-white debug flag
    color_a: vec4<f32>, // pointed silhouette
    color_b: vec4<f32>, // waisted silhouette
    base: vec4<f32>,    // x = darkening at the base, y = normal lean
    // x = fade-in start, y = fade-in end, z = cull distance, w = height
    size: vec4<f32>,
}
@group(2) @binding(0) var<uniform> env: ImpostorEnv;

struct VsIn {
    // Quad vertex: x/z in -1..1 across the plane, y in 0..1 up it.
    @location(0) pos: vec3<f32>,
    // 0 at the base, 1 at the crown. The two waist vertices are at 0.5,
    // which is what the pointed silhouette drops to the base.
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
    let shade = f32((in.inst_hash >> 24u) & 0xFFu) / 255.0;

    // The silhouette IS the variant: a waist at half height, or dropped
    // to the base to make a point. One mesh, no collapsed geometry — the
    // version that carried both outlines shaded fourteen vertices per
    // instance to draw six of them.
    let is_waisted = step(0.5, h03);
    let tip = select(select(0.0, in.tip, in.tip != 0.5), in.tip, is_waisted > 0.5);
    var local = vec3<f32>(in.pos.x, tip, in.pos.z);

    let yaw = h01 * 6.2831853;
    let c = cos(yaw);
    let s = sin(yaw);
    var p = vec3<f32>(local.x * c - local.z * s, local.y, local.x * s + local.z * c);

    // Size from the hash: width scales with height so a big instance is
    // not a stretched small one.
    let height = env.size.w * (0.72 + h02 * 0.62);
    p.x *= height * 0.30;
    p.z *= height * 0.30;
    p.y *= height;

    let cam_rel_root = in.inst_pos - view.world_position;
    let dist = length(cam_rel_root);
    // Grow in as the near geometry stops, and collapse past the cull so
    // the far edge is a shrink rather than a wall of popping quads.
    let grow = smoothstep(env.size.x, env.size.y, dist);
    let cull = 1.0 - smoothstep(env.size.z * 0.82, env.size.z, dist);
    p *= grow * cull;

    // A slow lean, so a stand of these is not a field of identical
    // stamps.
    let t = globals.time;
    let phase = in.inst_pos.x * 0.05 + in.inst_pos.z * 0.037 + h03 * 6.28;
    p.x += sin(t * 0.5 + phase) * tip * tip * height * 0.02;

    let cam_rel = cam_rel_root + p;
    let view_space = (view.view_from_world * vec4<f32>(cam_rel, 0.0)).xyz;

    // Two colors by silhouette, darker toward the base where a solid
    // shape shades itself.
    //
    // A SHADE of the color, not a second color: almost none of an
    // impostor's area is whatever its base would be made of, and a
    // second authored hue drags the whole population toward it.
    let color_up = mix(env.color_a.rgb, env.color_b.rgb, is_waisted);
    let color = color_up * mix(env.base.x, 1.0, tip) * (0.45 + 0.55 * shade);

    var out: VsOut;
    out.clip = view.clip_from_view * vec4<f32>(view_space, 1.0);
    out.color = color;
    out.cam_rel = cam_rel;
    return out;
}

// Sun and ambient only, off Bevy's own light buffer.
//
// The impostor pass costs almost exactly what its fragment shader costs:
// cutting the mesh to eight vertices changed nothing, and so did thinning
// the population while holding covered area constant. Returning a
// constant from the fragment shader was worth the whole pass, so the
// fragment shader is the only place there is anything to win here.
//
// What is dropped relative to `apply_pbr_lighting`, and why it is safe
// for far scenery:
//   - clustered point/spot iteration — far scenery is lit by the sun;
//   - specular — for reflectance zero it contributed zero;
//   - environment map / `EnvBRDFApprox` — ambient here is a percent of
//     the sun;
//   - Burley — at roughness 1 its two scatter terms are within a few
//     percent of Lambert except at grazing angles, which a silhouette a
//     few pixels across cannot resolve;
//   - cascaded shadows — cascades stop long before impostor range, and
//     every instance carries a baked shade factor in its hash.
//
// The light VALUES are still Bevy's, read from the per-view lights
// buffer, so there is nothing here to drift out of step with the real
// geometry these hand over to — only the BRDF is simplified, not the
// lighting.
fn sun_and_ambient(base: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    // Per view, so this is already only the suns that light this view —
    // Bevy filters directional lights by render layer on the CPU.
    var direct = vec3<f32>(0.0);
    for (var i = 0u; i < lights.n_directional_lights; i = i + 1u) {
        let l = lights.directional_lights[i];
        let n_dot_l = saturate(dot(n, l.direction_to_light.xyz));
        direct += l.color.rgb * n_dot_l;
    }
    // Two factors that are easy to drop and both blow the result out:
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
    // Crossed planes have no meaningful normal of their own, but the
    // solid they stand for does: what a camera sees of it is the
    // hemisphere facing it. So the normal leans from up toward the
    // viewer, horizontally, and a stand is lit from the sun side and
    // unlit from the other.
    //
    // Shading off up alone is right until you walk around the
    // population: then every real object in front of you has turned its
    // shaded side to you and gone dark, and the impostors behind them
    // have not — measured four times too bright against the geometry
    // they hand over to. Up is only the correct normal from directly
    // overhead, which is where this degenerates and falls back to it.
    //
    // A smooth function of where the camera is, so the whole stand turns
    // together as you move with no instant at which anything pops.
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let flat = vec2<f32>(-cam_rel.x, -cam_rel.z);
    let flat_len = length(flat);
    // Scaled by how side-on the instance is seen: from overhead a solid
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
