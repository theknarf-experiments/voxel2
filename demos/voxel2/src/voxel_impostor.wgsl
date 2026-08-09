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
    mesh_view_bindings::{view, globals},
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    pbr_types,
    pbr_functions,
}

struct ImpostorEnv {
    flags: vec4<f32>,    // x = coverage-eval flag
    canopy_a: vec4<f32>, // species A canopy
    canopy_b: vec4<f32>, // species B canopy
    trunk: vec4<f32>,    // trunk/shadowed base
    // x = fade-in start, y = fade-in end, z = cull distance, w = height
    size: vec4<f32>,
}
@group(2) @binding(0) var<uniform> env: ImpostorEnv;

struct VsIn {
    // Quad vertex: x/z in -1..1 across the plane, y in 0..1 up it.
    @location(0) pos: vec3<f32>,
    // 0 at the base, 1 at the crown.
    @location(1) tip: f32,
    // 0 = conifer outline, 1 = broadleaf. The mesh holds both.
    @location(4) species: f32,
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

    // Collapse the silhouette this instance is not. Cheaper than two
    // draw calls and keeps the whole forest in one instance buffer.
    let is_broadleaf = step(0.5, h03);
    if (abs(in.species - is_broadleaf) > 0.5) {
        var dead: VsOut;
        dead.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        dead.color = vec3<f32>(0.0);
        dead.cam_rel = vec3<f32>(0.0);
        return dead;
    }

    let yaw = h01 * 6.2831853;
    let c = cos(yaw);
    let s = sin(yaw);
    var p = vec3<f32>(in.pos.x * c - in.pos.z * s, in.pos.y, in.pos.x * s + in.pos.z * c);

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
    p.x += sin(t * 0.5 + phase) * in.tip * in.tip * height * 0.02;

    let cam_rel = cam_rel_root + p;
    let view_space = (view.view_from_world * vec4<f32>(cam_rel, 0.0)).xyz;

    // Two species by hash, darker toward the base where a real canopy
    // shades its own trunk.
    let canopy = mix(env.canopy_a.rgb, env.canopy_b.rgb, is_broadleaf);
    let color = mix(env.trunk.rgb, canopy, in.tip) * (0.45 + 0.55 * shadow);

    var out: VsOut;
    out.clip = view.clip_from_view * vec4<f32>(view_space, 1.0);
    out.color = color;
    out.cam_rel = cam_rel;
    return out;
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    if (env.flags.x > 0.5) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    let world = vec3<f32>(
        view.world_position.x + in.cam_rel.x,
        view.world_position.y + in.cam_rel.y,
        view.world_position.z + in.cam_rel.z,
    );
    // Crossed planes have no meaningful normal; shade off up, like grass,
    // so a stand is evenly lit and the cascades still darken it.
    let n = vec3<f32>(0.0, 1.0, 0.0);

    var pbr_input = pbr_types::pbr_input_new();
    pbr_input.material.base_color = vec4<f32>(in.color, 1.0);
    pbr_input.material.perceptual_roughness = 1.0;
    pbr_input.material.metallic = 0.0;
    // Foliage is not a mirror. The default F0 of 0.04 over a large flat
    // quad facing straight up is enough specular to wash the canopy out
    // to near-white under full daylight.
    pbr_input.material.reflectance = vec3<f32>(0.0);
    pbr_input.material.flags = pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    pbr_input.frag_coord = in.clip;
    pbr_input.world_position = vec4<f32>(world, 1.0);
    pbr_input.world_normal = n;
    pbr_input.N = n;
    pbr_input.V = normalize(-in.cam_rel);
    pbr_input.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;

    let color = pbr_functions::apply_pbr_lighting(pbr_input);
    return pbr_functions::main_pass_post_lighting_processing(pbr_input, color);
}
