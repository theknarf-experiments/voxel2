// Instanced grass blades: one draw call over a procedural tuft mesh.
// Per-instance data carries a world position and a hash; the vertex shader
// rotates the tuft, scales it, applies wind sway, and fades it out with
// distance. Camera-relative transform matches the chunk draw shader.

#import bevy_pbr::{
    mesh_view_bindings::{view, globals},
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    pbr_types,
    pbr_functions,
}

// Blade look only — light, shadows, fog and tonemapping are Bevy's.
struct GrassEnv {
    flags: vec4<f32>,     // x = coverage-eval flag
    base_a: vec4<f32>,    // blade base hue A
    base_b: vec4<f32>,    // blade base hue B
    tip_a: vec4<f32>,     // blade tip hue A
    tip_b: vec4<f32>,     // blade tip hue B
    fade: vec4<f32>,      // fade start, fade end, -, -
}
@group(2) @binding(0) var<uniform> env: GrassEnv;

struct VsIn {
    // Blade vertex (tuft-local).
    @location(0) pos: vec3<f32>,
    // 0 at the root, 1 at the tip — drives sway and shading.
    @location(1) tip: f32,
    // Instance: world position + packed hash.
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
    // Baked terrain sun-shadow factor (top byte, written by the CPU).
    let shadow = f32((in.inst_hash >> 24u) & 0xFFu) / 255.0;

    // Tuft yaw + scale from the hash.
    let yaw = h01 * 6.2831853;
    let c = cos(yaw);
    let s = sin(yaw);
    var p = vec3<f32>(in.pos.x * c - in.pos.z * s, in.pos.y, in.pos.x * s + in.pos.z * c);
    let scale = 0.7 + h02 * 0.7;
    p *= scale;

    // Distance fade: shrink blades into the ground instead of popping.
    let cam_rel_root = in.inst_pos - view.world_position;
    let dist = length(cam_rel_root);
    let fade = 1.0 - smoothstep(env.fade.x, env.fade.y, dist);
    p.y *= fade;

    // Wind: two phases so neighbors don't sway in lockstep.
    let t = globals.time;
    let phase = in.inst_pos.x * 0.35 + in.inst_pos.z * 0.23 + h03 * 6.28;
    let sway = (sin(t * 1.7 + phase) * 0.6 + sin(t * 3.1 + phase * 1.7) * 0.25)
        * in.tip * in.tip * 0.12 * scale;
    p.x += sway;
    p.z += sway * 0.6;

    let cam_rel = cam_rel_root + p;
    let view_space = (view.view_from_world * vec4<f32>(cam_rel, 0.0)).xyz;

    // Color: rooted dark, tip lighter, hue variation by hash, dimmed by the
    // baked terrain shadow (ambient floor keeps shaded grass readable).
    let base = mix(env.base_a.rgb, env.base_b.rgb, h02);
    let tip_col = mix(env.tip_a.rgb, env.tip_b.rgb, h03);
    let color = mix(base, tip_col, in.tip) * (0.45 + 0.55 * shadow);

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
    // Blades are thin crossed geometry with no meaningful surface normal:
    // shade them off the up axis, which keeps a tuft evenly lit and lets
    // the cascades darken grass standing in a tree's shadow.
    let n = vec3<f32>(0.0, 1.0, 0.0);

    var pbr_input = pbr_types::pbr_input_new();
    pbr_input.material.base_color = vec4<f32>(in.color, 1.0);
    pbr_input.material.perceptual_roughness = 1.0;
    pbr_input.material.metallic = 0.0;
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
