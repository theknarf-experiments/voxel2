// Instanced grass blades: one draw call over a procedural tuft mesh.
// Per-instance data carries a world position and a hash; the vertex shader
// rotates the tuft, scales it, applies wind sway, and fades it out with
// distance. Camera-relative transform matches the chunk draw shader.

#import bevy_render::view::View
#import bevy_render::globals::Globals

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;

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

const FADE_START: f32 = 70.0;
const FADE_END: f32 = 110.0;

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
    let fade = 1.0 - smoothstep(FADE_START, FADE_END, dist);
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
    let base = mix(vec3<f32>(0.10, 0.22, 0.06), vec3<f32>(0.16, 0.30, 0.09), h02);
    let tip_col = mix(vec3<f32>(0.35, 0.52, 0.16), vec3<f32>(0.55, 0.62, 0.22), h03);
    let color = mix(base, tip_col, in.tip) * (0.45 + 0.55 * shadow);

    var out: VsOut;
    out.clip = view.clip_from_view * vec4<f32>(view_space, 1.0);
    out.color = color;
    out.cam_rel = cam_rel;
    return out;
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    // Match the terrain's sun + haze so grass sits in the scene.
    let sun_dir = normalize(vec3<f32>(0.55, 0.5, 0.32));
    let lit = in.color * (0.55 + 0.45 * max(sun_dir.y, 0.0));
    let dist = length(in.cam_rel);
    let haze_amount = 1.0 - exp(-dist * 0.00006);
    let haze_color = vec3<f32>(0.62, 0.72, 0.88);
    return vec4<f32>(mix(lit, haze_color, haze_amount), 1.0);
}
