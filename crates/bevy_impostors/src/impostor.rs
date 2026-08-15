//! The crossed-silhouette impostor: the far half of any large population
//! of standing things.
//!
//! A real prop entity costs entities and a transform hierarchy, which
//! tops out in the low thousands; an impostor costs 16 bytes in a vertex
//! buffer, so a population is bounded by how many placements the host
//! publishes rather than by the renderer.
//!
//! Two fixed crossed planes rather than a camera-facing billboard, on
//! purpose: a billboard has to be rotated per frame and swims as the
//! camera turns, while crossed planes read as a solid from any angle and
//! cost nothing to place.
//!
//! **The instance hash is the whole per-instance contract**, byte by
//! byte: bits 0–7 yaw, 8–15 size, 16–23 silhouette pick + sway phase,
//! 24–31 a baked shade factor (0 = fully shaded, 255 = fully lit) the
//! host computes however it likes — terrain sun occlusion, cave darkness,
//! nothing. One 32-bit word, because at a million instances every byte
//! is a megabyte.

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    camera::visibility::{self, VisibilityClass},
    prelude::*,
    render::{extract_component::ExtractComponent, render_resource::ShaderType},
};
use bytemuck::{Pod, Zeroable};

use crate::prop::{Prop, PropPlugin};

/// Marker entity anchoring one set's impostor draw. See [`Prop`].
#[derive(Clone, Copy, Component, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<Impostors>)]
pub struct Impostors {
    pub set: u32,
}

impl Prop for Impostors {
    type Env = ImpostorEnv;
    type Vertex = ImpostorVertex;
    type Style = ImpostorStyle;

    const NAME: &'static str = "impostor";
    const LAYOUT_LABEL: &'static str = "impostor_layout";
    const DRAW_LABEL: &'static str = "impostor_draw";

    fn set(&self) -> u32 {
        self.set
    }

    fn shader(assets: &AssetServer) -> Handle<Shader> {
        load_embedded_asset!(assets, "impostor.wgsl")
    }

    fn mesh() -> (Vec<ImpostorVertex>, Vec<u32>) {
        build_impostor()
    }

    fn env(flags: Vec4, style: &ImpostorStyle) -> ImpostorEnv {
        ImpostorEnv {
            flags,
            color_a: style.color_a,
            color_b: style.color_b,
            base: style.base,
            size: style.size,
        }
    }
}

/// Draws [`Impostors`]. The host spawns the markers and fills
/// [`crate::PropPoints<Impostors>`].
pub struct ImpostorPlugin;

impl Plugin for ImpostorPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "impostor.wgsl");
        app.add_plugins(PropPlugin::<Impostors>::default());
    }
}

/// Fraction of the cull distance the impostors start shrinking at. Twin
/// of the `env.size.z * 0.82` in `impostor.wgsl`.
///
/// Public because a host aligning the far edge with something of its own
/// — a ground texture taking over, a fog wall — needs to divide by it:
/// setting the cull to `handover / IMPOSTOR_FADE_FROM` puts the START of
/// the fade exactly at the handover.
pub const IMPOSTOR_FADE_FROM: f32 = 0.82;

// --- silhouette mesh ---------------------------------------------------------

/// Impostor vertex: unit-quad position + a 0..1 base-to-crown factor.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ImpostorVertex {
    pos: [f32; 3],
    tip: f32,
}

/// One crossed silhouette, shaped per instance: pointed or waisted, from
/// the same four points.
///
/// Silhouettes rather than rectangles: a rectangle reads as a billboard
/// nobody finished, and at impostor range the outline is the only thing
/// carrying the shape.
fn build_impostor() -> (Vec<ImpostorVertex>, Vec<u32>) {
    let mut verts: Vec<ImpostorVertex> = Vec::new();
    let mut indices = Vec::new();
    // ONE outline for both silhouettes, a diamond: bottom, waist, top,
    // waist. The pointed variant is the same four points with the waist
    // dropped to the base, which the vertex shader does per instance.
    //
    // The mesh used to carry both silhouettes and collapse the one an
    // instance was not, which meant shading fourteen vertices to draw
    // six. At half a million instances that is millions of vertex
    // invocations a frame thrown away, and impostor cost measured LINEAR
    // in instance count — the tell that it is per-instance work rather
    // than fill.
    let outline: [(f32, f32); 4] = [(0.0, 0.0), (-1.0, 0.5), (0.0, 1.0), (1.0, 0.5)];
    for axis in 0..2u32 {
        let (sx, sz) = if axis == 0 { (1.0, 0.0) } else { (0.0, 1.0) };
        let base = verts.len() as u32;
        for &(u, v) in &outline {
            verts.push(ImpostorVertex {
                pos: [u * sx, v, u * sz],
                tip: v,
            });
        }
        // Fan, and the same fan reversed: a crossed silhouette has to be
        // visible from behind without disabling culling globally.
        for i in 1..outline.len() as u32 - 1 {
            indices.extend([base, base + i, base + i + 1]);
            indices.extend([base, base + i + 1, base + i]);
        }
    }
    (verts, indices)
}

// --- look --------------------------------------------------------------------

/// Impostor look and reach: two silhouette colors, and the distances that
/// decide where something nearer hands over to these.
#[derive(Resource, Clone, Copy)]
pub struct ImpostorStyle {
    /// The pointed silhouette's color (linear).
    pub color_a: Vec4,
    /// The waisted silhouette's color (linear).
    pub color_b: Vec4,
    /// x = how dark a silhouette goes at its base, as a fraction of
    /// itself. y = how far the shading normal leans from up toward the
    /// viewer (see the fragment shader for why an impostor wants that).
    pub base: Vec4,
    /// x = fade-in start, y = fade-in end, z = cull, w = base height (m).
    pub size: Vec4,
}

impl Default for ImpostorStyle {
    fn default() -> Self {
        Self {
            // LINEAR, not sRGB. Two greens, because the first thing
            // anyone draws a million of is a forest — a host with its own
            // palette overwrites these.
            color_a: Vec4::new(0.0051, 0.0223, 0.0041, 0.0),
            color_b: Vec4::new(0.0137, 0.0304, 0.0041, 0.0),
            base: Vec4::new(0.35, 0.5, 0.0, 0.0),
            size: Vec4::new(85.0, 150.0, 4000.0, 7.0),
        }
    }
}

/// Uniform slice for the impostor shader (twin of `ImpostorEnv` in WGSL).
#[derive(ShaderType, Clone, Copy, Default)]
pub struct ImpostorEnv {
    /// See [`crate::PropFlags`]; x nonzero draws every fragment white.
    flags: Vec4,
    color_a: Vec4,
    color_b: Vec4,
    base: Vec4,
    size: Vec4,
}

#[cfg(test)]
mod tests {
    /// [`IMPOSTOR_FADE_FROM`] and the shader's multiplier are ONE number
    /// written twice, and the two are load-bearing in opposite
    /// directions: a host sets the cull distance to
    /// `handover / IMPOSTOR_FADE_FROM` so that the shader's fade-out —
    /// which begins at `cull * IMPOSTOR_FADE_FROM` — lands exactly on its
    /// handover point. Drift them and the impostors either fade across
    /// the wrong span or stop short of it.
    ///
    /// [`IMPOSTOR_FADE_FROM`]: super::IMPOSTOR_FADE_FROM
    #[test]
    fn the_shaders_fade_start_is_fade_from() {
        let wgsl = include_str!("impostor.wgsl");
        assert_eq!(
            wgsl.matches("env.size.z * ").count(),
            1,
            "a second `env.size.z *` appeared — this test would pin the wrong one"
        );
        let declared: f32 = wgsl
            .split("env.size.z * ")
            .nth(1)
            .and_then(|rest| rest.split([',', ')', ';']).next())
            .and_then(|n| n.trim().parse().ok())
            .expect("impostor.wgsl multiplies env.size.z by a literal");
        assert_eq!(
            declared,
            super::IMPOSTOR_FADE_FROM,
            "impostor.wgsl and IMPOSTOR_FADE_FROM disagree about where the fade starts"
        );
    }
}
