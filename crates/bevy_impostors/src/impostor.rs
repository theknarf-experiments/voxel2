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
//! **An impostor population is a VARIANT TABLE**, because the population
//! it stands in for has variants — species, size tiers — and a far tier
//! that flattens them hands over to near geometry of visibly the wrong
//! kind. Each instance names its variant in the hash; the variant's
//! [`ImpostorVariantStyle`] says its color and silhouette. Adding a
//! variant is writing a table row, not touching the renderer.
//!
//! **The instance hash is the whole per-instance contract**, byte by
//! byte: bits 0–7 yaw and sway phase, 8–15 size, 16–23 the VARIANT
//! index into the style's table, 24–31 a baked shade factor (0 = fully
//! shaded, 255 = fully lit) the host computes however it likes. One
//! 32-bit word, because at a million instances every byte is a megabyte.
//!
//! One impostor POPULATION is one [`ImpostorSet`] tag: `Impostors<Trees>`
//! and `Impostors<Litter>` are separate pipelines with separate styles,
//! fed separately. The tag is a unit struct the host declares; the crate
//! never learns what it means.

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    camera::visibility::{self, VisibilityClass},
    prelude::*,
    render::{extract_component::ExtractComponent, render_resource::ShaderType},
};
use bytemuck::{Pod, Zeroable};
use std::marker::PhantomData;

use crate::prop::{Prop, PropPlugin};

/// How many variants an impostor style can carry. An instance whose
/// variant byte reaches past the host's filled rows lands on a zeroed
/// row, whose zero height collapses it — invisible, not garbage.
pub const MAX_IMPOSTOR_VARIANTS: usize = 16;

/// One impostor population. The tag is the host's: a unit struct per
/// thing it draws this way, so each population gets its own pipeline,
/// its own [`ImpostorStyle`] and its own points.
pub trait ImpostorSet: Send + Sync + 'static {
    /// Prefix for buffer labels in a capture.
    const NAME: &'static str;
    /// Bind group layout labels have to be `&'static str`.
    const LAYOUT_LABEL: &'static str;
    const DRAW_LABEL: &'static str;
}

/// Marker entity anchoring one set's impostor draw. See [`Prop`].
#[derive(Component, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<Impostors<S>>)]
pub struct Impostors<S: ImpostorSet> {
    pub set: u32,
    marker: PhantomData<fn() -> S>,
}

impl<S: ImpostorSet> Impostors<S> {
    pub fn new(set: u32) -> Self {
        Self {
            set,
            marker: PhantomData,
        }
    }
}

// Manual, not derived: a derive would demand `S: Clone` of a tag that is
// never stored, only named.
impl<S: ImpostorSet> Clone for Impostors<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: ImpostorSet> Copy for Impostors<S> {}

impl<S: ImpostorSet> Prop for Impostors<S> {
    type Env = ImpostorEnv;
    type Vertex = ImpostorVertex;
    type Style = ImpostorStyle<S>;

    const NAME: &'static str = S::NAME;
    const LAYOUT_LABEL: &'static str = S::LAYOUT_LABEL;
    const DRAW_LABEL: &'static str = S::DRAW_LABEL;

    fn set(&self) -> u32 {
        self.set
    }

    fn shader(assets: &AssetServer) -> Handle<Shader> {
        load_embedded_asset!(assets, "impostor.wgsl")
    }

    fn mesh() -> (Vec<ImpostorVertex>, Vec<u32>) {
        build_impostor()
    }

    fn env(flags: Vec4, style: &ImpostorStyle<S>) -> ImpostorEnv {
        ImpostorEnv {
            flags,
            base: style.base,
            size: style.size,
            variants: style.variants,
        }
    }
}

/// Draws one [`ImpostorSet`]. The host spawns the markers and fills
/// [`crate::PropPoints<Impostors<S>>`].
pub struct ImpostorPlugin<S: ImpostorSet>(PhantomData<fn() -> S>);

impl<S: ImpostorSet> Default for ImpostorPlugin<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Every impostor population shares one shader; the FIRST plugin embeds
/// it.
#[derive(Resource)]
struct ImpostorShaderEmbedded;

impl<S: ImpostorSet> Plugin for ImpostorPlugin<S> {
    fn build(&self, app: &mut App) {
        if app
            .world()
            .get_resource::<ImpostorShaderEmbedded>()
            .is_none()
        {
            app.insert_resource(ImpostorShaderEmbedded);
            embedded_asset!(app, "impostor.wgsl");
        }
        app.add_plugins(PropPlugin::<Impostors<S>>::default());
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

/// One crossed silhouette, shaped per instance from its variant's
/// [`ImpostorVariantStyle::shape`]: the waist vertices land wherever the
/// variant says, so one mesh is a diamond, a cone, or anything between.
///
/// Silhouettes rather than rectangles: a rectangle reads as a billboard
/// nobody finished, and at impostor range the outline is the only thing
/// carrying the shape.
fn build_impostor() -> (Vec<ImpostorVertex>, Vec<u32>) {
    let mut verts: Vec<ImpostorVertex> = Vec::new();
    let mut indices = Vec::new();
    // ONE outline for every variant, a diamond: bottom, waist, top,
    // waist. The vertex shader moves the waist per instance. The mesh
    // used to carry two silhouettes and collapse the one an instance was
    // not, which meant shading fourteen vertices to draw six — at half a
    // million instances, millions of vertex invocations a frame thrown
    // away.
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

/// One variant's look: a color and a silhouette.
#[derive(ShaderType, Clone, Copy, Default)]
pub struct ImpostorVariantStyle {
    /// Linear color of the silhouette.
    pub color: Vec4,
    /// x = waist height in 0..1 (0 makes the diamond a cone, 0.5 keeps
    /// it a diamond, anything between is between). y = half-width as a
    /// fraction of height. z = height factor on the style's base height.
    /// w = spare.
    pub shape: Vec4,
}

impl ImpostorVariantStyle {
    /// A pointed silhouette — a cone — in the given color.
    pub fn pointed(color: Vec4) -> Self {
        Self {
            color,
            shape: Vec4::new(0.0, 0.30, 1.0, 0.0),
        }
    }

    /// A waisted silhouette — a diamond — in the given color.
    pub fn waisted(color: Vec4) -> Self {
        Self {
            color,
            shape: Vec4::new(0.5, 0.30, 1.0, 0.0),
        }
    }
}

/// Impostor look and reach: the variant table, and the distances that
/// decide where something nearer hands over to these.
#[derive(Resource)]
pub struct ImpostorStyle<S: ImpostorSet> {
    /// x = how dark a silhouette goes at its base, as a fraction of
    /// itself. y = how far the shading normal leans from up toward the
    /// viewer (see the fragment shader for why an impostor wants that).
    pub base: Vec4,
    /// x = fade-in start, y = fade-in end, z = cull, w = base height (m).
    pub size: Vec4,
    /// One row per variant, indexed by each instance's variant byte.
    /// Unfilled rows are zeroed, which collapses any instance that lands
    /// on them.
    pub variants: [ImpostorVariantStyle; MAX_IMPOSTOR_VARIANTS],
    marker: PhantomData<fn() -> S>,
}

impl<S: ImpostorSet> Default for ImpostorStyle<S> {
    fn default() -> Self {
        let mut variants = [ImpostorVariantStyle::default(); MAX_IMPOSTOR_VARIANTS];
        // Two greens, because the first thing anyone draws a million of
        // is a forest — a host with its own palette overwrites these.
        // LINEAR, not sRGB.
        variants[0] = ImpostorVariantStyle::pointed(Vec4::new(0.0051, 0.0223, 0.0041, 0.0));
        variants[1] = ImpostorVariantStyle::waisted(Vec4::new(0.0137, 0.0304, 0.0041, 0.0));
        Self {
            base: Vec4::new(0.35, 0.5, 0.0, 0.0),
            size: Vec4::new(85.0, 150.0, 4000.0, 7.0),
            variants,
            marker: PhantomData,
        }
    }
}

/// Uniform slice for the impostor shader (twin of `ImpostorEnv` in WGSL).
#[derive(ShaderType, Clone, Copy)]
pub struct ImpostorEnv {
    /// See [`crate::PropFlags`]; x nonzero draws every fragment white.
    flags: Vec4,
    base: Vec4,
    size: Vec4,
    variants: [ImpostorVariantStyle; MAX_IMPOSTOR_VARIANTS],
}

impl Default for ImpostorEnv {
    fn default() -> Self {
        Self {
            flags: Vec4::ZERO,
            base: Vec4::ZERO,
            size: Vec4::ZERO,
            variants: [ImpostorVariantStyle::default(); MAX_IMPOSTOR_VARIANTS],
        }
    }
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

    /// The variant count is a twin of the WGSL array length.
    #[test]
    fn the_shaders_variant_table_matches_max() {
        let wgsl = include_str!("impostor.wgsl");
        let declared: usize = wgsl
            .split("array<ImpostorVariant, ")
            .nth(1)
            .and_then(|rest| rest.split('u').next())
            .and_then(|n| n.trim().parse().ok())
            .expect("impostor.wgsl declares the variant array length");
        assert_eq!(declared, super::MAX_IMPOSTOR_VARIANTS);
    }
}
