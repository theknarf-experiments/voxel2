//! Grass: the ground-cover [`Prop`] population. A procedural blade tuft
//! drawn once per scatter point, re-uploaded only when the tile set
//! changes.
//!
//! Everything a prop renderer needs beyond this file — buffers, pipeline,
//! bind group, extract, prepare, draw, marker sync — is
//! `instanced::PropPlugin`.

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    camera::visibility::{self, VisibilityClass},
    prelude::*,
    render::{extract_component::ExtractComponent, render_resource::ShaderType},
};
use bytemuck::{Pod, Zeroable};

/// Marker entity anchoring one world's grass draw. See [`Prop`].
#[derive(Clone, Copy, Component, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<GrassMarker>)]
pub struct GrassMarker {
    pub world: voxel_engine::WorldId,
}

/// The scatter population this demo draws as grass. Just a name the
/// level and the demo agree on — the engine never sees it.
pub const GROUND_COVER_CLASS: &str = "groundcover";

impl crate::instanced::Prop for GrassMarker {
    type Env = GrassEnv;
    type Vertex = BladeVertex;
    type Style = GrassStyle;

    const CLASS: &'static str = GROUND_COVER_CLASS;
    const NAME: &'static str = "grass";
    const LAYOUT_LABEL: &'static str = "grass_layout";
    const DRAW_LABEL: &'static str = "grass_draw";

    fn anchor(world: voxel_engine::WorldId) -> Self {
        Self { world }
    }

    fn world(&self) -> voxel_engine::WorldId {
        self.world
    }

    fn shader(assets: &AssetServer) -> Handle<Shader> {
        load_embedded_asset!(assets, "voxel_grass.wgsl")
    }

    fn mesh() -> (Vec<BladeVertex>, Vec<u32>) {
        build_tuft()
    }

    fn env(flags: Vec4, style: &GrassStyle) -> GrassEnv {
        GrassEnv {
            flags,
            base_a: style.base_a,
            base_b: style.base_b,
            tip_a: style.tip_a,
            tip_b: style.tip_b,
            fade: style.fade,
        }
    }
}

pub struct GrassPlugin;

impl Plugin for GrassPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "voxel_grass.wgsl");
        app.add_plugins(crate::instanced::PropPlugin::<GrassMarker>::default());
    }
}

// --- tuft mesh ---------------------------------------------------------------

/// Blade vertex: position + tip factor.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BladeVertex {
    pos: [f32; 3],
    tip: f32,
}

/// A tuft: 6 thin bent triangles fanned around the root.
fn build_tuft() -> (Vec<BladeVertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    let blades = 6u32;
    for i in 0..blades {
        let angle = std::f32::consts::TAU * i as f32 / blades as f32 + i as f32 * 0.7;
        let (s, c) = angle.sin_cos();
        let dir = Vec3::new(c, 0.0, s);
        let root = dir * 0.06;
        let height = 0.35 + (i as f32 * 0.37).fract() * 0.3;
        let lean = dir * (0.10 + (i as f32 * 0.53).fract() * 0.12);
        let side = Vec3::new(-s, 0.0, c) * 0.035;
        let base = verts.len() as u32;
        verts.push(BladeVertex {
            pos: (root - side).to_array(),
            tip: 0.0,
        });
        verts.push(BladeVertex {
            pos: (root + side).to_array(),
            tip: 0.0,
        });
        verts.push(BladeVertex {
            pos: (root + lean + Vec3::Y * height).to_array(),
            tip: 1.0,
        });
        indices.extend([base, base + 1, base + 2]);
    }
    (verts, indices)
}

// --- look --------------------------------------------------------------------

/// Blade look: two base hues and two tip hues mixed per point, plus the
/// view-distance fade. Art direction, so the values are albedo (the sun
/// is physical daylight) and they live here, not in the level file.
#[derive(Resource, Clone, Copy)]
pub struct GrassStyle {
    pub base_a: Vec4,
    pub base_b: Vec4,
    pub tip_a: Vec4,
    pub tip_b: Vec4,
    /// x = fade start (m), y = fade end.
    pub fade: Vec4,
}

impl Default for GrassStyle {
    fn default() -> Self {
        Self {
            base_a: Vec4::new(0.0187, 0.0411, 0.0112, 0.0),
            base_b: Vec4::new(0.0299, 0.056, 0.0168, 0.0),
            tip_a: Vec4::new(0.0653, 0.0971, 0.0299, 0.0),
            tip_b: Vec4::new(0.1027, 0.1157, 0.0411, 0.0),
            fade: Vec4::new(70.0, 110.0, 0.0, 0.0),
        }
    }
}

/// Level environment slice for the grass shader (sun + haze + style).
#[derive(ShaderType, Clone, Copy, Default)]
pub struct GrassEnv {
    /// w = coverage-eval flag; lighting comes from Bevy.
    flags: Vec4,
    base_a: Vec4,
    base_b: Vec4,
    tip_a: Vec4,
    tip_b: Vec4,
    fade: Vec4,
}
