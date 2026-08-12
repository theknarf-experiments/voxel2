//! Tree impostors: the far-forest [`Prop`] population, a crossed
//! silhouette drawn once per scatter point.
//!
//! This is how a forest gets to millions. A prop entity costs three
//! entities and a transform hierarchy, which tops out in the low
//! thousands; an impostor costs 16 bytes in a vertex buffer, so the
//! population is bounded by how many placements the scatter layer cares
//! to generate rather than by the renderer.
//!
//! Everything shared with grass — buffers, pipeline, bind group, extract,
//! prepare, draw, marker sync — is `instanced::PropPlugin`. What is left
//! here is what only impostors have: the silhouette, and the palette and
//! reach they take from their neighbours.

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    camera::visibility::{self, VisibilityClass},
    prelude::*,
    render::{extract_component::ExtractComponent, render_resource::ShaderType, Extract},
};
use bytemuck::{Pod, Zeroable};

/// Marker entity anchoring one world's impostor draw. See [`Prop`].
#[derive(Clone, Copy, Component, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<ImpostorMarker>)]
pub struct ImpostorMarker {
    pub world: voxel_engine::WorldId,
}

/// The scatter population this demo draws as tree impostors. Just a name
/// the level and the demo agree on — the engine never sees it.
pub const IMPOSTOR_CLASS: &str = "treecover";

/// The prop class these impostors stand in for.
///
/// Their canopy colours are TAKEN from its variants rather than authored
/// again. They were authored twice, and the two drifted: the impostors
/// carried hand-converted linear values that had lost a third of their
/// green, so a stand handed over to real trees that were a different
/// species of green. One palette, and retuning the props retunes the
/// forest behind them.
pub const IMPOSTOR_STANDS_IN_FOR: &str = "tree";

impl crate::instanced::Prop for ImpostorMarker {
    type Env = ImpostorEnv;
    type Vertex = ImpostorVertex;
    type Style = ImpostorStyle;

    const CLASS: &'static str = IMPOSTOR_CLASS;
    const NAME: &'static str = "impostor";
    const LAYOUT_LABEL: &'static str = "impostor_layout";
    const DRAW_LABEL: &'static str = "impostor_draw";

    fn anchor(world: voxel_engine::WorldId) -> Self {
        Self { world }
    }

    fn world(&self) -> voxel_engine::WorldId {
        self.world
    }

    fn shader(assets: &AssetServer) -> Handle<Shader> {
        load_embedded_asset!(assets, "voxel_impostor.wgsl")
    }

    fn mesh() -> (Vec<ImpostorVertex>, Vec<u32>) {
        build_impostor()
    }

    fn env(flags: Vec4, style: &ImpostorStyle) -> ImpostorEnv {
        ImpostorEnv {
            flags,
            canopy_a: style.canopy_a,
            canopy_b: style.canopy_b,
            base: style.base,
            size: style.size,
        }
    }
}

pub struct ImpostorPlugin;

impl Plugin for ImpostorPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "voxel_impostor.wgsl");
        app.add_plugins(crate::instanced::PropPlugin::<ImpostorMarker>::default());

        let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
            return;
        };
        render_app.add_systems(ExtractSchedule, sync_impostor_style);
    }
}

/// Fraction of the cull distance the impostors start shrinking at. Twin
/// of the `env.size.z * 0.82` in `voxel_impostor.wgsl`.
const FADE_FROM: f32 = 0.82;

/// Take the impostors' colour from the trees they stand in for, and their
/// reach from the ground that paints them.
///
/// Both are the middle tier's whole job: it has a real forest on one side
/// and a painted one on the other, and it is the only thing that can be
/// wrong about either. Neither number is authored here, because both are
/// really somebody else's — see [`IMPOSTOR_STANDS_IN_FOR`] for the palette.
///
/// For the reach: the two tiers meet at one distance and it is the level's
/// to choose, but only one of them can honour the number as written. Paint
/// is decided per CHUNK, so it begins at whatever LOD boundary follows;
/// culling at the authored distance left a ring of ground with impostors
/// gone and paint not yet arrived. So the impostors are told where the
/// paint really starts and fade out ACROSS it — the texel grid appears
/// while they are still at full strength and is never seen bare.
fn sync_impostor_style(
    worlds: Extract<Res<voxel_engine::Worlds>>,
    props: Extract<Res<crate::WorldProps>>,
    mut style: ResMut<ImpostorStyle>,
) {
    // The cone silhouette is a conifer and the diamond is a broadleaf, so
    // each takes that prop variant's foliage. Matched by MODEL rather than
    // by index: which variant a level lists first is level dressing, and
    // reading it positionally would swap the two species the day someone
    // reorders them.
    for table in props.0.values() {
        let Some(class) = table.0.get(IMPOSTOR_STANDS_IN_FOR) else {
            continue;
        };
        let foliage = |model| {
            class
                .variants
                .iter()
                .find(|v| v.model == model)
                .map(|v| v.foliage.to_linear().to_vec4())
        };
        if let Some(c) = foliage(crate::props::Model::Conifer) {
            style.canopy_a = c;
        }
        if let Some(c) = foliage(crate::props::Model::Broadleaf) {
            style.canopy_b = c;
        }
    }

    // Every frame rather than on change: this is a fold over at most
    // `MAX_WORLDS` levels, and a change tick compared across worlds is a
    // subtlety to get wrong for no saving.
    let reach = worlds
        .iter()
        .filter_map(|world| {
            let def = world
                .query
                .planner_as::<crate::planning::RegionPlanner>()?
                .populations
                .iter()
                .find(|def| def.class == IMPOSTOR_CLASS)?;
            let from = def.cover.as_ref()?.from_m;
            Some(crate::surface_paint::cover_starts_m(&world.config, from) / FADE_FROM)
        })
        .fold(f32::NEG_INFINITY, f32::max);
    if reach.is_finite() {
        style.size.z = reach;
    }
}

// --- silhouette mesh ---------------------------------------------------------

/// Impostor vertex: unit-quad position + a 0..1 base-to-crown factor.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ImpostorVertex {
    pos: [f32; 3],
    tip: f32,
}

/// One crossed silhouette, shaped per instance: a cone for conifers, a
/// diamond for broadleaves, from the same four points.
///
/// Silhouettes rather than rectangles: a rectangle reads as a billboard
/// nobody finished, and at impostor range the outline is the only thing
/// carrying the shape.
fn build_impostor() -> (Vec<ImpostorVertex>, Vec<u32>) {
    let mut verts: Vec<ImpostorVertex> = Vec::new();
    let mut indices = Vec::new();
    // ONE outline for both species, a diamond: bottom, waist, top, waist.
    // A conifer is the same four points with the waist dropped to the
    // base, which the vertex shader does per instance.
    //
    // The mesh used to carry both silhouettes and collapse the one an
    // instance was not, which meant shading fourteen vertices to draw
    // six. At half a million trees that is several million vertex
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

/// Impostor look and reach. Art direction plus the two distances that
/// decide where the real props hand over to these.
#[derive(Resource, Clone, Copy)]
pub struct ImpostorStyle {
    pub canopy_a: Vec4,
    pub canopy_b: Vec4,
    /// x = how dark the canopy goes at its base, as a fraction of itself.
    /// y = how far the shading normal leans from up toward the viewer.
    pub base: Vec4,
    /// x = fade-in start, y = fade-in end, z = cull, w = base height (m).
    pub size: Vec4,
}

impl Default for ImpostorStyle {
    fn default() -> Self {
        Self {
            // LINEAR, not sRGB — and overwritten from the prop table every
            // frame anyway, so these only have to be sane for the frames
            // before a world has loaded. Authoring them here is what went
            // wrong before: the same greens converted by hand, once, and
            // then left behind when the props were retuned.
            canopy_a: Vec4::new(0.0051, 0.0223, 0.0041, 0.0),
            canopy_b: Vec4::new(0.0137, 0.0304, 0.0041, 0.0),
            base: Vec4::new(0.35, 0.5, 0.0, 0.0),
            // Real prop trees cover the first 120 m; impostors take over
            // across the next 60. Where they STOP is not authored here —
            // see `sync_impostor_style`.
            size: Vec4::new(85.0, 150.0, 4000.0, 7.0),
        }
    }
}

/// Uniform slice for the impostor shader (twin of `ImpostorEnv` in WGSL).
#[derive(ShaderType, Clone, Copy, Default)]
pub struct ImpostorEnv {
    /// x = coverage-eval flag; lighting comes from Bevy.
    flags: Vec4,
    canopy_a: Vec4,
    canopy_b: Vec4,
    base: Vec4,
    size: Vec4,
}

#[cfg(test)]
mod tests {
    /// [`FADE_FROM`] and the shader's multiplier are ONE number written
    /// twice, and the two are load-bearing in opposite directions: the
    /// host sets the cull distance to `paint_starts / FADE_FROM` so that
    /// the shader's fade-out — which begins at `cull * FADE_FROM` — lands
    /// exactly where the ground paint begins. Drift them and the
    /// impostors either fade across the wrong span or stop before the
    /// paint arrives, which is the ring of bare ground the whole
    /// arrangement exists to prevent.
    ///
    /// Every other twin in this repo has a guard (the op table, the
    /// layout regions, the water shader's world table). This one was a
    /// comment.
    #[test]
    fn the_shaders_fade_start_is_fade_from() {
        let wgsl = include_str!("voxel_impostor.wgsl");
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
            .expect("voxel_impostor.wgsl multiplies env.size.z by a literal");
        assert_eq!(
            declared,
            super::FADE_FROM,
            "voxel_impostor.wgsl and FADE_FROM disagree about where the fade starts"
        );
    }
}
