//! Shared plumbing for pipelines that shade through Bevy's PBR.
//!
//! Our draws are not `Mesh3d` entities, but they can still use Bevy's
//! view data: groups 0 and 1 are the mesh view bind group (lights,
//! cascaded shadow maps, clusters, fog, tonemapping LUTs), bound with
//! the public `SetMeshViewBindGroup` / `SetMeshViewBindingArrayBindGroup`
//! render commands. Group 2 is where Bevy puts per-mesh data, so each
//! pipeline puts its own uniform there and leaves the material group at
//! 3 free.
//!
//! The catch is that the view *layout*, the shader defs and the color
//! target format must all agree with the bind group Bevy built for that
//! view. Bevy derives all three from one `MeshPipelineKey`, so we derive
//! ours from the same key, the same way.

use bevy::{
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    ecs::query::Has,
    light::ShadowFilteringMethod,
    pbr::{
        tonemapping_pipeline_key, MeshPipelineKey, MeshPipelineViewLayoutKey,
        MeshPipelineViewLayouts,
    },
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_resource::RenderPipelineDescriptor,
        view::{ExtractedView, RenderVisibleEntities},
    },
    shader::ShaderDefVal,
};

/// Everything a pipeline key depends on, mirroring the inputs of Bevy's
/// own view key so both stay in step.
pub(crate) type PbrViewQuery = (
    &'static ExtractedView,
    Option<&'static ExtractedCamera>,
    &'static RenderVisibleEntities,
    &'static Msaa,
    Option<&'static Tonemapping>,
    Option<&'static DebandDither>,
    Option<&'static ShadowFilteringMethod>,
    Has<DistanceFog>,
);

/// Build the key for a view exactly as `bevy_pbr` does.
pub(crate) fn view_key(
    view: &ExtractedView,
    camera: Option<&ExtractedCamera>,
    msaa: &Msaa,
    tonemapping: Option<&Tonemapping>,
    dither: Option<&DebandDither>,
    shadow_filter_method: Option<&ShadowFilteringMethod>,
    distance_fog: bool,
) -> MeshPipelineKey {
    let mut key = MeshPipelineKey::from_msaa_samples(msaa.samples())
        | MeshPipelineKey::from_target_format(view.target_format);
    if !camera.is_some_and(|camera| camera.hdr) {
        if let Some(tonemapping) = tonemapping {
            key |= MeshPipelineKey::TONEMAP_IN_SHADER;
            key |= tonemapping_pipeline_key(*tonemapping);
        }
        if let Some(DebandDither::Enabled) = dither {
            key |= MeshPipelineKey::DEBAND_DITHER;
        }
    }
    if distance_fog {
        key |= MeshPipelineKey::DISTANCE_FOG;
    }
    match shadow_filter_method.copied().unwrap_or_default() {
        ShadowFilteringMethod::Hardware2x2 => {
            key |= MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2;
        }
        ShadowFilteringMethod::Gaussian => {
            key |= MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN;
        }
        ShadowFilteringMethod::Temporal => {
            key |= MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL;
        }
    }
    key
}

/// Point a descriptor's groups 0/1 at Bevy's view layouts for this key and
/// give it the matching shader defs and color target. Descriptors must
/// already have three layout slots; the first two are overwritten.
pub(crate) fn specialize_for_view(
    layouts: &MeshPipelineViewLayouts,
    key: MeshPipelineKey,
    descriptor: &mut RenderPipelineDescriptor,
) {
    descriptor.multisample.count = key.msaa_samples();

    let view_layout = layouts.get_view_layout(MeshPipelineViewLayoutKey::from(key));
    descriptor.layout[0] = view_layout.main_layout.clone();
    descriptor.layout[1] = view_layout.binding_array_layout.clone();

    let defs = shader_defs(key);
    descriptor.vertex.shader_defs = defs.clone();
    if let Some(fragment) = descriptor.fragment.as_mut() {
        fragment.shader_defs = defs;
        if let Some(Some(target)) = fragment.targets.first_mut() {
            target.format = key.target_format();
        }
    }
}

/// The subset of Bevy's mesh shader defs that our shaders and the view
/// bindings they touch depend on.
fn shader_defs(key: MeshPipelineKey) -> Vec<ShaderDefVal> {
    let mut defs = vec![
        // `bevy_pbr::pbr_functions` pulls in the material bindings even
        // though we never sample them; the group index has to resolve.
        ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 3),
    ];
    if key.msaa_samples() > 1 {
        defs.push("MULTISAMPLED".into());
    }
    if key.contains(MeshPipelineKey::DISTANCE_FOG) {
        defs.push("DISTANCE_FOG".into());
    }
    // Without a filter method `sample_shadow_map` returns 0 — every lit
    // surface reads as fully shadowed.
    let filter = key.intersection(MeshPipelineKey::SHADOW_FILTER_METHOD_RESERVED_BITS);
    if filter == MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2 {
        defs.push("SHADOW_FILTER_METHOD_HARDWARE_2X2".into());
    } else if filter == MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL {
        defs.push("SHADOW_FILTER_METHOD_TEMPORAL".into());
    } else {
        defs.push("SHADOW_FILTER_METHOD_GAUSSIAN".into());
    }
    if key.contains(MeshPipelineKey::TONEMAP_IN_SHADER) {
        defs.push("TONEMAP_IN_SHADER".into());
        defs.push(ShaderDefVal::UInt(
            "TONEMAPPING_LUT_TEXTURE_BINDING_INDEX".into(),
            bevy::pbr::TONEMAPPING_LUT_TEXTURE_BINDING_INDEX,
        ));
        defs.push(ShaderDefVal::UInt(
            "TONEMAPPING_LUT_SAMPLER_BINDING_INDEX".into(),
            bevy::pbr::TONEMAPPING_LUT_SAMPLER_BINDING_INDEX,
        ));
        defs.push(tonemap_method_def(key));
        // Debanding is tied to tonemapping in the shader.
        if key.contains(MeshPipelineKey::DEBAND_DITHER) {
            defs.push("DEBAND_DITHER".into());
        }
    }
    defs
}

/// The tonemapping method def matching the key's reserved bits — the LUT
/// path in `main_pass_post_lighting_processing` switches on it.
fn tonemap_method_def(key: MeshPipelineKey) -> ShaderDefVal {
    let method = key.intersection(MeshPipelineKey::TONEMAP_METHOD_RESERVED_BITS);
    if method == MeshPipelineKey::TONEMAP_METHOD_NONE {
        "TONEMAP_METHOD_NONE".into()
    } else if method == MeshPipelineKey::TONEMAP_METHOD_REINHARD {
        "TONEMAP_METHOD_REINHARD".into()
    } else if method == MeshPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE {
        "TONEMAP_METHOD_REINHARD_LUMINANCE".into()
    } else if method == MeshPipelineKey::TONEMAP_METHOD_ACES_FITTED {
        "TONEMAP_METHOD_ACES_FITTED".into()
    } else if method == MeshPipelineKey::TONEMAP_METHOD_AGX {
        "TONEMAP_METHOD_AGX".into()
    } else if method == MeshPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM {
        "TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM".into()
    } else if method == MeshPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC {
        "TONEMAP_METHOD_BLENDER_FILMIC".into()
    } else if method == MeshPipelineKey::TONEMAP_METHOD_PBR_NEUTRAL {
        "TONEMAP_METHOD_PBR_NEUTRAL".into()
    } else {
        "TONEMAP_METHOD_TONY_MC_MAPFACE".into()
    }
}
