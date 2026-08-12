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
    core_pipeline::{
        core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey},
        tonemapping::{DebandDither, Tonemapping},
    },
    ecs::query::Has,
    light::ShadowFilteringMethod,
    pbr::{
        tonemapping_pipeline_key, MeshPipelineKey, MeshPipelineViewLayoutKey,
        MeshPipelineViewLayouts,
    },
    prelude::*,
    render::{
        camera::{DirtySpecializations, ExtractedCamera, PendingQueues},
        mesh::allocator::MeshSlabs,
        render_phase::{
            BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, ViewBinnedRenderPhases,
        },
        render_resource::{
            Canonical, RenderPipeline, RenderPipelineDescriptor, Specializer, SpecializerKey,
            Variants,
        },
        view::{ExtractedView, RenderVisibleEntities},
    },
    shader::ShaderDefVal,
};
use std::marker::PhantomData;

/// Everything a pipeline key depends on, mirroring the inputs of Bevy's
/// own view key so both stay in step.
pub type PbrViewQuery = (
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
pub fn view_key(
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
pub fn specialize_for_view(
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
        // `bevy_pbr::pbr_functions` pulls in `pbr_bindings`, which declares
        // a `StandardMaterial` at this group. We never sample it (so naga
        // strips it), but it must not land on group 3, where our own
        // material lives.
        ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 9),
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

// --- drawing through it ------------------------------------------------------

/// Replaces a descriptor's groups 0 and 1 with the view's own, per key.
///
/// One specializer for every pipeline that shades this way: terrain
/// chunks, water, and each instanced prop population had a copy under
/// four names, and a copy is a chance for one of them to stop agreeing
/// with the view bind group Bevy built.
pub struct ViewSpecializer {
    pub view_layouts: MeshPipelineViewLayouts,
}

/// The whole key: a specialized pipeline differs only in what the view
/// bind group contains, and Bevy derives all of that from this.
#[derive(Copy, Clone, PartialEq, Eq, Hash, SpecializerKey)]
pub struct ViewKey(pub MeshPipelineKey);

impl Specializer<RenderPipeline> for ViewSpecializer {
    type Key = ViewKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut RenderPipelineDescriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        specialize_for_view(&self.view_layouts, key.0, descriptor);
        Ok(key)
    }
}

/// The pipeline that draws one marker's entities, and the layout of the
/// bind group it wants at group 2.
///
/// Keyed by the MARKER rather than by the renderer, so a renderer that
/// builds its own descriptor still gets the resource and
/// [`queue_by_marker`] without declaring a struct of its own. Four of
/// them declared this same pair of fields.
#[derive(Resource)]
pub struct DrawPipeline<M> {
    pub layout: bevy::render::render_resource::BindGroupLayoutDescriptor,
    pub variants: Variants<RenderPipeline, ViewSpecializer>,
    marker: PhantomData<fn() -> M>,
}

impl<M> DrawPipeline<M> {
    pub fn new(
        view_layouts: &MeshPipelineViewLayouts,
        layout: bevy::render::render_resource::BindGroupLayoutDescriptor,
        descriptor: RenderPipelineDescriptor,
    ) -> Self {
        Self {
            layout,
            variants: Variants::new(
                ViewSpecializer {
                    view_layouts: view_layouts.clone(),
                },
                descriptor,
            ),
            marker: PhantomData,
        }
    }
}

/// Per-view queue bookkeeping, one set per marker type.
#[derive(Resource, Deref, DerefMut)]
pub struct PendingDrawQueues<M> {
    #[deref]
    pub queues: PendingQueues,
    marker: PhantomData<fn() -> M>,
}

impl<M> Default for PendingDrawQueues<M> {
    fn default() -> Self {
        Self {
            queues: PendingQueues::default(),
            marker: PhantomData,
        }
    }
}

/// Put every `M` a view can see into the opaque phase, drawn by `D`.
///
/// This was four copies of seventy-odd lines — terrain, grass, impostors,
/// water — differing only in which entities a view should look for, which
/// pipeline to specialize and which draw to run. Those are the two type
/// parameters and the marker on the pipeline.
pub fn queue_by_marker<M, D>(
    pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
    pipeline: Option<ResMut<DrawPipeline<M>>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    views: Query<PbrViewQuery>,
    dirty_specializations: Res<DirtySpecializations>,
    mut pending_queues: ResMut<PendingDrawQueues<M>>,
) where
    M: Component,
    D: 'static,
{
    let Some(mut pipeline) = pipeline else {
        return;
    };
    let draw_function = opaque_draw_functions.read().id::<D>();

    for (
        view,
        camera,
        view_visible_entities,
        msaa,
        tonemapping,
        dither,
        shadow_filter_method,
        distance_fog,
    ) in views.iter()
    {
        let mesh_key = view_key(
            view,
            camera,
            msaa,
            tonemapping,
            dither,
            shadow_filter_method,
            distance_fog,
        );
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let Some(visible) = view_visible_entities.get::<M>() else {
            continue;
        };
        let view_pending = pending_queues.prepare_for_new_frame(view.retained_view_entity);

        for &main_entity in
            dirty_specializations.iter_to_dequeue(view.retained_view_entity, visible)
        {
            opaque_phase.remove(main_entity);
        }
        for (render_entity, main_entity) in dirty_specializations.iter_to_queue(
            view.retained_view_entity,
            visible,
            &view_pending.prev_frame,
        ) {
            let Ok(pipeline_id) = pipeline
                .variants
                .specialize(&pipeline_cache, ViewKey(mesh_key))
            else {
                continue;
            };
            opaque_phase.add(
                Opaque3dBatchSetKey {
                    draw_function,
                    pipeline: pipeline_id,
                    material_bind_group_index: None,
                    lightmap_slab: None,
                    slabs: MeshSlabs::default(),
                },
                Opaque3dBinKey {
                    asset_id: AssetId::<Mesh>::invalid().untyped(),
                },
                (*render_entity, *main_entity),
                InputUniformIndex::default(),
                BinnedRenderPhaseType::NonMesh,
            );
        }
    }
}
