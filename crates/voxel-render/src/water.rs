//! Procedural ocean: one draw call, no buffers — the vertex shader builds a
//! camera-following power-warped grid from `vertex_index` and displaces it
//! with analytic waves. Shorelines/foam come from evaluating the coarse
//! terrain heightfield per fragment.

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    camera::{
        primitives::Aabb,
        visibility::{self, VisibilityClass},
    },
    core_pipeline::core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey, CORE_3D_DEPTH_FORMAT},
    ecs::{
        query::ROQueryItem,
        system::{lifetimeless::SRes, SystemParamItem},
    },
    prelude::*,
    render::{
        camera::{DirtySpecializations, PendingQueues},
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        globals::{GlobalsBuffer, GlobalsUniform},
        mesh::allocator::MeshSlabs,
        render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, PhaseItem,
            RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
            ViewBinnedRenderPhases,
        },
        render_resource::{
            binding_types::uniform_buffer, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor,
            BindGroupLayoutEntries, Canonical, ColorTargetState, ColorWrites, CompareFunction,
            DepthStencilState, FragmentState, PipelineCache, RenderPipeline,
            RenderPipelineDescriptor, ShaderStages, ShaderType, Specializer, SpecializerKey,
            TextureFormat, UniformBuffer, Variants, VertexState,
        },
        renderer::{RenderDevice, RenderQueue},
        view::{
            ExtractedView, RenderVisibleEntities, ViewUniform, ViewUniformOffset, ViewUniforms,
        },
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
    },
};

const GRID_N: u64 = 192;
const SEA_SNAP: f64 = 64.0;

#[derive(ShaderType, Clone, Copy, Default)]
struct WaterParams {
    origin: Vec4,
}

/// Marker entity anchoring the water draw.
#[derive(Clone, Component, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<WaterMarker>)]
pub struct WaterMarker;

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/voxel_water.wgsl");
        app.add_plugins(ExtractComponentPlugin::<WaterMarker>::default())
            .add_systems(Startup, spawn_water_marker);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<PendingWaterQueues>()
            .init_resource::<WaterBindGroupRes>()
            .init_resource::<ExtractedWaterCamera>()
            .add_render_command::<Opaque3d, DrawWaterCommands>()
            .add_systems(RenderStartup, init_water_pipeline)
            .add_systems(ExtractSchedule, extract_water_camera)
            .add_systems(
                Render,
                prepare_water_bind_group.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Render, queue_water.in_set(RenderSystems::Queue));
    }
}

fn spawn_water_marker(mut commands: Commands) {
    commands.spawn((
        WaterMarker,
        Visibility::default(),
        Transform::default(),
        Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(1.0e9),
        },
    ));
}

#[derive(Resource)]
struct WaterPipeline {
    layout: BindGroupLayoutDescriptor,
    variants: Variants<RenderPipeline, WaterSpecializer>,
}

#[derive(Resource, Default)]
struct WaterBindGroupRes {
    bind_group: Option<BindGroup>,
    params: UniformBuffer<WaterParams>,
    tuning: UniformBuffer<crate::chunks::WorldTuning>,
}

#[derive(Resource, Default)]
struct ExtractedWaterCamera(Vec3);

fn extract_water_camera(
    cameras: Extract<Query<&GlobalTransform, With<Camera3d>>>,
    mut out: ResMut<ExtractedWaterCamera>,
) {
    if let Some(t) = cameras.iter().next() {
        out.0 = t.translation();
    }
}

fn init_water_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let layout = BindGroupLayoutDescriptor::new(
        "water_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer::<GlobalsUniform>(false),
                uniform_buffer::<WaterParams>(false),
                uniform_buffer::<crate::chunks::WorldTuning>(false),
            ),
        ),
    );
    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_water.wgsl");
    let base_descriptor = RenderPipelineDescriptor {
        label: Some("water_draw".into()),
        layout: vec![layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        depth_stencil: Some(DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::GreaterEqual),
            stencil: default(),
            bias: default(),
        }),
        ..default()
    };
    commands.insert_resource(WaterPipeline {
        layout,
        variants: Variants::new(WaterSpecializer, base_descriptor),
    });
}

struct WaterSpecializer;

#[derive(Copy, Clone, PartialEq, Eq, Hash, SpecializerKey)]
struct WaterKey(Msaa);

impl Specializer<RenderPipeline> for WaterSpecializer {
    type Key = WaterKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut RenderPipelineDescriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        descriptor.multisample.count = key.0.samples();
        Ok(key)
    }
}

#[derive(Default, Deref, DerefMut, Resource)]
struct PendingWaterQueues(PendingQueues);

#[allow(clippy::too_many_arguments)]
fn prepare_water_bind_group(
    view_uniforms: Res<ViewUniforms>,
    globals: Res<GlobalsBuffer>,
    pipeline: Option<Res<WaterPipeline>>,
    camera: Res<ExtractedWaterCamera>,
    tuning: Res<crate::chunks::WorldTuning>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut res: ResMut<WaterBindGroupRes>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    let (Some(view_binding), Some(globals_binding)) =
        (view_uniforms.uniforms.binding(), globals.buffer.binding())
    else {
        return;
    };
    // Snap the grid origin so vertices never swim as the camera moves.
    let ox = (camera.0.x as f64 / SEA_SNAP).floor() * SEA_SNAP;
    let oz = (camera.0.z as f64 / SEA_SNAP).floor() * SEA_SNAP;
    res.params.set(WaterParams {
        origin: Vec4::new(ox as f32, 0.0, oz as f32, 0.0),
    });
    res.params.write_buffer(&render_device, &render_queue);
    res.tuning.set(*tuning);
    res.tuning.write_buffer(&render_device, &render_queue);
    let (Some(params_binding), Some(tuning_binding)) = (res.params.binding(), res.tuning.binding())
    else {
        return;
    };
    res.bind_group = Some(render_device.create_bind_group(
        "water_bg",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            view_binding,
            globals_binding,
            params_binding,
            tuning_binding,
        )),
    ));
}

fn queue_water(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<ResMut<WaterPipeline>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities, &Msaa)>,
    dirty_specializations: Res<DirtySpecializations>,
    mut pending_queues: ResMut<PendingWaterQueues>,
) {
    let Some(mut pipeline) = pipeline else {
        return;
    };
    let draw_function = opaque_draw_functions.read().id::<DrawWaterCommands>();

    for (view, view_visible_entities, msaa) in views.iter() {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let Some(visible) = view_visible_entities.get::<WaterMarker>() else {
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
                .specialize(&pipeline_cache, WaterKey(*msaa))
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

type DrawWaterCommands = (SetItemPipeline, DrawWater);

struct DrawWater;

impl<P> RenderCommand<P> for DrawWater
where
    P: PhaseItem,
{
    type Param = SRes<WaterBindGroupRes>;
    type ViewQuery = &'static ViewUniformOffset;
    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        view_offset: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        res: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(bg) = &res.into_inner().bind_group else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(0, bg, &[view_offset.offset]);
        let cells = (GRID_N - 1) as u32;
        pass.draw(0..(cells * cells * 6), 0..1);
        RenderCommandResult::Success
    }
}
