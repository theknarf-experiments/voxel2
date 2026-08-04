//! M2 prototype: one hardcoded chunk, generated and meshed entirely on the
//! GPU (sphere SDF → surface nets → indexed indirect draw). Zero CPU
//! geometry. This module exists to de-risk the Bevy render-world
//! integration; the real multi-chunk pipeline (slab allocators, batching)
//! replaces it in M4.

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    camera::{
        primitives::Aabb,
        visibility::{self, VisibilityClass},
    },
    core_pipeline::{
        core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey, CORE_3D_DEPTH_FORMAT},
        schedule::camera_driver,
    },
    ecs::{
        query::ROQueryItem,
        system::{lifetimeless::SRes, SystemParamItem},
    },
    mesh::VertexBufferLayout,
    prelude::*,
    render::{
        camera::{DirtySpecializations, PendingQueues},
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::allocator::MeshSlabs,
        render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, PhaseItem,
            RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
            ViewBinnedRenderPhases,
        },
        render_resource::{
            binding_types::{storage_buffer_sized, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            Buffer, BufferDescriptor, BufferUsages, CachedComputePipelineId, Canonical,
            ColorTargetState, ColorWrites, CompareFunction, ComputePassDescriptor,
            ComputePipelineDescriptor, DepthStencilState, FragmentState, IndexFormat,
            PipelineCache, RenderPipeline, RenderPipelineDescriptor, ShaderStages, Specializer,
            SpecializerKey, TextureFormat, Variants, VertexAttribute, VertexFormat, VertexState,
            VertexStepMode,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph},
        view::{ExtractedView, RenderVisibleEntities, ViewUniform, ViewUniformOffset, ViewUniforms},
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};

const SAMPLES: u32 = 36;
const CELLS: u32 = 32;
const MAX_VERTS: u64 = 65_536;
const MAX_INDICES: u64 = 393_216;
/// pos.xyz + normal.xyz, tightly packed f32s.
const VERTEX_FLOATS: u64 = 6;

/// Marker for the prototype chunk entity (spawned by the demo app).
#[derive(Clone, Component, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<VoxelProtoChunk>)]
pub struct VoxelProtoChunk;

pub struct VoxelPrototypePlugin;

impl Plugin for VoxelPrototypePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/voxel_density_proto.wgsl");
        embedded_asset!(app, "shaders/voxel_mesh_sn.wgsl");
        embedded_asset!(app, "shaders/voxel_chunk_draw.wgsl");

        app.add_plugins(ExtractComponentPlugin::<VoxelProtoChunk>::default())
            .add_systems(Startup, spawn_proto_chunk);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<PendingVoxelProtoQueues>()
            .add_render_command::<Opaque3d, DrawVoxelProtoCommands>()
            .add_systems(RenderStartup, init_proto_resources)
            .add_systems(
                Render,
                prepare_view_bind_group.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Render, queue_proto_chunk.in_set(RenderSystems::Queue))
            .add_systems(RenderGraph, dispatch_proto_compute.before(camera_driver));
    }
}

fn spawn_proto_chunk(mut commands: Commands) {
    // The chunk occupies world meters 0..32 on each axis.
    commands.spawn((
        VoxelProtoChunk,
        Visibility::default(),
        Transform::default(),
        Aabb {
            center: Vec3A::splat(16.0),
            half_extents: Vec3A::splat(16.0),
        },
    ));
}

// --- GPU resources -----------------------------------------------------------

#[derive(Resource)]
struct VoxelProtoBuffers {
    density: Buffer,
    cell_indices: Buffer,
    vertices: Buffer,
    indices: Buffer,
    counts: Buffer,
    indirect: Buffer,
}

#[derive(Resource)]
struct VoxelProtoPipelines {
    density_layout: BindGroupLayoutDescriptor,
    mesh_layout: BindGroupLayoutDescriptor,
    density: CachedComputePipelineId,
    sn_vertices: CachedComputePipelineId,
    sn_quads: CachedComputePipelineId,
    sn_finalize: CachedComputePipelineId,
}

#[derive(Resource)]
struct VoxelProtoDrawPipeline {
    view_layout: BindGroupLayoutDescriptor,
    variants: Variants<RenderPipeline, VoxelProtoSpecializer>,
}

#[derive(Resource, Default)]
struct VoxelProtoViewBindGroup(Option<BindGroup>);

fn init_proto_resources(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let buffer = |label, size, usage| {
        render_device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let buffers = VoxelProtoBuffers {
        density: buffer(
            "voxel_proto_density",
            (SAMPLES as u64).pow(3) * 4,
            BufferUsages::STORAGE,
        ),
        cell_indices: buffer(
            "voxel_proto_cell_indices",
            (CELLS as u64).pow(3) * 4,
            BufferUsages::STORAGE,
        ),
        vertices: buffer(
            "voxel_proto_vertices",
            MAX_VERTS * VERTEX_FLOATS * 4,
            BufferUsages::STORAGE | BufferUsages::VERTEX,
        ),
        indices: buffer(
            "voxel_proto_indices",
            MAX_INDICES * 4,
            BufferUsages::STORAGE | BufferUsages::INDEX,
        ),
        counts: buffer(
            "voxel_proto_counts",
            8,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ),
        indirect: buffer(
            "voxel_proto_indirect",
            20,
            BufferUsages::STORAGE | BufferUsages::INDIRECT,
        ),
    };
    commands.insert_resource(buffers);

    let density_layout = BindGroupLayoutDescriptor::new(
        "voxel_proto_density_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (storage_buffer_sized(false, None),),
        ),
    );
    let mesh_layout = BindGroupLayoutDescriptor::new(
        "voxel_proto_mesh_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, None), // density
                storage_buffer_sized(false, None), // cell_indices
                storage_buffer_sized(false, None), // vertices
                storage_buffer_sized(false, None), // indices
                storage_buffer_sized(false, None), // counts
                storage_buffer_sized(false, None), // indirect
            ),
        ),
    );

    let density_shader = load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_density_proto.wgsl");
    let mesh_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_mesh_sn.wgsl");

    let compute = |label: &'static str, shader: &Handle<Shader>, entry: &'static str, layout: &BindGroupLayoutDescriptor| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    };
    let pipelines = VoxelProtoPipelines {
        density: compute("voxel_proto_density", &density_shader, "density_main", &density_layout),
        sn_vertices: compute("voxel_proto_sn_vertices", &mesh_shader, "sn_vertices", &mesh_layout),
        sn_quads: compute("voxel_proto_sn_quads", &mesh_shader, "sn_quads", &mesh_layout),
        sn_finalize: compute("voxel_proto_sn_finalize", &mesh_shader, "sn_finalize", &mesh_layout),
        density_layout,
        mesh_layout,
    };
    commands.insert_resource(pipelines);

    // Draw pipeline.
    let view_layout = BindGroupLayoutDescriptor::new(
        "voxel_proto_view_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (uniform_buffer::<ViewUniform>(true),),
        ),
    );
    let draw_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_chunk_draw.wgsl");
    let base_descriptor = RenderPipelineDescriptor {
        label: Some("voxel_proto_draw".into()),
        layout: vec![view_layout.clone()],
        vertex: VertexState {
            shader: draw_shader.clone(),
            entry_point: Some("vertex".into()),
            buffers: vec![VertexBufferLayout {
                array_stride: VERTEX_FLOATS as u64 * 4,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                ],
            }],
            ..default()
        },
        fragment: Some(FragmentState {
            shader: draw_shader,
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
            // Bevy uses reversed-Z.
            depth_compare: Some(CompareFunction::GreaterEqual),
            stencil: default(),
            bias: default(),
        }),
        ..default()
    };
    commands.insert_resource(VoxelProtoDrawPipeline {
        view_layout,
        variants: Variants::new(VoxelProtoSpecializer, base_descriptor),
    });
    commands.init_resource::<VoxelProtoViewBindGroup>();
}

// --- Compute dispatch --------------------------------------------------------

fn dispatch_proto_compute(
    mut render_context: RenderContext,
    pipelines: Option<Res<VoxelProtoPipelines>>,
    buffers: Option<Res<VoxelProtoBuffers>>,
    pipeline_cache: Res<PipelineCache>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    let (Some(pipelines), Some(buffers)) = (pipelines, buffers) else {
        return;
    };
    let (Some(density), Some(verts), Some(quads), Some(finalize)) = (
        pipeline_cache.get_compute_pipeline(pipelines.density),
        pipeline_cache.get_compute_pipeline(pipelines.sn_vertices),
        pipeline_cache.get_compute_pipeline(pipelines.sn_quads),
        pipeline_cache.get_compute_pipeline(pipelines.sn_finalize),
    ) else {
        return; // pipelines still compiling
    };

    let density_bg = render_context.render_device().create_bind_group(
        "voxel_proto_density_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.density_layout),
        &BindGroupEntries::sequential((buffers.density.as_entire_buffer_binding(),)),
    );
    let mesh_bg = render_context.render_device().create_bind_group(
        "voxel_proto_mesh_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.mesh_layout),
        &BindGroupEntries::sequential((
            buffers.density.as_entire_buffer_binding(),
            buffers.cell_indices.as_entire_buffer_binding(),
            buffers.vertices.as_entire_buffer_binding(),
            buffers.indices.as_entire_buffer_binding(),
            buffers.counts.as_entire_buffer_binding(),
            buffers.indirect.as_entire_buffer_binding(),
        )),
    );

    let encoder = render_context.command_encoder();
    encoder.clear_buffer(&buffers.counts, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("voxel_proto_meshing"),
            ..default()
        });
        pass.set_bind_group(0, &density_bg, &[]);
        pass.set_pipeline(density);
        pass.dispatch_workgroups(SAMPLES / 6, SAMPLES / 6, SAMPLES / 6);

        pass.set_bind_group(0, &mesh_bg, &[]);
        pass.set_pipeline(verts);
        pass.dispatch_workgroups(CELLS / 4, CELLS / 4, CELLS / 4);
        pass.set_pipeline(quads);
        pass.dispatch_workgroups(CELLS / 4, CELLS / 4, CELLS / 4);
        pass.set_pipeline(finalize);
        pass.dispatch_workgroups(1, 1, 1);
    }
    *done = true;
    info!("voxel prototype chunk meshed on GPU");
}

// --- Drawing -----------------------------------------------------------------

struct VoxelProtoSpecializer;

#[derive(Copy, Clone, PartialEq, Eq, Hash, SpecializerKey)]
struct VoxelProtoKey(Msaa);

impl Specializer<RenderPipeline> for VoxelProtoSpecializer {
    type Key = VoxelProtoKey;

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
struct PendingVoxelProtoQueues(PendingQueues);

fn prepare_view_bind_group(
    view_uniforms: Res<ViewUniforms>,
    pipeline: Option<Res<VoxelProtoDrawPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut bind_group: ResMut<VoxelProtoViewBindGroup>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };
    bind_group.0 = Some(render_device.create_bind_group(
        "voxel_proto_view_bg",
        &pipeline_cache.get_bind_group_layout(&pipeline.view_layout),
        &BindGroupEntries::sequential((view_binding,)),
    ));
}

fn queue_proto_chunk(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<ResMut<VoxelProtoDrawPipeline>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities, &Msaa)>,
    dirty_specializations: Res<DirtySpecializations>,
    mut pending_queues: ResMut<PendingVoxelProtoQueues>,
) {
    let Some(mut pipeline) = pipeline else {
        return;
    };
    let draw_function = opaque_draw_functions.read().id::<DrawVoxelProtoCommands>();

    for (view, view_visible_entities, msaa) in views.iter() {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let Some(visible) = view_visible_entities.get::<VoxelProtoChunk>() else {
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
                .specialize(&pipeline_cache, VoxelProtoKey(*msaa))
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

type DrawVoxelProtoCommands = (SetItemPipeline, DrawVoxelProto);

struct DrawVoxelProto;

impl<P> RenderCommand<P> for DrawVoxelProto
where
    P: PhaseItem,
{
    type Param = (SRes<VoxelProtoBuffers>, SRes<VoxelProtoViewBindGroup>);
    type ViewQuery = &'static ViewUniformOffset;
    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        view_offset: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (buffers, view_bind_group): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffers = buffers.into_inner();
        let Some(view_bg) = &view_bind_group.into_inner().0 else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(0, view_bg, &[view_offset.offset]);
        pass.set_vertex_buffer(0, buffers.vertices.slice(..));
        pass.set_index_buffer(buffers.indices.slice(..), IndexFormat::Uint32);
        pass.draw_indexed_indirect(&buffers.indirect, 0);
        RenderCommandResult::Success
    }
}
