//! Instanced grass rendering: one draw call over a procedural blade-tuft
//! mesh with a per-instance buffer (world position + hash). Instances are
//! produced by the main world (vegetation streaming) and re-uploaded only
//! when the tile set changes.

use std::sync::Mutex;

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
    mesh::VertexBufferLayout,
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
            BindGroupLayoutEntries, Buffer, BufferInitDescriptor, BufferUsages, Canonical,
            ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, FragmentState,
            IndexFormat, PipelineCache, PrimitiveState, RenderPipeline, RenderPipelineDescriptor,
            ShaderStages, Specializer, SpecializerKey, TextureFormat, Variants, VertexAttribute,
            VertexFormat, VertexState, VertexStepMode,
        },
        renderer::RenderDevice,
        view::{
            ExtractedView, RenderVisibleEntities, ViewUniform, ViewUniformOffset, ViewUniforms,
        },
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};

/// One grass tuft instance.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GrassInstance {
    pub pos: [f32; 3],
    pub hash: u32,
}

/// Main-world resource: the current instance set plus a dirty flag.
/// Interior mutability so extraction (read-only) can clear the flag.
#[derive(Resource, Default)]
pub struct GrassInstances {
    inner: Mutex<GrassShared>,
}

#[derive(Default)]
struct GrassShared {
    instances: Vec<GrassInstance>,
    dirty: bool,
}

impl GrassInstances {
    pub fn set(&self, instances: Vec<GrassInstance>) {
        let mut inner = self.inner.lock().unwrap();
        inner.instances = instances;
        inner.dirty = true;
    }
}

/// Marker entity anchoring the grass draw.
#[derive(Clone, Component, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<GrassMarker>)]
pub struct GrassMarker;

pub struct GrassPlugin;

impl Plugin for GrassPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/voxel_grass.wgsl");
        app.init_resource::<GrassInstances>()
            .add_plugins(ExtractComponentPlugin::<GrassMarker>::default())
            .add_systems(Startup, spawn_grass_marker);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<GrassBuffers>()
            .init_resource::<PendingGrassQueues>()
            .init_resource::<GrassBindGroupRes>()
            .init_resource::<GrassEnvUniform>()
            .add_render_command::<Opaque3d, DrawGrassCommands>()
            .add_systems(RenderStartup, init_grass_pipeline)
            .add_systems(ExtractSchedule, extract_grass_instances)
            .add_systems(
                Render,
                prepare_grass_bind_group.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Render, queue_grass.in_set(RenderSystems::Queue));
    }
}

fn spawn_grass_marker(mut commands: Commands) {
    commands.spawn((
        GrassMarker,
        Visibility::default(),
        Transform::default(),
        Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(1.0e9),
        },
    ));
}

// --- tuft mesh ---------------------------------------------------------------

/// Blade vertex: position + tip factor.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct BladeVertex {
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

// --- render-world resources --------------------------------------------------

#[derive(Resource, Default)]
struct GrassBuffers {
    tuft_vertices: Option<Buffer>,
    tuft_indices: Option<Buffer>,
    tuft_index_count: u32,
    instances: Option<Buffer>,
    instance_count: u32,
}

#[derive(Resource)]
struct GrassPipeline {
    layout: BindGroupLayoutDescriptor,
    variants: Variants<RenderPipeline, GrassSpecializer>,
}

#[derive(Resource, Default)]
struct GrassBindGroupRes(Option<BindGroup>);

/// Level environment slice for the grass shader (sun + haze).
#[derive(bevy::render::render_resource::ShaderType, Clone, Copy, Default)]
struct GrassEnv {
    sun_dir: Vec4,
    haze: Vec4,
    haze_tint: Vec4,
}

#[derive(Resource, Default)]
struct GrassEnvUniform(bevy::render::render_resource::UniformBuffer<GrassEnv>);

fn init_grass_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    mut buffers: ResMut<GrassBuffers>,
) {
    // Static tuft geometry.
    let (verts, indices) = build_tuft();
    buffers.tuft_vertices = Some(
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("grass_tuft_vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        }),
    );
    buffers.tuft_indices = Some(
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("grass_tuft_indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        }),
    );
    buffers.tuft_index_count = indices.len() as u32;

    let layout = BindGroupLayoutDescriptor::new(
        "grass_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer::<GlobalsUniform>(false),
                uniform_buffer::<GrassEnv>(false), // level sun + haze
            ),
        ),
    );
    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_grass.wgsl");
    let base_descriptor = RenderPipelineDescriptor {
        label: Some("grass_draw".into()),
        layout: vec![layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            buffers: vec![
                VertexBufferLayout {
                    array_stride: std::mem::size_of::<BladeVertex>() as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                },
                VertexBufferLayout {
                    array_stride: std::mem::size_of::<GrassInstance>() as u64,
                    step_mode: VertexStepMode::Instance,
                    attributes: vec![
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 2,
                        },
                        VertexAttribute {
                            format: VertexFormat::Uint32,
                            offset: 12,
                            shader_location: 3,
                        },
                    ],
                },
            ],
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
        // Blades are visible from both sides.
        primitive: PrimitiveState {
            cull_mode: None,
            ..default()
        },
        ..default()
    };
    commands.insert_resource(GrassPipeline {
        layout,
        variants: Variants::new(GrassSpecializer, base_descriptor),
    });
}

fn extract_grass_instances(
    instances: Extract<Res<GrassInstances>>,
    mut buffers: ResMut<GrassBuffers>,
    render_device: Res<RenderDevice>,
) {
    let mut inner = instances.inner.lock().unwrap();
    if !inner.dirty {
        return;
    }
    inner.dirty = false;
    buffers.instance_count = inner.instances.len() as u32;
    if inner.instances.is_empty() {
        buffers.instances = None;
        return;
    }
    buffers.instances = Some(
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("grass_instances"),
            contents: bytemuck::cast_slice(&inner.instances),
            usage: BufferUsages::VERTEX,
        }),
    );
}

// --- drawing -----------------------------------------------------------------

struct GrassSpecializer;

#[derive(Copy, Clone, PartialEq, Eq, Hash, SpecializerKey)]
struct GrassKey(Msaa);

impl Specializer<RenderPipeline> for GrassSpecializer {
    type Key = GrassKey;

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
struct PendingGrassQueues(PendingQueues);

#[allow(clippy::too_many_arguments)]
fn prepare_grass_bind_group(
    view_uniforms: Res<ViewUniforms>,
    globals: Res<GlobalsBuffer>,
    pipeline: Option<Res<GrassPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    env: Res<crate::chunks::EnvParams>,
    mut env_uniform: ResMut<GrassEnvUniform>,
    mut bind_group: ResMut<GrassBindGroupRes>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    env_uniform.0.set(GrassEnv {
        sun_dir: env.sun_dir,
        haze: env.haze,
        haze_tint: env.haze_tint,
    });
    env_uniform.0.write_buffer(&render_device, &render_queue);
    let (Some(view_binding), Some(globals_binding), Some(env_binding)) = (
        view_uniforms.uniforms.binding(),
        globals.buffer.binding(),
        env_uniform.0.binding(),
    ) else {
        return;
    };
    bind_group.0 = Some(render_device.create_bind_group(
        "grass_bg",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((view_binding, globals_binding, env_binding)),
    ));
}

fn queue_grass(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<ResMut<GrassPipeline>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities, &Msaa)>,
    dirty_specializations: Res<DirtySpecializations>,
    mut pending_queues: ResMut<PendingGrassQueues>,
) {
    let Some(mut pipeline) = pipeline else {
        return;
    };
    let draw_function = opaque_draw_functions.read().id::<DrawGrassCommands>();

    for (view, view_visible_entities, msaa) in views.iter() {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let Some(visible) = view_visible_entities.get::<GrassMarker>() else {
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
                .specialize(&pipeline_cache, GrassKey(*msaa))
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

type DrawGrassCommands = (SetItemPipeline, DrawGrass);

struct DrawGrass;

impl<P> RenderCommand<P> for DrawGrass
where
    P: PhaseItem,
{
    type Param = (SRes<GrassBuffers>, SRes<GrassBindGroupRes>);
    type ViewQuery = &'static ViewUniformOffset;
    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        view_offset: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (buffers, bind_group): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffers = buffers.into_inner();
        let Some(bg) = &bind_group.into_inner().0 else {
            return RenderCommandResult::Skip;
        };
        let (Some(vb), Some(ib), Some(inst)) = (
            &buffers.tuft_vertices,
            &buffers.tuft_indices,
            &buffers.instances,
        ) else {
            return RenderCommandResult::Success;
        };
        if buffers.instance_count == 0 {
            return RenderCommandResult::Success;
        }
        pass.set_bind_group(0, bg, &[view_offset.offset]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_vertex_buffer(1, inst.slice(..));
        pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..buffers.tuft_index_count, 0, 0..buffers.instance_count);
        RenderCommandResult::Success
    }
}
