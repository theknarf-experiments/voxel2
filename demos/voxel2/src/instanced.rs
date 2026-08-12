//! The instanced-prop renderer, once.
//!
//! A prop population is a small static mesh drawn once per
//! [`ScatterPoint`], with one uniform of look parameters at group 2 and
//! Bevy's view groups at 0 and 1. Grass and tree impostors were that
//! renderer written out twice — buffers, pipeline, bind group, extract,
//! prepare, draw and marker sync, four hundred lines each, identical
//! down to a comment in the impostor copy that still talked about
//! grass, which is what copies do.
//!
//! A population is now a [`Prop`] impl: the mesh, the shader, the
//! uniform's contents and the scatter class it reads. Everything else is
//! [`PropPlugin`].
//!
//! Water is not a `Prop` — it has no mesh and no instances — but it draws
//! the same way, so it shares [`PropPipelineRes`], [`PropSpecializer`] and
//! [`queue_props`] from here.

use bevy::{
    asset::AssetServer,
    camera::primitives::Aabb,
    core_pipeline::core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey, CORE_3D_DEPTH_FORMAT},
    ecs::{
        query::ROQueryItem,
        system::{lifetimeless::SRes, SystemParamItem},
    },
    mesh::VertexBufferLayout,
    pbr::{
        MeshPipelineKey, MeshPipelineViewLayouts, SetMeshViewBindGroup,
        SetMeshViewBindingArrayBindGroup,
    },
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
            binding_types::uniform_buffer, encase::internal::WriteInto, BindGroup,
            BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            BufferInitDescriptor, BufferUsages, Canonical, ColorTargetState, ColorWrites,
            CompareFunction, DepthStencilState, FragmentState, IndexFormat, PipelineCache,
            PrimitiveState, RenderPipeline, RenderPipelineDescriptor, ShaderStages, ShaderType,
            Specializer, SpecializerKey, TextureFormat, UniformBuffer, Variants, VertexAttribute,
            VertexFormat, VertexState, VertexStepMode,
        },
        renderer::{RenderDevice, RenderQueue},
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use bytemuck::Pod;
use std::marker::PhantomData;
use voxel_render::{ScatterPoint, ScatterPoints};

/// Replaces groups 0 and 1 with the view's own, per pipeline key.
pub struct PropSpecializer {
    pub view_layouts: MeshPipelineViewLayouts,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, SpecializerKey)]
pub struct PropKey(pub MeshPipelineKey);

impl Specializer<RenderPipeline> for PropSpecializer {
    type Key = PropKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut RenderPipelineDescriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        voxel_render::pbr_view::specialize_for_view(&self.view_layouts, key.0, descriptor);
        Ok(key)
    }
}

/// A static mesh on the GPU, and how many indices it has.
pub struct PropMesh {
    pub vertices: Buffer,
    pub indices: Buffer,
    pub index_count: u32,
}

/// Upload one prop's geometry. It never changes, so this runs once.
pub fn upload_mesh<V: Pod>(
    device: &RenderDevice,
    name: &str,
    vertices: &[V],
    indices: &[u32],
) -> PropMesh {
    PropMesh {
        vertices: device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(&format!("{name}_vertices")),
            contents: bytemuck::cast_slice(vertices),
            usage: BufferUsages::VERTEX,
        }),
        indices: device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(&format!("{name}_indices")),
            contents: bytemuck::cast_slice(indices),
            usage: BufferUsages::INDEX,
        }),
        index_count: indices.len() as u32,
    }
}

/// The bind group layout and base pipeline for one instanced prop.
///
/// `Env` is the renderer's own uniform — the only thing at group 2, and
/// the only part of this that differs between them. Both vertex formats
/// are a position and one spare float, which is why the layout is here
/// rather than passed in: a prop that needed a third attribute would be
/// asking for a different pipeline, not a parameter.
/// The labels are parameters rather than derived from one name because a
/// bind group layout's has to be `&'static str`, and a formatted one is
/// not.
pub fn prop_pipeline<Env: ShaderType>(
    layout_label: &'static str,
    draw_label: &'static str,
    shader: Handle<Shader>,
    vertex_stride: u64,
) -> (BindGroupLayoutDescriptor, RenderPipelineDescriptor) {
    // Groups 0/1 are Bevy's view bind group (see `pbr_view`); ours is the
    // per-mesh slot at 2 and carries only look parameters — a prop takes
    // its light from the app's lights like any other surface.
    let layout = BindGroupLayoutDescriptor::new(
        layout_label,
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (uniform_buffer::<Env>(false),),
        ),
    );
    let descriptor = RenderPipelineDescriptor {
        label: Some(draw_label.into()),
        // Groups 0/1 are replaced per key by the specializer.
        layout: vec![layout.clone(), layout.clone(), layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            buffers: vec![
                VertexBufferLayout {
                    array_stride: vertex_stride,
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
                    array_stride: std::mem::size_of::<ScatterPoint>() as u64,
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
        // Both props are flat cards, visible from both sides.
        primitive: PrimitiveState {
            cull_mode: None,
            ..default()
        },
        ..default()
    };
    (layout, descriptor)
}

/// A pipeline to draw one marker's population with.
///
/// Keyed by the marker rather than by the renderer, so a renderer that
/// builds its own descriptor (water) still gets the resource, the queue
/// system and the specializer from here.
#[derive(Resource)]
pub struct PropPipelineRes<M> {
    pub layout: BindGroupLayoutDescriptor,
    pub variants: Variants<RenderPipeline, PropSpecializer>,
    marker: PhantomData<fn() -> M>,
}

impl<M> PropPipelineRes<M> {
    pub fn new(
        view_layouts: &MeshPipelineViewLayouts,
        layout: BindGroupLayoutDescriptor,
        descriptor: RenderPipelineDescriptor,
    ) -> Self {
        Self {
            layout,
            variants: Variants::new(
                PropSpecializer {
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
pub struct PendingPropQueues<M> {
    #[deref]
    pub queues: PendingQueues,
    marker: std::marker::PhantomData<fn() -> M>,
}

impl<M> Default for PendingPropQueues<M> {
    fn default() -> Self {
        Self {
            queues: PendingQueues::default(),
            marker: std::marker::PhantomData,
        }
    }
}

/// Put every marker this view can see into the opaque phase.
///
/// Grass, impostors and water had this system three times, at
/// seventy-six lines each and byte-identical bar the names. What differs
/// between them is only WHICH entities to look for, WHICH pipeline to
/// specialize and WHICH draw to run — so those are the three parameters,
/// and the rest is written once.
pub fn queue_props<M, D>(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<ResMut<PropPipelineRes<M>>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    views: Query<voxel_render::pbr_view::PbrViewQuery>,
    dirty_specializations: Res<DirtySpecializations>,
    mut pending_queues: ResMut<PendingPropQueues<M>>,
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
        let mesh_key = voxel_render::pbr_view::view_key(
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
                .specialize(&pipeline_cache, PropKey(mesh_key))
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

// --- one population ----------------------------------------------------------

/// One instanced scatter population: what a renderer of props still gets
/// to decide.
///
/// The implementing type IS the marker component — one entity per loaded
/// world, on that world's render layer, so Bevy's visible-entity
/// filtering picks the right one per view and the draw binds that world's
/// instances. Props are world content: they are seated on a heightfield
/// and worlds share coordinates, so the wrong world's do not merely look
/// out of place, they stand in mid-air or bury themselves in rock.
pub trait Prop: Component + ExtractComponent + Copy {
    /// The uniform at group 2 — the only binding that differs between
    /// populations, and a layout twin of the shader's `Env` struct.
    type Env: ShaderType + WriteInto + Default + Send + Sync + 'static;
    /// One vertex of the static mesh: position, then one spare float.
    type Vertex: Pod;
    /// The look parameters an app edits; folded into [`Prop::Env`] every
    /// frame, so they can be written by anything (see the impostors'
    /// palette, taken from the props they stand in for).
    type Style: Resource + Default;

    /// The scatter class the level publishes this population's points
    /// under. A name the level and the demo agree on — the engine never
    /// sees it.
    const CLASS: &'static str;
    /// Prefix for buffer labels in a capture.
    const NAME: &'static str;
    /// A bind group layout's label has to be `&'static str`, and one
    /// formatted from [`Prop::NAME`] is not.
    const LAYOUT_LABEL: &'static str;
    const DRAW_LABEL: &'static str;

    fn anchor(world: voxel_engine::WorldId) -> Self;
    fn world(&self) -> voxel_engine::WorldId;
    /// `load_embedded_asset!` resolves against the CALLING file, so this
    /// cannot move here.
    fn shader(assets: &AssetServer) -> Handle<Shader>;
    fn mesh() -> (Vec<Self::Vertex>, Vec<u32>);
    fn env(flags: Vec4, style: &Self::Style) -> Self::Env;
}

/// The shared mesh, and where this population stands in each world.
#[derive(Resource)]
pub struct PropBuffers<P: Prop> {
    mesh: Option<PropMesh>,
    /// One instance buffer per world. The geometry is shared; where the
    /// props stand is not.
    instances: crate::instancing::InstanceBuffers,
    marker: PhantomData<fn() -> P>,
}

impl<P: Prop> Default for PropBuffers<P> {
    fn default() -> Self {
        Self {
            mesh: None,
            instances: default(),
            marker: PhantomData,
        }
    }
}

#[derive(Resource)]
pub struct PropBindGroup<P: Prop>(Option<BindGroup>, PhantomData<fn() -> P>);

impl<P: Prop> Default for PropBindGroup<P> {
    fn default() -> Self {
        Self(None, PhantomData)
    }
}

#[derive(Resource)]
pub struct PropEnv<P: Prop>(UniformBuffer<P::Env>, PhantomData<fn() -> P>);

impl<P: Prop> Default for PropEnv<P> {
    fn default() -> Self {
        Self(default(), PhantomData)
    }
}

/// Give every loaded world an anchor. Not a `Startup` one-shot: a world
/// can arrive at any time, because opening a portal loads one.
fn sync_prop_markers<P: Prop>(
    mut commands: Commands,
    worlds: Res<voxel_engine::Worlds>,
    // Bookkeeping, not a query: a spawn is not visible to a query until
    // commands apply, so two changes to `Worlds` before the flush would
    // give one world two markers and draw its props twice.
    mut spawned: Local<std::collections::HashSet<voxel_engine::WorldId>>,
) {
    if !worlds.is_changed() {
        return;
    }
    for world in worlds.iter() {
        if !spawned.insert(world.id) {
            continue;
        }
        commands.spawn((
            P::anchor(world.id),
            crate::OfWorld::scene(world.id),
            Visibility::default(),
            Transform::default(),
            Aabb {
                center: Vec3A::ZERO,
                half_extents: Vec3A::splat(1.0e9),
            },
        ));
    }
}

fn init_prop_pipeline<P: Prop>(
    mut commands: Commands,
    view_layouts: Res<MeshPipelineViewLayouts>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    mut buffers: ResMut<PropBuffers<P>>,
) {
    let (verts, indices) = P::mesh();
    buffers.mesh = Some(upload_mesh(&render_device, P::NAME, &verts, &indices));

    let (layout, descriptor) = prop_pipeline::<P::Env>(
        P::LAYOUT_LABEL,
        P::DRAW_LABEL,
        P::shader(&asset_server),
        std::mem::size_of::<P::Vertex>() as u64,
    );
    commands.insert_resource(PropPipelineRes::<P>::new(&view_layouts, layout, descriptor));
}

fn extract_prop_instances<P: Prop>(
    instances: Extract<Res<ScatterPoints>>,
    mut buffers: ResMut<PropBuffers<P>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(per_world) = instances.take_class_if_dirty(P::CLASS) else {
        return;
    };
    buffers.instances.publish(
        &format!("{}_instances", P::NAME),
        per_world,
        &render_device,
        &render_queue,
    );
}

#[allow(clippy::too_many_arguments)]
fn prepare_prop_bind_group<P: Prop>(
    pipeline: Option<Res<PropPipelineRes<P>>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    env: Res<voxel_render::EnvParams>,
    style: Res<P::Style>,
    mut env_uniform: ResMut<PropEnv<P>>,
    mut bind_group: ResMut<PropBindGroup<P>>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    env_uniform.0.set(P::env(env.flags, &style));
    env_uniform.0.write_buffer(&render_device, &render_queue);
    let Some(env_binding) = env_uniform.0.binding() else {
        return;
    };
    bind_group.0 = Some(render_device.create_bind_group(
        format!("{}_bg", P::NAME).as_str(),
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((env_binding,)),
    ));
}

pub struct DrawProp<P: Prop>(PhantomData<fn() -> P>);

impl<P: Prop, I: PhaseItem> RenderCommand<I> for DrawProp<P> {
    type Param = (SRes<PropBuffers<P>>, SRes<PropBindGroup<P>>);
    type ViewQuery = ();
    /// The marker says which world this draw is for. Which markers a view
    /// sees is already decided — they are ordinary entities filtered by
    /// render layer — so this only has to bind what that one asked for.
    type ItemQuery = bevy::ecs::system::lifetimeless::Read<P>;

    fn render<'w>(
        _: &I,
        _: ROQueryItem<'w, '_, Self::ViewQuery>,
        marker: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (buffers, bind_group): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffers = buffers.into_inner();
        let Some(marker) = marker else {
            return RenderCommandResult::Skip;
        };
        let Some(bg) = &bind_group.into_inner().0 else {
            return RenderCommandResult::Skip;
        };
        let (Some(mesh), Some(slot)) = (&buffers.mesh, buffers.instances.get(marker.world()))
        else {
            return RenderCommandResult::Success;
        };
        pass.set_bind_group(2, bg, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, slot.buffer.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.index_count, 0, 0..slot.count);
        RenderCommandResult::Success
    }
}

type PropCommands<P> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    DrawProp<P>,
);

/// Everything a [`Prop`] population needs to draw.
///
/// A renderer's own plugin embeds its shader — `embedded_asset!` resolves
/// against the calling file — adds this, and adds whatever else only it
/// has.
pub struct PropPlugin<P: Prop>(PhantomData<fn() -> P>);

impl<P: Prop> Default for PropPlugin<P> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<P: Prop> Plugin for PropPlugin<P> {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScatterPoints>()
            .add_plugins(ExtractComponentPlugin::<P>::default())
            .add_systems(Update, sync_prop_markers::<P>);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<PropBuffers<P>>()
            .init_resource::<PendingPropQueues<P>>()
            .init_resource::<PropBindGroup<P>>()
            .init_resource::<PropEnv<P>>()
            .init_resource::<P::Style>()
            .add_render_command::<Opaque3d, PropCommands<P>>()
            .add_systems(
                RenderStartup,
                init_prop_pipeline::<P>.after(bevy::pbr::init_mesh_pipeline_view_layouts),
            )
            .add_systems(ExtractSchedule, extract_prop_instances::<P>)
            .add_systems(
                Render,
                prepare_prop_bind_group::<P>.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                queue_props::<P, PropCommands<P>>.in_set(RenderSystems::Queue),
            );
    }
}
