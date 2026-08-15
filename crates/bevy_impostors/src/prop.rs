//! The instanced-prop renderer, once.
//!
//! A population is a [`Prop`] impl: the mesh, the shader, the uniform's
//! contents. Everything else — buffers, pipeline, bind group, extract,
//! prepare, queue, draw — is [`PropPlugin`], written once for every
//! population. It began life as two four-hundred-line renderers that were
//! the same code under different names, identical down to a comment in
//! one copy that still talked about the other.
//!
//! What is shared with EVERY pipeline that shades through Bevy's PBR is
//! `bevy_pbr_view`: the view-key specializer, `DrawPipeline<M>` and
//! `queue_by_marker`.

use bevy::{
    asset::AssetServer,
    core_pipeline::core_3d::{Opaque3d, CORE_3D_DEPTH_FORMAT},
    ecs::{
        query::ROQueryItem,
        system::{lifetimeless::SRes, SystemParamItem},
    },
    mesh::VertexBufferLayout,
    pbr::{MeshPipelineViewLayouts, SetMeshViewBindGroup, SetMeshViewBindingArrayBindGroup},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_phase::{
            AddRenderCommand, PhaseItem, RenderCommand, RenderCommandResult, SetItemPipeline,
            TrackedRenderPass,
        },
        render_resource::{
            binding_types::uniform_buffer, encase::internal::WriteInto, BindGroup,
            BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            BufferDescriptor, BufferInitDescriptor, BufferUsages, ColorTargetState, ColorWrites,
            CompareFunction, DepthStencilState, FragmentState, IndexFormat, PipelineCache,
            PrimitiveState, RenderPipelineDescriptor, ShaderStages, ShaderType, TextureFormat,
            UniformBuffer, VertexAttribute, VertexFormat, VertexState, VertexStepMode,
        },
        renderer::{RenderDevice, RenderQueue},
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use bevy_pbr_view as pbr_view;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Mutex;

/// One instance: a world position plus a per-point hash the population's
/// shader uses for variation (yaw, size, shape, tint, phase — its call).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct PropInstance {
    pub pos: [f32; 3],
    pub hash: u32,
}

/// Where a population's points come in: the host publishes, per SET, and
/// the renderer re-uploads only when something changed.
///
/// A set is whatever partition the host draws under different views — a
/// world, a map, a floor. Sets matter because instances are content
/// seated somewhere: drawn under the wrong view they do not merely look
/// out of place, they stand in mid-air.
///
/// Interior mutability so extraction — which runs read-only — can clear
/// the dirty flag.
#[derive(Resource)]
pub struct PropPoints<P: Prop> {
    inner: Mutex<PointsShared>,
    marker: PhantomData<fn() -> P>,
}

#[derive(Default)]
struct PointsShared {
    sets: HashMap<u32, Vec<PropInstance>>,
    dirty: bool,
}

impl<P: Prop> Default for PropPoints<P> {
    fn default() -> Self {
        Self {
            inner: Mutex::new(PointsShared::default()),
            marker: PhantomData,
        }
    }
}

impl<P: Prop> PropPoints<P> {
    /// Replace every set's points at once.
    ///
    /// Wholesale on purpose: a set that has no points RIGHT NOW must stop
    /// drawing, and a per-set API would make "this set fell silent" a
    /// case the host has to remember to express.
    pub fn replace(&self, per_set: impl IntoIterator<Item = (u32, Vec<PropInstance>)>) {
        let mut inner = self.inner.lock().unwrap();
        inner.sets.clear();
        inner.sets.extend(per_set);
        inner.dirty = true;
    }

    /// Take everything if anything changed since the last take.
    fn take_if_dirty(&self) -> Option<Vec<(u32, Vec<PropInstance>)>> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.dirty {
            return None;
        }
        inner.dirty = false;
        Some(inner.sets.iter().map(|(s, p)| (*s, p.clone())).collect())
    }
}

/// Host-writable flags folded into every population's uniform — a debug
/// overlay bit, an effect toggle; the crate gives them no meaning. Render
/// world; write it in `ExtractSchedule`.
#[derive(Resource, Default, Clone, Copy)]
pub struct PropFlags(pub Vec4);

// --- gpu plumbing ------------------------------------------------------------

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

/// One set's instances: a buffer with room to spare, and how much of it
/// is live.
pub struct InstanceBuffer {
    pub buffer: Buffer,
    /// Points the allocation can hold. Only ever grows: a population of
    /// half a million is megabytes of instances, and reallocating on
    /// every publish is a frame spike rather than a cost.
    capacity: u32,
    /// Points to draw. Zero means empty right now, which is not the same
    /// as gone — the allocation is kept.
    pub count: u32,
}

/// Every set's instances for one population.
#[derive(Default)]
pub struct InstanceBuffers(HashMap<u32, InstanceBuffer>);

impl InstanceBuffers {
    pub fn get(&self, set: u32) -> Option<&InstanceBuffer> {
        self.0.get(&set).filter(|slot| slot.count > 0)
    }

    /// Replace what every set draws. Counts are zeroed before filling
    /// rather than buffers dropped, for the same reason capacity only
    /// grows.
    pub fn publish(
        &mut self,
        label: &str,
        per_set: impl IntoIterator<Item = (u32, Vec<PropInstance>)>,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) {
        for slot in self.0.values_mut() {
            slot.count = 0;
        }
        for (set, points) in per_set {
            if points.is_empty() {
                continue;
            }
            let needed = points.len() as u32;
            let fits = self.0.get(&set).is_some_and(|s| s.capacity >= needed);
            if !fits {
                // Rounded up, so a population that grows by a few points
                // a second does not reallocate a few times a second.
                let capacity = needed.next_power_of_two();
                let buffer = device.create_buffer(&BufferDescriptor {
                    label: Some(label),
                    size: u64::from(capacity) * size_of::<PropInstance>() as u64,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.0.insert(
                    set,
                    InstanceBuffer {
                        buffer,
                        capacity,
                        count: 0,
                    },
                );
            }
            let slot = self.0.get_mut(&set).expect("present or just inserted");
            queue.write_buffer(&slot.buffer, 0, bytemuck::cast_slice(&points));
            slot.count = needed;
        }
    }
}

/// The bind group layout and base pipeline for one instanced prop.
///
/// `Env` is the renderer's own uniform — the only thing at group 2, and
/// the only part of this that differs between populations. Every vertex
/// format is a position and one spare float, which is why the layout is
/// here rather than passed in: a prop that needed a third attribute would
/// be asking for a different pipeline, not a parameter.
pub fn prop_pipeline<Env: ShaderType>(
    layout_label: &'static str,
    draw_label: &'static str,
    shader: Handle<Shader>,
    vertex_stride: u64,
) -> (BindGroupLayoutDescriptor, RenderPipelineDescriptor) {
    // Groups 0/1 are Bevy's view bind group (see `bevy_pbr_view`); ours
    // is the per-mesh slot at 2 and carries only look parameters — a prop
    // takes its light from the app's lights like any other surface.
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
                    array_stride: std::mem::size_of::<PropInstance>() as u64,
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
        // Props are thin cards and blades, visible from both sides.
        primitive: PrimitiveState {
            cull_mode: None,
            ..default()
        },
        ..default()
    };
    (layout, descriptor)
}

// --- one population ----------------------------------------------------------

/// One instanced scatter population: what a renderer of props still gets
/// to decide.
///
/// The implementing type IS the marker component — the host spawns one
/// per instance set, with whatever visibility filtering (render layers)
/// its views need. Bevy's visible-entity filtering then picks the right
/// marker per view and the draw binds that set's instances.
pub trait Prop: Component + ExtractComponent + Copy {
    /// The uniform at group 2 — the only binding that differs between
    /// populations, and a layout twin of the shader's `Env` struct.
    type Env: ShaderType + WriteInto + Default + Send + Sync + 'static;
    /// One vertex of the static mesh: position, then one spare float.
    type Vertex: Pod;
    /// The look parameters an app edits; folded into [`Prop::Env`] every
    /// frame, so they can be written by anything.
    type Style: Resource + Default;

    /// Prefix for buffer labels in a capture.
    const NAME: &'static str;
    /// A bind group layout's label has to be `&'static str`, and one
    /// formatted from [`Prop::NAME`] is not.
    const LAYOUT_LABEL: &'static str;
    const DRAW_LABEL: &'static str;

    /// Which instance set the marker being queried draws.
    fn set(&self) -> u32;
    /// `load_embedded_asset!` resolves against the CALLING file, so this
    /// cannot move into the plugin.
    fn shader(assets: &AssetServer) -> Handle<Shader>;
    fn mesh() -> (Vec<Self::Vertex>, Vec<u32>);
    fn env(flags: Vec4, style: &Self::Style) -> Self::Env;
}

/// The shared mesh, and where this population stands in each set.
#[derive(Resource)]
pub struct PropBuffers<P: Prop> {
    mesh: Option<PropMesh>,
    /// One instance buffer per set. The geometry is shared; where the
    /// props stand is not.
    instances: InstanceBuffers,
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
    commands.insert_resource(pbr_view::DrawPipeline::<P>::new(
        &view_layouts,
        layout,
        descriptor,
    ));
}

fn extract_prop_instances<P: Prop>(
    points: Extract<Res<PropPoints<P>>>,
    mut buffers: ResMut<PropBuffers<P>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(per_set) = points.take_if_dirty() else {
        return;
    };
    buffers.instances.publish(
        &format!("{}_instances", P::NAME),
        per_set,
        &render_device,
        &render_queue,
    );
}

#[allow(clippy::too_many_arguments)]
fn prepare_prop_bind_group<P: Prop>(
    pipeline: Option<Res<pbr_view::DrawPipeline<P>>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    flags: Res<PropFlags>,
    style: Res<P::Style>,
    mut env_uniform: ResMut<PropEnv<P>>,
    mut bind_group: ResMut<PropBindGroup<P>>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    env_uniform.0.set(P::env(flags.0, &style));
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
    /// The marker says which set this draw is for. Which markers a view
    /// sees is already decided — they are ordinary entities filtered by
    /// the host's visibility setup — so this only has to bind what that
    /// one asked for.
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
        let (Some(mesh), Some(slot)) = (&buffers.mesh, buffers.instances.get(marker.set())) else {
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
/// A population's own plugin embeds its shader — `embedded_asset!`
/// resolves against the calling file — adds this, and adds whatever else
/// only it has. The HOST spawns the marker entities (one per set, under
/// its own visibility scheme) and fills [`PropPoints`].
pub struct PropPlugin<P: Prop>(PhantomData<fn() -> P>);

impl<P: Prop> Default for PropPlugin<P> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<P: Prop> Plugin for PropPlugin<P> {
    fn build(&self, app: &mut App) {
        app.init_resource::<PropPoints<P>>()
            .add_plugins(ExtractComponentPlugin::<P>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<PropBuffers<P>>()
            .init_resource::<pbr_view::PendingDrawQueues<P>>()
            .init_resource::<PropBindGroup<P>>()
            .init_resource::<PropEnv<P>>()
            .init_resource::<PropFlags>()
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
                pbr_view::queue_by_marker::<P, PropCommands<P>>.in_set(RenderSystems::Queue),
            );
    }
}
