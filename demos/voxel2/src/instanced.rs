//! What grass and impostors are both made of.
//!
//! Two renderers of the same shape: a small static mesh drawn once per
//! [`ScatterPoint`], one uniform of look parameters at group 2, and
//! Bevy's view groups at 0 and 1. They were a hundred and ten identical
//! lines each — identical down to a comment in the impostor copy that
//! still talked about grass, which is what copies do.
//!
//! What is NOT here is anything either of them decides: the mesh, the
//! shader, the uniform's contents and when to re-upload instances are
//! per-renderer and stay where they are read.

use bevy::{
    core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT,
    mesh::VertexBufferLayout,
    pbr::{MeshPipelineKey, MeshPipelineViewLayouts},
    prelude::*,
    render::{
        render_resource::{
            binding_types::uniform_buffer, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            Buffer, BufferInitDescriptor, BufferUsages, Canonical, ColorTargetState, ColorWrites,
            CompareFunction, DepthStencilState, FragmentState, PrimitiveState, RenderPipeline,
            RenderPipelineDescriptor, ShaderStages, ShaderType, Specializer, SpecializerKey,
            TextureFormat, VertexAttribute, VertexFormat, VertexState, VertexStepMode,
        },
        renderer::RenderDevice,
    },
};
use bytemuck::Pod;
use voxel_render::ScatterPoint;

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
