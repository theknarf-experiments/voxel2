//! Instanced tree impostors: one draw call over a crossed-quad mesh with
//! a per-instance buffer (world position + hash).
//!
//! This is how a forest gets to millions. A prop entity costs three
//! entities and a transform hierarchy, which tops out in the low
//! thousands; an impostor costs 16 bytes in a vertex buffer, so the
//! population is bounded by how many placements the scatter layer cares
//! to generate rather than by the renderer.
//!
//! Deliberately the same shape as `grass`: same bind group layout, same
//! specializer, same marker-per-world and `ItemQuery` dispatch. When a
//! third instanced population appears these two should be one pipeline
//! with a table of (class, mesh, style) — two is not yet enough evidence
//! to say what that table wants.

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
            binding_types::uniform_buffer, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor,
            BindGroupLayoutEntries, Buffer, BufferInitDescriptor, BufferUsages, Canonical,
            ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, FragmentState,
            IndexFormat, PipelineCache, PrimitiveState, RenderPipeline, RenderPipelineDescriptor,
            ShaderStages, Specializer, SpecializerKey, TextureFormat, Variants, VertexAttribute,
            VertexFormat, VertexState, VertexStepMode,
        },
        renderer::RenderDevice,
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};
use voxel_render::{ScatterPoint, ScatterPoints};

/// Marker entity anchoring one world's grass draw.
///
/// One per loaded world, each on its world's render layer, so Bevy's
/// visible-entity filtering picks the right one per view and the draw
/// command binds that world's instance buffer. Grass is world content:
/// it is seated on a heightfield, and worlds share coordinates, so the
/// wrong world's blades do not merely look out of place — they stand in
/// mid-air or bury themselves in rock.
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

pub struct ImpostorPlugin;

impl Plugin for ImpostorPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "voxel_impostor.wgsl");
        app.init_resource::<ScatterPoints>()
            .add_plugins(ExtractComponentPlugin::<ImpostorMarker>::default())
            .add_systems(Update, sync_impostor_markers);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ImpostorBuffers>()
            .init_resource::<PendingImpostorQueues>()
            .init_resource::<ImpostorBindGroupRes>()
            .init_resource::<ImpostorEnvUniform>()
            .init_resource::<ImpostorStyle>()
            .add_render_command::<Opaque3d, DrawImpostorCommands>()
            .add_systems(
                RenderStartup,
                init_impostor_pipeline.after(bevy::pbr::init_mesh_pipeline_view_layouts),
            )
            .add_systems(
                ExtractSchedule,
                (extract_impostor_instances, sync_impostor_style),
            )
            .add_systems(
                Render,
                prepare_impostor_bind_group.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Render, queue_impostor.in_set(RenderSystems::Queue));
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
                .level
                .scatter
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

/// Give every loaded world a grass anchor. Not a `Startup` one-shot: a
/// world can arrive at any time, because opening a portal loads one.
fn sync_impostor_markers(
    mut commands: Commands,
    worlds: Res<voxel_engine::Worlds>,
    // Bookkeeping, not a query: a spawn is not visible to a query until
    // commands apply, so two changes to `Worlds` before the flush would
    // give one world two markers and draw its grass twice.
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
            ImpostorMarker { world: world.id },
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

// --- tuft mesh ---------------------------------------------------------------

/// Impostor vertex: unit-quad position + a 0..1 base-to-crown factor.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ImpostorVertex {
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

// --- render-world resources --------------------------------------------------

#[derive(Resource, Default)]
struct ImpostorBuffers {
    mesh_vertices: Option<Buffer>,
    mesh_indices: Option<Buffer>,
    mesh_index_count: u32,
    /// One instance buffer per world. The tuft geometry is shared; where
    /// the tufts stand is not.
    instances: crate::instancing::InstanceBuffers,
}

#[derive(Resource)]
struct ImpostorPipeline {
    layout: BindGroupLayoutDescriptor,
    variants: Variants<RenderPipeline, ImpostorSpecializer>,
}

#[derive(Resource, Default)]
struct ImpostorBindGroupRes(Option<BindGroup>);

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
#[derive(bevy::render::render_resource::ShaderType, Clone, Copy, Default)]
struct ImpostorEnv {
    /// x = coverage-eval flag; lighting comes from Bevy.
    flags: Vec4,
    canopy_a: Vec4,
    canopy_b: Vec4,
    base: Vec4,
    size: Vec4,
}

#[derive(Resource, Default)]
struct ImpostorEnvUniform(bevy::render::render_resource::UniformBuffer<ImpostorEnv>);

fn init_impostor_pipeline(
    mut commands: Commands,
    view_layouts: Res<MeshPipelineViewLayouts>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    mut buffers: ResMut<ImpostorBuffers>,
) {
    // Static tuft geometry.
    let (verts, indices) = build_impostor();
    buffers.mesh_vertices = Some(
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("impostor_vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        }),
    );
    buffers.mesh_indices = Some(
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("impostor_indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        }),
    );
    buffers.mesh_index_count = indices.len() as u32;

    // Groups 0/1 are Bevy's view bind group (see `pbr_view`); ours is the
    // per-mesh slot at 2 and carries only look parameters — grass takes
    // its light from the app's lights like any other surface.
    let layout = BindGroupLayoutDescriptor::new(
        "impostor_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (uniform_buffer::<ImpostorEnv>(false),),
        ),
    );
    let shader: Handle<Shader> = load_embedded_asset!(asset_server.as_ref(), "voxel_impostor.wgsl");
    let base_descriptor = RenderPipelineDescriptor {
        label: Some("impostor_draw".into()),
        // Groups 0/1 are replaced per key by the specializer.
        layout: vec![layout.clone(), layout.clone(), layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            buffers: vec![
                VertexBufferLayout {
                    array_stride: std::mem::size_of::<ImpostorVertex>() as u64,
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
        // Blades are visible from both sides.
        primitive: PrimitiveState {
            cull_mode: None,
            ..default()
        },
        ..default()
    };
    commands.insert_resource(ImpostorPipeline {
        layout,
        variants: Variants::new(
            ImpostorSpecializer {
                view_layouts: view_layouts.clone(),
            },
            base_descriptor,
        ),
    });
}

fn extract_impostor_instances(
    instances: Extract<Res<ScatterPoints>>,
    mut buffers: ResMut<ImpostorBuffers>,
    render_device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
) {
    let Some(per_world) = instances.take_class_if_dirty(IMPOSTOR_CLASS) else {
        return;
    };
    buffers.instances.publish(
        "impostor_instances",
        per_world,
        &render_device,
        &render_queue,
    );
}

// --- drawing -----------------------------------------------------------------

struct ImpostorSpecializer {
    view_layouts: MeshPipelineViewLayouts,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, SpecializerKey)]
struct ImpostorKey(MeshPipelineKey);

impl Specializer<RenderPipeline> for ImpostorSpecializer {
    type Key = ImpostorKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut RenderPipelineDescriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        voxel_render::pbr_view::specialize_for_view(&self.view_layouts, key.0, descriptor);
        Ok(key)
    }
}

#[derive(Default, Deref, DerefMut, Resource)]
struct PendingImpostorQueues(PendingQueues);

#[allow(clippy::too_many_arguments)]
fn prepare_impostor_bind_group(
    pipeline: Option<Res<ImpostorPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    env: Res<voxel_render::EnvParams>,
    style: Res<ImpostorStyle>,
    mut env_uniform: ResMut<ImpostorEnvUniform>,
    mut bind_group: ResMut<ImpostorBindGroupRes>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    env_uniform.0.set(ImpostorEnv {
        flags: env.flags,
        canopy_a: style.canopy_a,
        canopy_b: style.canopy_b,
        base: style.base,
        size: style.size,
    });
    env_uniform.0.write_buffer(&render_device, &render_queue);
    let Some(env_binding) = env_uniform.0.binding() else {
        return;
    };
    bind_group.0 = Some(render_device.create_bind_group(
        "impostor_bg",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((env_binding,)),
    ));
}

fn queue_impostor(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<ResMut<ImpostorPipeline>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    views: Query<voxel_render::pbr_view::PbrViewQuery>,
    dirty_specializations: Res<DirtySpecializations>,
    mut pending_queues: ResMut<PendingImpostorQueues>,
) {
    let Some(mut pipeline) = pipeline else {
        return;
    };
    let draw_function = opaque_draw_functions.read().id::<DrawImpostorCommands>();

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
        let Some(visible) = view_visible_entities.get::<ImpostorMarker>() else {
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
                .specialize(&pipeline_cache, ImpostorKey(mesh_key))
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

type DrawImpostorCommands = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    DrawImpostor,
);

struct DrawImpostor;

impl<P> RenderCommand<P> for DrawImpostor
where
    P: PhaseItem,
{
    type Param = (SRes<ImpostorBuffers>, SRes<ImpostorBindGroupRes>);
    type ViewQuery = ();
    /// The marker says which world this draw is for. Which markers a view
    /// sees is already decided — they are ordinary entities filtered by
    /// render layer — so this only has to bind what that one asked for.
    type ItemQuery = bevy::ecs::system::lifetimeless::Read<ImpostorMarker>;

    fn render<'w>(
        _: &P,
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
        let (Some(vb), Some(ib), Some(slot)) = (
            &buffers.mesh_vertices,
            &buffers.mesh_indices,
            buffers.instances.get(marker.world),
        ) else {
            return RenderCommandResult::Success;
        };
        pass.set_bind_group(2, bg, &[]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_vertex_buffer(1, slot.buffer.slice(..));
        pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..buffers.mesh_index_count, 0, 0..slot.count);
        RenderCommandResult::Success
    }
}
