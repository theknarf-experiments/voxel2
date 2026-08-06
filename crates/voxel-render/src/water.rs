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
            binding_types::{storage_buffer_read_only_sized, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            Buffer, BufferInitDescriptor, BufferUsages, Canonical, ColorTargetState,
            ColorWrites, CompareFunction, DepthStencilState,
            FragmentState, PipelineCache, RenderPipeline, RenderPipelineDescriptor, ShaderStages,
            ShaderType, Specializer, SpecializerKey, TextureFormat, UniformBuffer, Variants,
            VertexState,
        },
        renderer::{RenderDevice, RenderQueue},
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
    },
};

const GRID_N: u64 = 192;
const SEA_SNAP: f64 = 64.0;

#[derive(ShaderType, Clone, Copy, Default)]
struct WaterParams {
    origin: Vec4,
    /// x = ocean enabled (0/1), y = river segment count.
    counts: Vec4,
}

/// One river water segment for the GPU (layout twins the WGSL RiverSeg).
#[derive(Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RiverSegGpu {
    /// a.xz | b.xz (world meters).
    pub ab: [f32; 4],
    /// half width | level at a | level at b | unused.
    pub geo: [f32; 4],
    /// river tint rgb | unused.
    pub color: [f32; 4],
}

/// River segments near the camera, maintained by the engine's streamer;
/// drawn by the water pipeline so rivers share the ocean's shading.
/// `generation` bumps on change so the render world re-uploads.
#[derive(Resource, Clone, Default)]
pub struct RiverWater {
    pub segments: Vec<RiverSegGpu>,
    pub generation: u64,
}

/// Marker entity anchoring the water draw.
#[derive(Clone, Component, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<WaterMarker>)]
pub struct WaterMarker;

/// The generator's water surface (from its `water` op): presence and sea
/// level. Runtime (not build-time) so a hot-reload can switch worlds.
#[derive(Resource, Clone, Copy, Default)]
pub struct WaterSurface {
    pub enabled: bool,
    pub level: f32,
}

fn apply_water_toggle(
    surface: Res<WaterSurface>,
    rivers: Res<RiverWater>,
    mut markers: Query<&mut Visibility, With<WaterMarker>>,
) {
    // The one water draw covers the ocean AND rivers: visible if either
    // has something to show.
    let show = surface.enabled || !rivers.segments.is_empty();
    for mut visibility in &mut markers {
        *visibility = if show {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

fn extract_water_surface(
    surface: Extract<Res<WaterSurface>>,
    rivers: Extract<Res<RiverWater>>,
    mut commands: Commands,
) {
    commands.insert_resource(**surface);
    if rivers.is_changed() {
        commands.insert_resource((**rivers).clone());
    }
}

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/voxel_water.wgsl");
        app.init_resource::<WaterSurface>()
            .init_resource::<RiverWater>()
            .add_plugins(ExtractComponentPlugin::<WaterMarker>::default())
            .add_systems(Startup, spawn_water_marker)
            .add_systems(Update, apply_water_toggle);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<PendingWaterQueues>()
            .init_resource::<WaterBindGroupRes>()
            .init_resource::<ExtractedWaterCamera>()
            .init_resource::<WaterSurface>()
            .init_resource::<RiverWater>()
            .add_render_command::<Opaque3d, DrawWaterCommands>()
            .add_systems(
                RenderStartup,
                init_water_pipeline.after(bevy::pbr::init_mesh_pipeline_view_layouts),
            )
            .add_systems(
                ExtractSchedule,
                (extract_water_camera, extract_water_surface),
            )
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
    /// Uploaded river segments (buffer, generation, count).
    river_buffer: Option<(Buffer, u64, u32)>,
}

#[derive(Resource, Default)]
struct ExtractedWaterCamera(Vec3);

fn extract_water_camera(
    cameras: Extract<Query<&GlobalTransform, crate::PlayerCameraFilter>>,
    mut out: ResMut<ExtractedWaterCamera>,
) {
    if let Some(t) = cameras.iter().next() {
        out.0 = t.translation();
    }
}

fn init_water_pipeline(
    mut commands: Commands,
    view_layouts: Res<MeshPipelineViewLayouts>,
    asset_server: Res<AssetServer>,
) {
    // Groups 0/1 are Bevy's view bind group (see `pbr_view`); ours sits in
    // the per-mesh slot at 2.
    let layout = BindGroupLayoutDescriptor::new(
        "water_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<WaterParams>(false),
                // The level's generator program (shoreline height ops);
                // shared with the chunk pipeline.
                storage_buffer_read_only_sized(false, None),
                // River water segments (RiverSeg twin).
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );
    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_water.wgsl");
    let base_descriptor = RenderPipelineDescriptor {
        label: Some("water_draw".into()),
        // Groups 0/1 are replaced per key by the specializer.
        layout: vec![layout.clone(), layout.clone(), layout.clone()],
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
        variants: Variants::new(
            WaterSpecializer {
                view_layouts: view_layouts.clone(),
            },
            base_descriptor,
        ),
    });
}

struct WaterSpecializer {
    view_layouts: MeshPipelineViewLayouts,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, SpecializerKey)]
struct WaterKey(MeshPipelineKey);

impl Specializer<RenderPipeline> for WaterSpecializer {
    type Key = WaterKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut RenderPipelineDescriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        crate::pbr_view::specialize_for_view(&self.view_layouts, key.0, descriptor);
        Ok(key)
    }
}

#[derive(Default, Deref, DerefMut, Resource)]
struct PendingWaterQueues(PendingQueues);

#[allow(clippy::too_many_arguments)]
fn prepare_water_bind_group(
    pipeline: Option<Res<WaterPipeline>>,
    camera: Res<ExtractedWaterCamera>,
    surface: Res<WaterSurface>,
    rivers: Res<RiverWater>,
    gpu: Option<Res<crate::chunks::ChunkGpuResources>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut res: ResMut<WaterBindGroupRes>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    // Upload river segments when the streamer's generation moves on.
    let rivers_changed = res
        .river_buffer
        .as_ref()
        .is_none_or(|(_, generation, _)| *generation != rivers.generation);
    if rivers_changed {
        // A dummy zeroed segment keeps the binding valid when empty (the
        // draw count is 0 then; nothing samples it).
        let dummy = [RiverSegGpu::default()];
        let contents: &[RiverSegGpu] = if rivers.segments.is_empty() {
            &dummy
        } else {
            &rivers.segments
        };
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("water_river_segs"),
            contents: bytemuck::cast_slice(contents),
            usage: BufferUsages::STORAGE,
        });
        res.river_buffer = Some((buffer, rivers.generation, rivers.segments.len() as u32));
    }
    let seg_count = res.river_buffer.as_ref().map_or(0, |(_, _, n)| *n);

    // Snap the grid origin so vertices never swim as the camera moves.
    let ox = (camera.0.x as f64 / SEA_SNAP).floor() * SEA_SNAP;
    let oz = (camera.0.z as f64 / SEA_SNAP).floor() * SEA_SNAP;
    res.params.set(WaterParams {
        origin: Vec4::new(ox as f32, surface.level, oz as f32, 0.0),
        counts: Vec4::new(
            if surface.enabled { 1.0 } else { 0.0 },
            seg_count as f32,
            0.0,
            0.0,
        ),
    });
    res.params.write_buffer(&render_device, &render_queue);
    // The chunk pipeline owns and writes the program buffer each frame.
    let program_binding = gpu.as_ref().and_then(|g| g.program_buffer.binding());
    let (Some(params_binding), Some(program_binding), Some((river_buffer, _, _))) = (
        res.params.binding(),
        program_binding,
        res.river_buffer.as_ref(),
    ) else {
        return;
    };
    let river_binding = river_buffer.as_entire_buffer_binding();
    res.bind_group = Some(render_device.create_bind_group(
        "water_bg",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((params_binding, program_binding, river_binding)),
    ));
}

fn queue_water(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<ResMut<WaterPipeline>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    views: Query<crate::pbr_view::PbrViewQuery>,
    dirty_specializations: Res<DirtySpecializations>,
    mut pending_queues: ResMut<PendingWaterQueues>,
) {
    let Some(mut pipeline) = pipeline else {
        return;
    };
    let draw_function = opaque_draw_functions.read().id::<DrawWaterCommands>();

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
        let mesh_key = crate::pbr_view::view_key(
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
                .specialize(&pipeline_cache, WaterKey(mesh_key))
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

type DrawWaterCommands = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    DrawWater,
);

struct DrawWater;

impl<P> RenderCommand<P> for DrawWater
where
    P: PhaseItem,
{
    type Param = SRes<WaterBindGroupRes>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        _: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        res: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let res = res.into_inner();
        let Some(bg) = &res.bind_group else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(2, bg, &[]);
        let cells = (GRID_N - 1) as u32;
        let river_indices = res.river_buffer.as_ref().map_or(0, |(_, _, n)| *n * 6);
        pass.draw(0..(cells * cells * 6 + river_indices), 0..1);
        RenderCommandResult::Success
    }
}
