//! Multi-chunk GPU terrain pipeline (M4, fixed LOD 0).
//!
//! Per-chunk flow, spread over frames:
//!   request → density+count dispatch (into an arena slot) → counts copied to
//!   a staging ring and mapped → CPU reads exact counts → slab allocation →
//!   mesh dispatch into the slab → drawn per frame with a camera-relative
//!   offset uniform.
//!
//! The density arena slot is held from generation until meshing, then
//! recycled (slots freed during planning become reusable the *next* frame so
//! a same-frame generation can never overwrite a slot a mesh pass still
//! reads). The counts buffer serves both the count pass (slots assigned from
//! 0 upward) and the mesh pass cursors (slots from the top downward); it is
//! cleared at the start of each frame's compute work.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    camera::{
        primitives::{Aabb, Frustum},
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
    math::DVec3,
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
            Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages,
            CachedComputePipelineId, Canonical,
            ColorTargetState, ColorWrites, CompareFunction, ComputePassDescriptor,
            ComputePipelineDescriptor, DepthStencilState, DynamicUniformBuffer, FragmentState,
            IndexFormat, MapMode, PipelineCache, RenderPipeline,
            RenderPipelineDescriptor, ShaderStages, ShaderType, Specializer, SpecializerKey,
            TextureFormat, Variants, VertexAttribute, VertexFormat, VertexState, VertexStepMode,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        view::{ExtractedView, RenderVisibleEntities, ViewUniform, ViewUniformOffset, ViewUniforms},
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
    },
};

use voxel_core::csg::CsgOp;
use voxel_core::ChunkKey;

use crate::slab::{SlabAlloc, SlabAllocator};

/// Density samples per axis: 33 corners + apron covering corners -2..=35
/// (one extra low corner for coarse-parity stitching).
const SAMPLES: u32 = 38;
const CELLS: u32 = 32;
/// Compressed vertex: 12 bytes (unorm16 pos ×4 incl. pad, snorm16 oct normal).
const VERTEX_BYTES: u64 = 12;

const ARENA_SLOTS: u32 = 128;
const COUNTS_SLOTS: u32 = 128;
const GEN_BUDGET: usize = 48;
const MESH_BUDGET: usize = 64;
const STAGING_BUFFERS: usize = 3;

// --- main-world <-> render-world plumbing ------------------------------------

/// Chunk lifecycle commands from the main-world LOD controller.
#[derive(Clone, Debug)]
pub enum ChunkCommand {
    /// Generate (and mesh, if non-empty) this chunk. `show_on_ready` makes
    /// it visible as soon as it is drawable; otherwise it stays hidden until
    /// an explicit [`ChunkCommand::Show`] (ready-before-swap LOD flips).
    /// `ops` are planning-layer CSG operations applied by the density pass.
    Request {
        key: ChunkKey,
        show_on_ready: bool,
        ops: Option<Arc<Vec<CsgOp>>>,
    },
    Show(ChunkKey),
    Free(ChunkKey),
}

/// Main-world queue of chunk lifecycle commands (filled by the LOD
/// controller in voxel-engine, drained by extraction). Interior mutability
/// because extraction system params must be read-only.
#[derive(Resource, Default)]
pub struct ChunkCommandQueue {
    inner: Mutex<Vec<ChunkCommand>>,
}

impl ChunkCommandQueue {
    pub fn push(&self, command: ChunkCommand) {
        self.inner.lock().unwrap().push(command);
    }
}

/// Render→main notifications: a requested chunk became drawable (meshed) or
/// was classified empty. The LOD controller uses these for
/// ready-before-swap.
#[derive(Resource, Clone)]
pub struct ChunkReadyChannel {
    pub rx: crossbeam_channel::Receiver<ChunkKey>,
}

#[derive(Resource, Clone)]
struct ChunkReadySender(crossbeam_channel::Sender<ChunkKey>);

/// Shared render statistics for the debug HUD (written by the render world,
/// read by the main world).
#[derive(Resource, Clone, Default)]
pub struct SharedRenderStats(pub Arc<Mutex<RenderStats>>);

#[derive(Default)]
pub struct RenderStats {
    pub tracked: usize,
    pub meshed: usize,
    pub empty_classified: usize,
    pub awaiting: usize,
    pub arena_free: u32,
    pub slab_occupancy: [(u32, u32); 4],
    pub drawn: usize,
    pub culled: usize,
}

/// Marker entity that anchors the terrain draw in the render phases.
#[derive(Clone, Component, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<VoxelTerrainMarker>)]
pub struct VoxelTerrainMarker;

/// Which world the density/draw shaders generate.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum WorldKind {
    #[default]
    Planet,
    Megastructure,
}

#[derive(Default)]
pub struct VoxelChunksPlugin {
    pub world: WorldKind,
}

impl Plugin for VoxelChunksPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/voxel_terrain_density.wgsl");
        embedded_asset!(app, "shaders/voxel_mega_density.wgsl");
        embedded_asset!(app, "shaders/voxel_mesh_chunks.wgsl");
        embedded_asset!(app, "shaders/voxel_chunk_draw.wgsl");

        let (ready_tx, ready_rx) = crossbeam_channel::unbounded();
        app.init_resource::<ChunkCommandQueue>()
            .init_resource::<SharedRenderStats>()
            .insert_resource(ChunkReadyChannel { rx: ready_rx })
            .add_plugins(ExtractComponentPlugin::<VoxelTerrainMarker>::default())
            .add_systems(Startup, spawn_terrain_marker);

        let stats = app.world().resource::<SharedRenderStats>().clone();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(stats)
            .insert_resource(self.world)
            .insert_resource(ChunkReadySender(ready_tx))
            .init_resource::<ExtractedChunkCommands>()
            .init_resource::<ChunkTable>()
            .init_resource::<FrameBatches>()
            .init_resource::<VoxelDrawList>()
            .init_resource::<PendingVoxelQueues>()
            .init_resource::<ViewBindGroupRes>()
            .add_render_command::<Opaque3d, DrawVoxelChunksCommands>()
            .add_systems(RenderStartup, init_chunk_resources)
            .add_systems(ExtractSchedule, (extract_chunk_commands, extract_camera_pos))
            .add_systems(Render, plan_frame.in_set(RenderSystems::Prepare))
            .add_systems(
                Render,
                prepare_view_bind_group.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Render, queue_voxel_chunks.in_set(RenderSystems::Queue))
            .add_systems(RenderGraph, dispatch_chunk_work.before(camera_driver));
    }
}

fn spawn_terrain_marker(mut commands: Commands) {
    commands.spawn((
        VoxelTerrainMarker,
        Visibility::default(),
        Transform::default(),
        // Effectively infinite: chunk-level culling is a later milestone.
        Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(1.0e9),
        },
    ));
}

// --- render-world state ------------------------------------------------------

#[derive(Resource, Default)]
struct ExtractedChunkCommands(Vec<ChunkCommand>);

#[derive(Resource, Default)]
struct ExtractedCameraPos(DVec3);

#[derive(Resource, Default)]
struct ExtractedFrustum(Option<Frustum>);

enum ChunkState {
    /// Waiting for budget / an arena slot.
    QueuedGen,
    /// Density generated (or generating this frame); counts on their way back.
    CountsInFlight { slot: u32 },
    /// Freed while counts were in flight; readback must recycle the slot.
    Cancelled { slot: u32 },
    /// Counts known but the slab had no space; retry allocation.
    AwaitingAlloc { slot: u32, verts: u32, indices: u32 },
    /// Classified all-air/all-solid: drawable as nothing.
    Empty,
    /// In the slab and drawable.
    Meshed { alloc: SlabAlloc, index_count: u32 },
}

struct RenderChunk {
    state: ChunkState,
    visible: bool,
    show_on_ready: bool,
    /// Planning-layer ops, held until the density pass consumes them.
    ops: Option<Arc<Vec<CsgOp>>>,
}

#[derive(Resource, Default)]
struct ChunkTable {
    chunks: HashMap<ChunkKey, RenderChunk>,
    empty_classified: usize,
}

struct GenEntry {
    uniform_offset: u32,
}

struct MeshEntry {
    uniform_offset: u32,
}

#[derive(Resource, Default)]
struct FrameBatches {
    gen: Vec<GenEntry>,
    mesh: Vec<MeshEntry>,
    /// Staging buffer index the gen batch's counts get copied into.
    staging_idx: Option<usize>,
}

#[derive(Clone, Copy)]
struct DrawEntry {
    uniform_offset: u32,
    base_vertex: u32,
    first_index: u32,
    index_count: u32,
}

#[derive(Resource, Default)]
struct VoxelDrawList(Vec<DrawEntry>);

#[derive(ShaderType, Clone, Copy)]
struct ChunkParams {
    origin: Vec4,
    slot: u32,
    base_vertex: u32,
    first_index: u32,
    counts_slot: u32,
    /// Range into this frame's concatenated CSG op buffer.
    csg_offset: u32,
    csg_count: u32,
    _pad: UVec2,
}

#[derive(ShaderType, Clone, Copy)]
struct ChunkDrawUniform {
    offset: Vec4,
}

enum StagingState {
    Free,
    /// Copy recorded this frame; mapping requested next frame.
    PendingMap { entries: Vec<(ChunkKey, u32)> },
    /// map_async issued; waiting for the callback.
    Mapping { entries: Vec<(ChunkKey, u32)> },
}

struct StagingSlot {
    buffer: Buffer,
    state: StagingState,
}

#[derive(Resource)]
struct ChunkGpuResources {
    density_arena: Buffer,
    cell_scratch: Buffer,
    vertex_slab: Buffer,
    index_slab: Buffer,
    counts: Buffer,
    /// This frame's concatenated planning ops (None → bind the dummy).
    csg_buffer: Option<Buffer>,
    csg_dummy: Buffer,
    staging: Vec<StagingSlot>,
    arena_free: Vec<u32>,
    slab: SlabAllocator,
    gen_uniforms: DynamicUniformBuffer<ChunkParams>,
    draw_uniforms: DynamicUniformBuffer<ChunkDrawUniform>,
    map_tx: crossbeam_channel::Sender<usize>,
    map_rx: crossbeam_channel::Receiver<usize>,
}

#[derive(Resource)]
struct ChunkPipelines {
    gen_layout: BindGroupLayoutDescriptor,
    mesh_layout: BindGroupLayoutDescriptor,
    density: CachedComputePipelineId,
    count: CachedComputePipelineId,
    vertices: CachedComputePipelineId,
    quads: CachedComputePipelineId,
}

#[derive(Resource)]
struct ChunkDrawPipeline {
    view_layout: BindGroupLayoutDescriptor,
    chunk_layout: BindGroupLayoutDescriptor,
    variants: Variants<RenderPipeline, VoxelChunksSpecializer>,
}

#[derive(Resource, Default)]
struct ViewBindGroupRes {
    view: Option<BindGroup>,
    chunk: Option<BindGroup>,
}

fn init_chunk_resources(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    world_kind: Res<WorldKind>,
) {
    let buffer = |label: &str, size: u64, usage: BufferUsages| {
        render_device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };

    let (map_tx, map_rx) = crossbeam_channel::unbounded();
    let staging = (0..STAGING_BUFFERS)
        .map(|i| StagingSlot {
            buffer: render_device.create_buffer(&BufferDescriptor {
                label: Some(&format!("voxel_counts_staging_{i}")),
                size: COUNTS_SLOTS as u64 * 8,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            state: StagingState::Free,
        })
        .collect();

    commands.insert_resource(ChunkGpuResources {
        density_arena: buffer(
            "voxel_density_arena",
            ARENA_SLOTS as u64 * (SAMPLES as u64).pow(3) * 4,
            BufferUsages::STORAGE,
        ),
        cell_scratch: buffer(
            "voxel_cell_scratch",
            // Cells -1..=32 per axis (seam-free overlap meshing).
            ARENA_SLOTS as u64 * (CELLS as u64 + 2).pow(3) * 4,
            BufferUsages::STORAGE,
        ),
        vertex_slab: buffer(
            "voxel_vertex_slab",
            SlabAllocator::total_vertices() * VERTEX_BYTES,
            BufferUsages::STORAGE | BufferUsages::VERTEX,
        ),
        index_slab: buffer(
            "voxel_index_slab",
            // u16 indices.
            SlabAllocator::total_indices() * 2,
            BufferUsages::STORAGE | BufferUsages::INDEX,
        ),
        counts: buffer(
            "voxel_counts",
            COUNTS_SLOTS as u64 * 8,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ),
        csg_buffer: None,
        csg_dummy: render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("voxel_csg_dummy"),
            contents: bytemuck::bytes_of(&CsgOp::boxy(
                bevy::math::Vec3::ZERO,
                bevy::math::Vec3::ZERO,
                0.0,
                0,
                true,
            )),
            usage: BufferUsages::STORAGE,
        }),
        staging,
        arena_free: (0..ARENA_SLOTS).rev().collect(),
        slab: SlabAllocator::new(),
        gen_uniforms: DynamicUniformBuffer::default(),
        draw_uniforms: DynamicUniformBuffer::default(),
        map_tx,
        map_rx,
    });

    let gen_layout = BindGroupLayoutDescriptor::new(
        "voxel_gen_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, None),     // density arena
                uniform_buffer::<ChunkParams>(true),   // per-chunk params
                storage_buffer_sized(false, None),     // planning CSG ops
            ),
        ),
    );
    let mesh_layout = BindGroupLayoutDescriptor::new(
        "voxel_mesh_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, None),     // density arena
                uniform_buffer::<ChunkParams>(true),   // per-chunk params
                storage_buffer_sized(false, None),     // cell_indices scratch
                storage_buffer_sized(false, None),     // vertex slab
                storage_buffer_sized(false, None),     // index slab
                storage_buffer_sized(false, None),     // counts
            ),
        ),
    );

    let density_shader: Handle<Shader> = match *world_kind {
        WorldKind::Planet => {
            load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_terrain_density.wgsl")
        }
        WorldKind::Megastructure => {
            load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_mega_density.wgsl")
        }
    };
    let mesh_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_mesh_chunks.wgsl");

    let compute_defs: Vec<bevy::shader::ShaderDefVal> = match *world_kind {
        WorldKind::Planet => vec![],
        WorldKind::Megastructure => vec!["MEGASTRUCTURE".into()],
    };
    let compute = |label: &'static str,
                   shader: &Handle<Shader>,
                   entry: &'static str,
                   layout: &BindGroupLayoutDescriptor| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            shader_defs: compute_defs.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    };
    commands.insert_resource(ChunkPipelines {
        density: compute("voxel_density", &density_shader, "density_main", &gen_layout),
        count: compute("voxel_sn_count", &mesh_shader, "sn_count", &mesh_layout),
        vertices: compute("voxel_sn_vertices", &mesh_shader, "sn_vertices", &mesh_layout),
        quads: compute("voxel_sn_quads", &mesh_shader, "sn_quads", &mesh_layout),
        gen_layout,
        mesh_layout,
    });

    // Draw pipeline.
    let view_layout = BindGroupLayoutDescriptor::new(
        "voxel_chunks_view_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (uniform_buffer::<ViewUniform>(true),),
        ),
    );
    let chunk_layout = BindGroupLayoutDescriptor::new(
        "voxel_chunks_chunk_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (uniform_buffer::<ChunkDrawUniform>(true),),
        ),
    );
    let draw_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_chunk_draw.wgsl");
    let shader_defs = match *world_kind {
        WorldKind::Planet => vec![],
        WorldKind::Megastructure => vec!["MEGASTRUCTURE".into()],
    };
    let base_descriptor = RenderPipelineDescriptor {
        label: Some("voxel_chunks_draw".into()),
        layout: vec![view_layout.clone(), chunk_layout.clone()],
        vertex: VertexState {
            shader: draw_shader.clone(),
            shader_defs: shader_defs.clone(),
            entry_point: Some("vertex".into()),
            buffers: vec![VertexBufferLayout {
                array_stride: VERTEX_BYTES,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    VertexAttribute {
                        format: VertexFormat::Unorm16x4,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Snorm16x2,
                        offset: 8,
                        shader_location: 1,
                    },
                ],
            }],
            ..default()
        },
        fragment: Some(FragmentState {
            shader: draw_shader,
            shader_defs,
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
    commands.insert_resource(ChunkDrawPipeline {
        view_layout,
        chunk_layout,
        variants: Variants::new(VoxelChunksSpecializer, base_descriptor),
    });
}

// --- extraction --------------------------------------------------------------

fn extract_chunk_commands(
    queue: Extract<Res<ChunkCommandQueue>>,
    mut extracted: ResMut<ExtractedChunkCommands>,
) {
    extracted.0.append(&mut queue.inner.lock().unwrap());
}

fn extract_camera_pos(
    cameras: Extract<Query<(&GlobalTransform, &Frustum), With<Camera3d>>>,
    mut commands: Commands,
) {
    let (pos, frustum) = cameras
        .iter()
        .next()
        .map(|(t, f)| (t.translation().as_dvec3(), Some(*f)))
        .unwrap_or_default();
    commands.insert_resource(ExtractedCameraPos(pos));
    commands.insert_resource(ExtractedFrustum(frustum));
}

// --- planning (Prepare) ------------------------------------------------------

fn make_params(
    key: ChunkKey,
    slot: u32,
    alloc: Option<&SlabAlloc>,
    counts_slot: u32,
    csg: (u32, u32),
) -> ChunkParams {
    let origin = key.min_corner_m();
    ChunkParams {
        origin: Vec4::new(
            origin.x as f32,
            origin.y as f32,
            origin.z as f32,
            key.voxel_size_m() as f32,
        ),
        slot,
        base_vertex: alloc.map_or(0, |a| a.base_vertex),
        first_index: alloc.map_or(0, |a| a.first_index),
        counts_slot,
        csg_offset: csg.0,
        csg_count: csg.1,
        _pad: UVec2::ZERO,
    }
}

/// Generation priority: distance to the chunk's AABB normalized by its edge
/// length — coarse chunks and near chunks first, uniformly across levels.
fn gen_priority(key: ChunkKey, camera: DVec3) -> f64 {
    let min = key.min_corner_m();
    let max = min + DVec3::splat(key.edge_m());
    let closest = camera.clamp(min, max);
    camera.distance(closest) / key.edge_m()
}

#[allow(clippy::too_many_arguments)]
fn plan_frame(
    mut gpu: ResMut<ChunkGpuResources>,
    mut table: ResMut<ChunkTable>,
    mut extracted: ResMut<ExtractedChunkCommands>,
    mut batches: ResMut<FrameBatches>,
    mut draw_list: ResMut<VoxelDrawList>,
    camera: Res<ExtractedCameraPos>,
    frustum: Res<ExtractedFrustum>,
    stats: Res<SharedRenderStats>,
    ready_tx: Res<ChunkReadySender>,
    pipelines: Option<Res<ChunkPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let gpu = &mut *gpu;
    // Plan no GPU work until every compute pipeline is compiled; this keeps
    // dispatch unconditional and avoids unwinding half-planned state.
    let pipelines_ready = pipelines.is_some_and(|p| {
        [p.density, p.count, p.vertices, p.quads]
            .iter()
            .all(|id| pipeline_cache.get_compute_pipeline(*id).is_some())
    });
    batches.gen.clear();
    batches.mesh.clear();
    batches.staging_idx = None;
    gpu.gen_uniforms.clear();
    gpu.draw_uniforms.clear();
    draw_list.0.clear();

    // 0. Request mapping for staging copies recorded *last* frame. Doing
    //    this here (not in last frame's Cleanup) guarantees the copy has
    //    been submitted — mapping a buffer before its copy submits is a
    //    wgpu validation error.
    for (idx, slot) in gpu.staging.iter_mut().enumerate() {
        if matches!(slot.state, StagingState::PendingMap { .. }) {
            let StagingState::PendingMap { entries } =
                std::mem::replace(&mut slot.state, StagingState::Free)
            else {
                unreachable!()
            };
            let tx = gpu.map_tx.clone();
            slot.buffer.slice(..).map_async(MapMode::Read, move |res| {
                if res.is_ok() {
                    let _ = tx.send(idx);
                }
            });
            slot.state = StagingState::Mapping { entries };
        }
    }

    // 1. Apply commands from the LOD controller.
    for command in extracted.0.drain(..) {
        match command {
            ChunkCommand::Request { key, show_on_ready, ops } => {
                match table.chunks.get_mut(&key) {
                    None => {
                        table.chunks.insert(
                            key,
                            RenderChunk {
                                state: ChunkState::QueuedGen,
                                visible: false,
                                show_on_ready,
                                ops,
                            },
                        );
                    }
                    // Freed-while-in-flight chunk re-requested: resurrect it,
                    // the pending readback will complete it normally.
                    Some(chunk) => {
                        if let ChunkState::Cancelled { slot } = chunk.state {
                            chunk.state = ChunkState::CountsInFlight { slot };
                            chunk.visible = false;
                            chunk.show_on_ready = show_on_ready;
                        }
                    }
                }
            }
            ChunkCommand::Show(key) => {
                if let Some(chunk) = table.chunks.get_mut(&key) {
                    chunk.visible = true;
                }
            }
            ChunkCommand::Free(key) => match table.chunks.remove(&key) {
                Some(RenderChunk { state: ChunkState::CountsInFlight { slot }, .. }) => {
                    // Result still coming; readback will recycle the slot.
                    table.chunks.insert(
                        key,
                        RenderChunk {
                            state: ChunkState::Cancelled { slot },
                            visible: false,
                            show_on_ready: false,
                            ops: None,
                        },
                    );
                }
                Some(RenderChunk { state: ChunkState::AwaitingAlloc { slot, .. }, .. }) => {
                    gpu.arena_free.push(slot)
                }
                Some(RenderChunk { state: ChunkState::Meshed { alloc, .. }, .. }) => {
                    gpu.slab.free(alloc)
                }
                _ => {}
            },
        }
    }

    // 2. Drain finished count readbacks.
    while let Ok(staging_idx) = gpu.map_rx.try_recv() {
        let slot_entry = &mut gpu.staging[staging_idx];
        let StagingState::Mapping { entries } =
            std::mem::replace(&mut slot_entry.state, StagingState::Free)
        else {
            continue;
        };
        let data = slot_entry.buffer.slice(..).get_mapped_range();
        let counts: &[u32] = bytemuck::cast_slice(&data);
        for (key, counts_slot) in &entries {
            let verts = counts[(counts_slot * 2) as usize];
            let quads = counts[(counts_slot * 2 + 1) as usize];
            let table = &mut *table;
            let Some(chunk) = table.chunks.get_mut(key) else {
                continue;
            };
            match chunk.state {
                ChunkState::Cancelled { slot } => {
                    gpu.arena_free.push(slot);
                    table.chunks.remove(key);
                }
                ChunkState::CountsInFlight { slot } => {
                    let max_verts = *crate::slab::CLASS_VERTS.last().unwrap();
                    let max_indices = max_verts * crate::slab::INDEX_FACTOR;
                    let too_big = verts > max_verts || quads * 6 > max_indices;
                    if too_big {
                        warn!("chunk {key:?} exceeds largest slab class ({verts} verts); dropped");
                    }
                    if too_big || verts == 0 || quads == 0 {
                        gpu.arena_free.push(slot);
                        table.empty_classified += 1;
                        chunk.state = ChunkState::Empty;
                        let _ = ready_tx.0.send(*key);
                    } else {
                        chunk.state = ChunkState::AwaitingAlloc { slot, verts, indices: quads * 6 };
                    }
                }
                _ => {}
            }
        }
        drop(data);
        slot_entry.buffer.unmap();
    }

    // 3. Schedule mesh dispatches for chunks whose slab slot is ready.
    //    Counts-buffer cursors for meshing are assigned from the top so they
    //    never collide with this frame's count batch (assigned from 0).
    let mut mesh_counts_slot = COUNTS_SLOTS;
    let mut freed_slots = Vec::new();
    let mesh_keys: Vec<ChunkKey> = if !pipelines_ready {
        Vec::new()
    } else {
        table
            .chunks
            .iter()
            .filter(|(_, c)| matches!(c.state, ChunkState::AwaitingAlloc { .. }))
            .map(|(k, _)| *k)
            .take(MESH_BUDGET.min((COUNTS_SLOTS as usize).saturating_sub(GEN_BUDGET)))
            .collect()
    };
    for key in mesh_keys {
        let chunk = table.chunks.get_mut(&key).unwrap();
        let ChunkState::AwaitingAlloc { slot, verts, indices } = chunk.state else {
            unreachable!()
        };
        let Some(alloc) = gpu.slab.alloc(verts, indices) else {
            // Slab full: keep waiting (arena slot stays held; visible in HUD).
            continue;
        };
        mesh_counts_slot -= 1;
        let offset = gpu
            .gen_uniforms
            .push(&make_params(key, slot, Some(&alloc), mesh_counts_slot, (0, 0)));
        batches.mesh.push(MeshEntry { uniform_offset: offset });
        // The mesh compute is recorded later this frame, before the main
        // pass, so the chunk is immediately drawable.
        chunk.state = ChunkState::Meshed { alloc, index_count: indices };
        if chunk.show_on_ready {
            chunk.visible = true;
        }
        let _ = ready_tx.0.send(key);
        freed_slots.push(slot);
    }

    // 4. Schedule new density+count work if a staging buffer is available,
    //    nearest-first (normalized by chunk edge so levels interleave fairly).
    let staging_idx = if pipelines_ready {
        gpu.staging
            .iter()
            .position(|s| matches!(s.state, StagingState::Free))
    } else {
        None
    };
    if let Some(staging_idx) = staging_idx {
        let mut entries = Vec::new();
        let mut frame_ops: Vec<CsgOp> = Vec::new();
        let mut queued: Vec<(ChunkKey, f64)> = table
            .chunks
            .iter()
            .filter(|(_, c)| matches!(c.state, ChunkState::QueuedGen))
            .map(|(k, _)| (*k, gen_priority(*k, camera.0)))
            .collect();
        queued.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (key, _) in queued.into_iter().take(GEN_BUDGET) {
            let Some(slot) = gpu.arena_free.pop() else {
                break;
            };
            let chunk = table.chunks.get_mut(&key).unwrap();
            // Consume this chunk's planning ops into the frame buffer.
            let csg = match chunk.ops.take() {
                Some(ops) => {
                    let offset = frame_ops.len() as u32;
                    frame_ops.extend_from_slice(&ops);
                    (offset, ops.len() as u32)
                }
                None => (0, 0),
            };
            let counts_slot = entries.len() as u32;
            let offset = gpu
                .gen_uniforms
                .push(&make_params(key, slot, None, counts_slot, csg));
            batches.gen.push(GenEntry { uniform_offset: offset });
            entries.push((key, counts_slot));
            chunk.state = ChunkState::CountsInFlight { slot };
        }
        if !entries.is_empty() {
            batches.staging_idx = Some(staging_idx);
            gpu.staging[staging_idx].state = StagingState::PendingMap { entries };
        }
        // Upload this frame's op set (dummy is kept bound when empty).
        gpu.csg_buffer = if frame_ops.is_empty() {
            None
        } else {
            Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("voxel_csg_ops"),
                contents: bytemuck::cast_slice(&frame_ops),
                usage: BufferUsages::STORAGE,
            }))
        };
    }

    // 5. Arena slots freed by meshing become reusable next frame.
    gpu.arena_free.append(&mut freed_slots);

    // 6. Build the draw list (visible chunks only, frustum-culled) with
    //    camera-relative offsets.
    let mut culled = 0usize;
    for (key, chunk) in &table.chunks {
        let ChunkState::Meshed { alloc, index_count } = &chunk.state else {
            continue;
        };
        if !chunk.visible {
            continue;
        }
        if let Some(f) = &frustum.0 {
            // World-space AABB, inflated by the skirt depth. f32 is fine at
            // current view ranges; camera-relative culling comes with M6.
            let half = key.edge_m() * 0.5 + key.voxel_size_m() * 6.0;
            let aabb = Aabb {
                center: (key.min_corner_m() + DVec3::splat(key.edge_m() * 0.5))
                    .as_vec3()
                    .into(),
                half_extents: Vec3A::splat(half as f32),
            };
            if !f.intersects_obb_identity(&aabb) {
                culled += 1;
                continue;
            }
        }
        let rel = key.min_corner_m() - camera.0;
        let offset = gpu.draw_uniforms.push(&ChunkDrawUniform {
            offset: Vec4::new(
                rel.x as f32,
                rel.y as f32,
                rel.z as f32,
                key.voxel_size_m() as f32,
            ),
        });
        draw_list.0.push(DrawEntry {
            uniform_offset: offset,
            base_vertex: alloc.base_vertex,
            first_index: alloc.first_index,
            index_count: *index_count,
        });
    }

    gpu.gen_uniforms.write_buffer(&render_device, &render_queue);
    gpu.draw_uniforms.write_buffer(&render_device, &render_queue);

    // 7. HUD stats.
    if let Ok(mut s) = stats.0.lock() {
        s.tracked = table.chunks.len();
        s.meshed = table
            .chunks
            .values()
            .filter(|c| matches!(c.state, ChunkState::Meshed { .. }))
            .count();
        s.empty_classified = table.empty_classified;
        s.awaiting = table
            .chunks
            .values()
            .filter(|c| {
                !matches!(c.state, ChunkState::Meshed { .. } | ChunkState::Empty)
            })
            .count();
        s.arena_free = gpu.arena_free.len() as u32;
        s.slab_occupancy = gpu.slab.occupancy();
        s.drawn = draw_list.0.len();
        s.culled = culled;
    }
}

// --- compute dispatch (render graph) -----------------------------------------

fn dispatch_chunk_work(
    mut render_context: RenderContext,
    pipelines: Option<Res<ChunkPipelines>>,
    gpu: Option<ResMut<ChunkGpuResources>>,
    batches: Res<FrameBatches>,
    pipeline_cache: Res<PipelineCache>,
) {
    let (Some(pipelines), Some(mut gpu)) = (pipelines, gpu) else {
        return;
    };
    if batches.gen.is_empty() && batches.mesh.is_empty() {
        return;
    }
    // Planning is gated on pipeline readiness, so these always resolve.
    let (Some(density), Some(count), Some(vertices), Some(quads)) = (
        pipeline_cache.get_compute_pipeline(pipelines.density),
        pipeline_cache.get_compute_pipeline(pipelines.count),
        pipeline_cache.get_compute_pipeline(pipelines.vertices),
        pipeline_cache.get_compute_pipeline(pipelines.quads),
    ) else {
        return;
    };
    let gpu = &mut *gpu;

    let Some(gen_uniform_binding) = gpu.gen_uniforms.binding() else {
        return;
    };

    let csg = gpu.csg_buffer.as_ref().unwrap_or(&gpu.csg_dummy);
    let gen_bg = render_context.render_device().create_bind_group(
        "voxel_gen_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.gen_layout),
        &BindGroupEntries::sequential((
            gpu.density_arena.as_entire_buffer_binding(),
            gen_uniform_binding.clone(),
            csg.as_entire_buffer_binding(),
        )),
    );
    let mesh_bg = render_context.render_device().create_bind_group(
        "voxel_mesh_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.mesh_layout),
        &BindGroupEntries::sequential((
            gpu.density_arena.as_entire_buffer_binding(),
            gen_uniform_binding,
            gpu.cell_scratch.as_entire_buffer_binding(),
            gpu.vertex_slab.as_entire_buffer_binding(),
            gpu.index_slab.as_entire_buffer_binding(),
            gpu.counts.as_entire_buffer_binding(),
        )),
    );

    let encoder = render_context.command_encoder();
    encoder.clear_buffer(&gpu.counts, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("voxel_chunk_work"),
            ..default()
        });

        let gen_groups = SAMPLES.div_ceil(6);
        let cell_groups = CELLS / 4;
        // Count and vertex passes cover the extended cell range -1..=32.
        let ext_groups = (CELLS + 2).div_ceil(4);

        // Density for the whole gen batch, then counts.
        pass.set_pipeline(density);
        for entry in &batches.gen {
            pass.set_bind_group(0, &gen_bg, &[entry.uniform_offset]);
            pass.dispatch_workgroups(gen_groups, gen_groups, gen_groups);
        }
        pass.set_pipeline(count);
        for entry in &batches.gen {
            pass.set_bind_group(0, &mesh_bg, &[entry.uniform_offset]);
            pass.dispatch_workgroups(ext_groups, ext_groups, ext_groups);
        }

        // Mesh batch: vertices then quads.
        pass.set_pipeline(vertices);
        for entry in &batches.mesh {
            pass.set_bind_group(0, &mesh_bg, &[entry.uniform_offset]);
            pass.dispatch_workgroups(ext_groups, ext_groups, ext_groups);
        }
        pass.set_pipeline(quads);
        for entry in &batches.mesh {
            pass.set_bind_group(0, &mesh_bg, &[entry.uniform_offset]);
            pass.dispatch_workgroups(cell_groups, cell_groups, cell_groups);
        }
    }

    // Copy the gen batch's counts out for readback.
    if let Some(idx) = batches.staging_idx {
        let staging = &gpu.staging[idx].buffer;
        encoder.copy_buffer_to_buffer(&gpu.counts, 0, staging, 0, COUNTS_SLOTS as u64 * 8);
    }
}

// --- drawing -----------------------------------------------------------------

struct VoxelChunksSpecializer;

#[derive(Copy, Clone, PartialEq, Eq, Hash, SpecializerKey)]
struct VoxelChunksKey(Msaa);

impl Specializer<RenderPipeline> for VoxelChunksSpecializer {
    type Key = VoxelChunksKey;

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
struct PendingVoxelQueues(PendingQueues);

fn prepare_view_bind_group(
    view_uniforms: Res<ViewUniforms>,
    pipeline: Option<Res<ChunkDrawPipeline>>,
    gpu: Option<Res<ChunkGpuResources>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut bind_groups: ResMut<ViewBindGroupRes>,
) {
    let (Some(pipeline), Some(gpu)) = (pipeline, gpu) else {
        return;
    };
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };
    bind_groups.view = Some(render_device.create_bind_group(
        "voxel_chunks_view_bg",
        &pipeline_cache.get_bind_group_layout(&pipeline.view_layout),
        &BindGroupEntries::sequential((view_binding,)),
    ));
    bind_groups.chunk = gpu.draw_uniforms.binding().map(|b| {
        render_device.create_bind_group(
            "voxel_chunks_chunk_bg",
            &pipeline_cache.get_bind_group_layout(&pipeline.chunk_layout),
            &BindGroupEntries::sequential((b,)),
        )
    });
}

fn queue_voxel_chunks(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<ResMut<ChunkDrawPipeline>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities, &Msaa)>,
    dirty_specializations: Res<DirtySpecializations>,
    mut pending_queues: ResMut<PendingVoxelQueues>,
) {
    let Some(mut pipeline) = pipeline else {
        return;
    };
    let draw_function = opaque_draw_functions.read().id::<DrawVoxelChunksCommands>();

    for (view, view_visible_entities, msaa) in views.iter() {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let Some(visible) = view_visible_entities.get::<VoxelTerrainMarker>() else {
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
                .specialize(&pipeline_cache, VoxelChunksKey(*msaa))
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

type DrawVoxelChunksCommands = (SetItemPipeline, DrawVoxelChunks);

struct DrawVoxelChunks;

impl<P> RenderCommand<P> for DrawVoxelChunks
where
    P: PhaseItem,
{
    type Param = (
        SRes<ChunkGpuResources>,
        SRes<ViewBindGroupRes>,
        SRes<VoxelDrawList>,
    );
    type ViewQuery = &'static ViewUniformOffset;
    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        view_offset: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (gpu, bind_groups, draw_list): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu = gpu.into_inner();
        let bind_groups = bind_groups.into_inner();
        let draw_list = draw_list.into_inner();
        let (Some(view_bg), Some(chunk_bg)) = (&bind_groups.view, &bind_groups.chunk) else {
            return RenderCommandResult::Skip;
        };
        if draw_list.0.is_empty() {
            return RenderCommandResult::Success;
        }
        pass.set_bind_group(0, view_bg, &[view_offset.offset]);
        pass.set_vertex_buffer(0, gpu.vertex_slab.slice(..));
        pass.set_index_buffer(gpu.index_slab.slice(..), IndexFormat::Uint16);
        for entry in &draw_list.0 {
            pass.set_bind_group(1, chunk_bg, &[entry.uniform_offset]);
            pass.draw_indexed(
                entry.first_index..entry.first_index + entry.index_count,
                entry.base_vertex as i32,
                0..1,
            );
        }
        RenderCommandResult::Success
    }
}
