//! The DEMO's water look: a procedural ocean and its rivers.
//!
//! This is host content, not engine code. One draw call, no buffers — the
//! vertex shader builds a camera-following power-warped grid from
//! `vertex_index` and displaces it with analytic waves; shorelines come
//! from replaying the world's height ops per fragment. Every constant
//! here (wave sizes, tints, foam, roughness) is art direction.
//!
//! The engine supplies only generic ribbon data and the pieces a host
//! needs to shade like the terrain:
//! `voxel_render::pbr_view` for Bevy's view bind group and the world
//! program buffer for the shoreline.

/// One ribbon segment as the water shader wants it (layout twins the
/// WGSL `RiverSeg`). The engine hands out `RibbonSeg`s; turning them into
/// GPU data is the host's business, because the layout belongs to the
/// host's shader.
#[derive(Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RiverSegGpu {
    /// a.xz | b.xz (world meters).
    pub ab: [f32; 4],
    /// half width | level at a | level at b | unused.
    pub geo: [f32; 4],
    /// tint rgb | unused.
    pub color: [f32; 4],
}

/// One world's ribbon segments near the camera, maintained by the demo's
/// streamer. `generation` bumps on change so the render world re-uploads.
#[derive(Clone, Default)]
pub struct WorldRivers {
    pub segments: Vec<RiverSegGpu>,
    pub generation: u64,
}

/// Every loaded world's rivers, keyed by world.
///
/// Worlds share coordinates, so one flat list was not merely the wrong
/// world's rivers — it laid a second level's courses across the ground of
/// whichever level you were standing in, at that level's heights.
#[derive(Resource, Clone, Default)]
pub struct RiverWater(pub HashMap<voxel_engine::WorldId, WorldRivers>);

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    camera::{
        primitives::Aabb,
        visibility::{self, VisibilityClass},
    },
    core_pipeline::core_3d::{Opaque3d, CORE_3D_DEPTH_FORMAT},
    ecs::{
        query::ROQueryItem,
        system::{lifetimeless::SRes, SystemParamItem},
    },
    pbr::{MeshPipelineViewLayouts, SetMeshViewBindGroup, SetMeshViewBindingArrayBindGroup},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_phase::{
            AddRenderCommand, PhaseItem, RenderCommand, RenderCommandResult, SetItemPipeline,
            TrackedRenderPass,
        },
        render_resource::{
            binding_types::{storage_buffer_read_only_sized, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            BufferInitDescriptor, BufferUsages, ColorTargetState, ColorWrites, CompareFunction,
            DepthStencilState, FragmentState, PipelineCache, RenderPipelineDescriptor,
            ShaderStages, ShaderType, TextureFormat, UniformBuffer, VertexState,
        },
        renderer::{RenderDevice, RenderQueue},
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use std::collections::HashMap;

const GRID_N: u64 = 192;
const SEA_SNAP: f64 = 64.0;

#[derive(ShaderType, Clone, Copy, Default)]
struct WaterParams {
    origin: Vec4,
    /// x = ocean enabled (0/1), y = river segment count, z = world index
    /// into the shared program buffer (the shoreline reads that world's
    /// height ops).
    counts: Vec4,
}

/// Marker entity anchoring ONE world's water draw.
///
/// One per loaded world, each on its world's render layer, so a view sees
/// only the water of the world it is looking at and the draw command
/// binds that world's parameters and river segments.
#[derive(Clone, Copy, Component, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<WaterMarker>)]
pub struct WaterMarker {
    pub world: voxel_engine::WorldId,
    /// Sea level, or `None` in a world with no ocean — its rivers still
    /// draw through the same pipeline.
    pub sea_level: Option<f32>,
}

/// Give every loaded world a water anchor, and keep its sea level and
/// visibility in step with the scene it is dressed by.
///
/// Not a `Startup` one-shot: opening a portal loads a world at any time.
fn sync_water_markers(
    mut commands: Commands,
    scenes: Res<crate::WorldScenes>,
    rivers: Res<RiverWater>,
    mut markers: Query<(&mut WaterMarker, &mut Visibility)>,
    // Bookkeeping, not the query: a spawn is not visible to a query until
    // commands apply, so a second change before the flush would give one
    // world two surfaces drawing over each other.
    mut spawned: Local<std::collections::HashSet<voxel_engine::WorldId>>,
) {
    if !scenes.is_changed() && !rivers.is_changed() {
        return;
    }
    for (world, scene) in &scenes.0 {
        // The one water draw covers the ocean AND rivers: visible if
        // either has something to show.
        let has_rivers = rivers.0.get(world).is_some_and(|r| !r.segments.is_empty());
        let want = if scene.sea_level.is_some() || has_rivers {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        match markers.iter_mut().find(|(m, _)| m.world == *world) {
            Some((mut marker, mut visibility)) => {
                if marker.sea_level != scene.sea_level {
                    marker.sea_level = scene.sea_level;
                }
                if *visibility != want {
                    *visibility = want;
                }
            }
            None => {
                if !spawned.insert(*world) {
                    continue;
                }
                commands.spawn((
                    WaterMarker {
                        world: *world,
                        sea_level: scene.sea_level,
                    },
                    crate::OfWorld::scene(*world),
                    want,
                    Transform::default(),
                    Aabb {
                        center: Vec3A::ZERO,
                        half_extents: Vec3A::splat(1.0e9),
                    },
                ));
            }
        }
    }
}

fn extract_water_rivers(rivers: Extract<Res<RiverWater>>, mut commands: Commands) {
    if rivers.is_changed() {
        commands.insert_resource((**rivers).clone());
    }
}

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "voxel_water.wgsl");
        app.init_resource::<RiverWater>()
            .add_plugins(ExtractComponentPlugin::<WaterMarker>::default())
            .add_systems(Update, sync_water_markers);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<voxel_render::pbr_view::PendingDrawQueues<WaterMarker>>()
            .init_resource::<WaterGpu>()
            .init_resource::<ExtractedWaterCamera>()
            .init_resource::<RiverWater>()
            .add_render_command::<Opaque3d, DrawWaterCommands>()
            .add_systems(
                RenderStartup,
                init_water_pipeline.after(bevy::pbr::init_mesh_pipeline_view_layouts),
            )
            .add_systems(
                ExtractSchedule,
                (extract_water_camera, extract_water_rivers),
            )
            .add_systems(
                Render,
                prepare_water_bind_group.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                voxel_render::pbr_view::queue_by_marker::<WaterMarker, DrawWaterCommands>
                    .in_set(RenderSystems::Queue),
            );
    }
}

/// What one world's water draw binds.
#[derive(Default)]
struct WorldWaterGpu {
    bind_group: Option<BindGroup>,
    params: UniformBuffer<WaterParams>,
    /// Uploaded river segments (buffer, generation, count).
    river_buffer: Option<(Buffer, u64, u32)>,
}

#[derive(Resource, Default)]
struct WaterGpu(HashMap<voxel_engine::WorldId, WorldWaterGpu>);

#[derive(Resource, Default)]
struct ExtractedWaterCamera(Vec3);

fn extract_water_camera(
    cameras: Extract<Query<&GlobalTransform, voxel_render::PlayerCameraFilter>>,
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
    let shader: Handle<Shader> = load_embedded_asset!(asset_server.as_ref(), "voxel_water.wgsl");
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
    commands.insert_resource(voxel_render::pbr_view::DrawPipeline::<WaterMarker>::new(
        &view_layouts,
        layout,
        base_descriptor,
    ));
}

#[allow(clippy::too_many_arguments)]
fn prepare_water_bind_group(
    pipeline: Option<Res<voxel_render::pbr_view::DrawPipeline<WaterMarker>>>,
    camera: Res<ExtractedWaterCamera>,
    markers: Query<&WaterMarker>,
    rivers: Res<RiverWater>,
    gpu: Option<Res<voxel_render::ChunkGpuResources>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut res: ResMut<WaterGpu>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    // The chunk pipeline owns and writes the program buffer each frame; it
    // holds every world's ops, and `WaterParams.counts.z` says which slice
    // this world's shoreline reads.
    let Some(program_binding) = gpu.as_ref().and_then(|g| g.program_buffer.binding()) else {
        return;
    };
    let live: std::collections::HashSet<_> = markers.iter().map(|m| m.world).collect();
    res.0.retain(|world, _| live.contains(world));

    for marker in &markers {
        let world = res.0.entry(marker.world).or_default();
        let rivers = rivers.0.get(&marker.world).cloned().unwrap_or_default();
        // Upload river segments when this world's generation moves on.
        let stale = world
            .river_buffer
            .as_ref()
            .is_none_or(|(_, generation, _)| *generation != rivers.generation);
        if stale {
            // A dummy zeroed segment keeps the binding valid when empty
            // (the draw count is 0 then; nothing samples it).
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
            world.river_buffer = Some((buffer, rivers.generation, rivers.segments.len() as u32));
        }
        let seg_count = world.river_buffer.as_ref().map_or(0, |(_, _, n)| *n);

        // Snap the grid origin so vertices never swim as the camera moves.
        let ox = (camera.0.x as f64 / SEA_SNAP).floor() * SEA_SNAP;
        let oz = (camera.0.z as f64 / SEA_SNAP).floor() * SEA_SNAP;
        world.params.set(WaterParams {
            origin: Vec4::new(ox as f32, marker.sea_level.unwrap_or(0.0), oz as f32, 0.0),
            counts: Vec4::new(
                if marker.sea_level.is_some() { 1.0 } else { 0.0 },
                seg_count as f32,
                f32::from(marker.world),
                0.0,
            ),
        });
        world.params.write_buffer(&render_device, &render_queue);
        let (Some(params_binding), Some((river_buffer, _, _))) =
            (world.params.binding(), world.river_buffer.as_ref())
        else {
            continue;
        };
        let river_binding = river_buffer.as_entire_buffer_binding();
        world.bind_group = Some(render_device.create_bind_group(
            "water_bg",
            &pipeline_cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((params_binding, program_binding.clone(), river_binding)),
        ));
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
    type Param = SRes<WaterGpu>;
    type ViewQuery = ();
    /// The marker says which world this draw is for; which markers a view
    /// sees is already decided by render layer.
    type ItemQuery = bevy::ecs::system::lifetimeless::Read<WaterMarker>;

    fn render<'w>(
        _: &P,
        _: ROQueryItem<'w, '_, Self::ViewQuery>,
        marker: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        res: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(marker) = marker else {
            return RenderCommandResult::Skip;
        };
        let Some(world) = res.into_inner().0.get(&marker.world) else {
            return RenderCommandResult::Skip;
        };
        let Some(bg) = &world.bind_group else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(2, bg, &[]);
        let cells = (GRID_N - 1) as u32;
        let river_indices = world.river_buffer.as_ref().map_or(0, |(_, _, n)| *n * 6);
        pass.draw(0..(cells * cells * 6 + river_indices), 0..1);
        RenderCommandResult::Success
    }
}

#[cfg(test)]
mod tests {
    /// The world table's LENGTH is part of the layout twin, not a detail
    /// of it: `ops` begins after the table, so a shader that declares a
    /// shorter one reads every op at the wrong index. The water shader
    /// declared 4 while the engine wrote 8, which silently moved the
    /// shoreline's height replay off the front of the op list.
    #[test]
    fn the_water_shaders_world_table_is_as_long_as_the_engine_writes() {
        let wgsl = include_str!("voxel_water.wgsl");
        let declared = wgsl
            .split("array<WorldHeader,")
            .nth(1)
            .and_then(|rest| rest.split('>').next())
            .and_then(|n| n.trim().parse::<usize>().ok())
            .expect("voxel_water.wgsl declares array<WorldHeader, N>");
        assert_eq!(
            declared,
            voxel_render::MAX_WORLDS,
            "voxel_water.wgsl's world table must match voxel_render::MAX_WORLDS"
        );
    }

    /// Every world's ops live in one buffer, so the shoreline has to be
    /// told which slice to replay rather than assuming the first.
    #[test]
    fn the_shoreline_reads_the_worlds_own_ops() {
        let wgsl = include_str!("voxel_water.wgsl");
        assert!(
            !wgsl.contains("prog.worlds[0]"),
            "the shoreline must index by the world in params, not world 0"
        );
        assert!(wgsl.contains("prog.worlds[u32(params.counts.z)]"));
    }
}
