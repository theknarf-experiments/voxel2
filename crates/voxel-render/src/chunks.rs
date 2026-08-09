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
    pbr::{
        MeshPipelineKey, MeshPipelineViewLayouts, SetMaterialBindGroup, SetMeshViewBindGroup,
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
            binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            AsBindGroup, BufferDescriptor, BufferInitDescriptor, BufferUsages,
            CachedComputePipelineId,
            Canonical, ColorTargetState, ColorWrites, CompareFunction, ComputePassDescriptor,
            ComputePipelineDescriptor, DepthStencilState, DynamicUniformBuffer, FragmentState,
            Face, IndexFormat, MapMode, PipelineCache, PrimitiveState, RenderPipeline,
            RenderPipelineDescriptor,
            ShaderStages, ShaderType, Specializer, SpecializerKey, StorageBuffer, TextureFormat,
            UniformBuffer, Variants, VertexAttribute, VertexFormat, VertexState, VertexStepMode,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},

        Extract, Render, RenderApp, RenderStartup, RenderSystems,
    },
};

use voxel_core::csg::CsgOp;
use voxel_core::ChunkKey;

use crate::material::VoxelSurfaceMaterial;
use crate::slab::{SlabAlloc, SlabAllocator};

/// Density samples per axis: 33 corners + apron covering corners -2..=35
/// (one extra low corner for coarse-parity stitching).
const SAMPLES: u32 = 38;

/// Samples one density invocation walks down its column. Twin of
/// `Y_PER_THREAD` in `voxel_world_density.wgsl`: the height chain is
/// evaluated once per run, so this trades redundant work against how
/// many threads are in flight.
const DENSITY_Y_PER_THREAD: u32 = 4;
const CELLS: u32 = 32;
/// Compressed vertex: 12 bytes (unorm16 pos ×4 incl. pad, snorm16 oct normal).
const VERTEX_BYTES: u64 = 12;

// Depth of the pipeline, in chunks. Doubling these used to make a cold
// start WORSE, because the streamer fed the pipeline a level at a time
// and drained it before refilling: extra slots only sat empty for longer.
// Once the LOD pass stopped waiting per level and kept the queue full,
// the same doubling started paying (`gen_starved` 35 -> 15). The cost is
// GPU memory — the density and cell arenas are sized from ARENA_SLOTS, so
// this is ~96 MB more.
const ARENA_SLOTS: u32 = 512;
const COUNTS_SLOTS: u32 = 512;
// Per frame, keeping each frame's GPU batch small enough not to blow a
// ~8 ms vsync slot (spiky batches read as missed-vsync 17 ms frames even
// when average load is fine).
//
// MEASURED at 45 m/s, vsync off, median of five samples. The batch size
// is the whole difference between a smooth frame and a stutter, and it
// costs nothing to make it small:
//
//   budget   p50    p95    worst   awaiting   settle
//      320  14.1   57.6     80.7          0    2.20s
//       96  17.7   32.6     38.6          0    2.13s
//       48  16.4   26.4     31.3          0    2.25s
//       24  13.4   21.1     28.9        500    2.1s
//       12  11.7   19.0     25.0       2157      -
//
// Re-measured after the frame got cheaper elsewhere. SPOT SAMPLES ARE
// USELESS HERE: the autopilot crosses ocean and forest, and where it is
// moves p95 by more than the budget does. These aggregate EVERY frame of
// a 110 s flight, ~8-9k of them, which is the only way the orderings
// below stayed put when repeated.
//
//   budget   p50    p90    p99   over 16.67ms   backlog   boot settle
//       24   8.3   16.1   22.9      7.85%       drains
//       16   7.9   14.5   18.9      4.55%       drains       2.5s
//       12   7.7   13.0   17.6      1.66%       drains       2.8s
//        8   7.5   11.4   17.3    1.93/2.36%    drains       3.5s
//        4   7.2   10.0   17.0      1.18%      NEVER         -
//
// `over 16.67ms` is the share of frames that miss 60 fps, which is the
// number the target is actually about; p50 barely moves across the whole
// range, so a median reading hides all of this.
//
// 12 is the knee. Below it the frame stops improving — 8 measured 1.93%
// and 2.36% on two runs, so it is not distinguishable from 12's 1.66% —
// while settle keeps getting worse, 2.8s to 3.5s. Paying a second of
// boot for nothing is the wrong side of the trade.
//
// The old reasoning was wrong twice over. It read "below 48 the backlog
// grows, which is a smoother picture of the wrong world" — but `awaiting`
// is a transient, not a floor. At 12 it peaks in the thousands and still
// returns to zero, and `drawn` is within a few percent of what 48 draws.
// What a smaller batch does is spread the same work over more frames
// instead of landing it in one lump.
//
// 4 is where that stops being true and the old warning comes good: the
// backlog sits at ~2000 and never returns to zero, the world is
// permanently behind the camera, and it never reports settled. So the
// floor is set by whether the queue drains, not by the frame time, which
// keeps improving past the point where the picture stops being right.
//
// At walking and stationary speeds none of this binds; there is no queue.
fn env_budget(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Chunks whose density is dispatched per frame. Env-overridable because
/// how much GPU work fits in a frame is a property of the machine as much
/// as of the code, and re-measuring the table above on another one should
/// not need a rebuild.
static GEN_BUDGET: std::sync::LazyLock<usize> =
    std::sync::LazyLock::new(|| env_budget("VOXEL_GEN_BUDGET", 12));
/// Deferred chunks re-admitted per frame. Each one costs a density pass
/// it has already paid once, so recovery is deliberately gradual: the
/// alternative is a frame that regenerates everything the slab just
/// rejected and rejects it again.
const RETRY_BUDGET: usize = 32;
static MESH_BUDGET: std::sync::LazyLock<usize> =
    std::sync::LazyLock::new(|| env_budget("VOXEL_MESH_BUDGET", 12));
const STAGING_BUFFERS: usize = 3;

// --- main-world <-> render-world plumbing ------------------------------------

/// Chunk lifecycle commands from the main-world LOD controller.
#[derive(Clone, Debug)]
pub enum ChunkCommand {
    /// Generate (and mesh, if non-empty) this chunk. `show_on_ready` makes
    /// it visible as soon as it is drawable; otherwise it stays hidden
    /// until [`ChunkCommand::Commit`], which is how a set of chunks is
    /// revealed together.
    /// `ops` are planning-layer CSG operations applied by the density pass.
    Request {
        key: ChunkKey,
        show_on_ready: bool,
        /// In-place regens only: keep drawing the old mesh after the new
        /// one is ready and swap on [`ChunkCommand::Commit`] — lets the LOD
        /// controller land a whole epoch of seam-coupled meshes in one
        /// frame (readiness is reported when the held mesh is drawable).
        hold: bool,
        ops: Option<Arc<Vec<CsgOp>>>,
        /// 2 bits per face (+x,-x,+y,-y,+z,-z): 0 equal/none, 1 = neighbor
        /// coarser, 2 = neighbor finer. Drives seam ownership + band blend.
        face_mask: u32,
    },
    /// Make visible and swap in any held regen result.
    Commit(ChunkKey),
    Free(ChunkKey),
}

/// Sentinel gen mask for meshes whose seam mask is unknown (resurrected
/// pre-free results): never equals a real 12-bit mask or the empty
/// accept-any marker, so readiness gates can't be satisfied by them.
const STALE_MASK: u32 = u32::MAX - 1;

/// Main-world queue of chunk lifecycle commands (filled by the chunk
/// generation service in voxel-engine, drained by extraction). Interior
/// mutability because extraction system params must be read-only; a shared
/// handle because the service that owns it hands it to generation threads.
#[derive(Resource, Default, Clone)]
pub struct ChunkCommandQueue {
    inner: Arc<Mutex<Vec<ChunkCommand>>>,
}

impl ChunkCommandQueue {
    pub fn push(&self, command: ChunkCommand) {
        self.inner.lock().unwrap().push(command);
    }

    /// Take everything queued so far.
    pub fn take(&self) -> Vec<ChunkCommand> {
        std::mem::take(&mut *self.inner.lock().unwrap())
    }
}

/// Waiters for a chunk becoming drawable, keyed by chunk.
///
/// The epoch machine polls readiness as a batch, which suits a planner
/// deciding what to swap. A layer that *owns* a chunk needs the opposite:
/// to ask for one and block until it exists, because `create` is where a
/// chunk's resources are acquired. Both read the same notifications, so
/// there is exactly one drain — the chunk generation service in
/// voxel-engine — and it fans out to whoever is waiting.
#[derive(Resource, Default, Clone)]
pub struct ChunkWaiters(Arc<Mutex<HashMap<ChunkKey, Vec<crossbeam_channel::Sender<u32>>>>>);

impl ChunkWaiters {
    /// A receiver that fires when `key` next becomes drawable, with the
    /// seam mask its mesh was built with.
    pub fn wait_for(&self, key: ChunkKey) -> crossbeam_channel::Receiver<u32> {
        let (tx, rx) = crossbeam_channel::bounded(1);
        self.0.lock().unwrap().entry(key).or_default().push(tx);
        rx
    }

    /// Called by whoever drains the ready channel.
    pub fn notify(&self, key: ChunkKey, mask: u32) {
        if let Some(waiters) = self.0.lock().unwrap().remove(&key) {
            for tx in waiters {
                let _ = tx.send(mask);
            }
        }
    }

    /// Give up on a chunk that will never arrive (its request was
    /// cancelled), so a blocked create cannot wait forever.
    pub fn abandon(&self, key: ChunkKey) {
        self.0.lock().unwrap().remove(&key);
    }

    pub fn pending(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

/// Render→main notifications: a requested chunk became drawable (meshed) or
/// was classified empty. Drained by the chunk generation service, which
/// the LOD controller reads for ready-before-swap.
#[derive(Resource, Clone)]
pub struct ChunkReadyChannel {
    /// (key, face_mask baked into the drawable mesh; u32::MAX for
    /// empty-classified chunks, which any seam configuration accepts).
    pub rx: crossbeam_channel::Receiver<(ChunkKey, u32)>,
}

#[derive(Resource, Clone)]
struct ChunkReadySender(crossbeam_channel::Sender<(ChunkKey, u32)>);

/// Shared render statistics for the debug HUD (written by the render world,
/// read by the main world).
#[derive(Resource, Clone, Default)]
pub struct SharedRenderStats(pub Arc<Mutex<RenderStats>>);

#[derive(Default)]
pub struct RenderStats {
    /// Mask each currently-drawn (visible, meshed) chunk's on-screen mesh
    /// was built with — ground truth for seam validation (the ready
    /// channel reports held meshes before they swap in).
    pub drawn_masks: Vec<(ChunkKey, u32)>,
    /// Draw-list length per world — is a world being drawn at all?
    pub drawn_per_world: Vec<usize>,
    pub tracked: usize,
    pub meshed: usize,
    pub empty_classified: usize,
    pub awaiting: usize,
    pub arena_free: u32,
    /// Slots in use and free per class, both spelled out, and what the
    /// allocator had to do to keep up. A class at zero free is a working
    /// set, not a problem; pressure is the number that means something.
    pub slab_used_pages: u32,
    pub slab_total_pages: u32,
    /// High-water mark over the session. A reading taken standing still
    /// is one sample of a process that depends on where the camera is,
    /// so this is what a budget should be judged against.
    pub slab_peak_pages: u32,
    /// Longest free run. Well below the maximum chunk size while pages
    /// are plentiful means fragmentation, not exhaustion.
    pub slab_longest_free_run: u32,
    /// Live allocations by run length, and its high-water mark — the
    /// working set's shape, which no fixed partition could predict.
    pub slab_runs: Vec<u32>,
    pub slab_peak_runs: Vec<u32>,
    /// Chunks the slab can hold, from what chunks have actually cost at
    /// their peak. Admission control's budget, and 0 before the render
    /// world has run (callers fall back to the configured page count).
    pub slab_capacity_chunks: usize,
    pub slab_pressure: crate::slab::SlabPressure,
    pub drawn: usize,
    pub culled: usize,
    /// Chunks that have entered each pipeline stage since start. Rates
    /// come from differencing these: which stage is the narrowest is not
    /// something the budgets tell you, because a budget only says what a
    /// frame is ALLOWED to do.
    pub gen_started: u64,
    pub mesh_started: u64,
    pub reported_ready: u64,
    /// Frames the gen batch stopped early for want of an arena slot.
    pub gen_starved: u64,
    /// Wedge forensics: (state name, count) over all tracked chunks,
    /// including the pending track — every arena-slot holder is visible.
    pub state_counts: Vec<(&'static str, usize)>,
}

/// The level's generator program — the data that *is* the world — plus the
/// seed mixed into its hashes and the sun direction for the shadow bake.
/// Set by the level presenter; extracted every frame so hot-reloads apply.
/// The density shader interprets it; the mesh (shadow bake) and water
/// (shoreline) shaders read its height ops.
#[derive(Resource, Clone)]
pub struct WorldProgram {
    pub ops: std::sync::Arc<Vec<voxel_core::worldop::WorldOp>>,
    pub seed: u32,
    pub sun_dir: Vec3,
}

/// Everything the renderer knows about ONE loaded world.
///
/// One record rather than a resource per aspect. Four parallel `Vec`s
/// indexed by world id is four chances to push to three of them: the
/// portal did exactly that, registering a program and a config while the
/// material table stayed global, and the only thing standing between two
/// levels and each other's shading was an `assert!` in the host asking
/// them not to reuse a material id.
#[derive(Clone, Default)]
pub struct RenderWorld {
    pub program: WorldProgram,
    /// This world's recipes, indexed by the material ids ITS generator
    /// ops emit. Per world because ids are level data: two levels
    /// authored apart both use id 1 and mean different things by it.
    pub materials: Vec<WorldMaterial>,
    /// What this world's host painted on its own ground.
    pub surface_map: SurfaceMap,
    /// Half-spaces this world's chunks are clipped to; empty when it is
    /// drawn whole. A host showing one world through an opening in
    /// another uses these to keep it on the far side of the opening.
    pub clip: Vec<Vec4>,
}

/// Every loaded world, indexed BY world id.
///
/// The index is the id — `ChunkKey::world`, the GPU header array slot and
/// this vector are the same number by construction, which is the whole
/// reason a chunk needs to carry nothing but its key.
#[derive(Resource, Default, Clone)]
pub struct RenderWorlds(Vec<RenderWorld>);

impl RenderWorlds {
    /// Register a world and return its id. The ONE way to grow the set:
    /// everything per-world arrives together or not at all.
    pub fn register(&mut self, world: RenderWorld) -> voxel_core::WorldId {
        let id = self.0.len();
        assert!(
            id < MAX_WORLDS,
            "at most {MAX_WORLDS} worlds fit one program buffer",
        );
        self.0.push(world);
        id as voxel_core::WorldId
    }

    pub fn get(&self, id: voxel_core::WorldId) -> Option<&RenderWorld> {
        self.0.get(usize::from(id))
    }

    pub fn get_mut(&mut self, id: voxel_core::WorldId) -> Option<&mut RenderWorld> {
        self.0.get_mut(usize::from(id))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &RenderWorld> {
        self.0.iter()
    }

    /// Clip planes for a world, empty when it has none or does not
    /// exist.
    pub fn clip(&self, id: voxel_core::WorldId) -> &[Vec4] {
        self.get(id).map_or(&[], |w| &w.clip[..])
    }

    /// Drop every world's clip planes. The host republishes the ones
    /// that still apply each frame, so a mask it stops setting stops
    /// being applied.
    pub fn clear_clips(&mut self) {
        for world in &mut self.0 {
            world.clip.clear();
        }
    }
}

/// Which world the camera is in. The host sets it; it drives the main
/// view's [`ViewWorld`] and render layer.
#[derive(Resource, Default, Clone, Copy)]
pub struct CameraWorld(pub voxel_core::WorldId);

/// Which world a VIEW looks at.
///
/// Per view, not global, because a portal renders a second view of a
/// DIFFERENT world into the same frame. A camera without one looks at
/// world 0.
#[derive(Component, Default, Clone, Copy, ExtractComponent)]
pub struct ViewWorld(pub voxel_core::WorldId);

impl Default for WorldProgram {
    fn default() -> Self {
        Self {
            ops: std::sync::Arc::new(Vec::new()),
            seed: 0,
            sun_dir: Vec3::new(0.55, 0.5, 0.32),
        }
    }
}

/// Where the per-material threshold table starts, and how long it is: one
/// `f32` per material id a texel can hold. Twin of the mesh shader's
/// `SURFACE_MAP_THRESHOLDS`.
const SURFACE_MAP_THRESHOLDS: usize = 8;
const SURFACE_MAP_MATERIALS: usize = 256;

/// Words of one world's [`SurfaceMap`] header before its texels. Twin of
/// the mesh shader's `SURFACE_MAP_HEADER`.
const SURFACE_MAP_HEADER: usize = SURFACE_MAP_THRESHOLDS + SURFACE_MAP_MATERIALS;

/// A raster of surface material ids the mesh pass paints onto up-facing
/// vertices, without carving anything.
///
/// The mechanism a feature needs once it is smaller than a voxel. A road
/// is 0.5 m thick, so cutting it stops doing anything about 100 m from
/// the camera — but a road is not an object standing ON the ground, it IS
/// the ground, so at every distance past that it is a material rather
/// than a shape. One fetch per VERTEX, independent of how many roads
/// there are, where serving them as ops costs a loop per SAMPLE.
///
/// The engine owns the mechanism and nothing about what is painted: the
/// host rasterizes whatever its layers planned, and material 0 means
/// "leave the terrain's own material alone".
#[derive(Resource, Clone, Default)]
pub struct SurfaceMap {
    /// Material ids, one byte per texel, four per word, row-major.
    pub texels: std::sync::Arc<Vec<u32>>,
    /// World xz of texel (0, 0).
    pub origin: Vec2,
    pub texel_m: f32,
    /// Texels per side. 0 disables the map.
    pub size: u32,
    /// Only paint chunks whose voxels are at least this big.
    ///
    /// The map is what a feature becomes when it is smaller than a voxel.
    /// Where the voxels are finer than that, the real thing is there —
    /// carved, at full detail — and painting over it would replace
    /// geometry with a texel grid. So the host says how coarse a chunk
    /// has to be before the approximation is the better answer.
    pub min_voxel_m: f32,
    /// Per-material overrides of [`Self::min_voxel_m`], as
    /// `(material id, min voxel)`.
    ///
    /// The handover scale belongs to the FEATURE, not to the map. A road
    /// is only ever ground, so its paint can take over the moment the
    /// carve stops resolving; a water course is also drawn as a surface
    /// by whoever owns it, out to a range the map knows nothing about, and
    /// painting inside that range draws the same river twice. A material
    /// with no entry here uses the default.
    pub coarse_from: Vec<(u32, f32)>,
    /// Bumped when the raster changes, so the GPU copy is rebuilt only
    /// then.
    pub generation: u64,
}

impl SurfaceMap {
    /// Header + payload, as the shader reads one world's section. The
    /// placement travels in the buffer rather than in a uniform so that no
    /// layout twin grows a field: `ChunkParams` is already mirrored in two
    /// shaders and Rust.
    ///
    /// Header ONLY: the texels are uploaded straight out of the `Arc` that
    /// already holds them. Returning header and payload together meant
    /// building a 16 MB `Vec` to copy a 24-word preamble onto the front of
    /// a buffer that already existed.
    fn header_words(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(SURFACE_MAP_HEADER);
        out.push(self.size);
        out.push(self.texel_m.to_bits());
        out.push(self.origin.x.to_bits());
        out.push(self.origin.y.to_bits());
        out.push(self.min_voxel_m.to_bits());
        // Every material defaults to the map's own threshold, so a host
        // that never names one behaves exactly as before.
        out.resize(SURFACE_MAP_THRESHOLDS, 0);
        out.resize(SURFACE_MAP_HEADER, self.min_voxel_m.to_bits());
        for &(id, min_voxel) in &self.coarse_from {
            out[SURFACE_MAP_THRESHOLDS + (id as usize % SURFACE_MAP_MATERIALS)] =
                min_voxel.to_bits();
        }
        out
    }

    /// Words this world occupies in the shared buffer, header included.
    fn section_words(&self) -> usize {
        if self.size == 0 {
            0
        } else {
            SURFACE_MAP_HEADER + self.texels.len()
        }
    }
}

/// Every world's surface map in ONE buffer: a table of per-world offsets,
/// then the sections themselves.
///
/// Per world because the map is painted in world-space xz and nothing in
/// it says which world it belongs to. One global raster meant the near
/// level's rivers and roads were painted onto the far level's ground
/// wherever their coordinates happened to overlap — and worlds share
/// coordinates by design, so that is everywhere.
///
/// Offset 0 means "this world paints nothing": the table itself occupies
/// word 0, so no real section can start there.
/// Where each world's section starts, and how many words the whole buffer
/// needs. Computed without touching a texel.
fn surface_map_layout(worlds: &RenderWorlds) -> ([u32; MAX_WORLDS], usize) {
    let mut offsets = [0u32; MAX_WORLDS];
    let mut cursor = MAX_WORLDS;
    for (id, world) in worlds.iter().enumerate().take(MAX_WORLDS) {
        let words = world.surface_map.section_words();
        if words == 0 {
            continue;
        }
        offsets[id] = cursor as u32;
        cursor += words;
    }
    (offsets, cursor)
}

/// The buffer image the upload produces, assembled in one piece.
///
/// Test-only, and deliberately built from the SAME two functions the
/// upload writes from, so it cannot agree with a layout the GPU never
/// sees. Production writes these sections straight into the queue at
/// their offsets and never holds the whole thing in memory.
#[cfg(test)]
fn surface_map_words(worlds: &RenderWorlds) -> Vec<u32> {
    let (offsets, total) = surface_map_layout(worlds);
    let mut out = vec![0u32; total];
    out[..MAX_WORLDS].copy_from_slice(&offsets);
    for (id, world) in worlds.iter().enumerate().take(MAX_WORLDS) {
        let words = world.surface_map.section_words();
        if words == 0 {
            continue;
        }
        let at = offsets[id] as usize;
        out[at..at + SURFACE_MAP_HEADER].copy_from_slice(&world.surface_map.header_words());
        out[at + SURFACE_MAP_HEADER..at + words].copy_from_slice(&world.surface_map.texels);
    }
    out
}

/// Bumped whenever any world's raster changes, so the GPU copy is rebuilt
/// only then. Summed rather than compared per world: generations only ever
/// increase, so a change in any of them changes the sum.
fn surface_map_generation(worlds: &RenderWorlds) -> u64 {
    worlds
        .iter()
        .map(|w| w.surface_map.generation)
        .fold(0u64, u64::wrapping_add)
}

/// Renders a uniform base color modulated by grain, pour/mortar bands,
/// grime, drip streaks, moss in upward crevices, and optional emissive
/// ceiling light strips.
pub const MAT_KIND_SURFACE: u32 = 0;
/// Renders altitude-zoned natural terrain (low/mid/high/peak colors with
/// noisy borders) with a slope override to the high-zone color.
pub const MAT_KIND_ZONED: u32 = 1;
/// Forested zoned terrain: crown-noise canopy with normal perturbation
/// and AO between the low and rock zones, strata-bumped rock above.
pub const MAT_KIND_CANOPY: u32 = 2;

/// One material recipe, GPU form (128 B). The draw shader indexes the
/// material table with the per-vertex material id the generator ops emit —
/// shading is level data, not engine code.
///
/// Layout by kind — `surface`:
/// c0 = base rgb | grain amp, c1 = grime tint | amount,
/// c2 = moss rgb | amount, c3 = emissive rgb | intensity,
/// p0 = band (freq, amp, lo, hi), p1 = (band warp, streaks, strip spacing,
/// strip level spacing), p2 = (strip chance, strip glow, detail fade, -).
///
/// `zoned`: c0..c3 = low/mid/high/peak rgb | (mid start, high start, peak
/// start, border amp), p0 = mid-b rgb | mid width, p1 = high-b rgb | high
/// width, p2 = (peak width, steep hi, steep lo, detail fade).
#[derive(ShaderType, Clone, Copy, Debug, PartialEq)]
pub struct WorldMaterial {
    /// kind, unused ×3
    pub head: UVec4,
    pub c0: Vec4,
    pub c1: Vec4,
    pub c2: Vec4,
    pub c3: Vec4,
    pub p0: Vec4,
    pub p1: Vec4,
    pub p2: Vec4,
}

impl WorldMaterial {
    /// Assemble from `voxel_core::layout::MatPack`, which is what decides
    /// which parameter lives in which component. Nothing here knows what
    /// any of the eight `vec4`s means — that is the point.
    pub fn from_packed((kind, slots): (u32, [[f32; 4]; 7])) -> Self {
        let v = |i: usize| Vec4::from_array(slots[i]);
        Self {
            head: UVec4::new(kind, 0, 0, 0),
            c0: v(0),
            c1: v(1),
            c2: v(2),
            c3: v(3),
            p0: v(4),
            p1: v(5),
            p2: v(6),
        }
    }
}

impl Default for WorldMaterial {
    fn default() -> Self {
        // Neutral gray surface — what an unassigned material id renders as.
        Self {
            head: UVec4::new(MAT_KIND_SURFACE, 0, 0, 0),
            c0: Vec4::new(0.5, 0.5, 0.5, 0.3),
            c1: Vec4::ZERO,
            c2: Vec4::ZERO,
            c3: Vec4::ZERO,
            p0: Vec4::new(0.0, 0.0, 0.0, 1.0),
            p1: Vec4::ZERO,
            p2: Vec4::new(0.0, 0.0, 0.002, 0.0),
        }
    }
}

/// How many material ids a world can use.
///
/// The recipes themselves live in runtime-sized storage buffers and are
/// not capped at all. What this bounds is the id → slab slot map the
/// draw shader indirects through, whose entries are packed one BYTE
/// each — so `MAX_WORLDS * MATERIAL_SLOTS` must stay within 256 distinct
/// slots. At 8 worlds that puts the ceiling here at exactly 32.
///
/// Raising it costs only the map: 8 x 32 entries is 256 bytes of uniform.
/// Past that, widen the packed slot (and the vertex's material byte,
/// `(extra >> 8u) & 0xFFu`, which is the real 256-per-chunk ceiling).
pub const MATERIAL_SLOTS: usize = 32;

/// A world's material table, indexed by material id and padded to
/// [`MATERIAL_SLOTS`] so the id → slot arithmetic is uniform.
pub fn material_table(recipes: impl IntoIterator<Item = (u32, WorldMaterial)>) -> Vec<WorldMaterial> {
    let mut table = vec![WorldMaterial::default(); MATERIAL_SLOTS];
    for (id, recipe) in recipes {
        let id = id as usize;
        if id < MATERIAL_SLOTS {
            table[id] = recipe;
        } else {
            warn!("material id {id} out of range (max {})", MATERIAL_SLOTS - 1);
        }
    }
    table
}

/// Per-view render flags the voxel shaders still need from the engine.
/// Lighting and atmosphere are NOT here: voxel surfaces shade through
/// Bevy's PBR, so the app's lights, ambient and `DistanceFog` drive them
/// like any other surface.
#[derive(Resource, ShaderType, Clone, Copy, Debug)]
pub struct EnvParams {
    /// x = coverage-eval mode (monotone geometry over a magenta clear).
    pub flags: Vec4,
    /// `(world, material id)` → slot in the bindless material slab, one id
    /// per component, world-major. Filled in the render world, where the
    /// slots are known.
    ///
    /// Indexed by world because a material id means whatever its own level
    /// says: planet's 1 and megastructure's 1 are different recipes that
    /// must be able to coexist.
    pub material_slots: [UVec4; MATERIAL_SLOT_VEC4S],
}

// Hand-written because the array outgrew the length `Default` is derived
// for; `[UVec4; N]` only implements it up to 32.
impl Default for EnvParams {
    fn default() -> Self {
        Self {
            flags: Vec4::ZERO,
            material_slots: [UVec4::ZERO; MATERIAL_SLOT_VEC4S],
        }
    }
}

/// `vec4`s needed to hold one slot per (world, material id) pair.
pub const MATERIAL_SLOT_VEC4S: usize = MAX_WORLDS * MATERIAL_SLOTS / 4;

/// Where a world's material id lands in [`EnvParams::material_slots`].
/// Twin of the draw shader's `material_for`.
pub fn material_slot_index(world: voxel_core::WorldId, id: u32) -> usize {
    usize::from(world) * MATERIAL_SLOTS + (id as usize).min(MATERIAL_SLOTS - 1)
}

/// GPU layout twin of `voxel_core::worldop::WorldOp` (64 B).
#[derive(ShaderType, Clone, Copy, Default)]
pub(crate) struct GpuWorldOp {
    /// kind, flags, material, unused
    meta: UVec4,
    p0: Vec4,
    p1: Vec4,
    p2: Vec4,
}

/// How many worlds one program buffer can describe.
///
/// Fixed because `encase` allows a single runtime-sized array per struct,
/// and that one is the ops. Raising it costs a `GpuWorldHeader` (32 B)
/// and eight material slots per world in a uniform — nothing — so the
/// real ceiling on loaded worlds is the slab budget they are admitted
/// against, not this. Twin of the `array<WorldHeader, N>` in the density
/// and mesh shaders.
pub const MAX_WORLDS: usize = 8;

/// One world's slice of the shared program buffer.
/// `count = (op offset, op count, height ops, seed)`.
#[derive(ShaderType, Clone, Copy, Default)]
pub struct GpuWorldHeader {
    count: UVec4,
    sun: Vec4,
}

/// Every loaded world's program, in ONE buffer.
///
/// Worlds are concatenated and addressed by a per-world (offset, count),
/// which is the same shape the CSG ops already use: a chunk carries a
/// range rather than the pipeline carrying a world. That is what lets one
/// density dispatch serve chunks of different worlds in the same frame,
/// and it is why the portal can show two levels at once without a second
/// set of GPU resources.
#[derive(ShaderType, Clone, Default)]
pub struct GpuWorldProgram {
    worlds: [GpuWorldHeader; MAX_WORLDS],
    #[shader(size(runtime))]
    ops: Vec<GpuWorldOp>,
}

impl GpuWorldProgram {
    fn from_worlds(render_worlds: &RenderWorlds) -> Self {
        let mut gpu_ops: Vec<GpuWorldOp> = Vec::new();
        let mut worlds = [GpuWorldHeader::default(); MAX_WORLDS];
        for (world, program) in render_worlds
            .iter()
            .map(|w| &w.program)
            .take(MAX_WORLDS)
            .enumerate()
        {
            let offset = gpu_ops.len() as u32;
            let height_ops = program.ops.iter().filter(|op| op.is_height_op()).count() as u32;
            gpu_ops.extend(program.ops.iter().map(|op| GpuWorldOp {
                meta: UVec4::new(op.kind, op.flags, op.material, 0),
                p0: Vec4::from_array(op.p0),
                p1: Vec4::from_array(op.p1),
                p2: Vec4::from_array(op.p2),
            }));
            worlds[world] = GpuWorldHeader {
                count: UVec4::new(offset, program.ops.len() as u32, height_ops, program.seed),
                sun: program.sun_dir.extend(0.0),
            };
        }
        // Runtime-sized arrays must not be empty.
        if gpu_ops.is_empty() {
            gpu_ops.push(GpuWorldOp::default());
        }
        Self {
            worlds,
            ops: gpu_ops,
        }
    }
}

/// Marker entity that anchors the terrain draw in the render phases.
#[derive(Clone, Component, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<VoxelTerrainMarker>)]
pub struct VoxelTerrainMarker;

#[derive(Default)]
pub struct VoxelChunksPlugin {
    /// How big the mesh slab is and how finely it is divided. A game
    /// whose chunks are denser than this demo's raises it here rather
    /// than editing the crate — and what a world costs depends on where
    /// the camera is, so the allocator reports peaks to size it against
    /// instead of asking anyone to predict it.
    pub slab: crate::slab::SlabConfig,
}

impl Plugin for VoxelChunksPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/voxel_world_density.wgsl");
        embedded_asset!(app, "shaders/voxel_mesh_chunks.wgsl");
        embedded_asset!(app, "shaders/voxel_chunk_draw.wgsl");

        let slab_config = self.slab;
        let (ready_tx, ready_rx) = crossbeam_channel::unbounded();
        app.init_resource::<RenderWorlds>()
            .init_resource::<CameraWorld>()
            .init_resource::<EnvParams>();
        app.init_resource::<ChunkCommandQueue>()
            .init_resource::<SharedRenderStats>()
            .insert_resource(ChunkReadyChannel { rx: ready_rx })
            .init_resource::<ChunkWaiters>()
            .add_plugins((
                ExtractComponentPlugin::<VoxelTerrainMarker>::default(),
                ExtractComponentPlugin::<ViewWorld>::default(),
                // Gives the terrain material the usual asset lifecycle:
                // extraction, prepared bind groups, a bind group allocator.
                MaterialPlugin::<VoxelSurfaceMaterial>::default(),
            ))
            .init_resource::<TerrainMaterials>()
            .add_systems(Startup, spawn_terrain_marker)
            .add_systems(Update, sync_terrain_materials);

        let stats = app.world().resource::<SharedRenderStats>().clone();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<RenderWorlds>()
            .init_resource::<CameraWorld>()
            .init_resource::<WorldViews>()
            .init_resource::<EnvParams>()
            .insert_resource(stats)
            .insert_resource(slab_config)
            .insert_resource(ChunkReadySender(ready_tx))
            .init_resource::<ExtractedChunkCommands>()
            .init_resource::<ChunkTable>()
            .init_resource::<FrameBatches>()
            .init_resource::<VoxelDrawLists>()
            .init_resource::<PendingVoxelQueues>()
            .init_resource::<ViewBindGroupRes>()
            .add_render_command::<Opaque3d, DrawVoxelChunksCommands>()
            // The draw pipeline's layout comes from Bevy's view layouts,
            // which are built in the same schedule — order is otherwise
            // ambiguous and only sometimes lands the right way.
            .add_systems(
                RenderStartup,
                init_chunk_resources.after(bevy::pbr::init_mesh_pipeline_view_layouts),
            )
            .init_resource::<ExtractedTerrainMaterials>()
            .add_systems(
                ExtractSchedule,
                (
                    extract_chunk_commands,
                    extract_camera_pos,
                    extract_worlds,
                    extract_terrain_materials,
                ),
            )
            // Slots must be resolved before `plan_frame` writes the uniform.
            .add_systems(
                Render,
                resolve_material_slots
                    .in_set(RenderSystems::Prepare)
                    .before(plan_frame),
            )
            .add_systems(Render, plan_frame.in_set(RenderSystems::Prepare))
            .add_systems(
                Render,
                prepare_view_bind_group.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Render, queue_voxel_chunks.in_set(RenderSystems::Queue))
            .add_systems(RenderGraph, dispatch_chunk_work.before(camera_driver));
    }
}

/// Surface materials as assets: one per (world, material id). Held so the
/// render world can look up which slab slot each landed in.
#[derive(Resource, Default, Clone)]
pub struct TerrainMaterials(pub Vec<Vec<Handle<VoxelSurfaceMaterial>>>);

/// Publish the level's recipes as assets, one per id. Handles are reused
/// across reloads so re-shading never respawns anything.
fn sync_terrain_materials(
    mut commands: Commands,
    worlds: Res<RenderWorlds>,
    mut materials: ResMut<TerrainMaterials>,
    mut assets: ResMut<Assets<VoxelSurfaceMaterial>>,
    markers: Query<Entity, With<VoxelTerrainMarker>>,
) {
    if !worlds.is_changed() && materials.0.len() == worlds.len() {
        return;
    }
    // Handles are created once per (world, id) and reused: re-adding a
    // material component is not free — it re-specializes the phase item
    // the whole terrain draw hangs off — so a world arriving must not
    // disturb the worlds already drawing.
    let had = materials.0.len();
    materials.0.resize_with(worlds.len(), Vec::new);
    for (slot, world) in materials.0.iter_mut().zip(worlds.iter()) {
        slot.resize_with(world.materials.len(), || {
            assets.add(VoxelSurfaceMaterial::default())
        });
        for (handle, recipe) in slot.iter().zip(&world.materials) {
            // ONLY when it actually differs. `get_mut` marks the asset
            // modified, and a material modified every frame never settles:
            // Bevy re-prepares it, the bind group is rebuilt under the
            // draw, and `SetMaterialBindGroup` skips the whole terrain.
            // That is invisible — no error, no validation warning, a full
            // draw list and no ground.
            if assets.get(handle).is_some_and(|m| m.recipe == *recipe) {
                continue;
            }
            if let Some(mut material) = assets.get_mut(handle) {
                material.recipe = *recipe;
            }
        }
    }
    // The marker's material only decides which slab the draw binds; the
    // per-vertex id picks the recipe out of it. Inserted ONCE, on the
    // frame the first world appears — re-inserting it whenever any world's
    // recipes changed made registering a second world re-specialize the
    // anchor of every world's terrain draw.
    if had > 0 {
        return;
    }
    if let Some(first) = materials.0.first().and_then(|w| w.first()) {
        for entity in &markers {
            commands
                .entity(entity)
                .insert(MeshMaterial3d(first.clone()));
        }
    }
}

/// The anchor the chunk phase item hangs off — not content. Visible from
/// every world's camera; which world's chunks get drawn is decided per
/// chunk by `key.world`. On layer 0 alone, a camera in world 1 drew no
/// terrain at all because it could not see the anchor.
fn spawn_terrain_marker(mut commands: Commands) {
    commands.spawn((
        VoxelTerrainMarker,
        crate::all_world_layers(),
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

/// Where each world is being looked at FROM, and through what frustum.
///
/// Per world, not global: a portal renders a second world from a
/// different eye in the same frame, and chunk offsets are camera-relative
/// (for precision at planet scale) while culling is per frustum. Sharing
/// one camera's values across both views draws the far world at the near
/// eye's offsets — off screen — and culls it against the wrong frustum.
///
/// One entry per world holds only while a world is shown by at most one
/// view, which is true of a portal. Recursion would need one per view.
type WorldView = Option<(DVec3, Option<Frustum>)>;

/// Cameras, with the world each one looks at.
type WorldViewQuery<'w, 's> =
    Query<'w, 's, (&'static GlobalTransform, &'static Frustum, Option<&'static ViewWorld>), With<Camera3d>>;

#[derive(Resource, Default)]
struct WorldViews(Vec<WorldView>);

#[derive(Resource, Default)]
struct ExtractedFrustum(Option<Frustum>);

enum ChunkState {
    /// Waiting for budget / an arena slot.
    QueuedGen,
    /// Density generated (or generating this frame); counts on their way back.
    CountsInFlight { slot: u32 },
    /// Freed while counts were in flight; readback must recycle the slot.
    Cancelled { slot: u32 },
    /// Arena slot AND slab space both in hand; the mesh dispatch is next.
    ///
    /// A chunk never holds an arena slot without somewhere to put the
    /// result. That is the whole wedge: waiting here for a slab slot held
    /// the arena slot, the arena starved, no chunk could generate, so
    /// nothing was ever freed to release a slab slot.
    ReadyToMesh { slot: u32, alloc: SlabAlloc, index_count: u32 },
    /// Counted, but the slab had no room. Holds NOTHING — retried when
    /// space frees, at the cost of re-running density.
    Deferred { verts: u32, indices: u32 },
    /// Classified all-air/all-solid: drawable as nothing.
    Empty,
    /// In the slab and drawable.
    Meshed { alloc: SlabAlloc, index_count: u32 },
}

struct RenderChunk {
    state: ChunkState,
    visible: bool,
    show_on_ready: bool,
    /// Hold a finished in-place regen until [`ChunkCommand::Commit`].
    hold: bool,
    /// Planning-layer ops, held until the density pass consumes them.
    ops: Option<Arc<Vec<CsgOp>>>,
    /// Seam ownership mask (see [`ChunkCommand::Request`]); baked into both
    /// the gen and mesh params of one generation so passes always agree.
    face_mask: u32,
    /// The mask the currently in-flight/most recent generation was built
    /// with (face_mask may have been updated by a newer request since).
    gen_mask: u32,
    /// The mask of the mesh currently drawn (updates only when a mesh
    /// swaps in — gen_mask may already describe a held successor).
    drawn_mask: u32,
    /// In-place regeneration (a neighbor's LOD changed): the old mesh keeps
    /// drawing until the replacement is ready, then swaps atomically.
    pending: Option<Pending>,
    /// A Request superseded this chunk's in-flight regen: when that
    /// regen lands, discard its result and regenerate with the stored
    /// mask/ops (otherwise the requesting epoch waits forever).
    requeue: bool,
}

enum Pending {
    Queued,
    CountsInFlight { slot: u32 },
    ReadyToMesh { slot: u32, alloc: SlabAlloc, index_count: u32 },
    /// Counted, but the slab had no room. Holds nothing; the OLD mesh
    /// keeps drawing, so this costs a stale seam rather than a hole.
    Deferred { verts: u32, indices: u32 },
    /// Regen meshed but held: the old mesh keeps drawing until Commit.
    Held { alloc: SlabAlloc, index_count: u32 },
    /// Regen classified empty but held: emptiness applies at Commit.
    HeldEmpty,
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
    /// Allocated index range (index units), cleared before emission so any
    /// count-vs-emit divergence yields degenerate triangles, not stale
    /// indices from the slot's previous occupant.
    first_index: u32,
    index_count: u32,
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
struct VoxelDrawLists(Vec<Vec<DrawEntry>>);

impl VoxelDrawLists {
    fn clear(&mut self) {
        self.0.resize_with(MAX_WORLDS, Vec::new);
        for list in &mut self.0 {
            list.clear();
        }
    }

    fn total(&self) -> usize {
        self.0.iter().map(Vec::len).sum()
    }
}

#[derive(ShaderType, Clone, Copy)]
struct ChunkParams {
    origin: Vec4,
    /// Chunk minimum corner in integer world-voxel units at this chunk's
    /// own scale (pos × 32); w = which WORLD's program to interpret, an
    /// index into the program buffer's per-world headers. Density sample positions derive
    /// from these EXACT integers so two chunks sharing a world sample
    /// compute a bit-identical position — `origin + idx × vs` rounds
    /// differently per chunk whenever the voxel size is not an exact
    /// binary float (0.1 m is not), and a single ULP flips signs where a
    /// surface grazes a sample: deterministic seam cracks.
    origin_voxels: bevy::math::IVec4,
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
/// Per-chunk draw data. Entries are dynamic-offset allocated and so
/// already padded to the device's uniform alignment (256 B on Metal),
/// which is why the clip planes ride here for free rather than needing a
/// per-view bind group.
struct ChunkDrawUniform {
    offset: Vec4,
    /// x = how many of `clip` are active, y = which WORLD this chunk is
    /// in. The fragment shader needs the world to pick a material table:
    /// a material id means whatever the chunk's own level says it means.
    head: UVec4,
    /// World-space half-spaces this chunk is clipped to; a fragment
    /// survives where `dot(n, p) + d >= 0` for all of them — see
    /// [`RenderWorld::clip`].
    ///
    /// The engine owns the mechanism and not what it is for. A host
    /// showing one world through an opening in another wants the far
    /// world only beyond the opening's plane, and Bevy's depth texture is
    /// `Depth32Float` with no stencil aspect, so the mask cannot be a
    /// stencil; a half-space is exact for a planar boundary and costs a
    /// dot product.
    clip: [Vec4; MAX_CLIP_PLANES],
}

/// Half-spaces one chunk draw can carry. Enough for a convex opening's
/// four sides and its own plane.
pub const MAX_CLIP_PLANES: usize = 5;

enum StagingState {
    Free,
    /// Copy recorded this frame; mapping requested next frame.
    PendingMap {
        entries: Vec<(ChunkKey, u32)>,
    },
    /// map_async issued; waiting for the callback.
    Mapping {
        entries: Vec<(ChunkKey, u32)>,
    },
}

struct StagingSlot {
    buffer: Buffer,
    state: StagingState,
}

#[derive(Resource)]
pub struct ChunkGpuResources {
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
    /// The world's generator program. Public so a host pipeline can
    /// replay its height ops (shorelines, water depth) on the GPU without
    /// hand-copying a CPU/GPU twin.
    pub program_buffer: StorageBuffer<GpuWorldProgram>,
    /// The surface map's words, and the generation they were built from.
    surface_map: Option<(Buffer, u64)>,
    /// A header saying "size 0", for when no host has painted anything.
    /// Aliasing another buffer would read its contents as a raster.
    surface_map_dummy: Buffer,
    env_uniform: UniformBuffer<EnvParams>,
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
    chunk_layout: BindGroupLayoutDescriptor,
    variants: Variants<RenderPipeline, VoxelChunksSpecializer>,
}

#[derive(Resource, Default)]
struct ViewBindGroupRes {
    chunk: Option<BindGroup>,
}

fn init_chunk_resources(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    view_layouts: Res<MeshPipelineViewLayouts>,
    slab_config: Res<crate::slab::SlabConfig>,
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
            slab_config.total_vertices() * VERTEX_BYTES,
            BufferUsages::STORAGE | BufferUsages::VERTEX,
        ),
        index_slab: buffer(
            "voxel_index_slab",
            // u16 indices.
            slab_config.total_indices() * 2,
            BufferUsages::STORAGE | BufferUsages::INDEX | BufferUsages::COPY_DST,
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
        slab: SlabAllocator::new(*slab_config),
        gen_uniforms: DynamicUniformBuffer::default(),
        program_buffer: StorageBuffer::default(),
        surface_map: None,
        surface_map_dummy: render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("voxel_surface_map_dummy"),
            contents: bytemuck::cast_slice(&[0u32; MAX_WORLDS]),
            usage: BufferUsages::STORAGE,
        }),
        env_uniform: UniformBuffer::default(),
        draw_uniforms: DynamicUniformBuffer::default(),
        map_tx,
        map_rx,
    });

    let gen_layout = BindGroupLayoutDescriptor::new(
        "voxel_gen_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, None),           // density arena
                uniform_buffer::<ChunkParams>(true),         // per-chunk params
                storage_buffer_sized(false, None),           // planning CSG ops
                storage_buffer_read_only_sized(false, None), // generator program
            ),
        ),
    );
    let mesh_layout = BindGroupLayoutDescriptor::new(
        "voxel_mesh_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, None),           // density arena
                uniform_buffer::<ChunkParams>(true),         // per-chunk params
                storage_buffer_sized(false, None),           // cell_indices scratch
                storage_buffer_sized(false, None),           // vertex slab
                storage_buffer_sized(false, None),           // index slab
                storage_buffer_sized(false, None),           // counts
                storage_buffer_read_only_sized(false, None), // generator program
            ),
        ),
    );

    let density_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_world_density.wgsl");
    let mesh_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_mesh_chunks.wgsl");

    let compute = |label: &'static str,
                   shader: &Handle<Shader>,
                   entry: &'static str,
                   layout: &BindGroupLayoutDescriptor| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    };
    commands.insert_resource(ChunkPipelines {
        density: compute(
            "voxel_density",
            &density_shader,
            "density_main",
            &gen_layout,
        ),
        count: compute("voxel_sn_count", &mesh_shader, "sn_count", &mesh_layout),
        vertices: compute(
            "voxel_sn_vertices",
            &mesh_shader,
            "sn_vertices",
            &mesh_layout,
        ),
        quads: compute("voxel_sn_quads", &mesh_shader, "sn_quads", &mesh_layout),
        gen_layout,
        mesh_layout,
    });

    // Draw pipeline. Groups 0 and 1 are Bevy's mesh view bind groups
    // (lights, shadow maps, clusters, fog, tonemapping LUTs); group 2 is
    // where Bevy puts per-mesh data, so ours slots in without disturbing
    // the material group at 3.
    let chunk_layout = BindGroupLayoutDescriptor::new(
        "voxel_chunks_chunk_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ChunkDrawUniform>(true),    // per-chunk offset
                uniform_buffer::<EnvParams>(false),          // render flags
                storage_buffer_read_only_sized(false, None), // surface material map
            ),
        ),
    );
    let draw_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "shaders/voxel_chunk_draw.wgsl");
    let base_descriptor = RenderPipelineDescriptor {
        label: Some("voxel_chunks_draw".into()),
        // Groups 0/1 are replaced per key by the specializer (the view
        // layout depends on msaa, fog and tonemapping).
        layout: vec![
            // 0/1 replaced per key by the specializer (Bevy's view layouts).
            chunk_layout.clone(),
            chunk_layout.clone(),
            chunk_layout.clone(),
            // 3: the terrain material, exactly where Bevy puts materials.
            VoxelSurfaceMaterial::bind_group_layout_descriptor(&render_device),
        ],
        vertex: VertexState {
            shader: draw_shader.clone(),
            shader_defs: vec![],
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
        },
        fragment: Some(FragmentState {
            shader: draw_shader,
            shader_defs: vec![],
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        depth_stencil: Some(DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::GreaterEqual),
            stencil: default(),
            bias: default(),
        }),
        // The surface is CLOSED and consistently wound — `write_quad`
        // flips the winding on the sign of the density, so every triangle
        // faces out of the solid. Without this it defaults to `None` and
        // every one of them is rasterized from behind as well, which on a
        // fill-bound view is the whole back half of the world being
        // shaded and then depth-rejected.
        primitive: PrimitiveState {
            cull_mode: Some(Face::Back),
            ..default()
        },
        ..default()
    };
    commands.insert_resource(ChunkDrawPipeline {
        chunk_layout,
        variants: Variants::new(
            VoxelChunksSpecializer {
                view_layouts: view_layouts.clone(),
            },
            base_descriptor,
        ),
    });
}

// --- extraction --------------------------------------------------------------

fn extract_chunk_commands(
    queue: Extract<Res<ChunkCommandQueue>>,
    mut extracted: ResMut<ExtractedChunkCommands>,
) {
    extracted.0.append(&mut queue.take());
}

/// Asset ids of the world's surface materials, in material-id order.
#[derive(Resource, Default)]
struct ExtractedTerrainMaterials(Vec<Vec<AssetId<VoxelSurfaceMaterial>>>);

fn extract_terrain_materials(
    materials: Extract<Res<TerrainMaterials>>,
    mut extracted: ResMut<ExtractedTerrainMaterials>,
) {
    extracted.0.clear();
    extracted.0.extend(
        materials
            .0
            .iter()
            .map(|world| world.iter().map(|h| h.id()).collect()),
    );
}

/// Resolve (world, material id) → slab slot. Bindless packs recipes into
/// slabs, so the shader needs the slot to pick one per vertex; the mapping
/// is only knowable here, after Bevy has prepared the materials.
fn resolve_material_slots(
    extracted: Res<ExtractedTerrainMaterials>,
    prepared: Res<bevy::render::erased_render_asset::ErasedRenderAssets<bevy::pbr::PreparedMaterial>>,
    mut env: ResMut<EnvParams>,
    mut warned: Local<bool>,
) {
    let mut slots = [UVec4::ZERO; MATERIAL_SLOT_VEC4S];
    // The draw binds ONE bind group — the marker material's — and every
    // slot is an index into it. Recipes landing in different groups would
    // shade the terrain with whatever occupies that slot in the bound
    // one: wrong colours, no error. Per-world tables multiplied the recipe
    // count by the number of loaded worlds, so say so rather than let it
    // present as "the far world looks odd".
    let mut group = None;
    for (world, ids) in extracted.0.iter().enumerate().take(MAX_WORLDS) {
        for (id, asset_id) in ids.iter().enumerate().take(MATERIAL_SLOTS) {
            let Some(material) = prepared.get(*asset_id) else {
                continue;
            };
            let i = material_slot_index(world as voxel_core::WorldId, id as u32);
            slots[i / 4][i % 4] = *material.binding.slot;
            let group = group.get_or_insert(material.binding.group);
            if *group != material.binding.group && !*warned {
                *warned = true;
                error!(
                    "terrain materials split across bindless groups ({:?} and {:?}): \
                     world {world} material {id} will shade with another recipe. \
                     Too many (world, material) pairs for one slab.",
                    *group, material.binding.group,
                );
            }
        }
    }
    env.material_slots = slots;
}

/// One clone per frame, so hot reloads and portal masks apply. Cheap: the
/// ops and the surface raster are behind `Arc`s, and the rest is a handful
/// of `vec4`s per world.
fn extract_worlds(
    worlds: Extract<Res<RenderWorlds>>,
    camera_world: Extract<Res<CameraWorld>>,
    env: Extract<Res<EnvParams>>,
    mut commands: Commands,
) {
    commands.insert_resource((**worlds).clone());
    commands.insert_resource(**camera_world);
    commands.insert_resource(**env);
}

fn extract_camera_pos(
    cameras: Extract<Query<(&GlobalTransform, &Frustum), crate::PlayerCameraFilter>>,
    worlds: Extract<WorldViewQuery>,
    mut commands: Commands,
) {
    let (pos, frustum) = cameras
        .iter()
        .next()
        .map(|(t, f)| (t.translation().as_dvec3(), Some(*f)))
        .unwrap_or_default();
    commands.insert_resource(ExtractedCameraPos(pos));
    commands.insert_resource(ExtractedFrustum(frustum));
    let mut per_world = vec![None; MAX_WORLDS];
    for (transform, frustum, world) in worlds.iter() {
        let id = usize::from(world.map_or(0, |w| w.0));
        if let Some(slot) = per_world.get_mut(id) {
            // First view of a world wins; there is only ever one today.
            slot.get_or_insert((transform.translation().as_dvec3(), Some(*frustum)));
        }
    }
    commands.insert_resource(WorldViews(per_world));
}

// --- planning (Prepare) ------------------------------------------------------

fn make_params(
    key: ChunkKey,
    slot: u32,
    alloc: Option<&SlabAlloc>,
    counts_slot: u32,
    csg: (u32, u32),
    face_mask: u32,
) -> ChunkParams {
    let origin = key.min_corner_m();
    ChunkParams {
        origin: Vec4::new(
            origin.x as f32,
            origin.y as f32,
            origin.z as f32,
            key.voxel_size_m() as f32,
        ),
        origin_voxels: (key.pos * 32).extend(i32::from(key.world)),
        slot,
        base_vertex: alloc.map_or(0, |a| a.base_vertex),
        first_index: alloc.map_or(0, |a| a.first_index),
        counts_slot,
        csg_offset: csg.0,
        csg_count: csg.1,
        _pad: UVec2::new(
            face_mask,
            std::env::var("VOXEL_EVAL_HOLES").map_or(0, |_| 1),
        ),
    }
}

/// Where a counted chunk goes next.
///
/// The one place the slab is consulted, and it is consulted the moment
/// the counts land — NOT later, from a queue of chunks sitting on arena
/// slots. Deciding here is what makes "holds an arena slot" and "can
/// still finish" the same condition.
enum Placement {
    /// Space reserved. Dispatch the mesh.
    Mesh(SlabAlloc),
    /// Nothing to draw here.
    Empty,
    /// Bigger than the largest class can ever hold.
    TooBig,
    /// No room right now. Release everything and retry when the slab
    /// drains; never wait holding a resource somebody else needs.
    Defer,
}

fn place(slab: &mut crate::slab::SlabAllocator, verts: u32, quads: u32) -> Placement {
    if verts == 0 || quads == 0 {
        return Placement::Empty;
    }
    let indices = quads.saturating_mul(6);
    if slab.pages_for(verts, indices) > slab.config().max_pages_per_chunk {
        return Placement::TooBig;
    }
    match slab.alloc(verts, indices) {
        Some(alloc) => Placement::Mesh(alloc),
        None => Placement::Defer,
    }
}

/// One chunk's draw data: where it is relative to the eye that sees it,
/// which world it belongs to, and the planes masking that world.
///
/// The clip belongs to the WORLD, not the chunk — a world seen through a
/// portal is masked to the opening wherever it is drawn — and the world
/// travels with the chunk because one draw list can only be bound to one
/// material table otherwise.
fn draw_uniform(key: ChunkKey, eye: DVec3, planes: &[Vec4]) -> ChunkDrawUniform {
    let rel = key.min_corner_m() - eye;
    let count = planes.len().min(MAX_CLIP_PLANES);
    let mut clip = [Vec4::ZERO; MAX_CLIP_PLANES];
    clip[..count].copy_from_slice(&planes[..count]);
    ChunkDrawUniform {
        offset: Vec4::new(
            rel.x as f32,
            rel.y as f32,
            rel.z as f32,
            key.voxel_size_m() as f32,
        ),
        head: UVec4::new(count as u32, u32::from(key.world), 0, 0),
        clip,
    }
}

/// Generation priority: distance to the chunk's AABB normalized by its edge
/// length — coarse chunks and near chunks first, uniformly across levels.
fn gen_priority(key: ChunkKey, camera: DVec3) -> f64 {
    let min = key.min_corner_m();
    let max = min + DVec3::splat(key.edge_m());
    let closest = camera.clamp(min, max);
    // Distance normalized by chunk size: the chunk the player stands in
    // always generates first, and far coverage chunks (huge edges) still
    // rank early. (A coarse-first level bias lived here briefly for
    // progressive load-in; the genesis bootstrap made it obsolete and it
    // pushed the player's chunk to the back of the queue.)
    camera.distance(closest) / key.edge_m()
}

/// Times `plan_frame` without wrapping every caller's borrow: the shim
/// takes the system's parameters and hands them straight on.
#[allow(clippy::too_many_arguments)]
fn plan_frame(
    gpu: ResMut<ChunkGpuResources>,
    table: ResMut<ChunkTable>,
    extracted: ResMut<ExtractedChunkCommands>,
    batches: ResMut<FrameBatches>,
    draw_lists: ResMut<VoxelDrawLists>,
    camera: Res<ExtractedCameraPos>,
    world_registries: (Res<WorldViews>, Res<RenderWorlds>),
    env: Res<EnvParams>,
    frustum: Res<ExtractedFrustum>,
    stats: Res<SharedRenderStats>,
    ready_tx: Res<ChunkReadySender>,
    pipelines: Option<Res<ChunkPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    voxel_core::timed!(
        "render::plan_frame",
        4.0,
        plan_frame_inner(
            gpu,
            table,
            extracted,
            batches,
            draw_lists,
            camera,
            world_registries,
            env,
            frustum,
            stats,
            ready_tx,
            pipelines,
            pipeline_cache,
            render_device,
            render_queue,
        )
    );
}

#[allow(clippy::too_many_arguments)]
fn plan_frame_inner(
    mut gpu: ResMut<ChunkGpuResources>,
    mut table: ResMut<ChunkTable>,
    mut extracted: ResMut<ExtractedChunkCommands>,
    mut batches: ResMut<FrameBatches>,
    mut draw_lists: ResMut<VoxelDrawLists>,
    camera: Res<ExtractedCameraPos>,
    // Grouped: Bevy caps a system at 16 parameters.
    (world_views, worlds): (Res<WorldViews>, Res<RenderWorlds>),
    env: Res<EnvParams>,
    frustum: Res<ExtractedFrustum>,
    stats: Res<SharedRenderStats>,
    ready_tx: Res<ChunkReadySender>,
    pipelines: Option<Res<ChunkPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {    // Stage counters: which stage is narrowest is not something the
    // budgets can tell you, since a budget only says what a frame is
    // ALLOWED to do.
    let mut readies = 0u64;

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
    draw_lists.clear();

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
            ChunkCommand::Request {
                key,
                show_on_ready,
                hold,
                ops,
                face_mask,
            } => {
                match table.chunks.get_mut(&key) {
                    None => {
                        table.chunks.insert(
                            key,
                            RenderChunk {
                                state: ChunkState::QueuedGen,
                                visible: false,
                                show_on_ready,
                                hold,
                                ops,
                                face_mask,
                                gen_mask: face_mask,
                                drawn_mask: face_mask,
                                pending: None,
                                requeue: false,
                            },
                        );
                    }
                    Some(chunk) => match chunk.state {
                        // Freed-while-in-flight chunk re-requested. The
                        // in-flight counts were produced under the OLD
                        // mask; meshing them under any other mask breaks
                        // the count/emit agreement (overflowing the slab
                        // alloc). Leave the state Cancelled — the readback
                        // discards the stale result — and mark a fresh
                        // generation with the requested mask and ops.
                        ChunkState::Cancelled { .. } => {
                            chunk.visible = false;
                            chunk.show_on_ready = show_on_ready;
                            chunk.hold = hold;
                            chunk.ops = ops;
                            chunk.face_mask = face_mask;
                            chunk.pending = Some(Pending::Queued);
                        }
                        // Not yet generating: the new mask/ops just apply.
                        ChunkState::QueuedGen => {
                            chunk.ops = ops;
                            chunk.hold = hold;
                            chunk.face_mask = face_mask;
                        }
                        // Live or mid-generation: regenerate in place with
                        // the new mask; the old mesh draws until it's ready.
                        _ => {
                            chunk.ops = ops;
                            chunk.hold = hold;
                            chunk.face_mask = face_mask;
                            // A superseded held result is stale: drop it and
                            // regenerate with the new mask.
                            match chunk.pending.take() {
                                Some(Pending::Held { alloc, .. }) => gpu.slab.free(alloc),
                                Some(p @ (Pending::CountsInFlight { .. }
                                | Pending::ReadyToMesh { .. })) => {
                                    // The in-flight regen carries the OLD
                                    // mask; without a requeue its report
                                    // never matches the new request and
                                    // the epoch stalls to abort.
                                    chunk.pending = Some(p);
                                    chunk.requeue = true;
                                }
                                _ => {}
                            }
                            if chunk.pending.is_none() {
                                chunk.pending = Some(Pending::Queued);
                            }
                        }
                    },
                }
            }
            ChunkCommand::Commit(key) => {
                if let Some(chunk) = table.chunks.get_mut(&key) {
                    chunk.visible = true;
                    match chunk.pending.take() {
                        Some(Pending::Held { alloc, index_count }) => {
                            if let ChunkState::Meshed { alloc: old, .. } = chunk.state {
                                gpu.slab.free(old);
                            }
                            chunk.state = ChunkState::Meshed { alloc, index_count };
                            chunk.drawn_mask = chunk.gen_mask;
                        }
                        Some(Pending::HeldEmpty) => {
                            if let ChunkState::Meshed { alloc, .. } = chunk.state {
                                gpu.slab.free(alloc);
                            }
                            chunk.state = ChunkState::Empty;
                        }
                        other => chunk.pending = other,
                    }
                }
            }
            ChunkCommand::Free(key) => {
                let Some(chunk) = table.chunks.remove(&key) else {
                    continue;
                };
                // A pending regen may hold its own arena slot.
                let mut inflight_slot = None;
                match chunk.pending {
                    Some(Pending::CountsInFlight { slot }) => inflight_slot = Some(slot),
                    Some(Pending::ReadyToMesh { slot, alloc, .. }) => {
                        gpu.arena_free.push(slot);
                        gpu.slab.free(alloc);
                    }
                    Some(Pending::Held { alloc, .. }) => gpu.slab.free(alloc),
                    _ => {}
                }
                match chunk.state {
                    ChunkState::CountsInFlight { slot } => inflight_slot = Some(slot),
                    ChunkState::ReadyToMesh { slot, alloc, .. } => {
                        gpu.arena_free.push(slot);
                        gpu.slab.free(alloc);
                    }
                    ChunkState::Meshed { alloc, .. } => gpu.slab.free(alloc),
                    _ => {}
                }
                if let Some(slot) = inflight_slot {
                    // Result still coming; readback will recycle the slot.
                    table.chunks.insert(
                        key,
                        RenderChunk {
                            state: ChunkState::Cancelled { slot },
                            visible: false,
                            show_on_ready: false,
                            hold: false,
                            ops: None,
                            face_mask: 0,
                            gen_mask: 0,
                            drawn_mask: STALE_MASK,
                            pending: None,
                            requeue: false,
                        },
                    );
                }
            }
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
            // A pending regen's counts route to the pending track.
            if let Some(Pending::CountsInFlight { slot }) = chunk.pending {
                match place(&mut gpu.slab, verts, quads) {
                    Placement::TooBig => {
                        warn!("chunk {key:?} regen exceeds largest slab class; kept old mesh");
                        gpu.arena_free.push(slot);
                        chunk.pending = None;
                        // Report anyway so an epoch waiting on this chunk
                        // can complete instead of wedging on a
                        // pathological mesh.
                        let _ = ready_tx.0.send((*key, chunk.gen_mask));
                        readies += 1;
                    }
                    Placement::Empty => {
                        gpu.arena_free.push(slot);
                        if chunk.hold {
                            chunk.pending = Some(Pending::HeldEmpty);
                        } else {
                            chunk.pending = None;
                            if let ChunkState::Meshed { alloc, .. } = chunk.state {
                                gpu.slab.free(alloc);
                            }
                            chunk.state = ChunkState::Empty;
                        }
                        let _ = ready_tx.0.send((*key, u32::MAX));
                        readies += 1;
                    }
                    Placement::Mesh(alloc) => {
                        chunk.pending = Some(Pending::ReadyToMesh {
                            slot,
                            alloc,
                            index_count: quads * 6,
                        });
                    }
                    Placement::Defer => {
                        // The OLD mesh keeps drawing, so this costs a
                        // stale seam, not a hole. Report it: a waiter
                        // blocking here would hold the whole pass.
                        gpu.arena_free.push(slot);
                        chunk.pending = Some(Pending::Deferred {
                            verts,
                            indices: quads * 6,
                        });
                        let _ = ready_tx.0.send((*key, chunk.gen_mask));
                        readies += 1;
                    }
                }
                // Superseded mid-flight: the result above belongs to the
                // old request — drop it and regenerate with the stored
                // mask/ops. (An AwaitingAlloc result still holds its
                // arena slot; return it.)
                if chunk.requeue {
                    chunk.requeue = false;
                    match chunk.pending.take() {
                        Some(Pending::ReadyToMesh { slot, alloc, .. }) => {
                            gpu.arena_free.push(slot);
                            gpu.slab.free(alloc);
                        }
                        Some(Pending::Deferred { .. })
                        | Some(Pending::HeldEmpty)
                        | Some(Pending::Queued)
                        | None => {}
                        Some(other) => chunk.pending = Some(other),
                    }
                    chunk.pending = Some(Pending::Queued);
                }
                continue;
            }
            match chunk.state {
                ChunkState::Cancelled { slot } => {
                    gpu.arena_free.push(slot);
                    if chunk.pending.take().is_some() {
                        // Resurrected while these counts were in flight:
                        // stale result discarded, generate fresh with the
                        // requested mask/ops.
                        chunk.state = ChunkState::QueuedGen;
                    } else {
                        table.chunks.remove(key);
                    }
                }
                ChunkState::CountsInFlight { slot } => match place(&mut gpu.slab, verts, quads) {
                    Placement::TooBig => {
                        warn!("chunk {key:?} exceeds largest slab class ({verts} verts); dropped");
                        gpu.arena_free.push(slot);
                        table.empty_classified += 1;
                        chunk.state = ChunkState::Empty;
                        let _ = ready_tx.0.send((*key, u32::MAX));
                        readies += 1;
                    }
                    Placement::Empty => {
                        gpu.arena_free.push(slot);
                        table.empty_classified += 1;
                        chunk.state = ChunkState::Empty;
                        let _ = ready_tx.0.send((*key, u32::MAX));
                        readies += 1;
                    }
                    Placement::Mesh(alloc) => {
                        chunk.state = ChunkState::ReadyToMesh {
                            slot,
                            alloc,
                            index_count: quads * 6,
                        };
                    }
                    Placement::Defer => {
                        // A hole, and reported as one so the pass that
                        // asked for it completes. Blocking here is what
                        // turned slab pressure into a frozen world.
                        gpu.arena_free.push(slot);
                        chunk.state = ChunkState::Deferred {
                            verts,
                            indices: quads * 6,
                        };
                        let _ = ready_tx.0.send((*key, u32::MAX));
                        readies += 1;
                    }
                },
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
            .filter(|(_, c)| {
                matches!(c.state, ChunkState::ReadyToMesh { .. })
                    || matches!(c.pending, Some(Pending::ReadyToMesh { .. }))
            })
            .map(|(k, _)| *k)
            .take((*MESH_BUDGET).min((COUNTS_SLOTS as usize).saturating_sub(*GEN_BUDGET)))
            .collect()
    };
    if let Ok(mut st) = stats.0.lock() {
        st.mesh_started += mesh_keys.len() as u64;
    }
    for key in mesh_keys {
        let chunk = table.chunks.get_mut(&key).unwrap();
        // Space was reserved when the counts landed, so this cannot fail
        // and there is nothing here to wait on.
        let (slot, alloc, indices, is_pending) = match (&chunk.state, &chunk.pending) {
            (
                _,
                Some(Pending::ReadyToMesh {
                    slot,
                    alloc,
                    index_count,
                }),
            ) => (*slot, *alloc, *index_count, true),
            (
                ChunkState::ReadyToMesh {
                    slot,
                    alloc,
                    index_count,
                },
                _,
            ) => (*slot, *alloc, *index_count, false),
            _ => unreachable!(),
        };
        mesh_counts_slot -= 1;
        let offset = gpu.gen_uniforms.push(&make_params(
            key,
            slot,
            Some(&alloc),
            mesh_counts_slot,
            (0, 0),
            chunk.gen_mask,
        ));
        batches.mesh.push(MeshEntry {
            uniform_offset: offset,
            first_index: alloc.first_index,
            index_count: indices,
        });
        // The mesh compute is recorded later this frame, before the main
        // pass, so the chunk is immediately drawable. A pending regen swaps
        // its new mesh in atomically (the old one drew until now).
        if is_pending {
            if chunk.hold {
                // Held: old mesh keeps drawing; Commit swaps. Readiness is
                // reported now — the mesh compute records this frame.
                chunk.pending = Some(Pending::Held {
                    alloc,
                    index_count: indices,
                });
                if chunk.requeue {
                    // Superseded mid-flight: this held result carries the
                    // old mask — discard it and regenerate.
                    chunk.requeue = false;
                    if let Some(Pending::Held { alloc, .. }) = chunk.pending.take() {
                        gpu.slab.free(alloc);
                    }
                    chunk.pending = Some(Pending::Queued);
                } else {
                    let _ = ready_tx.0.send((key, chunk.gen_mask));
                    readies += 1;
                }
                freed_slots.push(slot);
                continue;
            }
            if let ChunkState::Meshed { alloc: old, .. } = chunk.state {
                gpu.slab.free(old);
            }
            chunk.pending = None;
            if chunk.requeue {
                chunk.requeue = false;
                chunk.pending = Some(Pending::Queued);
            }
            let _ = ready_tx.0.send((key, chunk.gen_mask));
        } else {
            if chunk.show_on_ready {
                chunk.visible = true;
            }
            let _ = ready_tx.0.send((key, chunk.gen_mask));
        }
        chunk.drawn_mask = chunk.gen_mask;
        chunk.state = ChunkState::Meshed {
            alloc,
            index_count: indices,
        };
        freed_slots.push(slot);
    }

    // 3b. Deferred chunks retry once the slab can hold them again.
    //
    //     A retry costs a full density pass — the arena slot was released
    //     rather than held, which is the point — so this is bounded and
    //     nearest-first. Under sustained pressure the frame must not be
    //     spent re-running work that will defer again.
    {
        let mut retry: Vec<(ChunkKey, f64)> = table
            .chunks
            .iter()
            .filter_map(|(key, chunk)| match (&chunk.state, &chunk.pending) {
                (_, Some(Pending::Deferred { verts, indices }))
                | (ChunkState::Deferred { verts, indices }, _) => Some((*key, *verts, *indices)),
                _ => None,
            })
            .filter(|(_, verts, indices)| gpu.slab.would_fit(*verts, *indices))
            .map(|(key, _, _)| (key, gen_priority(key, camera.0)))
            .collect();
        retry.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (key, _) in retry.into_iter().take(RETRY_BUDGET) {
            let chunk = table.chunks.get_mut(&key).unwrap();
            if matches!(chunk.pending, Some(Pending::Deferred { .. })) {
                chunk.pending = Some(Pending::Queued);
            } else {
                chunk.state = ChunkState::QueuedGen;
            }
        }
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
            .filter(|(_, c)| {
                matches!(c.state, ChunkState::QueuedGen)
                    || (matches!(c.pending, Some(Pending::Queued))
                        && matches!(c.state, ChunkState::Meshed { .. } | ChunkState::Empty))
            })
            .map(|(k, _)| (*k, gen_priority(*k, camera.0)))
            .collect();
        queued.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut started = 0u64;
        let mut starved = false;
        for (key, _) in queued.into_iter().take(*GEN_BUDGET) {
            let Some(slot) = gpu.arena_free.pop() else {
                starved = true;
                break;
            };
            started += 1;
            let chunk = table.chunks.get_mut(&key).unwrap();
            // Copy this chunk's planning ops into the frame buffer. Kept
            // on the chunk (not consumed): any later regen of the same
            // generation request must re-apply them, or CSG content
            // silently vanishes from the reissued mesh.
            let csg = match &chunk.ops {
                Some(ops) => {
                    let offset = frame_ops.len() as u32;
                    frame_ops.extend_from_slice(ops);
                    (offset, ops.len() as u32)
                }
                None => (0, 0),
            };
            let counts_slot = entries.len() as u32;
            let offset = gpu.gen_uniforms.push(&make_params(
                key,
                slot,
                None,
                counts_slot,
                csg,
                chunk.face_mask,
            ));
            batches.gen.push(GenEntry {
                uniform_offset: offset,
            });
            entries.push((key, counts_slot));
            chunk.gen_mask = chunk.face_mask;
            if matches!(chunk.pending, Some(Pending::Queued)) {
                chunk.pending = Some(Pending::CountsInFlight { slot });
            } else {
                chunk.state = ChunkState::CountsInFlight { slot };
            }
        }
        if !entries.is_empty() {
            batches.staging_idx = Some(staging_idx);
            gpu.staging[staging_idx].state = StagingState::PendingMap { entries };
        }
        // Upload this frame's op set (dummy is kept bound when empty).
        gpu.csg_buffer = if frame_ops.is_empty() {
            None
        } else {
            Some(
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("voxel_csg_ops"),
                    contents: bytemuck::cast_slice(&frame_ops),
                    usage: BufferUsages::STORAGE,
                }),
            )
        };
        if let Ok(mut st) = stats.0.lock() {
            st.gen_started += started;
            st.gen_starved += u64::from(starved);
        }
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
        // The eye and frustum of whoever is looking at THIS world.
        let (eye, view_frustum) = world_views
            .0
            .get(usize::from(key.world))
            .and_then(|v| v.as_ref())
            .map_or((camera.0, frustum.0), |(p, f)| (*p, *f));
        if let Some(f) = &view_frustum {
            // World-space AABB, inflated by the skirt depth. f32 is fine at
            // current view ranges; camera-relative culling comes with M6.
            let half = key.edge_m() * 0.5 + key.voxel_size_m() * 6.0;
            let aabb = Aabb {
                center: (key.min_corner_m() + DVec3::splat(key.edge_m() * 0.5))
                    .as_vec3()
                    .into(),
                half_extents: Vec3A::splat(half as f32),
            };
            // intersect_far must be false: with the infinite reversed-Z
            // projection the far half-space is degenerate, and testing it
            // culls everything beyond a few km (bevy's own visibility
            // culling skips it too).
            if !f.intersects_obb(&aabb, &bevy::math::Affine3A::IDENTITY, true, false) {
                culled += 1;
                continue;
            }
        }
        let offset = gpu
            .draw_uniforms
            .push(&draw_uniform(*key, eye, worlds.clip(key.world)));
        draw_lists.0[usize::from(key.world)].push(DrawEntry {
            uniform_offset: offset,
            base_vertex: alloc.base_vertex,
            first_index: alloc.first_index,
            index_count: *index_count,
        });
    }

    gpu.gen_uniforms.write_buffer(&render_device, &render_queue);
    gpu.draw_uniforms
        .write_buffer(&render_device, &render_queue);
    gpu.program_buffer.set(GpuWorldProgram::from_worlds(&worlds));
    gpu.program_buffer
        .write_buffer(&render_device, &render_queue);
    gpu.env_uniform.set(*env);
    gpu.env_uniform.write_buffer(&render_device, &render_queue);

    // The surface map is rebuilt wholesale when the host repaints it —
    // once per kilometre of travel, or when the plan under it changes.
    //
    // WRITTEN, not recreated. Allocating a fresh 16 MB buffer each time
    // measured 10-13 ms on the render thread, which is a frame and a half
    // at the rate this is trying to hold. The size only changes when a
    // world starts or stops painting, so the allocation is the rare case.
    let generation = surface_map_generation(&worlds);
    if generation > 0
        && gpu
            .surface_map
            .as_ref()
            .is_none_or(|(_, uploaded)| *uploaded != generation)
    {
        // SECTION BY SECTION, never concatenated. Building one `Vec` of
        // the whole thing walked 16 MB four times per repaint — into
        // `to_words`, into `sections`, into `offsets`, then into the
        // staging belt — and measured 5-7 ms on the render thread, 38
        // times in 100 s of flight. The texels are behind an `Arc` that
        // already holds them in exactly the layout the shader wants, so
        // the only copy that has to happen is the one into the queue.
        let (offsets, total_words) = surface_map_layout(&worlds);
        let size = (total_words * size_of::<u32>()) as u64;
        let buffer = match gpu.surface_map.as_ref().filter(|(b, _)| b.size() == size) {
            Some((buffer, _)) => buffer.clone(),
            None => render_device.create_buffer(&BufferDescriptor {
                label: Some("voxel_surface_map"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };
        render_queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&offsets));
        for (id, world) in worlds.iter().enumerate().take(MAX_WORLDS) {
            if world.surface_map.section_words() == 0 {
                continue;
            }
            let at = u64::from(offsets[id]) * size_of::<u32>() as u64;
            let header = world.surface_map.header_words();
            render_queue.write_buffer(&buffer, at, bytemuck::cast_slice(&header));
            render_queue.write_buffer(
                &buffer,
                at + (SURFACE_MAP_HEADER * size_of::<u32>()) as u64,
                bytemuck::cast_slice(&world.surface_map.texels),
            );
        }
        gpu.surface_map = Some((buffer, generation));
    }

    // 7. HUD stats.
    if let Ok(mut s) = stats.0.lock() {
        s.reported_ready += readies;
        s.drawn_masks.clear();
        for (key, c) in &table.chunks {
            if c.visible && matches!(c.state, ChunkState::Meshed { .. }) {
                s.drawn_masks.push((*key, c.drawn_mask));
            }
        }
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
            .filter(|c| !matches!(c.state, ChunkState::Meshed { .. } | ChunkState::Empty))
            .count();
        s.arena_free = gpu.arena_free.len() as u32;
        s.slab_used_pages = gpu.slab.used_pages();
        s.slab_total_pages = gpu.slab.config().total_pages;
        s.slab_peak_pages = gpu.slab.peak_used_pages();
        s.slab_longest_free_run = gpu.slab.longest_free_run();
        s.slab_runs = gpu.slab.run_histogram().to_vec();
        s.slab_peak_runs = gpu.slab.peak_run_histogram().to_vec();
        s.slab_capacity_chunks = gpu.slab.capacity_chunks();
        s.slab_pressure = gpu.slab.pressure();
        let mut counts: std::collections::HashMap<&'static str, usize> = Default::default();
        for (key, c) in table.chunks.iter() {
            let state = match c.state {
                ChunkState::QueuedGen => "queued_gen",
                ChunkState::CountsInFlight { .. } => "counts_in_flight",
                ChunkState::Cancelled { .. } => "cancelled",
                ChunkState::ReadyToMesh { .. } => "ready_to_mesh",
                ChunkState::Deferred { .. } => "deferred",
                ChunkState::Empty => "empty",
                ChunkState::Meshed { .. } => "meshed",
            };
            *counts.entry(state).or_default() += 1;
            let pending = match c.pending {
                None => None,
                Some(Pending::Queued) => Some("p_queued"),
                Some(Pending::CountsInFlight { .. }) => Some("p_counts_in_flight"),
                Some(Pending::ReadyToMesh { .. }) => Some("p_ready_to_mesh"),
                Some(Pending::Deferred { .. }) => Some("p_deferred"),
                Some(Pending::Held { .. }) => Some("p_held"),
                Some(Pending::HeldEmpty) => Some("p_held_empty"),
            };
            if let Some(p) = pending {
                *counts.entry(p).or_default() += 1;
            }
            // Per level, so a loose emptiness bound can be attributed to a
            // scale rather than guessed at. A chunk the GPU generated only
            // to classify empty cost a full density pass for nothing.
            let level = key.level.min(15) as usize;
            let bucket: &'static str = match c.state {
                ChunkState::Empty => EMPTY_BY_LEVEL[level],
                ChunkState::Meshed { .. } => MESHED_BY_LEVEL[level],
                _ => continue,
            };
            *counts.entry(bucket).or_default() += 1;
        }
        let with_ops = table.chunks.values().filter(|c| c.ops.is_some()).count();
        let total_ops: usize = table
            .chunks
            .values()
            .filter_map(|c| c.ops.as_ref().map(|o| o.len()))
            .sum();
        counts.insert("with_ops", with_ops);
        counts.insert("total_ops", total_ops);
        s.state_counts = counts.into_iter().collect();
        s.drawn = draw_lists.total();
        s.drawn_per_world = draw_lists.0.iter().map(Vec::len).collect();
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

    let Some(program_binding) = gpu.program_buffer.binding() else {
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
            program_binding.clone(),
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
            program_binding,
        )),
    );

    let encoder = render_context.command_encoder();
    encoder.clear_buffer(&gpu.counts, 0, None);
    // Zero the index ranges this frame's mesh batch will fill: emitted
    // quads overwrite; any shortfall draws degenerate (0,0,0) triangles
    // instead of the previous occupant's stale indices.
    for entry in &batches.mesh {
        encoder.clear_buffer(
            &gpu.index_slab,
            entry.first_index as u64 * 2,
            Some(entry.index_count as u64 * 2),
        );
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("voxel_chunk_work"),
            ..default()
        });

        // Density walks a RUN of y per invocation: a grid over xz, and
        // one layer of it per run. Twin of Y_PER_THREAD in the shader.
        let gen_groups = SAMPLES.div_ceil(8);
        let gen_y_groups = SAMPLES.div_ceil(DENSITY_Y_PER_THREAD);
        // Count and vertex passes cover the extended cell range -1..=32.
        let ext_groups = (CELLS + 2).div_ceil(4);

        // Density for the whole gen batch, then counts.
        pass.set_pipeline(density);
        for entry in &batches.gen {
            pass.set_bind_group(0, &gen_bg, &[entry.uniform_offset]);
            pass.dispatch_workgroups(gen_groups, gen_groups, gen_y_groups);
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
            pass.dispatch_workgroups(ext_groups, ext_groups, ext_groups);
        }
    }

    // Copy the gen batch's counts out for readback.
    if let Some(idx) = batches.staging_idx {
        let staging = &gpu.staging[idx].buffer;
        encoder.copy_buffer_to_buffer(&gpu.counts, 0, staging, 0, COUNTS_SLOTS as u64 * 8);
    }
}

// --- drawing -----------------------------------------------------------------

struct VoxelChunksSpecializer {
    view_layouts: MeshPipelineViewLayouts,
}

/// Keyed by Bevy's own mesh pipeline key: the view bind group layout, the
/// shader defs and the color target must all agree with what the mesh
/// view bind group actually contains, and Bevy derives all three from it.
#[derive(Copy, Clone, PartialEq, Eq, Hash, SpecializerKey)]
struct VoxelChunksKey(MeshPipelineKey);

impl Specializer<RenderPipeline> for VoxelChunksSpecializer {
    type Key = VoxelChunksKey;

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
struct PendingVoxelQueues(PendingQueues);

fn prepare_view_bind_group(
    pipeline: Option<Res<ChunkDrawPipeline>>,
    gpu: Option<Res<ChunkGpuResources>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut bind_groups: ResMut<ViewBindGroupRes>,
) {
    let (Some(pipeline), Some(gpu)) = (pipeline, gpu) else {
        return;
    };
    let (Some(chunk_binding), Some(env_binding)) =
        (gpu.draw_uniforms.binding(), gpu.env_uniform.binding())
    else {
        bind_groups.chunk = None;
        return;
    };
    bind_groups.chunk = Some(render_device.create_bind_group(
        "voxel_chunks_chunk_bg",
        &pipeline_cache.get_bind_group_layout(&pipeline.chunk_layout),
        &BindGroupEntries::sequential((
            chunk_binding,
            env_binding,
            gpu.surface_map
                .as_ref()
                .map_or(&gpu.surface_map_dummy, |(b, _)| b)
                .as_entire_buffer_binding(),
        )),
    ));
}

#[allow(clippy::too_many_arguments)]
fn queue_voxel_chunks(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<ResMut<ChunkDrawPipeline>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    views: Query<crate::pbr_view::PbrViewQuery>,
    dirty_specializations: Res<DirtySpecializations>,
    mut pending_queues: ResMut<PendingVoxelQueues>,
) {
    let Some(mut pipeline) = pipeline else {
        return;
    };
    let draw_function = opaque_draw_functions.read().id::<DrawVoxelChunksCommands>();

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
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let Some(visible) = view_visible_entities.get::<VoxelTerrainMarker>() else {
            continue;
        };
        let view_pending = pending_queues.prepare_for_new_frame(view.retained_view_entity);
        let mesh_key = crate::pbr_view::view_key(
            view,
            camera,
            msaa,
            tonemapping,
            dither,
            shadow_filter_method,
            distance_fog,
        );

        for &main_entity in
            dirty_specializations.iter_to_dequeue(view.retained_view_entity, visible)
        {
            opaque_phase.remove(main_entity);
        }
        for (render_entity, main_entity) in dirty_specializations
            .iter_to_queue(view.retained_view_entity, visible, &view_pending.prev_frame)
            .map(|(re, me)| (*re, *me))
        {
            let Ok(pipeline_id) = pipeline
                .variants
                .specialize(&pipeline_cache, VoxelChunksKey(mesh_key))
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
                (render_entity, main_entity),
                InputUniformIndex::default(),
                BinnedRenderPhaseType::NonMesh,
            );
        }
    }
}

type DrawVoxelChunksCommands = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMaterialBindGroup<3>,
    DrawVoxelChunks,
);

struct DrawVoxelChunks;

impl<P> RenderCommand<P> for DrawVoxelChunks
where
    P: PhaseItem,
{
    type Param = (
        SRes<ChunkGpuResources>,
        SRes<ViewBindGroupRes>,
        SRes<VoxelDrawLists>,
    );
    /// The VIEW decides which world it draws — that is the whole portal.
    type ViewQuery = Option<bevy::ecs::system::lifetimeless::Read<ViewWorld>>;
    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        view_world: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (gpu, bind_groups, draw_lists): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu = gpu.into_inner();
        let bind_groups = bind_groups.into_inner();
        let world = view_world.map_or(0, |w| w.0);
        let Some(draw_list) = draw_lists.into_inner().0.get(usize::from(world)) else {
            return RenderCommandResult::Success;
        };
        let Some(chunk_bg) = &bind_groups.chunk else {
            return RenderCommandResult::Skip;
        };
        if draw_list.is_empty() {
            return RenderCommandResult::Success;
        }
        pass.set_vertex_buffer(0, gpu.vertex_slab.slice(..));
        pass.set_index_buffer(gpu.index_slab.slice(..), IndexFormat::Uint16);
        for entry in draw_list {
            pass.set_bind_group(2, chunk_bg, &[entry.uniform_offset]);
            pass.draw_indexed(
                entry.first_index..entry.first_index + entry.index_count,
                entry.base_vertex as i32,
                0..1,
            );
        }
        RenderCommandResult::Success
    }
}

/// Per-level histogram bucket names. Static strings because the stats map
/// is keyed by `&'static str`; the index is the LOD level.
const EMPTY_BY_LEVEL: [&str; 16] = [
    "empty_L0", "empty_L1", "empty_L2", "empty_L3", "empty_L4", "empty_L5",
    "empty_L6", "empty_L7", "empty_L8", "empty_L9", "empty_L10", "empty_L11",
    "empty_L12", "empty_L13", "empty_L14", "empty_L15",
];
const MESHED_BY_LEVEL: [&str; 16] = [
    "mesh_L0", "mesh_L1", "mesh_L2", "mesh_L3", "mesh_L4", "mesh_L5",
    "mesh_L6", "mesh_L7", "mesh_L8", "mesh_L9", "mesh_L10", "mesh_L11",
    "mesh_L12", "mesh_L13", "mesh_L14", "mesh_L15",
];

/// Slab exhaustion must degrade, never wedge.
#[cfg(test)]
mod slab_pressure_tests {
    use super::*;
    use crate::slab::{SlabAllocator, SlabConfig};

    /// Counts that fit one page.
    const SMALL: (u32, u32) = (100, 100);

    /// A deliberately tiny slab: these tests are about what the
    /// ALLOCATOR does at its limit, and the limit should be reachable in
    /// a few hundred allocations rather than eleven thousand.
    fn tiny() -> SlabConfig {
        SlabConfig {
            page_verts: 2_048,
            index_factor: 6,
            total_pages: 256,
            max_pages_per_chunk: 26,
        }
    }

    fn tiny_slab() -> SlabAllocator {
        SlabAllocator::new(tiny())
    }

    #[test]
    fn placement_reads_the_counts_the_way_the_pipeline_does() {
        let (c, mut slab) = (tiny(), tiny_slab());
        assert!(matches!(place(&mut slab, 0, 10), Placement::Empty));
        assert!(matches!(place(&mut slab, 10, 0), Placement::Empty));
        assert!(matches!(place(&mut slab, 100, 100), Placement::Mesh(_)));
        // Past the longest run one chunk may take, in either dimension.
        let max = c.max_pages_per_chunk * c.page_verts;
        assert!(matches!(place(&mut slab, max + 1, 1), Placement::TooBig));
        assert!(matches!(
            place(&mut slab, 1, max * c.index_factor / 6 + 1),
            Placement::TooBig
        ));
    }

    /// A deferral must take NOTHING. If it consumed a slot the retry
    /// would leak one per attempt and the slab would drain itself.
    #[test]
    fn a_deferral_holds_no_slab_slot() {
        let mut slab = tiny_slab();
        let mut held = Vec::new();
        while let Placement::Mesh(alloc) = place(&mut slab, SMALL.0, SMALL.1) {
            held.push(alloc);
        }
        assert_eq!(slab.free_pages(), 0, "the fixture must exhaust the slab");
        for _ in 0..100 {
            assert!(matches!(place(&mut slab, SMALL.0, SMALL.1), Placement::Defer));
        }
        assert_eq!(slab.free_pages(), 0, "deferring must take nothing");
        // And freeing one makes exactly one placement possible again.
        slab.free(held.pop().unwrap());
        assert!(matches!(place(&mut slab, SMALL.0, SMALL.1), Placement::Mesh(_)));
        assert!(matches!(place(&mut slab, SMALL.0, SMALL.1), Placement::Defer));
    }

    #[test]
    fn would_fit_agrees_with_alloc() {
        let (c, mut slab) = (tiny(), tiny_slab());
        let mut held = Vec::new();
        loop {
            let fits = slab.would_fit(SMALL.0, SMALL.1);
            match slab.alloc(SMALL.0, SMALL.1) {
                Some(a) => {
                    assert!(fits, "would_fit said no and alloc succeeded");
                    held.push(a);
                }
                None => {
                    assert!(!fits, "would_fit said yes and alloc failed");
                    break;
                }
            }
        }
        // Oversized requests are never claimed to fit.
        assert!(!slab.would_fit(c.max_pages_per_chunk * c.page_verts + 1, 1));
    }

    /// THE WEDGE, reproduced against the real allocator and the real
    /// placement rule.
    ///
    /// Demand far past slab capacity. Every chunk takes an arena slot to
    /// generate density, and what it does when the slab says no is the
    /// whole question: holding the slot to wait for space starves the
    /// arena, so no chunk can generate, so nothing is ever freed to make
    /// space — a cycle that only a 10-second timeout broke, once per
    /// pass, forever.
    ///
    /// The invariant: **every arena slot comes back**, whatever the slab
    /// says. A chunk holds an arena slot only while it can still finish.
    #[test]
    fn slab_exhaustion_returns_every_arena_slot() {
        let (c, mut slab) = (tiny(), tiny_slab());
        let arena_capacity = 512usize;
        let mut arena_free = arena_capacity;
        let mut meshed: Vec<SlabAlloc> = Vec::new();
        let mut deferred = 0usize;

        // Ten times more chunks than the slab can ever hold.
        let demand: usize = c.total_pages as usize * 10;
        for _ in 0..demand {
            // Density needs an arena slot.
            if arena_free == 0 {
                panic!("arena starved: a chunk is holding a slot it cannot use");
            }
            arena_free -= 1;
            // Counts land; the slab decides.
            match place(&mut slab, SMALL.0, SMALL.1) {
                Placement::Mesh(alloc) => {
                    meshed.push(alloc);
                    arena_free += 1; // dispatched; slot recycled
                }
                Placement::Empty | Placement::TooBig => arena_free += 1,
                Placement::Defer => {
                    arena_free += 1; // holds NOTHING — this is the fix
                    deferred += 1;
                }
            }
        }

        assert_eq!(arena_free, arena_capacity, "every arena slot came back");
        assert!(deferred > 0, "the fixture must actually exhaust the slab");
        assert_eq!(
            meshed.len(),
            c.total_pages as usize,
            "and the slab was filled to capacity, not left short",
        );
        assert_eq!(slab.pressure().failed as usize, deferred);

        // Freeing the resident set lets the deferred work through.
        for alloc in meshed.drain(..) {
            slab.free(alloc);
        }
        assert!(slab.would_fit(SMALL.0, SMALL.1), "recovery is possible");
    }
}

#[cfg(test)]
mod surface_map_tests {
    use super::*;

    /// The layout table generates the SHADER side of `ChunkParams`; this
    /// struct is still hand-written because it carries `ShaderType`.
    /// Sizes agreeing is not proof the fields do, but it catches the
    /// change that actually happens — a field added to one, not the other.
    #[test]
    fn the_chunk_params_table_is_the_size_of_this_struct() {
        let bytes: usize = voxel_core::layout::CHUNK_PARAMS
            .iter()
            .map(|f| match f.ty {
                "vec4<f32>" | "vec4<i32>" | "vec4<u32>" => 16,
                "vec2<u32>" | "vec2<f32>" => 8,
                "u32" | "i32" | "f32" => 4,
                other => panic!("size of `{other}` is not known here"),
            })
            .sum();
        assert_eq!(
            bytes,
            core::mem::size_of::<ChunkParams>(),
            "voxel_core::layout::CHUNK_PARAMS and ChunkParams disagree"
        );
    }

    /// Reads a `const NAME: u32 = N u;` out of the draw shader, which is
    /// where the map is read: the mesh pass chose the material per VERTEX
    /// and threw away everything finer than the vertex spacing.
    fn shader_const(name: &str) -> usize {
        let src = include_str!("shaders/voxel_chunk_draw.wgsl");
        let prefix = format!("const {name}:");
        let line = src
            .lines()
            .find(|l| l.trim_start().starts_with(&prefix))
            .unwrap_or_else(|| panic!("{name} is not declared in the draw shader"));
        line.rsplit('=')
            .next()
            .unwrap()
            .trim()
            .trim_end_matches(';')
            .trim_end_matches('u')
            .parse()
            .unwrap()
    }

    /// The header is a layout twin: Rust writes it, WGSL indexes it. Drift
    /// does not fail to compile — the shader reads texels out of the
    /// threshold table and paints the world with aliased float bits.
    #[test]
    fn the_header_layout_matches_the_shader() {
        assert_eq!(shader_const("SURFACE_MAP_THRESHOLDS"), SURFACE_MAP_THRESHOLDS);
        assert_eq!(shader_const("SURFACE_MAP_HEADER"), SURFACE_MAP_HEADER);
    }

    /// Every material answers, whether or not the host named it, and at
    /// the index the shader will read it from.
    #[test]
    fn a_named_material_hands_over_later_than_the_rest() {
        let map = SurfaceMap {
            size: 1,
            min_voxel_m: 3.2,
            coarse_from: vec![(4, 6.4)],
            texels: std::sync::Arc::new(vec![0]),
            ..Default::default()
        };
        let words = map.header_words();
        assert_eq!(words.len(), SURFACE_MAP_HEADER);
        assert_eq!(
            map.section_words(),
            SURFACE_MAP_HEADER + 1,
            "the texel follows the header",
        );
        let at = |id: usize| f32::from_bits(words[SURFACE_MAP_THRESHOLDS + id]);
        assert_eq!(at(3), 3.2, "an unnamed material keeps the default");
        assert_eq!(at(4), 6.4, "a named one takes over only when coarser");
        assert_eq!(at(255), 3.2, "the table covers every id a texel can hold");
    }
}

/// Worlds coexist without touching each other.
///
/// Every defect these pin was live at once, and none of them failed a
/// test, because not one test loaded a second world. Two worlds share
/// coordinates and one set of GPU resources by design, so anything the
/// renderer keeps globally is silently shared between levels that have
/// never heard of each other.
#[cfg(test)]
mod multi_world_tests {
    use super::*;
    use voxel_core::worldop::WorldOp;

    fn world(ops: usize, seed: u32) -> RenderWorld {
        RenderWorld {
            program: WorldProgram {
                ops: std::sync::Arc::new((0..ops).map(|i| WorldOp::new(i as u32)).collect()),
                seed,
                sun_dir: Vec3::new(0.0, 1.0, 0.0),
            },
            materials: material_table([]),
            ..Default::default()
        }
    }

    fn painted(material: u32) -> SurfaceMap {
        SurfaceMap {
            size: 2,
            texel_m: 4.0,
            min_voxel_m: 1.0,
            texels: std::sync::Arc::new(vec![material]),
            generation: 1,
            ..Default::default()
        }
    }

    /// Ids are handed out in order and are the index in both directions.
    /// The GPU header array, `ChunkKey::world` and this vector are the
    /// same number, which is the only reason a chunk can carry nothing but
    /// its key.
    #[test]
    fn registration_hands_out_the_index_as_the_id() {
        let mut worlds = RenderWorlds::default();
        assert_eq!(worlds.register(world(1, 10)), 0);
        assert_eq!(worlds.register(world(2, 20)), 1);
        assert_eq!(worlds.len(), 2);
        assert_eq!(worlds.get(1).unwrap().program.seed, 20);
        assert!(worlds.get(2).is_none(), "an unregistered world is absent");
    }

    #[test]
    #[should_panic(expected = "at most")]
    fn registering_past_the_program_buffer_says_so() {
        let mut worlds = RenderWorlds::default();
        for _ in 0..=MAX_WORLDS {
            worlds.register(world(1, 0));
        }
    }

    /// A world arriving must not disturb the worlds already drawing.
    ///
    /// This is the shape of the bug that made opening a portal wipe the
    /// near world's ground: registering world 1 reached global state that
    /// world 0 was already using.
    #[test]
    fn a_new_world_leaves_the_others_untouched() {
        let mut worlds = RenderWorlds::default();
        worlds.register(world(3, 7));
        worlds.get_mut(0).unwrap().surface_map = painted(4);
        worlds.get_mut(0).unwrap().clip = vec![Vec4::X];
        let before = worlds.get(0).unwrap().clone();

        worlds.register(world(5, 9));

        let after = worlds.get(0).unwrap();
        assert_eq!(after.program.seed, before.program.seed);
        assert_eq!(after.program.ops.len(), before.program.ops.len());
        assert_eq!(after.clip, before.clip);
        assert_eq!(after.surface_map.size, before.surface_map.size);
        assert_eq!(after.materials, before.materials);
        // And the packed GPU program still gives world 0 its own ops.
        let gpu = GpuWorldProgram::from_worlds(&worlds);
        assert_eq!(gpu.worlds[0].count.x, 0, "world 0 still starts at 0");
        assert_eq!(gpu.worlds[0].count.y, 3);
    }

    /// Worlds are concatenated into one op buffer and addressed by
    /// (offset, count). A world reading another's range renders the wrong
    /// level's terrain with no error anywhere.
    #[test]
    fn the_program_buffer_gives_each_world_its_own_range() {
        let mut worlds = RenderWorlds::default();
        worlds.register(world(3, 7));
        worlds.register(world(5, 9));
        let gpu = GpuWorldProgram::from_worlds(&worlds);

        assert_eq!((gpu.worlds[0].count.x, gpu.worlds[0].count.y), (0, 3));
        assert_eq!((gpu.worlds[1].count.x, gpu.worlds[1].count.y), (3, 5));
        assert_eq!(gpu.worlds[0].count.w, 7, "each world keeps its own seed");
        assert_eq!(gpu.worlds[1].count.w, 9);
        assert_eq!(gpu.ops.len(), 8);
        // Ranges do not overlap and cover the buffer.
        assert_eq!(gpu.ops[3].meta.x, 0, "world 1's ops start over at kind 0");
        // An unregistered world is empty, not a view of world 0.
        assert_eq!(gpu.worlds[2].count.y, 0);
    }

    /// A material id means whatever ITS level says. Planet's 1 and the
    /// megastructure's 1 are different recipes, and before the tables were
    /// per world the only thing keeping two levels apart was an assert in
    /// the host asking them not to collide.
    #[test]
    fn a_material_id_means_something_different_in_each_world() {
        let red = WorldMaterial {
            c0: Vec4::new(1.0, 0.0, 0.0, 0.0),
            ..Default::default()
        };
        let blue = WorldMaterial {
            c0: Vec4::new(0.0, 0.0, 1.0, 0.0),
            ..Default::default()
        };

        let mut worlds = RenderWorlds::default();
        worlds.register(RenderWorld {
            materials: material_table([(1, red)]),
            ..world(1, 0)
        });
        worlds.register(RenderWorld {
            materials: material_table([(1, blue)]),
            ..world(1, 0)
        });

        assert_eq!(worlds.get(0).unwrap().materials[1].c0, red.c0);
        assert_eq!(worlds.get(1).unwrap().materials[1].c0, blue.c0);
        // Every id answers, so an unassigned one is neutral gray rather
        // than whatever the other world put there.
        assert_eq!(
            worlds.get(1).unwrap().materials[3],
            WorldMaterial::default()
        );
    }

    /// The (world, id) → slab slot map is a layout twin: Rust fills it,
    /// the draw shader indexes it. Drift compiles and shades the terrain
    /// with another world's recipe.
    #[test]
    fn the_slot_index_matches_the_draw_shader() {
        let src = include_str!("shaders/voxel_chunk_draw.wgsl");
        assert!(
            src.contains(&format!(
                "let i = chunk.head.y * {MATERIAL_SLOTS}u + min(id, {}u);",
                MATERIAL_SLOTS - 1
            )),
            "material_for must index (world, id) world-major with {MATERIAL_SLOTS} \
             ids per world — twin of material_slot_index",
        );
        assert!(
            src.contains(&format!("array<vec4<u32>, {MATERIAL_SLOT_VEC4S}>")),
            "the shader's material_slots must hold {MATERIAL_SLOT_VEC4S} vec4s",
        );
        // The packed slot map is one byte per entry, so every
        // (world, id) pair must fit a u8. A const assert, because
        // raising either constant should fail the BUILD, not a test run.
        const _: () = assert!(
            MAX_WORLDS * MATERIAL_SLOTS <= 256,
            "the slot map packs one byte per (world, id)"
        );
        assert_eq!(material_slot_index(0, 0), 0);
        assert_eq!(material_slot_index(0, 7), 7);
        assert_eq!(material_slot_index(1, 0), MATERIAL_SLOTS);
        assert_eq!(material_slot_index(3, 7), 3 * MATERIAL_SLOTS + 7);
        // Out-of-range ids clamp INTO their own world, never into the next.
        assert_eq!(material_slot_index(1, 99), 2 * MATERIAL_SLOTS - 1);
    }

    /// Each world's painted raster lives in its own section, found through
    /// the offset table. One global raster painted this level's rivers
    /// onto every other loaded level, because worlds share coordinates.
    #[test]
    fn each_world_finds_its_own_surface_map() {
        let mut worlds = RenderWorlds::default();
        worlds.register(world(1, 0));
        worlds.register(world(1, 0));
        worlds.register(world(1, 0));
        worlds.get_mut(0).unwrap().surface_map = painted(0xAA);
        // World 1 paints nothing.
        worlds.get_mut(2).unwrap().surface_map = painted(0xBB);

        let words = surface_map_words(&worlds);
        let base = |world: usize| words[world] as usize;
        assert_eq!(base(0), MAX_WORLDS, "the first section follows the table");
        assert_eq!(base(1), 0, "0 means this world paints nothing");
        assert_ne!(base(2), 0);
        assert_ne!(base(0), base(2), "sections must not overlap");
        // Each section starts with its own size and holds its own texels.
        assert_eq!(words[base(0)], 2);
        assert_eq!(words[base(0) + SURFACE_MAP_HEADER], 0xAA);
        assert_eq!(words[base(2) + SURFACE_MAP_HEADER], 0xBB);
        // An unregistered world reads the table's zero, so the shader's
        // "base == 0 means nothing painted" covers it.
        assert_eq!(words[3], 0);
    }

    /// The draw shader indexes the offset table by the chunk's world, and
    /// treats 0 as "nothing painted". Both halves are load-bearing.
    #[test]
    fn the_surface_map_table_matches_the_draw_shader() {
        let src = include_str!("shaders/voxel_chunk_draw.wgsl");
        assert!(
            src.contains("let base = surface_map[chunk.head.y];"),
            "painted_material must index the table by the chunk's world",
        );
        assert!(
            src.contains("if (base == 0u) {"),
            "offset 0 must mean this world paints nothing",
        );
        // The table has to be long enough for every world an id can name.
        let worlds = RenderWorlds::default();
        assert_eq!(surface_map_words(&worlds).len(), MAX_WORLDS);
    }

    /// Only a world that was given planes is masked. A portal masks the
    /// world it shows; the world you are standing in must keep drawing
    /// everywhere.
    #[test]
    fn a_clip_masks_only_the_world_it_was_set_for() {
        let mut worlds = RenderWorlds::default();
        worlds.register(world(1, 0));
        worlds.register(world(1, 0));
        worlds.get_mut(1).unwrap().clip = vec![Vec4::X, Vec4::Y];

        assert!(worlds.clip(0).is_empty(), "the near world is unmasked");
        assert_eq!(worlds.clip(1).len(), 2);
        assert!(worlds.clip(9).is_empty(), "an absent world clips nothing");

        let near = draw_uniform(ChunkKey::new(0, IVec3::ZERO), DVec3::ZERO, worlds.clip(0));
        assert_eq!(near.head.x, 0);
        let far = draw_uniform(
            ChunkKey::in_world(1, 0, IVec3::ZERO),
            DVec3::ZERO,
            worlds.clip(1),
        );
        assert_eq!(far.head.x, 2);

        // Closing the portal stops the mask: republished every frame.
        worlds.clear_clips();
        assert!(worlds.clip(1).is_empty());
    }

    /// A chunk's draw carries its own world, because one draw list is
    /// bound to one material table and the fragment shader has no other
    /// way to know which level it is shading.
    #[test]
    fn a_chunk_draw_carries_its_world() {
        let key = ChunkKey::in_world(2, 3, IVec3::new(1, 0, -1));
        let uniform = draw_uniform(key, DVec3::ZERO, &[]);
        assert_eq!(uniform.head.y, 2);
        // And the offset is relative to the eye that sees THAT world, not
        // to a single global camera: two worlds are viewed from different
        // points in the same frame.
        let eye = key.min_corner_m();
        assert_eq!(draw_uniform(key, eye, &[]).offset.truncate(), Vec3::ZERO);
    }

    /// A material asset must not be touched when its recipe has not
    /// changed.
    ///
    /// `get_mut` marks the asset modified, Bevy re-prepares it, and the
    /// bind group is rebuilt under the draw. Do it every frame and it
    /// never settles: `SetMaterialBindGroup` skips, and the terrain
    /// vanishes with a full draw list, no error and no validation
    /// warning. That is exactly what shipped — one host system taking
    /// `RenderWorlds` mutably every frame was enough.
    #[test]
    fn unchanged_recipes_do_not_touch_their_assets() {
        use bevy::ecs::system::RunSystemOnce;

        let mut app = bevy::app::App::new();
        app.init_resource::<Assets<VoxelSurfaceMaterial>>()
            .init_resource::<TerrainMaterials>()
            .init_resource::<RenderWorlds>();
        app.world_mut()
            .resource_mut::<RenderWorlds>()
            .register(world(1, 0));
        app.world_mut().spawn(VoxelTerrainMarker);

        // First pass builds the handles and writes every recipe.
        app.world_mut()
            .run_system_once(sync_terrain_materials)
            .unwrap();
        assert_eq!(
            app.world().resource::<TerrainMaterials>().0[0].len(),
            MATERIAL_SLOTS,
        );

        // Second pass, with the same recipes but the resource marked
        // changed — the shape a host system that touches `RenderWorlds`
        // every frame creates.
        app.world_mut().clear_trackers();
        app.world_mut().resource_mut::<RenderWorlds>().set_changed();
        app.world_mut()
            .run_system_once(sync_terrain_materials)
            .unwrap();
        assert!(
            !app.world()
                .resource_ref::<Assets<VoxelSurfaceMaterial>>()
                .is_changed(),
            "a recipe that did not change must not modify its asset",
        );

        // But a recipe that DID change still applies.
        app.world_mut().clear_trackers();
        {
            let mut worlds = app.world_mut().resource_mut::<RenderWorlds>();
            worlds.get_mut(0).unwrap().materials[1].c0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        }
        app.world_mut()
            .run_system_once(sync_terrain_materials)
            .unwrap();
        let handle = app.world().resource::<TerrainMaterials>().0[0][1].clone();
        let assets = app.world().resource::<Assets<VoxelSurfaceMaterial>>();
        assert_eq!(assets.get(&handle).unwrap().recipe.c0.x, 1.0);
    }

    /// More planes than the uniform holds must truncate, not overrun.
    #[test]
    fn clip_planes_saturate_at_the_uniform_size() {
        let planes = vec![Vec4::X; MAX_CLIP_PLANES + 3];
        let uniform = draw_uniform(ChunkKey::new(0, IVec3::ZERO), DVec3::ZERO, &planes);
        assert_eq!(uniform.head.x as usize, MAX_CLIP_PLANES);
    }
}
