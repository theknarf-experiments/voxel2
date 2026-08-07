//! The dependency-graph runtime.
//!
//! Generation is driven from the top down and nowhere else: a
//! [`TopDep`] declares "this layer, this level, this much around this
//! point", and everything else exists because something above it needs it.
//! Reads never generate. If a layer reads data it did not declare a
//! dependency for, it gets a diagnostic naming the padding it should have
//! declared — not silent work on the reading thread.
//!
//! When a top dependency moves, the new closure is generated *before* the
//! old one is released, so nothing a consumer still needs disappears
//! underneath it.

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use glam::{DVec3, IVec3};
use voxel_core::seed::{chunk_seed, Rng};

use crate::layer::{layer_key, IAabb, LayerKey};
use crate::layer::{chunk_bounds as bounds_of, chunk_range as range_of};
use crate::traits::{Dep, Layer, LayerChunk, FINAL_LEVEL};
use crate::store::{ChunkSlot, ErasedChunk, Provider, Usage};

type CreateFn = Box<dyn Fn(&LayerGraph, &Arc<ChunkSlot>, u32) + Send + Sync>;
type NewChunkFn = Box<dyn Fn() -> ErasedChunk + Send + Sync>;

struct LayerEntry {
    name: String,
    type_id: TypeId,
    extent: DVec3,
    levels: u32,
    /// `level_padding(l)` captured at registration.
    level_pads: Vec<IVec3>,
    /// Dependencies per level, with `FINAL_LEVEL` already resolved.
    deps: Vec<Vec<Dep>>,
    grid: RwLock<HashMap<IVec3, Arc<ChunkSlot>>>,
    /// Chunk objects returned by `destroy`, ready to be filled again.
    /// Chunks are long-lived at scale and their buffers dominate; letting
    /// a layer keep its capacity is why `destroy` should clear rather than
    /// replace.
    pool: Mutex<Vec<ErasedChunk>>,
    created: AtomicUsize,
    destroyed: AtomicUsize,
    create_nanos: AtomicU64,
    /// Serializes creates of a level that reads its own layer across
    /// chunks (`level_padding != 0`). Without it two neighbours can
    /// deadlock: each holds its own chunk for writing while asking to read
    /// the other's earlier level. Only such levels pay; level 0, which is
    /// all any single-level layer has, is fully parallel.
    self_reading: Vec<bool>,
    level_locks: Vec<Mutex<()>>,
    new_chunk: NewChunkFn,
    create: CreateFn,
    destroy: CreateFn,
}

thread_local! {
    /// Create stack, for cycle detection — a cycle would otherwise
    /// deadlock on a level lock.
    static GEN_STACK: RefCell<Vec<(LayerKey, u32, IVec3)>> = const { RefCell::new(Vec::new()) };
}

/// Owns the registered layers and every resident chunk.
pub struct LayerGraph {
    world_seed: u64,
    /// Per-world value handed to every chunk's create (the world's
    /// generator, host data, …). Opaque here: this crate knows nothing
    /// about what a world is, which is what lets one process host several.
    context: Arc<dyn Any + Send + Sync>,
    layers: HashMap<LayerKey, LayerEntry>,
    /// Helper threads still available to parallel creates. Bounds total
    /// concurrency across nested ensures — without it, a create that
    /// ensures its providers would spawn a fresh pool at every level of
    /// the graph.
    free_workers: AtomicUsize,
    reads_missed: AtomicUsize,
    /// Set when the world is going away. Checked inside the generation
    /// loops so a teardown or a level reload does not have to wait for a
    /// whole pass over a resident set it is about to discard.
    aborting: AtomicBool,
}

/// What one layer instance has cost and is holding.
#[derive(Debug, Clone)]
pub struct LayerStats {
    pub name: String,
    pub resident: usize,
    pub created: usize,
    pub destroyed: usize,
    /// Total time inside this layer's `create`.
    pub create_time: std::time::Duration,
}

impl LayerGraph {
    pub fn new(world_seed: u64) -> Self {
        Self::with_context(world_seed, Arc::new(()))
    }

    pub fn with_context(world_seed: u64, context: Arc<dyn Any + Send + Sync>) -> Self {
        let threads = std::env::var("VOXEL_LAYER_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                // Leave cores for the render and main threads: this runs
                // while the app is drawing.
                std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(1, 6))
            });
        Self {
            world_seed,
            context,
            layers: HashMap::new(),
            free_workers: AtomicUsize::new(threads.saturating_sub(1)),
            reads_missed: AtomicUsize::new(0),
            aborting: AtomicBool::new(false),
        }
    }

    /// Cap on helper threads for parallel creates. Tests pin this to
    /// prove generation is order-independent.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.free_workers = AtomicUsize::new(threads.saturating_sub(1));
        self
    }

    /// Register the default instance of a layer (instance name = `L::NAME`).
    pub fn register<L: Layer>(&mut self, layer: L) {
        self.register_as(L::NAME, layer);
    }

    /// Register a NAMED instance: several differently-parameterized
    /// instances of one layer type, each with its own chunks and seed
    /// stream. Dependencies must already be registered, which makes the
    /// graph a DAG by construction.
    pub fn register_as<L: Layer>(&mut self, instance: &str, layer: L) {
        let key = layer_key(instance);
        assert!(
            !self.layers.contains_key(&key),
            "layer instance {instance:?} registered twice"
        );
        let extent = layer.chunk_extent();
        assert!(
            extent.cmpge(DVec3::ZERO).all() && extent.is_finite(),
            "chunk_extent must be non-negative and finite"
        );
        let levels = layer.levels();
        assert!(levels >= 1, "a layer needs at least one level");

        let mut deps = Vec::with_capacity(levels as usize);
        for level in 0..levels {
            let mut level_deps = layer.dependencies(level);
            for dep in &mut level_deps {
                let entry = self.layers.get(&dep.key).unwrap_or_else(|| {
                    panic!(
                        "layer {instance:?} level {level} depends on an unregistered layer; \
                         register dependencies first"
                    )
                });
                if dep.level == FINAL_LEVEL {
                    dep.level = entry.levels - 1;
                }
                assert!(
                    dep.level < entry.levels,
                    "layer {instance:?} depends on level {} of {:?}, which has {} level(s)",
                    dep.level,
                    entry.name,
                    entry.levels
                );
            }
            deps.push(level_deps);
        }
        let level_pads: Vec<IVec3> = (0..levels).map(|l| layer.level_padding(l)).collect();
        let self_reading: Vec<bool> = (0..levels as usize)
            .map(|l| l > 0 && level_pads[l] != IVec3::ZERO)
            .collect();

        let layer = Arc::new(layer);
        let create_layer = layer.clone();
        let destroy_layer = layer;
        let create: CreateFn = Box::new(move |graph, slot, level| {
            let ctx = ChunkCtx {
                graph,
                layer: &*create_layer,
                key,
                coord: slot.coord,
                level,
                _marker: PhantomData,
            };
            let mut data = slot.data.write().unwrap();
            let chunk = data
                .downcast_mut::<L::Chunk>()
                .expect("layer chunk type mismatch");
            chunk.create(&ctx, level);
        });
        let destroy: CreateFn = Box::new(move |graph, slot, level| {
            let ctx = ChunkCtx {
                graph,
                layer: &*destroy_layer,
                key,
                coord: slot.coord,
                level,
                _marker: PhantomData,
            };
            let mut data = slot.data.write().unwrap();
            let chunk = data
                .downcast_mut::<L::Chunk>()
                .expect("layer chunk type mismatch");
            chunk.destroy(&ctx, level);
        });

        self.layers.insert(
            key,
            LayerEntry {
                name: instance.to_string(),
                type_id: TypeId::of::<L>(),
                extent,
                levels,
                level_pads,
                deps,
                grid: RwLock::new(HashMap::new()),
                pool: Mutex::new(Vec::new()),
                created: AtomicUsize::new(0),
                destroyed: AtomicUsize::new(0),
                create_nanos: AtomicU64::new(0),
                self_reading,
                level_locks: (0..levels).map(|_| Mutex::new(())).collect(),
                new_chunk: Box::new(|| Box::new(L::Chunk::default())),
                create,
                destroy,
            },
        );
    }

    fn entry(&self, key: LayerKey) -> &LayerEntry {
        self.layers
            .get(&key)
            .unwrap_or_else(|| panic!("layer instance is not registered"))
    }

    /// Every registered instance name, for tooling that wants to hold
    /// residency over a whole graph.
    pub fn instances(&self) -> Vec<String> {
        self.layers.values().map(|e| e.name.clone()).collect()
    }

    /// Top level of a registered instance.
    pub fn top_level(&self, instance: &str) -> u32 {
        self.entry(layer_key(instance)).levels - 1
    }

    // ---------------------------------------------------------------- ensure

    /// Generate every chunk of `key` covering `bounds` up to `level`,
    /// resolving each one's declared dependency closure first, and record
    /// them all in `usage` — which is what keeps them resident.
    fn ensure(&self, key: LayerKey, bounds: IAabb, level: u32, usage: &mut Usage) {
        self.ensure_shell(key, bounds, None, level, usage);
    }

    /// `ensure`, minus any chunk whose own bounds lie entirely inside
    /// `hole`. A chunk straddling the hole's edge is kept: it is partly
    /// wanted, and half a chunk is not a thing.
    fn ensure_shell(
        &self,
        key: LayerKey,
        bounds: IAabb,
        hole: Option<IAabb>,
        level: u32,
        usage: &mut Usage,
    ) {
        let entry = self.entry(key);
        let (lo, hi) = range_of(entry.extent, bounds);
        let wanted = |coord: IVec3| -> bool {
            hole.is_none_or(|hole| !hole.contains(bounds_of(entry.extent, coord)))
        };

        // 1. Slots for every covered coordinate, creating empty ones as
        //    needed. Only this step touches the grid lock.
        let mut slots: Vec<Arc<ChunkSlot>> = Vec::new();
        {
            let existing = entry.grid.read().unwrap();
            let mut all_present = true;
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        let coord = IVec3::new(x, y, z);
                        if !wanted(coord) {
                            continue;
                        }
                        match existing.get(&coord) {
                            Some(slot) => slots.push(slot.clone()),
                            None => {
                                all_present = false;
                            }
                        }
                    }
                }
            }
            if !all_present {
                drop(existing);
                slots.clear();
                let mut grid = entry.grid.write().unwrap();
                for z in lo.z..=hi.z {
                    for y in lo.y..=hi.y {
                        for x in lo.x..=hi.x {
                            let coord = IVec3::new(x, y, z);
                            if !wanted(coord) {
                                continue;
                            }
                            let slot = grid.entry(coord).or_insert_with(|| {
                                let data = entry
                                    .pool
                                    .lock()
                                    .unwrap()
                                    .pop()
                                    .unwrap_or_else(|| (entry.new_chunk)());
                                Arc::new(ChunkSlot::new(key, coord, entry.levels as usize, data))
                            });
                            slots.push(slot.clone());
                        }
                    }
                }
            }
        }

        // 2. Generate what is missing, nearest-first — the order a player
        //    notices.
        let center = IVec3::new(
            bounds.min.x.saturating_add(bounds.max.x) / 2,
            bounds.min.y.saturating_add(bounds.max.y) / 2,
            bounds.min.z.saturating_add(bounds.max.z) / 2,
        );
        let extent = entry.extent.max(DVec3::ONE);
        let center_coord = IVec3::new(
            (center.x as f64 / extent.x).floor() as i32,
            (center.y as f64 / extent.y).floor() as i32,
            (center.z as f64 / extent.z).floor() as i32,
        );
        let mut missing: Vec<Arc<ChunkSlot>> = slots
            .iter()
            .filter(|slot| !slot.has_level(level))
            .cloned()
            .collect();
        if !missing.is_empty() {
            missing.sort_by_key(|slot| {
                let d = (slot.coord - center_coord).clamp(IVec3::splat(-30_000), IVec3::splat(30_000));
                d.x * d.x + d.y * d.y + d.z * d.z
            });
            self.create_all(key, &missing, level);
        }

        // 3. Everything covered is now a provider of whatever asked for it,
        //    whether this call generated it or found it.
        for slot in slots {
            slot.add_user(level);
            usage.providers.push((slot, level));
        }
    }

    /// Create `missing` in parallel when there is enough of it to pay for
    /// the threads, and when the worker budget has any left — nested
    /// ensures share one global budget.
    fn create_all(&self, key: LayerKey, missing: &[Arc<ChunkSlot>], level: u32) {
        let entry = self.entry(key);
        if entry.self_reading[level as usize] {
            let _serial = entry.level_locks[level as usize].lock().unwrap();
            for slot in missing {
                self.create_level(key, slot, level);
            }
            return;
        }
        const PARALLEL_MIN: usize = 8;
        let helpers = if missing.len() < PARALLEL_MIN {
            0
        } else {
            self.reserve_workers(missing.len().min(8) - 1)
        };
        if helpers == 0 {
            for slot in missing {
                if self.aborting.load(Ordering::Relaxed) {
                    return;
                }
                self.create_level(key, slot, level);
            }
            return;
        }
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..helpers {
                scope.spawn(|| {
                    while let Some(slot) = missing.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if self.aborting.load(Ordering::Relaxed) {
                            return;
                        }
                        self.create_level(key, slot, level);
                    }
                });
            }
            while let Some(slot) = missing.get(next.fetch_add(1, Ordering::Relaxed)) {
                if self.aborting.load(Ordering::Relaxed) {
                    break;
                }
                self.create_level(key, slot, level);
            }
        });
        self.free_workers.fetch_add(helpers, Ordering::Relaxed);
    }

    fn reserve_workers(&self, want: usize) -> usize {
        let mut free = self.free_workers.load(Ordering::Relaxed);
        loop {
            let take = want.min(free);
            if take == 0 {
                return 0;
            }
            match self.free_workers.compare_exchange_weak(
                free,
                free - take,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return take,
                Err(observed) => free = observed,
            }
        }
    }

    /// Bring one chunk up to `level`: its providers first, then the chunk.
    fn create_level(&self, key: LayerKey, slot: &Arc<ChunkSlot>, level: u32) {
        let entry = self.entry(key);
        let _guard = slot.level_locks[level as usize].lock().unwrap();
        if slot.has_level(level) {
            return; // another worker got here first
        }

        let stack_key = (key, level, slot.coord);
        GEN_STACK.with(|stack| {
            assert!(
                !stack.borrow().contains(&stack_key),
                "layer dependency cycle at {:?} level {level} {}",
                entry.name,
                slot.coord
            );
            stack.borrow_mut().push(stack_key);
        });

        // Providers, exactly as declared. The previous level of this same
        // layer comes first: reaching level N always goes through N-1, so
        // a chunk can never skip a generation pass.
        let own = bounds_of(entry.extent, slot.coord);
        let mut providers = Usage::default();
        if level > 0 {
            let pad = entry.level_pads[level as usize];
            self.ensure(key, dep_bounds(own, pad), level - 1, &mut providers);
        }
        for dep in &entry.deps[level as usize] {
            self.ensure(dep.key, dep_bounds(own, dep.padding), dep.level, &mut providers);
        }

        let started = std::time::Instant::now();
        (entry.create)(self, slot, level);
        entry
            .create_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        entry.created.fetch_add(1, Ordering::Relaxed);

        slot.levels.lock().unwrap()[level as usize].providers = providers.providers;
        slot.level.store(level as i32, Ordering::Release);

        GEN_STACK.with(|stack| {
            stack.borrow_mut().pop();
        });
    }

    // --------------------------------------------------------------- release

    /// Give up a usage record. Chunk levels nothing depends on any more
    /// are destroyed, and their own providers released in turn — the
    /// cascade is iterative, so it holds no lock while it runs.
    pub fn release(&self, usage: Usage) {
        let mut pending: Vec<Provider> = usage.providers;
        while let Some((slot, level)) = pending.pop() {
            let Some(providers) = slot.drop_user(level) else {
                continue; // still wanted by someone else
            };
            self.destroy_level(&slot, level);
            pending.extend(providers);
        }
    }

    /// Generate `level` of a chunk again, keeping whoever depends on it.
    ///
    /// A chunk's content is normally a pure function of its coordinate and
    /// its dependencies, and nothing here would exist if that were always
    /// true. But a voxel chunk is meshed against the LOD of its
    /// neighbours, which follows the camera: the chunk is the same chunk,
    /// at the same coordinate, needing to be built again. The alternative
    /// is pretending it is a different chunk, which would leak its
    /// identity into a coordinate and defeat the sharing this design is
    /// for.
    ///
    /// Its providers are re-resolved before the old set is released —
    /// ensure-new-then-release-old, the same rule a moving top dependency
    /// follows — so a dependency the rebuild still needs is never
    /// destroyed and immediately regenerated.
    pub fn invalidate(&self, instance: &str, coord: IVec3, level: u32) {
        let key = layer_key(instance);
        let entry = self.entry(key);
        let Some(slot) = entry.grid.read().unwrap().get(&coord).cloned() else {
            return; // not resident; nothing to rebuild
        };
        let released = {
            let _guard = slot.level_locks[level as usize].lock().unwrap();
            if !slot.has_level(level) {
                return; // never built, or already being rebuilt
            }
            (entry.destroy)(self, &slot, level);
            slot.level.store(level as i32 - 1, Ordering::Release);
            std::mem::take(&mut slot.levels.lock().unwrap()[level as usize].providers)
        };
        self.create_level(key, &slot, level);
        self.release(Usage::from_providers(released));
    }

    fn destroy_level(&self, slot: &Arc<ChunkSlot>, level: u32) {
        let entry = self.entry(slot.layer);
        {
            let _guard = slot.level_locks[level as usize].lock().unwrap();
            (entry.destroy)(self, slot, level);
            slot.level.store(level as i32 - 1, Ordering::Release);
        }
        entry.destroyed.fetch_add(1, Ordering::Relaxed);
        if level == 0 {
            let removed = entry.grid.write().unwrap().remove(&slot.coord);
            // Only the last holder of the slot may recycle its chunk; a
            // reader that still has the Arc would otherwise see it refilled
            // underneath it.
            if let Some(slot) = removed {
                if let Ok(slot) = Arc::try_unwrap(slot) {
                    entry.pool.lock().unwrap().push(slot.data.into_inner().unwrap());
                }
            }
        }
    }

    // ------------------------------------------------------------ top deps

    /// Bring a top dependency in line with its current focus: generate the
    /// new closure, THEN release the old one. That order is what makes a
    /// moving focus safe — nothing a consumer still needs is ever released
    /// before its replacement exists.
    pub fn process_top(&self, dep: &mut TopDep) {
        if !dep.changed {
            return;
        }
        dep.changed = false;
        let old = dep.current.take();
        if dep.active {
            let mut usage = Usage::default();
            self.ensure_shell(dep.key, dep.bounds(), dep.hole_bounds(), dep.level, &mut usage);
            dep.current = Some(usage);
        }
        if let Some(old) = old {
            self.release(old);
        }
    }

    // ----------------------------------------------------------- read access

    /// Chunks of a named instance covering `bounds`, at its top level.
    /// Resident chunks only — this never generates.
    pub fn view<L: Layer>(&self, instance: &str, bounds: IAabb) -> View<L> {
        let key = layer_key(instance);
        self.view_at(key, bounds, self.entry(key).levels - 1)
    }

    fn view_at<L: Layer>(&self, key: LayerKey, bounds: IAabb, level: u32) -> View<L> {
        let entry = self.entry(key);
        assert_eq!(
            entry.type_id,
            TypeId::of::<L>(),
            "layer instance {:?} is not a {}",
            entry.name,
            std::any::type_name::<L>()
        );
        let (lo, hi) = range_of(entry.extent, bounds);
        let grid = entry.grid.read().unwrap();
        let mut chunks = Vec::new();
        let mut missing = 0usize;
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let coord = IVec3::new(x, y, z);
                    match grid.get(&coord) {
                        Some(slot) if slot.has_level(level) => chunks.push((coord, slot.clone())),
                        _ => missing += 1,
                    }
                }
            }
        }
        if missing > 0 {
            let n = self.reads_missed.fetch_add(missing, Ordering::Relaxed);
            // A miss means someone's working set is not covered. Name the
            // instance and the box, capped so a systemic miss cannot
            // drown the log.
            if n < 40 && std::env::var_os("VOXEL_LOG_LAYERS").is_some() {
                eprintln!(
                    "read miss: {:?} level {level} wanted {:?}..{:?}, {missing} of {} chunks absent",
                    entry.name,
                    bounds.min,
                    bounds.max,
                    missing + chunks.len(),
                );
            }
        }
        View {
            chunks,
            missing,
            _marker: PhantomData,
        }
    }

    /// Chunk levels read that were not resident. Anything but zero means a
    /// consumer's working set is not covered by a top dependency, or a
    /// layer under-declared its padding.
    pub fn reads_missed(&self) -> usize {
        self.reads_missed.load(Ordering::Relaxed)
    }

    pub fn reset_reads_missed(&self) {
        self.reads_missed.store(0, Ordering::Relaxed);
    }

    /// Resident chunks across all layers. Equals the transitive dependency
    /// closure of the active top dependencies — that is the invariant this
    /// design exists to provide, and the eviction timer it replaces.
    pub fn resident_chunks(&self) -> usize {
        self.layers
            .values()
            .map(|entry| entry.grid.read().unwrap().len())
            .sum()
    }

    /// Ask in-flight generation to stop as soon as it can. Irreversible:
    /// the world is being torn down.
    pub fn abort(&self) {
        self.aborting.store(true, Ordering::Relaxed);
    }

    /// Per-layer residency and cost, for the HUD and for deciding whether
    /// a layer is worth what it costs.
    pub fn layer_stats(&self) -> Vec<LayerStats> {
        let mut out: Vec<LayerStats> = self
            .layers
            .values()
            .map(|entry| LayerStats {
                name: entry.name.clone(),
                resident: entry.grid.read().unwrap().len(),
                created: entry.created.load(Ordering::Relaxed),
                destroyed: entry.destroyed.load(Ordering::Relaxed),
                create_time: std::time::Duration::from_nanos(
                    entry.create_nanos.load(Ordering::Relaxed),
                ),
            })
            .collect();
        out.sort_by_key(|s| std::cmp::Reverse(s.create_time));
        out
    }

    /// Resident chunks of one instance.
    pub fn resident_in(&self, instance: &str) -> usize {
        self.entry(layer_key(instance)).grid.read().unwrap().len()
    }
}

/// A root of the generation graph: "this layer, this level, this much
/// around this point". Nothing generates without one.
pub struct TopDep {
    key: LayerKey,
    level: u32,
    size: IVec3,
    /// Optional inner box, centred on the focus, that this dependency
    /// does *not* want.
    ///
    /// The reference has no such thing: a top dependency is a box, and
    /// its LOD sample keeps every level resident in its own nested ball.
    /// That works at four levels and does not at twelve — an interior
    /// world would hold roughly 5k meshed chunks against 3.6k slots.
    /// What LOD actually wants is the annulus where a level is the finest
    /// one covering a point, and a box cannot say that. So a top
    /// dependency can have a hole.
    hole: Option<IVec3>,
    focus: IVec3,
    /// Chunk index range the current focus resolves to. Movement within
    /// one chunk index is not a change — that quantization IS the
    /// hysteresis, and it is why a walking player does not re-plan every
    /// frame.
    indices: Option<(IVec3, IVec3)>,
    current: Option<Usage>,
    changed: bool,
    active: bool,
}

impl TopDep {
    /// A dependency on the top level of `instance`, covering `size` meters
    /// centered on the focus.
    pub fn new(graph: &LayerGraph, instance: &str, size: IVec3) -> Self {
        let level = graph.top_level(instance);
        Self::at_level(instance, level, size)
    }

    pub fn at_level(instance: &str, level: u32, size: IVec3) -> Self {
        Self {
            key: layer_key(instance),
            level,
            size,
            hole: None,
            focus: IVec3::ZERO,
            indices: None,
            current: None,
            changed: true,
            active: true,
        }
    }

    /// Half-open, and never degenerate: a size of 1 covers exactly the
    /// chunk the focus is in. `focus - size/2 .. + size`, matching
    /// LayerProcGen's `GridBounds(focus - size / 2, size)`.
    /// Keep only the shell: chunks fully inside `hole` are not wanted.
    /// Sized like `size`, so a level whose ring runs from `a` to `b` uses
    /// `size = 2b`, `hole = 2a`.
    pub fn with_hole(mut self, hole: IVec3) -> Self {
        self.hole = Some(hole);
        self.changed = true;
        self
    }

    /// The inner box this dependency does not want, in world space.
    ///
    /// A zero on an axis means "unrestricted there", matching how a
    /// collapsed axis is read everywhere else — a planar layer's chunks
    /// span all of y, so a hole that did not also span all of y could
    /// never contain one.
    fn hole_bounds(&self) -> Option<IAabb> {
        let hole = self.hole?;
        if hole.cmple(IVec3::ZERO).all() {
            return None;
        }
        let axis = |size: i32, focus: i32| -> (i32, i32) {
            if size <= 0 {
                (i32::MIN, i32::MAX)
            } else {
                let min = focus - size / 2;
                (min, min.saturating_add(size))
            }
        };
        let (x0, x1) = axis(hole.x, self.focus.x);
        let (y0, y1) = axis(hole.y, self.focus.y);
        let (z0, z1) = axis(hole.z, self.focus.z);
        Some(IAabb::new(
            IVec3::new(x0, y0, z0),
            IVec3::new(x1, y1, z1),
        ))
    }

    pub fn bounds(&self) -> IAabb {
        let size = self.size.max(IVec3::ONE);
        let min = self.focus - size / 2;
        IAabb::new(min, min + size)
    }

    /// Move the focus. Only marks the dependency changed when the covered
    /// chunk indices actually differ.
    pub fn set_focus(&mut self, graph: &LayerGraph, focus: IVec3) {
        self.focus = focus;
        let extent = graph.entry(self.key).extent;
        let indices = range_of(extent, self.bounds());
        if self.indices != Some(indices) {
            self.indices = Some(indices);
            self.changed = true;
        }
    }

    pub fn set_size(&mut self, size: IVec3) {
        if self.size != size {
            self.size = size;
            self.indices = None;
            self.changed = true;
        }
    }

    /// Deactivate: the next `process_top` releases everything this held.
    pub fn set_active(&mut self, active: bool) {
        if self.active != active {
            self.active = active;
            self.changed = true;
        }
    }

    pub fn size(&self) -> IVec3 {
        self.size
    }

    pub fn changed(&self) -> bool {
        self.changed
    }

    /// Chunk levels this dependency is currently holding resident.
    pub fn held(&self) -> usize {
        self.current.as_ref().map_or(0, Usage::len)
    }
}

/// Read access handed to [`LayerChunk::create`]: this chunk's identity and
/// seed, its layer's configuration, and padded views of declared
/// dependencies.
pub struct ChunkCtx<'a, L: Layer> {
    graph: &'a LayerGraph,
    layer: &'a L,
    key: LayerKey,
    coord: IVec3,
    level: u32,
    _marker: PhantomData<L>,
}

impl<L: Layer> ChunkCtx<'_, L> {
    /// The layer this chunk belongs to — its configuration lives there.
    pub fn layer(&self) -> &L {
        self.layer
    }

    pub fn coord(&self) -> IVec3 {
        self.coord
    }

    /// Which level is being generated (0-based).
    pub fn level(&self) -> u32 {
        self.level
    }

    /// World-space bounds of this chunk.
    pub fn chunk_bounds(&self) -> IAabb {
        bounds_of(self.graph.entry(self.key).extent, self.coord)
    }

    /// Deterministic seed for this chunk, distinct per instance and level.
    pub fn seed(&self) -> u64 {
        chunk_seed(
            self.graph.world_seed,
            self.key ^ (self.level as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            self.coord,
        )
    }

    pub fn rng(&self) -> Rng {
        Rng::new(self.seed())
    }

    /// The graph's per-world context, downcast to `C`.
    pub fn context<C: 'static>(&self) -> &C {
        self.graph.context.downcast_ref::<C>().unwrap_or_else(|| {
            panic!(
                "layer {:?} asked for a {} context, but the graph carries a different type",
                L::NAME,
                std::any::type_name::<C>()
            )
        })
    }

    /// Read a declared dependency's default instance.
    pub fn get<D: Layer>(&self, bounds: IAabb) -> View<D> {
        self.get_named::<D>(D::NAME, bounds)
    }

    /// Read a declared dependency instance covering `bounds`.
    ///
    /// Panics if the instance was not declared for this level, or if
    /// `bounds` exceeds this chunk's bounds inflated by the declared
    /// padding — the containment rule determinism rests on. The message
    /// carries the padding that would have covered the read.
    pub fn get_named<D: Layer>(&self, instance: &str, bounds: IAabb) -> View<D> {
        let entry = self.graph.entry(self.key);
        let dep_key = layer_key(instance);
        let dep = entry.deps[self.level as usize]
            .iter()
            .find(|d| d.key == dep_key)
            .unwrap_or_else(|| {
                panic!(
                    "layer {:?} level {} reads {instance:?} without declaring it as a dependency",
                    entry.name, self.level,
                )
            });
        let own = self.chunk_bounds();
        // The read may span a collapsed axis freely — what is *ensured*
        // on one is bounded (see `dep_bounds`), so asking beyond it shows
        // up as a miss, which is the honest report.
        let allowed = own.inflate(dep.padding);
        assert!(
            allowed.contains(bounds),
            "layer {:?} reads {instance:?} outside its declared padding: allowed {:?}, \
             requested {:?} — declare Dep::named({instance:?}, {:?})",
            entry.name,
            allowed,
            bounds,
            needed_padding(own, bounds, dep.padding),
        );
        self.graph.view_at::<D>(dep_key, bounds, dep.level)
    }

    /// Read level `level() - 1` of this same layer, EXCLUDING this chunk —
    /// which the create method already has as `self`. Panics at level 0 or
    /// outside the declared [`Layer::level_padding`].
    pub fn get_self(&self, bounds: IAabb) -> View<L> {
        assert!(
            self.level > 0,
            "layer {:?} level 0 has no previous level to read",
            L::NAME
        );
        let own = self.chunk_bounds();
        let allowed = own.inflate(self.layer.level_padding(self.level));
        assert!(
            allowed.contains(bounds),
            "layer {:?} level {} reads outside its declared level padding: allowed {:?}, \
             requested {:?}",
            L::NAME,
            self.level,
            allowed,
            bounds
        );
        let mut view = self.graph.view_at::<L>(self.key, bounds, self.level - 1);
        view.chunks.retain(|(coord, _)| *coord != self.coord);
        view
    }
}

/// The region a chunk may read from a dependency: its own bounds inflated
/// by the declared padding — except on an axis where it is *collapsed*.
///
/// A planar layer in a 3D world spans all of y, so inflating would ask a
/// volumetric dependency to exist at every height there has ever been. On
/// a collapsed axis the padding is therefore read as an absolute band
/// about the origin: a planar layer declaring 2560 m of y padding means
/// "I read 2560 m above and below", which is the only thing it could
/// sensibly mean.
pub fn dep_bounds(own: IAabb, padding: IVec3) -> IAabb {
    let mut bounds = own.inflate(padding);
    let axis = |unbounded: bool, pad: i32, lo: &mut i32, hi: &mut i32| {
        if unbounded {
            *lo = -pad;
            *hi = pad;
        }
    };
    let (mut min, mut max) = (bounds.min, bounds.max);
    axis(own.min.x == i32::MIN, padding.x, &mut min.x, &mut max.x);
    axis(own.min.y == i32::MIN, padding.y, &mut min.y, &mut max.y);
    axis(own.min.z == i32::MIN, padding.z, &mut min.z, &mut max.z);
    bounds = IAabb::new(min, max);
    bounds
}

/// Per-axis padding that would have covered `requested` from `own`, never
/// shrinking what is already declared — the number to put in the `Dep`.
fn needed_padding(own: IAabb, requested: IAabb, declared: IVec3) -> IVec3 {
    let axis = |own_min: i32, own_max: i32, req_min: i32, req_max: i32, have: i32| {
        own_min
            .saturating_sub(req_min)
            .max(req_max.saturating_sub(own_max))
            .max(have)
    };
    IVec3::new(
        axis(own.min.x, own.max.x, requested.min.x, requested.max.x, declared.x),
        axis(own.min.y, own.max.y, requested.min.y, requested.max.y, declared.y),
        axis(own.min.z, own.max.z, requested.min.z, requested.max.z, declared.z),
    )
}

/// Resident chunks of one layer covering a requested region.
///
/// Iteration borrows each chunk in turn rather than handing out a
/// collection: the chunks are live objects, and holding them all borrowed
/// at once would block the create of anything that shares them.
pub struct View<L: Layer> {
    chunks: Vec<(IVec3, Arc<ChunkSlot>)>,
    missing: usize,
    _marker: PhantomData<L>,
}

/// A borrowed chunk. Holds the read lock for as long as it is alive, so
/// iteration releases each chunk before touching the next.
pub struct ChunkRef<'a, C> {
    guard: std::sync::RwLockReadGuard<'a, ErasedChunk>,
    _marker: PhantomData<C>,
}

impl<C: 'static> std::ops::Deref for ChunkRef<'_, C> {
    type Target = C;
    fn deref(&self) -> &C {
        self.guard
            .downcast_ref::<C>()
            .expect("layer chunk type mismatch")
    }
}

impl<L: Layer> View<L> {
    /// The resident chunks, in deterministic (z, y, x ascending) order.
    pub fn iter(&self) -> impl Iterator<Item = (IVec3, ChunkRef<'_, L::Chunk>)> {
        self.chunks.iter().map(|(coord, slot)| {
            (
                *coord,
                ChunkRef {
                    guard: slot.data.read().unwrap(),
                    _marker: PhantomData,
                },
            )
        })
    }

    pub fn for_each(&self, mut f: impl FnMut(IVec3, &L::Chunk)) {
        for (coord, slot) in &self.chunks {
            let data = slot.data.read().unwrap();
            let chunk = data
                .downcast_ref::<L::Chunk>()
                .expect("layer chunk type mismatch");
            f(*coord, chunk);
        }
    }


    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Chunks that should have been covered but were not resident.
    pub fn missing(&self) -> usize {
        self.missing
    }

    pub fn is_complete(&self) -> bool {
        self.missing == 0
    }
}
