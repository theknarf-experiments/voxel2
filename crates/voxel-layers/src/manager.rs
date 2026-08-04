//! The layer manager: registration, recursive on-demand generation with
//! cross-thread deduplication, and the chunk cache.

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};

use glam::IVec3;
use voxel_core::seed::{chunk_seed, splitmix64, Rng};

use crate::layer::{chunk_bounds, chunk_range, Dep, IAabb, Layer};

type ErasedChunk = Arc<dyn Any + Send + Sync>;
type Slot = Arc<OnceLock<ErasedChunk>>;
type GenerateFn = Box<dyn Fn(&LayerManager, IVec3) -> ErasedChunk + Send + Sync>;

struct LayerEntry {
    name: &'static str,
    stable_id: u64,
    extent: IVec3,
    deps: Vec<Dep>,
    generate: GenerateFn,
}

/// Owns all registered layers and their generated chunks.
///
/// Generation is blocking and recursive: requesting a chunk generates its
/// dependency chunks first (deduplicated across threads — concurrent
/// requests for the same chunk generate it exactly once). Everything is
/// regenerable, so the cache can be dropped at any time.
pub struct LayerManager {
    world_seed: u64,
    layers: HashMap<TypeId, LayerEntry>,
    cache: Mutex<HashMap<(TypeId, IVec3), Slot>>,
}

thread_local! {
    /// Generation stack for cycle detection (a cycle would otherwise
    /// deadlock on the chunk's `OnceLock`).
    static GEN_STACK: RefCell<Vec<(TypeId, IVec3)>> = const { RefCell::new(Vec::new()) };
}

impl LayerManager {
    pub fn new(world_seed: u64) -> Self {
        Self {
            world_seed,
            layers: HashMap::new(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Register a layer. Its dependencies must already be registered, which
    /// makes the layer graph a DAG by construction.
    pub fn register<L: Layer>(&mut self, layer: L) {
        let deps = layer.dependencies();
        for dep in &deps {
            assert!(
                self.layers.contains_key(&dep.layer),
                "layer {:?} depends on an unregistered layer; register dependencies first",
                L::NAME
            );
        }
        assert!(
            !self.layers.contains_key(&TypeId::of::<L>()),
            "layer {:?} registered twice",
            L::NAME
        );

        let extent = layer.chunk_extent();
        assert!(
            extent.cmpge(IVec3::ZERO).all(),
            "chunk_extent must be non-negative"
        );
        let arc = Arc::new(layer);
        let generate: GenerateFn = Box::new(move |mgr, coord| {
            let ctx = LayerCtx::<L> {
                mgr,
                coord,
                _layer: PhantomData,
            };
            Arc::new(arc.generate(&ctx, coord))
        });
        let mut id_hash = splitmix64(0xC0FFEE);
        for b in L::NAME.bytes() {
            id_hash = splitmix64(id_hash ^ b as u64);
        }
        self.layers.insert(
            TypeId::of::<L>(),
            LayerEntry {
                name: L::NAME,
                stable_id: id_hash,
                extent,
                deps,
                generate,
            },
        );
    }

    /// All chunks of `L` covering `bounds` (world meters), generating any
    /// that don't exist yet. This is the top-level entry point streaming
    /// code uses.
    pub fn get<L: Layer>(&self, bounds: IAabb) -> LayerView<L> {
        let entry = self.entry_of::<L>();
        let (lo, hi) = chunk_range(entry.extent, bounds);
        let mut chunks = Vec::new();
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let coord = IVec3::new(x, y, z);
                    chunks.push((coord, self.get_chunk::<L>(coord)));
                }
            }
        }
        LayerView { chunks }
    }

    /// A single chunk of `L`, generating it (and its dependencies) if needed.
    pub fn get_chunk<L: Layer>(&self, coord: IVec3) -> Arc<L::Chunk> {
        let erased = self.get_chunk_erased(TypeId::of::<L>(), coord);
        erased
            .downcast::<L::Chunk>()
            .expect("layer chunk type mismatch")
    }

    fn get_chunk_erased(&self, type_id: TypeId, coord: IVec3) -> ErasedChunk {
        let key = (type_id, coord);
        let slot: Slot = {
            let mut cache = self.cache.lock().unwrap();
            cache.entry(key).or_default().clone()
        };
        if let Some(chunk) = slot.get() {
            return chunk.clone();
        }

        GEN_STACK.with(|stack| {
            assert!(
                !stack.borrow().contains(&key),
                "layer dependency cycle detected at {:?} {coord}",
                self.layers[&type_id].name
            );
            stack.borrow_mut().push(key);
        });
        let result = slot
            .get_or_init(|| (self.layers[&type_id].generate)(self, coord))
            .clone();
        GEN_STACK.with(|stack| {
            stack.borrow_mut().pop();
        });
        result
    }

    /// Number of cached chunks (all layers).
    pub fn cached_chunks(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Drop every cached chunk. Safe at any quiescent point — everything is
    /// regenerable. (Finer-grained rolling eviction arrives with streaming.)
    pub fn evict_all(&self) {
        self.cache.lock().unwrap().clear();
    }

    fn entry_of<L: Layer>(&self) -> &LayerEntry {
        self.layers
            .get(&TypeId::of::<L>())
            .unwrap_or_else(|| panic!("layer {:?} is not registered", L::NAME))
    }
}

/// Read access handed to [`Layer::generate`]: the chunk's own identity/seed
/// plus padded views of declared dependency layers.
pub struct LayerCtx<'a, L: Layer> {
    mgr: &'a LayerManager,
    coord: IVec3,
    _layer: PhantomData<L>,
}

impl<L: Layer> LayerCtx<'_, L> {
    /// Deterministic seed for this chunk.
    pub fn seed(&self) -> u64 {
        let entry = self.mgr.entry_of::<L>();
        chunk_seed(self.mgr.world_seed, entry.stable_id, self.coord)
    }

    /// Deterministic RNG for this chunk.
    pub fn rng(&self) -> Rng {
        Rng::new(self.seed())
    }

    /// World-space bounds of this chunk.
    pub fn chunk_bounds(&self) -> IAabb {
        chunk_bounds(self.mgr.entry_of::<L>().extent, self.coord)
    }

    /// Read chunks of dependency layer `D` covering `bounds`.
    ///
    /// Panics if `D` was not declared in [`Layer::dependencies`], or if
    /// `bounds` exceeds this chunk's bounds inflated by the declared padding
    /// — the LayerProcGen containment rule that determinism rests on.
    pub fn get<D: Layer>(&self, bounds: IAabb) -> LayerView<D> {
        let entry = self.mgr.entry_of::<L>();
        let dep = entry
            .deps
            .iter()
            .find(|d| d.layer == TypeId::of::<D>())
            .unwrap_or_else(|| {
                panic!(
                    "layer {:?} reads {:?} without declaring it as a dependency",
                    L::NAME,
                    D::NAME
                )
            });
        let allowed = self.chunk_bounds().inflate(dep.padding);
        assert!(
            allowed.contains(bounds),
            "layer {:?} reads {:?} outside its declared padding: allowed {:?}, requested {:?}",
            L::NAME,
            D::NAME,
            allowed,
            bounds
        );
        self.mgr.get::<D>(bounds)
    }
}

/// Chunks of one layer covering a requested region, in deterministic
/// (z, y, x ascending) order.
pub struct LayerView<L: Layer> {
    chunks: Vec<(IVec3, Arc<L::Chunk>)>,
}

impl<L: Layer> LayerView<L> {
    pub fn iter(&self) -> impl Iterator<Item = (IVec3, &L::Chunk)> {
        self.chunks.iter().map(|(c, chunk)| (*c, chunk.as_ref()))
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}
