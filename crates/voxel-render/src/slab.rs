//! Bucketed slab allocation for chunk meshes.
//!
//! One big vertex buffer and one big index buffer are partitioned into
//! fixed-size-class regions. Chunks get a slot in the smallest class that
//! fits their exact (pre-counted) vertex/index counts; frees are O(1)
//! free-list pushes. Geometry never moves after allocation.

/// Vertex capacities per size class. Index capacity per slot is
/// `INDEX_FACTOR ×` the class vertex capacity. Allocation checks the exact
/// counted vertex AND index requirements, so a chunk with an unusual
/// index/vertex ratio simply lands in a larger class — the factor is a
/// sizing heuristic, not a correctness bound. The largest class covers the
/// theoretical 34³ extended-cell maximum with skirt twins.
pub const CLASS_VERTS: [u32; 4] = [2_048, 6_144, 16_384, 53_248];
/// Slots per class, sized for ground-level LOD (~2000 live surface chunks
/// across all levels): terrain chunks are mostly ≤2k, architectural chunks
/// (a floor slab + pillars + skirt twins) typically 4–6k.
pub const CLASS_SLOTS: [u32; 4] = [2_048, 1_024, 384, 96];
pub const INDEX_FACTOR: u32 = 6;

/// A granted allocation: ranges into the shared vertex/index buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabAlloc {
    pub class: u8,
    pub slot: u32,
    /// First vertex (in vertices, not bytes).
    pub base_vertex: u32,
    /// First index (in indices, not bytes).
    pub first_index: u32,
}

pub struct SlabAllocator {
    free: [Vec<u32>; 4],
    /// Base offsets (in vertices / indices) of each class region.
    class_vertex_base: [u32; 4],
    class_index_base: [u32; 4],
}

impl Default for SlabAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl SlabAllocator {
    pub fn new() -> Self {
        let mut class_vertex_base = [0u32; 4];
        let mut class_index_base = [0u32; 4];
        let mut vcursor = 0u32;
        let mut icursor = 0u32;
        for c in 0..4 {
            class_vertex_base[c] = vcursor;
            class_index_base[c] = icursor;
            vcursor += CLASS_VERTS[c] * CLASS_SLOTS[c];
            icursor += CLASS_VERTS[c] * INDEX_FACTOR * CLASS_SLOTS[c];
        }
        Self {
            free: core::array::from_fn(|c| (0..CLASS_SLOTS[c]).rev().collect()),
            class_vertex_base,
            class_index_base,
        }
    }

    /// Total vertex capacity of all classes (buffer sizing).
    pub fn total_vertices() -> u64 {
        (0..4)
            .map(|c| CLASS_VERTS[c] as u64 * CLASS_SLOTS[c] as u64)
            .sum()
    }

    /// Total index capacity of all classes (buffer sizing).
    pub fn total_indices() -> u64 {
        Self::total_vertices() * INDEX_FACTOR as u64
    }

    /// Allocate the smallest slot fitting the exact counts, or `None` if the
    /// counts exceed the largest class or every fitting class is exhausted.
    pub fn alloc(&mut self, vertex_count: u32, index_count: u32) -> Option<SlabAlloc> {
        for (class, &class_verts) in CLASS_VERTS.iter().enumerate() {
            if vertex_count > class_verts || index_count > class_verts * INDEX_FACTOR {
                continue;
            }
            if let Some(slot) = self.free[class].pop() {
                return Some(SlabAlloc {
                    class: class as u8,
                    slot,
                    base_vertex: self.class_vertex_base[class] + slot * CLASS_VERTS[class],
                    first_index: self.class_index_base[class]
                        + slot * CLASS_VERTS[class] * INDEX_FACTOR,
                });
            }
            // Class fits but is full — try the next larger one.
        }
        None
    }

    pub fn free(&mut self, alloc: SlabAlloc) {
        let class = alloc.class as usize;
        debug_assert!(alloc.slot < CLASS_SLOTS[class]);
        debug_assert!(!self.free[class].contains(&alloc.slot), "double free");
        self.free[class].push(alloc.slot);
    }

    /// (free, total) slots per class, for the debug HUD.
    pub fn occupancy(&self) -> [(u32, u32); 4] {
        core::array::from_fn(|c| (self.free[c].len() as u32, CLASS_SLOTS[c]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_smallest_fitting_class() {
        let mut slab = SlabAllocator::new();
        assert_eq!(slab.alloc(100, 600).unwrap().class, 0);
        assert_eq!(slab.alloc(CLASS_VERTS[0] + 1, 600).unwrap().class, 1);
        assert_eq!(slab.alloc(CLASS_VERTS[1] + 1, 600).unwrap().class, 2);
        // Index count alone can push into a bigger class.
        assert_eq!(
            slab.alloc(100, CLASS_VERTS[0] * INDEX_FACTOR + 1)
                .unwrap()
                .class,
            1
        );
        // Too big entirely.
        assert!(slab.alloc(CLASS_VERTS[3] + 1, 1).is_none());
    }

    #[test]
    fn ranges_are_disjoint_and_stable() {
        let mut slab = SlabAllocator::new();
        let a = slab.alloc(1_000, 6_000).unwrap();
        let b = slab.alloc(1_000, 6_000).unwrap();
        assert_ne!(a.base_vertex, b.base_vertex);
        assert_ne!(a.first_index, b.first_index);
        // Slots within a class never overlap.
        assert!(a.base_vertex.abs_diff(b.base_vertex) >= CLASS_VERTS[0]);
        assert!(a.first_index.abs_diff(b.first_index) >= CLASS_VERTS[0] * INDEX_FACTOR);
    }

    #[test]
    fn exhaustion_overflows_to_larger_class_and_free_recycles() {
        let mut slab = SlabAllocator::new();
        let mut allocs = Vec::new();
        for _ in 0..CLASS_SLOTS[0] {
            allocs.push(slab.alloc(10, 60).unwrap());
        }
        // Class 0 exhausted → next alloc lands in class 1.
        let spill = slab.alloc(10, 60).unwrap();
        assert_eq!(spill.class, 1);
        // Freeing one class-0 slot makes it available again.
        let freed = allocs.pop().unwrap();
        slab.free(freed);
        assert_eq!(slab.alloc(10, 60).unwrap(), freed);
    }

    #[test]
    fn fuzz_alloc_free_never_overlaps() {
        use voxel_core::seed::Rng;
        let mut slab = SlabAllocator::new();
        let mut live: Vec<(SlabAlloc, u32, u32)> = Vec::new();
        let mut rng = Rng::new(0xF422);
        for _ in 0..20_000 {
            if rng.next_f32() < 0.55 || live.is_empty() {
                let verts = 1 + rng.next_range(CLASS_VERTS[3]);
                let indices = 1 + rng.next_range(CLASS_VERTS[3] * INDEX_FACTOR);
                if let Some(a) = slab.alloc(verts, indices) {
                    // The granted ranges must hold the request…
                    assert!(CLASS_VERTS[a.class as usize] >= verts);
                    // …and never overlap any live allocation.
                    for (b, bv, bi) in &live {
                        let av_end = a.base_vertex + verts;
                        let bv_end = b.base_vertex + bv;
                        assert!(
                            av_end <= b.base_vertex || bv_end <= a.base_vertex,
                            "vertex overlap"
                        );
                        let ai_end = a.first_index + indices;
                        let bi_end = b.first_index + bi;
                        assert!(
                            ai_end <= b.first_index || bi_end <= a.first_index,
                            "index overlap"
                        );
                    }
                    live.push((a, verts, indices));
                }
            } else {
                let i = rng.next_range(live.len() as u32) as usize;
                let (a, _, _) = live.swap_remove(i);
                slab.free(a);
            }
        }
    }

    #[test]
    fn capacity_math_matches_class_tables() {
        let expected: u64 = (0..4)
            .map(|c| CLASS_VERTS[c] as u64 * CLASS_SLOTS[c] as u64)
            .sum();
        assert_eq!(SlabAllocator::total_vertices(), expected);
        assert_eq!(
            SlabAllocator::total_indices(),
            expected * INDEX_FACTOR as u64
        );
    }
}
