//! Page-based slab allocation for chunk meshes.
//!
//! One big vertex buffer and one big index buffer are divided into
//! fixed-size PAGES. A chunk takes a contiguous run of as many pages as
//! its exact (pre-counted) vertex and index counts need; frees are O(1)
//! bit clears. Geometry never moves after allocation.
//!
//! **There is no fixed partition, on purpose.** This used to be four
//! size classes with a hardcoded slot count each, and the split was
//! wrong for every level that was not the one it had been measured on.
//! Terrain chunks are overwhelmingly one page; the megastructure
//! interior is the opposite shape, spread across three sizes. With a
//! static split, terrain pegged the small class at 100% and spilled
//! chunks into slots three times the size they needed while two whole
//! classes sat untouched — and loading two worlds at once, whose
//! demands ADD, made it worse in a way no single split could answer.
//!
//! Levels are data and a level designer can write a shape nobody
//! anticipated, so the allocator does not get to assume one. Pages make
//! the split a runtime consequence of what chunks actually ask for, and
//! [`SlabConfig`] makes the remaining numbers a host's to choose.
//!
//! Placement is size-directed to keep long runs available: single pages
//! are taken from the low end, multi-page runs from the high end. That
//! is what stops a churn of small allocations from perforating the
//! buffer so that no long run exists while a third of it is free.

/// How big the slab is and how finely it is divided. A host passes this
/// to `VoxelChunksPlugin`; a game whose chunks are denser or sparser
/// than this demo's changes it without touching the allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, bevy::prelude::Resource)]
pub struct SlabConfig {
    /// Vertices per page. Also the granularity of internal waste: a
    /// chunk with 100 vertices still occupies one page.
    pub page_verts: u32,
    /// Index capacity per page is `index_factor x` its vertex capacity.
    /// Allocation checks the counted vertex AND index requirements, so a
    /// chunk with an unusual ratio simply takes more pages — the factor
    /// is a sizing heuristic, not a correctness bound.
    pub index_factor: u32,
    /// Pages in the buffer: the memory budget, in units of `page_verts`.
    pub total_pages: u32,
    /// Longest run one chunk may take. A chunk asking for more is
    /// refused rather than allowed to swallow the buffer; it must cover
    /// the mesher's theoretical maximum for one chunk.
    pub max_pages_per_chunk: u32,
}

impl Default for SlabConfig {
    /// The demo's budget. `total_pages` is the same vertex capacity the
    /// four-class partition had, so this change is a re-shaping and not
    /// a memory increase; `max_pages_per_chunk` covers the theoretical
    /// 34-cubed extended-cell maximum.
    ///
    /// These are a STARTING POINT, not a measurement. What a world costs
    /// depends on where you are standing in it, so the allocator reports
    /// peaks (see [`SlabAllocator::peak_used_pages`]) rather than
    /// expecting anyone to have predicted them.
    fn default() -> Self {
        Self {
            page_verts: 2_048,
            index_factor: 6,
            total_pages: 11_392,
            max_pages_per_chunk: 26,
        }
    }
}

impl SlabConfig {
    /// Total vertex capacity (buffer sizing).
    pub fn total_vertices(&self) -> u64 {
        self.total_pages as u64 * self.page_verts as u64
    }

    /// Total index capacity (buffer sizing).
    pub fn total_indices(&self) -> u64 {
        self.total_vertices() * self.index_factor as u64
    }

    /// Pages a chunk with these counts needs.
    pub fn pages_for(&self, vertex_count: u32, index_count: u32) -> u32 {
        vertex_count
            .div_ceil(self.page_verts)
            .max(index_count.div_ceil(self.page_verts * self.index_factor))
            .max(1)
    }
}

/// A granted allocation: ranges into the shared vertex/index buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabAlloc {
    /// First page of the run.
    pub page: u32,
    /// Pages in the run.
    pub pages: u32,
    /// First vertex (in vertices, not bytes).
    pub base_vertex: u32,
    /// First index (in indices, not bytes).
    pub first_index: u32,
}

/// What the allocator has had to do to keep up.
///
/// Pages removed the old `oversized` counter outright: a run is exactly
/// as long as the chunk needs, so there is no larger-slot-than-required
/// case left to count. What remains is the one that matters.
#[derive(Debug, Default, Clone, Copy)]
pub struct SlabPressure {
    /// Allocations that found no run long enough. These become
    /// AwaitingAlloc, which holds an arena slot while it waits.
    pub failed: u64,
    /// Failures where enough pages were free but not CONTIGUOUS — the
    /// signal that separates "the buffer is full" from "the buffer is
    /// fragmented", which want opposite fixes.
    pub fragmented: u64,
}

pub struct SlabAllocator {
    cfg: SlabConfig,
    /// One bit per page; set means allocated. The source of truth.
    used: Vec<u64>,
    used_pages: u32,
    /// Free single pages, as hints. May name a page a multi-page run has
    /// since taken, so entries are verified against `used` on pop.
    singles: Vec<u32>,
    /// Where the next low-end scan starts.
    low_cursor: u32,
    /// Live allocations by run length, indexed by `pages - 1`. The shape
    /// of the working set, which is the thing a partition would have to
    /// have guessed.
    histogram: Vec<u32>,
    /// The same, at its high-water mark.
    ///
    /// A reading taken standing still is one sample of a process that
    /// depends entirely on where you are: fly somewhere denser and the
    /// shape changes. Peaks accumulate over a session, so `mise run fly`
    /// produces evidence a stationary camera cannot.
    peak_histogram: Vec<u32>,
    peak_used_pages: u32,
    peak_live: u32,
    live: u32,
    pressure: SlabPressure,
}

impl Default for SlabAllocator {
    fn default() -> Self {
        Self::new(SlabConfig::default())
    }
}

impl SlabAllocator {
    pub fn new(cfg: SlabConfig) -> Self {
        let runs = cfg.max_pages_per_chunk as usize;
        Self {
            used: vec![0; cfg.total_pages.div_ceil(64) as usize],
            cfg,
            used_pages: 0,
            singles: Vec::new(),
            low_cursor: 0,
            histogram: vec![0; runs],
            peak_histogram: vec![0; runs],
            peak_used_pages: 0,
            peak_live: 0,
            live: 0,
            pressure: SlabPressure::default(),
        }
    }

    pub fn config(&self) -> SlabConfig {
        self.cfg
    }

    /// Pages a chunk with these counts needs.
    pub fn pages_for(&self, vertex_count: u32, index_count: u32) -> u32 {
        self.cfg.pages_for(vertex_count, index_count)
    }

    /// Chunks that can hold a mesh at once — what admission control
    /// compares a world's demand against.
    ///
    /// Derived from what chunks have actually cost rather than declared,
    /// and from the PEAK rather than the present: mean pages per chunk
    /// is ~1 where terrain is thin and several times that in dense
    /// geometry, so admitting against the current camera's number would
    /// over-admit the moment you fly somewhere busier. A world loaded
    /// into an empty slab is admitted against the optimistic one-page
    /// assumption and re-fitted as evidence arrives, which is the shape
    /// the rest of admission control already has.
    pub fn capacity_chunks(&self) -> usize {
        let mean = if self.peak_live == 0 {
            1.0
        } else {
            (self.peak_used_pages as f64 / self.peak_live as f64).max(1.0)
        };
        (self.cfg.total_pages as f64 / mean) as usize
    }

    fn is_free(&self, page: u32) -> bool {
        self.used[(page / 64) as usize] & (1u64 << (page % 64)) == 0
    }

    fn set_range(&mut self, page: u32, pages: u32, allocated: bool) {
        for p in page..page + pages {
            let (word, bit) = ((p / 64) as usize, 1u64 << (p % 64));
            if allocated {
                self.used[word] |= bit;
            } else {
                self.used[word] &= !bit;
            }
        }
    }

    /// First free run of `pages` at or after `from`, scanning upward.
    fn find_run_low(&self, pages: u32, from: u32) -> Option<u32> {
        let mut start = from;
        while start + pages <= self.cfg.total_pages {
            match (start..start + pages).find(|&p| !self.is_free(p)) {
                // Everything before the blocker is too short; resume past it.
                Some(blocked) => start = blocked + 1,
                None => return Some(start),
            }
        }
        None
    }

    /// Last free run of `pages`, scanning downward from the top. Keeps
    /// long runs away from the churn of single pages at the bottom.
    fn find_run_high(&self, pages: u32) -> Option<u32> {
        let mut end = self.cfg.total_pages;
        while end >= pages {
            let start = end - pages;
            match (start..end).rev().find(|&p| !self.is_free(p)) {
                Some(blocked) => end = blocked,
                None => return Some(start),
            }
        }
        None
    }

    /// Allocate a run fitting the exact counts, or `None` if nothing
    /// long enough is free.
    pub fn alloc(&mut self, vertex_count: u32, index_count: u32) -> Option<SlabAlloc> {
        let pages = self.cfg.pages_for(vertex_count, index_count);
        if pages > self.cfg.max_pages_per_chunk {
            self.pressure.failed += 1;
            return None;
        }
        let page = if pages == 1 {
            // Hint list first; entries can be stale, so verify.
            loop {
                match self.singles.pop() {
                    Some(p) if self.is_free(p) => break Some(p),
                    Some(_) => continue,
                    None => break self.find_run_low(1, self.low_cursor).or_else(|| {
                        // The cursor only moves forward, so a wrap is
                        // how freed low pages are found again.
                        self.find_run_low(1, 0)
                    }),
                }
            }
        } else {
            self.find_run_high(pages)
        };
        let Some(page) = page else {
            self.pressure.failed += 1;
            if self.cfg.total_pages - self.used_pages >= pages {
                self.pressure.fragmented += 1;
            }
            return None;
        };
        self.set_range(page, pages, true);
        self.used_pages += pages;
        self.live += 1;
        self.histogram[(pages - 1) as usize] += 1;
        let run = (pages - 1) as usize;
        self.peak_histogram[run] = self.peak_histogram[run].max(self.histogram[run]);
        self.peak_used_pages = self.peak_used_pages.max(self.used_pages);
        self.peak_live = self.peak_live.max(self.live);
        if pages == 1 {
            self.low_cursor = page + 1;
        }
        Some(SlabAlloc {
            page,
            pages,
            base_vertex: page * self.cfg.page_verts,
            first_index: page * self.cfg.page_verts * self.cfg.index_factor,
        })
    }

    /// Could `alloc` succeed right now? A HINT, not a reservation:
    /// another chunk can take the run first, and the caller simply
    /// defers again. Used to decide which deferred chunks are worth
    /// re-running density for.
    pub fn would_fit(&self, vertex_count: u32, index_count: u32) -> bool {
        let pages = self.cfg.pages_for(vertex_count, index_count);
        if pages > self.cfg.max_pages_per_chunk {
            return false;
        }
        if pages == 1 {
            return self.used_pages < self.cfg.total_pages;
        }
        self.find_run_high(pages).is_some()
    }

    pub fn used_pages(&self) -> u32 {
        self.used_pages
    }

    pub fn free_pages(&self) -> u32 {
        self.cfg.total_pages - self.used_pages
    }

    /// Live allocations by run length (index 0 is one page). What a
    /// fixed partition would have had to predict, reported so nobody has
    /// to predict it again.
    pub fn run_histogram(&self) -> &[u32] {
        &self.histogram
    }

    /// The same at its high-water mark over the session. This, not a
    /// reading taken standing still, is what a budget should be judged
    /// against.
    pub fn peak_run_histogram(&self) -> &[u32] {
        &self.peak_histogram
    }

    pub fn peak_used_pages(&self) -> u32 {
        self.peak_used_pages
    }

    /// Live allocations.
    pub fn live_chunks(&self) -> u32 {
        self.live
    }

    /// Longest run currently free. Falling far below
    /// [`self.cfg.max_pages_per_chunk`] while plenty of pages are free is
    /// fragmentation, and it is the only failure mode pages introduce
    /// that classes did not have.
    pub fn longest_free_run(&self) -> u32 {
        let (mut best, mut run) = (0, 0);
        for page in 0..self.cfg.total_pages {
            if self.is_free(page) {
                run += 1;
                best = best.max(run);
            } else {
                run = 0;
            }
        }
        best
    }

    pub fn pressure(&self) -> SlabPressure {
        self.pressure
    }

    pub fn free(&mut self, alloc: SlabAlloc) {
        debug_assert!(alloc.page + alloc.pages <= self.cfg.total_pages);
        debug_assert!(
            (alloc.page..alloc.page + alloc.pages).all(|p| !self.is_free(p)),
            "double free"
        );
        self.set_range(alloc.page, alloc.pages, false);
        self.used_pages -= alloc.pages;
        self.live -= 1;
        self.histogram[(alloc.pages - 1) as usize] -= 1;
        if alloc.pages == 1 {
            self.singles.push(alloc.page);
            self.low_cursor = self.low_cursor.min(alloc.page);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slab() -> SlabAllocator {
        SlabAllocator::new(SlabConfig::default())
    }

    fn cfg() -> SlabConfig {
        SlabConfig::default()
    }

    #[test]
    fn a_run_is_exactly_as_long_as_the_counts_need() {
        let (c, mut slab) = (cfg(), slab());
        assert_eq!(slab.alloc(100, 600).unwrap().pages, 1);
        assert_eq!(
            slab.alloc(c.page_verts, c.page_verts * c.index_factor)
                .unwrap()
                .pages,
            1
        );
        assert_eq!(slab.alloc(c.page_verts + 1, 600).unwrap().pages, 2);
        // Index count alone can lengthen the run.
        assert_eq!(
            slab.alloc(100, c.page_verts * c.index_factor + 1).unwrap().pages,
            2
        );
        // Too big entirely.
        assert!(slab
            .alloc((c.max_pages_per_chunk + 1) * c.page_verts, 1)
            .is_none());
    }

    #[test]
    fn ranges_are_disjoint_and_derived_from_the_page() {
        let (c, mut slab) = (cfg(), slab());
        let a = slab.alloc(1_000, 6_000).unwrap();
        let b = slab.alloc(1_000, 6_000).unwrap();
        assert_ne!(a.base_vertex, b.base_vertex);
        assert_eq!(a.base_vertex, a.page * c.page_verts);
        assert_eq!(a.first_index, a.page * c.page_verts * c.index_factor);
        assert!(a.base_vertex.abs_diff(b.base_vertex) >= c.page_verts);
    }

    /// The size table is the host's, not the allocator's: a game with
    /// denser chunks says so instead of editing this crate.
    #[test]
    fn the_host_chooses_the_budget() {
        let c = SlabConfig {
            page_verts: 512,
            index_factor: 4,
            total_pages: 64,
            max_pages_per_chunk: 8,
        };
        let mut slab = SlabAllocator::new(c);
        assert_eq!(c.total_vertices(), 64 * 512);
        assert_eq!(c.total_indices(), 64 * 512 * 4);
        assert_eq!(slab.alloc(513, 1).unwrap().pages, 2);
        assert!(slab.alloc(9 * 512, 1).is_none(), "beyond this host's maximum");
        for _ in 0..62 {
            assert!(slab.alloc(1, 1).is_some());
        }
        assert!(slab.alloc(1, 1).is_none(), "64 pages, all spoken for");
    }

    /// The shape the four-class partition could not serve: a world whose
    /// chunks are nearly all one page. It used to peg the smallest class
    /// at 100% and spill into slots three times too big while later
    /// classes sat idle; now such a world can use the whole buffer.
    #[test]
    fn a_world_of_small_chunks_can_use_the_whole_buffer() {
        let (c, mut slab) = (cfg(), slab());
        for _ in 0..c.total_pages {
            assert!(slab.alloc(10, 60).is_some());
        }
        assert_eq!(slab.free_pages(), 0);
        assert!(slab.alloc(10, 60).is_none());
        assert_eq!(slab.pressure().failed, 1);
    }

    /// The opposite shape — multi-page chunks — has to keep working
    /// alongside it, since two worlds are loaded at once and their
    /// demands add. The counts are illustrative of the two SHAPES, not a
    /// budget: what a world costs depends on where the camera is.
    #[test]
    fn opposite_shapes_coexist_because_neither_owns_a_partition() {
        let (c, mut slab) = (cfg(), slab());
        for (count, pages) in [(1_135u32, 1u32), (971, 3), (421, 8)] {
            for _ in 0..count {
                let a = slab
                    .alloc(pages * c.page_verts, pages * c.page_verts * c.index_factor)
                    .expect("a multi-page world must fit");
                assert_eq!(a.pages, pages);
            }
        }
        for _ in 0..1_892 {
            assert!(slab.alloc(10, 60).is_some(), "single-page world alongside it");
        }
        assert_eq!(slab.pressure().failed, 0);
        assert_eq!(slab.run_histogram()[0], 1_135 + 1_892);
        assert_eq!(slab.run_histogram()[2], 971);
        assert_eq!(slab.run_histogram()[7], 421);
    }

    /// Single pages come from the bottom and long runs from the top, so
    /// a churn of small allocations cannot perforate the space long runs
    /// need. Without the split placement this leaves no long run at all.
    #[test]
    fn small_churn_does_not_starve_long_runs() {
        let (c, mut slab) = (cfg(), slab());
        let mut live = Vec::new();
        for i in 0..4_000 {
            let a = slab.alloc(10, 60).unwrap();
            if i % 2 == 0 {
                live.push(a);
            } else {
                slab.free(a);
            }
        }
        assert!(
            slab.alloc(c.max_pages_per_chunk * c.page_verts, 1).is_some(),
            "a maximum-size chunk must still find a run"
        );
        assert_eq!(slab.pressure().fragmented, 0);
    }

    #[test]
    fn freeing_recycles_and_the_histogram_tracks_the_live_set() {
        let (c, mut slab) = (cfg(), slab());
        let a = slab.alloc(10, 60).unwrap();
        let b = slab.alloc(3 * c.page_verts, 60).unwrap();
        assert_eq!(slab.live_chunks(), 2);
        assert_eq!(slab.run_histogram()[0], 1);
        assert_eq!(slab.run_histogram()[2], 1);
        slab.free(a);
        slab.free(b);
        assert_eq!(slab.live_chunks(), 0);
        assert_eq!(slab.used_pages(), 0);
        assert!(slab.run_histogram().iter().all(|&n| n == 0));
        assert_eq!(slab.alloc(10, 60).unwrap().page, a.page, "recycled");
    }

    /// Standing still is one sample of a moving process. Peaks are what
    /// survive the camera moving somewhere denser and back.
    #[test]
    fn peaks_outlive_the_moment_they_were_measured() {
        let (c, mut slab) = (cfg(), slab());
        let dense: Vec<_> = (0..100)
            .map(|_| slab.alloc(8 * c.page_verts, 60).unwrap())
            .collect();
        assert_eq!(slab.peak_used_pages(), 800);
        assert_eq!(slab.peak_run_histogram()[7], 100);
        for a in dense {
            slab.free(a);
        }
        // The live set is empty; the evidence is not.
        assert_eq!(slab.used_pages(), 0);
        assert_eq!(slab.peak_used_pages(), 800);
        assert_eq!(slab.peak_run_histogram()[7], 100);
    }

    /// Capacity is measured, not declared: the same buffer holds far
    /// more single-page chunks than multi-page ones, and admission
    /// control has to be told which world it is looking at.
    #[test]
    fn capacity_follows_what_chunks_actually_cost() {
        let (c, mut slab) = (cfg(), slab());
        assert_eq!(
            slab.capacity_chunks(),
            c.total_pages as usize,
            "an empty slab assumes one page per chunk"
        );
        for _ in 0..500 {
            slab.alloc(10, 60).unwrap();
        }
        assert_eq!(slab.capacity_chunks(), c.total_pages as usize);
        for _ in 0..500 {
            slab.alloc(8 * c.page_verts, 60).unwrap();
        }
        // Mean run is now 4.5 pages, so far fewer chunks fit.
        assert!(slab.capacity_chunks() < c.total_pages as usize / 4);
    }

    #[test]
    fn fuzz_alloc_free_never_overlaps() {
        use voxel_core::seed::Rng;
        let (c, mut slab) = (cfg(), slab());
        let mut live: Vec<SlabAlloc> = Vec::new();
        let mut rng = Rng::new(0xF422);
        for _ in 0..20_000 {
            if rng.next_f32() < 0.55 || live.is_empty() {
                let verts = 1 + rng.next_range(c.max_pages_per_chunk * c.page_verts);
                let indices =
                    1 + rng.next_range(c.max_pages_per_chunk * c.page_verts * c.index_factor);
                if let Some(a) = slab.alloc(verts, indices) {
                    // The granted run must hold the request…
                    assert!(a.pages * c.page_verts >= verts);
                    assert!(a.pages * c.page_verts * c.index_factor >= indices);
                    // …and never overlap any live allocation.
                    for b in &live {
                        assert!(
                            a.page + a.pages <= b.page || b.page + b.pages <= a.page,
                            "page overlap"
                        );
                    }
                    live.push(a);
                }
            } else {
                let i = rng.next_range(live.len() as u32) as usize;
                slab.free(live.swap_remove(i));
            }
        }
        let held: u32 = live.iter().map(|a| a.pages).sum();
        assert_eq!(slab.used_pages(), held, "accounting must match the live set");
    }
}
