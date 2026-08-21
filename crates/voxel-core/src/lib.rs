//! Core voxel types shared by every other crate: chunk keys, voxel formats,
//! quantization, morton indexing, and deterministic seeding.
//!
//! This crate deliberately has no Bevy dependency.

pub mod branch;
pub mod csg;
pub mod interval;
pub mod key;
pub mod layout;
pub mod morton;
pub mod opgen;
pub mod patch;
pub mod pos;
pub mod seed;
pub mod voxel;
pub mod worldop;

pub use key::{ChunkKey, WorldId};
pub use pos::GlobalPos;
pub use voxel::Voxel;

/// Cells per chunk axis, at every LOD level.
pub const CHUNK_CELLS: u32 = 32;

/// Voxel edge length in meters at LOD 0.
pub const BASE_VOXEL_M: f64 = 0.1;

/// SDF narrow band half-width, in units of the voxel size at the chunk's LOD.
/// Stored SDF values are clamped to `±SDF_BAND`.
pub const SDF_BAND: f32 = 4.0;

/// Wall time and call count charged to one named stage, summed across
/// every thread that ran it.
///
/// Attribution for a SETTLE, which the per-frame [`timed`] cannot give:
/// the work is spread over dozens of worker threads and hundreds of
/// frames, so what matters is the total charged to each stage, not
/// whether any one frame overran. A sampling profiler answers "which
/// symbol" and this answers "which stage", which is the question when
/// deciding what to cut.
///
/// Always on: two relaxed atomics on paths that already do far more, and
/// a number nobody reads costs nothing. Read them with
/// `voxctl status` -> `stages`.
#[derive(Default)]
pub struct Stage {
    nanos: core::sync::atomic::AtomicU64,
    calls: core::sync::atomic::AtomicU64,
}

impl Stage {
    pub const fn new() -> Self {
        Self {
            nanos: core::sync::atomic::AtomicU64::new(0),
            calls: core::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Run `f`, charging its wall time here.
    pub fn time<T>(&self, f: impl FnOnce() -> T) -> T {
        use core::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let out = f();
        self.nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
        self.calls.fetch_add(1, Relaxed);
        out
    }

    /// Charge `n` to the call count without timing anything — for the
    /// sizes that explain a stage's cost (ops walked, cells built).
    pub fn count(&self, n: u64) {
        self.calls
            .fetch_add(n, core::sync::atomic::Ordering::Relaxed);
    }

    /// (milliseconds, calls) so far.
    pub fn read(&self) -> (f64, u64) {
        use core::sync::atomic::Ordering::Relaxed;
        (
            self.nanos.load(Relaxed) as f64 / 1.0e6,
            self.calls.load(Relaxed),
        )
    }

    pub fn reset(&self) {
        use core::sync::atomic::Ordering::Relaxed;
        self.nanos.store(0, Relaxed);
        self.calls.store(0, Relaxed);
    }
}

/// Is per-system cost attribution switched on? `VOXEL_COST=1`.
///
/// Read once. See [`timed`] for why it is off by default.
pub fn cost_logging() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VOXEL_COST").is_some())
}

/// Warn when a block of work overruns a frame budget, naming it —
/// under `VOXEL_COST=1`.
///
/// Attribution for a stutter has to be per-SYSTEM: a frame-time graph
/// says a frame was slow and nothing about which of forty systems did it,
/// and a chrome trace needs a dependency this workspace cannot fetch.
///
/// OPT-IN, because a budget is a guess and a wrong guess is worse than no
/// instrument at all: the one call site's 4 ms is under what its system
/// costs in an ordinary frame, so it warned several times a second
/// forever and buried the log it was supposed to help read. A number that
/// fires every frame reports nothing. Left in rather than deleted because
/// what it does when you ask for it is still the only per-system
/// attribution this workspace has.
#[macro_export]
macro_rules! timed {
    ($name:literal, $budget_ms:expr, $body:expr) => {{
        let __started = $crate::cost_logging().then(std::time::Instant::now);
        let __out = $body;
        if let Some(__started) = __started {
            let __ms = __started.elapsed().as_secs_f32() * 1000.0;
            if __ms > $budget_ms {
                bevy::log::warn!("COST {} {:.1}ms", $name, __ms);
            }
        }
        __out
    }};
}

/// Serde `default` stubs: a name, a type and a value.
///
/// `#[serde(default = "...")]` names a FUNCTION, so every authored default
/// needs one to exist. Fifty-three of them were three lines each to say a
/// single number, and their names are the documentation — `d_scatter_tile`
/// and `d_floor_step` are both `0.5` and must stay two names, so this
/// collapses the syntax and nothing else.
#[macro_export]
macro_rules! defaults {
    ($($vis:vis $name:ident: $ty:ty = $value:expr;)*) => {
        $($vis fn $name() -> $ty { $value })*
    };
}
