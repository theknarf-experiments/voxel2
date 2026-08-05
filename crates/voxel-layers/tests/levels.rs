//! Internal layer levels (LayerProcGen): one layer, multiple passes —
//! level 1 chunks read level 0 data from their padded neighborhood.
//! Integrity: chunked results must equal a global brute-force oracle.

use glam::{IVec3, Vec2};
use voxel_layers::{IAabb, Layer, LayerCtx, LayerManager};

const CHUNK_M: i32 = 200;
const POINTS: usize = 5;
const RELAX_R: f32 = 90.0;

/// Two levels: 0 = scattered points, 1 = each point averaged with every
/// level-0 point within RELAX_R (context reaches into neighbor chunks).
struct RelaxLayer;

#[derive(Clone)]
struct RelaxChunk {
    points: Vec<Vec2>,
}

fn scatter(seed_rng: &mut voxel_core::seed::Rng, bounds: IAabb) -> Vec<Vec2> {
    let origin = Vec2::new(bounds.min.x as f32, bounds.min.z as f32);
    (0..POINTS)
        .map(|_| origin + Vec2::new(seed_rng.next_f32(), seed_rng.next_f32()) * CHUNK_M as f32)
        .collect()
}

fn relax(own: &[Vec2], context: &[Vec2]) -> Vec<Vec2> {
    own.iter()
        .map(|p| {
            let mut sum = Vec2::ZERO;
            let mut n = 0.0;
            for q in context {
                if p.distance(*q) < RELAX_R {
                    sum += *q;
                    n += 1.0;
                }
            }
            sum / n
        })
        .collect()
}

impl Layer for RelaxLayer {
    type Chunk = RelaxChunk;
    const NAME: &'static str = "test/relax";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CHUNK_M, 0, CHUNK_M)
    }

    fn levels(&self) -> u32 {
        2
    }

    fn level_padding(&self, _level: u32) -> IVec3 {
        IVec3::new(RELAX_R.ceil() as i32, 0, RELAX_R.ceil() as i32)
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> RelaxChunk {
        let bounds = ctx.chunk_bounds();
        match ctx.level() {
            0 => RelaxChunk {
                points: scatter(&mut ctx.rng(), bounds),
            },
            _ => {
                let pad = self.level_padding(1);
                let view = ctx.get_self(bounds.inflate(pad));
                let context: Vec<Vec2> =
                    view.iter().flat_map(|(_, c)| c.points.clone()).collect();
                let own = ctx.get_self(bounds);
                let own_points: Vec<Vec2> = own
                    .iter()
                    .flat_map(|(_, c)| c.points.clone())
                    .filter(|p| {
                        p.x >= bounds.min.x as f32
                            && p.x < bounds.max.x as f32
                            && p.y >= bounds.min.z as f32
                            && p.y < bounds.max.z as f32
                    })
                    .collect();
                RelaxChunk {
                    points: relax(&own_points, &context),
                }
            }
        }
    }
}

#[test]
fn relaxation_matches_global_oracle_and_is_deterministic() {
    let mgr = LayerManager::new(42);
    let mgr2 = LayerManager::new(42);
    let (mgr, mgr2) = {
        let mut a = mgr;
        let mut b = mgr2;
        a.register(RelaxLayer);
        b.register(RelaxLayer);
        (a, b)
    };

    // Query the top level over a 3x3 chunk area (well inside a wider
    // level-0 support so the oracle has full context).
    let bounds = IAabb::new(IVec3::new(0, 0, 0), IVec3::new(3 * CHUNK_M, 1, 3 * CHUNK_M));
    let view = mgr.get::<RelaxLayer>(bounds);
    let view2 = mgr2.get::<RelaxLayer>(bounds);

    // Determinism across managers.
    let a: Vec<Vec<[f32; 2]>> = view
        .iter()
        .map(|(_, c)| c.points.iter().map(|p| [p.x, p.y]).collect())
        .collect();
    let b: Vec<Vec<[f32; 2]>> = view2
        .iter()
        .map(|(_, c)| c.points.iter().map(|p| [p.x, p.y]).collect())
        .collect();
    assert_eq!(a, b);

    // Integrity: brute-force oracle over a super-region of level-0 data.
    let oracle_mgr = LayerManager::new(42);
    let oracle_mgr = {
        let mut m = oracle_mgr;
        m.register(RelaxLayer);
        m
    };
    let super_bounds = bounds.inflate(IVec3::new(2 * CHUNK_M, 0, 2 * CHUNK_M));
    let level0 = oracle_mgr.get_at_level::<RelaxLayer>(super_bounds, 0);
    let all_points: Vec<Vec2> = level0.iter().flat_map(|(_, c)| c.points.clone()).collect();
    for (coord, chunk) in view.iter() {
        let cb = IAabb::new(
            coord * IVec3::new(CHUNK_M, 0, CHUNK_M),
            (coord + IVec3::ONE) * IVec3::new(CHUNK_M, 1, CHUNK_M),
        );
        let own: Vec<Vec2> = all_points
            .iter()
            .copied()
            .filter(|p| {
                p.x >= cb.min.x as f32
                    && p.x < cb.max.x as f32
                    && p.y >= cb.min.z as f32
                    && p.y < cb.max.z as f32
            })
            .collect();
        let expect = relax(&own, &all_points);
        assert_eq!(
            chunk.points.len(),
            expect.len(),
            "chunk {coord:?} point count"
        );
        for (got, want) in chunk.points.iter().zip(&expect) {
            assert!(
                got.distance(*want) < 1e-4,
                "chunk {coord:?}: relaxed point {got:?} != oracle {want:?}"
            );
        }
    }
}

/// Reading level-0 data beyond the declared level padding must panic with
/// a diagnostic (the LayerProcGen containment rule).
struct GreedyLayer;

impl Layer for GreedyLayer {
    type Chunk = RelaxChunk;
    const NAME: &'static str = "test/greedy";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(CHUNK_M, 0, CHUNK_M)
    }

    fn levels(&self) -> u32 {
        2
    }

    fn level_padding(&self, _level: u32) -> IVec3 {
        IVec3::new(10, 0, 10)
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> RelaxChunk {
        match ctx.level() {
            0 => RelaxChunk { points: Vec::new() },
            _ => {
                // Requests far more than the declared 10 m padding.
                let too_big = ctx.chunk_bounds().inflate(IVec3::new(500, 0, 500));
                let _ = ctx.get_self(too_big);
                RelaxChunk { points: Vec::new() }
            }
        }
    }
}

#[test]
#[should_panic(expected = "outside its declared level padding")]
fn self_read_outside_level_padding_panics() {
    let mut mgr = LayerManager::new(1);
    mgr.register(GreedyLayer);
    mgr.get::<GreedyLayer>(IAabb::new(IVec3::ZERO, IVec3::new(1, 1, 1)));
}
