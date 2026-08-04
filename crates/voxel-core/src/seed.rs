//! Deterministic seeding. Every piece of procedural randomness in the engine
//! derives from `(world_seed, layer_id, chunk_coord)` through these mixers, so
//! generation is reproducible regardless of thread count or generation order.

use glam::IVec3;

/// SplitMix64 mixer — the finalizer alone is a good hash, and the sequence a
/// good small PRNG.
#[inline]
pub fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Seed for one chunk of one layer.
pub fn chunk_seed(world_seed: u64, layer_id: u64, coord: IVec3) -> u64 {
    let c = (coord.x as u32 as u64) | ((coord.y as u32 as u64) << 32);
    let mut s = splitmix64(world_seed ^ splitmix64(layer_id));
    s = splitmix64(s ^ c);
    splitmix64(s ^ (coord.z as u32 as u64))
}

/// Minimal deterministic PRNG over a SplitMix64 stream.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Uniform in `[0, n)`.
    #[inline]
    pub fn next_range(&mut self, n: u32) -> u32 {
        ((self.next_u64() >> 32).wrapping_mul(n as u64) >> 32) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_seed_is_deterministic_and_spread() {
        let a = chunk_seed(42, 1, IVec3::new(0, 0, 0));
        assert_eq!(a, chunk_seed(42, 1, IVec3::new(0, 0, 0)));
        // Neighboring coords, other layers, and other worlds all differ.
        assert_ne!(a, chunk_seed(42, 1, IVec3::new(1, 0, 0)));
        assert_ne!(a, chunk_seed(42, 1, IVec3::new(0, 1, 0)));
        assert_ne!(a, chunk_seed(42, 1, IVec3::new(0, 0, 1)));
        assert_ne!(a, chunk_seed(42, 2, IVec3::new(0, 0, 0)));
        assert_ne!(a, chunk_seed(43, 1, IVec3::new(0, 0, 0)));
        // Negative coords are distinct from positive ones.
        assert_ne!(
            chunk_seed(42, 1, IVec3::new(-1, 0, 0)),
            chunk_seed(42, 1, IVec3::new(1, 0, 0))
        );
    }

    #[test]
    fn rng_ranges() {
        let mut rng = Rng::new(123);
        for _ in 0..10_000 {
            let f = rng.next_f32();
            assert!((0.0..1.0).contains(&f));
            let r = rng.next_range(7);
            assert!(r < 7);
        }
    }

    #[test]
    fn rng_streams_from_different_seeds_differ() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let same = (0..64).filter(|_| a.next_u64() == b.next_u64()).count();
        assert_eq!(same, 0);
    }
}
