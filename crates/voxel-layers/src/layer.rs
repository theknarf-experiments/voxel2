//! Chunk geometry and layer identity: the coordinate system every
//! layer shares.

use glam::IVec3;

/// Integer axis-aligned box in world meters, `max` exclusive.
/// `i32::MIN`/`i32::MAX` bounds mean "unbounded along that axis".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IAabb {
    pub min: IVec3,
    pub max: IVec3,
}

impl IAabb {
    pub fn new(min: IVec3, max: IVec3) -> Self {
        Self { min, max }
    }

    /// Grow by `pad` meters on every side (saturating, so unbounded axes
    /// stay unbounded).
    pub fn inflate(self, pad: IVec3) -> Self {
        Self {
            min: IVec3::new(
                self.min.x.saturating_sub(pad.x),
                self.min.y.saturating_sub(pad.y),
                self.min.z.saturating_sub(pad.z),
            ),
            max: IVec3::new(
                self.max.x.saturating_add(pad.x),
                self.max.y.saturating_add(pad.y),
                self.max.z.saturating_add(pad.z),
            ),
        }
    }

    pub fn contains(self, other: IAabb) -> bool {
        self.min.cmple(other.min).all() && self.max.cmpge(other.max).all()
    }

    /// Half-open interval overlap on every axis.
    pub fn intersects(self, other: IAabb) -> bool {
        self.min.cmplt(other.max).all() && other.min.cmplt(self.max).all()
    }
}

/// Stable instance key: hash of the instance name. Also seeds the
/// instance's RNG streams, so renaming an instance reshuffles its
/// randomness — treat like a save-format change.
pub type LayerKey = u64;

pub fn layer_key(name: &str) -> LayerKey {
    let mut h = voxel_core::seed::splitmix64(0xC0FFEE);
    for b in name.bytes() {
        h = voxel_core::seed::splitmix64(h ^ b as u64);
    }
    h
}

/// World-space bounds of a layer chunk, honoring collapsed axes.
pub fn chunk_bounds(extent: IVec3, coord: IVec3) -> IAabb {
    let axis = |e: i32, c: i32| -> (i32, i32) {
        if e == 0 {
            (i32::MIN, i32::MAX)
        } else {
            // Chunk coords stay small enough that this cannot overflow in
            // practice (extent ≥ 1 m, world spans ± 2^31 m).
            (c * e, (c + 1) * e)
        }
    };
    let (min_x, max_x) = axis(extent.x, coord.x);
    let (min_y, max_y) = axis(extent.y, coord.y);
    let (min_z, max_z) = axis(extent.z, coord.z);
    IAabb::new(
        IVec3::new(min_x, min_y, min_z),
        IVec3::new(max_x, max_y, max_z),
    )
}

/// Range of chunk coordinates (inclusive) covering `bounds`.
pub fn chunk_range(extent: IVec3, bounds: IAabb) -> (IVec3, IVec3) {
    let axis = |e: i32, min: i32, max: i32| -> (i32, i32) {
        if e == 0 {
            (0, 0)
        } else {
            (min.div_euclid(e), (max - 1).div_euclid(e))
        }
    };
    let (x0, x1) = axis(extent.x, bounds.min.x, bounds.max.x);
    let (y0, y1) = axis(extent.y, bounds.min.y, bounds.max.y);
    let (z0, z1) = axis(extent.z, bounds.min.z, bounds.max.z);
    (IVec3::new(x0, y0, z0), IVec3::new(x1, y1, z1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_bounds_and_range_roundtrip() {
        let extent = IVec3::new(256, 0, 256);
        let b = chunk_bounds(extent, IVec3::new(-1, 0, 2));
        assert_eq!(b.min, IVec3::new(-256, i32::MIN, 512));
        assert_eq!(b.max, IVec3::new(0, i32::MAX, 768));

        let (lo, hi) = chunk_range(
            extent,
            IAabb::new(IVec3::new(-10, 5, 0), IVec3::new(10, 6, 257)),
        );
        assert_eq!(lo, IVec3::new(-1, 0, 0));
        assert_eq!(hi, IVec3::new(0, 0, 1));
    }

    #[test]
    fn inflate_saturates_on_collapsed_axes() {
        let b = IAabb::new(IVec3::new(0, i32::MIN, 0), IVec3::new(10, i32::MAX, 10));
        let inflated = b.inflate(IVec3::splat(100));
        assert_eq!(inflated.min.y, i32::MIN);
        assert_eq!(inflated.max.y, i32::MAX);
        assert_eq!(inflated.min.x, -100);
    }
}
