//! Chunk geometry and layer identity: the coordinate system every
//! layer shares.

use glam::{DVec3, IVec3};

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
///
/// Rounded OUTWARD, because a grid spacing need not be a whole number of
/// meters: a voxel LOD level's chunk edge is `3.2 · 2^level`, and a layer
/// that could only be spaced integrally could never align with one. The
/// box a fractional grid reports is therefore a conservative superset of
/// the cell — one meter at most, against paddings of tens — so an ensure
/// covers a little extra and a containment check is a little permissive.
/// Both err toward more data than declared, never less.
pub fn chunk_bounds(extent: DVec3, coord: IVec3) -> IAabb {
    let axis = |e: f64, c: i32| -> (i32, i32) {
        if e <= 0.0 {
            (i32::MIN, i32::MAX)
        } else {
            // Chunk coords stay small enough that this cannot overflow in
            // practice (extent ≥ 1 m, world spans ± 2^31 m).
            ((c as f64 * e).floor() as i32, ((c + 1) as f64 * e).ceil() as i32)
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

/// Range of chunk coordinates (inclusive) whose cell overlaps `bounds`.
pub fn chunk_range(extent: DVec3, bounds: IAabb) -> (IVec3, IVec3) {
    let axis = |e: f64, min: i32, max: i32| -> (i32, i32) {
        if e <= 0.0 {
            (0, 0)
        } else {
            (
                (min as f64 / e).floor() as i32,
                (max as f64 / e).ceil() as i32 - 1,
            )
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
        let extent = DVec3::new(256.0, 0.0, 256.0);
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

    /// A voxel LOD level's chunk edge is `3.2 · 2^level` — never a whole
    /// number of meters. A layer must still be able to put exactly one
    /// chunk per cell on that grid, which is the whole reason cells are
    /// spaced by a float.
    #[test]
    fn a_fractional_grid_still_maps_one_cell_per_coordinate() {
        for level in 0..12u32 {
            let edge = 0.1 * 32.0 * (1u64 << level) as f64;
            let extent = DVec3::splat(edge);
            for coord in [-1000, -1, 0, 1, 7, 1000] {
                let c = IVec3::splat(coord);
                let b = chunk_bounds(extent, c);
                // The reported box contains the true cell, and rounding
                // never loses a meter of it.
                assert!((b.min.x as f64) <= coord as f64 * edge);
                assert!((b.max.x as f64) >= (coord + 1) as f64 * edge);
                // Round trip: the cell's own box resolves to the cell,
                // and to no other.
                let (lo, hi) = chunk_range(extent, b);
                assert!(lo.cmple(c).all() && hi.cmpge(c).all(), "level {level} coord {coord}");
                assert!(hi.x - lo.x <= 2, "outward rounding pulled in extra cells");
            }
        }
    }

    /// Every voxel chunk belongs to exactly one cell: a grid that skipped
    /// or doubled one would silently drop or twice-generate terrain.
    #[test]
    fn a_fractional_grid_partitions_its_axis() {
        let edge = 0.1 * 32.0 * 4.0; // level 2: 12.8 m
        let extent = DVec3::splat(edge);
        for m in -2000..2000 {
            let point = IAabb::new(IVec3::splat(m), IVec3::splat(m + 1));
            let (lo, hi) = chunk_range(extent, point);
            let owners: Vec<i32> = (lo.x..=hi.x)
                .filter(|c| {
                    let b = chunk_bounds(extent, IVec3::splat(*c));
                    b.min.x <= m && m < b.max.x
                })
                .collect();
            assert!(!owners.is_empty(), "meter {m} belongs to no cell");
        }
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
