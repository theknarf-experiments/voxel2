//! Integer chunk addressing. Chunk keys are exact — no floating point is ever
//! involved in identifying a chunk.

use glam::{DVec3, IVec3};
use serde::{Deserialize, Serialize};

use crate::{BASE_VOXEL_M, CHUNK_CELLS};

/// Address of a chunk in the LOD octree.
///
/// `level` 0 is the finest LOD (voxel = [`BASE_VOXEL_M`]); each level doubles
/// the voxel size while keeping [`CHUNK_CELLS`]³ cells per chunk. `pos` is the
/// chunk coordinate at that level, i.e. the chunk covers world meters
/// `pos * edge_m() ..= (pos + 1) * edge_m()` per axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct ChunkKey {
    pub level: u8,
    pub pos: IVec3,
}

impl ChunkKey {
    pub const fn new(level: u8, pos: IVec3) -> Self {
        Self { level, pos }
    }

    /// Voxel edge length in meters at this key's LOD.
    pub fn voxel_size_m(&self) -> f64 {
        BASE_VOXEL_M * (1u64 << self.level) as f64
    }

    /// Chunk edge length in meters.
    pub fn edge_m(&self) -> f64 {
        self.voxel_size_m() * CHUNK_CELLS as f64
    }

    /// Minimum (most negative) corner of the chunk in world meters.
    pub fn min_corner_m(&self) -> DVec3 {
        self.pos.as_dvec3() * self.edge_m()
    }

    /// Center of the chunk in world meters.
    pub fn center_m(&self) -> DVec3 {
        self.min_corner_m() + DVec3::splat(self.edge_m() * 0.5)
    }

    /// The chunk one LOD level coarser that contains this chunk.
    pub fn parent(&self) -> ChunkKey {
        // Arithmetic shift right floors toward negative infinity, which is the
        // correct containment rule for negative chunk coordinates.
        ChunkKey::new(self.level + 1, IVec3::new(self.pos.x >> 1, self.pos.y >> 1, self.pos.z >> 1))
    }

    /// The eight children one LOD level finer. Panics at level 0.
    pub fn children(&self) -> [ChunkKey; 8] {
        assert!(self.level > 0, "level-0 chunks have no children");
        let base = self.pos * 2;
        core::array::from_fn(|i| {
            let offset = IVec3::new(i as i32 & 1, (i as i32 >> 1) & 1, (i as i32 >> 2) & 1);
            ChunkKey::new(self.level - 1, base + offset)
        })
    }

    /// Key of the chunk containing the world-space point at the given level.
    pub fn containing(point_m: DVec3, level: u8) -> ChunkKey {
        let edge = BASE_VOXEL_M * (1u64 << level) as f64 * CHUNK_CELLS as f64;
        let p = (point_m / edge).floor();
        ChunkKey::new(level, IVec3::new(p.x as i32, p.y as i32, p.z as i32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_contains_children() {
        for level in 1..=17u8 {
            let key = ChunkKey::new(level, IVec3::new(-3, 7, -1));
            for child in key.children() {
                assert_eq!(child.parent(), key);
                assert_eq!(child.level, level - 1);
            }
        }
    }

    #[test]
    fn parent_of_negative_coords_floors() {
        assert_eq!(ChunkKey::new(0, IVec3::new(-1, -2, 1)).parent().pos, IVec3::new(-1, -1, 0));
        assert_eq!(ChunkKey::new(0, IVec3::new(-2, 3, -3)).parent().pos, IVec3::new(-1, 1, -2));
    }

    #[test]
    fn containing_roundtrip() {
        for level in [0u8, 3, 10, 17] {
            let key = ChunkKey::new(level, IVec3::new(-5, 2, 9));
            // Points strictly inside the chunk map back to the same key.
            let inside = key.min_corner_m() + DVec3::splat(key.edge_m() * 0.25);
            assert_eq!(ChunkKey::containing(inside, level), key);
            assert_eq!(ChunkKey::containing(key.center_m(), level), key);
        }
    }

    #[test]
    fn edge_scales_with_level() {
        let k0 = ChunkKey::new(0, IVec3::ZERO);
        let k17 = ChunkKey::new(17, IVec3::ZERO);
        assert_eq!(k0.edge_m(), 32.0);
        assert_eq!(k17.edge_m(), 32.0 * 131072.0); // ~4194 km
    }
}
