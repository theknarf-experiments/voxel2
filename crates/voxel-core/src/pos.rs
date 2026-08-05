//! World-space positions for an effectively infinite world: integer chunk
//! coordinate plus a chunk-local f32 offset. Used for all persistent state;
//! f64 vectors appear only transiently (camera math, LOD distances).

use glam::{DVec3, IVec3, Vec3};
use serde::{Deserialize, Serialize};

use crate::{BASE_VOXEL_M, CHUNK_CELLS};

/// Chunk edge in meters at LOD 0.
const CHUNK_EDGE_M: f64 = BASE_VOXEL_M * CHUNK_CELLS as f64;

/// A precise world position: LOD-0 chunk coordinate + local offset in meters.
///
/// `local` is kept in `[0, 32)` per axis by [`GlobalPos::normalize`]. With i32
/// chunk coords this spans ±6.8e10 m — thousands of planet radii — while
/// keeping full f32 precision locally.
#[derive(Clone, Copy, PartialEq, Debug, Default, Serialize, Deserialize)]
pub struct GlobalPos {
    pub chunk: IVec3,
    pub local: Vec3,
}

impl GlobalPos {
    pub fn new(chunk: IVec3, local: Vec3) -> Self {
        Self { chunk, local }.normalize()
    }

    /// Rewraps `local` into `[0, 32)`, carrying overflow into `chunk`.
    pub fn normalize(mut self) -> Self {
        let carry = (self.local.as_dvec3() / CHUNK_EDGE_M).floor();
        let carry_i = IVec3::new(carry.x as i32, carry.y as i32, carry.z as i32);
        self.chunk += carry_i;
        self.local -= (carry * CHUNK_EDGE_M).as_vec3();
        self
    }

    /// Lossy conversion to a single f64 vector (fine for camera/LOD math).
    pub fn to_dvec3(self) -> DVec3 {
        self.chunk.as_dvec3() * CHUNK_EDGE_M + self.local.as_dvec3()
    }

    /// Construct from an f64 world position.
    pub fn from_dvec3(p: DVec3) -> Self {
        let chunk = (p / CHUNK_EDGE_M).floor();
        let chunk_i = IVec3::new(chunk.x as i32, chunk.y as i32, chunk.z as i32);
        Self {
            chunk: chunk_i,
            local: (p - chunk * CHUNK_EDGE_M).as_vec3(),
        }
    }

    /// Position relative to a chunk origin, exact in integers then f32.
    /// This is the camera-relative transform used for rendering.
    pub fn relative_to_chunk(self, origin_chunk: IVec3) -> Vec3 {
        ((self.chunk - origin_chunk).as_vec3()) * CHUNK_EDGE_M as f32 + self.local
    }

    /// Offset by a small (f32-sized) displacement.
    pub fn offset(self, delta: Vec3) -> Self {
        Self {
            chunk: self.chunk,
            local: self.local + delta,
        }
        .normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_wraps_local() {
        let edge = (crate::BASE_VOXEL_M * crate::CHUNK_CELLS as f64) as f32;
        let p = GlobalPos::new(IVec3::ZERO, Vec3::new(33.0, -1.0, 64.5));
        let expect = |v: f32| (v / edge).floor() as i32;
        assert_eq!(p.chunk, IVec3::new(expect(33.0), expect(-1.0), expect(64.5)));
        assert!(p.local.x >= 0.0 && p.local.x < edge);
        assert!(p.local.y >= 0.0 && p.local.y < edge);
        assert!((p.to_dvec3() - DVec3::new(33.0, -1.0, 64.5)).length() < 1e-4);
    }

    #[test]
    fn dvec3_roundtrip_far_from_origin() {
        // 1000 km from origin — f64 roundtrip must stay sub-millimeter.
        let p = DVec3::new(1.0e6, -2.5e5, 7.77e5) + DVec3::new(0.123, 0.456, 0.789);
        let g = GlobalPos::from_dvec3(p);
        assert!((g.to_dvec3() - p).length() < 1e-3);
        assert!(
            g.local.min_element() >= 0.0
                && (g.local.max_element() as f64) < crate::BASE_VOXEL_M * crate::CHUNK_CELLS as f64
        );
    }

    #[test]
    fn relative_to_chunk_is_precise_far_out() {
        // Two points ~1 m apart, 1e9 m from origin: their camera-relative
        // difference must be exact even though f32 world coords could not
        // represent it.
        let chunk = IVec3::new(31_250_000, 0, 0); // 1e9 m
        let a = GlobalPos::new(chunk, Vec3::new(1.0, 2.0, 3.0));
        let b = GlobalPos::new(chunk, Vec3::new(2.0, 2.0, 3.0));
        let ra = a.relative_to_chunk(chunk);
        let rb = b.relative_to_chunk(chunk);
        assert_eq!(rb - ra, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn offset_carries_across_chunks() {
        let edge = (crate::BASE_VOXEL_M * crate::CHUNK_CELLS as f64) as f32;
        let p = GlobalPos::new(IVec3::new(5, 0, 0), Vec3::new(edge - 0.5, 0.0, 0.0));
        let q = p.offset(Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(q.chunk, IVec3::new(6, 0, 0));
        assert!((q.local.x - 0.5).abs() < 1e-5);
    }
}
