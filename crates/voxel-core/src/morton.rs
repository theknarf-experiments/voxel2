//! Morton (Z-order) indexing for voxels within a chunk. 10 bits per axis
//! (up to 1024³), which comfortably covers the 36-sample grid and any
//! future in-chunk addressing.

use glam::UVec3;

/// Spread the low 10 bits of `x` so there are two zero bits between each.
#[inline]
fn split3(x: u32) -> u32 {
    let mut x = x & 0x3FF;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

/// Inverse of [`split3`].
#[inline]
fn compact3(x: u32) -> u32 {
    let mut x = x & 0x09249249;
    x = (x | (x >> 2)) & 0x030C30C3;
    x = (x | (x >> 4)) & 0x0300F00F;
    x = (x | (x >> 8)) & 0x030000FF;
    x = (x | (x >> 16)) & 0x3FF;
    x
}

/// Interleave three 10-bit coordinates into a 30-bit morton code.
#[inline]
pub fn morton_encode(p: UVec3) -> u32 {
    split3(p.x) | (split3(p.y) << 1) | (split3(p.z) << 2)
}

/// Inverse of [`morton_encode`].
#[inline]
pub fn morton_decode(code: u32) -> UVec3 {
    UVec3::new(compact3(code), compact3(code >> 1), compact3(code >> 2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_exhaustive_low() {
        for x in 0..40 {
            for y in 0..40 {
                for z in 0..40 {
                    let p = UVec3::new(x, y, z);
                    assert_eq!(morton_decode(morton_encode(p)), p);
                }
            }
        }
    }

    #[test]
    fn roundtrip_max_coords() {
        for p in [UVec3::splat(1023), UVec3::new(1023, 0, 511), UVec3::new(7, 1000, 3)] {
            assert_eq!(morton_decode(morton_encode(p)), p);
        }
    }

    #[test]
    fn ordering_is_z_order() {
        // The unit cube corners 0..8 must enumerate x-fastest.
        assert_eq!(morton_encode(UVec3::new(1, 0, 0)), 1);
        assert_eq!(morton_encode(UVec3::new(0, 1, 0)), 2);
        assert_eq!(morton_encode(UVec3::new(0, 0, 1)), 4);
        assert_eq!(morton_encode(UVec3::new(1, 1, 1)), 7);
    }
}
