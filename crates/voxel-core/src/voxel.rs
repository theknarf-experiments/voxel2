//! The 4-byte voxel format shared bit-exactly between CPU and GPU:
//! `f16 sdf | u8 material | u8 flags` packed into a `u32`.
//!
//! The SDF value is stored in units of the chunk's voxel size and clamped to
//! the narrow band `±SDF_BAND`. Material 0 means air. Rust's `f16` is still
//! unstable, so the half-float conversion is done manually here; WGSL reads
//! the same bits natively.

use bytemuck::{Pod, Zeroable};

use crate::SDF_BAND;

/// One voxel, packed exactly as the GPU sees it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Pod, Zeroable)]
#[repr(transparent)]
pub struct Voxel(pub u32);

impl Voxel {
    pub const AIR: Voxel = Voxel(0x7C00); // sdf = +inf(f16), material 0, flags 0

    /// Pack an SDF value (in voxel-size units), material, and flags.
    pub fn new(sdf_voxels: f32, material: u8, flags: u8) -> Self {
        let clamped = sdf_voxels.clamp(-SDF_BAND, SDF_BAND);
        Voxel(f32_to_f16_bits(clamped) as u32 | (material as u32) << 16 | (flags as u32) << 24)
    }

    /// SDF in voxel-size units.
    pub fn sdf(self) -> f32 {
        f16_bits_to_f32((self.0 & 0xFFFF) as u16)
    }

    pub fn material(self) -> u8 {
        (self.0 >> 16) as u8
    }

    pub fn flags(self) -> u8 {
        (self.0 >> 24) as u8
    }

    pub fn is_solid(self) -> bool {
        self.sdf() < 0.0
    }
}

/// f32 → IEEE 754 half, round-to-nearest-even, with overflow → ±inf.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf / NaN
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    // Rebias 127 → 15.
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7C00; // overflow → inf
    }
    if unbiased >= -14 {
        // Normal half. Round mantissa 23 → 10 bits, nearest-even.
        let mut m = mant >> 13;
        let rest = mant & 0x1FFF;
        if rest > 0x1000 || (rest == 0x1000 && (m & 1) == 1) {
            m += 1;
        }
        let mut e = (unbiased + 15) as u32;
        if m == 0x400 {
            // Mantissa rounding overflowed into the exponent.
            m = 0;
            e += 1;
            if e >= 31 {
                return sign | 0x7C00;
            }
        }
        return sign | ((e as u16) << 10) | m as u16;
    }
    if unbiased >= -24 {
        // Subnormal half: value = m × 2⁻²⁴ with m = full × 2^(unbiased+1),
        // i.e. a right shift by s = -(unbiased + 1) ∈ [14, 23].
        let full = mant | 0x0080_0000; // mantissa with implicit leading 1
        let s = (-(unbiased + 1)) as u32;
        let mut m = full >> s;
        let rest = full & ((1 << s) - 1);
        let half = 1u32 << (s - 1);
        if rest > half || (rest == half && (m & 1) == 1) {
            m += 1;
        }
        return sign | m as u16; // may carry into exponent bit 10 — that is correct
    }
    sign // underflow → ±0
}

/// IEEE 754 half → f32.
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: value = mant × 2⁻²⁴. Normalize the leading 1 up to
            // bit 10, drop it, and set the f32 exponent accordingly.
            let shift = mant.leading_zeros() - 21; // 10 - highest_bit(mant)
            let m = (mant << shift) & 0x3FF; // leading 1 lands on bit 10 and is masked off
            let e = 113 - shift; // biased f32 exp = highest_bit(mant) + 103
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_exact_values() {
        // Values exactly representable in f16 must roundtrip bit-perfectly.
        for x in [
            0.0f32, 1.0, -1.0, 0.5, 2.0, 4.0, -4.0, 0.25, 1.5, -3.75, 100.0,
        ] {
            assert_eq!(f16_bits_to_f32(f32_to_f16_bits(x)), x, "value {x}");
        }
    }

    #[test]
    fn f16_roundtrip_error_bound_in_band() {
        // Across the narrow band, relative error must be within f16 epsilon.
        let mut i = 0;
        while i <= 8000 {
            let x = -4.0 + i as f32 * 0.001;
            let r = f16_bits_to_f32(f32_to_f16_bits(x));
            let tol = (x.abs() * 0.001).max(1e-4); // f16 eps ≈ 9.8e-4 relative
            assert!((r - x).abs() <= tol, "x={x} r={r}");
            i += 1;
        }
    }

    #[test]
    fn f16_specials() {
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xFC00);
        assert_eq!(f32_to_f16_bits(1.0e6), 0x7C00); // overflow → inf
        assert_eq!(f16_bits_to_f32(0x7C00), f32::INFINITY);
        assert!(f16_bits_to_f32(0x3C00) == 1.0);
        // Subnormal halves roundtrip.
        let tiny = f16_bits_to_f32(0x0001);
        assert!(tiny > 0.0 && tiny < 1e-7);
        assert_eq!(f32_to_f16_bits(tiny), 0x0001);
    }

    #[test]
    fn f16_all_bit_patterns_roundtrip() {
        // decode → encode must be the identity for every finite/inf half.
        for bits in 0..=0xFFFFu16 {
            let f = f16_bits_to_f32(bits);
            if f.is_nan() {
                assert!(f32_to_f16_bits(f) & 0x7C00 == 0x7C00); // stays NaN/inf class
                continue;
            }
            assert_eq!(f32_to_f16_bits(f), bits, "bits {bits:#06x} → {f}");
        }
    }

    #[test]
    fn voxel_pack_unpack() {
        let v = Voxel::new(-1.25, 7, 3);
        assert_eq!(v.sdf(), -1.25);
        assert_eq!(v.material(), 7);
        assert_eq!(v.flags(), 3);
        assert!(v.is_solid());

        // Out-of-band values clamp.
        assert_eq!(Voxel::new(100.0, 1, 0).sdf(), SDF_BAND);
        assert_eq!(Voxel::new(-100.0, 1, 0).sdf(), -SDF_BAND);

        assert!(!Voxel::AIR.is_solid());
        assert_eq!(Voxel::AIR.material(), 0);
    }
}
