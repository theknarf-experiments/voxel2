//! Concrete world-generation: CPU mirrors of the GPU terrain functions plus
//! (eventually) the LayerProcGen planning layers.
//!
//! `terrain_height` MUST stay bit-compatible with
//! `voxel-render/src/shaders/voxel_terrain_density.wgsl` — vegetation and
//! gameplay place things on the surface the GPU generates.

use glam::Vec2;

/// Mirrors the WGSL `hash2`.
fn hash2(p: glam::IVec2) -> f32 {
    let mut h: u32 = (p.x as u32)
        .wrapping_mul(374_761_393)
        .wrapping_add((p.y as u32).wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0xFF_FFFF) as f32 / 16_777_216.0
}

/// Mirrors the WGSL `value_noise` (quintic smoothstep).
fn value_noise(p: Vec2) -> f32 {
    let i = p.floor();
    let f = p - i;
    let i = glam::IVec2::new(i.x as i32, i.y as i32);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = hash2(i);
    let b = hash2(i + glam::IVec2::new(1, 0));
    let c = hash2(i + glam::IVec2::new(0, 1));
    let d = hash2(i + glam::IVec2::new(1, 1));
    let ab = a + (b - a) * u.x;
    let cd = c + (d - c) * u.x;
    ab + (cd - ab) * u.y
}

fn band_fade(wavelength: f32, voxel_size: f32) -> f32 {
    let lo = 2.0 * voxel_size;
    let hi = 4.0 * voxel_size;
    let t = ((wavelength - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn fbm(p: Vec2, base_scale: f32, octaves: i32, voxel_size: f32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = base_scale;
    for _ in 0..octaves {
        let fade = band_fade(1.0 / freq, voxel_size);
        sum += amp * fade * (value_noise(p * freq) - 0.5);
        amp *= 0.5;
        freq *= 2.0;
    }
    sum
}

/// Terrain height in meters at a world XZ position, evaluated at the given
/// voxel size (pass 1.0 for full detail). Mirrors the WGSL exactly.
pub fn terrain_height(xz: Vec2, voxel_size: f32) -> f32 {
    let continents = fbm(xz, 0.00005, 3, voxel_size) * 800.0;
    let mountains = fbm(xz + Vec2::new(510.0, -770.0), 0.0008, 5, voxel_size) * 420.0;
    let rolling = fbm(xz + Vec2::new(1337.0, 55.0), 0.01, 5, voxel_size) * 36.0;
    let detail = fbm(xz + Vec2::new(37.0, 91.0), 0.06, 4, voxel_size) * 5.0;
    continents + mountains + rolling + detail - 8.0
}

/// Forest density in [0, 1]: slow spatial noise so woods come in coherent
/// patches with clearings.
pub fn forest_density(xz: Vec2) -> f32 {
    let n = fbm(xz + Vec2::new(-4200.0, 8800.0), 0.004, 3, 1.0) + 0.5;
    (n * 1.6 - 0.35).clamp(0.0, 1.0)
}

/// Soft sun shadow at a world position: horizon march over the band-limited
/// heightfield. Mirrors the WGSL bake in voxel_mesh_chunks.wgsl (sun
/// direction and falloff must stay in sync).
pub fn sun_shadow(pos: glam::Vec3) -> f32 {
    let sun = glam::Vec3::new(0.55, 0.5, 0.32).normalize();
    let mut occ = 0.0f32;
    let mut t = 8.0f32;
    for _ in 0..9 {
        let sp = pos + sun * t;
        let dh = terrain_height(Vec2::new(sp.x, sp.z), 8.0) - sp.y;
        occ = occ.max(dh / t);
        t *= 1.8;
    }
    let x = (occ / 0.2).clamp(0.0, 1.0);
    1.0 - x * x * (3.0 - 2.0 * x)
}

/// Approximate surface normal Y (up-ness) via central differences.
pub fn terrain_up(xz: Vec2, voxel_size: f32) -> f32 {
    let e = 2.0;
    let hx = terrain_height(xz + Vec2::new(e, 0.0), voxel_size)
        - terrain_height(xz - Vec2::new(e, 0.0), voxel_size);
    let hz = terrain_height(xz + Vec2::new(0.0, e), voxel_size)
        - terrain_height(xz - Vec2::new(0.0, e), voxel_size);
    let n = glam::Vec3::new(-hx, 2.0 * e, -hz).normalize();
    n.y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_finite() {
        for i in 0..2000 {
            let p = Vec2::new((i * 37) as f32 * 13.7, (i * 91) as f32 * -7.3);
            let h = terrain_height(p, 1.0);
            assert!(h.is_finite());
            assert_eq!(h, terrain_height(p, 1.0));
            // Sum of amplitude bounds: |h| ≤ 0.5·(800+420+36+5) + 8.
            assert!(h.abs() < 640.0, "h={h} at {p}");
        }
    }

    #[test]
    fn has_relief() {
        // Not a constant function: sample variance must be significant.
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for i in 0..500 {
            let p = Vec2::new(i as f32 * 977.0, i as f32 * -613.0);
            let h = terrain_height(p, 1.0);
            min = min.min(h);
            max = max.max(h);
        }
        assert!(max - min > 200.0, "relief only {}", max - min);
    }

    #[test]
    fn coarse_voxels_only_lose_detail() {
        // Band-limited coarse evaluation stays within the fine-detail
        // amplitude envelope of the full signal.
        for i in 0..200 {
            let p = Vec2::new(i as f32 * 311.0, i as f32 * 157.0);
            let fine = terrain_height(p, 1.0);
            let coarse = terrain_height(p, 64.0);
            // Bands below 256 m wavelength carry at most ~±30 m.
            assert!((fine - coarse).abs() < 60.0, "fine {fine} vs coarse {coarse}");
        }
    }

    #[test]
    fn up_vector_sane() {
        for i in 0..200 {
            let p = Vec2::new(i as f32 * 53.0, i as f32 * -29.0);
            let up = terrain_up(p, 1.0);
            assert!((0.0..=1.0).contains(&up));
        }
    }
}
