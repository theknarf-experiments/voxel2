//! Concrete world-generation: the CPU twin of the GPU generator-program
//! interpreter ([`program`]) plus the LayerProcGen planning layers.
//!
//! [`program::eval`] MUST stay bit-compatible with
//! `voxel-render/src/shaders/voxel_world_density.wgsl` — vegetation and
//! gameplay place things on the surface the GPU generates.

pub mod program;
pub mod path;
pub mod flow;
pub mod stack;
pub mod structure;

use glam::Vec2;

/// Mirrors the WGSL `hash2` (level seed mixed in; 0 = identity).
pub(crate) fn hash2(seed: u32, p: glam::IVec2) -> f32 {
    let mut h: u32 = (p.x as u32)
        .wrapping_mul(374_761_393)
        .wrapping_add((p.y as u32).wrapping_mul(668_265_263))
        .wrapping_add(seed.wrapping_mul(2_654_435_769));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0xFF_FFFF) as f32 / 16_777_216.0
}

/// Mirrors the WGSL `value_noise` (quintic smoothstep).
fn value_noise(seed: u32, p: Vec2) -> f32 {
    let i = p.floor();
    let f = p - i;
    let i = glam::IVec2::new(i.x as i32, i.y as i32);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = hash2(seed, i);
    let b = hash2(seed, i + glam::IVec2::new(1, 0));
    let c = hash2(seed, i + glam::IVec2::new(0, 1));
    let d = hash2(seed, i + glam::IVec2::new(1, 1));
    let ab = a + (b - a) * u.x;
    let cd = c + (d - c) * u.x;
    ab + (cd - ab) * u.y
}

/// The generator is unbanded: a pure function of position, so all LODs
/// sample identical values and seams cannot disagree. (Kept as a hook —
/// per-LOD band-limiting must never return without a seam-exactness
/// story.)
pub(crate) fn band_fade(_wavelength: f32, _voxel_size: f32) -> f32 {
    1.0
}

pub(crate) fn fbm(seed: u32, p: Vec2, base_scale: f32, octaves: i32, voxel_size: f32) -> f32 {
    fbm_mode(seed, p, base_scale, octaves, voxel_size, 0)
}

/// FBM with a per-octave shaping mode: 0 plain, 1 ridged (sharp crests),
/// 2 billow (rounded mounds). Mirrors the WGSL exactly.
pub(crate) fn fbm_mode(
    seed: u32,
    p: Vec2,
    base_scale: f32,
    octaves: i32,
    voxel_size: f32,
    mode: u32,
) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = base_scale;
    for _ in 0..octaves {
        let fade = band_fade(1.0 / freq, voxel_size);
        let n = value_noise(seed, p * freq);
        let v = match mode {
            1 => 0.5 - (2.0 * n - 1.0).abs(),
            2 => (2.0 * n - 1.0).abs() - 0.5,
            _ => n - 0.5,
        };
        sum += amp * fade * v;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum
}

/// A world's generator: its op program, seed and sun direction — the
/// whole CPU-side world in one value.
///
/// Everything that samples the world takes one of these, so an app can
/// host several worlds at once (two planets, a planet and an interior,
/// a preview world beside the live one). Clone is cheap: the ops are
/// shared.
#[derive(Clone)]
pub struct Generator {
    ops: std::sync::Arc<Vec<voxel_core::worldop::WorldOp>>,
    seed: u32,
    sun: glam::Vec3,
    has_height: bool,
}

impl Default for Generator {
    fn default() -> Self {
        Self::new(program::planet_program(), 0, program::DEFAULT_SUN_DIR)
    }
}

impl Generator {
    pub fn new(ops: Vec<voxel_core::worldop::WorldOp>, seed: u32, sun: glam::Vec3) -> Self {
        let has_height = ops.iter().any(|op| op.is_height_op());
        Self {
            ops: std::sync::Arc::new(ops),
            seed,
            sun,
            has_height,
        }
    }

    pub fn ops(&self) -> &[voxel_core::worldop::WorldOp] {
        &self.ops
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Direction the sun comes from (normalized).
    pub fn sun_direction(&self) -> glam::Vec3 {
        self.sun.normalize_or(glam::Vec3::Y)
    }

    /// Signed distance and material at a point — the CPU twin of the
    /// density shader.
    pub fn sample(&self, p: glam::Vec3, voxel_size: f32) -> (f32, u32) {
        program::eval(&self.ops, self.seed, p, voxel_size)
    }

    /// Heightfield (meters) at a world XZ, evaluated at `voxel_size`
    /// (1.0 = full detail). Mirrors the GPU exactly.
    pub fn height(&self, xz: Vec2, voxel_size: f32) -> f32 {
        program::eval_height(&self.ops, self.seed, xz, voxel_size)
    }

    /// Field registers at a column (prop densities, gameplay queries).
    pub fn fields(&self, xz: Vec2) -> [f32; voxel_core::worldop::FIELD_SLOTS] {
        program::eval_fields(&self.ops, self.seed, xz, 4.0)
    }

    /// The structural Y-lattice spacing, if the program has one.
    pub fn lattice_y_spacing(&self) -> Option<f32> {
        program::lattice_y_spacing(&self.ops)
    }


    /// Patch density in [0, 1]: slow spatial noise so scattered props
    /// come in coherent patches with clearings.
    pub fn patch_density(&self, xz: Vec2, scale: f32, offset: Vec2, contrast: f32, bias: f32) -> f32 {
        let n = fbm(self.seed, xz + offset, scale, 3, 1.0) + 0.5;
        (n * contrast + bias).clamp(0.0, 1.0)
    }

    /// Soft sun shadow: horizon march over the band-limited heightfield.
    /// Mirrors the WGSL bake in voxel_mesh_chunks.wgsl.
    pub fn sun_shadow(&self, pos: glam::Vec3) -> f32 {
        // Twin of the GPU gate: a heightless program (interiors) casts
        // no terrain shadow.
        if !self.has_height {
            return 1.0;
        }
        let sun = self.sun_direction();
        let mut occ = 0.0f32;
        let mut t = 8.0f32;
        for _ in 0..9 {
            let sp = pos + sun * t;
            let dh = self.height(Vec2::new(sp.x, sp.z), 8.0) - sp.y;
            occ = occ.max(dh / t);
            t *= 1.8;
        }
        let x = (occ / 0.2).clamp(0.0, 1.0);
        1.0 - x * x * (3.0 - 2.0 * x)
    }

    /// Approximate surface up-ness (normal Y) via central differences.
    pub fn up(&self, xz: Vec2, voxel_size: f32) -> f32 {
        self.normal(xz, voxel_size).y
    }

    /// Approximate surface normal via central differences.
    pub fn normal(&self, xz: Vec2, voxel_size: f32) -> glam::Vec3 {
        let e = 2.0;
        let hx = self.height(xz + Vec2::new(e, 0.0), voxel_size)
            - self.height(xz - Vec2::new(e, 0.0), voxel_size);
        let hz = self.height(xz + Vec2::new(0.0, e), voxel_size)
            - self.height(xz - Vec2::new(0.0, e), voxel_size);
        glam::Vec3::new(-hx, 2.0 * e, -hz).normalize()
    }
}
