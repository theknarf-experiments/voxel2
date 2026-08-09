//! World-generation program ops: the data that *is* a world. A level's base
//! generator is an ordered list of `WorldOp`s evaluated by two twin
//! interpreters — the GPU density shader and the CPU mirror in
//! voxel-worldgen — over a small register file (height accumulator, SDF
//! accumulator, Y-lattice locals, shaft locals). Layout is shared
//! bit-for-bit with the WGSL `WorldOp` struct (64 bytes).

use bytemuck::{Pod, Zeroable};

/// Band-limited 2D FBM added to the height register.
/// `p0 = (offset_x, offset_z, cycles_per_m, amplitude_m)`, `p1.x = octaves`.
pub const WOP_HEIGHT_FBM: u32 = 0;
/// Constant added to the height register. `p0.x = meters`.
pub const WOP_HEIGHT_OFFSET: u32 = 1;
/// Merge the heightfield surface into the SDF: `d = y - H`.
pub const WOP_HEIGHT_SURFACE: u32 = 2;
/// Merge "solid everywhere" (the structure reads as a solid mass at coarse
/// LODs; pair with `COARSE_ONLY` and a later cut).
pub const WOP_COARSE_SOLID: u32 = 3;
/// Set the Y-lattice registers: `level = round(y / spacing)`,
/// `fy = y - level * spacing`. `p0.x = spacing`.
pub const WOP_LATTICE_Y: u32 = 4;
/// Merge horizontal slabs on the Y lattice. `p0.x = half thickness`.
pub const WOP_SLABS_Y: u32 = 5;
/// Cut hash-gated holes on an XZ grid, one roll per (cell, lattice level).
/// `p0 = (cell_m, chance)`, `p1 = (half_x, half_y, half_z)`; the cut box is
/// centered on the cell at the current lattice plane.
pub const WOP_GRID_HOLES: u32 = 6;
/// Merge square columns on a jittered XZ grid.
/// `p0 = (spacing, jitter_m, girth_base, girth_var)`.
pub const WOP_PILLARS_XZ: u32 = 7;
/// Merge hash-gated axis-aligned walls with optional doorway cuts.
/// `p0 = (spacing, half_thickness, chance, axis)` (axis 0 = x-normal,
/// 1 = z-normal), `p1 = (gate_salt, door_cell, door_chance, door_level_salt)`,
/// `p2 = (door_half x/y/z, door_y_offset)`.
pub const WOP_WALLS: u32 = 8;
/// Compute the shaft registers (jittered XZ grid of vertical cylinders);
/// does not touch the SDF. `p0 = (spacing, jitter_m, radius_base, radius_var)`.
pub const WOP_SHAFTS_XZ: u32 = 9;
/// Carve the computed shafts out of the SDF.
pub const WOP_SHAFTS_CUT: u32 = 10;
/// Merge catwalk beams bridging the shafts along X on every Nth lattice
/// level. `p0 = (every_n, half_width, y_offset, half_height)`, `p1.x = reach`.
pub const WOP_BEAMS: u32 = 11;
// Kinds 12 and 13 are retired: 12 was a water meta op and 13 the
// vegetation one. Neither belongs here — the engine has no water or
// vegetation concept, only geometry ops. Sea level is a host constant
// and spawners are level data.
/// Domain-warp the XZ coordinate that later height ops sample.
/// `p0 = (cycles_per_m, amplitude_m, offset_x, offset_z)`, `p1.x = octaves`.
pub const WOP_WARP_XZ: u32 = 14;
/// Meta op (no SDF effect): accumulate a 2D FBM band into a field
/// register — named world data consumed by spawner densities and other
/// CPU-side queries, never by the density itself (the GPU interpreter
/// skips it via its default arm). Multiple field ops targeting one slot
/// accumulate, and domain warps applied by earlier `WOP_WARP_XZ` ops
/// affect field samples exactly like height samples.
/// `p0 = (offset_x, offset_z, cycles_per_m, amplitude)`,
/// `p1 = (octaves, noise_mode, slot, bias)`.
pub const WOP_FIELD: u32 = 17;

/// Number of field registers.
pub const FIELD_SLOTS: usize = 4;

/// Sample the two region axes into the `ta`/`tb` registers.
///
/// Every band op in a program tests the same two axes, so they are
/// computed ONCE per sample here rather than per band: three regions
/// cost two noise evaluations, not six. Must precede any op that reads
/// them, which in practice means first.
///
/// Deliberately unwarped — a region is its own field, not a feature of
/// the terrain, so it does not follow an earlier `WOP_WARP_XZ`.
///
/// `p0 = (a_offset_x, a_offset_z, a_cycles_per_m, b_cycles_per_m)`,
/// `p1 = (b_offset_x, b_offset_z, octaves, -)`.
pub const WOP_REGION_AXES: u32 = 19;

/// Add an FBM band to the height, faded in by how firmly a point is
/// inside a region — terrain character per region.
///
/// The fade is smooth, unlike [`WOP_MATERIAL_BAND`]'s hard test: two
/// materials cannot blend but two heights must, or every region border
/// would be a cliff. Regions whose bands overlap simply sum, which is
/// what makes a transition read as one landscape becoming another.
///
/// The FBM is ZERO-MEAN, so on its own a region digs as much as it
/// raises — a "mountain range" built from noise alone sits half below
/// the ground around it and fills with water. `lift` is the constant
/// that makes a region a massif or a basin; the noise then shapes it.
///
/// `p0 = (offset_x, offset_z, cycles_per_m, amplitude_m)`,
/// `p1 = (octaves, noise_mode, feather, lift_m)`,
/// `p2 = (a0, a1, b0, b1)` — the region, in the axes `WOP_REGION_AXES`
/// sampled.
pub const WOP_HEIGHT_BAND_FBM: u32 = 20;

/// Repaint the surface material where two low-frequency noise samples
/// both fall inside a band: `if mat == from && a in [a0,a1) && b in
/// [b0,b1) { mat = to }`.
///
/// The mechanism, not a use for it. Two independent noise axes and a box
/// in their product is enough to carve a plane into regions, and a level
/// that wants regions named after climate, faction or fallout writes the
/// bands and the names in its own file — the interpreter only ever sees
/// numbers.
///
/// Reads the `ta`/`tb` registers [`WOP_REGION_AXES`] filled, so the
/// noise is paid for once however many regions a level declares.
///
/// `head.z = to material`, `p0 = (a0, a1, b0, b1)`,
/// `p1.z = from material`.
pub const WOP_MATERIAL_BAND: u32 = 18;

/// Cliff step: adds `amp * smoothstep(start, end, h)` to the height
/// register — terrain crossing the band grows a wall (iq's Rainforest
/// cliff term). `p0 = (start_m, end_m, amp_m)`.
pub const WOP_HEIGHT_STEP: u32 = 16;
/// Anisotropic 3D FBM solid: union or carve by a noise iso-surface —
/// caves, overhangs, floating islands. `p0 = (cycles_per_m_xz,
/// cycles_per_m_y, threshold, width_m)`, `p1 = (offset x, y, z, mode)`
/// (mode 0 = union, 1 = carve), `p2.x = octaves`.
pub const WOP_FBM3: u32 = 15;

/// Noise modes for `WOP_HEIGHT_FBM` (`p1.y`).
pub const NOISE_MODE_FBM: u32 = 0;
/// Sharp ridges (mountain crests, dune fields).
pub const NOISE_MODE_RIDGED: u32 = 1;
/// Rounded billows (rolling hills, cloudy blobs).
pub const NOISE_MODE_BILLOW: u32 = 2;

/// Skip this op at coarse LODs (voxel size >= the structural cutoff).
pub const WOP_FLAG_FINE_ONLY: u32 = 1;
/// Skip this op at fine LODs.
pub const WOP_FLAG_COARSE_ONLY: u32 = 2;

/// Voxel size (meters) at and above which structural detail ops are dropped
/// and `COARSE_ONLY` ops apply. Must match the WGSL interpreter.
pub const WOP_COARSE_VOXEL_M: f32 = 4.0;

/// Pack a region gate into the header word: four band edges in 0..1, one
/// byte each, in the order `(a0, a1, b0, b1)`.
///
/// A byte is 1/255 of an axis, and an axis is kilometres wide — finer than
/// any boundary a level can meaningfully author, and it costs no space at
/// all, which matters more: the op is 64 bytes and every one of them is
/// spoken for.
///
/// Zero is the ungated sentinel and cannot collide with a real gate,
/// because it decodes to the empty band `a in [0, 0)`.
pub fn pack_region(band: [f32; 4]) -> u32 {
    let q = |v: f32| ((v.clamp(0.0, 1.0) * 255.0).round() as u32) & 0xFF;
    q(band[0]) | (q(band[1]) << 8) | (q(band[2]) << 16) | (q(band[3]) << 24)
}

/// Inverse of [`pack_region`]. The twin of the WGSL unpack in each
/// interpreter's gate.
pub fn unpack_region(packed: u32) -> [f32; 4] {
    let u = |s: u32| ((packed >> s) & 0xFF) as f32 / 255.0;
    [u(0), u(8), u(16), u(24)]
}

/// One generator op, 64 bytes, `#[repr(C)]` — uploaded verbatim.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct WorldOp {
    pub kind: u32,
    pub flags: u32,
    pub material: u32,
    /// Region gate, packed by [`pack_region`]; 0 applies the op
    /// everywhere. Tested against the `ta`/`tb` registers before the op
    /// runs at all, so ANY op can be confined to a region without the op
    /// itself knowing regions exist.
    ///
    /// Hard-edged, unlike [`WOP_HEIGHT_BAND_FBM`]'s fade, because what
    /// this gates is structure: a wall is present or it is not, and there
    /// is no half of one to blend to. Where a level wants the ground to
    /// change gradually it uses the band ops; where it wants the
    /// ARCHITECTURE to change it uses this, and an abrupt seam is the
    /// point.
    pub region: u32,
    pub p0: [f32; 4],
    pub p1: [f32; 4],
    pub p2: [f32; 4],
}

impl WorldOp {
    pub fn new(kind: u32) -> Self {
        Self {
            kind,
            flags: 0,
            material: 0,
            region: 0,
            p0: [0.0; 4],
            p1: [0.0; 4],
            p2: [0.0; 4],
        }
    }

    /// Confine this op to a box in the two region axes.
    pub fn region(mut self, band: [f32; 4]) -> Self {
        self.region = pack_region(band);
        self
    }

    pub fn flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }

    pub fn material(mut self, material: u32) -> Self {
        self.material = material;
        self
    }

    pub fn p0(mut self, v: [f32; 4]) -> Self {
        self.p0 = v;
        self
    }

    pub fn p1(mut self, v: [f32; 4]) -> Self {
        self.p1 = v;
        self
    }

    pub fn p2(mut self, v: [f32; 4]) -> Self {
        self.p2 = v;
        self
    }

    /// True for ops that CONTRIBUTE to the height — which is what tells a
    /// world it has a heightfield at all, and so whether the horizon
    /// shadow bake has anything to bake.
    ///
    /// Not the same as appearing in the height-only replay:
    /// `WOP_REGION_AXES` is replayed there (a height band needs its
    /// registers) but adds nothing to `h`, and a sunless interior that
    /// divides itself into districts still has no heightfield.
    pub fn is_height_op(&self) -> bool {
        self.kind == WOP_HEIGHT_FBM
            || self.kind == WOP_HEIGHT_OFFSET
            || self.kind == WOP_HEIGHT_STEP
            || self.kind == WOP_HEIGHT_BAND_FBM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_is_64_bytes() {
        assert_eq!(std::mem::size_of::<WorldOp>(), 64);
    }
}
