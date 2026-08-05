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
/// Meta op (no SDF effect): the world has a water surface. `p0.x = sea
/// level (m)`. Drives the ocean draw and shoreline shading.
pub const WOP_WATER: u32 = 12;
/// Meta op (no SDF effect): vegetation grows on the heightfield surface.
/// `p0.x = density multiplier`. Drives tree/grass streaming.
pub const WOP_VEGETATION: u32 = 13;

/// Skip this op at coarse LODs (voxel size >= the structural cutoff).
pub const WOP_FLAG_FINE_ONLY: u32 = 1;
/// Skip this op at fine LODs.
pub const WOP_FLAG_COARSE_ONLY: u32 = 2;

/// Voxel size (meters) at and above which structural detail ops are dropped
/// and `COARSE_ONLY` ops apply. Must match the WGSL interpreter.
pub const WOP_COARSE_VOXEL_M: f32 = 4.0;

/// One generator op, 64 bytes, `#[repr(C)]` — uploaded verbatim.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct WorldOp {
    pub kind: u32,
    pub flags: u32,
    pub material: u32,
    pub _pad: u32,
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
            _pad: 0,
            p0: [0.0; 4],
            p1: [0.0; 4],
            p2: [0.0; 4],
        }
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

    /// True for ops that only touch the height register (the height-only
    /// interpreters in the mesh/water shaders evaluate exactly these).
    pub fn is_height_op(&self) -> bool {
        self.kind == WOP_HEIGHT_FBM || self.kind == WOP_HEIGHT_OFFSET
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
