//! Single-source GPU struct layouts: the twins `opgen` does not cover.
//!
//! `opgen` killed the drift class for generator ops by writing each op
//! body once. Two layouts were left doing it by hand:
//!
//! - [`CHUNK_PARAMS`], the per-chunk uniform, written out in Rust and in
//!   two shaders.
//! - [`MATERIALS`], which is worse than a twin. A material recipe is eight
//!   `vec4`s whose MEANING depends on the kind in `head.x`, so a named
//!   parameter like a canopy's rock start is `c1.w` in the packer and
//!   `m.c1.w` in the shader, agreed by nothing but a comment. Adding one
//!   field meant reading the comment to find a free component and hoping.
//!
//! Both are emitted as WGSL and spliced by `tools/genops`, with guard
//! tests in voxel-render. The slot is written once, here.

// --- per-chunk uniform --------------------------------------------------------

/// One field of a GPU struct: the WGSL type, and why it is there.
pub struct Field {
    pub name: &'static str,
    pub ty: &'static str,
    pub doc: &'static str,
}

/// The per-chunk uniform, twin of `ChunkParams` in voxel-render.
///
/// Read by the density pass and the mesh pass, which is why it was three
/// copies. `origin_voxels.w` carries the world id in both, and the mesh
/// pass reads a seam mask out of `_pad.x` the density pass ignores — the
/// doc says so once rather than differently in each shader.
pub const CHUNK_PARAMS: &[Field] = &[
    Field {
        name: "origin",
        ty: "vec4<f32>",
        doc: "xyz = chunk minimum corner in world meters, w = voxel size in meters.",
    },
    Field {
        name: "origin_voxels",
        ty: "vec4<i32>",
        doc: "Minimum corner in integer world-voxel units (pos * 32, this chunk's\n\
              scale); w = which WORLD's program to interpret. Sample positions\n\
              derive from these EXACT integers so two chunks sharing a sample\n\
              compute a bit-identical position at any voxel size — `origin + idx\n\
              * vs` rounds differently per chunk whenever the voxel size is not\n\
              an exact binary float (0.1 m is not), and one ULP flips a sign\n\
              where a surface grazes a sample: deterministic seam cracks.",
    },
    Field {
        name: "slot",
        ty: "u32",
        doc: "Density arena slot this chunk's samples live in.",
    },
    Field {
        name: "base_vertex",
        ty: "u32",
        doc: "",
    },
    Field {
        name: "first_index",
        ty: "u32",
        doc: "",
    },
    Field {
        name: "counts_slot",
        ty: "u32",
        doc: "",
    },
    Field {
        name: "csg_offset",
        ty: "u32",
        doc: "Range into this frame's concatenated CSG op buffer.",
    },
    Field {
        name: "csg_count",
        ty: "u32",
        doc: "",
    },
    Field {
        name: "aux",
        ty: "vec4<u32>",
        doc: "x = seam mask, 2 bits per face (+x,-x,+y,-y,+z,-z): 1 = neighbour\n\
              coarser, 2 = neighbour finer. Read by the mesh pass only.\n\
              y = draw the cells a snap failed in (VOXEL_EVAL_HOLES).\n\
              z = base of this chunk's per-cell op index in `csg_cells`, or\n\
              0 when the chunk carries none and every op is walked.\n\
              w = unused. It was `_pad` and already carried the first two.",
    },
];

/// The WGSL declaration of a struct, from its field table.
pub fn wgsl_struct(name: &str, fields: &[Field]) -> String {
    let mut out = format!("struct {name} {{\n");
    for f in fields {
        for line in f.doc.lines().filter(|l| !l.trim().is_empty()) {
            out.push_str(&format!("    // {}\n", line.trim()));
        }
        out.push_str(&format!("    {}: {},\n", f.name, f.ty));
    }
    out.push_str("}\n");
    out
}

// --- surface map texel order --------------------------------------------------

/// Texels per side of a surface-map tile.
///
/// The map is stored in tiles rather than rows because it is read PER
/// FRAGMENT and neighbouring fragments land on neighbouring texels — which
/// row-major puts a whole row apart vertically (4 KB at 4096 wide), so a
/// screen-space quad straddles cache lines every time it moves down one
/// pixel. An 8x8 tile is 64 bytes: one line, one neighbourhood. Measured
/// 0.7 ms a frame.
pub const TEXEL_TILE: u32 = 8;

/// Where a texel lives in the raster. `size` is the map's width in texels
/// and must be a multiple of [`TEXEL_TILE`].
pub fn texel_index(size: u32, x: u32, z: u32) -> u32 {
    let tiles_per_row = size / TEXEL_TILE;
    let tile = (z / TEXEL_TILE) * tiles_per_row + (x / TEXEL_TILE);
    tile * TEXEL_TILE * TEXEL_TILE + (z % TEXEL_TILE) * TEXEL_TILE + (x % TEXEL_TILE)
}

/// The shader's twin of [`texel_index`], generated so the two orders
/// cannot disagree. They fail SILENTLY if they do: the map still reads,
/// it just reads somewhere else, and the world is painted with a scramble
/// of its own data.
pub fn wgsl_texel_index() -> String {
    format!(
        "const TEXEL_TILE: u32 = {TEXEL_TILE}u;\n\
         fn surface_texel_index(size: u32, x: u32, z: u32) -> u32 {{\n\
         \x20   let tiles_per_row = size / TEXEL_TILE;\n\
         \x20   let tile = (z / TEXEL_TILE) * tiles_per_row + (x / TEXEL_TILE);\n\
         \x20   return tile * TEXEL_TILE * TEXEL_TILE + (z % TEXEL_TILE) * TEXEL_TILE\n\
         \x20       + (x % TEXEL_TILE);\n\
         }}\n"
    )
}

#[cfg(test)]
mod texel_tests {
    use super::*;

    /// Every texel of a small map gets a distinct slot, and the slots fill
    /// the map exactly. A tiling that overlaps or leaves holes is the
    /// failure that would not announce itself.
    #[test]
    fn the_tiled_order_is_a_permutation() {
        let size = 32;
        let mut seen = vec![false; (size * size) as usize];
        for z in 0..size {
            for x in 0..size {
                let i = texel_index(size, x, z) as usize;
                assert!(!seen[i], "({x},{z}) collides at {i}");
                seen[i] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "the order leaves holes");
    }

    /// The point of the order: a row of eight neighbours is contiguous.
    #[test]
    fn a_tile_is_one_run_of_memory() {
        let base = texel_index(4096, 64, 64);
        for x in 0..TEXEL_TILE {
            assert_eq!(texel_index(4096, 64 + x, 64), base + x);
        }
    }
}

// --- material recipes ---------------------------------------------------------

/// Which of the eight `vec4`s a parameter lives in. `head` is the kind
/// tag and belongs to no parameter, so the payload slots are these seven.
pub const SLOTS: [&str; 7] = ["c0", "c1", "c2", "c3", "p0", "p1", "p2"];

/// Which part of a slot a parameter occupies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Comp {
    X,
    Y,
    Z,
    W,
    /// `xyz` — a colour. Occupies X, Y and Z together.
    Rgb,
}

impl Comp {
    pub fn wgsl(self) -> &'static str {
        match self {
            Comp::X => "x",
            Comp::Y => "y",
            Comp::Z => "z",
            Comp::W => "w",
            Comp::Rgb => "rgb",
        }
    }

    pub fn ty(self) -> &'static str {
        match self {
            Comp::Rgb => "vec3<f32>",
            _ => "f32",
        }
    }

    /// The float lanes this occupies, for the overlap check.
    fn lanes(self) -> &'static [usize] {
        match self {
            Comp::X => &[0],
            Comp::Y => &[1],
            Comp::Z => &[2],
            Comp::W => &[3],
            Comp::Rgb => &[0, 1, 2],
        }
    }
}

/// One named material parameter, and where it sits.
pub struct MatParam {
    pub name: &'static str,
    /// Index into [`SLOTS`].
    pub slot: usize,
    pub comp: Comp,
}

/// One material kind: what `head.x` selects, and its parameter names.
pub struct MatKind {
    pub name: &'static str,
    pub id: u32,
    pub params: &'static [MatParam],
}

const fn p(name: &'static str, slot: usize, comp: Comp) -> MatParam {
    MatParam { name, slot, comp }
}

/// Every material kind's slot map.
///
/// Order within a kind is presentation only; the slot is the contract.
/// A parameter the shader has stopped reading stays listed, because the
/// packer still writes it and the next person to want a free component
/// needs to see that it is taken.
pub const MATERIALS: &[MatKind] = &[
    MatKind {
        name: "surface",
        id: 0,
        params: &[
            p("base", 0, Comp::Rgb),
            p("grain", 0, Comp::W),
            p("grime_tint", 1, Comp::Rgb),
            p("grime_amount", 1, Comp::W),
            p("moss_color", 2, Comp::Rgb),
            p("moss_amount", 2, Comp::W),
            p("emissive_color", 3, Comp::Rgb),
            p("emissive_intensity", 3, Comp::W),
            p("band_freq", 4, Comp::X),
            p("band_amp", 4, Comp::Y),
            p("band_lo", 4, Comp::Z),
            p("band_hi", 4, Comp::W),
            p("band_warp", 5, Comp::X),
            p("streaks", 5, Comp::Y),
            p("strip_spacing", 5, Comp::Z),
            p("strip_level_spacing", 5, Comp::W),
            p("strip_chance", 6, Comp::X),
            p("strip_glow", 6, Comp::Y),
            p("detail_fade", 6, Comp::Z),
        ],
    },
    MatKind {
        name: "zoned",
        id: 1,
        params: &[
            p("low", 0, Comp::Rgb),
            p("mid_start", 0, Comp::W),
            p("mid_a", 1, Comp::Rgb),
            p("high_start", 1, Comp::W),
            p("high_a", 2, Comp::Rgb),
            p("peak_start", 2, Comp::W),
            p("peak", 3, Comp::Rgb),
            p("border", 3, Comp::W),
            p("mid_b", 4, Comp::Rgb),
            p("mid_width", 4, Comp::W),
            p("high_b", 5, Comp::Rgb),
            p("high_width", 5, Comp::W),
            p("peak_width", 6, Comp::X),
            p("steep_hi", 6, Comp::Y),
            p("steep_lo", 6, Comp::Z),
            p("detail_fade", 6, Comp::W),
        ],
    },
    MatKind {
        name: "canopy",
        id: 2,
        params: &[
            p("canopy_a", 0, Comp::Rgb),
            p("canopy_start", 0, Comp::W),
            p("canopy_b", 1, Comp::Rgb),
            p("rock_start", 1, Comp::W),
            p("rock", 2, Comp::Rgb),
            p("rock_width", 2, Comp::W),
            p("patch", 3, Comp::Rgb),
            p("border", 3, Comp::W),
            p("low", 4, Comp::Rgb),
            p("canopy_width", 4, Comp::W),
            p("crown_scale", 5, Comp::X),
            p("crown_relief", 5, Comp::Y),
            p("strata_scale", 5, Comp::Z),
            p("strata_relief", 5, Comp::W),
            p("steep_hi", 6, Comp::X),
            p("steep_lo", 6, Comp::Y),
            p("detail_fade", 6, Comp::Z),
            p("patch_amount", 6, Comp::W),
        ],
    },
];

impl MatKind {
    pub fn param(&self, name: &str) -> Option<&MatParam> {
        self.params.iter().find(|p| p.name == name)
    }
}

/// The kind of a given name, for the packer and the tests.
pub fn material_kind(name: &str) -> &'static MatKind {
    MATERIALS
        .iter()
        .find(|k| k.name == name)
        .unwrap_or_else(|| panic!("no material kind `{name}`"))
}

/// Named accessors for every material parameter.
///
/// The shader reads `canopy_rock_start(m)`, never `m.c1.w`, so moving a
/// parameter is one edit in [`MATERIALS`] and nothing else. Emitted for
/// every kind at once because the draw shader switches on `head.x` and
/// needs all of them.
pub fn wgsl_material_accessors() -> String {
    let mut out = String::new();
    for kind in MATERIALS {
        out.push_str(&format!(
            "// --- {} (head.x == {}u) ---\n",
            kind.name, kind.id
        ));
        out.push_str(&format!(
            "const MAT_KIND_{}: u32 = {}u;\n",
            kind.name.to_uppercase(),
            kind.id
        ));
        for param in kind.params {
            out.push_str(&format!(
                "fn {}_{}(m: WorldMaterial) -> {} {{ return m.{}.{}; }}\n",
                kind.name,
                param.name,
                param.comp.ty(),
                SLOTS[param.slot],
                param.comp.wgsl(),
            ));
        }
    }
    out
}

/// Build a material's eight `vec4`s by NAME.
///
/// The packer that feeds the GPU and the shader that reads it now agree by
/// construction: both go through [`MATERIALS`], so neither can move a
/// parameter without the other following.
pub struct MatPack {
    kind: &'static MatKind,
    slots: [[f32; 4]; 7],
}

impl MatPack {
    pub fn new(kind: &str) -> Self {
        Self {
            kind: material_kind(kind),
            slots: [[0.0; 4]; 7],
        }
    }

    fn at(&self, name: &str) -> &'static MatParam {
        self.kind.param(name).unwrap_or_else(|| {
            panic!(
                "material kind `{}` has no parameter `{name}`",
                self.kind.name
            )
        })
    }

    /// Set a scalar parameter.
    pub fn set(&mut self, name: &str, v: f32) -> &mut Self {
        let param = self.at(name);
        assert!(param.comp != Comp::Rgb, "`{name}` is a colour — use `rgb`");
        self.slots[param.slot][param.comp.lanes()[0]] = v;
        self
    }

    /// Set a colour parameter.
    pub fn rgb(&mut self, name: &str, v: [f32; 3]) -> &mut Self {
        let param = self.at(name);
        assert!(param.comp == Comp::Rgb, "`{name}` is a scalar — use `set`");
        self.slots[param.slot][..3].copy_from_slice(&v);
        self
    }

    /// Set a colour and the scalar sharing its slot, which is how most of
    /// them are authored: a colour with one number about it.
    pub fn rgb_w(&mut self, rgb: &str, v: [f32; 3], w: &str, s: f32) -> &mut Self {
        self.rgb(rgb, v).set(w, s)
    }

    /// `head.x`, then the seven payload slots in [`SLOTS`] order.
    pub fn finish(&self) -> (u32, [[f32; 4]; 7]) {
        (self.kind.id, self.slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two parameters of a kind must never share a float.
    ///
    /// This is the check that was a comment. Finding a free component used
    /// to mean reading the layout note and counting, which is exactly the
    /// job a machine should have.
    #[test]
    fn no_two_parameters_of_a_kind_share_a_component() {
        for kind in MATERIALS {
            let mut taken: [[Option<&str>; 4]; 7] = [[None; 4]; 7];
            for param in kind.params {
                for &lane in param.comp.lanes() {
                    if let Some(other) = taken[param.slot][lane] {
                        panic!(
                            "{}: `{}` and `{}` both use {}.{}",
                            kind.name,
                            other,
                            param.name,
                            SLOTS[param.slot],
                            ["x", "y", "z", "w"][lane],
                        );
                    }
                    taken[param.slot][lane] = Some(param.name);
                }
            }
        }
    }

    /// Kind ids are what `head.x` switches on, so they must be distinct
    /// and must be the indices the engine's `MAT_KIND_*` constants use.
    #[test]
    fn material_kind_ids_are_distinct_and_dense() {
        let mut ids: Vec<u32> = MATERIALS.iter().map(|k| k.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..MATERIALS.len() as u32).collect::<Vec<_>>());
    }

    #[test]
    fn the_packer_writes_where_the_accessor_reads() {
        let mut pack = MatPack::new("canopy");
        pack.rgb("canopy_a", [1.0, 2.0, 3.0]).set("rock_start", 9.0);
        let (kind, slots) = pack.finish();
        assert_eq!(kind, 2);
        assert_eq!(slots[0][..3], [1.0, 2.0, 3.0], "canopy_a is c0.rgb");
        assert_eq!(slots[1][3], 9.0, "rock_start is c1.w");
        // And the shader is told the same thing.
        let wgsl = wgsl_material_accessors();
        assert!(
            wgsl.contains("fn canopy_canopy_a(m: WorldMaterial) -> vec3<f32> { return m.c0.rgb; }")
        );
        assert!(wgsl.contains("fn canopy_rock_start(m: WorldMaterial) -> f32 { return m.c1.w; }"));
    }

    #[test]
    #[should_panic(expected = "no parameter `nope`")]
    fn packing_an_unknown_parameter_is_a_bug_not_a_silent_zero() {
        MatPack::new("surface").set("nope", 1.0);
    }
}
