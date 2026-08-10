//! Single-source generator-op semantics: every interpreter op body is
//! written ONCE here, in a dialect that is valid WGSL and valid Rust
//! after token substitution. Emitters produce:
//!
//! - the CPU arms (`voxel-worldgen`'s build script → `include!` into
//!   `program::eval` / `program::eval_height`), and
//! - the GPU arms (`tools/genops` splices them between `GENOPS` markers
//!   in the density / mesh shaders, and in any host shader that opts in).
//!
//! This kills the twin-drift class mechanically: an op edited or added
//! here lands in all five interpreter sites at once. Guard tests in
//! voxel-render fail when the spliced shader text goes stale.
//!
//! Dialect rules (kept deliberately tiny):
//! - `@p0.x`..`@p2.w`, `@p0.xy`, `@p0.zw`, `@p1.xyz`, `@p2.xyz` read op
//!   params; `@mat` reads the op's material id.
//! - `@FBM(...)` is the band-limited FBM entry point; `@VS@` expands to
//!   `vs, ` in full-interpreter contexts and nothing in height-replay
//!   contexts (whose FBM wrapper has no vs parameter).
//! - `var x = ...;` declares a mutable local (becomes `let mut` in Rust).
//! - Everything else is shared syntax: `let`, paren-less `if`, `f32`
//!   math, and the helper set below (`v2/v3/iv2/iv3/to_i/to_u/to_v2/
//!   to_iv2/floor2` plus the twinned `hash2/hash3/sd_box/fbm3/
//!   round_half_up/smoothstep/abs/max/min/length` builtins or shims).

use crate::worldop::*;

/// A value the interpreter carries between ops — the register file, named
/// as the things the registers MEAN.
///
/// One variant per group of registers that move together: an op that
/// writes the SDF always writes the material with it, and the Y lattice is
/// `level` and `fy` or neither. Naming the group rather than the register
/// is what lets a level wire two ops together without knowing that a
/// register file exists at all — the level says `"in": {"lattice": "floors"}`
/// and the compiler lowers that back to `level`/`fy`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Value {
    /// `h` — the heightfield accumulator.
    Height,
    /// `d` and `mat` — the SDF accumulator and the material it carries.
    /// One value, because nothing writes one without the other.
    Sdf,
    /// `ta`, `tb` — the two region axes, paid for once per sample.
    Regions,
    /// `warp` — the XZ offset later height and field ops sample through.
    Warp,
    /// `level`, `fy` — the Y lattice.
    Lattice,
    /// `sxz`, `sr`, `shaft` — the shaft locals.
    Shafts,
    /// A named scalar field. CPU-side consumers only: the GPU interpreter
    /// skips the op that fills it.
    Field,
}

impl Value {
    /// The interpreter registers this value occupies.
    ///
    /// The guard test reads these back off the op bodies, so a body that
    /// touches `fy` without declaring [`Value::Lattice`] fails the build
    /// rather than compiling into a wiring the level cannot see.
    pub const fn registers(self) -> &'static [&'static str] {
        match self {
            Value::Height => &["h"],
            Value::Sdf => &["d", "mat"],
            Value::Regions => &["ta", "tb"],
            Value::Warp => &["warp"],
            Value::Lattice => &["level", "fy"],
            Value::Shafts => &["sxz", "sr", "shaft"],
            Value::Field => &["fields"],
        }
    }

    /// Every value, for the guard test and for the compiler's allocator.
    pub const ALL: &'static [Value] = &[
        Value::Height,
        Value::Sdf,
        Value::Regions,
        Value::Warp,
        Value::Lattice,
        Value::Shafts,
        Value::Field,
    ];
}

/// A named port on an op: what a level wires to, and what type it carries.
pub type Port = (&'static str, Value);

/// Ops with no GPU twin, and so no entry in [`OPS`]: the interpreter arm is
/// hand-written in `voxel_worldgen::program` and the density shader skips
/// them through its default arm. They still have a wiring surface, so a
/// level can name one and the compiler can check it.
pub const META_PORTS: &[(u32, &[Port], &[Port])] = &[(
    WOP_FIELD,
    &[("warp", Value::Warp)],
    &[("field", Value::Field)],
)];

/// Every op kind's ports, whether or not it has a GPU twin.
///
/// One lookup for the compiler, so nothing has to know which ops are
/// spliced into shaders and which are CPU-side meta ops.
pub fn ports(kind: u32) -> Option<(&'static [Port], &'static [Port])> {
    if let Some(op) = OPS.iter().find(|o| o.kind == kind) {
        return Some((op.ins, op.outs));
    }
    META_PORTS
        .iter()
        .find(|(k, ..)| *k == kind)
        .map(|(_, ins, outs)| (*ins, *outs))
}

/// One generator op's single-source definition.
pub struct OpDef {
    /// Rust const name (must exist in `voxel_core::worldop`).
    pub name: &'static str,
    pub kind: u32,
    /// Part of the height-only replay (shadow bake, seabed, eval_height).
    pub height: bool,
    /// What this op consumes and produces.
    ///
    /// The ports are the op's PUBLIC surface — a level names them and the
    /// compiler checks them — while the registers below are the private
    /// lowering. Declared here rather than inferred from the body because
    /// the body is a text dialect, and a wiring surface read out of text
    /// is a wiring surface that changes when someone renames a local.
    pub ins: &'static [Port],
    pub outs: &'static [Port],
    /// Body in the shared dialect.
    pub body: &'static str,
    /// How this op bounds the registers over a BOX, in plain Rust against
    /// [`crate::interval::Interval`], with the same `@p0.x` param tokens.
    ///
    /// Range analysis runs on the CPU only, so this needs no WGSL twin:
    /// it is one more line in this table rather than a third dialect.
    /// `None` means nobody has bounded this op yet, and a program
    /// containing one declines to be analysed — unbounded costs pruning,
    /// never correctness, which is what lets a heightfield be pruned and
    /// an infinite megastructure simply not be.
    ///
    /// In scope: `d` and `h` (`Interval`, the SDF and height registers)
    /// and `py` (the box's y extent). Must be CONSERVATIVE: the result
    /// has to contain every value the op could produce over the box.
    pub range: Option<&'static str>,
}

pub const OPS: &[OpDef] = &[
    OpDef {
        name: "WOP_HEIGHT_FBM",
        kind: WOP_HEIGHT_FBM,
        height: true,
        ins: &[("height", Value::Height), ("warp", Value::Warp)],
        outs: &[("height", Value::Height)],
        body: "\
h += @FBM(pxz + warp + @p0.xy, @p0.z, to_i(@p1.x), @VS@to_u(@p1.y)) * @p0.w;",
        range: Some("\
            // Bounded over the box rather than everywhere: the amplitude
            // alone is the whole world's range and decides nothing near
            // the ground.
            h = h + frange(pxz_lo + @p0.xy, pxz_hi + @p0.xy, @p0.z, to_i(@p1.x), to_u(@p1.y)) * @p0.w;"),
    },
    OpDef {
        name: "WOP_HEIGHT_OFFSET",
        kind: WOP_HEIGHT_OFFSET,
        height: true,
        ins: &[("height", Value::Height)],
        outs: &[("height", Value::Height)],
        body: "h += @p0.x;",
        range: Some("\
            h = h + @p0.x;"),
    },
    OpDef {
        name: "WOP_HEIGHT_STEP",
        kind: WOP_HEIGHT_STEP,
        height: true,
        ins: &[("height", Value::Height)],
        outs: &[("height", Value::Height)],
        body: "h += @p0.z * smoothstep(@p0.x, @p0.y, h);",
        range: Some("\
            // smoothstep is in [0, 1], but which part of it depends on
            // the height so far: below the ramp the step adds nothing, and
            // above it the step adds all of itself. Only a height that
            // straddles the ramp is uncertain.
            if h.hi <= @p0.x {
            } else if h.lo >= @p0.y {
                h = h + @p0.z;
            } else {
                h = h + Interval::new(0.0, @p0.z);
            }"),
    },
    OpDef {
        name: "WOP_WARP_XZ",
        kind: WOP_WARP_XZ,
        height: true,
        ins: &[("warp", Value::Warp)],
        outs: &[("warp", Value::Warp)],
        body: "\
let q = pxz + @p0.zw;
let oct = to_i(@p1.x);
warp.x += @FBM(q, @p0.x, oct, @VS@0) * @p0.y;
warp.y += @FBM(q + v2(713.0, -337.0), @p0.x, oct, @VS@0) * @p0.y;",
        range: Some("\
            // The warp is smooth noise, not an arbitrary displacement, so
            // bound what it can be OVER THIS BOX rather than by its
            // amplitude. The amplitude is the whole world's warp: using it
            // turned a 3 m chunk into a 40 m one, and a fine chunk sitting
            // well clear of the ground could not be decided at all.
            let wlo = pxz_lo + @p0.zw;
            let whi = pxz_hi + @p0.zw;
            let off = Vec2::new(713.0, -337.0);
            let wx = frange(wlo, whi, @p0.x, to_i(@p1.x), 0) * @p0.y;
            let wy = frange(wlo + off, whi + off, @p0.x, to_i(@p1.x), 0) * @p0.y;
            pxz_lo += Vec2::new(wx.lo, wy.lo);
            pxz_hi += Vec2::new(wx.hi, wy.hi);"),
    },
    OpDef {
        name: "WOP_FBM3",
        kind: WOP_FBM3,
        height: false,
        ins: &[("sdf", Value::Sdf)],
        outs: &[("sdf", Value::Sdf)],
        body: "\
let q = p + @p1.xyz;
let n = fbm3(q, @p0.x, @p0.y, to_i(@p2.x), vs);
let sd = (@p0.z - n) * @p0.w;
if @p1.w < 0.5 {
    if sd < d { d = sd; mat = @mat; }
} else {
    d = max(d, -sd);
}",
        range: None,
    },
    OpDef {
        name: "WOP_HEIGHT_SURFACE",
        kind: WOP_HEIGHT_SURFACE,
        height: false,
        ins: &[("height", Value::Height), ("sdf", Value::Sdf)],
        outs: &[("sdf", Value::Sdf)],
        body: "\
let nd = p.y - h;
if nd < d { d = nd; mat = @mat; }",
        range: Some("\
            // Solid below the height, air above it.
            d = d.min(py - h);"),
    },
    OpDef {
        name: "WOP_REGION_AXES",
        kind: WOP_REGION_AXES,
        height: true,
        ins: &[],
        outs: &[("regions", Value::Regions)],
        body: "\
ta = @FBM(pxz + @p0.xy, @p0.z, to_i(@p1.z), @VS@0) + 0.5;
tb = @FBM(pxz + @p1.xy, @p0.w, to_i(@p1.z), @VS@0) + 0.5;",
        range: Some("\
            // Both axes over the box, so a later band can tell whether
            // this box is inside its region, outside it, or straddling.
            ta = frange(pxz_lo + @p0.xy, pxz_hi + @p0.xy, @p0.z, to_i(@p1.z), 0) + 0.5;
            tb = frange(pxz_lo + @p1.xy, pxz_hi + @p1.xy, @p0.w, to_i(@p1.z), 0) + 0.5;"),
    },
    OpDef {
        name: "WOP_HEIGHT_BAND_FBM",
        kind: WOP_HEIGHT_BAND_FBM,
        height: true,
        ins: &[("height", Value::Height), ("warp", Value::Warp), ("regions", Value::Regions)],
        outs: &[("height", Value::Height)],
        body: "\
let fa = @p1.z;
let wa = smoothstep(@p2.x - fa, @p2.x + fa, ta) * (1.0 - smoothstep(@p2.y - fa, @p2.y + fa, ta));
let wb = smoothstep(@p2.z - fa, @p2.z + fa, tb) * (1.0 - smoothstep(@p2.w - fa, @p2.w + fa, tb));
h += min(wa, wb) * (@p1.w + @FBM(pxz + warp + @p0.xy, @p0.z, to_i(@p1.x), @VS@to_u(@p1.y)) * @p0.w);",
        range: Some("\
            // A box entirely outside the region contributes NOTHING, and
            // saying so is the whole point: otherwise the tallest region
            // in a level widens the bound of every box in it.
            let f = crate::program::band_feather([@p2.x, @p2.y]);
            let g = crate::program::band_feather([@p2.z, @p2.w]);
            let touches = ta.hi >= @p2.x - f && ta.lo <= @p2.y + f
                && tb.hi >= @p2.z - g && tb.lo <= @p2.w + g;
            if touches {
                let band = frange(pxz_lo + @p0.xy, pxz_hi + @p0.xy, @p0.z, to_i(@p1.x), to_u(@p1.y)) * @p0.w
                    + @p1.w;
                h = h + Interval::new(band.lo.min(0.0), band.hi.max(0.0));
            }"),
    },
    OpDef {
        name: "WOP_MATERIAL_BAND",
        kind: WOP_MATERIAL_BAND,
        height: false,
        ins: &[("sdf", Value::Sdf), ("regions", Value::Regions)],
        outs: &[("sdf", Value::Sdf)],
        body: "\
if mat == to_u(@p1.z) && ta >= @p0.x && ta < @p0.y && tb >= @p0.z && tb < @p0.w { mat = @mat; }",
        // Repaints only; the SDF and the height are untouched, so a box
        // containing one of these is bounded exactly as it would be
        // without it.
        range: Some("// material only"),
    },
    OpDef {
        name: "WOP_COARSE_SOLID",
        kind: WOP_COARSE_SOLID,
        height: false,
        ins: &[("sdf", Value::Sdf)],
        outs: &[("sdf", Value::Sdf)],
        body: "if SOLID < d { d = SOLID; mat = @mat; }",
        range: None,
    },
    OpDef {
        name: "WOP_LATTICE_Y",
        kind: WOP_LATTICE_Y,
        height: false,
        ins: &[],
        outs: &[("lattice", Value::Lattice)],
        body: "\
level = round_half_up(p.y / @p0.x);
fy = p.y - level * @p0.x;",
        range: None,
    },
    OpDef {
        name: "WOP_SLABS_Y",
        kind: WOP_SLABS_Y,
        height: false,
        ins: &[("sdf", Value::Sdf), ("lattice", Value::Lattice)],
        outs: &[("sdf", Value::Sdf)],
        body: "\
let nd = abs(fy) - @p0.x;
if nd < d { d = nd; mat = @mat; }",
        range: None,
    },
    OpDef {
        name: "WOP_GRID_HOLES",
        kind: WOP_GRID_HOLES,
        height: false,
        ins: &[("sdf", Value::Sdf), ("lattice", Value::Lattice)],
        outs: &[("sdf", Value::Sdf)],
        body: "\
let cell = @p0.x;
let c = to_iv2(floor2(pxz / cell));
if hash3(iv3(c.x, to_i(level), c.y)) < @p0.y {
    let oc = (to_v2(c) + 0.5) * cell;
    let cut = sd_box(v3(p.x - oc.x, fy, p.z - oc.y), @p1.xyz);
    d = max(d, -cut);
}",
        range: None,
    },
    OpDef {
        name: "WOP_PILLARS_XZ",
        kind: WOP_PILLARS_XZ,
        height: false,
        ins: &[("sdf", Value::Sdf)],
        outs: &[("sdf", Value::Sdf)],
        body: "\
let sp = @p0.x;
let c = iv2(to_i(round_half_up(pxz.x / sp)), to_i(round_half_up(pxz.y / sp)));
let jit = v2(hash2(c) - 0.5, hash2(c + iv2(311, 77)) - 0.5) * @p0.y;
let q = pxz - to_v2(c) * sp - jit;
let girth = @p0.z + hash2(c + iv2(9, -4)) * @p0.w;
let nd = max(abs(q.x), abs(q.y)) - girth;
if nd < d { d = nd; mat = @mat; }",
        range: None,
    },
    OpDef {
        name: "WOP_WALLS",
        kind: WOP_WALLS,
        height: false,
        ins: &[("sdf", Value::Sdf), ("lattice", Value::Lattice)],
        outs: &[("sdf", Value::Sdf)],
        body: "\
let sp = @p0.x;
var a = p.x;
var b = p.z;
if @p0.w > 0.5 { a = p.z; b = p.x; }
let wi = round_half_up(a / sp);
let w = a - wi * sp;
if hash2(iv2(to_i(wi) + to_i(@p1.x), to_i(level))) < @p0.z {
    var wall = abs(w) - @p0.y;
    let dc = @p1.y;
    let ci = round_half_up(b / dc);
    let cl = b - ci * dc;
    if hash3(iv3(to_i(wi), to_i(ci), to_i(level) + to_i(@p1.w))) < @p1.z {
        let doorway = sd_box(v3(w, fy + @p2.w, cl), @p2.xyz);
        wall = max(wall, -doorway);
    }
    if wall < d { d = wall; mat = @mat; }
}",
        range: None,
    },
    OpDef {
        name: "WOP_SHAFTS_XZ",
        kind: WOP_SHAFTS_XZ,
        height: false,
        ins: &[],
        outs: &[("shafts", Value::Shafts)],
        body: "\
let sp = @p0.x;
let c = iv2(to_i(round_half_up(pxz.x / sp)), to_i(round_half_up(pxz.y / sp)));
let jit = v2(hash2(c + iv2(41, 13)) - 0.5, hash2(c + iv2(-7, 99)) - 0.5) * @p0.y;
sxz = pxz - to_v2(c) * sp - jit;
sr = @p0.z + hash2(c) * @p0.w;
shaft = length(sxz) - sr;",
        range: None,
    },
    OpDef {
        name: "WOP_SHAFTS_CUT",
        kind: WOP_SHAFTS_CUT,
        height: false,
        ins: &[("sdf", Value::Sdf), ("shafts", Value::Shafts)],
        outs: &[("sdf", Value::Sdf)],
        body: "d = max(d, -shaft);",
        range: None,
    },
    OpDef {
        name: "WOP_BEAMS",
        kind: WOP_BEAMS,
        height: false,
        ins: &[("sdf", Value::Sdf), ("lattice", Value::Lattice), ("shafts", Value::Shafts)],
        outs: &[("sdf", Value::Sdf)],
        body: "\
let n = @p0.x;
if abs(level - round_half_up(level / n) * n) < 0.5 {
    let beam = max(max(abs(sxz.y) - @p0.y, abs(fy + @p0.z) - @p0.w), length(sxz) - (sr + @p1.x));
    if beam < d { d = beam; mat = @mat; }
}",
        range: None,
    },
];

/// Emission context: the full interpreter or the height-only replay.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Ctx {
    Full,
    Height,
    /// Everything the height replay does NOT do.
    ///
    /// The height chain depends only on xz, so a volumetric evaluator can
    /// run it once per COLUMN and this per sample. The split is exactly
    /// `OpDef::height`, and it is sound because no `height: false` arm
    /// reads a register the height arms own except `h`, `ta` and `tb`,
    /// which the column pass hands over.
    Sample,
}

fn substitute(body: &str, ctx: Ctx, wgsl: bool) -> String {
    let mut s = body.to_string();
    // FBM entry point + optional vs argument.
    let fbm = match (ctx, wgsl) {
        (Ctx::Full | Ctx::Sample, false) => "fbm_mode",
        (Ctx::Full | Ctx::Sample, true) => "fbm",
        (Ctx::Height, _) => "hfbm",
    };
    s = s.replace("@FBM", fbm);
    s = s.replace("@VS@", if ctx == Ctx::Height { "" } else { "vs, " });
    s = s.replace("@mat", if wgsl { "op.head.z" } else { "op.material" });
    // Param tokens, longest suffix first.
    for reg in ["p0", "p1", "p2"] {
        for (suffix, idx) in [("xyz", 0usize), ("xy", 0), ("zw", 2)] {
            let token = format!("@{reg}.{suffix}");
            let repl = if wgsl {
                format!("op.{reg}.{suffix}")
            } else {
                match suffix {
                    "xyz" => format!("Vec3::new(op.{reg}[0], op.{reg}[1], op.{reg}[2])"),
                    "xy" => format!("Vec2::new(op.{reg}[{}], op.{reg}[{}])", idx, idx + 1),
                    "zw" => format!("Vec2::new(op.{reg}[{}], op.{reg}[{}])", idx, idx + 1),
                    _ => unreachable!(),
                }
            };
            s = s.replace(&token, &repl);
        }
        for (i, comp) in ["x", "y", "z", "w"].iter().enumerate() {
            let token = format!("@{reg}.{comp}");
            let repl = if wgsl {
                format!("op.{reg}.{comp}")
            } else {
                format!("op.{reg}[{i}]")
            };
            s = s.replace(&token, &repl);
        }
    }
    if !wgsl {
        s = s.replace("var ", "let mut ");
    }
    assert!(!s.contains('@'), "unresolved dialect token in: {s}");
    s
}

fn indent(text: &str, by: &str) -> String {
    text.lines()
        .map(|l| {
            if l.is_empty() {
                String::new()
            } else {
                format!("{by}{l}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn ops_for(ctx: Ctx) -> impl Iterator<Item = &'static OpDef> {
    OPS.iter().filter(move |op| match ctx {
        Ctx::Full => true,
        Ctx::Height => op.height,
        Ctx::Sample => !op.height,
    })
}

/// The Rust `match op.kind { ... }` statement for `include!`.
pub fn rust_arms(ctx: Ctx) -> String {
    let mut out = String::from("match op.kind {\n");
    for op in ops_for(ctx) {
        out.push_str(&format!(
            "    {} => {{\n{}\n    }}\n",
            op.name,
            indent(&substitute(op.body, ctx, false), "        ")
        ));
    }
    out.push_str("    _ => {}\n}\n");
    out
}

/// The Rust `match op.kind { ... }` for the RANGE interpreter.
///
/// Three outcomes, and the difference between them is the whole safety
/// argument:
///
/// - an op with a rule bounds the registers;
/// - an op in this table WITHOUT one returns `None`, because it can put
///   solid somewhere this analysis cannot see;
/// - an op that is not in this table at all is not part of the SDF —
///   `eval` ignores it too, by the same `_ => {}`.
pub fn rust_range_arms() -> String {
    let mut out = String::from("match op.kind {\n");
    for op in OPS {
        match op.range {
            Some(rule) => out.push_str(&format!(
                "    {} => {{\n{}\n    }}\n",
                op.name,
                indent(&substitute(rule, Ctx::Full, false), "        ")
            )),
            None => out.push_str(&format!(
                "    {} => return None, // unbounded: no rule yet\n",
                op.name
            )),
        }
    }
    out.push_str("    _ => {} // not an SDF op\n}\n");
    out
}

/// The WGSL `case Nu: { ... }` list to splice inside a `switch` (the
/// shells keep the loop, flag gating, and `default {}`).
pub fn wgsl_arms(ctx: Ctx) -> String {
    let mut out = String::new();
    for op in ops_for(ctx) {
        out.push_str(&format!(
            "case {}u: {{ // {}\n{}\n}}\n",
            op.kind,
            op.name,
            indent(&substitute(op.body, ctx, true), "    ")
        ));
    }
    out
}

/// Module-level WGSL helper shims the generated arms call. `Height`
/// contexts additionally need the vs-less `hfbm` wrapper, whose body
/// differs per shader (coarse_fbm), so shells define that one.
/// The height arms in the FULL dialect, for a volumetric evaluator that
/// runs the height chain once per column. Same ops as `Ctx::Height`, but
/// the vs-aware FBM the density shader has rather than the replay shells'.
pub fn wgsl_column_arms() -> String {
    let mut out = String::new();
    for op in OPS.iter().filter(|op| op.height) {
        out.push_str(&format!(
            "case {}u: {{ // {}\n{}\n}}\n",
            op.kind,
            op.name,
            indent(&substitute(op.body, Ctx::Full, true), "    ")
        ));
    }
    out
}

pub fn wgsl_helpers(ctx: Ctx) -> String {
    let mut out = String::from(
        "\
fn v2(x: f32, y: f32) -> vec2<f32> { return vec2<f32>(x, y); }
fn to_i(x: f32) -> i32 { return i32(x); }
fn to_u(x: f32) -> u32 { return u32(x); }
// Region gate (WorldOp::region). Emitted here rather than written into
// each shader so the three files cannot drift: every interpreter loop
// calls this before its switch, and 0 means the op is ungated.
//
// An integer form that compared the axes quantized to bytes, skipping
// the unpack, measured no faster (megastructure settle 2.12 -> 2.30 s,
// inside the run-to-run noise): what a gated-out op costs is the loop
// iteration and the 64-byte read, not the arithmetic. Left in the
// obvious form.
fn region_gate(packed: u32, ta: f32, tb: f32) -> bool {
    if packed == 0u { return true; }
    let a0 = f32(packed & 0xFFu) / 255.0;
    let a1 = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b0 = f32((packed >> 16u) & 0xFFu) / 255.0;
    let b1 = f32((packed >> 24u) & 0xFFu) / 255.0;
    return ta >= a0 && ta < a1 && tb >= b0 && tb < b1;
}
",
    );
    if ctx != Ctx::Height {
        out.push_str(
            "\
fn v3(x: f32, y: f32, z: f32) -> vec3<f32> { return vec3<f32>(x, y, z); }
fn iv2(x: i32, y: i32) -> vec2<i32> { return vec2<i32>(x, y); }
fn iv3(x: i32, y: i32, z: i32) -> vec3<i32> { return vec3<i32>(x, y, z); }
fn to_v2(v: vec2<i32>) -> vec2<f32> { return vec2<f32>(v); }
fn to_iv2(v: vec2<f32>) -> vec2<i32> { return vec2<i32>(v); }
fn floor2(v: vec2<f32>) -> vec2<f32> { return floor(v); }
",
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitters_cover_all_ops_and_resolve_all_tokens() {
        // substitute() asserts no '@' survives; force it on every op in
        // every context.
        let full_rust = rust_arms(Ctx::Full);
        let full_wgsl = wgsl_arms(Ctx::Full);
        let height_rust = rust_arms(Ctx::Height);
        let height_wgsl = wgsl_arms(Ctx::Height);
        for op in OPS {
            assert!(full_rust.contains(op.name), "{} missing (rust)", op.name);
            assert!(
                full_wgsl.contains(&format!("case {}u:", op.kind)),
                "{} missing (wgsl)",
                op.name
            );
            assert_eq!(
                height_rust.contains(op.name),
                op.height,
                "{} height selection (rust)",
                op.name
            );
            assert_eq!(
                height_wgsl.contains(&format!("case {}u:", op.kind)),
                op.height,
                "{} height selection (wgsl)",
                op.name
            );
        }
        // Height arms never reference vs (their FBM wrapper owns it).
        assert!(!height_rust.contains("vs,"));
        assert!(!height_wgsl.contains("vs,"));
        // Kind constants in the table match the worldop consts.
        for op in OPS {
            assert!(
                full_wgsl.contains(&format!("case {}u: {{ // {}", op.kind, op.name)),
                "kind/name mismatch for {}",
                op.name
            );
        }
    }
}

#[cfg(test)]
mod port_tests {
    use super::*;

    /// Does `body` use `ident` as a whole word?
    ///
    /// `h` must not match `hash`, and `d` must not match `sd_box` — the
    /// registers are one and two letters long, so anything short of a real
    /// token scan reports every op as touching everything.
    fn touches(body: &str, ident: &str) -> bool {
        let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
        // `@mat` is a read of the op's own material PARAM, not the `mat`
        // register; it appears in ops that never read the register.
        let body = body.replace("@mat", "");
        body.split(|c: char| !is_word(c))
            .any(|token| token == ident)
    }

    /// An op body may touch no register its ports do not carry.
    ///
    /// This is what makes the ports a real declaration rather than a
    /// comment: the wiring a level sees and the registers the interpreter
    /// moves are the same set, checked, so an op that quietly starts
    /// reading the lattice cannot also quietly stop being wired to it.
    #[test]
    fn an_op_touches_only_the_registers_its_ports_carry() {
        for op in OPS {
            let declared: Vec<&str> = op
                .ins
                .iter()
                .chain(op.outs)
                .flat_map(|(_, v)| v.registers().iter().copied())
                .collect();
            for value in Value::ALL {
                for reg in value.registers() {
                    if touches(op.body, reg) && !declared.contains(reg) {
                        panic!(
                            "{} touches `{reg}` but no port of it carries {value:?} \
                             (declares {declared:?})",
                            op.name
                        );
                    }
                }
            }
        }
    }

    /// And every port an op declares is one the body actually uses.
    ///
    /// The other direction, because a spurious port is a dependency a
    /// level would be made to wire for nothing.
    #[test]
    fn an_op_declares_no_port_it_does_not_use() {
        for op in OPS {
            for (port, value) in op.ins.iter().chain(op.outs) {
                let used = value.registers().iter().any(|r| touches(op.body, r));
                assert!(
                    used,
                    "{} declares port `{port}` carrying {value:?}, which its body never touches",
                    op.name
                );
            }
        }
    }

    /// Every kind in the table is reachable through one lookup, meta ops
    /// included.
    #[test]
    fn ports_are_found_for_table_ops_and_meta_ops_alike() {
        for op in OPS {
            assert!(ports(op.kind).is_some(), "{}", op.name);
        }
        assert!(ports(WOP_FIELD).is_some(), "WOP_FIELD is CPU-only, not absent");
    }
}
