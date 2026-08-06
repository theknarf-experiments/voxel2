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

/// One generator op's single-source definition.
pub struct OpDef {
    /// Rust const name (must exist in `voxel_core::worldop`).
    pub name: &'static str,
    pub kind: u32,
    /// Part of the height-only replay (shadow bake, seabed, eval_height).
    pub height: bool,
    /// Body in the shared dialect.
    pub body: &'static str,
}

pub const OPS: &[OpDef] = &[
    OpDef {
        name: "WOP_HEIGHT_FBM",
        kind: WOP_HEIGHT_FBM,
        height: true,
        body: "\
h += @FBM(pxz + warp + @p0.xy, @p0.z, to_i(@p1.x), @VS@to_u(@p1.y)) * @p0.w;",
    },
    OpDef {
        name: "WOP_HEIGHT_OFFSET",
        kind: WOP_HEIGHT_OFFSET,
        height: true,
        body: "h += @p0.x;",
    },
    OpDef {
        name: "WOP_HEIGHT_STEP",
        kind: WOP_HEIGHT_STEP,
        height: true,
        body: "h += @p0.z * smoothstep(@p0.x, @p0.y, h);",
    },
    OpDef {
        name: "WOP_WARP_XZ",
        kind: WOP_WARP_XZ,
        height: true,
        body: "\
let q = pxz + @p0.zw;
let oct = to_i(@p1.x);
warp.x += @FBM(q, @p0.x, oct, @VS@0) * @p0.y;
warp.y += @FBM(q + v2(713.0, -337.0), @p0.x, oct, @VS@0) * @p0.y;",
    },
    OpDef {
        name: "WOP_FBM3",
        kind: WOP_FBM3,
        height: false,
        body: "\
let q = p + @p1.xyz;
let n = fbm3(q, @p0.x, @p0.y, to_i(@p2.x), vs);
let sd = (@p0.z - n) * @p0.w;
if @p1.w < 0.5 {
    if sd < d { d = sd; mat = @mat; }
} else {
    d = max(d, -sd);
}",
    },
    OpDef {
        name: "WOP_HEIGHT_SURFACE",
        kind: WOP_HEIGHT_SURFACE,
        height: false,
        body: "\
let nd = p.y - h;
if nd < d { d = nd; mat = @mat; }",
    },
    OpDef {
        name: "WOP_COARSE_SOLID",
        kind: WOP_COARSE_SOLID,
        height: false,
        body: "if SOLID < d { d = SOLID; mat = @mat; }",
    },
    OpDef {
        name: "WOP_LATTICE_Y",
        kind: WOP_LATTICE_Y,
        height: false,
        body: "\
level = round_half_up(p.y / @p0.x);
fy = p.y - level * @p0.x;",
    },
    OpDef {
        name: "WOP_SLABS_Y",
        kind: WOP_SLABS_Y,
        height: false,
        body: "\
let nd = abs(fy) - @p0.x;
if nd < d { d = nd; mat = @mat; }",
    },
    OpDef {
        name: "WOP_GRID_HOLES",
        kind: WOP_GRID_HOLES,
        height: false,
        body: "\
let cell = @p0.x;
let c = to_iv2(floor2(pxz / cell));
if hash3(iv3(c.x, to_i(level), c.y)) < @p0.y {
    let oc = (to_v2(c) + 0.5) * cell;
    let cut = sd_box(v3(p.x - oc.x, fy, p.z - oc.y), @p1.xyz);
    d = max(d, -cut);
}",
    },
    OpDef {
        name: "WOP_PILLARS_XZ",
        kind: WOP_PILLARS_XZ,
        height: false,
        body: "\
let sp = @p0.x;
let c = iv2(to_i(round_half_up(pxz.x / sp)), to_i(round_half_up(pxz.y / sp)));
let jit = v2(hash2(c) - 0.5, hash2(c + iv2(311, 77)) - 0.5) * @p0.y;
let q = pxz - to_v2(c) * sp - jit;
let girth = @p0.z + hash2(c + iv2(9, -4)) * @p0.w;
let nd = max(abs(q.x), abs(q.y)) - girth;
if nd < d { d = nd; mat = @mat; }",
    },
    OpDef {
        name: "WOP_WALLS",
        kind: WOP_WALLS,
        height: false,
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
    },
    OpDef {
        name: "WOP_SHAFTS_XZ",
        kind: WOP_SHAFTS_XZ,
        height: false,
        body: "\
let sp = @p0.x;
let c = iv2(to_i(round_half_up(pxz.x / sp)), to_i(round_half_up(pxz.y / sp)));
let jit = v2(hash2(c + iv2(41, 13)) - 0.5, hash2(c + iv2(-7, 99)) - 0.5) * @p0.y;
sxz = pxz - to_v2(c) * sp - jit;
sr = @p0.z + hash2(c) * @p0.w;
shaft = length(sxz) - sr;",
    },
    OpDef {
        name: "WOP_SHAFTS_CUT",
        kind: WOP_SHAFTS_CUT,
        height: false,
        body: "d = max(d, -shaft);",
    },
    OpDef {
        name: "WOP_BEAMS",
        kind: WOP_BEAMS,
        height: false,
        body: "\
let n = @p0.x;
if abs(level - round_half_up(level / n) * n) < 0.5 {
    let beam = max(max(abs(sxz.y) - @p0.y, abs(fy + @p0.z) - @p0.w), length(sxz) - (sr + @p1.x));
    if beam < d { d = beam; mat = @mat; }
}",
    },
];

/// Emission context: the full interpreter or the height-only replay.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Ctx {
    Full,
    Height,
}

fn substitute(body: &str, ctx: Ctx, wgsl: bool) -> String {
    let mut s = body.to_string();
    // FBM entry point + optional vs argument.
    let fbm = match (ctx, wgsl) {
        (Ctx::Full, false) => "fbm_mode",
        (Ctx::Full, true) => "fbm",
        (Ctx::Height, _) => "hfbm",
    };
    s = s.replace("@FBM", fbm);
    s = s.replace("@VS@", if ctx == Ctx::Full { "vs, " } else { "" });
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
    OPS.iter().filter(move |op| ctx == Ctx::Full || op.height)
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
pub fn wgsl_helpers(ctx: Ctx) -> String {
    let mut out = String::from(
        "\
fn v2(x: f32, y: f32) -> vec2<f32> { return vec2<f32>(x, y); }
fn to_i(x: f32) -> i32 { return i32(x); }
fn to_u(x: f32) -> u32 { return u32(x); }
",
    );
    if ctx == Ctx::Full {
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
