//! Guards for the generated WGSL interpreter arms (voxel-core::opgen):
//! the spliced shader regions must match the op table exactly, and the
//! generated arms must be valid, well-typed WGSL. Run `mise run genops`
//! after editing the op table to refresh the shaders.

use voxel_core::layout::{wgsl_material_accessors, wgsl_struct, wgsl_texel_index, CHUNK_PARAMS};
use voxel_core::opgen::{wgsl_arms, wgsl_column_arms, wgsl_helpers, Ctx};

/// (path, helper dialect, arms ctx, has a separate column block).
///
/// The density shader splits the program: the height chain runs once per
/// COLUMN and the rest per sample, so its arms are `Ctx::Sample` and its
/// column block is the height ops in the full dialect.
const SHADERS: &[(&str, Ctx, Ctx, bool)] = &[
    (
        "src/shaders/voxel_world_density.wgsl",
        Ctx::Full,
        Ctx::Sample,
        true,
    ),
    (
        "src/shaders/voxel_mesh_chunks.wgsl",
        Ctx::Height,
        Ctx::Height,
        false,
    ),
    (
        "../../demos/voxel2/src/voxel_water.wgsl",
        Ctx::Height,
        Ctx::Height,
        false,
    ),
];

fn region<'a>(text: &'a str, begin: &str, end: &str) -> &'a str {
    let b = text.find(begin).expect("begin marker");
    let b = text[b..].find('\n').unwrap() + b + 1;
    let e = text.find(end).expect("end marker");
    let e = text[..e].rfind('\n').unwrap() + 1;
    &text[b..e]
}

fn indented(content: &str, indent: &str) -> String {
    content
        .lines()
        .map(|l| {
            if l.is_empty() {
                String::from("\n")
            } else {
                format!("{indent}{l}\n")
            }
        })
        .collect()
}

#[test]
fn spliced_shader_regions_match_the_op_table() {
    for (path, helpers, arms, column) in SHADERS {
        let full = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), path);
        let text = std::fs::read_to_string(&full).unwrap();
        assert_eq!(
            region(&text, "// GENOPS HELPERS BEGIN", "// GENOPS HELPERS END"),
            indented(&wgsl_helpers(*helpers), ""),
            "{path}: helpers stale — run `mise run genops`"
        );
        assert_eq!(
            region(&text, "// GENOPS ARMS BEGIN", "// GENOPS ARMS END"),
            indented(&wgsl_arms(*arms), "            "),
            "{path}: arms stale — run `mise run genops`"
        );
        if *column {
            assert_eq!(
                region(
                    &text,
                    "// GENOPS COLUMN ARMS BEGIN",
                    "// GENOPS COLUMN ARMS END"
                ),
                indented(&wgsl_column_arms(), "            "),
                "{path}: column arms stale — run `mise run genops`"
            );
        }
    }
}

/// Every op is in exactly one of the density shader's two passes, or a
/// program would silently skip it.
#[test]
fn the_two_passes_partition_the_op_table() {
    let height = wgsl_column_arms();
    let sample = wgsl_arms(Ctx::Sample);
    let full = wgsl_arms(Ctx::Full);
    for line in full.lines().filter(|l| l.starts_with("case ")) {
        let in_height = height.lines().any(|h| h == line);
        let in_sample = sample.lines().any(|s| s == line);
        assert!(
            in_height != in_sample,
            "{line} is in {} passes, must be exactly one",
            u8::from(in_height) + u8::from(in_sample)
        );
    }
}

/// The generated arms compile as WGSL: a minimal standalone module with
/// the register shell and stub twins stands in for the real shaders
/// (which need naga_oil imports).
#[test]
fn generated_arms_are_valid_wgsl() {
    let harness = |ctx: Ctx| -> String {
        let mut s = String::from(
            "\
struct WorldOp { head: vec4<u32>, p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32> }
struct WorldProgram { count: vec4<u32>, sun: vec4<f32>, anchor: vec4<f32>, field: vec4<f32>, ops: array<WorldOp, 8> }
var<private> prog: WorldProgram;
const BIG: f32 = 1.0e6;
const SOLID: f32 = -1.0e5;
fn hash2(p: vec2<i32>) -> f32 { return f32(p.x + p.y) * 0.001; }
fn hash3(p: vec3<i32>) -> f32 { return f32(p.x + p.y + p.z) * 0.001; }
fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 { return length(max(abs(p) - b, vec3<f32>(0.0))); }
fn fbm(q: vec2<f32>, s: f32, o: i32, vs: f32, m: u32) -> f32 { return q.x + s + f32(o) + vs + f32(m); }
fn fbm3(q: vec3<f32>, a: f32, b: f32, o: i32, vs: f32) -> f32 { return q.x + a + b + f32(o) + vs; }
fn hfbm(q: vec2<f32>, s: f32, o: i32, m: u32) -> f32 { return q.x + s + f32(o) + f32(m); }
fn round_half_up(x: f32) -> f32 { return floor(x + 0.5); }
",
        );
        s.push_str(&wgsl_helpers(ctx));
        s.push_str(
            "\
fn interpret(p: vec3<f32>, vs: f32) -> f32 {
    var h = 0.0;
    var d = BIG;
    var mat = 1u;
    var level = 0.0;
    var fy = p.y;
    var sxz = vec2<f32>(0.0);
    var sr = 0.0;
    var shaft = BIG;
    var warp = vec2<f32>(0.0);
    var ta = 0.0;
    var tb = 0.0;
    let pxz = p.xz;
    for (var i = 0u; i < prog.count.x; i++) {
        let op = prog.ops[i];
        switch op.head.x {
",
        );
        s.push_str(&wgsl_arms(ctx));
        s.push_str(
            "\
            default {}
        }
    }
    return d + h + f32(mat) + level + fy + sxz.x + sr + shaft + warp.x;
}
",
        );
        s
    };
    for ctx in [Ctx::Full, Ctx::Height] {
        let src = harness(ctx);
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("generated WGSL does not parse: {e}\n{src}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("generated WGSL does not validate: {e:?}"));
    }
}

/// The GPU struct layouts (voxel-core::layout) spliced into the shaders
/// that carry their markers.
///
/// Separate from the op-table guard because these are twins of a struct
/// rather than of an interpreter: the per-chunk uniform, and the named
/// accessors that keep a material parameter's slot out of the shader.
#[test]
fn spliced_layout_regions_match_the_tables() {
    let at = |path: &str| {
        std::fs::read_to_string(format!("{}/{}", env!("CARGO_MANIFEST_DIR"), path)).unwrap()
    };
    for path in [
        "src/shaders/voxel_world_density.wgsl",
        "src/shaders/voxel_mesh_chunks.wgsl",
    ] {
        assert_eq!(
            region(
                &at(path),
                "// GENMAT CHUNKPARAMS BEGIN",
                "// GENMAT CHUNKPARAMS END"
            ),
            wgsl_struct("ChunkParams", CHUNK_PARAMS),
            "{path}: ChunkParams stale — run `mise run genops`"
        );
    }
    assert_eq!(
        region(
            &at("src/shaders/voxel_chunk_draw.wgsl"),
            "// GENMAT ACCESSORS BEGIN",
            "// GENMAT ACCESSORS END"
        ),
        wgsl_material_accessors(),
        "material accessors stale — run `mise run genops`"
    );
    assert_eq!(
        region(
            &at("src/shaders/voxel_chunk_draw.wgsl"),
            "// GENMAT TEXELORDER BEGIN",
            "// GENMAT TEXELORDER END"
        ),
        wgsl_texel_index(),
        "texel order stale — run `mise run genops`"
    );
}
