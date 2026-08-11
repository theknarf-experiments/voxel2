//! Splice the generated WGSL interpreter arms/helpers (from
//! voxel-core::opgen) into the marked regions of the three shaders.
//! Run from the workspace root: `mise run genops`. A guard test in
//! voxel-render fails when the spliced text goes stale.

use voxel_core::layout::{wgsl_material_accessors, wgsl_struct, wgsl_texel_index, CHUNK_PARAMS};
use voxel_core::opgen::{wgsl_arms, wgsl_column_struct, wgsl_helpers, Ctx};

const HELPERS_BEGIN: &str = "// GENOPS HELPERS BEGIN";
const HELPERS_END: &str = "// GENOPS HELPERS END";
const ARMS_BEGIN: &str = "// GENOPS ARMS BEGIN";
const ARMS_END: &str = "// GENOPS ARMS END";
/// The xz-only arms, spliced separately where an evaluator runs them once
/// per column instead of once per sample. Which ops those are is derived
/// (`opgen::Axis`), not listed here.
const COLUMN_BEGIN: &str = "// GENOPS COLUMN ARMS BEGIN";
const COLUMN_END: &str = "// GENOPS COLUMN ARMS END";
/// The handover: what the column pass carries, how it is packed, and how
/// the sample pass reads it back.
const CSTRUCT_BEGIN: &str = "// GENOPS COLUMN STRUCT BEGIN";
const CSTRUCT_END: &str = "// GENOPS COLUMN STRUCT END";
const CRETURN_BEGIN: &str = "// GENOPS COLUMN RETURN BEGIN";
const CRETURN_END: &str = "// GENOPS COLUMN RETURN END";
const CUNPACK_BEGIN: &str = "// GENOPS COLUMN UNPACK BEGIN";
const CUNPACK_END: &str = "// GENOPS COLUMN UNPACK END";
/// GPU struct layouts (voxel-core::layout) rather than op bodies: the
/// per-chunk uniform, and the material recipes' named slot accessors.
const PARAMS_BEGIN: &str = "// GENMAT CHUNKPARAMS BEGIN";
const PARAMS_END: &str = "// GENMAT CHUNKPARAMS END";
const TEXEL_BEGIN: &str = "// GENMAT TEXELORDER BEGIN";
const TEXEL_END: &str = "// GENMAT TEXELORDER END";
const MAT_BEGIN: &str = "// GENMAT ACCESSORS BEGIN";
const MAT_END: &str = "// GENMAT ACCESSORS END";

fn splice(text: &str, begin: &str, end: &str, content: &str, indent: &str) -> String {
    let b = text.find(begin).expect("begin marker");
    let b_line_end = text[b..].find('\n').expect("marker line") + b + 1;
    let e = text.find(end).expect("end marker");
    let e_line_start = text[..e].rfind('\n').expect("end line") + 1;
    let indented: String = content
        .lines()
        .map(|l| {
            if l.is_empty() {
                String::from("\n")
            } else {
                format!("{indent}{l}\n")
            }
        })
        .collect();
    format!(
        "{}{}{}",
        &text[..b_line_end],
        indented,
        &text[e_line_start..]
    )
}

/// Splice only where the marker exists. A shader opts into a region by
/// carrying its markers, so the water shader — which has an interpreter
/// but no per-chunk uniform — needs no entry in a table to say so.
fn splice_if_present(text: &str, begin: &str, end: &str, content: &str, indent: &str) -> String {
    if text.contains(begin) {
        splice(text, begin, end, content, indent)
    } else {
        text.to_string()
    }
}

fn main() {
    // (path, helper dialect, arms ctx, does the file split into two
    // passes).
    let targets = [
        (
            "crates/voxel-render/src/shaders/voxel_world_density.wgsl",
            Ctx::Full,
            Ctx::Sample,
            true,
            "            ",
        ),
        (
            "crates/voxel-render/src/shaders/voxel_mesh_chunks.wgsl",
            Ctx::Height,
            Ctx::Height,
            false,
            "            ",
        ),
        (
            "demos/voxel2/src/voxel_water.wgsl",
            Ctx::Height,
            Ctx::Height,
            false,
            "            ",
        ),
    ];
    for (path, helpers, arms, column, arm_indent) in targets {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let text = splice(
            &text,
            HELPERS_BEGIN,
            HELPERS_END,
            &wgsl_helpers(helpers),
            "",
        );
        let text = splice(&text, ARMS_BEGIN, ARMS_END, &wgsl_arms(arms), arm_indent);
        // The column pass runs inside the volumetric evaluator, so it
        // keeps the full dialect's vs-aware FBM rather than the replay
        // shells'.
        let text = if column {
            let (cstruct, creturn, cunpack) = wgsl_column_struct();
            let text = splice(
                &text,
                COLUMN_BEGIN,
                COLUMN_END,
                &wgsl_arms(Ctx::Column),
                arm_indent,
            );
            let text = splice(&text, CSTRUCT_BEGIN, CSTRUCT_END, &cstruct, "");
            let text = splice(&text, CRETURN_BEGIN, CRETURN_END, &creturn, "    ");
            splice(&text, CUNPACK_BEGIN, CUNPACK_END, &cunpack, "    ")
        } else {
            text
        };
        let text = splice_if_present(
            &text,
            PARAMS_BEGIN,
            PARAMS_END,
            &wgsl_struct("ChunkParams", CHUNK_PARAMS),
            "",
        );
        std::fs::write(path, text).unwrap();
        println!("spliced {path}");
    }
    // The material table is read by the draw shader alone, which has no
    // interpreter in it at all.
    let draw = "crates/voxel-render/src/shaders/voxel_chunk_draw.wgsl";
    let text = std::fs::read_to_string(draw).unwrap_or_else(|e| panic!("{draw}: {e}"));
    let text = splice(&text, TEXEL_BEGIN, TEXEL_END, &wgsl_texel_index(), "");
    let text = splice(&text, MAT_BEGIN, MAT_END, &wgsl_material_accessors(), "");
    std::fs::write(draw, text).unwrap();
    println!("spliced {draw}");
}
