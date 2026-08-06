//! Splice the generated WGSL interpreter arms/helpers (from
//! voxel-core::opgen) into the marked regions of the three shaders.
//! Run from the workspace root: `mise run genops`. A guard test in
//! voxel-render fails when the spliced text goes stale.

use voxel_core::opgen::{wgsl_arms, wgsl_helpers, Ctx};

const HELPERS_BEGIN: &str = "// GENOPS HELPERS BEGIN";
const HELPERS_END: &str = "// GENOPS HELPERS END";
const ARMS_BEGIN: &str = "// GENOPS ARMS BEGIN";
const ARMS_END: &str = "// GENOPS ARMS END";

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
    format!("{}{}{}", &text[..b_line_end], indented, &text[e_line_start..])
}

fn main() {
    let targets = [
        (
            "crates/voxel-render/src/shaders/voxel_world_density.wgsl",
            Ctx::Full,
            "            ",
        ),
        (
            "crates/voxel-render/src/shaders/voxel_mesh_chunks.wgsl",
            Ctx::Height,
            "            ",
        ),
        (
            "demos/voxel2/src/voxel_water.wgsl",
            Ctx::Height,
            "            ",
        ),
    ];
    for (path, ctx, arm_indent) in targets {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let text = splice(&text, HELPERS_BEGIN, HELPERS_END, &wgsl_helpers(ctx), "");
        let text = splice(&text, ARMS_BEGIN, ARMS_END, &wgsl_arms(ctx), arm_indent);
        std::fs::write(path, text).unwrap();
        println!("spliced {path}");
    }
}
