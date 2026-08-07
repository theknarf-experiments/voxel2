//! Generate the CPU interpreter arms from the single-source op table
//! (voxel-core::opgen). program.rs `include!`s the output, so the CPU
//! and GPU interpreters cannot drift op-by-op.

use std::io::Write;

fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    let write = |name: &str, text: String| {
        let mut f = std::fs::File::create(format!("{out}/{name}")).unwrap();
        f.write_all(text.as_bytes()).unwrap();
    };
    write(
        "op_arms_full.rs",
        voxel_core::opgen::rust_arms(voxel_core::opgen::Ctx::Full),
    );
    write(
        "op_arms_height.rs",
        voxel_core::opgen::rust_arms(voxel_core::opgen::Ctx::Height),
    );
    write("op_arms_range.rs", voxel_core::opgen::rust_range_arms());
    println!("cargo:rerun-if-changed=build.rs");
}
