//! WGSL is compiled when a PIPELINE is built, not when the crate is, so
//! `cargo build` says nothing about whether a shader is even parseable.
//! A deleted struct or a renamed field costs a full run to discover, and
//! presents as geometry silently missing rather than as an error — which
//! is exactly how `struct MatSample` went missing once.
//!
//! These shaders are import-free, so naga can take them whole. The ones
//! that `#import bevy_pbr::...` need naga_oil and Bevy's own module
//! sources to compose first; they are not covered here.

const STANDALONE: &[&str] = &[
    "src/shaders/voxel_world_density.wgsl",
    "src/shaders/voxel_mesh_chunks.wgsl",
];

/// Parse and TYPE-CHECK, not just parse: an undefined identifier, a field
/// that moved, or a `vec4` indexed past its end are all validation
/// errors rather than parse errors, and those are the mistakes a layout
/// twin actually makes.
#[test]
fn standalone_shaders_are_valid_wgsl() {
    for path in STANDALONE {
        let full = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), path);
        let src = std::fs::read_to_string(&full)
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        assert!(
            !src.contains("#import"),
            "{path} gained an #import — it can no longer be validated standalone, \
             so either move it out of STANDALONE or compose it with naga_oil"
        );
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("{path} does not parse:\n{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{path} is not valid WGSL:\n{e:?}"));
    }
}
