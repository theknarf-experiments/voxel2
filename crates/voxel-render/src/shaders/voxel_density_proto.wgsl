// M2 prototype density pass: fills one chunk's 36³ sample buffer with a
// sphere SDF. Voxel format matches voxel-core: f16 sdf | u8 material | u8
// flags packed in a u32. Sample index i holds the value at cell corner i - 1.

const SAMPLES: u32 = 36u;

@group(0) @binding(0) var<storage, read_write> density: array<u32>;

fn sample_index(p: vec3<u32>) -> u32 {
    return p.x + SAMPLES * (p.y + SAMPLES * p.z);
}

@compute @workgroup_size(6, 6, 6)
fn density_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(SAMPLES))) {
        return;
    }
    // Corner coordinate in voxel units (chunk-local; voxel size 1 m at LOD 0).
    let p = vec3<f32>(vec3<i32>(id) - vec3<i32>(1));
    let center = vec3<f32>(16.0, 16.0, 16.0);
    let sdf = clamp(length(p - center) - 12.0, -4.0, 4.0);
    let material = select(0u, 1u, sdf < 0.0);
    let packed = (pack2x16float(vec2<f32>(sdf, 0.0)) & 0xFFFFu) | (material << 16u);
    density[sample_index(id)] = packed;
}
