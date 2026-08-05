// Scout for ruin sites near the scenic default spawn.
fn main() {
    let mut found = 0;
    for cz in -160..-120 {
        for cx in -120..-90 {
            let min = glam::Vec3::new(cx as f32 * 256.0, -500.0, cz as f32 * 256.0);
            let max = min + glam::Vec3::new(256.0, 1000.0, 256.0);
            let ops = voxel_worldgen::ruins::ruins_ops(min, max);
            if ops.len() >= 8 {
                let c = ops[0].center;
                let h = voxel_worldgen::terrain_height(glam::Vec2::new(c[0], c[2]), 1.0);
                println!("ruin: {:.0},{:.0},{:.0}  ops={} ground={h:.0}", c[0], c[1], c[2], ops.len());
                found += 1;
                if found >= 10 { return; }
            }
        }
    }
}
