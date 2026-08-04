// Scout for scenic start positions.
fn main() {
    let mut found = 0;
    'outer: for zi in -100..100 {
        for xi in -100..100 {
            let x = xi as f32 * 400.0;
            let z = zi as f32 * 400.0;
            let p = glam::Vec2::new(x, z);
            let h = voxel_worldgen::terrain_height(p, 1.0);
            if !(15.0..60.0).contains(&h) { continue; }
            if voxel_worldgen::terrain_up(p, 1.0) < 0.92 { continue; }
            if voxel_worldgen::forest_density(p) < 0.75 { continue; }
            // Mountain within 4 km?
            let mut peak = 0.0f32;
            for dz in -8..8 { for dx in -8..8 {
                let q = p + glam::Vec2::new(dx as f32 * 500.0, dz as f32 * 500.0);
                peak = peak.max(voxel_worldgen::terrain_height(q, 1.0));
            }}
            // Water within 4 km?
            let mut low = f32::MAX;
            for dz in -8..8 { for dx in -8..8 {
                let q = p + glam::Vec2::new(dx as f32 * 500.0, dz as f32 * 500.0);
                low = low.min(voxel_worldgen::terrain_height(q, 1.0));
            }}
            if peak > 300.0 && low < -5.0 {
                println!("start: {x},{},{z}  h={h:.0} peak={peak:.0} low={low:.0}", h + 30.0);
                found += 1;
                if found >= 8 { break 'outer; }
            }
        }
    }
}
