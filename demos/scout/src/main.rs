// Scout: find steep high terrain (cliff faces) to screenshot.
fn main() {
    let mut best: Vec<(f32, f32, f32, f32)> = Vec::new(); // (score, x, z, h)
    for gz in -420..-340 {
        for gx in -320..-220 {
            let x = gx as f32 * 100.0;
            let z = gz as f32 * 100.0;
            let p = glam::Vec2::new(x, z);
            let h = voxel_worldgen::terrain_height(p, 1.0);
            if h < 300.0 {
                continue;
            }
            let up = voxel_worldgen::terrain_up(p, 1.0);
            let score = h * (1.0 - up);
            best.push((score, x, z, h));
        }
    }
    best.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (s, x, z, h) in best.iter().take(8) {
        println!("cliff score {s:.0}: x={x:.0} z={z:.0} h={h:.0}");
    }
}
