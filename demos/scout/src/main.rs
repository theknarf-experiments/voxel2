// Scout: list cave mouths near the test area.
fn main() {
    for cz in -160..-130 {
        for cx in -120..-90 {
            if let Some(m) = voxel_worldgen::caves::cave_mouth(0, 0.45, cx, cz) {
                println!("mouth: {:.0} {:.0} {:.0}", m.x, m.y, m.z);
            }
        }
    }
}
