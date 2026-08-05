// Scout: find cave ops included in one chunk's padded query box but
// whose influence reaches a neighbor that culled them.
use glam::Vec3;

fn main() {
    let base = 0.1_f32;
    for level in 0..4u32 {
        let vs = base * (1 << level) as f32;
        let edge = 32.0 * vs;
        let pad = 4.0 * vs;
        // chunks around the crack area
        let center = Vec3::new(-26828.0, 84.0, -40701.0);
        let c0 = (center / edge).floor();
        for dz in -2..=2 {
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let min = (c0 + glam::Vec3::new(dx as f32, dy as f32, dz as f32)) * edge;
                    let max = min + Vec3::splat(edge);
                    let ops_here =
                        voxel_worldgen::caves::caves_ops(0, 0.45, [2.2, 3.6], min - pad, max + pad);
                    for dir in [
                        Vec3::new(edge, 0.0, 0.0),
                        Vec3::new(-edge, 0.0, 0.0),
                        Vec3::new(0.0, edge, 0.0),
                        Vec3::new(0.0, -edge, 0.0),
                        Vec3::new(0.0, 0.0, edge),
                        Vec3::new(0.0, 0.0, -edge),
                    ] {
                        let nmin = min + dir;
                        let nmax = nmin + Vec3::splat(edge);
                        let ops_n = voxel_worldgen::caves::caves_ops(
                            0, 0.45, [2.2, 3.6], nmin - pad, nmax + pad,
                        );
                        for op in &ops_here {
                            if ops_n.contains(op) {
                                continue;
                            }
                            let sample_reach = 3.0 * vs;
                            let smin = nmin - Vec3::splat(2.0 * vs);
                            let smax = nmax + Vec3::splat(sample_reach);
                            let c = Vec3::from(op.center);
                            let r = op.half[0];
                            let closest = c.clamp(smin, smax);
                            if (closest - c).length() < r {
                                println!(
                                    "L{level} chunk {min:?}: op {c:?} r={r:.2} reaches {dir:?} neighbor but culled"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    println!("scan done");
}
