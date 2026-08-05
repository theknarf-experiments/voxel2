// Scout: find a road midpoint to screenshot.
fn main() {
    let mgr = voxel_worldgen::roads::planning_layers(0);
    let bounds = voxel_layers::IAabb::new(
        glam::IVec3::new(-30000, 0, -42000),
        glam::IVec3::new(-22000, 1, -34000),
    );
    for (_, chunk) in mgr.get::<voxel_worldgen::roads::RoadsLayer>(bounds).iter() {
        for &(a, b) in &chunk.roads {
            let m = (a + b) * 0.5;
            let h = voxel_worldgen::terrain_height(m, 1.0);
            println!(
                "road: {:.0},{:.0} <-> {:.0},{:.0}  mid {:.0},{:.0},{:.0} len={:.0}",
                a.x,
                a.y,
                b.x,
                b.y,
                m.x,
                h + 25.0,
                m.y,
                a.distance(b)
            );
        }
    }
}
