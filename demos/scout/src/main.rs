// Scout: find river mouths/segments near the test area.
use glam::IVec3;
use voxel_layers::IAabb;

fn main() {
    let mgr = voxel_worldgen::rivers::planning_layers(0, 0.5);
    let bounds = IAabb::new(IVec3::new(-30000, 0, -42000), IVec3::new(-22000, 1, -34000));
    for (_, chunk) in mgr.get::<voxel_worldgen::rivers::RiversLayer>(bounds).iter() {
        for river in &chunk.rivers {
            let a = river.waypoints[0];
            let b = *river.waypoints.last().unwrap();
            let ha = voxel_worldgen::terrain_height(a, 8.0);
            let hb = voxel_worldgen::terrain_height(b, 8.0);
            println!(
                "river: spring {:.0},{:.0},{:.0} -> mouth {:.0},{:.0},{:.0} len {} wp",
                a.x, ha, a.y, b.x, hb, b.y, river.waypoints.len()
            );
        }
    }
}
