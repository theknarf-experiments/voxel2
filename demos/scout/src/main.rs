// Scout: find roads with large climbs (switchback candidates).
use glam::{IVec3, Vec2};
use voxel_layers::IAabb;

fn main() {
    let mgr = voxel_worldgen::roads::planning_layers(0, 0.32, 700.0);
    let bounds = IAabb::new(IVec3::new(-30000, 0, -42000), IVec3::new(-22000, 1, -34000));
    let mut best: Vec<(f32, Vec2, usize)> = Vec::new();
    for (_, chunk) in mgr.get::<voxel_worldgen::roads::RoadsLayer>(bounds).iter() {
        for road in &chunk.roads {
            let hs: Vec<f32> = road
                .waypoints
                .iter()
                .map(|p| voxel_worldgen::terrain_height(*p, 8.0))
                .collect();
            let span = hs.iter().cloned().fold(f32::MIN, f32::max)
                - hs.iter().cloned().fold(f32::MAX, f32::min);
            let mid = road.waypoints[road.waypoints.len() / 2];
            best.push((span, mid, road.waypoints.len()));
        }
    }
    best.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (span, mid, n) in best.iter().take(5) {
        let h = voxel_worldgen::terrain_height(*mid, 8.0);
        println!("climb {span:.0}m mid {:.0},{:.0},{:.0} ({n} wp)", mid.x, h, mid.y);
    }
}
