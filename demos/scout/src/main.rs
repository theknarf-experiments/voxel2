// Scout: find stack rivers (flow courses) near the test area, matching
// the shipped planet.json stack configuration.
use glam::IVec3;
use voxel_layers::{IAabb, LayerManager};
use voxel_worldgen::stack::{FlowCfg, FlowCourses, ScatterCfg, ScatterSites};

fn main() {
    let mut mgr = LayerManager::new(0);
    mgr.register_as(
        "sites:springs",
        ScatterSites {
            cfg: ScatterCfg {
                cell_m: 512,
                chance: 0.5,
                altitude: [60.0, 400.0],
                ..Default::default()
            },
        },
    );
    mgr.register_as(
        "flow:rivers",
        FlowCourses {
            cfg: FlowCfg {
                source: "sites:springs".into(),
                ..Default::default()
            },
            cell_m: 512,
        },
    );
    let bounds = IAabb::new(IVec3::new(-30000, 0, -42000), IVec3::new(-22000, 1, -34000));
    for (_, chunk) in mgr.get_named::<FlowCourses>("flow:rivers", bounds).iter() {
        for (wp, levels) in &chunk.courses {
            let a = wp[0];
            let b = *wp.last().unwrap();
            let mid = wp[wp.len() / 2];
            println!(
                "river: spring {:.0},{:.0} -> mouth {:.0},{:.0} | mid {:.0},{:.0} level {:.1} | {} wp",
                a.x, a.y, b.x, b.y, mid.x, mid.y, levels[wp.len() / 2], wp.len()
            );
        }
    }
}
