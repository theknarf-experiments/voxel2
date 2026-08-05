//! The generic planning-stack vocabulary (unification M2): a small set of
//! parameterized layer kinds — scatter, connect, flow, worm, emit — that
//! level JSON composes into ONE LayerManager per level. Features (ruins,
//! roads, rivers, caves, dungeons, districts) are configurations of these
//! kinds, not engine code.

use glam::{IVec3, Vec2, Vec3};
use voxel_core::csg::CsgOp;
use voxel_layers::{IAabb, Layer, LayerCtx};

use crate::{terrain_height, terrain_up};

/// A water surface segment (river reach): endpoints, half width, and the
/// (monotone) water level at each end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSeg {
    pub a: Vec2,
    pub b: Vec2,
    pub half_w: f32,
    pub levels: [f32; 2],
}

/// A point of interest emitted by the stack (dungeon entrance, bridge,
/// spawn anchor...). `kind` is a level-defined tag.
#[derive(Clone, Debug, PartialEq)]
pub struct Marker {
    pub pos: Vec3,
    pub kind: String,
}

/// What planning layers ultimately emit; the world-query facade serves
/// these bucketed by index cells.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PatchSet {
    pub ops: Vec<CsgOp>,
    pub water: Vec<WaterSeg>,
    pub clearance: Vec<[Vec2; 2]>,
    pub markers: Vec<Marker>,
}

impl PatchSet {
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
            && self.water.is_empty()
            && self.clearance.is_empty()
            && self.markers.is_empty()
    }

    pub fn extend(&mut self, other: PatchSet) {
        self.ops.extend(other.ops);
        self.water.extend(other.water);
        self.clearance.extend(other.clearance);
        self.markers.extend(other.markers);
    }
}

/// Configuration of a `scatter` stack layer: hash-gated candidate sites
/// per cell, filtered by terrain.
#[derive(Clone, Debug)]
pub struct ScatterCfg {
    pub cell_m: i32,
    /// Chance a cell hosts a site.
    pub chance: f32,
    /// Margin (meters) keeping sites away from cell borders.
    pub margin_m: f32,
    /// Altitude band sites may occupy.
    pub altitude: [f32; 2],
    /// Up-ness interval (1 = flat).
    pub up: [f32; 2],
}

impl Default for ScatterCfg {
    fn default() -> Self {
        Self {
            cell_m: 256,
            chance: 0.3,
            margin_m: 32.0,
            altitude: [f32::MIN, f32::MAX],
            up: [0.0, 1.0],
        }
    }
}

/// Generic site scatter: the sites layer every other kind consumes.
/// Register one instance per feature ("sites:ruins", "sites:springs"...).
#[derive(Clone)]
pub struct ScatterSites {
    pub cfg: ScatterCfg,
}

pub struct SitesChunk {
    pub sites: Vec<Vec2>,
}

impl Layer for ScatterSites {
    type Chunk = SitesChunk;
    const NAME: &'static str = "stack/scatter";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(self.cfg.cell_m, 0, self.cfg.cell_m)
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> SitesChunk {
        let mut rng = ctx.rng();
        if rng.next_f32() > self.cfg.chance {
            return SitesChunk { sites: Vec::new() };
        }
        let b = ctx.chunk_bounds();
        let cell = self.cfg.cell_m as f32;
        let m = self.cfg.margin_m.min(cell * 0.45);
        let p = Vec2::new(
            b.min.x as f32 + m + rng.next_f32() * (cell - 2.0 * m),
            b.min.z as f32 + m + rng.next_f32() * (cell - 2.0 * m),
        );
        let h = terrain_height(p, 8.0);
        let up = terrain_up(p, 8.0);
        if !(self.cfg.altitude[0]..self.cfg.altitude[1]).contains(&h)
            || !(self.cfg.up[0]..=self.cfg.up[1]).contains(&up)
        {
            return SitesChunk { sites: Vec::new() };
        }
        SitesChunk { sites: vec![p] }
    }
}

/// Configuration of a `connect` stack layer: pathfound links between
/// sites of a scatter instance (roads, patrol routes, power lines...).
#[derive(Clone, Debug)]
pub struct ConnectCfg {
    /// Scatter instance supplying the sites.
    pub source: String,
    /// Max link distance (m); each site links to its nearest neighbor.
    pub reach_m: f32,
    /// Pathfinding corridor half-width around the endpoint box.
    pub corridor_m: f32,
    pub slope_penalty: f32,
}

impl Default for ConnectCfg {
    fn default() -> Self {
        Self {
            source: String::new(),
            reach_m: 700.0,
            corridor_m: 192.0,
            slope_penalty: 60.0,
        }
    }
}

/// Generic pathfound connections (owned by the link midpoint's cell).
#[derive(Clone)]
pub struct ConnectPaths {
    pub cfg: ConnectCfg,
    pub cell_m: i32,
}

pub struct PathsChunk {
    pub paths: Vec<Vec<Vec2>>,
}

impl Layer for ConnectPaths {
    type Chunk = PathsChunk;
    const NAME: &'static str = "stack/connect";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(self.cell_m, 0, self.cell_m)
    }

    fn dependencies(&self) -> Vec<voxel_layers::Dep> {
        let pad = (self.cfg.reach_m + self.cfg.corridor_m) as i32;
        vec![voxel_layers::Dep::named(
            &self.cfg.source,
            IVec3::new(pad, 0, pad),
        )]
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> PathsChunk {
        let own = ctx.chunk_bounds();
        let pad = (self.cfg.reach_m + self.cfg.corridor_m) as i32;
        let view = ctx.get_named::<ScatterSites>(&self.cfg.source, own.inflate(IVec3::new(pad, 0, pad)));
        let sites: Vec<Vec2> = view.iter().flat_map(|(_, c)| c.sites.iter().copied()).collect();
        let in_own = |p: Vec2| {
            p.x >= own.min.x as f32
                && p.x < own.max.x as f32
                && p.y >= own.min.z as f32
                && p.y < own.max.z as f32
        };
        let mut paths = Vec::new();
        for &a in &sites {
            let Some(&b) = sites
                .iter()
                .filter(|&&b| b != a && a.distance(b) < self.cfg.reach_m)
                .min_by(|x, y| a.distance_squared(**x).total_cmp(&a.distance_squared(**y)))
            else {
                continue;
            };
            let (lo, hi) = if (a.x, a.y) <= (b.x, b.y) { (a, b) } else { (b, a) };
            if !in_own((lo + hi) * 0.5) {
                continue;
            }
            let clo = lo.min(hi) - Vec2::splat(self.cfg.corridor_m);
            let chi = lo.max(hi) + Vec2::splat(self.cfg.corridor_m);
            let params = crate::path::PathParams {
                slope_penalty: self.cfg.slope_penalty,
                ..Default::default()
            };
            let waypoints = crate::path::find_path(
                &|p| terrain_height(p, 8.0),
                lo,
                hi,
                clo,
                chi,
                &params,
            )
            .unwrap_or_else(|| vec![lo, hi]);
            if !paths.contains(&waypoints) {
                paths.push(waypoints);
            }
        }
        PathsChunk { paths }
    }
}

/// Configuration of a `flow` stack layer: descent courses from sites
/// (rivers, lava, mudslides).
#[derive(Clone, Debug)]
pub struct FlowCfg {
    pub source: String,
    pub max_steps: usize,
    pub max_spill_rise: f32,
    /// Half width at the source and at the end (linear growth).
    pub width: [f32; 2],
}

impl Default for FlowCfg {
    fn default() -> Self {
        Self {
            source: String::new(),
            max_steps: 400,
            max_spill_rise: 7.0,
            width: [2.0, 7.0],
        }
    }
}

#[derive(Clone)]
pub struct FlowCourses {
    pub cfg: FlowCfg,
    pub cell_m: i32,
}

pub struct CoursesChunk {
    pub courses: Vec<(Vec<Vec2>, Vec<f32>)>,
}

impl Layer for FlowCourses {
    type Chunk = CoursesChunk;
    const NAME: &'static str = "stack/flow";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(self.cell_m, 0, self.cell_m)
    }

    fn dependencies(&self) -> Vec<voxel_layers::Dep> {
        vec![voxel_layers::Dep::named(&self.cfg.source, IVec3::ZERO)]
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> CoursesChunk {
        let own = ctx.chunk_bounds();
        let view = ctx.get_named::<ScatterSites>(&self.cfg.source, own);
        let mut courses = Vec::new();
        for (_, chunk) in view.iter() {
            for &start in &chunk.sites {
                let params = crate::rivers::FlowParams {
                    max_steps: self.cfg.max_steps,
                    max_spill_rise: self.cfg.max_spill_rise,
                    ..Default::default()
                };
                let waypoints =
                    crate::rivers::flow_path(&|p| terrain_height(p, 8.0), start, &params);
                if waypoints.len() < 6 {
                    continue;
                }
                let mut level = f32::MAX;
                let levels: Vec<f32> = waypoints
                    .iter()
                    .map(|p| {
                        level = level.min(terrain_height(*p, 8.0) - 0.35);
                        level
                    })
                    .collect();
                courses.push((waypoints, levels));
            }
        }
        CoursesChunk { courses }
    }
}

/// Configuration of a `worm` stack layer: noise-steered burrows from
/// sites (caves, lava tubes).
#[derive(Clone, Debug)]
pub struct WormCfg {
    pub source: String,
    pub steps: u32,
    pub radius: [f32; 2],
    /// Keep tunnels this many radii under the surface.
    pub burial_radii: f32,
}

impl Default for WormCfg {
    fn default() -> Self {
        Self {
            source: String::new(),
            steps: 70,
            radius: [2.2, 3.6],
            burial_radii: 2.4,
        }
    }
}

#[derive(Clone)]
pub struct WormBurrows {
    pub cfg: WormCfg,
    pub cell_m: i32,
}

pub struct WormsChunk {
    /// Each worm: sphere centers with radii.
    pub worms: Vec<Vec<(Vec3, f32)>>,
}

impl Layer for WormBurrows {
    type Chunk = WormsChunk;
    const NAME: &'static str = "stack/worm";

    fn chunk_extent(&self) -> IVec3 {
        IVec3::new(self.cell_m, 0, self.cell_m)
    }

    fn dependencies(&self) -> Vec<voxel_layers::Dep> {
        vec![voxel_layers::Dep::named(&self.cfg.source, IVec3::ZERO)]
    }

    fn generate(&self, ctx: &LayerCtx<'_, Self>, _coord: IVec3) -> WormsChunk {
        let own = ctx.chunk_bounds();
        let view = ctx.get_named::<ScatterSites>(&self.cfg.source, own);
        let mut worms = Vec::new();
        for (_, chunk) in view.iter() {
            for &mouth_xz in &chunk.sites {
                let mut rng = ctx.rng();
                let ground = terrain_height(mouth_xz, 8.0);
                let base_r =
                    self.cfg.radius[0] + rng.next_f32() * (self.cfg.radius[1] - self.cfg.radius[0]);
                let mut yaw = rng.next_f32() * std::f32::consts::TAU;
                let mut pitch = -0.4 - rng.next_f32() * 0.2;
                let mut pos = Vec3::new(mouth_xz.x, ground + base_r * 0.6, mouth_xz.y);
                let mut worm = Vec::new();
                for _ in 0..self.cfg.steps {
                    let r = base_r * (0.8 + 0.45 * rng.next_f32());
                    worm.push((pos, r));
                    yaw += (rng.next_f32() - 0.5) * 0.55;
                    pitch += (rng.next_f32() - 0.5) * 0.3 - pitch * 0.15;
                    pitch = pitch.clamp(-0.55, 0.25);
                    let ceiling =
                        terrain_height(Vec2::new(pos.x, pos.z), 8.0) - r * self.cfg.burial_radii;
                    if pos.y > ceiling {
                        pitch = (pitch - 0.2).min(-0.35);
                    }
                    let dir = Vec3::new(
                        yaw.cos() * pitch.cos(),
                        pitch.sin(),
                        yaw.sin() * pitch.cos(),
                    );
                    pos += dir * (r * 0.9);
                }
                worms.push(worm);
            }
        }
        WormsChunk { worms }
    }
}

/// Convenience: sites of a named scatter instance within bounds.
pub fn sites_in(
    mgr: &voxel_layers::LayerManager,
    instance: &str,
    bounds: IAabb,
) -> Vec<Vec2> {
    mgr.get_named::<ScatterSites>(instance, bounds)
        .iter()
        .flat_map(|(_, c)| c.sites.iter().copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_layers::LayerManager;

    fn bounds(r: i32) -> IAabb {
        IAabb::new(IVec3::new(-r, 0, -r), IVec3::new(r, 1, r))
    }

    /// A region of the reference planet known to be land (the test area
    /// around the shipped level's start) — world origin is open ocean and
    /// altitude-filtered scatters would be vacuously empty there.
    fn land_bounds(r: i32) -> IAabb {
        let c = IVec3::new(-27000, 0, -38000);
        IAabb::new(
            IVec3::new(c.x - r, 0, c.z - r),
            IVec3::new(c.x + r, 1, c.z + r),
        )
    }

    #[test]
    fn scatter_instances_are_independent_and_deterministic() {
        let mut mgr = LayerManager::new(3);
        mgr.register_as(
            "sites:common",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.9,
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "sites:rare",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.05,
                    ..Default::default()
                },
            },
        );
        let common = sites_in(&mgr, "sites:common", bounds(4096));
        let rare = sites_in(&mgr, "sites:rare", bounds(4096));
        assert!(common.len() > rare.len() * 3, "chance config ignored: {} vs {}", common.len(), rare.len());

        let mut mgr2 = LayerManager::new(3);
        mgr2.register_as(
            "sites:common",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.9,
                    ..Default::default()
                },
            },
        );
        assert_eq!(common, sites_in(&mgr2, "sites:common", bounds(4096)));
    }

    #[test]
    fn connect_paths_join_sites_within_reach() {
        let mut mgr = LayerManager::new(5);
        mgr.register_as(
            "sites:towns",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.6,
                    altitude: [3.0, 400.0],
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "roads",
            ConnectPaths {
                cfg: ConnectCfg {
                    source: "sites:towns".into(),
                    ..Default::default()
                },
                cell_m: 256,
            },
        );
        let b = land_bounds(4096);
        let mut total = 0;
        for (_, c) in mgr.get_named::<ConnectPaths>("roads", b).iter() {
            for path in &c.paths {
                total += 1;
                assert!(path.len() >= 2);
                let (a, z) = (path[0], *path.last().unwrap());
                assert!(a.distance(z) < 700.0 + 1.0);
                // Corridor containment: the emitter's padding contract.
                let lo = a.min(z) - Vec2::splat(192.0);
                let hi = a.max(z) + Vec2::splat(192.0);
                for w in path {
                    assert!(w.x >= lo.x && w.x <= hi.x && w.y >= lo.y && w.y <= hi.y);
                }
            }
        }
        assert!(total > 0, "no connections in 8 km x 8 km");
    }

    #[test]
    fn flow_and_worm_kinds_generate_from_scatter_instances() {
        let mut mgr = LayerManager::new(5);
        mgr.register_as(
            "sites:springs",
            ScatterSites {
                cfg: ScatterCfg {
                    cell_m: 512,
                    chance: 0.6,
                    altitude: [60.0, 400.0],
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "rivers",
            FlowCourses {
                cfg: FlowCfg {
                    source: "sites:springs".into(),
                    ..Default::default()
                },
                cell_m: 512,
            },
        );
        mgr.register_as(
            "sites:mouths",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 0.6,
                    altitude: [6.0, 500.0],
                    ..Default::default()
                },
            },
        );
        mgr.register_as(
            "caves",
            WormBurrows {
                cfg: WormCfg {
                    source: "sites:mouths".into(),
                    ..Default::default()
                },
                cell_m: 256,
            },
        );
        let b = land_bounds(4096);
        let mut courses = 0;
        for (_, c) in mgr.get_named::<FlowCourses>("rivers", b).iter() {
            for (wp, levels) in &c.courses {
                courses += 1;
                assert_eq!(wp.len(), levels.len());
                // Water line is monotone non-increasing.
                for w in levels.windows(2) {
                    assert!(w[1] <= w[0] + 1e-4);
                }
            }
        }
        assert!(courses > 0, "no rivers");
        let mut worms = 0;
        for (_, c) in mgr.get_named::<WormBurrows>("caves", b).iter() {
            for worm in &c.worms {
                worms += 1;
                assert!(worm.len() as u32 == WormCfg::default().steps);
            }
        }
        assert!(worms > 0, "no worms");
    }

    #[test]
    fn scatter_respects_terrain_filters() {
        let mut mgr = LayerManager::new(3);
        mgr.register_as(
            "sites:highland",
            ScatterSites {
                cfg: ScatterCfg {
                    chance: 1.0,
                    altitude: [120.0, 10_000.0],
                    up: [0.8, 1.0],
                    ..Default::default()
                },
            },
        );
        let found = sites_in(&mgr, "sites:highland", land_bounds(6000));
        assert!(!found.is_empty(), "no highland sites on the land region");
        for p in found {
            let h = terrain_height(p, 8.0);
            let up = terrain_up(p, 8.0);
            assert!(h >= 120.0, "lowland site at {p:?} h={h}");
            assert!(up >= 0.8, "steep site at {p:?} up={up}");
        }
    }
}
