//! Live tooling over the Bevy Remote Protocol: a dev build adds this
//! unconditionally, starting an HTTP JSON-RPC server the `voxctl` CLI
//! drives —
//! teleport the camera, query planning data (ribbons, markers), and dump
//! offscreen screenshots from a RUNNING game instead of relaunching it
//! for every look.

use bevy::prelude::*;
use bevy::remote::{error_codes, http::RemoteHttpPlugin, BrpError, BrpResult, RemotePlugin};
use serde_json::{json, Value};


pub struct VoxelRemotePlugin {
    pub port: u16,
}

impl Plugin for VoxelRemotePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HostCommands>().add_plugins(
            RemotePlugin::default()
                .with_method_main("voxel/status", status)
                .with_method_main("voxel/teleport", teleport)
                .with_method_main("voxel/inspect", inspect)
                .with_method_main("voxel/ops", ops)
                .with_method_main("voxel/scan", scan)
                .with_method_main("voxel/viz", viz)
                .with_method_main("voxel/world", world_switch)
                .with_method_main("voxel/host", host_command)
                .with_method_main("voxel/screenshot", screenshot),
        )
        .add_plugins(RemoteHttpPlugin::default().with_port(self.port));
    }
}

/// Percentiles of a frame-time history, in milliseconds.
///
/// `worst` is the single slowest frame in the window rather than a
/// percentile: one 40 ms frame in 120 is exactly the stutter this is for,
/// and every percentile short of the maximum averages it away.
fn frame_stats(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let round = |x: f64| (x * 100.0).round() / 100.0;
    json!({
        "n": sorted.len(),
        "mean": round(mean),
        "p50": round(at(0.5)),
        "p95": round(at(0.95)),
        "p99": round(at(0.99)),
        "worst": round(at(1.0)),
        // What the mean fps WOULD be without the tail, so a run can be
        // compared against the 100 fps target without reading four
        // numbers.
        "mean_fps": round(1000.0 / mean),
    })
}

fn err(msg: impl Into<String>) -> BrpError {
    BrpError {
        code: error_codes::INTERNAL_ERROR,
        message: msg.into(),
        data: None,
    }
}

/// World-coordinate clamp for remote inputs: far beyond any world, small
/// enough that downstream tile math cannot overflow i32.
const REMOTE_POS_M: f32 = 1.0e7;
/// Query radius clamp: a stray `radius: 1e9` must not enumerate the
/// whole plane through on-demand layer generation on the main thread.
const REMOTE_RADIUS_M: f64 = 8_192.0;

fn f32s(v: &Value, key: &str, n: usize) -> Result<Vec<f32>, BrpError> {
    let arr = v
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| err(format!("missing param {key:?} (array of {n})")))?;
    if arr.len() != n {
        return Err(err(format!("{key:?} must have {n} elements")));
    }
    arr.iter()
        .map(|x| {
            x.as_f64()
                .map(|v| (v as f32).clamp(-REMOTE_POS_M, REMOTE_POS_M))
                .ok_or_else(|| err(format!("{key:?} has a non-numeric element")))
        })
        .collect()
}

fn radius(params: &Value, default: f64) -> f32 {
    params
        .get("radius")
        .and_then(Value::as_f64)
        .unwrap_or(default)
        .clamp(1.0, REMOTE_RADIUS_M) as f32
}

type PlayerCamera<'w, 's, T> =
    Query<'w, 's, T, (With<crate::FreeCamera>, With<Camera3d>)>;

fn status(
    In(_): In<Option<Value>>,
    cams: PlayerCamera<&Transform>,
    stats: Option<Res<voxel_render::SharedRenderStats>>,
    probe: Option<Res<voxel_engine::streaming::StreamProbe>>,
    worlds: Option<Res<voxel_engine::Worlds>>,
    camera_world: Res<voxel_render::CameraWorld>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
) -> BrpResult {
    let t = cams.single().map_err(|_| err("no player camera"))?;
    let f = t.forward();
    // Smoothed frame rate: the shipped floor is 100 fps, and shading
    // changes are exactly the kind that quietly cost it.
    let fps = diagnostics
        .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed());
    // The DISTRIBUTION, not the average. An average hides the whole
    // problem: a frame that misses its deadline every twentieth frame
    // reads as a fine mean and a visible stutter, and the two are told
    // apart by p99 against p50 and by nothing else. Milliseconds, over
    // whatever history the diagnostic keeps (120 frames).
    let frame_ms: Vec<f64> = diagnostics
        .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .map(|d| d.values().copied().collect())
        .unwrap_or_default();
    let mut out = json!({
        "pos": [t.translation.x, t.translation.y, t.translation.z],
        "look": [f.x, f.y, f.z],
        "fps": fps,
        "frame_ms": frame_stats(&frame_ms),
    });
    if let Some(p) = probe {
        out["stream"] = json!({
            "resident": p.resident,
            "generating": p.generating,
            "stalled": p.stalled,
            "pruned": p.pruned,
            "unpruned": p.unpruned,
            "unpruned_with_ops": p.unpruned_with_ops,
            "reads_missed": p.reads_missed,
            "settled": p.settled,
            "settling_s": p.settling_s,
            "last_settle_s": p.last_settle_s,
            "worst_settle_s": p.worst_settle_s,
        });
    }
    // The world the camera is in: its planning is the one whose residency
    // and missed reads say whether what you are looking at is covered.
    if let Some(world) = worlds.as_ref().and_then(|w| w.query(camera_world.0)) {
        let planning = world.stats();
        out["planning"] = json!({
            "resident_chunks": planning.resident_chunks,
            "reads_missed": planning.reads_missed,
            "generating": planning.generating,
            "layers": planning
                .layers
                .iter()
                .map(|l| json!({
                    "name": l.name,
                    "resident": l.resident,
                    "created": l.created,
                    "destroyed": l.destroyed,
                    "create_ms": l.create_time.as_secs_f64() * 1000.0,
                }))
                .collect::<Vec<_>>(),
        });
    }
    if let Some(s) = stats {
        let s = s.0.lock().unwrap();
        out["chunks"] = json!({
            "gen_started": s.gen_started,
            "mesh_started": s.mesh_started,
            "reported_ready": s.reported_ready,
            "gen_starved": s.gen_starved,
            "drawn_per_world": s.drawn_per_world,
            "tracked": s.tracked,
            "meshed": s.meshed,
            "awaiting": s.awaiting,
            "drawn": s.drawn,
            "arena_free": s.arena_free,
            "slab_used_pages": s.slab_used_pages,
            "slab_total_pages": s.slab_total_pages,
            "slab_peak_pages": s.slab_peak_pages,
            "slab_longest_free_run": s.slab_longest_free_run,
            // Live allocations by run length, and the session high-water
            // mark. Standing still samples one point of a process that
            // depends on where the camera is; the peak is what a budget
            // has to answer to.
            "slab_runs": s.slab_runs,
            "slab_peak_runs": s.slab_peak_runs,
            "slab_capacity_chunks": s.slab_capacity_chunks,
            "slab_failed": s.slab_pressure.failed,
            "slab_fragmented": s.slab_pressure.fragmented,
            "states": s.state_counts.iter().cloned().collect::<std::collections::HashMap<_,_>>(),
        });
    }
    Ok(out)
}

/// `{"pos": [x, y, z], "look": [dx, dy, dz]?}` — move the fly camera.
fn teleport(In(params): In<Option<Value>>, mut cams: PlayerCamera<&mut Transform>) -> BrpResult {
    let params = params.ok_or_else(|| err("params required"))?;
    let pos = f32s(&params, "pos", 3)?;
    let pos = Vec3::new(pos[0], pos[1], pos[2]);
    let mut t = cams.single_mut().map_err(|_| err("no player camera"))?;
    t.translation = pos;
    if params.get("look").is_some() {
        let look = f32s(&params, "look", 3)?;
        let look = Vec3::new(look[0], look[1], look[2]).normalize_or_zero();
        if look != Vec3::ZERO {
            let up = if look.dot(Vec3::Y).abs() > 0.9 {
                Vec3::Z
            } else {
                Vec3::Y
            };
            t.look_at(pos + look * 100.0, up);
        }
    }
    Ok(json!({"ok": true}))
}


/// The world the player is standing in.
///
/// Every introspection command answers about THAT world: `voxctl ribbons`
/// after stepping through a portal must describe where you are, not the
/// level the app was launched with. Worlds share coordinates, so a
/// command answering from the wrong one looks plausible and is wrong.
fn here<'a>(
    worlds: &'a voxel_engine::Worlds,
    camera: &voxel_render::CameraWorld,
) -> Result<&'a voxel_engine::WorldQuery, BrpError> {
    worlds
        .query(camera.0)
        .ok_or_else(|| err("no world loaded"))
}

/// Anything the HOST's planner cares to answer, carried verbatim.
///
/// The engine has no ribbons and no markers, so this endpoint does not
/// either: it forwards the query and returns the reply. `voxctl ribbons`
/// and `voxctl markers` are `{"kind": "ribbons"}` and
/// `{"kind": "markers"}` to a host that knows what those are.
fn inspect(
    In(params): In<Option<Value>>,
    worlds: Res<voxel_engine::Worlds>,
    camera: Res<voxel_render::CameraWorld>,
) -> BrpResult {
    let world = here(&worlds, &camera)?;
    let params = params.ok_or_else(|| err("params required"))?;
    // Introspection, not a working set — an absent answer out here is by
    // design and must not count against `reads_missed`.
    let _peek = world.peek();
    Ok(world.inspect(&params))
}

/// `{"center": [x, y, z], "radius": r, "edge": chunk_edge_m?}` — CSG ops
/// the provider would serve around a point (debugging what a chunk sees).
fn ops(
    In(params): In<Option<Value>>,
    worlds: Res<voxel_engine::Worlds>,
    camera: Res<voxel_render::CameraWorld>,
) -> BrpResult {
    let world = here(&worlds, &camera)?;
    let params = params.ok_or_else(|| err("params required"))?;
    let c = f32s(&params, "center", 3)?;
    let r = radius(&params, 40.0);
    let edge = params.get("edge").and_then(Value::as_f64).unwrap_or(12.8) as f32;
    let center = Vec3::new(c[0], c[1], c[2]);
    // Introspection: what the provider WOULD serve here. An absent chunk
    // is part of the answer, not a consumer failing to declare coverage.
    let _peek = world.peek();
    let found = world.ops_in(center - Vec3::splat(r), center + Vec3::splat(r), edge);
    let adds = found.iter().filter(|o| o.kind & 1 == 0).count();
    let sample: Vec<Value> = found
        .iter()
        .take(12)
        .map(|o| {
            json!({
                "kind": o.kind,
                "center": o.center,
                "half": o.half,
                "material": o.material,
            })
        })
        .collect();
    Ok(json!({
        "count": found.len(),
        "adds": adds,
        "cuts": found.len() - adds,
        "sample": sample,
    }))
}

/// `{"chunks": bool?, "layers": bool|number?}` — toggle the debug
/// overlays. `layers` takes a radius in meters, or a bool for the near
/// range.
/// Anything the HOST defines, passed through untouched.
///
/// The engine's tooling has no business knowing what a portal is, so it
/// carries an opaque value and the host decides what it means. Queued
/// rather than applied,
/// because the host reads it from its own systems.
#[derive(Resource, Default)]
pub struct HostCommands(pub Vec<Value>);

fn host_command(In(params): In<Option<Value>>, mut queue: ResMut<HostCommands>) -> BrpResult {
    let params = params.ok_or_else(|| err("params required"))?;
    queue.0.push(params.clone());
    Ok(json!({ "queued": params }))
}

/// `{"world": n}` — which world the camera is in, and so which one is
/// drawn. Every registered world stays resident either way; this only
/// chooses the view, which is how you can tell two levels are genuinely
/// live at once rather than being swapped.
fn world_switch(
    In(params): In<Option<Value>>,
    mut camera_world: ResMut<voxel_render::CameraWorld>,
) -> BrpResult {
    if let Some(w) = params
        .as_ref()
        .and_then(|p| p.get("world"))
        .and_then(Value::as_u64)
    {
        camera_world.0 = w as u8;
    }
    Ok(json!({ "world": camera_world.0 }))
}

fn viz(In(params): In<Option<Value>>, mut viz: ResMut<crate::viz::DebugViz>) -> BrpResult {
    let params = params.ok_or_else(|| err("params required"))?;
    if let Some(v) = params.get("chunks").and_then(Value::as_bool) {
        viz.chunks = v;
    }
    match params.get("layers") {
        Some(Value::Bool(v)) => {
            viz.layer_radius_m = if *v { crate::viz::LAYER_VIZ_NEAR_M } else { 0.0 }
        }
        Some(v) => {
            if let Some(r) = v.as_f64() {
                viz.layer_radius_m = r as f32;
            }
        }
        None => {}
    }
    Ok(json!({
        "chunks": viz.chunks,
        "layers_radius_m": viz.layer_radius_m,
        "lines_drawn": viz.drawn,
        "lines_wanted": viz.wanted,
    }))
}

/// Scenic-spot scoring: local relief (how dramatically the terrain
/// drops around the point) plus an altitude bonus — the "steep high
/// terrain" heuristic the old offline scout binary used.
fn scan_terrain(
    generator: &voxel_worldgen::Generator,
    center: Vec2,
    radius: f32,
    step: f32,
    top: usize,
) -> Vec<(Vec3, f32, f32)> {
    // Bound the grid so a wide scan cannot stall the main thread.
    let step = step.max(radius * 2.0 / 128.0).max(8.0);
    let n = (radius * 2.0 / step) as i32;
    let height = |p: Vec2| generator.height(p, 8.0);
    let mut spots: Vec<(Vec3, f32, f32)> = Vec::new();
    for gz in 0..=n {
        for gx in 0..=n {
            let p = center - Vec2::splat(radius)
                + Vec2::new(gx as f32, gz as f32) * step;
            let h = height(p);
            if h <= 1.0 {
                continue; // sea floor is never scenic
            }
            let mut relief = 0.0f32;
            for (dx, dz) in [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)] {
                relief = relief.max((h - height(p + Vec2::new(dx, dz))).abs());
            }
            let score = relief + h * 0.15;
            spots.push((Vec3::new(p.x, h, p.y), relief, score));
        }
    }
    spots.sort_by(|a, b| b.2.total_cmp(&a.2));
    spots.truncate(top);
    spots
}

/// `{"center": [x, z], "radius": r?, "step": s?, "top": n?}` — scan the
/// terrain mirror for scenic spots (steep, high ground), ranked. The
/// offline scout binary's job, minus the hand-copied level config.
fn scan(
    In(params): In<Option<Value>>,
    worlds: Res<voxel_engine::Worlds>,
    camera: Res<voxel_render::CameraWorld>,
) -> BrpResult {
    let world = here(&worlds, &camera)?;
    let params = params.ok_or_else(|| err("params required"))?;
    let c = f32s(&params, "center", 2)?;
    let r = radius(&params, 4_096.0);
    let step = params.get("step").and_then(Value::as_f64).unwrap_or(32.0) as f32;
    let top = params
        .get("top")
        .and_then(Value::as_u64)
        .unwrap_or(10)
        .min(64) as usize;
    let spots: Vec<Value> = scan_terrain(world.generator(), Vec2::new(c[0], c[1]), r, step, top)
        .iter()
        .map(|(pos, relief, score)| {
            json!({
                "pos": [pos.x, pos.y, pos.z],
                "relief": relief,
                "score": score,
            })
        })
        .collect();
    Ok(json!({"count": spots.len(), "spots": spots}))
}

/// `{"path": "shot.png"}` — dump the next rendered frame through the
/// offscreen mirror (works with an occluded window).
fn screenshot(
    In(params): In<Option<Value>>,
    mut requests: ResMut<crate::ScreenshotRequest>,
) -> BrpResult {
    let params = params.ok_or_else(|| err("params required"))?;
    let path = params
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| err("missing param \"path\""))?;
    let window = params
        .get("window")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    requests.0.push(crate::ScreenshotWant {
        path: path.to_string(),
        window,
    });
    Ok(json!({"queued": path, "window": window}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_ranks_bounded_spots_within_radius() {
        let generator = voxel_worldgen::Generator::new(
            voxel_worldgen::program::planet_program(),
            0,
            voxel_worldgen::program::DEFAULT_SUN_DIR,
        );
        // Land region of the reference planet.
        let center = Vec2::new(-27000.0, -38000.0);
        let spots = scan_terrain(&generator, center, 4096.0, 64.0, 10);
        assert!(!spots.is_empty() && spots.len() <= 10);
        for w in spots.windows(2) {
            assert!(w[0].2 >= w[1].2, "not sorted by score");
        }
        for (pos, relief, score) in &spots {
            assert!(Vec2::new(pos.x, pos.z).distance(center) <= 4096.0 * 1.5);
            assert!(pos.y > 1.0, "sea-floor spot");
            assert!(*score >= *relief);
        }
        // Deterministic.
        assert_eq!(spots, scan_terrain(&generator, center, 4096.0, 64.0, 10));
    }
}
