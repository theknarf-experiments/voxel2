//! Live tooling over the Bevy Remote Protocol: `VOXEL_REMOTE=1` (or a
//! port number) starts an HTTP JSON-RPC server the `voxctl` CLI drives —
//! teleport the camera, query planning data (water, markers), and dump
//! offscreen screenshots from a RUNNING game instead of relaunching it
//! for every look.

use bevy::prelude::*;
use bevy::remote::{error_codes, http::RemoteHttpPlugin, BrpError, BrpResult, RemotePlugin};
use serde_json::{json, Value};

use crate::level::WorldQuery;

pub struct VoxelRemotePlugin {
    pub port: u16,
}

impl Plugin for VoxelRemotePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            RemotePlugin::default()
                .with_method_main("voxel/status", status)
                .with_method_main("voxel/teleport", teleport)
                .with_method_main("voxel/water", water)
                .with_method_main("voxel/markers", markers)
                .with_method_main("voxel/ops", ops)
                .with_method_main("voxel/viz", viz)
                .with_method_main("voxel/screenshot", screenshot),
        )
        .add_plugins(RemoteHttpPlugin::default().with_port(self.port));
    }
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
    Query<'w, 's, T, (With<voxel_debug::FreeCamera>, With<Camera3d>)>;

fn status(
    In(_): In<Option<Value>>,
    cams: PlayerCamera<&Transform>,
    stats: Option<Res<voxel_render::SharedRenderStats>>,
    probe: Option<Res<crate::streaming::StreamProbe>>,
) -> BrpResult {
    let t = cams.single().map_err(|_| err("no player camera"))?;
    let f = t.forward();
    let mut out = json!({
        "pos": [t.translation.x, t.translation.y, t.translation.z],
        "look": [f.x, f.y, f.z],
    });
    if let Some(p) = probe {
        out["stream"] = json!({
            "leaves": p.leaves,
            "planning": p.planning,
            "replan_needed": p.replan_needed,
            "epoch_waits": p.epoch_waits,
            "epoch_to_request": p.epoch_to_request,
            "epoch_age_s": p.epoch_age_s,
        });
    }
    if let Some(s) = stats {
        let s = s.0.lock().unwrap();
        out["chunks"] = json!({
            "tracked": s.tracked,
            "meshed": s.meshed,
            "awaiting": s.awaiting,
            "drawn": s.drawn,
            "arena_free": s.arena_free,
            "slabs": s.slab_occupancy,
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

/// `{"center": [x, z], "radius": r}` — planning water segments near a
/// point (find the rivers).
fn water(In(params): In<Option<Value>>, world: Res<WorldQuery>) -> BrpResult {
    let params = params.ok_or_else(|| err("params required"))?;
    let c = f32s(&params, "center", 2)?;
    let r = radius(&params, 512.0);
    let (min, max) = (Vec2::new(c[0] - r, c[1] - r), Vec2::new(c[0] + r, c[1] + r));
    let segs: Vec<Value> = world
        .water_in(min, max)
        .iter()
        .map(|s| {
            json!({
                "a": [s.a.x, s.a.y],
                "b": [s.b.x, s.b.y],
                "half_w": s.half_w,
                "levels": s.levels,
            })
        })
        .collect();
    Ok(json!({"count": segs.len(), "segments": segs}))
}

/// `{"center": [x, z], "radius": r, "kind": "ruin"?}` — stack markers
/// near a point (findable content).
fn markers(In(params): In<Option<Value>>, world: Res<WorldQuery>) -> BrpResult {
    let params = params.ok_or_else(|| err("params required"))?;
    let c = f32s(&params, "center", 2)?;
    let r = radius(&params, 2048.0);
    let kind = params.get("kind").and_then(Value::as_str);
    let (min, max) = (Vec2::new(c[0] - r, c[1] - r), Vec2::new(c[0] + r, c[1] + r));
    let found: Vec<Value> = world
        .markers_in(min, max, kind)
        .iter()
        .map(|m| json!({"pos": [m.pos.x, m.pos.y, m.pos.z], "kind": m.kind}))
        .collect();
    Ok(json!({"count": found.len(), "markers": found}))
}

/// `{"center": [x, y, z], "radius": r, "edge": chunk_edge_m?}` — CSG ops
/// the provider would serve around a point (debugging what a chunk sees).
fn ops(In(params): In<Option<Value>>, world: Res<WorldQuery>) -> BrpResult {
    let params = params.ok_or_else(|| err("params required"))?;
    let c = f32s(&params, "center", 3)?;
    let r = radius(&params, 40.0);
    let edge = params.get("edge").and_then(Value::as_f64).unwrap_or(12.8) as f32;
    let center = Vec3::new(c[0], c[1], c[2]);
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

/// `{"chunks": bool?, "layers": bool?}` — toggle the debug overlays.
fn viz(In(params): In<Option<Value>>, mut viz: ResMut<crate::debug_viz::DebugViz>) -> BrpResult {
    let params = params.ok_or_else(|| err("params required"))?;
    if let Some(v) = params.get("chunks").and_then(Value::as_bool) {
        viz.chunks = v;
    }
    if let Some(v) = params.get("layers").and_then(Value::as_bool) {
        viz.layers = v;
    }
    Ok(json!({"chunks": viz.chunks, "layers": viz.layers}))
}

/// `{"path": "shot.png"}` — dump the next rendered frame through the
/// offscreen mirror (works with an occluded window).
fn screenshot(
    In(params): In<Option<Value>>,
    mut requests: ResMut<voxel_debug::ScreenshotRequest>,
) -> BrpResult {
    let params = params.ok_or_else(|| err("params required"))?;
    let path = params
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| err("missing param \"path\""))?;
    requests.0.push(path.to_string());
    Ok(json!({"queued": path}))
}
