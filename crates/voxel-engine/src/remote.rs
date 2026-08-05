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

fn f32s(v: &Value, key: &str, n: usize) -> Result<Vec<f32>, BrpError> {
    let arr = v
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| err(format!("missing param {key:?} (array of {n})")))?;
    if arr.len() != n {
        return Err(err(format!("{key:?} must have {n} elements")));
    }
    Ok(arr
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
        .collect())
}

type PlayerCamera<'w, 's, T> =
    Query<'w, 's, T, (With<voxel_debug::FreeCamera>, With<Camera3d>)>;

fn status(In(_): In<Option<Value>>, cams: PlayerCamera<&Transform>) -> BrpResult {
    let t = cams.single().map_err(|_| err("no player camera"))?;
    let f = t.forward();
    Ok(json!({
        "pos": [t.translation.x, t.translation.y, t.translation.z],
        "look": [f.x, f.y, f.z],
    }))
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
    let r = params.get("radius").and_then(Value::as_f64).unwrap_or(512.0) as f32;
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
    let r = params.get("radius").and_then(Value::as_f64).unwrap_or(2048.0) as f32;
    let kind = params.get("kind").and_then(Value::as_str);
    let (min, max) = (Vec2::new(c[0] - r, c[1] - r), Vec2::new(c[0] + r, c[1] + r));
    let found: Vec<Value> = world
        .markers_in(min, max, kind)
        .iter()
        .map(|m| json!({"pos": [m.pos.x, m.pos.y, m.pos.z], "kind": m.kind}))
        .collect();
    Ok(json!({"count": found.len(), "markers": found}))
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
