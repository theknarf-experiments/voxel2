//! voxctl — drive a running voxel2 instance over the Bevy Remote
//! Protocol (start the game with `VOXEL_REMOTE=1`). Interactive
//! verification without relaunching: teleport, query planning data,
//! dump offscreen screenshots.
//!
//! Usage:
//!   voxctl status
//!   voxctl goto X Y Z [DX DY DZ]
//!   voxctl ribbons X Z [RADIUS]
//!   voxctl scan X Z [RADIUS] [STEP]   # scenic-spot ranking (ex-scout)
//!   voxctl markers X Z [RADIUS] [KIND]
//!   voxctl shot PATH
//!   voxctl get [FIELD_PATH]          # read the live level
//!   voxctl set FIELD_PATH VALUE      # write it; applies without a relaunch
//!   voxctl raw METHOD [PARAMS_JSON]
//!
//! e.g. `voxctl set materials[7].base '[0.021,0.032,0.0087]'`
//!
//! `VOXCTL_PORT` overrides the default port (15702).

use std::io::{Read, Write};
use std::net::TcpStream;

use serde_json::{json, Value};

fn call(method: &str, params: Value) -> Result<Value, String> {
    let port: u16 = std::env::var("VOXCTL_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(15702);
    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    })
    .to_string();
    let mut stream = TcpStream::connect(("127.0.0.1", port)).map_err(|e| {
        format!("connect 127.0.0.1:{port}: {e} (is the game running with VOXEL_REMOTE=1?)")
    })?;
    let http = format!(
        "POST / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request}",
        request.len()
    );
    stream
        .write_all(http.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|e| e.to_string())?;
    let response = String::from_utf8_lossy(&response);
    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, b)| b)
        .unwrap_or(&response);
    // Tolerate chunked transfer encoding: take the largest {...} span.
    let json_body = match (body.find('{'), body.rfind('}')) {
        (Some(a), Some(b)) if b > a => &body[a..=b],
        _ => return Err(format!("no JSON in response: {body:?}")),
    };
    let v: Value = serde_json::from_str(json_body).map_err(|e| format!("bad JSON: {e}"))?;
    if let Some(error) = v.get("error") {
        return Err(error.to_string());
    }
    Ok(v.get("result").cloned().unwrap_or(Value::Null))
}

/// The live level. voxctl is voxel2's own tool, so it may name it.
const LEVEL: &str = "voxel_engine::level::LevelDef";

/// Reflect paths start at a field, so `materials[7].base[0]` needs a
/// leading dot. Accepted either way: nobody typing one at a shell prompt
/// wants to remember that, and the leading dot reads like a typo.
fn normalize_path(path: &str) -> String {
    if path.starts_with('.') || path.starts_with('[') {
        path.to_string()
    } else {
        format!(".{path}")
    }
}

/// Walk a reflect-style path through the JSON a resource read returns.
///
/// `get` and `set` must take the SAME path string, and they do not go
/// through the same machinery: `set` hands the path to `GetPath` on the
/// server, while a resource read has no path parameter at all and returns
/// the whole thing. So the walk here reproduces what a reflect path does,
/// which is not quite what the JSON looks like:
///
/// * An enum serializes as `{"Surface": {..}}`, but a reflect path reaches
///   the active variant's fields directly — so a lone variant wrapper is
///   transparent.
/// * `Option` is an enum whose `Some` is a TUPLE variant, so reflect wants
///   `.cover.0.full_at`; the serializer has already unwrapped it. A numeric
///   segment against a non-array is therefore that unwrap, not an index.
fn walk<'a>(root: &'a Value, path: &str) -> Result<&'a Value, String> {
    let mut node = root;
    let mut rest = path.trim_start_matches('.');
    while !rest.is_empty() {
        let (seg, next) = match rest.find(['.', '[']) {
            Some(0) if rest.starts_with('[') => {
                let end = rest.find(']').ok_or("unclosed `[`")?;
                (&rest[1..end], rest[end + 1..].trim_start_matches('.'))
            }
            Some(i) => (&rest[..i], rest[i..].trim_start_matches('.')),
            None => (rest, ""),
        };
        node = match (node, seg.parse::<usize>()) {
            (Value::Array(items), Ok(i)) => items.get(i).ok_or(format!("no index {i}"))?,
            // `Some`, which the serializer already unwrapped.
            (_, Ok(_)) => node,
            (Value::Object(map), Err(_)) if map.contains_key(seg) => &map[seg],
            // A variant wrapper stands between the path and the fields it
            // names. Stepped through only when the field is genuinely on
            // the other side of it — a one-field struct is not a wrapper,
            // and treating it as one walked straight past `cover`.
            (Value::Object(map), Err(_)) if map.len() == 1 => map
                .values()
                .next()
                .filter(|child| child.get(seg).is_some())
                .and_then(|child| child.get(seg))
                .ok_or(format!("no field `{seg}`"))?,
            _ => return Err(format!("cannot take `{seg}` of {node}")),
        };
        rest = next;
    }
    Ok(node)
}

fn parse_f64(s: &str) -> f64 {
    s.parse().unwrap_or_else(|_| {
        eprintln!("not a number: {s}");
        std::process::exit(2);
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let strs: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = match strs.as_slice() {
        ["status"] => call("voxel/status", Value::Null),
        ["goto", x, y, z] => call(
            "voxel/teleport",
            json!({"pos": [parse_f64(x), parse_f64(y), parse_f64(z)]}),
        ),
        ["goto", x, y, z, dx, dy, dz] => call(
            "voxel/teleport",
            json!({
                "pos": [parse_f64(x), parse_f64(y), parse_f64(z)],
                "look": [parse_f64(dx), parse_f64(dy), parse_f64(dz)],
            }),
        ),
        ["ribbons", x, z, rest @ ..] => {
            let radius = rest.first().map(|r| parse_f64(r)).unwrap_or(512.0);
            // The engine has no ribbons; `voxel/inspect` forwards the
            // question to whatever host is loaded, which decides what a
            // ribbon is and what to say about one.
            call(
                "voxel/inspect",
                json!({"kind": "ribbons", "center": [parse_f64(x), parse_f64(z)], "radius": radius}),
            )
        }
        ["markers", x, z, rest @ ..] => {
            let radius = rest.first().map(|r| parse_f64(r)).unwrap_or(2048.0);
            let mut params = json!({"kind": "markers", "center": [parse_f64(x), parse_f64(z)], "radius": radius});
            if let Some(of) = rest.get(1) {
                params["of"] = json!(of);
            }
            call("voxel/inspect", params)
        }
        ["scan", x, z, rest @ ..] => {
            let radius = rest.first().map(|r| parse_f64(r)).unwrap_or(4096.0);
            let step = rest.get(1).map(|s| parse_f64(s)).unwrap_or(32.0);
            call(
                "voxel/scan",
                json!({"center": [parse_f64(x), parse_f64(z)], "radius": radius, "step": step}),
            )
        }
        // Terrain height at a point, and a camera height safely above it.
        //
        // Use this before EVERY `goto` that picks a new xz. A camera
        // placed at a guessed y lands inside the ground about as often
        // as not, and an underground shot does not look like a bad
        // camera — it looks like a rendering bug, so the whole test is
        // wasted twice. `scan` is a scenic-spot ranker and answers a
        // different question.
        ["ground", x, z, rest @ ..] => {
            let eye = rest.first().map(|e| parse_f64(e)).unwrap_or(2.0);
            call(
                "voxel/inspect",
                json!({"kind": "ground", "x": parse_f64(x), "z": parse_f64(z), "eye": eye}),
            )
        }
        ["shot", path] => call("voxel/screenshot", json!({"path": path})),
        // What the PLAYER sees, not the offscreen mirror — a different
        // render path, and black while the window is backgrounded.
        ["shot", path, "--window"] => {
            call("voxel/screenshot", json!({"path": path, "window": true}))
        }
        // Opens (or moves) the portal in front of the camera. The engine
        // carries the command without knowing what a portal is; the host
        // decides.
        // Toggles the opening onto level N (default 0), the same as the
        // F1/F2/... keys. The engine carries the command without knowing
        // what a portal is; the host reads the queue.
        ["portal"] => call("voxel/host", json!({"cmd": "portal", "level": 0})),
        ["portal", n] => call(
            "voxel/host",
            json!({"cmd": "portal", "level": n.parse::<u64>().unwrap_or(0)}),
        ),
        ["world", w] => call("voxel/world", json!({"world": parse_f64(w) as u64})),
        ["inspect", params] => match serde_json::from_str(params) {
            Ok(p) => call("voxel/inspect", p),
            Err(e) => Err(format!("bad params JSON: {e}")),
        },
        // Read and write the LIVE level by field path. The engine applies
        // whatever writes the resource, so a set here takes the same route
        // a file edit does — a material is a table upload, a node change
        // rebuilds what that node reaches. Tuning a colour used to be a
        // relaunch.
        // A resource read takes no path, so the walk happens here. See
        // `walk` for why that is not the same as indexing the JSON.
        ["get", path] => call("world.get_resources", json!({"resource": LEVEL}))
            .and_then(|v| walk(v.get("value").unwrap_or(&v), path).cloned()),
        ["get"] => call("world.get_resources", json!({"resource": LEVEL})),
        ["set", path, value] => match serde_json::from_str::<Value>(value) {
            Ok(v) => call(
                "world.mutate_resources",
                json!({"resource": LEVEL, "path": normalize_path(path), "value": v}),
            ),
            // A bare word is the common case for an enum variant, and
            // quoting it through a shell is a papercut nobody needs.
            Err(_) => call(
                "world.mutate_resources",
                json!({"resource": LEVEL, "path": normalize_path(path), "value": value}),
            ),
        },
        ["raw", method] => call(method, Value::Null),
        ["raw", method, params] => match serde_json::from_str(params) {
            Ok(p) => call(method, p),
            Err(e) => Err(format!("bad params JSON: {e}")),
        },
        _ => {
            eprintln!(
                "usage: voxctl status | goto X Y Z [DX DY DZ] | ribbons X Z [R] | \
                 markers X Z [R] [KIND] | scan X Z [R] [STEP] | shot PATH [--window] | \
                 inspect JSON | get [PATH] | set PATH VALUE | \
                 portal [N] | world N | raw METHOD [JSON]"
            );
            std::process::exit(2);
        }
    };
    match result {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `get` and `set` must accept the same path string, and only one of
    /// them goes through `GetPath` on the server. These are the three
    /// places the JSON does not look like the reflect path that reaches it.
    #[test]
    fn a_reflect_path_walks_the_serialized_form() {
        let level = json!({
            "lod": {"split_k": 2.5},
            // An enum: reflect names the variant's fields directly.
            "materials": [{"Surface": {"base": [0.5, 0.25, 0.125]}}],
            // An Option: reflect wants `.0`, the serializer inlined it.
            "placements": [{"prefab": "monolith"}],
        });
        let at = |p: &str| walk(&level, p).unwrap().clone();

        assert_eq!(at("lod.split_k"), json!(2.5));
        assert_eq!(at(".lod.split_k"), json!(2.5), "a leading dot is optional");
        assert_eq!(at("materials[0].base[1]"), json!(0.25), "through a variant");
        assert_eq!(
            at("placements[0].prefab.0"),
            json!("monolith"),
            "through Some"
        );
        // Naming the variant explicitly still has to work: it is what the
        // JSON actually contains, and what someone reading a dump will try.
        assert_eq!(at("materials[0].Surface.base[0]"), json!(0.5));

        assert!(walk(&level, "lod.nope").is_err());
        assert!(walk(&level, "materials[9].base").is_err());
    }
}
