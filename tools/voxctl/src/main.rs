//! voxctl — drive a running voxel2 instance over the Bevy Remote
//! Protocol (start the game with `VOXEL_REMOTE=1`). Interactive
//! verification without relaunching: teleport, query planning data,
//! dump offscreen screenshots.
//!
//! Usage:
//!   voxctl status
//!   voxctl goto X Y Z [DX DY DZ]
//!   voxctl water X Z [RADIUS]
//!   voxctl markers X Z [RADIUS] [KIND]
//!   voxctl shot PATH
//!   voxctl raw METHOD [PARAMS_JSON]
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
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .map_err(|e| format!("connect 127.0.0.1:{port}: {e} (is the game running with VOXEL_REMOTE=1?)"))?;
    let http = format!(
        "POST / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request}",
        request.len()
    );
    stream.write_all(http.as_bytes()).map_err(|e| e.to_string())?;
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
        ["water", x, z, rest @ ..] => {
            let radius = rest.first().map(|r| parse_f64(r)).unwrap_or(512.0);
            call(
                "voxel/water",
                json!({"center": [parse_f64(x), parse_f64(z)], "radius": radius}),
            )
        }
        ["markers", x, z, rest @ ..] => {
            let radius = rest.first().map(|r| parse_f64(r)).unwrap_or(2048.0);
            let mut params = json!({"center": [parse_f64(x), parse_f64(z)], "radius": radius});
            if let Some(kind) = rest.get(1) {
                params["kind"] = json!(kind);
            }
            call("voxel/markers", params)
        }
        ["shot", path] => call("voxel/screenshot", json!({"path": path})),
        ["raw", method] => call(method, Value::Null),
        ["raw", method, params] => match serde_json::from_str(params) {
            Ok(p) => call(method, p),
            Err(e) => Err(format!("bad params JSON: {e}")),
        },
        _ => {
            eprintln!(
                "usage: voxctl status | goto X Y Z [DX DY DZ] | water X Z [R] | \
                 markers X Z [R] [KIND] | shot PATH | raw METHOD [JSON]"
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
