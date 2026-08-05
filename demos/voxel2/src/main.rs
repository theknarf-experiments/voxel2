//! voxel2: loads and presents a JSON level definition.
//!
//!     cargo run -p voxel2 -- levels/planet.json
//!     cargo run -p voxel2 -- levels/megastructure.json

use bevy::prelude::*;
use voxel_debug::prelude::*;
use voxel_engine::{LevelDef, LevelPlugin};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "levels/planet.json".to_string());
    let json = match std::fs::read_to_string(&path) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("failed to read level '{path}': {e}");
            std::process::exit(1);
        }
    };
    let level = match LevelDef::from_json(&json) {
        Ok(level) => level,
        Err(e) => {
            eprintln!("failed to parse level '{path}': {e}");
            std::process::exit(1);
        }
    };

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: format!("voxel2 — {}", level.name),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((
            VoxelDebugPlugin,
            LevelPlugin {
                def: level,
                source: Some(path.into()),
            },
        ))
        .run();
}
