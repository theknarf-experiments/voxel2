//! A second level, live alongside the first.
//!
//! The point of a portal is that both worlds exist at once — you can see
//! into one from the other and walk between them — so the far side cannot
//! be a level swap. It is registered as its own world: its own generator,
//! its own LOD field, its own anchor, streaming continuously whether or
//! not you are looking at it.
//!
//! Worlds share coordinates. World 1 is not "over there"; it is the same
//! space, and which one you see is `CameraWorld`. That is why only the
//! camera's world is drawn — two levels rendered together would interleave
//! two solids rather than show two places.

use bevy::prelude::*;
use voxel_engine::{LevelDef, StreamedWorlds};
use voxel_render::WorldPrograms;

/// Coarsest level the far world streams while it has no portal to be
/// seen through. See `register_far_world`.
const FAR_MAX_LEVEL: u8 = 5;

/// Level file for the far side, if the host asked for one.
#[derive(Resource)]
pub struct FarLevel(pub LevelDef);

pub struct PortalPlugin;

impl Plugin for PortalPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, register_far_world);
    }
}

/// Register the far level as world 1: generator into the shared program
/// buffer, LOD field into the streamer.
fn register_far_world(
    far: Option<Res<FarLevel>>,
    mut worlds: ResMut<StreamedWorlds>,
    mut programs: ResMut<WorldPrograms>,
    mut materials: ResMut<voxel_render::WorldMaterials>,
    near: Res<LevelDef>,
) {
    let Some(far) = far else {
        return;
    };
    let level = &far.0;
    let generator = std::sync::Arc::new(level.generator(0));
    // INTERIM, and the number that matters most here: a second world
    // roughly doubles the meshed working set, and the slab is one fixed
    // GPU allocation sized for one world — at the far level's own config
    // every class hit 100% and 427 chunks wedged in AwaitingAlloc.
    //
    // The real fix is the portal: the far side only needs to be resident
    // where you can SEE it, which is a cone through a surface, not a
    // sphere around the camera. Until that exists, hold it to a small
    // field so the thing streams and settles.
    let mut config = voxel_engine::LodConfig::from(&level.lod);
    config.max_level = config.max_level.min(FAR_MAX_LEVEL);
    config.top_radius = 1;
    let id = worlds.add(config, generator.clone());
    programs.0.push(voxel_render::WorldProgram {
        ops: std::sync::Arc::new(generator.ops().to_vec()),
        seed: generator.seed(),
        sun_dir: generator.sun_direction(),
    });
    // One material table serves every world. It is INDEXED by the id a
    // chunk's vertex carries, and ids are level data, so two levels that
    // coexist simply must not reuse one — planet is 1/3/4, the
    // megastructure is 2. A collision would silently repaint one world
    // with the other's recipe, so it is worth the assert.
    for def in &level.materials {
        let id = def.id() as usize;
        assert!(
            id < materials.0.len(),
            "far level material id {id} is out of range",
        );
        assert!(
            !near.materials.iter().any(|n| n.id() == def.id()),
            "far level reuses material id {id}, which the near level already defines",
        );
        materials.0[id] = def.pack();
    }
    info!("far level registered as world {id}");
}
