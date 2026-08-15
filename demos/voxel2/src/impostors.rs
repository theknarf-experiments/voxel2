//! Tree impostors: what this demo feeds `bevy_impostors` and where it
//! takes the look from.
//!
//! The renderer — silhouette mesh, shader, buffers, draw — is the crate's.
//! What is left here is what only this game knows: which scatter class is
//! the far forest, and that the impostors' palette and reach are really
//! somebody else's numbers.

use bevy::prelude::*;
use bevy::render::Extract;
use bevy_impostors::{ImpostorStyle, Impostors, IMPOSTOR_FADE_FROM};

use crate::prop_worlds::{WorldProp, WorldPropPlugin};

/// The scatter population this demo draws as tree impostors. Just a name
/// the level and the demo agree on — the engine never sees it.
pub const IMPOSTOR_CLASS: &str = "treecover";

/// The prop class these impostors stand in for.
///
/// Their canopy colours are TAKEN from its variants rather than authored
/// again. They were authored twice, and the two drifted: the impostors
/// carried hand-converted linear values that had lost a third of their
/// green, so a stand handed over to real trees that were a different
/// species of green. One palette, and retuning the props retunes the
/// forest behind them.
pub const IMPOSTOR_STANDS_IN_FOR: &str = "tree";

impl WorldProp for Impostors {
    const CLASS: &'static str = IMPOSTOR_CLASS;

    fn anchor(world: voxel_engine::WorldId) -> Self {
        Self {
            set: u32::from(world),
        }
    }
}

pub struct ImpostorPlugin;

impl Plugin for ImpostorPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            bevy_impostors::impostor::ImpostorPlugin,
            WorldPropPlugin::<Impostors>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
            return;
        };
        render_app.add_systems(ExtractSchedule, sync_impostor_style);
    }
}

/// Take the impostors' colour from the trees they stand in for, and their
/// reach from the ground that paints them.
///
/// Both are the middle tier's whole job: it has a real forest on one side
/// and a painted one on the other, and it is the only thing that can be
/// wrong about either. Neither number is authored here, because both are
/// really somebody else's — see [`IMPOSTOR_STANDS_IN_FOR`] for the palette.
///
/// For the reach: the two tiers meet at one distance and it is the level's
/// to choose, but only one of them can honour the number as written. Paint
/// is decided per CHUNK, so it begins at whatever LOD boundary follows;
/// culling at the authored distance left a ring of ground with impostors
/// gone and paint not yet arrived. So the impostors are told where the
/// paint really starts and fade out ACROSS it — the texel grid appears
/// while they are still at full strength and is never seen bare.
fn sync_impostor_style(
    worlds: Extract<Res<voxel_engine::Worlds>>,
    props: Extract<Res<crate::WorldProps>>,
    mut style: ResMut<ImpostorStyle>,
) {
    // The cone silhouette is a conifer and the diamond is a broadleaf, so
    // each takes that prop variant's foliage. Matched by MODEL rather than
    // by index: which variant a level lists first is level dressing, and
    // reading it positionally would swap the two species the day someone
    // reorders them.
    for table in props.0.values() {
        let Some(class) = table.0.get(IMPOSTOR_STANDS_IN_FOR) else {
            continue;
        };
        let foliage = |model| {
            class
                .variants
                .iter()
                .find(|v| v.model == model)
                .map(|v| v.foliage.to_linear().to_vec4())
        };
        if let Some(c) = foliage(crate::props::Model::Conifer) {
            style.color_a = c;
        }
        if let Some(c) = foliage(crate::props::Model::Broadleaf) {
            style.color_b = c;
        }
    }

    // Every frame rather than on change: this is a fold over at most
    // `MAX_WORLDS` levels, and a change tick compared across worlds is a
    // subtlety to get wrong for no saving.
    let reach = worlds
        .iter()
        .filter_map(|world| {
            let def = world
                .query
                .planner_as::<crate::planning::RegionPlanner>()?
                .populations
                .iter()
                .find(|def| def.class == IMPOSTOR_CLASS)?;
            let from = def.cover.as_ref()?.from_m;
            Some(crate::surface_paint::cover_starts_m(&world.config, from) / IMPOSTOR_FADE_FROM)
        })
        .fold(f32::NEG_INFINITY, f32::max);
    if reach.is_finite() {
        style.size.z = reach;
    }
}
