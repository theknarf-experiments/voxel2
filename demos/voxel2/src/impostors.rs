//! Tree impostors: what this demo feeds `bevy_impostors` and where it
//! takes the look from.
//!
//! The renderer — silhouette mesh, shader, buffers, draw — is the
//! crate's. What is left here is what only this game knows: which
//! scatter class is the far forest, what silhouette each prop MODEL
//! reads as at a distance, and that the impostors' palette and reach are
//! really somebody else's numbers.

use bevy::prelude::*;
use bevy::render::Extract;
use bevy_impostors::{
    ImpostorSet, ImpostorStyle, ImpostorVariantStyle, Impostors, IMPOSTOR_FADE_FROM,
    MAX_IMPOSTOR_VARIANTS,
};

use crate::prop_worlds::{WorldProp, WorldPropPlugin};

/// The impostor population standing in for the far forest.
pub struct Trees;

impl ImpostorSet for Trees {
    const NAME: &'static str = "tree_impostor";
    const LAYOUT_LABEL: &'static str = "tree_impostor_layout";
    const DRAW_LABEL: &'static str = "tree_impostor_draw";
}

pub type TreeImpostors = Impostors<Trees>;

/// The scatter population this demo draws as tree impostors — the SAME
/// class the near entity trees are drawn from, which is the point: one
/// placement draw, so every real tree stands where an impostor stands,
/// at the same variant.
///
/// Each variant's canopy colour is TAKEN from the class's prop table
/// rather than authored again. They were authored twice, and the two
/// drifted: the impostors carried hand-converted linear values that had
/// lost a third of their green, so a stand handed over to real trees
/// that were a different species of green. One palette, and retuning the
/// props retunes the forest behind them.
pub const IMPOSTOR_CLASS: &str = "tree";

impl WorldProp for TreeImpostors {
    const CLASS: &'static str = IMPOSTOR_CLASS;

    fn anchor(world: voxel_engine::WorldId) -> Self {
        Impostors::new(u32::from(world))
    }
}

pub struct ImpostorPlugin;

impl Plugin for ImpostorPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            bevy_impostors::ImpostorPlugin::<Trees>::default(),
            WorldPropPlugin::<TreeImpostors>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
            return;
        };
        render_app.add_systems(ExtractSchedule, sync_impostor_style);
    }
}

/// What each prop model reads as at impostor range.
///
/// This demo's half of the variant table: the CRATE knows silhouettes,
/// the PROPS know models, and this is the one place the two vocabularies
/// meet. A new model gets a row here and every impostor population picks
/// it up; a model without one falls back to a plain diamond.
fn silhouette(model: crate::props::Model, color: Vec4) -> ImpostorVariantStyle {
    use crate::props::Model;
    match model {
        Model::Conifer => ImpostorVariantStyle::pointed(color),
        Model::Broadleaf => ImpostorVariantStyle::waisted(color),
        // Narrow and a little taller than the others, like its mesh.
        Model::Birch => ImpostorVariantStyle {
            color,
            shape: Vec4::new(0.5, 0.18, 1.15, 0.0),
        },
        // Low clumps: squat diamonds, wider than tall.
        Model::Bush | Model::Rock => ImpostorVariantStyle {
            color,
            shape: Vec4::new(0.5, 0.55, 0.30, 0.0),
        },
        _ => ImpostorVariantStyle::waisted(color),
    }
}

/// Take the impostors' variant table from the props they stand in for,
/// and their reach from the ground that paints them.
///
/// Both are the middle tier's whole job: it has a real forest on one side
/// and a painted one on the other, and it is the only thing that can be
/// wrong about either. Neither number is authored here, because both are
/// really somebody else's — see [`IMPOSTOR_CLASS`] for the palette.
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
    mut style: ResMut<ImpostorStyle<Trees>>,
) {
    // Row i of the table IS variant i of the prop class: the placement's
    // variant byte indexes both, which is what keeps an impostor and the
    // tree it becomes the same species. By INDEX here, not by model —
    // the alignment between the level's variants and the prop table's is
    // already this demo's invariant for entity spawning.
    for table in props.0.values() {
        let Some(class) = table.0.get(IMPOSTOR_CLASS) else {
            continue;
        };
        for (i, v) in class
            .variants
            .iter()
            .take(MAX_IMPOSTOR_VARIANTS)
            .enumerate()
        {
            style.variants[i] = silhouette(v.model, v.foliage.to_linear().to_vec4());
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
