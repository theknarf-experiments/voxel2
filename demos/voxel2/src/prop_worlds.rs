//! How this demo's worlds feed `bevy_impostors` populations.
//!
//! The crate draws instanced props per SET and has no idea what a set is;
//! here a set is a world. This module is the translation: one marker per
//! loaded world on that world's render layer, and each population's
//! points bridged from the engine's [`ScatterPoints`] classes into the
//! crate's [`PropPoints`]. The engine never sees the class names and the
//! crate never sees the worlds.

use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
use bevy_impostors::{PropFlags, PropPoints};
use std::marker::PhantomData;
use voxel_render::ScatterPoints;

/// A [`bevy_impostors::Prop`] this demo feeds from a scatter class, one
/// marker per world.
pub trait WorldProp: bevy_impostors::Prop {
    /// The scatter class the level publishes this population's points
    /// under. A name the level and the demo agree on — the engine never
    /// interprets it.
    const CLASS: &'static str;
    fn anchor(world: voxel_engine::WorldId) -> Self;
}

/// The host half of a population: anchors and points. The render half is
/// the crate's `PropPlugin`, which the population's own plugin adds.
pub struct WorldPropPlugin<P>(PhantomData<fn() -> P>);

impl<P> Default for WorldPropPlugin<P> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<P: WorldProp> Plugin for WorldPropPlugin<P> {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScatterPoints>()
            .add_systems(Update, (sync_anchors::<P>, bridge_points::<P>));

        // The flags sync is one system for ALL populations — they share
        // the [`PropFlags`] resource — so only the first of these plugins
        // adds it.
        if app.world().contains_resource::<FlagsSyncAdded>() {
            return;
        }
        app.insert_resource(FlagsSyncAdded);
        if let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) {
            render_app.add_systems(ExtractSchedule, sync_prop_flags);
        }
    }
}

/// See [`WorldPropPlugin`]: the flags sync must be added exactly once.
#[derive(Resource)]
struct FlagsSyncAdded;

/// Give every loaded world an anchor. Not a `Startup` one-shot: a world
/// can arrive at any time, because opening a portal loads one.
fn sync_anchors<P: WorldProp>(
    mut commands: Commands,
    worlds: Res<voxel_engine::Worlds>,
    // Bookkeeping, not a query: a spawn is not visible to a query until
    // commands apply, so two changes to `Worlds` before the flush would
    // give one world two markers and draw its props twice.
    mut spawned: Local<std::collections::HashSet<voxel_engine::WorldId>>,
) {
    if !worlds.is_changed() {
        return;
    }
    for world in worlds.iter() {
        if !spawned.insert(world.id) {
            continue;
        }
        commands.spawn((
            P::anchor(world.id),
            crate::OfWorld::scene(world.id),
            Visibility::default(),
            Transform::default(),
            Aabb {
                center: Vec3A::ZERO,
                half_extents: Vec3A::splat(1.0e9),
            },
        ));
    }
}

/// Hand one class's points across the seam whenever the streamer
/// republishes it. Both sides are a position and a hash; only the names
/// differ.
fn bridge_points<P: WorldProp>(scatter: Res<ScatterPoints>, points: Res<PropPoints<P>>) {
    let Some(per_world) = scatter.take_class_if_dirty(P::CLASS) else {
        return;
    };
    points.replace(per_world.into_iter().map(|(world, pts)| {
        (
            u32::from(world),
            pts.iter()
                .map(|p| bevy_impostors::PropInstance {
                    pos: p.pos,
                    hash: p.hash,
                })
                .collect(),
        )
    }));
}

/// Every population folds [`voxel_render::EnvParams::flags`] into its
/// uniform through the crate's [`PropFlags`]; this keeps the two equal.
fn sync_prop_flags(env: Res<voxel_render::EnvParams>, mut flags: ResMut<PropFlags>) {
    if env.is_changed() {
        flags.0 = env.flags;
    }
}
