//! The panel: open it, and respawn its rows whenever what they show has
//! changed.
//!
//! Bevy's UI is retained and the document is not, so the panel is rebuilt
//! rather than diffed. Rebuilding is affordable because the walk only
//! descends into expanded paths — the rows on screen, not the several
//! thousand a level contains — and it is CORRECT, which diffing a tree
//! whose shape changes under it (a list gains an item, an enum changes
//! variant) is not without more machinery than the rebuild costs.

use bevy::feathers::constants::{fonts, size};
use bevy::feathers::containers::{pane, pane_body, pane_header};
use bevy::feathers::font_styles::InheritableFont;
use bevy::feathers::theme::ThemedText;
use bevy::prelude::*;
use bevy::text::FontWeight;
use bevy::ui::{px, percent, Display, FlexDirection, Node, Overflow, OverflowAxis, PositionType};

use crate::row;
use crate::walk;
use crate::{EditorRoots, EditorState, TOGGLE_KEY};

/// The panel's root entity. One at a time.
#[derive(Component)]
pub struct EditorPanel;

/// What the rows on screen were built from.
///
/// The panel respawns when this stops matching, so it has to be everything
/// the rows depend on: which document, what is open in it, and whether the
/// document itself changed.
#[derive(Resource, PartialEq)]
pub struct Shown {
    root: usize,
    expanded: Vec<String>,
    /// The change tick of the document resource the rows were read from.
    tick: u32,
    /// The panel entity, so finding it again is not a fresh `QueryState`
    /// built and matched against every archetype in the world — of which a
    /// streaming voxel world has a great many.
    entity: Entity,
}

/// Is there anything for the panel systems to do?
///
/// A closed dev tool has to cost NOTHING. These systems are exclusive, so
/// running them every frame splits the schedule at two more sync points,
/// and the frames that costs come out of the streamer's budget: `fly`
/// exhausted the mesh slabs on planet with the panel shut, because chunks
/// are released per frame and there were fewer frames to do it in.
pub fn active(state: Res<EditorState>, shown: Option<Res<Shown>>) -> bool {
    state.open || shown.is_some()
}

pub fn toggle(keys: Res<ButtonInput<KeyCode>>, mut state: ResMut<EditorState>) {
    if keys.just_pressed(TOGGLE_KEY) {
        state.open = !state.open;
    }
}

/// Open or close whatever a clicked disclosure names.
///
/// The widget is not told to update itself: the panel respawns from the
/// expansion set, so the set is the only state and the checkbox's own
/// `Checked` is rendered from it. Retained UI and a rebuilt panel agree
/// exactly when nothing is stored in both.
pub fn on_disclosure(
    change: On<bevy::ui_widgets::ValueChange<bool>>,
    toggles: Query<&row::TogglesPath>,
    mut state: ResMut<EditorState>,
) {
    let Ok(row::TogglesPath(path)) = toggles.get(change.event_target()) else {
        return;
    };
    if change.value {
        state.expanded.insert(path.clone());
    } else {
        state.expanded.remove(path);
    }
}

/// Respawn the panel when what it shows has changed.
///
/// An exclusive system because reading an arbitrary reflected resource
/// needs the whole `World` and the type registry at once — the price of
/// the panel not knowing what it is showing.
pub fn rebuild(world: &mut World) {
    let open = world.resource::<EditorState>().open;
    if !open {
        if let Some(shown) = world.remove_resource::<Shown>() {
            world.entity_mut(shown.entity).despawn();
        }
        return;
    }

    let Some((label, tick, rows)) = read_document(world) else {
        return;
    };

    let state = world.resource::<EditorState>();
    let mut expanded: Vec<String> = state.expanded.iter().cloned().collect();
    expanded.sort();
    let mut wanted = Shown {
        root: state.root,
        expanded,
        tick,
        entity: Entity::PLACEHOLDER,
    };
    // The entity is an OUTPUT of this rebuild, not part of what decides
    // whether one is needed.
    if let Some(shown) = world.get_resource::<Shown>() {
        wanted.entity = shown.entity;
        if *shown == wanted {
            return;
        }
        world.entity_mut(shown.entity).despawn();
    }
    let header = format!("{label}  —  {} rows  (F10)", rows.len());
    // `impl SceneList for Vec<S: Scene>` is what lets a panel whose shape
    // is only known at runtime be expressed in a static macro.
    let row_scenes: Vec<_> = rows.iter().map(row::scene).collect();

    let panel = world
        .spawn_scene(bsn! {
            pane()
            // Docked to the right edge, full height. The debug HUD lives
            // in the top-left corner and the two used to overlap.
            Node {
                position_type: PositionType::Absolute,
                right: px(0),
                top: px(0),
                bottom: px(0),
                width: percent(40),
                min_width: px(560),
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
            }
            Children [
                (pane_header() Children [ (Text({header}) ThemedText) ]),
                (
                    pane_body()
                    // Feathers propagates `TextFont` only from an ancestor
                    // carrying this; without it every row rendered at
                    // Bevy's default 20px and a level's worth of them was
                    // three screens tall.
                    InheritableFont {
                        font: fonts::MONO,
                        font_size: size::SMALL_FONT,
                        weight: FontWeight::NORMAL,
                    }
                    Node {
                        // The body takes the rest of the height and
                        // scrolls, or a level with an open section is a
                        // list that runs off the bottom of the screen.
                        flex_grow: 1.0,
                        min_height: px(0),
                        flex_direction: FlexDirection::Column,
                        // Clip across, scroll down: a long doc line must
                        // not paint over the world beside the panel.
                        overflow: Overflow { x: OverflowAxis::Clip, y: OverflowAxis::Scroll },
                    }
                    Children [ {row_scenes} ]
                ),
            ]
        })
        .map(|e| e.id());
    match panel {
        Ok(panel) => {
            world.entity_mut(panel).insert(EditorPanel);
            wanted.entity = panel;
            world.insert_resource(wanted);
        }
        // A scene that fails to resolve is a bug in this crate, not in the
        // level; keeping the old `Shown` makes it retry rather than sit
        // silently on an empty panel.
        Err(e) => warn_once!("editor panel did not spawn: {e}"),
    }

}

/// Read the selected document and walk it into rows.
///
/// A resource is an entity in Bevy 0.19 and `ReflectResource` is only a
/// marker implying `ReflectComponent`, so this reads the document off that
/// entity — no raw pointer and no `unsafe` to reach a resource whose type
/// is not known until runtime.
///
/// Returns `None` when the root is not a registered reflected resource.
/// Reported, because a tab that is silently empty is exactly the failure
/// this crate exists to stop making.
fn read_document(world: &mut World) -> Option<(String, u32, Vec<walk::Row>)> {
    let index = world.resource::<EditorState>().root;
    let root = world.resource::<EditorRoots>().0.get(index)?.clone();
    let expanded = world.resource::<EditorState>().expanded.clone();

    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let Some(registration) = registry.get_with_type_path(root.type_path) else {
        warn_once!(
            "editor root '{}' is not in the type registry — nothing to show",
            root.type_path
        );
        return None;
    };
    let Some(reflect) = registration.data::<ReflectComponent>() else {
        warn_once!(
            "editor root '{}' is registered but not as a resource — it needs \
             #[reflect(Resource)]",
            root.type_path
        );
        return None;
    };

    let component_id = world.components().get_id(registration.type_id())?;
    let tick = world
        .get_resource_change_ticks_by_id(component_id)?
        .changed
        .get();
    let entity = world.resource_entities().get(component_id)?;
    let value = reflect.reflect(world.entity(entity))?;

    let rows = walk::rows(value.as_partial_reflect(), &expanded);
    Some((root.label, tick, rows))
}
