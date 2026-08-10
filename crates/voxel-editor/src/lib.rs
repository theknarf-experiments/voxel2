//! A level editor built out of the level's own reflection.
//!
//! There is no widget code per field and no schema restated here. A
//! document is a reflected resource; [`walk`] turns it into rows by asking
//! `TypeInfo` what is in it and `voxel_engine::schema` how to show it, and
//! every row carries the reflect path it came from — the same path
//! `world.mutate_resources` uses over BRP. Editing is therefore one
//! observer rather than one per field, and a type this crate has never
//! heard of is editable the moment it is annotated.

use bevy::feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins};
use bevy::platform::collections::HashSet;
use bevy::prelude::*;

pub mod edit;
mod panel;
mod path;
mod style;
mod row;
mod walk;

pub use style::PanelStyle;
pub use walk::{rows, Num, Row, RowKind};

/// The key that opens the editor.
///
/// F1..F6 are the portal keys and F8/F9 the debug overlays, so this is the
/// first one free.
pub const TOGGLE_KEY: KeyCode = KeyCode::F10;

/// One editable document.
///
/// Reached through `ReflectResource`, so a root is any reflected resource
/// — the engine's [`LevelDef`] and a host's own parsed planning block are
/// the same kind of thing to this crate, which is what keeps its
/// vocabulary free of either one's nouns.
///
/// [`LevelDef`]: voxel_engine::LevelDef
#[derive(Clone)]
pub struct Root {
    /// Tab label.
    pub label: String,
    /// Fully-qualified type path of the resource.
    pub type_path: &'static str,
}

/// The documents this app lets the editor edit, in tab order.
#[derive(Resource, Default, Clone)]
pub struct EditorRoots(pub Vec<Root>);

/// Which root is showing, and which paths are open.
///
/// Expansion is by PATH rather than by entity: the rows are respawned
/// whenever anything changes, so state tied to a row would be lost every
/// time the document was edited — including by the edit that came from
/// that row.
///
/// Reflected so tooling can drive the editor the way it drives everything
/// else here — `voxctl raw world.mutate_resources` on `.open` opens the
/// panel, which is how an offscreen screenshot of it is taken at all.
#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct EditorState {
    pub open: bool,
    pub root: usize,
    pub expanded: HashSet<String>,
    /// Panel width in logical pixels, dragged by the grip on its inner
    /// edge. Here rather than on the node because the panel is respawned
    /// whenever the document changes.
    pub width: f32,
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            open: false,
            root: 0,
            // The root itself is always open, or the panel is one row.
            expanded: HashSet::from_iter([String::new()]),
            width: 620.0,
        }
    }
}

/// Draws [`EditorRoots`] as a panel over the running game.
#[derive(Default)]
pub struct EditorPlugin {
    roots: Vec<Root>,
}

impl EditorPlugin {
    /// Add a reflected resource as an editable document.
    ///
    /// `T` must be registered and carry `#[reflect(Resource)]`; without it
    /// the tab appears and is empty, which is reported at startup rather
    /// than left to be discovered.
    pub fn root<T: Resource + Reflect + TypePath>(mut self, label: impl Into<String>) -> Self {
        self.roots.push(Root {
            label: label.into(),
            type_path: T::type_path(),
        });
        self
    }
}

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<bevy::input_focus::tab_navigation::TabNavigationPlugin>() {
            app.add_plugins(FeathersPlugins);
        }
        // An EMPTY theme, not a missing one, is the untouched state:
        // `UiTheme` defaults to no colours at all and `UiTheme::color`
        // answers every token with fuchsia. Testing for the resource
        // instead of for its contents leaves the panel bright magenta,
        // which is exactly what it did.
        let unthemed = app
            .world()
            .get_resource::<UiTheme>()
            .is_none_or(|t| t.0.color.is_empty());
        if unthemed {
            app.insert_resource(UiTheme(create_dark_theme()));
        }

        app.insert_resource(EditorRoots(self.roots.clone()))
            .init_resource::<EditorState>()
            .init_resource::<edit::Pending>()
            .init_resource::<PanelStyle>()
            .register_type::<EditorState>()
            .register_type::<PanelStyle>()
            .add_observer(panel::on_disclosure)
            .add_observer(panel::on_grip_drag)
            .add_observer(edit::on_f32)
            .add_observer(edit::on_bool)
            .add_systems(
                Update,
                (
                    panel::toggle,
                    // Ordered: an edit queued this frame is applied before
                    // the panel is rebuilt from it, so a row never shows
                    // the value it had a frame after you changed it. Gated
                    // so a shut panel costs no frames at all — see
                    // `panel::active`.
                    (
                        edit::apply,
                        panel::rebuild,
                        // After the rebuild, so a panel respawned this
                        // frame gets the dragged width and not the one it
                        // was authored with.
                        panel::apply_width,
                        panel::on_wheel,
                    )
                        .chain()
                        .run_if(panel::active),
                )
                    .chain(),
            );
    }
}
