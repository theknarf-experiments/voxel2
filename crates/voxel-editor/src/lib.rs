//! A level editor built out of the level's own reflection.
//!
//! There is no widget code per field and no schema restated here. A
//! document is a reflected resource; [`walk`] turns it into rows by asking
//! `TypeInfo` what is in it and `voxel_engine::schema` how to show it, and
//! every row carries the reflect path it came from — the same path
//! `world.mutate_resources` uses over BRP. Editing is therefore one
//! observer per value TYPE rather than one per field, and a type this
//! crate has never heard of is editable the moment it is annotated.
//!
//! What a value can be told to become: a number drags (at the speed its
//! field declares, or one percent of itself), a bool toggles, a reference
//! offers what the document actually contains, and an enum offers its own
//! variants — switching one builds the new variant's fields from the
//! registry. A colour opens into the numbers it is made of. Text and the
//! structure of a document — adding, removing, reordering — are not
//! editable yet, and the rows say so by being inert rather than by
//! pretending.
//!
//! Two views. [`View::Rows`] is the generic one, and works on anything.
//! [`View::Graph`] is a picture of `voxel_engine::graph` specifically —
//! boxes, ports, wires, a frame per scope — because a graph view needs
//! ports and wires and those belong to that language. Selecting a box
//! shows its fields in the same column any other selection uses.
//!
//! Three things guard the document. Edits queue and apply through one
//! exclusive system, which is where [`edit::History`] takes its snapshot,
//! so undo is a whole document rather than an inverse per widget. The
//! panel refuses to respawn while a widget is being dragged, or the drag
//! would despawn its own widget. And a document that does not compile is
//! shown with the compiler's complaint across the top, because the panel
//! is now able to make one.

use bevy::feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins};
use bevy::platform::collections::HashSet;
use bevy::prelude::*;

pub mod canvas;
pub mod edit;
pub mod graph;
pub mod handles;
mod panel;
mod path;
mod row;
pub mod shapes;
mod style;
mod walk;

pub use canvas::{GraphViewport, SelectsNode, PLAIN_BORDER};
pub use edit::{on_drag, on_drag_done, on_pick, on_typed, undo, History};
pub use panel::on_tab;
pub use panel::{hover, on_pinch, on_select, on_wheel, save};
pub use row::{DragsNum, FieldPath, PicksOption, SelectsRoot, WritesNum};
pub use style::PanelStyle;
pub use walk::{rows, rows_at, rows_in, Num, Row, RowKind};

/// The panel asked to save what it is editing.
///
/// Generic on purpose: this crate edits any reflected resource and has no
/// idea what a document IS, let alone where it came from. The host says
/// what saving means — for a level, `voxel_engine::level::SaveLevel`.
#[derive(Message, Debug, Default, Clone, Copy)]
pub struct SaveRequested;

/// The key that opens the editor.
///
/// F1..F6 are the portal keys and F8/F9 the debug overlays, so this is the
/// first one free.
pub const TOGGLE_KEY: KeyCode = KeyCode::F10;

/// One editable document.
///
/// Reached through `ReflectResource`, so a root is any reflected resource
/// — the engine's [`LevelDef`] and a host's own settings are the same kind
/// of thing to this crate, which is what keeps its vocabulary free of
/// either one's nouns.
///
/// [`LevelDef`]: voxel_engine::LevelDef
#[derive(Clone)]
pub struct Root {
    /// Tab label.
    pub label: String,
    /// Fully-qualified type path of the resource.
    pub type_path: &'static str,
    /// Which of the document's top-level sections this tab shows.
    pub sections: Sections,
    /// How this tab draws them.
    pub view: View,
}

/// How a tab draws its document.
///
/// The rows work on ANY annotated type; the graph is a view of
/// `voxel_engine::graph` specifically, because a picture of a graph needs
/// ports, wires and scopes and those belong to that language.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum View {
    #[default]
    Rows,
    /// The node list as a picture, with the selected node's own fields
    /// beside it. [`Sections`] says what the LEVEL's own box holds — the
    /// parts of the document that are not nodes.
    Graph,
}

/// Which top-level fields a tab shows.
///
/// A tab is a VIEW of a document, not a document: a level's node list is
/// most of what there is to edit and nothing like the rest of it, so the
/// two want separate tabs while remaining one resource, one change tick
/// and one set of reflect paths.
#[derive(Clone, PartialEq, Debug, Default)]
pub enum Sections {
    #[default]
    All,
    Only(Vec<String>),
    Except(Vec<String>),
}

impl Sections {
    /// Does this tab show the document's `field` section?
    pub fn shows(&self, field: &str) -> bool {
        match self {
            Sections::All => true,
            Sections::Only(names) => names.iter().any(|n| n == field),
            Sections::Except(names) => !names.iter().any(|n| n == field),
        }
    }
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
    /// Where the graph view is looked at from.
    pub camera: canvas::GraphCamera,
    /// Reflect path of the node the graph view is inspecting, if any.
    /// A path rather than an entity: the graph is respawned whenever the
    /// document changes, and the same node is the same path afterwards.
    pub selected: Option<String>,
    /// Set to ask for a save; cleared as soon as the ask is sent. A flag
    /// rather than only a keystroke so tooling can drive it, which is also
    /// the only way this is testable on a running panel.
    pub save: bool,
    /// Set to put the document back as it was before the last edit.
    pub undo: bool,
    /// Set to put back what [`EditorState::undo`] took away.
    pub redo: bool,
    /// The document has been edited HERE since the last save was asked
    /// for. Optimistic — the host does the saving and says if it failed —
    /// but it is the difference between knowing your work is on disk and
    /// guessing.
    pub edited: bool,
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
            camera: canvas::GraphCamera::default(),
            selected: None,
            save: false,
            undo: false,
            redo: false,
            edited: false,
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
            sections: Sections::All,
            view: View::Rows,
        });
        self
    }

    /// Draw the root just added as a 2D graph rather than as rows.
    pub fn as_graph(mut self) -> Self {
        match self.roots.last_mut() {
            Some(root) => root.view = View::Graph,
            None => warn!("editor: a view declared before any root — ignored"),
        }
        self
    }

    /// Show only these sections of the root just added.
    pub fn only(self, sections: &[&str]) -> Self {
        self.sections(Sections::Only(
            sections.iter().map(|s| s.to_string()).collect(),
        ))
    }

    /// Show everything of the root just added EXCEPT these sections.
    ///
    /// Paired with [`EditorPlugin::only`] on another tab, this is how a
    /// document is split without either half having to list the other's
    /// contents — add a section to the level and it appears in the tab
    /// that did not name it.
    pub fn except(self, sections: &[&str]) -> Self {
        self.sections(Sections::Except(
            sections.iter().map(|s| s.to_string()).collect(),
        ))
    }

    fn sections(mut self, sections: Sections) -> Self {
        match self.roots.last_mut() {
            Some(root) => root.sections = sections,
            None => warn!("editor: sections declared before any root — ignored"),
        }
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

        app.add_message::<SaveRequested>()
            .insert_resource(EditorRoots(self.roots.clone()))
            .init_resource::<EditorState>()
            .init_resource::<edit::Pending>()
            .init_resource::<panel::Dragging>()
            .init_resource::<edit::History>()
            .init_resource::<PanelStyle>()
            .init_resource::<graph::GraphStyle>()
            .add_plugins(handles::plugin)
            .register_type::<EditorState>()
            .register_type::<PanelStyle>()
            .register_type::<graph::GraphStyle>()
            .add_observer(panel::on_disclosure)
            .add_observer(panel::on_tab)
            .add_observer(panel::on_select)
            .add_observer(panel::on_grip_drag)
            .add_observer(panel::on_drag_start)
            .add_observer(panel::on_drag_end)
            .add_observer(edit::on_f32)
            .add_observer(edit::on_bool)
            .add_observer(edit::on_pick)
            .add_observer(edit::on_drag)
            .add_observer(edit::on_drag_done)
            .add_systems(
                Update,
                (
                    panel::toggle,
                    panel::save,
                    // The 3D manipulator queues into the same `Pending`
                    // the rows do, so it sits in the same chain and gets
                    // undo, saving and the partial rebuild for nothing.
                    (
                        handles::on_click,
                        handles::preview_drag,
                        handles::commit_drag,
                        handles::on_nudge,
                        handles::count_sharing,
                    )
                        .chain()
                        .before(edit::apply)
                        .run_if(panel::active),
                    // Ordered: an edit queued this frame is applied before
                    // the panel is rebuilt from it, so a row never shows
                    // the value it had a frame after you changed it. Gated
                    // so a shut panel costs no frames at all — see
                    // `panel::active`.
                    (
                        edit::on_typed,
                        edit::apply,
                        edit::undo,
                        panel::rebuild,
                        panel::apply_camera,
                        // After the rebuild, so a panel respawned this
                        // frame gets the dragged width and not the one it
                        // was authored with.
                        panel::apply_width,
                        panel::hover,
                        panel::on_wheel,
                        panel::on_pinch,
                        // Last: what it draws is the document as it is
                        // AFTER this frame's edit landed.
                        handles::draw,
                    )
                        .chain()
                        .run_if(panel::active),
                )
                    .chain(),
            );
    }
}
