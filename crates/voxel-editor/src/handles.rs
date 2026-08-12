//! Drawing the authored shapes, and dragging them.
//!
//! [`crate::shapes`] decided all the geometry; this is the part that needs
//! a window. It draws every placement's ops as wireframes where they
//! actually are, picks the one under the pointer, and turns a drag into
//! the same [`crate::edit::Edit`] a row would have queued — so undo,
//! Cmd+S and the partial rebuild all work without knowing a handle exists.
//!
//! **What moves is the PREFAB.** A placement puts an object somewhere; the
//! handles change the object. Since a prefab file can back several levels,
//! [`ShapeTool::shared_with`] names the others, because an edit that
//! quietly changed a second level would be worse than one that says it is
//! about to.

use bevy::camera::RenderTarget;
use bevy::gizmos::config::{GizmoConfigGroup, GizmoConfigStore};
use bevy::prelude::*;
use voxel_core::csg::{CsgOp, CSG_KIND_CYLINDER_ADD, CSG_KIND_CYLINDER_CUT};
use voxel_engine::level::LevelDef;

use crate::edit::{Edit, Pending, Value};
use crate::shapes::{self, Drag, Handle, Ray};
use crate::walk::Num;
use crate::EditorState;

/// Its own gizmo group so the shapes can be styled — and switched off —
/// without touching the planning overlay's.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct ShapeGizmos;

/// What the 3D manipulator is doing.
///
/// Reflected, like [`EditorState`], so tooling can drive it: setting
/// `selected` and `nudge` over BRP performs exactly the write a drag
/// performs, which is the only way any of this is testable on a running
/// app.
#[derive(Resource, Reflect, Default, Debug)]
#[reflect(Resource)]
pub struct ShapeTool {
    /// Which prefab, and which of its ops, the handles are on.
    pub selected: Option<[usize; 2]>,
    /// Move the selected op by this many LOCAL meters, then clear.
    ///
    /// The pointer drives the same write through the same queue; this is
    /// how a script does it, and how a test does.
    pub nudge: [f32; 3],
    /// Grow the selected op by this many LOCAL meters, then clear.
    pub grow: [f32; 3],
    /// Levels other than the open one whose `prefabs` name the same file.
    /// Read-only; recomputed when the selection changes.
    pub shared_with: Vec<String>,
    /// The drag in flight, and the value it has reached. Not reflected:
    /// a pointer gesture half-restored from a snapshot would be a drag
    /// nobody is making.
    ///
    /// The value is held HERE rather than written to the document,
    /// because a placement edit restreams the world — 4.7 s on the
    /// planet — and doing that once a frame makes a drag impossible. The
    /// wireframe follows the pointer off this; the document learns about
    /// it once, on release. Which is what `CommitOnRelease` does for the
    /// rows, for the same reason.
    #[reflect(ignore)]
    drag: Option<Drag>,
    #[reflect(ignore)]
    preview: Option<[f32; 3]>,
}

/// Faint for context, bright for the one you are working on.
const IDLE: Color = Color::srgb(0.35, 0.42, 0.55);
const LIVE: Color = Color::srgb(1.0, 0.75, 0.2);
/// x, y, z — the usual three, so an axis reads without a legend.
const AXIS: [Color; 3] = [
    Color::srgb(0.95, 0.35, 0.35),
    Color::srgb(0.45, 0.95, 0.45),
    Color::srgb(0.40, 0.60, 1.0),
];

/// Every placement of the open level, in world space.
///
/// Through `voxel_engine::level::place`, which is what the world is
/// actually carved from — a second copy of that arithmetic would draw the
/// handles next to the shape rather than on it.
/// World 0, always: [`LevelDef`] IS world 0's document, and a portal
/// world is a different level this resource says nothing about.
fn placed(
    level: &LevelDef,
    worlds: &voxel_engine::Worlds,
    live: Option<(usize, usize, Handle, [f32; 3])>,
) -> Vec<(Option<usize>, Vec<CsgOp>)> {
    let Some(query) = worlds.query(0) else {
        return Vec::new();
    };
    let generator = query.generator();
    let ground = |xz: Vec2| generator.height(xz, 1.0);
    voxel_engine::level::authored(level)
        .map(|(p, local)| {
            // Which prefab these ops came from, so a click can name the
            // path to edit. An INLINE placement has none: it is drawn,
            // because it is there, but its handles would have nothing to
            // write to.
            let prefab = level
                .prefabs
                .iter()
                .position(|f| Some(&f.name) == p.prefab.as_ref());
            // A drag in flight is applied HERE, to a copy, so the shape
            // follows the pointer without the document — and therefore
            // the streamed world — hearing about every frame of it.
            match live {
                Some((f, op, handle, value)) if Some(f) == prefab => {
                    let mut local = local.to_vec();
                    if let Some(def) = local.get_mut(op) {
                        apply_local(def, handle, value);
                    }
                    (prefab, voxel_engine::level::place(p, &local, ground))
                }
                _ => (prefab, voxel_engine::level::place(p, local, ground)),
            }
        })
        .collect()
}

/// Which field of a shape a handle writes.
///
/// One decision, read twice: [`apply_local`] uses it to preview a drag and
/// [`write`] to commit one. Two copies of it would let the shape jump on
/// release, which is the one moment the two must agree.
enum Target {
    Center(usize),
    Half(usize),
    Radius,
    HalfHeight,
}

fn target(def: &voxel_engine::level::CsgOpDef, handle: Handle) -> Target {
    let axis = handle.axis();
    if !handle.is_size() {
        return Target::Center(axis);
    }
    if def.shape != "cylinder" {
        return Target::Half(axis);
    }
    // A cylinder is round: either horizontal handle is its radius.
    match axis {
        1 => Target::HalfHeight,
        _ => Target::Radius,
    }
}

/// The drag applied to a copy of the shape, for drawing it.
fn apply_local(def: &mut voxel_engine::level::CsgOpDef, handle: Handle, value: [f32; 3]) {
    match target(def, handle) {
        Target::Center(a) => def.center[a] = value[a],
        Target::Half(a) => def.half[a] = value[a],
        Target::Radius => def.radius = value[handle.axis()],
        Target::HalfHeight => def.half_height = value[1],
    }
}

/// Wireframes for every authored shape, and handles on the selected one.
pub fn draw(
    mut gizmos: Gizmos<ShapeGizmos>,
    state: Res<EditorState>,
    tool: Res<ShapeTool>,
    level: Option<Res<LevelDef>>,
    worlds: Option<Res<voxel_engine::Worlds>>,
    cameras: Cameras,
) {
    let (Some(level), Some(worlds)) = (level, worlds) else {
        return;
    };
    if !state.open {
        return;
    }
    let Some((_, eye)) = window_camera(&cameras) else {
        return;
    };
    let eye = eye.translation();

    let live = tool
        .drag
        .zip(tool.preview)
        .zip(tool.selected)
        .map(|((drag, value), [f, i])| (f, i, drag.handle, value));
    for (prefab, ops) in placed(&level, &worlds, live) {
        for (i, op) in ops.iter().enumerate() {
            let live = prefab.is_some_and(|f| tool.selected == Some([f, i]));
            outline(&mut gizmos, op, if live { LIVE } else { IDLE });
            if live {
                handles(&mut gizmos, op, eye);
            }
        }
    }
}

/// One shape, drawn as the shape it is.
///
/// Picking uses each op's bounding box, which is the right thing to aim
/// at; DRAWING one as a box is not — the monolith's 9.5 m base plate read
/// as a room-sized crate. A round thing gets a round outline.
fn outline(gizmos: &mut Gizmos<ShapeGizmos>, op: &CsgOp, color: Color) {
    let centre = Vec3::from(op.center);
    let half = Vec3::from(op.half);
    let turn = shapes::rotation(op.yaw);
    if !matches!(op.kind, CSG_KIND_CYLINDER_ADD | CSG_KIND_CYLINDER_CUT) {
        gizmos.cube(
            Transform::from_translation(centre)
                .with_rotation(turn)
                .with_scale(half * 2.0),
            color,
        );
        return;
    }
    // A cylinder's bounds carry its radius in x and z and its height in y.
    let radius = half.x;
    let flat = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    for end in [half.y, -half.y] {
        gizmos.circle(Isometry3d::new(centre + Vec3::Y * end, flat), radius, color);
    }
    // Four uprights, so it reads as a solid rather than two loose rings.
    for i in 0..4 {
        let a = std::f32::consts::FRAC_PI_2 * i as f32;
        let at = centre + Vec3::new(a.cos(), 0.0, a.sin()) * radius;
        gizmos.line(at + Vec3::Y * half.y, at - Vec3::Y * half.y, color);
    }
}

/// Six arrows: three to slide it, three to grow it.
fn handles(gizmos: &mut Gizmos<ShapeGizmos>, op: &CsgOp, eye: Vec3) {
    let centre = Vec3::from(op.center);
    for handle in Handle::ALL {
        let at = shapes::handle_at(op, handle);
        let color = AXIS[handle.axis()];
        gizmos.line(centre, at, color);
        // A size handle is a cube on the face; a move handle is a ball
        // beyond it. Two shapes rather than two colours, because the
        // colours are already saying which axis.
        let r = shapes::grab_radius(at.distance(eye));
        if handle.is_size() {
            gizmos.cube(
                Transform::from_translation(at).with_scale(Vec3::splat(r * 1.4)),
                color,
            );
        } else {
            gizmos.sphere(at, r, color);
        }
    }
}

/// The camera that draws into the window — not a portal's mirror, which
/// renders to a texture and would answer with a ray nobody is pointing.
///
/// In 0.19 the target is a component, and a camera without one draws to
/// the primary window: no component IS the window camera.
fn window_camera<'a>(cameras: &'a Cameras) -> Option<(&'a Camera, &'a GlobalTransform)> {
    cameras
        .iter()
        .find(|(c, _, target)| {
            c.is_active && target.is_none_or(|t| matches!(t, RenderTarget::Window(_)))
        })
        .map(|(c, t, _)| (c, t))
}

/// Every camera, and where it draws.
type Cameras<'w, 's> = Query<
    'w,
    's,
    (
        &'static Camera,
        &'static GlobalTransform,
        Option<&'static RenderTarget>,
    ),
>;

/// The ray under the pointer, or `None` when it is over the panel.
fn pointer_ray(cameras: &Cameras, windows: &Query<&Window>, panel_width: f32) -> Option<Ray> {
    let (camera, transform) = window_camera(cameras)?;
    let window = windows.iter().next()?;
    let cursor = window.cursor_position()?;
    // The panel is an absolute strip down the right edge; a drag that
    // started on the world must not keep steering when the pointer
    // crosses onto it.
    if cursor.x > window.width() - panel_width {
        return None;
    }
    let ray = camera.viewport_to_world(transform, cursor).ok()?;
    Some(Ray {
        origin: ray.origin,
        dir: *ray.direction,
    })
}

/// Click to select a shape, or to grab one of its handles.
#[allow(clippy::too_many_arguments)]
pub fn on_click(
    mouse: Res<ButtonInput<MouseButton>>,
    state: Res<EditorState>,
    mut tool: ResMut<ShapeTool>,
    level: Option<Res<LevelDef>>,
    worlds: Option<Res<voxel_engine::Worlds>>,
    cameras: Cameras,
    windows: Query<&Window>,
) {
    if !state.open || !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let (Some(level), Some(worlds)) = (level, worlds) else {
        return;
    };
    let Some(ray) = pointer_ray(&cameras, &windows, state.width) else {
        return;
    };
    let all = placed(&level, &worlds, None);

    // A handle on the CURRENT selection wins over selecting something
    // else: the handles stand outside the shape, and reaching for one
    // must not pick whatever is behind it.
    if let Some([prefab, i]) = tool.selected {
        if let Some(op) = all
            .iter()
            .find(|(p, _)| *p == Some(prefab))
            .and_then(|(_, ops)| ops.get(i))
        {
            if let Some(handle) = shapes::pick_handle(op, ray) {
                let axis = shapes::axes(op.yaw)[handle.axis()];
                if let Some(from) = shapes::along_axis(shapes::handle_at(op, handle), axis, ray) {
                    // The LOCAL value the drag starts from, so a long
                    // drag is one multiplication rather than an
                    // accumulation that drifts.
                    let start = local_start(&level, prefab, i, handle);
                    tool.drag = Some(Drag {
                        handle,
                        from,
                        start,
                    });
                }
                return;
            }
        }
    }

    // Otherwise select whatever is under the pointer, nearest first.
    let mut best: Option<(f32, [usize; 2])> = None;
    for (prefab, ops) in &all {
        // An inline placement is drawn but not selectable: its ops belong
        // to the placement, and the handles write a prefab's path.
        let Some(prefab) = *prefab else { continue };
        if let Some(i) = shapes::pick_op(ops, ray) {
            let d = Vec3::from(ops[i].center).distance(ray.origin);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, [prefab, i]));
            }
        }
    }
    tool.selected = best.map(|(_, s)| s);
    tool.drag = None;
}

/// The local value a drag starts from: whatever field its handle writes.
///
/// Through `target`, so which field that is gets decided once. A cylinder
/// carries its extents as a radius and a half height rather than three
/// numbers, and saying so twice is how the preview and the commit come to
/// disagree.
fn local_start(level: &LevelDef, prefab: usize, op: usize, handle: Handle) -> [f32; 3] {
    let Some(def) = level.prefabs.get(prefab).and_then(|p| p.ops.get(op)) else {
        return [0.0; 3];
    };
    match target(def, handle) {
        Target::Center(_) => def.center,
        Target::Half(_) => def.half,
        Target::Radius | Target::HalfHeight => [def.radius, def.half_height, def.radius],
    }
}

/// While the button is down, steer the selected op — and write NOTHING.
///
/// No `Pending` in this signature, on purpose. A placement edit restreams
/// the chunks it touches, and doing that once a frame floods the log and
/// makes the drag useless; the rule is "preview while held, commit on
/// release", the same one `CommitOnRelease` gives the panel's rows. It
/// was a comment once and the comment was wrong — a refactor left a
/// `write` in this path and nothing could catch it. Now the queue is not
/// reachable from here.
#[allow(clippy::too_many_arguments)]
pub fn preview_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    state: Res<EditorState>,
    mut tool: ResMut<ShapeTool>,
    level: Option<Res<LevelDef>>,
    worlds: Option<Res<voxel_engine::Worlds>>,
    cameras: Cameras,
    windows: Query<&Window>,
) {
    if !mouse.pressed(MouseButton::Left) {
        return;
    }
    let (Some(drag), Some([prefab, i])) = (tool.drag, tool.selected) else {
        return;
    };
    let (Some(level), Some(worlds)) = (level, worlds) else {
        return;
    };
    let Some(ray) = pointer_ray(&cameras, &windows, state.width) else {
        return;
    };
    // The UNDRAGGED shape: the grab was measured against it, so steering
    // off the preview would let the drag chase itself.
    let all = placed(&level, &worlds, None);
    let Some(op) = all
        .iter()
        .find(|(p, _)| *p == Some(prefab))
        .and_then(|(_, ops)| ops.get(i))
    else {
        return;
    };
    let scale = level
        .placements
        .iter()
        .find(|p| p.prefab.as_deref() == level.prefabs.get(prefab).map(|f| f.name.as_str()))
        .map_or(1.0, |p| p.scale);
    if let Some(value) = shapes::to_local(op, drag, ray, scale) {
        tool.preview = Some(value);
    }
}

/// Let go: the document hears about the whole drag, once.
pub fn commit_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    state: Res<EditorState>,
    mut tool: ResMut<ShapeTool>,
    level: Option<Res<LevelDef>>,
    mut pending: ResMut<Pending>,
) {
    // A button released without a drag in flight, or a drag abandoned
    // because the pointer left the world, both end the same way: forget
    // it. Only a drag that reached a value writes one.
    if mouse.pressed(MouseButton::Left) {
        return;
    }
    let taken = (tool.drag.take(), tool.preview.take());
    let (Some(drag), Some(value)) = taken else {
        return;
    };
    let (Some([prefab, i]), Some(level)) = (tool.selected, level) else {
        return;
    };
    write(
        &mut pending,
        &level,
        state.root,
        prefab,
        i,
        drag.handle,
        value,
    );
}

/// A nudge or a grow set from outside — the tooling path, and the tested
/// one. Applied once and cleared, so setting it twice moves it twice.
pub fn on_nudge(
    mut tool: ResMut<ShapeTool>,
    state: Res<EditorState>,
    level: Option<Res<LevelDef>>,
    mut pending: ResMut<Pending>,
) {
    let Some(level) = level else { return };
    let Some([prefab, i]) = tool.selected else {
        return;
    };
    const MOVES: [Handle; 3] = [Handle::MoveX, Handle::MoveY, Handle::MoveZ];
    const SIZES: [Handle; 3] = [Handle::SizeX, Handle::SizeY, Handle::SizeZ];
    for (delta, axes) in [(tool.nudge, MOVES), (tool.grow, SIZES)] {
        if delta == [0.0; 3] {
            continue;
        }
        let start = local_start(&level, prefab, i, axes[0]);
        let value = [
            start[0] + delta[0],
            start[1] + delta[1],
            start[2] + delta[2],
        ];
        // Queued together, so a nudge on two axes is one undo step.
        for (axis, handle) in axes.into_iter().enumerate() {
            if delta[axis] != 0.0 {
                write(&mut pending, &level, state.root, prefab, i, handle, value);
            }
        }
    }
    tool.nudge = [0.0; 3];
    tool.grow = [0.0; 3];
}

/// Queue the edit a handle makes, in the level's own vocabulary.
///
/// A move writes `center`; a size writes `half` on a box and
/// `radius`/`half_height` on a cylinder, which is the one place the
/// difference between those two shapes shows up here.
fn write(
    pending: &mut Pending,
    level: &LevelDef,
    root: usize,
    prefab: usize,
    op: usize,
    handle: Handle,
    value: [f32; 3],
) {
    let Some(def) = level.prefabs.get(prefab).and_then(|p| p.ops.get(op)) else {
        return;
    };
    let base = shapes::op_path(prefab, op);
    let (path, v) = match target(def, handle) {
        Target::Center(a) => (format!("{base}.center[{a}]"), value[a]),
        Target::Half(a) => (format!("{base}.half[{a}]"), value[a]),
        Target::Radius => (format!("{base}.radius"), value[handle.axis()]),
        Target::HalfHeight => (format!("{base}.half_height"), value[1]),
    };
    pending.0.push(Edit {
        root,
        path,
        value: Value::Num(v as f64, Num::F32),
    });
}

/// Which other levels use the selected prefab's file.
///
/// Read off the levels beside the open one, and only when the selection
/// moves: a prefab is shared by whoever names its path, and nothing in a
/// loaded level knows who else named it.
pub fn count_sharing(
    mut tool: ResMut<ShapeTool>,
    level: Option<Res<LevelDef>>,
    source: Option<Res<voxel_engine::level::LevelPath>>,
) {
    if !tool.is_changed() && level.as_ref().is_none_or(|l| !l.is_changed()) {
        return;
    }
    let (Some(level), Some(source)) = (level, source) else {
        return;
    };
    let Some([prefab, _]) = tool.selected else {
        tool.shared_with.clear();
        return;
    };
    let Some(rel) = level.prefabs.get(prefab).and_then(|p| p.from.as_deref()) else {
        tool.shared_with.clear();
        return;
    };
    tool.shared_with = voxel_engine::level::prefab::users(&source.0, rel);
}

/// Resources and gizmo styling. The systems are added by
/// [`crate::EditorPlugin`], in the same chain the rows use, so a drag's
/// edit lands before the panel is rebuilt from the document.
pub fn plugin(app: &mut App) {
    app.init_resource::<ShapeTool>()
        .register_type::<ShapeTool>()
        .init_gizmo_group::<ShapeGizmos>()
        .add_systems(Startup, configure);
}

fn configure(mut store: ResMut<GizmoConfigStore>) {
    let (config, _) = store.config_mut::<ShapeGizmos>();
    // Over the world: a handle you cannot see behind the shape it belongs
    // to is a handle you cannot use.
    config.depth_bias = -1.0;
    config.line.width = 2.0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_engine::level::{CsgOpDef, PrefabDef};

    /// A level with one prefab holding one box, which is all `write`
    /// needs to name a path.
    fn level() -> LevelDef {
        // Through the parser, because a level's serde defaults are not
        // its `Default` and a fixture built the other way is a level
        // nothing would load.
        let mut level = LevelDef::from_json(
            r#"{"lod":{"max_level":8,"top_radius":3,"top_y":[-1,0],
                "split_k":2.5,"merge_k":3.0},"nodes":[]}"#,
            &voxel_engine::graph::registry::engine_kinds(),
        )
        .unwrap();
        level.prefabs.push(PrefabDef {
            name: "thing".into(),
            ops: vec![CsgOpDef {
                shape: "box".into(),
                op: "add".into(),
                center: [0.0; 3],
                half: [1.0; 3],
                radius: 0.0,
                half_height: 0.0,
                yaw_deg: 0.0,
                material: voxel_engine::level::d_op_material(),
                blend: 0.0,
            }],
            from: None,
        });
        level
    }

    fn app(pressed: bool) -> App {
        let mut app = App::new();
        let mut mouse = ButtonInput::<MouseButton>::default();
        if pressed {
            mouse.press(MouseButton::Left);
        } else {
            // Pressed then released, so `just_released` is true and the
            // button reads as up — a real frame of letting go.
            mouse.press(MouseButton::Left);
            mouse.release(MouseButton::Left);
        }
        app.insert_resource(mouse)
            .insert_resource(EditorState::default())
            .insert_resource(level())
            .init_resource::<Pending>()
            .insert_resource(ShapeTool {
                selected: Some([0, 0]),
                drag: Some(Drag {
                    handle: Handle::MoveX,
                    from: 0.0,
                    start: [0.0; 3],
                }),
                preview: Some([7.0, 0.0, 0.0]),
                ..default()
            })
            .add_systems(Update, commit_drag);
        app
    }

    /// The rule the whole thing turns on: a placement edit restreams the
    /// chunks it touches, so a drag must write once, at the end.
    ///
    /// `preview_drag` cannot break this — it has no `Pending` to write
    /// to — so what is left to check is that letting go DOES write, and
    /// that holding on does not.
    #[test]
    fn a_held_drag_writes_nothing_and_releasing_writes_once() {
        let mut held = app(true);
        held.update();
        assert!(
            held.world().resource::<Pending>().0.is_empty(),
            "the document heard about a drag still in flight"
        );
        // And the drag is still live, so the next frame keeps steering.
        assert!(held.world().resource::<ShapeTool>().drag.is_some());

        let mut let_go = app(false);
        let_go.update();
        let queued = &let_go.world().resource::<Pending>().0;
        assert_eq!(queued.len(), 1, "one edit for the whole gesture");
        assert_eq!(queued[0].path, ".prefabs[0].ops[0].center[0]");
        // Cleared, so a second frame with the button up writes nothing.
        let_go.update();
        assert_eq!(let_go.world().resource::<Pending>().0.len(), 1);
    }
}
