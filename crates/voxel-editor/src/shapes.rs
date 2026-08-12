//! Where the handles are: picking and dragging authored shapes in 3D.
//!
//! Pure arithmetic over [`CsgOp`], with no Bevy systems in it, for the
//! same reason [`crate::graph`] is: the part of a manipulator that can be
//! WRONG is the geometry, and the part that needs a window is the drawing.
//! Everything here is testable at a terminal.
//!
//! A prefab's ops are authored in LOCAL space and carved in world space,
//! and the difference is a placement's translate/yaw/scale
//! (`voxel_engine::level::place`). Handles are drawn and dragged in world
//! space because that is where you can see them; what an edit writes is
//! the local value, which is why every drag ends in [`to_local`].

use bevy::math::{Mat3, Quat, Vec2, Vec3};
use bevy::prelude::Reflect;
use voxel_core::csg::CsgOp;

/// A ray through the world, from the camera.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    /// Normalized.
    pub dir: Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// One draggable arrow: an axis of the shape's own frame.
///
/// Axes rather than planes or a trackball, because an authored object is
/// built out of boxes and cylinders that are square to their own yaw:
/// every edit anyone actually makes to one is along an axis of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Reflect, Default)]
pub enum Handle {
    /// Slide the shape along an axis.
    #[default]
    MoveX,
    MoveY,
    MoveZ,
    /// Grow the shape along an axis, about its own centre.
    SizeX,
    SizeY,
    SizeZ,
}

impl Handle {
    pub const ALL: [Handle; 6] = [
        Handle::MoveX,
        Handle::MoveY,
        Handle::MoveZ,
        Handle::SizeX,
        Handle::SizeY,
        Handle::SizeZ,
    ];

    /// Which axis of the shape's frame, as 0/1/2.
    pub fn axis(self) -> usize {
        match self {
            Handle::MoveX | Handle::SizeX => 0,
            Handle::MoveY | Handle::SizeY => 1,
            Handle::MoveZ | Handle::SizeZ => 2,
        }
    }

    pub fn is_size(self) -> bool {
        matches!(self, Handle::SizeX | Handle::SizeY | Handle::SizeZ)
    }
}

/// The shape's own axes in world space: yaw about Y, so X and Z turn and
/// Y does not.
///
/// The sense is `CsgOp::sdf`'s, which turns the QUERY POINT by `-yaw` and
/// so places the shape at `x' = x cos - z sin`. Getting that backwards
/// draws every yawed box mirrored, which is what it did.
pub fn axes(yaw: f32) -> [Vec3; 3] {
    let (sin, cos) = yaw.sin_cos();
    [Vec3::new(cos, 0.0, sin), Vec3::Y, Vec3::new(-sin, 0.0, cos)]
}

/// The shape's frame as a rotation, for drawing it.
///
/// Built FROM [`axes`] rather than from `Quat::from_rotation_y`, whose
/// sense is the opposite of the one the SDF uses. One convention in one
/// place, or the outline is mirrored and the handles point along axes the
/// shape has not got.
pub fn rotation(yaw: f32) -> Quat {
    let [x, y, z] = axes(yaw);
    Quat::from_mat3(&Mat3::from_cols(x, y, z))
}

/// Where a handle's grab point sits, in world space.
///
/// A size handle sits ON the face, which is the thing it moves. A move
/// handle stands just clear of it — proportional, with a floor, because
/// twice the half-extent put the arrow for a 9.5 m plate nineteen metres
/// out in the grass where it read as unrelated to the shape.
pub fn handle_at(op: &CsgOp, handle: Handle) -> Vec3 {
    let centre = Vec3::from(op.center);
    let half = Vec3::from(op.half);
    let axis = axes(op.yaw)[handle.axis()];
    let reach = half[handle.axis()].max(0.2);
    let out = if handle.is_size() {
        reach
    } else {
        reach * 1.25 + 0.5
    };
    centre + axis * out
}

/// How big a handle's grab sphere is at this distance from the eye.
///
/// Constant on SCREEN rather than in the world: a handle you can hit is
/// one you can see, and a shape 400 m away is a few pixels of it.
pub fn grab_radius(distance: f32) -> f32 {
    (distance * 0.02).clamp(0.05, 8.0)
}

/// The handle this ray grabs, nearest first, or `None`.
pub fn pick_handle(op: &CsgOp, ray: Ray) -> Option<Handle> {
    let mut best: Option<(f32, Handle)> = None;
    for handle in Handle::ALL {
        let at = handle_at(op, handle);
        let t = (at - ray.origin).dot(ray.dir);
        if t <= 0.0 {
            continue;
        }
        let miss = (ray.at(t) - at).length();
        if miss <= grab_radius(t) && best.is_none_or(|(bt, _)| t < bt) {
            best = Some((t, handle));
        }
    }
    best.map(|(_, h)| h)
}

/// Which op this ray hits first, as an index into `ops`.
///
/// Against each shape's own oriented box — a cylinder included, because
/// its bounds are what you are aiming at and half a metre of corner is
/// not worth a second intersector.
pub fn pick_op(ops: &[CsgOp], ray: Ray) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    for (i, op) in ops.iter().enumerate() {
        let Some(t) = hit_box(op, ray) else { continue };
        if best.is_none_or(|(bt, _)| t < bt) {
            best = Some((t, i));
        }
    }
    best.map(|(_, i)| i)
}

/// Ray against an oriented box: the slab test in the shape's own frame.
fn hit_box(op: &CsgOp, ray: Ray) -> Option<f32> {
    let axes = axes(op.yaw);
    let rel = ray.origin - Vec3::from(op.center);
    let half = Vec3::from(op.half);
    let (mut near, mut far) = (f32::NEG_INFINITY, f32::INFINITY);
    for a in 0..3 {
        let o = rel.dot(axes[a]);
        let d = ray.dir.dot(axes[a]);
        let h = half[a].max(1e-4);
        if d.abs() < 1e-6 {
            // Parallel to this pair of faces: a miss unless already
            // between them.
            if o.abs() > h {
                return None;
            }
            continue;
        }
        let (t0, t1) = ((-h - o) / d, (h - o) / d);
        let (t0, t1) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        near = near.max(t0);
        far = far.min(t1);
        if near > far {
            return None;
        }
    }
    // Standing inside it still counts as hitting it.
    if far < 0.0 {
        None
    } else {
        Some(near.max(0.0))
    }
}

/// How far along a handle's axis a drag has moved, in world meters.
///
/// The pointer ray and the handle axis are skew lines, so this takes the
/// point on the AXIS closest to the ray — the standard closest-approach
/// solution. Returns `None` when the two are near parallel, where that
/// point runs off to infinity and a drag would jump.
pub fn along_axis(anchor: Vec3, axis: Vec3, ray: Ray) -> Option<f32> {
    let w = anchor - ray.origin;
    let a = axis.dot(axis);
    let b = axis.dot(ray.dir);
    let c = ray.dir.dot(ray.dir);
    let d = axis.dot(w);
    let e = ray.dir.dot(w);
    let denom = a * c - b * b;
    if denom.abs() < 1e-5 {
        return None;
    }
    Some((b * e - c * d) / denom)
}

/// A drag in progress.
#[derive(Clone, Copy, Debug)]
pub struct Drag {
    pub handle: Handle,
    /// Where along the axis the grab started.
    pub from: f32,
    /// The op's local value when the drag began — centre for a move,
    /// half-extents for a size. Deltas apply to THIS rather than
    /// accumulating, so a drag cannot drift.
    pub start: [f32; 3],
}

/// The op's new local value after dragging to `ray`, or `None` if the
/// pointer is too near the axis to say.
///
/// `scale` is the placement's, because the drag is measured in world
/// meters and what gets written is local.
pub fn to_local(op: &CsgOp, drag: Drag, ray: Ray, scale: f32) -> Option<[f32; 3]> {
    let axis = axes(op.yaw)[drag.handle.axis()];
    let now = along_axis(handle_at(op, drag.handle), axis, ray)?;
    let world = now - drag.from;
    let local = world / if scale.abs() < 1e-4 { 1.0 } else { scale };
    let mut out = drag.start;
    let a = drag.handle.axis();
    if drag.handle.is_size() {
        // A shape with no thickness cannot be grabbed back, so a size
        // handle stops before it gets there rather than at zero.
        out[a] = (out[a] + local).max(0.01);
    } else {
        out[a] += local;
    }
    Some(out)
}

/// The reflect path of one op inside a level.
pub fn op_path(prefab: usize, op: usize) -> String {
    format!(".prefabs[{prefab}].ops[{op}]")
}

/// The xz of a placement, for asking the heightfield where the ground is.
pub fn ground_xz(position: [f32; 3]) -> Vec2 {
    Vec2::new(position[0], position[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxy(center: [f32; 3], half: [f32; 3], yaw: f32) -> CsgOp {
        CsgOp::boxy(Vec3::from(center), Vec3::from(half), yaw, 3, false)
    }

    fn down_x(from: Vec3) -> Ray {
        Ray {
            origin: from,
            dir: Vec3::NEG_X,
        }
    }

    #[test]
    fn a_ray_hits_the_box_it_points_at_and_misses_the_one_beside_it() {
        let ops = [
            boxy([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0),
            boxy([0.0, 0.0, 8.0], [1.0, 1.0, 1.0], 0.0),
        ];
        assert_eq!(pick_op(&ops, down_x(Vec3::new(10.0, 0.0, 0.0))), Some(0));
        assert_eq!(pick_op(&ops, down_x(Vec3::new(10.0, 0.0, 8.0))), Some(1));
        assert_eq!(pick_op(&ops, down_x(Vec3::new(10.0, 0.0, 4.0))), None);
    }

    /// Nearest wins, so clicking a stack of shapes selects the one you can
    /// actually see.
    #[test]
    fn the_nearest_box_is_the_one_picked() {
        let ops = [
            boxy([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0),
            boxy([5.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0),
        ];
        assert_eq!(pick_op(&ops, down_x(Vec3::new(20.0, 0.0, 0.0))), Some(1));
    }

    /// A yawed box is hit through its OWN faces.
    ///
    /// A long thin plank, quarter-turned so its length runs along z: a ray
    /// two metres off the centre line hits it turned and misses it
    /// straight, which no axis-aligned test could tell apart.
    #[test]
    fn a_yawed_box_is_picked_in_its_own_frame() {
        let half = [3.0, 1.0, 0.5];
        let straight = [boxy([0.0, 0.0, 0.0], half, 0.0)];
        let turned = [boxy([0.0, 0.0, 0.0], half, std::f32::consts::FRAC_PI_2)];
        let past_the_end = down_x(Vec3::new(10.0, 0.0, 2.0));
        assert_eq!(pick_op(&straight, past_the_end), None, "0.5 m deep in z");
        assert_eq!(pick_op(&turned, past_the_end), Some(0), "3 m long in z");
    }

    /// The rotation the outline is drawn with must put the box's corners
    /// where the SDF says solid is.
    ///
    /// Against `CsgOp::sdf` rather than against another formula, because
    /// the bug this pins WAS a second formula: drawing used
    /// `Quat::from_rotation_y(yaw)`, whose sense is the opposite of the
    /// one the SDF turns its query point by, so every yawed box was drawn
    /// mirrored about its own axis. Reverting `rotation` to that spelling
    /// fails this at turn 0.3.
    #[test]
    fn the_drawn_rotation_is_the_one_the_sdf_uses() {
        for turn in [0.3f32, 0.9, -1.2, std::f32::consts::FRAC_PI_4] {
            let half = Vec3::new(3.0, 1.0, 0.5);
            let op = boxy([0.0, 0.0, 0.0], half.to_array(), turn);
            let r = rotation(turn);
            // Well inside the drawn box, along each of its own axes.
            for corner in [
                Vec3::new(0.9, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.9),
                Vec3::new(0.9, 0.0, 0.9),
                Vec3::new(-0.9, 0.0, 0.9),
            ] {
                let at = r * (corner * half);
                assert!(
                    op.sdf(at) < 0.0,
                    "turn {turn}: {at:?} drawn inside, sdf says out"
                );
            }
            // And just past the long face is outside.
            let out = r * Vec3::new(half.x * 1.2, 0.0, 0.0);
            assert!(op.sdf(out) > 0.0, "turn {turn}: {out:?}");
        }
    }

    /// The drawn frame and the picked frame are one frame.
    #[test]
    fn the_rotation_agrees_with_the_axes() {
        for turn in [0.0f32, 0.7, -2.1] {
            let ax = axes(turn);
            let r = rotation(turn);
            for (i, unit) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
                assert!((r * unit - ax[i]).length() < 1e-5, "turn {turn} axis {i}");
            }
        }
    }

    #[test]
    fn handles_stand_off_the_faces_they_belong_to() {
        let op = boxy([0.0, 0.0, 0.0], [2.0, 1.0, 3.0], 0.0);
        assert_eq!(handle_at(&op, Handle::SizeX), Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(handle_at(&op, Handle::SizeZ), Vec3::new(0.0, 0.0, 3.0));
        // Just clear of the face it belongs to, not a multiple of it.
        let move_x = handle_at(&op, Handle::MoveX).x;
        assert!(move_x > 2.0 && move_x < 4.0, "{move_x}");
    }

    /// A yawed shape's handles turn with it, or they would pull along an
    /// axis the shape does not have.
    #[test]
    fn handles_turn_with_the_shape() {
        let op = boxy(
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 1.0],
            std::f32::consts::FRAC_PI_2,
        );
        let at = handle_at(&op, Handle::SizeX);
        assert!(at.z > 1.9 && at.x.abs() < 1e-5, "{at:?}");
    }

    #[test]
    fn a_handle_is_grabbed_when_the_ray_passes_near_it() {
        let op = boxy([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0);
        // Straight down the +X handle.
        let hit = Ray {
            origin: Vec3::new(20.0, 0.0, 0.0),
            dir: Vec3::NEG_X,
        };
        assert_eq!(pick_handle(&op, hit), Some(Handle::MoveX));
        // Well off to one side.
        let miss = Ray {
            origin: Vec3::new(20.0, 30.0, 0.0),
            dir: Vec3::NEG_X,
        };
        assert_eq!(pick_handle(&op, miss), None);
    }

    /// Dragging a move handle slides the shape by the distance the
    /// pointer travelled ALONG the axis, in local units.
    #[test]
    fn a_move_drag_slides_by_the_distance_along_the_axis() {
        let op = boxy([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0);
        let axis = axes(op.yaw)[0];
        let anchor = handle_at(&op, Handle::MoveX);
        let eye = Vec3::new(0.0, 20.0, 0.0);
        let start = along_axis(
            anchor,
            axis,
            Ray {
                origin: eye,
                dir: (anchor - eye).normalize(),
            },
        )
        .expect("the grab is on the axis");
        let drag = Drag {
            handle: Handle::MoveX,
            from: start,
            start: op.center,
        };
        // Point three metres further along +X.
        let target = anchor + Vec3::X * 3.0;
        let moved = to_local(
            &op,
            drag,
            Ray {
                origin: eye,
                dir: (target - eye).normalize(),
            },
            1.0,
        )
        .expect("not parallel");
        assert!((moved[0] - 3.0).abs() < 1e-3, "{moved:?}");
        assert_eq!([moved[1], moved[2]], [0.0, 0.0], "only its own axis moves");
    }

    /// A placement's scale is between the world the drag is measured in
    /// and the local value it writes.
    #[test]
    fn a_scaled_placement_writes_a_smaller_local_move() {
        let op = boxy([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0);
        let anchor = handle_at(&op, Handle::MoveX);
        let eye = Vec3::new(0.0, 20.0, 0.0);
        let axis = axes(op.yaw)[0];
        let from = along_axis(
            anchor,
            axis,
            Ray {
                origin: eye,
                dir: (anchor - eye).normalize(),
            },
        )
        .unwrap();
        let drag = Drag {
            handle: Handle::MoveX,
            from,
            start: op.center,
        };
        let target = anchor + Vec3::X * 4.0;
        let ray = Ray {
            origin: eye,
            dir: (target - eye).normalize(),
        };
        let at_one = to_local(&op, drag, ray, 1.0).unwrap();
        let at_two = to_local(&op, drag, ray, 2.0).unwrap();
        assert!(
            (at_one[0] - 2.0 * at_two[0]).abs() < 1e-3,
            "{at_one:?} {at_two:?}"
        );
    }

    /// A size handle cannot be dragged through zero into a mirrored shape
    /// nobody can grab back.
    #[test]
    fn a_size_drag_stops_before_nothing() {
        let op = boxy([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0);
        let drag = Drag {
            handle: Handle::SizeX,
            from: 0.0,
            start: op.half,
        };
        let eye = Vec3::new(0.0, 20.0, 0.0);
        // Far back along -X: a huge negative delta.
        let target = Vec3::new(-50.0, 0.0, 0.0);
        let sized = to_local(
            &op,
            drag,
            Ray {
                origin: eye,
                dir: (target - eye).normalize(),
            },
            1.0,
        )
        .unwrap();
        assert!(sized[0] >= 0.01, "{sized:?}");
    }

    /// Looking straight down an axis says nothing about position along
    /// it, and a drag that answered anyway would jump.
    #[test]
    fn a_drag_down_the_axis_declines() {
        let axis = Vec3::X;
        let ray = Ray {
            origin: Vec3::new(-10.0, 0.0, 0.0),
            dir: Vec3::X,
        };
        assert_eq!(along_axis(Vec3::ZERO, axis, ray), None);
    }
}
