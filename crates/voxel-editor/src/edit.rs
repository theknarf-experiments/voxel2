//! Widget → document.
//!
//! One observer per widget *value type*, not per field: a row carries the
//! reflect path it came from, so the observer's whole job is to look that
//! up and hand the value on. Adding a field to a level adds nothing here.
//!
//! Observers cannot reach an arbitrary reflected resource — that needs the
//! `World` and the type registry together — so they queue and an exclusive
//! system applies. The queue is also where undo will hang.

use bevy::prelude::*;
use bevy::ui_widgets::ValueChange;

use crate::path;
use crate::row::{CommitOnRelease, FieldPath, WritesNum};
use crate::walk::Num;
use crate::{EditorRoots, EditorState};

/// A value on its way into the document.
pub struct Edit {
    pub root: usize,
    pub path: String,
    pub value: Value,
}

pub enum Value {
    Num(f64, Num),
    Bool(bool),
}

/// Edits waiting for the exclusive system that can apply them.
#[derive(Resource, Default)]
pub struct Pending(pub Vec<Edit>);

/// A slider moved.
///
/// `is_final` gates the fields that restream the world: dragging emits a
/// value per frame, and rebuilding a streamed world at that rate makes the
/// drag useless and the change invisible. Which fields those are is the
/// LEVEL's declaration (`schema::Rebuilds`), never a guess here.
pub fn on_f32(
    change: On<ValueChange<f32>>,
    rows: Query<(&FieldPath, Option<&WritesNum>, Has<CommitOnRelease>)>,
    state: Res<EditorState>,
    mut pending: ResMut<Pending>,
) {
    let Ok((FieldPath(path), num, on_release)) = rows.get(change.event_target()) else {
        return;
    };
    if on_release && !change.is_final {
        return;
    }
    pending.0.push(Edit {
        root: state.root,
        path: path.clone(),
        value: Value::Num(change.value as f64, num.map_or(Num::F32, |n| n.0)),
    });
}

/// A checkbox or toggle switch flipped.
pub fn on_bool(
    change: On<ValueChange<bool>>,
    rows: Query<&FieldPath>,
    state: Res<EditorState>,
    mut pending: ResMut<Pending>,
) {
    let Ok(FieldPath(path)) = rows.get(change.event_target()) else {
        return;
    };
    pending.0.push(Edit {
        root: state.root,
        path: path.clone(),
        value: Value::Bool(change.value),
    });
}

/// Apply queued edits to the documents they name.
///
/// Exclusive because reaching a resource whose type is only known at
/// runtime needs the world and the registry at once — the same price the
/// panel pays to read it.
pub fn apply(world: &mut World) {
    // Read before taking: `resource_mut` marks the queue changed whether or
    // not there was anything in it.
    if world.resource::<Pending>().0.is_empty() {
        return;
    }
    let edits = std::mem::take(&mut world.resource_mut::<Pending>().0);
    let roots = world.resource::<EditorRoots>().clone();
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();

    for edit in edits {
        let Some(root) = roots.0.get(edit.root) else {
            continue;
        };
        let Some(reflect) = registry
            .get_with_type_path(root.type_path)
            .and_then(|r| r.data::<ReflectComponent>().map(|d| (d, r.type_id())))
        else {
            continue;
        };
        let (reflect, type_id) = reflect;
        let Some(component_id) = world.components().get_id(type_id) else {
            continue;
        };
        let Some(entity) = world.resource_entities().get(component_id) else {
            continue;
        };
        // A resource is an entity in 0.19, and going through `Mut` is what
        // marks it changed — which is the whole mechanism: `LevelDef`
        // changing is what makes `apply_level_change` rebuild the world.
        let Some(mut document) = reflect.reflect_mut(world.entity_mut(entity)) else {
            continue;
        };

        let slot = match path::resolve_mut(document.as_partial_reflect_mut(), &edit.path) {
            Ok(slot) => slot,
            // A path that does not resolve is a bug in the walk that made
            // it, not in the level. Say which one.
            Err(e) => {
                warn!("editor: {e}");
                continue;
            }
        };
        let wanted = typed(&edit.value);
        // An edit that changes nothing must not be applied.
        //
        // Not an optimisation — a correctness fix, and one any widget that
        // reports its own initial value will need again. Going through
        // `Mut` marks the document changed whether or not the bytes
        // differ, so a widget echoing back the value it was just given had
        // `apply_level_change` re-diffing and rebuilding the generator
        // EVERY FRAME with the panel merely open: 90 fps to 9.
        if slot.reflect_partial_eq(&*wanted) == Some(true) {
            continue;
        }
        if let Err(e) = slot.try_apply(&*wanted) {
            // `try_apply`, never `apply`: a type mismatch in a dev tool
            // must not take the session down with it.
            warn!("editor: '{}' would not take that value — {e}", edit.path);
        }
    }
}

/// The value at the field's own type.
///
/// Every numeric widget deals in `f32`; the field may be a `u8` level
/// count or a `usize` step budget, and `try_apply` refuses a mismatch
/// rather than rounding one into the other. Rounding here, once, is the
/// only place that conversion is allowed to happen.
fn typed(value: &Value) -> Box<dyn PartialReflect> {
    match *value {
        Value::Bool(b) => Box::new(b),
        Value::Num(v, num) => match num {
            Num::F32 => Box::new(v as f32),
            Num::F64 => Box::new(v),
            Num::U8 => Box::new(v.round().clamp(0.0, u8::MAX as f64) as u8),
            Num::U32 => Box::new(v.round().clamp(0.0, u32::MAX as f64) as u32),
            Num::I32 => Box::new(v.round() as i32),
            Num::Usize => Box::new(v.round().max(0.0) as usize),
        },
    }
}
