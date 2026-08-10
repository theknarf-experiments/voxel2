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
use bevy::reflect::{TypeInfo, TypeRegistry};
use bevy::text::EditableText;
use bevy::ui_widgets::ValueChange;

use crate::path;
use crate::row::{CommitOnRelease, DragsNum, FieldPath, PicksOption, WritesNum};
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
    Text(String),
    /// A different variant of the enum at this path. What the variant
    /// CONTAINS is built here, from the registry — see [`variant`].
    Variant(String),
}

/// Edits waiting for the exclusive system that can apply them.
#[derive(Resource, Default)]
pub struct Pending(pub Vec<Edit>);

/// Documents as they were before each batch of edits MADE HERE.
///
/// A value poked over BRP or a level reloaded from disk is not this
/// crate's to undo: they do not come through the queue, and a tool that
/// silently reverted somebody else's write would be worse than one that
/// admits it only owns its own.
///
/// Whole snapshots rather than inverse operations: an inverse has to know
/// what every widget MEANS, and a switched enum variant has no inverse at
/// all — the fields it replaced are gone. A document is small, and this is
/// a dev tool.
///
/// The dynamic form, because bevy can `reflect_clone` neither `[f32; 3]`
/// nor a boxed trait object, and a level is full of both. `try_apply`
/// takes it back.
#[derive(Resource)]
pub struct History {
    done: Vec<Step>,
    undone: Vec<Step>,
    /// Enough to undo a session's worth of tuning without holding every
    /// document a long session ever produced.
    pub depth: usize,
}

/// One document, as it was.
type Step = (usize, Box<dyn PartialReflect>);

impl Default for History {
    fn default() -> Self {
        Self {
            done: Vec::new(),
            undone: Vec::new(),
            depth: 64,
        }
    }
}

impl History {
    /// Remember the state before a batch, and forget any redo: the future
    /// that was undone is not the future of a document that has since been
    /// edited down a different path.
    fn remember(&mut self, step: Step) {
        self.done.push(step);
        let over = self.done.len().saturating_sub(self.depth);
        self.done.drain(..over);
        self.undone.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.done.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.undone.is_empty()
    }
}

/// Commit what has been typed, on Enter.
///
/// Not per keystroke. A name is a REFERENCE — the only way anything refers
/// to a node — so a half-typed one refers to nothing, and every
/// intermediate would be a document that does not compile and an entry in
/// the undo stack.
pub fn on_typed(
    keys: Res<ButtonInput<KeyCode>>,
    focus: Res<bevy::input_focus::InputFocus>,
    typed: Query<(&EditableText, &ChildOf)>,
    fields: Query<&FieldPath>,
    state: Res<EditorState>,
    mut pending: ResMut<Pending>,
) {
    if !keys.just_pressed(KeyCode::Enter) {
        return;
    }
    let Some(at) = focus.get() else { return };
    let Ok((text, parent)) = typed.get(at) else {
        return;
    };
    // The path is on the CONTAINER: the editable part is a child, which is
    // where focus lands.
    let Ok(FieldPath(path)) = fields.get(parent.parent()) else {
        return;
    };
    pending.0.push(Edit {
        root: state.root,
        path: path.clone(),
        value: Value::Text(text.value().to_string()),
    });
}

/// A number was dragged sideways.
///
/// The value is `from + distance * speed`, where the distance is the
/// drag's OWN total: a dropped frame or a re-entrant observer cannot make
/// it drift away from the pointer, the way accumulating deltas would.
pub fn on_drag(
    drag: On<Pointer<Drag>>,
    rows: Query<(
        &FieldPath,
        &DragsNum,
        Option<&WritesNum>,
        Has<CommitOnRelease>,
    )>,
    state: Res<EditorState>,
    mut pending: ResMut<Pending>,
) {
    let Ok((FieldPath(path), drags, num, on_release)) = rows.get(drag.event_target()) else {
        return;
    };
    // A field that restreams the world waits for the pointer to be let
    // go; dragging it would rebuild the streamed world once a frame.
    if on_release {
        return;
    }
    pending
        .0
        .push(dragged(&state, path, drags, num, drag.distance.x));
}

/// The end of a drag, for the fields that wait for it.
pub fn on_drag_done(
    drag: On<Pointer<DragEnd>>,
    rows: Query<(
        &FieldPath,
        &DragsNum,
        Option<&WritesNum>,
        Has<CommitOnRelease>,
    )>,
    state: Res<EditorState>,
    mut pending: ResMut<Pending>,
) {
    let Ok((FieldPath(path), drags, num, on_release)) = rows.get(drag.event_target()) else {
        return;
    };
    if !on_release {
        return;
    }
    pending
        .0
        .push(dragged(&state, path, drags, num, drag.distance.x));
}

fn dragged(
    state: &EditorState,
    path: &str,
    drags: &DragsNum,
    num: Option<&WritesNum>,
    distance: f32,
) -> Edit {
    Edit {
        root: state.root,
        path: path.to_string(),
        value: Value::Num(
            (drags.from + distance * drags.speed) as f64,
            num.map_or(Num::F32, |n| n.0),
        ),
    }
}

/// A reference was picked from its menu.
///
/// The row said what the field may hold — the level's own material ids,
/// its prefabs, its node names — so this only has to write the one that
/// was chosen. Whether it lands as a number or as text is the FIELD's
/// business, carried on the item: an id is spelled as a number and a name
/// as a string, and the menu that offers them is the same menu.
pub fn on_pick(
    activate: On<bevy::ui_widgets::Activate>,
    picks: Query<&PicksOption>,
    state: Res<EditorState>,
    mut pending: ResMut<Pending>,
) {
    let Ok(pick) = picks.get(activate.event_target()) else {
        return;
    };
    if pick.variant {
        pending.0.push(Edit {
            root: state.root,
            path: pick.path.clone(),
            value: Value::Variant(pick.value.clone()),
        });
        return;
    }
    let value = match pick.num {
        Some(num) => match pick.value.parse::<f64>() {
            Ok(v) => Value::Num(v, num),
            // The options come from the document, so a numeric reference
            // whose option is not a number is a bug in the pattern that
            // enumerated it, not in the level.
            Err(_) => {
                warn!("editor: '{}' is not a number", pick.value);
                return;
            }
        },
        None => Value::Text(pick.value.clone()),
    };
    pending.0.push(Edit {
        root: state.root,
        path: pick.path.clone(),
        value,
    });
}

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
/// Step the document back — or forward again.
///
/// Exclusive for the same reason `apply` is: reaching an arbitrary
/// reflected resource needs the `World` and the registry together.
pub fn undo(world: &mut World) {
    let state = world.resource::<EditorState>();
    let (undo, redo) = (state.undo, state.redo);
    if !(undo || redo) {
        return;
    }
    {
        let mut state = world.resource_mut::<EditorState>();
        state.undo = false;
        state.redo = false;
    }
    let mut history = world.resource_mut::<History>();
    let step = if undo {
        history.done.pop()
    } else {
        history.undone.pop()
    };
    let Some((root, was)) = step else {
        info!("editor: nothing to {}", if undo { "undo" } else { "redo" });
        return;
    };
    let Some(mut document) = document_mut(world, root) else {
        return;
    };
    // What is being replaced becomes the way back.
    let now = document.as_partial_reflect().to_dynamic();
    if let Err(e) = document.as_partial_reflect_mut().try_apply(was.as_ref()) {
        warn!("editor: that did not take — {e}");
        return;
    }
    let mut history = world.resource_mut::<History>();
    if undo {
        history.undone.push((root, now));
    } else {
        history.done.push((root, now));
    }
}

/// The document a root names, ready to be written.
fn document_mut(world: &mut World, root: usize) -> Option<Mut<'_, dyn Reflect>> {
    let roots = world.resource::<EditorRoots>().clone();
    let root = roots.0.get(root)?;
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let registration = registry.get_with_type_path(root.type_path)?;
    let reflect = registration.data::<ReflectComponent>()?;
    let component_id = world.components().get_id(registration.type_id())?;
    let entity = world.resource_entities().get(component_id)?;
    // A resource is an entity in 0.19, and going through `Mut` is what
    // marks it changed — which is the whole mechanism.
    reflect.reflect_mut(world.entity_mut(entity))
}

pub fn apply(world: &mut World) {
    // Read before taking: `resource_mut` marks the queue changed whether or
    // not there was anything in it.
    if world.resource::<Pending>().0.is_empty() {
        return;
    }
    let edits = std::mem::take(&mut world.resource_mut::<Pending>().0);
    // One step per BATCH, not per edit: a drag queues one a frame and
    // undoing it a pixel at a time would be its own kind of unusable.
    let mut remember = true;
    let mut history: Vec<(usize, Box<dyn PartialReflect>)> = Vec::new();
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
        // Before the first edit of this batch touches it.
        if remember {
            remember = false;
            let snapshot = document.as_partial_reflect().to_dynamic();
            history.push((edit.root, snapshot));
        }

        // Renaming a node is not a field write. A name is the only way
        // anything refers to a node, so writing one without its references
        // is the same edit as deleting the node — see `graph::rename`.
        if let Value::Text(to) = &edit.value {
            if let Some(renamed) = rename_node(&mut *document, &edit.path, to) {
                if renamed > 0 {
                    info!("editor: renamed, and {renamed} wires followed");
                }
                continue;
            }
        }
        let slot = match path::resolve_mut(document.as_partial_reflect_mut(), &edit.path) {
            Ok(slot) => slot,
            // A path that does not resolve is a bug in the walk that made
            // it, not in the level. Say which one.
            Err(e) => {
                warn!("editor: {e}");
                continue;
            }
        };
        let wanted = match &edit.value {
            // A variant needs its fields, and only the slot knows what
            // type it is. Everything else is the value as given.
            Value::Variant(name) => match variant(slot, name, &registry) {
                Ok(built) => built,
                Err(e) => {
                    warn!("editor: '{}' cannot become {name} — {e}", edit.path);
                    continue;
                }
            },
            value => typed(value),
        };
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
    for step in history {
        world.resource_mut::<History>().remember(step);
    }
}

/// If `path` names a NODE's name, rename it and every wire that said it.
///
/// `None` when the path is some other string — a class, a marker kind —
/// which is an ordinary field write.
fn rename_node(document: &mut dyn Reflect, path: &str, to: &str) -> Option<usize> {
    // `.nodes[3].name.0`, and nothing else. A scope's children live at
    // `.nodes[3].node.nodes[1].name.0`, which ends the same way.
    let suffix = ".name.0";
    if !path.starts_with(".nodes[") || !path.ends_with(suffix) {
        return None;
    }
    let level = document
        .as_any_mut()
        .downcast_mut::<voxel_engine::LevelDef>()?;
    let from = level
        .as_reflect()
        .reflect_path(path)
        .ok()?
        .try_downcast_ref::<String>()?
        .clone();
    if from == to {
        return Some(0);
    }
    Some(voxel_engine::graph::rename(&mut level.nodes, &from, to))
}

/// The value at the field's own type.
///
/// Every numeric widget deals in `f32`; the field may be a `u8` level
/// count or a `usize` step budget, and `try_apply` refuses a mismatch
/// rather than rounding one into the other. Rounding here, once, is the
/// only place that conversion is allowed to happen.
/// Build the named variant of the enum at `slot`, with every field at its
/// type's default.
///
/// Switching a recipe is a real edit — a `zoned` material is not a
/// `surface` one with different numbers — so the new variant arrives
/// EMPTY rather than carrying whatever happened to be in the old one at
/// the same position. A field whose type the registry cannot default is a
/// refusal with the type named, never a half-built value.
fn variant(
    slot: &dyn PartialReflect,
    name: &str,
    registry: &TypeRegistry,
) -> Result<Box<dyn PartialReflect>, String> {
    use bevy::reflect::enums::{DynamicEnum, DynamicVariant, VariantInfo};
    use bevy::reflect::structs::DynamicStruct;
    use bevy::reflect::tuple::DynamicTuple;
    let Some(TypeInfo::Enum(info)) = slot.get_represented_type_info() else {
        return Err("it is not an enum".into());
    };
    let Some(want) = info.variant(name) else {
        return Err("no such variant".into());
    };
    let default = |ty: &'static bevy::reflect::TypeInfo| empty(ty, registry);
    let built = match want {
        VariantInfo::Unit(_) => DynamicVariant::Unit,
        VariantInfo::Tuple(v) => {
            let mut fields = DynamicTuple::default();
            for i in 0..v.field_len() {
                let field = v.field_at(i).ok_or("missing field")?;
                fields.insert_boxed(default(field.type_info().ok_or("unknown field type")?)?);
            }
            DynamicVariant::Tuple(fields)
        }
        VariantInfo::Struct(v) => {
            let mut fields = DynamicStruct::default();
            for i in 0..v.field_len() {
                let field = v.field_at(i).ok_or("missing field")?;
                fields.insert_boxed(
                    field.name(),
                    default(field.type_info().ok_or("unknown field type")?)?,
                );
            }
            DynamicVariant::Struct(fields)
        }
    };
    let mut out = DynamicEnum::new(name, built);
    out.set_represented_type(slot.get_represented_type_info());
    Ok(Box::new(out))
}

/// A value of `ty` with nothing in it.
///
/// The type's own `Default` where it has one, and otherwise built from the
/// SHAPE the registry describes: bevy registers no `ReflectDefault` for
/// `[f32; 3]`, and refusing to switch a material recipe over a colour that
/// is three zeroes either way would be an odd place to stop.
fn empty(
    ty: &'static bevy::reflect::TypeInfo,
    registry: &TypeRegistry,
) -> Result<Box<dyn PartialReflect>, String> {
    use bevy::reflect::array::DynamicArray;
    use bevy::reflect::list::DynamicList;
    use bevy::reflect::map::DynamicMap;
    use bevy::reflect::set::DynamicSet;
    use bevy::reflect::structs::DynamicStruct;
    use bevy::reflect::tuple::DynamicTuple;
    use bevy::reflect::tuple_struct::DynamicTupleStruct;

    if let Some(default) =
        registry.get_type_data::<bevy::reflect::std_traits::ReflectDefault>(ty.type_id())
    {
        return Ok(default.default().into_partial_reflect());
    }
    let field = |ty: Option<&'static bevy::reflect::TypeInfo>| {
        ty.ok_or_else(|| "a field of unknown type".to_string())
            .and_then(|ty| empty(ty, registry))
    };
    Ok(match ty {
        TypeInfo::Struct(info) => {
            let mut out = DynamicStruct::default();
            for i in 0..info.field_len() {
                let f = info.field_at(i).ok_or("missing field")?;
                out.insert_boxed(f.name(), field(f.type_info())?);
            }
            Box::new(out)
        }
        TypeInfo::TupleStruct(info) => {
            let mut out = DynamicTupleStruct::default();
            for i in 0..info.field_len() {
                out.insert_boxed(field(info.field_at(i).ok_or("missing field")?.type_info())?);
            }
            Box::new(out)
        }
        TypeInfo::Tuple(info) => {
            let mut out = DynamicTuple::default();
            for i in 0..info.field_len() {
                out.insert_boxed(field(info.field_at(i).ok_or("missing field")?.type_info())?);
            }
            Box::new(out)
        }
        TypeInfo::Array(info) => {
            let mut items = Vec::with_capacity(info.capacity());
            for _ in 0..info.capacity() {
                items.push(field(info.item_info())?);
            }
            Box::new(DynamicArray::new(items.into_boxed_slice()))
        }
        // Empty is the right answer for anything that HOLDS things.
        TypeInfo::List(_) => Box::new(DynamicList::default()),
        TypeInfo::Map(_) => Box::new(DynamicMap::default()),
        TypeInfo::Set(_) => Box::new(DynamicSet::default()),
        TypeInfo::Enum(_) | TypeInfo::Opaque(_) => {
            return Err(format!("{} has no default", ty.type_path()))
        }
    })
}

fn typed(value: &Value) -> Box<dyn PartialReflect> {
    match *value {
        Value::Text(ref s) => Box::new(s.clone()),
        // Handled at the slot, which is the only thing that knows the type.
        Value::Variant(_) => unreachable!("a variant is built from the slot"),
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
