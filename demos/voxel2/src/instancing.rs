//! Per-world instance buffers for the point populations.
//!
//! Grass and tree impostors are the same shape — a class of
//! [`ScatterPoint`]s the streamer republishes as the camera moves, drawn
//! as one instanced call per world. What they share here is the BUFFER,
//! and specifically that it grows instead of being reallocated: a forest
//! of half a million impostors is 8 MB of instances, and rebuilding that
//! allocation every time the streamer publishes is a frame spike rather
//! than a cost.
//!
//! Not the pipelines. Those still differ in mesh, shader and style, and
//! two is not yet enough evidence to say what a shared one would want.

use bevy::prelude::*;
use bevy::render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
};
use std::collections::HashMap;
use voxel_render::ScatterPoint;

/// One world's instances: a buffer with room to spare, and how much of it
/// is live.
pub struct InstanceBuffer {
    pub buffer: Buffer,
    /// Points the allocation can hold. Only ever grows.
    capacity: u32,
    /// Points to draw. Zero means the population is empty right now,
    /// which is not the same as gone — the allocation is kept.
    pub count: u32,
}

/// Every world's instances for one population class.
#[derive(Default)]
pub struct InstanceBuffers(HashMap<voxel_engine::WorldId, InstanceBuffer>);

impl InstanceBuffers {
    pub fn get(&self, world: voxel_engine::WorldId) -> Option<&InstanceBuffer> {
        self.0.get(&world).filter(|slot| slot.count > 0)
    }

    /// Replace what every world draws.
    ///
    /// Counts are zeroed before filling rather than buffers dropped: a
    /// world that published nothing this time must stop drawing, but its
    /// allocation is worth keeping for when it has points again.
    pub fn publish(
        &mut self,
        label: &'static str,
        per_world: impl IntoIterator<Item = (voxel_engine::WorldId, Vec<ScatterPoint>)>,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) {
        for slot in self.0.values_mut() {
            slot.count = 0;
        }
        for (world, points) in per_world {
            if points.is_empty() {
                continue;
            }
            let needed = points.len() as u32;
            let fits = self.0.get(&world).is_some_and(|s| s.capacity >= needed);
            if !fits {
                // Rounded up, so a population that grows by a few points a
                // second does not reallocate a few times a second.
                let capacity = needed.next_power_of_two();
                let buffer = device.create_buffer(&BufferDescriptor {
                    label: Some(label),
                    size: u64::from(capacity) * size_of::<ScatterPoint>() as u64,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.0.insert(
                    world,
                    InstanceBuffer {
                        buffer,
                        capacity,
                        count: 0,
                    },
                );
            }
            let slot = self.0.get_mut(&world).expect("present or just inserted");
            queue.write_buffer(&slot.buffer, 0, bytemuck::cast_slice(&points));
            slot.count = needed;
        }
    }
}
