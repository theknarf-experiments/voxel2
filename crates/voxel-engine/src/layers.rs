//! Bevy wiring for the layer graph.
//!
//! [`MainThreadQueue`] is the other half of `create`/`destroy` symmetry.
//! Chunks are generated on worker threads, but a chunk that owns Bevy
//! entities cannot spawn or despawn them there. It queues the work
//! instead, and an exclusive system drains the queue under a time budget.
//! Layers reach the queue through the graph's per-world context, which is
//! why `voxel-layers` needs no Bevy dependency of its own.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::prelude::*;

/// One piece of deferred chunk work.
type MainThreadAction = Box<dyn FnOnce(&mut World) + Send>;

/// Work a layer chunk needs done on the main thread, in the order it was
/// queued. Cloneable, so it can live in a world context handed to layers.
#[derive(Resource, Clone, Default)]
pub struct MainThreadQueue(Arc<Mutex<VecDeque<MainThreadAction>>>);

impl MainThreadQueue {
    /// Queue work for the next frame. Called from generation threads.
    pub fn push(&self, action: impl FnOnce(&mut World) + Send + 'static) {
        self.0.lock().unwrap().push_back(Box::new(action));
    }

    pub fn pending(&self) -> usize {
        self.0.lock().unwrap().len()
    }

    fn pop(&self) -> Option<MainThreadAction> {
        self.0.lock().unwrap().pop_front()
    }
}

/// Main-thread time spent draining queued chunk work per frame. Generation
/// runs ahead of the camera, so a backlog costs latency rather than
/// correctness — which makes a frame budget the right trade.
#[derive(Resource, Clone, Copy)]
pub struct MainThreadBudget(pub Duration);

impl Default for MainThreadBudget {
    fn default() -> Self {
        Self(Duration::from_millis(2))
    }
}

/// Installs the main-thread queue and its drain.
pub struct VoxelLayersPlugin;

impl Plugin for VoxelLayersPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MainThreadQueue>()
            .init_resource::<MainThreadBudget>()
            .add_systems(Update, drain_main_thread_queue);
    }
}

/// Run queued chunk work until the frame budget is spent. Whatever is left
/// waits for the next frame, in order.
fn drain_main_thread_queue(world: &mut World) {
    let (queue, budget) = {
        let Some(queue) = world.get_resource::<MainThreadQueue>() else {
            return;
        };
        let budget = world
            .get_resource::<MainThreadBudget>()
            .copied()
            .unwrap_or_default();
        (queue.clone(), budget.0)
    };
    if queue.pending() == 0 {
        return;
    }
    let start = Instant::now();
    while let Some(action) = queue.pop() {
        action(world);
        if start.elapsed() >= budget {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Queued work runs on the main thread, in order, and stops at the
    /// budget rather than stalling the frame.
    #[test]
    fn queue_drains_in_order_within_budget() {
        let mut world = World::new();
        world.init_resource::<MainThreadQueue>();
        world.insert_resource(MainThreadBudget(Duration::from_millis(5)));
        let queue = world.resource::<MainThreadQueue>().clone();

        let order = Arc::new(Mutex::new(Vec::new()));
        for i in 0..4 {
            let order = order.clone();
            queue.push(move |_| order.lock().unwrap().push(i));
        }
        assert_eq!(queue.pending(), 4);

        drain_main_thread_queue(&mut world);
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(queue.pending(), 0);
    }

    /// A backlog is deferred, never dropped and never run out of order.
    #[test]
    fn work_past_the_budget_waits_for_the_next_frame() {
        let mut world = World::new();
        world.init_resource::<MainThreadQueue>();
        world.insert_resource(MainThreadBudget(Duration::ZERO));
        let queue = world.resource::<MainThreadQueue>().clone();

        let done = Arc::new(AtomicUsize::new(0));
        for _ in 0..3 {
            let done = done.clone();
            queue.push(move |_| {
                done.fetch_add(1, Ordering::Relaxed);
            });
        }

        // A zero budget still runs one item per frame: progress is
        // guaranteed, so a queue can never wedge.
        drain_main_thread_queue(&mut world);
        assert_eq!(done.load(Ordering::Relaxed), 1);
        assert_eq!(queue.pending(), 2);

        drain_main_thread_queue(&mut world);
        drain_main_thread_queue(&mut world);
        assert_eq!(done.load(Ordering::Relaxed), 3);
        assert_eq!(queue.pending(), 0);
    }

    /// The queue is what lets a layer chunk own entities.
    #[test]
    fn queued_work_can_spawn_and_despawn_entities() {
        let mut world = World::new();
        world.init_resource::<MainThreadQueue>();
        world.init_resource::<MainThreadBudget>();
        let queue = world.resource::<MainThreadQueue>().clone();

        let spawned = Arc::new(Mutex::new(None));
        {
            let spawned = spawned.clone();
            queue.push(move |world: &mut World| {
                *spawned.lock().unwrap() = Some(world.spawn_empty().id());
            });
        }
        drain_main_thread_queue(&mut world);
        let entity = spawned.lock().unwrap().expect("spawned");
        assert!(world.get_entity(entity).is_ok());

        queue.push(move |world: &mut World| {
            world.despawn(entity);
        });
        drain_main_thread_queue(&mut world);
        assert!(world.get_entity(entity).is_err());
    }
}
