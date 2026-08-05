//! Megastructure demo: an endless Blame!-style concrete interior — floors,
//! pillars, walls, and vast vertical shafts, generated and meshed entirely
//! on the GPU by the same engine that renders the planet.

use bevy::prelude::*;
use voxel_debug::prelude::*;
use voxel_engine::streaming::ChunkOpsProvider;
use voxel_engine::{LodConfig, VoxelEnginePlugin, WorldKind};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "voxel2 — megastructure".into(),
                ..default()
            }),
            ..default()
        }))
        // Match the interior gloom in the concrete shader.
        .insert_resource(ClearColor(Color::srgb(0.035, 0.045, 0.06)))
        .insert_resource(LodConfig {
            // Interior scale, matched to the fog: past ~1 km nothing is
            // visible anyway, so keep the streamed world tight and refine
            // less eagerly than open terrain (interiors self-occlude).
            max_level: 4,
            top_radius: 2,
            top_y: (-3, 3),
            split_k: 1.6,
            merge_k: 2.1,
        })
        // Planned variation: habitation pockets and light wells.
        .insert_resource(ChunkOpsProvider(Some(std::sync::Arc::new(
            |key: voxel_engine::ChunkKey| {
                if key.level > 2 {
                    return Vec::new();
                }
                let min = key.min_corner_m().as_vec3();
                let max = min + Vec3::splat(key.edge_m() as f32);
                voxel_worldgen::mega::pockets_ops(min, max)
            },
        ))))
        .add_plugins((
            VoxelDebugPlugin,
            VoxelEnginePlugin {
                world: WorldKind::Megastructure,
            },
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (autopilot, walk_mode).chain())
        .run();
}

/// `VOXEL_WALK=1`: gravity + capsule collision against the CPU mirror of
/// the megastructure SDF — walk the interior on foot.
fn walk_mode(
    mut cameras: Query<&mut Transform, With<Camera3d>>,
    time: Res<Time>,
    mut fall_speed: Local<f32>,
    mut spawned: Local<bool>,
) {
    if std::env::var("VOXEL_WALK").is_err() {
        return;
    }
    use voxel_worldgen::mega::{mega_sdf, mega_sdf_with_ops, pockets_ops};
    const RADIUS: f32 = 0.5;
    const EYE: f32 = 1.6;

    for mut t in &mut cameras {
        // Planned features near the player participate in collision.
        let local_ops = pockets_ops(
            t.translation - Vec3::splat(30.0),
            t.translation + Vec3::splat(30.0),
        );
        let sdf = |p: Vec3| mega_sdf_with_ops(p, &local_ops);
        let grad = |p: Vec3| {
            let e = 0.1;
            Vec3::new(
                sdf(p + Vec3::X * e) - sdf(p - Vec3::X * e),
                sdf(p + Vec3::Y * e) - sdf(p - Vec3::Y * e),
                sdf(p + Vec3::Z * e) - sdf(p - Vec3::Z * e),
            )
            .normalize_or_zero()
        };
        // First tick: relocate onto solid floor (spawn cells can be holes).
        if !*spawned {
            *spawned = true;
            'probe: for r in 0..40 {
                for (dx, dz) in [(1.0, 0.3), (-0.7, 1.0), (0.4, -1.0), (-1.0, -0.5)] {
                    let p =
                        t.translation + Vec3::new(dx * r as f32 * 4.0, 0.0, dz * r as f32 * 4.0);
                    let level = (p.y / 44.0).round() * 44.0;
                    let foot = Vec3::new(p.x, level, p.z);
                    if mega_sdf(foot) < -1.0 {
                        t.translation = Vec3::new(foot.x, level + 1.5 + EYE, foot.z);
                        break 'probe;
                    }
                }
            }
        }

        // Clamped dt + substeps so startup hitches can't tunnel through
        // 1.5 m floor slabs.
        let dt = time.delta_secs().min(0.033);
        *fall_speed = (*fall_speed - 22.0 * dt).max(-30.0);
        let mut body = t.translation - Vec3::Y * EYE;
        let mut remaining = *fall_speed * dt;
        while remaining.abs() > 0.0 {
            let step = remaining.clamp(-0.4, 0.4);
            body.y += step;
            remaining -= step;
            for _ in 0..4 {
                let d = sdf(body);
                if d < RADIUS {
                    let n = grad(body);
                    body += n * (RADIUS - d);
                    if n.y > 0.5 {
                        *fall_speed = 0.0;
                        remaining = 0.0;
                    }
                }
            }
        }
        // Resolve horizontal penetration from the flycam's own motion.
        for _ in 0..4 {
            let d = sdf(body);
            if d < RADIUS {
                body += grad(body) * (RADIUS - d);
            }
        }
        t.translation = body + Vec3::Y * EYE;
    }
}

fn setup(mut commands: Commands) {
    let start = std::env::var("VOXEL_START")
        .ok()
        .and_then(|s| {
            let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
        })
        .unwrap_or(Vec3::new(11.0, 12.0, 7.0));
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(start)
            .looking_at(start + Vec3::new(0.7, 0.05, 0.7) * 100.0, Vec3::Y),
        FreeCamera {
            walk_speed: 8.0,
            run_speed: 60.0,
            ..default()
        },
    ));

    // Faint cold ambient so interiors aren't pitch black up close.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.6, 0.7, 0.9),
        brightness: 120.0,
        ..default()
    });
}

/// Flies the camera forward when `VOXEL_AUTOPILOT` is set (m/s).
fn autopilot(mut cameras: Query<&mut Transform, With<Camera3d>>, time: Res<Time>) {
    let Ok(speed) = std::env::var("VOXEL_AUTOPILOT") else {
        return;
    };
    let speed: f32 = speed.parse().unwrap_or(30.0);
    for mut transform in &mut cameras {
        let dir = transform.forward();
        transform.translation += dir * speed * time.delta_secs();
    }
}
