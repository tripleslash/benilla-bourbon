//! **The wade probe** — a stand-in body that walks back and forth through water, so the foam that
//! only a wading unit can produce is reachable from a binary that has no player in it.
//!
//! `water_fx`'s `CWater0Ripple` emitter drives off units: the avatar through `view::Viewer`, and
//! everyone else through `(Transform, WorldUnit)`. `benilla-worldview` has neither — it is the
//! engine with no game attached, which is the whole point of it — so the ripple ring, the wake
//! chevron and the step-in splash were the one part of the water that could not be looked at
//! without launching the client, logging in, and walking to a lake.
//!
//! This is the smallest thing that closes that: one entity with the two components the emitter
//! reads, moved along a line, on a key. It is a *validation rig*, not a player — it has no
//! collision, no camera, no input beyond the toggle, and it does not pretend to be a unit for any
//! other system. It lives in `worldview` rather than in `water_fx` for that reason: the emitter
//! must keep taking its input from real bodies, and a fixture that ships inside it would be a
//! second source of units that the game would then have to know not to spawn.
//!
//! **Press `B`** to put it in the water in front of the camera and take it out again.

use bevy::prelude::*;

use crate::liquid::{water_surface_at, LiquidClaim, WaterChunkInfo};
use crate::view::WorldCamera;
use crate::world_unit::WorldUnit;
use benilla_assets::coords::{bevy_to_wow, wow_to_bevy};

/// A human's collision height in yards — the emitter's depth gate is `2 ·` this, and the ring's
/// radius scales off it, so a wrong value here is a rig that tests the wrong sizes.
const PROBE_HEIGHT: f32 = 2.03;

/// How far the probe walks either side of its anchor, in yards. About four body widths: far enough
/// to cross a shoreline and put the wake over both shallow and deep water.
const WALK_RADIUS: f32 = 6.0;

/// How deep the probe stands, in yards below the surface.
///
/// **Not zero, and that is the whole fix.** The emitter's depth is `surface − feet`, so a body with
/// its feet exactly at the waterline is in no water at all: every gate reads zero and nothing is
/// emitted. Knee-deep on a two-yard body is what a wade actually is, and it sits inside the
/// emission gate (`depth < max(2·height, 1)`) with room either side.
const WADE_DEPTH: f32 = 0.9;

/// Seconds for one full there-and-back. Slow enough to read, fast enough that the wake cadence
/// (one record per ~0.625 yd of travel) actually fires.
const WALK_PERIOD: f32 = 8.0;

/// Marks the probe body.
#[derive(Component)]
pub(super) struct WadeProbe {
    /// Where it walks around, in Bevy world space — the camera's ground position when it was
    /// switched on.
    anchor: Vec3,
    /// The horizontal direction it walks along, unit length.
    axis: Vec3,
    /// Seconds since it was switched on, so the walk starts at the anchor rather than mid-stride.
    age: f32,
}

/// **`$WOW_WADE_PROBE`** — put the probe in the water without a keypress.
///
/// The hotkey is the tool; this is how an unattended run reaches it. A timed launch has nobody to
/// press `B`, and a screenshot or a log line proving the ripple system fires is exactly the thing
/// worth having in a run with nobody watching. It searches outward from the camera for the first
/// XY the liquid query answers on, so it lands in real water rather than at a guessed offset.
pub(super) fn auto_spawn_probe(
    mut commands: Commands,
    existing: Query<Entity, With<WadeProbe>>,
    camera: Query<&Transform, With<WorldCamera>>,
    water: Query<&WaterChunkInfo>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut done: Local<bool>,
) {
    if *done || std::env::var_os("WOW_WADE_PROBE").is_none() || !existing.is_empty() {
        return;
    }
    let Ok(cam) = camera.single() else {
        return;
    };
    // A coarse outward scan, centred on the point the camera is LOOKING at rather than on the
    // camera itself. The difference matters for a rig whose whole job is to be watched: scanning
    // from the camera finds whichever water is nearest in any direction, which on a riverbank is
    // as likely to be behind you as in front, and a probe you cannot see validates nothing.
    let forward = cam.forward().as_vec3();
    let flat = Vec3::new(forward.x, 0.0, forward.z).normalize_or(Vec3::Z);
    let seed = cam.translation + flat * 10.0;
    let mut found = None;
    'scan: for ring in 0..30u32 {
        let r = ring as f32 * 4.0;
        for step in 0..(ring * 8).max(1) {
            let a = step as f32 / (ring * 8).max(1) as f32 * std::f32::consts::TAU;
            let p = seed + Vec3::new(a.cos() * r, 0.0, a.sin() * r);
            let wow = bevy_to_wow(p);
            if let Some(z) = water_surface_at(water.iter(), wow, LiquidClaim::Outdoors) {
                found = Some(wow_to_bevy([wow[0], wow[1], z]));
                break 'scan;
            }
        }
    }
    let Some(anchor) = found else {
        return; // no water streamed in yet — try again next frame
    };
    *done = true;
    spawn_probe(&mut commands, anchor, Vec3::X, &mut meshes, &mut materials);
    info!("wade probe: auto-spawned at {anchor:?} ($WOW_WADE_PROBE)");
}

/// `B` toggles the probe in and out.
pub(super) fn toggle_wade_probe(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    existing: Query<Entity, With<WadeProbe>>,
    camera: Query<&Transform, With<WorldCamera>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !keys.just_pressed(KeyCode::KeyB) {
        return;
    }
    if let Ok(entity) = existing.single() {
        commands.entity(entity).despawn();
        info!("wade probe: off");
        return;
    }
    let Ok(cam) = camera.single() else {
        return;
    };
    // In front of the camera and level with it in XZ; `drive_probe` drops it onto the water.
    let forward = cam.forward().as_vec3();
    let flat = Vec3::new(forward.x, 0.0, forward.z).normalize_or(Vec3::Z);
    let anchor = cam.translation + flat * 8.0;
    // The walk axis is across the view, so the probe crosses the shoreline in front of you rather
    // than marching away from you down it.
    let axis = Vec3::new(-flat.z, 0.0, flat.x);
    spawn_probe(&mut commands, anchor, axis, &mut meshes, &mut materials);
    info!("wade probe: on - walking {WALK_RADIUS} yd either side of the camera's forward point");
}

fn spawn_probe(
    commands: &mut Commands,
    anchor: Vec3,
    axis: Vec3,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    commands
        .spawn((
            WadeProbe {
                anchor,
                axis,
                age: 0.0,
            },
            // The two components `water_fx::emit_water_foam` reads for a non-viewer body. `wades` is
            // the whole point; `bound: None` keeps the world from culling a fixture by an extent it
            // was never given.
            WorldUnit {
                wades: true,
                scale: 1.0,
                height: PROBE_HEIGHT,
                bound: None,
            },
            Transform::from_translation(anchor),
            // The parent carries no mesh of its own, so nothing would give it the visibility component
            // its child's inherits from — Bevy warns B0004 and the child's visibility stops
            // propagating. Stated here rather than left to a required-component chain that does not
            // reach a mesh-less entity.
            Visibility::default(),
        ))
        // The body is a CHILD, lifted half its height. The parent's translation is the emitter's
        // contract — feet, because `water_fx` and `liquid::ripple` both measure depth as
        // `surface - position` — while a capsule's own origin is its middle, so a mesh on the parent
        // would stand buried to the neck and be invisible under its own ripples.
        .with_child((
            // Unlit and crude on purpose: this is a marker, and dressing it up would invite reading it
            // as a character.
            Mesh3d(meshes.add(Capsule3d::new(0.4, PROBE_HEIGHT - 0.8))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.9, 0.2, 0.2),
                unlit: true,
                ..default()
            })),
            Transform::from_xyz(0.0, PROBE_HEIGHT * 0.5, 0.0),
        ));
}

/// Walk the probe along its axis and keep its feet at the water surface.
///
/// The height query is what makes this a wade rather than a fly-past: the emitter gates on the
/// water's depth under the body, so a probe left at the camera's altitude is a unit standing in
/// mid-air over a lake and produces nothing. Where there is no water it simply holds its anchor
/// height — the rig has no terrain to stand on and does not need one.
pub(super) fn drive_probe(
    time: Res<Time>,
    mut probe: Query<(&mut WadeProbe, &mut Transform)>,
    water: Query<&WaterChunkInfo>,
) {
    let Ok((mut probe, mut transform)) = probe.single_mut() else {
        return;
    };
    probe.age += time.delta_secs();
    let phase = probe.age / WALK_PERIOD * std::f32::consts::TAU;
    let mut pos = probe.anchor + probe.axis * (phase.sin() * WALK_RADIUS);

    let wow = bevy_to_wow(pos);
    // `Outdoors` unconditionally: the rig is a fixture for open-world shorelines, and the interior
    // claim it would otherwise need is the camera's, which is not where the probe is standing.
    if let Some(surface) = water_surface_at(water.iter(), wow, LiquidClaim::Outdoors) {
        pos = wow_to_bevy([wow[0], wow[1], surface - WADE_DEPTH]);
    }
    transform.translation = pos;
    // Face the walk, so the wake chevron has a heading to point along.
    let heading = probe.axis * phase.cos();
    if heading.length_squared() > 1.0e-6 {
        transform.look_to(heading.normalize(), Vec3::Y);
    }
}
