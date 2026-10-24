//! fork (Everwood port): warm, slowly breathing point lights over streamed magma surfaces.
//!
//! Reads the same world-space wet grid swimming uses: MCLQ and placed, rotated MLIQ already publish
//! it on their rendered entity, so no light stands over a shore or a masked hole. Sampling and
//! nearest-first budgeting run on surface changes (or the knob's on/off edge), not on camera
//! motion; only the at most [`LAVA_LIGHTS_MAX`] slow intensity writes run each frame. Slime stays
//! dark.
//!
//! The lights are ordinary [`WorldPointLight`]s on the ordinary packer, in either lighting model.
//! A WMO pool's lights take the pool's own rooms ([`LightRooms`]), so a cave pool lights the
//! interior walls around it and goes out with its room; the fork measured that the interior
//! classification is the right one for cave pools (its Searing Gorge night captures: pinning them
//! to the exterior lane made the rock darker). An open-world pool names no room.

use benilla_assets::coords::wow_to_bevy;
use bevy::prelude::*;
use std::collections::HashMap;

use super::{LightRooms, WorldPointLight};
use crate::liquid::{LiquidSoundSource, LiquidSurface, WaterChunkInfo};
use crate::wmo_portal::WmoGroupVis;

/// A magma channel's footprint is one MCNK liquid layer per 33.3 yd chunk, so a pool that fits in
/// one chunk needs this many to use the budget it stands beside (the fork measured six leaving a
/// 289 yd gap in Searing Gorge).
pub const LAVA_LIGHTS_PER_SURFACE: usize = 12;
pub const LAVA_LIGHTS_MAX: usize = 64;
/// The sample lattice over a magma footprint, and the minimum spacing the budgeter keeps. Measured
/// by the fork against the terrain falloff `rgb / (0.7 d + 0.03 d²) × max(N·L, 0)`: ground beside a
/// river reads the sum of the lights in its selection, so the spacing is most of the amplitude.
const GRID_YD: f32 = 8.0;
const MERGE_YD: f32 = 7.0;
/// How far above the magma surface a light floats. For flat ground at the liquid's own level
/// `N·L` degenerates to `lift / d`, so a low light cannot light its own banks.
const LIFT_YD: f32 = 2.5;
const COLOR: [f32; 3] = [1.0, 0.42, 0.10];
/// The authored rung: a river of molten rock is a bonfire (3.0), not a brazier (2.0).
const AUTHORED: f32 = 3.0;
// `WorldPointLight` stores 4π times authored units, which the packer's `/(4π)` undoes.
const INTENSITY: f32 = AUTHORED * 4.0 * std::f32::consts::PI;
/// The ≤3-nearest selection radius, as every authored point light's.
const RANGE_YD: f32 = 48.0;

/// A lava light. Ownership is explicit, not `ChildOf`: reconciliation removes the lights with
/// their surface. The modern lamp atlas never gives one a cube shadow
/// (`modern::lights::update_lights`).
#[derive(Component)]
pub struct LavaLight {
    surface: Entity,
    position: Vec3,
    phase: f32,
}

#[derive(Default)]
struct Candidates {
    points: Vec<Vec3>,
    /// The pool's rooms, for a WMO surface.
    rooms: Option<WmoGroupVis>,
}

#[derive(Resource, Default)]
struct LavaSurfaces {
    candidates: HashMap<Entity, Candidates>,
    dirty: bool,
    enabled: bool,
}

/// The `lavaLight` knob as a gain: zero removes every lava light.
#[derive(Resource, Clone, Copy)]
pub struct LavaLightGain(pub f32);

impl LavaLightGain {
    /// The registered `lavaLight` default (on).
    pub const REGISTERED: f32 = 1.0;
}

impl Default for LavaLightGain {
    fn default() -> Self {
        Self(Self::REGISTERED)
    }
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<LavaLightGain>()
        .init_resource::<LavaSurfaces>()
        // `Update`, never `PostUpdate` (`modern/HOOKS.md`): the light packer runs in `PostUpdate`
        // and reads these the same frame.
        .add_systems(Update, (reconcile, breathe, census).chain());
}

// The wet vertices give the bounds, then the grid's own inverse sampler gives coverage and height:
// a rotated MLIQ's AABB is not its footprint. Partial edge cells use their own centre.
fn sample_grid(info: &WaterChunkInfo) -> Vec<Vec3> {
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    info.for_each_wet_cell(|corners| {
        for [x, y, _] in corners {
            lo = lo.min(Vec2::new(x, y));
            hi = hi.max(Vec2::new(x, y));
        }
    });
    if !lo.is_finite() || !hi.is_finite() {
        return Vec::new();
    }
    let mut points = Vec::new();
    for j in 0..((hi.y - lo.y) / GRID_YD).ceil() as usize {
        for i in 0..((hi.x - lo.x) / GRID_YD).ceil() as usize {
            let cell_lo = lo + Vec2::new(i as f32, j as f32) * GRID_YD;
            let centre = (cell_lo + (cell_lo + Vec2::splat(GRID_YD)).min(hi)) * 0.5;
            if let Some(z) = info.surface_z_at(centre.x, centre.y) {
                points.push(wow_to_bevy([centre.x, centre.y, z + LIFT_YD]));
            }
        }
    }
    points
}

fn select(candidates: &HashMap<Entity, Candidates>, camera: Vec3) -> Vec<(Entity, Vec3)> {
    let mut ranked: Vec<_> = candidates
        .iter()
        .flat_map(|(&surface, c)| c.points.iter().map(move |&p| (surface, p)))
        .collect();
    ranked.sort_by(|(a, p), (b, q)| {
        p.distance_squared(camera)
            .total_cmp(&q.distance_squared(camera))
            .then_with(|| a.to_bits().cmp(&b.to_bits()))
            .then_with(|| p.x.total_cmp(&q.x))
            .then_with(|| p.y.total_cmp(&q.y))
            .then_with(|| p.z.total_cmp(&q.z))
    });
    let mut selected: Vec<(Entity, Vec3)> = Vec::new();
    let mut counts = HashMap::<Entity, usize>::new();
    for (surface, p) in ranked {
        if counts.get(&surface).copied().unwrap_or(0) >= LAVA_LIGHTS_PER_SURFACE
            || selected
                .iter()
                .any(|(_, q)| p.distance_squared(*q) < MERGE_YD * MERGE_YD)
        {
            continue;
        }
        // Keep the nearer representative in place: averaging a cluster could put it on dry land.
        selected.push((surface, p));
        *counts.entry(surface).or_default() += 1;
        if selected.len() == LAVA_LIGHTS_MAX {
            break;
        }
    }
    selected
}

#[allow(clippy::type_complexity)]
fn reconcile(
    mut commands: Commands,
    gain: Res<LavaLightGain>,
    mut cache: ResMut<LavaSurfaces>,
    surfaces: Query<
        (
            Entity,
            Ref<WaterChunkInfo>,
            Ref<LiquidSoundSource>,
            Option<&WmoGroupVis>,
        ),
        With<LiquidSurface>,
    >,
    camera: Query<&GlobalTransform, With<crate::view::WorldCamera>>,
    lights: Query<(Entity, &LavaLight)>,
) {
    let old_len = cache.candidates.len();
    cache.candidates.retain(|e, _| surfaces.contains(*e));
    cache.dirty |= old_len != cache.candidates.len();
    for (surface, info, sound, rooms) in &surfaces {
        // Both spawn paths publish the same sound class: `nibble & 3 == 2` is magma. Water (0/1)
        // and slime (3) never emit.
        if sound.nibble & 3 != 2 {
            cache.dirty |= cache.candidates.remove(&surface).is_some();
        } else if info.is_changed()
            || sound.is_changed()
            || !cache.candidates.contains_key(&surface)
        {
            cache.candidates.insert(
                surface,
                Candidates {
                    points: sample_grid(&info),
                    rooms: rooms.cloned(),
                },
            );
            cache.dirty = true;
        }
    }
    let enabled = gain.0.is_finite() && gain.0 > 0.0;
    cache.dirty |= enabled != cache.enabled;
    cache.enabled = enabled;
    if !cache.dirty {
        return;
    }
    let mut selected = if enabled {
        let Some(camera) = camera.iter().next() else {
            // Retire orphaned lights even while the world camera is being replaced.
            for (e, light) in &lights {
                if !cache.candidates.contains_key(&light.surface) {
                    commands.entity(e).despawn();
                }
            }
            return; // stay dirty until nearest-to-camera has a real camera
        };
        select(&cache.candidates, camera.translation())
    } else {
        Vec::new()
    };
    cache.dirty = false;
    debug!(
        "lava light: magma surfaces {} candidates {} kept {} gain {}",
        cache.candidates.len(),
        cache
            .candidates
            .values()
            .map(|c| c.points.len())
            .sum::<usize>(),
        selected.len(),
        gain.0,
    );
    for (e, light) in &lights {
        if let Some(i) = selected
            .iter()
            .position(|&(s, p)| s == light.surface && p == light.position)
        {
            selected.swap_remove(i);
        } else {
            commands.entity(e).despawn();
        }
    }
    for (surface, position) in selected {
        let hash = position.x.to_bits().wrapping_mul(0x9e37_79b9)
            ^ position.y.to_bits().rotate_left(11)
            ^ position.z.to_bits().rotate_left(21);
        let mut light = commands.spawn((
            LavaLight {
                surface,
                position,
                phase: hash as f32 / u32::MAX as f32 * std::f32::consts::TAU,
            },
            WorldPointLight {
                color: COLOR,
                intensity: INTENSITY * gain.0,
                range: RANGE_YD,
            },
            Transform::from_translation(position),
            GlobalTransform::from_translation(position),
        ));
        if let Some(rooms) = cache.candidates.get(&surface).and_then(|c| c.rooms.clone()) {
            light.insert(LightRooms(rooms));
        }
    }
}

fn breathe(
    time: Res<Time>,
    gain: Res<LavaLightGain>,
    mut lights: Query<(&LavaLight, &mut WorldPointLight)>,
) {
    for (lava, mut light) in &mut lights {
        let wave = (time.elapsed_secs() * std::f32::consts::TAU * 0.4 + lava.phase).sin();
        let intensity = INTENSITY * gain.0.max(0.0) * (1.0 + 0.1 * wave);
        // A stationary clock and a sub-epsilon breath do not churn change detection.
        if (light.intensity - intensity).abs() > 1e-4 {
            light.intensity = intensity;
        }
    }
}

/// `WOW_POINTS_DUMP`: the lava lights alone, nearest first, with whether each names rooms. The
/// packer's own table prints only the nearest eight lights of any kind, which in a zone full of
/// braziers prints no lava row.
fn census(
    time: Res<Time>,
    gain: Res<LavaLightGain>,
    lights: Query<(&LavaLight, &WorldPointLight, Has<LightRooms>)>,
    camera: Query<&GlobalTransform, With<crate::view::WorldCamera>>,
    mut last: Local<f64>,
) {
    static DUMP: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(mode) = DUMP.get_or_init(|| crate::launch_env::var("WOW_POINTS_DUMP")) else {
        return;
    };
    let every = if mode == "frame" { 0.0 } else { 1.0 };
    let now = time.elapsed_secs_f64();
    if now - *last < every {
        return;
    }
    *last = now;
    let Some(camera) = camera.iter().next() else {
        return;
    };
    let cam = camera.translation();
    let mut rows: Vec<(f32, bool, f32, Vec3)> = lights
        .iter()
        .map(|(lava, light, rooms)| {
            (
                lava.position.distance(cam),
                rooms,
                light.intensity / (4.0 * std::f32::consts::PI),
                lava.position,
            )
        })
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let roomed = rows.iter().filter(|r| r.1).count();
    eprintln!(
        "[lava] {} light(s) ({roomed} in a WMO room / {} open), gain {:.2}, cam {:?}",
        rows.len(),
        rows.len() - roomed,
        gain.0,
        benilla_assets::coords::bevy_to_wow(cam),
    );
    for (d, roomed, i, p) in rows.iter().take(8) {
        eprintln!(
            "  d {d:6.2}  at {:?}  I {i:.2}  {}",
            benilla_assets::coords::bevy_to_wow(*p),
            if *roomed { "ROOM" } else { "OPEN" },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liquid::LiquidSource;
    use benilla_formats::LiquidKind;

    fn grid(cols: usize, rows: usize, wet: Vec<bool>, kind: LiquidKind) -> WaterChunkInfo {
        let positions = (0..rows)
            .flat_map(|j| {
                (0..cols)
                    .map(move |i| [i as f32 * 12.0, j as f32 * 12.0, i as f32 * 2.0 + j as f32])
            })
            .collect();
        WaterChunkInfo::new(LiquidSource::AdtChunk, kind, [cols, rows], positions, wet)
    }

    fn points(list: Vec<(Entity, Vec<Vec3>)>) -> HashMap<Entity, Candidates> {
        list.into_iter()
            .map(|(e, points)| {
                (
                    e,
                    Candidates {
                        points,
                        rooms: None,
                    },
                )
            })
            .collect()
    }

    // Denominated in the constants, so a tuning change does not fail the rule (mask and sampled
    // height) this protects.
    #[test]
    fn grid_sampling_respects_mask_and_surface_height() {
        let info = grid(3, 3, vec![true, false, false, true], LiquidKind::Magma);
        let points = sample_grid(&info);
        assert!(!points.is_empty());
        for point in &points {
            let p = benilla_assets::coords::bevy_to_wow(*point);
            // Every point is over a wet cell and stands `LIFT_YD` over the sampled surface: never
            // the chunk maximum, never a masked hole.
            assert_eq!(
                info.surface_z_at(p[0], p[1]).map(|z| z + LIFT_YD),
                Some(p[2]),
            );
        }
        // The lattice is the one the constant names: the first cell's centre is half a cell in.
        assert_eq!(
            points[0],
            wow_to_bevy([
                GRID_YD * 0.5,
                GRID_YD * 0.5,
                // `z = i·2 + j` over a 12 yd vertex spacing is `x/6 + y/12` at the sample.
                GRID_YD * 0.5 / 6.0 + GRID_YD * 0.5 / 12.0 + LIFT_YD,
            ]),
        );
        assert!(sample_grid(&grid(2, 2, vec![false], LiquidKind::Magma)).is_empty());
    }

    #[test]
    fn rotated_grid_does_not_emit_over_aabb_corners_or_holes() {
        let rot = Quat::from_rotation_z(0.37);
        let positions = (0..4)
            .flat_map(|j| {
                (0..4).map(move |i| {
                    (rot * Vec3::new(i as f32 * 12.0, j as f32 * 12.0, 3.0)).to_array()
                })
            })
            .collect();
        let info = WaterChunkInfo::new(
            LiquidSource::AdtChunk,
            LiquidKind::Magma,
            [4, 4],
            positions,
            vec![true, true, true, true, false, true, true, true, true],
        );
        let points = sample_grid(&info);
        assert!(!points.is_empty());
        for point in points {
            let p = benilla_assets::coords::bevy_to_wow(point);
            assert_eq!(
                info.surface_z_at(p[0], p[1]).map(|z| z + LIFT_YD),
                Some(p[2])
            );
            let local = rot.inverse() * Vec3::from_array(p);
            assert!(!(local.x > 12.0 && local.x < 24.0 && local.y > 12.0 && local.y < 24.0));
        }
    }

    #[test]
    fn merge_across_surfaces_keeps_nearest_and_accepts_the_spacing_exactly() {
        let mut world = World::new();
        let a = world.spawn_empty().id();
        let b = world.spawn_empty().id();
        // Inside the merge radius is dropped; exactly at it is kept (the test is `< MERGE_YD²`).
        let points = points(vec![
            (a, vec![Vec3::ZERO]),
            (b, vec![Vec3::X * (MERGE_YD - 1.0), Vec3::X * MERGE_YD]),
        ]);
        assert_eq!(
            select(&points, Vec3::ZERO),
            vec![(a, Vec3::ZERO), (b, Vec3::X * MERGE_YD)]
        );
    }

    #[test]
    fn caps_are_global_and_per_surface_with_nearest_winning() {
        let mut world = World::new();
        // Enough per surface that the per-surface cap bites before the global one, and enough
        // surfaces that the global cap bites too.
        let per_surface = LAVA_LIGHTS_PER_SURFACE + 3;
        let points = points(
            (0..10)
                .map(|s| {
                    let e = world.spawn_empty().id();
                    (
                        e,
                        (0..per_surface)
                            .map(|i| Vec3::new(s as f32 * 200.0, 0.0, i as f32 * (MERGE_YD + 1.0)))
                            .collect(),
                    )
                })
                .collect(),
        );
        let selected = select(&points, Vec3::ZERO);
        assert_eq!(selected.len(), LAVA_LIGHTS_MAX);
        let mut capped = 0;
        for &e in points.keys() {
            let n = selected.iter().filter(|(s, _)| *s == e).count();
            assert!(n <= LAVA_LIGHTS_PER_SURFACE);
            capped += usize::from(n == LAVA_LIGHTS_PER_SURFACE);
        }
        assert!(capped > 0, "the per-surface cap must actually bind here");
        // Nearest first: the budget is spent on the near surfaces, never scattered over far ones.
        let far = 200.0 * (LAVA_LIGHTS_MAX / LAVA_LIGHTS_PER_SURFACE + 1) as f32;
        assert!(selected.iter().all(|(_, p)| p.x <= far));
        assert_eq!(selected[0].1, Vec3::ZERO);
    }

    /// The budget goes to the magma the camera stands by, and a pool that fits in one MCNK liquid
    /// layer can take enough of it to read.
    #[test]
    fn the_budget_goes_to_the_pool_the_camera_is_standing_by() {
        let mut world = World::new();
        let near = world.spawn_empty().id();
        let far = world.spawn_empty().id();
        let line = |x: f32| {
            (0..LAVA_LIGHTS_PER_SURFACE * 2)
                .map(|i| Vec3::new(x, 0.0, i as f32 * (MERGE_YD + 1.0)))
                .collect::<Vec<_>>()
        };
        let points = points(vec![(near, line(0.0)), (far, line(289.0))]);
        let selected = select(&points, Vec3::ZERO);
        assert_eq!(
            selected.iter().filter(|(s, _)| *s == near).count(),
            LAVA_LIGHTS_PER_SURFACE,
            "the near pool fills its whole per-surface allowance",
        );
    }

    fn app() -> App {
        let mut app = App::new();
        app.init_resource::<Time>();
        register(&mut app);
        app.world_mut()
            .spawn((crate::view::WorldCamera, GlobalTransform::IDENTITY));
        app
    }

    fn surface(app: &mut App, kind: LiquidKind, nibble: u8) -> Entity {
        app.world_mut()
            .spawn((
                LiquidSurface,
                LiquidSoundSource { nibble },
                grid(3, 3, vec![true; 4], kind),
            ))
            .id()
    }

    fn lights(app: &mut App) -> usize {
        app.world_mut()
            .query_filtered::<Entity, With<LavaLight>>()
            .iter(app.world())
            .count()
    }

    /// How many lights one full test surface is worth, derived: the lattice constants move with
    /// the tuning and the count is not the subject of any test below.
    fn per_surface() -> usize {
        sample_grid(&grid(3, 3, vec![true; 4], LiquidKind::Magma)).len()
    }

    #[test]
    fn gain_zero_removes_all_and_positive_gain_restores_without_surface_churn() {
        let mut app = app();
        surface(&mut app, LiquidKind::Magma, 6);
        app.update();
        assert_eq!(lights(&mut app), per_surface());
        app.world_mut().resource_mut::<LavaLightGain>().0 = 0.0;
        app.update();
        assert_eq!(lights(&mut app), 0);
        app.world_mut().resource_mut::<LavaLightGain>().0 = 2.0;
        app.update();
        assert_eq!(lights(&mut app), per_surface());
        for light in app
            .world_mut()
            .query::<&WorldPointLight>()
            .iter(app.world())
        {
            assert!((INTENSITY * 1.8..=INTENSITY * 2.2).contains(&light.intensity));
        }
    }

    #[test]
    fn streaming_removes_orphans_and_slime_never_emits() {
        let mut app = app();
        let magma = surface(&mut app, LiquidKind::Magma, 2);
        surface(&mut app, LiquidKind::Slime, 3);
        app.update();
        assert_eq!(lights(&mut app), per_surface());
        for (e, marker) in app
            .world_mut()
            .query::<(Entity, &LavaLight)>()
            .iter(app.world())
        {
            assert_eq!(marker.surface, magma);
            // An open-world pool names no room, and nothing flickers a lava light but its breath.
            assert!(app.world().get::<LightRooms>(e).is_none());
            assert!(app.world().get::<super::super::FlameFlicker>(e).is_none());
            assert!(app.world().get::<ChildOf>(e).is_none());
        }
        app.world_mut().despawn(magma);
        app.update();
        assert_eq!(lights(&mut app), 0);
    }
}
