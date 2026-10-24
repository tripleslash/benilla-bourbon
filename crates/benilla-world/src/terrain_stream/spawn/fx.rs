//! A placed model's point lights, particle emitters and ribbon trails, baked at its placement.

use benilla_assets::coords::wow_to_bevy;
use benilla_assets::{ModelEmitter, ModelLight};
use benilla_formats::WmoLight;
use bevy::prelude::*;

use crate::lighting::WorldPointLight;
use crate::particles;

/// Unit intensity in the `PointLight` convention, which the packer's `/(4π)` undoes to exactly
/// `diffuse_color × diffuse_intensity`; `wow_model.wgsl` applies the falloff `1/(0.7d + 0.03d²)`.
const POINT_LIGHT_INTENSITY: f32 = 4.0 * std::f32::consts::PI;
/// The ≤3-nearest selection radius (yd) around the lit unit's anchor, not a cutoff (a GL light has
/// none; att(48 yd) ≈ 0.01), wide enough to reach a hall's fixtures ~20 yd from its centre.
const POINT_LIGHT_RANGE: f32 = 48.0;

/// The [`WorldPointLight`] for an authored M2 or WMO MOLT light, `color` its linear RGB. As in the
/// reference it lights terrain, M2s and WMO walls, each taking its unit's ≤3 nearest, diffuse only.
pub fn point_light(color: [f32; 3], intensity_scale: f32) -> WorldPointLight {
    WorldPointLight {
        color,
        intensity: POINT_LIGHT_INTENSITY * intensity_scale.max(0.0),
        range: POINT_LIGHT_RANGE,
    }
}

fn spawn_point_light(
    commands: &mut Commands,
    world: Vec3,
    color: [f32; 3],
    intensity_scale: f32,
    // The light's rooms inside a building, `None` outdoors: a torch prop in a culled room is not
    // in the reference's scene that frame, so it lights nothing.
    room: Option<crate::wmo_portal::WmoGroupVis>,
    // fork: whether this source burns ([`crate::lighting::flame_flicker`]) — inserted once, here, so
    // the entity's archetype is settled at spawn.
    flicker: Option<crate::lighting::FlameFlicker>,
    out: &mut Vec<Entity>,
) {
    let mut e = commands.spawn((
        point_light(color, intensity_scale),
        Transform::from_translation(world),
        // fork (perf): a world light, which the modern table may group with its neighbours.
        crate::lighting::PlacedLight,
    ));
    if let Some(room) = room {
        e.insert(crate::lighting::LightRooms(room));
    }
    if let Some(flicker) = flicker {
        e.insert(flicker);
    }
    out.push(e.id());
}

/// Spawns a light for each casting M2 point light (`type == 1`) at its rest position, since a
/// placed prop never animates its light bone; `room` comes off the placement's [`emitter_fade`].
pub(super) fn spawn_lights_for(
    commands: &mut Commands,
    lights: &[ModelLight],
    transform: Transform,
    room: Option<&crate::wmo_portal::WmoGroupVis>,
    out: &mut Vec<Entity>,
) {
    for l in lights.iter().map(|l| &l.def) {
        if !l.casts() {
            continue; // directional lights feed the ambient; a static `0` visibility key is dark
        }
        let world = transform.transform_point(wow_to_bevy(l.position));
        spawn_point_light(
            commands,
            world,
            l.diffuse_color,
            l.diffuse_intensity,
            room.cloned(),
            crate::lighting::flame_flicker(
                None,
                false,
                l.diffuse_color,
                l.diffuse_intensity,
                world,
            ),
            out,
        );
    }
}

/// fork: modern lighting (`FORK.md` P8) — **the lamp that is only a flame card.**
///
/// A great many braziers, candles, sconces and hearths author no M2 light at all: the reference drew
/// their fire as an additive sprite over an unlit wall and that was the look. On the emissive tiers
/// (Ultra and Cinematic since P8b) they emit, from a light [`crate::modern::lights::emissive`]
/// derives from the very ramp that sprite is drawn with. The light is spawned on every tier and
/// gated where the table is packed (`global_light`), so a tier change lights it without a re-stream.
///
/// It is a no-op for a model that authors its own light (that gate is `emissive::synthesize`'s
/// first line, and the reason it is spelled there rather than here), so the two spawners can never
/// both answer for one doodad.
///
/// A lamppost is the case that made this take `submeshes` as well as `emitters`: the glow around
/// an Elwynn or Darkshore lantern is an **additive billboard card**, not a particle emitter, so the
/// first cut of this looked straight past every one of them.
///
/// The light sits at the **glowing part's own centroid** in model space, not at the model origin: a
/// wall sconce's flame is a yard out from its mounting point and half a yard up, and a light at the
/// origin would put the glow inside the wall — the one placement error that reads immediately,
/// because the cube shadow then has that wall between the light and the room.
pub(super) fn spawn_emissive_light_for(
    commands: &mut Commands,
    // The model's file path: a light fixture whose glow is its unlit glass lights from that
    // ([`crate::modern::lights::emissive::fixture_glass`]).
    path: &str,
    lights: &[ModelLight],
    emitters: &[ModelEmitter],
    submeshes: &[benilla_assets::ModelSubmesh],
    transform: Transform,
    room: Option<&crate::wmo_portal::WmoGroupVis>,
    // The model's loader-idle file slot: only what fires or draws in it counts
    // ([`crate::modern::lights::emissive::emits_at_rest`]).
    idle: usize,
    out: &mut Vec<Entity>,
) {
    // **No tier gate here.** The light is spawned whatever the tier, tagged
    // [`SynthesizedLight`], and `global_light` drops it from the table when the tier does not want
    // it. Gating the spawn instead made a lamp's fate depend on whether it streamed in before or
    // after the lighting latch — see that type's note.
    use crate::modern::lights::emissive;
    if emissive::is_living_swarm(path) || emissive::is_light_shaft(path) {
        return;
    }
    // The centroid and fixture radius are `emissive`'s own law, shared with the entity lane —
    // a GameObject brazier is this same derivation over `EntityPart`s instead of `ModelSubmesh`es.
    let recipe = emissive::for_model(lights, emitters, submeshes, idle).and_then(|emissive| {
        let (local, radius) = emissive::glow_anchor(
            emitters
                .iter()
                .filter(|e| emissive::emits_at_rest(&e.def.timing, idle))
                .map(|e| e.def.position),
            submeshes
                .iter()
                .filter(|b| emissive::shown_at_rest(b.alpha_anim.as_deref(), idle))
                .map(|b| b.geometry.as_ref()),
        )?;
        Some((emissive, local, radius))
    });
    let Some((emissive, local, radius)) =
        recipe.or_else(|| emissive::fixture_glass(path, lights, submeshes, idle))
    else {
        return;
    };
    let before = out.len();
    let world = transform.transform_point(wow_to_bevy(local));
    spawn_point_light(
        commands,
        world,
        emissive.color,
        emissive.intensity_scale,
        room.cloned(),
        crate::lighting::flame_flicker(emissive.flame, true, emissive.color, 0.0, world),
        out,
    );
    // How far the fixture reaches around the light it houses, in world yards — the radius whose
    // geometry gets clipped out of this light's own cube, so a lantern does not project its frame
    // over the pool it is supposed to be casting. See `ShadowLight::near`.
    let scale = transform.scale.max_element().max(1e-3);
    for e in &out[before..] {
        commands.entity(*e).insert((
            crate::modern::lights::FixtureRadius(radius * scale),
            crate::modern::lights::SynthesizedLight,
        ));
    }
}

/// fork: modern lighting (P8) — **a building's lit windows, as lights**: one per pool of SIDN glass
/// ([`crate::modern::lights::emissive::window_glows`]), each tagged
/// [`crate::lighting::NightGlow`] so the packer scales it by the night fraction and it is out of the
/// table by day, and [`crate::modern::lights::SynthesizedLight`] so it lights only on the tiers that
/// derive light from what only ever glowed (Ultra and Cinematic).
///
/// Outdoor lights, deliberately (no rooms): they are drawn from exterior groups' glass and light
/// the street, and the surface lanes' indoor/outdoor test keeps them off the room behind the pane.
/// They never hold a cube slot (`modern::lights::update_lights`): a window is a soft fill, and a
/// city has more of them than the atlas has room for lamps.
pub(super) fn spawn_window_lights_for(
    commands: &mut Commands,
    submeshes: &[benilla_assets::ModelSubmesh],
    transform: Transform,
    out: &mut Vec<Entity>,
) {
    let glows = crate::modern::lights::emissive::window_glows(
        submeshes
            .iter()
            .map(|b| (b.sidn, b.interior, b.geometry.as_ref())),
    );
    for glow in glows {
        let before = out.len();
        spawn_point_light(
            commands,
            transform.transform_point(wow_to_bevy(glow.pos)),
            glow.color,
            glow.intensity_scale,
            None,
            None, // a lit window is glass: it glows, it does not burn
            out,
        );
        for e in &out[before..] {
            commands.entity(*e).insert((
                crate::modern::lights::SynthesizedLight,
                crate::lighting::NightGlow,
                crate::lighting::WindowPane,
            ));
        }
    }
    // fork: and the lamps in the building's own geometry — a street lamp's SIDN glass, one light
    // per lantern ([`crate::modern::lights::emissive::lamp_glass_glows`]). Night-only like the
    // glass, but a lamp: no `WindowPane`.
    let lamps = crate::modern::lights::emissive::lamp_glass_glows(
        submeshes
            .iter()
            .map(|b| (b.sidn, b.interior, b.geometry.as_ref())),
    );
    for lamp in lamps {
        let before = out.len();
        spawn_point_light(
            commands,
            transform.transform_point(wow_to_bevy(lamp.pos)),
            lamp.color,
            lamp.intensity_scale,
            None,
            None, // glass: it glows, it does not burn
            out,
        );
        for e in &out[before..] {
            commands.entity(*e).insert((
                crate::modern::lights::SynthesizedLight,
                crate::lighting::NightGlow,
            ));
        }
    }
}

/// Spawns a light for each omni MOLT light (`type == 0`) under the M2 falloff, which the disk
/// `attenStart`/`attenEnd` do not shape (`0x695c00`). A light belongs to the rooms whose MOLR
/// names it; one that no group names is ungated.
pub(super) fn spawn_wmo_lights_for(
    commands: &mut Commands,
    lights: &[WmoLight],
    group_light_refs: &[Vec<u16>],
    instance: Option<Entity>,
    transform: Transform,
    out: &mut Vec<Entity>,
) {
    for (i, l) in lights.iter().enumerate() {
        if !l.is_omni() {
            continue; // spot/directional/ambient MOLT types are not point sources
        }
        let i = i as u16;
        let groups: std::sync::Arc<[u16]> = group_light_refs
            .iter()
            .enumerate()
            .filter(|(_, refs)| refs.contains(&i))
            .map(|(g, _)| g as u16)
            .collect();
        let world = transform.transform_point(wow_to_bevy(l.position));
        let before = out.len();
        spawn_point_light(
            commands,
            world,
            l.color,
            l.intensity,
            instance
                .filter(|_| !groups.is_empty())
                .map(|instance| crate::wmo_portal::WmoGroupVis { instance, groups }),
            crate::lighting::flame_flicker(None, false, l.color, l.intensity, world),
            out,
        );
        // fork: no flame or glass is drawn for a MOLT light, so the fog gives it no halo.
        for e in &out[before..] {
            commands.entity(*e).insert(crate::lighting::Glowless);
        }
    }
}

/// A placed model's one draw-set gate, shared by its mesh, emitters, ribbons, lights and anim host.
/// `fade` is `(radius, model-space centre)`; `None` or empty `groups` means no room gate.
pub(super) fn emitter_fade(
    transform: Transform,
    fade: (f32, Vec3),
    instance: Option<Entity>,
    groups: Option<&std::sync::Arc<[u16]>>,
) -> particles::EmitterFade {
    let room = instance.zip(groups.filter(|g| !g.is_empty()));
    particles::EmitterFade {
        instance,
        room: room.map(|(instance, groups)| crate::wmo_portal::WmoGroupVis {
            instance,
            groups: groups.clone(),
        }),
        ..particles::EmitterFade::sphere(fade.0, transform.transform_point(fade.1))
    }
}

/// Spawns a model's emitters into `out`. With an anim `host` each rides its bone's anchor, which
/// holds a bone that never animates at its rest pivot, so riding is exact for every emitter.
pub(super) fn spawn_emitters_for(
    commands: &mut Commands,
    emitters: &[ModelEmitter],
    transform: Transform,
    host: Option<&super::assemble::PlacementHost>,
    fade: &particles::EmitterFade,
    out: &mut Vec<Entity>,
) {
    let arm = host.and_then(|h| h.arm);
    for em in emitters {
        let owner = host
            .and_then(|h| h.anchor(em.def.bone))
            .map(|a| (a, em.bone_pivot));
        if let Some(e) = particles::spawn_emitter(
            commands,
            em,
            transform,
            particles::EmitterFrames {
                owner,
                anchor: None, // anchor at the placement: an animated bone never drags the cloud
                // An animated doodad's rig goes only when its placement unloads, the model's end,
                // so the pool is freed rather than left draining at the old spot.
                on_owner_loss: particles::OwnerLoss::Free,
                ..default()
            },
            // A placed doodad re-rolls its variation every play window, so its emitters read the
            // slot and clip time off the host's live player, as the reference's animate kernel
            // (`0x714260`) does for every model.
            match arm {
                Some(root) => particles::EmitClock::Host(root),
                None => particles::EmitClock::Pinned,
            },
        ) {
            commands.entity(e).insert(fade.clone());
            out.push(e);
        }
    }
}

/// Spawns a model's ribbon trails, each riding its host bone's anchor or, on a static placement,
/// `fallback_owner`; a trail despawns with its owner, so none joins a despawn list.
pub(super) fn spawn_ribbons_for(
    commands: &mut Commands,
    ribbons: &[benilla_assets::ModelRibbon],
    transform: Transform,
    host: Option<&super::assemble::PlacementHost>,
    fallback_owner: Option<Entity>,
    // The emitters' gate, carried because a joint or submesh owner states no fade sphere or rooms.
    fade: &particles::EmitterFade,
) {
    let arm = host.and_then(|h| h.arm);
    for rb in ribbons {
        let (owner, use_pivot) = match host.and_then(|h| h.anchor(rb.def.bone)) {
            Some(a) => (a, true),
            None => match fallback_owner {
                Some(e) => (e, false),
                None => continue, // nothing to ride (an entity-less placement)
            },
        };
        // The `+0xc0` enable gate reads the live sequence, as the emitters do; an unanimated
        // placement has no clock and rests in `Stand`.
        crate::ribbons::spawn_ribbon(
            commands,
            rb,
            owner,
            use_pivot,
            transform.scale.max_element(),
            arm.map_or(
                crate::ribbons::RibbonSeq::Fixed(0),
                crate::ribbons::RibbonSeq::Host,
            ),
            // No model-alpha source: a placed prop / effect instance is always drawn.
            None,
            Some(fade.clone()),
        );
    }
}
