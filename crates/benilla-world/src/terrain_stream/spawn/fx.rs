//! Per-model light/particle dressing for placed models: the shared [`point_light`] recipe and the
//! world-baked spawners the terrain path rides (M2/WMO point lights, particle emitters, ribbon
//! trails). A transport prop takes the entity-owned shapes instead (`crate::entities`' `wmo_props`,
//! decision 0474) — riding owners and prop-local children, never a world bake.

use benilla_assets::coords::wow_to_bevy;
use benilla_assets::{ModelEmitter, ModelLight};
use benilla_formats::WmoLight;
use bevy::prelude::*;

use crate::lighting::WorldPointLight;
use crate::particles;

/// Light intensity for an M2 light of unit `diffuse_intensity`, in the units Bevy's `PointLight`
/// used when it carried this (`linear_color × intensity/(4π)`), so `4π` makes the packer read exactly
/// `diffuse_color × diffuse_intensity` — a clean, predictable base. `wow_model.wgsl` then applies the
/// faithful WoW falloff `1/(0.7d+0.03d²)` (decision 0016). There is no gain dial any
/// more: the 0273 `sh_c16.w` gain is retired and the commit is the reference's raw product
/// (`global_light::commit_raw`).
const POINT_LIGHT_INTENSITY: f32 = 4.0 * std::f32::consts::PI;
/// The packed per-light `range` (yd) — since 0285 this is the ≤3-nearest **selection-candidacy**
/// radius around the receiving unit's anchor, not an evaluation cutoff (a committed GL light has
/// none; the shader evaluates the selected lights at any distance, and att(48 yd) ≈ 0.01 is already
/// invisible). Generous so a hall-scale WMO group whose center sits ~20 yd from its wall fixtures
/// still gathers them — erring wide only ever admits negligible candidates.
const POINT_LIGHT_RANGE: f32 = 48.0;

/// The one [`WorldPointLight`] recipe for an authored WoW light source (M2 or WMO MOLT). The light only
/// carries position + colour into the shared packed table; every lit surface — terrain, M2
/// doodads/NPCs, and WMO walls/floors (decision 0273: the reference lights them all) — then selects
/// its unit's ≤3 NEAREST from that table and applies the faithful falloff in-shader (decision 0285;
/// the committed light is diffuse-only). `color` is the authored WoW RGB (linear, hue preserved);
/// `intensity_scale` folds in the per-light authored intensity.
///
/// `pub(crate)`: the terrain path spawns it at a world-baked point below; a transport prop's light
/// ([`crate::entities`]'s `wmo_props`) spawns it as a CHILD of the moving gameobject instead, so
/// propagation carries the source with the hull.
pub fn point_light(color: [f32; 3], intensity_scale: f32) -> WorldPointLight {
    WorldPointLight {
        color,
        intensity: POINT_LIGHT_INTENSITY * intensity_scale.max(0.0),
        range: POINT_LIGHT_RANGE,
    }
}

/// Spawn one [`WorldPointLight`] at `world` (Bevy space), appended to `out` so it rides the placement's
/// lifecycle. The light recipe is [`point_light`].
fn spawn_point_light(
    commands: &mut Commands,
    world: Vec3,
    color: [f32; 3],
    intensity_scale: f32,
    // The rooms this source belongs to, for a light inside a building — `None` outdoors. A torch in
    // a culled room is one the reference never instantiated, so it lights nothing
    // ([`crate::lighting::LightRooms`], decision 0689).
    room: Option<crate::wmo_portal::WmoGroupVis>,
    out: &mut Vec<Entity>,
) {
    let mut e = commands.spawn((
        point_light(color, intensity_scale),
        Transform::from_translation(world),
    ));
    if let Some(room) = room {
        e.insert(crate::lighting::LightRooms(room));
    }
    out.push(e.id());
}

/// Spawn a [`WorldPointLight`] for each casting `type==1` (point) **M2** light of a prop at `transform` (the
/// campfire/torch/brazier/lantern sources). Position is M2 model space at rest (bone matrix identity at
/// rest), placed via the prop's transform like the emitters — a placed prop never animates its light
/// bone, so the rest pose is exact here (the entity path, whose bones DO move, rides joints instead).
///
/// `room` is the prop's own rooms for a WMO prop (`None` for an ADT map doodad) — taken straight off
/// the [`emitter_fade`] built for this placement, so the prop's mesh, its flames, its streamers and
/// its glow are all admitted or refused by one value.
pub(super) fn spawn_lights_for(
    commands: &mut Commands,
    lights: &[ModelLight],
    transform: Transform,
    room: Option<&crate::wmo_portal::WmoGroupVis>,
    out: &mut Vec<Entity>,
) {
    for l in lights.iter().map(|l| &l.def) {
        if !l.casts() {
            continue; // directional lights feed an ambient term; a static `0` visibility key is dark
        }
        let world = transform.transform_point(wow_to_bevy(l.position));
        spawn_point_light(
            commands,
            world,
            l.diffuse_color,
            l.diffuse_intensity,
            room.cloned(),
            out,
        );
    }
}

/// fork: modern lighting (`FORK.md` P8) — **the lamp that is only a flame card.**
///
/// A great many braziers, candles, sconces and hearths author no M2 light at all: the reference drew
/// their fire as an additive sprite over an unlit wall and that was the look. On the Cinematic tier
/// they emit, from a light [`crate::modern::lights::emissive`] derives from the very ramp that
/// sprite is drawn with. Every other tier returns before touching anything, so this is dead weight
/// in a classic, Low, Medium, High or Ultra session — one atomic load per placement.
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
    lights: &[ModelLight],
    emitters: &[ModelEmitter],
    submeshes: &[benilla_assets::ModelSubmesh],
    transform: Transform,
    room: Option<&crate::wmo_portal::WmoGroupVis>,
    out: &mut Vec<Entity>,
) {
    // **No tier gate here.** The light is spawned whatever the tier, tagged
    // [`SynthesizedLight`], and `global_light` drops it from the table when the tier does not want
    // it. Gating the spawn instead made a lamp's fate depend on whether it streamed in before or
    // after the lighting latch — see that type's note.
    let Some(emissive) = crate::modern::lights::emissive::for_model(lights, emitters, submeshes)
    else {
        return;
    };
    // The centroid and fixture radius are `emissive`'s own law, shared with the entity lane —
    // a GameObject brazier is this same derivation over `EntityPart`s instead of `ModelSubmesh`es.
    let Some((local, radius)) = crate::modern::lights::emissive::glow_anchor(
        emitters.iter().map(|e| e.def.position),
        submeshes.iter().map(|b| b.geometry.as_ref()),
    ) else {
        return;
    };
    let before = out.len();
    spawn_point_light(
        commands,
        transform.transform_point(wow_to_bevy(local)),
        emissive.color,
        emissive.intensity_scale,
        room.cloned(),
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
/// table by day, and [`crate::modern::lights::SynthesizedLight`] so it lights only on Cinematic —
/// the tier that derives light from what only ever glowed.
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

/// Spawn a [`WorldPointLight`] for each omni (`type==0`) **WMO MOLT** light at the building's `transform` (the
/// interior fixture sources — forge fire, inn fireplaces, chapel candles/chandeliers). These radiate onto
/// the NPCs/doodads near each fixture and onto the building's own walls/floor (decision 0273). Position
/// is WMO model space. The MOLT curve is the same fixed WoW falloff as the M2 path — byte-verified
/// (wow-re wmo-molt-runtime: the disk attenStart/End do not shape the GL falloff) and confirmed live in
/// the reference GL trace (committed c/l/q = 0/0.7/0.03, diffuse = colour × intensity, ambient 0).
///
/// A fixture belongs to the rooms whose **MOLR** names it (`group_light_refs`, per absolute group
/// index — the same list an interior GameObject's committed light folds), so a dungeon's brazier
/// stops lighting the hillside above it the moment the flood leaves its room. The refs are inverted
/// here, once per placement: a MOLT light no group names keeps no rooms and stays ungated, the same
/// fail-open arm the prop path's empty MODR list takes.
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
        spawn_point_light(
            commands,
            world,
            l.color,
            l.intensity,
            instance
                .filter(|_| !groups.is_empty())
                .map(|instance| crate::wmo_portal::WmoGroupVis { instance, groups }),
            out,
        );
    }
}

/// The **one draw-set gate a placed model's emitters share** — the quad clouds and the ribbon
/// trails alike, built once per placement so the two families can never be handed different
/// answers about whether their model is in the frame's scene.
///
/// `fade` is the owner's bounding sphere as `(radius, MODEL-space centre)`; the world centre is the
/// placement applied to it — the same point the doodad's own visibility gate measures to.
/// `instance` is the WMO placement this model's doodad set belongs to (`None` for an ADT map
/// doodad), carrying the exterior-window exemption for a prop of the building the camera is
/// standing in (0786). `groups` are the rooms of that placement which name this prop — `None` or
/// empty ⇒ **no room gate**, which is the mesh path's own `d.groups.is_empty()` arm, spelled once
/// here so the two cannot diverge (0689/1289). It is borrowed rather than owned because the shared
/// `Arc` is exactly what wants cloning, and an ADT map doodad — the tens-of-thousands case — must
/// not allocate an empty one per placement to say "I have no rooms".
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

/// Spawn a particle-emitter entity for each of a model's emitters at `transform`, appending them to
/// `out` so they ride the placement's lifecycle (despawn when it streams out). Emitters with no
/// resolved texture are skipped by [`particles::spawn_emitter`].
///
/// `host` is the placement's [`PlacementHost`] when it animates (0130 phase 4): each emitter then
/// rides its host bone's ANCHOR (decision 1365 — the collapsed rig's stand-in for the joint,
/// pre-minted by the assembler for exactly these bones) — the torch flame follows the flame-bone
/// jiggle, the inn chandelier's candle flames swing with it. Riding is unconditional given a
/// host: for a bone whose chain never animates the anchor holds its telescoped rest pivot, so
/// `anchor · (position − pivot)` is byte-for-byte the static `placement · position` — no
/// per-emitter classification needed.
///
/// `armed_seq` is the FILE sequence slot this placement's own arm rolled, and the emitters sample
/// their rate/gate tracks against **that** slot. This site used to pass `EmitClock::Pinned` — slot 0,
/// unconditionally — on the reasoning that a placed doodad arms once, so slot 0 is all it can ever
/// play. The arm has rolled a frequency-weighted *variation* since 0130, so that stopped being true
/// and nothing noticed: an emitter whose rate is keyed only in a later variation sampled a flat zero
/// for ever and emitted nothing, on every placement, while still building, pooling and ticking.
/// `benilla-extract partslotscan` counts **947** such emitters across **184** models; the one that
/// surfaced it is the Blasted Lands lightning strike, whose whole burst lives in slot 1
/// (decision 0760, bug B63).
pub(super) fn spawn_emitters_for(
    commands: &mut Commands,
    emitters: &[ModelEmitter],
    transform: Transform,
    host: Option<&super::assemble::PlacementHost>,
    // The shared draw-set gate ([`emitter_fade`]), cloned onto every emitter this model spawns.
    fade: &particles::EmitterFade,
    out: &mut Vec<Entity>,
) {
    let arm = host.and_then(|h| h.arm);
    for em in emitters {
        // Static placements (no anim host — the ~90% tier, and every capture) have no owner to
        // follow; they despawn via the placement's entity list.
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
                // An animated doodad's rig goes when its placement unloads, and that IS this
                // model being destroyed — free the pool with it, the same rule the WMO-prop lane
                // beside this one already states (0826, missed here: a tile crossing left the
                // brazier's flame draining at the old spot).
                on_owner_loss: particles::OwnerLoss::Free,
                ..default()
            },
            // A placed doodad's arm is NOT one-time: it re-rolls its variation every play-window
            // (decision 0768), so the emitters read the slot AND the clip time off the host's live
            // player each frame — the same lane units and GameObjects use, and what the reference's
            // `m2_animate` does for every model without distinction. A pinned slot was right only
            // under the superseded "armed once at load" contract.
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

/// Spawn a ribbon trail for each of a model's ribbon emitters (wisp streamers, the Caverns-of-Time
/// energy trails). A trail rides its host bone's anchor when the placement animates (same ride as
/// [`spawn_emitters_for`]); a static placement's trail rides the first spawned submesh entity (its
/// transform IS the placement). Trails self-despawn when their owner goes, so they don't join
/// `out` — the placement's despawn cascades within a frame.
pub(super) fn spawn_ribbons_for(
    commands: &mut Commands,
    ribbons: &[benilla_assets::ModelRibbon],
    transform: Transform,
    host: Option<&super::assemble::PlacementHost>,
    fallback_owner: Option<Entity>,
    // The SAME draw-set gate the quad clouds one function up are given ([`emitter_fade`]) — a trail
    // is one of this model's emitters and is admitted by exactly the same five terms. It has to be
    // carried explicitly because a trail rides a joint or a submesh, and neither of those states
    // its owner model's fade sphere or its rooms ([`crate::ribbons::RibbonTrail::fade`]).
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
        // The `+0xc0` enable gate reads the placement's live sequence, like its emitters one
        // function up: an animated doodad re-rolls its variation every play window, so there is
        // no arm-once answer. An unanimated placement has no clock and rests in `Stand`.
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
            // No model-alpha source: a placed prop / effect instance is always drawn (0827).
            None,
            Some(fade.clone()),
        );
    }
}
