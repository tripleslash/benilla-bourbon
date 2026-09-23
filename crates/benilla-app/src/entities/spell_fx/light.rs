//! fork: **the light a luminous spell effect throws, and the envelope it throws it on** (ported from
//! pkuzic's benilla-lighting `spell_fx/lifecycle.rs`).
//!
//! Which effects are luminous at all (fire, holy and fel — never frost, nature, arcane or shadow),
//! what colour they burn and how bright is [`benilla_formats::spell_light`], decided per MODEL.
//! What lives here is per INSTANCE: when the light comes up, how it goes out, and how many may
//! exist at once. The light is a child of the effect's own root, so the reap that despawns the
//! root takes the light with it and there is no orphaning path; the envelope only shapes the
//! brightness inside that life.
//!
//! Packed on the modern tiers only (`benilla_world::lighting::SpellFxLight`), never a shadow
//! caster, never a flicker — a spell light already has a shape of its own.

use benilla_assets::coords::wow_to_bevy;
use benilla_world::lighting::{LightOwner, SpellFxLight, WorldPointLight};
use benilla_world::terrain_stream::point_light;
use bevy::prelude::*;

use super::lifecycle::FxDecay;
use crate::entities::DisplayModel;

/// Seconds a spell light takes to come up — short enough to read as a flash, long enough that a
/// light never appears at full strength on a single frame.
pub(crate) const SPELL_LIGHT_RAMP: f32 = 0.1;
/// The default burst fade ([`SpellLightMode::Burst`]) when the caller knows no better span.
pub(crate) const SPELL_BURST_SPAN: f32 = 0.6;
/// How fast a KIT light goes dark once its instance is reaped ([`FxDecay`]) — quicker than the
/// model's own `Decay`: the aura is over the moment the server said so.
pub(crate) const SPELL_REAP_FADE: f32 = 0.25;
/// How many spell lights may be alive at once, across every lane. The shared table packs the 256
/// nearest sources; a raid's worth of casts must never evict a city's torches.
pub(crate) const SPELL_LIGHTS_MAX: usize = 24;

/// The kill switch, `$WOW_SPELL_LIGHT=0` (browser `?spell_light=0`): no spell light is CREATED.
/// Read once.
pub(crate) fn spell_lights_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| benilla_world::launch_env::var("WOW_SPELL_LIGHT").as_deref() != Some("0"))
}

/// Which lifecycle a spell light follows — the one thing its spawn site knows that the model
/// cannot.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum SpellLightMode {
    /// A kit/aura effect on a unit: up on the ramp, HELD while the instance stands, out on the
    /// reap ([`FxDecay`] → [`SPELL_REAP_FADE`]).
    Kit,
    /// A missile: constant while it flies; its arrival despawn is its ending.
    Missile,
    /// An impact / destination effect: full at onset, gone `span` seconds later.
    Burst { span: f32 },
}

/// One live spell light, a child of the effect root it belongs to.
#[derive(Component)]
pub(crate) struct SpellLight {
    /// The [`WorldPointLight::intensity`] at full envelope — captured at spawn, only ever scaled.
    base: f32,
    /// Seconds from spawn before the light comes up — the emitter's own rate-track onset
    /// ([`benilla_formats::spell_light::emit_onset`]): a firework shell detonates half a second
    /// into its clip, and a light lit at spawn would flash the rocket's flight instead.
    onset: f32,
    mode: SpellLightMode,
    /// Seconds alive — also the budget's ordering (oldest dropped first).
    age: f32,
    /// Seconds since this light's host instance was reaped ([`SpellLightMode::Kit`] only).
    reaped: f32,
}

impl SpellLight {
    pub(crate) fn new(base: f32, onset: f32, mode: SpellLightMode) -> Self {
        Self {
            base,
            onset,
            mode,
            age: 0.0,
            reaped: 0.0,
        }
    }

    /// The envelope's current multiplier, `0..=1`: ramp in, the mode's own decay, the reap fade.
    fn envelope(&self) -> f32 {
        let t = self.age - self.onset;
        if t <= 0.0 {
            return 0.0; // the flame this light stands for does not exist yet
        }
        let up = (t / SPELL_LIGHT_RAMP).min(1.0);
        let down = match self.mode {
            // The fade starts where the ramp ended, so a burst's peak is a real (if brief) plateau
            // rather than a spike a low frame rate could skip.
            SpellLightMode::Burst { span } => {
                (1.0 - (t - SPELL_LIGHT_RAMP).max(0.0) / span.max(1e-3)).clamp(0.0, 1.0)
            }
            SpellLightMode::Kit | SpellLightMode::Missile => 1.0,
        };
        let reap = (1.0 - self.reaped / SPELL_REAP_FADE).clamp(0.0, 1.0);
        up * down * reap
    }
}

/// Spawn the light `dm` throws, if its model is a luminous spell, as a child of `root` at the
/// winning emitter's rest position. `owner` is the body it is cast on (a kit effect), whose room
/// gates the light like any carried light; `None` for a free-flying missile or a ground impact.
pub(crate) fn spawn_spell_light(
    commands: &mut Commands,
    path: &str,
    dm: &DisplayModel,
    root: Entity,
    owner: Option<Entity>,
    mode: SpellLightMode,
) {
    if !spell_lights_enabled() {
        return;
    }
    let Some(spell) = benilla_formats::spell_light::synthesize_spell_light(
        path,
        dm.emitters.iter().map(|e| &e.def),
    ) else {
        return;
    };
    let Some(emitter) = dm.emitters.get(spell.emitter) else {
        return;
    };
    let mut light = point_light(spell.color, spell.intensity);
    let base = light.intensity;
    light.intensity = 0.0; // the envelope brings it up
    let mut e = commands.spawn((
        light,
        SpellFxLight,
        SpellLight::new(base, spell.onset, mode),
        Transform::from_translation(wow_to_bevy(emitter.def.position)),
        Visibility::default(),
    ));
    if let Some(owner) = owner {
        e.insert(LightOwner(owner));
    }
    let id = e.id();
    commands.entity(root).add_child(id);
}

/// Run every live spell light's envelope. Writes [`WorldPointLight::intensity`] only on a real
/// change; the packer reads it in `PostUpdate`.
pub(crate) fn advance_spell_lights(
    time: Res<Time>,
    mut lights: Query<(&mut SpellLight, &mut WorldPointLight, Option<&ChildOf>)>,
    reaped: Query<Has<FxDecay>>,
) {
    let dt = time.delta_secs();
    for (mut light, mut point, parent) in &mut lights {
        light.age += dt;
        if light.mode == SpellLightMode::Kit
            && parent.is_some_and(|c| reaped.get(c.parent()).unwrap_or(false))
        {
            light.reaped += dt;
        }
        let want = light.base * light.envelope();
        if (point.intensity - want).abs() > 1e-3 {
            point.intensity = want;
        }
    }
}

/// The ceiling ([`SPELL_LIGHTS_MAX`]): drop the OLDEST beyond it — the newest cast is the one the
/// player is looking at, and age is the one ordering that does not churn frame to frame.
pub(crate) fn budget_spell_lights(mut commands: Commands, lights: Query<(Entity, &SpellLight)>) {
    let live = lights.iter().count();
    if live <= SPELL_LIGHTS_MAX {
        return;
    }
    let mut by_age: Vec<(Entity, f32)> = lights.iter().map(|(e, s)| (e, s.age)).collect();
    by_age.sort_by(|a, b| b.1.total_cmp(&a.1)); // oldest first
    for (light, _) in by_age.into_iter().take(live - SPELL_LIGHTS_MAX) {
        commands.entity(light).try_despawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GOLDEN — the envelope's four legs, on the pure function. The ONSET leg is a firework's
    /// fuse: a light lit at spawn flashes the rocket's flight instead of its burst.
    #[test]
    fn the_spell_envelope_ramps_holds_and_bursts() {
        let at = |mut l: SpellLight, age: f32| {
            l.age = age;
            l.envelope()
        };
        let kit = || SpellLight::new(100.0, 0.0, SpellLightMode::Kit);
        assert_eq!(at(kit(), 0.0), 0.0, "nothing on the spawn frame");
        assert!((at(kit(), SPELL_LIGHT_RAMP * 0.5) - 0.5).abs() < 1e-3);
        assert_eq!(at(kit(), SPELL_LIGHT_RAMP), 1.0);
        assert_eq!(
            at(kit(), 30.0),
            1.0,
            "an aura holds for as long as it stands"
        );
        assert_eq!(
            at(SpellLight::new(1.0, 0.0, SpellLightMode::Missile), 5.0),
            1.0
        );
        let burst = || SpellLight::new(1.0, 0.0, SpellLightMode::Burst { span: 0.6 });
        assert_eq!(at(burst(), SPELL_LIGHT_RAMP), 1.0);
        assert!((at(burst(), SPELL_LIGHT_RAMP + 0.3) - 0.5).abs() < 1e-3);
        assert_eq!(at(burst(), SPELL_LIGHT_RAMP + 0.6), 0.0);
        assert_eq!(at(burst(), 10.0), 0.0, "and stays gone");
        let fuse = || SpellLight::new(1.0, 1.5, SpellLightMode::Burst { span: 0.6 });
        assert_eq!(at(fuse(), 1.4), 0.0);
        assert_eq!(at(fuse(), 1.5 + SPELL_LIGHT_RAMP), 1.0);
        let mut reaped = kit();
        reaped.age = 5.0;
        reaped.reaped = SPELL_REAP_FADE * 0.5;
        assert!((reaped.envelope() - 0.5).abs() < 1e-3);
        reaped.reaped = SPELL_REAP_FADE;
        assert_eq!(reaped.envelope(), 0.0);
    }
}
