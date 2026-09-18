//! **Lights from things that only ever glowed** (fork-owned; `FORK.md` P8).
//!
//! Every tier up to Ultra lights from exactly the sources the reference commits: an M2 `type==1`
//! light block, a WMO `MOLT` omni, a torch in an NPC's hand. That is the whole point of those tiers —
//! "a modern frame lights from exactly the lights a classic frame would, it just stops being limited
//! to the three nearest".
//!
//! Cinematic is the tier that stops being true, on purpose, because **most of the things that look
//! like lights in 1.12 are not lights**. A weapon's enchant glow is pure particle emitters
//! (`BlueGlow_Med`: zero render batches, one additive emitter on `Flare.blp`); so are a great many
//! braziers, candles, sconces and hearths, which author a flame card and no light block at all. The
//! reference drew them as sprites over an unlit wall and that was the look. Asked for lamps that
//! light and swords that glow, there is nothing to switch on — the light has to be *derived* from
//! what the artist did author.
//!
//! **What it derives it from.** An emitter's over-life colour ramp ([`OverLife::color`]) is the
//! artist's own statement of what colour the thing emits and how bright, keyed three times across a
//! particle's life with the additive weight in `A`. [`synthesize`] takes the ramp's
//! opacity-weighted mean, so a flame that fades to nothing over its life contributes what it
//! actually puts in the frame rather than its hottest key, and scales it by the emitter's authored
//! particle size — a hearth's two-yard flames and a rune's inch-high sparks are not the same light.
//!
//! **What it refuses**, and this list is the reason the tier is affordable at all:
//! - a model that already authors a casting M2 light. The artist said what it emits; nothing here
//!   improves on that, and doubling it would make every torch in the world twice as bright.
//! - a non-additive emitter. Smoke, debris, splashes and alpha-keyed chips are not emission —
//!   `Add` is the blend an artist reaches for when they mean "this glows".
//! - a `lit` emitter: one the scene's light multiplies is by definition being lit, not emitting.
//! - a ramp too dark to matter once weighted ([`MIN_EMISSION`]).
//!
//! In a corpus sweep those four take out the overwhelming majority of emitters, which is what keeps
//! the 256-entry shared table from filling with dust motes.

use benilla_assets::{ModelEmitter, ModelLight};
use benilla_formats::{OverLife, ParticleBlend};

/// The weighted emission below which a source is not worth a table slot. In the authored 0..1 colour
/// space, after the opacity weighting — a dust mote's ramp lands two orders of magnitude under a
/// flame's.
const MIN_EMISSION: f32 = 0.02;

/// Authored particle half-extent (yd) that reads as a "one unit" emitter. A flame card is about
/// this; the scale below is relative to it, so a hearth outweighs a rune and neither is hard-coded.
const REFERENCE_SIZE: f32 = 0.35;

/// How much of the authored emission becomes light. Emission is a *sprite's* brightness, not a
/// photometric quantity, so this is a look-dev constant and not a derivation — it is set where a
/// campfire synthesized from its flame ramp lands within a stop of the campfires that do author a
/// light block, which is the only calibration available and the only one that matters: the two
/// families have to sit side by side in one street without reading as two different renderers.
const EMISSION_TO_INTENSITY: f32 = 0.55;

/// Cap on one synthesized source's intensity scale. A few emitters author a ramp that saturates
/// every channel at full opacity; unclamped they out-shine a forge.
const MAX_INTENSITY: f32 = 1.5;

/// A synthesized source, in exactly the terms `terrain_stream::point_light` takes: the authored
/// linear RGB and the intensity scale that multiplies it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emissive {
    pub color: [f32; 3],
    pub intensity_scale: f32,
}

/// One drawn thing, as the synthesis reads it: how it blends, whether the scene's light multiplies
/// it, and its over-life ramp. The three fields of a `ParticleEmitterDef` this cares about — taken
/// apart from the def so the law below can be stated over values a test can build, instead of over
/// a fifty-field parse record.
pub type Source<'a> = (ParticleBlend, bool, &'a OverLife);

/// The light a model emits by virtue of what it draws, or `None` when it emits nothing this tier
/// should stand in for.
///
/// `has_authored_light` is whether the model ships a **casting** M2 light of its own. When it does
/// the answer is always `None`: the artist already said what this emits, and a second light would
/// make every torch in the world twice as bright. That is the single most important line here.
pub fn synthesize<'a>(
    has_authored_light: bool,
    sources: impl IntoIterator<Item = Source<'a>>,
) -> Option<Emissive> {
    if has_authored_light {
        return None;
    }
    // Several emitters on one model (a fire's flame plus its embers plus its glow) are one source:
    // they occupy the same handful of inches and a separate light each would be three lights, three
    // table slots and three cube-shadow candidates for one hearth. Summed in energy, averaged in hue.
    let mut weight = 0.0f32;
    let mut sum = [0.0f32; 3];
    for (blend, lit, ramp) in sources {
        let Some((r, g, b, w)) = emission_of(blend, lit, ramp) else {
            continue;
        };
        weight += w;
        sum[0] += r * w;
        sum[1] += g * w;
        sum[2] += b * w;
    }
    if weight < MIN_EMISSION {
        return None;
    }
    // The hue is the energy-weighted mean of the contributors, kept at unit peak so `color` stays an
    // authored colour and all of the magnitude rides `intensity_scale` — which is the split
    // `point_light` expects (it recovers `colour × intensity` exactly).
    let peak = sum[0].max(sum[1]).max(sum[2]);
    if peak <= 0.0 {
        return None;
    }
    Some(Emissive {
        color: [sum[0] / peak, sum[1] / peak, sum[2] / peak],
        intensity_scale: (weight * EMISSION_TO_INTENSITY).min(MAX_INTENSITY),
    })
}

/// [`synthesize`] over a loaded model's own parts — the form both spawners call.
pub fn for_model(lights: &[ModelLight], emitters: &[ModelEmitter]) -> Option<Emissive> {
    synthesize(
        lights.iter().any(|l| l.def.casts()),
        emitters
            .iter()
            .map(|e| (e.def.blend, e.def.lit, &e.def.over_life)),
    )
}

/// One source's contribution: `(r, g, b, weight)`, or `None` if it is not emission at all.
fn emission_of(blend: ParticleBlend, lit: bool, ramp: &OverLife) -> Option<(f32, f32, f32, f32)> {
    // Additive only. `Add` is what an artist picks to mean "this glows"; `Alpha` is smoke,
    // `AlphaKey` and `Opaque` are chips and debris. None of the three emits.
    if blend != ParticleBlend::Add {
        return None;
    }
    // A `lit` emitter is multiplied by the scene's light — it is being lit, so it is not a light.
    if lit {
        return None;
    }
    // The ramp's opacity-weighted mean over the three keys. `A` is the additive weight, so a flame
    // that fades to nothing contributes what it actually puts in the frame and not its hottest key.
    let mut alpha_sum = 0.0f32;
    let mut rgb = [0.0f32; 3];
    for key in &ramp.color {
        let a = key[3].clamp(0.0, 1.0);
        alpha_sum += a;
        for c in 0..3 {
            rgb[c] += key[c].max(0.0) * a;
        }
    }
    if alpha_sum <= 0.0 {
        return None;
    }
    let mean = [rgb[0] / alpha_sum, rgb[1] / alpha_sum, rgb[2] / alpha_sum];
    let luminance = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2];
    // Scaled by how big the thing actually is. The size ramp's peak, not its mean: a flame's
    // authored size is what it reaches, and a two-yard hearth is not an inch-high rune.
    let size = ramp.scale.iter().copied().fold(0.0f32, f32::max);
    let scale = (size / REFERENCE_SIZE).clamp(0.25, 3.0);
    // The mean alpha carries "how much of its life this thing is actually bright".
    let weight = luminance * scale * (alpha_sum / ramp.color.len() as f32);
    (weight >= MIN_EMISSION).then_some((mean[0], mean[1], mean[2], weight))
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_formats::CellRamp;

    fn ramp(keys: [[f32; 4]; 3], size: f32) -> OverLife {
        OverLife {
            mid: 0.5,
            color: keys,
            scale: [size; 3],
            head_cells: [CellRamp::new(0, 0); 2],
            tail_cells: [CellRamp::new(0, 0); 2],
            repeat: [1.0; 2],
        }
    }

    fn glow() -> OverLife {
        ramp([[1.0, 0.6, 0.2, 1.0]; 3], REFERENCE_SIZE)
    }

    fn add(r: &OverLife) -> Source<'_> {
        (ParticleBlend::Add, false, r)
    }

    /// GOLDEN — **the authored answer always wins.** A model that ships a casting M2 light gets no
    /// synthesized one however bright its flames are. Without this every torch in the world lights
    /// twice: once from its light block and once from the flame card drawn on top of it.
    #[test]
    fn a_model_that_authors_a_light_synthesizes_nothing() {
        let r = glow();
        assert_eq!(synthesize(true, [add(&r)]), None);
        assert!(synthesize(false, [add(&r)]).is_some());
    }

    /// GOLDEN — the two refusals that keep the light table from filling with dust. Only an
    /// unlit ADDITIVE emitter is emission; smoke, chips and scene-lit clouds are not.
    #[test]
    fn only_unlit_additive_emitters_emit() {
        let r = glow();
        for blend in [
            ParticleBlend::Alpha,
            ParticleBlend::AlphaKey,
            ParticleBlend::Opaque,
        ] {
            assert_eq!(
                synthesize(false, [(blend, false, &r)]),
                None,
                "{blend:?} is not emission"
            );
        }
        assert_eq!(
            synthesize(false, [(ParticleBlend::Add, true, &r)]),
            None,
            "a scene-lit emitter is being lit, not emitting"
        );
    }

    /// The hue is the artist's, and all of the magnitude rides the intensity scale — the split
    /// `terrain_stream::point_light` expects, since it recovers `colour × intensity`.
    #[test]
    fn the_hue_is_authored_and_the_magnitude_is_separate() {
        let r = ramp([[0.2, 0.4, 1.0, 1.0]; 3], REFERENCE_SIZE);
        let blue = synthesize(false, [add(&r)]).expect("a bright blue glow emits");
        // Peak-normalised, so the channel the artist pushed to 1.0 is still 1.0.
        assert!((blue.color[2] - 1.0).abs() < 1e-5, "{:?}", blue.color);
        assert!((blue.color[0] - 0.2).abs() < 1e-5, "{:?}", blue.color);
        assert!(blue.intensity_scale > 0.0 && blue.intensity_scale <= MAX_INTENSITY);
    }

    /// A ramp that fades out contributes what it actually puts in the frame, not its hottest key —
    /// so a dying ember is dimmer than a steady flame of the same colour.
    #[test]
    fn a_fading_ramp_weighs_less_than_a_steady_one() {
        let s = glow();
        let f = ramp(
            [
                [1.0, 0.6, 0.2, 1.0],
                [1.0, 0.6, 0.2, 0.3],
                [1.0, 0.6, 0.2, 0.0],
            ],
            REFERENCE_SIZE,
        );
        let steady = synthesize(false, [add(&s)]).expect("steady");
        let fading = synthesize(false, [add(&f)]).expect("fading");
        assert!(
            fading.intensity_scale < steady.intensity_scale,
            "{} vs {}",
            fading.intensity_scale,
            steady.intensity_scale
        );
        // Same hue either way: only the weight moved.
        for c in 0..3 {
            assert!((fading.color[c] - steady.color[c]).abs() < 1e-4);
        }
    }

    /// Size is part of it: a hearth's two-yard flames are not an inch-high rune's sparks.
    #[test]
    fn a_bigger_flame_is_a_stronger_light() {
        let keys = [[1.0, 0.6, 0.2, 1.0]; 3];
        let (s, b) = (ramp(keys, 0.1), ramp(keys, 1.5));
        let small = synthesize(false, [add(&s)]).unwrap();
        let big = synthesize(false, [add(&b)]).unwrap();
        assert!(big.intensity_scale > small.intensity_scale);
    }

    /// A model's emitters are ONE source. Three flame cards on one brazier must not become three
    /// table entries — that is three cube-shadow candidates for one fire.
    #[test]
    fn several_emitters_on_one_model_are_one_light() {
        let r = glow();
        let single = synthesize(false, [add(&r)]).unwrap();
        let triple = synthesize(false, [add(&r), add(&r), add(&r)]).unwrap();
        assert!(triple.intensity_scale > single.intensity_scale);
        // Still one answer, and still the same hue.
        for c in 0..3 {
            assert!((triple.color[c] - single.color[c]).abs() < 1e-4);
        }
    }

    /// A ramp the eye cannot see is not a light. This is the cap on how many of the corpus's
    /// emitters can ever reach the table.
    #[test]
    fn a_dark_ramp_emits_nothing() {
        let dim = ramp([[0.01, 0.01, 0.01, 0.2]; 3], REFERENCE_SIZE);
        assert_eq!(synthesize(false, [add(&dim)]), None);
        let clear = ramp([[1.0, 1.0, 1.0, 0.0]; 3], REFERENCE_SIZE);
        assert_eq!(
            synthesize(false, [add(&clear)]),
            None,
            "fully transparent over its whole life"
        );
    }

    /// However bright the ramp, one source can never out-shine a forge.
    #[test]
    fn the_intensity_is_capped() {
        let blazing = ramp([[1.0, 1.0, 1.0, 1.0]; 3], 10.0);
        let e = synthesize(false, [add(&blazing); 8]).unwrap();
        assert!((e.intensity_scale - MAX_INTENSITY).abs() < 1e-5);
    }
}
