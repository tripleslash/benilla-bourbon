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
//! **Two kinds of source, because artists authored glow two ways.** A fire is particles; a lamp,
//! a corona, a rune or a lit window is usually an **additive render batch** — a billboard card with
//! a glow texture, blended on top of the scene ([`benilla_assets::ModelSubmesh::additive`], "M2 glow
//! cards / coronae"). The lampposts of Elwynn and Darkshore are the second kind, and a recipe that
//! read only emitters looked straight past them.
//!
//! **What it derives an emitter from.** An emitter's over-life colour ramp ([`OverLife::color`]) is the
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
//!
//! **What it derives a glow card from, and the one place this is an approximation.** A card's
//! colour lives in two places: its authored vertex tint, and its texture. The tint we have — a
//! constant M2 colour track is baked into the vertex colours at parse, so [`glow_emission`] reads
//! them straight off the batch. The texture we do **not**: world art is decoded straight into a
//! native mip chain with `RenderAssetUsages::RENDER_WORLD`, which the render world *takes* rather
//! than copies, so no main-world system can read a texel of it — the same wall P5 hit when it had
//! to voxelize on the GPU instead of baking on a worker.
//!
//! So a card whose tint is authored gets its authored hue, and a card that is white-on-white —
//! tint white, hue entirely in the texture — gets [`UNTINTED_GLOW`], a warm incandescent stand-in.
//! That is a guess, and it is the weakest number in this file: it is right for the lanterns,
//! braziers and hearths that are most of the population, and wrong for a blue arcane crystal, which
//! will light its surroundings amber. `?modern_emissive=0` turns the whole derivation off.

use benilla_assets::{ModelEmitter, ModelLight, ModelSubmesh};
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

/// The hue a glow card gets when it authors no tint of its own — see the module note. Warm
/// incandescent: the lanterns, torches, braziers and hearths this stands in for are most of the
/// population, and the texture that holds the real answer is render-world only.
const UNTINTED_GLOW: [f32; 3] = [1.0, 0.72, 0.38];
/// How near white a card's vertex tint has to be before it counts as "untinted" and takes
/// [`UNTINTED_GLOW`] instead. A card tinted deliberately is tinted well clear of this.
const UNTINTED_LIMIT: f32 = 0.08;
/// A glow card's own contribution scale relative to a particle emitter's. A card is a single
/// steady quad rather than a cloud, and reads brighter for its size.
const GLOW_WEIGHT: f32 = 0.9;
/// Card half-extent (yd) that reads as a "one unit" glow — a lamppost's corona is about this.
const REFERENCE_CARD: f32 = 0.5;

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

/// One authored **glow card**: its alpha-weighted mean vertex tint and its half-extent in yards.
/// [`glow_of`] builds it from a batch; a test builds it directly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glow {
    pub tint: [f32; 4],
    pub size: f32,
}

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
    combine(has_authored_light, sources, [])
}

/// The full recipe: a model's particle emitters and its glow cards together, since a lamp may have
/// both (a lantern's corona card and a candle flame) and they are one light.
pub fn combine<'a>(
    has_authored_light: bool,
    sources: impl IntoIterator<Item = Source<'a>>,
    glows: impl IntoIterator<Item = Glow>,
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
    for glow in glows {
        let Some((r, g, b, w)) = glow_emission(&glow) else {
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

/// [`combine`] over a loaded model's own parts — the form both spawners call.
pub fn for_model(
    lights: &[ModelLight],
    emitters: &[ModelEmitter],
    submeshes: &[ModelSubmesh],
) -> Option<Emissive> {
    combine(
        lights.iter().any(|l| l.def.casts()),
        emitters
            .iter()
            .map(|e| (e.def.blend, e.def.lit, &e.def.over_life)),
        submeshes.iter().filter_map(glow_of),
    )
}

/// One batch read as a glow card, or `None` when it is not one.
///
/// `additive` is the whole test, and it is the artist's own word: the M2 render flag that says
/// "add this on top of the scene rather than mixing it in", which is what a corona, a lamp's halo,
/// a rune and a lit window are and what a wall, a plank and a leaf are not.
pub fn glow_of(batch: &ModelSubmesh) -> Option<Glow> {
    glow_of_geometry(&batch.geometry)
}

/// [`glow_of`] over the parsed batch itself — what the entity lane has in hand (an `EntityPart`
/// keeps its `RenderSubmesh`, not the `ModelSubmesh` the world lane carries).
pub fn glow_of_geometry(geometry: &benilla_formats::RenderSubmesh) -> Option<Glow> {
    if !geometry.additive {
        return None;
    }
    // **An ENV-MAPPED layer is a reflection, not an emitter.** The shine/reflect pass authored over
    // a weapon, a shield or a polished fitting (`ARMORREFLECT3` on the ballista's bolt heads,
    // `BALISTASHINE02` on the loose bolts — 264 shared sections across 221 world doodads, see
    // `RenderSubmesh::section`) is additive exactly as a corona is, so `additive` alone cannot tell
    // the two apart. What can is that such a batch's texture coordinates are GENERATED from the
    // view-space reflection vector rather than authored ([`RenderSubmesh::env_map`]): it is showing
    // the world back to the camera, and a mirror is not a lamp.
    //
    // It is also the batch this recipe is least able to read. An env-mapped layer usually carries
    // no vertex colours — its UVs are frequently every vertex at one point, because the runtime
    // supplies them — so the alpha-weighted mean below degrades to opaque white, which is the
    // LARGEST weight the recipe can assign. That is why the symptom is not a subtle one: the
    // vendor's weapon rack outside Northshire Abbey threw light across the road, from a stand of
    // perfectly ordinary unenchanted swords.
    if geometry.env_map {
        return None;
    }
    if geometry.positions.is_empty() {
        return None;
    }
    // The alpha-weighted mean vertex tint. A constant M2 colour track is baked into the vertex
    // colours at parse (`ModelSubmesh::tint_anim`'s note: a `Some` tint means the static bake was
    // SKIPPED), so for the static case — which every placed lamp is — this is the authored colour.
    let mut tint = [0.0f32; 4];
    if geometry.vertex_colors.is_empty() {
        tint = [1.0; 4];
    } else {
        for c in &geometry.vertex_colors {
            for i in 0..4 {
                tint[i] += c[i];
            }
        }
        for slot in &mut tint {
            *slot /= geometry.vertex_colors.len() as f32;
        }
    }
    // The card's half-extent, from its own vertices — a lamppost's corona against a hearth's blaze.
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in &geometry.positions {
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    let size = (0..3)
        .map(|i| (hi[i] - lo[i]) * 0.5)
        .fold(0.0f32, f32::max);
    Some(Glow { tint, size })
}

/// **Where a model's glowing parts sit, and how far their own geometry reaches** — the centroid the
/// derived light is placed at and the fixture radius clipped out of its cube, in raw WoW model
/// space. `None` when the model has no glowing part at all.
///
/// Shared by both lanes, and that is the point of it living here rather than beside either one. The
/// world lane has `ModelSubmesh`es and the entity lane has `EntityPart`s — different types around
/// the same `RenderSubmesh` — so a centroid written beside one of them is a law the other has to
/// copy, and a copied law is one that drifts. Callers hand over emitter positions and card
/// geometries; everything either lane decides about them is decided here.
///
/// The light sits at the **glowing part's own centroid**, not at the model origin: a wall sconce's
/// flame is a yard out from its mounting point and half a yard up, and a light at the origin would
/// put the glow inside the wall — the one placement error that reads immediately, because the cube
/// shadow then has that wall between the light and the room.
///
/// Plain positions, not bone-composed: a placed prop's light bone is at rest by construction, and a
/// part riding a bone that *does* animate swings by inches, well inside the yard this light stands
/// for.
///
/// The radius is the furthest a glow card's own vertex gets from that centre. Emitters are points
/// and contribute none — a flame has no frame around it to clip.
pub fn glow_anchor<'a>(
    emitter_positions: impl IntoIterator<Item = [f32; 3]>,
    cards: impl IntoIterator<Item = &'a benilla_formats::RenderSubmesh> + Clone,
) -> Option<([f32; 3], f32)> {
    let mut sum = [0.0f32; 3];
    let mut n = 0.0f32;
    for pos in emitter_positions {
        for c in 0..3 {
            sum[c] += pos[c];
        }
        n += 1.0;
    }
    for geometry in cards.clone() {
        if glow_of_geometry(geometry).is_none() {
            continue;
        }
        let positions = &geometry.positions;
        if positions.is_empty() {
            continue;
        }
        let mut mid = [0.0f32; 3];
        for p in positions {
            for c in 0..3 {
                mid[c] += p[c];
            }
        }
        for c in 0..3 {
            sum[c] += mid[c] / positions.len() as f32;
        }
        n += 1.0;
    }
    if n <= 0.0 {
        return None;
    }
    let mid = [sum[0] / n, sum[1] / n, sum[2] / n];
    let mut radius = 0.0f32;
    for geometry in cards {
        if glow_of_geometry(geometry).is_none() {
            continue;
        }
        for p in &geometry.positions {
            let d = (0..3).map(|c| (p[c] - mid[c]).powi(2)).sum::<f32>().sqrt();
            radius = radius.max(d);
        }
    }
    Some((mid, radius))
}

/// A glow card's contribution: `(r, g, b, weight)`.
fn glow_emission(glow: &Glow) -> Option<(f32, f32, f32, f32)> {
    let alpha = glow.tint[3].clamp(0.0, 1.0);
    if alpha <= 0.0 {
        return None;
    }
    let rgb = [
        glow.tint[0].max(0.0),
        glow.tint[1].max(0.0),
        glow.tint[2].max(0.0),
    ];
    let peak = rgb[0].max(rgb[1]).max(rgb[2]);
    if peak <= 0.0 {
        return None;
    }
    // Untinted (white, or no vertex colours at all) means the hue is in a texture the main world
    // cannot read — see the module note. Take the warm stand-in and keep the card's own brightness.
    let spread = peak - rgb[0].min(rgb[1]).min(rgb[2]);
    let hue = if spread <= UNTINTED_LIMIT {
        UNTINTED_GLOW
    } else {
        [rgb[0] / peak, rgb[1] / peak, rgb[2] / peak]
    };
    let scale = (glow.size / REFERENCE_CARD).clamp(0.25, 3.0);
    let weight = peak * alpha * scale * GLOW_WEIGHT;
    (weight >= MIN_EMISSION).then_some((hue[0], hue[1], hue[2], weight))
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

    /// A glow card, as a batch: additive, with an authored tint and a corona's half-extent.
    fn glow_batch(tint: [f32; 4]) -> benilla_formats::RenderSubmesh {
        benilla_formats::RenderSubmesh {
            additive: true,
            positions: vec![
                [-REFERENCE_CARD, 0.0, -REFERENCE_CARD],
                [REFERENCE_CARD, 0.0, REFERENCE_CARD],
            ],
            vertex_colors: vec![tint; 2],
            ..Default::default()
        }
    }

    /// GOLDEN — **a mirror is not a lamp.** An ENV-MAPPED additive layer is the shine/reflect pass
    /// artists author over metal (`ARMORREFLECT3`, `BALISTASHINE02` — 264 shared sections across
    /// 221 world doodads), and it blends exactly like a corona, so `additive` alone reads it as
    /// one. It has to be refused on the one thing that does separate them: its texture coordinates
    /// are generated from the view reflection vector rather than authored.
    ///
    /// The director's report is the shape of the bug: the vendor's weapon rack outside Northshire
    /// Abbey threw light across the road, from a stand of ordinary unenchanted swords whose only
    /// "glow" was their env-map shine. And it was not subtle, for the reason the second assert
    /// pins — such a batch carries no vertex colours, so the tint recipe degrades to opaque white,
    /// the largest weight it can assign. An env-mapped layer is the brightest possible false
    /// positive, not a marginal one.
    #[test]
    fn an_env_mapped_shine_layer_is_not_a_glow_card() {
        let shine = benilla_formats::RenderSubmesh {
            env_map: true,
            ..glow_batch([1.0; 4])
        };
        assert_eq!(
            glow_of_geometry(&shine),
            None,
            "a weapon's reflection pass was read as a light source"
        );
        // The same batch WITHOUT the env-map flag is the brightest card this recipe knows — which
        // is exactly what the rack was contributing before the refusal.
        let lamp = glow_batch([1.0; 4]);
        let read = glow_of_geometry(&lamp).expect("an untinted additive card IS a glow card");
        assert_eq!(read.tint, [1.0; 4], "white at full alpha — the maximum weight");
        assert!(
            combine(false, [], [read]).is_some(),
            "and it synthesizes a real light, so refusing it is what stops the rack glowing"
        );
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

    fn card(tint: [f32; 4], size: f32) -> Glow {
        Glow { tint, size }
    }

    /// GOLDEN — **the lamppost.** The glow around an Elwynn or Darkshore lantern is an additive
    /// billboard card, not a particle emitter, and the first cut of this recipe looked straight
    /// past every one of them. A model with no emitters at all still emits if it carries a card.
    #[test]
    fn a_glow_card_alone_emits() {
        let e = combine(false, [], [card([1.0, 0.75, 0.4, 1.0], REFERENCE_CARD)])
            .expect("a lantern's corona is a light");
        assert!((e.color[0] - 1.0).abs() < 1e-5, "{:?}", e.color);
        assert!(e.color[1] < e.color[0] && e.color[2] < e.color[1], "warm");
        assert!(e.intensity_scale > 0.0);
    }

    /// A card whose tint is white has its hue in a texture the main world cannot read, so it takes
    /// the warm stand-in rather than lighting the street bluish-white. The weakest assumption in
    /// the file, and the one most worth pinning.
    #[test]
    fn an_untinted_card_takes_the_warm_stand_in() {
        let white = combine(false, [], [card([1.0, 1.0, 1.0, 1.0], REFERENCE_CARD)]).unwrap();
        for c in 0..3 {
            assert!(
                (white.color[c] - UNTINTED_GLOW[c]).abs() < 1e-4,
                "{:?} vs {UNTINTED_GLOW:?}",
                white.color
            );
        }
        // An authored tint is NOT overridden: a blue rune stays blue.
        let blue = combine(false, [], [card([0.2, 0.45, 1.0, 1.0], REFERENCE_CARD)]).unwrap();
        assert!((blue.color[2] - 1.0).abs() < 1e-5);
        assert!(blue.color[0] < 0.3, "{:?}", blue.color);
    }

    /// The authored light still wins over a card, exactly as it does over an emitter — a lamp that
    /// ships both a light block and a corona must not light twice.
    #[test]
    fn a_card_cannot_double_an_authored_light() {
        assert_eq!(
            combine(true, [], [card([1.0, 0.75, 0.4, 1.0], REFERENCE_CARD)]),
            None
        );
    }

    /// A bigger card is a stronger light, and a transparent one is none.
    #[test]
    fn a_cards_size_and_alpha_carry() {
        let small = combine(false, [], [card([1.0, 0.75, 0.4, 1.0], 0.1)]).unwrap();
        let big = combine(false, [], [card([1.0, 0.75, 0.4, 1.0], 2.0)]).unwrap();
        assert!(big.intensity_scale > small.intensity_scale);
        assert_eq!(combine(false, [], [card([1.0, 0.75, 0.4, 0.0], 1.0)]), None);
    }

    /// A lamp with both a corona and a candle flame is still ONE light, and both contribute.
    #[test]
    fn cards_and_emitters_combine_into_one_light() {
        let r = glow();
        let both = combine(false, [add(&r)], [card([1.0, 0.75, 0.4, 1.0], REFERENCE_CARD)])
            .expect("both");
        let flame_only = combine(false, [add(&r)], []).unwrap();
        assert!(both.intensity_scale > flame_only.intensity_scale);
    }

    /// However bright the ramp, one source can never out-shine a forge.
    #[test]
    fn the_intensity_is_capped() {
        let blazing = ramp([[1.0, 1.0, 1.0, 1.0]; 3], 10.0);
        let e = synthesize(false, [add(&blazing); 8]).unwrap();
        assert!((e.intensity_scale - MAX_INTENSITY).abs() < 1e-5);
    }
}
