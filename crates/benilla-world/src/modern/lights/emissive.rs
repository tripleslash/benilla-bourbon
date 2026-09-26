//! **Lights from things that only ever glowed** (fork-owned; `FORK.md` P8).
//!
//! Every tier up to High lights from exactly the sources the reference commits: an M2 `type==1`
//! light block, a WMO `MOLT` omni, a torch in an NPC's hand. That is the whole point of those tiers —
//! "a modern frame lights from exactly the lights a classic frame would, it just stops being limited
//! to the three nearest".
//!
//! Ultra and Cinematic (P8b; Cinematic alone before it) are the tiers that stop that being true,
//! on purpose, because **most of the things that look like lights in 1.12 are not lights**. A
//! weapon's enchant glow is pure particle emitters (`BlueGlow_Med`: zero render batches, one
//! additive emitter on `Flare.blp`); so are a great many
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
use bevy::math::Vec3;

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
/// fork: the smallest authored particle (yd, the size ramp's peak) that counts as a FLAME rather
/// than a SPARK — pkuzic's corpus measurement (benilla-lighting `fire_light::MIN_FLAME_SIZE`). The
/// real candles sit at 0.056–0.069 yd (`GeneralCandelabra01`, `CandelabraTallWall01`); the ember and
/// exhaust trails that share their additive, unlit look — a zeppelin's cinders, a mount's embers,
/// the sparks on a flaming helm — sit at 0.019–0.028 and were the whole false-positive population,
/// each one a table light. The floor goes just under the smallest real candle.
const MIN_FLAME_SIZE: f32 = 0.04;

/// A synthesized source, in exactly the terms `terrain_stream::point_light` takes: the authored
/// linear RGB and the intensity scale that multiplies it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emissive {
    pub color: [f32; 3],
    pub intensity_scale: f32,
    /// fork: the light BURNS — its energy came mostly from particle emitters rather than glow cards,
    /// and its hue is a warm flame's — and this is the kind of fire it is, filed by its strongest
    /// emitter's size ([`crate::lighting::FlameKind::from_flame_size`]). `None` for a corona, a
    /// rune, a pane of glass or a cold magic glow: those do not flicker.
    pub flame: Option<crate::lighting::FlameKind>,
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
    // fork: how much of the energy is particles, and the size of the strongest emitter — whether
    // this light is a flame, and which kind ([`Emissive::flame`]).
    let (mut particles, mut strongest) = (0.0f32, (0.0f32, 0.0f32));
    for (blend, lit, ramp) in sources {
        let Some((r, g, b, w, size)) = emission_of(blend, lit, ramp) else {
            continue;
        };
        weight += w;
        particles += w;
        if w > strongest.0 {
            strongest = (w, size);
        }
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
    let color = [sum[0] / peak, sum[1] / peak, sum[2] / peak];
    let flame = (particles * 2.0 > weight && crate::lighting::is_warm(color))
        .then(|| crate::lighting::FlameKind::from_flame_size(strongest.1));
    Some(Emissive {
        color,
        intensity_scale: (weight * EMISSION_TO_INTENSITY).min(MAX_INTENSITY),
        flame,
    })
}

/// [`combine`] over a loaded model's own parts — the form the doodad spawners call. `idle` is the
/// model's loader-idle file slot, the sequence a placed prop rests in.
pub fn for_model(
    lights: &[ModelLight],
    emitters: &[ModelEmitter],
    submeshes: &[ModelSubmesh],
    idle: usize,
) -> Option<Emissive> {
    combine(
        lights.iter().any(|l| l.def.casts()),
        emitters
            .iter()
            .filter(|e| emits_at_rest(&e.def.timing, idle))
            .map(|e| (e.def.blend, e.def.lit, &e.def.over_life)),
        submeshes
            .iter()
            .filter(|b| shown_at_rest(b.alpha_anim.as_deref(), idle))
            .filter_map(glow_of),
    )
}

/// fork: **whether an emitter fires at all in the sequence its model rests in** — gate on and rate
/// above zero somewhere in slot `idle` ([`benilla_formats::EmitTiming::first_burst`], the
/// reference's `enabled != 0 && rate > 0` birth test). An explosion's flames and a chest's
/// opening sparkle are keyed off in Stand and only fire in the sequence a script arms
/// (`G_BarrelExplode.m2`: all seven emitters off in Stand, on in 157/150); a light derived from
/// them lights a barrel that is not exploding.
pub fn emits_at_rest(timing: &benilla_formats::EmitTiming, idle: usize) -> bool {
    timing.first_burst(Some(idle)).is_some()
}

/// fork: **whether a batch is drawn at all in the sequence its model rests in** — its colour alpha
/// and transparency weight both reach above zero somewhere in slot `idle`. The reference skips a
/// batch whose alpha is `≤ 0` (`0x707680`), so a card keyed hidden in Stand draws nothing there
/// and emits nothing here. `None` is a batch with no animated alpha: drawn.
pub fn shown_at_rest(alpha: Option<&benilla_formats::AlphaAnim>, idle: usize) -> bool {
    let Some(alpha) = alpha else { return true };
    let slot = alpha.seq(Some(idle));
    let peak = |track: Option<&benilla_formats::ScalarAnim>| {
        track.filter(|t| !t.keys.is_empty()).map_or(1.0, |t| {
            t.keys.iter().map(|&(_, v)| v).fold(f32::MIN, f32::max)
        })
    };
    peak(slot.color.as_ref()) > 0.0 && peak(slot.weight.as_ref()) > 0.0
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
    let size = (0..3).map(|i| (hi[i] - lo[i]) * 0.5).fold(0.0f32, f32::max);
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

/// One source's contribution: `(r, g, b, weight, peak size)`, or `None` if it is not emission at all.
fn emission_of(
    blend: ParticleBlend,
    lit: bool,
    ramp: &OverLife,
) -> Option<(f32, f32, f32, f32, f32)> {
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
    // fork: a spark is not a flame ([`MIN_FLAME_SIZE`]) — it trails one, and lights nothing.
    if size < MIN_FLAME_SIZE {
        return None;
    }
    let scale = (size / REFERENCE_SIZE).clamp(0.25, 3.0);
    // The mean alpha carries "how much of its life this thing is actually bright".
    let weight = luminance * scale * (alpha_sum / ramp.color.len() as f32);
    (weight >= MIN_EMISSION).then_some((mean[0], mean[1], mean[2], weight, size))
}

/// fork: P8 — the edge (yd) of the cell a building's lit windows are pooled over. One light per
/// window would be hundreds for a city and a table full of them; one per building would sit in the
/// middle of it and light nothing that is near a window. A storey's worth of wall is the scale at
/// which a row of windows reads as one pool of light on the street below it.
const WINDOW_CELL: f32 = 8.0;
/// Glass (square yards) a pool needs before it is worth a light — a single slit of lit glass is a
/// sprite, not a source.
const MIN_WINDOW_AREA: f32 = 0.6;
/// The glass area (square yards) that reads as a "one unit" window pool.
const REFERENCE_WINDOW_AREA: f32 = 4.0;
/// How much of a lit window becomes light. Look-dev, calibrated against the derived lampposts: a
/// window is a soft glow through a pane, a pool on the cobbles well under a lamp's.
const WINDOW_WEIGHT: f32 = 0.45;
/// How far in front of the glass the light stands (yd). A light IN the pane lights the wall it is
/// set into edge-on and the street hardly at all; a yard out, it lights the sill, the wall around
/// the frame and the ground beneath, which is what a lit window at night does.
const WINDOW_STANDOFF: f32 = 1.0;
/// The most window lights one building contributes. Stormwind's city is one WMO with hundreds of
/// lit panes; past this the biggest pools win and the rest stay sprites.
pub const MAX_WINDOW_LIGHTS: usize = 48;

/// One pool of lit windows, in the building's own (WoW) model space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowGlow {
    /// Where the light stands: the glass's area-weighted centre, [`WINDOW_STANDOFF`] out along it.
    pub pos: [f32; 3],
    /// Linear RGB, from the material's SIDN colour.
    pub color: [f32; 3],
    pub intensity_scale: f32,
}

/// One batch as the window derivation reads it: its SIDN colour (`None` = not a night-glow
/// material), whether it belongs to an interior group, and its geometry.
pub type WindowBatch<'a> = (Option<[u8; 3]>, bool, &'a benilla_formats::RenderSubmesh);

/// fork: P8 — **the light a building's lit windows cast at night.**
///
/// A WMO window is a MOMT **SIDN** material: an authored colour the client adds as GL emission,
/// scaled by the night fraction, so the pane glows after dark. The reference commits no light for
/// it — the glow never reaches the street — and at Cinematic that is the most conspicuous missing
/// light in any town at night. This derives one: the glass of every SIDN batch in an EXTERIOR group
/// (an interior batch's SIDN is dead on its unlit lane, `RenderSubmesh::sidn`) pooled into
/// [`WINDOW_CELL`] cells, each pool a light at its glass's centre stood a yard out in front of it,
/// its colour the SIDN colour and its strength its glass area.
///
/// The night fraction is NOT applied here: the spawn site tags each light `NightGlow`, and the
/// packer scales it per frame, so a window's light follows the pane's own glow through dusk.
pub fn window_glows<'a>(batches: impl IntoIterator<Item = WindowBatch<'a>>) -> Vec<WindowGlow> {
    #[derive(Default)]
    struct Pool {
        area: f32,
        centre: [f32; 3],
        normal: [f32; 3],
        color: [f32; 3],
    }
    let mut pools: std::collections::HashMap<[i32; 3], Pool> = std::collections::HashMap::new();
    for (sidn, interior, geometry) in batches {
        let Some(rgb) = sidn else { continue };
        if interior || rgb == [0, 0, 0] {
            continue;
        }
        // SIDN is gamma bytes; the light table is linear.
        let color = rgb.map(|c| (c as f32 / 255.0).powf(2.2));
        let pos = &geometry.positions;
        for &[ia, ib, ic] in geometry.indices.as_chunks::<3>().0 {
            let [a, b, c] = [ia, ib, ic].map(|i| pos.get(i as usize).copied());
            let (Some(a), Some(b), Some(c)) = (a, b, c) else {
                continue;
            };
            let [a, b, c] = [a, b, c].map(Vec3::from);
            let cross = (b - a).cross(c - a);
            let area = cross.length() * 0.5;
            if area <= 1e-5 {
                continue;
            }
            // The authored normals say which way the pane faces; the winding may not, so they win
            // when present.
            let normal = if geometry.normals.len() == pos.len() {
                let n: Vec3 = [ia, ib, ic]
                    .iter()
                    .map(|&i| Vec3::from(geometry.normals[i as usize]))
                    .sum();
                n.normalize_or_zero()
            } else {
                cross / (2.0 * area)
            };
            let mid = (a + b + c) / 3.0;
            let key = (mid / WINDOW_CELL).floor().as_ivec3().to_array();
            let pool = pools.entry(key).or_default();
            pool.area += area;
            for k in 0..3 {
                pool.centre[k] += mid[k] * area;
                pool.normal[k] += normal[k] * area;
                pool.color[k] += color[k] * area;
            }
        }
    }
    let mut out: Vec<(f32, WindowGlow)> = pools
        .into_values()
        .filter(|p| p.area >= MIN_WINDOW_AREA)
        .map(|p| {
            let centre = Vec3::from(p.centre) / p.area;
            // Panes facing opposite ways in one cell (both sides of a thin wall) cancel, and the
            // light then stays on the glass rather than guessing a side.
            let facing = Vec3::from(p.normal) / p.area;
            let out_dir = if facing.length() > 0.3 {
                facing.normalize()
            } else {
                Vec3::ZERO
            };
            let color = Vec3::from(p.color) / p.area;
            let size = (p.area / REFERENCE_WINDOW_AREA).sqrt().clamp(0.4, 1.6);
            (
                p.area,
                WindowGlow {
                    pos: (centre + out_dir * WINDOW_STANDOFF).to_array(),
                    color: color.to_array(),
                    intensity_scale: WINDOW_WEIGHT * size,
                },
            )
        })
        .collect();
    // Biggest pools first, so a building over the cap keeps the windows that light the most.
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.truncate(MAX_WINDOW_LIGHTS);
    out.into_iter().map(|(_, g)| g).collect()
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
        assert_eq!(
            read.tint, [1.0; 4],
            "white at full alpha — the maximum weight"
        );
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

    /// fork: a SPARK is not a flame. The ember and exhaust trails that ride real fires (a zeppelin's
    /// cinders, a flaming helm's sparks) author 0.02–0.03 yd particles; the smallest real candle
    /// authors 0.056. Below [`MIN_FLAME_SIZE`] an emitter lights nothing, however bright its ramp.
    #[test]
    fn a_spark_is_not_a_light() {
        let keys = [[1.0, 0.6, 0.2, 1.0]; 3];
        let (spark, candle) = (ramp(keys, 0.03), ramp(keys, 0.06));
        assert_eq!(synthesize(false, [add(&spark)]), None);
        assert!(synthesize(false, [add(&candle)]).is_some());
    }

    /// fork: a warm light made of particles BURNS, and is filed by its flame's size; a cold one,
    /// and a corona made of a glow card, does not flicker.
    #[test]
    fn only_a_warm_particle_light_is_a_flame() {
        use crate::lighting::FlameKind;
        let fire = ramp([[1.0, 0.6, 0.2, 1.0]; 3], 0.6);
        assert_eq!(
            synthesize(false, [add(&fire)]).unwrap().flame,
            Some(FlameKind::Brazier)
        );
        let cold = ramp([[0.2, 0.4, 1.0, 1.0]; 3], 0.6);
        assert_eq!(synthesize(false, [add(&cold)]).unwrap().flame, None);
        let corona = combine(false, [], [card([1.0, 0.6, 0.2, 1.0], 0.5)]).unwrap();
        assert_eq!(corona.flame, None, "a glow card is glass, not fire");
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
        for (got, want) in white.color.iter().zip(UNTINTED_GLOW) {
            assert!(
                (got - want).abs() < 1e-4,
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
        let both = combine(
            false,
            [add(&r)],
            [card([1.0, 0.75, 0.4, 1.0], REFERENCE_CARD)],
        )
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

    /// A pane `w`×`h` yards in the wall `x = x0`, facing +X, as one quad.
    fn pane(x0: f32, y: f32, z: f32, w: f32, h: f32) -> benilla_formats::RenderSubmesh {
        benilla_formats::RenderSubmesh {
            positions: vec![
                [x0, y, z],
                [x0, y + w, z],
                [x0, y + w, z + h],
                [x0, y, z + h],
            ],
            normals: vec![[1.0, 0.0, 0.0]; 4],
            indices: vec![0, 1, 2, 0, 2, 3],
            ..Default::default()
        }
    }

    /// GOLDEN — **a lit window lights the street in front of it, at night only.** The report:
    /// "WMOs like streetlamps and windows at night do not emit lights on the cinematic tier". A
    /// SIDN pane stands one light a yard out in FRONT of the glass (its authored normal's side),
    /// in its own SIDN colour; a non-SIDN batch and an interior group's SIDN batch derive nothing.
    #[test]
    fn a_lit_window_stands_a_light_in_front_of_its_glass() {
        let glass = pane(10.0, 0.0, 0.0, 2.0, 2.0);
        let warm = Some([255u8, 200, 120]);
        let glows = window_glows([(warm, false, &glass)]);
        assert_eq!(glows.len(), 1, "{glows:?}");
        let g = glows[0];
        assert!(
            (g.pos[0] - (10.0 + WINDOW_STANDOFF)).abs() < 1e-4,
            "stands out in front: {g:?}"
        );
        assert!(
            (g.pos[1] - 1.0).abs() < 1e-4 && (g.pos[2] - 1.0).abs() < 1e-4,
            "{g:?}"
        );
        assert!(
            g.color[0] > g.color[1] && g.color[1] > g.color[2],
            "keeps the SIDN hue: {g:?}"
        );
        assert!(g.intensity_scale > 0.0);

        assert!(
            window_glows([(None, false, &glass)]).is_empty(),
            "not a night-glow material"
        );
        assert!(
            window_glows([(warm, true, &glass)]).is_empty(),
            "an interior batch's SIDN is dead"
        );
        assert!(
            window_glows([(Some([0, 0, 0]), false, &glass)]).is_empty(),
            "a black SIDN colour emits nothing"
        );
    }

    /// A row of panes in one storey is ONE pool of light, not one light each — and a building
    /// with more lit glass than [`MAX_WINDOW_LIGHTS`] pools keeps the biggest.
    #[test]
    fn a_buildings_windows_pool_and_are_capped() {
        let warm = Some([255u8, 200, 120]);
        let row: Vec<_> = (0..3)
            .map(|i| pane(10.0, 0.5 + i as f32 * 2.2, 1.0, 1.5, 1.5))
            .collect();
        let glows = window_glows(row.iter().map(|b| (warm, false, b)));
        assert_eq!(
            glows.len(),
            1,
            "one storey of one wall is one pool: {glows:?}"
        );

        let street: Vec<_> = (0..200)
            .map(|i| {
                pane(
                    0.0,
                    i as f32 * WINDOW_CELL,
                    0.0,
                    1.5,
                    1.5 + (i % 7) as f32 * 0.2,
                )
            })
            .collect();
        let glows = window_glows(street.iter().map(|b| (warm, false, b)));
        assert_eq!(glows.len(), MAX_WINDOW_LIGHTS);
        // Kept biggest-first: every survivor is at least as bright as the dimmest of them.
        let weakest = glows
            .iter()
            .map(|g| g.intensity_scale)
            .fold(f32::INFINITY, f32::min);
        assert!(glows.iter().all(|g| g.intensity_scale >= weakest));
    }

    /// A slit of glass too small to matter is a sprite, not a light.
    #[test]
    fn a_sliver_of_glass_is_not_a_light() {
        let slit = pane(0.0, 0.0, 0.0, 0.2, 0.5);
        assert!(window_glows([(Some([255u8, 255, 255]), false, &slit)]).is_empty());
    }

    /// An emitter counts only in a slot where its gate is on and its rate above zero: an
    /// explosion keyed off in the rest sequence emits nothing there.
    #[test]
    fn an_emitter_gated_off_at_rest_does_not_count() {
        let timing = benilla_formats::EmitTiming::gated(20.0, &[true, false]);
        assert!(emits_at_rest(&timing, 0), "gate on in slot 0");
        assert!(!emits_at_rest(&timing, 1), "gate off in slot 1");
        let silent = benilla_formats::EmitTiming::gated(0.0, &[true]);
        assert!(!emits_at_rest(&silent, 0), "a zero rate births nothing");
    }

    /// A card whose alpha is keyed to zero across the rest sequence is not drawn there, so it is
    /// no glow; a card with no animated alpha is always drawn.
    #[test]
    fn a_card_hidden_at_rest_does_not_count() {
        let held = |v: f32| benilla_formats::ScalarAnim {
            period: 0.0,
            step: true,
            wrap: true,
            gseq: false,
            keys: vec![(0.0, v)],
        };
        let alpha = benilla_formats::AlphaAnim::new(vec![
            benilla_formats::AlphaSeq {
                color: Some(held(0.0)),
                weight: None,
            },
            benilla_formats::AlphaSeq {
                color: None,
                weight: Some(held(0.6)),
            },
        ])
        .expect("an animated alpha");
        assert!(!shown_at_rest(Some(&alpha), 0), "hidden in slot 0");
        assert!(shown_at_rest(Some(&alpha), 1), "drawn in slot 1");
        assert!(shown_at_rest(None, 0), "no animated alpha: drawn");
    }

    /// GOLDEN — **the barrel that is not exploding.** `G_BarrelExplode.m2` (GameObject display
    /// 472) authors seven emitters, the flames and flare additive and unlit, every one keyed off in
    /// Stand and fired only by the explode sequences (157, 150). None of them emits at rest, while
    /// the Elwynn campfire's flames do.
    #[test]
    fn an_explosion_models_emitters_are_off_at_rest() {
        let data = benilla_formats::wow_data_or_skip!();
        let mut chain = benilla_formats::open_chain(&data).expect("open vanilla patch chain");
        let at_rest = |chain: &mut benilla_formats::Chain, path: &str| -> Vec<bool> {
            let bytes = chain.read_file(path).expect("read model");
            let stand = benilla_formats::parse_m2_animations(&bytes)
                .iter()
                .find(|a| a.anim_id == 0)
                .map(|a| a.seq_index)
                .expect("a Stand sequence");
            benilla_formats::parse_m2_particle_emitters(&bytes)
                .expect("parse emitters")
                .iter()
                .map(|e| emits_at_rest(&e.timing, stand))
                .collect()
        };
        let barrel = at_rest(&mut chain, "World\\Goober\\G_BarrelExplode.m2");
        assert_eq!(barrel.len(), 7, "seven emitters");
        assert!(barrel.iter().all(|on| !on), "all off in Stand: {barrel:?}");
        let campfire = at_rest(
            &mut chain,
            "World\\Azeroth\\Elwynn\\PassiveDoodads\\Campfire\\ElwynnCampfire.m2",
        );
        assert!(
            campfire.iter().all(|&on| on),
            "a fire burns at rest: {campfire:?}"
        );
    }
}
