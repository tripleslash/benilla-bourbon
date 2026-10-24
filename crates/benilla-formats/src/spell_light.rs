//! fork: **the light a luminous SPELL throws** — which effect models emit (fire, holy and fel burn;
//! frost, nature, arcane and shadow do not), what colour, how bright, and when. Ported from
//! pkuzic's benilla-lighting (`fire_light.rs`'s spell route), which calibrated every rule below
//! against the shipped corpus.
//!
//! Pure rules over a model's path and its particle emitters, so they are testable without a world;
//! the per-instance lifecycle (ramp, hold, burst, reap fade, budget) is the app's
//! (`entities::spell_fx::light`), and the light is only packed on the modern tiers
//! (`lighting::global_light`).
//!
//! The judgement is made from the model's NAME first (the artists named the corpus by school) and
//! from the winning emitter's HUE second. The emitter gate is ADDITIVE + a real particle size, not a
//! texture list: a spell's flame art is `Clouds8x8`, `STAR5A` or `FLARE`, none of which reads as
//! fire by name.

use crate::{OverLife, ParticleBlend, ParticleEmitterDef};

/// The fallback hue for a FIRE effect whose ramp yields none: plain warm orange.
pub const DEFAULT_WARM: [f32; 3] = [1.0, 0.55, 0.2];

/// Below this saturation (`max−min` of the linear RGB) a ramp key carries no hue worth taking.
const MIN_SATURATION: f32 = 0.15;

/// The smallest authored particle (yd, the size ramp's peak) that counts as a FLAME rather than a
/// SPARK. The corpus's real candles sit at 0.056–0.069; the ember trails that share their art sit at
/// 0.019–0.028 and were the whole false-positive population.
const MIN_FLAME_SIZE: f32 = 0.04;

/// Is this model path a SPELL effect? Matched on the path segment so `WORLD\…\SPELLSTONE.M2`
/// cannot trip it.
pub fn is_spell_path(path: &str) -> bool {
    let p = path.replace('/', "\\").to_ascii_uppercase();
    p.starts_with("SPELLS\\") || p.contains("\\SPELLS\\")
}

/// The emitter's peak authored particle half-extent (the over-life size ramp's largest key, yards).
pub fn peak_size(def: &ParticleEmitterDef) -> f32 {
    def.over_life.scale.iter().copied().fold(0.0, f32::max)
}

/// The flat gain on every synthesised SPELL light, folded into the intensity at synthesis. `1.0`
/// would sit a spell light on a torch's rungs; half of that keeps a Fireball, a burning body or a
/// cast hand a subtle pool around the effect rather than a light over the whole scene.
pub const SPELL_LIGHT_GAIN: f32 = 0.5;

/// The HOLY fallback hue: a warm white with a gold cast. Used when a holy effect's ramp yields no
/// usable key, which is the common case rather than the exception — Holy Light's strongest emitter
/// ramps `warm-gold (a 0) -> white (a 0.13) -> acid yellow (a 0)`, so the only key anyone ever SEES
/// is the white one and the only SATURATED one is invisible. Taking the ramp literally there gives
/// a yellow-green flash for the paladin's signature heal; this is the colour the effect reads as.
pub const DEFAULT_HOLY: [f32; 3] = [1.0, 0.95, 0.78];

/// The FEL fallback hue: warlock green. Same role as [`DEFAULT_HOLY`] for the demonic family,
/// whose art is green fire — the imp's Firebolt, Fel Fire Nova, a Felhunter's sacrifice.
pub const DEFAULT_FEL: [f32; 3] = [0.45, 1.0, 0.30];

/// Which luminous school a spell effect belongs to — the whole yes/no of this route, and the
/// user's stated rule: **fire, holy light and fel burn; frost, nature, arcane and shadow do not.**
///
/// The kind decides only WHETHER a light is derived and what its FALLBACK hue is; the light's
/// actual colour still comes from the artist's own over-life ramp wherever the ramp has one
/// ([`spell_color`]), so a blue firework lights blue even though its kind is `Fire`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpellLightKind {
    /// Burning: fireball, immolate, fire nova, a firework's powder, a rocket's exhaust.
    Fire,
    /// Radiant: Holy Light, Smite, a blessing, a resurrection — white-gold.
    Holy,
    /// Demonic green fire: the imp's flame, Fel Fire Nova, Hellfire's ring.
    Fel,
    /// Emits nothing. Frost, nature, arcane, shadow — and everything unrecognised, because the
    /// safe answer for an effect we cannot name is the reference's own behaviour (no light).
    None,
}

impl SpellLightKind {
    /// Does this school light the world at all?
    pub fn lights(self) -> bool {
        !matches!(self, SpellLightKind::None)
    }

    /// The hue to use when the emitter's ramp yields none ([`spell_color`] returned `None`).
    pub fn default_color(self) -> [f32; 3] {
        match self {
            SpellLightKind::Holy => DEFAULT_HOLY,
            SpellLightKind::Fel => DEFAULT_FEL,
            _ => DEFAULT_WARM,
        }
    }

    /// Short label for the offline readout / trace.
    pub fn label(self) -> &'static str {
        match self {
            SpellLightKind::Fire => "fire",
            SpellLightKind::Holy => "holy",
            SpellLightKind::Fel => "fel",
            SpellLightKind::None => "none",
        }
    }
}

/// The NAME rules, **in evaluation order — first match wins, and the order is the whole design**.
///
/// The dark schools come FIRST because their names collide with the luminous ones and the collision
/// is always resolved the same way: `LightningBoltIvus` contains `LIGHT` (holy) and must not be
/// one — `LIGHTNING` ahead of it takes it; `Starfire` and `FaerieFire` contain `FIRE` and are
/// druid/nature effects whose art is a green-white glow, not a flame; `ShadowFlame` contains
/// `FLAME` and is shadow. Putting the vetoes first means a new luminous key can never accidentally
/// re-admit one of them.
///
/// After the vetoes the luminous keys run. Note `FIRE` catches `FIREWORK`, `FIRECRACKER`,
/// `SOULFIRE`, `HELLFIRE`, `RAINOFFIRE`, `FIREBALL`, `FIREBOLT` and `FIREBLAST` in one entry,
/// which is why the list is this short: the artists named the corpus by school already.
const SPELL_NAME_RULES: [(&str, SpellLightKind); 32] = [
    // --- the dark schools, and the name collisions they must win ---
    ("LIGHTNING", SpellLightKind::None),
    ("STARFIRE", SpellLightKind::None),
    ("FAERIE", SpellLightKind::None),
    ("SHADOW", SpellLightKind::None),
    ("FROST", SpellLightKind::None),
    ("ICE", SpellLightKind::None),
    ("FREEZE", SpellLightKind::None),
    ("CHILL", SpellLightKind::None),
    ("BLIZZ", SpellLightKind::None),
    ("SNOW", SpellLightKind::None),
    ("WINTER", SpellLightKind::None),
    ("NATURE", SpellLightKind::None),
    ("THUNDER", SpellLightKind::None),
    ("WRATH", SpellLightKind::None),
    ("ENTANGL", SpellLightKind::None),
    ("POISON", SpellLightKind::None),
    ("ARCANE", SpellLightKind::None),
    ("POLYMORPH", SpellLightKind::None),
    ("CURSE", SpellLightKind::None),
    ("FEAR", SpellLightKind::None),
    ("DRAIN", SpellLightKind::None),
    ("DEATHCOIL", SpellLightKind::None),
    // --- the luminous schools ---
    ("FIRE", SpellLightKind::Fire),
    ("FLAME", SpellLightKind::Fire),
    ("IMMOLAT", SpellLightKind::Fire),
    ("PYRO", SpellLightKind::Fire),
    ("INCINER", SpellLightKind::Fire),
    ("LAVA", SpellLightKind::Fire),
    ("HOLY", SpellLightKind::Holy),
    ("DIVINE", SpellLightKind::Holy),
    ("SMITE", SpellLightKind::Holy),
    ("FEL", SpellLightKind::Fel),
];

/// The school named by the model's **full path**, uppercased — `None` when no rule matches (the
/// hue then decides, [`spell_hue_kind`]).
///
/// Matched on the whole path and not just the basename, because the corpus nests by family
/// (`SPELLS\Enchantments\Shaman_Fire.m2`) and a directory naming the school is exactly as
/// authoritative as a file naming it.
pub fn spell_name_kind(path: &str) -> Option<SpellLightKind> {
    let p = path.replace('/', "\\").to_ascii_uppercase();
    SPELL_NAME_RULES
        .iter()
        .find(|(k, _)| p.contains(k))
        .map(|&(_, kind)| kind)
}

/// The school of an **unnamed** effect, from the colour its winning emitter burns.
///
/// Deliberately conservative — an unrecognised effect defaults to dark, so every arm here has to
/// EARN its light:
/// - **Fel** — green-dominant and saturated. The imp's flame is green fire.
/// - **Holy** — bright and near-white with at most a warm cast (`r` close to `g`, both high, blue
///   not above green). A white-gold flash is holy; a white-BLUE one is frost's burnout, refused by
///   keeping blue under green.
/// - **Fire** — warm and descending (`r >= g >= b`) with a real red-to-blue spread. That is a
///   flame ramp and nothing else here: frost is blue-dominant, nature green-dominant, shadow
///   desaturated-dark, arcane violet (`r` close to `b` above `g`, which the `b <= g` clause
///   rejects).
pub fn spell_hue_kind(c: [f32; 3]) -> SpellLightKind {
    let (r, g, b) = (c[0], c[1], c[2]);
    if g > r && g > b && g - r.max(b) > 0.25 {
        return SpellLightKind::Fel; // green fire
    }
    if r >= 0.75 && g >= 0.75 && b <= g && (r - g).abs() <= 0.25 {
        return SpellLightKind::Holy; // white / white-gold
    }
    if r >= 0.5 && g <= r && b <= g && r - b >= 0.35 {
        return SpellLightKind::Fire; // orange-red
    }
    SpellLightKind::None
}

/// The spell route's emitter gate: ADDITIVE, a real particle size, and a rate that ever rises.
///
/// No texture test — see the module note: the spell corpus's flame art is `Clouds8x8`, `STAR5A`,
/// `FLARE` and `lensflare1`, none of which is in the world route's flame vocabulary, and a
/// texture gate here would light nothing at all. [`MIN_FLAME_SIZE`] is shared with the world route
/// and does the same job it does there: it keeps SPARK emitters (0.02–0.03 yd) from being mistaken
/// for the flame they trail.
pub fn spell_emitter(def: &ParticleEmitterDef) -> bool {
    def.blend == ParticleBlend::Add
        && peak_size(def) >= MIN_FLAME_SIZE
        && def.timing.peak_rate() > 0.0
}

/// An over-life ramp's peak additive weight — the `A` channel's largest key. A spell emitter whose
/// every key is transparent is drawn as nothing and must not decide the light.
pub fn peak_alpha(def: &ParticleEmitterDef) -> f32 {
    def.over_life.color.iter().map(|k| k[3]).fold(0.0, f32::max)
}

/// Which emitter of a spell model IS the effect: `peak size × sqrt(rate) × peak alpha`.
///
/// Three factors, each answering a way a naive pick took the WRONG emitter on a real model:
/// - **size** — the honest one, as in pkuzic's world flame route.
/// - **sqrt(rate)** — a burst of 300 is a bigger event than a trickle of 10, but under a square
///   root so a dense spark shower can't outrank the flame. (The world route's FOURTH root is tuned
///   for steady props; a spell's whole vocabulary is bursts, and the fourth root flattens them.)
/// - **peak alpha** — the one this route adds, and the one that gets fireworks right. Every
///   firework model authors a pale muzzle FLASH at `a 0.39` beside its coloured BURST at `a 1.0`;
///   size and rate alone crown the flash, and `Fireworks_Blue` then lights the sky pale yellow.
pub fn spell_strength(def: &ParticleEmitterDef) -> f32 {
    peak_size(def) * def.timing.peak_rate().max(0.0).sqrt() * peak_alpha(def)
}

/// The effect's colour: the ramp key with the largest **saturation × additive weight**,
/// peak-normalised. `None` when no key is both saturated and visible.
///
/// The `× alpha` is the difference from the world flame route's colour pick, and it is not a refinement — it is the
/// rule. A spell ramp is authored `invisible hue -> VISIBLE colour -> invisible burnout`, i.e. its
/// most saturated key is routinely one drawn at `a 0`: Fireball's tail ramps to pure red at
/// `a 0.16` while it BURNS gold at `a 1.0`, and Holy Light's acid-yellow burnout is authored at
/// `a 0` throughout. Weighting by the additive weight asks "what colour is actually on screen",
/// which is the only question a light derived from it can be answering.
///
/// The floor is a QUARTER of [`MIN_SATURATION`], not the whole of it: the world route's floor is
/// applied to a ramp key at full opacity, and applying the same number to a product that already
/// carries the alpha would reject every faint-but-coloured spell key (Holy Light's tinted cloud
/// sits at `a 0.09`).
pub fn spell_color(over_life: &OverLife) -> Option<[f32; 3]> {
    let score = |k: &[f32; 4]| {
        let (lo, hi) = k[..3]
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        (hi - lo) * k[3]
    };
    let best = over_life
        .color
        .iter()
        .filter(|k| score(k) >= MIN_SATURATION * 0.25)
        .max_by(|a, b| score(a).total_cmp(&score(b)))?;
    let peak = best[0].max(best[1]).max(best[2]);
    (peak > 1e-4).then(|| [best[0] / peak, best[1] / peak, best[2] / peak])
}

/// Bucket a spell light onto the shared intensity rungs (`m2_light_reach`: 0.6 -> 6 yd, 1.5 -> 12,
/// 2.0 -> 16) from the winning emitter's **particle size**, promoted one rung by a BURST.
///
/// Size and not [`spell_strength`], because the strength score exists to RANK emitters within one
/// model and its rate term saturates across models: every spell emitter runs 50–500/s, so a
/// strength bucket would file Holy Light, Fireball and a firework on the same top rung. Size
/// separates them as authored — a missile's core is 0.4–0.8 yd, a nova's cloud 1.1, a spark 0.05.
///
/// The burst promotion is what gets a firework to the brazier rung its 16 yd shell deserves: a
/// `>= 100/s` rate is a detonation, not a flame, and its light should reach further than its
/// (small) particles suggest. Nothing here reaches the 3.0 bonfire rung — a spell must not
/// out-blaze a forge.
pub fn spell_intensity(size: f32, rate: f32) -> (f32, &'static str) {
    let (intensity, bucket) = match size {
        s if s < 0.25 => (0.6, "candle"),
        s if s < 0.75 => (1.5, "torch"),
        _ => (2.0, "brazier"),
    };
    match (rate >= 100.0, bucket) {
        (true, "candle") => (1.5, "torch"),
        (true, "torch") => (2.0, "brazier"),
        _ => (intensity, bucket),
    }
}

/// When the emitter's rate track first reaches half its peak, in seconds from the sequence start —
/// the light's **onset**, i.e. how long after the effect spawns the flame it stands for exists.
///
/// A firework is the whole reason this is read. `G_Firework03Red` is a ~0.7 s model whose shell
/// only detonates at `t = 0.567` (`rate 0 -> 200` there); `SPELLS\Fireworks_Blue` bursts at 1.5 s.
/// A light lit at spawn flashes the rocket's flight, which is exactly backwards. Everything with a
/// constant rate — every missile, every kit glow — answers `0.0` and is unaffected.
pub fn emit_onset(def: &ParticleEmitterDef) -> f32 {
    let peak = def.timing.peak_rate();
    if peak <= 0.0 {
        return 0.0;
    }
    def.timing
        .slot_views()
        .first()
        .and_then(|(_, rate, _)| *rate)
        .and_then(|keys| {
            keys.iter()
                .find(|&&(_, v)| v >= peak * 0.5)
                .map(|&(t, _)| t.max(0.0))
        })
        .unwrap_or(0.0)
}

/// One synthesised spell light — the winning emitter, plus the school it was
/// judged to be and the onset its emitter's rate track names.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticSpell {
    /// Index into the emitter slice handed in — the winning ([`spell_strength`]) emitter. The
    /// caller takes that emitter's model-space position and bone, exactly as the flame route does.
    pub emitter: usize,
    /// The school ([`SpellLightKind`]) — never [`SpellLightKind::None`] here (that is a `None`).
    pub kind: SpellLightKind,
    /// Peak-normalised linear RGB ([`spell_color`], else the kind's default).
    pub color: [f32; 3],
    /// The light's `diffuse_intensity` ([`spell_intensity`] × [`SPELL_LIGHT_GAIN`]).
    pub intensity: f32,
    /// The intensity bucket's name — readout only, comparable with the other two routes'.
    pub bucket: &'static str,
    /// Seconds from the instance's birth before the light comes up ([`emit_onset`]).
    pub onset: f32,
}

/// Is this model a FIREWORK — one of the `World\Goober\G_Firework0*` GameObject shells, or the
/// `SPELLS\Firework*` / `Firecracker` family?
///
/// They are the one luminous population that is NOT under `SPELLS\` (the GameObject shells live in
/// `World\Goober\`), so without this the user's "fireworks" case would be exactly half done: the
/// spell-cast rockets would light and the Darkmoon show would not.
///
/// The LAUNCHER is excluded deliberately. `G_FireworkLauncher01` is permanent scenery that plays
/// its muzzle flash on trigger; every lane here keys its light to the INSTANCE's birth, so a
/// launcher would flash once when it streamed in and then stand dark forever — a light that is
/// wrong in both directions. Its rockets do the lighting.
pub fn is_firework_path(path: &str) -> bool {
    let p = path.replace('/', "\\").to_ascii_uppercase();
    !p.contains("LAUNCHER")
        && (p.contains("FIREWORK") || p.contains("FIRECRACKER") || p.contains("ROMANCANDLE"))
}

/// Should this model take the SPELL route rather than the two world ones? `SPELLS\` content and
/// the firework family, and nothing else — see [`is_spell_path`] / [`is_firework_path`].
pub fn is_spell_light_path(path: &str) -> bool {
    is_spell_path(path) || is_firework_path(path)
}

/// Derive **at most one** light for a spell effect / firework model: pick the strongest emitter
/// ([`spell_strength`]), judge its school (NAME first, then HUE — a name rule always wins, so a
/// frost effect with one red spark stays dark), and take its colour, bucket and onset.
///
/// `None` when the model is not spell content, authors no additive emitter, or belongs to a school
/// that does not emit light — which is the great majority of the corpus, and is the point.
///
/// One light per model, for the reason the flame route gives: a Fireball authors four emitters and
/// every one of them that passed would stack a full light on the same spot.
pub fn synthesize_spell_light<'a>(
    path: &str,
    emitters: impl IntoIterator<Item = &'a ParticleEmitterDef>,
) -> Option<SyntheticSpell> {
    if !is_spell_light_path(path) {
        return None;
    }
    let (emitter, def, _) = emitters
        .into_iter()
        .enumerate()
        .filter(|(_, d)| spell_emitter(d))
        .map(|(i, d)| (i, d, spell_strength(d)))
        // First-wins on a tie, as in the world flame route — mirrored emitter pairs are real
        // here too (a firework shell authors three identical bursts on three bones).
        .fold(
            None,
            |best: Option<(usize, &ParticleEmitterDef, f32)>, cur| match best {
                Some(b) if b.2 >= cur.2 => Some(b),
                _ => Some(cur),
            },
        )?;
    let ramp = spell_color(&def.over_life);
    // The NAME wins over the HUE wherever both speak: an artist naming a model `Frostbolt` has
    // said more about it than one warm key in its ramp has. The hue decides for the unnamed
    // remainder only, and it defaults to dark.
    let kind =
        spell_name_kind(path).unwrap_or_else(|| ramp.map_or(SpellLightKind::None, spell_hue_kind));
    if !kind.lights() {
        return None;
    }
    let (intensity, bucket) = spell_intensity(peak_size(def), def.timing.peak_rate());
    Some(SyntheticSpell {
        emitter,
        kind,
        color: ramp.unwrap_or_else(|| kind.default_color()),
        intensity: intensity * SPELL_LIGHT_GAIN,
        bucket,
        onset: emit_onset(def),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CellRamp, EmitParams, EmitTiming, ParamsNow, ParticleShape};

    /// A minimal emitter def — only the fields the fire rule reads carry meaning.
    fn def(
        texture: Option<&str>,
        blend: ParticleBlend,
        keys: [[f32; 4]; 3],
        size: f32,
        rate: f32,
    ) -> ParticleEmitterDef {
        ParticleEmitterDef {
            flags: 0,
            position: [0.0, 0.0, 1.0],
            bone: 0,
            shape: ParticleShape::Plane,
            blend,
            lit: false,
            texture: texture.map(str::to_string),
            tile_rows: 1,
            tile_cols: 1,
            head_tail: 0,
            timing: EmitTiming::constant(rate),
            params: EmitParams::constant(ParamsNow::default()),
            drag: 0.0,
            tail_time: 0.0,
            spline: None,
            geometry_model: None,
            recursion_model: None,
            angular_velocity_min: [0.0; 3],
            angular_velocity_max: [0.0; 3],
            inherit_scale: 0.0,
            follow_speed1: 0.0,
            follow_scale1: 0.0,
            follow_speed2: 0.0,
            follow_scale2: 0.0,
            twinkle_speed: 0.0,
            twinkle_percent: 1.0,
            twinkle_min: 0.0,
            twinkle_max: 0.0,
            spin: 0.0,
            over_life: OverLife {
                mid: 0.5,
                color: keys,
                scale: [size; 3],
                head_cells: [CellRamp::new(0, 0); 2],
                tail_cells: [CellRamp::new(0, 0); 2],
                repeat: [1.0; 2],
            },
        }
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2e-3)
    }

    /// GOLDEN — the classification table, school by school, on the models the user named. The
    /// LUMINOUS half must light and the DARK half must not; nothing else about the feature matters
    /// if this table is wrong.
    #[test]
    fn spell_school_decides_who_lights() {
        // Fireball's winning emitter (the shockwave tail): gold -> red -> red, all additive.
        let fireball = def(
            Some("Creature\\Spells\\Shockwave4white.blp"),
            ParticleBlend::Add,
            [
                [1.0, 0.847, 0.0, 1.0],
                [0.98, 0.02, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.161],
            ],
            0.833,
            36.0,
        );
        let fire = synthesize_spell_light("Spells\\Fireball_Missile_High.m2", [&fireball])
            .expect("a fireball burns");
        assert_eq!(fire.kind, SpellLightKind::Fire);
        // The colour is the key that is actually ON SCREEN (gold at full alpha), not the pure red
        // burnout the world route's most-saturated rule would have taken.
        assert!(close(fire.color, [1.0, 0.847, 0.0]));
        assert_eq!(fire.bucket, "brazier"); // a 0.83 yd core

        // Frostbolt — blue/cyan, and NAMED frost: dark twice over.
        let frostbolt = def(
            Some("Spells\\Clouds8x8.blp"),
            ParticleBlend::Add,
            [
                [0.384, 0.106, 0.957, 0.671],
                [0.106, 0.435, 0.929, 0.502],
                [0.71, 0.871, 0.984, 0.0],
            ],
            0.3167,
            60.0,
        );
        assert_eq!(
            synthesize_spell_light("Spells\\Frostbolt.m2", [&frostbolt]),
            None
        );

        // Holy Light's strongest emitter: every saturated key is authored INVISIBLE, so the ramp
        // yields nothing and the holy default (warm white) stands in. A most-saturated-key rule
        // would flash the paladin acid yellow.
        let holy = def(
            Some("Spells\\Clouds8x8Fade.blp"),
            ParticleBlend::Add,
            [
                [1.0, 0.729, 0.302, 0.0],
                [1.0, 1.0, 1.0, 0.133],
                [0.965, 1.0, 0.0, 0.0],
            ],
            1.111,
            10.0,
        );
        let light = synthesize_spell_light("Spells\\HolyLight_Impact_Head.m2", [&holy])
            .expect("holy light shines");
        assert_eq!(light.kind, SpellLightKind::Holy);
        assert_eq!(light.color, DEFAULT_HOLY);

        // Shadow Bolt — shadow is the ABSENCE of light, whatever its ramp says. This ramp is the
        // violet one; the NAME is what settles it either way.
        let shadow = def(
            Some("Spells\\Clouds8x8.blp"),
            ParticleBlend::Add,
            [
                [0.4, 0.0, 0.6, 1.0],
                [0.2, 0.0, 0.4, 1.0],
                [0.1, 0.0, 0.2, 0.0],
            ],
            0.5,
            60.0,
        );
        assert_eq!(
            synthesize_spell_light("Spells\\ShadowBolt_Missile.m2", [&shadow]),
            None
        );

        // Entangling Roots / Wrath — nature. Named, so no hue can rescue them.
        let green = def(
            Some("Spells\\Clouds8x8.blp"),
            ParticleBlend::Add,
            [[0.2, 1.0, 0.2, 1.0]; 3],
            0.5,
            40.0,
        );
        assert_eq!(
            synthesize_spell_light("Spells\\EntanglingRoots_State.m2", [&green]),
            None
        );
        assert_eq!(
            synthesize_spell_light("Spells\\Wrath_Missile.m2", [&green]),
            None
        );

        // The imp's Firebolt: NAMED fire (the `FIRE` key), and its green ramp still supplies the
        // colour — a green flame is a flame with odd chemistry, exactly as the world route holds.
        let imp = synthesize_spell_light("Spells\\FireBolt_Missile_Low.m2", [&green])
            .expect("the imp's flame burns");
        assert_eq!(imp.kind, SpellLightKind::Fire);
        assert!(close(imp.color, [0.2, 1.0, 0.2]));

        // …and the demonic family reached by NAME rather than by flame: `Fel_FireNova`.
        assert_eq!(
            synthesize_spell_light("Spells\\Fel_FireNova_Area.m2", [&green])
                .expect("fel fire burns")
                .kind,
            // `FIRE` precedes `FEL` in the table and both light — the kind is a label here, and
            // the light is identical either way.
            SpellLightKind::Fire
        );
    }

    /// GOLDEN — FIREWORKS, both families, and the emitter pick that gets their colour right.
    ///
    /// `Fireworks_Blue_01` authors a pale yellow muzzle FLASH beside its blue SHELL. The flash is
    /// smaller but faster; only the alpha term in [`spell_strength`] puts the shell on top, and
    /// without it the user's "coloured burst" is a yellow one.
    #[test]
    fn a_firework_lights_in_its_own_colour() {
        // SPELLS\Fireworks_Blue_01 #1 — the blue shell (0.667 yd, 50/s, opaque).
        let shell = def(
            Some("ITEM\\OBJECTCOMPONENTS\\WEAPON\\FLARE.BLP"),
            ParticleBlend::Add,
            [
                [0.753, 0.949, 1.0, 1.0],
                [0.078, 0.337, 1.0, 1.0],
                [0.157, 0.631, 1.0, 0.0],
            ],
            0.667,
            50.0,
        );
        // …and #0, the pale flash: smaller, twice the rate.
        let flash = def(
            Some("ITEM\\OBJECTCOMPONENTS\\WEAPON\\FLARE.BLP"),
            ParticleBlend::Add,
            [
                [0.941, 0.98, 0.678, 0.392],
                [0.953, 0.918, 0.525, 0.392],
                [0.973, 0.965, 0.796, 0.0],
            ],
            0.444,
            100.0,
        );
        let fw = synthesize_spell_light("SPELLS\\Fireworks_Blue_01.m2", [&flash, &shell])
            .expect("a firework burns");
        assert_eq!(fw.kind, SpellLightKind::Fire);
        assert_eq!(fw.emitter, 1, "the SHELL wins, not the flash");
        assert!(close(fw.color, [0.078, 0.337, 1.0]));

        // The GameObject family is NOT under `SPELLS\` and reaches the route by name alone.
        let red = def(
            Some("SPELLS\\STAR5A.BLP"),
            ParticleBlend::Add,
            [
                [0.698, 0.0, 0.0, 1.0],
                [0.922, 0.0, 0.0, 1.0],
                [0.875, 0.176, 0.176, 0.0],
            ],
            0.611,
            200.0,
        );
        let go = synthesize_spell_light("World\\Goober\\G_Firework03Red.m2", [&red])
            .expect("the Darkmoon show burns");
        assert!(close(go.color, [1.0, 0.0, 0.0]));
        // 0.61 yd is the torch rung; the 200/s detonation promotes it to the brazier's 16 yd.
        assert_eq!(go.bucket, "brazier");

        // The LAUNCHER is scenery, not a firework: it must stay dark (it would otherwise flash
        // once on stream-in and never again).
        assert!(!is_firework_path("World\\Goober\\G_FireworkLauncher01.m2"));
        assert!(is_firework_path("World/Goober/G_Firework01Blue.m2"));
    }

    /// GOLDEN — the NAME rules beat the HUE rules, which is the asymmetry the whole table rests
    /// on: an artist's filename is evidence about the SCHOOL, a ramp key is evidence about one
    /// particle. The three collisions below are all real shipped names.
    #[test]
    fn a_name_beats_a_hue() {
        assert_eq!(
            spell_name_kind("SPELLS\\LightningBoltIvus_Missile.m2"),
            Some(SpellLightKind::None),
            "LIGHTNING must not read as LIGHT"
        );
        assert_eq!(
            spell_name_kind("SPELLS\\Starfire_Area.m2"),
            Some(SpellLightKind::None),
            "a druid's Starfire is not a fire"
        );
        assert_eq!(
            spell_name_kind("Spells\\FaerieFire_Impact.m2"),
            Some(SpellLightKind::None)
        );
        assert_eq!(
            spell_name_kind("Spells\\HolyLight_Impact_Head.m2"),
            Some(SpellLightKind::Holy)
        );
        assert_eq!(
            spell_name_kind("Spells\\Immolate_State.m2"),
            Some(SpellLightKind::Fire)
        );
        assert_eq!(
            spell_name_kind("Spells\\ArcaneExplosion_Base.m2"),
            Some(SpellLightKind::None)
        );
        assert_eq!(spell_name_kind("Spells\\Doomguard_Summon.m2"), None); // hue decides

        // A FROST effect carrying one warm spark stays dark — the name arm is unconditional.
        let spark = def(
            Some("Spells\\Clouds8x8.blp"),
            ParticleBlend::Add,
            [[1.0, 0.4, 0.0, 1.0]; 3],
            0.5,
            60.0,
        );
        assert_eq!(
            synthesize_spell_light("Spells\\FrostNova_Base.m2", [&spark]),
            None
        );
        // …the same emitter under an unnamed model lights, by hue alone.
        assert_eq!(
            synthesize_spell_light("Spells\\Doomguard_Summon.m2", [&spark])
                .expect("a warm unnamed effect burns")
                .kind,
            SpellLightKind::Fire
        );
    }

    /// The hue classifier's own table — the arms that must NOT fire are the point of it.
    #[test]
    fn the_hue_classifier_defaults_to_dark() {
        assert_eq!(spell_hue_kind([1.0, 0.55, 0.2]), SpellLightKind::Fire);
        assert_eq!(spell_hue_kind([1.0, 0.95, 0.78]), SpellLightKind::Holy);
        assert_eq!(spell_hue_kind([0.2, 1.0, 0.2]), SpellLightKind::Fel);
        assert_eq!(
            spell_hue_kind([0.1, 0.44, 0.93]),
            SpellLightKind::None,
            "frost"
        );
        assert_eq!(
            spell_hue_kind([0.71, 0.87, 0.98]),
            SpellLightKind::None,
            "ice white-blue"
        );
        assert_eq!(
            spell_hue_kind([0.7, 0.4, 1.0]),
            SpellLightKind::None,
            "arcane violet"
        );
        assert_eq!(
            spell_hue_kind([0.3, 0.3, 0.3]),
            SpellLightKind::None,
            "grey"
        );
    }

    /// The gates: a non-spell path never reaches the route (the world routes own it), a
    /// non-additive or spark-sized emitter never wins it, and an emitter-less model yields
    /// nothing.
    #[test]
    fn the_spell_gate_refuses_everything_else() {
        let flame = def(
            Some("Spells\\Clouds8x8.blp"),
            ParticleBlend::Add,
            [[1.0, 0.4, 0.0, 1.0]; 3],
            0.5,
            60.0,
        );
        assert_eq!(
            synthesize_spell_light("World\\Goober\\ElwynnCampfire.m2", [&flame]),
            None,
            "a placed campfire is the FLAME route's, not this one's"
        );
        let smoke = def(
            Some("Spells\\Clouds8x8.blp"),
            ParticleBlend::Alpha,
            [[1.0, 0.4, 0.0, 1.0]; 3],
            0.5,
            60.0,
        );
        assert_eq!(
            synthesize_spell_light("Spells\\Fire_Nova.m2", [&smoke]),
            None
        );
        let sparks = def(
            Some("Spells\\Clouds8x8.blp"),
            ParticleBlend::Add,
            [[1.0, 0.4, 0.0, 1.0]; 3],
            0.02,
            60.0,
        );
        assert_eq!(
            synthesize_spell_light("Spells\\Fire_Nova.m2", [&sparks]),
            None
        );
        assert_eq!(synthesize_spell_light("Spells\\Fire_Nova.m2", []), None);
        // A constant-rate emitter has no fuse: the light is up at t=0.
        assert_eq!(emit_onset(&flame), 0.0);
    }

    /// The reach ladder, and the burst promotion that puts a firework shell on the brazier rung.
    #[test]
    fn spell_reach_buckets_on_size_and_burst() {
        assert_eq!(spell_intensity(0.08, 20.0), (0.6, "candle"));
        assert_eq!(spell_intensity(0.5, 50.0), (1.5, "torch"));
        assert_eq!(spell_intensity(1.1, 10.0), (2.0, "brazier"));
        // A detonation is promoted exactly one rung, and the top rung stays the brazier's:
        // nothing a spell does may out-blaze a forge.
        assert_eq!(spell_intensity(0.08, 300.0), (1.5, "torch"));
        assert_eq!(spell_intensity(0.5, 200.0), (2.0, "brazier"));
        assert_eq!(spell_intensity(1.1, 500.0), (2.0, "brazier"));
    }
}
