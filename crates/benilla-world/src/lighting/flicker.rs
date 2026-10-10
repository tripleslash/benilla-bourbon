//! fork: **flame flicker** — the per-light brightness wobble that makes a fire read as FIRE. Ported
//! from pkuzic's benilla-lighting (`lighting/flicker.rs`), which measured and tuned it.
//!
//! Every flame in the world is a CONSTANT otherwise. An authored M2 light block is sampled at its
//! first animation key and never again, and a light this fork derives from a flame emitter
//! (`modern::lights::emissive`) has no track at all. So a wall torch, a candelabra and a bonfire
//! all light their room with the steadiness of a light bulb, which is the one cue that reads as
//! "this is not fire".
//!
//! It costs nothing because of where it lands: `global_light::build_light_data` repacks the whole
//! point table every frame anyway, so a multiplier folded in at pack time is one hash and three
//! sines per packed entry and zero GPU work. Nothing is written back onto the
//! [`super::WorldPointLight`], which keeps the flicker out of every OTHER reader of its intensity:
//! the lamp-shadow ranking (`modern::lights::update_lights`) scores the unmodulated light, so a
//! flame breathing at 5 Hz can never make a cube slot thrash, and the grid's eight-a-cell ranking
//! (`light_bin.wgsl`) reads the pack's steady strengths ([`super::LightRank`]) rather than the
//! committed colour, so it can never trade two flames in and out of a full cell either.
//!
//! **What it must not be** is a strobe. The waveform is band-limited by construction rather than
//! tuned by eye:
//!
//! * **Amplitude is small and ordered by mass.** A candle is a thumbnail of burning gas that a
//!   draught can visibly bend (±12%); a bonfire is a cubic yard of it with a thermal time constant
//!   (±6%). The big fires are STEADIER, not just slower.
//! * **Frequency is bounded, and the fast partials carry the LEAST weight.** Three sines spread
//!   across the kind's band, weighted `0.42 / 0.28 / 0.18` low→high, plus value noise
//!   ([`FlameKind::max_rate`] states the slew bound the test asserts).
//! * **Weights sum to exactly 1**, and every term is zero-mean, so the MEAN multiplier is exactly
//!   1.0 and no room's calibrated brightness moves.
//!
//! It is a function of ABSOLUTE time and a per-light seed only: frame-rate independent,
//! deterministic, and — because the seed hashes the light's own position — never in sync between
//! two candles on the same table. The live gain is `$WOW_FIRE_FLICKER` (browser `?fire_flicker=`):
//! 0 off, 1 the default amplitudes, 2 doubled ([`flicker_gain`]).

use bevy::prelude::*;

/// Which fire this is, and therefore how it burns. The four rungs are pkuzic's intensity ladder
/// (candle 0.6 / torch 1.5 / brazier 2.0 / bonfire 3.0), which the authored M2 fire family's
/// `diffuse_intensity` spans; a derived flame is filed onto it by its emitter's size
/// ([`FlameKind::from_flame_size`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlameKind {
    Candle,
    Torch,
    Brazier,
    Bonfire,
}

impl FlameKind {
    /// The rung an AUTHORED light's `diffuse_intensity` sits on. Cut points are the midpoints of the
    /// rungs (0.6 / 1.5 / 2.0 / 3.0): the authored records carry brightness honestly, so how bright
    /// the artist made the fire is what kind of fire it is.
    pub fn from_intensity(i: f32) -> Self {
        match i {
            v if v < 1.0 => Self::Candle,
            v if v < 1.75 => Self::Torch,
            v if v < 2.5 => Self::Brazier,
            _ => Self::Bonfire,
        }
    }

    /// fork: the rung a DERIVED flame sits on, from its emitter's peak particle size (yd) — the one
    /// honest magnitude a flame emitter carries. pkuzic's corpus sweep: a candle authors ~0.1 yd
    /// particles, a wall torch ~0.3, a brazier ~0.6, a bonfire ~1.5; the cuts sit between them.
    pub fn from_flame_size(size: f32) -> Self {
        match size {
            v if v < 0.2 => Self::Candle,
            v if v < 0.45 => Self::Torch,
            v if v < 1.0 => Self::Brazier,
            _ => Self::Bonfire,
        }
    }

    /// Peak excursion of the intensity multiplier at `$WOW_FIRE_FLICKER 1` — `0.12` = ±12%.
    ///
    /// Ordered candle > torch > brazier > bonfire because FLAME MASS is what damps a fire: a candle
    /// tip is grams of gas and every draught in the room shows in it, a bonfire is a slow thermal
    /// body whose brightness barely moves. Reading the order the other way (big fire = big flicker)
    /// is the intuitive mistake, and it looks like a fault rather than like fire.
    pub fn amplitude(self) -> f32 {
        match self {
            Self::Candle => 0.12,
            Self::Torch => 0.10,
            Self::Brazier => 0.08,
            Self::Bonfire => 0.06,
        }
    }

    /// The band (Hz) the three sine partials are spread across — again mass-ordered, and again for
    /// the physical reason: a small flame's turbulence is fast, a big one's is slow. The top of the
    /// candle band (7 Hz) is the highest rate anything here runs at, and it is the LEAST weighted
    /// partial, so nothing in the feature approaches a strobe.
    pub fn band(self) -> (f32, f32) {
        match self {
            Self::Candle => (4.0, 7.0),
            Self::Torch => (3.0, 5.0),
            Self::Brazier => (2.0, 4.0),
            Self::Bonfire => (1.5, 3.0),
        }
    }

    /// A hard upper bound on `|d/dt|` of the intensity multiplier, in units of multiplier per
    /// second — the number that makes "band-limited" an assertion rather than an adjective.
    ///
    /// Every partial's slew is `amp × weight × 2π × f`, bounded by the top of the band; the value
    /// noise adds at most `amp × weight × 3 × f` (a smoothstep's max slope is 1.5, over a span of
    /// at most 2). Loose by design — it bounds the worst possible phase alignment, which the
    /// waveform will not actually reach — but that is what a bound is for.
    pub fn max_rate(self, gain: f32) -> f32 {
        let (_, hi) = self.band();
        let sines = std::f32::consts::TAU * hi * (W_SLOW + W_MID + W_FAST);
        let noise = 3.0 * hi * W_NOISE;
        self.amplitude() * gain.max(0.0) * (sines + noise)
    }
}

// The four term weights. They SUM TO EXACTLY 1, which is the invariant that makes
// [`FlameKind::amplitude`] a true peak bound (all four terms at +1 gives exactly `1 + amp`), and
// the ordering slow > mid > fast > noise is what keeps the eye's impression on the low frequencies
// — the difference between "the flame breathes" and "the lamp is faulty".
const W_SLOW: f32 = 0.42;
const W_MID: f32 = 0.28;
const W_FAST: f32 = 0.18;
const W_NOISE: f32 = 0.12;

/// Peak wobble on the BLUE channel alone at `$WOW_FIRE_FLICKER 1` — the colour-temperature term.
///
/// A flame's colour genuinely moves with its brightness (a starved flame is redder, a fed one
/// whiter), and 3% on blue alone is the cheapest honest expression of that: it needs no colour
/// space, cannot push the light out of gamut, and at this size reads as warmth rather than as a
/// hue shift. It runs at its own slow rate and its own phase, so it never simply tracks the
/// intensity — a flame that dims and reddens in lockstep looks like a dimmer knob.
const BLUE_WOBBLE: f32 = 0.03;

/// The blue term's rate as a fraction of the kind's slowest partial — deliberately SLOWER than any
/// brightness partial. Colour temperature is a property of the whole flame body, not of the tip
/// that flickers.
const BLUE_RATE: f32 = 0.37;

/// fork: the live flicker gain — `$WOW_FIRE_FLICKER` (browser `?fire_flicker=`): `0` off (every
/// flame steady, exactly the pre-flicker look), `1` the default amplitudes, `2` doubled. Read once.
pub fn flicker_gain() -> f32 {
    static GAIN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *GAIN.get_or_init(|| {
        crate::launch_env::var("WOW_FIRE_FLICKER")
            .and_then(|v| v.trim().parse::<f32>().ok())
            .map_or(1.0, |g| g.clamp(0.0, 2.0))
    })
}

/// fork: marks a [`super::WorldPointLight`] that burns rather than shines, with everything the
/// modulation needs — `kind` (how deep and how fast) and `seed` (whose phase).
///
/// Inserted ONCE, at spawn, and never removed or changed: the flicker is a pure function of the
/// component plus the clock, so there is no per-frame write, no `Changed` traffic and no archetype
/// churn — the entity's archetype is settled by the end of the spawn command queue exactly as it
/// was before this component existed.
#[derive(Component, Clone, Copy)]
pub struct FlameFlicker {
    pub kind: FlameKind,
    /// The phase seed — [`flicker_seed`] of the light's own position, so no two flames in a room
    /// are ever in step. Two candles flickering together is the single most artificial thing this
    /// feature could do, and it is what a shared global phase would produce.
    pub seed: u32,
}

/// One frame's modulation: multipliers, not values, so the caller keeps its authored colour and
/// intensity intact and folds these in at the last moment.
#[derive(Clone, Copy, Debug)]
pub struct FlickerMod {
    /// Multiplier on the light's committed intensity. Mean exactly 1.0.
    pub intensity: f32,
    /// Multiplier on the BLUE channel only — the colour-temperature wobble. Mean exactly 1.0.
    pub blue: f32,
}

impl FlickerMod {
    /// The identity — what a non-flame light gets, and what every flame gets at `$WOW_FIRE_FLICKER 0`.
    pub const STEADY: Self = Self {
        intensity: 1.0,
        blue: 1.0,
    };
}

impl FlameFlicker {
    pub fn new(kind: FlameKind, seed: u32) -> Self {
        Self { kind, seed }
    }

    /// This flame's multipliers at absolute time `t` (seconds) and live gain `gain`
    /// ([`flicker_gain`]: 0 off, 1 the authored amplitudes, 2 doubled).
    ///
    /// `t` is ABSOLUTE elapsed time, never a delta: that is what makes the result frame-rate
    /// independent (a 30 fps and a 240 fps client show the same flame at the same instant) and what
    /// lets the whole thing be stateless — no per-light accumulator to keep, restore or desync.
    pub fn at(&self, t: f32, gain: f32) -> FlickerMod {
        if gain <= 0.0 {
            return FlickerMod::STEADY;
        }
        let (lo, hi) = self.kind.band();
        let span = hi - lo;
        // The three partials at the bottom / middle / top of the band, each JITTERED by up to ±12%
        // of the span from its own seed word. Without the jitter every candle in the game would
        // share the same three frequencies and differ only in phase, which beats into a visible
        // collective rhythm across a room full of them; with it, no two flames ever repeat.
        let f = |k: u32, frac: f32| {
            let jitter = (unit(self.seed ^ mix(0x9e37_79b9u32.wrapping_mul(k + 1))) - 0.5) * 0.24;
            (lo + span * (frac + jitter)).clamp(lo * 0.85, hi)
        };
        let phase = |k: u32| unit(mix(self.seed.wrapping_add(0x85eb_ca6b).wrapping_mul(k + 3)));
        let s = |k: u32, frac: f32| (std::f32::consts::TAU * (f(k, frac) * t + phase(k))).sin();
        // Three sines plus one smoothed value-noise term. The noise is what stops the sum reading
        // as a sum of sines — a pure harmonic stack has an audible-to-the-eye periodicity at the
        // beat of its partials, and a fire has none. It runs at the SLOW edge of the band so it
        // contributes shape, not speed.
        let w = W_SLOW * s(0, 0.0)
            + W_MID * s(1, 0.5)
            + W_FAST * s(2, 1.0)
            + W_NOISE * value_noise(t * lo, self.seed ^ 0x1234_5678);
        let amp = self.kind.amplitude() * gain;
        // The blue term is independent: its own phase, its own (slower) rate. See [`BLUE_WOBBLE`].
        let b = (std::f32::consts::TAU * (lo * BLUE_RATE * t + phase(7))).sin();
        FlickerMod {
            // `max(0)` is belt-and-braces: with `Σw == 1` and the cvar clamped to 2 the worst case
            // is `1 − 0.24`, and a negative multiplier could not be committed anyway.
            intensity: (1.0 + amp * w).max(0.0),
            blue: (1.0 + BLUE_WOBBLE * gain * b).max(0.0),
        }
    }
}

/// fork: does this light burn, and as what? The one rule every spawn lane calls, so a candle cannot
/// flicker as a candle in a WMO and as a bonfire on a transport.
///
/// * `flame` — a derived light whose winning source was a FLAME EMITTER
///   (`modern::lights::emissive::Emissive::flame`, which also files its kind). That is a fire by
///   construction, whatever colour it burns: a purple ogre torch and a green magic brazier are
///   flames with strange chemistry. **The route decides, not the hue.**
/// * `synthetic` without `flame` — a light derived from a glow card or a lit window: a pane of
///   glass. Behind glass a flame does not visibly flicker, and a street lamp that did would read as
///   a fault. **No flicker.**
/// * An AUTHORED light block or MOLT fixture — flicker iff its colour is WARM ([`is_warm`]),
///   filed by its intensity. The authored corpus is overwhelmingly candles and torches at that hue;
///   what it is NOT are neutral fill lights and cold magic sources, which the warmth test excludes.
///   A heuristic, and the reason `$WOW_FIRE_FLICKER=0` exists.
pub fn flame_kind_for(
    flame: Option<FlameKind>,
    synthetic: bool,
    color: [f32; 3],
    intensity: f32,
) -> Option<FlameKind> {
    if flame.is_some() {
        return flame;
    }
    if synthetic {
        return None; // the lamp/emissive route — glass, not flame
    }
    is_warm(color).then(|| FlameKind::from_intensity(intensity))
}

/// The warmth test for an AUTHORED source: strictly descending `r > g > b` with a saturation
/// (`max − min`, which for a descending triple is `r − b`) over `0.2`.
///
/// Both halves earn their place. The ORDER alone admits `(0.9, 0.88, 0.86)` — the near-white fill
/// light, which must not breathe. The SATURATION alone admits a cold blue fixture. Together they
/// name the warm family and nothing else; the shipped warm fixtures clear the floor by a wide
/// margin (a candle's authored `(1.0, 0.72, 0.40)` sits at 0.60).
pub fn is_warm(c: [f32; 3]) -> bool {
    c[0] > c[1] && c[1] > c[2] && c[0] - c[2] > 0.2
}

/// fork: the flicker component a light spawned at `pos` should carry, if any — [`flame_kind_for`]
/// with the seed taken from the position. The one call every spawn site makes.
pub fn flame_flicker(
    flame: Option<FlameKind>,
    synthetic: bool,
    color: [f32; 3],
    intensity: f32,
    pos: Vec3,
) -> Option<FlameFlicker> {
    flame_kind_for(flame, synthetic, color, intensity)
        .map(|kind| FlameFlicker::new(kind, flicker_seed(pos)))
}

/// fork: a light's phase seed from WHERE IT IS.
///
/// Position is the only identity a light has that is stable across a stream-out/stream-in (entity
/// ids are not), so a torch keeps its phase when you walk out of the room and back — a re-seeded
/// flame would visibly jump. Quantised to 1/16 yd before hashing so floating-point noise in the
/// placement transform cannot change the seed between two spawns of the same fixture.
pub fn flicker_seed(p: Vec3) -> u32 {
    let q = |v: f32| (v * 16.0).round() as i32 as u32;
    mix(q(p.x) ^ mix(q(p.y) ^ mix(q(p.z))))
}

/// A 32-bit integer finaliser (the `lowbias32` mix) — the whole "randomness" budget of this
/// feature. Not a PRNG: a hash, so every value it produces is reproducible from its input alone,
/// which is what makes the flicker deterministic per light rather than per session.
fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// A hash word as a `[0, 1)` float — 24 bits, the f32 mantissa, so the conversion is exact.
fn unit(x: u32) -> f32 {
    (mix(x) >> 8) as f32 / 16_777_216.0
}

/// Smoothed value noise in `[-1, 1]`: hash a value at each integer step of `t` and smoothstep
/// between them.
///
/// Zero-mean by construction (the per-step values are uniform on `[-1, 1]` and the interpolation is
/// a convex blend of two of them), and band-limited by construction too — its steepest slope is
/// `1.5 × span` per unit of `t`, which is what [`FlameKind::max_rate`] budgets for. That
/// combination is the reason it is value noise and not a random walk or a filtered PRNG: both of
/// those drift off the mean and neither has a slope bound.
fn value_noise(t: f32, seed: u32) -> f32 {
    let i = t.floor();
    let u = t - i;
    let u = u * u * (3.0 - 2.0 * u); // smoothstep — C1 at the knots, so no slope discontinuity
    let k = i as i64 as u32;
    let a = unit(seed ^ mix(k)) * 2.0 - 1.0;
    let b = unit(seed ^ mix(k.wrapping_add(1))) * 2.0 - 1.0;
    a + (b - a) * u
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [FlameKind; 4] = [
        FlameKind::Candle,
        FlameKind::Torch,
        FlameKind::Brazier,
        FlameKind::Bonfire,
    ];

    /// The weight budget is the invariant everything else rests on: it makes
    /// [`FlameKind::amplitude`] a true peak bound and the mean exactly 1.
    #[test]
    fn the_term_weights_sum_to_one() {
        assert!((W_SLOW + W_MID + W_FAST + W_NOISE - 1.0).abs() < 1e-6);
    }

    /// GOLDEN — the tuned exposure must survive the feature. A flicker whose mean is not 1 is a
    /// brightness change wearing a flicker's clothes, and it would silently retune every interior.
    #[test]
    fn the_mean_multiplier_is_one_over_a_long_window() {
        for kind in KINDS {
            for seed in [1u32, 0xdead_beef, 12345, 7] {
                let f = FlameFlicker::new(kind, seed);
                let n = 60_000;
                let (mut si, mut sb) = (0.0f64, 0.0f64);
                for k in 0..n {
                    let m = f.at(k as f32 * 0.005, 1.0); // 300 s at 200 Hz
                    si += f64::from(m.intensity);
                    sb += f64::from(m.blue);
                }
                let (mi, mb) = (si / f64::from(n), sb / f64::from(n));
                assert!((mi - 1.0).abs() < 0.005, "{kind:?}/{seed}: mean {mi}");
                assert!((mb - 1.0).abs() < 0.002, "{kind:?}/{seed}: blue mean {mb}");
            }
        }
    }

    /// GOLDEN — the anti-strobe assertion. Excursion inside the stated amplitude, and slew inside
    /// the stated bound: this is the test that would fail if someone widened a band or reweighted
    /// the partials toward the fast end.
    #[test]
    fn the_waveform_is_bounded_in_amplitude_and_in_slew() {
        for kind in KINDS {
            for gain in [1.0f32, 2.0] {
                let f = FlameFlicker::new(kind, 0xabcd_1234);
                let dt = 0.0005;
                let bound = kind.max_rate(gain);
                let mut prev = f.at(0.0, gain).intensity;
                for k in 1..40_000 {
                    let m = f.at(k as f32 * dt, gain).intensity;
                    let amp = kind.amplitude() * gain;
                    assert!(
                        (m - 1.0).abs() <= amp + 1e-4,
                        "{kind:?} gain {gain}: excursion {} over amp {amp}",
                        m - 1.0
                    );
                    let rate = (m - prev).abs() / dt;
                    assert!(
                        rate <= bound,
                        "{kind:?} gain {gain}: {rate}/s over {bound}/s"
                    );
                    prev = m;
                }
            }
        }
    }

    /// A steadier fire for a bigger fire — the ordering that is the whole "plausible" claim, held
    /// both in the authored amplitude and in the bound the waveform actually reaches.
    #[test]
    fn a_bigger_fire_flickers_less() {
        let amps: Vec<f32> = KINDS.iter().map(|k| k.amplitude()).collect();
        assert!(amps.windows(2).all(|w| w[0] > w[1]), "{amps:?}");
        let rates: Vec<f32> = KINDS.iter().map(|k| k.max_rate(1.0)).collect();
        assert!(rates.windows(2).all(|w| w[0] > w[1]), "{rates:?}");
    }

    /// Deterministic per seed, and DIFFERENT per seed — two candles on one table must not breathe
    /// together, which is the failure a shared global phase would produce.
    #[test]
    fn the_phase_is_deterministic_and_unshared() {
        let a = FlameFlicker::new(FlameKind::Candle, flicker_seed(Vec3::new(1.0, 2.0, 3.0)));
        let b = FlameFlicker::new(FlameKind::Candle, flicker_seed(Vec3::new(1.5, 2.0, 3.0)));
        assert_ne!(a.seed, b.seed);
        let same = FlameFlicker::new(FlameKind::Candle, flicker_seed(Vec3::new(1.0, 2.0, 3.0)));
        let mut apart = 0.0f32;
        for k in 0..2000 {
            let t = k as f32 * 0.01;
            assert_eq!(a.at(t, 1.0).intensity, same.at(t, 1.0).intensity);
            apart = apart.max((a.at(t, 1.0).intensity - b.at(t, 1.0).intensity).abs());
        }
        assert!(
            apart > 0.05,
            "two neighbouring candles ran in step: {apart}"
        );
    }

    /// `$WOW_FIRE_FLICKER 0` is the off switch — a hard identity, not merely a small wobble.
    #[test]
    fn gain_zero_is_steady_and_gain_two_is_double() {
        let f = FlameFlicker::new(FlameKind::Torch, 99);
        for k in 0..500 {
            let t = k as f32 * 0.02;
            assert_eq!(f.at(t, 0.0).intensity, 1.0);
            assert_eq!(f.at(t, 0.0).blue, 1.0);
            // Every term is linear in the gain, so doubling it doubles the excursion exactly.
            let (a, b) = (f.at(t, 1.0).intensity, f.at(t, 2.0).intensity);
            assert!(((b - 1.0) - 2.0 * (a - 1.0)).abs() < 1e-5);
        }
    }

    /// The route rule: a purple magic flame flickers, a lamppost does not, a white fill light does
    /// not — and the intensity ladder files each on its own rung.
    #[test]
    fn only_flames_flicker() {
        // Flame route wins over any colour test: a green magic brazier is still a fire.
        assert_eq!(
            flame_kind_for(Some(FlameKind::Brazier), true, [0.4, 1.0, 0.45], 2.0),
            Some(FlameKind::Brazier)
        );
        // Glow-card / window route: glass, never a flicker — even at a warm hue.
        assert_eq!(flame_kind_for(None, true, [1.0, 0.72, 0.40], 1.5), None);
        // Authored + warm => flickers, filed by intensity.
        assert_eq!(
            flame_kind_for(None, false, [1.0, 0.72, 0.40], 0.6),
            Some(FlameKind::Candle)
        );
        // Authored + near-white fill, and authored + cold: neither burns.
        assert_eq!(flame_kind_for(None, false, [0.9, 0.88, 0.86], 1.5), None);
        assert_eq!(flame_kind_for(None, false, [0.45, 0.65, 1.0], 1.5), None);
        // The ladder itself, at the rungs.
        for (i, k) in [
            (0.6, FlameKind::Candle),
            (1.5, FlameKind::Torch),
            (2.0, FlameKind::Brazier),
            (3.0, FlameKind::Bonfire),
        ] {
            assert_eq!(FlameKind::from_intensity(i), k, "rung {i}");
        }
        // And a derived flame's size ladder, at pkuzic's measured landmarks.
        for (size, k) in [
            (0.1, FlameKind::Candle),
            (0.3, FlameKind::Torch),
            (0.6, FlameKind::Brazier),
            (1.5, FlameKind::Bonfire),
        ] {
            assert_eq!(FlameKind::from_flame_size(size), k, "size {size}");
        }
    }
}
