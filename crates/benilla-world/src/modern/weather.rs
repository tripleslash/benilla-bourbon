//! **Modern weather** (fork-owned): how a modern frame draws the reference's weather.
//!
//! The simulation is upstream's and untouched: the same mist puffs are born, streamed and retired
//! by the same laws (`weather::precip::mist`). Only a puff's look differs. The reference's puff is
//! a 12 yd sheet of the fog colour, opaque from 18 yd, and up to 128 of them stream through the
//! camera: against the classic fog wall they are the wall's near edge, but a modern frame's air is
//! a thin haze the scene shows through, and there the sheets alone hid it. A modern puff is a
//! smaller, fainter wisp of that same haze. The storm's hold on the distance is the fog medium's
//! (`volumetrics::media`, `STORM_HAZE`).
//!
//! The storm's light is reworked too ([`storm_atmosphere`]), and heavy rain past it goes darker and
//! colder and throws lightning ([`apply_heavy_rain`]): neither is in 1.12.1, whose weather has no
//! flash and whose storm row is one grey for every hour.

use std::sync::atomic::{AtomicU32, Ordering};

use benilla_formats::Atmosphere;
use bevy::prelude::*;

use crate::lighting::{LightingConsumeSet, WowLighting};
use crate::weather::{WeatherKind, WeatherState};

use super::volumetrics::media;

/// What one mist puff looks like (`weather::precip::mist::push_mist`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PuffLook {
    /// Half the puff quad's edge (yd).
    pub half: f32,
    /// The per-corner distance fade: nothing nearer than `near`, whole from `far` (yd to the eye).
    pub near: f32,
    pub far: f32,
    /// The fade-in and fade-out ramps of a puff's life (s).
    pub fade: f32,
    /// The most a puff covers, over its fades and its texture.
    pub alpha: f32,
    /// Coverage from a smooth analytic falloff (`effect_points::puff_falloff`) instead of
    /// `snowmist01.blp`, whose 4-bit alpha steps in five rings to a peak of 1/3.
    pub procedural: bool,
}

/// A modern frame's puff: 10 yd (0.69 of the reference's area), clear within 8 yd and whole from
/// 20, on 0.7 s ramps, so nothing solid crosses the camera; a procedural `(1 − r²)³` wisp peaking
/// at 0.14, a mean coverage of 0.027 over the quad against the reference texture's 0.087.
/// Together about a sixth of the reference's veil.
pub const PUFF: PuffLook = PuffLook {
    half: 5.0,
    near: 8.0,
    far: 20.0,
    fade: 0.7,
    alpha: 0.14,
    procedural: true,
};

/// The share of a band's colour (its distance from its own luma) a full storm swaps for the storm
/// row's: the rest stays the clear row's hue.
pub const STORM_DESAT: f32 = 0.5;
/// A full storm leaves each band at most this share of its clear luma.
pub const STORM_DARKEN: f32 = 0.85;

/// Rec. 709 luma of a gamma colour.
fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// One band at full storm: the storm row's luma capped at [`STORM_DARKEN`] of the clear's, carrying
/// the clear row's colour scaled with it, [`STORM_DESAT`] of it traded for the storm row's tint.
fn storm_band(clear: [f32; 3], stormy: [f32; 3]) -> [f32; 3] {
    let (yc, ys) = (luma(clear), luma(stormy));
    let y = ys.min(STORM_DARKEN * yc);
    let scale = if yc > 1e-5 { (y / yc).min(1.0) } else { 0.0 };
    [0, 1, 2].map(|i| {
        let tint = (clear[i] - yc) * scale * (1.0 - STORM_DESAT) + (stormy[i] - ys) * STORM_DESAT;
        (y + tint).clamp(0.0, 1.0)
    })
}

/// The storm blend a modern frame lights with. The reference lerps every band toward the storm
/// `LightParams`, and a zone's storm row is one grey for all hours (Elwynn: ambient 78/78/95, fog
/// 82/84/82, zenith 111 at noon and at midnight), so a night storm turns the blue dark grey and
/// brighter than the clear night. Here the storm row's colours only ever darken the clear row's
/// and keep half its hue, and the fog keeps the clear row's distances, so the view reaches as far
/// as on a clear day (the storm's veil is the medium's `STORM_HAZE`); cloud cover and glow blend
/// as the reference's.
pub fn storm_atmosphere(clear: &Atmosphere, stormy: &Atmosphere, t: f32) -> Atmosphere {
    let target = Atmosphere {
        fog_color: storm_band(clear.fog_color, stormy.fog_color),
        sun_diffuse: storm_band(clear.sun_diffuse, stormy.sun_diffuse),
        sun_color: storm_band(clear.sun_color, stormy.sun_color),
        ambient: storm_band(clear.ambient, stormy.ambient),
        sky: std::array::from_fn(|i| storm_band(clear.sky[i], stormy.sky[i])),
        cloud_colors: std::array::from_fn(|i| {
            storm_band(clear.cloud_colors[i], stormy.cloud_colors[i])
        }),
        fog_end: clear.fog_end,
        fog_start_frac: clear.fog_start_frac,
        ..*stormy
    };
    clear.lerp(&target, t)
}

/// fork: **rain clears around the eye.** A streak keeps its whole size from this many yards out
/// and shrinks to nothing inside a quarter of it (`effect_points.wgsl`): the reference's streaks
/// are fixed world-size triangles, so the few drops beside the camera were the widest marks on the
/// screen and a downpour hid the view behind them.
pub const STREAK_NEAR_YD: f32 = 4.0;

/// The streaks' near fade on a modern frame ([`STREAK_NEAR_YD`]); 0 in a classic one, which keeps
/// the reference's.
pub fn streak_near() -> f32 {
    if super::frame_is_modern() {
        STREAK_NEAR_YD
    } else {
        0.0
    }
}

/// The puff a modern frame draws; `None` in a classic one, which keeps the reference's.
pub fn puff_look() -> Option<PuffLook> {
    super::frame_is_modern().then_some(PUFF)
}

/// The colour (gamma) a modern puff takes for the scene's fog colour: the haze's own, the zone
/// look's tint under the same saturation and level caps as the medium (`media::subtle`).
pub fn puff_color(fog_color: [f32; 3]) -> [f32; 3] {
    let tint = super::look::fog_tint();
    media::subtle([0, 1, 2].map(|c| fog_color[c] * tint[c]))
}

/// This frame's weather intensity (any precipitation's effect density, 0 fine) and [`heavy_rain`],
/// for readers without the weather (the cloud deck).
static INTENSITY_NOW: AtomicU32 = AtomicU32::new(0);
static HEAVY_NOW: AtomicU32 = AtomicU32::new(0);

/// This frame's weather intensity on a modern frame, 0 (fine) to 1 (0 in a classic one).
pub fn intensity_now() -> f32 {
    f32::from_bits(INTENSITY_NOW.load(Ordering::Relaxed))
}

/// This frame's rain heaviness on a modern frame, 0 to 1 (0 in a classic one).
pub fn heavy_now() -> f32 {
    f32::from_bits(HEAVY_NOW.load(Ordering::Relaxed))
}

/// How heavy the rain is, 0 to 1, from the effect's ramped density (`(grade − 0.25)·4/3`): eased
/// in from [`HEAVY_FROM`] and whole at [`HEAVY_DENSITY`], across the server's medium and heavy rain
/// (grade 0.70, vmangos `GetWeatherState`). The storm light weight `min(1, 4·density)` is whole at a
/// quarter, so it cannot tell a shower from a downpour.
pub fn heavy_rain(weather: &WeatherState) -> f32 {
    heaviness(weather.effect_kind, weather.effect_density)
}

/// The effect density where the look is whole, grade 0.8, and where it starts easing in, grade 0.5
/// (fork: from 0.625 to 0.70, asked in the viewer: the rain light swapped in too late and too fast;
/// the server's heavy rain, grade 0.70, now takes about three quarters of it).
const HEAVY_DENSITY: f32 = 0.55 * 4.0 / 3.0;
const HEAVY_FROM: f32 = 1.0 / 3.0;

fn heaviness(kind: WeatherKind, density: f32) -> f32 {
    if kind != WeatherKind::Rain {
        return 0.0;
    }
    let t = ((density - HEAVY_FROM) / (HEAVY_DENSITY - HEAVY_FROM)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// What heavy rain leaves of each band's luma.
pub const HEAVY_DARKEN: f32 = 0.4;
/// The share of each band heavy rain turns to [`HEAVY_BLUE_HUE`] at the same luma.
pub const HEAVY_BLUE: f32 = 0.8;
/// The cold hue heavy rain casts (gamma): a deep teal-blue, green well over red (asked in the
/// viewer: a thunderstorm should sink the zone into it).
pub const HEAVY_BLUE_HUE: [f32; 3] = [0.18, 0.62, 1.0];

/// The humid veil heavy rain adds to the air's shaft haze (1/yd), on top of the storm's
/// (`media::STORM_HAZE`): with Elwynn's look the air keeps half the scene at 200 yd.
pub const HEAVY_HAZE: f32 = 0.0008;
/// How much more haze the lamp halos' air takes under heavy rain, as a share of its own: the
/// lamps bloom into the wet air (`volumetrics`' `halo_air`). By day only: the night air already
/// thickens their glow.
pub const HEAVY_HALO_GAIN: f32 = 1.5;

/// One band under rain of heaviness `h`: darkened, then part turned to the cold hue.
fn heavy_band(c: [f32; 3], h: f32) -> [f32; 3] {
    cold_band(c, HEAVY_DARKEN, HEAVY_BLUE, HEAVY_BLUE_HUE, h)
}

/// One band at weight `h` of a cold grade: `darken` of its luma left, then `blue` of it turned to
/// the cold `hue` at the same luma.
fn cold_band(c: [f32; 3], darken: f32, blue: f32, hue: [f32; 3], h: f32) -> [f32; 3] {
    let dark = c.map(|v| v * (1.0 + (darken - 1.0) * h));
    let hue = hue.map(|v| v * luma(dark) / luma(hue));
    let k = blue * h;
    [0, 1, 2].map(|i| (dark[i] + (hue[i] - dark[i]) * k).clamp(0.0, 1.0))
}

/// fork: **a bright night goes dark and cold**, as heavy rain does, a little more so. A zone whose
/// night fog is authored grey and light (Darkshore's 0.15 luma against Elwynn's blue 0.05) read
/// as a pale dusk at 03:00. Its night takes the heavy-rain grade at [`NIGHT_DARKEN`] and
/// [`NIGHT_BLUE`] toward [`NIGHT_BLUE_HUE`], weighted by the night fraction and by how far the
/// resolved fog sits between [`NIGHT_FOG_DARK`] (none: Elwynn's night as it is) and
/// [`NIGHT_FOG_BRIGHT`] (whole).
pub const NIGHT_DARKEN: f32 = 0.45;
pub const NIGHT_BLUE: f32 = 0.7;
pub const NIGHT_BLUE_HUE: [f32; 3] = [0.40, 0.55, 0.90];
pub const NIGHT_FOG_DARK: f32 = 0.06;
pub const NIGHT_FOG_BRIGHT: f32 = 0.14;

/// The bright-night grade's weight: the night fraction times the fog's place between
/// [`NIGHT_FOG_DARK`] and [`NIGHT_FOG_BRIGHT`].
fn bright_night(night: f32, fog_color: [f32; 3]) -> f32 {
    let t =
        ((luma(fog_color) - NIGHT_FOG_DARK) / (NIGHT_FOG_BRIGHT - NIGHT_FOG_DARK)).clamp(0.0, 1.0);
    night.clamp(0.0, 1.0) * t * t * (3.0 - 2.0 * t)
}

/// A strike's flash pulses: one in two strikes flashes once, the rest two or three times, each
/// later pulse this many seconds after the one before at this share of the first's peak.
const PULSE_GAP_S: (f32, f32) = (0.08, 0.25);
const PULSE_PEAK: (f32, f32) = (0.5, 0.9);
/// Each flash pulse's decay time constant (s).
const STRIKE_DECAY_S: f32 = 0.12;
/// The share of a whole flash the scene keeps while the bolt is up ([`Strike::bolt`]).
const FLASH_SUSTAIN: f32 = 0.6;
/// The bolt holds whole this long past the last pulse, then fades out over the next span.
const BOLT_HOLD_S: f32 = 0.8;
const BOLT_FADE_S: f32 = 1.5;
/// Seconds between strikes at full heaviness, drawn uniformly; lighter rain waits longer, and none
/// strikes short of [`STRIKE_HEAVINESS`]. `$WOW_LIGHTNING_EVERY=<s>` fixes the gap, for a capture.
const STRIKE_GAP_S: (f32, f32) = (3.0, 10.0);
/// Strikes need the server's heavy rain: [`heavy_rain`] whole, give or take its ramp's last step.
const STRIKE_HEAVINESS: f32 = 0.95;
/// The strike clock's pace at the edge of heavy rain, as a share of a full storm's.
const STRIKE_RATE_FLOOR: f32 = 0.15;

/// How far past heavy the rain is, 0 to 1: from the server's heavy rain (grade 0.70) to a full
/// storm (grade 1), which paces the strikes.
fn storminess(weather: &WeatherState) -> f32 {
    if weather.effect_kind != WeatherKind::Rain {
        return 0.0;
    }
    ((weather.effect_density - HEAVY_DENSITY) / (1.0 - HEAVY_DENSITY)).clamp(0.0, 1.0)
}
/// What a whole flash adds to the ambient, the key, the sky and the fog (gamma): a cold white
/// that lifts the storm without lighting a day; the key carries most. Under heavy rain it takes the
/// rain's own cold hue at the same luma (`apply_heavy_rain`): a near-neutral white over the teal
/// storm turned the scene grey for as long as a bolt was up, and back between strikes.
const FLASH_AMBIENT: [f32; 3] = [0.12, 0.14, 0.19];
const FLASH_KEY: [f32; 3] = [0.45, 0.50, 0.60];
const FLASH_SKY: [f32; 3] = [0.10, 0.12, 0.16];
const FLASH_FOG: [f32; 3] = [0.06, 0.07, 0.10];

/// One strike: its age and its flash pulses, (seconds in, peak).
struct Strike {
    age: f32,
    pulses: Vec<(f32, f32)>,
}

impl Strike {
    /// When the last pulse fires.
    fn last_pulse(&self) -> f32 {
        self.pulses.last().map_or(0.0, |p| p.0)
    }

    /// The whole strike's length: the bolt's hold and fade after its last pulse.
    fn len(&self) -> f32 {
        self.last_pulse() + BOLT_HOLD_S + BOLT_FADE_S
    }

    /// The flash brightness now, at most 1: the pulses over a glow that holds and fades with the
    /// bolt, so the scene stays lit while the bolt is up.
    fn flash(&self) -> f32 {
        let t = self.age;
        let pulses = self
            .pulses
            .iter()
            .filter(|(at, _)| t >= *at)
            .map(|(at, peak)| peak * (-(t - at) / STRIKE_DECAY_S).exp())
            .sum::<f32>();
        pulses.max(FLASH_SUSTAIN * self.bolt()).min(1.0)
    }

    /// The bolt's opacity now: whole through the pulses and the hold, then an eased fade.
    fn bolt(&self) -> f32 {
        let f = ((self.age - self.last_pulse() - BOLT_HOLD_S) / BOLT_FADE_S).clamp(0.0, 1.0);
        1.0 - f * f * (3.0 - 2.0 * f)
    }
}

/// Where a bolt stands: this share of the farclip out (yd, clamped), within ±this of the view's
/// heading, its top this share of its distance above the eye and its foot below the horizon, or
/// for [`BOLT_AIR`] of strikes in the air ([`BOLT_AIR_FOOT`]).
const BOLT_REACH: f32 = 0.45;
const BOLT_DIST_YD: (f32, f32) = (120.0, 350.0);
const BOLT_SPREAD_DEG: f32 = 25.0;
const BOLT_TOP: f32 = 2.0;
const BOLT_FOOT_YD: f32 = 40.0;
/// The chance a strike ends in the air, and where its foot stops then: this share of the top's
/// height above the eye, from the lowest to the highest — about 6 to 22 degrees over the horizon,
/// so the end shows in a level view (the top stands at ~63 degrees, off the top of the frame).
const BOLT_AIR: f32 = 0.7;
const BOLT_AIR_FOOT: (f32, f32) = (0.05, 0.2);
/// The bolt's sideways jag, a share of each split segment, over this many halvings: a few long
/// straight runs between sharp kinks.
const BOLT_JAG: f32 = 0.14;
const BOLT_SPLITS: u32 = 5;
/// The chance a strike forks once.
const BOLT_FORK: f32 = 0.25;
/// Half widths (yd) of the widest strike's core and its glow, and the glow's opacity.
const BOLT_CORE_YD: f32 = 1.5;
/// Each strike's share of [`BOLT_CORE_YD`], from the thinnest to the widest, drawn as
/// `lo + (hi − lo)·r²`: most strikes are thin, a few as heavy as the widest.
const BOLT_WIDTH: (f32, f32) = (0.3, 1.0);
const BOLT_GLOW_YD: f32 = 8.0;
const BOLT_GLOW_ALPHA: f32 = 0.35;
/// What is left of a line's brightness at its far end: a bolt dims toward the ground, and one that
/// ends in the air ([`BOLT_AIR`]) fades out to nothing.
const BOLT_FOOT_ALPHA: f32 = 0.6;
const BOLT_AIR_FOOT_ALPHA: f32 = 0.0;
const BOLT_CORE: [f32; 3] = [0.88, 0.92, 1.0];
const BOLT_GLOW: [f32; 3] = [0.55, 0.65, 1.0];

/// One strike's bolt: its jagged lines, each with its core half width and what is left of its
/// brightness at its far end.
struct Bolt {
    lines: Vec<(Vec<Vec3>, f32, f32)>,
}

/// The lightning clock: seconds to the next strike, and the live strike's age and bolt.
#[derive(Resource)]
struct Lightning {
    rng: u32,
    wait: f32,
    every: Option<f32>,
    strike: Option<Strike>,
    bolt: Option<Bolt>,
}

impl Default for Lightning {
    fn default() -> Self {
        let every = crate::launch_env::var("WOW_LIGHTNING_EVERY")
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|s| *s > 0.0);
        Self {
            rng: 0x2545_f491,
            wait: every.unwrap_or(STRIKE_GAP_S.0),
            every,
            strike: None,
            bolt: None,
        }
    }
}

impl Lightning {
    fn rand01(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng >> 8) as f32 / (1u32 << 24) as f32
    }

    /// `rand01` centred, in [−1, 1).
    fn rand_signed(&mut self) -> f32 {
        2.0 * self.rand01() - 1.0
    }

    /// Advances by `dt` under heaviness `h` and storminess `storm` ([`storminess`]); true when a
    /// strike begins this frame.
    fn step(&mut self, dt: f32, h: f32, storm: f32) -> bool {
        if let Some(strike) = self.strike.as_mut() {
            strike.age += dt;
            if strike.age >= strike.len() {
                self.strike = None;
                self.bolt = None;
            }
        }
        if h < STRIKE_HEAVINESS {
            return false;
        }
        // The clock runs at [`STRIKE_RATE_FLOOR`] at the edge of heavy rain and whole at a full
        // storm, so a downpour at grade 0.75 strikes a few times a minute and grade 1 every few
        // seconds.
        self.wait -= dt * (STRIKE_RATE_FLOOR + (1.0 - STRIKE_RATE_FLOOR) * storm);
        if self.wait > 0.0 || self.strike.is_some() {
            return false;
        }
        let roll = self.rand01();
        let count = if roll < 0.5 {
            1
        } else if roll < 0.8 {
            2
        } else {
            3
        };
        let mut pulses = vec![(0.0, 1.0)];
        for _ in 1..count {
            let at = pulses.last().map_or(0.0, |p: &(f32, f32)| p.0)
                + PULSE_GAP_S.0
                + (PULSE_GAP_S.1 - PULSE_GAP_S.0) * self.rand01();
            pulses.push((
                at,
                PULSE_PEAK.0 + (PULSE_PEAK.1 - PULSE_PEAK.0) * self.rand01(),
            ));
        }
        self.strike = Some(Strike { age: 0.0, pulses });
        self.wait = match self.every {
            Some(every) => every,
            None => STRIKE_GAP_S.0 + (STRIKE_GAP_S.1 - STRIKE_GAP_S.0) * self.rand01(),
        };
        true
    }

    /// The live strike's flash brightness.
    fn flash(&self) -> f32 {
        self.strike.as_ref().map_or(0.0, Strike::flash)
    }

    /// The live bolt's opacity.
    fn bolt_alpha(&self) -> f32 {
        self.strike.as_ref().map_or(0.0, Strike::bolt)
    }

    /// A line from `from` to `to`, halved [`BOLT_SPLITS`] times with each midpoint pushed sideways
    /// across the eye line: the chain-lightning centre line (`0x7af6d0`) as a fractal.
    fn jag(&mut self, from: Vec3, to: Vec3, eye: Vec3) -> Vec<Vec3> {
        let mut line = vec![from, to];
        for _ in 0..BOLT_SPLITS {
            let mut next = Vec::with_capacity(line.len() * 2);
            for w in line.windows(2) {
                let (a, b) = (w[0], w[1]);
                let mid = (a + b) * 0.5;
                let side = (b - a).cross(eye - mid).normalize_or_zero();
                next.push(a);
                next.push(mid + side * (b - a).length() * BOLT_JAG * self.rand_signed());
            }
            next.push(to);
            line = next;
        }
        line
    }

    /// A bolt for an eye at `eye` looking along `forward`, inside `farclip`.
    fn make_bolt(&mut self, eye: Vec3, forward: Vec3, farclip: f32) -> Bolt {
        let flat = Vec3::new(forward.x, 0.0, forward.z).normalize_or(Vec3::NEG_Z);
        let dir = Quat::from_rotation_y(BOLT_SPREAD_DEG.to_radians() * self.rand_signed()) * flat;
        let dist = (BOLT_REACH * farclip).clamp(BOLT_DIST_YD.0, BOLT_DIST_YD.1);
        let across = dir.cross(Vec3::Y);
        let top = eye + dir * dist + Vec3::Y * (BOLT_TOP * dist);
        let (drop, fade) = if self.rand01() < BOLT_AIR {
            let share = BOLT_AIR_FOOT.0 + (BOLT_AIR_FOOT.1 - BOLT_AIR_FOOT.0) * self.rand01();
            (-BOLT_TOP * dist * share, BOLT_AIR_FOOT_ALPHA)
        } else {
            (BOLT_FOOT_YD, BOLT_FOOT_ALPHA)
        };
        let foot = eye + dir * dist + across * (0.15 * dist * self.rand_signed()) - Vec3::Y * drop;
        let main = self.jag(top, foot, eye);
        let r = self.rand01();
        let core = BOLT_CORE_YD * (BOLT_WIDTH.0 + (BOLT_WIDTH.1 - BOLT_WIDTH.0) * r * r);
        let mut lines = Vec::new();
        // Some strikes fork once, from the upper half of the main line.
        if self.rand01() < BOLT_FORK {
            let at = main[(main.len() as f32 * (0.15 + 0.35 * self.rand01())) as usize];
            let end = at + across * (0.2 * dist * self.rand_signed()) - Vec3::Y * (0.3 * dist);
            lines.push((self.jag(at, end, eye), 0.5 * core, fade));
        }
        lines.insert(0, (main, core, fade));
        Bolt { lines }
    }
}

/// A bright night's grade ([`bright_night`]), then heavy rain's and its lightning, over the
/// resolved light, after the sun arc has keyed it, so noon goes dark and cold as much as night: a
/// flash lifts the ambient, key, sky and fog for its moment. The key keeps its direction, so the
/// shadows hold still under the flash. A modern frame only.
fn apply_heavy_rain(
    weather: Option<Res<WeatherState>>,
    time: Res<Time>,
    view: Res<crate::view::ViewDistance>,
    cam: Query<&GlobalTransform, With<crate::view::WorldCamera>>,
    mut lightning: ResMut<Lightning>,
    mut lighting: ResMut<WowLighting>,
) {
    if !super::frame_is_modern() {
        INTENSITY_NOW.store(0, Ordering::Relaxed);
        HEAVY_NOW.store(0, Ordering::Relaxed);
        return;
    }
    let n = bright_night(lighting.sidn_night, lighting.fog_color);
    if n > 0.0 {
        let l = &mut *lighting;
        for c in [&mut l.ambient, &mut l.diffuse, &mut l.fog_color]
            .into_iter()
            .chain(l.sky.iter_mut())
            .chain(l.cloud_colors.iter_mut())
        {
            *c = cold_band(*c, NIGHT_DARKEN, NIGHT_BLUE, NIGHT_BLUE_HUE, n);
        }
    }
    let h = weather.as_deref().map_or(0.0, heavy_rain);
    HEAVY_NOW.store(h.to_bits(), Ordering::Relaxed);
    let intensity = weather
        .as_deref()
        .filter(|w| w.effect_kind != WeatherKind::Fine)
        .map_or(0.0, |w| w.effect_density.clamp(0.0, 1.0));
    INTENSITY_NOW.store(intensity.to_bits(), Ordering::Relaxed);
    let storm = weather.as_deref().map_or(0.0, storminess);
    if lightning.step(time.delta_secs(), h, storm) {
        if let Ok(eye) = cam.single() {
            let bolt = lightning.make_bolt(eye.translation(), *eye.forward(), view.farclip);
            lightning.bolt = Some(bolt);
        }
    }
    let flash = lightning.flash();
    if h <= 0.0 && flash <= 0.0 {
        return;
    }
    let add = |c: [f32; 3], f: [f32; 3]| {
        let f = cold_band(f, 1.0, HEAVY_BLUE, HEAVY_BLUE_HUE, h);
        [0, 1, 2].map(|i| (c[i] + f[i] * flash).min(1.0))
    };
    let l = &mut *lighting;
    l.ambient = add(heavy_band(l.ambient, h), FLASH_AMBIENT);
    l.diffuse = add(heavy_band(l.diffuse, h), FLASH_KEY);
    l.fog_color = add(heavy_band(l.fog_color, h), FLASH_FOG);
    for c in l.sky.iter_mut().chain(l.cloud_colors.iter_mut()) {
        *c = add(heavy_band(*c, h), FLASH_SKY);
    }
}

/// The live bolt into the effect lane: a camera-facing additive ribbon along each line, a soft
/// glow around an opaque core, held and then faded ([`Strike::bolt`]). Hidden indoors and under water.
fn push_bolt(
    lightning: Res<Lightning>,
    indoors: Res<crate::weather::WeatherIndoors>,
    underwater: Option<Res<crate::liquid::Underwater>>,
    cam: Query<(Entity, &GlobalTransform), With<crate::view::WorldCamera>>,
    mut draw: crate::particles::buffer::WorldEffectDraw,
) {
    let alpha = lightning.bolt_alpha();
    let Some(bolt) = lightning.bolt.as_ref().filter(|_| alpha > 0.0) else {
        return;
    };
    if indoors.0 || underwater.is_some_and(|u| u.0.any()) {
        return;
    }
    let Ok((cam, eye)) = cam.single() else {
        return;
    };
    let eye = eye.translation();
    let main = &bolt.lines[0].0;
    let mut batch = draw
        .batch(cam, Handle::<Image>::default().id())
        .additive()
        .anchored(main[main.len() / 2]);
    for (line, core, fade) in &bolt.lines {
        let glow = BOLT_GLOW_YD * core / BOLT_CORE_YD;
        ribbon(batch.verts_mut(), line, eye, *core, BOLT_CORE, alpha, *fade);
        ribbon(
            batch.verts_mut(),
            line,
            eye,
            glow,
            BOLT_GLOW,
            BOLT_GLOW_ALPHA * alpha,
            *fade,
        );
    }
    batch.quads();
}

/// A camera-facing strip along `line`, `half` yd each side, opaque on the line and clear at its
/// edges and fading to `fade` of `alpha` along it, as quads: the two halves of each span.
fn ribbon(
    out: &mut Vec<crate::particles::buffer::EffectVertex>,
    line: &[Vec3],
    eye: Vec3,
    half: f32,
    rgb: [f32; 3],
    alpha: f32,
    fade: f32,
) {
    let side = |i: usize| {
        let along = line[(i + 1).min(line.len() - 1)] - line[i.saturating_sub(1)];
        along.cross(eye - line[i]).normalize_or_zero() * half
    };
    let v = |p: Vec3, a: f32| crate::particles::buffer::EffectVertex {
        pos: p.to_array(),
        uv: [0.5, 0.5],
        color: [rgb[0], rgb[1], rgb[2], a],
    };
    let last = line.len().saturating_sub(1).max(1) as f32;
    let at = |i: usize| alpha * (1.0 + (fade - 1.0) * i as f32 / last);
    for i in 0..line.len().saturating_sub(1) {
        let (a, b) = (line[i], line[i + 1]);
        let (sa, sb) = (side(i), side(i + 1));
        for s in [1.0, -1.0] {
            out.extend([
                v(a + sa * s, 0.0),
                v(a, at(i)),
                v(b, at(i + 1)),
                v(b + sb * s, 0.0),
            ]);
        }
    }
}

/// Heavy rain's rewrite of the resolved light; every reader of it runs after.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct HeavyRainSet;

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<Lightning>()
        .configure_sets(Update, LightingConsumeSet.after(HeavyRainSet))
        .add_systems(
            Update,
            apply_heavy_rain
                .in_set(HeavyRainSet)
                .after(super::sun_arc::SunArcSet),
        )
        .add_systems(
            PostUpdate,
            push_bolt.after(crate::particles::buffer::begin_effect_frame),
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(c: [u8; 3]) -> [f32; 3] {
        c.map(|v| f32::from(v) / 255.0)
    }

    /// Elwynn's midnight rows: the storm's grey would brighten the night; the modern band stays
    /// under the clear luma and keeps the blue.
    #[test]
    fn a_night_storm_darkens_and_keeps_its_blue() {
        for (clear, stormy) in [
            (bytes([29, 60, 84]), bytes([78, 78, 95])),  // ambient
            (bytes([0, 14, 33]), bytes([82, 84, 82])),   // fog
            (bytes([49, 86, 123]), bytes([82, 83, 82])), // horizon
        ] {
            let band = storm_band(clear, stormy);
            assert!(luma(band) <= STORM_DARKEN * luma(clear) + 1e-5, "{band:?}");
            assert!(
                band[2] > band[1] && band[1] > band[0],
                "{band:?} keeps the blue"
            );
        }
        assert_eq!(
            storm_band([0.0; 3], bytes([111, 111, 111])),
            [0.0; 3],
            "a black zenith"
        );
    }

    /// Elwynn's noon ambient: the storm row is darker than the clear one, so its luma is taken.
    #[test]
    fn a_day_storm_takes_the_storm_rows_luma() {
        let (clear, stormy) = (bytes([104, 130, 154]), bytes([78, 78, 95]));
        let band = storm_band(clear, stormy);
        assert!((luma(band) - luma(stormy)).abs() < 2.0 / 255.0, "{band:?}");
        assert!(band[2] > band[0], "{band:?}");
    }

    /// A modern storm keeps the clear day's fog distances; the reference's pulls Elwynn's end from
    /// 500 to 278 yd and its start behind the eye.
    #[test]
    fn a_modern_storm_keeps_the_clear_distances() {
        let clear = Atmosphere::DEFAULT;
        let stormy = Atmosphere {
            fog_end: 278.0,
            fog_start_frac: -0.5,
            ..Atmosphere::DEFAULT
        };
        let a = storm_atmosphere(&clear, &stormy, 1.0);
        assert_eq!(
            (a.fog_end, a.fog_start_frac),
            (clear.fog_end, clear.fog_start_frac)
        );
    }

    /// A shower is not heavy; a downpour is, and only rain is.
    #[test]
    fn only_a_downpour_is_heavy() {
        let grade = |g: f32| (g - 0.25) * 4.0 / 3.0;
        assert_eq!(heaviness(WeatherKind::Rain, grade(0.5)), 0.0, "medium rain");
        assert_eq!(heaviness(WeatherKind::Rain, grade(0.8)), 1.0);
        assert!((heaviness(WeatherKind::Rain, grade(0.65)) - 0.5).abs() < 1e-5);
        let server = heaviness(WeatherKind::Rain, grade(0.70));
        assert!(
            server > 0.7 && server < 0.8,
            "the server's heavy rain: {server}"
        );
        assert_eq!(heaviness(WeatherKind::Snow, 1.0), 0.0);
    }

    /// Elwynn's noon ambient under a downpour: darker, bluer than it is green, and greener than red.
    #[test]
    fn a_downpour_darkens_and_cools_noon() {
        let noon = bytes([104, 130, 154]);
        let wet = heavy_band(noon, 1.0);
        assert!(luma(wet) < 0.75 * luma(noon), "{wet:?}");
        let cool = |c: [f32; 3]| c[2] / c[0];
        assert!(cool(wet) > cool(noon), "{wet:?}");
        assert!(wet[2] > wet[1] && wet[1] > wet[0], "teal-blue: {wet:?}");
        assert_eq!(heavy_band(noon, 0.0), noon);
    }

    /// A strike flashes whole and decays; its bolt holds, then fades out; a shower never strikes.
    #[test]
    fn a_strike_flashes_and_its_bolt_lingers_then_fades() {
        let at = |age: f32| Strike {
            age,
            pulses: vec![(0.0, 1.0)],
        };
        assert_eq!(at(0.0).flash(), 1.0);
        assert!(
            (at(0.6).flash() - FLASH_SUSTAIN).abs() < 1e-3,
            "lit while the bolt holds"
        );
        assert_eq!(at(0.8).bolt(), 1.0, "held past the flash");
        assert!((at(1.55).bolt() - 0.5).abs() < 1e-5, "half-way out");
        assert_eq!(at(at(0.0).len()).bolt(), 0.0);
        let mut shower = Lightning::default();
        assert!((0..6000).all(|_| !shower.step(1.0 / 60.0, 0.4, 0.0)));
        let mut downpour = Lightning::default();
        let strikes = (0..60 * 60)
            .filter(|_| downpour.step(1.0 / 60.0, 1.0, 1.0))
            .count();
        assert!(strikes > 0, "a minute of downpour strikes");
    }

    /// Strikes pace with how far past heavy the rain is: rare at its edge, every few seconds in a
    /// full storm.
    #[test]
    fn a_full_storm_strikes_far_more_often_than_the_edge_of_heavy_rain() {
        let strikes = |storm: f32| {
            let mut l = Lightning {
                every: None,
                ..Lightning::default()
            };
            (0..60 * 600)
                .filter(|_| l.step(1.0 / 60.0, 1.0, storm))
                .count()
        };
        let (edge, full) = (strikes(0.0), strikes(1.0));
        assert!(edge > 0, "heavy rain still strikes");
        assert!(
            full > 3 * edge,
            "{full} strikes in a full storm vs {edge} at its edge"
        );
    }

    /// A bolt stands in front of the view and climbs well over the horizon, so it runs off the top
    /// of the frame rather than stopping in the sky; most strikes end in the air, the rest below
    /// the horizon.
    #[test]
    fn a_bolt_stands_ahead_and_climbs_steeply() {
        let mut l = Lightning::default();
        let eye = Vec3::new(10.0, 50.0, -20.0);
        let (mut air, mut ground) = (0, 0);
        for _ in 0..40 {
            let bolt = l.make_bolt(eye, Vec3::NEG_Z, 777.0);
            let foot = *bolt.lines[0].0.last().unwrap() - eye;
            if foot.y > 0.0 {
                air += 1;
            } else {
                ground += 1;
            }
            let top = bolt.lines[0].0[0] - eye;
            let flat = Vec3::new(top.x, 0.0, top.z);
            assert!(top.y > 1.5 * flat.length(), "top {top}");
            let off_axis = flat.normalize().angle_between(Vec3::NEG_Z).to_degrees();
            assert!(
                off_axis <= BOLT_SPREAD_DEG + 1e-3,
                "{off_axis} deg off the view"
            );
        }
        assert!(
            ground > 0 && air > ground,
            "{air} in the air, {ground} to the ground"
        );
    }

    /// A bright, saturated authored fog (a sandstorm's) is held to the haze's caps, hue kept; a
    /// grey storm fog passes through.
    #[test]
    fn a_puff_is_never_brighter_or_more_saturated_than_the_haze() {
        let sand = puff_color([0.95, 0.80, 0.45]);
        let (hi, lo) = (sand[0], sand[2]);
        assert!(hi <= media::MAX_FOG_LEVEL + 1e-5);
        assert!((hi - lo) / hi <= media::MAX_FOG_SATURATION + 1e-4);
        assert!(sand[0] >= sand[1] && sand[1] >= sand[2], "hue kept");
        assert_eq!(puff_color([0.4, 0.42, 0.45]), [0.4, 0.42, 0.45]);
    }
}
