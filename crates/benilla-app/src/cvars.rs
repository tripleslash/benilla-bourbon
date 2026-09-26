//! The CVar registry, host side: the reference's engine-side table (`ConsoleVar.cpp`, 0xc4-byte
//! records of name, value, default, latch slot and the change callback handed to `CVar::Register`,
//! `0x63db90`). [`Cvars`] holds every registered row's live value, survives a VM replacement and is
//! what `config.toml` is composed from; the script VM keeps a mirror for Lua's synchronous
//! `GetCVar`/`SetCVar` ([`benilla_ui::script::UiScript::seed_cvars`]) whose writes queue back here.
//!
//! - [`REGISTERED`] holds only vars something reads, a host knob or a Lua consumer. A row's default
//!   is the reference's, and [`Registered::reference`] says where it stands against it.
//! - The change callback is a Bevy observer on [`CvarChanged`], beside the knob it writes; the
//!   registry applies nothing itself.
//! - A latched row's write is staged in [`Row::pending`] until [`Cvars::commit_latched`], the
//!   reference's `0x639ec0` inside `RestartGx`. A stage never committed is dropped at exit, as the
//!   reference's `SaveConfig` writes only the applied value (`rec+0x20`).
//! - Boot folds `benilla-config/config.toml` in and fires the observers before anything after
//!   [`CvarLoad`]; each frame drains the VM's writes in and pushes the registry's out; a dirty
//!   registry saves after one quiet second or at exit, writing only values off their default and
//!   keeping unknown keys verbatim.
//!
//! Env overrides (`WOW_UI_SCALE`, `WOW_FARCLIP`, …) win for the session and never touch the file
//! ([`Cvars::own_for_session`]).

use std::collections::{BTreeMap, HashMap, HashSet};

use bevy::platform::time::Instant;

use bevy::prelude::*;

use crate::ui_script::VmMemo;
use benilla_ui::script::{SeededCvar, UiScript};
use benilla_world::liquid::WaterDetail;
use benilla_world::modern::LightingSettings;

/// One host-backed CVar: its name, benilla's default, and where that default stands against the
/// reference's ([`Reference`]).
pub(crate) struct Registered {
    /// The registered name, in the reference's own spelling.
    pub(crate) name: &'static str,
    /// What a fresh `benilla-config` runs at, and what `GetCVar` answers until the player moves it.
    pub(crate) default: &'static str,
    /// Where `default` stands against the reference's. Read only by the tests: it records the
    /// reference, never a value this client acts on.
    #[allow(dead_code)]
    pub(crate) reference: Reference,
    /// Registered with flag bit1 (`rec+0x1c & 0x2`, `flags` 2 or 3 at the register site): a write
    /// is staged in [`Row::pending`] until [`Cvars::commit_latched`].
    pub(crate) latched: bool,
}

impl Registered {
    /// Mark the row latched.
    pub(crate) const fn latched(self) -> Self {
        Self {
            latched: true,
            ..self
        }
    }
}

/// benilla's default against the reference's: the string the reference's `CVar::Register`
/// (`0x63db90`) passes for the name, or, for a setting 1.12 keeps in FrameXML, the value
/// `UIOptionsFrame.lua` boots it at. Not a `Config.wtf` line (`SaveConfig 0x63d980` writes only
/// values off their default), and not always what a fresh install runs at: `hwDetect` rewrites
/// sixteen video CVars from `VideoHardware.dbc` before the first frame ([`Reference::Overridden`]).
/// Each row's register site or FrameXML line is cited above it.
#[allow(dead_code)] // read only by the tests
pub(crate) enum Reference {
    /// The reference registers this default and benilla ships it; the test compares the two.
    Same(&'static str),
    /// The reference registers `registered`, but its own boot code overwrites it before the first
    /// frame; `default` is where that lands, and `why` is the override.
    Overridden {
        registered: &'static str,
        why: &'static str,
    },
    /// The reference ships `value` and benilla ships another; `why` is the reason.
    Deviates {
        value: &'static str,
        why: &'static str,
    },
    /// No reference setting to match: benilla's own knob, or a later-era name for something 1.12
    /// never made settable. `why` says which, and what the reference does instead.
    Ours(&'static str),
}

/// A row whose default is the reference's own registered string.
const fn same(name: &'static str, default: &'static str) -> Registered {
    Registered {
        name,
        default,
        reference: Reference::Same(default),
        latched: false,
    }
}

/// A row whose default follows the reference's boot-time override of its registered string.
const fn overridden(
    name: &'static str,
    default: &'static str,
    registered: &'static str,
    why: &'static str,
) -> Registered {
    Registered {
        name,
        default,
        reference: Reference::Overridden { registered, why },
        latched: false,
    }
}

/// A row that ships something other than the reference's `value`, for `why`.
const fn deviates(
    name: &'static str,
    default: &'static str,
    value: &'static str,
    why: &'static str,
) -> Registered {
    Registered {
        name,
        default,
        reference: Reference::Deviates { value, why },
        latched: false,
    }
}

/// A row the reference has no counterpart for.
const fn ours(name: &'static str, default: &'static str, why: &'static str) -> Registered {
    Registered {
        name,
        default,
        reference: Reference::Ours(why),
        latched: false,
    }
}

/// The table as the script VM's registrar wants it: `(name, default)` pairs in table order.
pub(crate) fn registered_pairs() -> impl Iterator<Item = (&'static str, &'static str)> {
    REGISTERED.iter().map(|r| (r.name, r.default))
}

/// The host-backed CVars, one row per knob that has a reader.
///
/// Not registered for want of a reader, though pfUI's `hdgraphic` writes them: `lodDist`
/// (`0x688524`, "100.0", read at `0x6afb1d` for the doodad LOD swap), `footstepBias` (`0x6888b4`,
/// "0.125", read at `0x68fcb6`), `mapObjLightLOD` (`0x6886ec`, "0") and `SkyCloudLOD` (`0x6d1d33`,
/// "0"). `DistCull` (`0x688570`) and `texLodBias` (`0x6885e2`, whose sink `0x672640` is `ret 4`)
/// have no reader in the reference either. `maxLOD` is no 1.12 CVar.
pub(crate) const REGISTERED: &[Registered] = &[
    // `realmName` (`0x83f2d0`): registered `""` (`0x882748`), help "Last realm connected to"
    // (`0x85d684`); the client builds its SavedVariables path from it (`0x5ab7d0`). Written from
    // the session's realm when addons load. `Ace/AceState.lua:27` trims it at
    // PLAYER_ENTERING_WORLD, so a nil breaks every Ace addon.
    same("realmName", ""),
    // The logon server address (register site `0x5ab6a6`), a string row judged by
    // `realmlist::on_cvar`.
    deviates(
        crate::realmlist::CVAR_REALMLIST,
        crate::realmlist::DEFAULT_REALMLIST,
        "us.logon.worldofwarcraft.com:3724",
        "that host has not resolved since 2019, so shipping it makes every first launch a \
         DNS failure; benilla dials the machine it is running on",
    ),
    // `autoClearAFK` (`0x5e24d4`, "1" `0x82e748`, handle `[0xc4d68c]` set at `0x5e24ef`, read at
    // `0x5eb84b`) gates five implicit AFK clears: any chat send but type `0x14`, Jump,
    // forward/back, strafe and turn (`0x513d36`/`0x514e23`/`0x514f0b`/`0x514fca`). Off, the clear
    // does nothing at all: no echo, no mirror write, no packet.
    same("autoClearAFK", "1"),
    same("MasterVolume", "1"),
    same("SoundVolume", "1"),
    same("MusicVolume", "0.4"),
    same("AmbienceVolume", "0.6"),
    // Registered "1" at `0x45737a`/`0x45739b`/`0x460a9d`. `MasterSoundEffects` is the Enable All
    // Sound box (`SoundOptionsFrame.lua:6`), which pauses the whole sound engine, not an SFX
    // toggle.
    same("MasterSoundEffects", "1"),
    same("EnableMusic", "1"),
    same("EnableAmbience", "1"),
    // The race/sex refusal voice lines (`0x457877`), `SoundOptionsFrame.lua:3`; the master enable
    // greys it.
    same("EnableErrorSpeech", "1"),
    // Not a 1.12 CVar or checkbox; the later-era name. The reference mutes on losing focus, music
    // included (`WM_ACTIVATE` to `0x7a4860`'s `FSOUND_SetMute(-3, active ? 0 : 1)`), so "0" is its
    // behaviour. The knob is `SoundConfig::background_sound`.
    same("Sound_EnableSoundWhenGameIsInBG", "0"),
    // Registered "1" (`0x4573be`); `SoundConfig::reverb` carries the evidence for shipping "0".
    deviates(
        "SoundReverb",
        "0",
        "1",
        "the reference's reverb is EAX-over-hardware, and that hardware has not existed since \
         Vista, so \"1\" would ship audio the real client has never actually produced on any \
         machine a player runs today",
    ),
    // FMOD 3's mix-ahead buffer in ms (`0x4571ca`, flags 2: read once at sound init); here it sizes
    // the render thread's ring ahead of the IO callback (`sound::output`). `0x457520` registers
    // "50" or "100" by a host probe (`0x835e10`/`0x835e0c`), which one on a current machine
    // untraced; ours is the larger, since the depth has to hide a whole stalled IO cycle.
    same("SoundBufferSize", "100").latched(),
    // benilla's own, not a 1.12 CVar. Deviation: the reference clips at full scale and has no
    // headroom mechanism (its SFX-bus duck `0x457960` is a sidechain armed only by server-pushed
    // voice lines); benilla limits because every SFX is mastered to full scale and overlapping
    // kits clip.
    ours(
        "SoundOutputLimiter",
        "1",
        "benilla's own — the reference sums at full scale and clips; every WoW SFX is mastered \
         to full scale, so overlapping kits need a limiter to keep from distorting",
    ),
    overridden(
        "uiScale",
        "0.9",
        "1.0",
        "a fresh reference client never consults this CVar: `useUiScale` registers \"0\" \
         (`0x48fce4`), and the OFF leg `0x492f70` computes clamp(768/height, 0.9, 1.0) instead — \
         0.9 at 854 px tall and up, which is every window we ship against. It is 1.0 at 768 and \
         below, where our flat 0.9 does diverge; `ui_script::DEFAULT_UI_SCALE` carries that. \
         See `useUiScale` below, whose row this one used to say did not exist",
    ),
    // `useUiScale` (`0x8430c0`), the switch the `uiScale` override gates on:
    // `ContainerFrame.lua:483` and `UIDropDownMenu.lua:525` branch on it and `OptionsFrame.lua:13`
    // gives it a checkbox.
    same("useUiScale", "0"),
    // fork: modern lighting (FORK.md) — the reference registers 350. The modern
    // look's long-range height fog shows the world past it (Orgrimmar's gate from the Durotar road
    // sits ~450 yd out), so the default view reaches 1000 yd; the slider still lowers it.
    ours(
        "farclip",
        "1000",
        "the modern look's horizon: detailed terrain to 1000 yd (reference registers 350, clamps at 777)",
    ),
    // `nearclip` (`0x68867a`: name `0x84ffb0`, default `0x84fb48` "0.1", flags 1, callback
    // `0x688d90`, record `[0xc7f348]`). The camera re-reads the record every frame: `0x511bc0`
    // (sole caller `0x483094`) stamps `[cam+0x38]` from it through the handle `[0xbe1078]` that
    // `0x50b728` caches, overwriting the ctor's 1/9 (`0x3de38e39`);
    // `benilla_world::view::stamp_near_clip` is that.
    // The callback's derived global `[0xc7b480]` has no reader. pfUI's `hdgraphic` writes
    // 0.06..0.30, inside `[0.01, 0.33]`.
    same("nearclip", "0.1"),
    // `deselectOnClick` and `mouseInvertPitch` are 1.12's own (`UIOptionsFrame.lua:8,4`);
    // `autoLootDefault` is the later-era name, 1.12 having only the shift gesture.
    same("deselectOnClick", "1"),
    // `BlockTrades` (`0x842fbc`), `UIOptionsFrame.lua:11`: the refusal leg `0x4bf7bc` fires only
    // when it is set. The knob is [`crate::ui_trade::BlockTrades`].
    same("BlockTrades", "0"),
    // `autoSelfCast` (register site `0x6e731d`, record `[0xceac34]`, read at `0x6e53d7`; `0x870dc0`
    // is its name string): a friendly cast that binds nothing falls back to the caster.
    // `TOGGLEAUTOSELFCAST` toggles it.
    deviates(
        "autoSelfCast",
        "1",
        "0",
        "with it off, an unbindable friendly cast falls into the reference's \
         targeting-cursor machine, which is unmodeled — leaving no path at all. Flip when that \
         machine lands",
    ),
    // The five saved camera views and the live index, at the reference's names and default strings;
    // owned by [`crate::player::camera_view`]. Registered so a `SaveView` persists.
    same(
        crate::player::camera_view::CVAR_ACTIVE_VIEW,
        crate::player::camera_view::ACTIVE_VIEW_DEFAULT,
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[0][0],
        crate::player::camera_view::VIEW_DEFAULTS[0][0],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[0][1],
        crate::player::camera_view::VIEW_DEFAULTS[0][1],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[0][2],
        crate::player::camera_view::VIEW_DEFAULTS[0][2],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[1][0],
        crate::player::camera_view::VIEW_DEFAULTS[1][0],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[1][1],
        crate::player::camera_view::VIEW_DEFAULTS[1][1],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[1][2],
        crate::player::camera_view::VIEW_DEFAULTS[1][2],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[2][0],
        crate::player::camera_view::VIEW_DEFAULTS[2][0],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[2][1],
        crate::player::camera_view::VIEW_DEFAULTS[2][1],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[2][2],
        crate::player::camera_view::VIEW_DEFAULTS[2][2],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[3][0],
        crate::player::camera_view::VIEW_DEFAULTS[3][0],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[3][1],
        crate::player::camera_view::VIEW_DEFAULTS[3][1],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[3][2],
        crate::player::camera_view::VIEW_DEFAULTS[3][2],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[4][0],
        crate::player::camera_view::VIEW_DEFAULTS[4][0],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[4][1],
        crate::player::camera_view::VIEW_DEFAULTS[4][1],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[4][2],
        crate::player::camera_view::VIEW_DEFAULTS[4][2],
    ),
    same("mouseInvertPitch", "0"),
    ours(
        "autoLootDefault",
        "0",
        "1.12 has no auto-loot CVar at all — vanilla offers only the shift gesture, so OFF \
         IS the reference's own behaviour; the spelling is era's",
    ),
    // The overhead-name gates, registered at `0x6c7470` into mask `0xce8720`: `UnitNamePlayer`
    // (`0x86c694`) "1" (`0x82e748`), `UnitNameNPC` (`0x86c6a4`) and `UnitNameOwn` (`0x86c6b0`) "0"
    // (`0x82e570`).
    same("UnitNamePlayer", "1"),
    same("UnitNameNPC", "0"),
    same("UnitNameOwn", "0"),
    // `UnitNamePlayerGuild` (`0x86c680`, "1", mask bit `0x10`) is not a show gate: `ShouldShowName`
    // (`0x6070a0`) reads bits `0x1/0x2/0x4`, and this gates the `"\n<%s>"` guild line at
    // `0x609085`. `UnitNamePlayerPVPTitle` (bit `0x20`, "1") has no row: nothing here draws the
    // rank prefix.
    same("UnitNamePlayerGuild", "1"),
    // 1.12 has no nameplate CVar: the bitmask `[0xc4da34]` (bits 0 and 3) persists through
    // FrameXML's `NAMEPLATES_ON`/`FRIENDNAMEPLATES_ON`, so these take the later-era names. Both
    // boot off (`UIOptionsFrame.lua:180-183`, `:769-775`): no plates until V.
    same(crate::vplates::CVAR_ENEMIES, "0"),
    same(crate::vplates::CVAR_FRIENDS, "0"),
    // `WorldDetail` is no 1.12 CVar but the `GetWorldDetail`/`SetWorldDetail` verb name
    // (`OptionsFrame.lua:27`); stops 0/1/2 are `frillDensity` 16/32/48. `SetWorldDetail 0x488dd0`
    // also writes `SmallCull` {0.07, 0.04, 0.01}, and `GetWorldDetail` reads only `SmallCull`,
    // whose registered 0.04 is stop 1: the reference's slider boots at Medium. Welded to
    // `ClutterConfig::default().density` (stop = density - 1).
    deviates(
        "WorldDetail",
        "2",
        "1",
        "the reference's own boot pair is inconsistent and its own slider reads Medium; ours starts \
         at High because ground cover is most of what the world looks like, and the stop is one \
         drag away either way",
    ),
    // `frillDensity` (`0x68862e`: name `0x8423d8`, default `0x864644` "16", flags 1, callback
    // `0x688de0`, record `[0xc7f2f4]`): detail-doodad cells visited per chunk, clamped to [1, 256]
    // and handed through `0x6725a0` to `[0xc7b494]`, the bound of the scatter loop at
    // `0x6bfcfb`/`0x6bff1c`. One knob with `WorldDetail`, each keeping its own clamp. `hwDetect`
    // (`0x639a60`) sets it from `VideoHardware.dbc` field `+0x18`, 24 on videoID 170. pfUI's
    // `hdgraphic` reads `GetCVar("frillDensity") > 48` and writes up to 256. Ours ships 48, not
    // 32: `cvars::tests` welds it to `(WorldDetail + 1) x FRILL_DENSITY`, so it follows the High stop.
    deviates(
        "frillDensity",
        "48",
        "16",
        "the reference's registered 16 is stop 0 and its post-`hwDetect` 24 is on no \
         stop at all, so every stop diverges; this is whatever stop `WorldDetail` starts on, in \
         the reference's own cells-per-chunk, and the two are welded rather than chosen apart",
    ),
    // ── Combat log display ranges, in yards ──────────────────────────────────────────────────────
    //
    // Registered by `0x626d00` from the `{name, default}` pairs at `0x8629e0`, read as the record's
    // float (`+0x24`). Classes 0 and 1, you and your pet, have no CVar there, only the `100000.0`
    // sentinel.
    same("CombatLogRangeParty", "50"),
    same("CombatLogRangePartyPet", "50"),
    same("CombatLogRangeFriendlyPlayers", "50"),
    same("CombatLogRangeFriendlyPlayersPets", "50"),
    same("CombatLogRangeHostilePlayers", "50"),
    same("CombatLogRangeHostilePlayersPets", "50"),
    same("CombatLogRangeCreature", "30"),
    // Outside that table (`0x626d5f`, "60" at `0x862e14`): `0x62c160` reads it first and falls back
    // to the per-class range only when the lookup fails.
    same(crate::ui_chat::combat::DEATH_LOG_RANGE_CVAR, "60"),
    // ── Floating combat text ─────────────────────────────────────────────────────────────────────
    //
    // `CombatDamage` (`0x6032df`, record `[0xc4d944]`) is the master: its two readers, the word
    // emitter `0x607140` and the number emitter `0x6128b0`, both return at "0", so nothing floats.
    // The `Pet*` rows gate only the owned-by-you branch; the stock Pet Melee Damage box writes both
    // (`UIOptionsFrame.lua:335-336`).
    same("CombatDamage", "1"),
    same("PetMeleeDamage", "1"),
    same("PetSpellDamage", "1"),
    // `CombatLogPeriodicSpells` (`0x6033b3`): the handle is discarded and every use looks it up by
    // name; read as the record's int.
    same(crate::ui_chat::combat::LOG_PERIODIC_CVAR, "1"),
    // ── Sound-panel check buttons ────────────────────────────────────────────────────────────────
    //
    // Category-7 registrations that keep no handle: the reference looks each up by name at use.
    //
    // `SoundListenerAtCharacter` (`0x457890`): the listener at the character, else at the camera
    // (`update_audio_listener`).
    same("SoundListenerAtCharacter", "1"),
    // `EmoteSounds` (`0x4573b9`): the received text-emote voice kit, and only that.
    same("EmoteSounds", "1"),
    // `SoundZoneMusicNoDelay` (`0x4578b3`): `next_track_time`'s immediate path.
    same("SoundZoneMusicNoDelay", "0"),
    // `assistAttack` (`0x48fc50`, record `[0xb4d8f8]`): `/assist` also starts the swing. The "3"
    // beside it is the next registration's, `minimapZoom`: stock `/assist` selects without
    // swinging.
    same("assistAttack", "0"),
    // ── Mouse-look speed, per axis ───────────────────────────────────────────────────────────────
    //
    // `cameraYawMoveSpeed` is the MOUSE_LOOK_SPEED slider (`UIOptionsFrame.lua:89`); a nil there
    // raises in `slider:SetValue` (`0x790980`) and stops `UIOptionsFrame_Load`. The stock Save
    // writes `cameraPitchMoveSpeed` as half of it (`:355-356`). The reference integrates
    // OS-accelerated pixels where we take raw device deltas, so the unit factor lives in
    // `camera::LOOK_YAW_PER_SPEED` and these defaults stay the reference's. The validator
    // `0x50c000` → `0x50b330` rejects values outside [0.1, 360] rather than clamping, and
    // `player::camera::on_cvar` does the same.
    same("cameraYawMoveSpeed", "180"),
    same("cameraPitchMoveSpeed", "90"),
    // MOUSE_SENSITIVITY (`UIOptionsFrame.lua:87`): FrameXML's spelling of the binary's `mouseSpeed`
    // (`0x402c7b`); lookups are case-insensitive (`CVar::Lookup 0x63de30`), here as there. The
    // reference's default is `SPI_GETMOUSESPEED × 0.1`, "1.0" on stock Windows, and its record
    // `[0x882704]` has no reader: the slider sets the OS pointer speed (`0x402ec0`, [0.1, 2.0]).
    // The default matches; Deviation: here the dial multiplies the camera's own rate, because
    // benilla does not change the OS pointer speed, a system-wide setting.
    same("mousespeed", "1"),
    // MAX_FOLLOW_DIST (`UIOptionsFrame.lua:90`), a factor over `cameraDistanceMax`'s 15 yd
    // (`0x84fbd0`); registered "1.0" (`0x82e92c`).
    same("cameraDistanceMaxFactor", "1"),
    // `cameraSmoothStyle` (`0x50ba92`, default `[0x84f4f4]` "1"), the auto-return behind the
    // character. The engine's enum is 0 Never, 1 Smart, 2 Always, as the stock dropdown writes it
    // (`UIOptionsFrame.lua:525,536,547`); 3, the Never entry's position, is the validator's upper
    // bound (`0x50b330(v, 0, 3)`). See `FollowStyle`.
    same("cameraSmoothStyle", "1"),
    // Read instead of `cameraSmoothStyle` while the state mask holds Track or Fear; no panel row.
    same("cameraSmoothTrackingStyle", "1"),
    // AUTO_FOLLOW_SPEED (`UIOptionsFrame.lua:88`), deg/s, registered "180.0" (`[0xbe1070]`): it
    // sets the transition's duration (`|dyaw| / rate * factor`), an average rate, not a slew. The
    // knob clamps it to `FOLLOW_SPEED_RANGE`. The stock Save also writes `cameraPitchSmoothSpeed`
    // at a quarter (`:353`), unregistered here: `FollowRig` has one rate.
    same("cameraYawSmoothSpeed", "180"),
    // `cameraPivot` `[0xbe10a4]` "1" (`0x50bda3`), smart pivot: gate `0x510690`, routing
    // `0x50fee0`, release `0x5107f0`; ours is `player::camera_dynamics::SmartPivot`.
    same("cameraPivot", "1"),
    // Read by the routing (`0x50fff5`/`0x510004`) in radians of camera rotation, so they carry to
    // benilla's raw-device units unchanged.
    same("cameraPivotDXMax", "0.05"),
    same("cameraPivotDYMin", "0"),
    // The pitch bias's ease-back once the pivot lets go, deg/s (`[0xbe0fc8]`; `0x512a50` divides
    // |Δ| by rate · π/180 for the duration); no panel row.
    same("cameraTargetSmoothSpeed", "90"),
    // `cameraWaterCollision` `[0xbe1088]` "1" (`0x50bd63`, `0x82e748`): one register, two consumers
    // that must ship together. `0x50e5ec` builds it; its `0xf0000` nibble joins the trace mask of
    // all three `0x50e570` queries (reaching `0x69cc13`), and `0x50e629` tests it to lift the sweep
    // origin to `surface + 2/9`. Ours: `benilla_world::collision::camera_filter` and
    // `player::camera_water`.
    same("cameraWaterCollision", "1"),
    // `cameraTerrainTilt` `[0xbe0fd4]` "0" (`0x50bcfd`), Follow Terrain: probe and staircase
    // `0x50d900`, arm `0x50dbc0`; ours is `player::camera_dynamics::TerrainTilt`.
    same("cameraTerrainTilt", "0"),
    // Rate in deg/s (`[0xbe0fc0]`), duration bounds in seconds (`[0xbe1050]`/`[0xbe1054]`). The 3 s
    // floor always binds (20° at 7.5°/s is 2.67 s), so the camera leans rather than tracks.
    same("cameraGroundSmoothSpeed", "7.5"),
    same("cameraTerrainTiltTimeMin", "3"),
    same("cameraTerrainTiltTimeMax", "10"),
    // `cameraBobbing` `[0xbe10c0]` "0" (`0x50b76d`), head bob: kernel `0x511920`, gate `0x5105e0`;
    // ours is `player::camera_dynamics::HeadBob`.
    same("cameraBobbing", "0"),
    // Amplitudes in the CVar's units, scaled by 1/36 (`[0x7ff9d0]`) to yards.
    // `cameraBobbingSmoothSpeed` is the decay rate, read only in the disarm `0x51113a`, which
    // divides the largest component by it for the ramp's duration (~0.069 s at these defaults).
    same("cameraBobbingLRAmplitude", "2"),
    same("cameraBobbingUDAmplitude", "2"),
    same("cameraBobbingFrequency", "0.8"),
    same("cameraBobbingSmoothSpeed", "0.8"),
    // `statusBarText` (`0x48fc34`, record `[0xb4d904]`, no engine reader), read by
    // `TextStatusBar.lua:47,97`: "0" shows the numbers on hover only.
    same("statusBarText", "0"),
    // Enhanced Tooltips (`UIOptionsFrame.lua:15`), registered "1" at `0x48fddd` (`0x82e748`); read
    // only by the stock interface.
    same("UberTooltips", "1"),
    // Registered at `0x603280`: `ChatBubbles` "1", `ChatBubblesParty` "0".
    same("ChatBubbles", "1"),
    same("ChatBubblesParty", "0"),
    // Registered "1" (`0x82e748`), category 4. `profanityFilter` (`0x402e68`, name `0x82e7f4`,
    // callback `0x403570`) masks `ChatProfanity.dbc` spans inside the shared masker `0x4a1a66`,
    // covering all thirteen call sites; `spamFilter` (`0x402e8e`, name `0x82e7d4`, callback
    // `0x4035b0`) silently drops a line matching `SpamMessages.dbc`. The knob is
    // [`crate::text_filter::TextFilterSwitches`].
    same("profanityFilter", "1"),
    same("spamFilter", "1"),
    // Registered by `CGlueMgr::EnterWorld` (`0x46b633` `gameTip` "0", `0x46b658` `showGameTips`
    // "1"), category 5. `gameTip` is the cursor and holds the next tip, not the one on screen;
    // `crate::game_tip` advances it.
    same("gameTip", "0"),
    same("showGameTips", "1"),
    // `showLootSpam` (`0x48fd1c`, name `0x8430a0`, "1" `0x82e748`, record `0xb4e2bc`): off, group
    // loot-roll lines are hidden and only the winner shows. Its three readers are the roll-line
    // composers; the knob is [`crate::ui_loot::LootConfig::show_loot_spam`].
    same("showLootSpam", "1"),
    // `guildMemberNotify` (`0x5e24c7`, "0" `0x82e570`, record `0xc4d3c4`): guildmate log on/off
    // lines, read only in `SMSG_GUILD_EVENT`'s handler. The knob is
    // [`crate::ui_guild::GuildMemberNotify`].
    same("guildMemberNotify", "0"),
    // Both registered "3" (`0x48fc6c`, `0x48fc88`); the minimap's +/- buttons write them through
    // `Minimap:SetZoom`. The knob is [`crate::minimap::MinimapZoom`].
    same("minimapZoom", "3"),
    same("minimapInsideZoom", "3"),
    // Load out of date AddOns, inverted (`0x402c3b`): "1" enforces the check. Read by the load walk
    // ([`Cvars::addon_version_check`]) and live by the VM's gate.
    same("checkAddonVersion", "1"),
    // `gxApi` (`0x63a833`: name `0x842a64`, default `0x864f7c` "direct3d", flags 3, callback
    // `0x63b030`, record `[0xc4ea94]`): the reference builds D3D9 unless it reads "OpenGl"
    // (`0x63a3c4`, `0x842a5c`). Here it reports the wgpu backend (`wgpu::Backend::to_str`), pushed
    // from `RenderAdapterInfo` and owned by the session, so it is never persisted. pfUI's
    // `panel.lua:185` concatenates it.
    deviates(
        "gxApi",
        "",
        "direct3d",
        "descriptive, not a selector — benilla renders through wgpu, which has no D3D9 \
         backend and no chooser; the value is the live adapter's own and is never persisted",
    )
    .latched(),
    // `gxVSync` (`0x63a859`, "1", flags 3), `OptionsFrame.lua:9`. The knob is
    // [`crate::video::VideoConfig::vsync`], which the window's present mode follows at the
    // `RestartGx` commit. `$WOW_NOVSYNC=1` overrides it for the session.
    same("gxVSync", "1").latched(),
    // Benilla's opt-in Wrath-style character shadow-map path. Static world assets keep their
    // authored shading; players, NPCs, creatures and mounts cast realtime silhouettes.
    // **Default "1" since 2026-09-08** (the director's call), where it was "0". The old note said
    // "0 is the default no frame moves at", which was the right reason for a path that had just
    // landed and is not the question any more: what the client should LOOK like out of the box is,
    // and the answer is with the shadows on. Welded to `VideoConfig::default().world_shadows`
    // below, so the registered string and the knob cannot drift.
    ours(
        "worldShadows",
        "1",
        "benilla's own — 1.12 has no shadow map to switch on; the unit shadow it does draw is \
         the blob decal `blob_shadow.rs` still owns at 0",
    ),
    // `gxWindow` (`0x63a889`): "0" on enUS; zhCN registers "1", as it does `gxMaximize`
    // (`0x63a8e0`), and koKR `AutoInteract` (`0x603390`). The knob is
    // [`crate::video::VideoConfig::display`]. Deviation: "0" raises a borderless fullscreen window
    // instead of mode-setting the display, because Wayland and macOS offer no mode-set and X11's
    // leaves the desktop changed after a crash ([`crate::video`]).
    same("gxWindow", "0").latched(),
    // `gxResolution`, a string row parsed by `video::on_cvar`. Here it is only the windowed size:
    // fullscreen is the monitor's own and no mode list is offered.
    deviates(
        "gxResolution",
        "1600x900",
        "640x480",
        "narrowed to the WINDOWED size only — fullscreen is the monitor's own and we expose \
         no mode list, and 640x480 is not a window anyone would ship a client at",
    )
    .latched(),
    // benilla's own: a body pane's doll renders at half the frame rate while the pane is open; the
    // reference draws it in the main pass. The knob is [`crate::portrait::PaneRate`].
    ours(
        "boothHalfRate",
        "1",
        "benilla's own — the reference draws its doll inside the main pass and has no \
         second view to rate-limit",
    ),
    // `gxMultisample` (`0x63a950`, flags 3). The reference formats its default from field 21 of the
    // `VideoHardware.dbc` row `DetectHardware` (`0x641260`) matches, which is 1, no multisampling,
    // on every fallback row a modern GPU reaches. The knob is [`benilla_world::view::MsaaSetting`],
    // read once at the camera's spawn; `$WOW_MSAA` overrides it for the session.
    same("gxMultisample", "1").latched(),
    // `GetCurrentMultisampleFormat 0x48c580` looks up all three of the Video dropdown's values by
    // name, so these must exist. They describe the swapchain's own pair and steer nothing;
    // `SetMultisampleFormat` writes them as `0x48c640` does.
    deviates(
        "gxColorBits",
        "32",
        "16",
        "these describe, they do not steer — the pair is our swapchain's own, and every \
         format `MsaaFormats` publishes carries it",
    )
    .latched(),
    deviates(
        "gxDepthBits",
        "32",
        "16",
        "as `gxColorBits` — the depth half of the same descriptive pair",
    )
    .latched(),
    // `trilinear` and `anisotropic`, over `benilla_assets::TexFilterSetting` (`tex_filter.rs`
    // carries the derivation). A change applies at the next launch, since a sampler is baked into
    // each texture at load; the reference's UI says "enabled upon restart".
    // `$WOW_TRILINEAR`/`$WOW_ANISO` override for the session.
    //
    // `trilinear` registers "0", but `hwDetect` runs `DetectHardware 0x641260` and sets it from the
    // matched `VideoHardware.dbc` row before the first frame; the fallback rows an unlisted GPU
    // reaches are 168/169/170, and 169 and 170 give 1. The reference install's `gx.log` resolves to
    // videoID 170.
    overridden(
        "trilinear",
        "1",
        "0",
        "`hwDetect` sets it from `VideoHardware.dbc` field 9 before the first frame, and \
         that field is 1 on both fallback rows an unlisted modern GPU can reach — measured on the \
         reference's own `Logs/gx.log` (`videoID: 170`)",
    ),
    // `anisotropic` registers `"1"` (off) and `hwDetect` never sets it (`0xc7f2e4` is absent from
    // `[0x639a60, 0x639b80)`).
    deviates(
        "anisotropic",
        "16",
        "1",
        "maintainer's call, reversing 1642: at mode 4 an oblique surface takes the mip of its long \
         axis and smears; 16x anisotropy costs nothing measurable on a current GPU",
    ),
    // Weather Intensity (`OptionsFrame.lua:33`), registered "2" (`0x67b806`, flags 0, callback
    // `0x67b870`, name `0x8685ac`). The reader is `benilla_world::weather::WeatherState`, which
    // scales the precipitation spawn rate by `0x67b870`'s table {0.1, 0.33, 0.66, 1.0}; rendering
    // only.
    deviates(
        "weatherDensity",
        "3",
        "2",
        "every precipitation rate in `benilla-world`'s own precipitation module was \
         derived and graded against the reference install's own apitrace captures, and that \
         install runs \
         `SET weatherDensity \"3\"` (K = 1.0) — so 3 is the value a benilla-vs-reference \
         side-by-side is correct at, and the registered 2 would thin every rate to 0.66 against \
         the only client we compare with. The slider is how a player takes it back down",
    ),
    // `gamma` (`0x402d70`: name `0x82e924` "Gamma", default `0x82e92c` "1.0", flags 0, callback
    // `0x4034d0`). The reference uploads `pow(i/255, gamma)` (`0x591680`) through
    // `SetDeviceGammaRamp`, except when windowed (`byte[dev+0x20b]`). Deviation: benilla, which
    // has no exclusive mode, applies the same curve in the composite pass
    // ([`crate::ui_gamma::DisplayGamma`]), because the skipped upload would make a slider that
    // moves no pixel. Spelled "1.000000" because `SetGamma 0x4891f0` formats with `"%f"`, and
    // Restore Defaults must compare equal to the default; [`sync_cvars`] seeds it the same way.
    same("gamma", "1.000000"),
    // benilla's own: the world renders at `window × this` while the UI stays native. The knob is
    // [`crate::world_backdrop::RenderScale`], clamped to `RENDER_SCALE_RANGE`; at "1" nothing is
    // resampled. `$WOW_RENDER_SCALE` overrides it for the session.
    ours(
        "renderScale",
        "1",
        "benilla's own — the reference has no off-screen buffer to hang a resolution dial \
         on; its nearest equivalent, `gxResolution`, drops the interface with the world",
    ),
    // **Screen-space indirect lighting** — benilla's own pair, no 1.12 counterpart in the same
    // sense `renderScale` has none: the reference renders straight to the backbuffer with no depth
    // it could gather over, and its ambient IS the authored `LightParams` row. One pass computes
    // both halves off one swept horizon (`benilla_world::ssilvb`), so they are two CVars rather
    // than one only because a player wants to move them apart: contact darkening is a fidelity
    // choice, colour bleed is a taste one.
    //
    // `ssaoStrength` — how much of the gathered occlusion to apply. `0` is off, and off is exact:
    // the fragment returns its source untouched rather than through a lossy decode/encode pair.
    //
    // **Registered at 0 since 2026-09-08** (the director's call), where it was 0.5. It now sits with
    // `worldShadows` and `renderScale`: a default at the value where no frame moves, and an effect a
    // player opts into rather than one they have to find and switch off. The reasoning lives on the
    // engine constant; the engine owns the number ([`AO_DEFAULT`]) and this string is welded to it
    // below, so the two cannot drift. `$WOW_SSAO` overrides session-only.
    // fork: 0.4 since 2026-09-17 (the director's call), welded to `AO_DEFAULT` below.
    ours(
        "ssaoStrength",
        "0.4",
        "benilla's own — the reference renders straight to the backbuffer with no depth to \
         gather over, and its ambient IS the authored `LightParams` row",
    ),
    // `ssgiStrength` — how much of the single bounce to add back, tinted by what bounced it. `0` is
    // off and is exact, on the same terms. `$WOW_SSGI` overrides session-only.
    ours(
        "ssgiStrength",
        "0",
        "benilla's own, on `ssaoStrength`'s terms — one swept horizon feeds both halves, \
         split across two rows only so a player can move them apart",
    ),
    // ── fork-owned block: modern lighting (FORK.md) ──
    // `lightingModel` — 0 classic (upstream's 1.12.1 reproduction), 1 modern (the default).
    // `lightingQuality` — the modern tier, 0 low … 4 cinematic (the default, on every platform).
    // Cinematic (P8) is the local-light tier: every authored lamp and every glowing
    // weapon emits, twelve of them cast soft cube shadows, and entities cast into those cubes.
    // Both apply to the next frame in the world.
    // `$WOW_LIGHTING` / `$WOW_LIGHTING_QUALITY` override session-only.
    ours(
        "lightingModel",
        "1",
        "the fork's own — the reference has one lighting model; this selects the parallel one",
    ),
    ours(
        "lightingQuality",
        "4",
        "the fork's own — the modern model's quality tier",
    ),
    // ── end fork-owned block ──
    // **`waterDetail` — which liquid renderer draws the world's water** (the Cataclysm liquid port).
    //
    // The name and the value space are the REFERENCE's, taken from a later client rather than
    // invented here: `Liquid::CMaterialWater::InitResources` (WoW 4.1.0) switches on exactly this
    // CVar, binding `Water` at 0, `MediumWater` at 1 and `ProcWaterAbove`/`ProcWaterBelow` at 2.
    // benilla implements 0 (its 1.12.1 renderer, `liquid.wgsl`) and 2 (`liquid_cata.wgsl`), and
    // deliberately leaves the 1 slot EMPTY rather than renumbering to a dense 0/1 — a medium tier
    // is a real thing that could land later, and it should land on its own number instead of
    // shifting a value players already have in `config.toml`. (fork: 1 is since the SSR-only cata
    // tier, and 3 the experimental modern combine — `liquid::WATER_DETAIL_EXPERIMENTAL`.)
    //
    // **Registered at 2**, so cata water is what the client looks like out of the box. That is a
    // benilla choice, not a reference default (1.12.1 has no such CVar at all: its water is the
    // only water it has), which is why this is an `ours` row. `$WOW_WATER_DETAIL` overrides
    // session-only.
    ours(
        "waterDetail",
        "3",
        "the reference's own CVar and value space, from 4.1.0's liquid system - benilla ports          its tier 2 (ProcWater + ProcWaterAbove/Below), ships the fork's Modern (Experimental) tier 3 by default, and keeps 1.12.1's renderer on tier 0",
    ),
    // **`hdCharacterModels` — the updated-character-model layer, ON by default.**
    //
    // benilla's own, on `waterDetail`'s terms but for a different kind of thing: the rows above
    // choose between renderers we ship, and this one chooses whether a body of ART is in the
    // composite at all. `patch-A.MPQ` natively, a second BENPACK container (`/data-hd`) in the
    // browser; either way [`benilla_formats::layers`] is what reads it and
    // [`benilla_formats::Chain::open`] is where it lands.
    //
    // **Registered "1", so the row is an OPT-OUT.** It was registered "0" until 2026-09-09, on
    // three recorded reasons; the flip answers each rather than overruling it:
    //
    //  1. *"On the web build the layer is another 1.7 GiB download, and a player who never asked
    //     for it must never pay for it."* Still true, and still the only reason the layer is a
    //     separate container at all — turning the box OFF fetches no index and no pack, exactly as
    //     before. What changed is which way round the question is asked: this site ships the
    //     layer, so the art a player gets by default is the art the site is for. The tooltip
    //     states the size before the click either way.
    //  2. *"Every visual golden in this tree is denominated in the original character models."*
    //     Still true, and untouched — the goldens do not read this row. `benilla_formats::layers`'
    //     own static stays **off**, so every test, tool and `benilla-worldview` probe keeps
    //     asserting 1.12 art unless `$WOW_HD_MODELS` says otherwise. This row moves the CLIENT's
    //     answer and nothing else.
    //  3. *"A player with no `patch-A.MPQ` and no `/data-hd` would take a warning line every
    //     boot."* Measured: they do not. Natively the archive is simply not in the folder and
    //     `Chain::open`'s layer filter has nothing to skip — no warning, original art. On the web
    //     `Chain::open_layered` warns **once to the console** and runs on the original art, which
    //     is the state that arm was written for.
    //
    // **Read at boot** ([`crate::cvars::boot_cvar`], like `gxWindow`), for the chains the client
    // opens then. On the web build the layer is a live mount afterwards too: a change of the row
    // swaps it at that edge ([`crate::hd_models`]). A native chain decides its archives as it
    // opens, so there the next start reads it, and the tooltip says which. `$WOW_HD_MODELS`
    // overrides session-only, which is how the model test suite runs against the layer.
    ours(
        "hdCharacterModels",
        "1",
        "benilla's own — 1.12 has no optional content layer to switch on; the reference mounts          whatever `patch-?.MPQ` archives it finds and never asks",
    ),
    // benilla's own: `/console fpsJournal 1` appends a per-second row of position, frame cost and
    // per-pass GPU time to `benilla-config/Diagnostics/fps-journal.csv`. The knob is
    // [`crate::perf::FpsJournalSetting`].
    ours(
        "fpsJournal",
        "0",
        "benilla's own — 1.12 has no player-side perf log; its nearest thing is the \
         Ctrl+R framerate label, a number with no file behind it",
    ),
    // `lastCharacterIndex` (`0x402d93`, "0" `0x82e570`, category 4, handle `[0x882674]`), help
    // "Last character selected": a 0-based row (the selection cell `[0x83856c]` under `"%d"`), so
    // "0" is the first character. It mirrors [`crate::char_select::Roster::pending_index`].
    same(crate::char_select::CVAR_LAST_CHARACTER, "0"),
];

/// `config.toml`: a `[cvars]` table of `Name = "value"` strings, sorted so every save is stable.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct LocalConfig {
    #[serde(default)]
    cvars: BTreeMap<String, String>,
}

// ─── The registry ────────────────────────────────────────────────────────────────────────────

/// An accepted move of a CVar's applied value: the reference's change callback, as a Bevy event.
/// Not fired for a no-op write, a staged latched value, or a session override.
#[derive(Event, Clone, Debug, PartialEq, Eq)]
pub(crate) struct CvarChanged {
    /// The registered spelling (`MasterVolume`).
    pub(crate) name: String,
    pub(crate) old: String,
    pub(crate) new: String,
}

impl CvarChanged {
    /// Case-insensitive, like every lookup the client makes.
    pub(crate) fn is(&self, name: &str) -> bool {
        self.name.eq_ignore_ascii_case(name)
    }

    /// The lowercased name, as observers' `match` arms spell it.
    pub(crate) fn key(&self) -> String {
        self.name.to_ascii_lowercase()
    }

    /// The new value as a number; `0` on a string row, whose observer reads [`Self::new`]. The
    /// registry refuses an unparseable write to a numeric row ([`Cvars::set`]).
    pub(crate) fn num(&self) -> f32 {
        self.new.trim().parse().unwrap_or(0.0)
    }

    /// The new value as the client's flag: int-parse, then `!= 0`.
    pub(crate) fn flag(&self) -> bool {
        self.num() != 0.0
    }
}

/// One row of the live registry: the parts of the reference's `CVar` record benilla keeps.
#[derive(Clone, Debug)]
pub(crate) struct Row {
    /// The registered spelling.
    pub(crate) name: String,
    pub(crate) default: String,
    /// The applied value: what `GetCVar` answers and what the file is composed from.
    pub(crate) value: String,
    /// A latched row's staged value (`rec+0x38`), applied by [`Cvars::commit_latched`].
    pub(crate) pending: Option<String>,
    /// The reference's flag bit1.
    pub(crate) latched: bool,
    /// Declared by an addon's `RegisterCVar`: persisted like any row and re-seeded into every later
    /// VM, so the addon's re-declaration is a no-op.
    pub(crate) addon: bool,
}

impl Row {
    /// A row whose default parses as a number refuses a write that does not ([`Cvars::set`]).
    fn numeric(&self) -> bool {
        self.default.trim().parse::<f32>().is_ok()
    }
}

/// What a write did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SetOutcome {
    /// No such row (warned).
    Unknown,
    /// A numeric row and a value that does not parse: refused, the applied value stands (warned).
    Refused,
    /// Already the applied value (or already the staged one).
    Unchanged,
    /// A latched row: staged, not applied; nothing fires until the commit.
    Staged,
    /// Applied: a [`CvarChanged`] is queued for the next flush and the config is dirty.
    Changed,
}

/// The engine-side CVar table.
#[derive(Resource)]
pub(crate) struct Cvars {
    rows: Vec<Row>,
    /// Lowercased name → row.
    index: HashMap<String, usize>,
    /// The file's `[cvars]` entries in their own spelling, the merge base of every save.
    file: BTreeMap<String, String>,
    /// Lowercased names the session owns rather than the player: the env levers, and `gxApi`, the
    /// render backend. Never saved; the file's entry is left as found.
    session_owned: HashSet<String>,
    /// The session-owned rows a player's own write takes back (fork: web build) — see
    /// [`Self::fallback_for_session`] and [`Self::cap_for_session`] — each with the value it
    /// displaced, which [`Self::release_fallbacks`] restores. Keys are a subset of `session_owned`.
    yielding: HashMap<String, String>,
    /// A player's write moved a [`GRAPHICS_CVARS`] row since [`sync_cvars`] last looked (fork: web
    /// build) — the Graphics page's Apply or Defaults, which retires the GPU recovery tier.
    graphics_applied: bool,
    /// Accepted moves not yet triggered, flushed by [`sync_cvars`], the boot load, the session-edge
    /// fold, or a caller of [`Cvars::take_events`].
    events: Vec<CvarChanged>,
    /// Host-side writes the VM's mirror has not seen; cleared by a seed.
    outbox: Vec<(String, String)>,
    /// A change since the last save; `last_change` drives the one-quiet-second debounce.
    dirty: bool,
    last_change: Option<Instant>,
}

impl Default for Cvars {
    fn default() -> Self {
        let mut cvars = Self {
            rows: Vec::with_capacity(REGISTERED.len()),
            index: HashMap::with_capacity(REGISTERED.len()),
            file: BTreeMap::new(),
            session_owned: HashSet::new(),
            yielding: HashMap::new(),
            graphics_applied: false,
            events: Vec::new(),
            outbox: Vec::new(),
            dirty: false,
            last_change: None,
        };
        for r in REGISTERED {
            cvars.insert_row(Row {
                name: r.name.to_string(),
                default: r.default.to_string(),
                value: r.default.to_string(),
                pending: None,
                latched: r.latched,
                addon: false,
            });
        }
        cvars
    }
}

impl Cvars {
    fn insert_row(&mut self, row: Row) {
        let key = row.name.to_ascii_lowercase();
        debug_assert!(
            !self.index.contains_key(&key),
            "{}: registered twice",
            row.name
        );
        self.index.insert(key, self.rows.len());
        self.rows.push(row);
    }

    fn slot(&self, name: &str) -> Option<usize> {
        self.index.get(&name.to_ascii_lowercase()).copied()
    }

    /// The row, matched case-insensitively.
    pub(crate) fn row(&self, name: &str) -> Option<&Row> {
        self.slot(name).map(|i| &self.rows[i])
    }

    /// Every row, in registration order.
    pub(crate) fn rows(&self) -> impl Iterator<Item = &Row> {
        self.rows.iter()
    }

    pub(crate) fn get(&self, name: &str) -> Option<&str> {
        self.row(name).map(|r| r.value.as_str())
    }

    pub(crate) fn num(&self, name: &str) -> Option<f32> {
        self.get(name).and_then(|v| v.trim().parse().ok())
    }

    /// The applied value as the client's flag (int-parse, `!= 0`).
    pub(crate) fn flag(&self, name: &str) -> Option<bool> {
        self.num(name).map(|v| v != 0.0)
    }

    pub(crate) fn default_of(&self, name: &str) -> Option<&str> {
        self.row(name).map(|r| r.default.as_str())
    }

    pub(crate) fn is_session_owned(&self, name: &str) -> bool {
        self.session_owned.contains(&name.to_ascii_lowercase())
    }

    /// The persisted `checkAddonVersion` the addon load walk gates on; registered on.
    pub(crate) fn addon_version_check(&self) -> bool {
        self.flag("checkAddonVersion").unwrap_or(true)
    }

    fn touch(&mut self) {
        self.dirty = true;
        self.last_change = Some(Instant::now());
    }

    /// A write from either side of the VM boundary. A `from_vm` write is not echoed back, but a
    /// refusal is, because the mirror already stored the refused value.
    fn write(&mut self, name: &str, value: &str, from_vm: bool) -> SetOutcome {
        let Some(i) = self.slot(name) else {
            warn!("cvar {name}: not registered — write ignored");
            return SetOutcome::Unknown;
        };
        let row = &mut self.rows[i];
        if row.numeric() && value.trim().parse::<f32>().is_err() {
            warn!(
                "cvar {}: unparseable value '{value}' refused (still {:?})",
                row.name, row.value
            );
            if from_vm {
                self.outbox.push((row.name.clone(), row.value.clone()));
            }
            return SetOutcome::Refused;
        }
        if row.latched {
            // The reference's `Set 0x63df50` on flag bit1 stores `latchedValue` and skips
            // `InternalSet`: no dirty, no callback. The stage lives here, not in the mirror, which
            // only ever learns applied values.
            let staged = (value != row.value).then(|| value.to_string());
            if row.pending == staged {
                return SetOutcome::Unchanged;
            }
            row.pending = staged;
            return if row.pending.is_some() {
                SetOutcome::Staged
            } else {
                SetOutcome::Unchanged // the stage cleared: the boundary has nothing to do
            };
        }
        // A player's write takes a yielding row back: from here on it is theirs and is saved.
        let key = row.name.to_ascii_lowercase();
        let reclaimed = from_vm && self.yielding.remove(&key).is_some();
        if reclaimed {
            self.session_owned.remove(&key);
        }
        if row.value == value {
            if reclaimed {
                self.touch();
            }
            return SetOutcome::Unchanged;
        }
        let old = std::mem::replace(&mut row.value, value.to_string());
        let name = row.name.clone();
        self.events.push(CvarChanged {
            name: name.clone(),
            old,
            new: value.to_string(),
        });
        if !from_vm {
            self.outbox.push((name, value.to_string()));
        }
        self.touch();
        SetOutcome::Changed
    }

    /// A host-side write (minimap zoom, camera views, the remembered character, the tip cursor):
    /// mirrored into the VM, persisted and observed.
    pub(crate) fn set(&mut self, name: &str, value: &str) -> SetOutcome {
        self.write(name, value, false)
    }

    /// A write the VM's mirror already made, drained from its change queue.
    pub(crate) fn set_from_vm(&mut self, name: &str, value: &str) -> SetOutcome {
        let outcome = self.write(name, value, true);
        if matches!(outcome, SetOutcome::Changed | SetOutcome::Staged)
            && GRAPHICS_CVARS.iter().any(|c| c.eq_ignore_ascii_case(name))
        {
            self.graphics_applied = true;
        }
        outcome
    }

    /// Whether a player's write moved a Graphics row since the last call — and clear it.
    pub(crate) fn take_graphics_applied(&mut self) -> bool {
        std::mem::take(&mut self.graphics_applied)
    }

    /// Follow a value the engine already applied, for a second spelling of one knob
    /// (`WorldDetail`/`frillDensity`): the row moves, persists and reaches the mirror, but no
    /// observer fires, since a full write would queue an event that lands a flush late and wins
    /// stale. Returns whether the row moved.
    pub(crate) fn mirror(&mut self, name: &str, value: &str) -> bool {
        let Some(i) = self.slot(name) else {
            warn!("cvar {name}: not registered — mirror ignored");
            return false;
        };
        let row = &mut self.rows[i];
        if row.value == value {
            return false;
        }
        row.value = value.to_string();
        row.pending = None;
        self.outbox.push((row.name.clone(), value.to_string()));
        self.touch();
        true
    }

    /// The latch boundary `RestartGx` crosses: the reference's `0x639ec0..0x639f5a`, which calls
    /// `CVar::Commit` (`0x63e060`) on the fourteen gx records `[0xc4ea90]` … `[0xc4eab4]`. Each
    /// staged `gx*` value is applied, fires, persists and reaches the mirror; returns how many
    /// moved.
    ///
    /// Only `gx*` rows: `SoundBufferSize`'s register site (`0x4571ca`) discards its record, so
    /// nothing commits it and its stage is lost at exit, as in the reference. The reference runs a
    /// gx callback at `SetCVar` time (`0x63df50`); ours fire here, before the device rebuild reads
    /// them.
    pub(crate) fn commit_latched(&mut self) -> usize {
        let mut moved = 0;
        for row in &mut self.rows {
            if !row.name.starts_with("gx") {
                continue;
            }
            let Some(staged) = row.pending.take() else {
                continue;
            };
            if staged == row.value {
                continue;
            }
            let old = std::mem::replace(&mut row.value, staged.clone());
            self.events.push(CvarChanged {
                name: row.name.clone(),
                old,
                new: staged.clone(),
            });
            self.outbox.push((row.name.clone(), staged));
            moved += 1;
        }
        if moved > 0 {
            self.touch();
        }
        moved
    }

    /// The session owns this row: it is never saved and the file's entry is left alone. `value` is
    /// what it answers this session; `None` marks it without moving it. No observer fires: the knob
    /// already read the env.
    pub(crate) fn own_for_session(&mut self, name: &str, value: Option<&str>) {
        let key = name.to_ascii_lowercase();
        self.session_owned.insert(key.clone());
        let Some(value) = value else {
            return;
        };
        let Some(&i) = self.index.get(&key) else {
            warn!("cvar {name}: not registered — session value ignored");
            return;
        };
        let row = &mut self.rows[i];
        if row.value != value {
            row.value = value.to_string();
            row.pending = None;
            self.outbox.push((row.name.clone(), value.to_string()));
        }
    }

    /// **This session owns the row, and the observers hear the value** — [`Self::own_for_session`]
    /// for a lever no knob read at its `Default`: the page's `?name=value` query string and the
    /// hosted `config.json` (fork: web build). Session-owned, so never saved and the file's entry
    /// is kept; unlike `own_for_session`, the move is queued as a [`CvarChanged`] because nothing
    /// else would carry it to the knob. A numeric row refuses a value that does not parse, as a
    /// write does. Returns whether the row moved.
    pub(crate) fn override_for_session(&mut self, name: &str, value: &str) -> bool {
        let key = name.to_ascii_lowercase();
        let Some(&i) = self.index.get(&key) else {
            warn!("cvar {name}: not registered — session value ignored");
            return false;
        };
        self.session_owned.insert(key);
        let row = &mut self.rows[i];
        if row.numeric() && value.trim().parse::<f32>().is_err() {
            warn!(
                "cvar {}: unparseable session value '{value}' ignored",
                row.name
            );
            return false;
        }
        row.pending = None;
        if row.value == value {
            return false;
        }
        let old = std::mem::replace(&mut row.value, value.to_string());
        self.events.push(CvarChanged {
            name: row.name.clone(),
            old,
            new: value.to_string(),
        });
        self.outbox.push((row.name.clone(), value.to_string()));
        true
    }

    /// **A value for the session only where the player has none** (fork: web build) — the page's
    /// GPU recovery tier. It applies like [`Self::override_for_session`] (session-owned, never
    /// saved), but only when the file holds no value of the player's own, and the player's next
    /// write through the VM takes the row back ([`Self::write`]), so a scale chosen in the Graphics
    /// page is applied, saved, and outranks the tier on every later load. Returns whether the row
    /// moved.
    pub(crate) fn fallback_for_session(&mut self, name: &str, value: &str) -> bool {
        let key = name.to_ascii_lowercase();
        if self.session_owned.contains(&key)
            || self.file.keys().any(|k| k.eq_ignore_ascii_case(name))
        {
            return false;
        }
        self.yield_for_session(name, value)
    }

    /// **A ceiling for the session only** (fork: web build) — the GPU recovery tier's lighting
    /// quality. Unlike [`Self::fallback_for_session`] it holds a saved value down as well: a tier
    /// is the machine's evidence, and a saved Cinematic is exactly what it must not reload into.
    /// A row already at or under the ceiling, or owned by a lever, is left alone. Session-owned
    /// and never saved (the file keeps the player's value), and taken back by the player's own
    /// write, or restored by [`Self::release_fallbacks`]. Returns whether the row moved.
    pub(crate) fn cap_for_session(&mut self, name: &str, cap: &str) -> bool {
        if self.is_session_owned(name) {
            return false;
        }
        let (Some(current), Ok(ceiling)) = (self.num(name), cap.trim().parse::<f32>()) else {
            return false;
        };
        if current <= ceiling {
            return false;
        }
        self.yield_for_session(name, cap)
    }

    /// The shared half of the two: take the row for the session, remembering what it displaced.
    fn yield_for_session(&mut self, name: &str, value: &str) -> bool {
        let key = name.to_ascii_lowercase();
        let Some(prior) = self.get(name).map(str::to_string) else {
            warn!("cvar {name}: not registered — session value ignored");
            return false;
        };
        let moved = self.override_for_session(name, value);
        if self.session_owned.contains(&key) {
            self.yielding.insert(key, prior);
        }
        moved
    }

    /// **Every row the recovery tier still holds, back to the player** (fork: web build): each
    /// leaves the session's hands and returns to the value it displaced — the registered default
    /// under a fallback, the saved or default value under a cap — which is what a load without the
    /// tier gives. The observers hear the move. Returns how many rows moved.
    pub(crate) fn release_fallbacks(&mut self) -> usize {
        let mut moved = 0;
        for (key, prior) in std::mem::take(&mut self.yielding) {
            self.session_owned.remove(&key);
            let Some(&i) = self.index.get(&key) else {
                continue;
            };
            let row = &mut self.rows[i];
            row.pending = None;
            if row.value == prior {
                continue;
            }
            let old = std::mem::replace(&mut row.value, prior.clone());
            self.events.push(CvarChanged {
                name: row.name.clone(),
                old,
                new: prior.clone(),
            });
            self.outbox.push((row.name.clone(), prior));
            moved += 1;
        }
        moved
    }

    /// An addon's `RegisterCVar`: a row of its own at the file's value for the name, else at the
    /// declared default. A name already registered is a no-op.
    pub(crate) fn learn_addon_row(&mut self, name: &str, default: &str) {
        if self.slot(name).is_some() {
            return;
        }
        let saved = self
            .file
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone());
        self.insert_row(Row {
            name: name.to_string(),
            default: default.to_string(),
            value: saved.unwrap_or_else(|| default.to_string()),
            pending: None,
            latched: false,
            addon: true,
        });
    }

    /// Fold the file in: a known, player-owned key becomes its row's applied value and fires (the
    /// reference's `Register` on a record `Config.wtf` created calls the callback); an unknown key
    /// is kept for the save and warned; a session-owned one is skipped. Nothing dirties.
    fn load_file(&mut self, file: BTreeMap<String, String>) {
        for (name, value) in &file {
            let key = name.to_ascii_lowercase();
            let Some(&i) = self.index.get(&key) else {
                warn!("config: unknown cvar '{name}' — preserved, not applied");
                continue;
            };
            if self.session_owned.contains(&key) {
                info!("config: {name} is owned by this session, not the file (file value kept)");
                continue;
            }
            let row = &mut self.rows[i];
            if row.numeric() && value.trim().parse::<f32>().is_err() {
                warn!("config: {name}: unparseable value '{value}' ignored");
                continue;
            }
            if row.value == *value {
                continue;
            }
            let old = std::mem::replace(&mut row.value, value.clone());
            self.events.push(CvarChanged {
                name: row.name.clone(),
                old,
                new: value.clone(),
            });
        }
        self.file = file;
    }

    /// The file's entries no row claims: a newer build's keys, or an addon's not yet registered.
    /// Handed to the VM as its saved base, so an addon's `RegisterCVar` starts at the saved value.
    pub(crate) fn orphans(&self) -> Vec<(String, String)> {
        self.file
            .iter()
            .filter(|(k, _)| !self.index.contains_key(&k.to_ascii_lowercase()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// The whole table as a VM's mirror is seeded from it.
    pub(crate) fn vm_seed(&self) -> Vec<SeededCvar> {
        self.rows
            .iter()
            .map(|r| SeededCvar {
                name: r.name.clone(),
                value: r.value.clone(),
                default: r.default.clone(),
                latched: r.latched,
            })
            .collect()
    }

    /// Read before taking, so a quiet frame never deref-muts the registry.
    pub(crate) fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    /// The accepted moves since the last flush, for the caller to trigger.
    pub(crate) fn take_events(&mut self) -> Vec<CvarChanged> {
        std::mem::take(&mut self.events)
    }

    fn has_outbox(&self) -> bool {
        !self.outbox.is_empty()
    }

    fn take_outbox(&mut self) -> Vec<(String, String)> {
        std::mem::take(&mut self.outbox)
    }

    /// The file to save: the previous file as the merge base, each row off its default at its
    /// applied value (a staged one is not the player's yet), each at its default removed;
    /// session-owned and unknown keys untouched.
    fn compose(&self) -> BTreeMap<String, String> {
        let mut out = self.file.clone();
        for row in &self.rows {
            let key = row.name.to_ascii_lowercase();
            if self.session_owned.contains(&key) {
                continue;
            }
            // Match any existing entry case-insensitively so a hand-edited spelling doesn't fork.
            let existing = out
                .keys()
                .find(|k| k.eq_ignore_ascii_case(&row.name))
                .cloned();
            if row.value == row.default {
                if let Some(k) = existing {
                    out.remove(&k);
                }
            } else {
                out.insert(
                    existing.unwrap_or_else(|| row.name.clone()),
                    row.value.clone(),
                );
            }
        }
        out
    }

    /// A registry already holding one stored value, as if `config.toml` said so.
    #[cfg(test)]
    pub(crate) fn with_value(name: &str, value: &str) -> Self {
        let mut cvars = Self::default();
        cvars.load_file(BTreeMap::from([(name.to_string(), value.to_string())]));
        cvars.events.clear();
        cvars
    }
}

/// How long a dirty config waits before saving: long enough to coalesce a slider drag.
const SAVE_QUIET: std::time::Duration = std::time::Duration::from_secs(1);

/// The startup fold of `config.toml` into the registry and the knobs ([`load_config`]). A set
/// because the world camera reads `gxMultisample` once at spawn, so `setup_player` must order after
/// it; without the constraint that order is the executor's choice.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CvarLoad;

pub(crate) struct CvarPlugin;

impl Plugin for CvarPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Cvars>()
            .add_systems(
                Startup,
                (load_config, publish_filter_policy)
                    .chain()
                    .in_set(CvarLoad),
            )
            // After the tick, so a `SetCVar` from this frame reaches the registry and its observers
            // before the frame's drains; `video::drain_restart_gx` orders after this so the commit
            // finds the stage.
            .add_systems(
                Update,
                (sync_cvars, save_config)
                    .chain()
                    .after(crate::ui_script::UiInput),
            );
        // The save runs on the exit edge and beside its feed. The close button's `AppExit` is
        // written in `PostUpdate`, so the `Last` instance below catches the exit frame; a closed
        // browser tab has no exit frame at all, so the page keeps its settings through this
        // quiet-second save. Whichever runs first clears `dirty`.
        crate::shutdown::on_app_exit(app, save_config.into_configs());
    }
}

/// What the environment took for this session and its value, read off the knobs the env levers
/// seeded (`RenderScale::default()` reads `$WOW_RENDER_SCALE`, …). A lever whose resource is absent
/// still marks its row, so the file cannot apply over the env.
fn session_values(world: &World) -> Vec<(&'static str, Option<String>)> {
    // `webenv::var`, not `std::env`: natively the same read, and in a browser tab the page's own
    // env object and query string — so `?farclip=` is session-only exactly as `$WOW_FARCLIP` is.
    let set = |k: &str| crate::webenv::var(k).is_some();
    let flag = |b: bool| if b { "1" } else { "0" }.to_string();
    let mut out: Vec<(&'static str, Option<String>)> = Vec::new();
    if set("WOW_UI_SCALE") {
        let v = world.get_resource::<crate::ui_script::UiScaleCvar>();
        out.push(("uiScale", v.map(|s| s.0.to_string())));
    }
    if set("WOW_FARCLIP") {
        let v = world.get_resource::<benilla_world::view::ViewDistance>();
        out.push(("farclip", v.map(|s| s.farclip.to_string())));
    }
    // The clutter lever takes both spellings of its knob; an off-grid multiplier seeds off-grid.
    if set("WOW_CLUTTER_DENSITY") {
        let v = world.get_resource::<benilla_world::clutter::ClutterConfig>();
        out.push(("WorldDetail", v.map(|c| (c.density - 1.0).to_string())));
        out.push(("frillDensity", v.map(|c| c.frill_density().to_string())));
    }
    if crate::video::novsync_env() {
        let v = world.get_resource::<crate::video::VideoConfig>();
        out.push(("gxVSync", v.map(|c| flag(c.vsync))));
    }
    let tex = world.get_resource::<benilla_assets::TexFilterSetting>();
    if set("WOW_TRILINEAR") {
        out.push(("trilinear", tex.map(|t| flag(t.trilinear))));
    }
    if set("WOW_ANISO") {
        out.push(("anisotropic", tex.map(|t| t.aniso.to_string())));
    }
    // `$WOW_WIN`, a capture scenario or an instrumented run owns the window geometry.
    if crate::video::windowed_env() {
        let v = world.get_resource::<crate::video::VideoConfig>();
        out.push((
            "gxWindow",
            v.map(|c| flag(c.display == crate::video::DisplayMode::Windowed)),
        ));
        out.push((
            "gxResolution",
            v.map(|c| format!("{}x{}", c.windowed.x, c.windowed.y)),
        ));
    }
    // The multisampling and render-scale levers: an instrument run must not pin 4× into the file.
    if set("WOW_MSAA") {
        let v = world.get_resource::<benilla_world::view::MsaaSetting>();
        out.push(("gxMultisample", v.map(|m| m.samples.to_string())));
    }
    if set("WOW_RENDER_SCALE") {
        let v = world.get_resource::<crate::world_backdrop::RenderScale>();
        out.push(("renderScale", v.map(|r| r.0.to_string())));
    }
    // `$WOW_SSAO` / `$WOW_SSGI` are the indirect-lighting A/B levers, and doubly session-only:
    // they are how `benilla-worldview` (which has no CVar host) drives the pass at all, so a value
    // that stuck in the file would silently denominate every later look at the same scene.
    let indirect = world.get_resource::<benilla_world::ssilvb::IndirectLighting>();
    if set("WOW_SSAO") {
        out.push(("ssaoStrength", indirect.map(|i| i.ao.to_string())));
    }
    if set("WOW_SSGI") {
        out.push(("ssgiStrength", indirect.map(|i| i.gi.to_string())));
    }
    // fork: modern lighting — `$WOW_LIGHTING` / `$WOW_LIGHTING_QUALITY`, on the same law.
    // `LightingSettings::default()` has already taken them.
    let lighting = world.get_resource::<LightingSettings>();
    if set("WOW_LIGHTING") {
        out.push((
            "lightingModel",
            lighting.map(|l| l.model.as_cvar().to_string()),
        ));
    }
    if set("WOW_LIGHTING_QUALITY") {
        out.push((
            "lightingQuality",
            lighting.map(|l| l.quality.as_cvar().to_string()),
        ));
    }
    // `$WOW_WATER_DETAIL` is the liquid-renderer A/B lever: every side-by-side of the two water
    // paths sets it, and a value that stuck would leave the client on whichever side the last
    // comparison wanted.
    if set("WOW_WATER_DETAIL") {
        let v = world.get_resource::<WaterDetail>();
        out.push(("waterDetail", v.map(|w| w.0.to_string())));
    }
    // `$WOW_HD_MODELS` is the content-layer lever — how the model test suite and
    // `benilla-worldview` reach the layer. `apply_boot_layers` has already taken it; a value that
    // stuck in the file would leave a player mounting 1.7 GiB of art because a test run wanted it.
    if set("WOW_HD_MODELS") {
        out.push(("hdCharacterModels", None));
    }
    // `$WOW_HOST` is the session's realmlist, which a test run must never write into the file.
    if set("WOW_HOST") {
        let v = world.get_resource::<crate::realmlist::Realmlist>();
        out.push((
            crate::realmlist::CVAR_REALMLIST,
            v.map(|r| r.address().to_string()),
        ));
    }
    out
}

/// Startup: mark what the session owns, fold `config.toml` in (no file means all defaults) and fire
/// the observers now, so everything after [`CvarLoad`] finds its knob written. The VM does not
/// exist yet; [`sync_cvars`] seeds its mirror.
fn load_config(world: &mut World) {
    let session = session_values(world);
    let stored = stored_config();
    let events = {
        let mut cvars = world.resource_mut::<Cvars>();
        for (name, value) in session {
            cvars.own_for_session(name, value.as_deref());
        }
        // `gxApi` is the render backend, a fact about the machine that `sync_cvars` pushes live, so
        // it is never persisted.
        cvars.own_for_session("gxApi", None);
        // The hosted page's realmlist, applied BEFORE the file is read: it marks `realmlist`
        // session-owned, and `load_file` skips every session-owned key. That ordering is the
        // whole of "`/config.json` is the only source" — see the function.
        apply_hosted_realmlist(&mut cvars);
        match stored {
            StoredConfig::Absent => {} // no file, hermetic capture, or no install
            StoredConfig::Bad(msg) => {
                // A malformed file is kept: nothing loads, and nothing saves over it until a
                // change.
                warn!("{msg}");
            }
            StoredConfig::Table(table) => cvars.load_file(table),
        }
        apply_query_overrides(&mut cvars);
        apply_gpu_tier(&mut cvars);
        cvars.take_events()
    };
    for event in events {
        world.trigger(event);
    }
}

/// `/config.json`'s `realmlist` — on a hosted page, **the only source the address comes from**.
///
/// It used to be the bottom of the precedence stack: a default, applied only when `config.toml`
/// carried no `realmList` of its own, because the login screen had a control a visitor could type
/// their own realm into. That control is gone (see `login/screen.rs`), and with it the reason for
/// the deference — a saved value can now only be one an *older* build wrote, and letting it win
/// would mean an operator re-uploading `config.json` silently fails to move exactly the returning
/// visitors, which is the whole population that matters.
///
/// So it registers as **session-overridden**, the same standing `$WOW_HOST` and every `$WOW_*`
/// A/B lever have: the file's value is not applied over it, and nothing writes it back. Two things
/// still outrank it, both deliberately — `$WOW_HOST`/`?host=` (the harness pin every probe and
/// smoke run sets, which `Realmlist` refuses to be moved off) and a Lua `SetCVar("realmList")`,
/// which is a live session act and not a stored one.
///
/// Native is untouched: [`crate::webenv::config`] is `None` off the page, so `config.toml` remains
/// the realmlist's home there — the seam the reference itself edits through `realmlist.wtf`.
fn apply_hosted_realmlist(cvars: &mut Cvars) {
    if cvars.is_session_owned(crate::realmlist::CVAR_REALMLIST) {
        return; // `$WOW_HOST` owns the session.
    }
    let Some(address) = crate::webenv::config("realmlist") else {
        return;
    };
    // Marked whatever the value, and deliberately: a `config.json` that names an unusable address
    // has still spoken for this key, and falling back to a saved `config.toml` one would be the
    // two-source confusion this exists to end.
    cvars.override_for_session(crate::realmlist::CVAR_REALMLIST, &address);
    info!("config: realmList = {address} from the hosted config.json");
}

/// **Browser build only:** any registered CVar named in the page's query string
/// (`?farclip=150&worlddetail=0&renderScale=0.75`) overrides the stored value for this session —
/// the web twin of the `$WOW_*` A/B levers, and the only way to hand a slow device a setting from
/// a link. Session-owned for the same reason, so the file keeps the player's own value.
#[cfg(target_arch = "wasm32")]
fn apply_query_overrides(cvars: &mut Cvars) {
    for (name, _) in registered_pairs() {
        if cvars.is_session_owned(name) {
            continue; // a lever already took it, at the value its knob read
        }
        // `webenv::var` strips `WOW_` and lowercases, so the query key is the CVar's own name.
        let Some(value) = crate::webenv::var(&format!("WOW_{name}")) else {
            continue;
        };
        cvars.override_for_session(name, &value);
        info!("config: {name} = {value} from the page URL (this session only)");
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_query_overrides(_cvars: &mut Cvars) {}

/// fork: web build — what `web/gpu.js` steps down to after the GPU device was lost. The render
/// scale (`__wenilla_env.gpu_render_scale`) is a fallback, so a render scale the player saved
/// wins; the lighting quality (`gpu_lighting_quality`) is a ceiling, so it holds a saved Cinematic
/// down too. Both are this session's only, either is the player's again from their own write to
/// it, and any Graphics Apply retires both ([`release_gpu_tier`]).
fn apply_gpu_tier(cvars: &mut Cvars) {
    if let Some(value) = crate::webenv::var("WOW_GPU_RENDER_SCALE") {
        if cvars.fallback_for_session("renderScale", &value) {
            info!("config: renderScale = {value} from the GPU recovery tier (until you set one)");
        }
    }
    if let Some(value) = crate::webenv::var("WOW_GPU_LIGHTING_QUALITY") {
        if cvars.cap_for_session("lightingQuality", &value) {
            info!("config: lightingQuality capped at {value} by the GPU recovery tier");
        }
    }
}

/// fork: web build — the CVars the Graphics page writes: its CVar rows, and `gamma` behind
/// Brightness's `SetGamma`. A player's write to any of them is "applied new graphics settings",
/// which is what retires `web/gpu.js`'s recovery tier.
pub(crate) const GRAPHICS_CVARS: &[&str] = &[
    "uiScale",
    "farclip",
    "WorldDetail",
    "gamma",
    "weatherDensity",
    "gxVSync",
    "worldShadows",
    "hdCharacterModels",
    "ssaoStrength",
    "ssgiStrength",
    "waterDetail",
    "lightingModel",
    "lightingQuality",
    "renderScale",
];

/// fork: web build — the player applied Graphics settings of their own, so the GPU recovery tier
/// is over: the page persists tier 0 and lifts its pixel budget now ([`crate::webgpu`]), and a
/// render scale the tier still holds returns to the player's default. Without it a single device
/// loss held the lower tier until it decayed, storage was cleared, or `?gputier=0`.
fn release_gpu_tier(cvars: &mut Cvars) {
    let released = cvars.release_fallbacks();
    if crate::webgpu::settings_applied() || released > 0 {
        info!("config: Graphics settings applied — GPU recovery tier cleared");
    }
}

/// What the one read of `config.toml` found.
enum StoredConfig {
    /// No file, no install, or a hermetic capture: every value is its registered default.
    Absent,
    Table(BTreeMap<String, String>),
    /// Unreadable or malformed; the message is carried because this read runs before `LogPlugin`
    /// exists.
    Bad(String),
}

/// Read `config.toml`, for [`load_config`] and for the primary window in [`crate::run`], which
/// needs `gxWindow`/`gxResolution` before it exists. Not cached: a process-wide cache would answer
/// every test from the first one's file.
fn stored_config() -> StoredConfig {
    let Some(path) = crate::local_state::config_path() else {
        return StoredConfig::Absent; // hermetic capture or no install: session-only state
    };
    let text = match crate::local_state::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return StoredConfig::Absent,
        Err(e) => return StoredConfig::Bad(format!("config: cannot read {}: {e}", path.display())),
    };
    match toml::from_str::<LocalConfig>(&text) {
        Ok(cfg) => StoredConfig::Table(cfg.cvars),
        Err(e) => StoredConfig::Bad(format!(
            "config: {} is malformed ({e}) — running on defaults",
            path.display()
        )),
    }
}

/// One CVar as `config.toml` holds it, before the `App` exists: for the primary window, which is
/// built with its display mode resolved. Everything else reads [`Cvars::get`].
pub(crate) fn boot_cvar(name: &str) -> Option<String> {
    match stored_config() {
        StoredConfig::Table(t) => t
            .into_iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v),
        StoredConfig::Absent | StoredConfig::Bad(_) => None,
    }
}

/// One CVar's **registered** default, matched case-insensitively — what the registry starts the
/// row at, for the boot-time readers that run before the registry exists.
///
/// [`boot_cvar`] answers from `config.toml` and only from it, so it is `None` for a player who has
/// never changed a setting — which is every player's first launch. A boot reader that treats that
/// `None` as "off" is not reading the default, it is inventing a second one, and the two drift the
/// moment the row is re-registered. [`apply_boot_layers`] is where that bit.
pub(crate) fn registered_default(name: &str) -> Option<&'static str> {
    REGISTERED
        .iter()
        .find(|r| r.name.eq_ignore_ascii_case(name))
        .map(|r| r.default)
}

/// **Which optional content layers this run mounts**, resolved and published to
/// `benilla_formats::layers` before the `App` exists.
///
/// A sibling of [`boot_cvar`]'s single caller and for the same structural reason: the client opens
/// two chains — the Bevy `AssetReader`'s and the world store's — and both are built during
/// plugin setup, long before a resource could carry the answer. A layer is a MOUNT, and a mounted
/// chain has already decided; there is nothing a later read of the CVar could act on.
///
/// Environment first, file second, the row's own registered default if neither says otherwise —
/// the ladder every other `$WOW_*` override in this module follows. **The last rung is
/// [`registered_default`], not a literal**: `boot_cvar` answers only from `config.toml`, so a
/// player who has never touched a setting has no row at all, and a hardcoded `false` there would
/// leave the layer off for exactly the players the default is for.
pub(crate) fn apply_boot_layers() {
    // `webenv::var`, not `std::env` — `$WOW_HD_MODELS` natively, `?hd_models=1` (or
    // `__wenilla_env`) in a browser tab. The parse is `benilla_formats::layers::layer_switch`'s,
    // shared with the page and the tools: three readers of one setting that disagree about what
    // "on" means is a client that mounts what the page did not download.
    let hd = crate::webenv::var("WOW_HD_MODELS")
        .and_then(|v| benilla_formats::layers::layer_switch(&v))
        .unwrap_or_else(|| {
            boot_cvar("hdCharacterModels")
                .or_else(|| registered_default("hdCharacterModels").map(str::to_string))
                .and_then(|v| benilla_formats::layers::layer_switch(&v))
                .unwrap_or(false)
        });
    benilla_formats::set_hd_character_models(hd);
}

/// Per frame: seed a new VM's mirror, drain its registrations and writes into the registry, push
/// the registry's writes into the mirror, and trigger every accepted move.
pub(crate) fn sync_cvars(
    script: Option<NonSendMut<UiScript>>,
    mut cvars: ResMut<Cvars>,
    adapter: Option<Res<bevy::render::renderer::RenderAdapterInfo>>,
    msaa_formats: Option<Res<benilla_world::view::MsaaFormats>>,
    mut seeded: Local<VmMemo<bool>>,
    mut commands: Commands,
) {
    // The live render backend; absent headless, where the registered `""` stands.
    if let Some(adapter) = adapter.as_deref() {
        let backend = adapter.backend.to_str();
        if cvars.get("gxApi") != Some(backend) {
            cvars.own_for_session("gxApi", Some(backend));
        }
    }
    // A Graphics write drained last frame (or folded at a session edge) retires the recovery
    // tier; its render-scale move flushes with this frame's events below.
    if cvars.take_graphics_applied() {
        release_gpu_tier(&mut cvars);
    }
    let Some(mut script) = script else {
        // Nothing to mirror into; a later VM is seeded from the table, which carries it all.
        if cvars.has_outbox() {
            cvars.take_outbox();
        }
        if cvars.has_events() {
            for event in cvars.take_events() {
                commands.trigger(event);
            }
        }
        return;
    };
    // The VM's writes before the seed: an addon's `SetCVar` at load is already queued, and seeding
    // first would overwrite the mirror with the older value. Registrations before writes, since an
    // addon declares a row and sets it together.
    let registrations = script.take_cvar_registrations();
    let changes = script.take_cvar_changes();
    if !registrations.is_empty() || !changes.is_empty() {
        for (name, default) in registrations {
            cvars.learn_addon_row(&name, &default);
        }
        for (name, value) in changes {
            cvars.set_from_vm(&name, &value);
        }
    }
    if seeded.claim(&script) {
        // The file's unclaimed entries first, so an addon's `RegisterCVar` starts at the saved
        // value; then the table.
        script.set_cvar_saved_base(cvars.orphans());
        script.seed_cvars(cvars.vm_seed());
        if cvars.has_outbox() {
            cvars.take_outbox(); // the seed just carried everything
        }
        // The Video dropdown's menu: what this device accepts, enumerated by
        // `view::MsaaSupportPlugin`.
        script.set_multisample_formats(
            msaa_formats
                .as_deref()
                .map(|f| {
                    f.formats
                        .iter()
                        .map(|&(color_bits, depth_bits, samples)| {
                            benilla_ui::script::MultisampleFormat {
                                color_bits,
                                depth_bits,
                                samples,
                            }
                        })
                        .collect()
                })
                .unwrap_or_default(),
        );
        // `GetVideoCaps`, the seven values `OptionsFrame_Load` destructures. Shaders, trilinear,
        // anisotropy and the hardware cursor are always there (`crate::cursor` composites the
        // reference's cursors into an OS cursor); `max_anisotropy` is reported raw because
        // `OptionsFrame.lua:122` matches it against `ANISOTROPIC_VALUES`. Triple buffering is
        // false: wgpu owns the surface's buffering, so the stock frame hides check button 13
        // (`OptionsFrame.lua:163-165`).
        script.set_video_caps(benilla_ui::script::VideoCaps {
            anisotropic: true,
            pixel_shaders: true,
            vertex_shaders: true,
            trilinear: true,
            triple_buffering: false,
            max_anisotropy: *benilla_assets::ANISO_RANGE.end(),
            hardware_cursor: true,
        });
    }
    if cvars.has_outbox() {
        for (name, value) in cvars.take_outbox() {
            script.set_cvar_host(&name, &value);
        }
    }
    if cvars.has_events() {
        for event in cvars.take_events() {
            commands.trigger(event);
        }
    }
}

/// Fold the dying VM's last writes into the registry. Called from
/// [`crate::ui_script::end_ui_session`] after the shutdown events (a `PLAYER_LOGOUT` handler may
/// `SetCVar`, and the reference keeps that write) and before the VM is replaced, so a final-frame
/// write is not lost to the next VM's seed. The observers fire here.
pub(crate) fn fold_dying_vm_cvars(world: &mut World) {
    // No registry: a test or stripped world without the plugin.
    if !world.contains_resource::<Cvars>() {
        return;
    }
    let (registrations, changes) = {
        let Some(mut script) = world.get_non_send_resource_mut::<UiScript>() else {
            return;
        };
        (script.take_cvar_registrations(), script.take_cvar_changes())
    };
    let events = {
        let mut cvars = world.resource_mut::<Cvars>();
        for (name, default) in registrations {
            cvars.learn_addon_row(&name, &default);
        }
        for (name, value) in changes {
            cvars.set_from_vm(&name, &value);
        }
        if cvars.has_outbox() {
            cvars.take_outbox(); // the VM this was for is going away
        }
        if cvars.has_events() {
            cvars.take_events()
        } else {
            Vec::new()
        }
    };
    for event in events {
        world.trigger(event);
    }
}

/// The file's header comment.
const HEADER: &str = "\
# benilla local config — CVar values that moved off their defaults.
# Managed by the client; hand edits are read on next launch and preserved on save.
";

/// Dirty and one quiet second, or the app exiting: rewrite `config.toml` atomically from the
/// registry, so a session with no VM saves what it changed.
fn save_config(mut cvars: ResMut<Cvars>, mut exits: MessageReader<AppExit>) {
    let exiting = exits.read().next().is_some();
    if !cvars.dirty {
        return;
    }
    let quiet = cvars.last_change.is_none_or(|t| t.elapsed() >= SAVE_QUIET);
    if !(quiet || exiting) {
        return;
    }
    let Some(path) = crate::local_state::config_path() else {
        cvars.dirty = false; // hermetic/session-only: nothing to write, stop retrying
        return;
    };
    let file = cvars.compose();
    let body = toml::to_string(&LocalConfig {
        cvars: file.clone(),
    })
    .expect("string map serializes");
    match crate::local_state::write_atomic(&path, &format!("{HEADER}{body}")) {
        Ok(()) => {
            cvars.file = file;
            cvars.dirty = false;
        }
        Err(e) => {
            warn!("config: cannot write {}: {e}", path.display());
            cvars.dirty = false; // don't retry every frame into the same error
        }
    }
}

/// Freeze the texture filter policy for the process and log the mode, so a player's log says which
/// mode the run was in. Its own system after [`load_config`], so it publishes whatever the file
/// held.
fn publish_filter_policy(filter: Res<benilla_assets::TexFilterSetting>) {
    benilla_assets::publish_tex_filter(*filter);
    let mode = filter.mode();
    let name = match mode {
        3 => "bilinear + nearest-mip select, aniso off",
        4 => "trilinear, aniso off",
        _ => "trilinear + aniso",
    };
    info!(
        "texture filter: mode {mode} ({name}) — trilinear={} anisotropic={}",
        u8::from(filter.trilinear),
        filter.aniso
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_bubble::BubbleConfig;
    use crate::minimap::MinimapZoom;
    use crate::nameplates::NameConfig;
    use crate::player::camera::{
        FollowConfig, FollowStyle, LookConfig, ZoomLimit, FOLLOW_SPEED_RANGE,
    };
    use crate::portrait::PaneRate;
    use crate::sound::SoundConfig;
    use crate::target::ClickConfig;
    use crate::ui_loot::LootConfig;
    use crate::ui_script::{UiScaleCvar, DEFAULT_UI_SCALE};
    use crate::video::VideoConfig;
    use crate::vplates::VPlateMode;
    use crate::world_backdrop::{RenderScale, RENDER_SCALE_RANGE};
    use benilla_ui::widget::MINIMAP_ZOOM_LEVELS;
    use benilla_world::clutter::ClutterConfig;
    use benilla_world::modern::{LightingModel, LightingQuality};
    use benilla_world::ssilvb::{IndirectLighting, SSILVB_RANGE};
    use benilla_world::view::{MsaaSetting, ViewDistance, FARCLIP_RANGE, MSAA_RANGE};
    // The engine's shipped indirect-lighting numbers. Named only here: the registered strings are
    // welded to them, and the fixtures below seed the resource with them rather than with
    // `IndirectLighting::default()`, which reads $WOW_SSAO / $WOW_SSGI.
    use benilla_world::ssilvb::{AO_DEFAULT, GI_DEFAULT};

    /// Every row's [`Reference`] column holds, both ways: a `Same` row that drifted fails, and so
    /// does a `Deviates` or `Overridden` row back in agreement. Numbers parse-compare, so "1" and
    /// "1.0" agree.
    #[test]
    fn defaults_stand_where_the_reference_column_says() {
        /// Numeric when both parse, textual otherwise.
        fn agrees(ours: &str, theirs: &str) -> bool {
            match (ours.parse::<f32>(), theirs.parse::<f32>()) {
                (Ok(a), Ok(b)) => a == b,
                _ => ours == theirs,
            }
        }
        for row in REGISTERED {
            let name = row.name;
            match &row.reference {
                Reference::Same(value) => assert!(
                    agrees(row.default, value),
                    "{name}: the row claims the reference registers {value:?} and we ship the \
                     same, but our default is {:?}. If the reference really does differ, this is \
                     a `deviates` row and owes a reason.",
                    row.default,
                ),
                Reference::Deviates { value, why } => {
                    assert!(
                        !agrees(row.default, value),
                        "{name}: a `deviates` row that no longer deviates — our {:?} IS the \
                         reference's. Demote it to `same`; a stale deviation hides that we are \
                         faithful again.",
                        row.default,
                    );
                    assert!(!why.trim().is_empty(), "{name}: a deviation owes a reason");
                }
                Reference::Overridden { registered, why } => {
                    assert!(
                        !agrees(row.default, registered),
                        "{name}: an `overridden` row whose default is just the registered string \
                         {registered:?} — that is `same`, and saying otherwise buries a real \
                         override behind a false one.",
                    );
                    assert!(
                        !why.trim().is_empty(),
                        "{name}: an override owes its mechanism"
                    );
                }
                Reference::Ours(why) => assert!(
                    !why.trim().is_empty(),
                    "{name}: a CVar the reference does not have owes the reason it exists",
                ),
            }
        }
    }

    #[test]
    fn the_options_that_leave_the_reference_are_this_list_and_no_other() {
        let mut names: Vec<&str> = REGISTERED
            .iter()
            .filter(|r| matches!(r.reference, Reference::Deviates { .. }))
            .map(|r| r.name)
            .collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "SoundReverb",
                // 2026-09-08: the Environment Detail row starts at High where the reference's own
                // slider reads Medium. A look call, argued at the row; it is here because a stop a
                // player did not choose is exactly the kind of difference this list exists to keep
                // visible.
                "WorldDetail",
                "autoSelfCast",
                "frillDensity",
                "gxApi",
                "gxColorBits",
                "gxDepthBits",
                "gxResolution",
                "realmList",
                "weatherDensity",
            ],
        );
    }

    /// Every numeric registered default equals the code constant it mirrors; the string rows are
    /// checked in [`the_string_valued_cvars_are_the_realm_and_the_windowed_size`].
    #[test]
    fn registered_defaults_mirror_the_code_truths() {
        let d: BTreeMap<&str, f32> = REGISTERED
            .iter()
            .filter_map(|r| r.default.parse::<f32>().ok().map(|f| (r.name, f)))
            .collect();
        let sound = SoundConfig::default();
        assert_eq!(d["MasterVolume"], sound.master);
        assert_eq!(d["SoundVolume"], sound.sfx);
        assert_eq!(d["MusicVolume"], sound.music);
        assert_eq!(d["AmbienceVolume"], sound.ambience);
        assert_eq!(d["MasterSoundEffects"] != 0.0, sound.enabled);
        assert_eq!(d["EnableMusic"] != 0.0, sound.music_enabled);
        assert_eq!(d["EnableAmbience"] != 0.0, sound.ambience_enabled);
        assert_eq!(d["EnableErrorSpeech"] != 0.0, sound.error_speech);
        assert_eq!(
            d["Sound_EnableSoundWhenGameIsInBG"] != 0.0,
            sound.background_sound
        );
        assert!(
            !sound.background_sound,
            "the reference goes quiet in the background and offers no way out"
        );
        assert_eq!(d["SoundReverb"] != 0.0, sound.reverb);
        assert_eq!(d["SoundOutputLimiter"] != 0.0, sound.limiter);
        assert!(sound.limiter, "the output limiter ships on");
        assert!(!sound.reverb, "zone reverb ships off");
        assert_eq!(d["uiScale"], DEFAULT_UI_SCALE);
        // `ViewDistance::default()` reads `$WOW_FARCLIP`, so this is the env-less literal (the
        // fork's 1000 yd horizon; the reference registers 350).
        assert_eq!(d["farclip"], 1000.0);
        assert_eq!(d["nearclip"], benilla_world::view::NEARCLIP_DEFAULT);
        assert_eq!(
            d["nearclip"],
            ViewDistance::default().nearclip,
            "the registered default and the resource's own must be one number"
        );
        // `MsaaSetting::default()` reads `$WOW_MSAA`, so this is the env-less literal.
        assert_eq!(d["gxMultisample"], 1.0);
        assert_eq!(
            d["deselectOnClick"] != 0.0,
            ClickConfig::default().deselect_on_click
        );
        assert_eq!(
            d["mouseInvertPitch"] != 0.0,
            LookConfig::default().invert_pitch
        );
        assert_eq!(d["mousespeed"], LookConfig::default().sensitivity);
        assert_eq!(d["cameraDistanceMaxFactor"], ZoomLimit::default().factor());
        let follow = FollowConfig::default();
        assert_eq!(
            d["cameraSmoothStyle"],
            follow.style.cvar().parse::<f32>().unwrap()
        );
        assert_eq!(
            d["cameraSmoothTrackingStyle"],
            follow.tracking_style.cvar().parse::<f32>().unwrap()
        );
        assert_eq!(d["cameraYawSmoothSpeed"], follow.yaw_speed);
        assert_eq!(FollowStyle::default(), FollowStyle::Smart);
        assert_eq!(d["autoLootDefault"] != 0.0, LootConfig::default().auto_loot);
        assert_eq!(
            d["showLootSpam"] != 0.0,
            LootConfig::default().show_loot_spam
        );
        assert_eq!(
            d["guildMemberNotify"] != 0.0,
            crate::ui_guild::GuildMemberNotify::default().0
        );
        assert_eq!(d["guildMemberNotify"], 0.0, "the binary registers \"0\"");
        assert_eq!(
            d["BlockTrades"] != 0.0,
            crate::ui_trade::BlockTrades::default().0
        );
        assert_eq!(d["BlockTrades"], 0.0, "an unset BlockTrades allows trades");
        let names = NameConfig::default();
        assert_eq!(d["UnitNamePlayer"] != 0.0, names.player);
        assert_eq!(d["UnitNameNPC"] != 0.0, names.npc);
        assert_eq!(d["UnitNameOwn"] != 0.0, names.own);
        assert_eq!(d["UnitNamePlayerGuild"] != 0.0, names.player_guild);
        assert!(
            names.player && !names.npc && !names.own && names.player_guild,
            "the binary registers UnitNamePlayer \"1\", NPC \"0\", Own \"0\", \
             PlayerGuild \"1\""
        );
        let camera_opts = crate::player::camera_dynamics::CameraOptions::default();
        assert_eq!(d["cameraPivot"] != 0.0, camera_opts.pivot);
        assert!(camera_opts.pivot, "the binary registers cameraPivot \"1\"");
        assert_eq!(
            d["cameraWaterCollision"] != 0.0,
            camera_opts.water_collision
        );
        assert!(
            camera_opts.pivot && camera_opts.water_collision,
            "the binary registers cameraPivot and cameraWaterCollision both \"1\""
        );
        assert_eq!(d["cameraTerrainTilt"] != 0.0, camera_opts.terrain_tilt);
        assert!(
            !camera_opts.terrain_tilt,
            "the binary registers cameraTerrainTilt \"0\""
        );
        assert_eq!(
            d["cameraGroundSmoothSpeed"],
            camera_opts.ground_smooth_speed
        );
        assert_eq!(d["cameraTerrainTiltTimeMin"], camera_opts.tilt_time_min);
        assert_eq!(d["cameraTerrainTiltTimeMax"], camera_opts.tilt_time_max);
        assert_eq!(d["cameraBobbing"] != 0.0, camera_opts.bobbing);
        assert!(
            !camera_opts.bobbing && !camera_opts.terrain_tilt,
            "the binary registers cameraBobbing and cameraTerrainTilt both \"0\""
        );
        assert_eq!(d["cameraBobbingLRAmplitude"], camera_opts.bob_lr_amplitude);
        assert_eq!(d["cameraBobbingUDAmplitude"], camera_opts.bob_ud_amplitude);
        assert_eq!(d["cameraBobbingFrequency"], camera_opts.bob_frequency);
        assert_eq!(d["cameraBobbingSmoothSpeed"], camera_opts.bob_smooth_speed);
        assert_eq!(d["cameraPivotDXMax"], camera_opts.pivot_dx_max);
        assert_eq!(d["cameraPivotDYMin"], camera_opts.pivot_dy_min);
        assert_eq!(
            d["cameraTargetSmoothSpeed"],
            camera_opts.target_smooth_speed
        );
        assert_eq!(
            d["weatherDensity"],
            f32::from(benilla_world::weather::WeatherState::default().weather_density)
        );
        let plates = VPlateMode::default();
        assert_eq!(d[crate::vplates::CVAR_ENEMIES] != 0.0, plates.enemies);
        assert_eq!(d[crate::vplates::CVAR_FRIENDS] != 0.0, plates.friends);
        assert!(
            !plates.enemies && !plates.friends,
            "a fresh 1.12 client draws no plates until V is pressed"
        );
        // `ClutterConfig::default()` reads `$WOW_CLUTTER_DENSITY`; stop 1 is its env-less ×2.
        assert_eq!(d["WorldDetail"], 2.0);
        assert_eq!(
            d["frillDensity"],
            (d["WorldDetail"] + 1.0) * benilla_formats::FRILL_DENSITY as f32
        );
        let bubbles = BubbleConfig::default();
        assert_eq!(d["ChatBubbles"] != 0.0, bubbles.all);
        assert_eq!(d["ChatBubblesParty"] != 0.0, bubbles.party);
        assert!(bubbles.all && !bubbles.party, "the binary's own pair");
        let zoom = MinimapZoom::default();
        assert_eq!(d["minimapZoom"], f32::from(zoom.outdoor));
        assert_eq!(d["minimapInsideZoom"], f32::from(zoom.inside));
        assert_eq!(zoom.outdoor, benilla_ui::widget::MINIMAP_DEFAULT_ZOOM);
        // `video::tests` welds `VideoConfig::vsync` to the window's boot present mode.
        assert_eq!(d["gxVSync"] != 0.0, VideoConfig::default().vsync);
        assert_eq!(
            d["worldShadows"] != 0.0,
            VideoConfig::default().world_shadows
        );
        assert_eq!(d["boothHalfRate"] != 0.0, PaneRate::default().half);
        // Every visual golden assumes a 1:1 backdrop.
        assert_eq!(d["renderScale"], 1.0);
        // The indirect-lighting pair welds to the ENGINE's shipped numbers, not to a literal
        // repeated here: `benilla_world::ssilvb` owns what the pass ships at. (Not
        // `IndirectLighting::default()`, which reads $WOW_SSAO / $WOW_SSGI.) The water tier welds
        // to the engine's tier constant the same way.
        assert_eq!(
            d["waterDetail"],
            f32::from(benilla_world::liquid::WATER_DETAIL_EXPERIMENTAL)
        );
        assert_eq!(d["ssaoStrength"], AO_DEFAULT);
        assert_eq!(d["ssgiStrength"], GI_DEFAULT);
        // The lighting pair welds to the engine's shipped model and tier: Modern at Cinematic.
        assert_eq!(
            d["lightingModel"],
            f32::from(LightingModel::default().as_cvar())
        );
        assert_eq!(
            d["lightingQuality"],
            f32::from(LightingQuality::DEFAULT.as_cvar())
        );
    }

    #[test]
    fn the_observers_apply_every_arm() {
        let mut app = cvar_app();
        apply(&mut app, "MusicVolume", "0.7");
        assert_eq!(res::<SoundConfig>(&app).music, 0.7);
        // A string row reaches its knob; a value that is not an address leaves the knob alone.
        apply(&mut app, "realmList", "logon.example.org:3724");
        assert_eq!(
            res::<crate::realmlist::Realmlist>(&app).address(),
            "logon.example.org:3724"
        );
        apply(
            &mut app,
            "realmlist",
            r#"SET realmlist "elsewhere.example.org""#,
        );
        assert_eq!(
            res::<crate::realmlist::Realmlist>(&app).address(),
            "elsewhere.example.org"
        );
        apply(&mut app, "realmList", "not an address");
        assert_eq!(
            res::<crate::realmlist::Realmlist>(&app).address(),
            "elsewhere.example.org",
            "a known key with a bad value is consumed, and the resource keeps its truth",
        );
        // Clamps are the knob's own: volume to [0,1], farclip to FARCLIP_RANGE.
        apply(&mut app, "mastervolume", "7");
        assert_eq!(res::<SoundConfig>(&app).master, 1.0);
        apply(&mut app, "farclip", "50");
        assert_eq!(res::<ViewDistance>(&app).farclip, *FARCLIP_RANGE.start());
        // `nearclip` clamps to the callback's `[0.01, 0.33]` (`0x688d90`).
        apply(&mut app, "nearclip", "0.001");
        assert_eq!(
            res::<ViewDistance>(&app).nearclip,
            0.01,
            "[0x8029d0], the callback's low bound"
        );
        apply(&mut app, "nearclip", "9");
        assert_eq!(
            res::<ViewDistance>(&app).nearclip,
            0.33,
            "[0x808300], its high bound"
        );
        apply(&mut app, "nearclip", "0.3");
        assert_eq!(res::<ViewDistance>(&app).nearclip, 0.3);
        // A sample count, clamped to the reference's [1, 16]; 1 is none.
        apply(&mut app, "gxMultisample", "4");
        assert_eq!(res::<MsaaSetting>(&app).samples, 4);
        apply(&mut app, "anisotropic", "99");
        assert_eq!(
            res::<benilla_assets::TexFilterSetting>(&app).aniso,
            *benilla_assets::ANISO_RANGE.end()
        );
        apply(&mut app, "anisotropic", "0");
        assert_eq!(
            res::<benilla_assets::TexFilterSetting>(&app).aniso,
            *benilla_assets::ANISO_RANGE.start()
        );
        // The knob ships on, so only the flip to 0 proves the arm.
        apply(&mut app, "trilinear", "0");
        assert!(!res::<benilla_assets::TexFilterSetting>(&app).trilinear);
        apply(&mut app, "trilinear", "1");
        assert!(res::<benilla_assets::TexFilterSetting>(&app).trilinear);
        // Clamped to the device's ceiling too: this GPU offers 4, and wgpu refuses a count it
        // lacks.
        apply(&mut app, "gxmultisample", "99");
        assert_eq!(res::<MsaaSetting>(&app).samples, 4);
        // The realistic route in: a config written where 8x exists, opened where it does not.
        apply(&mut app, "gxMultisample", "8");
        assert_eq!(
            res::<MsaaSetting>(&app).samples,
            4,
            "a device that stops at 4x must never be handed an 8"
        );
        apply(&mut app, "gxmultisample", "2");
        assert_eq!(res::<MsaaSetting>(&app).samples, 2);
        apply(&mut app, "gxmultisample", "0");
        assert_eq!(res::<MsaaSetting>(&app).samples, *MSAA_RANGE.start());
        apply(&mut app, "renderScale", "0.75");
        assert_eq!(res::<RenderScale>(&app).0, 0.75);
        apply(&mut app, "renderscale", "9");
        assert_eq!(res::<RenderScale>(&app).0, *RENDER_SCALE_RANGE.end());
        apply(&mut app, "renderscale", "0");
        assert_eq!(res::<RenderScale>(&app).0, *RENDER_SCALE_RANGE.start());
        // The indirect-lighting pair: independent knobs, each clamped at both ends, and each
        // reaching exactly ZERO — the off state the render node checks for and the shader's own
        // no-op arm. A clamp that floored at some epsilon would make "0" not mean off.
        apply(&mut app, "ssaoStrength", "0.25");
        assert_eq!(res::<IndirectLighting>(&app).ao, 0.25);
        assert_eq!(
            res::<IndirectLighting>(&app).gi,
            GI_DEFAULT,
            "the two halves are independent"
        );
        apply(&mut app, "ssaostrength", "9");
        assert_eq!(res::<IndirectLighting>(&app).ao, *SSILVB_RANGE.end());
        apply(&mut app, "ssaostrength", "-1");
        assert_eq!(res::<IndirectLighting>(&app).ao, *SSILVB_RANGE.start());
        // `waterDetail` MAPS rather than clamps: the value space is a set of tiers, and everything
        // that is not the 1.12.1 renderer resolves to cata. `1` — the reference's medium tier —
        // is the fork's SSR-only cata tier: the same renderer without the planar mirror render.
        apply(&mut app, "waterDetail", "0");
        assert!(!res::<WaterDetail>(&app).is_cata());
        apply(&mut app, "waterDetail", "1");
        assert!(
            res::<WaterDetail>(&app).is_cata(),
            "the medium tier must resolve to a renderer that exists"
        );
        assert!(!res::<WaterDetail>(&app).has_planar_reflection());
        apply(&mut app, "waterdetail", "2");
        assert!(res::<WaterDetail>(&app).has_planar_reflection());
        assert!(!res::<WaterDetail>(&app).is_experimental());
        // `3` is the fork's experimental modern combine: the full tier, mirror and all. Only a 3
        // asks for it — an out-of-range value still reads as plain cata.
        apply(&mut app, "waterDetail", "3");
        assert!(res::<WaterDetail>(&app).is_experimental());
        assert!(res::<WaterDetail>(&app).has_planar_reflection());
        apply(&mut app, "waterDetail", "7");
        assert!(res::<WaterDetail>(&app).is_cata());
        assert!(!res::<WaterDetail>(&app).is_experimental());
        apply(&mut app, "waterDetail", "-3");
        assert!(!res::<WaterDetail>(&app).is_cata());
        // The fork's shadow switch and lighting pair.
        apply(&mut app, "worldShadows", "0");
        assert!(!res::<VideoConfig>(&app).world_shadows);
        apply(&mut app, "lightingModel", "0");
        assert_eq!(res::<LightingSettings>(&app).model, LightingModel::Classic);
        apply(&mut app, "lightingModel", "1");
        assert_eq!(
            res::<LightingSettings>(&app).model,
            LightingModel::from_cvar(1.0)
        );
        apply(&mut app, "lightingQuality", "3");
        assert_eq!(
            res::<LightingSettings>(&app).quality,
            LightingQuality::from_cvar(3.0)
        );
        assert!(!res::<crate::perf::FpsJournalSetting>(&app).0);
        apply(&mut app, "fpsJournal", "1");
        assert!(res::<crate::perf::FpsJournalSetting>(&app).0);
        apply(&mut app, "fpsjournal", "0");
        assert!(!res::<crate::perf::FpsJournalSetting>(&app).0);
        // Enable flags: any nonzero is on, zero is off (the client's int-parse + != 0).
        apply(&mut app, "EnableMusic", "0");
        assert!(!res::<SoundConfig>(&app).music_enabled);
        apply(&mut app, "mastersoundeffects", "1");
        assert!(res::<SoundConfig>(&app).enabled);
        apply(&mut app, "deselectonclick", "0");
        assert!(!res::<ClickConfig>(&app).deselect_on_click);
        apply(&mut app, "MouseInvertPitch", "1");
        assert!(res::<LookConfig>(&app).invert_pitch);
        apply(&mut app, "mousespeed", "1.4");
        assert_eq!(res::<LookConfig>(&app).sensitivity, 1.4);
        apply(&mut app, "mousespeed", "9");
        assert_eq!(res::<LookConfig>(&app).sensitivity, 1.5);
        // The engine's enum (0 Never, 1 Smart, 2 Always); 3, the validator's upper bound, is Never.
        apply(&mut app, "cameraSmoothStyle", "0");
        assert_eq!(res::<FollowConfig>(&app).style, FollowStyle::Never);
        apply(&mut app, "camerasmoothstyle", "2");
        assert_eq!(res::<FollowConfig>(&app).style, FollowStyle::Always);
        apply(&mut app, "cameraSmoothStyle", "3");
        assert_eq!(res::<FollowConfig>(&app).style, FollowStyle::Never);
        apply(&mut app, "cameraSmoothStyle", "1");
        assert_eq!(res::<FollowConfig>(&app).style, FollowStyle::Smart);
        apply(&mut app, "cameraSmoothTrackingStyle", "2");
        assert_eq!(
            res::<FollowConfig>(&app).tracking_style,
            FollowStyle::Always
        );
        assert_eq!(
            res::<FollowConfig>(&app).style,
            FollowStyle::Smart,
            "and only that one"
        );
        apply(&mut app, "cameraYawSmoothSpeed", "270");
        assert_eq!(res::<FollowConfig>(&app).yaw_speed, 270.0);
        apply(&mut app, "cameraYawSmoothSpeed", "9000");
        assert_eq!(
            res::<FollowConfig>(&app).yaw_speed,
            *FOLLOW_SPEED_RANGE.end()
        );
        // The max-orbit factor lands as YARDS on the knob (base 15 x factor), clamped to 1..2.
        apply(&mut app, "cameraDistanceMaxFactor", "1");
        assert_eq!(res::<ZoomLimit>(&app).max, 15.0);
        apply(&mut app, "cameradistancemaxfactor", "5");
        assert_eq!(res::<ZoomLimit>(&app).max, 30.0);
        apply(&mut app, "autoLootDefault", "1");
        assert!(res::<LootConfig>(&app).auto_loot);
        apply(&mut app, "showLootSpam", "0");
        assert!(!res::<LootConfig>(&app).show_loot_spam);
        apply(&mut app, "guildMemberNotify", "1");
        assert!(res::<crate::ui_guild::GuildMemberNotify>(&app).0);
        apply(&mut app, "BlockTrades", "1");
        assert!(res::<crate::ui_trade::BlockTrades>(&app).0);
        apply(&mut app, "UnitNameNPC", "0");
        assert!(!res::<NameConfig>(&app).npc);
        apply(&mut app, "unitnameown", "1");
        assert!(res::<NameConfig>(&app).own);
        apply(&mut app, crate::vplates::CVAR_ENEMIES, "0");
        assert!(!res::<VPlateMode>(&app).enemies);
        apply(&mut app, "nameplateshowfriends", "1");
        assert!(res::<VPlateMode>(&app).friends);
        apply(&mut app, "ChatBubbles", "0");
        assert!(!res::<BubbleConfig>(&app).all);
        apply(&mut app, "chatbubblesparty", "0");
        assert!(!res::<BubbleConfig>(&app).party);
        // WorldDetail: stop 0/1/2 is density ×1/×2/×3, clamped to the slider's range.
        apply(&mut app, "WorldDetail", "0");
        assert_eq!(res::<ClutterConfig>(&app).density, 1.0);
        apply(&mut app, "worlddetail", "7");
        assert_eq!(res::<ClutterConfig>(&app).density, 3.0);
        // `frillDensity` is the same field in cells per chunk, with its own clamp `[1, 256]`
        // (`0x688de0`); the stops round-trip through both spellings.
        apply(&mut app, "frillDensity", "48");
        assert_eq!(res::<ClutterConfig>(&app).density, 3.0);
        apply(&mut app, "frilldensity", "16");
        assert_eq!(res::<ClutterConfig>(&app).density, 1.0);
        // Past the top stop is honoured; pfUI's `hdgraphic` writes up to 256.
        apply(&mut app, "frillDensity", "256");
        assert_eq!(res::<ClutterConfig>(&app).density, 16.0);
        // `0` is not clutter-off: the callback pins it to 1.
        apply(&mut app, "frillDensity", "9000");
        assert_eq!(res::<ClutterConfig>(&app).density, 16.0);
        apply(&mut app, "frillDensity", "0");
        assert_eq!(res::<ClutterConfig>(&app).density, 1.0 / 16.0);
        apply(&mut app, "WorldDetail", "1");
        assert_eq!(res::<ClutterConfig>(&app).density, 2.0);
        for (wrote, want) in [
            ("0", 0u8),
            ("1", 1),
            ("2", 2),
            ("3", 3),
            ("2.9", 2),
            ("9", 3),
            ("-4", 0),
        ] {
            apply(&mut app, "weatherDensity", wrote);
            assert_eq!(
                res::<benilla_world::weather::WeatherState>(&app).weather_density,
                want,
                "weatherDensity {wrote}"
            );
        }
        // `gamma` is the ramp exponent: `SetGamma` applies `1 - v` before it arrives here. The
        // consumer clamps what the reference accepts unclamped (`SetGamma(5)` writes -4 there).
        for (wrote, want) in [("1.000000", 1.0), ("0.500000", 0.5), ("1.500000", 1.5)] {
            apply(&mut app, "gamma", wrote);
            assert_eq!(
                res::<crate::ui_gamma::DisplayGamma>(&app).0,
                want,
                "gamma {wrote}"
            );
        }
        apply(&mut app, "gamma", "-4.000000");
        assert_eq!(
            res::<crate::ui_gamma::DisplayGamma>(&app).0,
            *crate::ui_gamma::GAMMA_RANGE.start(),
            "a negative exponent clamps at the consumer, where it cannot blank the screen"
        );
        apply(&mut app, "gamma", "99");
        assert_eq!(
            res::<crate::ui_gamma::DisplayGamma>(&app).0,
            *crate::ui_gamma::GAMMA_RANGE.end()
        );
        assert_eq!(res::<ClutterConfig>(&app).frill_density(), 32.0);
        // Both spellings reach the field from a state neither holds; `$WOW_CLUTTER_DENSITY` takes
        // both for the session. Two values, since writing the sibling's mirrored value is a no-op.
        for (key, value) in [
            (benilla_ui::script::CVAR_WORLD_DETAIL, "2"),
            (benilla_ui::script::CVAR_FRILL_DENSITY, "16"),
        ] {
            app.world_mut().resource_mut::<ClutterConfig>().density = 0.5;
            assert_eq!(apply(&mut app, key, value), SetOutcome::Changed);
            assert_ne!(
                res::<ClutterConfig>(&app).density,
                0.5,
                "{key}: reached no knob"
            );
        }
        // `benilla-ui`'s `SetWorldDetail`/`GetWorldDetail` name these CVars by const and cannot see
        // this table, so both must stay registered and the stops must stay `frillDensity`'s unit
        // times n.
        assert!(REGISTERED
            .iter()
            .any(|r| r.name == benilla_ui::script::CVAR_WORLD_DETAIL));
        assert!(REGISTERED
            .iter()
            .any(|r| r.name == benilla_ui::script::CVAR_FRILL_DENSITY));
        // The same for `GetGamma`/`SetGamma`, which read and write `1 - gamma`.
        assert!(REGISTERED
            .iter()
            .any(|r| r.name == benilla_ui::script::CVAR_GAMMA));
        // `RestoreVideoDefaults`' row list lives in `benilla-ui` too; an unregistered row would be
        // skipped silently.
        for key in benilla_ui::script::VIDEO_DEFAULT_CVARS {
            assert!(
                REGISTERED.iter().any(|r| r.name.eq_ignore_ascii_case(key)),
                "{key}: RestoreVideoDefaults would restore it, and nothing registers it"
            );
        }
        for (n, frill) in benilla_ui::script::WORLD_DETAIL_STOPS.iter().enumerate() {
            assert_eq!(
                *frill,
                benilla_formats::FRILL_DENSITY * (n as u32 + 1),
                "stop {n}: the reference's own 0x804518 entry must be this knob's unit times the stop"
            );
            apply(&mut app, "WorldDetail", &n.to_string());
            let by_stop = res::<ClutterConfig>(&app).density;
            // Off the stop first (the mirror already wrote this spelling), then the stop's cells.
            apply(&mut app, "frillDensity", "1");
            app.world_mut().resource_mut::<ClutterConfig>().density = 0.5;
            apply(&mut app, "frillDensity", &frill.to_string());
            assert_eq!(
                res::<ClutterConfig>(&app).density,
                by_stop,
                "stop {n}: the two spellings disagree"
            );
            assert_eq!(res::<ClutterConfig>(&app).frill_density(), *frill as f32);
        }
        apply(&mut app, "WorldDetail", "1");
        assert_eq!(res::<ClutterConfig>(&app).density, 2.0);
        apply(&mut app, "minimapZoom", "5");
        assert_eq!(res::<MinimapZoom>(&app).outdoor, 5);
        assert_eq!(
            res::<MinimapZoom>(&app).inside,
            3,
            "the two indices are independent"
        );
        apply(&mut app, "minimapinsidezoom", "9");
        assert_eq!(res::<MinimapZoom>(&app).inside, MINIMAP_ZOOM_LEVELS - 1);
        apply(&mut app, "minimapZoom", "-2");
        assert_eq!(res::<MinimapZoom>(&app).outdoor, 0);
        assert_eq!(apply(&mut app, "uiScale", "banana"), SetOutcome::Refused);
        assert_eq!(res::<UiScaleCvar>(&app).0, 0.9);
        assert_eq!(apply(&mut app, "bogus", "1"), SetOutcome::Unknown);
    }

    #[test]
    fn compose_writes_the_diff_and_preserves_what_it_does_not_own() {
        let mut cvars = Cvars::default();
        cvars.own_for_session("uiScale", Some("1.2")); // env-overridden this session
        cvars.load_file(
            [
                ("FutureKnob".to_string(), "3".to_string()), // a newer build's key: preserved
                ("uiScale".to_string(), "0.8".to_string()),  // the file's own, kept as found
                ("farclip".to_string(), "400".to_string()),  // will return to default
            ]
            .into(),
        );
        cvars.set("MusicVolume", "0.7"); // moved: written
        cvars.set("farclip", "1000"); // back to default: removed
        let out = cvars.compose();
        assert_eq!(out.get("MusicVolume").map(String::as_str), Some("0.7"));
        assert!(!out.contains_key("MasterVolume"), "at default: absent");
        assert_eq!(out.get("uiScale").map(String::as_str), Some("0.8"));
        assert!(!out.contains_key("farclip"));
        assert_eq!(out.get("FutureKnob").map(String::as_str), Some("3"));
    }

    #[test]
    fn a_lua_setcvar_lands_in_config_toml_end_to_end() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-cvar-e2e-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _u = EnvGuard::unset("WOW_UI_SCALE");
        let _f = EnvGuard::unset("WOW_FARCLIP");
        let _d = EnvGuard::unset("WOW_CLUTTER_DENSITY");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());
        crate::local_state::write_atomic(
            &tmp.join("config.toml"),
            "[cvars]\nMusicVolume = \"0.1\"\n",
        )
        .unwrap();

        let mut app = cvar_app();

        // Startup: the file's MusicVolume reaches the knob; Update: the VM table seeds from it.
        app.update();
        assert_eq!(app.world().resource::<SoundConfig>().music, 0.1);
        assert_eq!(
            app.world_mut()
                .non_send_resource_mut::<UiScript>()
                .cvar("MusicVolume")
                .as_deref(),
            Some("0.1")
        );

        // The Lua write (what a settings slider will do) reaches the knob on the next frame…
        app.world_mut()
            .non_send_resource_mut::<UiScript>()
            .run(r#"SetCVar("MusicVolume", 0.75)"#)
            .unwrap();
        app.update();
        assert_eq!(app.world().resource::<SoundConfig>().music, 0.75);

        // …and the exit flush writes the diff: the moved value, nothing at its default.
        app.world_mut().write_message(AppExit::Success);
        app.update();
        let text = std::fs::read_to_string(tmp.join("config.toml")).unwrap();
        assert!(text.contains("MusicVolume = \"0.75\""), "{text}");
        assert!(!text.contains("MasterVolume"), "defaults stay out:\n{text}");
        let back: LocalConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.cvars.len(), 1, "a diff, not a dump: {text}");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// **The web's whole persistence path, and native's crash-safety.** A `SetCVar` must reach
    /// `config.toml` after the one quiet second **with no `AppExit` ever fired**: on the page
    /// there IS no exit frame — the tab closes and the wasm is simply gone — so an exit-only
    /// flush means every settings change a browser player makes is silently lost on reload
    /// (found live on a hosted realm, 2026-08-31). Decision 1528 moved `save_config` from
    /// `Update` onto the exit edge to catch the close-button's late `AppExit`; the move fixed
    /// that frame and deleted the debounced mid-session save this test pins down.
    #[test]
    fn a_setcvar_is_saved_after_the_quiet_second_without_an_app_exit() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-cvar-quiet-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _u = EnvGuard::unset("WOW_UI_SCALE");
        let _f = EnvGuard::unset("WOW_FARCLIP");
        let _d = EnvGuard::unset("WOW_CLUTTER_DENSITY");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());

        let mut app = cvar_app();
        app.update();
        app.world_mut()
            .non_send_resource_mut::<UiScript>()
            .run(r#"SetCVar("MusicVolume", 0.33)"#)
            .unwrap();
        app.update(); // sync_cvars drains the change and stamps last_change
        assert!(
            !tmp.join("config.toml").exists(),
            "inside the quiet second nothing is written yet"
        );
        std::thread::sleep(SAVE_QUIET + std::time::Duration::from_millis(100));
        app.update(); // the debounce has elapsed — this frame must write, exit or not
        let text = std::fs::read_to_string(tmp.join("config.toml"))
            .expect("config.toml written by the quiet-second save, with no AppExit in sight");
        assert!(text.contains("MusicVolume = \"0.33\""), "{text}");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// **The page and the client must default the layer the same way** — the tripwire for the
    /// opt-in → opt-out flip.
    ///
    /// `web/packs.js`'s `hdWanted()` decides whether the PAGE fetches `/data-hd`;
    /// [`apply_boot_layers`] decides whether the CLIENT mounts it. Neither can detect a
    /// disagreement on its own: a page defaulting off against a client defaulting on makes every
    /// character read a cold round trip, and the other way round makes a visitor pay 1.7 GB for
    /// art they never see.
    ///
    /// Read the way a human reads it: `hdWanted`'s fallback legs are bare `return <bool>;` lines,
    /// while its env/query arms all end in `=== true;`, so "which way does it fall back" is a
    /// substring question with one answer.
    #[test]
    fn the_pages_layer_default_is_the_registered_one() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("workspace root");
        let js = std::fs::read_to_string(root.join("web/packs.js")).expect("web/packs.js");
        let body = js
            .split_once("function hdWanted() {")
            .expect("web/packs.js still defines hdWanted()")
            .1;
        let end = body.find("\n}").expect("hdWanted's closing brace");
        let body = &body[..end];

        let registered = registered_default("hdCharacterModels").expect("the row is registered");
        let on = benilla_formats::layers::layer_switch(registered)
            .expect("the registered default is a value, not an empty string");
        assert!(on, "this assertion is the flip itself: the row ships ON");

        assert!(
            !body.contains("return false;"),
            "hdCharacterModels is registered {registered:?} (on), but web/packs.js's hdWanted() \
             still falls back to false — the page would not fetch /data-hd for a player whose \
             client mounts it"
        );
        assert!(
            body.contains("return true;"),
            "hdWanted() has no fallback at all — every leg now depends on a saved setting"
        );
    }

    /// **A row with no host knob still reaches the file** — `hdCharacterModels`, and the bug it
    /// shipped with (reported 2026-09-08: "the toggle does nothing, even if I reload the page").
    /// The layer is a MOUNT, decided before the `App` exists, so this row has nothing live to
    /// write; the next start is its only reader, so the click has to land in `config.toml`.
    /// The click driven is the **opt-out** — the row is registered `"1"`.
    #[test]
    fn the_knobless_hd_models_row_lands_in_config_toml() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-cvar-hd-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _u = EnvGuard::unset("WOW_UI_SCALE");
        let _f = EnvGuard::unset("WOW_FARCLIP");
        let _d = EnvGuard::unset("WOW_CLUTTER_DENSITY");
        let _m = EnvGuard::unset("WOW_HD_MODELS");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());

        let mut app = cvar_app();
        app.update();
        app.world_mut()
            .non_send_resource_mut::<UiScript>()
            .run(r#"SetCVar("hdCharacterModels", 0)"#)
            .unwrap();
        app.update();
        app.world_mut().write_message(AppExit::Success);
        app.update();
        let text = std::fs::read_to_string(tmp.join("config.toml"))
            .expect("the click must reach config.toml — the next start is its only reader");
        assert!(text.contains("hdCharacterModels = \"0\""), "{text}");
        assert_eq!(
            benilla_formats::layers::layer_switch(
                &toml::from_str::<LocalConfig>(&text).unwrap().cvars["hdCharacterModels"]
            ),
            Some(false)
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// A client with a real CVar host: every knob resource an observer writes, every observer,
    /// [`CvarPlugin`] and a VM for the mirror.
    fn cvar_app() -> App {
        let mut app = App::new();
        app.add_plugins(bevy::MinimalPlugins)
            .insert_resource(SoundConfig::default())
            .insert_resource(UiScaleCvar(DEFAULT_UI_SCALE))
            .insert_resource(ViewDistance {
                farclip: 350.0,
                nearclip: benilla_world::view::NEARCLIP_DEFAULT,
            })
            .insert_resource(MsaaSetting { samples: 1 })
            // Literal: `TexFilterSetting::default()` reads `$WOW_TRILINEAR`/`$WOW_ANISO`.
            .insert_resource(benilla_assets::TexFilterSetting {
                trilinear: true,
                aniso: 16,
            })
            // A real-shaped device menu: `GetCurrentMultisampleFormat` needs rows to find.
            .insert_resource(benilla_world::view::MsaaFormats {
                formats: vec![(32, 32, 1), (32, 32, 2), (32, 32, 4)],
            })
            .init_resource::<LookConfig>()
            .init_resource::<crate::player::camera_dynamics::CameraOptions>()
            .init_resource::<crate::ui_chat::combat::CombatLogRanges>()
            .init_resource::<crate::combat_text::DamageTextGates>()
            .init_resource::<crate::ui_chat::combat::LogPeriodicSpells>()
            .init_resource::<benilla_world::weather::WeatherState>()
            .init_resource::<crate::ui_gamma::DisplayGamma>()
            .init_resource::<ClickConfig>()
            .init_resource::<crate::target::AssistAttack>()
            .init_resource::<LootConfig>()
            .init_resource::<NameConfig>()
            .init_resource::<VPlateMode>()
            .init_resource::<ClutterConfig>()
            .init_resource::<MinimapZoom>()
            .init_resource::<BubbleConfig>()
            .init_resource::<ZoomLimit>()
            .init_resource::<FollowConfig>()
            .init_resource::<VideoConfig>()
            // Literal, not Default: RenderScale::default() reads $WOW_RENDER_SCALE.
            .insert_resource(RenderScale(1.0))
            // Literal for the same reason: LightingSettings::default() reads $WOW_LIGHTING /
            // $WOW_LIGHTING_QUALITY, IndirectLighting::default() $WOW_SSAO / $WOW_SSGI, and
            // WaterDetail::default() $WOW_WATER_DETAIL.
            .insert_resource(LightingSettings {
                model: LightingModel::default(),
                quality: LightingQuality::DEFAULT,
            })
            .insert_resource(IndirectLighting {
                ao: AO_DEFAULT,
                gi: GI_DEFAULT,
            })
            .insert_resource(WaterDetail(benilla_world::liquid::WATER_DETAIL_CATA))
            // Literal: `Realmlist::default()` reads `$WOW_HOST`.
            .insert_resource(crate::realmlist::Realmlist::unpinned(
                crate::realmlist::DEFAULT_REALMLIST,
            ))
            .init_resource::<PaneRate>()
            .init_resource::<crate::ui_guild::GuildMemberNotify>()
            .init_resource::<crate::ui_trade::BlockTrades>()
            .init_resource::<crate::spell::AutoSelfCast>()
            .init_resource::<crate::perf::FpsJournalSetting>()
            .init_resource::<crate::text_filter::TextFilterSwitches>()
            .init_resource::<crate::game_tip::GameTipSetting>()
            .add_plugins(CvarPlugin);
        for observer in ALL_OBSERVERS {
            observer(&mut app);
        }
        app.insert_non_send_resource(UiScript::new().unwrap());
        app
    }

    /// Every CVar observer in the crate; a new one is a line here and an `add_observer` in its
    /// plugin.
    const ALL_OBSERVERS: &[fn(&mut App)] = &[
        |app| {
            app.add_observer(crate::sound::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_script::on_cvar);
        },
        |app| {
            app.add_observer(crate::video::on_cvar);
        },
        |app| {
            app.add_observer(crate::player::camera::on_cvar);
        },
        |app| {
            app.add_observer(crate::player::camera_dynamics::on_cvar);
        },
        |app| {
            app.add_observer(crate::target::on_cvar);
        },
        |app| {
            app.add_observer(crate::spell::cast_target::on_cvar);
        },
        |app| {
            app.add_observer(crate::combat_text::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_chat::combat::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_loot::on_cvar);
        },
        |app| {
            app.add_observer(crate::nameplates::on_cvar);
        },
        |app| {
            app.add_observer(crate::vplates::on_cvar);
        },
        |app| {
            app.add_observer(crate::game_tip::on_cvar);
        },
        |app| {
            app.add_observer(crate::text_filter::on_cvar);
        },
        |app| {
            app.add_observer(crate::chat_bubble::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_guild::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_trade::on_cvar);
        },
        |app| {
            app.add_observer(crate::minimap::on_cvar);
        },
        |app| {
            app.add_observer(crate::portrait::on_cvar);
        },
        |app| {
            app.add_observer(crate::perf::on_cvar);
        },
        |app| {
            app.add_observer(crate::realmlist::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_gamma::on_cvar);
        },
        |app| {
            app.add_observer(crate::world_backdrop::on_cvar);
        },
    ];

    /// A host write, its latch committed at once, and its observers run before this returns.
    fn apply(app: &mut App, name: &str, value: &str) -> SetOutcome {
        let world = app.world_mut();
        let (outcome, events) = {
            let mut cvars = world.resource_mut::<Cvars>();
            let outcome = cvars.set(name, value);
            cvars.commit_latched();
            (outcome, cvars.take_events())
        };
        for event in events {
            world.trigger(event);
        }
        outcome
    }

    fn res<T: Resource>(app: &App) -> &T {
        app.world().resource::<T>()
    }

    /// The last character entered survives a quit: two launches over one `benilla-config/` with the
    /// real [`CvarPlugin`] and [`crate::char_select`] systems, so the screen's host write must
    /// reach the file through the registry's dirty/compose path.
    #[test]
    fn entering_the_world_survives_the_quit_and_comes_back_selected() {
        use crate::char_select::{ClientState, Roster};
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-lastchar-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _u = EnvGuard::unset("WOW_UI_SCALE");
        let _f = EnvGuard::unset("WOW_FARCLIP");
        let _d = EnvGuard::unset("WOW_CLUTTER_DENSITY");
        let _w = EnvGuard::unset("WOW_CHAR");
        let _s = EnvGuard::unset("WOW_CHARSELECT_PICK");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());
        let roster = || {
            (1..=4)
                .map(|g| crate::char_select::test_character(g, &format!("Char{g}")))
                .collect::<Vec<_>>()
        };

        // ── Launch 1: the roster lands, and the player enters the world as the third row. ────
        let (tx, _rx) = async_channel::unbounded();
        let mut app = cvar_app();
        app.add_plugins(bevy::state::app::StatesPlugin);
        crate::char_select::add_test_systems(&mut app, tx);
        app.update(); // Startup loads the (absent) file; the first Update seeds the VM table
        app.world_mut().write_message(crate::net::CharListMessage {
            characters: roster(),
            realm: None,
        });
        app.update();
        assert_eq!(
            app.world().resource::<Roster>().selected(),
            Some(0),
            "nothing remembered yet, so the first row — the behaviour that was already right",
        );
        app.world_mut().resource_mut::<Roster>().pending_pick = Some(3); // guid 3 = row 2
        app.update();
        app.world_mut().write_message(AppExit::Success);
        app.update();

        let text = std::fs::read_to_string(tmp.join("config.toml")).unwrap();
        assert!(
            text.contains("lastCharacterIndex = \"2\""),
            "entering the world must reach the file, 0-based like Config.wtf:\n{text}"
        );

        // ── Launch 2: a fresh client over the same folder, and the roster arrives. ───────────
        let (tx, _rx2) = async_channel::unbounded();
        let mut app = cvar_app();
        app.add_plugins(bevy::state::app::StatesPlugin);
        crate::char_select::add_test_systems(&mut app, tx);
        app.update();
        app.world_mut().write_message(crate::net::CharListMessage {
            characters: roster(),
            realm: None,
        });
        app.update();

        assert_eq!(
            app.world().resource::<Roster>().selected(),
            Some(2),
            "the second launch must stand the SAME character on the stage — the whole report",
        );
        assert_eq!(
            *app.world().resource::<State<ClientState>>().get(),
            ClientState::CharSelect,
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_minimap_setzoom_reaches_the_knob_and_the_file() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-mmzoom-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _u = EnvGuard::unset("WOW_UI_SCALE");
        let _f = EnvGuard::unset("WOW_FARCLIP");
        let _d = EnvGuard::unset("WOW_CLUTTER_DENSITY");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());
        // The previous session left the outdoor map zoomed right in.
        crate::local_state::write_atomic(
            &tmp.join("config.toml"),
            "[cvars]\nminimapZoom = \"5\"\n",
        )
        .unwrap();

        let mut app = cvar_app();
        app.update();

        assert_eq!(app.world().resource::<MinimapZoom>().outdoor, 5);
        assert_eq!(app.world().resource::<MinimapZoom>().inside, 3);
        let seed = {
            let z = app.world().resource::<MinimapZoom>();
            (z.outdoor, z.inside)
        };
        {
            let mut script = app.world_mut().non_send_resource_mut::<UiScript>();
            assert_eq!(script.cvar("minimapZoom").as_deref(), Some("5"));
            // The widget is born first, then the persisted level is pushed into it: seeding a
            // widget that does not exist is a no-op.
            script.run(r#"m = CreateFrame("Minimap", "Mini")"#).unwrap();
            script.set_minimap_zoom(seed.0, seed.1);
            assert_eq!(script.eval::<u8>("return m:GetZoom()").unwrap(), 5);
            script.run("m:SetZoom(m:GetZoom() - 2)").unwrap();
        }
        app.update();
        assert_eq!(app.world().resource::<MinimapZoom>().outdoor, 3);

        app.world_mut().write_message(AppExit::Success);
        app.update();
        let text = std::fs::read_to_string(tmp.join("config.toml")).unwrap();
        assert!(
            !text.contains("minimapZoom"),
            "back at the registered default 3, so it leaves the diff entirely:\n{text}"
        );

        app.world_mut()
            .non_send_resource_mut::<UiScript>()
            .run("m:SetZoom(1)")
            .unwrap();
        app.update();
        app.world_mut().write_message(AppExit::Success);
        app.update();
        let text = std::fs::read_to_string(tmp.join("config.toml")).unwrap();
        assert!(text.contains("minimapZoom = \"1\""), "{text}");
        assert_eq!(app.world().resource::<MinimapZoom>().outdoor, 1);
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn the_toml_round_trips() {
        let cfg = LocalConfig {
            cvars: [("MusicVolume".to_string(), "0.7".to_string())].into(),
        };
        let text = format!("{HEADER}{}", toml::to_string(&cfg).unwrap());
        let back: LocalConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.cvars, cfg.cvars);
        let hand = "# my note\n[cvars]\nFarclip = \"500\"\n";
        let parsed: LocalConfig = toml::from_str(hand).unwrap();
        assert_eq!(parsed.cvars.get("Farclip").map(String::as_str), Some("500"));
    }

    /// The string-valued rows, a closed list, each default checked on its own terms: `realmName`
    /// (the reference's registered `""`, `0x882748`) and `gxApi` are empty until the session writes
    /// them; `gxResolution` and `realmList` pass their observers' parsers
    /// ([`crate::video::parse_resolution`], [`crate::realmlist::normalize`]).
    #[test]
    fn the_string_valued_cvars_are_the_realm_and_the_windowed_size() {
        let mut strings: Vec<&str> = REGISTERED
            .iter()
            .filter(|r| r.default.parse::<f32>().is_err())
            .map(|r| r.name)
            .collect();
        strings.sort_unstable(); // the list is the claim, not where the rows sit in the table
        assert_eq!(
            strings,
            vec!["gxApi", "gxResolution", "realmList", "realmName"]
        );
        let default_of = |name: &str| {
            REGISTERED
                .iter()
                .find(|r| r.name == name)
                .map(|r| r.default)
                .expect("registered")
        };
        assert_eq!(default_of("realmName"), "");
        assert_eq!(default_of("gxApi"), "");
        assert_eq!(
            crate::video::parse_resolution(default_of("gxResolution")),
            Some(crate::video::DEFAULT_WINDOWED)
        );
        // A default `realmlist::normalize` rejects would ship a client that cannot dial.
        assert_eq!(
            crate::realmlist::normalize(default_of(crate::realmlist::CVAR_REALMLIST)).as_deref(),
            Some(crate::realmlist::DEFAULT_REALMLIST),
        );
    }

    /// A registry loaded from `file`, with no disk.
    fn registry(file: &[(&str, &str)]) -> Cvars {
        let mut cvars = Cvars::default();
        cvars.load_file(
            file.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        );
        cvars
    }

    /// A latched row's write is staged: the applied value stands and nothing fires or dirties (the
    /// reference's `Set 0x63df50` stores `latchedValue` and skips `InternalSet`) until the commit
    /// (`0x63e060`) applies, fires, persists and mirrors it.
    #[test]
    fn a_latched_row_stages_the_write_until_the_boundary_commits_it() {
        let mut cvars = Cvars::default();
        assert!(
            cvars.row("gxVSync").unwrap().latched,
            "the reference's flags=3 row"
        );
        assert_eq!(cvars.set("gxVSync", "0"), SetOutcome::Staged);
        assert_eq!(cvars.get("gxVSync"), Some("1"), "applied value stands");
        assert_eq!(cvars.row("gxVSync").unwrap().pending.as_deref(), Some("0"));
        assert!(!cvars.has_events(), "nothing fires before the boundary");
        assert!(!cvars.dirty, "nothing to save before the boundary");
        assert_eq!(
            cvars.set("gxVSync", "0"),
            SetOutcome::Unchanged,
            "same stage"
        );
        assert!(
            !cvars.compose().contains_key("gxVSync"),
            "a stage that is never committed never reaches the file"
        );
        assert_eq!(cvars.commit_latched(), 1);
        assert_eq!(cvars.get("gxVSync"), Some("0"));
        assert_eq!(cvars.row("gxVSync").unwrap().pending, None);
        assert_eq!(
            cvars.take_events(),
            vec![CvarChanged {
                name: "gxVSync".into(),
                old: "1".into(),
                new: "0".into()
            }]
        );
        assert!(cvars.dirty);
        assert_eq!(
            cvars.compose().get("gxVSync").map(String::as_str),
            Some("0")
        );
        assert_eq!(
            cvars.take_outbox(),
            vec![("gxVSync".to_string(), "0".to_string())]
        );
        cvars.set("gxVSync", "1");
        assert_eq!(cvars.set("gxVSync", "0"), SetOutcome::Unchanged);
        assert_eq!(cvars.row("gxVSync").unwrap().pending, None);
        assert_eq!(cvars.commit_latched(), 0);
        // A latched row outside the gx set is not the video restart's to commit.
        assert_eq!(cvars.set("SoundBufferSize", "200"), SetOutcome::Staged);
        assert_eq!(cvars.commit_latched(), 0);
        assert_eq!(cvars.get("SoundBufferSize"), Some("100"));
        assert_eq!(
            cvars.row("SoundBufferSize").unwrap().pending.as_deref(),
            Some("200")
        );
        assert!(!cvars.compose().contains_key("SoundBufferSize"));
        assert_eq!(cvars.set("MusicVolume", "0.7"), SetOutcome::Changed);
        assert_eq!(cvars.get("MusicVolume"), Some("0.7"));
        assert_eq!(cvars.take_events().len(), 1);
    }

    #[test]
    fn a_numeric_row_refuses_what_does_not_parse_and_corrects_the_mirror() {
        let mut cvars = Cvars::default();
        assert_eq!(cvars.set_from_vm("uiScale", "banana"), SetOutcome::Refused);
        assert_eq!(cvars.get("uiScale"), Some("0.9"));
        assert!(!cvars.has_events());
        assert!(!cvars.dirty);
        assert_eq!(
            cvars.take_outbox(),
            vec![("uiScale".to_string(), "0.9".to_string())],
            "the VM stored 'banana' synchronously; the host writes the truth back"
        );
        assert_eq!(cvars.set("uiScale", "banana"), SetOutcome::Refused);
        assert!(
            cvars.take_outbox().is_empty(),
            "a host write has no mirror to correct"
        );
        assert_eq!(
            cvars.set("realmList", "not an address"),
            SetOutcome::Changed
        );
        assert_eq!(cvars.set("nosuchrow", "1"), SetOutcome::Unknown);
    }

    #[test]
    fn an_addon_row_persists_like_the_clients_own() {
        let mut cvars = registry(&[("myAddonKnob", "3")]);
        assert_eq!(
            cvars.orphans(),
            vec![("myAddonKnob".to_string(), "3".to_string())],
            "unclaimed until the addon declares it — the VM's saved base"
        );
        cvars.learn_addon_row("myAddonKnob", "1");
        assert_eq!(
            cvars.get("myAddonKnob"),
            Some("3"),
            "starts at the saved value"
        );
        assert_eq!(cvars.default_of("myAddonKnob"), Some("1"));
        assert!(cvars.orphans().is_empty());
        cvars.learn_addon_row("myAddonKnob", "9");
        assert_eq!(
            cvars.default_of("myAddonKnob"),
            Some("1"),
            "a re-declaration is a no-op"
        );
        assert!(!cvars.dirty);
        assert_eq!(cvars.set_from_vm("myAddonKnob", "5"), SetOutcome::Changed);
        assert!(cvars.dirty, "an addon-only change is a change");
        assert_eq!(
            cvars.compose().get("myAddonKnob").map(String::as_str),
            Some("5")
        );
        cvars.set_from_vm("myAddonKnob", "1");
        assert!(
            !cvars.compose().contains_key("myAddonKnob"),
            "back at the addon's default, it leaves the diff"
        );
        let cvars = registry(&[("FutureKnob", "3")]);
        assert_eq!(
            cvars.compose().get("FutureKnob").map(String::as_str),
            Some("3")
        );
    }

    #[test]
    fn a_session_owned_row_answers_the_env_and_never_reaches_the_file() {
        let mut cvars = Cvars::default();
        cvars.own_for_session("uiScale", Some("1.2"));
        cvars.load_file(BTreeMap::from([("uiScale".to_string(), "0.8".to_string())]));
        assert_eq!(
            cvars.get("uiScale"),
            Some("1.2"),
            "the env's, not the file's"
        );
        assert!(
            !cvars.has_events(),
            "the knob already read the env — nothing to apply"
        );
        assert_eq!(cvars.set("uiScale", "1.4"), SetOutcome::Changed);
        assert_eq!(
            cvars.compose().get("uiScale").map(String::as_str),
            Some("0.8"),
            "the file keeps what it said"
        );
        // A lever with no resource still marks the row, so the file cannot apply over the env.
        cvars.own_for_session("farclip", None);
        cvars.load_file(BTreeMap::from([("farclip".to_string(), "500".to_string())]));
        assert_eq!(cvars.get("farclip"), Some("1000"));
        assert!(cvars.is_session_owned("FARCLIP"));
    }

    /// fork: web build — **the GPU recovery tier's render scale is a fallback the player can take
    /// back.** It applies only where the file holds no render scale, is never saved, and the
    /// player's own write (the Graphics page's slider, through the VM) makes the row theirs again,
    /// so it is saved and outranks the tier from the next load on. Before this the tier arrived as
    /// an env override: the saved value was ignored and every in-game change was lost on reload.
    #[test]
    fn the_gpu_tier_render_scale_yields_to_the_player() {
        let mut cvars = Cvars::default();
        cvars.load_file(BTreeMap::new());
        assert!(cvars.fallback_for_session("renderScale", "0.55"));
        assert_eq!(cvars.get("renderScale"), Some("0.55"));
        assert!(cvars.has_events(), "the knob hears the tier");
        cvars.take_events();
        assert!(
            !cvars.compose().contains_key("renderScale"),
            "a recovery never pins itself into the file"
        );

        // The player moves the slider: applied, and theirs.
        assert_eq!(cvars.set_from_vm("renderScale", "1"), SetOutcome::Changed);
        assert!(!cvars.is_session_owned("renderScale"));
        cvars.set_from_vm("renderScale", "0.8");
        assert_eq!(
            cvars.compose().get("renderScale").map(String::as_str),
            Some("0.8")
        );

        // A saved render scale outranks the tier on the next load.
        let mut next = Cvars::default();
        next.load_file(BTreeMap::from([(
            "renderScale".to_string(),
            "0.8".to_string(),
        )]));
        assert!(!next.fallback_for_session("renderScale", "0.55"));
        assert_eq!(next.get("renderScale"), Some("0.8"));
        assert!(!next.is_session_owned("renderScale"));

        // A player who picks the tier's own value still takes the row back, and it is saved.
        let mut same = Cvars::default();
        same.load_file(BTreeMap::new());
        same.fallback_for_session("renderScale", "0.55");
        assert_eq!(
            same.set_from_vm("renderScale", "0.55"),
            SetOutcome::Unchanged
        );
        assert_eq!(
            same.compose().get("renderScale").map(String::as_str),
            Some("0.55")
        );
    }

    /// fork: web build — **any Graphics Apply retires the recovery tier**, not only a Render
    /// Scale move: the player's write to another Graphics row is flagged, and releasing the
    /// fallback returns the render scale to its default, heard by the knob and never saved. A
    /// write elsewhere, or a host write, is not the player applying Graphics settings.
    #[test]
    fn a_graphics_apply_releases_the_gpu_tier() {
        let mut cvars = Cvars::default();
        cvars.load_file(BTreeMap::new());
        cvars.fallback_for_session("renderScale", "0.55");
        cvars.take_events();

        cvars.set("farclip", "300");
        cvars.set_from_vm("MasterVolume", "0.5");
        assert!(!cvars.take_graphics_applied());

        assert_eq!(
            cvars.set_from_vm("lightingQuality", "2"),
            SetOutcome::Changed
        );
        assert!(cvars.take_graphics_applied());
        assert!(!cvars.take_graphics_applied(), "an edge, not a state");

        assert_eq!(cvars.release_fallbacks(), 1);
        let default = cvars.default_of("renderScale").unwrap().to_string();
        assert_eq!(cvars.get("renderScale"), Some(default.as_str()));
        assert!(!cvars.is_session_owned("renderScale"));
        assert!(cvars
            .take_events()
            .iter()
            .any(|e| e.name == "renderScale" && e.new == default));
        assert!(!cvars.compose().contains_key("renderScale"));
        assert_eq!(cvars.release_fallbacks(), 0, "released once");

        // The released row is the player's again: their next write is simply saved.
        assert_eq!(cvars.set_from_vm("renderScale", "0.8"), SetOutcome::Changed);
        assert_eq!(
            cvars.compose().get("renderScale").map(String::as_str),
            Some("0.8")
        );

        // A latched gx row stages on the write and still counts: its Okay commits it.
        let mut gx = Cvars::default();
        gx.load_file(BTreeMap::new());
        let flipped = if gx.flag("gxVSync") == Some(true) {
            "0"
        } else {
            "1"
        };
        gx.set_from_vm("gxVSync", flipped);
        assert!(gx.take_graphics_applied());
    }

    /// fork: web build — **the recovery tier caps the lighting quality, saved value or not.** A
    /// fallback would let a saved Cinematic reload straight back into the hang; the cap holds it
    /// down for the session, never writes itself into the file, and a Graphics Apply restores
    /// exactly what it displaced. A quality already under the cap, or a lever's, is left alone.
    #[test]
    fn the_gpu_tier_caps_the_lighting_quality() {
        // The shipped default, capped and restored.
        let mut fresh = Cvars::default();
        fresh.load_file(BTreeMap::new());
        assert!(fresh.cap_for_session("lightingQuality", "1"));
        assert_eq!(fresh.get("lightingQuality"), Some("1"));
        assert!(fresh
            .take_events()
            .iter()
            .any(|e| e.name == "lightingQuality"));
        assert!(!fresh.compose().contains_key("lightingQuality"));
        assert_eq!(fresh.release_fallbacks(), 1);
        assert_eq!(
            fresh.get("lightingQuality"),
            fresh.default_of("lightingQuality")
        );

        // A saved quality above the cap is held down, and the file keeps it.
        let saved = || BTreeMap::from([("lightingQuality".to_string(), "3".to_string())]);
        let mut cvars = Cvars::default();
        cvars.load_file(saved());
        assert!(cvars.cap_for_session("lightingQuality", "0"));
        assert_eq!(cvars.get("lightingQuality"), Some("0"));
        assert_eq!(
            cvars.compose().get("lightingQuality").map(String::as_str),
            Some("3")
        );
        cvars.set_from_vm("farclip", "300");
        assert!(cvars.take_graphics_applied());
        assert_eq!(cvars.release_fallbacks(), 1);
        assert_eq!(cvars.get("lightingQuality"), Some("3"));
        assert!(!cvars.is_session_owned("lightingQuality"));

        // The player's own pick under the tier is theirs, and saved.
        let mut pick = Cvars::default();
        pick.load_file(saved());
        pick.cap_for_session("lightingQuality", "0");
        assert_eq!(
            pick.set_from_vm("lightingQuality", "1"),
            SetOutcome::Changed
        );
        assert_eq!(pick.release_fallbacks(), 0);
        assert_eq!(
            pick.compose().get("lightingQuality").map(String::as_str),
            Some("1")
        );

        // Already under the cap: nothing moves, and nothing is held.
        let mut low = Cvars::default();
        low.load_file(BTreeMap::from([(
            "lightingQuality".to_string(),
            "1".to_string(),
        )]));
        assert!(!low.cap_for_session("lightingQuality", "2"));
        assert!(!low.is_session_owned("lightingQuality"));

        // A lever (`?lightingquality=`, `$WOW_LIGHTING_QUALITY`) outranks the tier.
        let mut lever = Cvars::default();
        lever.override_for_session("lightingQuality", "4");
        lever.load_file(BTreeMap::new());
        assert!(!lever.cap_for_session("lightingQuality", "0"));
        assert_eq!(lever.get("lightingQuality"), Some("4"));
    }

    /// The camera reads `gxMultisample` once at spawn, so the file must be applied inside
    /// [`CvarLoad`].
    #[test]
    fn the_loaded_file_reaches_a_knob_before_anything_ordered_after_cvar_load() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-cvar-load-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _u = EnvGuard::unset("WOW_UI_SCALE");
        let _f = EnvGuard::unset("WOW_FARCLIP");
        let _d = EnvGuard::unset("WOW_CLUTTER_DENSITY");
        let _m = EnvGuard::unset("WOW_MSAA");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());
        crate::local_state::write_atomic(
            &tmp.join("config.toml"),
            "[cvars]\nMusicVolume = \"0.1\"\ngxMultisample = \"4\"\n",
        )
        .unwrap();
        #[derive(Resource, Default)]
        struct SeenAtStartup(Option<(f32, u32)>);
        fn after_load(
            sound: Res<SoundConfig>,
            msaa: Res<MsaaSetting>,
            mut seen: ResMut<SeenAtStartup>,
        ) {
            seen.0 = Some((sound.music, msaa.samples));
        }
        let mut app = cvar_app();
        app.init_resource::<SeenAtStartup>()
            .add_systems(Startup, after_load.after(CvarLoad));
        app.update();
        assert_eq!(
            app.world().resource::<SeenAtStartup>().0,
            Some((0.1, 4)),
            "both the plain row and the latched one are applied by the time CvarLoad is over"
        );
        assert_eq!(
            app.world().resource::<Cvars>().get("gxMultisample"),
            Some("4"),
            "a file value is applied, not staged: the reference's LoadFile runs before Register"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Every registered row has a reader: a string literal naming it in this crate's or
    /// `benilla-ui`'s code, or a `LUA_ONLY` entry naming its stock reader. Test files do not count.
    #[test]
    fn every_registered_row_has_a_reader_in_the_source() {
        /// Rows whose only reader is the stock interface.
        const LUA_ONLY: &[(&str, &str)] = &[
            (
                "statusBarText",
                "TextStatusBar.lua reads it on each CVAR_UPDATE",
            ),
            (
                "UberTooltips",
                "GameTooltip's binding-line gate, stock and pfUI",
            ),
            ("gxApi", "pfUI's system tooltip names the backend"),
            (
                "hdCharacterModels",
                "not Lua: read before the App exists by `apply_boot_layers` in this file, which the \
                 scan skips — a mount has no later reader to observe it",
            ),
            (
                "useUiScale",
                "UIOptionsFrame.lua and OptionsFrame.lua branch on it to gate the uiScale slider",
            ),
        ];
        let app_src = crate::test_support::src_dir();
        let ui_src = app_src
            .parent()
            .and_then(std::path::Path::parent)
            .expect("crates/")
            .join("benilla-ui")
            .join("src");
        let mut code = String::new();
        for file in crate::test_support::rust_files(&app_src)
            .into_iter()
            .chain(crate::test_support::rust_files(&ui_src))
        {
            let rel = file.to_string_lossy().replace('\\', "/");
            if rel.ends_with("/cvars.rs") && rel.contains("benilla-app") || rel.contains("tests") {
                continue;
            }
            let text = std::fs::read_to_string(&file).expect("source is readable");
            for line in text.lines() {
                if !line.trim_start().starts_with("//") {
                    code.push_str(line);
                    code.push('\n');
                }
            }
        }
        let mut orphans = Vec::new();
        for row in REGISTERED {
            if LUA_ONLY.iter().any(|(n, _)| *n == row.name) {
                continue;
            }
            let exact = format!("\"{}\"", row.name);
            let lower = format!("\"{}\"", row.name.to_ascii_lowercase());
            if !(code.contains(&exact) || code.contains(&lower)) {
                orphans.push(row.name);
            }
        }
        assert!(
            orphans.is_empty(),
            "registered rows nothing in the source reads (an observer arm, a `cvars.get`, or a \
             `LUA_ONLY` entry with its stock reader): {orphans:?}"
        );
        for (name, _) in LUA_ONLY {
            assert!(
                REGISTERED.iter().any(|r| r.name == *name),
                "{name}: named Lua-only but not registered"
            );
        }
    }

    /// The rows the reference latches (`flags` 2 or 3 at the register site): the sound-init row and
    /// the `gx*` block `RestartGx` commits. `trilinear`, `anisotropic` and `farclip` register with
    /// `flags = 1` and apply live.
    #[test]
    fn the_latched_rows_are_the_references_own() {
        let mut latched: Vec<&str> = REGISTERED
            .iter()
            .filter(|r| r.latched)
            .map(|r| r.name)
            .collect();
        latched.sort_unstable();
        assert_eq!(
            latched,
            vec![
                "SoundBufferSize",
                "gxApi",
                "gxColorBits",
                "gxDepthBits",
                "gxMultisample",
                "gxResolution",
                "gxVSync",
                "gxWindow",
            ]
        );
    }
}
