//! `benilla`: a from-scratch World of Warcraft 1.12.1 client on Bevy. It opens the patch chain from
//! the install (`$WOW_DATA`, the project folder on a dev build, else beside the binary), streams
//! the world around the player through the `benilla-assets` `mpq://` pipeline, and runs the network
//! session on a background thread ([`net`]).
//!
//! The world is loaded when a character enters it and released when they leave; the glue screens
//! have no world behind them, so with no server the client sits at the login screen, as the
//! reference does. The capture harness (`$WOW_CAPTURE`) boots straight in-world.

// A player build's dead code is the seam working: with `--no-default-features` every symbol only an
// instrument calls goes unused, and a `cfg` per item would spread seam knowledge into gameplay
// modules. Dev builds warn normally.
#![cfg_attr(not(feature = "dev"), allow(dead_code))]
// **Nightly, but only on the one target that is already nightly.** The audio render worker waits on
// the worklet's read index with `core::arch::wasm32::memory_atomic_wait32`
// (`sound::web_output::render_loop`) — the wasm instruction a Web Worker is allowed to execute and
// the page's main thread is not, and the only way to block on a counter a JavaScript
// `Atomics.notify` will wake. It is still unstable (rust#77839).
//
// The `cfg_attr` is what keeps that off every other build: on stable, native or wasm without
// atomics, the predicate is false and no `feature` attribute is emitted at all. The threaded
// browser build is nightly regardless — `scripts/web-build-threads.sh` needs `-Zbuild-std` to
// compile std with `+atomics` in the first place — so this asks nothing of it that it did not
// already require.
//
// The stable alternative is `js_sys::Atomics::wait_with_timeout`, which is a JS call per wait on a
// path whose whole point is not to cross the boundary; it is the fallback if this feature is ever
// withdrawn rather than stabilised.
#![cfg_attr(
    all(target_arch = "wasm32", target_feature = "atomics"),
    feature(stdarch_wasm_atomic_wait)
)]

/// The realtime-audio allocation tripwire: in debug builds an allocation inside `sound::output`'s
/// `no_alloc` scopes (the output IO callback, the render pass) aborts the process; release builds
/// carry no wrapper.
#[cfg(debug_assertions)]
#[global_allocator]
static ALLOC: assert_no_alloc::AllocDisabler = assert_no_alloc::AllocDisabler;

pub mod addon_harness;
mod area;
mod area_poi;
mod area_trigger;
#[cfg(feature = "dev")]
mod asset_churn;
mod aura_visual;
mod bindings;
mod blob_shadow;
mod bowstring;
mod camera_shake;
#[cfg(feature = "dev")]
mod capture;
mod char_create;
mod char_select;
mod chat_bubble;
mod chr_classes;
mod cinematic;
mod combat_log;
mod combat_text;
mod console;
mod crash;
mod creature_anim;
mod cursor;
mod cvars;
mod death;
#[cfg(feature = "dev")]
mod debug_panel;
/// The dev/player seam: the instrument groups and the boundary rule. Always compiled; what it holds
/// is not.
mod dev;
mod doodad_events;
mod entities;
mod fishing_line;
mod flavor;
mod footprints;
mod game_plugins;
mod glue;
mod glue_strings;
mod go_anim;
mod go_templates;
mod hd_models;
#[cfg(feature = "dev")]
mod hover_log;
mod items;
mod loading_screen;
mod local_state;
mod login;
mod minimap;
mod nameplates;
mod names;
mod net;
mod npc_text;
mod pending_item_ops;
/// Ships in part (the FPS journal and the clocks it reads); the rest is `dev`.
mod perf;
mod pipe_warm;
mod player;
mod player_shadow;
mod poi_marker;
mod portrait;
#[cfg(feature = "dev")]
mod preflight;
#[cfg(feature = "dev")]
mod probe_shield;
mod query_cache;
mod quest_markers;
mod raid_marks;
mod ranged_flex;
mod realm_select;
mod realmlist;
mod run_mode;
mod screen_fade;
mod screenshot;
mod script_calls;
mod shaders;

/// Every boot-time reader of the chain rebuilds with it or says why not; read off the source.
#[cfg(test)]
mod data_load_guard_tests;
mod game_tip;
mod name_persist;
mod opaque2d;
/// Where "the client is going down" is observed, in `Last`; every system that persists state on the
/// way out registers through it.
mod shutdown;
mod smart_rect;
mod sound;
/// The two talent spell-modifier tables (`SMSG_SET_FLAT_/PCT_SPELL_MODIFIER`) and the read that
/// puts them on a number.
mod spell;
/// The melee swing refusal's latch + 4 s repeat (`SMSG_ATTACKSWING_*`).
mod swing_refusal;
mod target;
#[cfg(test)]
pub(crate) mod test_support;
mod text_filter;
mod text_reshape;
mod textinput;
mod transport;
mod tutorial;
mod ui_action;
mod ui_auction;
mod ui_aura;
mod ui_bank;
mod ui_battlefield;
mod ui_battlefield_positions;
mod ui_battlefield_score;
mod ui_bind_confirm;
mod ui_binder;
mod ui_cast;
mod ui_char;
mod ui_chat;
mod ui_craft;
mod ui_dialog_verbs;
mod ui_dressup;
mod ui_duel;
mod ui_follow;
mod ui_gamma;
mod ui_gm_ticket;
mod ui_gossip;
mod ui_guild;
mod ui_hide;
mod ui_honor;
mod ui_inspect;
mod ui_instance;
mod ui_item_text;
mod ui_items;
mod ui_layout;
mod ui_logout;
mod ui_loot;
mod ui_loot_roll;
mod ui_macro;
mod ui_mail;
mod ui_merchant;
mod ui_mirror;
mod ui_models;
mod ui_net;
mod ui_party;
mod ui_pass;
mod ui_pet;
mod ui_pet_book;
mod ui_pet_doll;
mod ui_pet_stats;
mod ui_petition;
mod ui_quest;
mod ui_quest_log;
mod ui_quest_share;
mod ui_reputation;
mod ui_saved;
mod ui_script;
mod ui_session;
mod ui_shapeshift;
mod ui_social;
mod ui_spellbook;
mod ui_stable;
mod ui_summon;
mod ui_tabard;
mod ui_talent;
mod ui_talent_wipe;
mod ui_taxi;
mod ui_text;
mod ui_tooltip;
mod ui_trade;
mod ui_tradeskill;
mod ui_trainer;
mod ui_unit;
mod ui_world_map;
mod video;
mod vplates;
mod weapon_trail;
/// The web build's env-var stand-in (a browser tab has no process environment) — see the
/// module doc. `pub` because the plan's "Env/config on web" interface names it as
/// `benilla_app::webenv::var` by that exact path; every in-crate caller still just writes
/// `crate::webenv::var` like any other module.
pub mod webenv;
// The browser stutter probe. NOT under `dev` (which is off in the web build, `--no-default-
// features --features webgpu`) — it is the web build's own instrument, and gating it behind the
// feature that compiles the desktop instrument stack out is what would make it unshippable.
mod webdiag;
mod webgpu;
mod webprogress;
// Web Worker threads for the browser build's task pools — the platform half of the forked
// `bevy_tasks`'s thread-spawner hook. Inert on every other target, and inert on a browser build
// without shared memory.
mod web_threads;
mod world_backdrop;
mod world_state;
mod world_state_ui;

use bevy::prelude::*;

// A launcher stamps the build id at compile time and hands it to [`run`] or [`run_with`];
// re-exported so the `benilla` shim needs no bevy dependency of its own.
pub use benilla_world::build_id::BuildId;
/// fork: copy the page's engine-side query keys (`?lighting=modern`, `?water_detail=1`, …) into
/// `benilla_world::launch_env`, which is how the engine reads them in the browser. Both entry
/// points call it: the client used to skip it, so `?lighting=modern` only reached the world view.
fn seed_launch_env() {
    const SEEDED: [&str; 41] = [
        "WOW_MAP",
        "WOW_MODERN_FOGLOG",
        "WOW_MODERN_LOOK",
        // fork: the Modern (Experimental) water's live look (`?water_look=`, `liquid::sync_water_look`).
        "WOW_WATER_LOOK",
        "WOW_FARCLIP",
        "WOW_WATER_DETAIL",
        "WOW_MODERN_ZONE",
        "WOW_WORLDVIEW_AT",
        "WOW_LIGHTING",
        "WOW_LIGHTING_QUALITY",
        "WOW_CLOCK",
        "WOW_MODERN_SUNARC",
        "WOW_MODERN_CPUHOOKS",
        "WOW_MODERN_SHADOWDEBUG",
        "WOW_MODERN_GI",
        "WOW_MODERN_GI_GAIN",
        "WOW_MODERN_LIGHTS",
        // fork: P8 — the Cinematic tier's emissive-source lever (`?modern_emissive=0`): the A/B for
        // "is this lamp's light authored, or did we derive it".
        "WOW_MODERN_EMISSIVE",
        // fork: the lamp-shadow bisect levers (`modern::lights::lamp_levers`): `?lamp_dynamic=0`,
        // `?lamp_taps=1`, `?lamp_tile=0.5`, `?lamp_shadowed=0`.
        "WOW_LAMP_DYNAMIC",
        "WOW_LAMP_TAPS",
        "WOW_LAMP_TILE",
        "WOW_LAMP_SHADOWED",
        // fork: the flame flicker's gain (`?fire_flicker=0` steady, `1` default, `2` doubled —
        // `lighting::flicker`) and the spell-light kill switch (`?spell_light=0`,
        // `entities::spell_fx::light`).
        "WOW_FIRE_FLICKER",
        "WOW_SPELL_LIGHT",
        "WOW_MODERN_MIRRORTIER",
        "WOW_MODERN_CASTERREACH",
        "WOW_WORLDVIEW_FACE",
        "WOW_WORLDVIEW_PITCH",
        // fork: P8 — the light-pool probe (`?points_dump=1`), `global_light`'s dump of what is
        // actually lighting this ground, which could not reach the browser at all: this list is
        // an allowlist, so a key missing from it is simply never seeded and the probe stays
        // silent — which reads exactly like the code not running.
        "WOW_POINTS_DUMP",
        // fork: modern lighting — the combine's deband dither (`?dither=1`), off by default like
        // the native knob. It could not be reached from the page at all before: the reader took
        // `std::env`, which on wasm is empty, so it read exactly like the dither not existing.
        "WOW_DITHER",
        // …and the emission front end's own (`?emit_dump=<label>[,period]`): which sequence slot an
        // emitter resolved and what its ten per-frame tracks sampled there. The dump that answers
        // "this effect moves wrong" without guessing at it.
        "WOW_EMIT_DUMP",
        // …and the viewer's own anchor (`?worldview_at=x,y[,z]`), without which the browser can
        // only ever look at the one default spot — and an interior like Northshire Abbey is
        // portal-culled from outside it, so the bug being chased was not on screen at all.
        "WOW_WORLDVIEW_AT",
        // fork: the perf-rig levers. Each names a pass the modern frame pays for, and none of them
        // reached the browser before — the readers took `std::env`, which on wasm32 always answers
        // NotPresent, so the one build whose GPU sets the project's budget had no way to A/B its
        // own passes. `?ssao=0`, `?msaa=`, `?static_gx=0` and the two SSILVB shape knobs.
        "WOW_SSAO",
        "WOW_SSGI",
        "WOW_SSILVB_RADIUS",
        "WOW_MSAA",
        "WOW_STATIC_GX",
        // …and the probe's own three (`benilla_world::perf_probe`): arm, resolution multiplier,
        // and the name this configuration answers to in the output line.
        "WOW_PERF",
        "WOW_PERF_WARMUP",
        "WOW_SUPERSCALE",
        "WOW_PERF_LABEL",
    ];
    benilla_world::launch_env::seed(
        SEEDED
            .iter()
            .filter_map(|k| webenv::var(k).map(|v| ((*k).to_string(), v))),
    );
}

/// The world viewer — `benilla-worldview`'s engine-only run, reachable from the browser too
/// (`?worldview=1`, fork-owned). The native shim calls `benilla_world::worldview::run` directly;
/// this wrapper adds what only the client crate can supply on the web: the worker pool, the query
/// string (copied into `benilla_world::launch_env` for the keys the engine reads), and the
/// loading overlay's `ready` edge, which the viewer has no glue screen to send.
pub fn run_worldview(build: BuildId) -> AppExit {
    web_threads::install();
    seed_launch_env();
    // fork: modern lighting — dev tuning knobs follow the live query string.
    benilla_world::launch_env::set_live_reader(webenv::var);
    webprogress::signal_ready_once();
    benilla_world::worldview::run(build)
}

/// Did the page ask for the world viewer instead of the client (`?worldview=1`)?
pub fn wants_worldview() -> bool {
    webenv::var("WOW_WORLDVIEW").as_deref() == Some("1")
}
pub use bevy::app::AppExit;

/// Take `LogPlugin` off the **web** build — always in a shipped one, and in a debug one while the
/// stutter probe is armed, so the browser console carries the probe's lines and nothing else.
///
/// A browser console has no `grep` and no scrollback a reader can pipe: the one thing a player can
/// do with it is select all and paste, and at `info` the client writes enough per second that a
/// probe line is unfindable in the paste. So diagnostic mode owns the console outright —
/// [`webdiag`] writes through `console.log` directly, precisely so it still speaks with no
/// subscriber installed. In a debug build `?diag=0` puts the ordinary log back (and silences the
/// probe); in a shipped one the log is gone either way and `?diag=1` is what turns the probe on.
///
/// Native is never touched: [`webdiag::armed`] is false off wasm32, so a `cargo run` logs as it always
/// has. Panics still reach the console either way — `wenilla`'s shim sets `console_error_panic_hook`
/// itself, before this crate is entered.
fn quiet_console_on_web(plugins: bevy::app::PluginGroupBuilder) -> bevy::app::PluginGroupBuilder {
    // **A shipped browser build writes nothing to the console.** Two reasons it is off rather than
    // merely quiet: every line is a wasm→JS bridge call, and `LogPlugin`'s web subscriber makes one
    // per event with no batching — so an `info!` in a per-frame system is a per-frame crossing that
    // nobody reads. `?diag=1` swaps the ordinary log for the probe (which owns the console when it
    // is on — see `webdiag::armed`), and neither is on by default here.
    //
    // A DEBUG wasm build keeps the old rule: the log unless the probe is armed. That is the build a
    // person is looking at a console for.
    let web_release = cfg!(target_arch = "wasm32") && !cfg!(debug_assertions);
    if web_release || webdiag::armed() {
        return plugins.disable::<bevy::log::LogPlugin>();
    }
    plugins
}

/// Builds and runs the client app. `build` is the launcher's compile-time git stamp, passed in as
/// data so the sha lives in the shim's fingerprint and a commit does not recompile this crate.
pub fn run(build: BuildId) -> AppExit {
    launch(build, None)
}

/// [`run`], plus `extend` adding a crate's own plugins: how a feature 1.12.1 lacks is built on top
/// of benilla (`examples/extended_launcher.rs`), the build marked `extended` wherever it is named.
/// `extend` gets the client fully built and runs before the probe fleet, which sees what it adds.
pub fn run_with(build: BuildId, extend: impl FnOnce(&mut App)) -> AppExit {
    launch(build, Some(Box::new(extend)))
}

/// What a crate on top of benilla adds to the built app ([`run_with`]).
type Extension<'a> = Box<dyn FnOnce(&mut App) + 'a>;

/// The one body behind [`run`] and [`run_with`], not generic, so a launcher compiles none of it.
fn launch(build: BuildId, extend: Option<Extension<'_>>) -> AppExit {
    // `WOW_HOVER_LOG_REPORT=<csv>` and `WOW_CAPTURE=list` print and exit before any setup.
    if let Ok(path) = std::env::var("WOW_HOVER_LOG_REPORT") {
        dev::report_recorded_hover_log(&path);
        return AppExit::Success;
    }
    if std::env::var("WOW_CAPTURE").as_deref() == Ok("list") {
        dev::print_scenario_names();
        return AppExit::Success;
    }
    // `WOW_PROBE=list` likewise prints the probe environment registry.
    if std::env::var("WOW_PROBE").as_deref() == Ok("list") {
        dev::print_probe_vars();
        return AppExit::Success;
    }

    // Every line that names the build says whether a crate on top extended it.
    let build = BuildId {
        extended: extend.is_some(),
        ..build
    };

    // From here a panic leaves `benilla-config/Diagnostics/crash-<unix>.txt`; armed before the
    // `App` exists, so a panic while plugins build is reported too.
    //
    // Native only (fork: web build). A page has no `benilla-config/Diagnostics/` to write a report
    // into, `std::time::Instant`/`SystemTime` panic on wasm32 (a panic *inside* the hook is an
    // abort), and `wenilla`'s shim already routes panics to the browser console through
    // `console_error_panic_hook` — which is where a web bug report's backtrace comes from.
    #[cfg(not(target_arch = "wasm32"))]
    crash::install(build);

    // **Before anything builds a task pool.** `TaskPoolPlugin` creates all three inside
    // `Plugin::build`, so the spawner and the thread count have to be in place ahead of
    // `add_plugins`, not merely early. Compiles to nothing off a shared-memory browser build.
    web_threads::install();
    // Before any resource reads `launch_env` (`LightingSettings`, `WaterDetail`, …).
    seed_launch_env();

    // The offload lever, read once and applied before the first asset can be asked for.
    // `?offload=0` puts every `spawn_pooled` caller back on the frame thread under the asset
    // budget — the exact behaviour the browser build had before it had workers — so a page that
    // turns out to disagree with the cross-worker await has a way out that is not a re-upload. See
    // `benilla_assets::frame_budget::set_offload_enabled` for why this one has a switch and the
    // rest of that module does not.
    benilla_assets::frame_budget::set_offload_enabled(
        webenv::var("WOW_OFFLOAD").as_deref() != Some("0"),
    );

    let mut app = App::new();
    // The panel footer and the preflight banner read the stamp back.
    app.insert_resource(build);
    // Static-scene transform tracking pinned on: the default threshold re-decides every frame with
    // two full scans of the rows the tracking exists to skip, and this scene is static-heavy. It
    // costs more only when most rows move in one frame (a load burst into a near-empty world).
    app.insert_resource(bevy::transform::systems::StaticTransformOptimizations::enabled());
    // Update runs single-threaded: most per-frame Update systems are non-Send (mlua's `UiScript`,
    // kira's audio handles) and serialize through the multi-threaded executor anyway, so its
    // dispatch is pure overhead (measured faster, tails flat). `WOW_MT_UPDATE=1` restores it.
    if std::env::var_os("WOW_MT_UPDATE").is_none() {
        app.edit_schedule(Update, |s| {
            s.set_executor_kind(bevy::ecs::schedule::ExecutorKind::SingleThreaded);
        });
    }
    // PostUpdate too: 208 systems paid about 10 µs each of multi-threaded dispatch, and
    // single-threaded measured faster parked and in motion. `WOW_MT_POSTUPDATE=1` restores it; the
    // log line names the config a run measured.
    if std::env::var_os("WOW_MT_POSTUPDATE").is_none() {
        app.edit_schedule(PostUpdate, |s| {
            s.set_executor_kind(bevy::ecs::schedule::ExecutorKind::SingleThreaded);
        });
        println!(
            "executor: PostUpdate -> single-threaded (the default; WOW_MT_POSTUPDATE=1 for MT)"
        );
    }
    // The build banner ships in every build: which build produced a log is the first thing a report
    // from another machine must establish.
    app.add_systems(Startup, benilla_world::build_id::banner);

    // With `$WOW_CAPTURE` set the app runs a deterministic, server-less capture (net off, so no
    // NPCs stream in) and exits.
    let capturing = run_mode::scenario_active();
    // Every instrumented run, captures and live probes, opens in the background so it never takes
    // over a person's screen; `WOW_BG` overrides.
    let background = benilla_world::bgwin::background_run();
    if capturing {
        // Ground clutter scatters with per-run randomness, so captures turn it off for byte-stable
        // baselines; set before plugins build so `ClutterConfig::from_env` reads it. The frame
        // clock, frozen in `capture`, is the other source of drift.
        std::env::set_var("WOW_CLUTTER_DENSITY", "0");
        // Anim-LOD park/wake (`creature_anim::lod::gate_rig_animation`) hangs on asset-load timing,
        // which the frozen clock does not control, so captures turn it off unless overridden. A rig
        // in frame should never be parked, so the shot keeps its subject.
        if std::env::var("WOW_NO_ANIM_LOD").is_err() {
            std::env::set_var("WOW_NO_ANIM_LOD", "1");
        }
    }

    // The WebSocket gateway, on the web target only. Until now the transport always built its URL
    // from the page's own origin, which was right while `wenilla-host` served the bundle and
    // proxied `/ws/{port}` from one process. On a static site those separate: the page comes from
    // an object store with no sockets, and the gateway is a box elsewhere whose address is supplied
    // after the client is built — hence `?ws=` / `__wenilla_env.ws` / `/config.json`, and not a
    // constant compiled into 90 MB of wasm. Absent, the origin default stands and local dev is
    // unchanged. (This is `set_ws_base`'s first caller; it had none before.)
    #[cfg(target_arch = "wasm32")]
    if let Some(base) = webenv::ws_base() {
        benilla_protocol::transport::web::set_ws_base(base);
    }

    // **Which optional content layers this run mounts**, resolved before ANY chain opens — the
    // `mpq://` source below is the first of two, and a chain decides its mounts at open. See
    // `cvars::apply_boot_layers` and `benilla_formats::layers`.
    crate::cvars::apply_boot_layers();

    // The `mpq://` source must be registered before `AssetPlugin` (in `DefaultPlugins`) builds. The
    // install is found by `benilla_formats::wow_data`, as everywhere; without one the source is
    // absent and terrain loads fail gracefully.
    match benilla_formats::wow_data() {
        Some(data_dir) => {
            if let Err(e) = benilla_assets::register_mpq_source(&mut app, &data_dir) {
                eprintln!("benilla-assets: mpq:// source unavailable ({e:#})");
            }
        }
        None => eprintln!(
            "benilla: no WoW install found — looked in {:?}",
            benilla_formats::candidates()
        ),
    }

    // No `game://` source: the five UI shaders are compiled in by `crate::shaders`
    // (`embedded://benilla_app/shaders/…`), so no build-machine path reaches the binary.

    app.add_plugins(quiet_console_on_web(
        benilla_world::boot::tuned_default_plugins(Window {
            title: "World of Warcraft".into(),
            // Born in the player's display mode (`gxWindow` read straight off `config.toml`) rather
            // than flipped into it at `Startup`, which would flash on every launch and, under
            // gamescope, spend the first second in the input state fullscreen is meant to end.
            // Instrumented runs stay windowed (`video::windowed_env`).
            mode: video::boot_window_mode(),
            // UI-fixture captures size the window to what they photograph: the action bar's 1024 px
            // plus 128 px end caps get a wide, short window, and vplates pins 1024×768, where one gx
            // unit is 1280 px and the plate lands at the border texture's native 128×32, diffable
            // against the BLP.
            resolution: video::at_requested_dpi(
                // Same opt-in as the UI load (`ui_script::lifecycle::ui_wanted`); the two must agree,
                // or the right content is captured at the wrong size.
                if capturing && crate::run_mode::capture_ui_opted_in() {
                    // `$WOW_WIN` overrides here too, for scale-dependent UI bugs.
                    if let Some(win) = video::requested_window_size() {
                        win
                    } else {
                        match std::env::var("WOW_CAPTURE").as_deref() {
                            Ok("ui-actionbar") => UVec2::new(1300, 260),
                            Ok("vplates") => UVec2::new(1024, 768),
                            // Short enough for the action bar to overlap the chat edit box.
                            Ok("ui-chatedit") => UVec2::new(566, 377),
                            // The map's chrome is a centered 1024×768 block, with margin.
                            Ok("ui-worldmap") => UVec2::new(1100, 800),
                            // The 920×724 era options window wants air on every side so the straddling
                            // right-edge tile and the hung close X stay in frame.
                            Ok("ui-options")
                            | Ok("ui-options-audio")
                            | Ok("ui-options-graphics") => UVec2::new(1200, 900),
                            _ => UVec2::new(640, 700),
                        }
                    }
                    .into()
                } else {
                    // `$WOW_WIN=WxH` (logical px) overrides the size. FFXGlow's blur is pinned in
                    // texels, so thin bright features self-amplify at high resolution; matching the
                    // reference's pixel density (`WOW_WIN=512x288` on a 2× display is 1024×576)
                    // isolates that term.
                    video::requested_window_size()
                        // A run that reads no pixels gets a small window: it is held `AlwaysOnTop`
                        // against the occlusion throttle (`capture::ProbeFocusPlugin`), and small and
                        // cornered keeps it out of the way. Pixel runs keep the full size.
                        .unwrap_or(if benilla_world::bgwin::no_pixel_run() {
                            UVec2::new(640, 360)
                        } else {
                            // The player's `gxResolution`; `bevy_winit` ignores it while fullscreen.
                            video::boot_windowed_size()
                        })
                        .into()
                },
            ),
            // `$WOW_NOVSYNC=1` uncaps presentation from boot so a headless FPS-journal run measures
            // frame cost, not the vsync ceiling; otherwise the player's `gxVSync` takes over from
            // `Startup` ([`crate::video`]).
            present_mode: video::present_mode(!video::novsync_env()),
            // A background run opens unfocused, so it cannot take keystrokes meant for another app, and
            // is born `AlwaysOnBottom` (`kCGNormalWindowLevel - 1`), so winit's two raises before the
            // first frame cannot flash it on top. `BgWinPlugin` promotes it to Normal once the launch
            // settles, so it can still be raised.
            focused: !background,
            window_level: if background {
                bevy::window::WindowLevel::AlwaysOnBottom
            } else {
                bevy::window::WindowLevel::Normal
            },
            ..default()
        }),
    ))
    .add_plugins(benilla_world::thread_qos::ThreadQosPlugin)
    // Undoes winit's forced macOS app activation for background runs; the window-side half is the
    // `Window` above.
    .add_plugins(benilla_world::bgwin::BgWinPlugin)
    // macOS `Cmd+Q` goes straight to `terminate:`, which never runs another frame, so the session
    // was never written; re-pointed at the window close, which [`shutdown`] sees.
    .add_plugins(benilla_world::mac_quit::MacQuitPlugin)
    // The engine as one group; `world_plugins.rs` has its load-bearing ordering edges and what it
    // leaves out (`pipe_warm`).
    .add_plugins(benilla_world::world_plugins::WorldPlugins)
    // The instruments, which the engine group does not carry; `--no-default-features` compiles them
    // all out (see `dev.rs`).
    .add_plugins(dev::DevToolsPlugin)
    // The browser stutter probe (`webdiag`), which the group above cannot carry: that group is
    // `dev`-gated and the web build compiles it out. After `WorldPlugins`, because the probe's
    // `Update` stamps order against `WorldStage` and its marker schedules splice around the
    // `StateTransition` that `StatesPlugin` seeded.
    .add_plugins(webdiag::plugin)
    // The FPS journal ships: `/console fpsJournal 1` appends a per-second row of position, frame
    // cost and the GPU's per-pass split to `benilla-config/Diagnostics/fps-journal.csv`;
    // `WOW_FPS_JOURNAL=<csv>` is the harness lever.
    .add_plugins(perf::FpsJournalPlugin)
    // The game as one group on top of the engine; `game_plugins.rs` has its members, its
    // load-bearing ordering edges and the test that builds it headless.
    .add_plugins(game_plugins::GamePlugins {
        connect: !capturing,
        start: run_mode::start_state(),
    });

    // benilla-assets' loaders go into the live `AssetServer`, so they register after `AssetPlugin`.
    benilla_assets::register_asset_loaders(&mut app);

    // `ExtractSchedule` runs single-threaded too: bevy_render leaves it multi-threaded, with no
    // non-Send member, and single-threaded measured faster. `WOW_MT_EXTRACT=1` restores it.
    //
    // `Render` stays multi-threaded: under pipelined rendering that executor is also bevy's route
    // for non-Send systems to the main thread, and `create_surfaces` needs it because macOS makes a
    // Metal layer only on the UI thread (the single-threaded executor panics at startup).
    //
    // **On wasm all of this is already true and none of it is a choice.** `ExecutorKind`'s default
    // is `SingleThreaded` on `target_arch = "wasm32"`, and `third_party/bevy_ecs` deliberately does
    // NOT touch that gate (only the `par_iter` ones) — so every schedule the fork does not name,
    // `Render` and `ExtractSchedule` included, is single-threaded there whatever this block does.
    // That is required rather than merely convenient: under `+atomics` `bevy_render`'s
    // `WgpuWrapper` is a `send_wrapper::SendWrapper`, which panics the moment its value is touched
    // or dropped off the thread that made it, and the render app's thread is the page's. The other
    // half of the same guarantee is upstream's: `PipelinedRenderingPlugin` is gated on a bare
    // `not(target_arch = "wasm32")`, so it never joins the graph and there is no render thread for
    // anything to be routed to.
    //
    // Set here because the render sub-app exists only once the plugin chain has built.
    if std::env::var_os("WOW_MT_EXTRACT").is_none() {
        if let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) {
            render_app.edit_schedule(bevy::render::ExtractSchedule, |s| {
                s.set_executor_kind(bevy::ecs::schedule::ExecutorKind::SingleThreaded);
            });
            println!(
                "executor: ExtractSchedule -> single-threaded (the default; WOW_MT_EXTRACT=1 for MT)"
            );
        } else {
            // A missing sub-app would silently flip nothing.
            eprintln!("executor: no render app — ExtractSchedule flip NOT applied");
        }
    }

    // Boot-progress bridge (webprogress.rs): tell the hosting page the Startup pile is done.
    // PostStartup runs at the tail of the same long first-update task the catalogs block in, so
    // this is the earliest signal the page can actually paint on.
    app.add_systems(PostStartup, webprogress::signal_startup);

    // A crate on top of benilla adds its plugins here ([`run_with`]).
    if let Some(extend) = extend {
        extend(&mut app);
    }

    // The probe fleet, last so it observes the fully-built app; compiled out by
    // `--no-default-features`.
    app.add_plugins(dev::DevProbesPlugin);

    // Returns the app's exit status: a failed capture sets `AppExit::error()`
    // (`capture::drive_capture`), which must not exit 0 with no PNG on disk.
    app.run()
}
