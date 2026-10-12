//! The UI session's lifecycle, all edges and nothing per-frame: `Startup` installs a boot VM
//! (strings and emote tokens), or the stock glue screens' VM under `WOW_GLUE_STOCK=1`; world entry
//! builds a fresh VM and loads the in-game UI and every addon onto it; leaving the world runs the
//! reference's shutdown tail and installs the glue-side VM again; `ReloadUI()` is the two edges
//! back to back without leaving the world.

use bevy::prelude::*;

use benilla_assets::LockRecover;
use benilla_ui::script::{GlueNotice, UiScript};

use super::manifest::silenced_ui_load;
use super::{load_ingame_ui, CursorPayloadHeld, PlayerUiHover, UiClock, UiKeyboardCapture};
use crate::ui_script::addons;

pub(crate) fn setup_script(world: &mut World) {
    install_glue_side_vm(world, super::glue::GlueScreen::Login);
}

/// Install the VM the world holds outside the world: the stock glue screens' under
/// [`super::glue::GlueStock`], put up on `screen`, else a boot VM.
///
/// The IO thread's roster policy follows the VM that is up: the screen asks for its roster only
/// when the stock glue VM was built and its character select screen loaded, so a run whose stock VM failed to build (logged as an error
/// by [`new_vm`]) falls back to the park's own enumeration instead of waiting on a screen that
/// never asks.
fn install_glue_side_vm(world: &mut World, screen: super::glue::GlueScreen) {
    let stock_asked = world
        .get_resource::<super::glue::GlueStock>()
        .is_some_and(|s| s.0);
    let stock_up = if stock_asked {
        install_glue_vm(world, screen)
    } else {
        install_boot_vm(world);
        false
    };
    if stock_asked && !stock_up {
        error!("ui_script: WOW_GLUE_STOCK=1 but the stock glue VM did not come up with its character select screen; the roster is fetched at the park");
    }
    if let Some(policy) = world.get_resource::<crate::net::RosterPolicy>() {
        policy
            .0
            .store(stock_up, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Build and install a boot VM, the one the world holds outside an in-game UI session: the
/// strings the feeds resolve lines from and the emote tokens `/`-commands are built from
/// (`ui_chat::commands`), no frames and no font objects. Installed at `Startup` and at every
/// [`end_ui_session`], unless the stock glue screens stand there instead ([`install_glue_vm`]).
fn install_boot_vm(world: &mut World) {
    if let Some(script) = strings_vm(world) {
        world.insert_non_send(script);
    }
}

/// A boot VM's contents on a VM of their own: `GlobalStrings.lua` and the emote tokens, for a
/// reader that needs them where the world holds the glue VM. `None` (the world's VM removed) when
/// the VM cannot start.
pub(crate) fn strings_vm(world: &mut World) -> Option<UiScript> {
    let script = new_vm(world, UiScript::new)?;
    load_global_strings(world, &script);
    load_emote_tokens(world, &script);
    Some(script)
}

/// Build the glue VM, load the stock GlueXML onto it, install it and put `screen` up: the glue
/// builder `0x46a7b0` then `SET_GLUE_SCREEN` (`0x402b65` at startup, `0x402090` back from the
/// world). It holds what the glue state holds and nothing of the boot VM's: no `GlobalStrings.lua`,
/// no emote tokens.
fn install_glue_vm(world: &mut World, screen: super::glue::GlueScreen) -> bool {
    let Some(mut script) = new_vm(world, UiScript::new_glue) else {
        return false;
    };
    seat_raster_seam_for_load(world, &mut script);
    seed_agreements(world, &mut script);
    let start = bevy::platform::time::Instant::now();
    let failures = super::glue::load_glue(&mut script);
    super::glue::finish_glue_load_log(&mut script);
    start_server_alert(world, &script);
    super::glue::set_glue_screen(&mut script, screen);
    info!(
        "ui_script: stock glue up in {:.0} ms on \"{}\" ({} load failure(s))",
        start.elapsed().as_secs_f32() * 1000.0,
        screen.key(),
        failures.len()
    );
    // Up means the screen that asks for the roster exists: the stock files loaded far enough to
    // define the character select frame and its `OnShow`, not merely that a VM started.
    let charselect = script
        .eval::<bool>(
            "return type(CharacterSelect) == 'table' and type(CharacterSelect_OnShow) == 'function'",
        )
        .unwrap_or(false);
    world.insert_non_send(script);
    charselect
}

/// Start the server alert's fetch, as the glue manager's init does after the GlueXML load and
/// before the first screen (`0x46a541`-`0x46a564`, `0x402b36`): the global `SERVER_ALERT_URL` as
/// the glue files left it, so `GlueLocalization` wins over `GlueStrings`, fetched once per
/// process ([`super::server_alert::begin_once`]).
fn start_server_alert(world: &mut World, script: &UiScript) {
    let url = script.global_text("SERVER_ALERT_URL");
    if let Some(fetch) = super::server_alert::begin_once(&url) {
        if let Some(mut mgr) = world.get_resource_mut::<super::glue_mgr::GlueMgr>() {
            mgr.set_alert_fetch(fetch);
        }
    }
}

/// Read the four agreement CVars into the glue VM and the manager before any screen is set, as
/// the reference does (`0x46a400` at `0x402b36`, then `SET_GLUE_SCREEN` at `0x402b65`): the login
/// screen's `OnShow` asks them at once, and nothing asks again.
fn seed_agreements(world: &mut World, script: &mut UiScript) {
    let Some(cvars) = world.get_resource::<crate::cvars::Cvars>() else {
        return;
    };
    let notices = super::glue_mgr::read_agreements(cvars);
    for notice in GlueNotice::ALL {
        script.set_glue_notice(notice, notices[notice as usize]);
    }
    if let Some(mut mgr) = world.get_resource_mut::<super::glue_mgr::GlueMgr>() {
        mgr.set_notices(notices);
    }
}

/// A fresh VM from `make` on the process clock with the addon asset probes wired, or `None` (the
/// world's VM removed) when the VM cannot start.
fn new_vm<E: std::fmt::Display>(
    world: &mut World,
    make: fn() -> Result<UiScript, E>,
) -> Option<UiScript> {
    let mut script = match make() {
        Ok(s) => s,
        Err(e) => {
            error!("ui_script: VM init failed: {e}");
            world.remove_non_send::<UiScript>();
            return None;
        }
    };
    // Before anything loads: the flavor decides the core's rows, `GetBuildInfo` and the script
    // kinds.
    let flavor = world
        .get_resource::<crate::flavor::ClientFlavor>()
        .copied()
        .unwrap_or_default();
    script.set_flavor(flavor.ui());
    seed_vm_clock(world, &mut script);
    install_addon_asset_resolvers(world, &mut script);
    info!("ui_script: new VM, session {}", script.session());
    Some(script)
}

/// Start the new VM's `GetTime()` where the process already is and re-anchor [`UiClock`] to it,
/// both off the `Time<Real>` it anchors on. The reference's `GetTime` is `GetTickCount` × 0.001
/// (`0x515ea0`, via `0x42c010` → `0x42b790`), an OS clock that never restarts; ours lives in the
/// VM, rebuilt at every login and `ReloadUI`, and a clock back at zero puts every converted
/// cooldown start in the past, which stock `CooldownFrame_SetTimer` hides (`Cooldown.lua:3`).
fn seed_vm_clock(world: &mut World, script: &mut UiScript) {
    let (anchor, elapsed) = world
        .get_resource::<Time<bevy::time::Real>>()
        .map_or_else(Default::default, |t| {
            (t.last_update(), t.elapsed_secs_f64())
        });
    script.set_now(elapsed);
    if let Some(mut clock) = world.get_resource_mut::<UiClock>() {
        *clock = UiClock {
            // `bevy::platform::time::Instant`, not `std::time::Instant`: `last_update()` hands
            // back the former, and on wasm32 the two are genuinely different types
            // (`web_time::Instant` there), so the std spelling does not even type-check on the
            // page — and would panic if it did. Same rule as [`UiClock::anchor`] itself.
            anchor: anchor.unwrap_or_else(bevy::platform::time::Instant::now),
            ui_now: elapsed,
        };
    }
}

/// Wire `Interface\AddOns\` art and fonts: the decoder's loose-file root and the VM's texture,
/// size and font probes, so the path forms of `SetTexture` and `SetFont` answer the reference's
/// 1|nil load verdict, and a region with no authored size on an axis takes it from its art as the
/// client's virtual size getters do. Each probe resolves [`addons::root`] through the store its
/// renderer reads, so the answer is what draws; they test existence, not decode.
fn install_addon_asset_resolvers(world: &mut World, script: &mut UiScript) {
    let root = addons::root();
    let Some(mut assets) = world.get_resource_mut::<benilla_assets::WorldAssets>() else {
        return;
    };
    assets.set_loose_addon_root(root.clone());
    let chain = assets.chain.clone();
    let size_chain = chain.clone();
    let size_root = root.clone();
    let font_chain = chain.clone();
    let font_root = root.clone();
    script.set_texture_probe(Box::new(move |path| {
        benilla_assets::sprite_candidates(path).iter().any(|c| {
            chain.lock_recover().contains(c)
                || root
                    .as_deref()
                    .is_some_and(|r| benilla_assets::loose_addon_file(r, c).is_some())
        })
    }));
    // Keyed by the path exactly as the region carries it, so a hit allocates nothing (this is
    // asked per zero-size textured region per layout pass); misses are cached too.
    let sizes: std::cell::RefCell<std::collections::HashMap<String, Option<(u32, u32)>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    script.set_texture_size_probe(Box::new(move |path| {
        if let Some(cached) = sizes.borrow().get(path) {
            return *cached;
        }
        match benilla_assets::sprite_dimensions(&size_chain, size_root.as_deref(), path) {
            benilla_assets::SpriteDims::Ready(w, h) => {
                sizes.borrow_mut().insert(path.to_string(), Some((w, h)));
                Some((w, h))
            }
            benilla_assets::SpriteDims::Missing => {
                sizes.borrow_mut().insert(path.to_string(), None);
                None
            }
            // In flight (web target): the region measures nothing this pass and is asked again on
            // the next one. Emphatically NOT memoised — this cache is for the session, and a number
            // taken before the art arrived would be the number the region keeps forever.
            benilla_assets::SpriteDims::Pending => None,
        }
    }));
    // Existence over the same two stores, in the same order, as `ui_text::engine`'s face loader,
    // memoised because a path's answer cannot change and an addon may call `SetFont` every frame.
    let seen: std::cell::RefCell<std::collections::HashMap<String, bool>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    script.set_font_probe(Box::new(move |path| {
        if let Some(&cached) = seen.borrow().get(path) {
            return cached;
        }
        let found = font_chain.lock_recover().contains(path)
            || font_root.as_deref().is_some_and(|r| {
                benilla_assets::loose_addon_file(r, &benilla_assets::normalize_path(path)).is_some()
            });
        seen.borrow_mut().insert(path.to_string(), found);
        found
    }));
}

/// Whether this run wants the player UI: always outside a capture, and in one only when opted in
/// ([`crate::run_mode::capture_ui_opted_in`]), since capture baselines test the world render.
fn ui_wanted(world: &World) -> bool {
    !world.contains_resource::<crate::run_mode::CaptureMode>()
        || crate::run_mode::capture_ui_opted_in()
}

/// The world-entry UI load still owed: armed at `OnEnter(InWorld)`, run by
/// [`run_pending_entry_load`] once the loading cover is on screen. `OnEnter(InWorld)` runs on the
/// frame that first presents the cover, so a load there would freeze the character screen for its
/// whole duration. The wait is [`crate::loading_screen::EntryCover`]'s, and the loading
/// screen does not clear while this exists, so the reveal never precedes the UI.
///
/// The load is a resumable state machine: [`ENTRY_LOAD_FRAME_BUDGET`] of it per frame, at least one
/// step each, so the cover keeps presenting, its bar moves ([`Self::fraction`]) and a browser tab
/// stays responsive. On the page the whole edge was one main-thread task of 3.0-3.4 s warm.
#[derive(Resource, Default)]
pub(crate) struct PendingEntryUiLoad {
    stage: EntryStage,
    /// Steps done / total, for the loading bar. `total` is known after the first step.
    done: usize,
    total: usize,
    started: Option<bevy::platform::time::Instant>,
    frames: u32,
    covering: bool,
    identity: Option<(String, String)>,
    /// Every character on this realm's list — the AddOn enable store's node set (decision 2311).
    /// Resolved with `identity` at the first step, because the addon step that needs it runs
    /// frames later and the roster resource may be gone by then.
    roster: Vec<String>,
    version_check: bool,
    /// The core's rows and the layer's files, resolved once at the first step.
    files: Vec<String>,
    /// How many of `files` are the core's (`None` for a chain without `FrameXML.toc`), and the
    /// core's `FrameXML.toc` load record, filled one row a step and closed, with the chain's
    /// `Bindings.xml` loaded, once the walk passes the core ([`super::manifest::close_core`]).
    core_rows: Option<usize>,
    core_toc: Option<benilla_ui::status::Status>,
    /// Frames the current step has spent waiting for its chain entry's pack to arrive
    /// (web only — see [`ENTRY_CHAIN_WAIT_FRAMES`]). Reset on every step that runs.
    waited: u32,
    /// The third-party walk, opened at its first step and stepped one addon per step.
    addons: Option<addons::ThirdParty>,
}

impl PendingEntryUiLoad {
    /// 0..1 of the load's steps done — the loading screen folds it into its bar.
    pub(crate) fn fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.done as f32 / self.total as f32).clamp(0.0, 1.0)
        }
    }
}

/// Where the sliced entry load is — see [`PendingEntryUiLoad`]. The order is
/// [`load_ingame_ui`]'s, one step per stage transition: what used to be one call is now the
/// same calls with frame boundaries allowed between them.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum EntryStage {
    /// Waiting for the cover, or about to take the first step.
    #[default]
    Armed,
    /// The core's rows, then the layer's files ([`super::manifest::ingame_entries`]), one per
    /// step.
    Builtin { next: usize },
    /// `UIParent_ManageFramePositions()`.
    Positions,
    /// The third-party addons, one per step: the walk keeps its dependency order and fires
    /// `ADDON_LOADED` per addon, and the VM stays parked between steps, so nothing runs between
    /// two addons that the reference's one burst would not have run.
    ThirdParty { next: usize },
    /// Saved variables, `VARIABLES_LOADED`, the failure count, the identity record.
    Finish,
}

/// How much of one frame the entry load may spend before yielding to the next. At least one
/// step always runs, so a single slow file still makes progress; the budget only decides
/// whether a *second* step starts. Small on purpose: the frames it yields to are the loading
/// cover's — cheap to render — and every one of them is a frame the browser can answer.
const ENTRY_LOAD_FRAME_BUDGET: std::time::Duration = std::time::Duration::from_millis(8);

/// How many frames one **chain-sourced** manifest entry may wait for its pack before the load
/// stops waiting and reads it blocking anyway.
///
/// Since the stock migration (1751) most of the manifest is `Interface\FrameXML\…` read off the
/// player's own install, and on the page that read is a synchronous fetch when the pack holding
/// it is not resident — which is the whole freeze the slicing exists to prevent, moved one layer
/// down. Asking [`benilla_formats::Chain::ready`] both answers "is it here?" and starts the warm,
/// so a step that yields on `false` converges without a scheduler of its own, with the loading
/// cover up and its bar still moving.
///
/// The cap is the backstop, not the mechanism: a pack that never arrives must not hang the login
/// forever, so past it the step runs and pays whatever the read costs. ~10 s at 60 fps, which is
/// far outside any warm this has been measured taking. Free on native, where `ready` is always
/// true on the first ask.
const ENTRY_CHAIN_WAIT_FRAMES: u32 = 600;

/// The boot VM, parked between the world-entry edge and the deferred load, out of every feed's
/// reach: each feed takes the VM as an `Option` and returns on `None`.
///
/// **The window is the whole sliced load, not just the wait before it.** The load spends itself
/// over frames now ([`PendingEntryUiLoad`]), and a VM with half a frame tree in it is no more
/// ready for a feed than one with none — so a slice that yields parks the VM again rather than
/// handing it back, and only [`EntryStage::Finish`] returns it to the world.
pub(crate) struct ParkedBootVm(UiScript);

/// `OnEnter(InWorld)`: arm the deferred entry load and park the boot VM.
pub(crate) fn arm_entry_ui_load(world: &mut World) {
    world.insert_resource(PendingEntryUiLoad::default());
    if let Some(vm) = world.remove_non_send::<UiScript>() {
        world.insert_non_send(ParkedBootVm(vm));
    }
}

/// Retire the glue VM and build the one the entry load runs on: the reference re-makes its Lua
/// state inside `UI_Init` (`0x48fe97`, in `0x48fbf0`), so no login shares a session id (and so a
/// [`super::VmMemo`]) with the character screen before it. The glue VM's CVars fold first, since
/// the character screen can write one (the AddOns panel's out-of-date toggle) and this is the last
/// moment its table exists. Everything else is re-seeded onto the new VM or is a
/// [`super::VmMemo`], which is meant to reset here. It starts bare: `FrameXML.toc` runs
/// `GlobalStrings.lua` and `Fonts.xml` as its first rows, and `ChatFrame.lua` the emote tokens.
fn mint_entry_vm(world: &mut World) {
    crate::cvars::fold_dying_vm_cvars(world);
    if let Some(script) = new_vm(world, UiScript::new) {
        world.insert_non_send(script);
    }
}

/// Rebuild the glue VM for a change of the data the chain answers from (a realm of the other
/// flavor entered, [`crate::flavor`]): GlobalStrings and the emote tokens come from the new
/// chain. The dying VM's CVars fold first, as at world entry.
pub(crate) fn reinstall_boot_vm(world: &mut World) {
    crate::cvars::fold_dying_vm_cvars(world);
    install_boot_vm(world);
}

/// Put the parked boot VM back into the world, if one is parked.
fn unpark_boot_vm(world: &mut World) {
    if let Some(ParkedBootVm(vm)) = world.remove_non_send::<ParkedBootVm>() {
        world.insert_non_send(vm);
    }
}

/// Run condition for every in-world feed: the in-game UI is up on the VM in the world, since a
/// feed that fires or drains into a VM with no frames loses what it sends. The reference likewise
/// processes no world state before its UI is up: world entry (`0x420d63`) runs after the same
/// iteration's inbound drain (`0x420d55`, its container `0x420c00` entered once per process),
/// socket threads only enqueue (`0x537b7c`; `SMSG_PONG` at `0x537b56` the one bypass), and the
/// login request waits for the UI (`0x46c272`).
///
/// Two terms, because the latch trails the wire by a frame: `apply_net_updates` drains `Connected`
/// and the login burst behind it in one pass, and the `InWorld` transition that arms the latch and
/// parks the VM runs at the next frame's `StateTransition`; the `InWorld` term covers that frame.
pub(crate) fn ingame_ui_up(
    pending: Option<Res<PendingEntryUiLoad>>,
    state: Option<Res<State<crate::char_select::ClientState>>>,
) -> bool {
    pending.is_none() && state.is_some_and(|s| *s.get() == crate::char_select::ClientState::InWorld)
}

/// `PreUpdate`, after [`run_pending_reload`] so a reload never interleaves an armed entry load:
/// run the armed entry load once the cover has presented, at once when there is no cover (a
/// capture booting straight `InWorld`). Leaving the world first drops the latch unrun, and
/// [`end_ui_session`] then skips the shutdown writes, so a UI-less VM never overwrites the saved
/// files.
pub(crate) fn run_pending_entry_load(world: &mut World) {
    if world.get_resource::<PendingEntryUiLoad>().is_none() {
        return;
    }
    let in_world = *world
        .resource::<State<crate::char_select::ClientState>>()
        .get()
        == crate::char_select::ClientState::InWorld;
    if !in_world {
        // Left the world before the load ran: nothing to build a UI for.
        world.remove_resource::<PendingEntryUiLoad>();
        unpark_boot_vm(world);
        return;
    }
    let covering = world
        .get_resource::<crate::loading_screen::LoadingScreen>()
        .is_some_and(|s| s.covering());
    // `presented()` is also true with no cover up: nothing to protect, so load now. Only the first
    // step waits; once the sliced load is under way the cover is on screen.
    if world.resource::<PendingEntryUiLoad>().stage == EntryStage::Armed
        && !world
            .get_resource::<crate::loading_screen::EntryCover>()
            .is_none_or(|c| c.presented())
    {
        return;
    }
    // The chain's `FrameXML.toc` is the walk's list, read at the first step: its pack is waited
    // for as a row's own is, with the glue VM still parked and nothing built yet.
    {
        let pending = world.resource::<PendingEntryUiLoad>();
        if pending.stage == EntryStage::Armed
            && ui_wanted(world)
            && pending.waited < ENTRY_CHAIN_WAIT_FRAMES
            && !super::reference_ui::ready(super::reference_ui::TOC)
        {
            world.resource_mut::<PendingEntryUiLoad>().waited += 1;
            return;
        }
    }
    // One frame's slice: steps until the budget is spent (at least one), then yield. The VM
    // leaves the world for the slice and comes back at its end, so every other system sees it
    // between frames exactly as it did around the old single burst.
    if !ui_wanted(world) {
        world.remove_resource::<PendingEntryUiLoad>();
        unpark_boot_vm(world);
        crate::vplates::clear_at_world_entry(world);
        return;
    }
    // **The first step builds the VM the load runs on** (decision 2226, [`mint_entry_vm`]): the
    // glue phase's VM goes back into the world only so its CVars can be folded, and a fresh boot
    // VM takes its place. Every later slice picks the parked, part-built one back up.
    if world.resource::<PendingEntryUiLoad>().stage == EntryStage::Armed {
        unpark_boot_vm(world);
        mint_entry_vm(world);
    }
    // The parked VM first (`ParkedBootVm` — the ordinary path since 1978, and every slice after
    // the first), then the world's own: the freshly minted entry VM on the first slice.
    let parked = world
        .remove_non_send::<ParkedBootVm>()
        .map(|ParkedBootVm(vm)| vm);
    let Some(mut script) = parked.or_else(|| world.remove_non_send::<UiScript>()) else {
        warn!("ui_script: entering the world with no VM — the in-game UI will not load");
        world.remove_resource::<PendingEntryUiLoad>();
        crate::vplates::clear_at_world_entry(world);
        return;
    };
    let frame_start = bevy::platform::time::Instant::now();
    world.resource_mut::<PendingEntryUiLoad>().frames += 1;
    loop {
        let stage = world.resource::<PendingEntryUiLoad>().stage;
        match stage {
            EntryStage::Armed => {
                // World entry: no player object yet, so the roster's record stands in.
                let (identity, roster, version_check) =
                    entry_prepare(world, &mut script, LoadSeat::Roster);
                // fork: upstream's load-edge sound bracket ([`silenced_ui_load`]), held open across
                // the slices: pushed here, before the first file, and popped at `Finish` after the
                // saved variables and the `PLAYER_LOGIN` arm. The depth lives on the VM, which is
                // parked between frames, so every file in between loads silent exactly as the
                // one-shot path's does.
                script.push_sound_suppression();
                script.set_instruction_budget(addons::LOAD_INSTRUCTION_BUDGET);
                let (files, core_rows) =
                    super::manifest::ingame_entries(&script, super::manifest::layer_enabled());
                let mut pending = world.resource_mut::<PendingEntryUiLoad>();
                pending.core_rows = core_rows;
                pending.core_toc = Some(benilla_ui::status::Status::default());
                pending.identity = identity;
                pending.roster = roster;
                pending.version_check = version_check;
                pending.covering = covering;
                pending.started = Some(frame_start);
                // Every core row (`GlobalStrings.lua` and `Fonts.xml` among the first) and the
                // layer's files, + positions + addons + finish.
                pending.total = files.len() + 3;
                pending.done = 0;
                pending.files = files;
                pending.stage = EntryStage::Builtin { next: 0 };
                pending.waited = 0;
                // Start the whole manifest's chain entries warming at once, before the first one
                // is read. They are the majority of the list since the stock migration, they sit
                // in one `Interface\FrameXML\` locality group, and the walk below reads them in
                // order — so one sweep here is the difference between a warm that overlaps the
                // load and forty warms discovered one blocking read at a time. Free on native.
                warm_chain_entries(world);
            }
            EntryStage::Builtin { next } => {
                let pending = world.resource::<PendingEntryUiLoad>();
                // The core's close, once, before the layer's first file (or at the end without
                // one): its `FrameXML.toc` record, then the chain's `Bindings.xml`, whose pack is
                // waited for as a row's is. Without a toc only the bindings load, as `UI_Init`
                // goes on to them (`0x48fff9`).
                if pending.core_toc.is_some() && next >= pending.core_rows.unwrap_or(0) {
                    if pending.waited < ENTRY_CHAIN_WAIT_FRAMES
                        && !super::reference_ui::ready(super::reference_ui::BINDINGS)
                    {
                        world.resource_mut::<PendingEntryUiLoad>().waited += 1;
                        break; // resume next frame, same step
                    }
                    let mut pending = world.resource_mut::<PendingEntryUiLoad>();
                    let toc = pending.core_toc.take().unwrap_or_default();
                    let has_core = pending.core_rows.is_some();
                    pending.waited = 0;
                    let _ = if has_core {
                        super::manifest::close_core(&script, toc)
                    } else {
                        super::manifest::load_core_bindings(&script)
                    };
                    continue;
                }
                if next < pending.files.len() {
                    let entry = pending.files[next].clone();
                    // A chain entry whose pack has not arrived is *waited for*, not read: the
                    // read would be synchronous and would blow this frame's budget on the one
                    // thing the slicing cannot slice. `ready` starts the warm, the cover keeps
                    // presenting, and the step runs on the frame the bytes land — or, past
                    // [`ENTRY_CHAIN_WAIT_FRAMES`], anyway.
                    if super::reference_ui::is_chain_entry(&entry)
                        && pending.waited < ENTRY_CHAIN_WAIT_FRAMES
                        && !super::reference_ui::ready(&entry)
                    {
                        world.resource_mut::<PendingEntryUiLoad>().waited += 1;
                        break; // resume next frame, same step
                    }
                    // Errors are logged by the loader itself and retained for `/errors`; the
                    // one-shot path discards them too.
                    let mut toc = world
                        .resource_mut::<PendingEntryUiLoad>()
                        .core_toc
                        .take()
                        .unwrap_or_default();
                    let _ = super::manifest::load_entry(&script, &entry, &mut toc);
                    let mut pending = world.resource_mut::<PendingEntryUiLoad>();
                    if next < pending.core_rows.unwrap_or(0) {
                        pending.core_toc = Some(toc);
                    }
                    pending.done += 1;
                    pending.stage = EntryStage::Builtin { next: next + 1 };
                    pending.waited = 0;
                } else {
                    world.resource_mut::<PendingEntryUiLoad>().stage = EntryStage::Positions;
                    continue; // free transition
                }
            }
            EntryStage::Positions => {
                let _ = super::manifest::bootstrap_positions(&script);
                let mut pending = world.resource_mut::<PendingEntryUiLoad>();
                pending.done += 1;
                pending.stage = EntryStage::ThirdParty { next: 0 };
            }
            EntryStage::ThirdParty { next } => {
                if world.resource::<PendingEntryUiLoad>().addons.is_none() {
                    let (identity, roster, version_check) = {
                        let p = world.resource::<PendingEntryUiLoad>();
                        (p.identity.clone(), p.roster.clone(), p.version_check)
                    };
                    let walk = addons::open_third_party(
                        &mut script,
                        identity.as_ref(),
                        &roster,
                        version_check,
                    );
                    let mut pending = world.resource_mut::<PendingEntryUiLoad>();
                    // The bar counted the addons as one step; it counts each now.
                    pending.total += walk.len().saturating_sub(1);
                    pending.addons = Some(walk);
                }
                let mut pending = world.resource_mut::<PendingEntryUiLoad>();
                let mut walk = pending.addons.take().expect("opened above");
                addons::load_one_addon(&mut walk, &mut script, next);
                pending.done += 1;
                if next + 1 < walk.len() {
                    pending.addons = Some(walk);
                    pending.stage = EntryStage::ThirdParty { next: next + 1 };
                } else {
                    let _ = walk.failures();
                    pending.stage = EntryStage::Finish;
                }
            }
            EntryStage::Finish => {
                let pending = world
                    .remove_resource::<PendingEntryUiLoad>()
                    .unwrap_or_default();
                entry_finish(world, &mut script, pending.identity.clone());
                script.pop_sound_suppression();
                world.insert_non_send(script);
                arm_leaving_world(world);
                // The world entry's own step after the UI load, which a `ReloadUI()` never takes.
                crate::vplates::clear_at_world_entry(world);
                // Every entry logs the load's cost, over how many frames it was spread, whether
                // the cover hid it, and the session every `VmMemo` keys on.
                info!(
                    "ui_script: in-game UI up in {:.0} ms over {} frame(s) (behind the cover: {}, \
                     vm session: {})",
                    pending
                        .started
                        .map_or(0.0, |s| s.elapsed().as_secs_f32() * 1000.0),
                    pending.frames,
                    pending.covering,
                    world
                        .get_non_send::<UiScript>()
                        .map_or(0, UiScript::session),
                );
                return;
            }
        }
        if frame_start.elapsed() >= ENTRY_LOAD_FRAME_BUDGET {
            break; // resume next frame
        }
    }
    // Back into the park, not into the world: the tree is still part-built, and 1978's whole
    // point is that no feed sees a VM in that state. `Finish` above is the one exit that
    // publishes it.
    world.insert_non_send(ParkedBootVm(script));
}

/// Start every chain-sourced manifest entry's pack warming, and return at once.
///
/// One sweep at the top of the load rather than a warm discovered per blocking read — see
/// [`ENTRY_CHAIN_WAIT_FRAMES`]. `ready` is the ask that starts a warm, and its answer is
/// discarded here on purpose: this is the warm, not the gate.
fn warm_chain_entries(world: &mut World) {
    let files = world.resource::<PendingEntryUiLoad>().files.clone();
    for entry in files
        .iter()
        .filter(|f| super::reference_ui::is_chain_entry(f))
    {
        let _ = super::reference_ui::ready(entry);
    }
    // The core's close reads the chain's `Bindings.xml` after the last row.
    let _ = super::reference_ui::ready(super::reference_ui::BINDINGS);
}

/// The entry load's opening — everything that has to be on the VM before the first interface file
/// runs: the addon array, the CVar table, the realm, the player record and seat, the version gate,
/// the per-VM seeds and the raster seam. Shared by the sliced entry load
/// ([`run_pending_entry_load`]) and the one-shot [`load_ingame_ui_on_world_entry`], so the two
/// cannot drift on what a fresh entry VM is told. `seat` is the `"player"` the edge has.
fn entry_prepare(
    world: &mut World,
    script: &mut UiScript,
    seat: LoadSeat,
) -> (Option<(String, String)>, Vec<String>, bool) {
    // The character whose AddOn enable state gates which addons run.
    let identity = world
        .get_resource::<crate::char_select::Roster>()
        .and_then(crate::ui_macro::identity);
    // The realm's character list is the enable store's node set: the reference keeps one
    // `ADDONSTATELIST` node per character, and an addon this one never set resolves from the rest.
    let roster: Vec<String> = world
        .get_resource::<crate::char_select::Roster>()
        .map(|r| r.chars.iter().map(|c| c.name.clone()).collect())
        .unwrap_or_default();
    // The addon array before any addon file runs: `SMSG_ADDON_INFO` lands during the handshake, so
    // the reference has it before `UI_Init`'s addon walk and `GetNumAddOns()` at file scope sees
    // it. A server that never answered leaves it empty for the session, as in the reference.
    if let Some(reply) = world
        .get_resource::<crate::net::AddonInfoReply>()
        .and_then(|r| r.0.clone())
    {
        script.note_addon_info_records(super::addons::addon_replies(&reply));
    }
    // The CVar table goes in before the UI loads: stock `UIOptionsFrame.xml`'s camera dropdowns
    // read `cameraSmoothStyle` and `cameraSmoothTrackingStyle` in their `OnLoad`, where a nil is a
    // concat error (`UIOptionsFrame.lua:379-384`, `:501-506`), and this VM was born inside this
    // call, before `cvars::sync_cvars`'s per-VM seed. Its order: the file's unclaimed entries
    // first, so an addon's `RegisterCVar` starts at the player's value, then the live table.
    match world.get_resource::<crate::cvars::Cvars>() {
        Some(cvars) => {
            script.set_cvar_saved_base(cvars.orphans());
            script.seed_cvars(cvars.vm_seed());
        }
        None => script.register_cvars(crate::cvars::registered_pairs()),
    }
    // The realm before the load, since addons read `GetRealmName()` at file scope; the roster
    // carries the realm-list entry this session connected to.
    let realm = world
        .get_resource::<crate::char_select::Roster>()
        .and_then(|r| r.realm.as_ref().map(|r| r.name.clone()))
        .unwrap_or_default();
    script.set_realm_name(&realm);
    // `CGGameUI::InitializeGame` flags these read-only before `UI_Init` loads anything
    // (`0x48f566`-`0x48f584`), so an in-game `SetCVar` of one raises, logout handlers included;
    // `ShutdownGame` clears them (`0x491240`), and here the flags die with this VM. The host's
    // own realm writes are engine writes, which the flag does not stop.
    for name in benilla_ui::script::IN_WORLD_READ_ONLY_CVARS {
        script.set_cvar_read_only(name, true);
    }
    // The player too, before the addon walk. The record first: the reference's
    // `UnitName`/`UnitRace`/`UnitClass`/`UnitSex` read only a copy of the char-enum row made at the
    // Enter World commit (`0x5abd9e`) and never cleared, and each rebuilt VM is told once. Then
    // the unit: the live player on a `/reload`, else the roster's stand-in until the descriptor
    // streams in.
    if let Some(record) = world
        .get_resource::<crate::char_select::Roster>()
        .and_then(record_from_roster)
    {
        script.set_player_record(record);
    }
    match (seat == LoadSeat::Live)
        .then(|| live_player_seat(world))
        .flatten()
    {
        Some(live) => {
            script.set_unit("player", Some(live.state));
            script.set_player_xp(live.xp.0, live.xp.1);
        }
        None => {
            if let Some(seat) = world
                .get_resource::<crate::char_select::Roster>()
                .and_then(seat_from_roster)
            {
                script.set_unit("player", Some(seat));
            }
        }
    }
    // The addon version gate, read from the persisted base the fold above just made current, so a
    // character-screen "Load out of date AddOns" click counts. A bare test world is check on.
    let version_check = world
        .get_resource::<crate::cvars::Cvars>()
        .is_none_or(crate::cvars::Cvars::addon_version_check);
    // Per-VM `Update` claims the load itself reads, seeded now: the zone-channel catalog (empty,
    // `General` would be filed and joined as a custom channel), the binding table (stock
    // `ActionButton_OnLoad` paints the hotkeys), the default language (`ChatFrame_OnEvent`'s
    // `PLAYER_ENTERING_WORLD` arm stores it, `ChatFrame.lua:1275-1276`) and the language table a
    // file-scope `SendChatMessage` names its tongue from.
    crate::ui_chat::seed_zone_channel_catalog(world, script);
    crate::bindings::seed_bindings_for_vm(world, script);
    crate::ui_unit::seed_default_language(world, script);
    crate::ui_unit::seed_language_table(world, script);
    // The world map's continent and zone lists: static DBC data in the reference, read at file
    // scope (Astrolabe builds its whole zone table from `GetMapContinents`/`GetMapZones` there).
    crate::ui_world_map::seed_world_map_catalog(world, script);
    // Screen size and measurer before the first `<OnLoad>`: this VM is loaded before `tick_script`
    // seats either, and the load measures text (the stock tab template's `OnLoad` sizes from its
    // text width, `UIPanelTemplates.xml:370-373`), which answers 0 with no measurer.
    seat_raster_seam_for_load(world, script);
    (identity, roster, version_check)
}

/// The entry load's closing — everything after the files: the dev commands, the minimap zoom
/// seed, saved variables + `VARIABLES_LOADED` with the chat-cache restore and the UI scale after
/// it, the `PLAYER_LOGIN` arm, the load log, the damage-text font, the budget disarm, the identity
/// record. Same sharing arrangement as [`entry_prepare`].
fn entry_finish(world: &mut World, script: &mut UiScript, identity: Option<(String, String)>) {
    crate::ui_chat::commands::register_dev_commands(script, crate::run_mode::dev_affordances());
    // The new Minimap's zoom indices from the persisted CVars, as the reference's minimap reset
    // copies each CVar into its live index; `Minimap:SetZoom` owns them from here.
    let zoom = {
        let z = world.resource::<crate::minimap::MinimapZoom>();
        (z.outdoor, z.inside)
    };
    script.set_minimap_zoom(zoom.0, zoom.1);
    // A plate setting `config.toml` still holds, carried into FrameXML's saved variables once.
    // Only by a load with the layer, which registers `FRIENDNAMEPLATES_ON` for save; a stock-UI
    // load leaves it for the next layered one. Read out of the world first: the call below borrows
    // `world` for the chat-cache restore.
    let legacy_plates = if super::manifest::layer_enabled() {
        world
            .get_resource_mut::<crate::cvars::Cvars>()
            .map(|mut cvars| crate::vplates::take_legacy_settings(&mut cvars))
            .unwrap_or_default()
    } else {
        Default::default()
    };
    let ui_scale = world
        .get_resource::<super::UiScaleCvar>()
        .map_or(1.0, |c| c.0);
    // The saved-variables chunk runs after the files set their defaults and before
    // `VARIABLES_LOADED`, the reference's order (`AddOn_Load` `0x51f240`); reversed, the defaults
    // always win. The chat cache restores after `VARIABLES_LOADED` and before any chat is routed:
    // its `UPDATE_CHAT_WINDOWS` registers each chat frame for its `CHAT_MSG_*` events
    // (`ChatFrame.lua:1261-1273`).
    finish_ui_load_with(
        script,
        // Before `VARIABLES_LOADED`, whose `UIParent_OnEvent` arm runs the first
        // `UpdateNameplates()` (`UIParent.lua:231-234`).
        |script| crate::vplates::seat_legacy_settings(script, &legacy_plates),
        |script| {
            crate::ui_chat::restore_chat_looks(world, script);
            // The UI scale lands on `UIParent` once the files, the addons and `VARIABLES_LOADED`
            // are done, after the chat reader (`0x4900d6`) and before the `PLAYER_LOGIN` arm
            // (`0x49011d`), through the display handler (`0x49010e` → `0x492e90`). So every
            // `OnLoad` sees `UIParent` at scale 1 and the root's size.
            let display = {
                let mut q = world.query_filtered::<&Window, With<bevy::window::PrimaryWindow>>();
                q.single(world).map_or((0, 0), |w| {
                    (w.width().round() as u32, w.height().round() as u32)
                })
            };
            let mut none = None;
            let mut cache = world.get_resource_mut::<super::UiAutoScaleCache>();
            let cache = cache.as_deref_mut().map_or(&mut none, |c| &mut c.0);
            script.seat_ui_scale(ui_scale, display, cache);
        },
    );
    // The load's record goes to `Logs\FrameXML.log` (`0x490187`-`0x4901b6`), after any
    // `LoadAddOn` block the load ran; nothing reaches chat.
    script.finish_ui_load_log();
    super::load_log::write(script.take_load_log_writes());
    // `DAMAGE_TEXT_FONT` binds at the end of the load, after every addon's `ADDON_LOADED` (where
    // MikScrollingBattleText and pfUI assign it). The reference reads it eagerly, once per world
    // entry (`0x6c8470`, called at `0x401620` in `0x401570`, after the UI load at `0x401602`),
    // and never again: `[0xce8820]` has one writer, and a `/reloadui` does not re-run it, where
    // this edge re-reads it on every reload.
    world.insert_resource(crate::combat_text::read_damage_text_font(script));
    // The load is over: disarm its instruction bound, so no handler is killed for being slow.
    script.clear_instruction_budget();
    world.insert_resource(AddOnIdentity(identity));
}

/// The end of a load edge, after the VM is back in the world.
fn arm_leaving_world(world: &mut World) {
    // Arm the world latch, the reference's `0x490168 call 0x4908c0`: a reload creates no entity
    // to re-arm it, so without this a second `/reload` would not fire `PLAYER_LEAVING_WORLD`. The
    // reference skips this leg on a fresh login (`0x490166 je`, the GUID still 0) and arms from
    // the player's create instead; both reach the same armed byte.
    if let Some(mut armed) = world.get_resource_mut::<LeavingWorldArmed>() {
        armed.arm();
    }
}

/// Seat the VM's screen size and text measurer for the load, under the current window: the
/// load-edge half of [`super::extract::tick_script`]'s per-frame pair. A fresh VM answers 1024×768
/// until then, and a value computed once at load keeps it: stock `WorldMapFrame_OnLoad` sizes the
/// full-screen `BlackoutWorld` from `GetScreenWidth()` and `GetScreenHeight()` and never again
/// (`WorldMapFrame.lua:19-28`), which answer the root while `UIParent` is still at scale 1. With no
/// atlas yet (it bakes on the first `Update`) the per-frame pass seats the measurer.
fn seat_raster_seam_for_load(world: &mut World, script: &mut UiScript) {
    let (w, h) = {
        let mut q = world.query_filtered::<&Window, With<bevy::window::PrimaryWindow>>();
        q.single(world)
            .map_or((0.0, 0.0), |win| (win.width(), win.height()))
    };
    let s = super::seam_scale(h);
    if w > 0.0 && h > 0.0 {
        // The VM lives in a 768-tall root, so logical size over the seam scale is its screen. No
        // `UIParent_ManageFramePositions` here, unlike `tick_script`: nothing is laid out yet.
        script.set_screen_size(w / s, h / s);
    }
    let Some(atlas) = world.get_resource::<crate::ui_text::UiFontAtlas>() else {
        return;
    };
    super::extract::seat_text_measurer(script, atlas, s);
}

/// Which `"player"` a UI load runs its files under, by the edge that runs it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LoadSeat {
    /// World entry: `UI_Init` runs before `CMSG_PLAYER_LOGIN` goes out (the load at `0x46c236`,
    /// the send in a later step at `0x46c272`), so no player object exists and the roster's
    /// record stands in ([`seat_from_roster`]).
    Roster,
    /// `/reload`: the player object stays in the world through the teardown and the load, so the
    /// live one is seated ([`crate::ui_unit::live_player_seat`]), the roster's if none exists.
    Live,
}

/// Load the in-game UI for this session onto a VM built for it, once per world entry. The
/// reference builds the UI in `UI_Init` (`0x48fbf0`, re-making the Lua state at `0x48fe97`) and
/// destroys it at `0x490bd0`, so every login runs every addon's file scope afresh. Safe on the
/// state edge because the initial transition runs after `PostStartup`: a capture booting straight
/// into `InWorld` still loads after [`benilla_assets::AssetSet::Open`].
///
/// fork: the tests' one-shot entry. A live world entry is sliced across frames
/// ([`run_pending_entry_load`]) and a `/reload` loads under the live player.
#[cfg(test)]
pub(crate) fn load_ingame_ui_on_world_entry(world: &mut World) {
    load_ingame_ui_seated(world, LoadSeat::Roster);
}

/// [`load_ingame_ui_on_world_entry`] under the `"player"` its edge has.
fn load_ingame_ui_seated(world: &mut World, seat: LoadSeat) {
    // The parked VM goes back into the world, where the CVar fold reads it; a reload has none
    // parked and uses the live one.
    unpark_boot_vm(world);
    if !ui_wanted(world) {
        return;
    }
    mint_entry_vm(world);
    let Some(mut script) = world.remove_non_send::<UiScript>() else {
        warn!("ui_script: entering the world with no VM — the in-game UI will not load");
        return;
    };
    let (identity, roster, version_check) = entry_prepare(world, &mut script, seat);
    // **The whole load edge is silent** — the TOC walk, every addon, the saved variables and the
    // `PLAYER_LOGIN` arm, the span `0x48fbf0` brackets itself across (`0x48fbfa` → `0x49016d`); see
    // [`silenced_ui_load`]. fork: the sliced load ([`run_pending_entry_load`]) holds the same
    // bracket open across its frames instead.
    silenced_ui_load(&mut script, |script| {
        let _ = load_ingame_ui(script, identity.as_ref(), &roster, version_check);
        entry_finish(world, script, identity);
    });
    world.insert_non_send(script);
    arm_leaving_world(world);
}

/// The live player's seat off the world, through the unit feed's own builder; `None` before our
/// own create. A world without the feed's resources (a bare test world) has none to read.
fn live_player_seat(world: &mut World) -> Option<crate::ui_unit::LivePlayerSeat> {
    world
        .run_system_cached(crate::ui_unit::live_player_seat)
        .unwrap_or_else(|e| {
            warn!("ui_script: no live player to seat for the load ({e}); the roster stands in");
            None
        })
}

/// The wire's gender byte (0 male, 1 female) on `UnitSex`'s 2/3 scale, as `ui_unit::snapshot`
/// maps the descriptor's; the record and the seat share it.
fn roster_sex(gender: u8) -> u8 {
    match gender {
        0 => 2,
        1 => 3,
        _ => 0,
    }
}

/// The local player record the UI loads under: our copy of the char-enum row the reference copies
/// at the Enter World commit (layout on [`benilla_ui::script::PlayerRecord`]). Race and class go
/// through the same lookups as [`seat_from_roster`]; the reference resolves them through
/// `ChrRaces`/`ChrClasses` at call time, the same answer. No level: the reference's record has
/// one at `+0x108`, but its accessor (`0x5abe00`) has no caller, so `UnitLevel("player")` reads
/// the descriptor.
pub(crate) fn record_from_roster(
    roster: &crate::char_select::Roster,
) -> Option<benilla_ui::script::PlayerRecord> {
    let row = roster.pending_row()?;
    let race = crate::ui_unit::race_names(row.race);
    let class = crate::ui_unit::class_names(row.class);
    Some(benilla_ui::script::PlayerRecord {
        name: row.name.clone(),
        race: race.map(|(n, f)| (n.to_string(), f.to_string())),
        class: class.map(|(n, f)| (n.to_string(), f.to_string())),
        sex: roster_sex(row.gender),
    })
}

/// The `"player"` snapshot the UI loads under, from the roster row of the pick in flight, until
/// the self descriptor streams in; `None` with no pick (a capture, a scenario, a test world).
/// [`record_from_roster`] answers name, race, class and sex; this carries what else addon file
/// scope reads, the faction group chief among it. Health and power stay zero, the descriptor's
/// to say.
pub(crate) fn seat_from_roster(
    roster: &crate::char_select::Roster,
) -> Option<benilla_ui::script::UnitState> {
    let row = roster.pending_row()?;
    let race = crate::ui_unit::race_names(row.race);
    let class = crate::ui_unit::class_names(row.class);
    let sex = roster_sex(row.gender);
    Some(benilla_ui::script::UnitState {
        // Before the player object exists the reference answers `UnitExists("player")`
        // (`0x515fb0`) nil: the resolver reads the GUID from the object (`0x515994`), and the
        // fallback `0x491900` bails on the zero GUID (`0x4e80aa`) to `0x516001 lua_pushnil`.
        // `UnitLevel` (`0x517fc0`) answers the number 0 (`0x51813e push 0`); the two misses differ.
        exists: false,
        name: Some(row.name.clone()),
        race: race.map(|(n, _)| n.to_string()),
        race_file: race.map(|(_, f)| f.to_string()),
        class: class.map(|(n, _)| n.to_string()),
        class_file: class.map(|(_, f)| f.to_string()),
        sex,
        is_player: true,
        player_controlled: true,
        // AceDB-2.0 concatenates `UnitFactionGroup("player")` at file scope, where nil is an error.
        faction_group: crate::ui_unit::race_faction_group(row.race).map(str::to_string),
        ..Default::default()
    })
}

/// The character the loaded AddOn state belongs to, kept so the shutdown writes go back to its
/// files after the roster's pick is gone.
#[derive(Resource, Default)]
pub(crate) struct AddOnIdentity(pub(crate) Option<(String, String)>);

/// The world latch, the reference's `[0xb4b424]`: set at `0x4908ce` in the world-enter cascade
/// `0x4908c0` and cleared at `0x490a8d` by the leave-world sweep `0x490a80`, which then fires
/// `PLAYER_LEAVING_WORLD` (`0x490b4d`, its one site), so the first departure in a world sweeps
/// and the rest are no-ops until the next entry. Ours is spent through [`LeaveWorldSweep`] or
/// [`run_leave_world_sweep`], by [`shutdown_ui_state`]'s roots and `ui_unit`'s worldport fire. A
/// cross-map worldport spends it without leaving `InWorld`, so a quit on its loading screen
/// fires nothing; a `/reload` fires, and its rebuild re-arms.
#[derive(Resource, Default)]
pub(crate) struct LeavingWorldArmed(bool);

/// The leave-world sweep `0x490a80` ran past its latch: a cross-map worldport (the local
/// player's destructor, `0x5dd543`), a `/reload` (`0x490c20`) or leaving the world. Its legs
/// past the clear read this, as the meeting stone's text drop (`0x490b2f` → `0x4c9f80`) does; a
/// same-map teleport destroys nothing and never sweeps.
#[derive(Message)]
pub(crate) struct WorldLeaveSweepMessage;

/// The latch and the sweep it gates, for a system: spending one writes the other.
#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct LeaveWorldSweep<'w> {
    armed: ResMut<'w, LeavingWorldArmed>,
    sweeps: MessageWriter<'w, WorldLeaveSweepMessage>,
}

impl LeaveWorldSweep<'_> {
    /// Run the sweep: `true`, and [`WorldLeaveSweepMessage`] written, at most once per world.
    pub(crate) fn run(&mut self) -> bool {
        let swept = self.armed.spend();
        if swept {
            self.sweeps.write(WorldLeaveSweepMessage);
        }
        swept
    }
}

/// [`LeaveWorldSweep::run`] for an exclusive edge; no latch is no world to leave.
pub(crate) fn run_leave_world_sweep(world: &mut World) -> bool {
    let swept = world
        .get_resource_mut::<LeavingWorldArmed>()
        .is_some_and(|mut l| l.spend());
    if swept {
        if let Some(mut sweeps) = world.get_resource_mut::<Messages<WorldLeaveSweepMessage>>() {
            sweeps.write(WorldLeaveSweepMessage);
        }
    }
    swept
}

impl LeavingWorldArmed {
    /// A world began. Idempotent, like the reference's `mov byte [0xb4b424],1`.
    pub(crate) fn arm(&mut self) {
        self.0 = true;
    }

    /// Take the one-shot: `true` at most once per world, `0x490a8d`'s clear folded in.
    fn spend(&mut self) -> bool {
        std::mem::take(&mut self.0)
    }

    /// Read without taking, for tests.
    #[cfg(test)]
    pub(crate) fn is_armed(&self) -> bool {
        self.0
    }
}

/// Arm the latch when the local player's entity appears: the reference's `0x5deb60` entry into
/// the world-enter cascade, which a fresh login and a worldport's new-world create both take
/// (`0x5deb49 call 0x468570` / `0x5deb50 jne 0x5deb6a`).
pub(crate) fn arm_leaving_world_on_self_create(
    created: Query<(), Added<crate::net::SelfPlayer>>,
    mut armed: ResMut<LeavingWorldArmed>,
) {
    if !created.is_empty() {
        armed.arm();
    }
}

/// The UI shutdown in the reference's order, `0x490bd0`'s tail: `PLAYER_LEAVING_WORLD` (273) →
/// `PLAYER_LOGOUT` (271, `0x490c2a`) → `layout-cache.txt` → the flat saved file (`0x490c7e`) →
/// the per-addon files (`0x490c83`) → `AddOns.txt` (`0x490c88`) → destroy the frame-script owner
/// (`0x490c97`) → nil all 216 C bindings out of `_G` (`0x490cba` → `0x490ce0`).
///
/// `PLAYER_LOGOUT` fires before any write, an addon's last chance to change a saved global. One
/// function on every root because the steps are ordered against each other; the layout save is
/// here because a `/reload` never leaves `InWorld`. No autosave, as the reference has none (the
/// layout cache's crash-only debounce in [`crate::ui_layout`] is ours).
pub(crate) fn shutdown_ui_state(
    script: &mut UiScript,
    identity: Option<&(String, String)>,
    leaving_world: bool,
) {
    // The one conditional step: the reference reaches `0x490c20 call 0x490a80` only past the
    // active-player guard (`0x490bee call 0x468550`, `0x490bf5 je 0x490c25`) and the world latch,
    // while `PLAYER_LOGOUT` (`0x490c2a`) fires on every root. The caller spends the latch.
    if leaving_world {
        script.fire_event("PLAYER_LEAVING_WORLD", vec![]);
    }
    script.fire_event("PLAYER_LOGOUT", vec![]);
    crate::ui_layout::save_now(script, identity);
    crate::ui_saved::save(script);
    addons::save_addon_variables(script, identity);
    addons::save_enable_state(script, identity);
}

/// `OnExit(InWorld)`, a `/logout` or a disconnect (two of the reference's five roots): run the
/// shutdown tail, then end this session's Lua state, so no frame, global or addon upvalue reaches
/// the next login. The reference's state outlives `0x490bd0` and is replaced twice through
/// `0x703b80`, in `ShutdownGame` (at `0x491231`) and then in the glue build (`0x46a7b0`), so the
/// character screen runs on a state of its own. Ours is a fresh boot VM, or the stock glue's
/// ([`install_glue_side_vm`]).
pub(crate) fn end_ui_session(world: &mut World) {
    close_ui_session(world);
    // `0x401ee0`'s glue leg: `"charselect"` after a logout, `"login"` after a disconnect
    // (`0x402084`); the state this exit enters says which.
    let screen = match world
        .get_resource::<State<crate::char_select::ClientState>>()
        .map(|s| *s.get())
    {
        Some(crate::char_select::ClientState::CharSelect) => super::glue::GlueScreen::CharSelect,
        _ => super::glue::GlueScreen::Login,
    };
    install_glue_side_vm(world, screen);
}

/// [`end_ui_session`] up to the new VM: the shutdown tail, the folds and the host values reset.
fn close_ui_session(world: &mut World) {
    // A VM still parked (the load never ran) goes back into the world first.
    unpark_boot_vm(world);
    // An entry load still armed means no in-game UI was built, and the tail would overwrite the
    // saved files with the boot VM's emptiness.
    let ui_never_loaded = world.remove_resource::<PendingEntryUiLoad>().is_some();
    let identity = world
        .get_resource::<AddOnIdentity>()
        .and_then(|id| id.0.clone());
    // Spent at the root: of this edge and the worldport's, the first to a departure owns it.
    let leaving_world = run_leave_world_sweep(world);
    if !ui_never_loaded {
        if let Some(mut script) = world.get_non_send_mut::<UiScript>() {
            shutdown_ui_state(&mut script, identity.as_ref(), leaving_world);
        }
    }
    // After the shutdown events (a `PLAYER_LOGOUT` handler may `SetCVar`, which in the reference
    // lands in an engine-side store that survives) and before the VM is replaced.
    crate::cvars::fold_dying_vm_cvars(world);
    // The chat cache folds here, not in its own `OnExit(InWorld)` system, which a `/reload` skips.
    crate::ui_chat::settings::fold_dying_vm_chat_cache(world);
    world.insert_resource(AddOnIdentity(None));

    // Host values taken from the dying frame tree go with it (a `VmMemo` resets on its own): the
    // hovered frame, the keyboard capture (`feed_ui_input` stops outside `InWorld`), the cursor
    // payload and the minimap's hole. Not [`UiClock`]: `GetTime` is the reference's OS clock, and
    // [`seed_vm_clock`] re-anchors it for the next VM.
    if let Some(mut hover) = world.get_resource_mut::<PlayerUiHover>() {
        hover.0 = None;
    }
    if let Some(mut keys) = world.get_resource_mut::<UiKeyboardCapture>() {
        keys.typing = false;
        keys.arrows_fall_through = false;
        keys.consumed.clear();
    }
    if let Some(mut held) = world.get_resource_mut::<CursorPayloadHeld>() {
        *held = CursorPayloadHeld::default();
    }
    if let Some(mut minimap) = world.get_resource_mut::<crate::minimap::MinimapWidget>() {
        minimap.0 = None;
    }
}

/// A `ReloadUI()` waiting to run: set when [`crate::ui_logout`] drains
/// [`benilla_ui::script::SessionRequest::ReloadUi`], run by [`run_pending_reload`] in the next
/// frame's `PreUpdate`. A flag, as in the reference (`[0xb4b3f4]`, written only by `0x491380` and
/// read only by the per-frame `0x495590`), so the VM that asked is never mid-call when destroyed.
#[derive(Resource, Default)]
pub(crate) struct ReloadUiPending(pub(crate) bool);

/// Run a pending `ReloadUI()`: the reference's teardown and rebuild (`0x495664 call 0x490bd0`,
/// `0x495669 call 0x48fbf0`), for us [`end_ui_session`] then the load without leaving the world,
/// so a `DisableAddOn` staged in the dying VM reaches `AddOns.txt` before the rebuild reads it.
/// The load runs under the live player, which never left. `UI_Init` then enters the world-enter
/// cascade itself (`0x490168`), firing the `PLAYER_LOGIN` it armed and `PLAYER_ENTERING_WORLD`;
/// ours fire from [`crate::ui_unit`]'s feed on the new VM's first `Update`, once it has re-seated
/// the player. In-world only, as the reference's gate (`0x494a50(0xa)`) is; elsewhere the request
/// is dropped.
pub(crate) fn run_pending_reload(world: &mut World) {
    if !std::mem::take(&mut world.resource_mut::<ReloadUiPending>().0) {
        return;
    }
    let in_world = *world
        .resource::<State<crate::char_select::ClientState>>()
        .get()
        == crate::char_select::ClientState::InWorld;
    if !in_world {
        info!("ui_script: ReloadUI outside the world — dropped");
        return;
    }
    info!("ui_script: ReloadUI — ending the UI session and building a new one");
    // A boot VM, never the glue's: the reload builds no glue (`0x495664` then `0x495669`), and the
    // load replaces this VM at once.
    close_ui_session(world);
    install_boot_vm(world);
    load_ingame_ui_seated(world, LoadSeat::Live);
    // `UI_Init` ends by re-entering the world-enter cascade (`0x490168`), its sends included, but
    // only with the active player set (`0x490166 je`): our own player's entity exists. Not
    // `SetActiveMover`, which only the player's create runs.
    let seated = world
        .query_filtered::<(), With<crate::net::SelfPlayer>>()
        .iter(world)
        .next()
        .is_some();
    if seated {
        world.write_message(crate::net::WorldEnterCascadeMessage);
    }
}

/// `AppExit`, the quit roots, read as a message because a quit from in-world never leaves
/// `InWorld`. A quit from the character screen, or on a worldport's loading screen, finds the
/// latch spent, so `PLAYER_LEAVING_WORLD` does not fire.
pub(crate) fn shutdown_on_exit(
    script: Option<NonSendMut<UiScript>>,
    id: Res<AddOnIdentity>,
    pending_entry: Option<Res<PendingEntryUiLoad>>,
    mut sweep: LeaveWorldSweep,
    mut exits: MessageReader<AppExit>,
) {
    if exits.read().next().is_none() {
        return;
    }
    // Same guard as [`end_ui_session`]: a quit inside the entry-load window has no UI to save.
    if pending_entry.is_some() {
        return;
    }
    let leaving_world = sweep.run();
    if let Some(mut script) = script {
        shutdown_ui_state(&mut script, id.0.as_ref(), leaving_world);
    }
}

/// The load's ordered tail without the host seams, for tests; production runs
/// [`finish_ui_load_with`]. The reference's `UI_Init` (`0x48fbf0`) loads every non-LoadOnDemand
/// addon, each firing `ADDON_LOADED` at `0x51f5ad` (`0x4900a3` → `0x51f600`), reads the flat
/// saved file and fires `VARIABLES_LOADED` (`0x4900b2` → `0x4913b0`), and arms `PLAYER_LOGIN`
/// (`[0xb4e260]`, set at `0x49011d`). The world-enter cascade `0x4908c0` fires `PLAYER_LOGIN` if
/// armed (read at `0x49094b`, fired at `0x490959`, disarmed at `0x49095e`), then
/// `PLAYER_ENTERING_WORLD` (`0x49096a`): [`UiScript::fire_world_enter`].
///
/// On a `/reload`, `UI_Init` enters the cascade itself (`0x490168`). On a fresh login that call is
/// skipped (`0x490166 je`, the active-player GUID still 0) and the cascade runs from the local
/// player's own create (`0x5deb60 call 0x4908c0`). Either way the player object exists when it
/// runs, so this only arms, and [`crate::ui_unit`]'s feed fires both once it has seated the
/// player.
#[cfg(test)]
pub(crate) fn finish_ui_load(script: &mut UiScript) {
    finish_ui_load_with(script, |_| {}, |_| {});
}

/// [`finish_ui_load`] with its two host seams. `host_settings` runs between the saved-variables
/// chunk and `VARIABLES_LOADED`, carrying over a setting `config.toml` held before it moved into
/// the saved variables. `between` runs after `VARIABLES_LOADED` and before the `PLAYER_LOGIN` arm
/// (so before the event too): the chat-cache restore's `UPDATE_CHAT_WINDOWS` +
/// `UPDATE_CHAT_COLOR` burst, where the reference registers its reader (`0x4900d6` → `0x498a20`),
/// then the UI scale (`0x49010e`). On a fresh login the reference's cache (`0x5afe50`) defers that
/// reader until the server's account data reconciles (`0x5afa42`/`0x5afd01`), after `UI_Init` has
/// returned. Either way a `VARIABLES_LOADED` handler sees the boot chat colours, not the file's.
pub(crate) fn finish_ui_load_with(
    script: &mut UiScript,
    host_settings: impl FnOnce(&mut UiScript),
    between: impl FnOnce(&mut UiScript),
) {
    // Still the load edge, so still bounded: re-armed so the saved variables and every
    // `VARIABLES_LOADED` handler get the full allowance; the entry edge disarms it after. The
    // world-enter events fire later, unbounded, as every event the feeds fire is.
    script.set_instruction_budget(addons::LOAD_INSTRUCTION_BUDGET);
    crate::ui_saved::load_saved_variables(script, host_settings);
    between(script);
    script.arm_player_login();
}

/// Run the patch chain's `Interface\FrameXML\GlobalStrings.lua`, the first file in the reference's
/// `FrameXML.toc` and the source of every localized string global. Failures are logged loudly:
/// without it every red error line is silently empty.
fn load_global_strings(world: &mut World, script: &UiScript) {
    let Some(assets) = world.get_resource::<benilla_assets::WorldAssets>() else {
        warn!("ui_script: no patch chain — GlobalStrings absent, error lines will be empty");
        return;
    };
    let bytes = {
        let mut chain = assets.chain.lock_recover();
        chain.read_file("Interface\\FrameXML\\GlobalStrings.lua")
    };
    let src = match bytes {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(e) => {
            error!("ui_script: GlobalStrings.lua read failed — error lines will be empty: {e:#}");
            return;
        }
    };
    if let Err(e) = script.run(&src) {
        error!("ui_script: GlobalStrings.lua failed to run: {e}");
        return;
    }
    // Sentinel: the cast-fail drain's own lookup, checked for presence only (any locale).
    let sentinel: Option<String> = script.lua().globals().get("SPELL_FAILED_NO_AMMO").ok();
    match sentinel {
        Some(s) if !s.is_empty() => info!("ui_script: GlobalStrings loaded"),
        other => {
            error!("ui_script: GlobalStrings sentinel missing ({other:?}) — error lines broken")
        }
    }
}

/// Run the emote token table from the patch chain's `ChatFrame.lua` (`EMOTE87_TOKEN = "SIT"`),
/// which maps each `GlobalStrings.lua` alias (`EMOTE87_CMD1 = "/sit"`) to its `EmotesText` name.
/// Only whole lines matching [`is_emote_token_line`] run, none of the file's code.
fn load_emote_tokens(world: &mut World, script: &UiScript) {
    let Some(assets) = world.get_resource::<benilla_assets::WorldAssets>() else {
        return; // already warned by `load_global_strings`
    };
    let bytes = {
        let mut chain = assets.chain.lock_recover();
        chain.read_file("Interface\\FrameXML\\ChatFrame.lua")
    };
    let src = match bytes {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(e) => {
            error!("ui_script: ChatFrame.lua read failed — emote commands will be dead: {e:#}");
            return;
        }
    };
    let table: Vec<&str> = src
        .lines()
        .map(str::trim)
        .filter(|l| is_emote_token_line(l))
        .collect();
    let count = table.len();
    if let Err(e) = script.run(&table.join("\n")) {
        error!("ui_script: emote token table failed to run: {e}");
        return;
    }
    // Sentinel: `EMOTE87` is `/sit`.
    let sentinel: Option<String> = script.lua().globals().get("EMOTE87_TOKEN").ok();
    match sentinel.as_deref() {
        Some("SIT") => info!("ui_script: {count} emote tokens loaded"),
        other => error!(
            "ui_script: emote token sentinel is {other:?}, not \"SIT\" ({count} lines) — \
             emote slash commands are broken"
        ),
    }
}

/// Whether a line is one of `ChatFrame.lua`'s `EMOTE<digits>_TOKEN = "<NAME>";` assignments, with
/// `NAME` in `[A-Z0-9_]`; that whole-line shape makes running it the same as reading data.
pub(crate) fn is_emote_token_line(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("EMOTE") else {
        return false;
    };
    let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    let Some(rest) = rest[digits..].strip_prefix("_TOKEN = \"") else {
        return false;
    };
    let Some(name) = rest.strip_suffix("\";") else {
        return false;
    };
    digits > 0
        && !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_entry_edge_parks_the_boot_vm_and_the_session_end_returns_it() {
        let mut world = World::new();
        let boot = UiScript::new().unwrap();
        let session = boot.session();
        world.insert_non_send(boot);
        arm_entry_ui_load(&mut world);
        assert!(
            world.get_non_send::<UiScript>().is_none(),
            "no VM in the deferral window"
        );
        assert!(world.get_resource::<PendingEntryUiLoad>().is_some());
        assert!(world.get_non_send::<ParkedBootVm>().is_some());
        // Left the world before the load ran: the VM comes back, untouched.
        unpark_boot_vm(&mut world);
        let vm = world
            .get_non_send::<UiScript>()
            .expect("the parked VM is back");
        assert_eq!(vm.session(), session, "the same VM, no session moved");
        assert!(world.get_non_send::<ParkedBootVm>().is_none());
    }

    /// The plate bits clear at the world entry (`0x401639`, in the entry setup `0x401570`) and
    /// not at a `ReloadUI()`, which never runs that setup. A capture world, so no UI is built.
    #[test]
    fn the_world_entry_clears_the_plate_bits_and_a_reload_does_not() {
        use crate::vplates::VPlateMode;
        let on = VPlateMode {
            enemies: true,
            friends: true,
        };
        let world_with = |on: VPlateMode| {
            let mut world = World::new();
            world.insert_resource(State::new(crate::char_select::ClientState::InWorld));
            world.insert_resource(crate::run_mode::CaptureMode);
            world.insert_resource(on);
            world
        };

        let mut world = world_with(on);
        world.insert_resource(PendingEntryUiLoad::default());
        run_pending_entry_load(&mut world);
        assert_eq!(*world.resource::<VPlateMode>(), VPlateMode::default());

        let mut world = world_with(on);
        world.insert_resource(ReloadUiPending(true));
        run_pending_reload(&mut world);
        assert!(!world.resource::<ReloadUiPending>().0, "the reload ran");
        assert_eq!(*world.resource::<VPlateMode>(), on, "a reload keeps them");
    }

    /// A `ReloadUI()` runs the leave-world sweep (`0x490c20`, past the armed latch) and then
    /// re-enters the world-enter cascade (`0x490168`), whose readers re-send the time, mail,
    /// battlefield and meeting-stone queries, only with the active player set (`0x490166 je`).
    #[test]
    fn a_reload_sweeps_and_reenters_the_world_enter_cascade_once_the_player_exists() {
        use crate::net::WorldEnterCascadeMessage;
        let reload = |seated: bool| {
            let mut world = World::new();
            world.insert_resource(State::new(crate::char_select::ClientState::InWorld));
            world.insert_resource(crate::run_mode::CaptureMode);
            world.init_resource::<Messages<WorldEnterCascadeMessage>>();
            world.init_resource::<Messages<WorldLeaveSweepMessage>>();
            world.init_resource::<LeavingWorldArmed>();
            world.resource_mut::<LeavingWorldArmed>().arm();
            if seated {
                world.spawn(crate::net::SelfPlayer);
            }
            world.insert_resource(ReloadUiPending(true));
            run_pending_reload(&mut world);
            assert!(!world.resource::<ReloadUiPending>().0, "the reload ran");
            let sweeps = world
                .resource_mut::<Messages<WorldLeaveSweepMessage>>()
                .drain()
                .count();
            let cascades = world
                .resource_mut::<Messages<WorldEnterCascadeMessage>>()
                .drain()
                .count();
            (sweeps, cascades)
        };
        assert_eq!(reload(true), (1, 1), "in the world");
        assert_eq!(
            reload(false).1,
            0,
            "no cascade before our own player's create"
        );
    }

    /// The IO thread's roster policy follows the VM that came up: the screen asks only when the
    /// stock glue VM stands, so a boot VM leaves the park to fetch the roster.
    #[test]
    fn the_roster_policy_follows_the_vm_that_stands() {
        use std::sync::atomic::Ordering;
        for stock in [true, false] {
            let mut world = World::new();
            let policy = crate::net::RosterPolicy(Default::default());
            policy.0.store(!stock, Ordering::SeqCst);
            world.insert_resource(policy.clone());
            world.insert_resource(super::super::glue::GlueStock(stock));
            world.insert_resource(State::new(crate::char_select::ClientState::Login));
            setup_script(&mut world);
            // The stock policy needs the stock files to have loaded: no install, no screen.
            let up = stock && benilla_formats::wow_data().is_some();
            assert_eq!(policy.0.load(Ordering::SeqCst), up, "stock {stock}");
        }
    }

    /// The agreements are read from their CVars before the first screen is set (`0x46a400` at
    /// `0x402b36`, `SET_GLUE_SCREEN` at `0x402b65`): a returning player's login shows the boxes,
    /// not the EULA, and a first launch still shows the EULA.
    #[test]
    fn the_login_screen_comes_up_on_the_accepted_agreements() {
        use crate::char_select::ClientState;
        let _ = benilla_formats::wow_data_or_skip!();
        let screens = |cvars: crate::cvars::Cvars| {
            let mut world = World::new();
            world.insert_resource(super::super::glue::GlueStock(true));
            world.insert_resource(State::new(ClientState::Login));
            world.insert_resource(cvars);
            world.init_resource::<super::super::glue_mgr::GlueMgr>();
            setup_script(&mut world);
            world
                .non_send::<UiScript>()
                .eval::<(bool, bool, bool)>(
                    "return TOSFrame:IsShown() == 1, AccountLoginUI:IsShown() == 1, \
                     AccountLoginAccountEdit:IsVisible() == 1",
                )
                .unwrap()
        };
        let mut accepted = crate::cvars::Cvars::with_value("readEULA", "1");
        accepted.set("readTOS", "1");
        assert_eq!(
            screens(accepted),
            (false, true, true),
            "accepted agreements go straight to the login boxes"
        );
        assert_eq!(
            screens(crate::cvars::Cvars::default()),
            (true, false, false),
            "a first launch is asked for the EULA"
        );
    }

    /// Under `WOW_GLUE_STOCK=1` the world holds the glue VM where it holds a boot VM, at startup
    /// on the login screen (`0x402b65`) and back from the world on the screen the exit enters
    /// (`0x402090`); a `ReloadUI()` never builds the glue.
    #[test]
    fn the_stock_glue_stands_where_the_boot_vm_does() {
        use crate::char_select::ClientState;
        use benilla_ui::script::VmKind;
        let shown = |world: &World, frame: &str| {
            benilla_formats::wow_data().is_none()
                || world
                    .non_send::<UiScript>()
                    .eval::<bool>(&format!("return {frame}:IsShown() == 1"))
                    .unwrap()
        };
        let mut world = World::new();
        world.insert_resource(super::super::glue::GlueStock(true));
        world.insert_resource(State::new(ClientState::Login));
        setup_script(&mut world);
        assert_eq!(world.non_send::<UiScript>().vm_kind(), VmKind::Glue);
        assert!(shown(&world, "AccountLogin"), "the boot screen is login");

        world.insert_resource(State::new(ClientState::CharSelect));
        world.init_resource::<PendingEntryUiLoad>();
        end_ui_session(&mut world);
        assert_eq!(world.non_send::<UiScript>().vm_kind(), VmKind::Glue);
        assert!(
            shown(&world, "CharacterSelect"),
            "a logout puts the character screen up"
        );

        world.insert_resource(State::new(ClientState::InWorld));
        world.insert_resource(crate::run_mode::CaptureMode);
        world.insert_resource(ReloadUiPending(true));
        run_pending_reload(&mut world);
        assert_eq!(
            world.non_send::<UiScript>().vm_kind(),
            VmKind::InGame,
            "the reload's VM is a boot VM, never the glue's"
        );
    }

    /// The reference re-makes its Lua state inside `UI_Init` (`0x48fe97`).
    #[test]
    fn the_world_entry_builds_its_own_vm_rather_than_adopting_the_glue_phases() {
        let mut world = World::new();
        let glue = UiScript::new().unwrap();
        let glue_session = glue.session();
        world.insert_non_send(glue);

        mint_entry_vm(&mut world);

        let entry = world
            .get_non_send::<UiScript>()
            .expect("the entry load has a VM to run on");
        assert_ne!(
            entry.session(),
            glue_session,
            "the entry VM is a new session — every VmMemo resets here, so a one-shot spent \
             against the frameless glue VM is spent again against the one that has an interface"
        );
    }

    /// A one-shot fired at the frameless glue VM reaches nobody (the reference's `0x703f50` drops
    /// an event with no listener), so the entry VM must be told again.
    #[test]
    fn a_one_shot_spent_against_the_character_screens_vm_is_spent_again_after_the_entry() {
        let mut world = World::new();
        world.insert_non_send(UiScript::new().expect("a glue VM"));
        let mut told: super::super::VmMemo<bool> = Default::default();

        assert!(
            told.claim(world.non_send::<UiScript>()),
            "the window: a feed fires its login one-shot at the frameless character-screen VM"
        );
        assert!(
            !told.claim(world.non_send::<UiScript>()),
            "and the memo has it as told — which is correct for THAT VM"
        );

        mint_entry_vm(&mut world);

        assert!(
            told.claim(world.non_send::<UiScript>()),
            "the VM the in-game UI loads onto has never been told, so the feed tells it again — \
             no run condition, no ordering edge, and nothing the feed had to know about"
        );
    }

    /// The reference's `GetTime` is `GetTickCount` × 0.001 (`0x515ea0`) and never restarts; stock
    /// `CooldownFrame_SetTimer` hides a cooldown whose start is not above 0.
    #[test]
    fn a_rebuilt_vm_inherits_the_running_gettime_clock() {
        use std::time::{Duration, Instant};

        let mut world = World::new();
        world.init_resource::<UiClock>();
        // The process has been up 90 s, on `Time<Real>` and on the dying VM alike.
        let startup = Instant::now() - Duration::from_secs(90);
        let mut time = Time::<bevy::time::Real>::new(startup);
        // `Time<Real>` measures `elapsed` from its first update, not from `startup`.
        time.update_with_instant(startup);
        time.update_with_instant(startup + Duration::from_secs(90));
        world.insert_resource(time);
        let mut dying = UiScript::new().unwrap();
        dying.set_now(90.0);
        world.insert_non_send(dying);
        // No in-game UI was loaded, so the shutdown tail and its file writes are skipped.
        world.init_resource::<PendingEntryUiLoad>();

        end_ui_session(&mut world);

        let vm = world
            .get_non_send::<UiScript>()
            .expect("the session end installs a fresh boot VM");
        assert_eq!(
            vm.now(),
            90.0,
            "the new VM's GetTime clock restarted at {} instead of the process's 90 s",
            vm.now()
        );
        // The conversion pair moved too: a 10-minute cooldown armed 30 s before the rebuild keeps
        // its real start.
        let clock = world.resource::<UiClock>();
        let armed = crate::spell::cooldowns::CooldownInfo {
            start: startup + Duration::from_secs(60),
            remaining_ms: 570_000,
            duration_ms: 600_000,
            enabled: true,
        };
        let triple = armed
            .ui_triple(clock.anchor, clock.ui_now)
            .expect("a running cooldown pushes a triple");
        assert_eq!(
            triple,
            (60_000, 600_000, true),
            "a cooldown armed before the rebuild derived start {} ms; stock Cooldown.lua hides \
             anything but `start > 0`",
            triple.0
        );
    }
}
