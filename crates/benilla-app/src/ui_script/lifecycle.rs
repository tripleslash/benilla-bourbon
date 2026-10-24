//! The UI session's lifecycle, all edges and nothing per-frame: `Startup` installs the glue
//! screens' VM; world entry builds a fresh VM and loads the in-game UI and every addon onto it;
//! leaving the world runs the reference's shutdown tail and installs the glue VM again; `ReloadUI()` is the two edges
//! back to back without leaving the world.

use bevy::prelude::*;

use benilla_assets::{LockRecover, LooseOpen};
use benilla_ui::script::UiScript;
use benilla_world::frame_pace::FrameStep;

use super::manifest::silenced_ui_load;
use super::{load_ingame_ui, CursorPayloadHeld, PlayerUiHover, UiClock, UiKeyboardCapture};
use crate::ui_script::addons;

pub(crate) fn setup_script(world: &mut World) {
    install_glue_side_vm(world, super::glue::GlueScreen::Login.key());
}

/// Install the VM the world holds outside the world: the glue screens', put up on the screen
/// named `screen`. A glue whose select screen did not load (logged as an error by [`new_vm`] or
/// the load) asks for no roster, and nothing else does.
pub(super) fn install_glue_side_vm(world: &mut World, screen: &str) {
    if !install_glue_vm(world, screen) {
        error!("ui_script: the glue VM did not come up with its character select screen");
    }
}

/// Build and install a boot VM, the one a `ReloadUI()` holds between its teardown and its load:
/// the strings the feeds resolve lines from and the emote tokens `/`-commands are built from
/// (`ui_chat::commands`), no frames and no font objects.
fn install_boot_vm(world: &mut World) {
    if let Some(script) = strings_vm(world) {
        world.insert_non_send(script);
    }
}

/// A boot VM: `GlobalStrings.lua` and the emote tokens, for a reader that needs them where the
/// world holds the glue VM. `None` (the world's VM removed) when
/// the VM cannot start.
pub(crate) fn strings_vm(world: &mut World) -> Option<UiScript> {
    let script = new_vm(world, UiScript::new)?;
    load_global_strings(world, &script);
    load_emote_tokens(world, &script);
    Some(script)
}

/// Build the glue VM, load the stock GlueXML onto it, install it and put `screen` up: the glue
/// builder `0x46a7b0` then `SET_GLUE_SCREEN` (`0x402b65` at startup, `0x402090` back from the
/// world). It holds what the glue state holds and nothing of a boot VM's: no `GlobalStrings.lua`,
/// no emote tokens.
fn install_glue_vm(world: &mut World, screen: &str) -> bool {
    let Some(mut script) = new_vm(world, UiScript::new_glue) else {
        return false;
    };
    // The build's size and aspect, and its clear of a pending rebuild (`0x46a7e0`, `0x46a81c`).
    super::glue::glue_built(world);
    seat_raster_seam_for_load(world, &mut script);
    // The CVar table before the load and the first screen, as the in-game build seeds its own:
    // this VM is born before `cvars::sync_cvars`'s per-VM seed, and the character screen's
    // `OnShow` names the realm `GetServerName` reads off `realmName` (`0x46d289`).
    if let Some(cvars) = world.get_resource::<crate::cvars::Cvars>() {
        cvars.seed_vm(&mut script);
    }
    seed_agreements(world, &mut script);
    let start = bevy::platform::time::Instant::now();
    let failures = super::glue::load_glue(&mut script);
    super::glue::finish_glue_load_log(&mut script);
    start_server_alert(world, &script);
    // What the glue carries across its rebuild, in place before the screen's `OnShow` reads it.
    super::glue_mgr::seat_glue_build(world, &mut script);
    // `SetCurrentScreen`'s buffer is the client's (`[0xb41478]`) and outlives the Lua state: a
    // rebuild signals what it holds (`0x46b9a2`), and the screen's handler writes the same name.
    script.set_glue_screen_name(screen);
    super::glue::signal_glue_screen(&mut script, screen);
    info!(
        "ui_script: stock glue up in {:.0} ms on \"{}\" ({} load failure(s))",
        start.elapsed().as_secs_f32() * 1000.0,
        screen,
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
        if let Some(shared) = world.get_non_send::<super::glue_mgr::SharedGlueMgr>() {
            shared.get_mut().set_alert_fetch(fetch);
        }
    }
}

/// Seat the glue VM with the four agreements' flag cells, which `0x46a400` reads off the `read*`
/// CVars once, at the first glue build (`0x402b36`, then `SET_GLUE_SCREEN` at `0x402b65`): the
/// login screen's `OnShow` asks them at once. Every glue build after it (`0x402090`, a size
/// change) seats the same cells unread, and only the accepts write them.
fn seed_agreements(world: &mut World, script: &mut UiScript) {
    let Some(shared) = world
        .get_non_send::<super::glue_mgr::SharedGlueMgr>()
        .cloned()
    else {
        return;
    };
    let Some(cvars) = world.get_resource::<crate::cvars::Cvars>() else {
        return;
    };
    let notices = shared
        .get_mut()
        .notices
        .get_or_insert_with(|| {
            benilla_ui::script::GlueNotices::new(super::glue_mgr::read_agreements(cvars))
        })
        .clone();
    script.set_glue_notices(notices);
}

/// A fresh VM from `make` on the process clock with the addon asset probes wired, or `None` (the
/// world's VM removed) when the VM cannot start.
fn new_vm<E: std::fmt::Display>(
    world: &mut World,
    make: fn() -> Result<UiScript, E>,
) -> Option<UiScript> {
    // The client globals the VM holds that no rebuild resets: `[0xcf4d24]`, written only by
    // `SetEuropeanNumbers` (`0x771c40`), outlives a `ReloadUI` and a logout.
    let european_numbers = world
        .get_non_send::<UiScript>()
        .is_some_and(UiScript::european_numbers);
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
    script.set_european_numbers(european_numbers);
    // `[0xc0e080]`, which start-up sets once from `Wow.ini`, so every VM answers `GetLocale` alike.
    script.set_locale(super::client_locale().index());
    // `[0x882730]`, `Wow.ini`'s `Region`, read once alike and handed to the name check.
    script.set_client_region(super::client_region());
    // The name checks' word lists, client globals start-up compiles once (`0x402b7f` → `0x6c91a0`).
    if let Some(names) = world.get_resource::<crate::text_filter::NameFilter>() {
        script.set_name_lists(std::rc::Rc::new(names.clone()));
    }
    seed_vm_clock(world, &mut script);
    seed_game_time(world, &mut script);
    // The reference seeds `rand` with the clock before `WinMain` (`0x5b800b`). Deviation: a
    // capture run (`scenario_active`) keeps the CRT's default seed, 1, so its frames reproduce.
    if crate::run_mode::scenario_active() {
        script.lua().srand(1);
    }
    install_addon_asset_resolvers(world, &mut script);
    info!("ui_script: new VM, session {}", script.session());
    Some(script)
}

/// Start the new VM's `GetTime()` where the process already is and re-anchor [`UiClock`] to it,
/// both off the [`FrameStep`] the VM ticks on, before this frame's step: every VM is built ahead
/// of the frame's tick (`Startup`, `PreUpdate`, a state edge), which then adds the step. The reference's `GetTime` is `GetTickCount` × 0.001
/// (`0x515ea0`, via `0x42c010` → `0x42b790`), an OS clock that never restarts; ours lives in the
/// VM, rebuilt at every login and `ReloadUI`, and a clock back at zero puts every converted
/// cooldown start in the past, which stock `CooldownFrame_SetTimer` hides (`Cooldown.lua:3`).
fn seed_vm_clock(world: &mut World, script: &mut UiScript) {
    let (anchor, now) = world
        .get_resource::<FrameStep>()
        .copied()
        .unwrap_or_default()
        .before_step();
    script.set_now(now);
    if let Some(mut clock) = world.get_resource_mut::<UiClock>() {
        *clock = UiClock {
            anchor,
            ui_now: now,
        };
    }
}

/// Start the new VM's `GetGameTime()` at the game clock as it stands: the reference's clock is one
/// object (`0xce8538`) built at start-up with every field -1 (`0x643300`, `0x642300`) and reset
/// by no logout or `ReloadUI`, so a VM answers -1, -1 before any server time and the last known
/// time after it, from the first line its addons run. fork: moved by the `serverTimeOffset` hours.
fn seed_game_time(world: &World, script: &mut UiScript) {
    if let Some(time) = world.get_resource::<crate::net::ServerTime>() {
        let offset = world
            .get_resource::<crate::net::TimeOffset>()
            .copied()
            .unwrap_or_default();
        let (hour, minute) = offset.shown(time.0.hour(), time.0.minute());
        script.set_game_time(hour, minute);
    }
}

/// Wire the loose files into art and fonts: the decoder's AddOns folder, beside the install's
/// index it already reads, and the VM's texture, size and font probes, so the path forms of
/// `SetTexture` and `SetFont` answer the reference's 1|nil load verdict, and a region with no
/// authored size on an axis takes it from its art as the client's virtual size getters do. Each
/// probe resolves through the store and the open its renderer reads (an interface texture's,
/// [`LooseOpen::Index`], and a font face's, [`LooseOpen::BareNameFirst`]), so the answer is what
/// draws; they test existence, not decode.
fn install_addon_asset_resolvers(world: &mut World, script: &mut UiScript) {
    let Some(mut assets) = world.get_resource_mut::<benilla_assets::WorldAssets>() else {
        return;
    };
    assets.set_loose_addon_root(addons::root());
    let loose = assets.loose().clone();
    let chain = assets.chain.clone();
    let size_chain = chain.clone();
    let size_loose = loose.clone();
    let cursor_chain = chain.clone();
    let cursor_loose = loose.clone();
    let font_chain = chain.clone();
    let font_loose = loose.clone();
    // `SetTexture` runs per frame in the addons that set an icon every `OnUpdate`: the probe
    // keeps what it read of a file.
    let probe = benilla_assets::SpriteProbe::default();
    script.set_texture_probe(Box::new(move |path| probe.found(&chain, &loose, path)));
    // Keyed by the path exactly as the region carries it, so a hit allocates nothing (this is
    // asked per zero-size textured region per layout pass); misses are cached too.
    let sizes: std::cell::RefCell<std::collections::HashMap<String, Option<(u32, u32)>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    script.set_texture_size_probe(Box::new(move |path| {
        if let Some(cached) = sizes.borrow().get(path) {
            return *cached;
        }
        match benilla_assets::sprite_dimensions(&size_chain, &size_loose, path) {
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
    // `SetCursor`'s 32×32 test reads the picture the cursor code opens (`0x449610`), its `.blp`
    // alone, decoded each call as `0x523c98` waits on the load.
    script.set_cursor_size_probe(Box::new(move |path| {
        benilla_assets::file_texture_rgba(&cursor_chain, &cursor_loose, path)
            .map(|(w, h, _)| (w, h))
    }));
    // Existence over the same stores, in the same order, as `ui_text::engine`'s face loader,
    // memoised because a path's answer cannot change and an addon may call `SetFont` every frame.
    let seen: std::cell::RefCell<std::collections::HashMap<String, bool>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    script.set_font_probe(Box::new(move |path| {
        if let Some(&cached) = seen.borrow().get(path) {
            return cached;
        }
        let found = font_found(&font_chain, &font_loose, path);
        seen.borrow_mut().insert(path.to_string(), found);
        found
    }));
}

/// Whether a font face's open of `path` finds a file: the loose files by bare name first, then the
/// chain, as `ui_text::engine`'s face loader reads them.
pub(super) fn font_found(
    chain: &std::sync::Mutex<benilla_formats::Chain>,
    loose: &benilla_assets::LooseFiles,
    path: &str,
) -> bool {
    loose.find(path, LooseOpen::BareNameFirst).is_some() || chain.lock_recover().contains(path)
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

/// The glue VM, parked between the world-entry edge and the deferred load, out of every feed's
/// reach: each feed takes the VM as an `Option` and returns on `None`.
///
/// **The window is the whole sliced load, not just the wait before it.** The load spends itself
/// over frames now ([`PendingEntryUiLoad`]), and a VM with half a frame tree in it is no more
/// ready for a feed than one with none — so a slice that yields parks the VM again rather than
/// handing it back, and only [`EntryStage::Finish`] returns it to the world.
pub(crate) struct ParkedGlueVm(UiScript);

/// `OnEnter(InWorld)`: arm the deferred entry load and park the glue VM.
pub(crate) fn arm_entry_ui_load(world: &mut World) {
    world.insert_resource(PendingEntryUiLoad::default());
    if let Some(vm) = world.remove_non_send::<UiScript>() {
        world.insert_non_send(ParkedGlueVm(vm));
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
    if let Some(mut script) = new_vm(world, UiScript::new) {
        crate::ui_macro::install_icon_source(world, &mut script);
        world.insert_non_send(script);
    }
}

/// Rebuild the glue VM for a change of the data the chain answers from (a realm of the other
/// flavor entered, [`crate::flavor`]): GlobalStrings and the emote tokens come from the new
/// chain. The dying VM's CVars fold first, as at world entry.
/// Under the stock glue screens the glue VM is the one standing; it is rebuilt off the new chain
/// (its GlueXML, Turtle WoW's own under the Turtle layer) once the manager is idle,
/// [`rebuild_stale_glue_vm`].
pub(crate) fn reinstall_boot_vm(world: &mut World) {
    let glue_up = world
        .get_non_send::<UiScript>()
        .is_some_and(|s| s.vm_kind() == benilla_ui::script::VmKind::Glue);
    if glue_up {
        return;
    }
    crate::cvars::fold_dying_vm_cvars(world);
    install_boot_vm(world);
}

/// Rebuild a stock glue VM built under the other flavor ([`crate::flavor`] switched the chain at a
/// realm entry) on the screen it shows, once that flavor is committed and the glue manager is idle
/// on `login` or `charselect`:
/// the same rebuild a logout makes (`0x46a7b0`), so nothing mid-operation is lost, and the new VM
/// runs the new chain's GlueXML and the new flavor's create catalog.
pub(crate) fn rebuild_stale_glue_vm(
    script: Option<NonSend<UiScript>>,
    flavor: Option<Res<crate::flavor::ClientFlavor>>,
    committed: Option<Res<crate::flavor::Committed>>,
    mgr: Option<NonSend<super::glue_mgr::SharedGlueMgr>>,
    mut commands: Commands,
) {
    let (Some(script), Some(mgr)) = (script, mgr) else {
        return;
    };
    let flavor = flavor.as_deref().copied().unwrap_or_default();
    // Only a settled flavor: the realm's roster arrived, or a failed entry switched back. Between
    // the switch and the roster the realm entry is in flight, and a rebuild would drop it.
    if committed.is_none_or(|c| c.0 != flavor) {
        return;
    }
    if script.vm_kind() != benilla_ui::script::VmKind::Glue || script.flavor() == flavor.ui() {
        return;
    }
    let Some(screen) = mgr.get().idle_screen(&script.glue_screen()) else {
        return;
    };
    info!(
        "ui_script: the glue VM predates the {flavor:?} chain; rebuilt on \"{}\"",
        screen.key()
    );
    commands.queue(move |world: &mut World| {
        crate::cvars::fold_dying_vm_cvars(world);
        install_glue_side_vm(world, screen.key());
    });
}

/// Put the parked boot VM back into the world, if one is parked.
fn unpark_glue_vm(world: &mut World) {
    if let Some(ParkedGlueVm(vm)) = world.remove_non_send::<ParkedGlueVm>() {
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
    state: Option<Res<State<crate::client_state::ClientState>>>,
) -> bool {
    pending.is_none()
        && state.is_some_and(|s| *s.get() == crate::client_state::ClientState::InWorld)
}

/// `PreUpdate`, after [`run_pending_reload`] so a reload never interleaves an armed entry load:
/// run the armed entry load once the cover has presented, at once when there is no cover (a
/// capture booting straight `InWorld`). Leaving the world first drops the latch unrun, and
/// [`shut_down_session`] then skips the shutdown writes, so a UI-less VM never overwrites the
/// saved files.
pub(crate) fn run_pending_entry_load(world: &mut World) {
    if world.get_resource::<PendingEntryUiLoad>().is_none() {
        return;
    }
    let in_world = *world
        .resource::<State<crate::client_state::ClientState>>()
        .get()
        == crate::client_state::ClientState::InWorld;
    if !in_world {
        // Left the world before the load ran: nothing to build a UI for.
        world.remove_resource::<PendingEntryUiLoad>();
        unpark_glue_vm(world);
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
    // `InitializeGame` flags the world session's read-only rows whether or not a UI then loads
    // (`0x48f566`-`0x48f584`); `end_ui_session` is `ShutdownGame`'s clear. A `/reload` keeps them.
    if world.resource::<PendingEntryUiLoad>().stage == EntryStage::Armed {
        flag_world_read_only(world);
    }
    if !ui_wanted(world) {
        world.remove_resource::<PendingEntryUiLoad>();
        unpark_glue_vm(world);
        crate::vplates::clear_at_world_entry(world);
        return;
    }
    // **The first step builds the VM the load runs on** (decision 2226, [`mint_entry_vm`]): the
    // glue phase's VM goes back into the world only so its CVars can be folded, and a fresh boot
    // VM takes its place. Every later slice picks the parked, part-built one back up.
    if world.resource::<PendingEntryUiLoad>().stage == EntryStage::Armed {
        unpark_glue_vm(world);
        mint_entry_vm(world);
    }
    // The parked VM first (`ParkedGlueVm` — the ordinary path since 1978, and every slice after
    // the first), then the world's own: the freshly minted entry VM on the first slice.
    let parked = world
        .remove_non_send::<ParkedGlueVm>()
        .map(|ParkedGlueVm(vm)| vm);
    let Some(mut script) = parked.or_else(|| world.remove_non_send::<UiScript>()) else {
        warn!("ui_script: entering the world with no VM — the in-game UI will not load");
        world.remove_resource::<PendingEntryUiLoad>();
        crate::vplates::clear_at_world_entry(world);
        return;
    };
    let frame_start = bevy::platform::time::Instant::now();
    world.resource_mut::<PendingEntryUiLoad>().frames += 1;
    // Every step that runs Lua runs with a call host lent, built off the world as a system's would
    // be, as the one-shot load's phases do: a call a file or a handler makes applies inside its
    // verb.

    let done_at_start = world.resource::<PendingEntryUiLoad>().done;
    loop {
        let stage = world.resource::<PendingEntryUiLoad>().stage;
        match stage {
            EntryStage::Armed => {
                // Before the first file, as the one-shot path does ([`load_ingame_ui_on_world_entry`]).
                crate::minimap::restore_default_mask(world);
                crate::ui_billing::arm_at_world_entry(world);
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
                    let _ = crate::script_calls::lent_by_world(world, &mut script, |script| {
                        if has_core {
                            super::manifest::close_core(script, toc)
                        } else {
                            super::manifest::load_core_bindings(script)
                        }
                    });
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
                    let _ = crate::script_calls::lent_by_world(world, &mut script, |script| {
                        super::manifest::load_entry(script, &entry, &mut toc)
                    });
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
                let _ = crate::script_calls::lent_by_world(world, &mut script, |script| {
                    super::manifest::bootstrap_positions(script)
                });
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
                    let walk = crate::script_calls::lent_by_world(world, &mut script, |script| {
                        addons::open_third_party(script, identity.as_ref(), &roster, version_check)
                    });
                    let mut pending = world.resource_mut::<PendingEntryUiLoad>();
                    // The bar counted the addons as one step; it counts each now.
                    pending.total += walk.len().saturating_sub(1);
                    pending.addons = Some(walk);
                }
                let mut walk = world
                    .resource_mut::<PendingEntryUiLoad>()
                    .addons
                    .take()
                    .expect("opened above");
                crate::script_calls::lent_by_world(world, &mut script, |script| {
                    addons::load_one_addon(&mut walk, script, next)
                });
                let mut pending = world.resource_mut::<PendingEntryUiLoad>();
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
                entry_finish(
                    world,
                    &mut script,
                    pending.identity.clone(),
                    LoadSeat::Roster,
                );
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
        // At least one load step a slice: the first slice's setup (the VM, the toc, the install's
        // loose-file index) can spend the budget alone, and a slice that loads nothing would make
        // an uncovered entry wait a frame for no reason.
        if frame_start.elapsed() >= ENTRY_LOAD_FRAME_BUDGET
            && world.resource::<PendingEntryUiLoad>().done > done_at_start
        {
            break; // resume next frame
        }
    }
    // Back into the park, not into the world: the tree is still part-built, and 1978's whole
    // point is that no feed sees a VM in that state. `Finish` above is the one exit that
    // publishes it.
    world.insert_non_send(ParkedGlueVm(script));
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
        .get_resource::<crate::glue_session::roster::Roster>()
        .and_then(crate::ui_macro::identity);
    // The realm's character list is the enable store's node set: the reference keeps one
    // `ADDONSTATELIST` node per character, and an addon this one never set resolves from the rest.
    let roster: Vec<String> = world
        .get_resource::<crate::glue_session::roster::Roster>()
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
    // call, before `cvars::sync_cvars`'s per-VM seed.
    match world.get_resource::<crate::cvars::Cvars>() {
        Some(cvars) => cvars.seed_vm(script),
        None => script.register_cvars(crate::cvars::registered_pairs()),
    }
    // The item locks before any file or event runs: a `/reload`'s survive, as on the reference's
    // item objects.
    crate::ui_items::locks::seed_vm_item_locks(world, script, seat == LoadSeat::Live);
    crate::ui_party::seed_vm_ready_check(world, script, seat == LoadSeat::Live);
    crate::ui_guild::seed_vm_guild(world, script, seat == LoadSeat::Live);
    // The realm before the load, since addons read `GetRealmName()` at file scope; the roster
    // carries the realm-list entry this session connected to.
    let realm = world
        .get_resource::<crate::glue_session::roster::Roster>()
        .and_then(|r| r.realm.as_ref().map(|r| r.name.clone()))
        .unwrap_or_default();
    script.set_realm_name(&realm);
    // `GetBillingTimeRested` reads the connection's word (`0x48ec59`), there for every VM.
    if let Some(billing) = world.get_resource::<crate::net::SessionBilling>() {
        script.set_billing_time_rested(billing.0.time_rested);
    }
    // `CGGameUI::InitializeGame` flags these read-only before `UI_Init` loads anything
    // (`0x48f566`-`0x48f584`), so an in-game `SetCVar` of one raises, logout handlers included,
    // and no other write stores; `ShutdownGame` clears them (`0x491240`). The VM's flags die with
    // it (the registry's are [`load_ingame_ui_on_world_entry`]'s). After `set_realm_name`, whose
    // write is the world entry's own.
    for name in benilla_ui::script::IN_WORLD_READ_ONLY_CVARS {
        script.set_cvar_read_only(name, true);
    }
    // The player too, before the addon walk. The record first: the reference's
    // `UnitName`/`UnitRace`/`UnitClass`/`UnitSex` read only a copy of the char-enum row made at the
    // Enter World commit (`0x5abd9e`) and never cleared, and each rebuilt VM is told once. Then
    // the unit: the live player on a `/reload`, else the roster's stand-in until the descriptor
    // streams in.
    if let Some(record) = world
        .get_resource::<crate::glue_session::roster::Roster>()
        .and_then(|roster| record_from_roster(roster, crate::ui_unit::world_unit_names(world)))
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
                .get_resource::<crate::glue_session::roster::Roster>()
                .and_then(|roster| {
                    seat_from_roster(roster, crate::ui_unit::world_unit_names(world))
                })
            {
                script.set_unit("player", Some(seat));
            }
        }
    }
    // A reload's group, before any file reads it: the reference's frames load over the engine's
    // roster and hear no group event for it ([`crate::ui_party::seed_reloaded_vm`]).
    if seat == LoadSeat::Live {
        crate::ui_party::seed_reloaded_vm(world, script);
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
    // The engine globals a previous VM folded out: the quest log's fold mask and the tradeskill
    // repeat.
    crate::ui_quest_log::seed_quest_folds(world, script);
    crate::ui_tradeskill::seed_repeat(world, script);
    crate::ui_unit::seed_default_language(world, script);
    crate::ui_unit::seed_language_table(world, script);
    // The world map's continent and zone lists: static DBC data in the reference, read at file
    // scope (Astrolabe builds its whole zone table from `GetMapContinents`/`GetMapZones` there).
    crate::ui_world_map::seed_world_map_catalog(world, script);
    // The auction house's category tree, the other static DBC data the reference holds from load:
    // `Blizzard_AuctionUI.lua` reads `GetAuctionItemClasses()` at file scope, and addons such as
    // Tmog build their item-type tables from `GetAuctionItemSubClasses` there, before any session.
    // `feed_auction` recreates it per open; a nil `cloth` from an empty tree fails such a load.
    {
        let classes = world.get_resource::<crate::ui_items::ItemClasses>();
        let subclasses = world.get_resource::<crate::ui_items::ItemSubClasses>();
        script.set_auction_item_classes(crate::ui_auction::categories(classes, subclasses));
    }
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
fn entry_finish(
    world: &mut World,
    script: &mut UiScript,
    identity: Option<(String, String)>,
    seat: LoadSeat,
) {
    // Read out of the world first, for the phases below.
    let ui_scale = world
        .get_resource::<super::UiScaleCvar>()
        .map_or(1.0, |c| c.0);
    let zoom = {
        let z = world.resource::<crate::minimap::MinimapZoom>();
        (z.outdoor, z.inside)
    };
    // A plate setting `config.toml` still holds, carried into FrameXML's saved variables once.
    // Only by a load with the layer, which registers `FRIENDNAMEPLATES_ON` for save; a stock-UI
    // load leaves it for the next layered one.
    let legacy_plates = if super::manifest::layer_enabled() {
        world
            .get_resource_mut::<crate::cvars::Cvars>()
            .map(|mut cvars| crate::vplates::take_legacy_settings(&mut cvars))
            .unwrap_or_default()
    } else {
        Default::default()
    };
    // Each phase runs with a call host lent, built off the world as a system's would be, so a call
    // an `ADDON_LOADED` or `VARIABLES_LOADED` handler makes applies inside its verb, as the
    // reference's do; the world reads between them are the load's own.
    crate::script_calls::lent_by_world(world, script, |script| {
        crate::ui_chat::commands::register_dev_commands(script, crate::run_mode::dev_affordances());
        // The new Minimap's zoom indices from the persisted CVars, as the reference's minimap
        // reset copies each CVar into its live index; `Minimap:SetZoom` owns them from here.
        script.set_minimap_zoom(zoom.0, zoom.1);
        // The saved-variables chunk runs after the files set their defaults and before
        // `VARIABLES_LOADED`, the reference's order (`AddOn_Load` `0x51f240`); reversed, the
        // defaults always win. Before `VARIABLES_LOADED` too, the plate setting, whose
        // `UIParent_OnEvent` arm runs the first `UpdateNameplates()` (`UIParent.lua:231-234`).
        read_saved_variables(script, |script| {
            crate::vplates::seat_legacy_settings(script, &legacy_plates)
        });
    });
    // The chat cache restores after `VARIABLES_LOADED` and before any chat is routed: its
    // `UPDATE_CHAT_WINDOWS` registers each chat frame for its `CHAT_MSG_*` events
    // (`ChatFrame.lua:1261-1273`).
    crate::ui_chat::restore_chat_looks(world, script);
    // The UI scale lands on `UIParent` once the files, the addons and `VARIABLES_LOADED` are
    // done, after the chat reader (`0x4900d6`) and before the `PLAYER_LOGIN` arm (`0x49011d`),
    // through the display handler (`0x49010e` → `0x492e90`). So every `OnLoad` sees `UIParent`
    // at scale 1 and the root's size.
    let display = {
        let mut q = world.query_filtered::<&Window, With<bevy::window::PrimaryWindow>>();
        q.single(world).map_or((0, 0), |w| {
            (w.width().round() as u32, w.height().round() as u32)
        })
    };
    let cached = world.get_resource::<super::UiAutoScaleCache>().map(|c| c.0);
    let mut cache = cached.flatten();
    crate::script_calls::lent_by_world(world, script, |script| {
        script.seat_ui_scale(ui_scale, display, &mut cache)
    });
    if let Some(mut slot) = world.get_resource_mut::<super::UiAutoScaleCache>() {
        slot.0 = cache;
    }
    script.arm_player_login();
    // The load's record goes to `Logs\FrameXML.log` (`0x490187`-`0x4901b6`), after any
    // `LoadAddOn` block the load ran; nothing reaches chat.
    script.finish_ui_load_log();
    super::load_log::write(script.take_load_log_writes());
    // `DAMAGE_TEXT_FONT` binds at the end of the world entry's load, after every addon's
    // `ADDON_LOADED` (where MikScrollingBattleText and pfUI assign it): `0x6c8470` is called only at
    // `0x401620` in `0x401570`, after the UI load at `0x401602`, and `0x401570` only from Enter
    // World (`0x46c236`). A `/reload` runs the teardown and the load alone (`0x495664`,
    // `0x495669`), so the face the login read stays until the next login.
    if seat == LoadSeat::Roster {
        world.insert_resource(crate::combat_text::read_damage_text_font(script));
    }
    // The load is over: disarm its instruction bound, so no handler is killed for being slow.
    script.clear_instruction_budget();
    world.insert_resource(AddOnIdentity(identity));
}

/// The end of a load edge, after the VM is back in the world.
fn arm_leaving_world(world: &mut World) {
    // Arm the world latch, the reference's `0x490168 call 0x4908c0`: a reload creates no entity
    // to re-arm it, so without this a second `/reload` would not fire `PLAYER_LEAVING_WORLD`. Only
    // with our own player present (`0x490166 je`, the GUID still 0 on a fresh login, which arms
    // from the player's create instead), so a load the server then refuses arms nothing.
    let seated = world
        .query_filtered::<(), With<crate::net::SelfPlayer>>()
        .iter(world)
        .next()
        .is_some();
    if let Some(mut armed) = world
        .get_resource_mut::<super::LeavingWorldArmed>()
        .filter(|_| seated)
    {
        armed.arm();
    }
}

/// Seat the VM's screen size and text measurer for the load, under the current window: the
/// load-edge half of [`super::extract::tick_script`]'s per-frame pair. A fresh VM answers 1024×768
/// until then, and a value computed once at load keeps it: stock `WorldMapFrame_OnLoad` sizes the
/// full-screen `BlackoutWorld` from `GetScreenWidth()` and `GetScreenHeight()` and never again
/// (`WorldMapFrame.lua:19-28`), which answer the root while `UIParent` is still at scale 1. The
/// atlas is built at `Startup` before the glue VM ([`crate::ui_text::FontAtlasInit`]); with none
/// (no window, no fonts) the per-frame pass seats the measurer once one exists.
fn seat_raster_seam_for_load(world: &mut World, script: &mut UiScript) {
    let (w, h) = {
        let mut q = world.query_filtered::<&Window, With<bevy::window::PrimaryWindow>>();
        q.single(world)
            .map_or((0.0, 0.0), |win| (win.width(), win.height()))
    };
    let s = super::seam_scale(h);
    if w > 0.0 && h > 0.0 {
        // The VM lives in a 768-tall root, `768·a` wide for the window's aspect `a`. No
        // `UIParent_ManageFramePositions` here, unlike `tick_script`: nothing is laid out yet.
        script.set_window_size(w, h);
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
    // Before the load, as `0x48f590` runs before `0x48fbf0`: an addon's file scope sets the mask
    // after it, and a `/reload` never takes this step.
    crate::minimap::restore_default_mask(world);
    // `InitializeGame` arms the play-time warning before it jumps to `UI_Init` (`0x48f636`-`0x48f681`).
    crate::ui_billing::arm_at_world_entry(world);
    flag_world_read_only(world);
    load_ingame_ui_seated(world, LoadSeat::Roster);
}

/// `InitializeGame`'s read-only flags on the registry's world-session rows (`0x48f566`-`0x48f584`),
/// before the UI gate; [`end_ui_session`] clears them as `ShutdownGame` does (`0x491240`).
fn flag_world_read_only(world: &mut World) {
    if let Some(mut cvars) = world.get_resource_mut::<crate::cvars::Cvars>() {
        for name in benilla_ui::script::IN_WORLD_READ_ONLY_CVARS {
            cvars.set_read_only(name, true);
        }
    }
}

/// [`load_ingame_ui_on_world_entry`] under the `"player"` its edge has.
fn load_ingame_ui_seated(world: &mut World, seat: LoadSeat) {
    // The parked VM goes back into the world, where the CVar fold reads it; a reload has none
    // parked and uses the live one.
    unpark_glue_vm(world);
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
    // The files run with a call host lent, built off the world as a system's would be, so a call
    // a file makes applies inside its verb; [`entry_finish`] lends one to each of its phases.
    silenced_ui_load(&mut script, |script| {
        crate::script_calls::lent_by_world(world, script, |script| {
            let _ = load_ingame_ui(script, identity.as_ref(), &roster, version_check);
        });
        entry_finish(world, script, identity, seat);
    });
    world.insert_non_send(script);
    arm_leaving_world(world);
}

/// The live player's seat off the world, through the unit feed's own builder; `None` before our
/// own create. A world without the feed's resources (a bare test world) has none to read.
fn live_player_seat(world: &mut World) -> Option<crate::ui_unit::LivePlayerSeat> {
    crate::script_calls::run_cached(world, crate::ui_unit::live_player_seat).unwrap_or_else(|e| {
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
    roster: &crate::glue_session::roster::Roster,
    names: crate::ui_unit::UnitNames<'_>,
) -> Option<benilla_ui::script::PlayerRecord> {
    let row = roster.pending_row()?;
    let race = names.race(row.race);
    let class = names.class(row.class);
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
    roster: &crate::glue_session::roster::Roster,
    names: crate::ui_unit::UnitNames<'_>,
) -> Option<benilla_ui::script::UnitState> {
    let row = roster.pending_row()?;
    let race = names.race(row.race);
    let class = names.class(row.class);
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
        faction_group: names.faction_group(row.race).map(str::to_string),
        ..Default::default()
    })
}

/// The character the loaded AddOn state belongs to, kept so the shutdown writes go back to its
/// files after the roster's pick is gone.
#[derive(Resource, Default)]
pub(crate) struct AddOnIdentity(pub(crate) Option<(String, String)>);

/// The UI shutdown's `PlaySound` suppression scope (`0x490bd1` → `0x458f50`, closed at
/// `0x490cd9` → `0x458f60`), around the whole of it, on the VM in the world: `true` when entered.
fn silence_ui_shutdown(world: &World) -> bool {
    world
        .get_non_send::<UiScript>()
        .map(UiScript::push_sound_suppression)
        .is_some()
}

/// Leave [`silence_ui_shutdown`]'s scope on the same VM.
fn unsilence_ui_shutdown(world: &World) {
    if let Some(script) = world.get_non_send::<UiScript>() {
        script.pop_sound_suppression();
    }
}

/// Which root runs the UI shutdown `0x490bd0`, which decides the order of its front.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ShutdownRoot {
    /// `/reload` (`0x495664` → `0x490bd0`): the world and its objects stay.
    Reload,
    /// The world's teardown `0x401ee0`, behind a logout, a disconnect, a refused login and an
    /// exit in the world: it destroys the objects first (`0x402011` → `0x467700`), the local
    /// player's destructor sweeping (`0x5dd543`), and nulls the object manager (`0x46771e`)
    /// before it calls the shutdown (`0x402039` → `0x491180` → `0x490bd0`). At the character
    /// screen it returns at once (`0x401ef3`, `[0x882734]` clear), so a quit there runs nothing.
    WorldTeardown,
}

/// The shutdown up to `PLAYER_LOGOUT`, in `root`'s order; `true` when it opened the suppression
/// scope (`0x490bd1`), which the caller closes after the tail (`0x490cd9`).
///
/// - `/reload`: the scope, the head with its sends and events, then past the active-player guard
///   (`0x490bee`-`0x490bf5`) the found arm's move-action cancel (`0x490c1b` → `0x60fb60(0, 1)`)
///   and the sweep (`0x490c20`).
/// - The world's teardown: the sweep from the local player's destructor, then the scope and a
///   silent head: with no object manager `SetSelection(0, 0)` only clears the selection
///   (`0x493549` → `0x467d00`, `0x4939d3`, returning at `0x4939df` before the send and the
///   event), and `CMSG_CANCEL_TRADE` still goes. The guard then finds no player.
///
/// The sweep runs only with our own player present: on a `/reload` the guard's `0x468550`; on the
/// teardown the destructor's own gate (`0x5dd71c`), which the world latch stands for, as a logout
/// despawns our avatar in its packet's frame, a frame before this runs: our own create arms it
/// (`0x5deb60`), so a refused login sweeps nothing.
fn run_shutdown_front(world: &mut World, root: ShutdownRoot) -> bool {
    let seated = world
        .query_filtered::<(), With<crate::net::SelfPlayer>>()
        .iter(world)
        .next()
        .is_some();
    match root {
        ShutdownRoot::Reload => {
            let silenced = silence_ui_shutdown(world);
            run_ui_shutdown_head(world, true);
            if seated {
                cancel_move_action(world);
                super::run_leave_world_sweep(world);
            }
            silenced
        }
        ShutdownRoot::WorldTeardown => {
            super::run_leave_world_sweep(world);
            let silenced = silence_ui_shutdown(world);
            run_ui_shutdown_head(world, false);
            silenced
        }
    }
}

/// `0x490bd0`'s head, ahead of its active-player guard: `SetSelection(0, 0)` (`0x490bdf`), then
/// `CMSG_CANCEL_TRADE` (`0x490be4` → `0x5d4310`), a send the sweep's trade leg repeats.
///
/// `loud`, with the object manager up (a `/reload`): the deselect's teardown (`0x493910(1)`)
/// closes a loot window on the target, sends `CMSG_SET_SELECTION` 0 and tells the VM with
/// `PLAYER_TARGET_CHANGED` (`0x493a2a`, `0x493a73`), unless the player is charmed, which makes
/// `SetSelection` a no-op (`0x493589`-`0x49359c`). Not `loud`, after the world's teardown: the
/// selection is only cleared ([`ShutdownRoot::WorldTeardown`]).
fn run_ui_shutdown_head(world: &mut World, loud: bool) {
    if loud {
        let looting = world
            .get_resource::<crate::ui_loot::LootState>()
            .and_then(crate::ui_loot::LootState::source);
        // A harness without the target stack has no selection to drop.
        let deselected =
            crate::script_calls::run_cached(world, crate::target::deselect_on_ui_shutdown)
                .ok()
                .flatten();
        // The teardown's own loot close, on a window looting the target (`0x493959`-`0x493974`).
        let loot_closed = looting.is_some()
            && world
                .get_resource::<crate::ui_loot::LootState>()
                .and_then(crate::ui_loot::LootState::source)
                .is_none();
        if loot_closed {
            let _ = crate::script_calls::run_cached(world, crate::ui_loot::tell_loot_closed);
        }
        // Then its signals, in place, after the loot close (`0x493a2a`, `0x493a73`).
        if let Some(select) = deselected {
            let _ =
                world.run_system_cached_with(crate::target::signal_ui_shutdown_deselect, select);
            crate::creature_anim::apply_unit_edits(world);
        }
    } else {
        let _ = crate::script_calls::run_cached(
            world,
            crate::target::clear_selection_on_world_teardown,
        );
    }
    if let Some(net) = world.get_resource::<crate::net::NetCommands>() {
        let _ = net.0.send(crate::net::ClientCommand::CancelTrade);
    }
}

/// The `/reload` guard's found arm (`0x490c0d`-`0x490c1b` → `0x60fb60(0, 1)`): a follow or a
/// Click to Move walk in flight is cancelled and the mover stopped, a follow telling the VM
/// `AUTOFOLLOW_END` (`0x60fb84`), so the new VM starts with none.
fn cancel_move_action(world: &mut World) {
    let followed =
        crate::script_calls::run_cached(world, crate::player::cancel_move_action_on_ui_shutdown)
            .unwrap_or(false);
    if followed {
        crate::script_calls::with_hosted_vm(world, |script| {
            if let Some(script) = script {
                script.fire_event("AUTOFOLLOW_END", vec![]);
            }
        });
    }
}

/// The UI shutdown in the reference's order, `0x490bd0`'s tail past the leave-world sweep (which
/// fires `PLAYER_LEAVING_WORLD`, [`super::run_leave_world_sweep`]): `PLAYER_LOGOUT` (271,
/// `0x490c2a`) → `layout-cache.txt` → the flat saved file (`0x490c7e`) → the per-addon files
/// (`0x490c83`) → `AddOns.txt` (`0x490c88`) → destroy the frame-script owner (`0x490c97`) → nil
/// all 216 C bindings out of `_G` (`0x490cba` → `0x490ce0`).
///
/// `PLAYER_LOGOUT` fires before any write, an addon's last chance to change a saved global, and on
/// every root, where the sweep runs only past the active-player guard (`0x490bee call 0x468550`,
/// `0x490bf5 je 0x490c25`) and the world latch. One function on every root because the steps are
/// ordered against each other; the layout save is here because a `/reload` never leaves
/// `InWorld`. No autosave, as the reference has none (the layout cache's crash-only debounce in
/// [`crate::ui_layout`] is ours).
pub(crate) fn shutdown_ui_state(script: &mut UiScript, identity: Option<&(String, String)>) {
    script.fire_event("PLAYER_LOGOUT", vec![]);
    crate::ui_layout::save_now(script, identity);
    crate::ui_saved::save(script);
    addons::save_addon_variables(script, identity);
    addons::save_enable_state(script, identity);
}

/// The session's end in `OnExit(InWorld)`, in its order: the shutdown with its `PLAYER_LOGOUT`
/// ([`shut_down_session`]), then the clears of what the dying state still shows (the chat
/// windows, the channels, the auras), then the glue VM ([`install_session_glue`]). In the
/// reference `0x490bd0` signals `PLAYER_LOGOUT` (`0x490c2a`) to the world's state while it is
/// whole, and that state is replaced only in `ShutdownGame` (`0x491231`).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SessionEnd {
    Shutdown,
    Clears,
    Glue,
}

/// The session's end at once, both halves with no clears between them, as a test runs it.
#[cfg(test)]
pub(crate) fn end_ui_session(world: &mut World) {
    shut_down_session(world);
    install_session_glue(world);
}

/// `OnExit(InWorld)`, a `/logout` or a disconnect (two of the reference's five roots), up to the
/// glue VM: the shutdown tail and the session's folds. [`install_session_glue`] then ends this
/// session's Lua state, so no frame, global or addon upvalue reaches the next login.
pub(crate) fn shut_down_session(world: &mut World) {
    close_ui_session(world, ShutdownRoot::WorldTeardown);
    // `ShutdownGame`'s clear of the world session's read-only flags (`0x491240`-`0x49125d`); a
    // `/reload` never comes here.
    if let Some(mut cvars) = world.get_resource_mut::<crate::cvars::Cvars>() {
        for name in benilla_ui::script::IN_WORLD_READ_ONLY_CVARS {
            cvars.set_read_only(name, false);
        }
    }
    // The session's end drops what the fold just carried, as the disconnect handler, a frame
    // earlier, dropped the rest: only a `/reload` seats it again.
    if let Some(mut log) = world.get_resource_mut::<crate::ui_quest_log::QuestLog>() {
        log.forget_folds();
    }
    if let Some(mut carry) = world.get_resource_mut::<crate::ui_tradeskill::TradeSkillRepeatCarry>()
    {
        carry.0 = 0;
    }
    if let Some(mut petition) = world.get_resource_mut::<crate::ui_petition::PetitionState>() {
        petition.forget_announced();
    }
}

/// The session's end's last step: the glue VM in place of the world's. The reference's state
/// outlives `0x490bd0` and is replaced twice through `0x703b80`, in `ShutdownGame` (at
/// `0x491231`) and then in the glue build (`0x46a7b0`), so the character screen runs on a state
/// of its own. Ours is a fresh glue VM ([`install_glue_side_vm`]).
pub(crate) fn install_session_glue(world: &mut World) {
    // `0x401ee0`'s glue leg: `"charselect"` after a logout, `"login"` after a disconnect
    // (`0x402084`); the state this exit enters says which.
    let screen = match world
        .get_resource::<State<crate::client_state::ClientState>>()
        .map(|s| *s.get())
    {
        Some(crate::client_state::ClientState::CharSelect) => super::glue::GlueScreen::CharSelect,
        _ => super::glue::GlueScreen::Login,
    };
    install_glue_side_vm(world, screen.key());
}

/// The shutdown a logout and a reload share, up to the new VM: the shutdown tail, the folds and
/// the host values reset.
fn close_ui_session(world: &mut World, root: ShutdownRoot) {
    // The front runs before the unpark, so a VM still parked (the load never ran) hears none of
    // it; the sweep's latch is spent at the root, so of this edge and the worldport's, the first
    // to a departure owns it.
    let silenced = run_shutdown_front(world, root);
    // A VM still parked goes back into the world first.
    unpark_glue_vm(world);
    // An entry load still armed means no in-game UI was built, and the tail would overwrite the
    // saved files with the glue VM's emptiness.
    let ui_never_loaded = world.remove_resource::<PendingEntryUiLoad>().is_some();
    let identity = world
        .get_resource::<AddOnIdentity>()
        .and_then(|id| id.0.clone());
    if !ui_never_loaded {
        crate::script_calls::with_hosted_vm(world, |script| {
            if let Some(script) = script {
                shutdown_ui_state(script, identity.as_ref());
            }
        });
    }
    if silenced {
        unsilence_ui_shutdown(world);
    }
    // After the shutdown events (a `PLAYER_LOGOUT` handler may `SetCVar`, which in the reference
    // lands in an engine-side store that survives) and before the VM is replaced.
    crate::cvars::fold_dying_vm_cvars(world);
    // The chat cache folds here, not in its own `OnExit(InWorld)` system, which a `/reload` skips.
    crate::ui_chat::settings::fold_dying_vm_chat_cache(world);
    // The item locks too: the reference keeps them on the item objects, which a `/reload` keeps.
    crate::ui_items::locks::fold_dying_vm_item_locks(world);
    // And the ready check, engine state the reference's reload never writes.
    crate::ui_party::fold_dying_vm_ready_check(world);
    // The guild roster's show-offline flag and selection, which only a login zeroes.
    crate::ui_guild::fold_dying_vm_guild(world);
    // So do the engine globals the VM holds for the session: the quest log's fold mask and the
    // tradeskill repeat, which no UI init resets.
    crate::ui_quest_log::fold_dying_vm_quest_folds(world);
    crate::ui_tradeskill::fold_dying_vm_repeat(world);
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
    // And the signals the frame's deaths parked in it, for the deaths lane.
    crate::target::drop_parked_deaths(world);
}

/// A `ReloadUI()` waiting to run: set when [`crate::ui_logout`] drains
/// [`benilla_ui::script::SessionRequest::ReloadUi`], run by [`run_pending_reload`] in the next
/// frame's `PreUpdate`. A flag, as in the reference (`[0xb4b3f4]`, written only by `0x491380` and
/// read only by the per-frame `0x495590`), so the VM that asked is never mid-call when destroyed.
#[derive(Resource, Default)]
pub(crate) struct ReloadUiPending(pub(crate) bool);

/// Run a pending `ReloadUI()`: the reference's teardown and rebuild (`0x495664 call 0x490bd0`,
/// `0x495669 call 0x48fbf0`), for us the session's shutdown then the load without leaving the
/// world, so a `DisableAddOn` staged in the dying VM reaches `AddOns.txt` before the rebuild
/// reads it.
/// The load runs under the live player, which never left. `UI_Init` then enters the world-enter
/// cascade itself (`0x490168`), firing the `PLAYER_LOGIN` it armed and `PLAYER_ENTERING_WORLD`;
/// ours fire from [`crate::ui_unit`]'s feed on the new VM's first `Update`, once it has re-seated
/// the player. In-world only, as the reference's gate (`0x494a50(0xa)`) is; elsewhere the request
/// is dropped, and the glue's own rebuild is [`super::glue::run_pending_glue_rebuild`]'s.
pub(crate) fn run_pending_reload(world: &mut World) {
    if !std::mem::take(&mut world.resource_mut::<ReloadUiPending>().0) {
        return;
    }
    let in_world = *world
        .resource::<State<crate::client_state::ClientState>>()
        .get()
        == crate::client_state::ClientState::InWorld;
    if !in_world {
        info!("ui_script: ReloadUI outside the world — dropped");
        return;
    }
    info!("ui_script: ReloadUI — ending the UI session and building a new one");
    // A boot VM, never the glue's: the reload builds no glue (`0x495664` then `0x495669`), and the
    // load replaces this VM at once.
    close_ui_session(world, ShutdownRoot::Reload);
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

/// `AppExit`, the quit root, read as a message because a quit from in-world never leaves
/// `InWorld`: the world's teardown ([`ShutdownRoot::WorldTeardown`]). A quit on a worldport's
/// loading screen finds the latch spent, so the sweep does not run; a quit from the character
/// screen runs nothing at all (`0x401ef3`).
pub(crate) fn shutdown_on_exit(
    world: &mut World,
    exits: &mut bevy::ecs::system::SystemState<MessageReader<AppExit>>,
) {
    if exits
        .get_mut(world)
        .expect("AppExit reader")
        .read()
        .next()
        .is_none()
    {
        return;
    }
    let in_world = world
        .get_resource::<State<crate::client_state::ClientState>>()
        .is_some_and(|s| *s.get() == crate::client_state::ClientState::InWorld);
    // Same guard as [`end_ui_session`]: a quit inside the entry-load window has no UI to save.
    if !in_world || world.contains_resource::<PendingEntryUiLoad>() {
        return;
    }
    let silenced = run_shutdown_front(world, ShutdownRoot::WorldTeardown);
    let identity = world
        .get_resource::<AddOnIdentity>()
        .and_then(|id| id.0.clone());
    crate::script_calls::with_hosted_vm(world, |script| {
        if let Some(script) = script {
            shutdown_ui_state(script, identity.as_ref());
        }
    });
    if silenced {
        unsilence_ui_shutdown(world);
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

/// [`finish_ui_load`] with its two host seams, in the order the client's load runs them
/// ([`load_ingame_ui_on_world_entry`] runs the same steps around its world reads).
/// `host_settings` runs between the saved-variables file and `VARIABLES_LOADED`, carrying over a
/// setting `config.toml` held before it moved into the saved variables. `between` runs after
/// `VARIABLES_LOADED` and before the `PLAYER_LOGIN` arm (so before the event too): the chat-cache
/// restore's `UPDATE_CHAT_WINDOWS` + `UPDATE_CHAT_COLOR` burst, where the reference registers its
/// reader (`0x4900d6` → `0x498a20`), then the UI scale (`0x49010e`). On a fresh login the
/// reference's cache (`0x5afe50`) defers that reader until the server's account data reconciles
/// (`0x5afa42`/`0x5afd01`), after `UI_Init` has returned. Either way a `VARIABLES_LOADED` handler
/// sees the boot chat colours, not the file's.
#[cfg(test)]
pub(crate) fn finish_ui_load_with(
    script: &mut UiScript,
    host_settings: impl FnOnce(&mut UiScript),
    between: impl FnOnce(&mut UiScript),
) {
    read_saved_variables(script, host_settings);
    between(script);
    script.arm_player_login();
}

/// The load's saved-variables step: the flat file, `host_settings`, then `VARIABLES_LOADED`
/// (`0x4900b2` → `0x4913b0`). Still the load edge, so still bounded: re-armed so the saved
/// variables and every `VARIABLES_LOADED` handler get the full allowance; the entry edge disarms
/// it after. The world-enter events fire later, unbounded, as every event the feeds fire is.
fn read_saved_variables(script: &mut UiScript, host_settings: impl FnOnce(&mut UiScript)) {
    script.set_instruction_budget(addons::LOAD_INSTRUCTION_BUDGET);
    crate::ui_saved::load_saved_variables(script, host_settings);
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

    /// The logout's `OnExit(InWorld)`: `0x490bd0` signals `PLAYER_LOGOUT` (`0x490c2a`) while the
    /// world's state is whole, so its handler reads the auras and the chat windows as they stood;
    /// the session's clears then run on that same state, before the glue VM replaces it.
    #[test]
    fn the_logout_clears_run_after_player_logout_and_on_the_world_vm() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let mut app = crate::game_plugins::schedule_tests::headless_client_unrun();
        // The whole schedule, with no world loaded: a system missing what world entry inserts
        // (the camera rig, the map) skips rather than panics.
        app.insert_resource(bevy::ecs::error::FallbackErrorHandler(
            bevy::ecs::error::ignore,
        ));
        let notes: Rc<RefCell<Vec<String>>> = Rc::default();
        let mut script = UiScript::new().expect("a VM");
        script.set_tracking(Some(benilla_ui::script::TrackingState {
            spell_id: 2580,
            name: None,
            icon: Some("TRACK".into()),
            cancelable: false,
        }));
        let sink = notes.clone();
        let note = script
            .lua()
            .create_function(move |_, text: String| {
                sink.borrow_mut().push(text);
                Ok(())
            })
            .expect("the note");
        script.lua().globals().set("Note", note).expect("Note");
        script
            .run(
                "for _, name in ipairs({'ChatFrame1', 'ChatFrame2'}) do \
                     setglobal(name, { Clear = function() Note('clear') end }) \
                 end \
                 local f = CreateFrame('Frame') f:RegisterEvent('PLAYER_LOGOUT') \
                 f:SetScript('OnEvent', function() \
                     Note('logout ' .. tostring(GetTrackingTexture())) \
                 end)",
            )
            .expect("the recorder installs");
        app.insert_non_send(script);
        app.world_mut()
            .run_schedule(OnExit(crate::client_state::ClientState::InWorld));
        assert_eq!(*notes.borrow(), ["logout TRACK", "clear", "clear"]);
    }

    /// A cinematic in flight at the logout: the shutdown's sweep drops it and tells the VM
    /// `CINEMATIC_STOP` (`0x490b69`-`0x490b77`) ahead of `PLAYER_LOGOUT` (`0x490c2a`), so the
    /// cinematic's own drop at the session's end runs after it and cannot swallow the event.
    #[test]
    fn a_cinematic_in_flight_at_the_logout_stops_in_the_shutdowns_sweep() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let mut app = crate::game_plugins::schedule_tests::headless_client_unrun();
        app.insert_resource(bevy::ecs::error::FallbackErrorHandler(
            bevy::ecs::error::ignore,
        ));
        app.insert_resource(crate::cinematic::Cinematic::playing_for_test());
        app.world_mut()
            .resource_mut::<super::super::LeavingWorldArmed>()
            .arm();
        let notes: Rc<RefCell<Vec<String>>> = Rc::default();
        let script = UiScript::new().expect("a VM");
        let sink = notes.clone();
        let note = script
            .lua()
            .create_function(move |_, text: String| {
                sink.borrow_mut().push(text);
                Ok(())
            })
            .expect("the note");
        script.lua().globals().set("Note", note).expect("Note");
        script
            .run(
                "local f = CreateFrame('Frame') f:RegisterEvent('CINEMATIC_STOP') \
                 f:RegisterEvent('PLAYER_LEAVING_WORLD') f:RegisterEvent('PLAYER_LOGOUT') \
                 f:SetScript('OnEvent', function() Note(event) end)",
            )
            .expect("the recorder installs");
        app.insert_non_send(script);
        app.world_mut()
            .run_schedule(OnExit(crate::client_state::ClientState::InWorld));
        assert_eq!(
            *notes.borrow(),
            ["PLAYER_LEAVING_WORLD", "CINEMATIC_STOP", "PLAYER_LOGOUT"]
        );
    }

    /// Every system of `OnExit(InWorld)` that fires into the dying interface or clears what it
    /// reads runs in the session's end order ([`SessionEnd`]); the rest touch no Lua-visible
    /// state (the sound's beds, kits and reverb, the camera's saved pose).
    #[test]
    fn every_session_end_system_that_reaches_the_lua_is_ordered() {
        use crate::game_plugins::schedule_tests::{census, type_names_available, SyncPoints};
        const NO_LUA: &[&str] = &[
            "::sound::kit::stop_all_channels",
            "::sound::reverb::leave_world",
            "::sound::interior::leave_world",
            "::sound::zone::leave_world",
            "::camera_saved::save_on_session_end",
            // The pass's end: the hosts' unit edits applied after them (`crate::vm_pass`).
            "::creature_anim::edits::flush_unit_edits",
        ];
        let mut app = crate::game_plugins::schedule_tests::headless_client();
        let exit = census(
            &mut app,
            OnExit(crate::client_state::ClientState::InWorld),
            SyncPoints::Declared,
        );
        if !type_names_available(&exit) {
            return;
        }
        let mut loose: Vec<&str> = exit
            .systems
            .values()
            .filter(|s| {
                !s.sets
                    .iter()
                    .any(|set| ["Shutdown", "Clears", "Glue"].contains(&set.as_str()))
            })
            .map(|s| s.name.as_str())
            .filter(|n| !NO_LUA.iter().any(|ok| n.ends_with(ok)))
            .collect();
        loose.sort();
        assert!(
            loose.is_empty(),
            "unordered against the session's end: {loose:?}"
        );
    }

    /// The roster's stand-in `"player"` and the record `UnitRace`/`UnitClass` read carry the
    /// table's names at the install's locale, and the tokens beside them.
    #[test]
    fn the_roster_seat_and_record_name_race_and_class_from_the_tables() {
        let mut c = crate::glue_session::roster::test_character(7, "Harness");
        c.race = 4; // Night Elf
        c.class = 5; // Priest
        let roster = crate::glue_session::roster::Roster::with_pending_pick(vec![c], 7);
        let korean = crate::ui_unit::TestNames::korean();
        let seat = seat_from_roster(&roster, korean.names()).expect("a pick seats");
        assert_eq!(seat.race.as_deref(), Some("나이트 엘프"));
        assert_eq!(seat.race_file.as_deref(), Some("NightElf"));
        assert_eq!(seat.class.as_deref(), Some("사제"));
        assert_eq!(seat.class_file.as_deref(), Some("PRIEST"));
        let record = record_from_roster(&roster, korean.names()).expect("a pick records");
        assert_eq!(
            record.race,
            Some(("나이트 엘프".to_string(), "NightElf".to_string()))
        );
        assert_eq!(
            record.class,
            Some(("사제".to_string(), "PRIEST".to_string()))
        );
    }

    /// A capture's VM draws the CRT's default sequence, `srand(1)`'s 41, 18467, ..., so its
    /// frames reproduce; a session's is seeded from the clock.
    #[test]
    fn a_capture_vm_draws_the_default_random_sequence() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _lock = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let draws = |capture: bool| {
            let _cap = if capture {
                EnvGuard::set("WOW_CAPTURE", "some-scenario")
            } else {
                EnvGuard::unset("WOW_CAPTURE")
            };
            let mut world = World::new();
            let script = new_vm(&mut world, UiScript::new).expect("a VM");
            script
                .eval::<Vec<i64>>(
                    "local r = math.random return {r(32767), r(32767), r(32767), r(32767)}",
                )
                .unwrap()
        };
        assert_eq!(draws(true), [42, 18468, 6335, 26501]);
        assert_ne!(draws(false), [42, 18468, 6335, 26501]);
    }

    /// Every VM, glue and in-game, answers `GetLocale` with the install's locale (`[0xc0e080]`).
    #[test]
    fn every_new_vm_is_seated_with_the_installs_locale() {
        let mut world = World::new();
        let want = super::super::client_locale();
        let in_game = new_vm(&mut world, UiScript::new).expect("a VM");
        let glue = new_vm(&mut world, UiScript::new_glue).expect("a glue VM");
        for script in [in_game, glue] {
            assert_eq!(script.locale(), want.index());
            let answer = script.eval::<String>("return GetLocale()").unwrap();
            assert_eq!(answer, want.name());
            assert_eq!(script.client_region(), super::super::client_region());
        }
    }

    /// The word lists are client globals start-up compiles once (`0x402b7f` → `0x6c91a0`), so a
    /// world VM's guild name check (`0x6c9b70`) searches them as the glue VM's name checks do: a
    /// NamesProfanity or ChatProfanity hit is refused as reserved (`0x6c9a51`, `0x4f5160`), and
    /// NamesReserved is not searched (`0x6c9b76`).
    #[test]
    fn a_world_vm_checks_a_guild_name_against_the_clients_word_lists() {
        use benilla_ui::script::PetitionCall;
        let list = |p: &str| {
            vec![benilla_formats::FilterPattern {
                id: 1,
                pattern: p.into(),
            }]
        };
        let mut world = World::new();
        world.insert_resource(
            crate::text_filter::TextFilter::new(&list("heck"), &[])
                .name_filter(&list("darn"), &list("thrall")),
        );
        let mut script = new_vm(&mut world, UiScript::new).expect("a VM");
        script.keep_calls();
        let refused = PetitionCall::NameRefused("ERR_GUILD_NAME_RESERVED");
        for (name, want) in [
            ("Darnwood", refused.clone()),
            ("Heckwood", refused),
            ("Thrallwood", PetitionCall::Buy("Thrallwood".into())),
        ] {
            script.run(&format!("BuyGuildCharter({name:?})")).unwrap();
            assert_eq!(script.take_petition_calls(), [want], "{name:?}");
        }
    }

    /// The game clock is one object built at start-up with every field -1 (`0x643300`, `0x642300`)
    /// that no logout or `ReloadUI` resets, and `GetGameTime` reads it unguarded (`0x515ee0`), so a
    /// VM answers -1, -1 before any server time and the last known time after it, from the first
    /// line its addons run.
    #[test]
    fn a_new_vm_answers_the_game_clock_as_it_stands() {
        let mut world = World::new();
        let answer = |world: &mut World| {
            new_vm(world, UiScript::new)
                .expect("a VM")
                .eval::<(f64, f64)>("return GetGameTime()")
                .unwrap()
        };
        assert_eq!(answer(&mut world), (-1.0, -1.0), "no clock resource");
        world.init_resource::<crate::net::ServerTime>();
        assert_eq!(answer(&mut world), (-1.0, -1.0), "no server time yet");
        let mut clock = crate::net::GameTime::new(0);
        clock.set(
            benilla_protocol::WowTime {
                minute: 45,
                hour: 13,
                ..benilla_protocol::WowTime::UNSET
            },
            0,
        );
        world.insert_resource(crate::net::ServerTime(clock));
        assert_eq!(answer(&mut world), (13.0, 45.0));
        let glue = new_vm(&mut world, UiScript::new_glue).expect("a glue VM");
        assert_eq!(
            glue.eval::<(f64, f64)>("return GetGameTime()").unwrap(),
            (13.0, 45.0)
        );
    }

    /// `[0xcf4d24]` has one writer, `SetEuropeanNumbers` (`0x771c40`), and nothing resets it, so
    /// the VM that replaces a dying one (a `ReloadUI`, a logout, a login) opens with its value.
    #[test]
    fn the_european_numbers_switch_outlives_the_vm_that_set_it() {
        let mut world = World::new();
        let first = UiScript::new().unwrap();
        assert!(!first.european_numbers(), "off at boot");
        world.insert_non_send(first);
        let text = |world: &World| {
            let script = world.non_send::<UiScript>();
            script
                .run(
                    "F = CreateFrame('Frame'):CreateFontString() \
                     F:SetFont('Fonts\\\\FRIZQT__.TTF', 12) F:SetText('1.5')",
                )
                .unwrap();
            script.eval::<String>("return F:GetText()").unwrap()
        };
        world
            .non_send::<UiScript>()
            .run("SetEuropeanNumbers(1)")
            .unwrap();
        for _ in 0..2 {
            let next = new_vm(&mut world, UiScript::new).expect("a VM");
            world.insert_non_send(next);
            assert_eq!(text(&world), "1,5");
        }
        world
            .non_send::<UiScript>()
            .run("SetEuropeanNumbers(false)")
            .unwrap();
        let next = new_vm(&mut world, UiScript::new).expect("a VM");
        world.insert_non_send(next);
        assert_eq!(text(&world), "1.5");
    }

    #[test]
    fn the_entry_edge_parks_the_glue_vm_and_the_session_end_returns_it() {
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
        assert!(world.get_non_send::<ParkedGlueVm>().is_some());
        // Left the world before the load ran: the VM comes back, untouched.
        unpark_glue_vm(&mut world);
        let vm = world
            .get_non_send::<UiScript>()
            .expect("the parked VM is back");
        assert_eq!(vm.session(), session, "the same VM, no session moved");
        assert!(world.get_non_send::<ParkedGlueVm>().is_none());
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
            world.insert_resource(State::new(crate::client_state::ClientState::InWorld));
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

    /// `InitializeGame` flags the world session's read-only rows on the registry before the UI gate
    /// (`0x48f566`-`0x48f584`), so a capture world that builds no UI has them too; a `/reload`
    /// keeps them, and the world's end clears them (`0x491240`).
    #[test]
    fn the_world_entry_flags_the_registry_read_only_with_or_without_a_ui() {
        let mut world = World::new();
        world.insert_resource(State::new(crate::client_state::ClientState::InWorld));
        world.insert_resource(crate::run_mode::CaptureMode);
        world.init_resource::<crate::cvars::Cvars>();
        world.insert_resource(PendingEntryUiLoad::default());
        run_pending_entry_load(&mut world);
        let set = |world: &mut World, name: &str| {
            world
                .resource_mut::<crate::cvars::Cvars>()
                .set(name, "x.example.org")
        };
        use crate::cvars::SetOutcome;
        for name in benilla_ui::script::IN_WORLD_READ_ONLY_CVARS {
            assert_eq!(set(&mut world, name), SetOutcome::ReadOnly, "{name}");
        }
        world.insert_resource(ReloadUiPending(true));
        run_pending_reload(&mut world);
        assert_eq!(set(&mut world, "realmName"), SetOutcome::ReadOnly);
        end_ui_session(&mut world);
        assert_eq!(set(&mut world, "realmName"), SetOutcome::Changed);
    }

    /// A `ReloadUI()` runs the leave-world sweep (`0x490c20`, past the armed latch) and then
    /// re-enters the world-enter cascade (`0x490168`), whose readers re-send the time, mail,
    /// battlefield and meeting-stone queries, only with the active player set (`0x490166 je`).
    #[test]
    fn a_reload_sweeps_and_reenters_the_world_enter_cascade_once_the_player_exists() {
        use crate::net::WorldEnterCascadeMessage;
        let reload = |seated: bool| {
            let mut world = World::new();
            world.insert_resource(State::new(crate::client_state::ClientState::InWorld));
            world.insert_resource(crate::run_mode::CaptureMode);
            world.init_resource::<Messages<WorldEnterCascadeMessage>>();
            world.init_resource::<super::super::SweptWorlds>();
            world.init_resource::<super::super::LeavingWorldArmed>();
            world
                .resource_mut::<super::super::LeavingWorldArmed>()
                .arm();
            if seated {
                world.spawn(crate::net::SelfPlayer);
            }
            world.insert_resource(ReloadUiPending(true));
            run_pending_reload(&mut world);
            assert!(!world.resource::<ReloadUiPending>().0, "the reload ran");
            let sweeps = world.resource::<super::super::SweptWorlds>().0;
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

    /// The agreements are read from their CVars before the first screen is set (`0x46a400` at
    /// `0x402b36`, `SET_GLUE_SCREEN` at `0x402b65`): a returning player's login shows the boxes,
    /// not the EULA. In this fork a first launch goes there too: the TOS and EULA stand accepted.
    #[test]
    fn the_login_screen_comes_up_on_the_accepted_agreements() {
        use crate::client_state::ClientState;
        let _ = benilla_formats::wow_data_or_skip!();
        let screens = |cvars: crate::cvars::Cvars| {
            let mut world = World::new();
            world.insert_resource(State::new(ClientState::Login));
            world.insert_resource(cvars);
            world.init_non_send::<super::super::glue_mgr::SharedGlueMgr>();
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
            (false, true, true),
            "this fork asks no first launch for the EULA"
        );
    }

    /// `0x46a400` reads the four `read*` options into the flag cells once, at `0x402b36`; after
    /// that only the accepts store them (`0x46aef0` and its siblings, a census of the eight
    /// cells). A glue rebuild (`0x46a7b0`) seats the same cells: a `read*` option changed by a
    /// console write is not read again, and an accept outlives the option it wrote.
    #[test]
    fn the_agreement_flags_are_seeded_once_and_a_rebuild_keeps_them() {
        use crate::client_state::ClientState;
        use benilla_ui::script::{GlueNotice, NoticeState};
        let accepted = NoticeState {
            accepted: true,
            show_notice: false,
        };
        let flags = |world: &World| {
            let script = world.non_send::<UiScript>();
            GlueNotice::ALL.map(|n| script.glue_notice(n))
        };
        // This fork stands every agreement accepted (`glue_mgr::read_agreements`), so the read is
        // seen through `show_notice`, which a negative option sets.
        let mut world = World::new();
        world.insert_resource(State::new(ClientState::Login));
        world.insert_resource(crate::cvars::Cvars::with_value("readTOS", "-1"));
        world.init_non_send::<super::super::glue_mgr::SharedGlueMgr>();
        setup_script(&mut world);
        let seeded = [
            NoticeState {
                accepted: true,
                show_notice: true,
            },
            accepted,
            accepted,
            accepted,
        ];
        assert_eq!(flags(&world), seeded, "the startup read");

        // The options move with no accept; the next glue build does not read them.
        {
            let mut cvars = world.resource_mut::<crate::cvars::Cvars>();
            cvars.set("readTOS", "0");
            cvars.set("readEULA", "-1");
        }
        install_glue_side_vm(&mut world, "login");
        assert_eq!(flags(&world), seeded, "a rebuild keeps the cells");

        // An accept stores its cell, and a rebuild after the option was reset keeps that too.
        world
            .non_send::<UiScript>()
            .run("AcceptScanning()")
            .unwrap();
        world
            .resource_mut::<crate::cvars::Cvars>()
            .set("readScanning", "0");
        install_glue_side_vm(&mut world, "login");
        let mut want = seeded;
        want[GlueNotice::Scanning as usize] = accepted;
        assert_eq!(
            flags(&world),
            want,
            "an accept outlives the option it wrote"
        );
    }

    /// The world holds the glue VM outside the world, at startup on the login screen
    /// (`0x402b65`) and back from the world on the screen the exit enters (`0x402090`); a
    /// `ReloadUI()` never builds the glue.
    #[test]
    fn the_glue_stands_outside_the_world() {
        use crate::client_state::ClientState;
        use benilla_ui::script::VmKind;
        let shown = |world: &World, frame: &str| {
            benilla_formats::wow_data().is_none()
                || world
                    .non_send::<UiScript>()
                    .eval::<bool>(&format!("return {frame}:IsShown() == 1"))
                    .unwrap()
        };
        let mut world = World::new();
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

    /// Only the in-game state registers the macro verbs (`RegisterMacroFunctions` `0x4f1b90`, from
    /// `0x490297`), so the world-entry VM gets the icon chooser's source and the glue VM and a
    /// reload's boot VM get none.
    #[test]
    fn only_the_entry_vm_gets_the_macro_icon_source() {
        use crate::client_state::ClientState;
        let has = |world: &World| world.non_send::<UiScript>().has_macro_icon_source();
        let mut world = World::new();
        world.insert_resource(State::new(ClientState::Login));
        setup_script(&mut world);
        assert!(!has(&world), "the glue VM");
        install_boot_vm(&mut world);
        assert!(!has(&world), "a boot VM");
        mint_entry_vm(&mut world);
        assert!(has(&world), "the world-entry VM");
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
        // The process has been up 90 s on the dying VM, and this frame's step, which the new VM's
        // first tick adds, takes the frame step's clock 16 ms past that.
        let startup = Instant::now() - Duration::from_secs(90);
        world.insert_resource(FrameStep {
            delta: Duration::from_millis(16),
            at: startup + Duration::from_millis(90_016),
            elapsed: Duration::from_millis(90_016),
        });
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

    const CORPSE: u64 = 0xF130_0000_0000_0042;
    const LEADER: u64 = 0x0000_0000_0000_0009;
    const FRONT_EVENTS: [&str; 4] = [
        "LOOT_CLOSED",
        "PLAYER_TARGET_CHANGED",
        "AUTOFOLLOW_END",
        "PLAYER_LEAVING_WORLD",
    ];

    /// A world in play for the shutdown's front: the corpse targeted and looted, a follow on,
    /// the latch armed, our own player present when `seated` (charmed when `charmed`), and the
    /// sweep with only its own event.
    fn front_world(
        seated: bool,
        charmed: bool,
    ) -> (
        World,
        crossbeam_channel::Receiver<crate::net::ClientCommand>,
    ) {
        front_world_in(World::new(), seated, charmed)
    }

    /// [`front_world`] on the whole client's world, headless, whose host the head's deselect
    /// lends its signals.
    fn hosted_front_world(
        seated: bool,
        charmed: bool,
    ) -> (
        World,
        crossbeam_channel::Receiver<crate::net::ClientCommand>,
    ) {
        let mut app = crate::game_plugins::schedule_tests::headless_client_unrun();
        front_world_in(std::mem::take(app.world_mut()), seated, charmed)
    }

    fn front_world_in(
        mut world: World,
        seated: bool,
        charmed: bool,
    ) -> (
        World,
        crossbeam_channel::Receiver<crate::net::ClientCommand>,
    ) {
        use benilla_protocol::messages::ObjectFields;
        let (tx, rx) = crossbeam_channel::unbounded();
        world.insert_resource(crate::net::NetCommands(tx));
        world.init_resource::<crate::target::Selection>();
        world.init_resource::<crate::spell::QueuedMeleeSpell>();
        world.init_resource::<crate::spell::PendingCast>();
        world.init_resource::<crate::spell::SpellTargeting>();
        world.init_resource::<crate::ui_cast::CastReports>();
        world.init_resource::<crate::spell::AutoRepeatActive>();
        world.init_resource::<crate::creature_anim::AttackStopInFlight>();
        world.init_resource::<crate::creature_anim::UnitEdits>();
        world.init_resource::<bevy::ecs::message::Messages<crate::player::StandStateRequest>>();
        world.init_resource::<Messages<crate::creature_anim::SheathRequest>>();
        world.init_resource::<crate::ui_loot::LootLatch>();
        world.init_resource::<crate::net::GuidIndex>();
        world.init_resource::<crate::player::Approach>();
        world.init_resource::<crate::player::FollowState>();
        world.resource_mut::<crate::player::FollowState>().guid = Some(LEADER);
        super::super::sweep::install_world_leave_sweep(&mut world, |_| {});
        world
            .resource_mut::<super::super::LeavingWorldArmed>()
            .arm();
        let mut loot = crate::ui_loot::LootState::default();
        loot.open(CORPSE, 1, 0, Vec::new());
        world.insert_resource(loot);
        let body = world.spawn_empty().id();
        {
            let mut selection = world.resource_mut::<crate::target::Selection>();
            selection.target = Some(body);
            selection.guid = Some(CORPSE);
        }
        if seated {
            // `UNIT_FIELD_CHARMEDBY` (10, 11).
            let fields = if charmed {
                ObjectFields::from_pairs(&[(10, 0x77), (11, 0)])
            } else {
                ObjectFields::from_pairs(&[])
            };
            world.spawn((crate::net::SelfPlayer, crate::net::ObjectStore(fields)));
        }
        world.insert_non_send(crate::ui_script::sweep_test::spied_vm(&FRONT_EVENTS));
        (world, rx)
    }

    fn front_seen(world: &World) -> Vec<String> {
        crate::ui_script::sweep_test::seen(world.non_send::<UiScript>())
    }

    /// `/reload` (`0x495664` → `0x490bd0`): the head with the object manager up drops the target
    /// loudly, its teardown closing the loot on it (`0x493959`-`0x493974`) and sending
    /// `CMSG_SET_SELECTION` 0 and `PLAYER_TARGET_CHANGED` (`0x493a2a`, `0x493a73`), then
    /// `CMSG_CANCEL_TRADE` (`0x490be4`); past the guard the found arm ends the follow
    /// (`0x60fb84`), then the sweep (`0x490c20`).
    #[test]
    fn a_reload_drops_the_target_loudly_ends_the_follow_then_sweeps() {
        use crate::net::ClientCommand;
        let (mut world, rx) = hosted_front_world(true, false);
        assert!(run_shutdown_front(&mut world, ShutdownRoot::Reload));
        assert_eq!(
            front_seen(&world),
            [
                "LOOT_CLOSED",
                "PLAYER_TARGET_CHANGED",
                "AUTOFOLLOW_END",
                "PLAYER_LEAVING_WORLD"
            ]
        );
        let sent: Vec<ClientCommand> = rx.try_iter().collect();
        assert!(
            matches!(
                sent.as_slice(),
                [
                    ClientCommand::LootRelease { guid: CORPSE },
                    ClientCommand::SetSelection { guid: 0 },
                    ClientCommand::CancelTrade,
                ]
            ),
            "{sent:?}"
        );
        assert_eq!(world.resource::<crate::target::Selection>().guid, None);
        assert_eq!(world.resource::<crate::player::FollowState>().guid, None);
    }

    /// The `/reload` head's deselect signals in place: its teardown's StopAttack (`0x493a08`)
    /// cancels a queued strike (`0x5ecb63` → `0x6e6f30`), whose stop reports after the loot close
    /// (`0x493959`) and ahead of `PLAYER_TARGET_CHANGED` (`0x493a73`).
    #[test]
    fn a_reload_reports_its_strike_cancel_before_the_target_change() {
        let (mut world, _rx) = hosted_front_world(true, false);
        world.insert_non_send(crate::ui_script::sweep_test::spied_vm(&[
            "LOOT_CLOSED",
            "SPELLCAST_STOP",
            "PLAYER_TARGET_CHANGED",
        ]));
        let me = world
            .query_filtered::<Entity, With<crate::net::SelfPlayer>>()
            .single(&world)
            .unwrap();
        world
            .entity_mut(me)
            .insert(crate::creature_anim::Engaged(CORPSE));
        world
            .resource_mut::<crate::spell::QueuedMeleeSpell>()
            .arm(78);
        world
            .resource_mut::<crate::spell::PendingCast>()
            .fill_record();
        run_shutdown_front(&mut world, ShutdownRoot::Reload);
        assert_eq!(
            front_seen(&world),
            ["LOOT_CLOSED", "SPELLCAST_STOP", "PLAYER_TARGET_CHANGED"]
        );
    }

    /// A charmed player's `SetSelection` is a no-op (`0x493589`-`0x49359c`): a `/reload` keeps
    /// the target and sends no `CMSG_SET_SELECTION`, and the trade cancel still goes.
    #[test]
    fn a_charmed_reload_keeps_the_target() {
        use crate::net::ClientCommand;
        let (mut world, rx) = front_world(true, true);
        run_shutdown_front(&mut world, ShutdownRoot::Reload);
        assert!(!front_seen(&world).contains(&"PLAYER_TARGET_CHANGED".to_string()));
        assert!(rx
            .try_iter()
            .all(|c| !matches!(c, ClientCommand::SetSelection { .. })));
        assert_eq!(
            world.resource::<crate::target::Selection>().guid,
            Some(CORPSE)
        );
    }

    /// A logout, a disconnect or an exit in the world (`0x401ee0`): the destructor's sweep first
    /// (`0x402011` → `0x5dd543`), then the head with no object manager, which clears the
    /// selection silently (`0x4939df`) and still cancels the trade; no follow cancel.
    #[test]
    fn a_world_teardown_sweeps_first_then_clears_the_target_silently() {
        use crate::net::ClientCommand;
        let (mut world, rx) = front_world(true, false);
        world
            .resource_mut::<crate::ui_loot::LootState>()
            .clear_session();
        assert!(run_shutdown_front(&mut world, ShutdownRoot::WorldTeardown));
        assert_eq!(front_seen(&world), ["PLAYER_LEAVING_WORLD"]);
        let sent: Vec<ClientCommand> = rx.try_iter().collect();
        assert!(
            matches!(sent.as_slice(), [ClientCommand::CancelTrade]),
            "{sent:?}"
        );
        let selection = world.resource::<crate::target::Selection>();
        assert_eq!((selection.guid, selection.last), (None, Some(CORPSE)));
    }

    /// A refused login has no player of ours to destroy, so no sweep: no create of ours armed the
    /// world latch; the head still cancels.
    #[test]
    fn a_refused_login_sweeps_nothing_and_cancels_the_trade() {
        use crate::net::ClientCommand;
        let (mut world, rx) = front_world(false, false);
        world.insert_resource(super::super::LeavingWorldArmed::default());
        run_shutdown_front(&mut world, ShutdownRoot::WorldTeardown);
        assert!(front_seen(&world).is_empty());
        let sent: Vec<ClientCommand> = rx.try_iter().collect();
        assert!(
            matches!(sent.as_slice(), [ClientCommand::CancelTrade]),
            "{sent:?}"
        );
    }

    /// A quit from the character screen runs nothing (`0x401ef3`): no head, no cancel, no
    /// `PLAYER_LOGOUT`.
    #[test]
    fn a_quit_at_the_character_screen_runs_nothing() {
        use bevy::ecs::system::RunSystemOnce;
        let (mut world, rx) = front_world(false, false);
        world.insert_resource(State::new(crate::client_state::ClientState::CharSelect));
        world.init_resource::<AddOnIdentity>();
        world.init_resource::<Messages<AppExit>>();
        world
            .non_send::<UiScript>()
            .run(
                "SEEN_LOGOUT = 0 local f = CreateFrame('Frame') f:RegisterEvent('PLAYER_LOGOUT') \
                 f:SetScript('OnEvent', function() SEEN_LOGOUT = SEEN_LOGOUT + 1 end)",
            )
            .unwrap();
        world.write_message(AppExit::Success);
        world.run_system_once(shutdown_on_exit).unwrap();
        assert_eq!(rx.try_iter().count(), 0);
        assert_eq!(
            world
                .non_send::<UiScript>()
                .eval::<i64>("return SEEN_LOGOUT")
                .unwrap(),
            0
        );
    }

    /// A one-shot the shutdown runs from its exclusive edge applies the unit edits it queued as it
    /// returns, as it applies its commands. The `/reload` head's loot close fires `LOOT_CLOSED`,
    /// whose handler stops a running auto-repeat (`SpellStopCasting`, `0x6e6e80` → `0x6ea080`):
    /// the arm is off our player before the head's next step.
    #[test]
    fn a_shutdown_one_shot_applies_its_unit_edits_as_it_returns() {
        const ME: u64 = 0x10;
        let mut app = crate::game_plugins::schedule_tests::headless_client_unrun();
        let (tx, _rx) = crossbeam_channel::unbounded();
        app.insert_resource(crate::net::NetCommands(tx));
        let world = app.world_mut();
        let me = world
            .spawn((
                crate::net::SelfPlayer,
                crate::net::Guid(ME),
                crate::creature_anim::AutoRepeatArmed,
            ))
            .id();
        world
            .resource_mut::<crate::net::GuidIndex>()
            .0
            .insert(ME, me);
        world.resource_mut::<crate::net::SelfGuid>().0 = Some(ME);
        // Auto Shot.
        world.resource_mut::<crate::spell::AutoRepeatActive>().0 = Some(75);
        let mut script = UiScript::new().expect("a VM");
        // The view the cast state pushed after the call that armed it.
        script.set_cast_view(benilla_ui::script::CastView {
            auto_repeat: Some(75),
            ..Default::default()
        });
        script
            .run(
                "local f = CreateFrame('Frame') f:RegisterEvent('LOOT_CLOSED') \
                 f:SetScript('OnEvent', function() SpellStopCasting() end)",
            )
            .unwrap();
        world.insert_non_send(script);
        crate::script_calls::run_cached(world, crate::ui_loot::tell_loot_closed)
            .expect("the loot close runs");
        assert_eq!(
            world.resource::<crate::spell::AutoRepeatActive>().0,
            None,
            "the handler's call applied"
        );
        assert!(
            !world
                .entity(me)
                .contains::<crate::creature_anim::AutoRepeatArmed>(),
            "and its edit, before the one-shot returned"
        );
    }
}
