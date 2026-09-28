//! The in-game interface: the core manifest, [`MANIFEST`], loaded in two phases, then benilla's
//! layer, [`LAYER_MANIFEST`]. A core entry with a path is the reference's own file off the
//! player's patch chain, a bare filename one we ship, and the manifest's order is the load order
//! across both; every layer entry is ours and loads after the whole core.

use bevy::prelude::*;

use benilla_ui::script::{UiFlavor, UiScript};
use benilla_ui::toc::Toc;

use super::addons::Addon;
use super::reference_ui;

/// The core interface's manifest, relative to `assets/ui`.
pub(super) const MANIFEST: &str = "benilla.toc";

/// benilla's own interface, one layer on the stock one, relative to `assets/ui`.
pub(super) const LAYER_MANIFEST: &str = "layer.toc";

/// Whether this run loads the layer: always in a player build; a dev build boots the stock UI
/// alone under `WOW_STOCK_UI=1`.
pub(super) fn layer_enabled() -> bool {
    layer_enabled_by(
        crate::run_mode::dev_affordances(),
        std::env::var("WOW_STOCK_UI").ok().as_deref(),
    )
}

/// [`layer_enabled`] over its two facts: a dev build, and the switch's value.
pub(super) fn layer_enabled_by(dev_build: bool, stock_ui: Option<&str>) -> bool {
    !(dev_build && stock_ui == Some("1"))
}

/// `/errors`: toggles the layer's script error log; a no-op when the run boots without the layer.
pub(crate) const ERRORS_TOGGLE: &str =
    "if BenillaScriptLog_Toggle then BenillaScriptLog_Toggle() end";

/// `/errors clear`: empties the layer's script error log, or does nothing without the layer.
pub(crate) const ERRORS_CLEAR: &str = "if BenillaScriptLog_Clear then BenillaScriptLog_Clear() end";

/// Hands a chord the layer's Keybindings page is capturing (`BenillaBindCapture`), a bind string
/// such as `CTRL-J`, to its handler; a no-op without the layer.
pub(crate) fn host_key_capture(chord: &str) -> String {
    format!("if KeyBindings_OnHostKey then KeyBindings_OnHostKey(\"{chord}\") end")
}

/// Every entry in load order: the core's, then the layer's.
#[cfg(test)]
pub(super) fn manifest_files() -> Vec<String> {
    let mut files = Addon::core().toc.files;
    files.extend(Addon::layer().toc.files);
    files
}

/// The manifest's entries that name a file we ship rather than one off the player's install.
#[cfg(test)]
pub(super) fn shipped_manifest_files() -> Vec<String> {
    manifest_files()
        .into_iter()
        .filter(|f| !reference_ui::is_chain_entry(f))
        .collect()
}

/// The reference's own FrameXML load order, on the chain.
const FRAMEXML_TOC: &str = "Interface\\FrameXML\\FrameXML.toc";

/// The in-game manifest for the VM's flavor: [`MANIFEST`]'s entries, and in a Turtle WoW VM the
/// rows Turtle's own `FrameXML.toc` adds to the stock one, spliced in ([`splice_flavor_rows`]).
/// Read once, at the entry edge; a 1.12.1 VM reads nothing.
pub(super) fn ingame_manifest(script: &UiScript) -> Vec<String> {
    let files = Addon::core().toc.files;
    if script.flavor() != UiFlavor::Turtle {
        return files;
    }
    let toc = |bytes: Vec<u8>| Toc::parse(&String::from_utf8_lossy(&bytes)).files;
    let stock = reference_ui::read_stock(FRAMEXML_TOC).map(toc);
    let flavor = reference_ui::read(FRAMEXML_TOC).map(toc);
    match (stock, flavor) {
        (Some(stock), Some(flavor)) => splice_flavor_rows(&files, &stock, &without_shop(flavor)),
        _ => {
            warn!(
                "ui_script: a FrameXML.toc is missing from the chain, so Turtle WoW's own \
                 interface rows do not load"
            );
            files
        }
    }
}

/// The folder of Turtle WoW's donation shop in its `FrameXML.toc`: `Turtle_ShopUI.xml` builds
/// `ShopFrame`, the minimap's `MinimapShopFrame` and `/shop`, and its templates file.
const SHOP_FOLDER: &str = "turtle_shopui\\";

/// `flavor` without Turtle WoW's donation shop rows ([`SHOP_FOLDER`]).
/// Deviation (this fork): the shop does not load. It sells Turtle WoW's own donation rewards, which
/// no other server offers; nothing else in its FrameXML reads it, and the layer's ESC menu has no
/// Donation Rewards rung to open it.
fn without_shop(flavor: Vec<String>) -> Vec<String> {
    flavor
        .into_iter()
        .filter(|row| {
            !row.replace('/', "\\")
                .to_ascii_lowercase()
                .starts_with(SHOP_FOLDER)
        })
        .collect()
}

/// Every in-game entry in load order: [`ingame_manifest`], then the layer's files when `layer` is
/// set. A layer file is a bare name, so [`load_manifest`] reads it from the shipped tree as it
/// reads the core's own; the sliced entry load (`lifecycle`) walks this list one file a step.
pub(super) fn ingame_entries(script: &UiScript, layer: bool) -> Vec<String> {
    let mut files = ingame_manifest(script);
    if layer {
        files.extend(Addon::layer().toc.files);
    } else {
        with_stock_menu(&mut files);
    }
    files
}

/// The stock ESC menu, which the core leaves to the layer's own `GameMenuFrame.xml`.
const STOCK_MENU: &str = "Interface\\FrameXML\\GameMenuFrame.xml";

/// `files` with the stock menu at its `FrameXML.toc` row, right after `UIPanelTemplates.xml`, for
/// a run without the layer: the stock UI's `ToggleGameMenu` and `UpdateMicroButtons` read
/// `GameMenuFrame` unguarded. Nothing is added when that row is missing.
fn with_stock_menu(files: &mut Vec<String>) {
    let seat = files
        .iter()
        .position(|f| f.eq_ignore_ascii_case("Interface\\FrameXML\\UIPanelTemplates.xml"));
    if let Some(i) = seat {
        files.insert(i + 1, STOCK_MENU.to_string());
    }
}

/// `ours` with every row of the `flavor` toc that the `stock` toc lacks, as an
/// `Interface\FrameXML\` entry, and without every stock row the flavor drops, which its client
/// never runs (Turtle WoW's drops `SoundOptionsFrame.xml` and `UIOptionsFrame.xml`, whose scripts
/// call the stock `OptionsFrame.lua` Turtle replaced). Each added row goes after the latest of
/// `ours` among the stock rows the flavor loads before it, so everything the flavor's client has
/// loaded by then is loaded here too, and rows seated at one place keep the flavor's order. With
/// no such row it goes after `GlobalStrings.lua`, and never ahead of entry 0, the font registry.
/// Rows compare case-insensitively; one `ours` already lists is not added twice.
pub(super) fn splice_flavor_rows(
    ours: &[String],
    stock: &[String],
    flavor: &[String],
) -> Vec<String> {
    let key = |row: &str| row.replace('/', "\\").to_ascii_lowercase();
    let chain_key = |row: &str| key(&format!("Interface\\FrameXML\\{row}"));
    let at: std::collections::HashMap<String, usize> =
        ours.iter().enumerate().map(|(i, r)| (key(r), i)).collect();
    let stock: std::collections::HashSet<String> = stock.iter().map(|r| key(r)).collect();
    let listed: std::collections::HashSet<String> = flavor.iter().map(|r| key(r)).collect();
    let dropped: std::collections::HashSet<String> = stock
        .iter()
        .filter(|r| !listed.contains(*r))
        .map(|r| chain_key(r))
        .collect();
    let home = at
        .get(&chain_key("GlobalStrings.lua"))
        .copied()
        .unwrap_or(0);

    // Walking the flavor's order: the latest of ours among the stock rows seen so far.
    let mut latest = home;
    let mut seated: Vec<Vec<String>> = vec![Vec::new(); ours.len()];
    for row in flavor {
        if stock.contains(&key(row)) {
            if let Some(&i) = at.get(&chain_key(row)) {
                latest = latest.max(i);
            }
        } else if !at.contains_key(&chain_key(row)) {
            if let Some(slot) = seated.get_mut(latest) {
                slot.push(format!("Interface\\FrameXML\\{row}"));
            }
        }
    }
    ours.iter()
        .zip(seated)
        .flat_map(|(row, after)| {
            let kept = (!dropped.contains(&key(row))).then(|| row.clone());
            kept.into_iter().chain(after)
        })
        .collect()
}

/// Runs `UIParent_ManageFramePositions` after the manifest has loaded, then installs the load-time
/// repairs below. Never after `Fonts.xml` alone: `UIParent.lua` defines the pass.
pub(super) fn bootstrap_positions(script: &UiScript) -> Vec<String> {
    if let Err(e) = script.run("UIParent_ManageFramePositions()") {
        error!("ui_script: managed-positions bootstrap: {e}");
        return vec![format!("managed-positions bootstrap: {e}")];
    }
    if let Err(e) = apply_buff_durations(script) {
        error!("ui_script: buff-duration layout: {e}");
        return vec![format!("buff-duration layout: {e}")];
    }
    Vec::new()
}

/// Runs `BuffButtons_UpdatePositions` at load, which `BuffFrame_OnLoad` never calls: until the
/// stock `UIOptionsFrame.lua`'s `VARIABLES_LOADED` arm runs it (l.206), the second buff row and
/// the debuff row keep `BuffFrame.xml`'s anchors whatever `SHOW_BUFF_DURATIONS` says.
pub(super) fn apply_buff_durations(script: &UiScript) -> Result<(), String> {
    script
        .run("if BuffButtons_UpdatePositions then BuffButtons_UpdatePositions() end")
        .map_err(|e| e.to_string())
}

/// What a screen-size change re-runs, from `extract::tick_script`'s resize arm: anchors follow the
/// screen, but not a size or seat computed from the old one. The managed pass re-wraps the open
/// bags, whose columns `updateContainerFrameAnchors` lays out by `GetScreenHeight()`.
/// Deviation: [`FULLSCREEN_QUADS_RESEAT`] re-sizes the full-screen quads, which the stock files
/// size only in their OnLoads, because a resize or `uiScale` change here keeps the loaded UI and
/// the blackout would stop short of the new screen. Host-side, not in the layer: it adapts the
/// engine's resizable window, and changes nothing the stock UI does at a fixed size.
pub(super) fn on_screen_resized(script: &UiScript) {
    let _ = script.run("if UIParent_ManageFramePositions then UIParent_ManageFramePositions() end");
    if let Err(e) = script.run(FULLSCREEN_QUADS_RESEAT) {
        error!("ui_script: full-screen quad re-seat: {e}");
    }
}

/// The sizing arithmetic of `WorldMapFrame_OnLoad` (`WorldMapFrame.lua:19-28`) and
/// `CinematicFrame_OnLoad` (`CinematicFrame.lua:6-21`), each existence-guarded: the glue screens
/// have neither. Below 4:3 the bars go back to `CinematicFrame.xml`'s declared 1024 x 128, where
/// the stock OnLoad leaves them.
const FULLSCREEN_QUADS_RESEAT: &str = r#"
if BlackoutWorld then
    local width = GetScreenWidth()
    local height = GetScreenHeight()
    if ( width / height < 4 / 3 ) then
        width = width * 1.25
        height = height * 1.25
    end
    BlackoutWorld:SetWidth( width )
    BlackoutWorld:SetHeight( height )
end
if UpperBlackBar and LowerBlackBar then
    local width = GetScreenWidth()
    local height = GetScreenHeight()
    local barWidth, blackBarHeight = 1024, 128
    if ( width / height > 4 / 3 ) then
        local desiredHeight = width / 2
        if ( desiredHeight > height ) then
            desiredHeight = height
        end
        barWidth = width
        blackBarHeight = ( height - desiredHeight ) / 2
    end
    UpperBlackBar:SetHeight( blackBarHeight )
    UpperBlackBar:SetWidth( barWidth )
    LowerBlackBar:SetHeight( blackBarHeight )
    LowerBlackBar:SetWidth( barWidth )
end
"#;

/// Runs a UI load in the counted sound-suppression scope, as `CGGameUI::Initialize` (`0x48fbf0`,
/// called at login `0x48f681` and `/reloadui` `0x495669`) runs its own: enter (`0x458f50`) at
/// `0x48fbfa`, leave (`0x458f60`) at `0x49016d`, and `PlaySoundByName` (`0x458030`) drops every
/// call in between. `PLAYER_LOGIN` fires inside it only on `/reloadui` (`0x490168`); a fresh login
/// fires it from the player's create (`0x5deb60` to `0x4908c0`), a cascade with its own scope
/// (`0x4908d5` to `0x490a56`). The stock `TargetFrame_OnHide` at load is one sound it swallows.
pub(super) fn silenced_ui_load<S: std::borrow::Borrow<UiScript>, R>(
    script: &mut S,
    body: impl FnOnce(&mut S) -> R,
) -> R {
    script.borrow().push_sound_suppression();
    let out = body(script);
    script.borrow().pop_sound_suppression();
    out
}

/// Loads every [`MANIFEST`] entry at once, then the layer unless [`layer_enabled`] says no, then
/// the load-time repairs. Production splits this load at boot ([`load_font_registry`],
/// [`load_ingame_ui`]), so the callers are the tests and [`crate::addon_harness`]. Returns every
/// failure; a loader error is tagged `"<Addon>/<file>: <error>"`.
pub(crate) fn load_default_ui(script: &UiScript) -> Vec<String> {
    // The client's CVar table first: the stock `UIOptionsFrame.xml` reads CVars in its OnLoads,
    // and a bare `UiScript::new()` has only `benilla-ui`'s own. A re-register only refreshes
    // defaults.
    script.register_cvars(crate::cvars::registered_pairs());
    let layer = layer_enabled();
    let mut failures = silenced_ui_load(&mut &*script, |script| {
        let mut core = Addon::core().toc.files;
        if !layer {
            with_stock_menu(&mut core);
        }
        let mut failures = load_manifest(script, &core);
        if layer {
            failures.extend(load_layer(script));
        }
        failures.extend(bootstrap_positions(script));
        failures
    });
    // A raise one call below the loader's own dispatch (an OnLoad that shows a frame whose OnShow
    // raises) lands in the VM's error list, not the loader's report; it is a load failure too.
    failures.extend(
        script
            .errors()
            .into_iter()
            .map(|e| format!("a handler raised during the load walk: {e}")),
    );
    failures
}

/// Loads a slice of [`MANIFEST`] entries, each from its own store and one at a time, so the
/// manifest's order is the load order across both.
/// Both stores answer `<Include>` / `<Script file=>` the same way: the loader resolves a reference
/// against the *including document's own* directory in its source's path space (1186), which for a
/// chain document means `Interface\FrameXML\ContainerFrame.xml`'s `<Script
/// file="ContainerFrame.lua"/>` reaches `Interface\FrameXML\ContainerFrame.lua` without anything
/// here knowing about it.
pub(super) fn load_manifest(script: &UiScript, files: &[String]) -> Vec<String> {
    let core = Addon::core();
    let reference = reference_ui::addon(
        files
            .iter()
            .filter(|f| reference_ui::is_chain_entry(f))
            .cloned()
            .collect(),
    );
    let mut failures = Vec::new();
    for file in files {
        let from = if reference_ui::is_chain_entry(file) {
            &reference
        } else {
            &core
        };
        failures.extend(from.load_files(script, std::slice::from_ref(file)));
    }
    failures
}

/// Every [`LAYER_MANIFEST`] entry, in order, after the whole core: the layer loads as more of the
/// stock load, with no `ADDON_LOADED` and no registry row, as FrameXML does (`0x48ffed`).
fn load_layer(script: &UiScript) -> Vec<String> {
    let layer = Addon::layer();
    layer.load_files(script, &layer.toc.files)
}

/// The font-object registry alone, `Fonts.xml`, loaded at `Startup`.
/// Deviation: it loads before the login screen because benilla's glue screens share its glyph
/// atlas; the reference's glue has its own `GlueFonts.xml`. The font objects declared outside it
/// (FRIZQT 13, 14 and 20, un-outlined) repeat combinations it has, so the atlas plan loses nothing.
pub(crate) fn load_font_registry(script: &UiScript) -> Vec<String> {
    load_manifest(script, Addon::core().toc.files.get(..1).unwrap_or_default())
}

/// The in-game UI, every core entry after `Fonts.xml`, then the layer unless [`layer_enabled`]
/// says no, then every third-party addon, on entering the world. The reference loads both in `CGGameUI::Initialize` (`0x48fbf0`), reached only from world
/// entry (`0x401570` from `0x46c236`), its addons at `0x4900a3` (`0x51f600`).
///
/// `identity`, `(realm, character)`, names the character's AddOn enable-state file, as the
/// reference keys `AddOns.txt` per character; `roster`, the realm's characters, decides an addon
/// this one has no row for. `version_check` is the persisted `checkAddonVersion`, since this VM
/// has no CVar table yet at load.
pub(crate) fn load_ingame_ui(
    script: &mut UiScript,
    identity: Option<&(String, String)>,
    roster: &[String],
    version_check: bool,
) -> Vec<String> {
    load_ingame_ui_with(script, identity, roster, version_check, layer_enabled())
}

/// [`load_ingame_ui`] with the layer decided by the caller.
pub(super) fn load_ingame_ui_with(
    script: &mut UiScript,
    identity: Option<&(String, String)>,
    roster: &[String],
    version_check: bool,
    layer: bool,
) -> Vec<String> {
    // Bounded, addons included: a chunk that never returns fails as a load error instead of
    // freezing the loading screen. The caller disarms the budget once the edge is done.
    script.set_instruction_budget(super::addons::LOAD_INSTRUCTION_BUDGET);
    let mut failures = load_manifest(
        script,
        ingame_entries(script, layer).get(1..).unwrap_or_default(),
    );
    failures.extend(bootstrap_positions(script));
    // Each addon's `ADDON_LOADED` fires as that addon finishes, as the reference does (`0x51f5ad`).
    failures.extend(super::addons::load_third_party(
        script,
        identity,
        roster,
        version_check,
    ));
    failures
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Fonts.xml` must stay entry 0: [`load_font_registry`] loads entry 0 at startup and
    /// [`load_ingame_ui`] the rest.
    #[test]
    fn the_manifest_is_a_toc_that_starts_with_the_font_registry() {
        let toc = Addon::core().toc;
        assert_eq!(toc.interface_versions(), vec![11200]);
        assert_eq!(toc.directive("Title"), Some("benilla"));
        assert_eq!(
            toc.files.first().map(String::as_str),
            Some("Interface\\FrameXML\\Fonts.xml"),
            "the font registry is the manifest's first entry — the loader splits there"
        );
    }

    fn rows(list: &[&str]) -> Vec<String> {
        list.iter().map(|r| r.to_string()).collect()
    }

    /// Turtle WoW's donation shop rows come off its toc, in either separator and any case; the
    /// rows around them stay, in order.
    #[test]
    fn turtle_s_shop_does_not_load() {
        let turtle = rows(&[
            "GameMenuFrame.xml",
            "Turtle_BarbershopUI\\Turtle_Barbershop.xml",
            "Turtle_ShopUI\\Turtle_ShopUITemplates.xml",
            "turtle_shopui/Turtle_ShopUI.xml",
            "Turtle_TransmogUI\\Turtle_TransmogUI.xml",
        ]);
        assert_eq!(
            without_shop(turtle),
            rows(&[
                "GameMenuFrame.xml",
                "Turtle_BarbershopUI\\Turtle_Barbershop.xml",
                "Turtle_TransmogUI\\Turtle_TransmogUI.xml",
            ])
        );
    }

    /// Turtle WoW's rows go after the latest of ours among the stock rows its toc loads before
    /// them; rows seated together keep its order; entry 0 stays the font registry; a stock row
    /// its toc drops is not loaded.
    #[test]
    fn a_flavor_s_own_rows_follow_every_stock_row_it_loads_before_them() {
        let ours = rows(&[
            "Interface\\FrameXML\\Fonts.xml",
            "Interface\\FrameXML\\GlobalStrings.lua",
            "Interface\\FrameXML\\UIParent.xml",
            "Ours.xml",
            "Interface\\FrameXML\\PartyFrame.xml",
            "Interface\\FrameXML\\Minimap.xml",
            "Interface\\FrameXML\\SoundOptionsFrame.xml",
            "Interface\\FrameXML\\TutorialFrame.xml",
            "GameMenuFrame.xml",
        ]);
        let stock = rows(&[
            "GlobalStrings.lua",
            "Fonts.xml",
            "UIParent.xml",
            "Minimap.xml",
            "PartyFrame.xml",
            "SoundOptionsFrame.xml",
            "TutorialFrame.xml",
            "GameMenuFrame.xml",
            "StatsFrame.xml",
        ]);
        let turtle = rows(&[
            "GlobalStrings.lua",
            "Globals.lua",
            "Overrides.lua",
            "Fonts.xml",
            "UIParent.xml",
            "Minimap.xml",
            "PartyFrame.xml",
            "json.lua",
            "TutorialFrame.xml",
            "GameMenuFrame.xml",
            "StatsFrame.xml",
            "Hooks.lua",
            "LFT/LFT.lua",
            "LFT\\LFT.xml",
        ]);
        assert_eq!(
            splice_flavor_rows(&ours, &stock, &turtle),
            rows(&[
                "Interface\\FrameXML\\Fonts.xml",
                "Interface\\FrameXML\\GlobalStrings.lua",
                "Interface\\FrameXML\\Globals.lua",
                "Interface\\FrameXML\\Overrides.lua",
                "Interface\\FrameXML\\UIParent.xml",
                "Ours.xml",
                "Interface\\FrameXML\\PartyFrame.xml",
                "Interface\\FrameXML\\Minimap.xml",
                // After Minimap.xml, which ours loads after PartyFrame.xml; SoundOptionsFrame.xml,
                // which the flavor drops, is gone.
                "Interface\\FrameXML\\json.lua",
                "Interface\\FrameXML\\TutorialFrame.xml",
                "Interface\\FrameXML\\Hooks.lua",
                "Interface\\FrameXML\\LFT/LFT.lua",
                "Interface\\FrameXML\\LFT\\LFT.xml",
                "GameMenuFrame.xml",
            ])
        );
    }

    /// The stock toc against itself splices nothing, so a 1.12.1 manifest is unchanged; a row
    /// ours already lists is never doubled.
    #[test]
    fn the_stock_toc_against_itself_changes_nothing() {
        let ours = Addon::core().toc.files;
        let stock = rows(&["GlobalStrings.lua", "Fonts.xml", "UIParent.xml"]);
        assert_eq!(splice_flavor_rows(&ours, &stock, &stock), ours);
        let listed = rows(&["GlobalStrings.lua", "UIParent.xml"]);
        assert_eq!(splice_flavor_rows(&ours, &stock[..1], &listed), ours);
    }

    /// A 1.12.1 VM's manifest is benilla.toc as written, with no chain read for a toc.
    #[test]
    fn a_vanilla_vm_loads_the_manifest_as_written() {
        let script = UiScript::new().unwrap();
        assert_eq!(ingame_manifest(&script), Addon::core().toc.files);
    }

    /// A shipped `.xml` the manifest does not list never loads, with no error to say so.
    #[test]
    fn the_manifest_lists_every_shipped_file_and_nothing_else() {
        let mut listed = shipped_manifest_files();
        let mut shipped: Vec<String> = super::super::content::shipped_files()
            .filter(|f| f.ends_with(".xml"))
            .map(str::to_owned)
            .collect();
        listed.sort();
        shipped.sort();
        assert_eq!(listed, shipped);
    }

    /// [`reference_ui::is_chain_entry`] reads a path separator as a chain entry, so a shipped file
    /// in a subdirectory would be looked for on the player's install instead.
    #[test]
    fn every_file_we_ship_is_a_bare_name_so_a_path_can_only_mean_the_chain() {
        for name in super::super::content::shipped_files() {
            assert!(
                !reference_ui::is_chain_entry(name),
                "assets/ui is flat, but ships {name}: a separator makes it a file off the install"
            );
        }
    }

    /// Every chain entry the manifest names is in the player's 1.12 patch chain.
    #[test]
    fn every_chain_entry_resolves_off_the_players_install() {
        let _data = benilla_formats::wow_data_or_skip!();
        let chain: Vec<String> = manifest_files()
            .into_iter()
            .filter(|f| reference_ui::is_chain_entry(f))
            .collect();
        for entry in &chain {
            assert!(
                reference_ui::read(entry).is_some(),
                "benilla.toc sources {entry} off the patch chain, which does not hold it"
            );
        }
    }

    /// The stock pass (`UIParent.lua:1592-1775`) indexes these frames unguarded, so a file that
    /// runs it at load must follow every file it reads, as the reference's toc orders them.
    #[test]
    fn every_load_time_runner_of_the_managed_pass_follows_what_it_reads() {
        let files = manifest_files();
        let at = |leaf: &str| {
            files
                .iter()
                .position(|f| f.ends_with(&format!("\\{leaf}")))
                .unwrap_or_else(|| panic!("the manifest lists {leaf}"))
        };
        // What the pass reads by name (`UIParent.lua:1598-1775`), and the file that declares it.
        const READS: &[(&str, &str)] = &[
            (
                "MultiBarLeft / MultiBarRight / MultiBarBottomLeft",
                "MultiActionBars.xml",
            ),
            (
                "PetActionBarFrame + SlidingActionBarTexture0/1",
                "PetActionBarFrame.xml",
            ),
            ("ReputationWatchBar", "ReputationFrame.xml"),
            (
                "MainMenuExpBar / MainMenuBarMaxLevelBar / MainMenuBar",
                "MainMenuBar.xml",
            ),
            ("CastingBarFrame", "CastingBarFrame.xml"),
            ("QuestTimerFrame", "QuestTimerFrame.xml"),
            ("QuestWatchFrame", "QuestLogFrame.xml"),
            ("DurabilityFrame + its three glyphs", "DurabilityFrame.xml"),
            ("MinimapCluster", "Minimap.xml"),
            (
                "ChatFrame1 / ChatFrame2 (+ FCF_DockUpdate)",
                "FloatingChatFrame.xml",
            ),
            // The shapeshift arm (l.1705-1732): the stance bar's file declares these and also runs
            // the pass, so it must precede the other runner.
            (
                "ShapeshiftBarLeft / Middle / Right",
                "BonusActionBarFrame.xml",
            ),
        ];
        // Who runs it at load: an OnLoad, or the OnShow of a frame its OnLoad shows.
        const RUNNERS: &[(&str, &str)] = &[
            (
                "ShapeshiftBar_OnLoad → Show → OnShow",
                "BonusActionBarFrame.xml",
            ),
            ("WorldStateAlwaysUpFrame OnLoad", "WorldStateFrame.xml"),
        ];
        for (what, runner) in RUNNERS {
            for (name, read) in READS {
                if read == runner {
                    continue; // its own declarations precede its own OnLoad
                }
                assert!(
                    at(read) < at(runner),
                    "{runner} ({what}) loads at {} but reads {name}, declared by {read} at {} — \
                     the pass raises at load and never seats the frame it was run for. Move the \
                     runner below the read, as the reference's toc has it.",
                    at(runner),
                    at(read)
                );
            }
        }
    }

    #[test]
    fn nothing_declares_a_uiparent_child_before_uiparent_itself_loads() {
        let files = manifest_files();
        let at = files
            .iter()
            .position(|f| f == r"Interface\FrameXML\UIParent.xml")
            .expect("the manifest lists UIParent.xml");
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/ui");
        for early in &files[..at] {
            // A chain entry is a stock file off the player's install, not one on disk here.
            if reference_ui::is_chain_entry(early) {
                continue;
            }
            let text = std::fs::read_to_string(dir.join(early)).unwrap();
            assert!(
                !text.contains(r#"parent="UIParent""#),
                "{early} loads before UIParent.xml (position {at}) but declares a \
                 UIParent child — the loader would warn and silently drop it. Move \
                 UIParent.xml up, or the declaration down."
            );
        }
    }
}
