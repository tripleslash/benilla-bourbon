//! Optional pfUI integration and feeds from the client's real state.

use benilla_ui::script::UiScript;
use bevy::prelude::*;

mod bags;
mod casts;
pub(crate) mod clicks;
mod combat_log;
mod icons;
pub(crate) mod input;
mod item_sets;
mod lockout;
mod loot;
#[cfg(feature = "nampower")]
mod nampower;
mod quests;
mod resurrect;
mod session_events;
mod social;
#[cfg(feature = "nampower")]
pub(crate) mod spell_queue;
mod spells;
mod swing;
#[cfg(test)]
pub(crate) mod test_support;
pub(crate) mod toc;
mod totems;
mod units;

pub(crate) fn prepare(script: &UiScript) -> Result<(), String> {
    let lua = script.lua();
    if lua
        .named_registry_value::<Option<bool>>("benilla.pfui.enabled")
        .map_err(|e| e.to_string())?
        != Some(true)
    {
        return Ok(());
    }
    if lua
        .named_registry_value::<Option<bool>>("benilla.pfui.data_installed")
        .map_err(|e| e.to_string())?
        == Some(true)
    {
        return Ok(());
    }
    let locale = script
        .eval::<String>("return GetLocale()")
        .map_err(|e| e.to_string())?;
    let reader = |path: &str| crate::ui_script::pfui_read(path);
    let creatures = benilla_pfui::creature::Catalog::read(&reader, &locale)
        .map_err(|e| format!("pfUI creature metadata: {e}"))?;
    let spells = benilla_pfui::spell::Catalog::read(&reader, &locale)
        .map_err(|e| format!("pfUI spell metadata: {e}"))?;
    // Without its tables the profession links answer nil; the rest of the data still installs.
    let trade_links = benilla_pfui::trade_link::Catalog::read(&reader).unwrap_or_else(|e| {
        warn!("pfUI profession links: {e}");
        benilla_pfui::trade_link::Catalog::default()
    });
    benilla_pfui::install(script).map_err(|e| e.to_string())?;
    // The feeds below hold the VM alone; what they raise runs with the host lent
    // ([`raise_held_events`]).
    benilla_ui::script::pfui_host::hold_unhosted_raises(lua);
    creatures.install(lua).map_err(|e| e.to_string())?;
    spells.install(lua).map_err(|e| e.to_string())?;
    trade_links
        .install(lua, &spells)
        .map_err(|e| e.to_string())?;
    static MAPS: std::sync::OnceLock<Result<benilla_pfui::maps::Catalog, String>> =
        std::sync::OnceLock::new();
    MAPS.get_or_init(|| benilla_pfui::maps::Catalog::read(&reader).map_err(|e| e.to_string()))
        .as_ref()
        .map_err(Clone::clone)?
        .install(lua)
        .map_err(|e| e.to_string())?;
    icons::prepare(script)?;
    item_sets::install(lua).map_err(|e| e.to_string())?;
    benilla_pfui::install_class_names(
        script,
        creatures
            .classes
            .iter()
            .map(|(token, name)| (token.as_str(), name.as_str())),
    )
    .map_err(|e| e.to_string())?;
    lua.set_named_registry_value("benilla.pfui.data_installed", true)
        .map_err(|e| e.to_string())
}

pub(crate) struct PfuiPlugin;

#[derive(Resource)]
struct Enabled;

#[derive(Resource, Default)]
struct SessionFocus(benilla_ui::script::pfui_host::units::FocusMemory);

pub(crate) fn enable_vm(world: &World, script: &UiScript) {
    if world.contains_resource::<Enabled>() {
        if let Some(focus) = world.get_resource::<SessionFocus>() {
            benilla_ui::script::pfui_host::units::retain_focus(script.lua(), focus.0.clone());
        }
        if let Err(error) = script
            .lua()
            .set_named_registry_value("benilla.pfui.enabled", true)
        {
            benilla_ui::script::pfui_host::report_error(
                script.lua(),
                (format!("pfUI VM activation: {error}")).to_string(),
            );
        }
        // Before any addon loads: pfUI reads `gxResolution` once, as it loads.
        if let Some(pixels) = world.get_resource::<WindowPixels>() {
            let _ = benilla_pfui::display::set_window_pixels(script.lua(), pixels.0, pixels.1);
        }
    }
}

/// The primary window's size in device pixels, tracked every frame so a VM built at any moment
/// starts with it.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
struct WindowPixels(u32, u32);

fn track_window_pixels(
    window: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut pixels: ResMut<WindowPixels>,
) {
    if let Ok(window) = window.single() {
        let now = WindowPixels(window.physical_width(), window.physical_height());
        if *pixels != now {
            *pixels = now;
        }
    }
}

/// Extension snapshots are visible before stock unit events dispatch to addons.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct InputSnapshots;

pub(crate) fn resolve_item_misses(
    script: &UiScript,
    pending: &mut std::collections::HashSet<u32>,
    items: &crate::items::Items,
) {
    let failed: Vec<_> = pending
        .iter()
        .copied()
        .filter(|id| items.template_answered_unknown(*id))
        .collect();
    for id in failed {
        pending.remove(&id);
        benilla_ui::script::pfui_host::item_result(script.lua(), id, false);
    }
}

impl Plugin for PfuiPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Enabled);
        app.init_resource::<SessionFocus>();
        app.init_resource::<crate::ui_items::pfui::DeferredItemUse>();
        app.configure_sets(
            Update,
            InputSnapshots
                .in_set(crate::ui_script::UiFeed)
                .before(crate::ui_unit::UnitFeed),
        );
        // After the lanes' drain: the frame's packets have fired their stock events, and the
        // tick is next.
        app.add_systems(
            Update,
            raise_held_events
                .in_set(crate::lanes::UiLanes)
                .after(crate::lanes::fire_in_order),
        );
        casts::register(app);
        input::register(app);
        combat_log::register(app);
        resurrect::register(app);
        lockout::register(app);
        #[cfg(feature = "nampower")]
        spell_queue::register(app);
        #[cfg(feature = "nampower")]
        nampower::register(app);
        bags::register(app);
        loot::register(app);
        quests::register(app);
        session_events::register(app);
        crate::ui_text::inline_texture::register(app);
        social::register(app);
        spells::register(app);
        swing::register(app);
        units::register(app);
        totems::register(app);
        crate::ui_unit::pfui::register(app);
        crate::ui_aura::pfui::register(app);
        crate::target::pfui::register(app);
        app.add_systems(Update, feed_clock.in_set(InputSnapshots));
        app.init_resource::<WindowPixels>();
        app.add_systems(Update, track_window_pixels);
        app.add_systems(
            Update,
            feed_window_pixels
                .in_set(InputSnapshots)
                .after(track_window_pixels),
        );
        app.add_systems(
            Update,
            inventory_events
                .after(crate::ui_unit::UnitFeed)
                .in_set(crate::ui_script::UiFeed)
                .run_if(crate::ui_script::ingame_ui_up),
        );
        app.add_systems(Update, feed_inventory.in_set(InputSnapshots));
        app.add_systems(
            Update,
            crate::ui_social::pfui::feed_friend_guids.in_set(crate::ui_script::UiFeed),
        );
    }
}

/// The raises the extension's feeds made with the VM alone, run with the host lent: a handler
/// reads the player and the units through the host and makes its calls inside it, as the lanes'
/// own handlers do. The VM's copies of the player's state are not kept current, so a handler run
/// without the host read the player as absent (pfUI's castbar never showed).
fn raise_held_events(mut vm: crate::script_calls::HostedVm) {
    vm.run(|script| {
        if let Some(script) = script {
            benilla_ui::script::pfui_host::raise_held(script.lua());
        }
    });
}

fn inventory_events(script: Option<NonSend<UiScript>>) {
    if let Some(script) = script {
        benilla_ui::script::pfui_host::units::announce_restored_focus(script.lua());
        benilla_ui::script::pfui_host::items::dispatch_inventory_events(script.lua());
    }
}

/// A resize reaches the running VM too.
fn feed_window_pixels(script: Option<NonSend<UiScript>>, pixels: Res<WindowPixels>) {
    if !pixels.is_changed() {
        return;
    }
    if let Some(script) = script {
        if let Err(error) =
            benilla_pfui::display::set_window_pixels(script.lua(), pixels.0, pixels.1)
        {
            benilla_ui::script::pfui_host::report_error(
                script.lua(),
                (format!("pfUI window size: {error}")).to_string(),
            );
        }
    }
}

fn feed_clock(script: Option<NonSend<UiScript>>, clock: Res<crate::net::ServerTime>) {
    let Some(script) = script else {
        return;
    };
    if let Err(error) = benilla_pfui::clock::update(script.lua(), clock.0.pfui_local_epoch()) {
        benilla_ui::script::pfui_host::report_error(
            script.lua(),
            (format!("pfUI clock: {error}")).to_string(),
        );
    }
}

fn feed_inventory(
    script: Option<NonSend<UiScript>>,
    self_q: Query<&crate::net::ObjectStore, With<crate::net::SelfPlayer>>,
    objects: crate::net::Objects,
    items: Res<crate::items::Items>,
    props: Option<Res<crate::items::RandomProperties>>,
) {
    let Some(script) = script else {
        return;
    };
    let mut instances = Vec::new();
    benilla_ui::script::pfui_host::items::observe_equipment_guids(
        script.lua(),
        std::array::from_fn(|slot| {
            self_q
                .single()
                .ok()
                .and_then(|player| player.0.player_inv_slot(slot as u8))
                .unwrap_or(0)
        }),
    );
    if let Ok(player) = self_q.single() {
        for bag in [
            benilla_ui::script::EQUIPMENT_BAG,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ] {
            let slots = match bag {
                benilla_ui::script::EQUIPMENT_BAG => 19,
                -2 | 0 => 16,
                -1 => 24,
                _ => {
                    let bag_guid = if bag <= 4 {
                        player.0.player_inv_slot(19 + bag as u8 - 1)
                    } else {
                        player.0.player_bank_bag_slot(bag as u8 - 5)
                    };
                    bag_guid
                        .and_then(|g| objects.object(g))
                        .and_then(|o| o.container_num_slots())
                        .unwrap_or(0)
                        .min(255)
                }
            };
            for slot0 in 0..slots as u8 {
                let Some(guid) = crate::ui_items::slot_guid(&player.0, bag, slot0, &objects) else {
                    continue;
                };
                let Some(item) = objects.object(guid) else {
                    continue;
                };
                let item_id = item.object_entry().unwrap_or(0);
                let count = item.item_stack_count().unwrap_or(1);
                let charges = item.item_spell_charges(0).unwrap_or(0);
                let name = items
                    .template(item_id, 0, crate::query_cache::Ask::Read)
                    .map(|t| {
                        crate::items::item_display_name(
                            &t.name,
                            item.item_random_properties_id() as i32,
                            props.as_deref(),
                        )
                    });
                instances.push((
                    (bag, u32::from(slot0) + 1),
                    benilla_ui::script::pfui_host::items::Instance {
                        guid,
                        item_id,
                        count,
                        name,
                        uses: count.saturating_mul(if charges < 0 {
                            charges.unsigned_abs()
                        } else {
                            1
                        }),
                    },
                ));
            }
        }
    }
    benilla_ui::script::pfui_host::items::set_instances(script.lua(), instances);
}

/// Queue `calls` for the host under the input events running now: made while it applies another
/// call, they apply right after it, inside the events that one was made under.
pub(crate) fn requeue(
    script: &mut benilla_ui::script::UiScript,
    calls: Vec<benilla_ui::script::ScriptCall>,
) {
    benilla_ui::script::pfui_host::queue_calls(script, calls);
}

/// A class id's English name and its `classFile` token, which the extension's caches key on; the
/// token is locale-independent, as `UnitClass`'s second return is.
pub(crate) fn class_names(class: u8) -> Option<(&'static str, &'static str)> {
    Some(match class {
        1 => ("Warrior", "WARRIOR"),
        2 => ("Paladin", "PALADIN"),
        3 => ("Hunter", "HUNTER"),
        4 => ("Rogue", "ROGUE"),
        5 => ("Priest", "PRIEST"),
        7 => ("Shaman", "SHAMAN"),
        8 => ("Mage", "MAGE"),
        9 => ("Warlock", "WARLOCK"),
        11 => ("Druid", "DRUID"),
        _ => return None,
    })
}
