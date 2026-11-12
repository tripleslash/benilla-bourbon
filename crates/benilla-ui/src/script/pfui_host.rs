//! Gated connections from the optional pfUI extension to the stock VM's state.
//! A feature-enabled build still uses the stock ABI until the extension installs its VM flag.

use mlua::{Function, IntoLuaMulti, Lua, MultiValue, Table, Value};

mod addons;
pub mod attributes;
pub mod auras;
mod bubbles;
pub mod casts;
pub(crate) mod chat_event;
mod cursor;
mod descriptors;
pub mod durations;
mod editbox_caret;
mod engine;
pub(crate) mod equipment_actions;
pub mod inline_texture;
pub mod items;
pub mod macro_verbs;
pub mod maps;
mod merchant;
mod message_history;
pub mod player_stats;
pub mod professions;
pub mod quests;
pub mod social;
pub(crate) mod tooltip_subjects;
pub mod totems;
pub mod units;
pub(crate) mod xml;

/// The key a frame registers under for an event the 1.12 table lacks: the name as given, once the
/// extension is installed in an in-game VM. ClassicAPI's events (`PLAYER_SWING`, the modern aura
/// and cast events) are the extension's to fire, and the stock table would drop their listeners.
pub(crate) fn extension_event(model: &super::Model, name: &str) -> Option<String> {
    (model.pfui_cursor.is_some() && model.vm == super::glue::VmKind::InGame)
        .then(|| name.to_string())
}

/// Timer callbacks are protected individually, but errors still belong in the VM error stream.
pub fn report_error(lua: &Lua, message: String) {
    lua.app_data_mut::<crate::script::ModelCell>()
        .expect("model")
        .record_error(message.clone());
}

/// The registry key of the `C_Timer` queue's pass, which the extension's timers set.
pub const TIMER_PASS: &str = "benilla.pfui.timer_pass";

/// ClassicAPI's `C_Timer` queue. Its timers ride the world tick `0x66fd50`, which the world frame's
/// render calls (`CGWorldFrame::OnRender` `0x482ea0`, at `0x483297`) in the draw pass: a frame's
/// timers fire after every `OnUpdate` of its walk, so a deferred scan sees what those updates made,
/// and none fire under the loading screen, where no draw pass runs.
pub(super) fn run_timers(lua: &Lua) {
    let Ok(Some(pass)) = lua.named_registry_value::<Option<Function>>(TIMER_PASS) else {
        return;
    };
    if let Err(e) = pass.call::<()>(()) {
        report_error(lua, e.to_string());
    }
}

/// Extension-owned event payloads use the stock protected dispatcher and legacy argN globals.
/// Every pfUI raise comes this way. ClassicAPI and Nampower raise these events from their own
/// per-frame hooks, part of the client's frame, so a raise here is the tick's
/// ([`super::origin::Origin::Tick`]). A raise made with no host lent and nothing held, a feed's
/// that holds the VM alone, waits for [`raise_held`] once the app asked for it
/// ([`hold_unhosted_raises`]): its handlers read the world through the host, as every stock
/// handler's do, where the VM's own copies of the player's state are not kept current.
pub fn fire_pfui_event(lua: &Lua, event: &str, args: Vec<super::ScriptValue>) {
    if held_for_the_host(lua) {
        if let Some(mut held) = lua.app_data_mut::<HeldRaises>() {
            held.raises.push(Held::Event(event.to_string(), args));
            return;
        }
    }
    super::origin::within(super::origin::Origin::Tick, || {
        super::tick::fire_event_into(lua, event, args);
    });
}

/// Run `dispatch` now with a host lent, else hold it for [`raise_held`] as
/// [`fire_pfui_event`] holds a raise: a dispatch whose handlers read a state it sets around its
/// raises (the combat log's current event) keeps it around them.
pub fn run_hosted(lua: &Lua, dispatch: impl FnOnce(&Lua) + Send + 'static) {
    if held_for_the_host(lua) {
        if let Some(mut held) = lua.app_data_mut::<HeldRaises>() {
            held.raises.push(Held::Dispatch(Box::new(dispatch)));
            return;
        }
    }
    dispatch(lua);
}

/// Whether a raise made now waits for [`raise_held`]: the app asked for it, no host is lent and
/// nothing is held, so its handlers would run with no host to read the world through.
fn held_for_the_host(lua: &Lua) -> bool {
    lua.app_data_ref::<HeldRaises>()
        .is_some_and(|held| held.enabled)
        && super::host::read_host(lua, |_| ()).is_none()
        && super::host::read_reader(lua, |_| ()).is_none()
        && !super::host::holding(lua)
}

/// One held raise: an event, or a dispatch that raises its own ([`run_hosted`]).
enum Held {
    Event(String, Vec<super::ScriptValue>),
    Dispatch(Box<dyn FnOnce(&Lua) + Send>),
}

/// The extension's raises made with no host lent, in the order made ([`fire_pfui_event`]).
#[derive(Default)]
struct HeldRaises {
    enabled: bool,
    raises: Vec<Held>,
}

/// Hold the raises made with no host lent for [`raise_held`]: the app's extension feeds hold the
/// VM without the host.
pub fn hold_unhosted_raises(lua: &Lua) {
    lua.set_app_data(HeldRaises {
        enabled: true,
        raises: Vec::new(),
    });
}

/// Raise what [`fire_pfui_event`] held, in the order made; the app calls it with the host lent.
pub fn raise_held(lua: &Lua) {
    let raises = lua
        .app_data_mut::<HeldRaises>()
        .map(|mut held| std::mem::take(&mut held.raises))
        .unwrap_or_default();
    for raise in raises {
        match raise {
            Held::Event(event, args) => {
                super::origin::within(super::origin::Origin::Tick, || {
                    super::tick::fire_event_into(lua, &event, args);
                });
            }
            Held::Dispatch(dispatch) => dispatch(lua),
        }
    }
}

/// The player's spellbook now: the lent host's, else the VM's copy. Upstream's getters read the
/// book through the host, and the copy is not kept current.
pub(crate) fn player_book(lua: &Lua) -> super::SpellBookState {
    super::live::ask(lua, &super::BookNow).unwrap_or_else(|| {
        lua.app_data_ref::<crate::script::ModelCell>()
            .expect("model")
            .spellbook
            .clone()
    })
}

/// The pet's spellbook now, as [`player_book`] reads the player's.
pub(crate) fn pet_book(lua: &Lua) -> super::PetBookState {
    super::live::ask(lua, &super::PetBookNow).unwrap_or_else(|| {
        lua.app_data_ref::<crate::script::ModelCell>()
            .expect("model")
            .pet_book
            .clone()
    })
}

/// The spell `text` names, `Name(Rank N)`, in the player's book and then the pet's, as
/// `CastSpellByName` resolves it, against the books as they stand at the call.
pub(crate) fn spell_named(lua: &Lua, text: &str) -> Option<super::NamedSpell> {
    let book = super::live::ask(lua, &super::BookNow);
    let pet = super::live::ask(lua, &super::PetBookNow);
    let model = lua
        .app_data_ref::<crate::script::ModelCell>()
        .expect("model");
    super::resolve_spell_in_books(
        book.as_ref().unwrap_or(&model.spellbook),
        pet.as_ref().unwrap_or(&model.pet_book),
        text,
    )
}

/// The item `token`'s unit shows at live-API id `slot`, as the `GetInventoryItem*` getters read
/// it: the token resolved at the call, then the player's own table or another player's visible
/// items.
pub(crate) fn inv_slot(
    lua: &Lua,
    model: &super::Model,
    token: &str,
    slot: usize,
) -> Option<super::InvSlotView> {
    let reader = super::char_stats::inventory_reader(lua, model, token)?;
    model.inv_slot_as(lua, reader, slot)
}

/// Queue `calls` for the host, in order, under the input events running now: made while the host
/// applies a call, they apply right after it, as a click action's steps (the target selected, the
/// cast, the selection put back) each apply as their own call.
pub fn queue_calls(script: &mut super::UiScript, calls: Vec<super::ScriptCall>) {
    let mut model = script.model_mut();
    for call in calls {
        model.queue_call(call);
    }
}

/// Whether `benilla-pfui::install` ran on this VM.
pub(crate) fn installed(lua: &Lua) -> mlua::Result<bool> {
    Ok(lua.named_registry_value::<Option<bool>>("benilla.pfui.installed")? == Some(true))
}

/// A handler call on pfUI's VM, by the stock convention (`this` and `arg1..` set around the call,
/// under the error function) with the modern arguments as well: `self`, then `event` for an
/// `OnEvent` (the walk's global), then `arg1..`. A unit event a frame filtered to other units
/// ([`RegisterUnitEvent`]) is dropped.
pub(crate) fn invoke_handler(
    lua: &Lua,
    wrapper: Table,
    handler: &Function,
    is_event: bool,
    extra: Vec<Value>,
) -> mlua::Result<()> {
    let g = lua.globals();
    let event = match is_event {
        true => match g.get::<Value>("event")? {
            Value::String(s) => Some(s),
            _ => None,
        },
        false => None,
    };
    if let (Some(event), Some(Value::String(unit))) = (&event, extra.first()) {
        let filters: Table = lua.named_registry_value("benilla.pfui.unit_events")?;
        if let Value::Table(events) = filters.raw_get::<Value>(wrapper.clone())? {
            if let Value::Table(units) = events.raw_get::<Value>(event.clone())? {
                let unit = unit.to_string_lossy();
                let matches = units
                    .sequence_values::<String>()
                    .collect::<mlua::Result<Vec<_>>>()?
                    .iter()
                    .any(|allowed| allowed.eq_ignore_ascii_case(&unit));
                if !matches {
                    return Ok(());
                }
            }
        }
    }
    let mut names = vec!["this".to_string()];
    names.extend((1..=extra.len()).map(|i| format!("arg{i}")));
    let saved: Vec<Value> = names
        .iter()
        .map(|name| g.get::<Value>(name.as_str()))
        .collect::<mlua::Result<_>>()?;
    let mut args = vec![Value::Table(wrapper.clone())];
    args.extend(event.map(Value::String));
    args.extend(extra.iter().cloned());
    g.set("this", wrapper)?;
    for (name, v) in names.iter().skip(1).zip(extra) {
        g.set(name.as_str(), v)?;
    }
    let error_function = super::stdlib::error_function(lua);
    let outcome = super::origin::within(super::origin::Origin::Script, || {
        super::stdlib::with_capture(|| {
            let args = MultiValue::from_vec(args);
            match &error_function {
                Some(h) => handler.call_under::<()>(h, args),
                None => handler.call::<()>(args),
            }
        })
    });
    for (name, v) in names.iter().zip(saved) {
        g.set(name.as_str(), v)?;
    }
    outcome
}

/// Publish the GUID resolved by the stock unit resolver. Its case folding and target hops are
/// reused, while the extension separately owns additional unit-token kinds.
pub fn install(lua: &Lua) -> mlua::Result<()> {
    let globals = lua.globals();
    if !globals.get::<Value>("UnitGUID")?.is_nil() {
        return Err(mlua::Error::runtime("UnitGUID already installed"));
    }
    globals.set(
        "UnitGUID",
        lua.create_function(|lua, token: String| {
            let guid = match units::custom_guid(lua, &token) {
                Some(guid) => guid,
                None => {
                    let model = lua
                        .app_data_ref::<crate::script::ModelCell>()
                        .expect("model");
                    super::unit::guid_of(lua, &model, &token)?
                }
            };
            Ok(guid
                .filter(|guid| *guid != 0)
                .map(|guid| format!("0x{guid:016X}")))
        })?,
    )?;
    let methods: Table = lua.named_registry_value(super::REG_FRAME_METHODS)?;
    methods.set(
        "IsEventRegistered",
        lua.create_function(|lua, (frame, event): (Table, String)| {
            let handle = super::object::frame_handle_of(lua, &frame)?;
            let registered = lua
                .app_data_ref::<crate::script::ModelCell>()
                .expect("model")
                .frame_events
                .get(&handle)
                .is_some_and(|events| events.contains(&event));
            let mut result = vec![Value::Boolean(registered)];
            if registered {
                let filters: Table = lua.named_registry_value("benilla.pfui.unit_events")?;
                if let Value::Table(events) = filters.raw_get::<Value>(frame)? {
                    if let Value::Table(units) = events.raw_get::<Value>(event)? {
                        result.extend(
                            units
                                .sequence_values::<Value>()
                                .collect::<mlua::Result<Vec<_>>>()?,
                        );
                    }
                }
            }
            Ok(MultiValue::from_vec(result))
        })?,
    )?;
    globals.set("INTERFACE_VERSION", super::addon_gate::CLIENT_INTERFACE)?;
    items::install(lua)?;
    cursor::install(lua)?;
    merchant::install(lua)?;
    addons::install(lua)?;
    bubbles::install(lua)?;
    descriptors::install(lua)?;
    engine::install(lua)?;
    durations::install(lua)?;
    xml::install(lua)?;
    message_history::install(lua)?;
    editbox_caret::install(lua)?;
    chat_event::install(lua)?;
    social::install(lua)?;
    units::install(lua)?;
    player_stats::install(lua)?;
    totems::install(lua)?;
    super::nameplate::pfui::install(lua)?;
    attributes::install(lua)?;
    auras::install(lua)?;
    super::tooltip_spell::pfui::install(lua)?;
    tooltip_subjects::install(lua)?;
    super::tooltip_item::pfui::install(lua)?;
    quests::install(lua)?;
    super::macros::pfui::install(lua)?;
    super::reputation::pfui::install(lua)?;
    macro_verbs::install(lua)?;
    Ok(())
}

pub(super) fn item_received(lua: &Lua, item_id: u32) {
    item_result(lua, item_id, true);
}

/// The host calls this for a server's negative item-query answer as well as successful loads.
pub fn item_result(lua: &Lua, item_id: u32, success: bool) {
    if lua
        .named_registry_value::<Option<bool>>("benilla.pfui.installed")
        .ok()
        .flatten()
        == Some(true)
    {
        fire_pfui_event(
            lua,
            "GET_ITEM_INFO_RECEIVED",
            vec![
                super::ScriptValue::Number(f64::from(item_id)),
                super::ScriptValue::Bool(success),
            ],
        );
    }
}

/// A load scope shared by TOC files, XML inclusions and synchronous LoadAddOn calls. Restoring
/// the enclosing scope keeps a dependent addon's private table from leaking into its caller.
pub struct AddonScope<'a> {
    lua: &'a Lua,
    previous: Value,
    active: bool,
    name: String,
}

impl Drop for AddonScope<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self
                .lua
                .set_named_registry_value("benilla.pfui.addon_context", self.previous.clone());
            // The extension's after-load hook (its memory measure).
            if let Ok(Some(bridge)) = self
                .lua
                .named_registry_value::<Option<Table>>("benilla.pfui.globals_bridge")
            {
                if let Ok(Some(after)) = bridge.get::<Option<Function>>("afterAddon") {
                    let _ = after.call::<()>(self.name.as_str());
                }
            }
        }
    }
}

pub fn addon_scope<'a>(lua: &'a Lua, name: &str) -> mlua::Result<AddonScope<'a>> {
    let active = lua.named_registry_value::<Option<bool>>("benilla.pfui.installed")? == Some(true);
    let previous: Value = lua.named_registry_value("benilla.pfui.addon_context")?;
    if active {
        let privates: Table = lua.named_registry_value("benilla.pfui.addon_privates")?;
        let private = match privates.raw_get::<Value>(name)? {
            Value::Table(table) => table,
            _ => {
                let table = lua.create_table()?;
                privates.raw_set(name, table.clone())?;
                table
            }
        };
        let context = lua.create_table()?;
        context.set(1, name)?;
        context.set(2, private)?;
        lua.set_named_registry_value("benilla.pfui.addon_context", context)?;
        // Stock globals a preceding load assigned are adapted before this addon reads them.
        if let Some(bridge) =
            lua.named_registry_value::<Option<Table>>("benilla.pfui.globals_bridge")?
        {
            if let Some(adapt) = bridge.get::<Option<Function>>("beforeAddon")? {
                adapt.call::<()>(name)?;
            }
        }
    }
    Ok(AddonScope {
        lua,
        previous,
        active,
        name: name.to_string(),
    })
}

/// The varargs an addon file's chunk runs with: `(addonName, privateTable)` inside an addon's
/// scope ([`addon_scope`]), none elsewhere.
pub(crate) fn chunk_args(lua: &Lua) -> Result<MultiValue, (mlua::Error, Vec<u8>)> {
    let args = || -> mlua::Result<MultiValue> {
        match lua.named_registry_value("benilla.pfui.addon_context")? {
            Value::Table(context) => {
                (context.get::<String>(1)?, context.get::<Table>(2)?).into_lua_multi(lua)
            }
            _ => Ok(MultiValue::new()),
        }
    };
    args().map_err(|e| (e, Vec::new()))
}
