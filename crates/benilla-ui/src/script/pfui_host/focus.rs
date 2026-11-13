//! Resolve extra tokens through live snapshots while reusing the stock read-only query ABI.
//! Aliases are private to one query and restored before addons regain control, including errors.

use super::{Metadata, UnitState};
use crate::script::{ScriptCall, SelectionRequest};

#[path = "unit_events.rs"]
mod events;
use mlua::{Function, Lua, MultiValue, Value};
use std::sync::{Arc, Mutex};

/// Session-owned focus survives a UI VM replacement, without saved-variable persistence.
#[derive(Clone, Default)]
pub struct FocusMemory(Arc<Mutex<Option<u64>>>);

impl FocusMemory {
    pub fn clear(&self) {
        self.set(None);
    }

    fn get(&self) -> Option<u64> {
        *self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn set(&self, guid: Option<u64>) {
        *self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = guid;
    }
}

pub(super) fn retained(lua: &Lua) -> Option<u64> {
    lua.app_data_ref::<FocusMemory>()
        .and_then(|memory| memory.get())
}

pub(super) fn resolve(lua: &Lua, token: &str) -> Option<Option<u64>> {
    if !token
        .get(..2)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("0x"))
        && !token
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("focus"))
        && !token
            .get(..9)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("nameplate"))
        && !token
            .get(..4)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("mark"))
    {
        return None;
    }
    let token = token.to_ascii_lowercase();
    if let Some(hex) = token.strip_prefix("0x") {
        return Some(if matches!(hex.len(), 8 | 16) {
            u64::from_str_radix(hex, 16).ok().filter(|guid| *guid != 0)
        } else {
            None
        });
    }
    let (mut guid, mut suffix) = if let Some(rest) = token.strip_prefix("nameplate") {
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        let index = rest[..digits].parse::<usize>().ok();
        (
            index.and_then(|index| super::super::super::nameplate::pfui::guid(lua, index)),
            &rest[digits..],
        )
    } else if let Some(rest) = token.strip_prefix("mark") {
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        let index = rest[..digits]
            .parse::<u8>()
            .ok()
            .filter(|index| (1..=8).contains(index));
        let guid = index.and_then(|index| {
            lua.app_data_ref::<Metadata>()?
                .states
                .values()
                .find(|state| state.has_object && state.raid_target == index)
                .map(|state| state.guid)
        });
        (guid, &rest[digits..])
    } else {
        (
            lua.app_data_ref::<Metadata>()?.focus,
            token.strip_prefix("focus")?,
        )
    };
    if !suffix.len().is_multiple_of(6)
        || !suffix.as_bytes().chunks(6).all(|chunk| chunk == b"target")
    {
        return Some(None);
    }
    let metadata = lua.app_data_ref::<Metadata>()?;
    while !suffix.is_empty() {
        guid = guid
            .and_then(|guid| metadata.targets.get(&guid).copied())
            .filter(|guid| *guid != 0);
        suffix = &suffix[6..];
    }
    Some(guid)
}

pub(super) fn focus(lua: &Lua, guid: Option<u64>) {
    let changed = {
        let mut metadata = lua.app_data_mut::<Metadata>().expect("unit metadata");
        let changed = metadata.focus != guid;
        metadata.focus = guid;
        metadata.restored_focus_pending = false;
        changed
    };
    if let Some(memory) = lua.app_data_ref::<FocusMemory>() {
        memory.set(guid);
    }
    if changed {
        crate::script::pfui_host::fire_pfui_event(lua, "PLAYER_FOCUS_CHANGED", vec![]);
    }
}

pub(super) fn state(lua: &Lua, guid: u64) -> Option<UnitState> {
    if let Some(state) = lua.app_data_ref::<Metadata>()?.states.get(&guid) {
        return Some(state.clone());
    }
    let model = lua.app_data_ref::<crate::script::ModelCell>()?;
    model
        .units_by_lower
        .values()
        .find(|state| state.guid == guid)
        .cloned()
        .or_else(|| model.units_by_guid.get(&guid).cloned())
}

pub(super) fn feed_states(lua: &Lua, states: impl IntoIterator<Item = (u64, UnitState, u64)>) {
    let before = events::snapshot(lua);
    {
        let Some(mut metadata) = lua.app_data_mut::<Metadata>() else {
            return;
        };
        metadata.states.clear();
        metadata.targets.clear();
        for (guid, state, target) in states {
            metadata.states.insert(guid, state);
            metadata.targets.insert(guid, target);
        }
    }
    events::dispatch(lua, before, events::snapshot(lua));
}

pub(super) fn announce_restore(lua: &Lua) {
    let restored = {
        let Some(mut metadata) = lua.app_data_mut::<Metadata>() else {
            return;
        };
        let restored = metadata.restored_focus_pending
            && metadata
                .focus
                .and_then(|guid| metadata.states.get(&guid))
                .is_some_and(|state| state.exists);
        if restored {
            metadata.restored_focus_pending = false;
        }
        restored
    };
    if restored {
        crate::script::pfui_host::fire_pfui_event(lua, "PLAYER_FOCUS_CHANGED", vec![]);
    }
}

struct SavedAlias {
    token: &'static str,
    guid: u64,
    state: Option<UnitState>,
    added_held: Option<u64>,
}
struct Aliases<'a> {
    lua: &'a Lua,
    saved: Vec<SavedAlias>,
    /// The guid this call pinned on `npc`, and the pin an enclosing call held before it.
    npc: Option<u64>,
    outer_npc: Option<AliasedNpc>,
}

/// The guid an alias holds on `npc` while the original getter runs. The live resolver reads `npc`
/// from the host's interaction global, which the alias cannot swap, so it asks for this first
/// ([`aliased_npc`]).
#[derive(Clone, Copy)]
struct AliasedNpc(u64);

/// The guid an extension token is aliased to on `npc` for the call under way, if any.
pub(crate) fn aliased_npc(lua: &Lua) -> Option<u64> {
    lua.app_data_ref::<AliasedNpc>().map(|pin| pin.0)
}

impl Drop for Aliases<'_> {
    fn drop(&mut self) {
        if self.npc.is_some() {
            match self.outer_npc.take() {
                Some(outer) => {
                    self.lua.set_app_data(outer);
                }
                None => {
                    self.lua.remove_app_data::<AliasedNpc>();
                }
            }
        }
        let mut model = self
            .lua
            .app_data_mut::<crate::script::ModelCell>()
            .expect("model");
        for saved in self.saved.drain(..).rev() {
            match saved.state {
                Some(state) => {
                    model.units_by_lower.insert(saved.token.into(), state);
                }
                None => {
                    model.units_by_lower.remove(saved.token);
                }
            }
            if saved.token == "npc" {
                model.unit_guids.npc = saved.guid;
            } else {
                model.unit_guids.mouseover = saved.guid;
            }
            if let Some(guid) = saved.added_held {
                model.unit_guids.held.remove(&guid);
            }
        }
    }
}

impl Aliases<'_> {
    fn token(&mut self, guid: Option<u64>) -> String {
        let Some(guid) = guid else {
            return String::new();
        };
        {
            let model = self
                .lua
                .app_data_ref::<crate::script::ModelCell>()
                .expect("model");
            for token in ["player", "pet", "target", "npc", "mouseover"] {
                if crate::script::unit::guid_of(self.lua, &model, token)
                    .ok()
                    .flatten()
                    == Some(guid)
                    && crate::script::unit::view(self.lua, &model, token).is_some()
                {
                    return token.into();
                }
            }
            for (i, party) in model.unit_guids.party.iter().enumerate() {
                if *party == guid {
                    return format!("party{}", i + 1);
                }
            }
            for (i, raid) in model.unit_guids.raid.iter().enumerate() {
                if *raid == guid {
                    return format!("raid{}", i + 1);
                }
            }
        }
        let token = if self.saved.is_empty() {
            "npc"
        } else {
            "mouseover"
        };
        let replacement = state(self.lua, guid);
        let mut model = self
            .lua
            .app_data_mut::<crate::script::ModelCell>()
            .expect("model");
        let old = model.units_by_lower.remove(token);
        if let Some(state) = replacement {
            model.units_by_lower.insert(token.into(), state);
        }
        let old_guid = if token == "npc" {
            self.npc = Some(guid);
            std::mem::replace(&mut model.unit_guids.npc, guid)
        } else {
            std::mem::replace(&mut model.unit_guids.mouseover, guid)
        };
        let added_held = if let std::collections::hash_map::Entry::Vacant(entry) =
            model.unit_guids.held.entry(guid)
        {
            entry.insert(0);
            Some(guid)
        } else {
            None
        };
        self.saved.push(SavedAlias {
            token,
            guid: old_guid,
            state: old,
            added_held,
        });
        token.into()
    }
}

pub(super) fn call(lua: &Lua, original: &Function, args: MultiValue) -> mlua::Result<MultiValue> {
    if !args.iter().any(|value| match value {
        Value::String(token) => resolve(lua, &token.to_string_lossy()).is_some(),
        _ => false,
    }) {
        return original.call::<MultiValue>(args);
    }
    let mut aliases = Aliases {
        lua,
        saved: vec![],
        npc: None,
        outer_npc: None,
    };
    let mut replaced = Vec::new();
    for arg in args {
        let custom = if let Value::String(token) = &arg {
            resolve(lua, &token.to_string_lossy())
        } else {
            None
        };
        replaced.push(if let Some(guid) = custom {
            Value::String(lua.create_string(aliases.token(guid))?)
        } else {
            arg
        });
    }
    if let Some(npc) = aliases.npc {
        aliases.outer_npc = lua.app_data_ref::<AliasedNpc>().map(|pin| *pin);
        lua.set_app_data(AliasedNpc(npc));
    }
    original.call::<MultiValue>(MultiValue::from_vec(replaced))
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let globals = lua.globals();
    for name in [
        "UnitExists",
        "UnitIsVisible",
        "UnitIsTapped",
        "UnitIsTappedByPlayer",
        "UnitIsPartyLeader",
        "UnitName",
        "UnitHealth",
        "UnitHealthMax",
        "UnitLevel",
        "UnitIsCorpse",
        "UnitCanAttack",
        "UnitCanAssist",
        "UnitIsDead",
        "UnitIsGhost",
        "UnitIsDeadOrGhost",
        "UnitReaction",
        "UnitIsEnemy",
        "UnitIsFriend",
        "UnitIsCivilian",
        "UnitPlayerControlled",
        "UnitIsPlayer",
        "UnitIsPlusMob",
        "UnitIsUnit",
        "UnitAffectingCombat",
        "UnitInRaid",
        "UnitInParty",
        "UnitCanCooperate",
        "UnitIsConnected",
        "UnitIsPVP",
        "UnitIsPVPFreeForAll",
        "UnitPVPName",
        "UnitPVPRank",
        "UnitFactionGroup",
        "UnitRace",
        "UnitClass",
        "UnitHasRelicSlot",
        "UnitSex",
        "UnitCreatureType",
        "UnitClassification",
        "UnitPowerType",
        "UnitMana",
        "UnitManaMax",
        "UnitIsCharmed",
        "UnitBuff",
        "UnitDebuff",
        "GetComboPoints",
        "GetGuildInfo",
        "GetRaidTargetIndex",
    ] {
        let original: Function = globals.get(name)?;
        if name == "UnitName" {
            // A token that names a unit by guid has a name on the reference even when no object
            // is held for it: the name cache's, else `UNKNOWNOBJECT` (`0x517216`); nil is for
            // GUID 0 alone. The stock binding answers nil there, and pfUI, having resolved a
            // token from a tooltip's guid, passes the name straight to `gsub`.
            globals.set(
                name,
                lua.create_function(move |lua, args: MultiValue| {
                    let token = match args.front() {
                        Some(Value::String(token)) => Some(token.to_string_lossy()),
                        _ => None,
                    };
                    let results = call(lua, &original, args)?;
                    if !matches!(results.front(), None | Some(Value::Nil)) {
                        return Ok(results);
                    }
                    let named = token.is_some_and(|token| {
                        super::guid(lua, &token).is_ok_and(|g| g.is_some_and(|g| g != 0))
                    });
                    if !named {
                        return Ok(results);
                    }
                    let unknown = lua.globals().get::<Value>("UNKNOWNOBJECT")?;
                    Ok(match unknown {
                        Value::String(_) => MultiValue::from_vec(vec![unknown, Value::Nil]),
                        _ => results,
                    })
                })?,
            )?;
            continue;
        }
        globals.set(
            name,
            lua.create_function(move |lua, args: MultiValue| call(lua, &original, args))?,
        )?;
    }
    globals.set(
        "FocusUnit",
        lua.create_function(|lua, token: Option<String>| {
            let guid = super::super::auras::unit_guid(lua, token.as_deref().unwrap_or("target"))
                .filter(|guid| state(lua, *guid).is_some_and(|state| state.exists));
            focus(lua, guid);
            Ok(())
        })?,
    )?;
    globals.set(
        "ClearFocus",
        lua.create_function(|lua, ()| {
            focus(lua, None);
            Ok(())
        })?,
    )?;
    let original: Function = globals.get("TargetUnit")?;
    globals.set(
        "TargetUnit",
        lua.create_function(move |lua, token: String| {
            if let Some(guid) = resolve(lua, &token) {
                if let Some(guid) = guid {
                    // The gate's tag 2, which the selection's bind asks while the cursor is up.
                    let may_cast =
                        crate::script::protected::gate(lua, crate::script::protected::Tag::CAST);
                    crate::script::calls::submit(
                        lua,
                        ScriptCall::Select {
                            request: SelectionRequest::PfuiGuid(guid),
                            may_cast,
                        },
                    )?;
                }
                Ok(())
            } else {
                original.call(token)
            }
        })?,
    )
}
