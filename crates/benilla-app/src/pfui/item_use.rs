//! Direct addon item use reuses the stock cast and bind gates without vendor/bank click modes.

use super::*;
use crate::net::Objects;
use crate::spell::cast_target::CastTargeting;

struct PendingTarget {
    item: u64,
    unit: Option<u64>,
    auto_self_cast: bool,
    session: u64,
}

/// Lives beside the stock pending use, so replacing the VM cannot lose its ownership marker.
#[derive(Resource, Default)]
pub(crate) struct DeferredItemUse {
    pending: Option<PendingTarget>,
    serial: u64,
}

impl DeferredItemUse {
    pub(crate) fn clear(&mut self) {
        self.pending = None;
        self.serial = self.serial.wrapping_add(1);
    }
}

fn position(
    store: &benilla_protocol::ObjectFields,
    objects: &Objects,
    guid: u64,
) -> Option<(i64, u32)> {
    for bag in [0, 1, 2, 3, 4, -2, EQUIPMENT_BAG] {
        let count = match bag {
            0 => 16,
            -2 => 16,
            EQUIPMENT_BAG => 19,
            _ => 36,
        };
        for slot in 0..count {
            if slot_guid(store, bag, slot, objects) == Some(guid) {
                return Some((bag, u32::from(slot) + 1));
            }
        }
    }
    None
}

/// The item's send data and the arm the use dispatcher takes for it (`0x5d8d00`): never the
/// equip fork, which is `UseContainerItem`'s own.
fn resolved(ladder: &crate::spell::CastLadder, guid: u64) -> Option<(Clicked, UseFork)> {
    let obj = ladder.objects.object(guid)?;
    let entry = obj.object_entry()?;
    let t = ladder
        .items
        .template(entry, guid, crate::query_cache::Ask::New(&ladder.commands))?;
    let item_text_id = obj.item_text_id().unwrap_or(0);
    let fork = dispatch_fork(t, obj.item_flags().unwrap_or(0), item_text_id);
    let clicked = Clicked {
        guid,
        entry,
        item_text_id,
        start_quest: t.start_quest,
        spell_index: t.use_spell_index().unwrap_or(0),
        use_spell: t.use_spell.map(|u| u.spell_id),
        page_text: t.page_text,
        is_charter: t.flags & benilla_protocol::messages::ITEM_FLAG_CHARTER != 0,
    };
    Some((clicked, fork))
}

fn request(c: Clicked, bag: i64, slot: u32) -> Option<super::super::ItemUse> {
    let (bag_index, slot) = wire_pos(bag, slot)?;
    Some(super::super::ItemUse {
        guid: Some(c.guid),
        start_quest: c.start_quest,
        bag_index,
        slot,
        entry: c.entry,
        spell_index: c.spell_index,
        use_spell: c.use_spell,
        on_object: None,
        is_charter: c.is_charter,
    })
}

impl ScriptItemUse<'_, '_> {
    /// A frame action's item use at `unit` (the selection when `None`): the stock fork's arms, and
    /// the open's item for the loot window to open, as the stock use returns it.
    pub(crate) fn pfui_use_item(
        &mut self,
        script: &mut UiScript,
        guid: u64,
        unit: Option<u64>,
    ) -> Option<crate::ui_loot::OpenItem> {
        let ctx = match unit {
            Some(unit) => self.targeting.pfui_context(unit),
            None => Some(self.targeting.context()),
        }?;
        let (bag, slot) = self
            .self_q
            .iter()
            .next()
            .and_then(|store| position(&store.0, &self.ladder.objects, guid))?;
        // A locked item is not used, as the stock call refuses it.
        if script.item_locks().contains(&guid) {
            return None;
        }
        let (c, fork) = resolved(&self.ladder, guid)?;
        let it = request(c, bag, slot)?;
        match fork {
            UseFork::Unwrap => {
                let _ = self.ladder.commands.0.send(ClientCommand::OpenItem {
                    bag_index: it.bag_index,
                    slot: it.slot,
                });
                return None;
            }
            UseFork::WrapPaper => {
                script.arm_gift_wrap(bag, slot);
                return None;
            }
            // The reader, as the stock arms open it; a re-click closes it.
            UseFork::Read => {
                self.item_text.read_pages(guid);
                return None;
            }
            UseFork::Letter => {
                self.item_text.read_letter(guid, c.item_text_id);
                return None;
            }
            // The loot window opens it, as the stock open arm hands it over.
            UseFork::Open => {
                return Some(crate::ui_loot::OpenItem {
                    guid,
                    bag_index: it.bag_index,
                    slot: it.slot,
                });
            }
            UseFork::Quest | UseFork::Plain | UseFork::Equip => {}
        }
        let serial = self.gate.pfui_item_use.as_ref().map(|state| state.serial);
        super::super::send_item_use(
            it,
            &ctx,
            &mut self.ladder,
            script,
            &mut self.gate,
            false,
            &self.ui_errors,
        );
        let deferred = self.gate.pfui_item_use.as_ref().map(|state| state.serial) != serial;
        if deferred
            && self
                .gate
                .on_use
                .0
                .is_some_and(|pending| pending.guid == Some(guid))
        {
            if let Some(state) = self.gate.pfui_item_use.as_mut() {
                state.pending = Some(PendingTarget {
                    item: guid,
                    unit: ctx.selection_guid,
                    auto_self_cast: ctx.auto_self_cast,
                    session: script.session(),
                });
            }
        }
        None
    }
}

/// Returns true when this was an extension-owned target, including a refusal after disappearance.
pub(super) fn confirm(
    script: &mut UiScript,
    it: super::super::ItemUse,
    targeting: &CastTargeting,
    ladder: &mut crate::spell::CastLadder,
    gate: &mut crate::ui_bind_confirm::BindGate,
    errors: &crate::ui_action::UiErrorKeys,
) -> bool {
    let Some(pending) = gate
        .pfui_item_use
        .as_mut()
        .and_then(|state| state.pending.take())
    else {
        return false;
    };
    if it.guid != Some(pending.item) {
        return false;
    }
    if pending.session != script.session() {
        return true;
    }
    let mut ctx = match pending.unit {
        Some(unit) => {
            let Some(ctx) = targeting.pfui_context(unit) else {
                return true;
            };
            ctx
        }
        None => {
            let mut ctx = targeting.context();
            ctx.selection_guid = None;
            ctx.rel.target_store = None;
            ctx.rel.target_owner_store = None;
            ctx.rel.group.target_guid = None;
            ctx.range.target_pos = None;
            ctx.range.target = None;
            ctx
        }
    };
    ctx.auto_self_cast = pending.auto_self_cast;
    let Some((bag, slot)) = ctx
        .rel
        .self_store
        .and_then(|store| position(&store.0, &ladder.objects, pending.item))
    else {
        return true;
    };
    if script.item_locks().contains(&pending.item) {
        return true;
    }
    let Some((c, _)) = resolved(ladder, pending.item) else {
        return true;
    };
    let Some(current) = request(c, bag, slot) else {
        return true;
    };
    // Moving the same instance is valid; transforming it does not inherit the old consent.
    if current.entry != it.entry
        || current.use_spell != it.use_spell
        || current.spell_index != it.spell_index
        || current.start_quest != it.start_quest
    {
        return true;
    }
    super::super::send_item_use(current, &ctx, ladder, script, gate, true, errors);
    true
}

#[cfg(test)]
#[path = "item_use_tests.rs"]
mod tests;
