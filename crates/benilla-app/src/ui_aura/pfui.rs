//! Preserve raw aura slots for the optional modern projection without changing stock DTOs.

use benilla_formats::SpellCatalog;
use benilla_ui::script::{
    pfui_host::auras::{self, Aura},
    AuraState, UiScript,
};
use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct VisibleFeed;

pub(crate) fn register(app: &mut App) {
    app.add_systems(
        Update,
        feed_visible
            .after(crate::ui_unit::UnitFeed)
            .in_set(VisibleFeed)
            .in_set(crate::ui_script::UiFeed),
    );
}

/// Whether the VM was fed yet, and each visible unit's aura inputs as last fed.
type VisibleMemo = (bool, HashMap<u64, Vec<super::AuraInput>>);

/// GUID/focus/plate queries include visible units that stock target/group tokens cannot name.
fn feed_visible(
    script: Option<NonSend<UiScript>>,
    spells: Option<Res<crate::ui_action::Spells>>,
    stores: Query<
        (&crate::net::Guid, Ref<crate::net::ObjectStore>),
        Without<crate::net::SelfPlayer>,
    >,
    mut memo: Local<crate::ui_script::VmMemo<VisibleMemo>>,
) {
    let Some(script) = script else { return };
    // Cleared against a new VM, which is told every list once, unchanged or not.
    let (fed, memo) = memo.get(&script);
    let fresh = !std::mem::replace(fed, true);
    let catalog_changed = spells.as_ref().is_some_and(|spells| spells.is_changed());
    let catalog = spells.as_ref().map(|spells| &spells.catalog);
    let mut present = HashSet::new();
    for (guid, store) in &stores {
        if !store.is_unit() {
            continue;
        }
        present.insert(guid.0);
        if !fresh && !catalog_changed && !store.is_changed() && memo.contains_key(&guid.0) {
            continue;
        }
        let inputs: Vec<_> =
            super::other_unit_inputs(&store, catalog, super::RaidFilter::default(), None).collect();
        if fresh || catalog_changed || memo.get(&guid.0) != Some(&inputs) {
            let list = super::aura_list(catalog, &inputs);
            unit(&script, guid.0, &inputs, &list, catalog);
            memo.insert(guid.0, inputs);
        }
    }
    memo.retain(|guid, _| {
        if present.contains(guid) {
            true
        } else {
            if !auras::has_group_alias(script.lua(), *guid) {
                remove(&script, *guid);
            }
            false
        }
    });
}

pub(super) fn player(
    script: &UiScript,
    slots: &[super::CachedAura],
    list: &[AuraState],
    catalog: Option<&SpellCatalog>,
) {
    let guid = benilla_ui::script::pfui_host::units::player_guid(script.lua());
    auras::set_auras(
        script.lua(),
        guid,
        Some(
            slots
                .iter()
                .zip(list)
                .map(|(raw, state)| project(raw.slot, raw.flags, state, catalog))
                .collect(),
        ),
    );
}

pub(super) fn unit(
    script: &UiScript,
    guid: u64,
    slots: &[super::AuraInput],
    list: &[AuraState],
    catalog: Option<&SpellCatalog>,
) {
    auras::set_auras(
        script.lua(),
        guid,
        Some(
            slots
                .iter()
                .zip(list)
                .map(|(raw, state)| project(raw.slot, raw.flags, state, catalog))
                .collect(),
        ),
    );
}

fn project(slot: u8, flags: u8, state: &AuraState, catalog: Option<&SpellCatalog>) -> Aura {
    use benilla_protocol::messages::AURA_FLAG_EFF_INDEX_MASK;
    let mut state = state.clone();
    let mut crowd_control = false;
    let (mut self_only, mut applies_aura) = (false, false);
    if let Some(spell) = catalog.and_then(|catalog| catalog.get(state.spell_id)) {
        // The nibble names active effects in reverse bit order (effect 0 is bit 3).
        // Roster-only records carry no flags, so all DBC effects remain candidates there.
        let active = |i: usize| flags & AURA_FLAG_EFF_INDEX_MASK == 0 || flags & (8 >> i) != 0;
        let control = |aura| matches!(aura, 2 | 5 | 6 | 7 | 12 | 25 | 26 | 27 | 33 | 60 | 67 | 128);
        let mechanic =
            |mechanic| matches!(mechanic, 1..=3 | 5 | 7..=14 | 17 | 18 | 20 | 23 | 24 | 27 | 30);
        let control_effect = (0..3).any(|i| active(i) && control(spell.effect_apply_aura[i]));
        // The wire has no polarity bit. Keep negative-range and cancelable slots authoritative;
        // identify known overflow debuffs from AURA_IS_DEBUFF or intrinsically harmful effects.
        let harmful = spell.attributes & 0x0400_0000 != 0
            || control_effect
            || (0..3).any(|i| active(i) && matches!(spell.effect_apply_aura[i], 3 | 53 | 64 | 89));
        if state.helpful && !state.cancelable && harmful {
            state.helpful = false;
        }
        let on_self = |i: usize| {
            matches!(spell.effect_implicit_target_a[i], 0 | 1)
                && matches!(spell.effect_implicit_target_b[i], 0 | 1)
        };
        self_only = spell.effects.iter().any(|effect| *effect != 0)
            && (0..3).all(|i| spell.effects[i] == 0 || on_self(i));
        // Apply aura, persistent area aura, party area aura.
        applies_aura = spell
            .effects
            .iter()
            .any(|effect| matches!(effect, 6 | 27 | 35));
        crowd_control = !state.helpful
            && (control_effect
                || mechanic(spell.mechanic)
                || (0..3).any(|i| active(i) && mechanic(spell.effect_mechanic[i])));
    }
    Aura {
        slot,
        state,
        source_guid: None,
        crowd_control,
        self_only,
        applies_aura,
    }
}

pub(super) fn remove(script: &UiScript, guid: u64) {
    auras::set_auras(script.lua(), guid, None);
}

#[cfg(test)]
#[path = "pfui_tests.rs"]
mod tests;
