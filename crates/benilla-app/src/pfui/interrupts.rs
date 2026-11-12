//! The `Spell.dbc` facts behind ClassicAPI's `notInterruptible` (`pfui_host::interrupts`): the
//! interrupts and silences of the known spells, each aura's immunity effects, and what stops a
//! cast.

use benilla_formats::{SpellCatalog, SpellDisplay};
use benilla_ui::script::pfui_host::interrupts::{Immunity, Stopper};

const EFFECT_APPLY_AURA: u32 = 6;
const EFFECT_INTERRUPT_CAST: u32 = 68;
const EFFECT_TRIGGER_SPELL: u32 = 64;
const AURA_MOD_SILENCE: u32 = 27;
const PREVENTION_TYPE_SILENCE: u32 = 1;
const INTERRUPT_FLAG_DAMAGE: u32 = 0x02;
const CHANNEL_FLAG_INTERRUPT: u32 = 0x04;
const ATTR_UNAFFECTED_BY_INVULNERABILITY: u32 = 0x2000_0000;
const ATTR_EX_DISPEL_AURAS_ON_IMMUNITY: u32 = 0x0000_8000;
const ATTR_EX_IMMUNITY_TO_HOSTILE_AND_FRIENDLY: u32 = 0x0001_0000;
const ATTR_EX2_NO_SCHOOL_IMMUNITIES: u32 = 0x0400_0000;

/// `(silenceable, interruptible)` of a cast of `spell`, as `Cast` carries them.
pub(super) fn cast_facts(spell: &SpellDisplay, channel: bool) -> (bool, bool) {
    let silenceable = spell.prevention_type == PREVENTION_TYPE_SILENCE;
    let flag = if channel {
        spell.channel_interrupt_flags & CHANNEL_FLAG_INTERRUPT
    } else {
        spell.interrupt_flags & INTERRUPT_FLAG_DAMAGE
    };
    (silenceable, silenceable && flag != 0)
}

fn stopper(spell: &SpellDisplay, effect: usize, silence: bool) -> Stopper {
    Stopper {
        silence,
        school_mask: 1u32.checked_shl(spell.school).unwrap_or(0),
        mechanic: spell.mechanic,
        dispel: spell.dispel,
        effect: spell.effects[effect],
        effect_aura: spell.effect_apply_aura[effect],
        effect_mechanic: spell.effect_mechanic[effect],
        ignores_school_immunity: spell.attributes & ATTR_UNAFFECTED_BY_INVULNERABILITY != 0
            || spell.attributes_ex & ATTR_EX_DISPEL_AURAS_ON_IMMUNITY != 0
            || spell.attributes_ex2 & ATTR_EX2_NO_SCHOOL_IMMUNITIES != 0,
    }
}

fn silences(spell: &SpellDisplay, out: &mut Vec<Stopper>) {
    for i in 0..3 {
        if spell.effects[i] == EFFECT_APPLY_AURA && spell.effect_apply_aura[i] == AURA_MOD_SILENCE {
            out.push(stopper(spell, i, true));
        }
    }
}

/// The interrupts and silences among `known`; a triggered silence (Spell Lock) counts as the
/// triggered spell's.
pub(super) fn stoppers(
    known: impl IntoIterator<Item = u32>,
    catalog: &SpellCatalog,
) -> Vec<Stopper> {
    let mut out = Vec::new();
    for id in known {
        let Some(spell) = catalog.get(id).filter(|spell| !spell.passive) else {
            continue;
        };
        for i in 0..3 {
            match spell.effects[i] {
                EFFECT_INTERRUPT_CAST => out.push(stopper(spell, i, false)),
                EFFECT_TRIGGER_SPELL => {
                    if let Some(triggered) = catalog.get(spell.effect_trigger_spell[i]) {
                        silences(triggered, &mut out);
                    }
                }
                _ => {}
            }
        }
        silences(spell, &mut out);
    }
    let mut unique: Vec<Stopper> = Vec::new();
    for stopper in out {
        if !unique.contains(&stopper) {
            unique.push(stopper);
        }
    }
    unique
}

/// The immunity effects among an aura's `active` effects.
pub(crate) fn immunities(spell: &SpellDisplay, active: impl Fn(usize) -> bool) -> Vec<Immunity> {
    (0..3)
        .filter(|&i| {
            active(i) && matches!(spell.effect_apply_aura[i], 37 | 38 | 39 | 41 | 77 | 147)
        })
        .map(|i| Immunity {
            aura: spell.effect_apply_aura[i],
            misc: spell.effect_misc_value[i],
            both_polarities: spell.attributes_ex & ATTR_EX_IMMUNITY_TO_HOSTILE_AND_FRIENDLY != 0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The client's own `Spell.dbc`: the class interrupts and silences, the immunity auras the
    /// API reference names, and which casts each kind stops. Skips without an install.
    #[test]
    fn spell_data_gives_the_stoppers_immunities_and_cast_facts() {
        let Some(data) = benilla_formats::wow_data() else {
            return;
        };
        let mut chain = benilla_formats::open_chain(&data).unwrap();
        let catalog = benilla_formats::load_spell_catalog(&mut chain).unwrap();
        let kinds =
            |id| -> Vec<bool> { stoppers([id], &catalog).iter().map(|s| s.silence).collect() };
        // Kick, Counterspell: an interrupt; Silence: a silence; Spell Lock: both, the silence
        // through the spell it triggers.
        assert_eq!(kinds(1766), [false]);
        assert_eq!(kinds(2139), [false]);
        assert_eq!(kinds(15487), [true]);
        assert_eq!(kinds(19244), [false, true]);
        assert_eq!(stoppers([1766], &catalog)[0].school_mask, 1);
        let all = |id: u32| {
            let spell = catalog.get(id).unwrap();
            immunities(spell, |_| true)
        };
        // Divine Shield and Ice Block: helpful school immunities over every school; Banish: one
        // that blocks hostile spells too.
        let schools = |list: &[benilla_ui::script::pfui_host::interrupts::Immunity]| {
            list.iter()
                .filter(|i| i.aura == 39)
                .fold(0, |mask, i| mask | i.misc)
        };
        assert_eq!(schools(&all(642)), 127);
        assert_eq!(schools(&all(11958)), 127);
        let banish = all(710);
        assert!(schools(&banish) == 127 && banish.iter().all(|i| i.both_polarities));
        let facts = |id: u32, channel| cast_facts(catalog.get(id).unwrap(), channel);
        // Greater Heal and Arcane Missiles: both stop them; First Aid: neither.
        assert_eq!(facts(2060, false), (true, true));
        assert_eq!(facts(5143, true), (true, true));
        assert_eq!(facts(746, true), (false, false));
    }
}
