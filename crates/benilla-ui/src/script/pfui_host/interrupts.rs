//! ClassicAPI's `notInterruptible`: whether none of the interrupts and silences the player knows
//! can stop a cast, by the spell's own data and the caster's immunity auras, as ClassicAPI
//! decides it: an interrupt or a silence stops a cast whose `PreventionType` is silence
//! (`Spell::EffectInterruptCast`, `Aura::HandleAuraModSilence`), unless an immunity aura of the
//! caster blocks it (`Unit::IsImmuneToSpell`, `Unit::IsImmuneToSpellEffect`). The server's
//! interrupt effect also tests the cast's interrupt flags, but in its own spell table, which a
//! server can patch apart from the client's `Spell.dbc` (ClassicAPI a16b990: stock 1.12's
//! Healing Ward lacks the bit that Turtle's and the server's carry), so the flags are not read.
//! A creature's own immunities are server data the client never sees, so such a cast reads
//! interruptible.

use super::auras::Aura;
use super::casts::Cast;
use mlua::Lua;

/// One known way to stop a cast: a spell's interrupt effect or silence aura, and the facts of
/// that spell an immunity is checked against.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stopper {
    /// `true` for a silence aura (`SPELL_AURA_MOD_SILENCE`), `false` for an interrupt effect.
    pub silence: bool,
    /// `1 << School`.
    pub school_mask: u32,
    pub mechanic: u32,
    pub dispel: u32,
    /// The effect's own type, applied aura and mechanic.
    pub effect: u32,
    pub effect_aura: u32,
    pub effect_mechanic: u32,
    /// The spell passes school immunities (`SPELL_ATTR_UNAFFECTED_BY_INVULNERABILITY`,
    /// `SPELL_ATTR_EX_DISPEL_AURAS_ON_IMMUNITY`, `SPELL_ATTR_EX2_NO_SCHOOL_IMMUNITIES`).
    pub ignores_school_immunity: bool,
}

/// One immunity effect of an aura: its aura type (37 effect, 38 state, 39 school, 41 dispel,
/// 77 mechanic, 147 mechanic mask) and misc value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Immunity {
    pub aura: u32,
    pub misc: i32,
    /// `SPELL_ATTR_EX_IMMUNITY_TO_HOSTILE_AND_FRIENDLY_EFFECTS`: it blocks a hostile spell even
    /// from a hostile aura (Banish).
    pub both_polarities: bool,
}

#[derive(Default)]
struct Stoppers(Vec<Stopper>);

/// Replace the ways the player knows to stop a cast.
pub fn set_stoppers(lua: &Lua, stoppers: Vec<Stopper>) {
    lua.set_app_data(Stoppers(stoppers));
}

fn blocks(immunity: &Immunity, helpful: bool, stopper: &Stopper) -> bool {
    // The server compares the polarities of the immunity's spell and the stopper, which is
    // always hostile; the aura's slot polarity stands for the first.
    if !(helpful || immunity.both_polarities) {
        return false;
    }
    let misc = immunity.misc as u32;
    let mechanic_bit = |mechanic: u32| mechanic != 0 && misc & (1 << (mechanic - 1)) != 0;
    match immunity.aura {
        39 => !stopper.ignores_school_immunity && misc & stopper.school_mask != 0,
        41 => misc == stopper.dispel,
        77 => {
            (stopper.mechanic != 0 && misc == stopper.mechanic)
                || (stopper.effect_mechanic != 0 && misc == stopper.effect_mechanic)
        }
        147 => mechanic_bit(stopper.mechanic) || mechanic_bit(stopper.effect_mechanic),
        37 => misc == stopper.effect,
        38 => stopper.effect_aura != 0 && misc == stopper.effect_aura,
        _ => false,
    }
}

/// Whether `stopper` stops `cast` on a caster wearing `auras`.
pub fn stops(stopper: &Stopper, cast: &Cast, auras: &[Aura]) -> bool {
    cast.silenceable
        && !auras.iter().any(|aura| {
            aura.immunities
                .iter()
                .any(|immunity| blocks(immunity, aura.state.helpful, stopper))
        })
}

/// `notInterruptible` of `cast` on `caster`: `false` when the player knows no way to stop a cast.
pub fn not_interruptible(lua: &Lua, caster: u64, cast: &Cast) -> bool {
    let Some(stoppers) = lua.app_data_ref::<Stoppers>() else {
        return false;
    };
    if stoppers.0.is_empty() {
        return false;
    }
    let auras = super::auras::of_guid(lua, caster);
    !stoppers
        .0
        .iter()
        .any(|stopper| stops(stopper, cast, &auras))
}
