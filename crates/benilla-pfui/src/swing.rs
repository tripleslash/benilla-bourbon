//! The player's swing clock, for ClassicAPI's `PLAYER_SWING(swingDuration, swingType)`.
//!
//! The server keeps one attack timer per weapon and never sends it. It is rebuilt here from the
//! packets that mark each reset: the player's own white hit, an on-next-swing ability or a
//! swing-resetting cast going off, an auto-repeat shot, and an attack the player parried. Every
//! reset only marks its weapon; [`flush`] then fires at most one event per weapon with the time
//! left, so several hits in one frame, an extra attack among them, announce one swing.
//!
//! The server's rules that no packet carries are vmangos's: a cast resets the melee timers when
//! it is not triggered and `InterruptFlags` has the auto-attack bit, and a parry cuts the
//! victim's nearer swing by 40% of the weapon speed, never below 20% of it. A realm with other
//! rules will drift until the next white hit. Not modelled, because nothing the client receives
//! marks them: the off-hand delay when an attack starts and the reset after a weapon swap.

use benilla_ui::script::{pfui_host, ScriptValue};
use mlua::Lua;
use std::collections::HashMap;

/// `Enum.PlayerSwingType`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Swing {
    MainHand = 0,
    OffHand = 1,
    Ranged = 2,
}

/// The weapon speeds in seconds, as the unit fields give them; 0 or `None` is no weapon.
#[derive(Clone, Copy, Debug, Default)]
pub struct Speeds {
    pub main: f64,
    pub off: Option<f64>,
    pub ranged: f64,
}

impl Speeds {
    fn of(&self, swing: Swing) -> f64 {
        match swing {
            Swing::MainHand => self.main,
            Swing::OffHand => self.off.unwrap_or(0.0),
            Swing::Ranged => self.ranged,
        }
    }
}

/// How a spell moves the clock when it goes off.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// `Attributes` `0x4`/`0x400`: it takes the main hand's next swing (Heroic Strike, Maul).
    NextSwing,
    /// `AttributesEx2` `0x20`: Auto Shot and a wand's Shoot.
    AutoRepeat,
    /// `InterruptFlags` `0x8`: a cast that restarts the melee timers.
    ResetsMelee,
}

impl Kind {
    /// Classify a `Spell.dbc` row; `None` leaves the clock alone.
    pub fn of(attributes: u32, attributes_ex2: u32, interrupt_flags: u32) -> Option<Kind> {
        if attributes & 0x404 != 0 {
            Some(Kind::NextSwing)
        } else if attributes_ex2 & 0x20 != 0 {
            Some(Kind::AutoRepeat)
        } else if interrupt_flags & 0x8 != 0 {
            Some(Kind::ResetsMelee)
        } else {
            None
        }
    }
}

#[derive(Default)]
pub struct Clock {
    /// When each weapon's next swing is due, in `GetTime` seconds.
    end: [Option<f64>; 3],
    dirty: [bool; 3],
    /// The cast the server announced with `SMSG_SPELL_START`, which it sends for no triggered
    /// spell: the proof that the matching `SMSG_SPELL_GO` is the player's own cast.
    started: Option<u32>,
    attacking: bool,
}

impl Clock {
    fn reset(&mut self, swing: Swing, now: f64, duration: f64) {
        // No weapon in this hand: nothing swings.
        if duration <= 0.0 {
            return;
        }
        self.end[swing as usize] = Some(now + duration);
        self.dirty[swing as usize] = true;
    }

    fn remaining(&self, swing: Swing, now: f64) -> Option<f64> {
        self.end[swing as usize]
            .map(|end| end - now)
            .filter(|left| *left > 0.0)
    }

    /// The player's white hit, landed or not: `SMSG_ATTACKERSTATEUPDATE` with the player attacking.
    pub fn swung(&mut self, now: f64, off_hand: bool, speeds: &Speeds) {
        let swing = if off_hand {
            Swing::OffHand
        } else {
            Swing::MainHand
        };
        self.reset(swing, now, speeds.of(swing));
    }

    pub fn attacking(&mut self, attacking: bool) {
        self.attacking = attacking;
    }

    /// `SMSG_SPELL_START` for the player.
    pub fn cast_started(&mut self, spell: u32) {
        self.started = Some(spell);
    }

    /// `SMSG_SPELL_GO` for the player.
    pub fn cast_went(&mut self, now: f64, spell: u32, kind: Option<Kind>, speeds: &Speeds) {
        let announced = self.started.take_if(|started| *started == spell).is_some();
        match kind {
            Some(Kind::NextSwing) => self.reset(Swing::MainHand, now, speeds.main),
            Some(Kind::AutoRepeat) => self.reset(Swing::Ranged, now, speeds.ranged),
            // A proc carries the same flags and resets nothing; and out of a fight there is no
            // swing to show.
            Some(Kind::ResetsMelee) if announced && self.attacking => {
                self.reset(Swing::MainHand, now, speeds.main);
                self.reset(Swing::OffHand, now, speeds.of(Swing::OffHand));
            }
            _ => (),
        }
    }

    /// The player parried an attack: the nearer melee swing comes sooner.
    pub fn parried(&mut self, now: f64, speeds: &Speeds) {
        let main = self.remaining(Swing::MainHand, now);
        let off = self.remaining(Swing::OffHand, now);
        let (swing, left) = match (main, off) {
            (Some(main), Some(off)) if off < main => (Swing::OffHand, off),
            (Some(main), _) => (Swing::MainHand, main),
            (None, Some(off)) => (Swing::OffHand, off),
            (None, None) => return,
        };
        let fifth = speeds.of(swing) * 0.2;
        if fifth <= 0.0 || left <= fifth {
            return;
        }
        let hastened = if left > fifth * 3.0 {
            left - fifth * 2.0
        } else {
            fifth
        };
        self.end[swing as usize] = Some(now + hastened);
        self.dirty[swing as usize] = true;
    }

    /// Each weapon reset since the last call, once, with the seconds until its next swing.
    pub fn take(&mut self, now: f64) -> Vec<(f64, Swing)> {
        let mut out = Vec::new();
        for swing in [Swing::MainHand, Swing::OffHand, Swing::Ranged] {
            if std::mem::take(&mut self.dirty[swing as usize]) {
                if let Some(left) = self.remaining(swing, now) {
                    out.push((left, swing));
                }
            }
        }
        out
    }
}

#[derive(Default)]
struct State {
    clock: Clock,
    kinds: HashMap<u32, Kind>,
}

/// Give the VM the spells that move the clock; the spell catalog calls this as it installs.
pub fn install_kinds(lua: &Lua, kinds: impl IntoIterator<Item = (u32, Kind)>) {
    let mut state = lua.remove_app_data::<State>().unwrap_or_default();
    state.kinds = kinds.into_iter().collect();
    lua.set_app_data(state);
}

fn now(lua: &Lua) -> f64 {
    lua.globals()
        .get::<mlua::Function>("GetTime")
        .and_then(|f| f.call::<f64>(()))
        .unwrap_or(0.0)
}

/// The player's weapon speeds off the stock stat queries.
fn speeds(lua: &Lua) -> Speeds {
    let call = |name: &str| {
        lua.globals()
            .get::<mlua::Function>(name)
            .and_then(|f| f.call::<(Option<f64>, Option<f64>)>("player"))
            .unwrap_or_default()
    };
    let (main, off) = call("UnitAttackSpeed");
    let (ranged, _) = call("UnitRangedDamage");
    Speeds {
        main: main.unwrap_or(0.0),
        off,
        ranged: ranged.unwrap_or(0.0),
    }
}

fn with(lua: &Lua, change: impl FnOnce(&mut State, f64, &Speeds)) {
    let (now, speeds) = (now(lua), speeds(lua));
    if let Some(mut state) = lua.app_data_mut::<State>() {
        change(&mut state, now, &speeds);
    }
}

pub fn swung(lua: &Lua, off_hand: bool) {
    with(lua, |s, now, speeds| s.clock.swung(now, off_hand, speeds));
}

pub fn parried(lua: &Lua) {
    with(lua, |s, now, speeds| s.clock.parried(now, speeds));
}

pub fn attacking(lua: &Lua, attacking: bool) {
    with(lua, |s, _, _| s.clock.attacking(attacking));
}

pub fn cast_started(lua: &Lua, spell: u32) {
    with(lua, |s, _, _| s.clock.cast_started(spell));
}

pub fn cast_went(lua: &Lua, spell: u32) {
    with(lua, |s, now, speeds| {
        let kind = s.kinds.get(&spell).copied();
        s.clock.cast_went(now, spell, kind, speeds);
    });
}

/// Forget the clock: a logout or a lost connection.
pub fn clear(lua: &Lua) {
    with(lua, |s, _, _| s.clock = Clock::default());
}

/// Fire `PLAYER_SWING` for each weapon reset since the last call.
pub fn flush(lua: &Lua) {
    let now = now(lua);
    let Some(resets) = lua.app_data_mut::<State>().map(|mut s| s.clock.take(now)) else {
        return;
    };
    for (left, swing) in resets {
        pfui_host::fire_pfui_event(
            lua,
            "PLAYER_SWING",
            vec![ScriptValue::Number(left), ScriptValue::Int(swing as i64)],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUAL: Speeds = Speeds {
        main: 2.0,
        off: Some(1.5),
        ranged: 3.0,
    };

    #[test]
    fn a_white_hit_resets_its_own_hand_and_a_burst_announces_once() {
        let mut clock = Clock::default();
        clock.swung(10.0, false, &DUAL);
        clock.swung(10.0, false, &DUAL);
        clock.swung(10.0, true, &DUAL);
        assert_eq!(
            clock.take(10.0),
            vec![(2.0, Swing::MainHand), (1.5, Swing::OffHand)]
        );
        assert!(clock.take(10.0).is_empty(), "nothing new to say");
    }

    #[test]
    fn an_empty_hand_swings_nothing() {
        let mut clock = Clock::default();
        let one_hand = Speeds {
            main: 2.0,
            off: None,
            ranged: 0.0,
        };
        clock.swung(1.0, true, &one_hand);
        clock.cast_went(1.0, 75, Some(Kind::AutoRepeat), &one_hand);
        assert!(clock.take(1.0).is_empty());
    }

    #[test]
    fn spells_move_the_clock_by_their_kind() {
        assert_eq!(Kind::of(0x4, 0, 0x8), Some(Kind::NextSwing));
        assert_eq!(Kind::of(0x400, 0, 0), Some(Kind::NextSwing));
        assert_eq!(Kind::of(0, 0x20, 0), Some(Kind::AutoRepeat));
        assert_eq!(Kind::of(0, 0, 0xf), Some(Kind::ResetsMelee));
        assert_eq!(Kind::of(0x40, 0x800, 0x7), None);

        let mut clock = Clock::default();
        clock.cast_went(5.0, 78, Some(Kind::NextSwing), &DUAL);
        clock.cast_went(5.0, 75, Some(Kind::AutoRepeat), &DUAL);
        clock.cast_went(5.0, 1, None, &DUAL);
        assert_eq!(
            clock.take(5.0),
            vec![(2.0, Swing::MainHand), (3.0, Swing::Ranged)]
        );
    }

    #[test]
    fn a_cast_resets_melee_only_when_announced_and_fighting() {
        let mut clock = Clock::default();
        let reset = Some(Kind::ResetsMelee);
        // A triggered spell goes off with no start.
        clock.attacking(true);
        clock.cast_went(1.0, 133, reset, &DUAL);
        assert!(clock.take(1.0).is_empty());
        // The start of another spell does not vouch for this one.
        clock.cast_started(116);
        clock.cast_went(1.0, 133, reset, &DUAL);
        assert!(clock.take(1.0).is_empty());
        // Out of a fight there is no swing to restart.
        clock.attacking(false);
        clock.cast_started(133);
        clock.cast_went(1.0, 133, reset, &DUAL);
        assert!(clock.take(1.0).is_empty());

        clock.attacking(true);
        clock.cast_started(133);
        clock.cast_went(4.0, 133, reset, &DUAL);
        assert_eq!(
            clock.take(4.0),
            vec![(2.0, Swing::MainHand), (1.5, Swing::OffHand)]
        );
        // One start vouches once.
        clock.cast_went(5.0, 133, reset, &DUAL);
        assert!(clock.take(5.0).is_empty());
    }

    #[test]
    fn a_parry_hastens_the_nearer_swing_within_the_servers_bounds() {
        let slow = Speeds {
            main: 3.0,
            off: None,
            ranged: 0.0,
        };
        let left = |since: f64| {
            let mut clock = Clock::default();
            clock.swung(0.0, false, &slow);
            clock.take(0.0);
            clock.parried(since, &slow);
            clock.take(since)
        };
        // 3.0 left of 3.0: more than 60%, so 40% of the speed comes off.
        let near = |got: Vec<(f64, Swing)>, want: f64, swing: Swing| {
            assert_eq!(got.len(), 1);
            assert!((got[0].0 - want).abs() < 1e-9, "{got:?} vs {want}");
            assert_eq!(got[0].1, swing);
        };
        near(left(0.0), 1.8, Swing::MainHand);
        // 1.5 left: between 20% and 60%, so it drops to 20%.
        near(left(1.5), 0.6, Swing::MainHand);
        // 0.5 left: already inside the last 20%, untouched and unannounced.
        assert!(left(2.5).is_empty());

        // With two weapons the one due sooner is the one hastened.
        let mut clock = Clock::default();
        clock.swung(0.0, false, &DUAL);
        clock.swung(0.0, true, &DUAL);
        clock.take(0.0);
        clock.parried(0.0, &DUAL);
        near(clock.take(0.0), 0.9, Swing::OffHand);
        // No swing running: nothing to hasten.
        let mut idle = Clock::default();
        idle.parried(0.0, &DUAL);
        assert!(idle.take(0.0).is_empty());
    }

    /// The VM path: speeds off the stock stat queries, spell kinds off the installed catalog, and
    /// one event per weapon with ClassicAPI's payload.
    #[test]
    fn the_vm_announces_each_reset_with_the_duration_and_the_swing_type() {
        use benilla_ui::script::UiScript;

        let mut script = UiScript::new().unwrap();

        script.keep_calls();
        crate::install(&script).unwrap();
        install_kinds(
            script.lua(),
            [(78, Kind::NextSwing), (75, Kind::AutoRepeat)],
        );
        script
            .run(
                r#"
            -- The stock stat queries read the lent host; this VM has none, so they answer here.
            function UnitAttackSpeed() return 2.9, 1.5 end
            function UnitRangedDamage() return 2.8 end
            seen = {}
            local frame = CreateFrame("Frame")
            frame:RegisterEvent("PLAYER_SWING")
            frame:SetScript("OnEvent", function(_, name, duration, kind)
              seen[#seen + 1] = string.format("%.1f:%d", duration, kind)
            end)
            function drain() local text = table.concat(seen, "|") seen = {} return text end
        "#,
            )
            .unwrap();
        let lua = script.lua();
        let drain = || script.eval::<String>("return drain()").unwrap();

        swung(lua, false);
        swung(lua, true);
        swung(lua, true);
        flush(lua);
        assert_eq!(drain(), "2.9:0|1.5:1");
        flush(lua);
        assert_eq!(drain(), "", "a reset is announced once");

        cast_went(lua, 78);
        cast_went(lua, 75);
        cast_went(lua, 133);
        flush(lua);
        assert_eq!(drain(), "2.9:0|2.8:2");
        assert_eq!(
            script
                .eval::<i64>("return Enum.PlayerSwingType.Ranged")
                .unwrap(),
            Swing::Ranged as i64
        );

        swung(lua, false);
        clear(lua);
        flush(lua);
        assert_eq!(drain(), "", "a logout forgets the clock");
    }

    /// The three `Spell.dbc` columns against spells whose behaviour is known; skipped without an
    /// install. It needs the audit feature's data reader.
    #[cfg(feature = "audit")]
    #[test]
    fn the_installs_spell_table_classifies_the_known_spells() {
        let Some(data) = benilla_formats::wow_data() else {
            return;
        };
        let Ok(chain) = benilla_formats::open_chain(&data) else {
            return;
        };
        let catalog = crate::spell::Catalog::read(&|path| chain.read(path).ok(), "enUS")
            .expect("the spell table");
        let kind = |id: u32| catalog.0.get(&id).and_then(|spell| spell.swing);
        assert_eq!(kind(78), Some(Kind::NextSwing), "Heroic Strike");
        assert_eq!(kind(845), Some(Kind::NextSwing), "Cleave");
        assert_eq!(kind(75), Some(Kind::AutoRepeat), "Auto Shot");
        assert_eq!(kind(5019), Some(Kind::AutoRepeat), "Shoot (wand)");
        assert_eq!(kind(133), Some(Kind::ResetsMelee), "Fireball");
        assert_eq!(kind(7386), None, "Sunder Armor");
        assert_eq!(kind(6673), None, "Battle Shout");
    }

    #[test]
    fn a_reset_reports_the_time_left_when_it_is_announced() {
        let mut clock = Clock::default();
        clock.swung(10.0, false, &DUAL);
        assert_eq!(clock.take(10.5), vec![(1.5, Swing::MainHand)]);
        clock.swung(20.0, false, &DUAL);
        assert!(
            clock.take(23.0).is_empty(),
            "a swing already past is no news"
        );
    }
}
