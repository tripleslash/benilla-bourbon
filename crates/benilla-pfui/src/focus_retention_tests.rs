use benilla_ui::script::{pfui_host::units, UiScript, UnitGuids, UnitState};

fn vm(memory: &units::FocusMemory) -> UiScript {
    let mut script = UiScript::new().unwrap();
    script.keep_calls();
    units::retain_focus(script.lua(), memory.clone());
    crate::install(&script).unwrap();
    script.set_unit_guids(&UnitGuids {
        player: 3,
        ..Default::default()
    });
    script.set_unit("player", Some(player()));
    script
}

fn player() -> UnitState {
    UnitState {
        guid: 3,
        exists: true,
        name: Some("Tester".into()),
        health: 42,
        ..Default::default()
    }
}

#[test]
fn focus_survives_vm_replacement_and_announces_restore_after_a_live_snapshot() {
    let memory = units::FocusMemory::default();
    let first = vm(&memory);
    first
        .run("FocusUnit('player');assert(UnitGUID('focus')==UnitGUID('player'))")
        .unwrap();
    drop(first);
    let second = vm(&memory);
    second.run("focusChanges=0;f=CreateFrame('Frame');f:RegisterEvent('PLAYER_FOCUS_CHANGED');f:SetScript('OnEvent',function() focusChanges=focusChanges+1;if UnitGUID('focus') then assert(UnitName('focus')=='Tester' and UnitHealth('focus')==42) end end)").unwrap();
    units::set_unit_states(second.lua(), []);
    units::announce_restored_focus(second.lua());
    second
        .run("assert(focusChanges==0 and UnitGUID('focus')=='0x0000000000000003')")
        .unwrap();
    units::set_unit_states(second.lua(), [(3, player(), 0)]);
    units::set_unit_states(second.lua(), [(3, player(), 0)]);
    second.run("assert(focusChanges==0)").unwrap();
    units::announce_restored_focus(second.lua());
    units::announce_restored_focus(second.lua());
    second
        .run("assert(focusChanges==1 and UnitExists('focus'))")
        .unwrap();
    second.run("ClearFocus()").unwrap();
    let third = vm(&memory);
    third.run("assert(UnitGUID('focus')==nil)").unwrap();
}

#[test]
fn session_reset_and_independent_sessions_do_not_inherit_focus() {
    let memory = units::FocusMemory::default();
    let first = vm(&memory);
    first.run("FocusUnit('player')").unwrap();
    vm(&units::FocusMemory::default())
        .run("assert(UnitGUID('focus')==nil)")
        .unwrap();
    memory.clear();
    units::clear_focus(first.lua());
    first.run("assert(UnitGUID('focus')==nil)").unwrap();
    vm(&memory).run("assert(UnitGUID('focus')==nil)").unwrap();
    UiScript::new()
        .unwrap()
        .run("assert(FocusUnit==nil)")
        .unwrap();
}
