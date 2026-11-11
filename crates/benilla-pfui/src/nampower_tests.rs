use super::*;
use benilla_ui::script::{
    pfui_host::casts::{self, Cast},
    UiScript, UnitGuids,
};

fn fixture() -> UiScript {
    let mut script = UiScript::new().unwrap();
    script.keep_calls();
    crate::install(&script).unwrap();
    crate::spell::Catalog(Default::default())
        .install(script.lua())
        .unwrap();
    script.set_unit_guids(&UnitGuids {
        player: 3,
        ..Default::default()
    });
    script
        .run(
            r#"
      rows={}
      listener=CreateFrame('Frame')
      listener:SetScript('OnEvent',function(self,evt,...)
        table.insert(rows,{evt,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9})
      end)
    "#,
        )
        .unwrap();
    script
}

#[test]
fn identity_cvars_and_events_are_per_vm_and_settings_survive_reload() {
    let script = fixture();
    script
        .run(
            r#"
      local a,b,c=GetNampowerVersion();assert(a==3 and b==0 and c==0)
      assert(BenillaNampowerCompatibility.emulated and not BenillaNampowerCompatibility.fullNampower)
      assert(type(QueueSpellByName)=='function' and DisenchantAll==nil)
      assert(GetCVar('NP_EnableSpellGoEvents')=='0')
      assert(GetCVarDefault('NP_QueueTargetingSpells')=='1')
      assert(GetCVarDefault('NP_QueueOnSwingSpells')=='0')
      assert(GetCVarDefault('NP_TargetingQueueWindowMs')=='500')
      assert(GetCVarDefault('NP_OnSwingBufferCooldownMs')=='500')
      assert(GetCVarDefault('NP_QuickcastTargetingSpells')=='0')
      assert(GetCVarDefault('NP_QuickcastOnDoubleCast')=='0')
      assert(GetCVarDefault('NP_RetryServerRejectedSpells')=='1')
      listener:RegisterEvent('SPELL_GO_SELF')
    "#,
        )
        .unwrap();
    fire(
        script.lua(),
        Some("NP_EnableSpellGoEvents"),
        "SPELL_GO_SELF",
        vec![ScriptValue::Int(139)],
    );
    script
        .run("assert(#rows==0);SetCVar('np_enablespellgoevents',1)")
        .unwrap();
    fire(
        script.lua(),
        Some("NP_EnableSpellGoEvents"),
        "SPELL_GO_SELF",
        vec![ScriptValue::Int(139)],
    );
    script
        .run(
            r#"
      assert(#rows==1 and rows[1][2]==139)
      assert(GetCVarDefault('NP_EnableSpellGoEvents')=='0')
      assert(GetCVar('NP_EnableSpellGoEvents')=='1')
      assert(BenillaPfuiDB.nampower.np_enablespellgoevents=='1')
      SetCVar('NP_SpellQueueWindowMs',700.9);assert(GetCVar('NP_SpellQueueWindowMs')=='700')
      error=function() end
      assert(not pcall(function() SetCVar('NP_SpellQueueWindowMs',-1) end))
      assert(not pcall(function() SetCVar('NP_SpellQueueWindowMs',{}) end))
      SetCVar('NP_EnableSpellGoEvents',0);assert(BenillaPfuiDB.nampower.np_enablespellgoevents==nil)
      BenillaPfuiDB={nampower={np_enablespellgoevents='1'}}
      assert(GetCVar('NP_EnableSpellGoEvents')=='1')
    "#,
        )
        .unwrap();
    UiScript::new().unwrap().run("assert(GetNampowerVersion==nil and GetCastInfo==nil and BenillaNampowerCompatibility==nil)").unwrap();
}

/// Every setting NampowerSettings' menu reads at version 3.0 answers, as Nampower 4.6.1 registers
/// it: `GetCVar` raises on a name no one registered, which broke the menu's toggles.
#[test]
fn every_setting_the_settings_menu_reads_answers_its_default() {
    let script = fixture();
    script
        .run(
            r#"
      local defaults = {
        NP_InterruptChannelsOutsideQueueWindow='0', NP_OptimizeBufferUsingPacketTimings='0',
        NP_PreventRightClickTargetChange='0', NP_PreventRightClickPvPAttack='0',
        NP_DoubleCastToEndChannelEarly='0', NP_SpamProtectionEnabled='1',
        NP_EnableUnitEventsPet='1', NP_EnableUnitEventsParty='1', NP_EnableUnitEventsRaid='1',
        NP_EnableUnitEventsMouseover='1', NP_EnableUnitEventsGuid='1',
        NP_EnableUnitEventsGuidFiltering='0', NP_EnableSpellEnergizeEvents='0',
        NP_PreventMountingWhenBuffCapped='1', NP_MaxBufferIncreaseMs='30',
        NP_ChannelLatencyReductionPercentage='75', NP_NameplateDistance='20',
      }
      for name, value in pairs(defaults) do
        assert(GetCVar(name) == value and GetCVarDefault(name) == value, name)
      end
      SetCVar('NP_EnableUnitEventsGuid', '0'); assert(GetCVar('NP_EnableUnitEventsGuid') == '0')
      SetCVar('NP_NameplateDistance', 41); assert(GetCVar('NP_NameplateDistance') == '41')
    "#,
        )
        .unwrap();
}

#[test]
fn cancellation_reaches_extension_queues_without_an_active_stock_cast() {
    use benilla_ui::script::{pfui_host::attributes::ClickAction, ScriptCall};
    let mut script = fixture();
    script
        .run("SpellStopCasting();SpellStopTargeting()")
        .unwrap();
    assert_eq!(
        script.take_script_calls(),
        vec![
            ScriptCall::PfuiClick(ClickAction::CancelQueue { ground_only: false }),
            ScriptCall::PfuiClick(ClickAction::CancelQueue { ground_only: true }),
        ]
    );
}

#[test]
fn queued_scripts_are_ordered_requests_with_a_validated_priority() {
    use benilla_ui::script::{pfui_host::attributes::ClickAction, ScriptCall};
    let mut script = fixture();
    script
        .run(
            r#"
      QueueScript('x=1');QueueScript('x=2',2);QueueScript('x=3',3);QueueScript('')
      assert(x==nil)
      error=function() end
      assert(not pcall(function() QueueScript({},1) end))
      assert(not pcall(function() QueueScript('x=4',4) end))
      assert(not pcall(function() QueueScript('x=4',0) end))
    "#,
        )
        .unwrap();
    assert_eq!(
        script.take_script_calls(),
        (1..=3)
            .map(|priority| {
                ScriptCall::PfuiClick(ClickAction::QueueScript {
                    body: format!("x={priority}"),
                    priority,
                })
            })
            .collect::<Vec<_>>()
    );
    UiScript::new()
        .unwrap()
        .run("assert(QueueScript==nil)")
        .unwrap();
}

#[test]
fn queue_overloads_keep_each_calls_policy_book_rank_and_unit() {
    use benilla_ui::script::{
        pfui_host::attributes::{ClickAction, QueuePolicy},
        PetBookState, ScriptCall, SpellBookState, SpellSlotView,
    };
    let mut script = fixture();
    script.set_spellbook(SpellBookState {
        slots: vec![
            SpellSlotView {
                spell_id: 2050,
                name: "Lesser Heal".into(),
                rank: Some("Rank 1".into()),
                ..Default::default()
            },
            SpellSlotView {
                spell_id: 2052,
                name: "Lesser Heal".into(),
                rank: Some("Rank 2".into()),
                ..Default::default()
            },
            SpellSlotView {
                spell_id: 99,
                name: "Passive".into(),
                passive: true,
                ..Default::default()
            },
        ],
        ..Default::default()
    });
    script.set_pet_book(PetBookState {
        slots: vec![SpellSlotView {
            spell_id: 16827,
            name: "Claw".into(),
            ..Default::default()
        }],
        ..Default::default()
    });
    script
        .run(
            r#"
      QueueSpellByName('Lesser Heal(Rank 1)')
      CastSpellByNameNoQueue('Lesser Heal', 'player')
      CastSpellNoQueue(1, 0, 1)
      CastSpellNoQueue('SPELLID:2052', 'spell')
      CastSpellNoQueue('Lesser Heal(Rank 1)', 'spell')
      CastSpellNoQueue(1, 'pet', 'player')
      CastSpellByNameNoQueue('Claw')
      CastSpellByName('Lesser Heal', 1)
      CastSpell('spellId:2050', 'spell', 'player')
      CastSpell('Claw', 'pet', 'player')
      QueueSpellByName('not learned')
      CastSpellNoQueue('spellId:99999', 'spell')
      CastSpellNoQueue(3, 'spell')
      CastSpellByNameNoQueue('Lesser Heal', 'party4')
      assert(not pcall(function() CastSpellNoQueue(0, 'spell') end))
      assert(not pcall(function() CastSpellNoQueue('spellId:bad', 'spell') end))
    "#,
        )
        .unwrap();
    let wrapped = |call, policy| {
        ScriptCall::PfuiClick(ClickAction::Cast {
            call: Box::new(call),
            policy,
        })
    };
    assert_eq!(
        script.take_script_calls(),
        vec![
            wrapped(
                ScriptCall::CastSpell {
                    spell_id: 2050,
                    on_self: false,
                    may_cast: true
                },
                QueuePolicy::Force
            ),
            wrapped(
                ScriptCall::PfuiClick(ClickAction::Spell {
                    spell_id: 2052,
                    target: Some(3)
                }),
                QueuePolicy::Never
            ),
            wrapped(
                ScriptCall::CastSpell {
                    spell_id: 2050,
                    on_self: true,
                    may_cast: true
                },
                QueuePolicy::Never
            ),
            wrapped(
                ScriptCall::CastSpell {
                    spell_id: 2052,
                    on_self: false,
                    may_cast: true
                },
                QueuePolicy::Never
            ),
            wrapped(
                ScriptCall::CastSpell {
                    spell_id: 2050,
                    on_self: false,
                    may_cast: true
                },
                QueuePolicy::Never
            ),
            wrapped(
                ScriptCall::PfuiClick(ClickAction::PetSpell {
                    spell_id: 16827,
                    target: 3
                }),
                QueuePolicy::Never
            ),
            wrapped(
                ScriptCall::CastPetSpell {
                    spell_id: 16827,
                    on_self: false
                },
                QueuePolicy::Never
            ),
            ScriptCall::CastSpell {
                spell_id: 2052,
                on_self: true,
                may_cast: true
            },
            ScriptCall::PfuiClick(ClickAction::Spell {
                spell_id: 2050,
                target: Some(3)
            }),
            ScriptCall::PfuiClick(ClickAction::PetSpell {
                spell_id: 16827,
                target: 3
            }),
        ]
    );
    UiScript::new().unwrap().run("assert(QueueSpellByName==nil and CastSpellNoQueue==nil and CastSpellByNameNoQueue==nil)").unwrap();
}

#[test]
fn aura_edges_keep_raw_slots_hidden_indices_and_stack_decrease_state() {
    let script = fixture();
    script.run(r#"
      for _,evt in ipairs({'BUFF_ADDED_SELF','BUFF_REMOVED_SELF','DEBUFF_ADDED_OTHER','DEBUFF_REMOVED_OTHER'}) do listener:RegisterEvent(evt) end
    "#).unwrap();
    let a = Aura {
        slot: 0,
        spell: 139,
        stacks: 1,
        level: 60,
        visible: true,
    };
    auras(script.lua(), 3, vec![a]);
    auras(script.lua(), 3, vec![a]);
    let d = Aura {
        slot: 36,
        spell: 589,
        stacks: 3,
        level: 57,
        visible: true,
    };
    let h = Aura {
        slot: 32,
        spell: 9000,
        stacks: 1,
        level: 0,
        visible: false,
    };
    auras(script.lua(), 4, vec![d, h]);
    auras(script.lua(), 4, vec![Aura { stacks: 2, ..d }, h]);
    auras(script.lua(), 4, vec![h]);
    auras(script.lua(), 3, vec![]);
    script.run(r#"
      assert(#rows==6)
      assert(rows[1][1]=='BUFF_ADDED_SELF' and rows[1][2]=='0x0000000000000003')
      assert(rows[1][3]==1 and rows[1][4]==139 and rows[1][6]==60 and rows[1][7]==0 and rows[1][8]==0)
      assert(rows[2][4]==9000 and rows[2][3]==0 and rows[2][7]==32)
      assert(rows[3][4]==589 and rows[3][3]==1 and rows[3][5]==3 and rows[3][7]==36)
      assert(rows[4][1]=='DEBUFF_REMOVED_OTHER' and rows[4][5]==2 and rows[4][8]==2)
      assert(rows[5][8]==1 and rows[5][5]==0 and rows[5][6]==0 and rows[6][1]=='BUFF_REMOVED_SELF')
    "#).unwrap();
}

#[test]
fn cast_query_shares_gettime_epoch_and_does_not_confuse_long_cooldowns_with_gcd() {
    let mut script = fixture();
    script.tick(10.0);
    casts::set_casts(
        script.lua(),
        [(
            3,
            Cast {
                spell_id: 2050,
                name: "Lesser Heal".into(),
                rank: "Rank 1".into(),
                icon: None,
                start_ms: 9000.0,
                end_ms: 11500.0,
                tradeskill: false,
                cast_guid: "Cast-0003-42".into(),
                delay_ms: 0,
                channel: false,
            },
        )],
    );
    target(script.lua(), 3, 4);
    set_gcd_end(script.lua(), 10.5);
    script.run(r#"
      local c=GetCastInfo();assert(c.castId==42 and c.spellId==2050 and c.guid=='0x0000000000000004')
      assert(c.castStartS==9 and c.castEndS==11.5 and c.castRemainingMs==1500 and c.castDurationMs==2500)
      assert(c.gcdRemainingMs==500 and c.gcdEndS==10.5 and c.castType==0)
      assert(GetWowTimeMs()==10000)
    "#).unwrap();
    casts::set_casts(script.lua(), []);
    script.run("assert(GetCastInfo()==nil)").unwrap();
}

#[test]
fn death_edges_require_a_live_to_dead_transition_and_reset_on_session_end() {
    let script = fixture();
    script.run("listener:RegisterEvent('UNIT_DIED')").unwrap();
    health(script.lua(), 4, 0);
    health(script.lua(), 4, 100);
    health(script.lua(), 4, 0);
    health(script.lua(), 4, 0);
    script
        .run("assert(#rows==1 and rows[1][2]=='0x0000000000000004')")
        .unwrap();
    reset(script.lua());
    health(script.lua(), 4, 0);
    script.run("assert(#rows==1)").unwrap();
}

#[test]
fn named_unit_casts_bind_guid_and_unknown_units_never_fall_back_to_selection() {
    use benilla_ui::script::{
        pfui_host::attributes::ClickAction, ScriptCall, SpellBookState, SpellSlotView,
    };
    let mut script = fixture();
    script.set_unit_guids(&UnitGuids {
        player: 3,
        target: 4,
        ..Default::default()
    });
    script.set_spellbook(SpellBookState {
        slots: vec![SpellSlotView {
            spell_id: 2050,
            name: "Lesser Heal".into(),
            rank: Some("Rank 1".into()),
            ..Default::default()
        }],
        ..Default::default()
    });
    script.run("CastSpellByName('Lesser Heal(Rank 1)','player');CastSpellByName('Lesser Heal','party4')").unwrap();
    assert_eq!(
        script.take_script_calls(),
        vec![ScriptCall::PfuiClick(ClickAction::Spell {
            spell_id: 2050,
            target: Some(3)
        })]
    );
    script.run("CastSpellByName('Lesser Heal',1)").unwrap();
    assert_eq!(
        script.take_script_calls(),
        vec![ScriptCall::CastSpell {
            spell_id: 2050,
            on_self: true,
            may_cast: true
        }]
    );
}

#[test]
fn observed_sources_enrich_real_auras_and_server_duration_wins_over_dbc() {
    use benilla_ui::script::{pfui_host::auras as host, AuraState};
    let mut script = fixture();
    script.tick(10.0);
    let raw = Aura {
        slot: 1,
        spell: 139,
        stacks: 1,
        level: 60,
        visible: true,
    };
    let feed = |duration, expiration_time| {
        host::set_auras(
            script.lua(),
            3,
            Some(vec![host::Aura {
                slot: 1,
                source_guid: None,
                crowd_control: false,
                self_only: false,
                applies_aura: false,
                state: AuraState {
                    spell_id: 139,
                    name: Some("Renew".into()),
                    helpful: true,
                    count: 1,
                    duration,
                    expiration_time,
                    ..Default::default()
                },
            }]),
        )
    };
    feed(0.0, 0.0);
    application(script.lua(), 4, 3, 139, 15000);
    auras(script.lua(), 3, vec![raw]);
    script
        .run(
            r#"
      local a=C_UnitAuras.GetAuraDataBySlot('player',1)
      assert(a.sourceGUID=='0x0000000000000004' and a.duration==15 and a.expirationTime==25)
    "#,
        )
        .unwrap();
    feed(13.0, 24.0);
    auras(script.lua(), 3, vec![raw]);
    script.run("local a=C_UnitAuras.GetAuraDataBySlot('player',1);assert(a.duration==13 and a.expirationTime==24)").unwrap();
    update_duration(script.lua(), 1, 139, 13000);
    script.run("local id,left,expires=GetPlayerAuraDuration(1);assert(id==139 and left==13000 and expires==23000);assert(GetPlayerAuraDuration(48)==nil)").unwrap();
    auras(script.lua(), 3, vec![Aura { spell: 1243, ..raw }]);
    script.run("local id,left,expires=GetPlayerAuraDuration(1);assert(id==1243 and left==0 and expires==0)").unwrap();
    // A duration that arrives before the new descriptor binds once, not on each slot reuse.
    update_duration(script.lua(), 1, 0, 15000);
    auras(script.lua(), 3, vec![raw]);
    script
        .run("local id,left=GetPlayerAuraDuration(1);assert(id==139 and left==15000)")
        .unwrap();
    auras(script.lua(), 3, vec![]);
    feed(0.0, 0.0);
    auras(script.lua(), 3, vec![raw]);
    script
        .run("assert(C_UnitAuras.GetAuraDataBySlot('player',1).sourceGUID==nil);local id,left,expires=GetPlayerAuraDuration(1);assert(id==139 and left==0 and expires==0)")
        .unwrap();
}
