use super::*;
use crate::items::Items;
use crate::pfui::test_support::{fixture as spell_fixture, run, with_unit_targets};
use benilla_protocol::messages::{ItemSpellEntry, ItemUseSpell, UseItemTarget};
use benilla_ui::script::{pfui_host::items, ContainerSlot, ContainerState};

const ITEM: u64 = 0x4000_0000_0000_0001;
const OTHER_ITEM: u64 = 0x4000_0000_0000_0002;
const ENTRY: u32 = 12345;

fn fixture(bind: bool) -> (App, crossbeam_channel::Receiver<ClientCommand>) {
    let (mut app, rx) = spell_fixture();
    with_unit_targets(&mut app);
    let world = app.world_mut();
    for guid in [ITEM, OTHER_ITEM] {
        crate::items::test_spawn_item(
            world,
            guid,
            benilla_protocol::ObjectFields::from_pairs(&[(3, ENTRY), (14, 1)]),
            false,
        );
    }
    let mut template = crate::items::test_template("Test item");
    template.bonding = if bind { 3 } else { 0 };
    template.use_spell = Some(ItemUseSpell {
        spell_id: 2050,
        cooldown_ms: -1,
        category: 0,
        category_cooldown_ms: -1,
    });
    template.spells = vec![ItemSpellEntry {
        index: 1,
        spell_id: 2050,
        trigger: 0,
        charges: 0,
        cooldown_ms: -1,
        category: 0,
        category_cooldown_ms: -1,
    }];
    world
        .resource_mut::<Items>()
        .insert_template(ENTRY, Some(template));
    set_slots(&mut app, ITEM, OTHER_ITEM);
    let mut script = app.world_mut().non_send_mut::<UiScript>();
    script.set_container(
        0,
        Some(ContainerState {
            num_slots: 16,
            slots: [
                (
                    1,
                    ContainerSlot {
                        item_id: ENTRY,
                        count: 1,
                        guid: ITEM,
                        ..Default::default()
                    },
                ),
                (
                    2,
                    ContainerSlot {
                        item_id: ENTRY,
                        count: 1,
                        guid: OTHER_ITEM,
                        ..Default::default()
                    },
                ),
            ]
            .into(),
            ..Default::default()
        }),
    );
    items::set_instances(
        script.lua(),
        [
            (
                (0, 1),
                items::Instance {
                    guid: ITEM,
                    item_id: ENTRY,
                    name: Some("Test item".into()),
                    count: 1,
                    uses: 0,
                },
            ),
            (
                (0, 2),
                items::Instance {
                    guid: OTHER_ITEM,
                    item_id: ENTRY,
                    name: Some("Test item".into()),
                    count: 1,
                    uses: 0,
                },
            ),
        ],
    );
    (app, rx)
}

fn set_slots(app: &mut App, first: u64, second: u64) {
    let world = app.world_mut();
    let me = world.resource::<crate::net::GuidIndex>().0[&3];
    world
        .entity_mut(me)
        .insert(ObjectStore(benilla_protocol::ObjectFields::from_pairs(&[
            (2, 0x19),
            (22, 100),
            (28, 100),
            (16, 4),
            (532, first as u32),
            (533, (first >> 32) as u32),
            (534, second as u32),
            (535, (second >> 32) as u32),
        ])));
}

fn uses(rx: &crossbeam_channel::Receiver<ClientCommand>) -> Vec<(u8, u8, u8, UseItemTarget)> {
    rx.try_iter()
        .filter_map(|c| match c {
            ClientCommand::UseItem {
                bag_index,
                slot,
                spell_index,
                target,
            } => Some((bag_index, slot, spell_index, target)),
            ClientCommand::SellItem { .. }
            | ClientCommand::AutoBankItem { .. }
            | ClientCommand::AutoEquipItem { .. } => {
                panic!("direct API took a bag-click mode: {c:?}")
            }
            _ => None,
        })
        .collect()
}

/// The confirm is a call, applied inside the verb.
fn confirm(app: &mut App) {
    run(app, "ConfirmBindOnUse()");
}

#[test]
fn item_uses_follow_ordered_target_changes_without_reselecting() {
    for (lua, target, selected) in [
        (
            "C_Item.UseItemByName(12345,'target');TargetUnit('player')",
            4,
            3,
        ),
        (
            "TargetUnit('player');C_Item.UseItemByName(12345,'target')",
            3,
            3,
        ),
        (
            "TargetUnit('player');C_Item.UseItemByName(12345,'targettarget')",
            4,
            3,
        ),
        ("C_Item.UseItemByName(12345,'player')", 3, 4),
    ] {
        let (mut app, rx) = fixture(false);
        run(&mut app, lua);
        assert_eq!(
            uses(&rx),
            vec![(255, 23, 1, UseItemTarget::Unit(target))],
            "{lua}"
        );
        assert_eq!(
            app.world().resource::<crate::target::Selection>().guid,
            Some(selected)
        );
    }
}

#[test]
fn direct_item_api_bypasses_vendor_bank_and_equip_modes() {
    for mode in [0, 1, 2] {
        let (mut app, rx) = fixture(false);
        match mode {
            0 => app
                .world_mut()
                .resource_mut::<crate::ui_merchant::MerchantOpen>()
                .open(50, vec![]),
            1 => app
                .world_mut()
                .resource_mut::<crate::ui_bank::BankOpen>()
                .open(51),
            _ => {
                let mut template = crate::items::test_template("Equippable item");
                template.inventory_type = 13;
                // No ON_USE: direct use must not auto-equip it.
                app.world_mut()
                    .resource_mut::<Items>()
                    .insert_template(ENTRY, Some(template));
            }
        }
        run(&mut app, "C_Item.UseItemByName(12345,'player')");
        assert_eq!(uses(&rx).len(), usize::from(mode < 2));
    }
}

#[test]
fn item_request_resolves_its_guid_at_the_new_slot() {
    let (mut app, rx) = fixture(false);
    // The item moved before the call: the call finds it where it is, as it applies.
    set_slots(&mut app, OTHER_ITEM, ITEM);
    run(&mut app, "C_Item.UseItemByName(12345,'player')");
    assert_eq!(uses(&rx), vec![(255, 24, 1, UseItemTarget::Unit(3))]);
}

#[test]
fn missing_target_item_or_pending_slot_refuses_before_a_bind_prompt() {
    for mode in [0, 1, 2] {
        let (mut app, rx) = fixture(true);
        // The target, the item or its slot gone before the call applies.
        match mode {
            0 => {
                app.world_mut()
                    .resource_mut::<crate::net::GuidIndex>()
                    .0
                    .remove(&4);
            }
            1 => set_slots(&mut app, 0, OTHER_ITEM),
            _ => app
                .world_mut()
                .non_send_mut::<UiScript>()
                .lock_item(ITEM),
        }
        run(&mut app, "C_Item.UseItemByName(12345,'target')");
        assert!(uses(&rx).is_empty());
        assert!(app
            .world()
            .resource::<crate::ui_bind_confirm::PendingBindOnUse>()
            .0
            .is_none());
    }
}

#[test]
fn bind_confirmation_retains_target_and_finds_the_moved_instance_only_once() {
    let (mut app, rx) = fixture(true);
    run(
        &mut app,
        "C_Item.UseItemByName(12345,'target');TargetUnit('player')",
    );
    assert!(uses(&rx).is_empty());
    assert!(app
        .world()
        .resource::<crate::ui_bind_confirm::PendingBindOnUse>()
        .0
        .is_some());
    set_slots(&mut app, OTHER_ITEM, ITEM);
    confirm(&mut app);
    assert_eq!(uses(&rx), vec![(255, 24, 1, UseItemTarget::Unit(4))]);
    confirm(&mut app);
    assert!(uses(&rx).is_empty());
}

#[test]
fn changed_items_units_locks_or_vm_invalidate_the_owned_confirmation() {
    for mode in [0, 1, 2, 3, 4] {
        let (mut app, rx) = fixture(true);
        run(
            &mut app,
            "C_Item.UseItemByName(12345,'target');TargetUnit('player')",
        );
        match mode {
            0 => {
                app.world_mut()
                    .resource_mut::<crate::net::GuidIndex>()
                    .0
                    .remove(&4);
            }
            1 => set_slots(&mut app, 0, OTHER_ITEM),
            2 => app
                .world_mut()
                .non_send_mut::<UiScript>()
                .lock_item(ITEM),
            3 => {
                let mut template = crate::items::test_template("Transformed item");
                template.start_quest = 42;
                app.world_mut()
                    .resource_mut::<Items>()
                    .insert_template(ENTRY, Some(template));
            }
            _ => {
                let script = UiScript::new().unwrap();
                benilla_pfui::install(&script).unwrap();
                app.world_mut().insert_non_send(script);
            }
        }
        confirm(&mut app);
        assert!(uses(&rx).is_empty(), "mode {mode}");
        assert!(app
            .world()
            .resource::<crate::ui_bind_confirm::PendingBindOnUse>()
            .0
            .is_none());
    }
}

#[test]
fn an_ordinary_deferred_use_replaces_the_extension_target_even_for_the_same_item() {
    let (mut app, rx) = fixture(true);
    run(
        &mut app,
        "C_Item.UseItemByName(12345,'target');TargetUnit('player');UseContainerItem(0,1)",
    );
    confirm(&mut app);
    assert_eq!(uses(&rx), vec![(255, 23, 1, UseItemTarget::Unit(3))]);
}

#[test]
fn an_implicit_target_is_also_retained_across_bind_confirmation() {
    let (mut app, rx) = fixture(true);
    run(&mut app, "C_Item.UseItemByName(12345);TargetUnit('player')");
    confirm(&mut app);
    assert_eq!(uses(&rx), vec![(255, 23, 1, UseItemTarget::Unit(4))]);
}
