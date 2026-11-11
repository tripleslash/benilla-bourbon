//! The pfUI app tests' world: the headless client with a player, a known book, the extension
//! installed, and the frame's steps a test runs Lua through.

#[cfg(feature = "nampower")]
use benilla_ui::script::pfui_host::casts;
use benilla_ui::script::{SpellBookState, SpellSlotView, UiScript, UnitGuids};
#[cfg(feature = "nampower")]
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

pub(crate) const HEAL: u32 = 2050;
pub(crate) const OTHER: u32 = 2052;
pub(crate) const ME: u64 = 3;

pub(crate) fn fixture() -> (App, crossbeam_channel::Receiver<crate::net::ClientCommand>) {
    let mut app = crate::game_plugins::schedule_tests::headless_client();
    let (tx, rx) = crossbeam_channel::unbounded();
    app.insert_resource(crate::net::NetCommands(tx));
    let mut spells = crate::ui_action::Spells::empty_for_tests();
    spells.catalog = benilla_formats::SpellCatalog::from_displays(
        [HEAL, OTHER]
            .map(|id| {
                (
                    id,
                    benilla_formats::SpellDisplay {
                        // The book's name, which `CastSpellByName` finds through the host.
                        name: format!("Heal{id}"),
                        start_recovery_category: 133,
                        start_recovery_ms: 1500,
                        implicit_target_a1: 1,
                        ..Default::default()
                    },
                )
            })
            .into(),
    );
    app.insert_resource(spells);
    let world = app.world_mut();
    let me = world
        .spawn((
            crate::net::SelfPlayer,
            crate::net::Guid(ME),
            crate::net::ObjectStore(benilla_protocol::ObjectFields::from_pairs(&[
                (22, 100),
                (28, 100),
            ])),
            Transform::default(),
            GlobalTransform::default(),
        ))
        .id();
    world
        .resource_mut::<crate::net::GuidIndex>()
        .0
        .insert(ME, me);
    world.resource_mut::<crate::net::SelfGuid>().0 = Some(ME);
    let mut script = UiScript::new().unwrap();
    // The host's CVar table, as `setup_script` seeds it: an unregistered name raises.
    script.register_cvars(crate::cvars::registered_pairs());
    benilla_pfui::install(&script).unwrap();
    benilla_pfui::spell::Catalog(Default::default())
        .install(script.lua())
        .unwrap();
    script.set_unit_guids(&UnitGuids {
        player: ME,
        ..Default::default()
    });
    script.set_spellbook(SpellBookState {
        slots: [HEAL, OTHER]
            .map(|id| SpellSlotView {
                spell_id: id,
                name: format!("Heal{id}"),
                ..Default::default()
            })
            .into(),
        ..Default::default()
    });
    script
        .run(
            r#"
      rows={}; listener=CreateFrame('Frame');listener:RegisterEvent('SPELL_QUEUE_EVENT')
      listener:SetScript('OnEvent',function() table.insert(rows,{arg1,arg2,arg3}) end)
    "#,
        )
        .unwrap();
    #[cfg(feature = "nampower")]
    script.run("SetCVar('NP_MinBufferTimeMs',0)").unwrap();
    world.insert_non_send(script);
    (app, rx)
}

/// The app learns the spells of the VM's test book: the host's book is the app's own (its known
/// spells over the catalog, whose entries carry the book's names), which a getter run with the
/// host lent reads.
fn teach(app: &mut App) {
    let world = app.world_mut();
    let book = world.non_send::<UiScript>().spellbook();
    world
        .resource_mut::<crate::ui_action::PlayerActions>()
        .spells
        .extend(book.slots.iter().map(|slot| slot.spell_id));
}

/// Run `lua` with the host lent, so its calls apply inside their verbs, then, with Nampower, the
/// frame's release of its queue.
pub(crate) fn run(app: &mut App, lua: &str) {
    teach(app);
    // The frame's pick point, as the queue reads it in a call.
    #[cfg(feature = "nampower")]
    app.world_mut()
        .run_system_once(crate::pfui::spell_queue::track_pick)
        .unwrap();
    crate::script_calls::with_hosted_vm(app.world_mut(), |script| {
        script.expect("a VM").run(lua).unwrap();
    });
    #[cfg(feature = "nampower")]
    app.world_mut()
        .run_system_once(crate::pfui::spell_queue::release)
        .unwrap();
}

#[cfg(feature = "nampower")]
pub(crate) fn casting(app: &App, end_ms: Option<f64>) {
    casts::set_casts(
        app.world().non_send::<UiScript>().lua(),
        end_ms.map(|end_ms| {
            (
                ME,
                casts::Cast {
                    spell_id: OTHER,
                    start_ms: 0.0,
                    end_ms,
                    name: "Heal2052".into(),
                    rank: String::new(),
                    icon: None,
                    tradeskill: false,
                    cast_guid: "Cast-3-1".into(),
                    delay_ms: 0,
                    channel: false,
                },
            )
        }),
    );
}

#[cfg(feature = "nampower")]
pub(crate) fn cast_packets(
    rx: &crossbeam_channel::Receiver<crate::net::ClientCommand>,
) -> Vec<u32> {
    rx.try_iter()
        .filter_map(|c| match c {
            crate::net::ClientCommand::CastSpell { spell_id, .. } => Some(spell_id),
            _ => None,
        })
        .collect()
}

pub(crate) fn with_unit_targets(app: &mut App) {
    let world = app.world_mut();
    let me = world.resource::<crate::net::GuidIndex>().0[&ME];
    world.entity_mut(me).insert(crate::net::ObjectStore(
        benilla_protocol::ObjectFields::from_pairs(&[(2, 0x19), (22, 100), (28, 100), (16, 4)]),
    ));
    let other = world
        .spawn((
            crate::net::Guid(4),
            crate::net::ObjectStore(benilla_protocol::ObjectFields::from_pairs(&[
                (2, 0x9),
                (22, 100),
                (28, 100),
            ])),
            Transform::default(),
            GlobalTransform::default(),
        ))
        .id();
    world
        .resource_mut::<crate::net::GuidIndex>()
        .0
        .insert(4, other);
    *world.resource_mut::<crate::target::Selection>() = crate::target::Selection {
        guid: Some(4),
        target: Some(other),
        ..Default::default()
    };
    world.resource_mut::<crate::ui_action::Spells>().catalog =
        benilla_formats::SpellCatalog::from_displays(
            [(
                HEAL,
                benilla_formats::SpellDisplay {
                    name: format!("Heal{HEAL}"),
                    targets: 2,
                    start_recovery_category: 133,
                    start_recovery_ms: 1500,
                    ..Default::default()
                },
            )]
            .into(),
        );
    world
        .non_send_mut::<UiScript>()
        .set_unit_guids(&UnitGuids {
            player: ME,
            target: 4,
            held: [(ME, 4), (4, 0)].into(),
            ..Default::default()
        });
}

#[cfg(feature = "nampower")]
pub(crate) fn targets(
    rx: &crossbeam_channel::Receiver<crate::net::ClientCommand>,
) -> Vec<Option<u64>> {
    rx.try_iter()
        .filter_map(|call| match call {
            crate::net::ClientCommand::CastSpell { target, .. } => Some(target),
            _ => None,
        })
        .collect()
}
