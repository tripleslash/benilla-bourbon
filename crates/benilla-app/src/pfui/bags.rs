//! Pace the addon batch over real descriptor confirmations, using the existing item sender.

use crate::net::NetHandlerApp;
use benilla_pfui::bags::{Executor, Slot};
use benilla_protocol::{SessionEvent, SessionEventKind};
use benilla_ui::script::{pfui_host::items, UiScript};
use bevy::prelude::*;

pub(super) fn register(app: &mut App) {
    app.net_handler(SessionEventKind::InventoryFailure, failure);
    // The moves the Lua asked for, drained with the UI's other asks.
    app.add_systems(Update, drain.in_set(crate::ui_script::UiDrain));
}

fn failure(In(event): In<SessionEvent>, script: Option<NonSend<UiScript>>) {
    if let (SessionEvent::InventoryFailure { item_guid, .. }, Some(script)) = (event, script) {
        items::bag_failure(script.lua(), item_guid);
    }
}

fn drain(
    script: Option<NonSend<UiScript>>,
    commands: Res<crate::net::NetCommands>,
    self_q: Query<&crate::net::ObjectStore, With<crate::net::SelfPlayer>>,
    objects: crate::net::Objects,
    templates: Res<crate::items::Items>,
    mut memo: Local<crate::ui_script::VmMemo<Executor>>,
    time: Res<Time>,
) {
    let Some(script) = script else { return };
    let worker = memo.get(&script);
    let incoming = items::take_bag_moves(script.lua());
    let failures = items::take_bag_failures(script.lua());
    let Ok(store) = self_q.single() else {
        worker.clear();
        items::set_bag_busy(script.lua(), false);
        return;
    };
    let mut error = worker.enqueue(incoming).err();
    for guid in failures {
        if worker.failed(guid) {
            error = Some("pfUI bag operation was rejected by the server".into());
        }
    }
    if error.is_none() {
        // The client's locks are per item now, by guid; an empty slot holds none.
        let locks = script.item_locks();
        let read = |bag, slot| {
            if !items::bag_position_available(script.lua(), bag, slot)
                || crate::ui_items::bag_num_slots(&store.0, bag, &objects).is_none_or(|n| slot > n)
            {
                return None;
            }
            let guid = crate::ui_items::slot_guid(
                &store.0,
                bag,
                u8::try_from(slot.checked_sub(1)?).ok()?,
                &objects,
            );
            let Some(guid) = guid else {
                return Some(Slot::default());
            };
            let fields = objects.object(guid)?;
            let id = fields.object_entry()?;
            let template = templates.template(id, guid, crate::query_cache::Ask::New(&commands))?;
            Some(Slot {
                guid,
                id,
                count: fields.item_stack_count()?,
                max_stack: template.stackable.max(1),
                locked: locks.contains(&guid),
            })
        };
        match worker.next(time.elapsed_secs_f64(), read) {
            // A move within the bags binds nothing, so it goes out as the stock sender sends a
            // whole stack or a split; the worker waits for both ends' descriptors.
            Ok(Some(movement)) => crate::ui_items::pfui::send_container_move(&commands, movement),
            Ok(None) => {}
            Err(message) => error = Some(message),
        }
    }
    items::set_bag_busy(script.lua(), worker.busy());
    if let Some(message) = error {
        benilla_ui::script::pfui_host::report_error(script.lua(), (message).to_string());
    }
}
