//! Area triggers: notice the player entering an `AreaTrigger.dbc` volume and send
//! `CMSG_AREATRIGGER`; the server decides what the trigger does.
//!
//! The reference's check (`0x5e2110`, containment `0x5e22d0`): only the current map's rows are
//! candidates (`0x5e2080`); while the player is still inside the latched trigger nothing is sent;
//! once they leave, the first containing row in file order is latched and sent; a map change
//! clears the latch. So a portal fires once per entry, and a portal's destination is authored
//! outside the return trigger, so a round trip is not a loop.
//!
//! The server re-checks the claim with 5 yd of slop and ignores it while taxi-flying
//! (vmangos `Handlers/MiscHandler.cpp:622`).
//!
//! A GM teleport straight into a volume sends this in the same instant as the teleport ack, and
//! the server may test it against the pre-teleport position and ignore it. Probe by moving into
//! the volume instead.
//!
//! Deviates: the reference sends once per entry. The server tests the claim against the last
//! position the movement stream reported, and the Turtle core clears an inn's rest whenever that
//! position is outdoors and outside the volume with no slop (`Player.cpp:7867`), so an entry
//! claimed a heartbeat ahead of the position can set rest and drop it at once, and the latch never
//! asks again: no rest until you walk out and back in. While still inside, not resting, and moved
//! `RETRY_MOVED` since the last claim, we re-send it from the newer position, at most `RETRIES`
//! times and no sooner than `RETRY_AFTER` after the last. A rest that came on and went again is
//! re-claimed the same way: the flag showing for a moment is not the claim holding.

use std::time::Duration;

use benilla_assets::coords::bevy_to_wow;
use benilla_assets::DataLoadAppExt;
use benilla_formats::AreaTriggerCatalog;
use bevy::platform::time::Instant;
use bevy::prelude::*;

use crate::net::{ClientCommand, NetCommands, ObjectStore, SelfPlayer};
use crate::player::Player;
use benilla_assets::{LockRecover, WorldAssets};
use benilla_world::schedule::WorldStage;
use benilla_world::world_map::CurrentMap;

/// The `AreaTrigger.dbc` catalog, bucketed by map; absent when the data did not load.
#[derive(Resource)]
pub(crate) struct AreaTriggers(pub(crate) AreaTriggerCatalog);

/// The latched trigger and its map: the reference's current-trigger pointer (`0xc4d73c`), cleared
/// on a map change and kept across a same-map reconnect, as the reference does.
#[derive(Resource, Default)]
pub(crate) struct InsideTrigger(Option<(u32, u32)>);

/// Re-sends of a latched claim that did not leave us resting (see the module's `Deviates`).
const RETRIES: u8 = 3;
/// Long enough for the movement stream to have reported the newer position.
const RETRY_AFTER: Duration = Duration::from_secs(1);
/// Yards moved since the last claim: standing still, the server would judge the same position.
const RETRY_MOVED: f32 = 1.0;
/// `PLAYER_FLAGS_RESTING`, set by the server while rested in an inn or a city.
const PLAYER_FLAGS_RESTING: u32 = 0x20;

/// The pending re-send for the latched trigger.
#[derive(Resource, Default)]
struct TriggerRetry(Option<Claim>);

/// The last claim of the latched trigger.
#[derive(Clone, Copy)]
struct Claim {
    map: u32,
    id: u32,
    at: Instant,
    from: [f32; 3],
    left: u8,
}

impl InsideTrigger {
    /// One check in the reference's order (`0x5e2110`): nothing while still inside the latched
    /// volume, else latch and return the first trigger on this map containing `p`.
    fn step(&mut self, triggers: &AreaTriggerCatalog, map_id: u32, p: [f32; 3]) -> Option<u32> {
        if let Some((latched_map, id)) = self.0 {
            let still_in = latched_map == map_id
                && triggers
                    .on_map(map_id)
                    .iter()
                    .find(|t| t.id == id)
                    .is_some_and(|t| t.contains(p));
            if still_in {
                return None;
            }
            self.0 = None;
        }
        let entered = triggers.first_containing(map_id, p)?;
        self.0 = Some((map_id, entered.id));
        Some(entered.id)
    }
}

pub(crate) struct AreaTriggerPlugin;

impl Plugin for AreaTriggerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InsideTrigger>()
            .init_resource::<TriggerRetry>()
            .add_data_loaders(load_area_triggers)
            // After Input has moved the avatar, so the position tested is the one the movement
            // stream reports, which the server re-checks the claim against.
            .add_systems(
                Update,
                check_area_triggers
                    .in_set(WorldStage::Stream)
                    .in_set(crate::client_state::InWorldGated),
            );
    }
}

fn load_area_triggers(mut commands: Commands, assets: Option<Res<WorldAssets>>) {
    let Some(assets) = assets else { return };
    let mut chain = assets.chain.lock_recover();
    match benilla_formats::load_area_trigger_catalog(&mut chain) {
        Ok(cat) => {
            info!("area_trigger: {} triggers", cat.len());
            commands.insert_resource(AreaTriggers(cat));
        }
        Err(e) => warn!("area_trigger: AreaTrigger.dbc failed to load: {e:#}"),
    }
}

/// The per-frame check (`0x5e2110`).
fn check_area_triggers(
    triggers: Option<Res<AreaTriggers>>,
    map: Option<Res<CurrentMap>>,
    player: Res<Player>,
    mut inside: ResMut<InsideTrigger>,
    mut retry: ResMut<TriggerRetry>,
    self_q: Query<&ObjectStore, With<SelfPlayer>>,
    net: Res<NetCommands>,
) {
    let (Some(triggers), Some(map)) = (triggers, map) else {
        return;
    };
    // Before the server has placed us, `pos` is not a server position.
    if !player.active {
        return;
    }
    let here = bevy_to_wow(player.pos);
    let Some(trigger_id) = inside.step(&triggers.0, map.0, here) else {
        // Still in the latched volume (or in none): re-claim it if the server left us unrested.
        let Some(claim) = retry.0 else {
            return;
        };
        if inside.0 != Some((claim.map, claim.id)) {
            retry.0 = None;
            return;
        }
        if claim.left == 0 {
            retry.0 = None;
            return;
        }
        // Resting now keeps the claim armed: the server can drop the rest a tick after setting it.
        let resting = self_q
            .single()
            .is_ok_and(|s| s.0.player_flags() & PLAYER_FLAGS_RESTING != 0);
        if resting {
            return;
        }
        let moved = here
            .iter()
            .zip(claim.from)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        if claim.at.elapsed() < RETRY_AFTER || moved < RETRY_MOVED {
            return;
        }
        let _ = net.0.send(ClientCommand::AreaTrigger {
            trigger_id: claim.id,
        });
        retry.0 = Some(Claim {
            at: Instant::now(),
            from: here,
            left: claim.left - 1,
            ..claim
        });
        info!(
            "area_trigger: re-sent {} on map {} at [{:.2}, {:.2}, {:.2}], still unrested",
            claim.id, claim.map, here[0], here[1], here[2]
        );
        return;
    };
    let _ = net.0.send(ClientCommand::AreaTrigger { trigger_id });
    retry.0 = Some(Claim {
        map: map.0,
        id: trigger_id,
        at: Instant::now(),
        from: here,
        left: RETRIES,
    });
    // At `info`: entries are rare, and this line tells whether the client saw the volume at all.
    info!(
        "area_trigger: entered {trigger_id} on map {} at [{:.2}, {:.2}, {:.2}]",
        map.0, here[0], here[1], here[2]
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_catalog() -> Option<AreaTriggerCatalog> {
        let data = benilla_formats::wow_data_or_skip!(None);
        let mut chain = benilla_formats::open_chain(&data).expect("open chain");
        Some(benilla_formats::load_area_trigger_catalog(&mut chain).expect("AreaTrigger.dbc"))
    }

    /// The state machine over the real table: the Darnassus portal pair and the Southshore inn's
    /// box trigger. Skips without client data.
    #[test]
    fn a_trigger_fires_once_per_entry_and_re_arms_on_leaving() {
        let Some(cat) = real_catalog() else { return };
        let mut inside = InsideTrigger::default();

        // Rut'theran Village's portal (542, a 10 yd sphere).
        let ruttheran = [8799.41, 969.787, 30.2409];
        assert_eq!(inside.step(&cat, 1, ruttheran), Some(542));
        assert_eq!(
            inside.step(&cat, 1, ruttheran),
            None,
            "standing in a portal must not re-report it — that is the teleport loop"
        );

        // The Darnassus landing spot lies outside the return trigger (527, 10 yd), 17.2 yd from
        // its centre: nothing fires and the latch re-arms.
        let darnassus_arrival = [9946.25, 2612.97, 1316.49];
        assert_eq!(inside.step(&cat, 1, darnassus_arrival), None);

        assert_eq!(inside.step(&cat, 1, [9947.48, 2630.04, 1318.6]), Some(527));

        // The Southshore inn's box trigger (708) on map 0 fires while one is latched on map 1.
        assert_eq!(
            inside.step(&cat, 0, [-854.547, -576.314, 18.4659]),
            Some(708)
        );
        assert_eq!(inside.step(&cat, 0, [-854.593, -576.207, 18.5563]), None);

        // The Deadmines entrance (78), an instance portal: a 7 yd sphere on map 0.
        assert_eq!(inside.step(&cat, 0, [-11208.5, 1685.34, 25.7612]), Some(78));
    }

    #[test]
    fn open_ground_and_unknown_maps_are_silent() {
        let Some(cat) = real_catalog() else { return };
        let mut inside = InsideTrigger::default();
        assert_eq!(inside.step(&cat, 0, [0.0, 0.0, 0.0]), None);
        assert_eq!(inside.step(&cat, 9999, [8799.41, 969.787, 30.2409]), None);
    }
}
