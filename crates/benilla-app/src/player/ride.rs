//! Riding a transport (boat, zeppelin, tram). [`carry`] recomposes the rider from the deck's pose
//! before any input integrates; [`update_attachment`] decides after the mover whether we are still
//! aboard, re-snapshots the deck-local pose and writes the `ride` trace tag.

use bevy::prelude::*;

use super::state::PendingBoard;
use super::{wrap_pi, FlyCam, Player, PlayerRide, TransportQuery};

/// A board or deboard line: the log, and on a `?diag=1` web build, which has no log, the console.
fn note(line: String) {
    info!("{line}");
    if crate::webdiag::armed() {
        crate::webdiag::out(&line);
    }
}

/// Recomposes the rider from the deck's this-frame pose.
pub(super) fn carry(player: &mut Player, cam: &mut FlyCam, transports: &TransportQuery) {
    // The transport tick runs on the Net→Input edge, so the deck pose is this frame's; a
    // streamed-out boat detaches into a fall. The yaw delta goes to aim, body and camera alike, on
    // top of this frame's mouse-look, as the reference's camera rides the transport-local rig: a
    // deck turn must reach neither the body chase, whose steps latch the turn-in-place shuffle,
    // nor `seat_camera`'s look-session gate.
    if let Some(ride) = player.ride.as_ref() {
        match transports.get(ride.entity) {
            Ok((boat, _, _)) => {
                let world = boat.translation + boat.rotation * ride.local_pos;
                let yaw_now = boat.rotation.to_euler(EulerRot::YXZ).0;
                let mut dyaw = yaw_now - ride.boat_yaw;
                // `to_euler` wraps to (−π, π]; a boat crossing that seam reads as a ±2π hop.
                dyaw = (dyaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                    - std::f32::consts::PI;
                player.pos = world;
                player.face_yaw += dyaw;
                player.model_yaw = wrap_pi(player.model_yaw + dyaw);
                cam.yaw += dyaw;
                if let Some(r) = player.ride.as_mut() {
                    r.boat_yaw = yaw_now;
                }
            }
            Err(_) => {
                note(format!(
                    "transport: deboard {:#x} (its entity is gone)",
                    ride.guid
                ));
                player.ride = None;
            }
        }
    }
}

/// A login aboard a transport: vmangos seats a passenger at the transport's current position plus
/// its saved offset (`Player::LoadFromDB` → `CalculatePassengerPosition`) and sends our create
/// block with `MOVEFLAG_ONTRANSPORT` and that offset, which the create path tags on our body as a
/// [`crate::transport::TransportRider`]. Our body is the mover's, not `compose_riders`', so the
/// tag becomes a pending board, and the ride starts once its transport is armed and posed. A
/// worldport that carries us with a transport we had not boarded yet (it changed continents during
/// the login load) leaves one too, by entry, and the transport is found on the new map.
pub(super) fn board_from_create(
    mut commands: Commands,
    mut player: ResMut<Player>,
    tagged: Query<(Entity, &crate::transport::TransportRider), With<crate::net::Embodied>>,
    decks: Query<
        (
            Entity,
            &crate::net::Guid,
            &Transform,
            Option<&crate::net::ObjectStore>,
        ),
        With<crate::transport::Transport>,
    >,
) {
    for (body, tail) in &tagged {
        commands
            .entity(body)
            .remove::<crate::transport::TransportRider>();
        if player.ride.is_none() {
            player.pending_board = Some(PendingBoard {
                guid: Some(tail.transport_guid),
                entry: None,
                local_pos: benilla_assets::coords::wow_to_bevy(tail.local_pos),
            });
        }
    }
    let Some(pending) = player.pending_board else {
        return;
    };
    if player.ride.is_some() {
        player.pending_board = None;
        return;
    }
    let found = decks.iter().find(|(_, g, _, store)| match pending.guid {
        Some(guid) => g.0 == guid,
        None => store.and_then(|s| s.0.object_entry()) == pending.entry,
    });
    let Some((entity, guid, boat, _)) = found else {
        return; // not streamed in or not armed yet: the hold keeps the body where it is
    };
    let (guid, local_pos) = (guid.0, pending.local_pos);
    player.pending_board = None;
    player.pos = boat.translation + boat.rotation * local_pos;
    player.ride = Some(PlayerRide {
        entity,
        guid,
        local_pos,
        boat_yaw: boat.rotation.to_euler(EulerRot::YXZ).0,
    });
    note(format!("transport: board {guid:#x} (logged in aboard)"));
}

/// Attaches or detaches against this frame's walkable support, then re-snapshots the deck-local
/// pose. `ground` is the mover's resolved support ([`super::mover::Outcome::ground`]).
pub(super) fn update_attachment(
    player: &mut Player,
    transports: &TransportQuery,
    child_of: &Query<&ChildOf>,
    ground: Option<Entity>,
    grounded: bool,
    swimming: bool,
) {
    // Attach on a transport's hull or a deck prop's collider child, found up the parent chain;
    // detach on world geometry or in water. Airborne keeps the attachment, so a jump above the
    // deck is deck-frame ballistics.
    let owning_transport = |mut e: Entity| {
        for _ in 0..4 {
            if let Ok((t, g, _)) = transports.get(e) {
                return Some((e, t, g));
            }
            e = child_of.get(e).ok()?.parent();
        }
        None
    };
    if swimming {
        if player.ride.take().is_some() {
            note("transport: deboard (entered the water)".into());
        }
    } else if grounded {
        match ground.and_then(owning_transport) {
            Some((entity, _, guid)) => {
                if player.ride.as_ref().map(|r| r.entity) != Some(entity) {
                    note(format!(
                        "transport: board {:#x} (support is its deck, collider {:?})",
                        guid.0, ground
                    ));
                }
                player.ride = Some(PlayerRide {
                    entity,
                    guid: guid.0,
                    local_pos: Vec3::ZERO, // filled by the snapshot just below
                    boat_yaw: 0.0,
                });
            }
            None => {
                if let Some(ride) = player.ride.take() {
                    // What the support was, up its parent chain, and where the feet stood on the
                    // deck: the one line that tells a deck seam from a world collider.
                    let mut chain = Vec::new();
                    let mut e = ground;
                    while let Some(x) = e {
                        chain.push(format!("{x:?}"));
                        e = child_of.get(x).ok().map(ChildOf::parent);
                        if chain.len() > 8 {
                            break;
                        }
                    }
                    let local = transports
                        .get(ride.entity)
                        .ok()
                        .map(|(t, _, _)| t.rotation.inverse().mul_vec3(player.pos - t.translation));
                    note(format!(
                        "transport: deboard {:#x} (support is not its deck) collider chain [{}] \
                         feet {:?} deck-local {:?} steep {} wedged {}",
                        ride.guid,
                        chain.join(" < "),
                        player.pos,
                        local,
                        player.steep_support,
                        player.wedged,
                    ));
                }
            }
        }
    }
    let feet = player.pos;
    // The `ride` trace (`WOW_MOVE_TRACE_TAGS=ride`): the deck-relative path, which no other
    // instrument records, a line a frame while attached.
    if benilla_assets::trace::enabled_for("ride") {
        let boat_pose = player
            .ride
            .as_ref()
            .and_then(|r| transports.get(r.entity).ok())
            .map(|(t, _, aabb)| {
                (
                    t.translation,
                    t.rotation.to_euler(EulerRot::YXZ).0,
                    aabb.copied(),
                )
            });
        if let (Some(ride), Some((bpos, byaw, baabb))) = (player.ride.as_ref(), boat_pose) {
            let local = Quat::from_euler(EulerRot::YXZ, byaw, 0.0, 0.0)
                .inverse()
                .mul_vec3(feet - bpos);
            // The deck's broad-phase box, which the mover probes' candidate query tests, trails the
            // deck by a frame: avian refreshes it in `FixedPostUpdate`, before `tick_transports`
            // moves the deck in `Update`. A positive `gap` hides the deck from the down-probe.
            let aabb_col = match baabb {
                Some(a) => format!(
                    " | aabbY[{:8.2},{:8.2}] gap{:+7.2}",
                    a.min.y,
                    a.max.y,
                    a.min.y - feet.y,
                ),
                None => " | aabbY[  absent]".to_string(),
            };
            benilla_assets::trace::line(
                "ride",
                &format!(
                    "on {:#x} deck({:8.2},{:7.2},{:8.2}) yaw{:+.3} | feet({:8.2},{:7.2},{:8.2}) \
                     local({:7.2},{:6.2},{:7.2}) | grounded={} support={} vy={:+6.2}{}",
                    ride.guid,
                    bpos.x,
                    bpos.y,
                    bpos.z,
                    byaw,
                    feet.x,
                    feet.y,
                    feet.z,
                    local.x,
                    local.y,
                    local.z,
                    grounded as u8,
                    match ground.and_then(owning_transport) {
                        Some(_) => "deck",
                        None if ground.is_some() => "world",
                        None => "NONE",
                    },
                    player.vel_y,
                    aabb_col,
                ),
            );
        } else if !swimming {
            // Not riding: a line only when standing on a transport, a missed attach.
            if let Some((_, _, guid)) = ground.and_then(owning_transport) {
                benilla_assets::trace::line(
                    "ride",
                    &format!(
                        "OFF but standing on {:#x} at ({:8.2},{:7.2},{:8.2}) grounded={}",
                        guid.0, feet.x, feet.y, feet.z, grounded as u8
                    ),
                );
            }
        }
    }
    if let Some(ride) = player.ride.as_mut() {
        if let Ok((boat, _, _)) = transports.get(ride.entity) {
            ride.local_pos = boat.compute_affine().inverse().transform_point3(feet);
            ride.boat_yaw = boat.rotation.to_euler(EulerRot::YXZ).0;
        }
    }
}
