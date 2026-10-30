//! Environmental-damage feedback: the fall dust puff, fired by two independent sources as in the
//! reference. `SMSG_ENVIRONMENTALDAMAGELOG` plays its damage type's kit on the victim and no sound
//! (`0x624f30`, through `0x60edf0`); the landing predictor `0x602d00`, run for every mover, plays
//! the wound vocal ([`crate::sound`]) and re-fires the fall kit at once ([`hard_landing_dust`]), so
//! a damaging fall puffs twice.

use bevy::prelude::*;

use benilla_assets::{LockRecover, WorldAssets};

/// `EnvironmentalDamage.dbc`'s six damage-type kits; absent, no environmental kit plays.
#[derive(Resource)]
pub(crate) struct EnvDamageTable(pub(crate) benilla_formats::EnvironmentalDamageTable);

/// Load the table off the patch chain at startup.
pub(super) fn load_env_damage_table(mut commands: Commands, assets: Option<Res<WorldAssets>>) {
    let Some(assets) = assets else { return };
    let loaded = {
        let mut chain = assets.chain.lock_recover();
        benilla_formats::load_environmental_damage(&mut chain)
    };
    match loaded {
        Ok(table) => commands.insert_resource(EnvDamageTable(table)),
        Err(e) => {
            warn!("anim: EnvironmentalDamage.dbc failed to load (no fall-damage dust): {e:#}")
        }
    }
}

/// The hard-landing floor, `[0x80c414]` = 13.0 yd of fall height (`0x602d00`); below it only the
/// land sound plays. Below `[0x80c418]` = 70.0 yd a dead or ghost mover lands soft (`0x605f30`),
/// which is not built. The server damages only from 14.57 yd (vmangos `Player.cpp:20968`), so a
/// shorter hard landing grunts and puffs with no damage packet.
pub(crate) const HARD_LANDING_DESCENT: f32 = 13.0;

/// A mover's landing, reported on every landing; consumers apply [`HARD_LANDING_DESCENT`].
#[derive(Message, Clone, Copy)]
pub(crate) struct HardLanding {
    pub(crate) entity: Entity,
    /// Fall height in yd from the arc's apex ([`fall_height`]).
    pub(crate) descent: f32,
}

/// `[0x81da6c]`, `1/(2g)`.
const HALF_INV_G: f32 = 0.025_918_681;

/// `0x7c60c0`: the fall start `start` (`+0x7c`) less the height `z`, plus the climb to the apex
/// `v²/(2g)` while the launch rose, `rise` its upward speed (the wire's `+0xa0` below −2⁻²²,
/// `0x7c60d7`), so a jump counts from its highest point, 1.64 yd up for a land jump; a negative
/// height is 0 (`0x7c60f8`). At 53 bits until the landing handler stores it (`0x602d24`).
pub(crate) fn fall_height(start: f32, z: f32, rise: f32) -> f32 {
    let mut h = f64::from(start) - f64::from(z);
    if rise > 1.0 / 4_194_304.0 {
        h += f64::from(rise) * f64::from(rise) * f64::from(HALF_INV_G);
    }
    if h < 0.0 {
        0.0
    } else {
        h as f32
    }
}

/// The predictor's dust leg: `0x602d00` re-fires `0x624f30(type 2, damage 0)` at the landing frame.
pub(super) fn hard_landing_dust(
    mut landings: MessageReader<HardLanding>,
    table: Option<Res<EnvDamageTable>>,
    mut seq: ResMut<super::PlaySeq>,
    mut pushes: MessageWriter<super::KitPush>,
) {
    for l in landings.read() {
        if l.descent <= HARD_LANDING_DESCENT {
            continue;
        }
        // Type 2, fall: the predictor's only type.
        if let Some(kit_id) = table.as_ref().and_then(|t| t.0.kit_id(2)) {
            debug!(
                "anim: hard landing ({:.1} yd) → predicted dust kit {kit_id}",
                l.descent
            );
            pushes.write(super::KitPush {
                entity: l.entity,
                kit_id,
                seq: seq.next(),
                // The landing frame's own prediction, after every packet.
                at: None,
            });
        }
    }
}
