//! The area sends: the area trigger and the zone update.

use anyhow::Result;

use crate::messages::{self, opcode};

use super::WorldWriter;

impl WorldWriter {
    /// Report walking into an `AreaTrigger.dbc` volume (`CMSG_AREATRIGGER`). There is no success
    /// reply; a refusal is `SMSG_AREA_TRIGGER_MESSAGE`, and most triggers answer nothing.
    pub fn area_trigger(&mut self, trigger_id: u32) -> Result<()> {
        self.send(
            opcode::CMSG_AREATRIGGER,
            &messages::area_trigger(trigger_id),
        )
    }

    /// `CMSG_ZONEUPDATE` (0x1f4, `0x67e940`): the zone id as a `u32` (`0x67e977`). vmangos ignores
    /// the payload and runs `UpdateZone` on receipt (`MiscHandler.cpp:381-385`).
    pub fn zone_update(&mut self, zone: u32) -> Result<()> {
        self.send(opcode::CMSG_ZONEUPDATE, &zone.to_le_bytes())
    }
}
