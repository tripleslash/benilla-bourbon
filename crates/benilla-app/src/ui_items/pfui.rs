//! Optional bag worker reuses the stock sender, bind checks and pending-operation locks.

pub(crate) use super::drain::pfui_use::DeferredItemUse;
pub(crate) use super::drain::send_container_move;
