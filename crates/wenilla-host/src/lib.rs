//! Library half of `wenilla-host` — split out purely so `tests/*` can build the exact routers
//! `main.rs` serves. An integration test can only reach a crate's *library* target (a bin-only
//! package has no seam a `tests/` file can `use`), so `main.rs` stays a thin CLI wrapper over
//! these modules.

pub mod data;
pub mod packs;
pub mod remote;
pub mod static_site;
pub mod ws;

/// The only ports the `/ws/{port}` proxy will ever dial — mangos realmd (login, 3724) and the two
/// world servers one shared realm list fronts: the 1.12.1 realm's (8085) and the Turtle WoW 1.18.1
/// realm's (8086). Fixed here, not a CLI flag: a gateway binds a
/// publicly reachable address, so the allowlist is a security boundary, not a convenience default
/// — with the loopback pin it is the entire reach of the relay. Public so anything that mounts
/// these routers uses the same set.
pub const ALLOWED_PORTS: [u16; 3] = [3724, 8085, 8086];
