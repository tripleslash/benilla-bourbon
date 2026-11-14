# `may-park` — benilla-bourbon's own crate, not a vendored one

`may-park` is written for this fork, not copied from crates.io; it sits in `third_party/` beside the
forks that depend on it. It is MIT OR Apache-2.0, as the rest of the tree.

## Why it exists

On `wasm32 + atomics` a contended `std::sync` lock parks with `memory.atomic.wait32`. A Web Worker
may execute that instruction; the browser page's thread may not: there it throws `Atomics.wait
cannot be called in this context` and the module traps. The page and the workers share locks (the
smol executor and its channels, bevy's `AssetServer`, the render worker's handshake), so no such
lock may park on the page. Every other thread should park, because a worker that spins burns a core
for as long as the holder runs.

The crate answers "may this thread park?" once per thread and offers `std` locks that ask:

- `forbid_parking_on_this_thread()`: the page calls it once, in `benilla_app::web_threads::install`,
  before any pool exists.
- `may_park()`: `false` on that thread, `true` everywhere else, and always `true` off
  `wasm32 + atomics`.
- `sync::Mutex` and `sync::RwLock`: std's own locks and guards, so a `Condvar` waits on them as usual.
  Their blocking calls loop on `try_*` where `may_park()` is `false`. Off `wasm32 + atomics` they
  are `std::sync`.

## Who uses it

The forks of `async-executor`, `event-listener`, `parking`, `bevy_platform` and `bevy_tasks`; each
one's `BENILLA.md` names the lock it routes through here. `src/lib.rs` is the whole crate.
