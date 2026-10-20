# benilla's `async-executor` fork — what differs from upstream, and how to check

Upstream: `async-executor 1.14.0` (crates.io), Apache-2.0 OR MIT — the version `bevy_tasks 0.18.1`
resolves to. Wired in through `[patch.crates-io]` in the workspace root. It is the third of the
threading forks, after `third_party/bevy_tasks` and `third_party/bevy_ecs`, and it is the one that
made the threaded browser build stop crashing.

## Why a fork exists at all

**The browser's main thread may not block, and this executor blocks.**

`async-executor` guards its state with `std::sync::Mutex` (`sleepers`, `active`) and
`std::sync::RwLock` (`local_queues`). On `wasm32` with `+atomics` those are futex-based, and a futex
wait is the `memory.atomic.wait32` instruction — which the main thread of a web page is forbidden to
execute. It does not return an error and it does not merely stall: it throws

```
Uncaught RuntimeError: Atomics.wait cannot be called in this context
```

out of the middle of whatever was running, and the client dies. (Downstream of that, winit's event
loop then panics with `RefCell already borrowed` while the first error unwinds — a red herring, and
the reason this took a stack trace to find.)

That matters because in the threaded browser build this executor is shared between the page's main
thread and one Web Worker per core. **Every path where the main thread meets the pool takes one of
these locks:**

| main-thread call | lock it takes |
|---|---|
| `Scope::spawn` (every `par_iter_mut` batch) | `active` |
| `TaskPool::spawn_pooled` (every collider build) | `active` |
| ticking the pool inside `TaskPool::scope` | `sleepers` |
| running/stealing inside a scope | `local_queues` |

With a worker per core hammering the same executor, contention is not a corner case — it is what a
`par_iter_mut` sweep does on every frame. Measured 2026-09-06: the client crashed there on the first
frame it drew.

## The delta, and the command that proves it

**One import changed. No logic is touched.**

| file | change |
|---|---|
| `src/lib.rs` | the `use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, TryLockError};` becomes a two-arm `cfg`: on `wasm32 + atomics` the locks come from `third_party/may-park` (and the error types from `std::sync`), `std::sync` everywhere else |
| `Cargo.toml` | the `may-park` path dependency, on that target only |

`may-park`'s `Mutex`/`RwLock` are std's own locks with std's own guards; only `lock`, `read` and
`write` ask first whether the calling thread may park, and loop on `try_*` on the one that may not:
the browser page's, which says so once (`benilla_app::web_threads::install`). Every call site —
`.lock().unwrap_or_else(PoisonError::into_inner)`, `.try_read()`, the `Debug` impls matching
`TryLockError::WouldBlock` — is upstream's, unmodified, and the executor's algorithm, queues,
work-stealing and sleeper bookkeeping are exactly the ones upstream ships and tests.

**Native is bit-for-bit upstream.** The `cfg` is false on every target that is not a shared-memory
wasm build, so `std::sync` is used and this fork is inert.

## Spin only where parking is illegal

The first version of this fork (2026-09) shipped its own spin locks (`src/wasm_spin.rs`) on *every*
thread. That kept the page alive, and it also made every Web Worker spin on these locks instead of
parking: a pool thread, and since 2026-10 the render worker, burned its core for as long as the
holder ran, including a holder descheduled mid-section. A worker may park, and std's futex lock
already spins briefly before it does, so parking is the right shape there. The page is the only
thread for which a trap is the alternative, and it is the only one that spins now.

Precedent for the page's half: Rust's own wasm allocator (`library/std/src/sys/alloc/wasm.rs`)
spins on its global lock because *"the main thread in a web browser cannot ever block, no
exceptions"*. `event-listener`'s `slab.rs` spin lock is the `no_std` path only (see
`third_party/event-listener`).

## What this does NOT fix

It does not make blocking legal on the page — it removes it from *this* crate. Anything else the main thread
shares with a worker has to satisfy the same rule, and **three separate crates turned out to need
the same treatment** before the client stopped throwing:

| crate | where | fork |
|---|---|---|
| `async-executor` | `sleepers`, `active`, `local_queues` — taken on every `Scope::spawn` and tick | this one |
| `event-listener` | `intrusive.rs`'s list — taken by every `async-channel` send/recv that registers or notifies | `third_party/event-listener` |
| `parking` | `Unparker::unpark` — taken whenever spawning work wakes a sleeping worker | `third_party/parking` |

Audited and clean: `concurrent-queue`, `crossbeam-queue` and `async-task` (lock-free), the global
allocator (std spins, above), `bevy_platform`'s `OnceLock`s (initialised on the main thread at boot,
before a worker exists), `bevy_ecs`'s `MultiThreadedExecutor` (a `std::sync::Mutex`, and the reason
`ExecutorKind`'s wasm default is deliberately left `SingleThreaded` — see
`third_party/bevy_ecs/BENILLA.md`), and `parking_lot` (reached through wgpu, which only the main
thread may touch, so its locks are never contended).

**Still conditional, and worth knowing:** `tracing-subscriber` takes locks a worker would contend
*if a worker logs*. Nothing in the threaded paths logs today — the collider build and the
`par_iter` sweeps are silent — but a `warn!` added inside one would reintroduce the hazard.

## Verifying against upstream

```sh
# 1.14.0 is the pinned version; adjust the path if cargo's registry hash differs.
diff -r third_party/async-executor ~/.cargo/registry/src/*/async-executor-1.14.0
```

That must print exactly a `diff` for `src/lib.rs` (the import) and `Cargo.toml` (the dependency),
`Only in third_party/async-executor: BENILLA.md`, and
`Only in <registry>: .cargo-ok`, `.cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig` — cargo's
packaging metadata, dropped here the same way the other `third_party` forks drop it. Anything else
crept in, and in this crate especially, anything else is a change to concurrent code that nobody
reviewed.
