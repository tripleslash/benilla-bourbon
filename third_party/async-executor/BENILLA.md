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

**One line changed, one file added. No logic is touched.**

| file | change |
|---|---|
| `src/lib.rs` | the `use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, TryLockError};` at line 54 becomes a two-arm `cfg`: `wasm_spin` on `wasm32 + atomics`, `std::sync` everywhere else |
| `src/wasm_spin.rs` | new — spinning `Mutex`/`RwLock` with the same surface. Has its own header |

Every call site — `.lock().unwrap_or_else(PoisonError::into_inner)`, `.try_read()`, the `Debug`
impls matching `TryLockError::WouldBlock` — is upstream's, unmodified. That is the whole point of
swapping the import rather than the code: the executor's algorithm, its unbounded queue, its
work-stealing and its sleeper bookkeeping are exactly the ones upstream ships and tests.

**Native is bit-for-bit upstream.** The `cfg` is false on every target that is not a shared-memory
wasm build, so `std::sync` is used and this fork is inert.

## Why spinning is the right trade here, and not in general

Every critical section guarded by these locks is a handful of instructions — a `Slab` insert or
remove, a `Vec` push, a few field updates — with no allocation under the lock and no nesting. The
worst case is a holder being descheduled mid-section, and the cost of that is a *spinning* waiter
rather than a *trapped* one. **A stall is recoverable; a trap is not.**

Two precedents, both of which this fork is deliberately imitating:

- **`event-listener` already does exactly this**, and for exactly this reason: its own `Mutex`
  (`event-listener/src/slab.rs`) spins and then gives up rather than blocking. It is why
  `async-channel` is safe on the main thread and this crate was not.
- **Rust's own wasm allocator does it too.** `library/std/src/sys/alloc/wasm.rs` spins on its global
  `dlmalloc` lock, and its comment is the clearest statement of the constraint anywhere:

  > The main thread in a web browser *cannot ever block*, no exceptions.

  Without that, every `malloc` racing a worker would have the same failure this fork fixes.

## What this does NOT fix

It does not make blocking legal — it removes it from *this* crate. Anything else the main thread
shares with a worker has to satisfy the same rule. As of this fork the audited set is:
`concurrent-queue` and `crossbeam-queue` (lock-free), `event-listener` (spins, above), the global
allocator (spins, above), `bevy_platform`'s `OnceLock`s (initialised on the main thread at boot,
before any worker exists), and `bevy_ecs`'s `MultiThreadedExecutor` (a `std::sync::Mutex`, and the
reason `ExecutorKind`'s wasm default is deliberately left `SingleThreaded` — see
`third_party/bevy_ecs/BENILLA.md`).

## Verifying against upstream

```sh
# 1.14.0 is the pinned version; adjust the path if cargo's registry hash differs.
diff -r third_party/async-executor ~/.cargo/registry/src/*/async-executor-1.14.0
```

That must print exactly a `diff` for `src/lib.rs` (the import), `Only in
third_party/async-executor/src: wasm_spin.rs`, `Only in third_party/async-executor: BENILLA.md`, and
`Only in <registry>: .cargo-ok`, `.cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig` — cargo's
packaging metadata, dropped here the same way the other `third_party` forks drop it. Anything else
crept in, and in this crate especially, anything else is a change to concurrent code that nobody
reviewed.
