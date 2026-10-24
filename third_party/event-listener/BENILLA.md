# benilla's `event-listener` fork — what differs from upstream, and how to check

Upstream: `event-listener 5.4.1` (crates.io), Apache-2.0 OR MIT. Wired in through
`[patch.crates-io]` in the workspace root. One of three sibling forks — with
`third_party/async-executor` and `third_party/parking` — that exist for a single rule:

> On `wasm32 + atomics`, the browser's main thread may not execute `memory.atomic.wait32`. Any
> blocking lock it takes while a worker holds the other side throws
> `Atomics.wait cannot be called in this context` and kills whatever was running.

## Why a fork exists at all

**This is the least obvious of the three, because it does not look like a lock at all.**

`Event` is what the smol family parks and wakes on. `async-channel` is built on it, so **every send
and receive that has to register or notify a listener takes this crate's mutex**. In benilla's
threaded browser build the page's main thread and a Web Worker per core sit on opposite ends of
those channels: the task pools' shutdown channel, every `Task` handed back by
`TaskPool::spawn_pooled`, and the app's own net and asset plumbing. Contention there is ordinary
traffic, not a corner case.

**And the crate reads as though it were already safe.** `src/slab.rs` carries its own spinning
`Mutex`, written for exactly this hazard — which is what made the first audit of this stack wrongly
conclude that `async-channel` was fine on the main thread. It is not: `slab.rs` is the `no_std`
path. With `std` on (which is always, here) `lib.rs` selects `intrusive.rs`, and that is backed by
`std::sync::Mutex` on **every** target, wasm included.

Symptom before the fork: a sporadic `Atomics.wait cannot be called in this context` with the whole
page frozen, and — because a trap out of a polled future kills the task that was being polled — a
login that opened its WebSocket and then never wrote a byte.

## The delta, and the command that proves it

**One `use` line changed. No logic is touched.**

| file | change |
|---|---|
| `src/lib.rs` | `mod sync`'s `pub(super) use std::sync::{Mutex, MutexGuard};` becomes a two-arm `cfg`: `may_park::sync` (`third_party/may-park`) on `wasm32 + atomics`, `std::sync` everywhere else |
| `Cargo.toml` | the `may-park` path dependency, on that target only |

`may-park`'s `Mutex` is std's own with std's guard; only `lock` asks whether the calling thread may
park, and loops on `try_lock` on the one that may not, the browser page's. `intrusive.rs` is
untouched: its `.lock().unwrap_or_else(|e| e.into_inner())`, its `try_lock().ok()` and its
`MutexGuard<'a, Inner<T>>` are std's.

**Native is bit-for-bit upstream** — the `cfg` is false anywhere that is not a shared-memory wasm
build.

## The trade

The page spins, every other thread parks. The first version of this fork (`src/wasm_spin.rs`,
2026-09) spun on every thread; a Web Worker may park, and one that spins burns its core while the
holder runs (or is descheduled). On the page a stall is recoverable and a trap is not, which is the
trade Rust's own wasm allocator makes (`library/std/src/sys/alloc/wasm.rs`: *"the main thread in a
web browser cannot ever block, no exceptions"*).

## Verifying against upstream

```sh
# 5.4.1 is the pinned version; adjust the path if cargo's registry hash differs.
diff -r third_party/event-listener ~/.cargo/registry/src/*/event-listener-5.4.1
```

That must print exactly a `diff` for `src/lib.rs` and `Cargo.toml`, `Only in
third_party/event-listener: BENILLA.md`, and `Only in <registry>:
.cargo-ok`, `.cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig`. Anything else crept in — and in
this crate, anything else is a change to concurrent code nobody reviewed.
