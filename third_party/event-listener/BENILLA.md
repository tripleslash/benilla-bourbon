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

**One `use` line changed, one file added. No logic is touched.**

| file | change |
|---|---|
| `src/lib.rs` | `mod sync`'s `pub(super) use std::sync::{Mutex, MutexGuard};` becomes a two-arm `cfg` — `crate::wasm_spin` on `wasm32 + atomics`, `std::sync` everywhere else — plus `mod wasm_spin;` |
| `src/wasm_spin.rs` | new — a spinning `Mutex` with the slice of the surface `intrusive.rs` uses. Has its own header |

`intrusive.rs` is untouched: its `.lock().unwrap_or_else(|e| e.into_inner())`, its `try_lock().ok()`
and its `MutexGuard<'a, Inner<T>>` all still compile because the stand-in keeps that shape.

**Native is bit-for-bit upstream** — the `cfg` is false anywhere that is not a shared-memory wasm
build.

## The trade

Spinning instead of sleeping. The critical sections here are a linked-list insert, remove or walk —
no allocation under the lock, no nesting, no I/O — so a waiter spins for nanoseconds. The worst case
is a holder descheduled mid-section, costing a spinning waiter rather than a trapped one. **A stall
is recoverable; a trap is not.** Rust's own wasm allocator makes the same trade for the same reason
(`library/std/src/sys/alloc/wasm.rs`: *"the main thread in a web browser cannot ever block, no
exceptions"*).

## Verifying against upstream

```sh
# 5.4.1 is the pinned version; adjust the path if cargo's registry hash differs.
diff -r third_party/event-listener ~/.cargo/registry/src/*/event-listener-5.4.1
```

That must print exactly a `diff` for `src/lib.rs`, `Only in third_party/event-listener/src:
wasm_spin.rs`, `Only in third_party/event-listener: BENILLA.md`, and `Only in <registry>:
.cargo-ok`, `.cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig`. Anything else crept in — and in
this crate, anything else is a change to concurrent code nobody reviewed.
