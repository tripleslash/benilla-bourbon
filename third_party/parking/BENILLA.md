# benilla's `parking` fork — what differs from upstream, and how to check

Upstream: `parking 2.2.1` (crates.io), Apache-2.0 OR MIT. Wired in through `[patch.crates-io]` in
the workspace root. The third of the sibling forks — with `third_party/async-executor` and
`third_party/event-listener` — obeying the same rule:

> On `wasm32 + atomics`, the browser's main thread may not execute `memory.atomic.wait32`. Any
> blocking lock it takes while a worker holds the other side throws
> `Atomics.wait cannot be called in this context` and kills whatever was running.

## Why a fork exists at all

`futures_lite::future::block_on` parks on this crate, and that is how every `bevy_tasks` worker
thread sleeps when the pool is idle — which is correct and must stay that way. **A worker may
block.** The problem is the other end.

`Unparker::unpark` is called by whichever thread just made work available. `async-executor` wakes a
sleeping worker from the thread that spawned the task, so on the threaded browser build **that
thread is routinely the page's main thread**. And `unpark`'s last step is:

```rust
drop(self.lock.lock().unwrap());
self.cvar.notify_one();
```

That `lock()` is the only blocking call an unparking thread makes, and its purpose is narrow —
upstream's own comment explains it: the parker sets `PARKED` and then calls `cvar.wait`, and a
notification sent in that window would be lost, so the unparker acquires the lock to wait the window
out. On the main thread it throws instead.

## The delta, and the command that proves it

**One import.**

| file | change |
|---|---|
| `src/lib.rs` | `Mutex` comes from `may_park::sync` (`third_party/may-park`) on `wasm32 + atomics`, `sync::Mutex` everywhere else; every call site is upstream's |
| `Cargo.toml` | the `may-park` path dependency, on that target only |

`may-park`'s `Mutex` is std's own with std's guard; only `lock` asks whether the calling thread may
park, and loops on `try_lock` on the one that may not, the browser page's. So `unpark`'s
acquire-and-release, which closes the window before a parker's `Condvar::wait`, spins when the page
wakes a worker and parks as upstream's does when a worker wakes one, and `park`'s `cvar.wait(guard)`
is unchanged.

The first version of this fork (2026-09) spun on `try_lock` in `unpark` on every thread instead;
a worker may park, and parking is the cheaper wait for it.

## Verifying against upstream

```sh
# 2.2.1 is the pinned version; adjust the path if cargo's registry hash differs.
diff -r third_party/parking ~/.cargo/registry/src/*/parking-2.2.1
```

That must print exactly a `diff` for `src/lib.rs` and `Cargo.toml`, `Only in third_party/parking:
BENILLA.md`,
plus `Only in <registry>: .cargo-ok`, `.cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig`.
