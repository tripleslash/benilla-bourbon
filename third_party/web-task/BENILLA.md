# benilla's `web-task` fork — what differs from upstream, and how to check

Upstream: `web-task 1.1.3` (crates.io), MIT OR Apache-2.0, the version `bevy_tasks 0.19.1` resolves
to. Wired in through `[patch.crates-io]` in the workspace root. Only the `+atomics` runtime
(`src/runtime/multithread.rs`) changes; the single-threaded build never compiles it.

## The bug

`spawn_local` parks a `!Send` task between polls on `Atomics.waitAsync(state, WAITING)` and chains
one `Closure` owned by the task's `LocalRunnable` onto that promise. After the poll that completes
the future, `run_to_completion` still sets up a wait whenever the `Task` handle is alive (bevy polls
its tasks from systems, so it usually is). Nothing ever notifies that waiter. When the handle drops,
the `LocalRunnable` is freed and its closure with it, and the waiter stays queued on a freed address.

The next allocation at that address inherits the waiter. A new task's `schedule`, or a std futex
that lands there, does `memory.atomic.notify(addr, 1)`. Waiters are woken oldest first, so the
stale one gets the wake-up. Its closure throws `closure invoked recursively or after being dropped`
(an `Uncaught (in promise)` in the console, through `__wbindgen_throw`). The waiter the notify was
meant for stays asleep, and its task is never polled again.

## The delta

| file | change |
|---|---|
| `src/runtime/multithread.rs` | `LocalRunnable` loses its `continuation` field. Each wait chains a `Closure::once` that holds only a `Weak` and hands it to the JS GC (`into_js_value`). `impl Drop for LocalRunnable` notifies every waiter on `state`. |

The `Weak` keeps the allocation, and so the address, reserved until its waiter has fired. The
`Drop` notify makes sure that waiter fires. When it does, the `Weak` no longer upgrades and the
continuation returns without touching anything. No `JsValue` is stored in the runnable any more,
which also stops a `Task` dropped on another thread from dropping a closure outside its own heap.

## Verifying against upstream

```sh
diff -r third_party/web-task ~/.cargo/registry/src/*/web-task-1.1.3
```

That must print exactly a `diff` for `src/runtime/multithread.rs` and
`Only in third_party/web-task: BENILLA.md`, plus `Only in <registry>: .cargo-ok`,
`.cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig`.
