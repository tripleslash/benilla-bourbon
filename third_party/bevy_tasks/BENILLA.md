# benilla's `bevy_tasks` fork — what differs from upstream, and how to check

Upstream: `bevy_tasks 0.19.1` (crates.io), MIT OR Apache-2.0 — the version bevy 0.19.1 resolves to.
Wired in through `[patch.crates-io]` in the workspace root. One patch entry covers the whole graph:
every bevy crate depends on this one, so patching it here patches `bevy_ecs`'s executor,
`bevy_asset`'s loader spawns and `bevy_app`'s `TaskPoolPlugin` at the same time.

## Why a fork exists at all

**So the browser build can have threads.** Everything else in the stack was already willing:

- `bevy_render` puts every wgpu handle behind `WgpuWrapper`, which becomes a
  `send_wrapper::SendWrapper` under `all(target_arch = "wasm32", target_feature = "atomics")`.
- `bevy_ecs`'s `ExecutorKind::MultiThreaded` is gated on `feature = "std"` alone; only its
  *default* is `target_arch`-gated, and `Schedule::set_executor_kind` is public.
- `tools/wasi-sdk` ships a `wasm32-wasip1-threads` sysroot, so the C Lua VM links beside a
  shared-memory module.

`bevy_tasks` was the one hard stop, in three separate ways, and none of them is a feature flag.

## The delta, and the command that proves it

**Three files: two hunked, one added.** Bevy 0.19 dropped its own web `Task` (it is
`async_task::Task` on every target), so 0.18's `src/task.rs` hunk has nothing left to patch.

| file | hunk | why |
|---|---|---|
| `src/lib.rs` | the `multi_threaded` alias predicate | Upstream: `all(not(target_arch = "wasm32"), feature = "multi_threaded")`. That makes "wasm" and "has no threads" the same statement, and shared memory ended that. Now `any(not(target_arch = "wasm32"), target_feature = "atomics")` — the honest question, unchanged on every other target. |
| `src/lib.rs` | `block_on` is the **spinning** `spin_block_on` on the browser's page thread (and on plain wasm), `futures_lite`'s parking one on any other thread of the threaded build (`may_park`, `third_party/may-park`), `bevy_platform::future::block_on` elsewhere | A parking `block_on` lowers to `memory.atomic.wait32`. The browser's **main thread may not execute that instruction** — it does not throw, it **traps**, and a trap takes the module down. `TaskPool::scope` calls `block_on` on whatever thread entered the scope, and on the web that is always the thread that draws. `spin_block_on` is `bevy_platform`'s own busy-wait fallback, lifted out. A Web Worker *may* park, and a spinning one burns a core: the pool's worker loop names `futures_lite`'s parking version directly, and since the render worker (which enters `scope` from render systems) `block_on` itself parks off the page. |
| `src/lib.rs` | `available_parallelism` consults `thread_spawner::hinted_parallelism()` first | On wasm `std::thread::available_parallelism` is the `unsupported` stub and answers `Err`, so upstream's `unwrap_or(1)` would build one-worker pools on an eight-core machine. The real number is `navigator.hardwareConcurrency`, which only the page can read. Unset on every other target, where upstream's line runs unchanged. |
| `src/lib.rs` | `mod thread_spawner;` + its public re-exports | see below |
| `src/task_pool.rs` | threads come from `thread_spawner::spawn` instead of `std::thread::Builder` | `std::thread::spawn` **does not exist on `wasm32-unknown-unknown` even with `+atomics`**: std gives that target `sleep` and takes `Thread`, `available_parallelism` and `yield_now` from `unsupported.rs`. A browser thread is a Web Worker, which only the page can create. The worker body itself is upstream's, unchanged. |
| `src/task_pool.rs` | a refused thread is survivable on wasm | Upstream `.expect("Failed to spawn thread.")`s. Here a refusal drops that one thread and the pool comes up smaller; on targets with real `std::thread` it still panics exactly as before. |
| `src/task_pool.rs` | `scope` forces `tick_task_pool_executor` on wasm | The other half of the same guarantee: a zero-worker pool must do the work on the calling thread rather than wait forever for a worker that never started. |
| `src/task_pool.rs` | `spawn` / `spawn_local` / `spawn_pooled` / `try_tick` | **The most important hunk. See below.** On the threaded web build `spawn` and `spawn_local` go to `web_task::spawn_local` (the microtask queue, as upstream's single-threaded pool does); `spawn_pooled` returns the worker executor's own `Task`. |
| `src/thread_spawner.rs` | new file | The spawner hook and the parallelism hint. Has its own header. |

**A dropped `Task` cancels on the web since 0.19** (upstream's own change, migration guide
"Dropping Tasks in Web Builds"). A `spawn_pooled` caller that means fire-and-forget must
`.detach()` the task.

### The `spawn` / `spawn_pooled` split, which is the whole design

On the web, `TaskPool::spawn` keeps upstream's **single-threaded** behaviour on purpose. It is not
an oversight and not a stepping stone:

`bevy_asset` spawns every load with `IoTaskPool::get().spawn(...)`. On this target an
`AssetReader::read` awaits `Chain::read_async`, which awaits `JsFuture`s and reaches `window`
(`benilla_formats::webpack::resident`). Those futures hold `JsValue`s. A `JsValue` belongs to one
agent and cannot cross to a worker at any price, and the resident pack map they read —
`window.__wenilla_packs`, all 3.3 GiB of it — lives on the main thread's JS heap by deliberate
design. **That is not a `Send` bound waiting to be relaxed; it is what the platform is.**

So the split is by name, not by inference:

- `spawn` takes a `ConditionalSendFuture` (no `Send` bound on wasm) and hands it to the browser's
  microtask queue, exactly as the single-threaded pool always did. Nothing that works today changes.
- `spawn_pooled` requires `Send` and reaches the worker threads. On every target with real threads
  it *is* `spawn`; on the web it is the difference between the two. The bound is what makes it safe
  rather than merely intended — JS-touching work does not compile here, which is the point.
- `scope` already required `Send` upstream (`Scope::spawn`), so **`par_iter_mut` and `bevy_ecs`'s
  multi-threaded executor become really parallel with no call-site change at all.**

## `block_on` was only half of the main-thread rule

The `block_on` hunk above stops *this crate* from parking the browser's main thread. It is not on
its own enough to make the main thread safe, and shipping it as though it were cost a crash in a
real browser: `async-executor`, which backs both `TaskPool`'s executor and `ThreadExecutor`, guards
its state with `std::sync::Mutex`/`RwLock`, and the main thread takes those on every `Scope::spawn`
and every tick. Under a worker per core that contends, blocks, and throws
`Atomics.wait cannot be called in this context`.

That is fixed in `third_party/async-executor` — read its `BENILLA.md` for the rule this whole
family of forks is obeying, and for the audited list of what else the main thread is allowed to
touch. **Anything added here that makes the main thread wait on a worker has to satisfy it too.**

## One trap, because it cost a full test run

**This crate has no `std` feature.** Its features are `async-io`, `async_executor`, `default`,
`futures-lite` and `multi_threaded` — nothing else. `std` reaches it as a *condition* inherited from
`bevy_platform`, and `crate::cfg::std!` is the only way to ask.

Writing `#[cfg(feature = "std")]` here compiles cleanly and is **silently always false**. The first
version of `thread_spawner.rs` gated its `std::thread::Builder` default that way, so every native
pool took the "this target has no threads" arm and panicked at `Failed to spawn thread.` — 30-odd
failures across `benilla-world`'s suite, all of them in tests that had never heard of threading.
The wasm build was unaffected and looked perfect throughout.

If a future hunk needs to ask about `std`, ask `crate::cfg::std!`.

## What is NOT being done

**This does not make wasm threads work by itself.** The crate still answers "no threads" unless a
host installs a spawner, and it deliberately contains no `wasm-bindgen`, no `web-sys` and no
knowledge of a page — which also keeps it clear of the tree's load-bearing `wasm-bindgen = "=0.2.126"`
pin. The browser half lives in benilla's own crates.

It also does not touch `conditional_send`, `usages.rs`, or `single_threaded_task_pool.rs`. Native
behaviour is unchanged in every hunk above; each one either reads a `target_feature` that is off, or
routes through a `thread_spawner` whose default *is* `std::thread::Builder`.

## Verifying against upstream

```sh
# The patched crate is never downloaded into cargo's registry, so fetch the pinned upstream.
mkdir -p /tmp/upstream && curl -sL https://static.crates.io/crates/bevy_tasks/bevy_tasks-0.19.1.crate   | tar xz -C /tmp/upstream
diff -r third_party/bevy_tasks /tmp/upstream/bevy_tasks-0.19.1
```

That must print exactly:

- `diff` for `src/lib.rs` and `src/task_pool.rs` — the hunks in the table, each carrying a
  `**benilla fork.**` comment saying why it is there;
- `Only in third_party/bevy_tasks/src: thread_spawner.rs` and `Only in third_party/bevy_tasks:
  BENILLA.md`;
- `Only in <upstream>: .cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig` — cargo's own
  packaging metadata, dropped here the same way `third_party/kira` drops it.

Anything else crept in. The two hunked files are **rustfmt-formatted** (`--edition 2024`), which
reorders one `use` list in `task_pool.rs`; that is the only cosmetic difference in the set.
