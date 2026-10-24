# benilla's `bevy_ecs` fork — what differs from upstream, and how to check

Upstream: `bevy_ecs 0.19.1` (crates.io), MIT OR Apache-2.0 — the version bevy 0.19.1 resolves to.
Wired in through `[patch.crates-io]` in the workspace root. It is the companion to
`third_party/bevy_tasks`, and neither is any use without the other.

## Why a fork exists at all

`bevy_tasks` was forked so the browser build can have worker threads. This one is forked because
**`Query::par_iter_mut` would not use them.** `query/par_iter.rs` and `query/state.rs` decide
between the parallel and the serial path by asking `target_arch == "wasm32"`, which was a correct
proxy for "has no threads" right up until the browser got shared memory.

Left un-forked, the fork of `bevy_tasks` would buy almost nothing: the two sweeps in this client that
were *written* for parallelism would keep running serially on the thread that draws —

- `benilla_world::model_render::visibility` — every model submesh in residency, ~100k in a city.
  Its own comment records that the serial walk "blew half the 16.7 ms budget (the Stormwind fps
  hunt)".
- `benilla_world::rig_anim::pose::evaluate_rig_poses` — ~400 rigs / ~12k bones, 0.4 ms of one
  thread on the critical path.

## The delta, and the command that proves it

**13 `#[cfg]` predicates, in two files, and one logic change (below).** Each predicate is one of
two mechanical rewrites:

| upstream | fork |
|---|---|
| `any(target_arch = "wasm32", not(feature = "multi_threaded"))` | `any(all(target_arch = "wasm32", not(target_feature = "atomics")), not(feature = "multi_threaded"))` |
| `all(not(target_arch = "wasm32"), feature = "multi_threaded")` | `all(any(not(target_arch = "wasm32"), target_feature = "atomics"), feature = "multi_threaded")` |

| file | sites |
|---|---|
| `src/query/par_iter.rs` | 3 serial-arm gates, 6 parallel-arm gates |
| `src/query/state.rs` | 4 parallel-arm gates (one `use`, three methods) |

Both files carry a `**benilla fork.**` comment at their first rewritten site saying why.

**One logic change, in `src/query/state.rs`'s `update_archetypes_unsafe_world_cell`.** When a query
has required components and the world has new archetypes, upstream walks every archetype that holds
the rarest required component and skips the already-seen ones one by one. The fork first compares
how many archetypes are new with the size of that set, and when fewer are new it tests only those,
as the no-required-components arm already does (`new_archetype` checks the match). Measured in a
crowd on the threaded browser build (JS self-profiler, page thread): ~67% of the samples in frames
of 100 ms or more sat in this walk, across dozens of queries, because every new archetype made each
of ~3,400 systems' queries re-walk hundreds of archetypes.

## What is deliberately NOT forked

- **`src/schedule/executor/mod.rs` — `ExecutorKind`'s default stays `SingleThreaded` on wasm.**
  Not an oversight, and not laziness: `benilla_app::run` sets the kind explicitly for `Update` and
  `PostUpdate` anyway, and it sets **single**-threaded, by measurement. Decision 1366 graded the
  multi-threaded executor at **−1.30 cpu_ms in favour of single-threaded** because ~60% of this
  app's per-frame `Update` systems are non-`Send` (mlua's `UiScript`, kira's audio handles) and
  serialize through the executor's one `local_thread_running` flag regardless; decision 1437 graded
  `PostUpdate` the same way at −0.49. A wasm build has fewer cores and the same non-`Send` majority,
  so flipping the default here would be shipping a configuration this project already measured as
  worse. `WOW_MT_UPDATE=1` / `WOW_MT_POSTUPDATE=1` remain the A/B levers if anyone wants to re-open
  it on the web.
- **`src/message/iterators.rs` and `src/message/mut_iterators.rs`.** They carry the same
  `target_arch` proxy for parallel message iteration. Nothing in this client uses it, and a serial
  fallback is correct rather than merely acceptable, so they stay upstream's.
- **`PipelinedRenderingPlugin`** needs nothing: `bevy_internal`'s
  `#[custom(cfg(all(not(target_arch = "wasm32"), feature = "multi_threaded")))]` is still a bare
  `target_arch` test, so the plugin stays out of the graph on wasm — which is what we want. The
  render app must not leave the main thread: `bevy_render`'s `WgpuWrapper` is a
  `send_wrapper::SendWrapper` under `+atomics`, and a `SendWrapper` touched or dropped off its
  creating thread panics.

## Verifying against upstream

```sh
# The patched crate is never downloaded into cargo's registry, so fetch the pinned upstream.
mkdir -p /tmp/upstream && curl -sL https://static.crates.io/crates/bevy_ecs/bevy_ecs-0.19.1.crate   | tar xz -C /tmp/upstream
diff -r third_party/bevy_ecs /tmp/upstream/bevy_ecs-0.19.1
```

That must print exactly a `diff` for `src/query/par_iter.rs` and `src/query/state.rs`, plus
`Only in <upstream>: .cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig` — cargo's packaging
metadata, dropped here the same way `third_party/kira` and `third_party/bevy_tasks` drop it, and
`Only in third_party/bevy_ecs: BENILLA.md`. Anything else crept in.
