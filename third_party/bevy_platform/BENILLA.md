# benilla's `bevy_platform` fork — what differs from upstream, and how to check

Upstream: `bevy_platform 0.19.1` (crates.io), MIT OR Apache-2.0. Wired in through `[patch.crates-io]`
in the workspace root, beside the other threading forks (`third_party/bevy_tasks`, `bevy_ecs`,
`bevy_render`, `bevy_winit`).

## The one delta: `Mutex` and `RwLock` do not park on the browser page's thread

`src/sync/mutex.rs` and `src/sync/rwlock.rs` choose their implementation by `cfg`. On
`wasm32 + atomics` (with `std`) they re-export `third_party/may-park`'s: std's own locks and guards,
whose `lock`/`read`/`write` loop on `try_*` on a thread that may not park and park as std's do on
every other. `src/sync/mod.rs` re-exports `may_park::forbid_parking_on_this_thread`, which the page
calls once (`benilla_app::web_threads::install`), and `Cargo.toml` gains the path dependency on that
target.

**Why.** A contended `std::sync` lock parks with `memory.atomic.wait32`, and on the page's thread
that instruction traps (`Atomics.wait cannot be called in this context`). With the render app on its
own Web Worker (`third_party/bevy_render/BENILLA.md`, "The render worker") the page and the worker
share bevy's locks, and they contend: bevy_pbr's material `prepare_asset` and benilla's static_gx
prepare call `AssetServer::load` on the worker, which write-locks the server's `infos`, while the
page's `handle_internal_asset_events` takes the same lock. Found live: the page trapped there a few
seconds after the login screen came up. Workers keep std's parking path: one that spun would burn
its core while the holder runs.

The same `may-park` locks replace the all-threads spin locks the `async-executor`, `event-listener`
and `parking` forks used to carry, and decide `bevy_tasks::block_on`. Everything else in
`bevy_platform::sync` (`Once`, `OnceLock`, `LazyLock`, `Barrier`, the poison types, `Arc`) is
unchanged.

## Verifying against upstream

```sh
mkdir -p /tmp/upstream && curl -sL https://static.crates.io/crates/bevy_platform/bevy_platform-0.19.1.crate | tar xz -C /tmp/upstream
diff -r third_party/bevy_platform /tmp/upstream/bevy_platform-0.19.1
```

That prints the `cfg` hunks in `src/sync/{mutex,rwlock,mod}.rs`, the `Cargo.toml` dependency, `Only in third_party/bevy_platform: BENILLA.md`, and cargo's packaging
metadata only upstream (`.cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig`).
