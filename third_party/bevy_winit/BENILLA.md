# benilla's `bevy_winit` fork — what differs from upstream, and how to check

Upstream: `bevy_winit 0.19.1` (crates.io), MIT OR Apache-2.0. Wired in through `[patch.crates-io]`
in the workspace root.

## The one delta: the runner keeps its per-frame query and system states

`WinitAppRunnerState` (`src/state.rs`) built three pieces of ECS state afresh on calls that happen
every frame:

- `window_event` ran `world.query::<(&mut Window, &mut CachedWindow)>()` after **every** window
  event to refresh the window cache — and `RedrawRequested` is a window event, so this is at least
  one per frame;
- `redraw_requested` built a `SystemState<(Res<WinitSettings>, Query<(Entity, &Window)>)>` to ask
  whether any window is focused;
- `about_to_wait` built a `SystemState<CreateMonitorParams>`.

A new `QueryState` initialises by matching **every archetype in the world**, and benilla's world has
thousands. In the browser world viewer (JS Self-Profiling, Elwynn) the first of these alone was ~3–4%
of the frame thread. The fork keeps all three on the runner, built once in `new` (the monitor state
on first use): a kept query state only matches the archetypes added since it last ran.

Semantics are unchanged. The cache refresh uses a dedicated `QueryState`, not the runner's existing
`windows_system_state`, because `QueryState::get_mut(world, …)` reads change ticks against the
world's last change tick exactly as a fresh `world.query()` did — a `SystemState` would compare
against its own last run instead. The focus read is read-only and copies `WinitSettings` out of the
borrow, since `redraw_requested` reads it again after mutating the runner.

## The second delta: a hook after the update

`AfterUpdate(fn(&mut World))` (`src/lib.rs`), a resource the runner looks up after each
`App::update` in `run_app_update` (`src/state.rs`) and calls with the main world. It runs in the
same task as the update, which on the web is the same `requestAnimationFrame` callback, after the
sub-apps have run. Frame generation (`benilla-world/src/fsr/cadence.rs`) uses it in the browser,
where there is no render thread: a held frame presents and then runs the next game update behind
that present, so the world frame after it only extracts and renders. Absent, nothing changes.

## The third delta: who finishes the plugins

`set_finish_plugins(fn(&mut App))` (`src/lib.rs`): when installed, the runner calls it instead of
`App::finish` + `App::cleanup` (both call sites, `new_events` and `winit_runner`, in `src/state.rs`).
The browser's render worker (`third_party/bevy_render/BENILLA.md`, eighth delta) installs one that
runs both on the worker owning the GPU device, since plugins' `finish` reads it and builds render
resources. Absent, nothing changes.

## The fourth delta: wake the event loop from another thread

`wake_event_loop()` (`src/lib.rs`) sends `WinitUserEvent::WakeUp` through a proxy kept from
`WinitPlugin::build`. The browser's render worker makes the GPU device off the page, and the runner
re-checks its plugins' readiness only when its event loop runs; an unfocused page sleeps through the
device's arrival without it (a black canvas until the first click).

## The fifth delta: forced startup updates wait for the app

`redraw_requested` (`src/state.rs`) spends `startup_forced_updates` only once
`plugins_state() == Cleaned`. Upstream spends one per loop iteration from the first, and before the
plugins are cleaned up `run_app_update` does nothing, so a renderer that becomes ready after the
loop is already running leaves none: an unfocused (reactive) window then sleeps until an input
event. Natively the device is made by `block_on` before the loop starts, so nothing is spent early;
the browser's render worker makes it off the page while the loop runs (found live: a black canvas
until the first click). For the same reason `new_events` treats the iteration that finishes the
plugins as an elapsed wait: woken by an event it reads as not elapsed, and a reactive window arms its
update timer only on an elapsed wait.

## The sixth delta: an update the app holds

`set_may_update` (`src/lib.rs`) installs a hook the runner asks, with the main world, before each
update (`run_app_update`, `src/state.rs`). `false` holds that update: nothing of it is consumed, the
runner asks for the next redraw, and the held update runs at the next frame whatever the update mode
(`update_deferred`). The browser's render worker uses it to pay an extract the last update owed
before the next one starts, without the page's thread waiting on the worker (which, between its
tasks, can be waiting on the page). Unset, every update runs. What it holds is settled even when no
update follows (a reactive window) by the seventh delta, which lets the worker take the extract while
the page waits.

## The seventh delta: the world lent between callbacks (web)

`set_lend_world` (`src/lib.rs`) installs two hooks, and on `wasm32` the runner state is spawned
inside `Lending` (`src/state.rs`), which calls `reclaim` before each `ApplicationHandler` callback and
`lend`, with the main world, after it (once the plugins are cleaned). Between the two the page is in
the browser's event loop and the runner does not touch the world, so the browser's render worker may
extract from it there: an update that ended while the worker was still drawing the last one is
extracted as soon as the worker is free, and the next update is not held for it. `exiting` only
reclaims. Unset, both are no-ops and `Lending` is the runner state as upstream.

Search for `benilla fork` to find every hunk.

## Verifying diff

```
diff -ru <crates.io bevy_winit-0.19.1> third_party/bevy_winit
```

The diff shows only the `src/state.rs` hunks above, `AfterUpdate`, the finish hook, `wake_event_loop`, `set_may_update` and `set_lend_world` in `src/lib.rs`, plus this file.
