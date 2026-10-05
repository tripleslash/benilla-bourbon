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

Search for `benilla fork` to find every hunk.

## Verifying diff

```
diff -ru <crates.io bevy_winit-0.19.1> third_party/bevy_winit
```

The diff shows only the `src/state.rs` hunks above, plus this file.
