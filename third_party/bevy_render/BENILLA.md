# benilla's `bevy_render` fork — what differs from upstream, and how to check

Upstream: `bevy_render 0.19.1` (crates.io), MIT OR Apache-2.0. Wired in through `[patch.crates-io]`
in the workspace root. The fourth and last of the threading forks, after `third_party/bevy_tasks`,
`third_party/bevy_ecs` and `third_party/async-executor`.

## Why a fork exists at all

**Two `ComputeTaskPool::scope`s hand wgpu handles to worker threads, and upstream guards every other
one.**

Under `wasm32 + atomics`, `WgpuWrapper` is a `send_wrapper::SendWrapper`, which panics the moment it
is dereferenced from a thread other than the one that created it:

```
Dereferenced SendWrapper<T> variable from a thread different to the one it has been created with.
```

That check is not paranoia and it must not be defeated. A wgpu handle on this target is a JS object
reached through wasm-bindgen's heap table, and **that table is per-thread** — the same index means
something different, or nothing, on a Web Worker. Sharing the handle does not merely race; it reads
the wrong object.

Upstream knows this: `src/renderer/wgpu_wrapper.rs` is where the `SendWrapper` comes from, and
since 0.19 the render graph is a schedule of systems whose command encoders live in the render
thread's `RenderContext`, so no command-buffer generation leaves that thread.

What it misses is `src/batching/gpu_preprocessing.rs`. `write_batched_instance_buffers` and
`write_indirect_parameters_buffers` take `render_device: Res<RenderDevice>` and
`render_queue: Res<RenderQueue>`, reborrow them as `&*`, and hand those references straight into
`scope.spawn`. On native that is correct and fast — the pool's threads run it and wgpu is genuinely
`Send`. On a threaded browser build it is the panic above, on the first frame that draws anything.

## The second delta: `Skip` cameras keep their output attachment

`prepare_view_attachments` in 0.19.1 skips every `CameraOutputMode::Skip` camera, and
`prepare_view_targets` then removes the `ViewTarget` of any camera with no output attachment and a
clear colour: such a camera never renders. 0.18.1 attached every camera's target. benilla's world
camera (its target a size-carrier image) and player-UI camera (the window; `ui_gamma` writes the
swapchain itself) are both `Skip` and clear, so on unpatched 0.19.1 the client and
`benilla-worldview`'s FSR mode draw nothing (found live, `WOW_FSR_DEBUG=views`: the world view
`target=false`). The hunk removes the three-line `Skip` early-out, restoring 0.18.1's loop.

Its twin is in `prepare_windows`, which 0.19.1 lets acquire a swap-chain texture only for a window
a `Write` camera targets (a Quest 3 fence-leak fix). At the glue screens the player-UI camera is
the window's only camera, so after the initial present the window had no texture and the browser
client stayed black at 140 fps (found live, `?diag=1`). The hunk drops the `Write` condition: any
camera targeting the window counts.

## The third delta: per-task canvas texture (web)

`extract_windows` keeps a swap-chain texture that was not presented for the next update, an
optimization that holds natively. A WebGPU canvas texture lives only until the end of the task
that acquired it, so on the web the next update submits a destroyed texture ("Destroyed texture
… used in a submit") and Bevy quits on the validation error. Frame generation's held frames are
updates that do not present the world view, which is how it was found (`?frame_generation=1`).
The hunk, `#[cfg(target_arch = "wasm32")]` only, drops the unpresented texture and its view, so
`prepare_windows` acquires a fresh one every update.

## The fourth delta: slab buffers are destroyed when let go (web)

The mesh allocator grows a slab by creating a larger buffer, copying the old one into it and
dropping the old one, and drops an emptied slab's buffer. Natively a drop frees the buffer once its
last use completes. The browser backend's drop does nothing (`wgpu`'s `WebBuffer` has a no-op
`Drop`), so the memory waits for the page's garbage collector to reach the `GPUBuffer` wrapper, and
Firefox's collector does not weigh GPU memory: while `?worldview=1` streamed Elwynn in, Firefox 155
created ~1.4 GB of mesh slabs, most of them abandoned by growth and none freed. The hunk,
`#[cfg(target_arch = "wasm32")]` only, calls `destroy()` on the buffer at both points. Both are in
`PrepareAssets`, before the frame encodes a pass, and the grow copy is submitted first.

## The fifth delta: pipelines are created asynchronously (web)

`render_resource/pipeline_cache.rs`, `#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]`
only. The shader modules and the layout are still made on the frame thread, but the pipeline is
asked for with wgpu's `create_render_pipeline_async` / `create_compute_pipeline_async`
(`third_party/wgpu`, `createRenderPipelineAsync`), and stays `CachedPipelineState::Creating` on a
`spawn_local` task until the browser's promise settles — the state native async compilation already
uses, so every consumer already skips a draw whose pipeline is not ready. The synchronous form
returned at once but compiled on the GPU process's queue, stalling every frame behind it (measured in
Chrome on D3D12: ~600 ms for one heavy fragment shader, against ~0 queue stall for the async form).
`block_on_render_pipeline` leaves such a pipeline pending rather than spinning on a promise that can
only settle once the frame returns; the upscaling blit, its one caller, skips its view until it
lands. `synchronous_pipeline_compilation` still forces the synchronous path.

## The sixth delta: a mesh's buffer slice is one lookup

`slab_allocator.rs` keeps every resident allocation's slab and element range in one map
(`SlabAllocator::resident_slice`), filled when the data becomes resident and dropped when it is
freed; a slab's growth copies at the same offsets, so an entry stays valid until its key is freed.
`mesh/allocator.rs`'s `mesh_vertex_slice` and `mesh_index_slice` — asked once per draw each — read
it instead of `key_to_slab`, the slab table and the slab's own allocation table, which hashed the
`MeshAllocationKey` twice per slice. Measured in the browser world viewer: those lookups were ~8% of
the frame thread.

## The seventh delta: the upload budget never holds a mesh back

`RenderAssetBytesPerFrame` (benilla sets it in the browser, `WEB_UPLOAD_BUDGET_MB`) makes
`prepare_assets` push an asset to a later frame once the frame's bytes are spent, and for an asset
that is being updated it removes the prepared one first. For an image that spreads a real upload.
For a mesh it spreads nothing: `allocate_and_free_meshes` runs before `prepare_assets::<RenderMesh>`
and copies every extracted mesh into its slab whatever the budget says, so only the `RenderMesh`
record waits, and while it waits nothing that draws the mesh is drawn. The player UI rewrites its
pooled batch meshes in place, so a rebuild on a frame whose budget the streaming world had spent
took the whole UI off the screen until the backlog cleared. A new mesh that waited had a second
cost: its entity is specialized only on the frame it is marked dirty, and the mesh was not there.

The hunk adds `RenderAsset::DEFERRABLE` (default `true`), which `prepare_assets` asks before both
deferrals, and `RenderMesh` says `false`. A mesh's bytes still count against the frame, so images
prepared after it wait as before.

## The eighth delta: the render worker (web, threaded)

Everything gated `all(target_arch = "wasm32", target_feature = "atomics")`; off that target, and on
it while the mode is off, the code paths are upstream's. `src/render_worker.rs` has the mechanism in
its header; this is the list of seams.

**The shape.** The page's one thread ran main, extract and `Render` in series. `pipelined_rendering`
cannot split them here: it moves the render `SubApp` across threads twice a frame and its handshake
blocks, and every wgpu handle is a `SendWrapper` around a JS object of one agent. So the render
world never moves. `benilla_app::render_worker` starts a dedicated worker (same module, same memory,
`web/renderworker.js`) and transfers the canvas to it as an `OffscreenCanvas`; device creation,
every plugin `finish`/`cleanup`, extract and the render schedule run there. For extract the *main*
world visits: the page parks a pointer, flips an atomic to `EXTRACT`, and spins only once the worker
has claimed it (`TAKEN`), for the extract itself. The main world's contents are `Send + Sync` by
trait bound; `NonSend` resources keep bevy's own thread check.

The page never waits on the worker's event loop. Measured in Chrome: after a task that submitted a
frame, the worker ran no task at all (no message, no timer) for as long as the page spun, 20 s, so a
page spinning on it waits for itself. An extract the worker does not claim within 4 ms is withdrawn;
after a render the worker is `SETTLING` until its next task marks it `IDLE` (`settle`, posted by
`web/renderworker.js`). An update that ends without an extract leaves it owed. The page then lends
its world to the worker while it waits in the browser's event loop (`lend`/`reclaim`, `bevy_winit`'s
`set_lend_world`, seventh delta there): the worker takes it as soon as its render is done, from its
settle or a wake, and the page takes it back before its next callback touches it, waiting only while
the worker is inside the extract. `set_may_update` (`before_update`) still holds the next update until
the extract is paid, which now happens only when the loan went untaken and the worker is still busy.
Timed (`?rwtrace=1`, Wailing Caverns at 2x, 144 Hz): the worker's settle lands ~0.07 ms after its
render, and a failed hand-over found it still *rendering* (the update ended early after a late
animation frame); held, that update chained into the next frame, a third of updates were owed
(`owed≈300 skipped≈330` per 5 s, ~108 renders/s). Lent, the worker takes those at its settle:
`skipped` 0-5 per 5 s, one render per update. The worker's host (`web/renderworker.js`) also settles
only once the GPU has at most two of its frames in hand (`onSubmittedWorkDone`): a GPU slower than the
display let the canvas run ~5 frames ahead (44-48 ms from a frame's submit to its completion, on the
page leg too); held to two it is as busy (~114 frames/s there, against ~113 on `?renderworker=0`),
the latency halves to ~24 ms, and the page holds its update while the GPU is behind instead of
queueing it. It is never dropped: bevy's messages (the `AssetEvent`s
`extract_render_asset` reads, removed components) live two updates, so an update that was never
extracted would lose asset changes, and every main update stays extracted exactly once and before
the next one runs, as on one thread.

| file | change |
|---|---|
| `src/render_worker.rs` | new — the state machine, the job queue, `run_blocking` for the runner's `finish`/`cleanup`, `RenderWorkerPlugin` (parks the `RenderApp` on the worker in `cleanup`, leaves `RenderWorkerExtractApp` and its non-blocking extract on the page), `before_update` (the runner's hold while an extract is owed), `lend`/`reclaim` (the world lent while the page waits), `settle` (which also takes a lent world), the page-side buffer creation and write queue, the graveyard, the stats `webdiag` prints, and `?rwtrace=1`'s timeline (`stamp`/`trace`, read by `__wenilla_render_trace`) |
| `src/lib.rs` | `pub mod render_worker`; `RenderPlugin::build` adds `RenderWorkerPlugin` when active, and does not insert the `TimeReceiver` (the main app keeps its own clock; the worker's `send_time` then finds the channel closed, which it ignores) |
| `src/settings.rs` | `create_render` posts `initialize_renderer` to the worker when active (no surface: the browser's adapter request has no use for one), marks its stages (`DEVICE_AT`, read by `describe`) and wakes the page when it has published the device (`set_wake_page`, `bevy_winit::wake_event_loop`); `unpack_into` records the device for page-made buffers |
| `src/view/window/mod.rs` | `create_surfaces` makes the surface from the worker's `OffscreenCanvas` (the window's raw handle names a page-heap object); the upstream path is wrapped in an `unwrap_or_else` and otherwise unchanged |
| `src/renderer/wgpu_wrapper.rs` | the value sits in a `ManuallyDrop<Held>`: `Local` (a `SendWrapper`) or `Shared` (`new_send`, for plain `Send + Sync` data). A `Local` dropped off its thread is buried and dropped on the worker after its next render, instead of panicking. Access off-thread still panics, naming the type (`used_off_thread`, `#[track_caller]`) |
| `src/renderer/render_context.rs` | `RenderContextState::rebind`: a system's `Deferred<RenderContextState>` made on the page (its schedule initialized during `build`: `remove_systems_in_set` initializes a changed schedule, which `benilla_world::ffx_glow` does to `Core2d`) is still empty, and the worker swaps in its own at first use |
| `src/diagnostic/internal.rs` | `bevy_platform::sync::Mutex` for std's: the page's `sync_diagnostics` and the worker's `finish_frame` share `RenderDiagnosticsMutex`, and a std lock the page finds held parks, which traps there (found live: `Atomics.wait cannot be called in this context` at the login screen) |
| `src/render_resource/buffer.rs` | `Buffer` holds an `Arc` of a lazy cell: a clone never touches wgpu (cloning a `SendWrapper` dereferences it), and a buffer the page creates is made on the worker at first use |
| `src/renderer/render_device.rs` | `features()`/`limits()` cached at construction; `create_buffer[_with_data]` off the worker returns a pending buffer |
| `src/renderer/mod.rs` | `RenderAdapterInfo` is built with `WgpuWrapper::new_send` (plain data the page reads: `benilla_app::cvars::sync_cvars`); `RenderQueue::write_buffer` (an inherent method, so it shadows `wgpu::Queue`'s through `Deref`) queues a page write, applied right after the next extract; `QueueWriteTarget` takes a bevy or a wgpu buffer by any reference |

**Why the page-side seams exist.** benilla's main world creates and writes buffers itself: the
shared light buffer at `Startup` (`benilla_world::assets`) and the UI-model light pool
(`benilla_app::ui_models`, `pane_light`), whose buffers sit in materials. Of the ~56
`Res<RenderDevice|RenderQueue>` in benilla's crates those are the only main-world ones; the rest run
in the render app. A write the page queues lands after the extract that closes the page update it was
made in, before that frame's own writes and submits: the order it had on one thread.

**Audited.** Every `Plugin::finish`/`cleanup` in the graph (bevy_render, bevy_pbr,
bevy_core_pipeline, bevy_anti_alias, bevy_gltf, bevy_egui, benilla's `gpu_blp`, `fsr`, `meter`,
`view`) reads the device or adapter, builds render resources, or writes plain main-world data, so all
of them run correctly on the worker. `RenderStartup` and device recovery run inside `update_state`,
which is part of extract, so on the worker. A render-world `webenv` read finds no `window` on the
worker and reads a snapshot the page published (`benilla_app::webenv::publish_for_workers`).

**Frame generation.** Every update is still extracted once and in order, so frame generation's web
cadence (`benilla_world::fsr::cadence`) holds frames as on one thread, and its split pair overlaps:
the worker draws the held frame while the page runs the next `Main`. That run-ahead waits when the
held frame's extract is still owed (`extract_owed`), so it never runs before that extract.

**What it does not do.** `web/index.html`'s two diagnostic
`requestDevice` wrappers (the WGSL error reporter, the sampler census) see only page devices; the
device watchdog is forwarded from the worker (`web/renderworker.js` → `gpu.js`).

## The ninth delta: released GPU objects are destroyed each frame (web)

The browser frees a dropped buffer, texture or query set only when its JS wrapper is collected, and
the render worker allocates almost no JS, so it almost never collects: dropped objects filled the
8 GB card until it paged (measured: ~10 MB/s, frames falling to 12 fps). The vendored `wgpu`
(`third_party/wgpu/PATCH.md`, "Explicit release") queues an object once its last handle, view and
bind group are gone; `destroy_released_gpu_resources` (`src/lib.rs`, wasm32 + `webgpu` only) destroys
them in `RenderSystems::PostCleanup`, after the frame's submits and the temporary entities' despawn,
and counts them for `benilla_app::webdiag` (`take_released_gpu_objects`). The slab allocator's own
`destroy()` (fourth delta) stays: it releases a buffer bind groups may still hold, earlier than a
guard would.

## The delta, and the command that proves it

**A few lines changed in five upstream files, one module added.**

| file | change |
|---|---|
| `src/view/mod.rs` | `prepare_view_attachments`: the `CameraOutputMode::Skip` early-out goes (above) |
| `src/view/window/mod.rs` | `prepare_windows`: a `Skip` camera counts as the window's target; `extract_windows`: the wasm32-only per-task canvas texture (above) |
| `src/slab_allocator.rs` | `reallocate_slab` and `free_empty_slabs` hand the buffer they let go to `release_buffer`, a wasm32-only `destroy()` (above) |
| `src/batching/gpu_preprocessing.rs` | two `ComputeTaskPool::get().scope(` (0.19.1 lines ~2355 and ~2657) become `crate::wasm_serial::compute_scope(`, and the now-unused `use bevy_tasks::ComputeTaskPool;` goes |
| `src/render_asset.rs` | `RenderAsset::DEFERRABLE`, asked by `prepare_assets` before each of its two deferrals (above) |
| `src/mesh/mod.rs` | `RenderMesh` sets `DEFERRABLE = false` (above) |
| `src/lib.rs` | `mod wasm_serial;`; the render worker's lines (eighth delta); `destroy_released_gpu_resources` and its counter (ninth delta) |
| `src/wasm_serial.rs` | new — the scope itself. Has its own header |
| `src/render_worker.rs`, `src/settings.rs`, `src/view/window/mod.rs`, `src/renderer/{wgpu_wrapper,render_device,mod}.rs`, `src/render_resource/buffer.rs` | the render worker (eighth delta, its own table) |

**The bodies of both systems are byte-for-byte upstream's**, which is the whole design of the fix:
the scope object a closure receives is inferred, so `compute_scope(|scope| { … })` reads identically
whether `scope` is a `bevy_tasks::Scope` or the serial stand-in. A fork that retyped ninety lines of
buffer writes would be a fork nobody could diff against upstream on the next bevy release.

Off `wasm32 + atomics`, `compute_scope` **is** `ComputeTaskPool::get().scope` — so native is
upstream, and the serial path costs nothing anywhere else.

## What this does NOT do

It does not make the render app single-threaded, and it is not a general fix for "wgpu on a worker".
It closes the two paths that exist in bevy 0.19.1. Anything added later — here or in a bevy upgrade
— that puts a `WgpuWrapper`-backed value inside a `scope.spawn` or a `par_iter` closure will panic
the same way, and the answer is to route it through `compute_scope` too.

Audited alongside this and found clean (parallel, but CPU-only — entity lists and uniform queues, no
wgpu handle in the closure): `bevy_pbr`'s `material.rs`, `render/mesh.rs` (×2) and `prepass/mod.rs`,
`bevy_sprite_render`'s `mesh2d/material.rs`, and the `par_iter`s in `bevy_light` and `bevy_camera`,
which run in the main world. New in 0.19.1 and also clean: `bevy_pbr`'s
`collect_meshes_for_gpu_building` (`render/mesh.rs` ~2472), whose `ComputeTaskPool` scope only
builds CPU-side instance lists and sends them over channels; the buffer uploads run after the scope,
on the render thread. `PipelineCache`'s async pipeline tasks are gated off wasm upstream.

## Verifying against upstream

```sh
# The patched crate is never downloaded into cargo's registry, so fetch the pinned upstream.
mkdir -p /tmp/upstream && curl -sL https://static.crates.io/crates/bevy_render/bevy_render-0.19.1.crate   | tar xz -C /tmp/upstream
diff -r third_party/bevy_render /tmp/upstream/bevy_render-0.19.1
```

That must print exactly a `diff` for `src/batching/gpu_preprocessing.rs` (three lines),
`src/view/mod.rs` (the `Skip` early-out), `src/view/window/mod.rs` (the two window hunks) and
`src/lib.rs` (the `mod`), `Only in third_party/bevy_render/src: wasm_serial.rs`, `Only in
third_party/bevy_render: BENILLA.md`, and `Only in <upstream>: .cargo_vcs_info.json`,
`Cargo.lock`, `Cargo.toml.orig` — cargo's packaging metadata, dropped the same way the other
`third_party` forks drop it.
