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

## The delta, and the command that proves it

**Three lines changed, one module added. No logic is touched.**

| file | change |
|---|---|
| `src/batching/gpu_preprocessing.rs` | two `ComputeTaskPool::get().scope(` (0.19.1 lines ~2355 and ~2657) become `crate::wasm_serial::compute_scope(`, and the now-unused `use bevy_tasks::ComputeTaskPool;` goes |
| `src/lib.rs` | `mod wasm_serial;` |
| `src/wasm_serial.rs` | new — the scope itself. Has its own header |

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

That must print exactly a `diff` for `src/batching/gpu_preprocessing.rs` (three lines) and
`src/lib.rs` (the `mod`), `Only in third_party/bevy_render/src: wasm_serial.rs`, `Only in
third_party/bevy_render: BENILLA.md`, and `Only in <upstream>: .cargo_vcs_info.json`,
`Cargo.lock`, `Cargo.toml.orig` — cargo's packaging metadata, dropped the same way the other
`third_party` forks drop it.
