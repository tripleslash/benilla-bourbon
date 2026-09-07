# benilla's `bevy_render` fork — what differs from upstream, and how to check

Upstream: `bevy_render 0.18.1` (crates.io), MIT OR Apache-2.0. Wired in through `[patch.crates-io]`
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

Upstream knows this. `src/renderer/mod.rs` runs command-buffer generation serially on exactly this
target (`#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]`, three times), with
exactly this reasoning, and `src/renderer/wgpu_wrapper.rs` is where the `SendWrapper` comes from.

What it misses is `src/batching/gpu_preprocessing.rs`. `write_batched_instance_buffers` and
`write_indirect_parameters_buffers` take `render_device: Res<RenderDevice>` and
`render_queue: Res<RenderQueue>`, reborrow them as `&*`, and hand those references straight into
`scope.spawn`. On native that is correct and fast — the pool's threads run it and wgpu is genuinely
`Send`. On a threaded browser build it is the panic above, on the first frame that draws anything.

## The delta, and the command that proves it

**Three lines changed, one module added. No logic is touched.**

| file | change |
|---|---|
| `src/batching/gpu_preprocessing.rs` | two `ComputeTaskPool::get().scope(` become `crate::wasm_serial::compute_scope(`, and the now-unused `use bevy_tasks::ComputeTaskPool;` goes |
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
It closes the two paths that exist in bevy 0.18.1. Anything added later — here or in a bevy upgrade
— that puts a `WgpuWrapper`-backed value inside a `scope.spawn` or a `par_iter` closure will panic
the same way, and the answer is to route it through `compute_scope` too.

Audited alongside this and found clean (parallel, but CPU-only — entity lists and uniform queues, no
wgpu handle in the closure): `bevy_pbr`'s `material.rs`, `render/mesh.rs` (×2) and `prepass/mod.rs`,
`bevy_sprite_render`'s `mesh2d/material.rs`, and the `par_iter`s in `bevy_light` and `bevy_camera`,
which run in the main world.

## Verifying against upstream

```sh
# 0.18.1 is the pinned version; adjust the path if cargo's registry hash differs.
diff -r third_party/bevy_render ~/.cargo/registry/src/*/bevy_render-0.18.1
```

That must print exactly a `diff` for `src/batching/gpu_preprocessing.rs` (three lines) and
`src/lib.rs` (the `mod`), `Only in third_party/bevy_render/src: wasm_serial.rs`, `Only in
third_party/bevy_render: BENILLA.md`, and `Only in <registry>: .cargo-ok`, `.cargo_vcs_info.json`,
`Cargo.lock`, `Cargo.toml.orig` — cargo's packaging metadata, dropped the same way the other
`third_party` forks drop it.
