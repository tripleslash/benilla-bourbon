# benilla's `wgpu-core` fork — what differs from upstream, and how to check

Upstream: [`gfx-rs/wgpu`](https://github.com/gfx-rs/wgpu) (`wgpu-core`), MIT OR Apache-2.0, version
`29.0.4`, the version the workspace lock resolves to (Bevy 0.19.1's wgpu 29). Wired in through
`[patch.crates-io]` in the workspace root. The crate is copied from the cargo registry as published
(`src/`, `build.rs`, `Cargo.toml`, the two licence files). Every hunk is marked `BENILLA:`. Its
partner is `third_party/wgpu-hal` (the DX12 half, "The second patch" there), and the change that
consumes both is `third_party/bevy_render` (the thirteenth delta).

## Why a fork exists at all

[gfx-rs/wgpu#7974](https://github.com/gfx-rs/wgpu/issues/7974), open: on DX12,
`multi_draw_indirect_count` and `multi_draw_indexed_indirect_count` run every draw with
`@builtin(instance_index)` starting at 0 and `@builtin(vertex_index)` without its
`first_vertex` / `base_vertex`. Upstream's own repro (a comment on the issue): two indexed draws
with `first_instance` 0 and 1 both draw as instance 0 on DX12 (WARP and AMD), correct on Vulkan.
Bevy 0.19.1 works around it by not using count draws on DX12 at all.

**The mechanism.** HLSL before Shader Model 6.8 has no `SV_StartInstanceLocation` /
`SV_StartVertexLocation`: `SV_InstanceID` and `SV_VertexID` start at 0 whatever the draw's start
values. naga's HLSL backend adds them back from a 3-dword root constant (`_NagaConstants`:
`first_vertex`, `first_instance`, `other`) that wgpu-hal sets before each direct draw. An
`ExecuteIndirect` can only set that constant per draw through a command signature that has a
`D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT` argument *before* the draw argument, so the argument
buffer must hold `[first_vertex, first_instance, 0]` in front of each draw's arguments (stride
28 non-indexed, 32 indexed) — a layout no caller writes. wgpu does it in two places:

- wgpu-hal creates such signatures for every pipeline layout made with
  `PipelineLayoutFlags::INDIRECT_BUILTIN_UPDATE`, which wgpu-core passes whenever its indirect
  validation exists.
- wgpu-core's indirect-draw validation (`indirect_validation/draw.rs`, `validate_draw.wgsl`)
  runs a compute pass before the render pass that copies each draw's arguments into a pooled
  destination buffer, and on DX12 (`write_d3d12_special_constants`) writes the three dwords in
  front of them.

Upstream sends plain `multi_draw_indirect` through that rewrite, but `multi_draw_indirect_count`
hands the caller's raw buffer to wgpu-hal, which executes it with the device's plain signature and
both constants set to 0. And the rewrite exists only while `InstanceFlags::VALIDATION_INDIRECT_CALL`
is set; without it even plain indirect draws read the builtins as 0 on DX12 (the flag's own doc
comment says so).

## The patch

| file | hunk |
|---|---|
| `src/device/resource.rs`, `Device::new` | indirect validation is created when the flag is set **or the backend is DX12**, so the rewrite (and the `INDIRECT_BUILTIN_UPDATE` layouts that expect it) always exist there. The flag stays what decides it on every other backend. |
| `src/command/render.rs`, `multi_draw_indirect_count` (and its call site) | takes the pass's `DrawBatcher`. On DX12, for `Draw` and `DrawIndexed` (not mesh tasks, which upstream does not rewrite either), it records the indirect buffer as a storage read, rewrites all `max_count` draws with `DrawBatcher::add_contiguous`, and calls `draw_*_indirect_count` on the destination buffer and offset with the caller's own count buffer. Off DX12 the function is upstream's. |
| `src/command/render.rs`, `RenderPassErrorInner` | `IndirectCountTooLargeForRewrite { size, max_binding_size }` (and its `webgpu_error_type` arm, a validation error): the rewritten arguments of one count draw must fit one storage binding, the device's `max_storage_buffer_binding_size` (wgpu-hal's DX12 adapter offers 2 GiB − 1; at the WebGPU default of 128 MiB it is still ~4.2 million indexed draws). |
| `src/indirect_validation/draw.rs`, `DrawBatcher` | `add` is split into the destination allocation and `record` (upstream's body unchanged); `add_contiguous` allocates one contiguous range for `count` draws and records each one, `src_stride` apart in the source and `dst_stride` apart in the destination. The batch key gains the destination's `BUFFER_SIZE` window (`dst_offset / BUFFER_SIZE`, always 0 in a pooled buffer), so a batch's metadata still fits one metadata buffer when a destination is larger than that. |
| `src/indirect_validation/draw.rs`, `DrawResources` | `get_dst_contiguous_subrange`: a range up to `BUFFER_SIZE` comes from the pool exactly as a single draw's does (`get_subrange_impl` never splits a range); a larger one gets a buffer of its own at that size, after which the next range starts a fresh pooled buffer. `Drop` destroys such buffers instead of pooling them; it runs when the command buffer has completed (`EncoderInFlight` keeps the resources alive until then), which is what makes pooling safe upstream as well. |
| `src/indirect_validation/draw.rs`, `BufferPoolEntry` | a `size` field, and `create_buffer_and_bind_group` becomes a wrapper over `create_sized_buffer_and_bind_group` (any size of at least `BUFFER_SIZE`, the layout's `min_binding_size`). |

**Why every `max_count` draw is rewritten.** The count lives in a GPU buffer, so the CPU cannot
know how many draws will run. Rewriting all of them costs one compute invocation each; the ones
past the count are never executed. The source range is already bounded by `max_count` and checked
against the buffer's size upstream, so the shader reads nothing out of range.

**The validation that comes with it.** The rewrite is wgpu's indirect validation: a draw whose
vertex, index or instance range exceeds the bound buffers becomes a no-op instead of undefined
behaviour. On DX12 that now holds for count draws as well, and holds with the flag off.

**What `@builtin(vertex_index)` gets.** The shader writes `src[2]` (`first_vertex`) for a
non-indexed draw and `src[3]` (`base_vertex`) for an indexed one, the same values wgpu-hal passes
to `prepare_draw` for direct draws, so indirect and direct draws agree.

## How to check

```sh
diff -ru ~/.cargo/registry/src/index.crates.io-*/wgpu-core-29.0.4 third_party/wgpu-core
```

shows exactly the hunks above (three files, every one marked `BENILLA:`), this file, and `Only
in` the registry copy for cargo's packaging files (`.cargo-ok`, `.cargo_vcs_info.json`,
`Cargo.lock`, `Cargo.toml.orig`). If the registry copy is absent, fetch
`https://static.crates.io/crates/wgpu-core/wgpu-core-29.0.4.crate`.

On a DX12 machine: `WOW_STATIC_GX_MDI=0` against `=1` (plain multi-draw), and Bevy's indirect lane
(`WOW_INDIRECT=1`, which now uses count draws on DX12), each pictured against Vulkan
(`WGPU_BACKEND=vulkan`) at the same spot, must match.

## When to drop it

When wgpu fixes [gfx-rs/wgpu#7974](https://github.com/gfx-rs/wgpu/issues/7974), or when the
workspace leaves wgpu 29: delete this directory and the `wgpu-core` line under
`[patch.crates-io]`, and check `third_party/wgpu-hal` and `third_party/bevy_render` (the
thirteenth delta) against the release.
