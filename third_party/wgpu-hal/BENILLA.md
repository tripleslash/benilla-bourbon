# benilla's `wgpu-hal` fork — what differs from upstream, and how to check

Upstream: [`gfx-rs/wgpu`](https://github.com/gfx-rs/wgpu) (`wgpu-hal`), MIT OR Apache-2.0, version
`29.0.4`, the version the workspace lock resolves to (Bevy 0.19.1's wgpu 29). Wired in through
`[patch.crates-io]` in the workspace root. The crate is copied from the cargo registry as published
(`src/`, `build.rs`, `Cargo.toml`, `README.md`, the two licence files), except that the three
`[[example]]` targets are dropped from `Cargo.toml` and `examples/` is not copied. Every hunk is
marked `BENILLA:`.

## Why a fork exists at all

Four patches: the Vulkan presentation stall below, DX12 multi-draw-indirect-count ("The second
patch", with `third_party/wgpu-core`), an async compute queue ("The third patch") and DX12
pipeline-statistics queries ("The fourth patch").

On Vulkan, the default on Windows and Linux, the swapchain's acquire (`NativeSwapchain::acquire`)
waits up to wgpu-core's 1 s frame timeout, and a presentation engine that stalls longer (a machine
loaded by other work) used to end the process:

- The post-acquire `vkWaitForFences` on the acquired image's fence returned `VK_TIMEOUT`, which
  went through `map_host_device_oom_and_lost_err` and became `DeviceError::Unexpected`. wgpu-core
  treats that as fatal and loses the device, and Bevy's `prepare_windows` panics with "Couldn't
  get swap chain texture … generic error".
- A `vkAcquireNextImageKHR` `VK_TIMEOUT` returned `SurfaceError::Timeout`, a skipped frame at
  best.

Upstream tracks it as [gfx-rs/wgpu#9029](https://github.com/gfx-rs/wgpu/issues/9029), still open
in wgpu 30. 1.12.1 has no such timeout: its frame pass presents when the swap chain is ready.
Found in `pkuzic/benilla-everwood_graphics@6fe464d896`.

## The patch — `src/vulkan/swapchain/native.rs`, `NativeSwapchain::acquire`

A `VK_TIMEOUT` is retried while the call has stalled for less than `STALL_BUDGET` (30 s, counted
from the start of the acquire) and the caller's timeout is not zero (a zero timeout is a poll,
which answers at once by design). Past the budget the result is what upstream returns.

- **`vkAcquireNextImageKHR` times out:** the acquire is issued again with the same semaphore and
  fence. The spec: when a call "does not successfully acquire an image, `semaphore` and `fence`
  are unaffected", so no signal operation is pending and the "unsignaled, no operations
  pending" valid usage still holds. `VK_TIMEOUT` itself is returned only for a timeout above zero
  and below `UINT64_MAX`.
- **`vkWaitForFences` on the acquired image times out:** the wait is issued again. The image is
  already acquired and its fence has a signal operation submitted, so waiting is the only
  correct move; the spec lets a wait be repeated and returns `VK_TIMEOUT` only when the fence is
  not yet signalled.
- **The wait on the previous submission, before the acquire, times out:** upstream drops
  `wait_for_fence`'s `Ok(false)` and acquires with a semaphore the GPU may still hold, which the
  spec forbids ("unsignaled, with no signal or wait operations pending"). The wait is issued
  again under the same budget; past it the acquire goes on as upstream's does.
- **Logging:** each retry logs at `error!` with the time stalled, so a stall shows in the log. A
  result that is not retried is logged raw (`{:?}` of the `vk::Result`) before it is mapped to
  `DeviceError::Unexpected` or `SurfaceError::Timeout`; the ordinary resize results (`NOT_READY`,
  `ERROR_OUT_OF_DATE_KHR`, `ERROR_SURFACE_LOST_KHR`) are not logged.

What changes in play: a stall under 30 s only delays the frame. A caller that passes a timeout
now waits up to the budget instead of one timeout, so `SurfaceError::Timeout` ("not ready, skip
the frame") comes only after 30 s.

## The second patch — `src/dx12/command.rs`, DX12 multi-draw-indirect-count

`draw_indirect_count` and `draw_indexed_indirect_count` now pick their command signature the way
`draw_indirect` and `draw_indexed_indirect` already do: where the bound pipeline layout has its own
signatures (`PipelineLayoutSpecialConstants::indirect_cmd_signatures`, made for layouts created
with `PipelineLayoutFlags::INDIRECT_BUILTIN_UPDATE`), they flush the vertex buffers and root
elements and execute with the layout's signature, whose `D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT`
argument sets the special-constants root constant (`first_vertex`, `first_instance`, `other`) from
the three dwords in front of each draw's arguments. Without such signatures they are upstream's:
`prepare_draw(0, 0)` and the device's plain signature.

Upstream always took the second path for count draws, so on DX12 every draw of a
`multi_draw_*indirect_count` read `@builtin(instance_index)` from 0 and `@builtin(vertex_index)`
without its base: HLSL before SM 6.8 has no `SV_StartInstanceLocation`, and naga adds the start
values back from that root constant
([gfx-rs/wgpu#7974](https://github.com/gfx-rs/wgpu/issues/7974)). The patch is only correct
together with `third_party/wgpu-core`, which rewrites a count draw's arguments into the
signature's layout on DX12 (`[first_vertex, first_instance, 0]` then the arguments, stride 28 or
32) before handing the buffer here; `third_party/wgpu-core/BENILLA.md` has the mechanism. Mesh-task
count draws are untouched on both sides.

## The third patch — an async compute queue (DX12, Vulkan)

A second, compute-only queue beside the main one, which wgpu-core drives
(`third_party/wgpu-core/BENILLA.md`, "Async compute") so benilla's froxel, light-binning and meter
compute can run beside the raster passes. Every hunk is marked `BENILLA: async compute`.

| file | hunk |
|---|---|
| `src/lib.rs` | `Device::create_async_compute_queue(&self) -> Option<Queue>` (default `None`; GL, Metal and noop keep it) and `Queue::wait_for_fence(&self, fence: &Fence, value: FenceValue) -> Result<(), DeviceError>` (default `Err(Unexpected)`): later submissions on the queue wait on the GPU until another queue's `fence` reaches `value`. `ASYNC_COMPUTE_BUFFER_MARK` / `ASYNC_COMPUTE_TEXTURE_MARK`: bit 15 of `wgt::BufferUses` / `wgt::TextureUses`, which wgpu-types 29 leaves unnamed (a `const` assertion fails the build if a bump names it), set by wgpu-core on a use the async queue left. |
| `src/dynamic/device.rs`, `src/dynamic/queue.rs` | the two methods on `DynDevice` / `DynQueue`. |
| `src/dx12/device.rs` | `create_async_compute_queue`: `CreateCommandQueue` with `D3D12_COMMAND_LIST_TYPE_COMPUTE`. `create_command_encoder` makes its allocator, and so its lists, of the queue's type. |
| `src/dx12/mod.rs`, `src/dx12/adapter.rs` | `Queue` and `CommandEncoder` carry their `D3D12_COMMAND_LIST_TYPE`; `Queue::wait_for_fence` is `ID3D12CommandQueue::Wait`. |
| `src/dx12/command.rs`, `src/dx12/conv.rs` | `transition_buffers` / `transition_textures` map a use through `compute_queue_state`: on a compute list, or for a marked use, without `PIXEL_SHADER_RESOURCE`, the one graphics-only state a compute-queue use maps to (the runtime fails `Close` on a compute list that names it: measured, `E_INVALIDARG`). The UAV-barrier test ignores the mark. |
| `src/vulkan/adapter.rs` | `open_with_callback` asks for a second queue: the first queue of a compute-only family where there is one, else queue 1 of the graphics family; none without timeline semaphores (the cross-queue wait is one). `device_from_raw` keeps its signature and forwards to `device_from_raw_with_async_queue`, which records the slot. |
| `src/vulkan/device.rs`, `src/vulkan/mod.rs` | `create_async_compute_queue` hands that queue out once, with its own relay semaphores. `Queue::wait_for_fence` queues a timeline wait (`ALL_COMMANDS`) that the next `submit` takes. Buffers and images are `VK_SHARING_MODE_CONCURRENT` over both families when the async queue is of another family (`DeviceShared::concurrent_families`). An encoder of the compute family masks its barriers' stages and accesses to what such a family supports (`conv::compute_queue_barrier`); the mark is stripped before layouts and stages are derived. |

**Why these resource rules.** DX12: wgpu-core records every transition *into* an async submission
on the main (direct) queue, which may name any state, and marks what the async lists write or
transition, so the transition *out* names the compute-queue state; a compute list itself only ever
names compute states. No common-state decay or promotion is relied on. Vulkan: a second family
needs either ownership transfers (a release on one queue matched by an acquire on the other, per
resource, with the layouts repeated) or concurrent sharing; concurrent sharing is one flag at
creation and nothing per frame, and the semaphore wait already orders the memory. Its possible cost is
drivers that disable compression for concurrent images (reported for some AMD generations; not
measured here).
The second-queue-of-the-graphics-family fallback needs neither.

## The fourth patch — pipeline-statistics queries on DX12

Upstream's DX12 adapter leaves `Features::PIPELINE_STATISTICS_QUERY` commented out: the query heap
type was already mapped (`D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS`), but `ResolveQueryData`
writes D3D12's whole `D3D12_QUERY_DATA_PIPELINE_STATISTICS` record per query, eleven `u64`s
(`IAVertices`, `IAPrimitives`, `VSInvocations`, `GSInvocations`, `GSPrimitives`, `CInvocations`,
`CPrimitives`, `PSInvocations`, `HSInvocations`, `DSInvocations`, `CSInvocations`), where wgpu
promises one `u64` per type the set was made for, in `PipelineStatisticsTypes` bit order. A
straight resolve would overrun the destination wgpu-core sized for that layout. Every hunk is
marked `BENILLA: pipeline statistics`.

| file | hunk |
|---|---|
| `src/dx12/adapter.rs` | the feature is exposed, in place of upstream's TODO |
| `src/dx12/mod.rs` | `QuerySet::stats`: the `PipelineStatisticsTypes` a statistics set was made for, empty for any other |
| `src/dx12/device.rs` | `create_query_set` records it |
| `src/dx12/command.rs` | `copy_query_results` sends a statistics set to `copy_pipeline_statistics`: the range resolves into an intermediate buffer (`COPY_DST`, then `COPY_SRC`), kept in `intermediate_copy_bufs` until `reset_all`, the same lifetime as the texture-copy intermediates; then one `CopyBufferRegion` per run of fields adjacent in both layouts. The timestamp `QuerySet` built for end-of-pass stamps gets an empty `stats`. |

The field mapping, `PIPELINE_STATISTICS_FIELDS`, is wgpu's bit order against the record's field
offsets (`offset_of!`): `VERTEX_SHADER_INVOCATIONS` is `VSInvocations`, `CLIPPER_INVOCATIONS`
`CInvocations`, `CLIPPER_PRIMITIVES_OUT` `CPrimitives`, `FRAGMENT_SHADER_INVOCATIONS`
`PSInvocations`, `COMPUTE_SHADER_INVOCATIONS` `CSInvocations`. These are the counters Vulkan's
`VK_QUERY_PIPELINE_STATISTIC_*` bits name for the same five types (`src/vulkan/conv.rs`), so a
pass reads the same five numbers on either backend. With all five types (bevy's render
diagnostics ask for all) a query is three copies: field 2, fields 5–7, field 10.
`pipeline_statistics_tests` checks the runs.

## How to check

`diff -u <registry>/wgpu-hal-29.0.4/src/vulkan/swapchain/native.rs third_party/wgpu-hal/src/vulkan/swapchain/native.rs`
shows the first patch and nothing else, and the same for `src/dx12/command.rs` shows the second,
third and fourth patches' hunks there, each marked `BENILLA:`; `diff -r` of the two trees shows only
the files the four patches name, the dropped
`[[example]]` blocks in `Cargo.toml`, and the files not copied (`examples/`, `Cargo.lock`,
`Cargo.toml.orig`, `.cargo_vcs_info.json`, `.cargo-ok`).
`video::tests::wgpu_hal_resolves_to_the_vendored_stall_patch` checks the lock resolves wgpu-hal to
this path, at the lock's version, with the hunks present.

## When to drop it

When wgpu-hal handles the timeout itself ([gfx-rs/wgpu#9029](https://github.com/gfx-rs/wgpu/issues/9029))
and wgpu fixes [gfx-rs/wgpu#7974](https://github.com/gfx-rs/wgpu/issues/7974), or when wgpu
leaves 29: delete this directory and the `wgpu-hal` line under `[patch.crates-io]`. Drop the
second patch only together with `third_party/wgpu-core`'s.
The fourth patch goes when an upstream DX12 adapter exposes `PIPELINE_STATISTICS_QUERY` itself;
without it, `WOW_GPU_STATS` reads nothing on DX12 (the device lacks the feature, and the instrument
says so at startup).
