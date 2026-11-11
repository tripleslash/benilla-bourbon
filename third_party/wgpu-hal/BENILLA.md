# benilla's `wgpu-hal` fork — what differs from upstream, and how to check

Upstream: [`gfx-rs/wgpu`](https://github.com/gfx-rs/wgpu) (`wgpu-hal`), MIT OR Apache-2.0, version
`30.0.1`, the version the workspace lock resolves to (Bevy 0.20.1's wgpu 30). Wired in through
`[patch.crates-io]` in the workspace root. The crate is copied from the cargo registry as published
(`src/`, `build.rs`, `Cargo.toml`, `README.md`, the two licence files), except that the three
`[[example]]` targets are dropped from `Cargo.toml` and `examples/` is not copied. Every hunk is
marked `BENILLA:`.

## Why a fork exists at all

Seven patches: the Vulkan presentation stall below, DX12 multi-draw-indirect-count ("The second
patch", with `third_party/wgpu-core`), an async compute queue ("The third patch") and DX12
pipeline-statistics queries ("The fourth patch"), the DX12 command list for NGX ("The fifth
patch"), a present interposer for Streamline ("The sixth patch") and the DX12 draw index for
mesh-task draws ("The seventh patch", with `third_party/wgpu-core` and `third_party/naga`).

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
- **The wait on the previous submission, before the acquire, times out:** wgpu-hal 30 answers it
  with `SurfaceError::Timeout` rather than acquiring with a semaphore the GPU may still hold
  (gfx-rs/wgpu#9405). The wait is issued again under the same budget; past it, upstream's
  `Timeout`.
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
draws are the seventh patch's.

## The third patch — an async compute queue (DX12, Vulkan)

A second, compute-only queue beside the main one, which wgpu-core drives
(`third_party/wgpu-core/BENILLA.md`, "Async compute") so benilla's froxel, light-binning and meter
compute can run beside the raster passes. Every hunk is marked `BENILLA: async compute`.

| file | hunk |
|---|---|
| `src/lib.rs` | `Device::create_async_compute_queue(&self) -> Option<Queue>` (default `None`; GL, Metal and noop keep it) and `Queue::wait_for_fence(&self, fence: &Fence, value: FenceValue) -> Result<(), DeviceError>` (default `Err(Unexpected)`): later submissions on the queue wait on the GPU until another queue's `fence` reaches `value`. `ASYNC_COMPUTE_BUFFER_MARK` / `ASYNC_COMPUTE_TEXTURE_MARK`, set by wgpu-core on a use the async queue left: bit 15 of `wgt::TextureUses`, which wgpu-types 30 leaves unnamed (a `const` assertion fails the build if a bump names it), and of `wgt::BufferUses`, which wgpu-types 30 fills with `RAY_TRACING_PIPELINE_SHADER_DATA`. The buffer mark shares that bit: a read-only use like the states it marks, which wgpu-core 30 never records (it has no ray-tracing pipelines); asserted in `lib.rs`. |
| `src/dynamic/device.rs`, `src/dynamic/queue.rs` | the two methods on `DynDevice` / `DynQueue`. |
| `src/dx12/device.rs` | `create_async_compute_queue`: `CreateCommandQueue` with `D3D12_COMMAND_LIST_TYPE_COMPUTE`. `create_command_encoder` makes its allocator, and so its lists, of the queue's type. |
| `src/dx12/mod.rs`, `src/dx12/adapter.rs` | `Queue` and `CommandEncoder` carry their `D3D12_COMMAND_LIST_TYPE`; `Queue::wait_for_fence` stages an `ID3D12CommandQueue::Wait` through upstream's `Queue::add_wait_fence` (wgpu-hal 30), issued before the next submit's `ExecuteCommandLists`; wgpu-core calls it right before that submit. The second queue gets 30's idle fence and staged-wait lists. |
| `src/dx12/command.rs`, `src/dx12/conv.rs` | `transition_buffers` / `transition_textures` map a use through `compute_queue_state`: on a compute list, or for a marked use, without `PIXEL_SHADER_RESOURCE`, the one graphics-only state a compute-queue use maps to (the runtime fails `Close` on a compute list that names it: measured, `E_INVALIDARG`). The UAV-barrier test ignores the mark. |
| `src/vulkan/adapter.rs` | `open_with_callback` asks for a second queue: the first queue of a compute-only family where there is one, else queue 1 of the graphics family; none without timeline semaphores (the cross-queue wait is one). `device_from_raw` keeps its signature and forwards to `device_from_raw_with_async_queue`, which records the slot. |
| `src/vulkan/device.rs`, `src/vulkan/mod.rs` | `create_async_compute_queue` hands that queue out once, with its own relay semaphores. `Queue::wait_for_fence` is upstream's `Queue::add_wait_semaphore` (wgpu-hal 30) with the timeline value, at `ALL_COMMANDS`: the next `submit` takes it; the fork's own wait list and submit hunk are gone. Buffers, images and the buffers behind acceleration structures (`create_acceleration_structure`: a ray query on the async queue reads a TLAS the main queue built) are `VK_SHARING_MODE_CONCURRENT` over both families when the async queue is of another family (`DeviceShared::concurrent_families`). An encoder of the compute family masks its barriers' stages and accesses to what such a family supports (`conv::compute_queue_barrier`); the mark is stripped before layouts and stages are derived. |

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

## The fifth patch — the DX12 command list, for NVIDIA NGX (`src/dx12/command.rs`)

`CommandEncoder::raw_command_list()` returns the open `ID3D12GraphicsCommandList`. DLSS on DX12
(`third_party/dlss_wgpu/src/d3d12.rs`) records `NVSDK_NGX_D3D12_CreateFeature` and
`NVSDK_NGX_D3D12_EvaluateFeature_C` into it, in an encoder of its own that nothing else records
into, so NGX's descriptor heaps and root signature need no restoring. Vulkan has the equivalent
upstream (`raw_handle()`). Additive: no existing path changes.

## The sixth patch — a present interposer (NVIDIA Streamline), DX12 and Vulkan

DLSS Frame Generation exists only inside Streamline's closed `sl.dlss_g` plugin, which must see the
swap chain. In manual-hooking mode (`ProgrammingGuideManualHooking.md`) the host routes the hooked
calls through Streamline's proxies:

- **DX12** (`src/dx12/mod.rs`, `interposer`; `src/dx12/adapter.rs`): with an `slUpgradeInterface`
  installed by `interposer::set_upgrade_interface` (and `slSetD3DDevice` by
  `interposer::set_device_hook`, handed the native device in `Adapter::open` before the first
  proxy is made, the guide's 5.1 order), the direct queue is created on the proxy device
  (`eID3D12Device_CreateCommandQueue`) and the HWND swap chain on the proxy factory
  (`eIDXGIFactory_CreateSwapChainForHwnd`); the swap chain wgpu then holds is the proxy, so
  `Present`, `GetBuffer`, `ResizeBuffers` and the rest reach Streamline. Each proxy keeps one
  reference for the process (`interposer::upgraded`): Streamline destroys a proxy at its last
  release, and the queue and swap chain made through it still lead back to it. The composition
  targets are not routed.
- **Vulkan** (`src/vulkan/mod.rs`, `interposer`; `src/vulkan/instance.rs`): with a loader path set
  by `interposer::set_loader`, `ash::Entry::load_from(sl.interposer.dll)` replaces
  `Entry::load()`, Streamline's documented Vulkan path (1.1): its `vkGet*ProcAddr` answer the
  driver's functions except the ones it intercepts.

Unset (every run that does not select DLSS frame generation at boot), both are one `OnceLock` read
at instance or swap-chain creation and nothing else.

## The seventh patch — the draw index on DX12 (`src/dx12/adapter.rs`, `device.rs`, `command.rs`)

static_gx's meshlet lane reads `@builtin(draw_index)` in its task stage, which naga 30's HLSL
backend lacks. The vendored naga reads it from the special constants' third word, `other`
(`third_party/naga/BENILLA.md`); this patch makes that word right in every draw:

| file | hunk |
|---|---|
| `src/dx12/adapter.rs` | the adapter reports `SHADER_DRAW_INDEX`. A direct draw's `other` is 0 (`prepare_draw`); every indirect draw goes through wgpu-core's rewrite on DX12, which writes the draw's index in its call there. |
| `src/dx12/device.rs`, `create_pipeline_layout` | an `INDIRECT_BUILTIN_UPDATE` layout on a device with mesh shaders gets a `draw_mesh` signature, `[CONSTANT(3), DISPATCH_MESH]` at stride 24, where upstream left it `None` ("Mesh pipelines don't support updating special constants"). |
| `src/dx12/command.rs`, `draw_mesh_tasks_indirect(_count)`, `mesh_cmd_signature` | a mesh-task indirect draw executes with the layout's `draw_mesh` signature where it has one (wgpu-core rewrote its arguments to `[0, 0, draw_index, x, y, z]`), else upstream's: `prepare_dispatch([0; 3])` and the device's plain signature. |

A direct `draw_mesh_tasks` keeps upstream's constants (the dispatch size, for a task stage's
`num_workgroups`), so a task stage there reads the dispatch's `z` as its draw index; the meshlet
lane draws only through counted multi-draws. The patch is only correct together with
`third_party/wgpu-core` ("The draw index" there), which rewrites every mesh-task indirect draw on
DX12 into the signature's layout.

## How to check

`diff -u <registry>/wgpu-hal-30.0.1/src/vulkan/swapchain/native.rs third_party/wgpu-hal/src/vulkan/swapchain/native.rs`
shows the first patch and nothing else, and the same for `src/dx12/command.rs` shows the second,
third, fourth, fifth and seventh patches' hunks there, each marked `BENILLA:`; `diff -r` of the two trees shows only
the files the patches name, the dropped
`[[example]]` blocks in `Cargo.toml`, and the files not copied (`examples/`, `Cargo.lock`,
`Cargo.toml.orig`, `.cargo_vcs_info.json`, `.cargo-ok`).
`video::tests::wgpu_hal_resolves_to_the_vendored_stall_patch` checks the lock resolves wgpu-hal to
this path, at the lock's version, with the hunks present.

## When to drop it

When wgpu-hal handles the timeout itself ([gfx-rs/wgpu#9029](https://github.com/gfx-rs/wgpu/issues/9029))
and wgpu fixes [gfx-rs/wgpu#7974](https://github.com/gfx-rs/wgpu/issues/7974), or when wgpu
leaves 30: delete this directory and the `wgpu-hal` line under `[patch.crates-io]`. Drop the
second and the seventh patches only together with `third_party/wgpu-core`'s (and the seventh with
`third_party/naga`).
The fourth patch goes when an upstream DX12 adapter exposes `PIPELINE_STATISTICS_QUERY` itself;
without it, `WOW_GPU_STATS` reads nothing on DX12 (the device lacks the feature, and the instrument
says so at startup).
