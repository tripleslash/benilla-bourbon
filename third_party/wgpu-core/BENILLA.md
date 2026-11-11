# benilla's `wgpu-core` fork — what differs from upstream, and how to check

Upstream: [`gfx-rs/wgpu`](https://github.com/gfx-rs/wgpu) (`wgpu-core`), MIT OR Apache-2.0, version
`30.0.1`, the version the workspace lock resolves to (Bevy 0.20.1's wgpu 30). Wired in through
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
Bevy 0.20.1 still works around it by not using count draws on DX12 at all.

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
| `src/command/render.rs`, `multi_draw_indirect_count` (and its call site) | takes the pass's `DrawBatcher`. On DX12, for `Draw` and `DrawIndexed` (not mesh tasks, which upstream does not rewrite either), it records the indirect buffer as a storage read and the count buffer as an indirect and a storage read, hands all `max_count` draws and the count's location to `DrawBatcher::add_contiguous`, and calls `draw_*_indirect_count` on the destination buffer and offset with the caller's own count buffer. Off DX12 the function is upstream's. |
| `src/command/render.rs`, `multi_draw_indirect` | `DrawContext::add` passes each draw's index in the call to `DrawBatcher::add`. |
| `src/command/bundle.rs`, `DrawIndirect` | passes `DrawBatcher::add` the draw index 0 (a bundle's indirect draw is a single draw). |
| `src/command/render.rs`, `RenderPassErrorInner` | `IndirectCountTooLargeForRewrite { size, max_binding_size }` (and its `webgpu_error_type` arm, a validation error): the rewritten arguments of one count draw must fit one storage binding, the device's `max_storage_buffer_binding_size` (wgpu-hal's DX12 adapter offers 2 GiB − 1; at the WebGPU default of 128 MiB it is still ~4.2 million indexed draws). |
| `src/indirect_validation/validate_draw.wgsl` | `MetadataEntry` gains `draw_index` (the draw's index in its call) and `count_offset` (its call's count in the new group 3, `counts`, or `NO_COUNT`); a draw at or past its count returns before reading or writing anything. |
| `src/indirect_validation/draw.rs`, `Draw` | the pipeline layout's group 3 is the source layout again, bound per batch to the count buffer's own `indirect_validation_bind_groups.draw` (any `INDIRECT` buffer has it) with the count's dynamic offset, or to the source's group where the batch has no count. The limits it needs become 4 bind groups, 2 dynamic storage buffers and 4 storage buffers a stage. `calculate_src_offsets` also takes a 4-byte datum, the count. |
| `src/indirect_validation/draw.rs`, `MetadataEntry`, `DrawBatcher` | the entry is 24 bytes (`draw_index`, `count_offset`). `add` takes the draw's index; `add_contiguous` allocates one contiguous range for `count` draws and makes their entries in one pass (`entry`: the limits read once, one batch lookup a run of same-keyed entries, `push`), `src_stride` apart in the source and `dst_stride` apart in the destination, with the count's location. The batch key (`BatchKey`) gains the destination's `DST_WINDOW` (as many 16-byte destinations as 24-byte entries fit one metadata buffer), so a batch's metadata always fits one metadata buffer, and the count buffer with its dynamic offset. |
| `src/indirect_validation/draw.rs`, `DrawResources` | `get_dst_contiguous_subrange`: a range up to `BUFFER_SIZE` comes from the pool exactly as a single draw's does (`get_subrange_impl` never splits a range); a larger one gets a buffer of its own at that size, after which the next range starts a fresh pooled buffer. `Drop` destroys such buffers instead of pooling them; it runs when the command buffer has completed (`EncoderInFlight` keeps the resources alive until then), which is what makes pooling safe upstream as well. |
| `src/indirect_validation/draw.rs`, `BufferPoolEntry` | a `size` field, and `create_buffer_and_bind_group` becomes a wrapper over `create_sized_buffer_and_bind_group` (any size of at least `BUFFER_SIZE`, the layout's `min_binding_size`). |

**Why every `max_count` draw has an entry, and only the counted ones are rewritten.** The count
lives in a GPU buffer, so the CPU cannot know how many draws will run: each of the `max_count`
gets a metadata entry and a compute invocation. The invocation reads the count first and stops at
a draw past it, which `ExecuteIndirect` never executes, so the rewrite reads and writes only the
draws that run. The source range is already bounded by `max_count` and checked against the
buffer's size upstream, so the shader reads nothing out of range.

**The validation that comes with it.** The rewrite is wgpu's indirect validation: a draw whose
vertex, index or instance range exceeds the bound buffers becomes a no-op instead of undefined
behaviour. On DX12 that now holds for count draws as well, and holds with the flag off.

**What `@builtin(vertex_index)` gets.** The shader writes `src[2]` (`first_vertex`) for a
non-indexed draw and `src[3]` (`base_vertex`) for an indexed one, the same values wgpu-hal passes
to `prepare_draw` for direct draws, so indirect and direct draws agree.

## Async compute — `src/device/async_queue.rs` and its callers

A second, compute-only queue (`AsyncComputeQueue`) beside the device's main queue, on the queue
wgpu-hal's third patch makes. Every hunk is marked `BENILLA: async compute`.

**The API** (`Global` is what the `wgpu` crate calls; `third_party/wgpu/PATCH.md` has the public one):

```rust
impl Device { pub fn create_async_compute_queue(self: &Arc<Self>) -> Option<Arc<AsyncComputeQueue>>; }
impl AsyncComputeQueue {
    pub fn submit(&self, command_buffers: &[Arc<CommandBuffer>])
        -> Result<SubmissionIndex, (SubmissionIndex, QueueSubmitError)>;
    pub fn wait_for_main(&self, index: SubmissionIndex) -> Result<(), DeviceError>;
    pub fn last_submission(&self) -> SubmissionIndex;
    pub fn get_timestamp_period(&self) -> f32;
}
impl Queue { pub fn wait_for_async(&self, q: &AsyncComputeQueue, index: SubmissionIndex) -> Result<(), DeviceError>; }
impl Global {
    pub fn device_create_async_compute_queue(&self, device_id: DeviceId) -> Option<Arc<AsyncComputeQueue>>;
    pub fn async_queue_create_command_encoder(&self, queue: &Arc<AsyncComputeQueue>,
        desc: &wgt::CommandEncoderDescriptor<Label>, id_in: Option<CommandEncoderId>)
        -> (CommandEncoderId, Option<DeviceError>);
    pub fn async_queue_submit(&self, queue: &Arc<AsyncComputeQueue>, command_buffer_ids: &[CommandBufferId])
        -> Result<SubmissionIndex, (SubmissionIndex, QueueSubmitError)>;
    pub fn queue_wait_for_async(&self, queue_id: QueueId, async_queue: &AsyncComputeQueue,
        index: SubmissionIndex) -> Result<(), DeviceError>;
}
```

**The synchronization scheme.**

- *One index space, two fences.* Both queues take submission indices from the device's one counter;
  each signals its own fence with its own. An index therefore names one submission on one queue,
  and a wait for index `n` (`Device::maintain`) waits for each queue's fence up to the lesser of `n`
  and that queue's newest.
- *Main before async.* `AsyncComputeQueue::submit` records the memory initialization and every
  transition into its command buffers' start states on a **main-queue** encoder (the "Async compute
  transit"), submits it after the pending writes as one main submission, and makes the async queue
  wait on the GPU for that. So an async submission never runs before main work submitted before
  it, sees every `write_buffer` / `write_texture` before it, and a compute list never names a
  graphics state.
- *Async before main.* Every main submission (the transit included) waits on the GPU for the
  newest async submission it conflicts with (`AsyncComputeQueue::conflicting_submission`): one
  that uses a resource the main submission uses, unless both used it only for reading, in one and
  the same state, with no transition inside either command buffer (`read_only_state`). Such a
  resource needs no barrier on either queue, so the two reads overlap. A pending write conflicts
  with every async use of its target.
- *States.* `Tracker::mark_for_async_queue` ORs `hal::ASYNC_COMPUTE_*_MARK` into the start and end
  states of every resource an async command buffer writes or transitions. The transit then
  transitions into the compute-queue state and the device tracker keeps it, so the next main
  transition starts from it (wgpu-hal maps a marked use without `PIXEL_SHADER_RESOURCE` on DX12,
  and ignores the mark elsewhere). An unmarked read-only resource stays in its full state, which
  a compute list may read without naming it. To tell the two apart, the command buffer trackers
  remember which resources they recorded a barrier for (`transitioned`).
- *Lifetimes.* Each queue's `LifetimeTracker` retires its own submissions against its own fence;
  a resource lives while either holds it (`Arc`). `map_async` on a buffer the async queue still
  uses is held there first (`map_if_used`), then handed to the main tracker (`Device::maintain`),
  which maps it or holds it for its own submissions. A destroyed resource both queues still use is
  split into two `TempResource::Shared` halves and freed when the later retires. `set_buffer_data`
  waits for both. `device.poll(Wait)` waits for both fences, and a queue is empty only when both
  are.

**Why a wait happened.** `RUST_LOG=wgpu_core::device::async_queue=debug` logs each conflict wait
with the first resource that caused it and the two uses; a buffer also gives its size and usage,
which names an unlabelled one.

**Limits.** Async command buffers come only from `async_queue_create_command_encoder`
(`CommandBufferMutable::async_compute`), go only to the async queue (`QueueSubmitError::WrongQueue`),
and may not record a render pass or a texture clear (`CommandEncoderError::AsyncComputeOnly`: a
clear may be a render pass), nor use a surface texture. Their hal encoders come from and return to
`Device::async_command_allocator`, so a compute-type list is never handed to the main queue. Query
sets are not tracked across the queues: a query an async pass writes must be resolved by a main
submission that waited for it (`Queue::wait_for_async`).

**What is not covered.** Mapped-at-creation buffers and buffer `unmap` copies are pending writes,
so they conflict as writes do. Ray-tracing acceleration structures are not tracked across the
queues.

| file | hunk |
|---|---|
| `src/device/async_queue.rs` | new: the queue, its submit, the conflict test, `schedule_destruction`, `Queue::wait_for_async`. |
| `src/device/resource.rs` | `Device::async_queue`, `async_command_allocator`, `get_async_queue`, `create_async_compute_queue`; `maintain` waits for and retires the async queue; `set_buffer_data` waits for it. |
| `src/device/queue.rs` | `Queue::submit` refuses async command buffers, and `submit_pending_submission` (wgpu 30's shared submit path) waits where its executions, pending writes included, conflict; `TempResource::Shared`; `QueueSubmitError::{WrongQueue, AsyncComputeSurfaceTexture}`; `validate_command_buffer`, `PendingWrites::pre_submit`, `dst_textures` and two `EncoderInFlight` fields `pub(crate)`. |
| `src/device/life.rs` | `map_if_used`, `take_ready_to_map`, `newest_matching`. |
| `src/device/global.rs`, `src/device/mod.rs` | the `Global` entry points; the module. |
| `src/resource.rs` | `map_async` and both `destroy`s route through the async queue. |
| `src/command/mod.rs` | `InnerCommandEncoder::async_compute` (which allocator its encoder returns to), `CommandBufferMutable::async_compute`, `CommandEncoder::new_for_queue`, the render-pass refusal in `encode_commands`, `CommandEncoderError::AsyncComputeOnly`. |
| `src/track/buffer.rs`, `src/track/texture.rs`, `src/track/mod.rs` | `transitioned`, `read_only_state`, `mark_for_async_queue`. |
| `src/lock/rank.rs` | `ASYNC_QUEUE_LIFE_TRACKER` wherever the main queue's life tracker may be taken, after it, and followed by what it may be; `ASYNC_QUEUE_FENCE` a leaf (wgpu 30 holds the device's own fence unlocked, so `command_indices`, held across both submits, orders the async submit against the main queue's). |

## How to check

```sh
diff -ru ~/.cargo/registry/src/index.crates.io-*/wgpu-core-30.0.1 third_party/wgpu-core
```

shows exactly the hunks above (every one marked `BENILLA:`; `src/device/async_queue.rs` is new), this
file, and `Only
in` the registry copy for cargo's packaging files (`.cargo-ok`, `.cargo_vcs_info.json`,
`Cargo.lock`, `Cargo.toml.orig`). If the registry copy is absent, fetch
`https://static.crates.io/crates/wgpu-core/wgpu-core-30.0.1.crate`.

What the rewrite costs a frame: under `WOW_GPU_PASSES`, each static_gx pass has a
`<pass>_rewrite` span beside it (`static_gx_prepass_rewrite`, `static_gx_rewrite`,
`static_gx_prepass_late_rewrite`), from an encoder timestamp right before the pass to the pass's
first, which brackets exactly the "Pre Pass" wgpu-core records ahead of it: the scope's barriers
and this rewrite. On Vulkan the same span is the barriers alone.

On a DX12 machine: `WOW_STATIC_GX_MDI=0` against `=1` (plain multi-draw), and Bevy's indirect lane
(`WOW_ENTITY_GPU_CULL=1`, which now uses count draws on DX12), each pictured against Vulkan
(`WGPU_BACKEND=vulkan`) at the same spot, must match.

## When to drop it

When wgpu fixes [gfx-rs/wgpu#7974](https://github.com/gfx-rs/wgpu/issues/7974), or when the
workspace leaves wgpu 30: delete this directory and the `wgpu-core` line under
`[patch.crates-io]`, and check `third_party/wgpu-hal` and `third_party/bevy_render` (the
thirteenth delta) against the release.
