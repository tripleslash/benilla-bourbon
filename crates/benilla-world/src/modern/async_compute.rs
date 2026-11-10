//! fork: perf — the modern compute on wgpu's async compute queue (native DX12 and Vulkan;
//! `third_party/wgpu-core/BENILLA.md`, "Async compute", has the synchronization). The light
//! binning ([`super::lights`]), the probe GI's trace ([`super::gi`]) and the exposure meter
//! ([`super::meter`]) record on it when it exists and on the main queue otherwise; what they
//! compute is the same either way.
//!
//! Where each overlaps, from what it reads:
//! - **Light binning** reads only CPU-written buffers (the light table, the ranks, the grid
//!   parameters), so it is recorded before the shadow raster and runs beside it and the probe
//!   trace. The shadow pass binds the light table too, for reading only. It writes a private grid,
//!   copied into the storage block (`MODERN_SHADOW_CASCADES`) after a flush in front of the froxels:
//!   every world material and the probe trace bind that block, and wgpu orders the two queues per
//!   buffer, not per range, so the binning writing the block's light region made them all wait for
//!   it (measured on DX12: `PERF_OVERLAP` 0.000 of 0.107 ms, every frame waiting on the block).
//! - **The probe trace** (classify, setup, rays, sums, blend; the voxel DDA or, at the ray-traced
//!   tier, the ray query) is recorded after `static_gx_pass`, once every input is this frame's
//!   (the cascades, the voxels, the TLAS, the light grid), and runs beside bevy's opaque pass, the
//!   fog composite, the water and the post-process. It keeps the probe field in a buffer of its own,
//!   copied into the block's probe region on the main queue at the next frame's start, for the same
//!   per-buffer reason: so the probes' readers see last frame's trace, and nothing on the main queue
//!   waits for it (`gi::node::gi_trace_pass_async`). `WOW_ASYNC_GI=0` keeps it on the main queue,
//!   in its old place, read the same frame. Not measured yet.
//! - **The froxels stay on the main queue.** They read this frame's shadow cascades, so on the
//!   async queue they ran beside the opaque and static_gx raster, writing a private volume copied
//!   into `MODERN_FOG_VOLUME` in front of the fog composite. On DX12 (RTX 3080 Laptop, Northshire,
//!   Cinematic, 2560x1440) the scatter then spanned 10.7 ms of GPU time against 0.6 ms on the main
//!   queue, time-sliced against the raster, and no frame got faster: p50 matched and p95/p99 rose
//!   by up to 2 ms. A high-priority compute queue left the span unchanged.
//! - **The meter** reads the scene before the combine exposes it, and runs beside the
//!   post-process. Its grid is read back a frame or more later in any case.
//!
//! `WOW_ASYNC_COMPUTE=0` keeps everything on the main queue; so does the browser, which has no
//! second queue, and so does `WOW_GPU_STATS` (below).

use bevy::ecs::system::{Deferred, SystemBuffer, SystemMeta, SystemParam};
use bevy::ecs::world::DeferredWorld;
use bevy::prelude::*;
use bevy::render::renderer::{
    PendingCommandBuffers, RenderAsyncComputeQueue, RenderDevice, RenderQueue,
};
use bevy::render::{RenderApp, RenderStartup};

/// Submits the main-queue command buffers queued before this system, at its place in graph
/// order. bevy's `FlushCommands` submits when the system runs, before the graph's buffers are
/// applied, so in the middle of a view's schedule it finds nothing to submit; this is applied with
/// them, after the ones in front of it and before its own system's `RenderContext` buffers.
#[derive(SystemParam)]
pub(crate) struct FlushMainQueue<'s> {
    state: Deferred<'s, FlushMainQueueState>,
}

impl FlushMainQueue<'_> {
    pub(crate) fn flush(&mut self) {
        self.state.0 = true;
    }
}

#[derive(Default)]
pub(crate) struct FlushMainQueueState(bool);

impl SystemBuffer for FlushMainQueueState {
    fn queue(&mut self, _system_meta: &SystemMeta, mut world: DeferredWorld) {
        if !std::mem::take(&mut self.0) {
            return;
        }
        let buffers: Vec<_> = world
            .resource_mut::<PendingCommandBuffers>()
            .finish()
            .collect();
        if !buffers.is_empty() {
            world.resource::<RenderQueue>().submit(buffers);
        }
    }
}

/// `RenderStartup`: the device's async compute queue, unless the lever is off or there is none.
/// [`super::caps::init`] runs after it and records the queue as `HwCaps::async_compute`.
pub(crate) fn init(mut commands: Commands, device: Res<RenderDevice>, queue: Res<RenderQueue>) {
    if cfg!(target_arch = "wasm32") {
        return;
    }
    if crate::launch_env::var("WOW_ASYNC_COMPUTE").as_deref() == Some("0") {
        info!("modern: async compute off (WOW_ASYNC_COMPUTE=0)");
        return;
    }
    // The render diagnostics' statistics queries count graphics stages, which a compute-only
    // Vulkan family may not (`perf_probe::gpu_stats`); the counts are the same on one queue.
    if crate::perf_probe::gpu_stats::armed() {
        info!("modern: async compute off under WOW_GPU_STATS; all on the main queue");
        return;
    }
    match RenderAsyncComputeQueue::create(&device, &queue) {
        Some(async_queue) => {
            info!("modern: async compute on — light binning, probe trace and meter on their own queue");
            commands.insert_resource(async_queue);
        }
        None => info!("modern: async compute unavailable on this device; all on the main queue"),
    }
}

pub(super) fn plugin(app: &mut App) {
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(RenderStartup, init);
    }
}
