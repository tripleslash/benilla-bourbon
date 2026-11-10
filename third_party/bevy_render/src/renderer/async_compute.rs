//! **benilla fork (BENILLA.md, "The fourteenth delta").** A render-graph system can record a
//! command buffer for wgpu's async compute queue (`wgpu::AsyncComputeQueue`, native DX12 and
//! Vulkan) through [`AsyncComputeContext`].
//!
//! When the system's buffers are applied (in topological order, like [`RenderContext`]'s), the
//! main-queue command buffers queued before it are submitted first, then its own on the async
//! queue: GPU order follows graph order, and the async work runs beside whatever the graph records
//! after it. wgpu orders the two queues itself: an async submission waits for the main queue's
//! earlier submissions, and a main submission waits only for async work it conflicts with, so a
//! consumer gets the result by being submitted after it. [`FlushCommands`] in front of the
//! consumer is what lets the work between the two overlap. The frame's last main submission
//! (diagnostics resolve, present) waits for all of the frame's async work, so a timestamp written
//! there is resolved after it ([`wait_for_async_compute`]).
//!
//! [`RenderContext`]: super::RenderContext
//! [`FlushCommands`]: super::FlushCommands

use super::{wgpu_wrapper, PendingCommandBuffers, RenderDevice, RenderQueue};
use crate::diagnostic::internal::DiagnosticsRecorder;
use crate::render_resource::CommandEncoder;
use alloc::{sync::Arc, vec::Vec};
use bevy_derive::Deref;
use bevy_ecs::prelude::*;
use bevy_ecs::system::{Deferred, SystemBuffer, SystemMeta, SystemParam};
use bevy_ecs::world::DeferredWorld;

/// The device's async compute queue, where the backend has one. Insert it with
/// [`RenderAsyncComputeQueue::create`]; [`AsyncComputeContext`] records for it while it exists.
#[derive(Resource, Clone, Deref)]
pub struct RenderAsyncComputeQueue(pub Arc<WgpuAsyncComputeQueue>);

wgpu_wrapper! {
    /// The `wgpu::AsyncComputeQueue` behind [`RenderAsyncComputeQueue`].
    pub struct WgpuAsyncComputeQueue(wgpu::AsyncComputeQueue);
}

impl RenderAsyncComputeQueue {
    /// The device's one async compute queue, or `None` (WebGPU, GL, Metal, no second queue, or
    /// already made).
    pub fn create(device: &RenderDevice, queue: &RenderQueue) -> Option<Self> {
        device
            .wgpu_device()
            .create_async_compute_queue(queue)
            .map(|q| Self(Arc::new(WgpuAsyncComputeQueue::new(q))))
    }
}

/// The newest async submission of this frame, for [`wait_for_async_compute`].
#[derive(Resource, Default)]
pub struct AsyncComputeFrame {
    last: Option<wgpu::SubmissionIndex>,
}

impl AsyncComputeFrame {
    /// Whether any async work was submitted this frame.
    pub fn submitted(&self) -> bool {
        self.last.is_some()
    }
}

#[derive(Default)]
struct AsyncComputeStateInner {
    encoder: Option<CommandEncoder>,
    queue: Option<RenderAsyncComputeQueue>,
}

/// The [`SystemBuffer`] behind [`AsyncComputeContext`].
pub struct AsyncComputeState(WgpuAsyncComputeStateInner);

wgpu_wrapper!(struct WgpuAsyncComputeStateInner(AsyncComputeStateInner));

impl Default for AsyncComputeState {
    fn default() -> Self {
        Self(WgpuAsyncComputeStateInner::new(
            AsyncComputeStateInner::default(),
        ))
    }
}

impl SystemBuffer for AsyncComputeState {
    fn queue(&mut self, _system_meta: &SystemMeta, mut world: DeferredWorld) {
        let inner = &mut *self.0;
        let queue = inner.queue.take();
        let (Some(encoder), Some(queue)) = (inner.encoder.take(), queue) else {
            return;
        };
        // What the graph recorded before this system goes first, so the async work sees it.
        let earlier: Vec<_> = world
            .resource_mut::<PendingCommandBuffers>()
            .finish()
            .collect();
        if !earlier.is_empty() {
            world.resource::<RenderQueue>().submit(earlier);
        }
        let index = queue.submit([encoder.finish()]);
        world.resource_mut::<AsyncComputeFrame>().last = Some(index);
    }
}

/// A system parameter that records a command buffer for the async compute queue. Without the
/// queue ([`RenderAsyncComputeQueue`] absent) [`Self::command_encoder`] is `None`, and the system
/// records on the main queue instead.
#[derive(SystemParam)]
pub struct AsyncComputeContext<'w, 's> {
    state: Deferred<'s, AsyncComputeState>,
    queue: Option<Res<'w, RenderAsyncComputeQueue>>,
    diagnostics_recorder: Option<Res<'w, DiagnosticsRecorder>>,
}

impl<'w, 's> AsyncComputeContext<'w, 's> {
    /// Whether the async queue exists.
    pub fn available(&self) -> bool {
        self.queue.is_some()
    }

    /// The async command encoder, created at first use; `None` without the queue. Compute passes
    /// and copies only (wgpu refuses render passes and texture clears on it).
    pub fn command_encoder(&mut self) -> Option<&mut CommandEncoder> {
        let queue = self.queue.as_deref()?.clone();
        let inner = &mut *self.state.0;
        if inner.queue.is_none() {
            inner.queue = Some(queue.clone());
        }
        Some(inner.encoder.get_or_insert_with(|| {
            queue.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("async_compute"),
            })
        }))
    }

    /// The diagnostics recorder, for pass spans on the async encoder.
    pub fn diagnostic_recorder(&self) -> Option<Res<'w, DiagnosticsRecorder>> {
        self.diagnostics_recorder.as_ref().map(Res::clone)
    }
}

/// After the graph, before the diagnostics resolve: submit what is queued, then make the main
/// queue wait for the frame's async work, so the frame's last submission (the timestamp resolve,
/// the present) follows it. Async work overlaps everything the graph recorded after it; only that
/// last submission waits.
pub fn wait_for_async_compute(
    mut frame: ResMut<AsyncComputeFrame>,
    async_queue: Option<Res<RenderAsyncComputeQueue>>,
    queue: Res<RenderQueue>,
    mut pending: ResMut<PendingCommandBuffers>,
) {
    let (Some(last), Some(async_queue)) = (frame.last.take(), async_queue) else {
        return;
    };
    let buffers: Vec<_> = pending.finish().collect();
    if !buffers.is_empty() {
        queue.submit(buffers);
    }
    queue.wait_for(&async_queue, &last);
}
