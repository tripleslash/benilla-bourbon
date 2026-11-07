//! benilla local patch (async compute): a second, compute-only queue on native DX12 and Vulkan.
//! `third_party/wgpu/PATCH.md`, "Async compute", has the scheme and the API.

use crate::*;

/// A queue that runs compute passes and copies beside the main [`Queue`], made by
/// [`Device::create_async_compute_queue`].
///
/// Its command buffers come from [`AsyncComputeQueue::create_command_encoder`] and go only to
/// [`AsyncComputeQueue::submit`]; they may not record render passes or texture clears.
///
/// Ordering is automatic in both directions: a submission here waits on the GPU for every main
/// submission made before it, and a main submission waits for the newest submission here that
/// wrote, or changed the state of, a resource it uses (or used one it writes). Two reads of a
/// resource in one unchanged state run side by side. [`Queue::wait_for`] and
/// [`AsyncComputeQueue::wait_for`] state a wait outright.
#[derive(Debug, Clone)]
pub struct AsyncComputeQueue {
    #[cfg(wgpu_core)]
    pub(crate) inner: alloc::sync::Arc<crate::backend::wgpu_core::CoreAsyncComputeQueue>,
    /// The main queue, which runs the command buffers' deferred callbacks.
    pub(crate) main: Queue,
}

#[cfg(send_sync)]
static_assertions::assert_impl_all!(AsyncComputeQueue: Send, Sync);

impl Device {
    /// benilla local patch (async compute). The device's one async compute queue, beside `queue`
    /// (this device's main queue). `None` on WebGPU, on GL and Metal, without a compute queue (a
    /// Vulkan device without timeline semaphores, or without a second compute-capable queue),
    /// and on every call after the first.
    pub fn create_async_compute_queue(&self, queue: &Queue) -> Option<AsyncComputeQueue> {
        #[cfg(wgpu_core)]
        if let Some(device) = self.inner.as_core_opt() {
            return device
                .create_async_compute_queue()
                .map(|inner| AsyncComputeQueue {
                    inner: alloc::sync::Arc::new(inner),
                    main: queue.clone(),
                });
        }
        let _ = queue;
        None
    }
}

impl AsyncComputeQueue {
    /// A command encoder for this queue: compute passes and copies only.
    pub fn create_command_encoder(&self, desc: &CommandEncoderDescriptor<'_>) -> CommandEncoder {
        #[cfg(wgpu_core)]
        return CommandEncoder {
            inner: self.inner.create_command_encoder(desc),
            actions: Default::default(),
        };
        #[cfg(not(wgpu_core))]
        unreachable!("no async compute queue without wgpu-core ({desc:?})")
    }

    /// Submit command buffers from [`Self::create_command_encoder`] on this queue. The index it
    /// returns is valid for [`Device::poll`] and [`Queue::wait_for`].
    pub fn submit<I: IntoIterator<Item = CommandBuffer>>(
        &self,
        command_buffers: I,
    ) -> SubmissionIndex {
        let mut actions = crate::api::DeferredCommandBufferActions::default();
        let mut command_buffers = command_buffers.into_iter().map(|comb| {
            actions.append(&mut comb.actions.lock());
            comb.buffer
        });
        #[cfg(wgpu_core)]
        let index = self.inner.submit(&mut command_buffers);
        #[cfg(not(wgpu_core))]
        let index = unreachable!("no async compute queue without wgpu-core");
        actions.execute(&self.main.inner);
        SubmissionIndex { index }
    }

    /// Make this queue wait, on the GPU, for main-queue submission `index`. Every submission
    /// here already waits for the main queue's newest; this states it.
    pub fn wait_for(&self, index: &SubmissionIndex) {
        #[cfg(wgpu_core)]
        self.inner.wait_for_main(index.index);
        let _ = index;
    }

    /// This queue's timestamp period, in nanoseconds per tick (may differ from the main queue's
    /// on DX12).
    pub fn get_timestamp_period(&self) -> f32 {
        #[cfg(wgpu_core)]
        return self.inner.timestamp_period();
        #[allow(unreachable_code)]
        0.0
    }
}

impl Queue {
    /// benilla local patch (async compute). Make this queue wait, on the GPU, until
    /// `async_queue` has finished its submission `index`.
    pub fn wait_for(&self, async_queue: &AsyncComputeQueue, index: &SubmissionIndex) {
        #[cfg(wgpu_core)]
        if let Some(queue) = self.inner.as_core_opt() {
            queue.wait_for_async(&async_queue.inner, index.index);
        }
        let _ = (async_queue, index);
    }
}
