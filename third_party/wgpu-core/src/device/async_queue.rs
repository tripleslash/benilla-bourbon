//! BENILLA: async compute — a second, compute-only queue beside the device's main queue.
//!
//! `third_party/wgpu-core/BENILLA.md` ("Async compute") has the scheme; in short:
//!
//! - **Indices.** Both queues draw submission indices from the device's one counter
//!   (`CommandIndices::active_submission_index`), so an index names one submission on one queue
//!   and indices compare across queues. Each queue signals its own fence with its own indices.
//! - **Main before async.** An async submission records its memory initialization and every
//!   transition into its command buffers' start states on the *main* queue (a graphics queue can
//!   name every state, a compute queue cannot), submits that with the pending writes as one main
//!   submission, and waits on the GPU for it. So the async queue never runs before anything the
//!   main queue was given earlier.
//! - **Async before main.** A main submission waits on the GPU for the newest async submission
//!   it conflicts with ([`AsyncComputeQueue::conflicting_submission`]): one that wrote or changed
//!   the state of a resource the main submission uses, or used a resource the main submission
//!   writes or transitions. Two reads in one unchanged state run side by side.
//! - **States.** What an async command buffer writes or transitions is marked
//!   ([`hal::ASYNC_COMPUTE_BUFFER_MARK`]) in the device tracker, so the transition out of it names
//!   the compute-queue state (on DX12, without `PIXEL_SHADER_RESOURCE`).
//! - **Lifetimes.** Each queue retires its own submissions against its own fence. A buffer mapped
//!   while either queue still uses it waits for both; a resource destroyed while both use it is
//!   freed when the later retires ([`TempResource::Shared`]).

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::{
    mem::{self, ManuallyDrop},
    sync::atomic::{AtomicU64, Ordering},
};

use smallvec::SmallVec;

use super::{
    life::LifetimeTracker,
    queue::{validate_command_buffer, EncoderInFlight, Queue, QueueSubmitError, TempResource},
    Device, DeviceError,
};
use crate::{
    command::{CommandBuffer, CommandEncoder, EncodingApi, InnerCommandEncoder},
    hal_label,
    lock::{rank, Mutex, MutexGuard, RwLock},
    resource::{Buffer, Labeled, TextureInner},
    resource_log,
    track::{self, Tracker},
    FastHashMap, SubmissionIndex,
};

/// A compute-only queue that runs beside the device's main queue. See the module docs.
pub struct AsyncComputeQueue {
    raw: ManuallyDrop<Box<dyn hal::DynQueue>>,
    /// Signalled with this queue's submission indices.
    fence: RwLock<ManuallyDrop<Box<dyn hal::DynFence>>>,
    life: Mutex<LifetimeTracker>,
    /// The newest submission index this queue was given.
    last_submission: AtomicU64,
    /// The newest of this queue's submissions the main queue has been made to wait for.
    main_waited: AtomicU64,
    pub(crate) device: Arc<Device>,
}

crate::impl_resource_type!(AsyncComputeQueue);
impl Labeled for AsyncComputeQueue {
    fn label(&self) -> &str {
        "async compute queue"
    }
}
crate::impl_parent_device!(AsyncComputeQueue);

impl core::fmt::Debug for AsyncComputeQueue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AsyncComputeQueue")
            .field("last_submission", &self.last_submission)
            .finish_non_exhaustive()
    }
}

impl AsyncComputeQueue {
    pub(crate) fn new(
        device: &Arc<Device>,
        raw: Box<dyn hal::DynQueue>,
        fence: Box<dyn hal::DynFence>,
    ) -> Self {
        Self {
            raw: ManuallyDrop::new(raw),
            fence: RwLock::new(rank::ASYNC_QUEUE_FENCE, ManuallyDrop::new(fence)),
            life: Mutex::new(rank::ASYNC_QUEUE_LIFE_TRACKER, LifetimeTracker::new()),
            last_submission: AtomicU64::new(0),
            main_waited: AtomicU64::new(0),
            device: device.clone(),
        }
    }

    pub(crate) fn raw(&self) -> &dyn hal::DynQueue {
        self.raw.as_ref()
    }

    #[track_caller]
    pub(crate) fn lock_life<'a>(&'a self) -> MutexGuard<'a, LifetimeTracker> {
        self.life.lock()
    }

    /// The newest submission index this queue was given (0 before the first).
    pub fn last_submission(&self) -> SubmissionIndex {
        self.last_submission.load(Ordering::Acquire)
    }

    /// The queue's timestamp period, which on DX12 may differ from the main queue's.
    pub fn get_timestamp_period(&self) -> f32 {
        unsafe { self.raw().get_timestamp_period() }
    }

    /// A command encoder whose command buffers only this queue takes.
    pub(crate) fn create_command_encoder(
        self: &Arc<Self>,
        label: &crate::Label,
    ) -> Result<Arc<CommandEncoder>, DeviceError> {
        let device = &self.device;
        device.check_is_valid()?;
        let encoder = device
            .async_command_allocator
            .acquire_encoder(device.raw(), self.raw())
            .map_err(|e| device.handle_hal_error(e))?;
        Ok(Arc::new(CommandEncoder::new_for_queue(
            encoder, device, label, true,
        )))
    }

    /// The newest of this queue's submissions not yet waited for by the main queue that the main
    /// submission made of `executions` conflicts with (see the module docs).
    pub(crate) fn conflicting_submission(
        &self,
        executions: &[&EncoderInFlight],
    ) -> Option<SubmissionIndex> {
        let after = self.main_waited.load(Ordering::Acquire);
        let mut reason = None;
        let index = self.lock_life().newest_matching(after, |a| {
            executions.iter().any(|m| {
                reason = conflict(a, m);
                reason.is_some()
            })
        });
        if let (Some(index), Some(reason)) = (index, reason) {
            // `RUST_LOG=wgpu_core::device::async_queue=debug` names what stops an overlap.
            log::debug!("async compute: main queue waits for submission {index}: {reason}");
        }
        index
    }

    /// Make the main queue wait, on the GPU, until this queue has finished `index`.
    pub(crate) fn make_main_wait(
        &self,
        main: &dyn hal::DynQueue,
        index: SubmissionIndex,
    ) -> Result<(), DeviceError> {
        let fence = self.fence.read();
        self.make_main_wait_on(main, fence.as_ref(), index)
    }

    /// [`Self::make_main_wait`] for a caller that holds this queue's fence.
    fn make_main_wait_on(
        &self,
        main: &dyn hal::DynQueue,
        fence: &dyn hal::DynFence,
        index: SubmissionIndex,
    ) -> Result<(), DeviceError> {
        if index <= self.main_waited.load(Ordering::Acquire) {
            return Ok(());
        }
        unsafe { main.wait_for_fence(fence, index) }
            .map_err(|e| self.device.handle_hal_error(e))?;
        self.main_waited.fetch_max(index, Ordering::AcqRel);
        Ok(())
    }

    /// Make this queue wait, on the GPU, until the main queue has finished `index`. Every async
    /// submission already waits for the main queue's newest submission before it; this is for a
    /// caller that wants it stated.
    pub fn wait_for_main(&self, index: SubmissionIndex) -> Result<(), DeviceError> {
        let fence = self.device.fence.read();
        unsafe { self.raw().wait_for_fence(fence.as_ref(), index) }
            .map_err(|e| self.device.handle_hal_error(e))
    }

    /// The fence value this queue has reached.
    pub(crate) fn completed(&self) -> Result<SubmissionIndex, DeviceError> {
        let fence = self.fence.read();
        unsafe { self.device.raw().get_fence_value(fence.as_ref()) }
            .map_err(|e| self.device.handle_hal_error(e))
    }

    /// Wait on the CPU until this queue has finished `index` (or its newest submission, if that is
    /// older), up to `timeout`.
    pub(crate) fn wait_idle_to(
        &self,
        index: SubmissionIndex,
        timeout: Option<core::time::Duration>,
    ) -> Result<bool, DeviceError> {
        let target = index.min(self.last_submission());
        if target == 0 {
            return Ok(true);
        }
        let fence = self.fence.read();
        unsafe { self.device.raw().wait(fence.as_ref(), target, timeout) }
            .map_err(|e| self.device.handle_hal_error(e))
    }

    /// Retire what has finished; returns the buffers whose mapping no longer waits on this
    /// queue, and whether it has nothing in flight.
    pub(crate) fn maintain(&self) -> Result<(Vec<Arc<Buffer>>, bool), DeviceError> {
        let done = self.completed()?;
        let mut life = self.lock_life();
        let closures = life.triage_submissions(done);
        debug_assert!(
            closures.is_empty(),
            "the async queue holds no work-done closures"
        );
        drop(closures);
        Ok((life.take_ready_to_map(), life.queue_empty()))
    }

    /// Submit async compute command buffers. See the module docs for what runs where.
    pub fn submit(
        &self,
        command_buffers: &[Arc<CommandBuffer>],
    ) -> Result<SubmissionIndex, (SubmissionIndex, QueueSubmitError)> {
        profiling::scope!("AsyncComputeQueue::submit");
        let device = &self.device;
        let Some(main) = device.get_queue() else {
            return Err((0, QueueSubmitError::Queue(DeviceError::Lost)));
        };

        let snatch_guard = device.snatchable_lock.read();
        let mut main_fence = device.fence.write();
        let mut async_fence = self.fence.write();
        let mut indices = device.command_indices.write();
        indices.active_submission_index += 1;
        let transit_index = indices.active_submission_index;
        indices.active_submission_index += 1;
        let submit_index = indices.active_submission_index;

        let res = 'error: {
            if let Err(e) = device.check_is_valid() {
                break 'error Err(e.into());
            }

            // The main-queue encoder that initializes memory and transitions into the async
            // command buffers' start states.
            let transit_raw = match device
                .command_allocator
                .acquire_encoder(device.raw(), main.raw())
            {
                Ok(raw) => raw,
                Err(e) => break 'error Err(device.handle_hal_error(e).into()),
            };
            let mut transit = InnerCommandEncoder {
                raw: ManuallyDrop::new(transit_raw),
                list: Vec::new(),
                device: device.clone(),
                is_open: false,
                api: EncodingApi::InternalUse,
                label: "(wgpu internal) Async compute transit".into(),
                async_compute: false,
            };
            if let Err(e) = transit.open_pass(hal_label(
                Some("(wgpu internal) Async compute transit"),
                device.instance_flags,
            )) {
                break 'error Err(e.into());
            }

            let mut executions = Vec::new();
            let mut first_error = None;
            let mut surface_owned = FastHashMap::default();
            let mut used_surface = track::TextureUsageScope::default();
            for command_buffer in command_buffers {
                used_surface.set_size(device.tracker_indices.textures.size());
                let data = command_buffer.take_finished();
                if first_error.is_some() {
                    continue;
                }
                let data = match data {
                    Ok(data) => data,
                    Err(e) => {
                        first_error = Some(e.into());
                        continue;
                    }
                };
                if !data.async_compute {
                    first_error = Some(QueueSubmitError::WrongQueue(false));
                    continue;
                }
                if let Err(e) = validate_command_buffer(
                    command_buffer,
                    &main,
                    &data,
                    &snatch_guard,
                    &mut surface_owned,
                    &mut used_surface,
                    &mut indices,
                ) {
                    first_error = Some(e);
                    continue;
                }
                if !surface_owned.is_empty() {
                    first_error = Some(QueueSubmitError::AsyncComputeSurfaceTexture);
                    continue;
                }
                data.set_acceleration_structure_dependencies(&snatch_guard);
                let mut baked = data.into_baked_commands();
                baked.trackers.mark_for_async_queue();

                // Initialize and transition on the main queue: lend the baked commands the transit
                // encoder for the two calls that record into theirs.
                mem::swap(&mut *baked.encoder.raw, &mut *transit.raw);
                let mut trackers = device.trackers.lock();
                let init = baked
                    .initialize_buffer_memory(&mut trackers, &snatch_guard)
                    .and_then(|()| {
                        baked.initialize_texture_memory(&mut trackers, device, &snatch_guard)
                    });
                if init.is_ok() {
                    CommandEncoder::insert_barriers_from_device_tracker(
                        baked.encoder.raw.as_mut(),
                        &mut trackers,
                        &baked.trackers,
                        &snatch_guard,
                    );
                }
                drop(trackers);
                mem::swap(&mut *baked.encoder.raw, &mut *transit.raw);
                if let Err(e) = init {
                    break 'error Err(e.into());
                }

                executions.push(EncoderInFlight {
                    inner: baked.encoder,
                    trackers: baked.trackers,
                    temp_resources: baked.temp_resources,
                    _indirect_draw_validation_resources: baked.indirect_draw_validation_resources,
                    pending_buffers: FastHashMap::default(),
                    pending_textures: FastHashMap::default(),
                    pending_blas_s: FastHashMap::default(),
                });
            }
            if let Some(e) = first_error {
                break 'error Err(e);
            }
            if let Err(e) = transit.close() {
                break 'error Err(e.into());
            }

            // The main submission: the pending writes, then the transit. It waits for whatever
            // earlier async work it conflicts with, as any main submission does.
            let mut pending_writes = main.pending_writes.lock();
            let mut main_surface_owned = FastHashMap::default();
            {
                used_surface = track::TextureUsageScope::default();
                used_surface.set_size(device.tracker_indices.textures.size());
                for texture in pending_writes.dst_textures.values() {
                    if let Ok(TextureInner::Surface { .. }) = texture.try_inner(&snatch_guard) {
                        main_surface_owned.insert(Arc::as_ptr(texture), texture.clone());
                        unsafe {
                            used_surface
                                .merge_single(texture, None, wgt::TextureUses::PRESENT)
                                .unwrap()
                        };
                    }
                }
                if !used_surface.is_empty() {
                    let mut trackers = device.trackers.lock();
                    let barriers = trackers
                        .textures
                        .set_from_usage_scope_and_drain_transitions(&used_surface, &snatch_guard)
                        .collect::<Vec<_>>();
                    unsafe {
                        pending_writes
                            .command_encoder
                            .transition_textures(&barriers)
                    };
                }
            }
            let mut main_executions = Vec::new();
            match pending_writes.pre_submit(&device.command_allocator, device, &main) {
                Ok(Some(pending)) => main_executions.push(pending),
                Ok(None) => {}
                Err(e) => break 'error Err(e.into()),
            }
            main_executions.push(EncoderInFlight {
                inner: transit,
                trackers: Tracker::new(device.ordered_buffer_usages, device.ordered_texture_usages),
                temp_resources: Vec::new(),
                _indirect_draw_validation_resources: crate::indirect_validation::DrawResources::new(
                    device.clone(),
                ),
                pending_buffers: FastHashMap::default(),
                pending_textures: FastHashMap::default(),
                pending_blas_s: FastHashMap::default(),
            });
            {
                // The transit changes the state of everything the async command buffers use, so
                // it conflicts as they do.
                let refs = main_executions
                    .iter()
                    .chain(executions.iter())
                    .collect::<Vec<_>>();
                if let Some(index) = self.conflicting_submission(&refs) {
                    let fence = async_fence.as_ref();
                    if let Err(e) = self.make_main_wait_on(main.raw(), fence, index) {
                        break 'error Err(e.into());
                    }
                }
            }
            let main_buffers = main_executions
                .iter()
                .flat_map(|e| e.inner.list.iter().map(|b| b.as_ref()))
                .collect::<Vec<_>>();
            let main_surfaces = main_surface_owned
                .values()
                .map(|texture| match texture.inner.get(&snatch_guard) {
                    Some(TextureInner::Surface { raw, .. }) => raw.as_ref(),
                    _ => unreachable!(),
                })
                .collect::<SmallVec<[_; 2]>>();
            if let Err(e) = unsafe {
                main.raw().submit(
                    &main_buffers,
                    &main_surfaces,
                    (main_fence.as_mut(), transit_index),
                )
            } {
                break 'error Err(device.handle_hal_error(e).into());
            }
            device
                .last_successful_submission_index
                .fetch_max(transit_index, Ordering::SeqCst);
            main.lock_life()
                .track_submission(transit_index, main_executions);
            drop(pending_writes);

            // The async submission, after the transit.
            if let Err(e) = unsafe {
                self.raw()
                    .wait_for_fence(main_fence.as_ref(), transit_index)
            } {
                break 'error Err(device.handle_hal_error(e).into());
            }
            let async_buffers = executions
                .iter()
                .flat_map(|e| e.inner.list.iter().map(|b| b.as_ref()))
                .collect::<Vec<_>>();
            if let Err(e) = unsafe {
                self.raw()
                    .submit(&async_buffers, &[], (async_fence.as_mut(), submit_index))
            } {
                break 'error Err(device.handle_hal_error(e).into());
            }
            self.last_submission
                .fetch_max(submit_index, Ordering::SeqCst);
            self.lock_life().track_submission(submit_index, executions);
            Ok(())
        };
        drop(indices);
        drop(async_fence);
        drop(main_fence);
        drop(snatch_guard);

        match res {
            Ok(()) => {
                // Retire what finished, as a main submission does.
                if let Err(e) = device.poll(wgt::PollType::Poll) {
                    log::warn!("async compute: poll after submit: {e}");
                }
                Ok(submit_index)
            }
            Err(e) => Err((submit_index, e)),
        }
    }
}

/// Why async execution `a` and main execution `m` must not overlap, if they must not: the first
/// resource they share other than for one unchanged read.
fn conflict(a: &EncoderInFlight, m: &EncoderInFlight) -> Option<alloc::string::String> {
    use alloc::format;
    for buffer in m.trackers.buffers.used_resources() {
        if a.trackers.buffers.contains(buffer) {
            match (
                a.trackers.buffers.read_only_state(buffer),
                m.trackers.buffers.read_only_state(buffer),
            ) {
                (Some(x), Some(y)) if x == y => {}
                (x, y) => {
                    return Some(format!(
                        "{} (async {x:?}, main {y:?})",
                        buffer.error_ident()
                    ))
                }
            }
        }
    }
    for texture in m.trackers.textures.used_resources() {
        if a.trackers.textures.contains(texture) {
            match (
                a.trackers.textures.read_only_state(texture),
                m.trackers.textures.read_only_state(texture),
            ) {
                (Some(x), Some(y)) if x == y => {}
                (x, y) => {
                    return Some(format!(
                        "{} (async {x:?}, main {y:?})",
                        texture.error_ident()
                    ))
                }
            }
        }
    }
    if let Some(b) = m
        .pending_buffers
        .values()
        .find(|b| a.trackers.buffers.contains(b))
    {
        return Some(format!("{} (a pending write)", b.error_ident()));
    }
    m.pending_textures
        .values()
        .find(|t| a.trackers.textures.contains(t))
        .map(|t| format!("{} (a pending write)", t.error_ident()))
}

/// Schedule `temp` (a destroyed resource) for release once neither queue's submissions use it:
/// `main_index` and `async_index` are the newest submission on each that does.
pub(crate) fn schedule_destruction(
    main: &mut LifetimeTracker,
    main_index: Option<SubmissionIndex>,
    async_life: Option<(&mut LifetimeTracker, SubmissionIndex)>,
    temp: TempResource,
) {
    match (main_index, async_life) {
        (Some(i), None) => main.schedule_resource_destruction(temp, i),
        (None, Some((life, j))) => life.schedule_resource_destruction(temp, j),
        (Some(i), Some((life, j))) => {
            let shared = Arc::new(temp);
            main.schedule_resource_destruction(TempResource::Shared(shared.clone()), i);
            life.schedule_resource_destruction(TempResource::Shared(shared), j);
        }
        (None, None) => drop(temp),
    }
}

impl Drop for AsyncComputeQueue {
    fn drop(&mut self) {
        resource_log!("Drop {}", self.error_ident());
        let last = self.last_submission();
        if last > 0 {
            match self.wait_idle_to(last, Some(core::time::Duration::from_secs(5))) {
                Ok(true) => {}
                Ok(false) => log::error!("async compute: queue still busy at drop"),
                Err(e) => log::error!("async compute: wait at drop failed: {e}"),
            }
        }
        let done = self.completed().unwrap_or(last);
        drop(self.life.lock().triage_submissions(done));
        let fence = unsafe { ManuallyDrop::take(&mut *self.fence.write()) };
        unsafe { self.device.raw().destroy_fence(fence) };
        unsafe { ManuallyDrop::drop(&mut self.raw) };
    }
}

impl Queue {
    /// BENILLA: async compute. Make this queue wait, on the GPU, until `async_queue` has finished
    /// `index`. Main submissions already wait where they conflict with async work; this is for a
    /// caller that wants it stated.
    pub fn wait_for_async(
        &self,
        async_queue: &AsyncComputeQueue,
        index: SubmissionIndex,
    ) -> Result<(), DeviceError> {
        async_queue.make_main_wait(self.raw(), index)
    }
}
