//! fork: parallel command recording for render systems that draw several independent passes.
//!
//! Bevy 0.19's render graph is a schedule: a system records into [`RenderContext`], whose buffer
//! is flushed to the queue in system order, and there is no command-buffer generation task any
//! more. So a system that draws independent passes (shadow layers, voxel jobs) fans them out
//! itself: each job records into its own encoder on the compute pool, and the finished buffers
//! are handed to the context in job order. `bevy_tasks`' scope returns results in spawn order, so
//! the GPU sees exactly the serial order.
//!
//! On wasm32 a wgpu handle must not leave the thread that made it (`bevy_render`'s
//! `wasm_serial`), so the jobs record one after another on the calling thread into one encoder,
//! as before. `WOW_SERIAL_ENCODE=1` forces that path on native, for an A/B.

use bevy::render::diagnostic::DiagnosticsRecorder;
use bevy::render::render_phase::TrackedRenderPass;
use bevy::render::render_resource::{
    CommandEncoder, CommandEncoderDescriptor, RenderPassDescriptor,
};
use bevy::render::renderer::{RenderContext, RenderDevice};

/// Where a job records: its own encoder on native, the system's shared one on the serial path.
/// Mirrors the part of [`RenderContext`] the nodes use, so a node body records into either.
pub(crate) struct Recorder<'d> {
    device: &'d RenderDevice,
    diagnostics: Option<&'d DiagnosticsRecorder>,
    encoder: Option<CommandEncoder>,
}

impl<'d> Recorder<'d> {
    fn new(device: &'d RenderDevice, diagnostics: Option<&'d DiagnosticsRecorder>) -> Self {
        Self {
            device,
            diagnostics,
            encoder: None,
        }
    }

    pub(crate) fn render_device(&self) -> &'d RenderDevice {
        self.device
    }

    /// The pass timer, as [`RenderContext::diagnostic_recorder`]. Its spans are kept per thread,
    /// so a span opened on a worker closes there.
    pub(crate) fn diagnostic_recorder(&self) -> Option<&'d DiagnosticsRecorder> {
        self.diagnostics
    }

    /// The job's encoder, made at first use.
    pub(crate) fn command_encoder(&mut self) -> &mut CommandEncoder {
        let device = self.device;
        self.encoder.get_or_insert_with(|| {
            device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("par_encode"),
            })
        })
    }

    /// A tracked render pass on the job's encoder, as [`RenderContext::begin_tracked_render_pass`].
    pub(crate) fn begin_tracked_render_pass<'a>(
        &'a mut self,
        descriptor: RenderPassDescriptor<'_>,
    ) -> TrackedRenderPass<'a> {
        let device = self.device;
        let pass = self.command_encoder().begin_render_pass(&descriptor);
        TrackedRenderPass::new(device, pass)
    }

    /// The recorded commands, or `None` when the job recorded nothing.
    fn finish(self) -> Option<wgpu::CommandBuffer> {
        self.encoder.map(CommandEncoder::finish)
    }
}

/// What a job, its result and its recorder cross threads with: `Send` (and [`JobSync`]'s `Sync`)
/// on native, nothing on wasm32, whose wgpu handles are neither and whose jobs never leave the
/// calling thread ([`serial`]).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) trait JobSend: Send {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send> JobSend for T {}
#[cfg(target_arch = "wasm32")]
pub(crate) trait JobSend {}
#[cfg(target_arch = "wasm32")]
impl<T> JobSend for T {}

/// The recorder's bound beside [`JobSend`]: `Sync` on native, nothing on wasm32.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) trait JobSync: Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Sync> JobSync for T {}
#[cfg(target_arch = "wasm32")]
pub(crate) trait JobSync {}
#[cfg(target_arch = "wasm32")]
impl<T> JobSync for T {}

/// Whether jobs record serially: always on wasm32, on native when `WOW_SERIAL_ENCODE=1`.
pub(crate) fn serial() -> bool {
    if cfg!(target_arch = "wasm32") {
        return true;
    }
    static SERIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SERIAL.get_or_init(|| crate::launch_env::var("WOW_SERIAL_ENCODE").as_deref() == Some("1"))
}

/// Record `jobs` with `f`, in parallel on native, and hand the buffers to `ctx` in job order
/// after whatever `ctx` already holds. Returns each job's result, in job order.
pub(crate) fn record<J, R, F>(ctx: &mut RenderContext, jobs: Vec<J>, f: F) -> Vec<R>
where
    J: JobSend,
    R: JobSend + 'static,
    F: Fn(&mut Recorder<'_>, J) -> R + JobSync,
{
    record_holding_last(ctx, jobs, false, f).0
}

/// [`record`], except that with `hold_last` the last job records on an encoder of its own and its
/// commands come back instead of going to `ctx`, for the caller to hand to a later system's
/// context: static_gx's occluders record its colour pass with its depth prepass and submit it in
/// the colour pass's own slot (`static_gx::render::static_gx_occluders`). Without `hold_last` it
/// is [`record`] exactly, and returns `None`.
pub(crate) fn record_holding_last<J, R, F>(
    ctx: &mut RenderContext,
    mut jobs: Vec<J>,
    hold_last: bool,
    f: F,
) -> (Vec<R>, Option<wgpu::CommandBuffer>)
where
    J: JobSend,
    R: JobSend + 'static,
    F: Fn(&mut Recorder<'_>, J) -> R + JobSync,
{
    let device = ctx.render_device().clone();
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    if jobs.len() > 1 && !serial() {
        #[cfg(not(target_arch = "wasm32"))]
        return record_parallel(ctx, &device, diagnostics, jobs, hold_last, &f);
    }
    let held = if hold_last { jobs.pop() } else { None };
    let mut rec = Recorder::new(&device, diagnostics);
    let mut out: Vec<R> = jobs.into_iter().map(|job| f(&mut rec, job)).collect();
    if let Some(buffer) = rec.finish() {
        ctx.add_command_buffer(buffer);
    }
    let held = held.and_then(|job| {
        let mut rec = Recorder::new(&device, diagnostics);
        out.push(f(&mut rec, job));
        rec.finish()
    });
    (out, held)
}

/// Each job on its own encoder on the compute pool; the buffers go to `ctx` in job order, the
/// last one back to the caller instead with `hold_last`.
#[cfg(not(target_arch = "wasm32"))]
fn record_parallel<J, R, F>(
    ctx: &mut RenderContext,
    device: &RenderDevice,
    diagnostics: Option<&DiagnosticsRecorder>,
    jobs: Vec<J>,
    hold_last: bool,
    f: &F,
) -> (Vec<R>, Option<wgpu::CommandBuffer>)
where
    J: Send,
    R: Send + 'static,
    F: Fn(&mut Recorder<'_>, J) -> R + Sync,
{
    let run = move |job: J| {
        let mut rec = Recorder::new(device, diagnostics);
        let out = f(&mut rec, job);
        (rec.finish(), out)
    };
    let run = &run;
    // The calling thread records the first job and never ticks the pool's shared executor, so it
    // picks up no unrelated task (`crate::fanout`). Results come back in spawn order.
    let done = bevy::tasks::ComputeTaskPool::get().scope_with_executor(false, None, |scope| {
        let mut jobs = jobs.into_iter();
        if let Some(job) = jobs.next() {
            scope.spawn_on_scope(async move { run(job) });
        }
        for job in jobs {
            scope.spawn(async move { run(job) });
        }
    });
    let last = done.len().wrapping_sub(1);
    let mut out = Vec::with_capacity(done.len());
    let mut held = None;
    for (i, (buffer, r)) in done.into_iter().enumerate() {
        if hold_last && i == last {
            held = buffer;
        } else if let Some(buffer) = buffer {
            ctx.add_command_buffer(buffer);
        }
        out.push(r);
    }
    (out, held)
}
