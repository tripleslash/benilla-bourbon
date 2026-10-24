//! **benilla fork.** A `ComputeTaskPool::scope` that stays on the calling thread when the calling
//! thread is the only one allowed to touch wgpu.
//!
//! # Why
//!
//! Under `wasm32 + atomics`, [`WgpuWrapper`](crate::renderer::WgpuWrapper) is a
//! `send_wrapper::SendWrapper`, and a `SendWrapper` **panics the moment it is dereferenced from a
//! thread other than the one that made it**:
//!
//! ```text
//! Dereferenced SendWrapper<T> variable from a thread different to the one it has been created with.
//! ```
//!
//! That check is not paranoia. A wgpu handle on this target is a JS object reached through
//! wasm-bindgen's heap table, and that table is **per-thread** — the same index means something
//! different, or nothing, on a Web Worker. Sharing the handle is not slow or unsound-in-theory; it
//! reads the wrong object.
//!
//! Upstream knows this and guards for it: `renderer/mod.rs` runs command-buffer generation serially
//! on exactly this target, with exactly this reasoning. What it misses is
//! `batching::gpu_preprocessing`, where `write_batched_instance_buffers` and
//! `write_indirect_parameters_buffers` hand `&RenderDevice` and `&RenderQueue` straight into
//! `scope.spawn`. On native those run on the pool's threads and wgpu is genuinely `Send`; on a
//! threaded browser build they are the panic above, on the first frame that draws anything.
//!
//! # How
//!
//! [`compute_scope`] has two bodies with one call shape. The scope object a closure receives is
//! inferred, so a call site reads identically either way and the bodies of those two systems are
//! **byte-for-byte upstream's** — which is the point. A fork that retyped 90 lines of buffer writes
//! would be a fork nobody could diff against upstream on the next bevy release.
//!
//! Off wasm this is `ComputeTaskPool::get().scope`, unchanged.

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
use core::future::Future;

/// The parallel scope, everywhere threads may touch wgpu.
#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
pub fn compute_scope<'env, F, T>(f: F) -> Vec<T>
where
    F: for<'scope> FnOnce(&'scope bevy_tasks::Scope<'scope, 'env, T>),
    T: Send + 'static,
{
    bevy_tasks::ComputeTaskPool::get().scope(f)
}

/// Stands in for `bevy_tasks::Scope`, running each spawned future where it was spawned.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub struct SerialScope;

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
impl SerialScope {
    /// Run `fut` to completion right now, on this thread.
    ///
    /// Every future these two systems spawn is synchronous work in an `async` block — a
    /// `write_buffer` call and a tracing span — so this completes on the first poll and
    /// `block_on` never actually waits. (On this target `bevy_tasks::block_on` is the spinning
    /// implementation anyway; the browser's main thread may not park.)
    pub fn spawn<T>(&self, fut: impl Future<Output = T>) -> T {
        bevy_tasks::block_on(fut)
    }
}

/// The serial scope, on the one target where a worker must not hold a wgpu handle.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub fn compute_scope<F, R>(f: F) -> R
where
    F: FnOnce(&SerialScope) -> R,
{
    f(&SerialScope)
}
