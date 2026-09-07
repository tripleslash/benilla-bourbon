#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]
#![no_std]

/// Configuration information for this crate.
pub mod cfg {
    pub(crate) use bevy_platform::cfg::*;

    pub use bevy_platform::cfg::{alloc, std, web};

    define_alias! {
        #[cfg(feature = "async_executor")] => {
            /// Indicates `async_executor` is used as the future execution backend.
            async_executor
        }

        // **benilla fork.** Upstream gates this on `target_arch`, which makes "wasm" and "has no
        // threads" the same statement. They stopped being the same statement when the browser got
        // shared memory: built with `-Ctarget-feature=+atomics` the module's linear memory is a
        // SharedArrayBuffer, and a Web Worker running that same module against that same memory is
        // a thread in every sense this crate cares about. `target_feature` is the honest question;
        // on every non-wasm target the condition is unchanged.
        //
        // What this does NOT do is conjure threads. `std::thread::spawn` is still `unsupported` on
        // `wasm32-unknown-unknown` even with atomics on — see `thread_spawner`, which is where the
        // pool's threads actually come from, and which answers "none" unless a host installs a
        // spawner.
        #[cfg(all(
            any(not(target_arch = "wasm32"), target_feature = "atomics"),
            feature = "multi_threaded"
        ))] => {
            /// Indicates multithreading support.
            multi_threaded
        }

        #[cfg(target_arch = "wasm32")] => {
            /// Indicates the current target requires additional `Send` bounds.
            conditional_send
        }

        #[cfg(feature = "async-io")] => {
            /// Indicates `async-io` will be used for the implementation of `block_on`.
            async_io
        }

        #[cfg(feature = "futures-lite")] => {
            /// Indicates `futures-lite` will be used for the implementation of `block_on`.
            futures_lite
        }
    }
}

cfg::std! {
    extern crate std;
}

extern crate alloc;

cfg::conditional_send! {
    if {
        /// Use [`ConditionalSend`] to mark an optional Send trait bound. Useful as on certain platforms (eg. Wasm),
        /// futures aren't Send.
        pub trait ConditionalSend {}
        impl<T> ConditionalSend for T {}
    } else {
        /// Use [`ConditionalSend`] to mark an optional Send trait bound. Useful as on certain platforms (eg. Wasm),
        /// futures aren't Send.
        pub trait ConditionalSend: Send {}
        impl<T: Send> ConditionalSend for T {}
    }
}

/// Use [`ConditionalSendFuture`] for a future with an optional Send trait bound, as on certain platforms (eg. Wasm),
/// futures aren't Send.
pub trait ConditionalSendFuture: Future + ConditionalSend {}

impl<T: Future + ConditionalSend> ConditionalSendFuture for T {}

use alloc::boxed::Box;

/// An owned and dynamically typed Future used when you can't statically type your result or need to add some indirection.
pub type BoxedFuture<'a, T> = core::pin::Pin<Box<dyn ConditionalSendFuture<Output = T> + 'a>>;

// Modules
mod executor;
pub mod futures;
mod iter;
mod slice;
mod task;
mod thread_spawner;
mod usages;

cfg::async_executor! {
    if {} else {
        mod edge_executor;
    }
}

// Exports
pub use iter::ParallelIterator;
pub use slice::{ParallelSlice, ParallelSliceMut};
pub use task::Task;
// **benilla fork.** How a host supplies threads on a platform whose `std` has none. See the
// module header; on every target with real `std::thread` this is inert.
pub use thread_spawner::{
    SpawnedThread, ThreadRequest, ThreadSpawner, set_available_parallelism, set_thread_spawner,
};
pub use usages::{AsyncComputeTaskPool, ComputeTaskPool, IoTaskPool};

pub use futures_lite;
pub use futures_lite::future::poll_once;

cfg::web! {
    if {} else {
        pub use usages::tick_global_task_pools_on_main_thread;
    }
}

cfg::multi_threaded! {
    if {
        mod task_pool;
        mod thread_executor;

        pub use task_pool::{Scope, TaskPool, TaskPoolBuilder};
        pub use thread_executor::{ThreadExecutor, ThreadExecutorTicker};
    } else {
        mod single_threaded_task_pool;

        pub use single_threaded_task_pool::{Scope, TaskPool, TaskPoolBuilder, ThreadExecutor};
    }
}

// **benilla fork.** wasm takes the spinning implementation, and this is a correctness fix rather
// than a tuning choice.
//
// `futures_lite`'s `block_on` parks the calling thread through `Mutex`/`Condvar`, which on wasm
// lowers to `memory.atomic.wait32`. The browser's MAIN thread is not allowed to execute that
// instruction: it does not throw a catchable error, it **traps**, and a trap takes the whole module
// down. `TaskPool::scope` calls `block_on` on whatever thread entered the scope, and on the web
// that is always the thread that draws — so the parking implementation is not merely slow there,
// it is a crash waiting for the first `par_iter_mut` that has to wait on a worker.
//
// Spinning is the right shape for the case it is actually in: a fork-join `scope` where the main
// thread has nothing to do until the workers finish, measured in a fraction of a frame. Worker
// threads never come through here — `task_pool.rs`'s worker loop names `futures_lite`'s parking
// `block_on` directly, because a worker MAY block and a spinning worker would burn a core.
cfg::switch! {
    #[cfg(target_arch = "wasm32")] => {
        pub use crate::spin_block_on as block_on;
    }
    cfg::async_io => {
        pub use async_io::block_on;
    }
    cfg::futures_lite => {
        pub use futures_lite::future::block_on;
    }
    _ => {
        pub use crate::spin_block_on as block_on;
    }
}

/// Blocks on the supplied `future` by polling it as fast as the CPU will allow.
///
/// This is upstream's fallback `block_on` — the one used when neither `async-io` nor `futures-lite`
/// is enabled — lifted out so it can also be named where a *parking* `block_on` is illegal. The
/// browser's main thread is that place: see the note above this function's re-export.
///
/// It never yields. Only call it where the wait is short and bounded by other threads finishing.
pub fn spin_block_on<T>(future: impl Future<Output = T>) -> T {
    use core::task::{Context, Poll};

    // Pin the future on the stack.
    let mut future = core::pin::pin!(future);

    // We don't care about the waker as we're just going to poll as fast as possible.
    let cx = &mut Context::from_waker(core::task::Waker::noop());

    // Keep polling until the future is ready.
    loop {
        match future.as_mut().poll(cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => core::hint::spin_loop(),
        }
    }
}

/// The tasks prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    #[doc(hidden)]
    pub use crate::{
        block_on,
        iter::ParallelIterator,
        slice::{ParallelSlice, ParallelSliceMut},
        usages::{AsyncComputeTaskPool, ComputeTaskPool, IoTaskPool},
    };
}

/// Gets the logical CPU core count available to the current process.
///
/// This is identical to `std::thread::available_parallelism`, except
/// it will return a default value of 1 if it internally errors out.
///
/// This will always return at least 1.
pub fn available_parallelism() -> usize {
    // **benilla fork.** A host may answer this itself, and on the browser it has to: there
    // `std::thread::available_parallelism` is the `unsupported` stub, returns `Err`, and the
    // `unwrap_or(1)` below would build one-worker pools on an eight-core machine. The real number
    // is `navigator.hardwareConcurrency`, which only the page can read — so it arrives through
    // `set_available_parallelism`. Unset on every other target, where the line below is upstream's.
    if let Some(threads) = thread_spawner::hinted_parallelism() {
        return threads;
    }
    cfg::switch! {{
        cfg::std => {
            std::thread::available_parallelism()
                .map(core::num::NonZero::<usize>::get)
                .unwrap_or(1)
        }
        _ => {
            1
        }
    }}
}
