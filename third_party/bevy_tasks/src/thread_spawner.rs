//! **benilla fork.** Where a [`TaskPool`](crate::TaskPool)'s threads come from, and how many of
//! them there should be.
//!
//! Upstream calls `std::thread::Builder::spawn` directly and reads
//! `std::thread::available_parallelism`. Neither works on `wasm32-unknown-unknown`, and not because
//! of a missing feature flag: the target takes its `Thread`, `available_parallelism` and
//! `yield_now` from std's `unsupported` backend even when `target_feature = "atomics"` is on
//! (`library/std/src/sys/thread/mod.rs` — the wasm arm implements `sleep` and nothing else). A
//! browser thread is a Web Worker, which only the page can create, so the platform half of this
//! cannot live in a crate that knows nothing about the page.
//!
//! So it is a hook. The host application installs a spawner once, before the first pool is built;
//! everything that has real threads keeps the `std` default and never notices this module exists.
//!
//! ## The failure mode this is shaped around
//!
//! A spawner is allowed to fail, and failing is not fatal. A pool whose threads all failed to start
//! has `thread_num() == 0`, and `TaskPool::scope` then does the work on the calling thread
//! (`task_pool.rs` forces `tick_task_pool_executor` for exactly this). That is the same behaviour
//! the single-threaded pool has always had on the web, so a page that cannot make workers degrades
//! to correct-but-serial rather than hanging. [`TaskPool::try_tick`](crate::TaskPool::try_tick) is
//! the other half of that guarantee — it is what lets a host drive `spawn`ed work when there are no
//! workers to run it.

use alloc::{boxed::Box, string::String};
use bevy_platform::sync::OnceLock;
use core::sync::atomic::{AtomicUsize, Ordering};

/// One thread the pool wants started.
///
/// Handed to a [`ThreadSpawner`], and handed *back* if it cannot be started — so a spawner never
/// has to drop a body on the floor, and a caller can tell "refused" from "started, then died".
pub struct ThreadRequest {
    /// A name for the thread, already numbered by the pool (`"Compute Task Pool (2)"`). Debuggers
    /// and browser thread lists show it; nothing reads it back.
    pub name: String,
    /// The stack size the pool asked for, if it asked. Worth honouring wherever the platform lets
    /// you: parry's collider build recurses, and a browser worker's default stack is small.
    pub stack_size: Option<usize>,
    /// The pool's worker loop. Runs until the pool is dropped, then returns.
    pub body: Box<dyn FnOnce() + Send + 'static>,
}

/// A host-provided way to start a pool thread.
///
/// A plain `fn` pointer rather than a boxed closure: it is installed once per process and read on
/// every pool build, so it needs neither an allocation nor a lock to read.
pub type ThreadSpawner = fn(ThreadRequest) -> Result<SpawnedThread, ThreadRequest>;

/// What a spawner hands back — the pool's end of a started thread.
#[derive(Debug)]
pub struct SpawnedThread(Kind);

// `crate::cfg::std!`, NOT `#[cfg(feature = "std")]`: this crate has no `std` feature of its own —
// it inherits the condition from `bevy_platform`, and the alias is the only way to ask. Spelling it
// as a feature test compiles fine and is silently ALWAYS FALSE, which turns the `std` spawner below
// into the "no threads here" one and panics every native pool at `Failed to spawn thread.`
// (measured: it did, across `benilla-world`'s suite, 2026-09-06).
crate::cfg::std! {
    if {
        #[derive(Debug)]
        enum Kind {
            /// A real OS thread, joined on pool drop exactly as upstream does.
            Std(std::thread::JoinHandle<()>),
            /// A thread the host owns and this crate cannot wait for — a Web Worker. **Not
            /// joined**, and that is deliberate rather than a shortcut: joining means blocking,
            /// blocking on the browser's main thread is `memory.atomic.wait32`, and that traps. The
            /// pool closes its shutdown channel before dropping the handles, so the worker's loop
            /// returns on its own.
            Detached,
        }

        impl SpawnedThread {
            /// Wrap a real OS thread.
            pub fn from_std(handle: std::thread::JoinHandle<()>) -> Self {
                Self(Kind::Std(handle))
            }

            /// Report a thread that was started but cannot be waited for.
            pub fn detached() -> Self {
                Self(Kind::Detached)
            }

            /// Wait for the thread to finish, where waiting is possible. A detached thread reports
            /// success without waiting — see `Kind::Detached`.
            pub(crate) fn join(self) -> Result<(), Box<dyn core::any::Any + Send + 'static>> {
                match self.0 {
                    Kind::Std(handle) => handle.join(),
                    Kind::Detached => Ok(()),
                }
            }
        }
    } else {
        #[derive(Debug)]
        enum Kind {
            /// The only kind there is without `std`: a thread some host started, which this crate
            /// cannot wait for.
            Detached,
        }

        impl SpawnedThread {
            /// Report a thread that was started but cannot be waited for.
            pub fn detached() -> Self {
                Self(Kind::Detached)
            }

            /// Nothing to wait for.
            pub(crate) fn join(self) -> Result<(), Box<dyn core::any::Any + Send + 'static>> {
                let Kind::Detached = self.0;
                Ok(())
            }
        }
    }
}

static SPAWNER: OnceLock<ThreadSpawner> = OnceLock::new();
/// 0 means "nobody said"; every real answer is at least 1.
static PARALLELISM: AtomicUsize = AtomicUsize::new(0);

/// Install the spawner every [`TaskPool`](crate::TaskPool) built from here on will use.
///
/// Call it once, before the first pool is built — under Bevy that means before `TaskPoolPlugin`,
/// which builds all three pools inside `Plugin::build`. Returns `Err` with the rejected spawner if
/// one is already installed: the first one keeps winning, so a second call cannot quietly change
/// where threads come from halfway through a run.
pub fn set_thread_spawner(spawner: ThreadSpawner) -> Result<(), ThreadSpawner> {
    SPAWNER.set(spawner)
}

/// Tell [`available_parallelism`](crate::available_parallelism) how many threads this machine can
/// actually run, for platforms that will not say.
///
/// The case this exists for is the browser: `std::thread::available_parallelism` is the
/// `unsupported` stub there and answers `Err`, which
/// [`available_parallelism`](crate::available_parallelism) turns into 1 — so an eight-core machine
/// would build one-worker pools. The real number is `navigator.hardwareConcurrency`, which only the
/// page can read.
///
/// Clamped to at least 1. A value set here wins over the platform's own answer on every target.
pub fn set_available_parallelism(threads: usize) {
    PARALLELISM.store(threads.max(1), Ordering::Relaxed);
}

/// The value [`set_available_parallelism`] was given, if it was given one.
pub(crate) fn hinted_parallelism() -> Option<usize> {
    match PARALLELISM.load(Ordering::Relaxed) {
        0 => None,
        n => Some(n),
    }
}

/// Start one pool thread through the installed spawner, or through `std` if none is installed.
pub(crate) fn spawn(request: ThreadRequest) -> Result<SpawnedThread, ThreadRequest> {
    match SPAWNER.get() {
        Some(spawner) => spawner(request),
        None => default_spawn(request),
    }
}

// Arm order matters. wasm32 is tested FIRST because it is the one target that has `std` and does
// not have `std::thread::spawn` — asking "do we have std?" would answer yes and then fail to
// compile. Everything after that is the ordinary question, asked through `cfg::std` rather than a
// `feature = "std"` test this crate does not have (see the note above `Kind`).
crate::cfg::switch! {
    // wasm32: no default. A host that wants threads installs a spawner; a host that does not gets
    // pools with no workers, which is a supported state — see this module's header.
    #[cfg(target_arch = "wasm32")] => {
        fn default_spawn(request: ThreadRequest) -> Result<SpawnedThread, ThreadRequest> {
            Err(request)
        }
    }
    // Every target that has `std::thread` keeps upstream's behaviour exactly, down to `name` and
    // `stack_size` being applied the same way.
    crate::cfg::std => {
        fn default_spawn(request: ThreadRequest) -> Result<SpawnedThread, ThreadRequest> {
            let ThreadRequest { name, stack_size, body } = request;
            let mut builder = std::thread::Builder::new().name(name.clone());
            if let Some(stack_size) = stack_size {
                builder = builder.stack_size(stack_size);
            }
            match builder.spawn(body) {
                Ok(handle) => Ok(SpawnedThread::from_std(handle)),
                // `Builder::spawn`'s error carries an `io::Error`, never the closure, so the body
                // is gone with the failed spawn. The pool only counts refusals, so the request it
                // gets back is a marker rather than something re-runnable.
                Err(_) => Err(ThreadRequest { name, stack_size, body: Box::new(|| {}) }),
            }
        }
    }
    // No `std`, so no threads to default to.
    _ => {
        fn default_spawn(request: ThreadRequest) -> Result<SpawnedThread, ThreadRequest> {
            Err(request)
        }
    }
}
