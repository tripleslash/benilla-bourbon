//! **benilla fork.** A non-blocking stand-in for `std::sync::Mutex`, for the one target where
//! blocking is not allowed.
//!
//! # Why
//!
//! On `wasm32` with `+atomics`, std's `Mutex` is futex-based, and a futex wait is the
//! `memory.atomic.wait32` instruction. **The browser's main thread is forbidden to execute it** — it
//! throws `Atomics.wait cannot be called in this context` out of the middle of whatever was
//! running, and the module dies with it.
//!
//! This crate is where that bites hardest, because it is not obvious that it is a lock at all.
//! `Event` is what `async-channel` (and `async-lock`, and everything else in the smol family) uses
//! to park and wake a waiter, so **every channel send and receive that has to register or notify a
//! listener takes this mutex** — `intrusive.rs`'s `List`. In benilla's threaded browser build the
//! page's main thread and a Web Worker per core are on opposite ends of those channels: the task
//! pools' shutdown channel, every `Task` handed back by `TaskPool::spawn_pooled`, and the app's own
//! net and asset plumbing. Contention there is ordinary, and on the main thread it was fatal.
//!
//! It is worth saying plainly that this crate reads as though it were already safe: `slab.rs` — the
//! `no_std` path — carries its own spinning `Mutex` written for exactly this hazard. But `slab.rs`
//! is only compiled *without* `std`, and with `std` on (which is always, here) the crate takes
//! `intrusive.rs` and `std::sync::Mutex` on every target, wasm included.
//!
//! # The trade
//!
//! This spins instead of sleeping. The critical sections it guards are a linked-list insert,
//! remove, or walk — no allocation under the lock, no nesting, no I/O — so a waiter spins for
//! nanoseconds. The worst case is a holder being descheduled mid-section, and the cost of that is a
//! spinning waiter rather than a trapped one. **A stall is recoverable; a trap is not.**
//!
//! Rust's own wasm allocator makes the same trade for the same reason, and says so:
//! `library/std/src/sys/alloc/wasm.rs` spins on its global `dlmalloc` lock because "the main thread
//! in a web browser *cannot ever block*, no exceptions".
//!
//! # Poisoning
//!
//! There is none. `PoisonError` exists so `intrusive.rs`'s
//! `.lock().unwrap_or_else(|e| e.into_inner())` — upstream's line, unmodified — keeps compiling.

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, Ordering};

/// The error a poisoned lock would return. Never constructed — see the module header.
pub struct PoisonError<T> {
    guard: T,
}

impl<T> PoisonError<T> {
    /// The guard, which is always valid because nothing is ever poisoned.
    pub fn into_inner(self) -> T {
        self.guard
    }
}

/// A spinning mutex with the slice of `std::sync::Mutex`'s surface this crate uses.
pub struct Mutex<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

// SAFETY: `locked` serialises access to the `UnsafeCell`; only the thread that moved it false->true
// may touch the value, and it restores false on guard drop.
unsafe impl<T: Send> Send for Mutex<T> {}
// SAFETY: as above — a shared reference only ever hands out an exclusive `MutexGuard`.
unsafe impl<T: Send> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    /// Create a new mutex.
    pub const fn new(value: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// Lock, spinning until it is free. Never blocks, so never traps.
    #[allow(clippy::result_unit_err)]
    pub fn lock(&self) -> Result<MutexGuard<'_, T>, PoisonError<MutexGuard<'_, T>>> {
        loop {
            if let Some(guard) = self.raw_try_lock() {
                return Ok(guard);
            }
            core::hint::spin_loop();
        }
    }

    /// Lock if it is free right now. `intrusive.rs` uses this for its `Debug`/`len` peek, where
    /// giving up is the correct answer.
    #[allow(clippy::result_unit_err)]
    pub fn try_lock(&self) -> Result<MutexGuard<'_, T>, ()> {
        self.raw_try_lock().ok_or(())
    }

    fn raw_try_lock(&self) -> Option<MutexGuard<'_, T>> {
        self.locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| MutexGuard { lock: self })
    }
}

/// Exclusive access to a [`Mutex`]'s value.
pub struct MutexGuard<'a, T> {
    lock: &'a Mutex<T>,
}

impl<T> Deref for MutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: holding this guard means `locked` is true and this thread set it.
        unsafe { &*self.lock.value.get() }
    }
}

impl<T> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as above, and `&mut self` makes this the only guard reference.
        unsafe { &mut *self.lock.value.get() }
    }
}

impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}
