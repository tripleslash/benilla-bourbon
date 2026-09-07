//! **benilla fork.** Non-blocking stand-ins for `std::sync::{Mutex, RwLock}`, for the one target
//! where blocking is not allowed.
//!
//! # Why
//!
//! On `wasm32` with `+atomics`, std's `Mutex` and `RwLock` are futex-based, and a futex wait is the
//! `memory.atomic.wait32` instruction. **The browser's main thread is forbidden to execute it.** It
//! does not return an error and it does not merely stall — it throws
//! `Atomics.wait cannot be called in this context` out of the middle of whatever was running, which
//! for a wasm module means a `RuntimeError` unwinding through code that has no idea what happened.
//!
//! That matters here because this executor is shared between the page's main thread and a pool of
//! Web Worker threads. Every path where the main thread meets the pool takes one of these locks:
//! `Executor::spawn` takes `active`, ticking takes `sleepers`, and running takes `local_queues`.
//! With one worker per core hammering the same executor, contention is not a corner case — it is
//! what a `par_iter_mut` sweep does on every frame. Measured: the client crashed there on the first
//! frame it drew.
//!
//! # The trade
//!
//! These spin instead of sleeping. That is the right shape for *this* executor and would be the
//! wrong shape for a general-purpose lock: every critical section guarded here is a handful of
//! instructions — a `Slab` insert or remove, a `Vec` push, a few field updates — with no allocation
//! under the lock and no nesting. The worst case is a holder being descheduled mid-section, and the
//! cost of that is a spinning waiter rather than a trapped one. **A stall is recoverable; a trap is
//! not.**
//!
//! This is the same trade `event-listener` already makes for the same reason: its own `Mutex`
//! (`event-listener/src/slab.rs`) is a spin lock that gives up rather than block. It is why that
//! crate is safe on the main thread and this one was not.
//!
//! # Poisoning
//!
//! There is none, and the types below still carry the shape of it. `PoisonError` and
//! `TryLockError::Poisoned` exist so the call sites in `lib.rs` — which are upstream's, unmodified —
//! keep compiling; `Poisoned` is simply never constructed. Nothing is lost by that: this target
//! builds with `panic = "abort"`, so a panic inside a critical section takes the module down rather
//! than leaving a lock poisoned for someone to observe.

use core::cell::UnsafeCell;
use core::fmt;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicUsize, Ordering};

/// Spin until `f` reports success.
///
/// `spin_loop` is the wasm `pause` equivalent; it is a hint to the CPU, not a yield, and there is
/// deliberately no yield here — the whole point is that this thread must not stop running.
#[inline]
fn spin_until<T>(mut f: impl FnMut() -> Option<T>) -> T {
    loop {
        if let Some(value) = f() {
            return value;
        }
        core::hint::spin_loop();
    }
}

/// The error a poisoned lock would return. Never constructed here — see the module header.
pub struct PoisonError<T> {
    guard: T,
}

impl<T> PoisonError<T> {
    /// The guard, which is always valid because nothing is ever poisoned.
    pub fn into_inner(self) -> T {
        self.guard
    }
}

impl<T> fmt::Debug for PoisonError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PoisonError")
    }
}

/// The error a `try_*` returns when it would have had to wait.
pub enum TryLockError<T> {
    /// Another thread holds the lock right now.
    WouldBlock,
    /// Never produced — see the module header.
    #[expect(dead_code, reason = "kept so upstream's match arms still compile")]
    Poisoned(PoisonError<T>),
}

/// `UNLOCKED`, or the number of readers, or [`WRITER`].
const UNLOCKED: usize = 0;
/// Sentinel for "one writer holds this", distinct from any reader count.
const WRITER: usize = usize::MAX;

/// A spinning mutex with `std::sync::Mutex`'s surface.
pub struct Mutex<T> {
    state: AtomicUsize,
    value: UnsafeCell<T>,
}

// SAFETY: the `AtomicUsize` is what serialises access to the `UnsafeCell`; only the thread that
// moved `state` from UNLOCKED to WRITER may touch the value, and it restores UNLOCKED on drop.
unsafe impl<T: Send> Send for Mutex<T> {}
// SAFETY: as above — shared references hand out `MutexGuard`s, which are exclusive.
unsafe impl<T: Send> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    /// Create a new mutex. `const`, because upstream's `State::new` is a `const fn`.
    pub const fn new(value: T) -> Self {
        Self {
            state: AtomicUsize::new(UNLOCKED),
            value: UnsafeCell::new(value),
        }
    }

    /// Lock, spinning until it is free. Never blocks.
    pub fn lock(&self) -> Result<MutexGuard<'_, T>, PoisonError<MutexGuard<'_, T>>> {
        Ok(spin_until(|| self.raw_try_lock()))
    }

    /// Lock if it is free right now.
    pub fn try_lock(&self) -> Result<MutexGuard<'_, T>, TryLockError<MutexGuard<'_, T>>> {
        self.raw_try_lock().ok_or(TryLockError::WouldBlock)
    }

    fn raw_try_lock(&self) -> Option<MutexGuard<'_, T>> {
        self.state
            .compare_exchange_weak(UNLOCKED, WRITER, Ordering::Acquire, Ordering::Relaxed)
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
        // SAFETY: holding this guard means `state == WRITER`, so no one else may touch the value.
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
        self.lock.state.store(UNLOCKED, Ordering::Release);
    }
}

/// A spinning reader-writer lock with `std::sync::RwLock`'s surface.
///
/// `state` is [`UNLOCKED`], a reader count, or [`WRITER`]. Writers are not queued and can in
/// principle be starved by a stream of readers; upstream takes the write lock only when a runner
/// registers or unregisters, which happens once per worker per scope, so there is nothing here for
/// starvation to bite.
pub struct RwLock<T> {
    state: AtomicUsize,
    value: UnsafeCell<T>,
}

// SAFETY: same argument as `Mutex`, with the reader count admitting shared references only.
unsafe impl<T: Send> Send for RwLock<T> {}
// SAFETY: `&T` is handed out only while `state` is a reader count, `&mut T` only under WRITER.
unsafe impl<T: Send + Sync> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    /// Create a new lock. `const`, for the same reason as [`Mutex::new`].
    pub const fn new(value: T) -> Self {
        Self {
            state: AtomicUsize::new(UNLOCKED),
            value: UnsafeCell::new(value),
        }
    }

    /// Take a read lock, spinning until no writer holds it.
    ///
    /// Unused today — upstream reaches `local_queues` through [`RwLock::try_read`] and
    /// [`RwLock::write`] only. Kept so this stays a faithful stand-in for `std::sync::RwLock`
    /// rather than a shim shaped around one caller: the day an upgrade starts calling `read`, the
    /// swap should keep working instead of failing to compile and inviting a hurried reimpl.
    #[expect(
        dead_code,
        reason = "completes the std::sync::RwLock surface; see the doc comment"
    )]
    pub fn read(&self) -> Result<RwLockReadGuard<'_, T>, PoisonError<RwLockReadGuard<'_, T>>> {
        Ok(spin_until(|| self.raw_try_read()))
    }

    /// Take a read lock if no writer holds it right now.
    pub fn try_read(&self) -> Result<RwLockReadGuard<'_, T>, TryLockError<RwLockReadGuard<'_, T>>> {
        self.raw_try_read().ok_or(TryLockError::WouldBlock)
    }

    /// Take the write lock, spinning until the lock is completely free.
    pub fn write(&self) -> Result<RwLockWriteGuard<'_, T>, PoisonError<RwLockWriteGuard<'_, T>>> {
        Ok(spin_until(|| {
            self.state
                .compare_exchange_weak(UNLOCKED, WRITER, Ordering::Acquire, Ordering::Relaxed)
                .ok()
                .map(|_| RwLockWriteGuard { lock: self })
        }))
    }

    fn raw_try_read(&self) -> Option<RwLockReadGuard<'_, T>> {
        let current = self.state.load(Ordering::Relaxed);
        // A writer holds it, or the count is one short of the writer sentinel (which cannot happen
        // with a sane number of threads, but must not be allowed to alias WRITER if it did).
        if current == WRITER || current == WRITER - 1 {
            return None;
        }
        self.state
            .compare_exchange_weak(current, current + 1, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| RwLockReadGuard { lock: self })
    }
}

/// Shared access to an [`RwLock`]'s value.
pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockReadGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: holding this guard means `state` is a reader count, so no writer can be present.
        unsafe { &*self.lock.value.get() }
    }
}

impl<T> Drop for RwLockReadGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.state.fetch_sub(1, Ordering::Release);
    }
}

/// Exclusive access to an [`RwLock`]'s value.
pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: holding this guard means `state == WRITER`.
        unsafe { &*self.lock.value.get() }
    }
}

impl<T> DerefMut for RwLockWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as above, and `&mut self` makes this the only guard reference.
        unsafe { &mut *self.lock.value.get() }
    }
}

impl<T> Drop for RwLockWriteGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.state.store(UNLOCKED, Ordering::Release);
    }
}
