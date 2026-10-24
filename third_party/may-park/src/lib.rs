//! **benilla.** Which threads of the threaded browser build may park, and `std::sync` locks that ask.
//!
//! On `wasm32 + atomics` a contended `std::sync` lock parks with `memory.atomic.wait32`. A Web
//! Worker may execute it; the browser page's thread may not: there it throws `Atomics.wait cannot be
//! called in this context` and the module traps. The page and the workers share locks (the smol
//! executor and its channels, bevy's `AssetServer`, the render worker's handshake), so every such
//! lock must not park on the page, and should park everywhere else: a worker that spins instead
//! burns a core for as long as the holder runs.
//!
//! The page says which thread it is, once, with [`forbid_parking_on_this_thread`], before it shares
//! a lock with a worker (`benilla_app::web_threads::install`, before any pool exists). Every thread
//! that does not say so may park. [`sync::Mutex`] and [`sync::RwLock`] are std's own, and their
//! guards are std's own (so a `Condvar` waits on them as usual); only their blocking calls ask
//! [`may_park`] first and loop on `try_*` on the page.
//!
//! Users: the `third_party` forks of `async-executor`, `event-listener`, `parking`, `bevy_platform`
//! and `bevy_tasks` (each one's BENILLA.md). Off `wasm32 + atomics` there is nothing to ask:
//! [`may_park`] is `true` and [`sync`] is `std::sync`.

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
std::thread_local! {
    static MAY_PARK: core::cell::Cell<bool> = const { core::cell::Cell::new(true) };
}

/// Mark the calling thread as one that may not park: the browser page's thread. From here on the
/// locks in [`sync`] spin on it rather than park. A no-op off `wasm32 + atomics`.
pub fn forbid_parking_on_this_thread() {
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    MAY_PARK.with(|p| p.set(false));
}

/// Whether the calling thread may park on a contended lock or a `block_on`.
#[inline]
pub fn may_park() -> bool {
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    return MAY_PARK.with(core::cell::Cell::get);
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
    true
}

/// `std::sync`'s `Mutex` and `RwLock`, asking [`may_park`] before they park.
#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
pub mod sync {
    pub use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
}

/// `std::sync`'s `Mutex` and `RwLock`, asking [`may_park`] before they park.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub mod sync {
    use core::ops::{Deref, DerefMut};
    use std::sync::{LockResult, TryLockError, TryLockResult};

    pub use std::sync::{MutexGuard, RwLockReadGuard, RwLockWriteGuard};

    /// A `try_*` result as its blocking call's, spinning while it would block.
    #[inline]
    fn spin<G>(mut attempt: impl FnMut() -> TryLockResult<G>) -> LockResult<G> {
        loop {
            match attempt() {
                Ok(guard) => return Ok(guard),
                Err(TryLockError::Poisoned(poisoned)) => return Err(poisoned),
                Err(TryLockError::WouldBlock) => core::hint::spin_loop(),
            }
        }
    }

    /// `std::sync::Mutex`, whose [`Mutex::lock`] spins where [`super::may_park`] is false. Every
    /// other method is std's, through `Deref`.
    #[derive(Debug, Default)]
    pub struct Mutex<T: ?Sized>(std::sync::Mutex<T>);

    impl<T> Mutex<T> {
        /// See [`std::sync::Mutex::new`].
        pub const fn new(t: T) -> Self {
            Self(std::sync::Mutex::new(t))
        }

        /// See [`std::sync::Mutex::into_inner`].
        pub fn into_inner(self) -> LockResult<T> {
            self.0.into_inner()
        }
    }

    impl<T: ?Sized> Mutex<T> {
        /// See [`std::sync::Mutex::lock`].
        #[inline]
        pub fn lock(&self) -> LockResult<MutexGuard<'_, T>> {
            if super::may_park() {
                self.0.lock()
            } else {
                spin(|| self.0.try_lock())
            }
        }
    }

    impl<T> From<T> for Mutex<T> {
        fn from(t: T) -> Self {
            Self::new(t)
        }
    }

    impl<T: ?Sized> Deref for Mutex<T> {
        type Target = std::sync::Mutex<T>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<T: ?Sized> DerefMut for Mutex<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    /// `std::sync::RwLock`, whose [`RwLock::read`] and [`RwLock::write`] spin where
    /// [`super::may_park`] is false. Every other method is std's, through `Deref`.
    #[derive(Debug, Default)]
    pub struct RwLock<T: ?Sized>(std::sync::RwLock<T>);

    impl<T> RwLock<T> {
        /// See [`std::sync::RwLock::new`].
        pub const fn new(t: T) -> Self {
            Self(std::sync::RwLock::new(t))
        }

        /// See [`std::sync::RwLock::into_inner`].
        pub fn into_inner(self) -> LockResult<T> {
            self.0.into_inner()
        }
    }

    impl<T: ?Sized> RwLock<T> {
        /// See [`std::sync::RwLock::read`].
        #[inline]
        pub fn read(&self) -> LockResult<RwLockReadGuard<'_, T>> {
            if super::may_park() {
                self.0.read()
            } else {
                spin(|| self.0.try_read())
            }
        }

        /// See [`std::sync::RwLock::write`].
        #[inline]
        pub fn write(&self) -> LockResult<RwLockWriteGuard<'_, T>> {
            if super::may_park() {
                self.0.write()
            } else {
                spin(|| self.0.try_write())
            }
        }
    }

    impl<T> From<T> for RwLock<T> {
        fn from(t: T) -> Self {
            Self::new(t)
        }
    }

    impl<T: ?Sized> Deref for RwLock<T> {
        type Target = std::sync::RwLock<T>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<T: ?Sized> DerefMut for RwLock<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }
}
