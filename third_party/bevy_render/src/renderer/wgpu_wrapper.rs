/// A wrapper to safely make `wgpu` types Send / Sync on web with atomics enabled.
///
/// On web with `atomics` enabled the inner value can only be accessed
/// or dropped on the `wgpu` thread or else a panic will occur.
/// On other platforms the wrapper simply contains the wrapped value.
// **benilla fork (BENILLA.md, "The render worker").** On this target the value sits in a
// `ManuallyDrop` so `Drop` below can choose where it is dropped: a wrapper whose last owner lets go
// of it on another thread is buried and dropped on its own thread, rather than panicking there.
// Access off-thread still panics: a dereference is a real bug, a drop is only bookkeeping. A value
// that is really `Send + Sync` (`new_send`) is held as it is and reads from any thread.
#[derive(Debug, Clone)]
pub struct WgpuWrapper<T>(
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))] T,
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    core::mem::ManuallyDrop<Held<T>>,
);

/// How a [`WgpuWrapper`] holds its value on this target.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[derive(Debug, Clone)]
enum Held<T> {
    /// A JS-backed handle: one thread only.
    Local(send_wrapper::SendWrapper<T>),
    /// Plain data, made with [`WgpuWrapper::new_send`].
    Shared(T),
}

// SAFETY: SendWrapper is always Send + Sync, and `Held::Shared` is only built from a `Send + Sync`
// value (`new_send`).
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[expect(unsafe_code, reason = "Blanket-impl Send requires unsafe.")]
unsafe impl<T> Send for WgpuWrapper<T> {}
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[expect(unsafe_code, reason = "Blanket-impl Sync requires unsafe.")]
unsafe impl<T> Sync for WgpuWrapper<T> {}

impl<T> WgpuWrapper<T> {
    /// Constructs a new instance of `WgpuWrapper` which will wrap the specified value.
    pub fn new(t: T) -> Self {
        #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
        return Self(t);
        #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
        return Self(core::mem::ManuallyDrop::new(Held::Local(
            send_wrapper::SendWrapper::new(t),
        )));
    }

    /// **benilla fork.** Wraps a value that is genuinely `Send + Sync` (plain data such as
    /// `wgpu::AdapterInfo`), which then reads from any thread. Elsewhere, `new`.
    pub fn new_send(t: T) -> Self
    where
        T: Send + Sync,
    {
        #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
        return Self(t);
        #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
        return Self(core::mem::ManuallyDrop::new(Held::Shared(t)));
    }

    /// **benilla fork.** Whether this holds a thread-bound value made on another thread.
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    pub(crate) fn owned_elsewhere(&self) -> bool {
        matches!(&*self.0, Held::Local(inner) if !inner.valid())
    }

    /// Unwraps the value.
    pub fn into_inner(self) -> T {
        #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
        return self.0;
        #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
        {
            let mut this = core::mem::ManuallyDrop::new(self);
            // SAFETY: `this` is never dropped, so the value is taken out exactly once.
            #[expect(unsafe_code, reason = "Moving out of a type with a Drop impl.")]
            let held = unsafe { core::mem::ManuallyDrop::take(&mut this.0) };
            match held {
                Held::Local(inner) => inner.take(),
                Held::Shared(inner) => inner,
            }
        }
    }
}

/// A buried wrapper: dropped by `render_worker`'s graveyard on the thread that made it.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
struct Buried<T>(send_wrapper::SendWrapper<T>);

// SAFETY: the value is only moved while buried, never touched; it is dropped by the graveyard
// only where `owned_here` (the SendWrapper's own thread check) says so.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[expect(
    unsafe_code,
    reason = "The graveyard moves the value between threads untouched."
)]
unsafe impl<T> Send for Buried<T> {}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
impl<T> crate::render_worker::Buried for Buried<T> {
    fn owned_here(&self) -> bool {
        self.0.valid()
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
impl<T> Drop for WgpuWrapper<T> {
    fn drop(&mut self) {
        // SAFETY: `drop` runs once, and `self.0` is not used after this.
        #[expect(unsafe_code, reason = "Taking the value out of ManuallyDrop in Drop.")]
        let held = unsafe { core::mem::ManuallyDrop::take(&mut self.0) };
        let inner = match held {
            Held::Shared(inner) => {
                drop(inner);
                return;
            }
            Held::Local(inner) => inner,
        };
        if inner.valid() || !core::mem::needs_drop::<T>() {
            drop(inner);
        } else {
            let buried: Box<dyn crate::render_worker::Buried + '_> = Box::new(Buried(inner));
            // SAFETY: every `T` wrapped is a wgpu handle that borrows nothing (`'static`); the
            // `Drop` impl cannot say so, because it must repeat the struct's bounds.
            #[expect(
                unsafe_code,
                reason = "Erasing a lifetime the wrapped handles do not have."
            )]
            let buried: Box<dyn crate::render_worker::Buried> =
                unsafe { core::mem::transmute(buried) };
            crate::render_worker::bury(buried);
        }
    }
}

/// The panic `SendWrapper` would raise, naming what was touched: on the render worker's build the
/// thread a value was made on is the first question about any such panic.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[cold]
#[track_caller]
fn used_off_thread<T>() -> ! {
    panic!(
        "a `{}` was used off the thread that made it (render worker: page vs worker)",
        core::any::type_name::<T>()
    );
}

impl<T> core::ops::Deref for WgpuWrapper<T> {
    type Target = T;

    #[track_caller]
    fn deref(&self) -> &Self::Target {
        #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
        return &self.0;
        #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
        match &*self.0 {
            Held::Local(inner) => {
                if !inner.valid() {
                    used_off_thread::<T>();
                }
                inner
            }
            Held::Shared(inner) => inner,
        }
    }
}

impl<T> core::ops::DerefMut for WgpuWrapper<T> {
    #[track_caller]
    fn deref_mut(&mut self) -> &mut Self::Target {
        #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
        return &mut self.0;
        #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
        match &mut *self.0 {
            Held::Local(inner) => {
                if !inner.valid() {
                    used_off_thread::<T>();
                }
                inner
            }
            Held::Shared(inner) => inner,
        }
    }
}
