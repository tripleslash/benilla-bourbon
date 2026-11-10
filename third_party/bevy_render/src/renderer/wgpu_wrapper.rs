/// A macro producing wrappers to safely make `wgpu` types Send / Sync on web with atomics enabled.
///
/// On web with `atomics` enabled the inner value can only be accessed
/// or dropped on the `wgpu` thread or else a panic will occur.
/// On other platforms the wrapper simply contains the wrapped value.
#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
macro_rules! wgpu_wrapper {
    // **benilla fork (BENILLA.md, "The render worker").** `@shared` marks plain `Send + Sync` data
    // the page reads off the render worker's thread; off `wasm32 + atomics` it is the plain arm.
    (@shared $($rest:tt)*) => {
        $crate::renderer::wgpu_wrapper! { $($rest)* }
    };
    ($( $(#[$($attrs:tt)*])* $vis:vis struct $name:ident ($wgputy:ty) );+ $(;)?) => {
        $(
            $( #[$($attrs)*] )*
            $vis struct $name ($wgputy);

            impl $name {
                /// Constructs a new instance of `WgpuWrapper` which will wrap the specified value.
                pub fn new(t: $wgputy) -> Self {
                    Self(t)
                }

                #[allow(clippy::allow_attributes, unused, reason = "This is not used on all wrappers.")]
                /// Unwraps the value.
                pub fn into_inner(self) -> $wgputy {
                    self.0
                }
            }

            impl ::core::ops::Deref for $name {
                type Target = $wgputy;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl ::core::ops::DerefMut for $name {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.0
                }
            }

            // Short-circuit the `Send + Sync` implementation.
            // At the type-level this does effectively nothing, however in the compiler
            // this creates a short-circuit for the trait solver that reduces recursion depth
            // and substantially improves compile times.
            const _: () = {
                const fn assert_sync_send<T: Sync + Send>() {}
                assert_sync_send::<$wgputy>()
            };
            // SAFETY: We just asserted that $wgputy is Send and Sync
            #[expect(unsafe_code, reason = "Blanket-impl Send requires unsafe.")]
            unsafe impl Send for $name {}
            // SAFETY: We just asserted that $wgputy is Send and Sync
            #[expect(unsafe_code, reason = "Blanket-impl Send requires unsafe.")]
            unsafe impl Sync for $name {}
        )+
    };
}

// **benilla fork (BENILLA.md, "The render worker").** On this target the value sits in a
// `ManuallyDrop<Held>` so `Drop` can choose where it is dropped: a wrapper whose last owner lets go
// of it on another thread is buried and dropped on its own thread, rather than panicking there.
// Access off-thread still panics, naming the type: a dereference is a real bug, a drop is only
// bookkeeping. `@shared` wraps a value that is really `Send + Sync`, held as it is and read from any
// thread.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
macro_rules! wgpu_wrapper {
    (@shared $( $(#[$($attrs:tt)*])* $vis:vis struct $name:ident ($wgputy:ty) );+ $(;)?) => {
        $(
            $( #[$($attrs)*] )*
            $vis struct $name (::core::mem::ManuallyDrop<$crate::renderer::Held<$wgputy>>);

            impl $name {
                /// Constructs a new instance of `WgpuWrapper` which will wrap the specified value.
                pub fn new(t: $wgputy) -> Self {
                    Self(::core::mem::ManuallyDrop::new($crate::renderer::Held::shared(t)))
                }
            }

            $crate::renderer::wgpu_wrapper!(@common $name ($wgputy));
        )+
    };
    (@common $name:ident ($wgputy:ty)) => {
        impl $name {
            #[allow(clippy::allow_attributes, unused, reason = "This is not used on all wrappers.")]
            /// Unwraps the value.
            pub fn into_inner(self) -> $wgputy {
                let mut this = ::core::mem::ManuallyDrop::new(self);
                // SAFETY: `this` is never dropped, so the value is taken out exactly once.
                #[expect(unsafe_code, reason = "Moving out of a type with a Drop impl.")]
                let held = unsafe { ::core::mem::ManuallyDrop::take(&mut this.0) };
                held.into_inner()
            }

            #[allow(clippy::allow_attributes, unused, reason = "This is not used on all wrappers.")]
            /// **benilla fork.** Whether this holds a thread-bound value made on another thread.
            pub(crate) fn owned_elsewhere(&self) -> bool {
                self.0.owned_elsewhere()
            }
        }

        impl ::core::ops::Deref for $name {
            type Target = $wgputy;

            #[track_caller]
            fn deref(&self) -> &Self::Target {
                self.0.get()
            }
        }

        impl ::core::ops::DerefMut for $name {
            #[track_caller]
            fn deref_mut(&mut self) -> &mut Self::Target {
                self.0.get_mut()
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                // SAFETY: `drop` runs once, and `self.0` is not used after this.
                #[expect(unsafe_code, reason = "Taking the value out of ManuallyDrop in Drop.")]
                let held = unsafe { ::core::mem::ManuallyDrop::take(&mut self.0) };
                held.release();
            }
        }
    };
    ($( $(#[$($attrs:tt)*])* $vis:vis struct $name:ident ($wgputy:ty) );+ $(;)?) => {
        $(
            $( #[$($attrs)*] )*
            $vis struct $name (::core::mem::ManuallyDrop<$crate::renderer::Held<$wgputy>>);

            impl $name {
                /// Constructs a new instance of `WgpuWrapper` which will wrap the specified value.
                pub fn new(t: $wgputy) -> Self {
                    Self(::core::mem::ManuallyDrop::new($crate::renderer::Held::local(t)))
                }
            }

            $crate::renderer::wgpu_wrapper!(@common $name ($wgputy));
        )+
    };
}

/// **benilla fork.** How a `wgpu_wrapper!` type holds its value on `wasm32 + atomics`.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[derive(Debug, Clone)]
pub enum Held<T> {
    /// A JS-backed handle: one thread only.
    Local(send_wrapper::SendWrapper<T>),
    /// Plain data, made by `wgpu_wrapper!(@shared ..)`.
    Shared(T),
}

// SAFETY: SendWrapper is always Send + Sync, and `Held::Shared` is only built from a `Send + Sync`
// value (`shared`).
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[expect(unsafe_code, reason = "Blanket-impl Send requires unsafe.")]
unsafe impl<T> Send for Held<T> {}
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[expect(unsafe_code, reason = "Blanket-impl Sync requires unsafe.")]
unsafe impl<T> Sync for Held<T> {}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
impl<T> Held<T> {
    /// A thread-bound value.
    pub fn local(t: T) -> Self {
        Self::Local(send_wrapper::SendWrapper::new(t))
    }

    /// A value that is genuinely `Send + Sync`, read from any thread.
    pub fn shared(t: T) -> Self
    where
        T: Send + Sync,
    {
        Self::Shared(t)
    }

    pub(crate) fn owned_elsewhere(&self) -> bool {
        matches!(self, Held::Local(inner) if !inner.valid())
    }

    /// The value; panics off the thread that made a thread-bound one.
    #[track_caller]
    pub fn get(&self) -> &T {
        match self {
            Held::Local(inner) => {
                if !inner.valid() {
                    used_off_thread::<T>();
                }
                inner
            }
            Held::Shared(inner) => inner,
        }
    }

    /// The value, mutably; panics off the thread that made a thread-bound one.
    #[track_caller]
    pub fn get_mut(&mut self) -> &mut T {
        match self {
            Held::Local(inner) => {
                if !inner.valid() {
                    used_off_thread::<T>();
                }
                inner
            }
            Held::Shared(inner) => inner,
        }
    }

    /// Unwraps the value.
    pub fn into_inner(self) -> T {
        match self {
            Held::Local(inner) => inner.take(),
            Held::Shared(inner) => inner,
        }
    }

    /// Drops the value here, or buries it for its own thread when it was made on another.
    pub fn release(self) {
        let inner = match self {
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
            // SAFETY: every `T` wrapped is a wgpu handle or render state that borrows nothing
            // (`'static`); the generic method cannot say so without bounding every wrapper.
            #[expect(
                unsafe_code,
                reason = "Erasing a lifetime the wrapped values do not have."
            )]
            let buried: Box<dyn crate::render_worker::Buried> =
                unsafe { core::mem::transmute(buried) };
            crate::render_worker::bury(buried);
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

pub(crate) use wgpu_wrapper;
