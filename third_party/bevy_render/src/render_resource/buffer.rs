use crate::renderer::WgpuWrapper;
use bevy_utils::define_atomic_id;
use core::ops::{Deref, RangeBounds};

define_atomic_id!(BufferId);

#[derive(Clone, Debug)]
pub struct Buffer {
    id: BufferId,
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
    value: WgpuWrapper<wgpu::Buffer>,
    // **benilla fork (BENILLA.md, "The render worker").** Shared, so a clone is an `Arc` clone that
    // never touches wgpu (cloning a `SendWrapper` dereferences it, which panics on the page when the
    // device lives on the render worker), and lazy, so the page can create one. The cell derefs to
    // the `wgpu::Buffer`, so every use below reads as upstream's.
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    value: alloc::sync::Arc<lazy::BufferCell>,
}

impl Buffer {
    #[inline]
    pub fn id(&self) -> BufferId {
        self.id
    }

    pub fn slice(&self, bounds: impl RangeBounds<wgpu::BufferAddress>) -> BufferSlice<'_> {
        BufferSlice {
            id: self.id,
            value: self.value.slice(bounds),
        }
    }

    #[inline]
    pub fn unmap(&self) {
        self.value.unmap();
    }
}

impl From<wgpu::Buffer> for Buffer {
    fn from(value: wgpu::Buffer) -> Self {
        Buffer {
            id: BufferId::new(),
            #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
            value: WgpuWrapper::new(value),
            #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
            value: alloc::sync::Arc::new(lazy::BufferCell::ready(WgpuWrapper::new(value))),
        }
    }
}

impl Deref for Buffer {
    type Target = wgpu::Buffer;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// **benilla fork (BENILLA.md, "The render worker").** A buffer the page asks for while the device
/// lives on the render worker: described now, created on the worker the first time it is used
/// there.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
impl Buffer {
    pub(crate) fn pending(desc: &wgpu::BufferDescriptor, contents: Option<&[u8]>) -> Self {
        assert!(
            !desc.mapped_at_creation,
            "a buffer mapped at creation cannot be made off the render worker"
        );
        Buffer {
            id: BufferId::new(),
            value: alloc::sync::Arc::new(lazy::BufferCell::pending(desc, contents)),
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
mod lazy {
    use crate::renderer::WgpuWrapper;
    use alloc::{string::String, vec::Vec};
    use std::sync::OnceLock;

    /// A `wgpu::Buffer`, made now or on first use on the render worker.
    pub struct BufferCell {
        value: OnceLock<WgpuWrapper<wgpu::Buffer>>,
        pending: Option<Pending>,
    }

    struct Pending {
        label: Option<String>,
        size: wgpu::BufferAddress,
        usage: wgpu::BufferUsages,
        contents: Option<Vec<u8>>,
    }

    impl core::fmt::Debug for BufferCell {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("BufferCell")
                .field("created", &self.value.get().is_some())
                .finish()
        }
    }

    impl BufferCell {
        pub fn ready(value: WgpuWrapper<wgpu::Buffer>) -> Self {
            Self {
                value: OnceLock::from(value),
                pending: None,
            }
        }

        pub fn pending(desc: &wgpu::BufferDescriptor, contents: Option<&[u8]>) -> Self {
            Self {
                value: OnceLock::new(),
                pending: Some(Pending {
                    label: desc.label.map(String::from),
                    size: desc.size,
                    usage: desc.usage,
                    contents: contents.map(<[u8]>::to_vec),
                }),
            }
        }
    }

    impl core::ops::Deref for BufferCell {
        type Target = wgpu::Buffer;

        fn deref(&self) -> &wgpu::Buffer {
            if let Some(value) = self.value.get() {
                return value;
            }
            assert!(
                crate::render_worker::is_render_thread(),
                "a page-made buffer was used on the page; it exists only on the render worker"
            );
            self.value.get_or_init(|| {
                let pending = self
                    .pending
                    .as_ref()
                    .expect("a buffer cell is either created or pending");
                WgpuWrapper::new(crate::render_worker::create_buffer(
                    &wgpu::BufferDescriptor {
                        label: pending.label.as_deref(),
                        size: pending.size,
                        usage: pending.usage,
                        mapped_at_creation: false,
                    },
                    pending.contents.as_deref(),
                ))
            })
        }
    }
}

#[derive(Clone, Debug)]
pub struct BufferSlice<'a> {
    id: BufferId,
    value: wgpu::BufferSlice<'a>,
}

impl<'a> BufferSlice<'a> {
    #[inline]
    pub fn id(&self) -> BufferId {
        self.id
    }
}

impl<'a> Deref for BufferSlice<'a> {
    type Target = wgpu::BufferSlice<'a>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
