//! **benilla fork.** A persistent wgpu pipeline cache for native builds.
//!
//! When the device has `Features::PIPELINE_CACHE` (wgpu offers it on Vulkan alone) and the app has
//! named a file for the adapter ([`set_cache_file`]), one `wgpu::PipelineCache` is created from
//! that file's bytes and every render and compute pipeline the
//! [`PipelineCache`](crate::render_resource::PipelineCache) creates is given it, so the driver
//! reuses the code a previous run compiled. Its data goes back to the file once a burst of new
//! pipelines settles (`save_settled`) and at exit (`save_on_exit`, and the last handle's drop),
//! written whole: a temporary file, then a rename over the old one.
//!
//! The browser keeps a shader cache of its own: on wasm32 the handle is empty and nothing here
//! runs.

#[cfg(not(target_arch = "wasm32"))]
pub use native::{set_cache_file, CacheFile};
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::{open, save_on_exit, save_settled, PipelineDiskCache};

/// What [`PipelineCache`](crate::render_resource::PipelineCache) hands each pipeline it creates:
/// the disk cache, when there is one.
#[derive(Clone, Default)]
pub(crate) struct SharedPipelineCache(
    #[cfg(not(target_arch = "wasm32"))] Option<alloc::sync::Arc<native::DiskCache>>,
);

#[cfg(not(target_arch = "wasm32"))]
impl SharedPipelineCache {
    /// The handle the render world's [`PipelineDiskCache`] holds.
    pub(crate) fn new(disk: &PipelineDiskCache) -> Self {
        Self(Some(disk.0.clone()))
    }

    /// The cache for a pipeline descriptor's `cache` field.
    pub(crate) fn get(&self) -> Option<&wgpu::PipelineCache> {
        self.0.as_deref().map(|disk| &disk.cache)
    }

    /// Count one pipeline created with [`Self::get`], so the next save writes its code.
    pub(crate) fn note_built(&self) {
        if let Some(disk) = &self.0 {
            disk.built.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl SharedPipelineCache {
    /// No cache on the page: the browser keeps its own.
    pub(crate) fn get(&self) -> Option<&wgpu::PipelineCache> {
        None
    }

    /// Nothing to count on the page.
    pub(crate) fn note_built(&self) {}
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use crate::renderer::RenderDevice;
    use alloc::sync::Arc;
    use bevy_app::AppExit;
    use bevy_ecs::{
        message::MessageReader,
        resource::Resource,
        system::{Local, Res},
    };
    use bevy_log::{debug, info, warn};
    use bevy_platform::time::Instant;
    use bevy_tasks::IoTaskPool;
    use core::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };
    use std::{
        path::{Path, PathBuf},
        sync::{Mutex, OnceLock, PoisonError},
    };

    /// The app's file for an adapter's pipeline cache; `None` keeps the cache off.
    pub type CacheFile = fn(&wgpu::AdapterInfo) -> Option<PathBuf>;

    static CACHE_FILE: OnceLock<CacheFile> = OnceLock::new();

    /// Name the file the pipeline cache lives in, per adapter, so a GPU or driver change can start
    /// a fresh one. Call before the render plugin finishes (before the app runs); the first call
    /// wins. Without a call there is no cache.
    pub fn set_cache_file(file: CacheFile) {
        let _ = CACHE_FILE.set(file);
    }

    /// The wgpu cache, its file, and how much of it is on disk.
    pub(crate) struct DiskCache {
        pub(super) cache: wgpu::PipelineCache,
        path: PathBuf,
        /// Pipelines created with `cache` so far.
        pub(super) built: AtomicUsize,
        /// `built` as of the last save; held while writing, so writers take turns.
        saved: Mutex<usize>,
    }

    impl DiskCache {
        /// Write the cache's data to its file if pipelines were created since the last save.
        fn save(&self) {
            let built = self.built.load(Ordering::Relaxed);
            let mut saved = self.saved.lock().unwrap_or_else(PoisonError::into_inner);
            if *saved == built {
                return;
            }
            *saved = built;
            let Some(data) = self.cache.get_data() else {
                return;
            };
            match write_whole(&self.path, &data) {
                Ok(()) => debug!(
                    "pipeline cache: {} KiB saved to {}",
                    data.len() / 1024,
                    self.path.display()
                ),
                Err(e) => warn!(
                    "pipeline cache: could not save {}: {e}",
                    self.path.display()
                ),
            }
        }
    }

    impl Drop for DiskCache {
        /// The last handle going (the worlds torn down at exit) saves what is unsaved.
        fn drop(&mut self) {
            self.save();
        }
    }

    /// `data` into `path` by a temporary file beside it and a rename, so a reader never meets a
    /// half-written file; the temporary name carries the process id, so two clients never share it.
    fn write_whole(path: &Path, data: &[u8]) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let temp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&temp, data)?;
        std::fs::rename(&temp, path).inspect_err(|_| {
            let _ = std::fs::remove_file(&temp);
        })
    }

    /// The disk cache, in the render world (for [`save_settled`]) and the main world (for
    /// [`save_on_exit`]).
    #[derive(Resource, Clone)]
    pub(crate) struct PipelineDiskCache(pub(super) Arc<DiskCache>);

    /// The disk cache for `device`, from the file the app names for `adapter_info`, or `None` when
    /// the device has no `PIPELINE_CACHE` or the app names no file.
    pub(crate) fn open(
        device: &RenderDevice,
        adapter_info: &wgpu::AdapterInfo,
    ) -> Option<PipelineDiskCache> {
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return None;
        }
        let file = CACHE_FILE.get()?;
        let path = file(adapter_info)?;
        let data = match std::fs::read(&path) {
            Ok(data) => Some(data),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                warn!("pipeline cache: could not read {}: {e}", path.display());
                None
            }
        };
        // SAFETY: the file holds only `get_data` output, written whole, for an adapter with this
        // file's key (the app's `CacheFile` names it by adapter and driver); `fallback: true`
        // makes wgpu start empty on data its header check or the driver rejects.
        let cache = unsafe {
            device
                .wgpu_device()
                .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("pipeline_disk_cache"),
                    data: data.as_deref(),
                    fallback: true,
                })
        };
        info!(
            "pipeline cache: {} ({} KiB read)",
            path.display(),
            data.as_ref().map_or(0, |d| d.len() / 1024)
        );
        Some(PipelineDiskCache(Arc::new(DiskCache {
            cache,
            path,
            built: AtomicUsize::new(0),
            saved: Mutex::new(0),
        })))
    }

    /// How long the count of created pipelines holds still before a save, so a load burst ends in
    /// one write.
    const SETTLE: Duration = Duration::from_secs(3);
    /// The longest a created pipeline goes unsaved while pipelines keep coming.
    const MAX_UNSAVED: Duration = Duration::from_secs(30);

    /// [`save_settled`]'s view of the count.
    #[derive(Default)]
    pub(crate) struct SaveClock {
        /// The count last seen.
        seen: usize,
        /// When it last changed.
        changed: Option<Instant>,
        /// When the oldest unsaved pipeline was first seen.
        unsaved_since: Option<Instant>,
    }

    /// Save the cache, off the render thread, once new pipelines have settled for [`SETTLE`] or
    /// have waited [`MAX_UNSAVED`].
    pub(crate) fn save_settled(disk: Option<Res<PipelineDiskCache>>, mut clock: Local<SaveClock>) {
        let Some(disk) = disk else {
            return;
        };
        let built = disk.0.built.load(Ordering::Relaxed);
        let now = Instant::now();
        if built != clock.seen {
            clock.seen = built;
            clock.changed = Some(now);
            clock.unsaved_since.get_or_insert(now);
        }
        let Some(unsaved_since) = clock.unsaved_since else {
            return;
        };
        let settled = clock
            .changed
            .is_some_and(|changed| now.duration_since(changed) >= SETTLE);
        if !settled && now.duration_since(unsaved_since) < MAX_UNSAVED {
            return;
        }
        clock.unsaved_since = None;
        let disk = disk.0.clone();
        match IoTaskPool::try_get() {
            Some(pool) => pool.spawn(async move { disk.save() }).detach(),
            None => disk.save(),
        }
    }

    /// Save the cache when the app is asked to exit.
    pub(crate) fn save_on_exit(
        disk: Option<Res<PipelineDiskCache>>,
        mut exits: MessageReader<AppExit>,
    ) {
        if exits.read().count() == 0 {
            return;
        }
        if let Some(disk) = disk {
            disk.0.save();
        }
    }
}
