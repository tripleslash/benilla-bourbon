//! Where the GPU pipeline cache lives: `<state folder>/shader-cache/pipelines-<key>.bin`, one file
//! named by the adapter (backend, vendor and device id, a digest of the driver's name and version),
//! so a new GPU or driver starts an empty one. `bevy_render`'s fork reads it at device creation and
//! writes it back as new pipelines settle and at exit (`third_party/bevy_render/BENILLA.md`, the
//! twelfth delta); only a Vulkan device has the cache, the browser keeps its own, and with no state
//! folder ([`crate::local_state::home`] is `None`) there is none.

/// Name the cache file for `bevy_render`, before the render plugin finishes.
pub(crate) fn install() {
    #[cfg(not(target_arch = "wasm32"))]
    bevy::render::pipeline_disk_cache::set_cache_file(native::file);
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::{Path, PathBuf};

    /// The folder under the state folder that holds the cache.
    const DIR: &str = "shader-cache";
    /// Every cache file's name starts so; anything else in the folder is not ours to clear.
    const PREFIX: &str = "pipelines-";

    /// `<state folder>/shader-cache/pipelines-<key>.bin` for `adapter`, its folder cleared of the
    /// files other adapters or drivers left, so it holds one cache.
    pub(super) fn file(adapter: &wgpu::AdapterInfo) -> Option<PathBuf> {
        let dir = crate::local_state::home()?.join(DIR);
        let name = file_name(
            adapter.backend.to_str(),
            adapter.vendor,
            adapter.device,
            &adapter.driver,
            &adapter.driver_info,
        );
        clear_others(&dir, &name);
        Some(dir.join(name))
    }

    /// `pipelines-<backend>-<vendor>-<device>-<driver digest>.bin`.
    fn file_name(
        backend: &str,
        vendor: u32,
        device: u32,
        driver: &str,
        driver_info: &str,
    ) -> String {
        let driver = fnv1a(&[driver.as_bytes(), &[0], driver_info.as_bytes()]);
        format!("{PREFIX}{backend}-{vendor:04x}-{device:04x}-{driver:016x}.bin")
    }

    /// FNV-1a over `parts`, a digest that stays the same across builds and toolchains.
    fn fnv1a(parts: &[&[u8]]) -> u64 {
        parts
            .iter()
            .flat_map(|part| part.iter())
            .fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    }

    /// Remove every `pipelines-*` file in `dir` but `keep`: a cache another adapter or driver made,
    /// or a temporary file a killed run left.
    fn clear_others(dir: &Path, keep: &str) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if name.starts_with(PREFIX) && name != keep {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The name carries the backend and both ids, and a driver update changes it.
        #[test]
        fn named_by_adapter_and_driver() {
            let name = file_name("vulkan", 0x10de, 0x2684, "NVIDIA", "566.36");
            assert!(
                name.starts_with("pipelines-vulkan-10de-2684-") && name.ends_with(".bin"),
                "{name}"
            );
            assert_ne!(name, file_name("vulkan", 0x10de, 0x2684, "NVIDIA", "572.16"));
            assert_ne!(name, file_name("vulkan", 0x10de, 0x2684, "NVIDIA5", "66.36"));
        }
    }
}
