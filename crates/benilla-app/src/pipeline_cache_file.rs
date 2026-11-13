//! Where the GPU pipeline cache lives: `<state folder>/shader-cache/pipelines-<key>.bin`, one file
//! named by the adapter (backend, vendor and device id, a digest of the driver's name and version)
//! and by the build (a digest of its version, sha, describe and profile), so a new GPU, driver or
//! build starts an empty one: the first launch after an update clears the last build's cache,
//! whose pipelines were compiled from other shaders and other layouts. `bevy_render`'s fork reads it at device creation and
//! writes it back as new pipelines settle and at exit (`third_party/bevy_render/BENILLA.md`, the
//! twelfth delta); only a Vulkan device has the cache, the browser keeps its own, and with no state
//! folder ([`crate::local_state::home`] is `None`) there is none.

/// Name the cache file for `bevy_render`, before the render plugin finishes, keyed to `build`.
pub(crate) fn install(build: &crate::BuildId) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        native::set_build(build);
        bevy::render::pipeline_disk_cache::set_cache_file(native::file);
    }
    #[cfg(target_arch = "wasm32")]
    let _ = build;
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::{Path, PathBuf};

    /// The folder under the state folder that holds the cache.
    const DIR: &str = "shader-cache";
    /// Every cache file's name starts so; anything else in the folder is not ours to clear.
    const PREFIX: &str = "pipelines-";

    /// The running build's digest ([`build_digest`]), set by [`super::install`]: `bevy_render`'s
    /// `CacheFile` is a plain `fn`, so the key reaches [`file`] through here.
    static BUILD: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

    pub(super) fn set_build(build: &crate::BuildId) {
        let _ = BUILD.set(build_digest(build));
    }

    /// A digest of what identifies the build: an update always brings another commit, so another
    /// sha, and a cache built by one build is never read by the next.
    fn build_digest(build: &crate::BuildId) -> u64 {
        fnv1a(&[
            build.version.as_bytes(),
            &[0],
            build.sha.as_bytes(),
            &[0],
            build.describe.as_bytes(),
            &[0],
            build.profile.as_bytes(),
        ])
    }

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
            BUILD.get().copied().unwrap_or(0),
        );
        clear_others(&dir, &name);
        Some(dir.join(name))
    }

    /// `pipelines-<backend>-<vendor>-<device>-<driver digest>-<build digest>.bin`.
    fn file_name(
        backend: &str,
        vendor: u32,
        device: u32,
        driver: &str,
        driver_info: &str,
        build: u64,
    ) -> String {
        let driver = fnv1a(&[driver.as_bytes(), &[0], driver_info.as_bytes()]);
        format!("{PREFIX}{backend}-{vendor:04x}-{device:04x}-{driver:016x}-{build:016x}.bin")
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

    /// Remove every `pipelines-*` file in `dir` but `keep`: a cache another adapter, driver or
    /// build made, or a temporary file a killed run left.
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
            let name = file_name("vulkan", 0x10de, 0x2684, "NVIDIA", "566.36", 7);
            assert!(
                name.starts_with("pipelines-vulkan-10de-2684-") && name.ends_with(".bin"),
                "{name}"
            );
            assert_ne!(
                name,
                file_name("vulkan", 0x10de, 0x2684, "NVIDIA", "572.16", 7)
            );
            assert_ne!(
                name,
                file_name("vulkan", 0x10de, 0x2684, "NVIDIA5", "66.36", 7)
            );
        }

        /// An update is another build: the same adapter names another file, and the first launch
        /// of the new build clears the old build's cache, keeping its own and anything not a cache.
        #[test]
        fn a_new_build_names_another_file_and_clears_the_old() {
            let build = |sha: &'static str| crate::BuildId {
                version: "0.2.0",
                sha,
                profile: "ship",
                ..Default::default()
            };
            let (old, new) = (build_digest(&build("aaaa")), build_digest(&build("bbbb")));
            assert_ne!(old, new);
            assert_eq!(old, build_digest(&build("aaaa")), "stable across runs");
            let named = |b| file_name("vulkan", 0x10de, 0x2684, "NVIDIA", "566.36", b);
            assert_ne!(named(old), named(new));

            let dir = std::env::temp_dir().join(format!("benilla-pcf-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            for f in [named(old), named(new), "notes.txt".to_string()] {
                std::fs::write(dir.join(f), b"x").unwrap();
            }
            clear_others(&dir, &named(new));
            assert!(!dir.join(named(old)).exists(), "the old build's cache goes");
            assert!(dir.join(named(new)).exists(), "this build's stays");
            assert!(dir.join("notes.txt").exists(), "nothing else is touched");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
