# benilla's `dlss_wgpu` fork: what differs from upstream, and how to check

Upstream: [`bevyengine/dlss_wgpu`](https://github.com/bevyengine/dlss_wgpu) v4.0.0 (commit
`323ba14`, the release on wgpu 29, DLSS SDK 310.5.3), MIT OR Apache-2.0 (`LICENSE-MIT`,
`LICENSE-APACHE`). Used directly by `benilla-world/src/dlss/` (not through Bevy's `DlssPlugin`,
which owns the camera, jitter and prepasses that benilla's FSR slot already owns). Every hunk is
marked `BENILLA:`.

## The SDK beside it

The NVIDIA NGX SDK is vendored at `third_party/nvidia-dlss` (DLSS SDK 310.9.16, proprietary, its own
`BENILLA.md`), with the Vulkan headers bindgen needs at `third_party/vulkan-headers`; `build.rs`
reads both by default, so a clone with libclang builds the real crate. The runtime DLLs are not
linked: the host loads them from the folder it names (`set_runtime_paths`).

## The deltas

1. **A mock build** (`build.rs`, `src/mock.rs`, `src/lib.rs`). Without an NGX library for the
   target, without libclang (bindgen's panic is caught), or off Windows/Linux, `cfg(dlss_mock)`
   builds the same public API with every constructor answering `FeatureNotSupported`, and a
   `cargo:warning` says why; the client falls back to FSR. `NGX_LINKED` says which build this is.
   Upstream panics.
2. **The vendored SDK by default, and the CRT the target links** (`build.rs`): `$DLSS_SDK` defaults
   to `../nvidia-dlss`, `$VULKAN_SDK` to `../vulkan-headers`; `nvsdk_ngx_s` when the target has
   `crt-static` (the desktop build), else `nvsdk_ngx_d`; both SDK layouts, the NVIDIA/DLSS
   repository's `lib/<platform>/x64/` and Streamline's `external/ngx-sdk/lib/<platform>/`.
3. **The D3D12 path** (`src/d3d12.rs`, `src/sdk.rs`, `src/super_resolution.rs`). `DlssSdk::new`
   reads the device's backend: Vulkan as upstream, or DX12 through `NVSDK_NGX_D3D12_*` on
   wgpu-hal's raw handles (`raw_device`, `raw_resource`, and the open command list from
   `third_party/wgpu-hal`'s `CommandEncoder::raw_command_list`, its fifth patch). The D3D12 entry
   points are declared by hand on opaque COM pointers (they sit in `nvsdk_ngx.h`'s `extern "C"`
   block) so bindgen needs no Windows SDK; the two `static inline` helpers
   (`NGX_D3D12_CREATE_DLSS_EXT`, `NGX_D3D12_EVALUATE_DLSS_EXT`) are restated in Rust as the named
   parameter writes they are. Shutdown waits idle on either backend.
4. **Shared public types** (`src/types.rs`): the flags, quality modes, errors and `FeatureSupport`
   without generated types (NGX's flag bits as literals, asserted against the bindings in a real
   build), `DlssBackend`, `DlssError::{UnsupportedBackend, Unknown}`, and
   `DlssPerfQualityMode::for_scale` for a host that fixes its render scale.
5. **DLSS 4.5's presets pinned** (`src/nvsdk_ngx.rs`, `pin_dlss_45_presets`): K for DLAA, Quality and
   Balanced, M for Performance, L for Ultra Performance (the 310.5 SDK's per-mode defaults), so an
   OTA update cannot move them (Programming Guide 3.12.1). Building against an SDK before 310.5
   fails to compile on the missing L/M constants, by design.
6. **A fixed render size** (`DlssSuperResolution::new`'s `fixed_render_resolution`): created at
   exactly the host's render size; `None` creates at the mode's maximum (dynamic resolution) instead
   of upstream's optimal size.
7. **The runtime and data folders** (`src/feature_info.rs`, `set_runtime_paths`) searched before
   `$DLSS_SDK`'s library folder; NGX's data (its logs) in the host's folder instead of the system
   temp directory.
8. **Upstream fixes after 4.0.0, backported**: `f7c75dc` (the enum-prefixed result names; the bare
   ones were bindings that matched every result) and `943e15d`'s shape (the Vulkan evaluate's
   resources held in named bindings until the call returns).
9. **Ray Reconstruction** stays Vulkan-only, as upstream, and answers `NotImplemented` on D3D12; it
   is the hook `benilla-world/src/dlss/HOOKS.md` describes. `DlssSdk::ray_reconstruction_available`
   reads NGX's `SuperSamplingDenoising.Available`. The OTA update check runs only once NGX reports
   DLSS supported.

## How to check

- Without libclang: `cargo check -p dlss_wgpu` prints the mock warning and builds.
- With LLVM installed (or `LIBCLANG_PATH` set): `cargo check -p dlss_wgpu` builds the bindings from
  the vendored SDK, no warning. `DLSS_SDK` / `VULKAN_SDK` point it elsewhere.
- Diff against upstream: `git -C <dlss_wgpu clone> show 323ba14:src/super_resolution.rs` beside
  `src/super_resolution.rs`; every difference carries `BENILLA:`.

## When to drop it

When upstream dlss_wgpu supports D3D12 on the wgpu version benilla pins, and builds without the SDK.
