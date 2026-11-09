# DLSS 4.5: hook points in upstream-owned files

DLSS (`FORK.md`) lives in fork-owned files:
- `benilla-world/src/dlss/` (this module)
- `third_party/dlss_wgpu/` (vendored, `BENILLA.md`)
- FSR's own files, which host its slot: `benilla-world/src/fsr/mod.rs`, `fsr/upscaler.rs`

Every edit it needs inside a file that upstream benilla also edits carries the marker
`fork: dlss (FORK.md)` and is listed here. After an upstream merge, check each row is still
present. `dlss::tests::every_hook_row_carries_its_marker` reads this table and fails on a file that
lost its marker. The vendored crates' deltas (`third_party/wgpu-hal`, `third_party/bevy_render`) are
in their own `BENILLA.md`.

## Live hooks

| File | Hook |
|---|---|
| `benilla-world/src/lib.rs` | `pub mod dlss;` |
| `benilla-world/src/world_plugins.rs` | `.add(crate::dlss::DlssPlugin)` after `FsrPlugin` |
| `benilla-world/Cargo.toml` | `dlss_wgpu`, `bevy_render`'s `raw_vulkan_init`, `uuid` (native); `libloading`, `windows-core`, `ash` (Windows) |
| `benilla-world/src/fsr/mod.rs` | `UpscalerKind`; `FsrView::upscaler`, `with_upscaler`; `FsrShaders::motion_only`, `motion_only_source`; `HudlessCompose` in `FsrView`'s sync targets |
| `benilla-world/src/fsr/upscaler.rs` | `Targets`/`UpscaleTargets` split by `UpscalerKind`; `FsrState::dlss`, `jitter`, `is_modern`, `kind`, `motion_vectors`; the `motion_only` pipeline; `prepare_fsr` is `pub(crate)`; `fsr_opaque_copy` skips a DLSS view; `fsr_upscale_pass`'s DLSS branch |
| `benilla-world/src/ffx_glow.rs` | the compose pair also for a view carrying `dlss::HudlessCompose` (DLSS-G's HUD-less world) |
| `benilla-app/src/world_backdrop.rs` | `Slot`, `resolve_upscaler`, `frame_generation_wanted`; `upscaler = off` in `track_render_size`/`retarget_world_camera`; `sync_fsr_view` sets `FsrView::upscaler`, `FgBlocked` under DLSS-G and `DlssFrameGeneration`; the `upscaler` arm of `on_cvar` |
| `benilla-app/src/cvars/table.rs` | the `upscaler` row in its fork-owned block |
| `benilla-app/src/cvars/mod.rs` | `GRAPHICS_CVARS`; the default weld and the observer arm in the tests |
| `benilla-app/src/cvars/callbacks.rs` | `ForkKnobs::upscaler`, passed to `world_backdrop::on_cvar` |
| `benilla-app/src/lib.rs` | `video::install_dlss` before the render plugin; `DLSS_RUNTIME_FILES`, `has_nvidia_gpu` for the desktop exe's fetch |
| `benilla-app/src/ui_script/lifecycle.rs` | `BENILLA_FG_MAX` and `BENILLA_REFLEX` beside `BENILLA_NATIVE`, for the multiplier and Reflex rows |
| `benilla-world/src/worldview.rs` | the boot half (`install_boot`) before the viewer's `RenderPlugin` |
| `benilla-app/src/video.rs` | `install_dlss`; `boot_render_backends` resolved once; `has_nvidia_gpu` |
| `benilla-app/assets/ui/OptionsFrame.xml` | `BENILLA_TOOLTIP_UPSCALER`, `RowUpscaler` in both windows, the Frame Generation tooltip |
| `benilla-app/src/ui_script/options_tests.rs` | the upscaler row's tooltip and staged-row entries |

## Ray Reconstruction hook (not built)

`DlssSupport::ray_reconstruction` (from NGX's `SuperSamplingDenoising.Available`) gates it and
`dlss::sr::DlssSdkRes` is the one NGX instance. A ray-traced shadow/GI pass creates
`dlss_wgpu::ray_reconstruction::DlssRayReconstruction` on it (Vulkan today; its D3D12 path is the
`d3d12.rs` pattern applied to `NGX_D3D12_CREATE_DLSSD_EXT` / `NGX_D3D12_EVALUATE_DLSSD_EXT`), and in
`fsr::upscaler::fsr_upscale_pass`'s DLSS branch replaces the one `dlss::sr::encode` call with its
evaluate, fed the guide buffers (albedo, normals/roughness, specular motion) it renders.
