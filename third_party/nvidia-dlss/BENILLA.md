# NVIDIA DLSS SDK, vendored: what is here and where it came from

Source: [`NVIDIA/DLSS`](https://github.com/NVIDIA/DLSS) at commit `9981c63`, DLSS SDK 310.9.16
(DLSS 4.5: Super Resolution presets K / M / L, Frame Generation). A subset, byte for byte:

- `include/`: the NGX headers `third_party/dlss_wgpu/build.rs` bindgens (`nvsdk_ngx*.h`).
- `lib/Windows_x86_64/x64/nvsdk_ngx_s.lib`, `nvsdk_ngx_d.lib`: the NGX loader, static- and
  dynamic-CRT; `build.rs` picks `_s` for the `+crt-static` desktop build.
- `lib/Windows_x86_64/rel/nvngx_dlss.dll`, `nvngx_dlssg.dll`: the production (signed) runtime
  snippets for Super Resolution and Frame Generation. Not linked: the site serves them
  (`scripts/stage-desktop.sh` → `dlss/<dll>.parts.json`) and the standalone exe fetches them into
  `wow-data/dlss/` (`crates/benilla-desktop/src/dlss.rs`); a dev build reads them from
  `benilla-config/dlss/` or `$WOW_DLSS_PATH`.

Left out: `nvngx_dlssd.dll` (Ray Reconstruction, a hook only), the development DLLs, the Linux
libraries, samples and docs.

## Licence

`LICENSE.txt`: the NVIDIA RTX SDKs licence, proprietary (`LicenseRef-NvidiaProprietary`), not
benilla's MIT OR Apache-2.0. It allows distributing the runtime DLLs with an application and linking
the loader into it, under its terms; a binary built with this SDK contains NVIDIA object code, and
modifications distributed in source carry the notice "This software contains source code provided
by NVIDIA Corporation." This copy lives in a personal fork; it is not for a public upstream.

## Updating

Check out the new tag of `NVIDIA/DLSS` and copy the same files over; keep the per-mode presets in
`dlss_wgpu`'s `pin_dlss_45_presets` in step with the SDK's release notes.
