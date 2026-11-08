# Vulkan headers, vendored: for bindgen only

Source: [`KhronosGroup/Vulkan-Headers`](https://github.com/KhronosGroup/Vulkan-Headers) `v1.3.281`
(the version `ash` 0.38 tracks): `include/vulkan/*.h` and `include/vk_video/*.h`, byte for byte.
Apache-2.0 OR MIT (`LICENSE.md`, `LICENSES/`).

Only `third_party/dlss_wgpu/build.rs` reads them, so bindgen can parse the NGX Vulkan headers without
a LunarG SDK installed (`$VULKAN_SDK` overrides). No Vulkan type is generated from them (`Vk*` is
blocklisted; `ash` supplies those) and nothing links against them.
