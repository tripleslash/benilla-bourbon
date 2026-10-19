# Vendored wgpu 29.0.4: benilla's local patch

This is the crates.io `wgpu` **29.0.4** source, unmodified except for the hunks below: **two entries added to one
table and the two enum names those entries need**, the browser WGSL fix-ups, the device's
limits and features read once, async pipeline creation and the per-frame labels. It is wired in through `[patch.crates-io]` in
the root `Cargo.toml`. Bevy 0.19.1 asks for `^29.0.3`, which this satisfies. The upstream
`README.md` is kept alongside this file.

All four lines serve the WebGPU (browser) backend alone. The native backends go through
`wgpu-core`, which is not vendored. Search for `benilla local patch` to find every change.

## Where the change is

**`src/backend/webgpu.rs`, `FEATURES_MAPPING`.** This table drives three things:
- `adapter.features()`
- `device.features()`
- the required-features list `request_device` sends

A WebGPU feature missing from it is invisible to the first two and cannot be requested through the
safe API.

**`src/backend/webgpu/webgpu_sys/gen_GpuFeatureName.rs`.** The vendored bindings predate both names
below, so the string enum gains `Subgroups = "subgroups"` and
`TextureFormatsTier1 = "texture-formats-tier1"`. wgpu 30's bindings have them.

## 1. `"subgroups"`

```rust
(wgt::Features::SUBGROUP, webgpu_sys::GpuFeatureName::Subgroups),
```

Carried from physics-rs's identical patch, so the browser build can reach subgroup intrinsics.

## 2. `"texture-formats-tier1"`

```rust
(
    wgt::Features {
        features_wgpu: wgt::FeaturesWGPU::empty(),
        features_webgpu: wgt::FeaturesWebGPU::from_bits_retain(1 << 18),
    },
    webgpu_sys::GpuFeatureName::TextureFormatsTier1,
),
```

`wgt::FeaturesWebGPU` has no flag for this feature. wgpu-types is not vendored, so the bit is
claimed locally: `1 << 18`, the next free bit after `PRIMITIVE_INDEX` (`1 << 17`).

Without it, the FSR 3.1.5 upscaler (`crates/benilla-world/src/fsr/`) cannot allocate one- and
two-channel 16-bit storage targets in the browser. It has to widen them to `Rgba16Float` /
`Rgba8Unorm` instead.

**The bit is a local convention, not an upstream constant.** It is mirrored by
`benilla_world::fsr::caps::TEXTURE_FORMATS_TIER1`. That constant carries a compile-time assertion
that `FeaturesWebGPU` still leaves the bit unnamed, so a wgpu-types bump that assigns it fails the
build instead of silently requesting the wrong feature.

## Browser WGSL (`src/backend/webgpu.rs`, `browser_wgsl`)

A module Bevy hands over as naga IR is written back to WGSL for the browser by naga 29's writer.
Two things it writes, the browser's compiler rejects, so `browser_wgsl` post-processes the text:

- `f32::MAX` comes out as `340282350000000000000000000000000000000f`, a decimal above the largest
  f32 ("value cannot be represented as 'f32'" in Tint); it is rewritten to `3.4028234e38f`, which is
  in range and rounds to exactly `f32::MAX`. The FSR shaders' `FSR3_FP32_MAX`/`FG_FLT_MAX` hit it.
- A module that uses subgroup or quad builtins (possible only because the `SUBGROUP` mapping above
  exposes the feature) needs `enable subgroups;`, which naga does not write; it is prepended when a
  `@builtin(subgroup_*)` or a `subgroupX(` / `quadX(` call appears.
  With it goes `diagnostic(off, subgroup_uniformity);`: Tint cannot prove the quad ops in Bevy's
  depth downsample (`downsample_second`) subgroup-uniform and rejects them; naga validated the module.

Remove it with the rest once naga's writer emits both correctly.

## Device limits and features read once (`src/backend/webgpu.rs`, `WebDevice`)

`DeviceInterface::limits()` and `::features()` rebuilt their answer from the JS
`GPUSupportedLimits` / `GPUSupportedFeatures` objects on every call — ~40 getters, each a crossing
of the wasm→JS bridge. Bevy treats `RenderDevice::limits()` as free: `bevy_pbr`'s
`SetMeshBindGroup` calls it once per mesh draw. Measured in the browser world viewer (JS
Self-Profiling, Elwynn, ~3,700 draws a frame): `map_wgt_limits` was 15% of the frame thread. A
device's limits and features are fixed at creation, so `WebDevice` keeps both in a `OnceCell`
filled on first read.

Upstream would take it as is; remove it with the rest once a wgpu release Bevy uses caches them.

## Async pipeline creation (`src/api/device.rs`, `src/backend/webgpu.rs`)

`Device::create_render_pipeline_async` / `create_compute_pipeline_async`, WebGPU backend only
(`None` elsewhere): the descriptor is mapped exactly as the synchronous path maps it
(`map_render_pipeline_descriptor`, `map_compute_pipeline_descriptor`, now shared) and handed to
`createRenderPipelineAsync` / `createComputePipelineAsync`; the returned future owns only the
promise. Measured in Chrome on D3D12: a synchronous `createRenderPipeline` returns at once in JS but
holds the GPU process's queue for the whole compile (~600 ms for one heavy fragment shader), so
every frame behind it stalls; the async form leaves the queue alone. `third_party/bevy_render`'s
pipeline cache is the caller.

## Labels on per-frame objects (`src/backend/webgpu.rs`, `HOT_LABELS`)

Bind groups, command encoders and their buffers, render and compute passes and texture views carry
their labels into the browser only in debug builds. Each label is a JS string made across the
bridge, and `CommandEncoder::finish` read the encoder's label back out of JS to relabel the buffer;
a frame makes ~60 bind groups, ~30 passes and ~25 encoders (~1% of the browser frame thread).
Pipelines, buffers, textures, samplers and layouts keep their labels in every build.

## Explicit release of buffers, textures and query sets (`src/backend/webgpu.rs`, `src/lib.rs`)

Upstream's `Drop` for `WebBuffer`, `WebTexture` and `WebQuerySet` is a no-op: the browser frees the
GPU memory when the JS wrapper is garbage-collected. A thread that allocates little JS collects
rarely, and the browser build's render worker allocates almost none, so dropped objects piled up
until the 8 GB card paged (measured: ~10 MB/s, frames falling to 12 fps a minute into play).

Each of the three now carries a `Release` guard (`Rc`), shared by every clone, held by every
`WebTextureView` made from the texture and by every `WebBindGroup` binding the buffer or view. When
the last of them drops, the JS object goes onto a per-thread list, and
`wgpu::destroy_released_resources()` (re-exported at the crate root, `webgpu` only) calls
`destroy()` on each. The caller runs it once the work recorded with them has been submitted:
`bevy_render` does so in `RenderSystems::PostCleanup` (its `BENILLA.md`, ninth delta). A surface's
texture gets a guard that owns nothing: it belongs to the canvas. Bind groups have no `destroy()` in
WebGPU and are still freed by the collector.

## Removal

Once a wgpu release that Bevy uses includes both mappings, with a real flag for tier 1:
1. Delete this directory and its `[patch.crates-io]` line.
2. Replace the local constant with the upstream one.

wgpu 30 has the bindings, but still lacks the mappings as of 30.0.1.

## Verifying diff

```
diff -ru <crates.io wgpu-29.0.4> third_party/wgpu
```

The diff shows only the hunks above, plus this file and the removed `Cargo.lock`.
