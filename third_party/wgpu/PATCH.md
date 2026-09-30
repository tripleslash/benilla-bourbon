# Vendored wgpu 29.0.4: benilla's local patch

This is the crates.io `wgpu` **29.0.4** source, unmodified except for **two entries added to one
table and the two enum names those entries need**. It is wired in through `[patch.crates-io]` in
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

## Removal

Once a wgpu release that Bevy uses includes both mappings, with a real flag for tier 1:
1. Delete this directory and its `[patch.crates-io]` line.
2. Replace the local constant with the upstream one.

wgpu 30 has the bindings, but still lacks the mappings as of 30.0.1.

## Verifying diff

```
diff -ru <crates.io wgpu-29.0.4> third_party/wgpu
```

The diff shows only the three hunks above, plus this file and the removed `Cargo.lock`.
