//! What the device offers the FSR chain: the narrow storage formats FidelityFX allocates its
//! single- and dual-channel targets in.
//!
//! WebGPU's core storage-texture set has nothing narrower than four channels. The optional
//! `"texture-formats-tier1"` feature adds `r16float`, `rg16float`, `r8unorm` and `rg8unorm`; on the
//! web that is [`TEXTURE_FORMATS_TIER1`]. On native, the adapter's own format features say it
//! (`TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`, which Bevy requests). Without either, every narrow
//! target widens to the nearest core format ([`Formats`]).

use bevy::render::render_resource::{TextureFormat, TextureUsages, WgpuFeatures};
use bevy::render::renderer::{RenderAdapter, RenderDevice};

/// The WebGPU `"texture-formats-tier1"` feature. wgpu 29 has no flag for it, so the bit is claimed
/// locally and mirrored by the vendored wgpu's `FEATURES_MAPPING` row that translates it to the
/// browser's feature string (`third_party/wgpu/PATCH.md`).
pub const TEXTURE_FORMATS_TIER1: WgpuFeatures = WgpuFeatures {
    features_wgpu: wgpu::FeaturesWGPU::empty(),
    features_webgpu: wgpu::FeaturesWebGPU::from_bits_retain(TIER1_BIT),
};

/// Bits 0–17 of `FeaturesWebGPU` are named in wgpu-types 29 (`PRIMITIVE_INDEX` is 1 << 17).
const TIER1_BIT: u64 = 1 << 18;

const _: () = assert!(
    wgpu::FeaturesWebGPU::all().bits() & TIER1_BIT == 0,
    "wgpu-types now names bit 18; use the upstream flag and drop the local one \
     (third_party/wgpu/PATCH.md)"
);

/// The four formats tier 1 adds, which the chain allocates when the device can write them.
const NARROW: [TextureFormat; 4] = [
    TextureFormat::R16Float,
    TextureFormat::Rg16Float,
    TextureFormat::R8Unorm,
    TextureFormat::Rg8Unorm,
];

/// Whether the narrow formats are writable storage on this device.
pub fn narrow_storage(device: &RenderDevice, adapter: &RenderAdapter) -> bool {
    if cfg!(target_arch = "wasm32") {
        return device.features().contains(TEXTURE_FORMATS_TIER1);
    }
    device
        .features()
        .contains(WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        && NARROW.iter().all(|&f| {
            adapter
                .get_texture_format_features(f)
                .allowed_usages
                .contains(TextureUsages::STORAGE_BINDING)
        })
}

/// Every storage format the chain writes, fixed once from [`narrow_storage`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Formats {
    /// Luminance, the instability factor, the farthest-depth mip.
    pub r16: TextureFormat,
    /// Dilated motion vectors and the shading-change pyramid.
    pub rg16: TextureFormat,
    /// The accumulation state, the shading-change factor, the locks.
    pub r8: TextureFormat,
    /// Frame generation's disocclusion mask.
    pub rg8: TextureFormat,
}

impl Formats {
    /// The reference's widths with tier 1; otherwise each widens to a core format, which costs
    /// footprint alone since only the leading channels are read or written.
    pub fn new(narrow: bool) -> Self {
        if narrow {
            Self {
                r16: TextureFormat::R16Float,
                rg16: TextureFormat::Rg16Float,
                r8: TextureFormat::R8Unorm,
                rg8: TextureFormat::Rg8Unorm,
            }
        } else {
            Self {
                r16: TextureFormat::Rgba16Float,
                rg16: TextureFormat::Rgba16Float,
                r8: TextureFormat::Rgba8Unorm,
                rg8: TextureFormat::Rgba8Unorm,
            }
        }
    }

    /// Resolves the `FSR3_FMT_*` placeholders and `FSR3_SHADING_CHANGE` in one pass's source.
    pub fn preprocess(self, source: &str) -> String {
        source
            // The two-channel tokens first: `FSR3_FMT_R16` is a prefix of neither, but keeping
            // the longer names ahead makes the order irrelevant to read.
            .replace("FSR3_FMT_RG16", wgsl_storage_format(self.rg16))
            .replace("FSR3_FMT_RG8", wgsl_storage_format(self.rg8))
            .replace("FSR3_FMT_R16", wgsl_storage_format(self.r16))
            .replace("FSR3_FMT_R8", wgsl_storage_format(self.r8))
            .replace(
                "FSR3_SHADING_CHANGE",
                if super::upscaler::SHADING_CHANGE_DETECTION {
                    "true"
                } else {
                    "false"
                },
            )
    }
}

/// The WGSL spelling of the storage formats [`Formats`] picks from.
fn wgsl_storage_format(format: TextureFormat) -> &'static str {
    match format {
        TextureFormat::R16Float => "r16float",
        TextureFormat::Rg16Float => "rg16float",
        TextureFormat::R8Unorm => "r8unorm",
        TextureFormat::Rg8Unorm => "rg8unorm",
        TextureFormat::Rgba16Float => "rgba16float",
        TextureFormat::Rgba8Unorm => "rgba8unorm",
        other => unreachable!("{other:?} is not one of the chain's storage formats"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bit the vendored wgpu maps to `"texture-formats-tier1"` is the one this constant sets,
    /// and it is the only one.
    #[test]
    fn the_tier1_flag_is_bit_18_alone() {
        assert_eq!(TEXTURE_FORMATS_TIER1.features_webgpu.bits(), 1 << 18);
        assert!(TEXTURE_FORMATS_TIER1.features_wgpu.is_empty());
    }

    /// Without tier 1 every narrow target widens to a format WebGPU's core set can bind as write-only
    /// storage, physics-rs's own fallback: `rgba16float` for the float targets, `rgba8unorm` for the
    /// unorm ones, so each keeps its range and precision.
    #[test]
    fn the_fallback_is_core_storage_of_the_same_kind() {
        let wide = Formats::new(false);
        assert_eq!(
            [wide.r16, wide.rg16, wide.r8, wide.rg8],
            [
                TextureFormat::Rgba16Float,
                TextureFormat::Rgba16Float,
                TextureFormat::Rgba8Unorm,
                TextureFormat::Rgba8Unorm,
            ]
        );
        let narrow = Formats::new(true);
        assert_eq!(
            [narrow.r16, narrow.rg16, narrow.r8, narrow.rg8],
            NARROW,
            "tier 1: the reference's own widths"
        );
    }

    /// Every placeholder resolves, in both format sets, and no token survives.
    #[test]
    fn preprocessing_leaves_no_placeholder() {
        let src = "FSR3_FMT_R16 FSR3_FMT_RG16 FSR3_FMT_R8 FSR3_FMT_RG8 FSR3_SHADING_CHANGE";
        for narrow in [false, true] {
            let out = Formats::new(narrow).preprocess(src);
            assert!(!out.contains("FSR3_"), "{out}");
        }
        assert_eq!(
            Formats::new(true).preprocess(src),
            "r16float rg16float r8unorm rg8unorm true"
        );
    }
}
