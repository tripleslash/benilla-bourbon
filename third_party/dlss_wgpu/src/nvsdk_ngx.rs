#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused)]
#![allow(unsafe_op_in_unsafe_fn)] // https://github.com/rust-lang/rust-bindgen/issues/3243
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

type PFN_vkGetDeviceProcAddr = ash::vk::PFN_vkGetDeviceProcAddr;
type PFN_vkGetInstanceProcAddr = ash::vk::PFN_vkGetInstanceProcAddr;
type VkBuffer = ash::vk::Buffer;
type VkCommandBuffer = ash::vk::CommandBuffer;
type VkDevice = ash::vk::Device;
type VkExtensionProperties = ash::vk::ExtensionProperties;
type VkFormat = ash::vk::Format;
type VkImage = ash::vk::Image;
type VkImageSubresourceRange = ash::vk::ImageSubresourceRange;
type VkImageView = ash::vk::ImageView;
type VkInstance = ash::vk::Instance;
type VkPhysicalDevice = ash::vk::PhysicalDevice;

use crate::types::{DlssError, DlssFeatureFlags, DlssPerfQualityMode};
use ash::vk::{
    ImageAspectFlags, ImageSubresourceRange, REMAINING_ARRAY_LAYERS, REMAINING_MIP_LEVELS,
};
use wgpu::{Adapter, TextureUsages, TextureView, wgc::api::Vulkan};

// BENILLA: `types.rs` spells NGX's flag bits as literals; they must be NGX's.
const _: () = {
    assert!(
        DlssFeatureFlags::HighDynamicRange.bits()
            == NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_IsHDR as i32
    );
    assert!(
        DlssFeatureFlags::LowResolutionMotionVectors.bits()
            == NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVLowRes as i32
    );
    assert!(
        DlssFeatureFlags::JitteredMotionVectors.bits()
            == NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVJittered as i32
    );
    assert!(
        DlssFeatureFlags::InvertedDepth.bits()
            == NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_DepthInverted as i32
    );
    assert!(
        DlssFeatureFlags::AutoExposure.bits()
            == NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_AutoExposure as i32
    );
    assert!(
        DlssFeatureFlags::AlphaUpscaling.bits()
            == NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_AlphaUpscaling as i32
    );
};

pub(crate) fn perf_quality_value(
    mode: DlssPerfQualityMode,
    upscaled_resolution: [u32; 2],
) -> NVSDK_NGX_PerfQuality_Value {
    match mode {
        DlssPerfQualityMode::Auto => {
            let mega_pixels =
                (upscaled_resolution[0] * upscaled_resolution[1]) as f32 / 1_000_000.0;
            if mega_pixels < 2.03 {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_DLAA
            } else if mega_pixels < 3.68 {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxQuality
            } else if mega_pixels < 8.29 {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf
            } else {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraPerformance
            }
        }
        DlssPerfQualityMode::Dlaa => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_DLAA,
        DlssPerfQualityMode::Quality => {
            NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxQuality
        }
        DlssPerfQualityMode::Balanced => {
            NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_Balanced
        }
        DlssPerfQualityMode::Performance => {
            NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf
        }
        DlssPerfQualityMode::UltraPerformance => {
            NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraPerformance
        }
    }
}

/// BENILLA: DLSS 4.5's presets, pinned rather than left at `Default` (which an OTA update may move,
/// DLSS Programming Guide 3.12.1): K for DLAA, Quality and Balanced; M for Performance; L for
/// Ultra Performance, the 310.5 SDK's own per-mode defaults. Set on the capability parameters
/// before the feature is created.
pub(crate) unsafe fn pin_dlss_45_presets(parameters: *mut NVSDK_NGX_Parameter) {
    let k = NVSDK_NGX_DLSS_Hint_Render_Preset_NVSDK_NGX_DLSS_Hint_Render_Preset_K as u32;
    let l = NVSDK_NGX_DLSS_Hint_Render_Preset_NVSDK_NGX_DLSS_Hint_Render_Preset_L as u32;
    let m = NVSDK_NGX_DLSS_Hint_Render_Preset_NVSDK_NGX_DLSS_Hint_Render_Preset_M as u32;
    for (name, preset) in [
        (&NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA[..], k),
        (&NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality[..], k),
        (&NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced[..], k),
        (&NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance[..], m),
        (&NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance[..], l),
    ] {
        NVSDK_NGX_Parameter_SetUI(parameters, name.as_ptr().cast(), preset);
    }
}

pub fn check_ngx_result(result: NVSDK_NGX_Result) -> Result<(), DlssError> {
    // BENILLA: upstream's f7c75dc — the enum-prefixed names bindgen generates; the bare names were
    // bindings, which matched every value.
    match result {
        NVSDK_NGX_Result_NVSDK_NGX_Result_Success => Ok(()),
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_FeatureNotSupported => {
            Err(DlssError::FeatureNotSupported)
        }
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_PlatformError => Err(DlssError::PlatformError),
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_FeatureAlreadyExists => {
            Err(DlssError::FeatureAlreadyExists)
        }
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_FeatureNotFound => Err(DlssError::FeatureNotFound),
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_InvalidParameter => {
            Err(DlssError::InvalidParameters)
        }
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall => {
            Err(DlssError::ScratchBufferTooSmall)
        }
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_NotInitialized => Err(DlssError::NotInitialized),
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_UnsupportedInputFormat => {
            Err(DlssError::UnsupportedInputFormat)
        }
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_RWFlagMissing => Err(DlssError::RWFlagMissing),
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_MissingInput => Err(DlssError::MissingInput),
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_UnableToInitializeFeature => {
            Err(DlssError::UnableToInitializeFeature)
        }
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_OutOfDate => Err(DlssError::OutOfDate),
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_OutOfGPUMemory => Err(DlssError::OutOfGPUMemory),
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_UnsupportedFormat => {
            Err(DlssError::UnsupportedFormat)
        }
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath => {
            Err(DlssError::UnableToWriteToAppDataPath)
        }
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_UnsupportedParameter => {
            Err(DlssError::UnsupportedParameter)
        }
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_Denied => Err(DlssError::Denied),
        NVSDK_NGX_Result_NVSDK_NGX_Result_FAIL_NotImplemented => Err(DlssError::NotImplemented),
        other => Err(DlssError::Unknown(other as i64)),
    }
}

pub fn texture_to_ngx(texture_view: &TextureView, adapter: &Adapter) -> NVSDK_NGX_Resource_VK {
    unsafe {
        let raw_view = texture_view.as_hal::<Vulkan>().unwrap().raw_handle();
        let texture = texture_view.texture();

        NVSDK_NGX_Create_ImageView_Resource_VK(
            raw_view,
            texture.as_hal::<Vulkan>().unwrap().raw_handle(),
            ImageSubresourceRange {
                aspect_mask: if texture.format().has_color_aspect() {
                    ImageAspectFlags::COLOR
                } else {
                    ImageAspectFlags::DEPTH
                },
                base_mip_level: 0,
                level_count: REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: REMAINING_ARRAY_LAYERS,
            },
            adapter
                .as_hal::<Vulkan>()
                .unwrap()
                .texture_format_as_raw(texture.format()),
            texture.width(),
            texture.height(),
            texture.usage().contains(TextureUsages::STORAGE_BINDING),
        )
    }
}

pub fn halton_sequence(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut result = 0.0;
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index = (index as f32 / base as f32).floor() as u32;
    }
    result
}
