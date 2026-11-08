//! BENILLA: DLSS through NGX's D3D12 API, on wgpu-hal's raw DX12 handles. Upstream dlss_wgpu is
//! Vulkan-only; this is the same feature on the same NGX library, driven through
//! `NVSDK_NGX_D3D12_*` instead of `NVSDK_NGX_VULKAN_*`.
//!
//! The entry points are declared here by hand on opaque COM pointers (`nvsdk_ngx.h`, inside its
//! `extern "C"` block), so bindgen needs no Windows SDK headers. The two `static inline` helpers
//! NGX ships for D3D12 (`NGX_D3D12_CREATE_DLSS_EXT`, `NGX_D3D12_EVALUATE_DLSS_EXT` in
//! `nvsdk_ngx_helpers_d3d.h`) are restated in Rust: they only set named parameters, then create or
//! evaluate.
//!
//! Resource states (DLSS Programming Guide 3.4): inputs in `NON_PIXEL_SHADER_RESOURCE`, the output
//! in `UNORDERED_ACCESS`; NGX returns every buffer to that state. wgpu's `TextureUses::RESOURCE`
//! is `PIXEL | NON_PIXEL_SHADER_RESOURCE` on DX12 and `STORAGE_READ_WRITE` is `UNORDERED_ACCESS`,
//! so the caller's transitions on its own encoder put them there, and wgpu's tracker still holds the
//! truth when the next command buffer runs.

use crate::nvsdk_ngx::*;
use std::ffi::{c_char, c_void};
use std::ptr;
use wgpu::{CommandEncoder, Device, Texture, TextureView, hal::api::Dx12};
use windows_core::Interface;

type ComPtr = *mut c_void;
type PFN_NVSDK_NGX_ProgressCallback_C = Option<unsafe extern "C" fn(f32, *mut bool)>;

unsafe extern "C" {
    fn NVSDK_NGX_D3D12_Init_with_ProjectID(
        in_project_id: *const c_char,
        in_engine_type: NVSDK_NGX_EngineType,
        in_engine_version: *const c_char,
        in_application_data_path: *const wchar_t,
        in_device: ComPtr,
        in_feature_info: *const NVSDK_NGX_FeatureCommonInfo,
        in_sdk_version: NVSDK_NGX_Version,
    ) -> NVSDK_NGX_Result;
    fn NVSDK_NGX_D3D12_Shutdown1(in_device: ComPtr) -> NVSDK_NGX_Result;
    fn NVSDK_NGX_D3D12_GetCapabilityParameters(
        out_parameters: *mut *mut NVSDK_NGX_Parameter,
    ) -> NVSDK_NGX_Result;
    fn NVSDK_NGX_D3D12_DestroyParameters(in_parameters: *mut NVSDK_NGX_Parameter)
    -> NVSDK_NGX_Result;
    fn NVSDK_NGX_D3D12_CreateFeature(
        in_cmd_list: ComPtr,
        in_feature_id: NVSDK_NGX_Feature,
        in_parameters: *mut NVSDK_NGX_Parameter,
        out_handle: *mut *mut NVSDK_NGX_Handle,
    ) -> NVSDK_NGX_Result;
    fn NVSDK_NGX_D3D12_ReleaseFeature(in_handle: *mut NVSDK_NGX_Handle) -> NVSDK_NGX_Result;
    fn NVSDK_NGX_D3D12_EvaluateFeature_C(
        in_cmd_list: ComPtr,
        in_feature_handle: *const NVSDK_NGX_Handle,
        in_parameters: *const NVSDK_NGX_Parameter,
        in_callback: PFN_NVSDK_NGX_ProgressCallback_C,
    ) -> NVSDK_NGX_Result;
    fn NVSDK_NGX_Parameter_SetD3d12Resource(
        in_parameter: *mut NVSDK_NGX_Parameter,
        in_name: *const c_char,
        in_value: ComPtr,
    );
}

/// `NVSDK_NGX_Parameter_*` names are NUL-terminated byte strings in the bindings.
fn name(n: &[u8]) -> *const c_char {
    n.as_ptr().cast()
}

/// The device's raw `ID3D12Device`, or `None` off DX12.
pub(crate) fn raw_device(device: &Device) -> Option<ComPtr> {
    unsafe { device.as_hal::<Dx12>().map(|d| d.raw_device().as_raw()) }
}

/// A texture's raw `ID3D12Resource`.
pub(crate) fn raw_resource(texture: &Texture) -> ComPtr {
    unsafe {
        texture
            .as_hal::<Dx12>()
            .map_or(ptr::null_mut(), |t| t.raw_resource().as_raw())
    }
}

fn view_resource(view: &TextureView) -> ComPtr {
    raw_resource(&view.texture())
}

/// Run `f` on the encoder's open `ID3D12GraphicsCommandList` (wgpu-hal's
/// `CommandEncoder::raw_command_list`, a `third_party/wgpu-hal` hook).
pub(crate) fn with_command_list<R>(
    encoder: &mut CommandEncoder,
    f: impl FnOnce(ComPtr) -> Result<R, crate::DlssError>,
) -> Result<R, crate::DlssError> {
    unsafe {
        encoder.as_hal_mut::<Dx12, _, _>(|hal| {
            let list = hal
                .and_then(|hal| hal.raw_command_list())
                .ok_or(crate::DlssError::UnsupportedBackend)?;
            f(list.as_raw())
        })
    }
}

/// `NVSDK_NGX_D3D12_Init_with_ProjectID`, then the capability parameters.
pub(crate) unsafe fn init(
    device: ComPtr,
    feature_info: &NVSDK_NGX_FeatureDiscoveryInfo,
) -> Result<*mut NVSDK_NGX_Parameter, crate::DlssError> {
    unsafe {
        check_ngx_result(NVSDK_NGX_D3D12_Init_with_ProjectID(
            feature_info.Identifier.v.ProjectDesc.ProjectId,
            NVSDK_NGX_EngineType_NVSDK_NGX_ENGINE_TYPE_CUSTOM,
            feature_info.Identifier.v.ProjectDesc.EngineVersion,
            feature_info.ApplicationDataPath,
            device,
            feature_info.FeatureInfo,
            NVSDK_NGX_Version_NVSDK_NGX_Version_API,
        ))?;
        let mut parameters = ptr::null_mut();
        if let Err(e) = check_ngx_result(NVSDK_NGX_D3D12_GetCapabilityParameters(&mut parameters))
        {
            let _ = NVSDK_NGX_D3D12_Shutdown1(device);
            return Err(e);
        }
        Ok(parameters)
    }
}

pub(crate) unsafe fn destroy(parameters: *mut NVSDK_NGX_Parameter, device: ComPtr) {
    unsafe {
        let _ = NVSDK_NGX_D3D12_DestroyParameters(parameters);
        let _ = NVSDK_NGX_D3D12_Shutdown1(device);
    }
}

pub(crate) unsafe fn destroy_parameters(parameters: *mut NVSDK_NGX_Parameter) {
    unsafe {
        let _ = NVSDK_NGX_D3D12_DestroyParameters(parameters);
    }
}

/// `NGX_D3D12_CREATE_DLSS_EXT`.
pub(crate) unsafe fn create_dlss(
    cmd_list: ComPtr,
    parameters: *mut NVSDK_NGX_Parameter,
    create: &NVSDK_NGX_DLSS_Create_Params,
) -> Result<*mut NVSDK_NGX_Handle, crate::DlssError> {
    unsafe {
        let p = parameters;
        NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_CreationNodeMask), 1);
        NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_VisibilityNodeMask), 1);
        NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_Width), create.Feature.InWidth);
        NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_Height), create.Feature.InHeight);
        NVSDK_NGX_Parameter_SetUI(
            p,
            name(NVSDK_NGX_Parameter_OutWidth),
            create.Feature.InTargetWidth,
        );
        NVSDK_NGX_Parameter_SetUI(
            p,
            name(NVSDK_NGX_Parameter_OutHeight),
            create.Feature.InTargetHeight,
        );
        NVSDK_NGX_Parameter_SetI(
            p,
            name(NVSDK_NGX_Parameter_PerfQualityValue),
            create.Feature.InPerfQualityValue as _,
        );
        NVSDK_NGX_Parameter_SetI(
            p,
            name(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags),
            create.InFeatureCreateFlags as _,
        );
        NVSDK_NGX_Parameter_SetI(
            p,
            name(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects),
            i32::from(create.InEnableOutputSubrects),
        );
        let mut handle = ptr::null_mut();
        check_ngx_result(NVSDK_NGX_D3D12_CreateFeature(
            cmd_list,
            NVSDK_NGX_Feature_NVSDK_NGX_Feature_SuperSampling,
            p,
            &mut handle,
        ))?;
        Ok(handle)
    }
}

/// What one DLSS evaluate reads and writes, on D3D12.
pub(crate) struct Eval<'a> {
    pub color: &'a TextureView,
    pub output: &'a TextureView,
    pub depth: &'a TextureView,
    pub motion_vectors: &'a TextureView,
    pub exposure: Option<&'a TextureView>,
    pub bias: Option<&'a TextureView>,
    pub jitter: [f32; 2],
    pub mv_scale: [f32; 2],
    pub render_size: [u32; 2],
    pub reset: bool,
    pub pre_exposure: f32,
    pub exposure_scale: f32,
}

/// `NGX_D3D12_EVALUATE_DLSS_EXT`, the members benilla's inputs fill; every optional resource the
/// helper sets to null is set to null here too, so a parameter block reused across features never
/// carries a stale pointer.
pub(crate) unsafe fn evaluate_dlss(
    cmd_list: ComPtr,
    handle: *mut NVSDK_NGX_Handle,
    parameters: *mut NVSDK_NGX_Parameter,
    e: &Eval,
) -> Result<(), crate::DlssError> {
    unsafe {
        let p = parameters;
        let res = |n: &[u8], r: ComPtr| NVSDK_NGX_Parameter_SetD3d12Resource(p, name(n), r);
        res(NVSDK_NGX_Parameter_Color, view_resource(e.color));
        res(NVSDK_NGX_Parameter_Output, view_resource(e.output));
        res(NVSDK_NGX_Parameter_Depth, view_resource(e.depth));
        res(NVSDK_NGX_Parameter_MotionVectors, view_resource(e.motion_vectors));
        res(
            NVSDK_NGX_Parameter_ExposureTexture,
            e.exposure.map_or(ptr::null_mut(), view_resource),
        );
        res(
            NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask,
            e.bias.map_or(ptr::null_mut(), view_resource),
        );
        for n in [
            &NVSDK_NGX_Parameter_TransparencyMask[..],
            &NVSDK_NGX_Parameter_MotionVectors3D[..],
            &NVSDK_NGX_Parameter_IsParticleMask[..],
            &NVSDK_NGX_Parameter_AnimatedTextureMask[..],
            &NVSDK_NGX_Parameter_DepthHighRes[..],
            &NVSDK_NGX_Parameter_Position_ViewSpace[..],
            &NVSDK_NGX_Parameter_RayTracingHitDistance[..],
            &NVSDK_NGX_Parameter_MotionVectorsReflection[..],
        ] {
            res(n, ptr::null_mut());
        }
        NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_Jitter_Offset_X), e.jitter[0]);
        NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_Jitter_Offset_Y), e.jitter[1]);
        NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_Sharpness), 0.0);
        NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_Reset), i32::from(e.reset));
        NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_MV_Scale_X), e.mv_scale[0]);
        NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_MV_Scale_Y), e.mv_scale[1]);
        NVSDK_NGX_Parameter_SetUI(
            p,
            name(NVSDK_NGX_Parameter_TonemapperType),
            NVSDK_NGX_ToneMapperType_NVSDK_NGX_TONEMAPPER_STRING as u32,
        );
        NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_FrameTimeDeltaInMsec), 0.0);
        for n in [
            &NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X[..],
            &NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y[..],
            &NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X[..],
            &NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y[..],
            &NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X[..],
            &NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y[..],
            &NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_X[..],
            &NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_Y[..],
            &NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_X[..],
            &NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_Y[..],
            &NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X[..],
            &NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y[..],
        ] {
            NVSDK_NGX_Parameter_SetUI(p, name(n), 0);
        }
        NVSDK_NGX_Parameter_SetUI(
            p,
            name(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width),
            e.render_size[0],
        );
        NVSDK_NGX_Parameter_SetUI(
            p,
            name(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height),
            e.render_size[1],
        );
        NVSDK_NGX_Parameter_SetF(
            p,
            name(NVSDK_NGX_Parameter_DLSS_Pre_Exposure),
            if e.pre_exposure == 0.0 { 1.0 } else { e.pre_exposure },
        );
        NVSDK_NGX_Parameter_SetF(
            p,
            name(NVSDK_NGX_Parameter_DLSS_Exposure_Scale),
            if e.exposure_scale == 0.0 { 1.0 } else { e.exposure_scale },
        );
        NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_DLSS_Indicator_Invert_X_Axis), 0);
        NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_DLSS_Indicator_Invert_Y_Axis), 0);
        check_ngx_result(NVSDK_NGX_D3D12_EvaluateFeature_C(cmd_list, handle, p, None))
    }
}

pub(crate) unsafe fn release(handle: *mut NVSDK_NGX_Handle) {
    unsafe {
        let _ = NVSDK_NGX_D3D12_ReleaseFeature(handle);
    }
}
