//! BENILLA: DLSS Frame Generation (DLSS-G) through NGX directly, on either backend. NGX's DLSS 4
//! SDK (310.9) exposes the feature (`NVSDK_NGX_Feature_FrameGeneration`, `nvngx_dlssg.dll`) with
//! the same create/evaluate shape as Super Resolution, so the host keeps its own present: each
//! evaluate takes the finished frame (no UI), depth and motion vectors, and writes the frame
//! halfway between it and the one evaluated before, which the host presents first.
//!
//! The helpers NGX ships (`NGX_D3D12_CREATE_DLSSG`, `NGX_D3D12_EVALUATE_DLSSG` and the `_VK` pair,
//! `nvsdk_ngx_helpers_dlssg_*.h`) are C++ and `static inline`: they only write named parameters,
//! then create or evaluate, so they are restated here and the DLSS-G headers stay out of bindgen.
//! Parameter names are `nvsdk_ngx_defs_dlssg.h`'s.
//!
//! Resource states as Super Resolution's: inputs readable (`TextureUses::RESOURCE`), the output a
//! storage target (`STORAGE_READ_WRITE`), set on the host's encoder before the evaluate.

use crate::{DlssSdk, nvsdk_ngx::*, types::*};
use std::{
    ffi::{CStr, c_void},
    iter, ptr,
    sync::{Arc, Mutex},
};
use wgpu::{
    Adapter, CommandBuffer, CommandEncoder, CommandEncoderDescriptor, Device, Queue, Texture,
    TextureFormat, TextureTransition, TextureUses, TextureView, hal::api::Vulkan,
};

pub use crate::types::{DlssFrameGenerationCamera, DlssFrameGenerationRenderParameters};

/// `nvsdk_ngx_defs_dlssg.h`'s parameter names.
mod names {
    use std::ffi::CStr;

    pub const BACKBUFFER_FORMAT: &CStr = c"DLSSG.BackbufferFormat";
    pub const WIDTH: &CStr = c"DLSSG.Width";
    pub const HEIGHT: &CStr = c"DLSSG.Height";
    pub const INTERNAL_WIDTH: &CStr = c"DLSSG.InternalWidth";
    pub const INTERNAL_HEIGHT: &CStr = c"DLSSG.InternalHeight";
    pub const DYNAMIC_RESOLUTION: &CStr = c"DLSSG.DynamicResolution";
    pub const NEVER_PROVIDED: &CStr = c"DLSSG.ResourceNeverProvidedFlags";

    pub const BACKBUFFER: &CStr = c"DLSSG.Backbuffer";
    pub const MVECS: &CStr = c"DLSSG.MVecs";
    pub const DEPTH: &CStr = c"DLSSG.Depth";
    pub const OUTPUT_INTERPOLATED: &CStr = c"DLSSG.OutputInterpolated";
    /// Optional resources benilla never provides, nulled each evaluate.
    pub const UNPROVIDED: [&CStr; 6] = [
        c"DLSSG.HUDLess",
        c"DLSSG.UI",
        c"DLSSG.UIAlpha",
        c"DLSSG.BidirectionalDistortionField",
        c"DLSSG.OutputReal",
        c"DLSSG.OutputDisableInterpolation",
    ];

    pub const MULTI_FRAME_COUNT: &CStr = c"DLSSG.MultiFrameCount";
    pub const MULTI_FRAME_INDEX: &CStr = c"DLSSG.MultiFrameIndex";
    pub const VIEW_TO_CLIP: &CStr = c"DLSSG.CameraViewToClip";
    pub const CLIP_TO_VIEW: &CStr = c"DLSSG.ClipToCameraView";
    pub const CLIP_TO_LENS_CLIP: &CStr = c"DLSSG.ClipToLensClip";
    pub const CLIP_TO_PREV_CLIP: &CStr = c"DLSSG.ClipToPrevClip";
    pub const PREV_CLIP_TO_CLIP: &CStr = c"DLSSG.PrevClipToClip";
    pub const JITTER: [&CStr; 2] = [c"DLSSG.JitterOffsetX", c"DLSSG.JitterOffsetY"];
    pub const MVEC_SCALE: [&CStr; 2] = [c"DLSSG.MvecScaleX", c"DLSSG.MvecScaleY"];
    pub const PINHOLE: [&CStr; 2] = [c"DLSSG.CameraPinholeOffsetX", c"DLSSG.CameraPinholeOffsetY"];
    pub const POS: [&CStr; 3] = [
        c"DLSSG.CameraPosX",
        c"DLSSG.CameraPosY",
        c"DLSSG.CameraPosZ",
    ];
    pub const UP: [&CStr; 3] = [c"DLSSG.CameraUpX", c"DLSSG.CameraUpY", c"DLSSG.CameraUpZ"];
    pub const RIGHT: [&CStr; 3] = [
        c"DLSSG.CameraRightX",
        c"DLSSG.CameraRightY",
        c"DLSSG.CameraRightZ",
    ];
    pub const FWD: [&CStr; 3] = [
        c"DLSSG.CameraFwdX",
        c"DLSSG.CameraFwdY",
        c"DLSSG.CameraFwdZ",
    ];
    pub const NEAR: &CStr = c"DLSSG.CameraNear";
    pub const FAR: &CStr = c"DLSSG.CameraFar";
    pub const FOV: &CStr = c"DLSSG.CameraFOV";
    pub const ASPECT: &CStr = c"DLSSG.CameraAspectRatio";
    pub const HDR: &CStr = c"DLSSG.ColorBuffersHDR";
    pub const DEPTH_INVERTED: &CStr = c"DLSSG.DepthInverted";
    pub const CAMERA_MOTION_INCLUDED: &CStr = c"DLSSG.CameraMotionIncluded";
    pub const RESET: &CStr = c"DLSSG.Reset";
    pub const AUTOMODE_OVERRIDE_RESET: &CStr = c"DLSSG.AutomodeOverrideReset";
    pub const NOT_RENDERING_GAME_FRAMES: &CStr = c"DLSSG.NotRenderingGameFrames";
    pub const ORTHO: &CStr = c"DLSSG.OrthoProjection";
    pub const MVEC_INVALID: &CStr = c"DLSSG.MvecInvalidValue";
    pub const MVEC_DILATED: &CStr = c"DLSSG.MvecDilated";
    pub const MENU_DETECTION: &CStr = c"DLSSG.MenuDetectionEnabled";
    pub const MIN_DEPTH_SEPARATION: &CStr = c"DLSSG.MinRelativeLinearDepthObjectSeparation";
    /// Every subrect, zero: the whole resource (the helpers' defaults).
    pub const SUBRECTS: [&CStr; 32] = [
        c"DLSSG.MVecsSubrectBaseX",
        c"DLSSG.MVecsSubrectBaseY",
        c"DLSSG.MVecsSubrectWidth",
        c"DLSSG.MVecsSubrectHeight",
        c"DLSSG.DepthSubrectBaseX",
        c"DLSSG.DepthSubrectBaseY",
        c"DLSSG.DepthSubrectWidth",
        c"DLSSG.DepthSubrectHeight",
        c"DLSSG.HUDLessSubrectBaseX",
        c"DLSSG.HUDLessSubrectBaseY",
        c"DLSSG.HUDLessSubrectWidth",
        c"DLSSG.HUDLessSubrectHeight",
        c"DLSSG.UISubrectBaseX",
        c"DLSSG.UISubrectBaseY",
        c"DLSSG.UISubrectWidth",
        c"DLSSG.UISubrectHeight",
        c"DLSSG.UIAlphaSubrectBaseX",
        c"DLSSG.UIAlphaSubrectBaseY",
        c"DLSSG.UIAlphaSubrectWidth",
        c"DLSSG.UIAlphaSubrectHeight",
        c"DLSSG.InputBackbufferSubrectBaseX",
        c"DLSSG.InputBackbufferSubrectBaseY",
        c"DLSSG.InputBackbufferSubrectWidth",
        c"DLSSG.InputBackbufferSubrectHeight",
        c"DLSSG.OutputInterpolatedSubrectBaseX",
        c"DLSSG.OutputInterpolatedSubrectBaseY",
        c"DLSSG.OutputInterpolatedSubrectWidth",
        c"DLSSG.OutputInterpolatedSubrectHeight",
        c"DLSSG.OutputRealSubrectBaseX",
        c"DLSSG.OutputRealSubrectBaseY",
        c"DLSSG.OutputRealSubrectWidth",
        c"DLSSG.OutputRealSubrectHeight",
    ];
}

/// `NVSDK_NGX_DLSSG_ResourceFlags`: HUDLess, UI, BidirectionalDistortionField, UIAlpha.
const NEVER_PROVIDED_FLAGS: u32 = (1 << 3) | (1 << 4) | (1 << 5) | (1 << 7);
/// `motionVectorsInvalidValue`: no vector is invalid (`FLT_MAX`, Streamline's `INVALID_FLOAT`).
const MVEC_INVALID: f32 = f32::MAX;
/// The identity, for `ClipToLensClip`: no lens distortion.
const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// One view's DLSS Frame Generation feature.
pub struct DlssFrameGeneration {
    display: [u32; 2],
    render: [u32; 2],
    format: TextureFormat,
    device: Device,
    sdk: Arc<Mutex<DlssSdk>>,
    feature: *mut NVSDK_NGX_Handle,
    backend: DlssBackend,
}

impl DlssFrameGeneration {
    /// Create the feature for a `display`-sized backbuffer read as `backbuffer_format` (a linear
    /// colour format; the output is the same) and `render`-sized depth and motion vectors.
    ///
    /// Expensive: cache it, and recreate only when a size or the format moves.
    /// [`DlssError::FeatureNotSupported`] where NGX reports no Frame Generation.
    pub fn new(
        display: [u32; 2],
        render: [u32; 2],
        backbuffer_format: TextureFormat,
        sdk: Arc<Mutex<DlssSdk>>,
        device: &Device,
        queue: &Queue,
        adapter: &Adapter,
    ) -> Result<Self, DlssError> {
        let locked = sdk.lock().unwrap();
        if !locked.frame_generation_available() {
            return Err(DlssError::FeatureNotSupported);
        }
        let backend = locked.backend;
        let native_format = match backend {
            // NGX reads the raw `VkFormat` value.
            DlssBackend::Vulkan => unsafe {
                adapter
                    .as_hal::<Vulkan>()
                    .ok_or(DlssError::UnsupportedBackend)?
                    .texture_format_as_raw(backbuffer_format)
                    .as_raw() as u32
            },
            #[cfg(windows)]
            DlssBackend::D3D12 => crate::d3d12::dxgi_format(backbuffer_format)
                .ok_or(DlssError::FeatureNotSupported)?,
            #[cfg(not(windows))]
            DlssBackend::D3D12 => return Err(DlssError::UnsupportedBackend),
        };

        let p = locked.parameters;
        unsafe {
            let ui = |n: &CStr, v: u32| NVSDK_NGX_Parameter_SetUI(p, n.as_ptr(), v);
            ui(c"CreationNodeMask", 1);
            ui(c"VisibilityNodeMask", 1);
            ui(c"Width", display[0]);
            ui(c"Height", display[1]);
            ui(names::WIDTH, display[0]);
            ui(names::HEIGHT, display[1]);
            ui(names::BACKBUFFER_FORMAT, native_format);
            ui(names::INTERNAL_WIDTH, render[0]);
            ui(names::INTERNAL_HEIGHT, render[1]);
            ui(names::DYNAMIC_RESOLUTION, 0);
            ui(names::NEVER_PROVIDED, NEVER_PROVIDED_FLAGS);
        }

        let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("dlss_frame_generation_context_creation"),
        });
        let feature = match backend {
            DlssBackend::Vulkan => unsafe {
                let raw_device = device
                    .as_hal::<Vulkan>()
                    .ok_or(DlssError::UnsupportedBackend)?
                    .raw_device()
                    .handle();
                command_encoder.as_hal_mut::<Vulkan, _, _>(|encoder| {
                    let mut handle = ptr::null_mut();
                    check_ngx_result(NVSDK_NGX_VULKAN_CreateFeature1(
                        raw_device,
                        encoder.ok_or(DlssError::UnsupportedBackend)?.raw_handle(),
                        NVSDK_NGX_Feature_NVSDK_NGX_Feature_FrameGeneration,
                        p,
                        &mut handle,
                    ))?;
                    Ok::<_, DlssError>(handle)
                })?
            },
            #[cfg(windows)]
            DlssBackend::D3D12 => {
                crate::d3d12::with_command_list(&mut command_encoder, |list| unsafe {
                    crate::d3d12::create_feature(
                        list,
                        NVSDK_NGX_Feature_NVSDK_NGX_Feature_FrameGeneration,
                        p,
                    )
                })?
            }
            #[cfg(not(windows))]
            DlssBackend::D3D12 => return Err(DlssError::UnsupportedBackend),
        };
        queue.submit([command_encoder.finish()]);

        Ok(Self {
            display,
            render,
            format: backbuffer_format,
            device: device.clone(),
            sdk: Arc::clone(&sdk),
            feature,
            backend,
        })
    }

    /// The backbuffer size the feature was created for.
    pub fn display_resolution(&self) -> [u32; 2] {
        self.display
    }

    /// The depth and motion-vector size the feature was created for.
    pub fn render_resolution(&self) -> [u32; 2] {
        self.render
    }

    /// The format the backbuffer is read as and the output written in.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Encode one evaluate: the frame between the last evaluated backbuffer and this one, into
    /// `output`. The same contract as Super Resolution's `render`: submit the returned buffer
    /// right after `command_encoder`, in the same submit.
    pub fn render(
        &mut self,
        params: DlssFrameGenerationRenderParameters,
        command_encoder: &mut CommandEncoder,
        adapter: &Adapter,
    ) -> Result<CommandBuffer, DlssError> {
        let sdk = self.sdk.lock().unwrap();
        command_encoder.transition_resources(iter::empty(), barrier_list(&params));
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("dlss_frame_generation"),
            });
        let p = sdk.parameters;
        // The matrices NGX reads through pointers live here until the evaluate returns.
        let camera = params.camera;
        unsafe { set_constants(p, &params, &camera) };
        match self.backend {
            DlssBackend::Vulkan => {
                let mut backbuffer = texture_to_ngx_as(params.backbuffer, adapter, self.format);
                let mut depth = texture_to_ngx(params.depth, adapter);
                let mut motion = texture_to_ngx(params.motion_vectors, adapter);
                let mut output = texture_to_ngx(params.output, adapter);
                unsafe {
                    let set = |n: &CStr, r: *mut NVSDK_NGX_Resource_VK| {
                        NVSDK_NGX_Parameter_SetVoidPointer(p, n.as_ptr(), r.cast::<c_void>())
                    };
                    set(names::BACKBUFFER, &mut backbuffer);
                    set(names::DEPTH, &mut depth);
                    set(names::MVECS, &mut motion);
                    set(names::OUTPUT_INTERPOLATED, &mut output);
                    for n in names::UNPROVIDED {
                        set(n, ptr::null_mut());
                    }
                    encoder.as_hal_mut::<Vulkan, _, _>(|encoder| {
                        check_ngx_result(NVSDK_NGX_VULKAN_EvaluateFeature_C(
                            encoder.ok_or(DlssError::UnsupportedBackend)?.raw_handle(),
                            self.feature,
                            p,
                            None,
                        ))
                    })?;
                }
            }
            #[cfg(windows)]
            DlssBackend::D3D12 => {
                unsafe {
                    use crate::d3d12::set_resource;
                    set_resource(p, names::BACKBUFFER, Some(params.backbuffer));
                    set_resource(p, names::DEPTH, Some(params.depth));
                    set_resource(p, names::MVECS, Some(params.motion_vectors));
                    set_resource(p, names::OUTPUT_INTERPOLATED, Some(params.output));
                    for n in names::UNPROVIDED {
                        set_resource(p, n, None);
                    }
                }
                let feature = self.feature;
                crate::d3d12::with_command_list(&mut encoder, |list| unsafe {
                    crate::d3d12::evaluate(list, feature, p)
                })?;
            }
            #[cfg(not(windows))]
            DlssBackend::D3D12 => return Err(DlssError::UnsupportedBackend),
        }
        Ok(encoder.finish())
    }
}

/// `NGX_*_EVALUATE_DLSSG`'s optional block, every member written so a parameter block shared with
/// Super Resolution never carries a stale value: one generated frame (2x), the camera, the flags.
unsafe fn set_constants(
    p: *mut NVSDK_NGX_Parameter,
    params: &DlssFrameGenerationRenderParameters,
    camera: &DlssFrameGenerationCamera,
) {
    unsafe {
        let ui = |n: &CStr, v: u32| NVSDK_NGX_Parameter_SetUI(p, n.as_ptr(), v);
        let f = |n: &CStr, v: f32| NVSDK_NGX_Parameter_SetF(p, n.as_ptr(), v);
        let m = |n: &CStr, v: &[[f32; 4]; 4]| {
            NVSDK_NGX_Parameter_SetVoidPointer(p, n.as_ptr(), v.as_ptr() as *mut c_void)
        };
        ui(names::MULTI_FRAME_COUNT, 1);
        ui(names::MULTI_FRAME_INDEX, 1);
        m(names::VIEW_TO_CLIP, &camera.view_to_clip);
        m(names::CLIP_TO_VIEW, &camera.clip_to_view);
        m(names::CLIP_TO_LENS_CLIP, &IDENTITY);
        m(names::CLIP_TO_PREV_CLIP, &camera.clip_to_prev_clip);
        m(names::PREV_CLIP_TO_CLIP, &camera.prev_clip_to_clip);
        for i in 0..2 {
            f(names::JITTER[i], params.jitter_offset[i]);
            f(names::MVEC_SCALE[i], params.motion_vector_scale[i]);
            f(names::PINHOLE[i], 0.0);
        }
        for i in 0..3 {
            f(names::POS[i], camera.position[i]);
            f(names::UP[i], camera.up[i]);
            f(names::RIGHT[i], camera.right[i]);
            f(names::FWD[i], camera.forward[i]);
        }
        f(names::NEAR, camera.near);
        f(names::FAR, camera.far);
        f(names::FOV, camera.fov_y);
        f(names::ASPECT, camera.aspect_ratio);
        ui(names::HDR, u32::from(params.hdr));
        ui(names::DEPTH_INVERTED, u32::from(params.depth_inverted));
        ui(names::CAMERA_MOTION_INCLUDED, 1);
        ui(names::RESET, u32::from(params.reset));
        ui(names::AUTOMODE_OVERRIDE_RESET, 0);
        ui(names::NOT_RENDERING_GAME_FRAMES, 0);
        ui(names::ORTHO, 0);
        f(names::MVEC_INVALID, MVEC_INVALID);
        ui(names::MVEC_DILATED, 0);
        ui(names::MENU_DETECTION, 0);
        f(names::MIN_DEPTH_SEPARATION, 40.0);
        for n in names::SUBRECTS {
            ui(n, 0);
        }
    }
}

fn barrier_list<'a>(
    params: &DlssFrameGenerationRenderParameters<'a>,
) -> impl Iterator<Item = TextureTransition<&'a Texture>> {
    let read = |view: &'a TextureView| TextureTransition {
        texture: view.texture(),
        selector: None,
        state: TextureUses::RESOURCE,
    };
    [
        read(params.backbuffer),
        read(params.depth),
        read(params.motion_vectors),
        TextureTransition {
            texture: params.output.texture(),
            selector: None,
            state: TextureUses::STORAGE_READ_WRITE,
        },
    ]
    .into_iter()
}

impl Drop for DlssFrameGeneration {
    fn drop(&mut self) {
        let sdk = self.sdk.lock().unwrap();
        sdk.wait_idle();
        unsafe {
            match self.backend {
                DlssBackend::Vulkan => {
                    let _ = check_ngx_result(NVSDK_NGX_VULKAN_ReleaseFeature(self.feature));
                }
                #[cfg(windows)]
                DlssBackend::D3D12 => crate::d3d12::release(self.feature),
                #[cfg(not(windows))]
                DlssBackend::D3D12 => {}
            }
        }
    }
}

unsafe impl Send for DlssFrameGeneration {}
unsafe impl Sync for DlssFrameGeneration {}
