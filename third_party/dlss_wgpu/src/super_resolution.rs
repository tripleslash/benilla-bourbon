use crate::{DlssSdk, nvsdk_ngx::*, types::*};
use std::{
    iter,
    ops::RangeInclusive,
    ptr,
    sync::{Arc, Mutex},
};
use wgpu::{
    Adapter, CommandBuffer, CommandEncoder, CommandEncoderDescriptor, Device, Queue, Texture,
    TextureTransition, TextureUses, TextureView, hal::api::Vulkan,
};

/// Camera-specific object for using DLSS Super Resolution.
pub struct DlssSuperResolution {
    upscaled_resolution: [u32; 2],
    min_render_resolution: [u32; 2],
    max_render_resolution: [u32; 2],
    device: Device,
    sdk: Arc<Mutex<DlssSdk>>,
    feature: *mut NVSDK_NGX_Handle,
    backend: DlssBackend,
}

impl DlssSuperResolution {
    /// Create a new [`DlssSuperResolution`] object.
    ///
    /// This is an expensive operation. The resulting object should be cached, and only recreated
    /// when settings change. BENILLA: on either backend, with DLSS 4.5's presets pinned.
    ///
    /// BENILLA: `fixed_render_resolution`, when the host fixes its render size itself (a render
    /// scale), creates the feature at exactly that input size; `None` creates it at the largest
    /// size the mode allows, so dynamic resolution within the range never recreates it.
    pub fn new(
        upscaled_resolution: [u32; 2],
        fixed_render_resolution: Option<[u32; 2]>,
        perf_quality_mode: DlssPerfQualityMode,
        feature_flags: DlssFeatureFlags,
        sdk: Arc<Mutex<DlssSdk>>,
        device: &Device,
        queue: &Queue,
    ) -> Result<Self, DlssError> {
        let locked_sdk = sdk.lock().unwrap();
        let backend = locked_sdk.backend;

        let perf_quality_value = perf_quality_value(perf_quality_mode, upscaled_resolution);

        let mut optimal_render_resolution = [0, 0];
        let mut min_render_resolution = [0, 0];
        let mut max_render_resolution = [0, 0];
        unsafe {
            let mut deprecated_sharpness = 0.0f32;
            check_ngx_result(NGX_DLSS_GET_OPTIMAL_SETTINGS(
                locked_sdk.parameters,
                upscaled_resolution[0],
                upscaled_resolution[1],
                perf_quality_value,
                &mut optimal_render_resolution[0],
                &mut optimal_render_resolution[1],
                &mut max_render_resolution[0],
                &mut max_render_resolution[1],
                &mut min_render_resolution[0],
                &mut min_render_resolution[1],
                &mut deprecated_sharpness,
            ))?;
            pin_dlss_45_presets(locked_sdk.parameters);
        }
        if perf_quality_mode == DlssPerfQualityMode::Dlaa {
            optimal_render_resolution = upscaled_resolution;
            min_render_resolution = upscaled_resolution;
            max_render_resolution = upscaled_resolution;
        }

        // BENILLA: the host's fixed size, or the largest the mode allows (Programming Guide
        // 3.2.2.1), so dynamic resolution within the range never recreates the feature.
        let input = fixed_render_resolution.unwrap_or([
            max_render_resolution[0].max(optimal_render_resolution[0]),
            max_render_resolution[1].max(optimal_render_resolution[1]),
        ]);
        if let Some(fixed) = fixed_render_resolution {
            min_render_resolution = fixed;
            max_render_resolution = fixed;
        }
        let mut create_params = NVSDK_NGX_DLSS_Create_Params {
            Feature: NVSDK_NGX_Feature_Create_Params {
                InWidth: input[0],
                InHeight: input[1],
                InTargetWidth: upscaled_resolution[0],
                InTargetHeight: upscaled_resolution[1],
                InPerfQualityValue: perf_quality_value,
            },
            InFeatureCreateFlags: feature_flags.as_flags() as _,
            InEnableOutputSubrects: feature_flags.contains(DlssFeatureFlags::OutputSubrect),
        };

        let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("dlss_super_resolution_context_creation"),
        });

        let mut feature = ptr::null_mut();
        match backend {
            DlssBackend::Vulkan => unsafe {
                command_encoder.as_hal_mut::<Vulkan, _, _>(|command_encoder| {
                    check_ngx_result(NGX_VULKAN_CREATE_DLSS_EXT(
                        command_encoder
                            .ok_or(DlssError::UnsupportedBackend)?
                            .raw_handle(),
                        1,
                        1,
                        &mut feature,
                        locked_sdk.parameters,
                        &mut create_params,
                    ))
                })?
            },
            #[cfg(windows)]
            DlssBackend::D3D12 => {
                let parameters = locked_sdk.parameters;
                feature = crate::d3d12::with_command_list(&mut command_encoder, |list| unsafe {
                    crate::d3d12::create_dlss(list, parameters, &create_params)
                })?;
            }
            #[cfg(not(windows))]
            DlssBackend::D3D12 => return Err(DlssError::UnsupportedBackend),
        }

        queue.submit([command_encoder.finish()]);

        Ok(Self {
            upscaled_resolution,
            min_render_resolution,
            max_render_resolution,
            device: device.clone(),
            sdk: Arc::clone(&sdk),
            feature,
            backend,
        })
    }

    /// Encode rendering commands for DLSS Super Resolution.
    ///
    /// The resulting command buffer should be submitted to a [`Queue`] in the same submit as the
    /// finished `command_encoder`, ordered immediately afterwards.
    /// ```compile_fail
    /// let mut my_command_encoder = device.create_command_encoder(descriptor);
    /// let dlss_command_buffer = dlss.render(render_parameters, &mut my_command_encoder, adapter).unwrap();
    /// queue.submit([my_command_encoder.finish(), dlss_command_buffer]);
    /// ```
    ///
    /// Failing to follow these rules is undefined behavior.
    pub fn render(
        &mut self,
        render_parameters: DlssSuperResolutionRenderParameters,
        command_encoder: &mut CommandEncoder,
        adapter: &Adapter,
    ) -> Result<CommandBuffer, DlssError> {
        let sdk = self.sdk.lock().unwrap();

        let partial_texture_size = render_parameters
            .partial_texture_size
            .unwrap_or(self.max_render_resolution);
        let mv_scale = render_parameters.motion_vector_scale.unwrap_or([1.0, 1.0]);

        command_encoder.transition_resources(iter::empty(), render_parameters.barrier_list());

        let mut dlss_command_encoder =
            self.device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("dlss_super_resolution"),
                });

        match self.backend {
            DlssBackend::Vulkan => {
                let (exposure, exposure_scale, pre_exposure) = match &render_parameters.exposure {
                    DlssSuperResolutionExposure::Manual {
                        exposure,
                        exposure_scale,
                        pre_exposure,
                    } => (
                        Some(texture_to_ngx(exposure, adapter)),
                        exposure_scale.unwrap_or(1.0),
                        pre_exposure.unwrap_or(0.0),
                    ),
                    DlssSuperResolutionExposure::Automatic => (None, 0.0, 0.0),
                };
                // Named bindings: the eval block holds pointers to them until the evaluate returns
                // (upstream's 943e15d).
                let mut exposure = exposure;
                let mut color = texture_to_ngx(render_parameters.color, adapter);
                let mut output = texture_to_ngx(render_parameters.dlss_output, adapter);
                let mut depth = texture_to_ngx(render_parameters.depth, adapter);
                let mut motion = texture_to_ngx(render_parameters.motion_vectors, adapter);
                let mut bias = render_parameters.bias.map(|b| texture_to_ngx(b, adapter));
                let mut eval_params = NVSDK_NGX_VK_DLSS_Eval_Params {
                    Feature: NVSDK_NGX_VK_Feature_Eval_Params {
                        pInColor: &mut color,
                        pInOutput: &mut output,
                        InSharpness: 0.0,
                    },
                    pInDepth: &mut depth,
                    pInMotionVectors: &mut motion,
                    InJitterOffsetX: render_parameters.jitter_offset[0],
                    InJitterOffsetY: render_parameters.jitter_offset[1],
                    InRenderSubrectDimensions: NVSDK_NGX_Dimensions {
                        Width: partial_texture_size[0],
                        Height: partial_texture_size[1],
                    },
                    InReset: render_parameters.reset as _,
                    InMVScaleX: mv_scale[0],
                    InMVScaleY: mv_scale[1],
                    pInTransparencyMask: ptr::null_mut(),
                    pInExposureTexture: exposure.as_mut().map_or(ptr::null_mut(), |e| e as *mut _),
                    pInBiasCurrentColorMask: bias.as_mut().map_or(ptr::null_mut(), |b| b as *mut _),
                    InColorSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
                    InDepthSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
                    InMVSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
                    InTranslucencySubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
                    InBiasCurrentColorSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
                    InOutputSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
                    InPreExposure: pre_exposure,
                    InExposureScale: exposure_scale,
                    InIndicatorInvertXAxis: 0,
                    InIndicatorInvertYAxis: 0,
                    GBufferSurface: NVSDK_NGX_VK_GBuffer {
                        pInAttrib: [ptr::null_mut(); 16],
                    },
                    InToneMapperType: NVSDK_NGX_ToneMapperType_NVSDK_NGX_TONEMAPPER_STRING,
                    pInMotionVectors3D: ptr::null_mut(),
                    pInIsParticleMask: ptr::null_mut(),
                    pInAnimatedTextureMask: ptr::null_mut(),
                    pInDepthHighRes: ptr::null_mut(),
                    pInPositionViewSpace: ptr::null_mut(),
                    InFrameTimeDeltaInMsec: 0.0,
                    pInRayTracingHitDistance: ptr::null_mut(),
                    pInMotionVectorsReflections: ptr::null_mut(),
                };
                unsafe {
                    dlss_command_encoder.as_hal_mut::<Vulkan, _, _>(|command_encoder| {
                        check_ngx_result(NGX_VULKAN_EVALUATE_DLSS_EXT(
                            command_encoder
                                .ok_or(DlssError::UnsupportedBackend)?
                                .raw_handle(),
                            self.feature,
                            sdk.parameters,
                            &mut eval_params,
                        ))
                    })?;
                }
            }
            #[cfg(windows)]
            DlssBackend::D3D12 => {
                let (exposure, exposure_scale, pre_exposure) = match &render_parameters.exposure {
                    DlssSuperResolutionExposure::Manual {
                        exposure,
                        exposure_scale,
                        pre_exposure,
                    } => (
                        Some(*exposure),
                        exposure_scale.unwrap_or(1.0),
                        pre_exposure.unwrap_or(0.0),
                    ),
                    DlssSuperResolutionExposure::Automatic => (None, 0.0, 0.0),
                };
                let eval = crate::d3d12::Eval {
                    color: render_parameters.color,
                    output: render_parameters.dlss_output,
                    depth: render_parameters.depth,
                    motion_vectors: render_parameters.motion_vectors,
                    exposure,
                    bias: render_parameters.bias,
                    jitter: render_parameters.jitter_offset,
                    mv_scale,
                    render_size: partial_texture_size,
                    reset: render_parameters.reset,
                    pre_exposure,
                    exposure_scale,
                };
                let (feature, parameters) = (self.feature, sdk.parameters);
                crate::d3d12::with_command_list(&mut dlss_command_encoder, |list| unsafe {
                    crate::d3d12::evaluate_dlss(list, feature, parameters, &eval)
                })?;
            }
            #[cfg(not(windows))]
            DlssBackend::D3D12 => return Err(DlssError::UnsupportedBackend),
        }
        Ok(dlss_command_encoder.finish())
    }

    /// Suggested subpixel camera jitter for a given frame.
    pub fn suggested_jitter(&self, frame_number: u32, render_resolution: [u32; 2]) -> [f32; 2] {
        let ratio = self.upscaled_resolution[0] as f32 / render_resolution[0] as f32;
        let phase_count = (8.0 * ratio * ratio) as u32;
        let i = frame_number % phase_count.max(1);

        [halton_sequence(i, 2) - 0.5, halton_sequence(i, 3) - 0.5]
    }

    /// Suggested mip bias to apply when sampling textures.
    pub fn suggested_mip_bias(&self, render_resolution: [u32; 2]) -> f32 {
        (render_resolution[0] as f32 / self.upscaled_resolution[0] as f32).log2() - 1.0
    }

    /// The upscaled resolution DLSS will output at.
    pub fn upscaled_resolution(&self) -> [u32; 2] {
        self.upscaled_resolution
    }

    /// The resolution the camera should render at, pre-upscaling.
    pub fn render_resolution(&self) -> [u32; 2] {
        self.min_render_resolution
    }

    /// Like [`Self::render_resolution`], but returns a range of values for use with dynamic
    /// resolution scaling.
    pub fn render_resolution_range(&self) -> RangeInclusive<[u32; 2]> {
        self.min_render_resolution..=self.max_render_resolution
    }
}

impl Drop for DlssSuperResolution {
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

unsafe impl Send for DlssSuperResolution {}
unsafe impl Sync for DlssSuperResolution {}

/// Inputs and output resources needed for rendering [`DlssSuperResolution`].
pub struct DlssSuperResolutionRenderParameters<'a> {
    /// Main color view of your camera.
    pub color: &'a TextureView,
    /// Depth buffer.
    pub depth: &'a TextureView,
    /// Motion vectors.
    pub motion_vectors: &'a TextureView,
    /// Camera exposure settings.
    pub exposure: DlssSuperResolutionExposure<'a>,
    /// Optional per-pixel bias to make DLSS more reactive.
    pub bias: Option<&'a TextureView>,
    /// The texture DLSS outputs to.
    pub dlss_output: &'a TextureView,
    /// Whether DLSS should reset temporal history, useful for camera cuts.
    pub reset: bool,
    /// Subpixel jitter that was applied to your camera.
    pub jitter_offset: [f32; 2],
    /// Optionally use only a specific subrect of the input textures.
    pub partial_texture_size: Option<[u32; 2]>,
    /// Optional scaling factor to apply to the values contained within [`Self::motion_vectors`].
    pub motion_vector_scale: Option<[f32; 2]>,
}

/// Camera exposure as input for [`DlssSuperResolution`].
pub enum DlssSuperResolutionExposure<'a> {
    /// Exposure controlled by the application: a 1×1 texture whose first channel is the value.
    Manual {
        exposure: &'a TextureView,
        exposure_scale: Option<f32>,
        pre_exposure: Option<f32>,
    },
    /// Auto-exposure handled by DLSS.
    Automatic,
}

impl<'a> DlssSuperResolutionRenderParameters<'a> {
    fn barrier_list(&self) -> impl Iterator<Item = TextureTransition<&'a Texture>> {
        fn resource_barrier(texture_view: &TextureView) -> TextureTransition<&Texture> {
            TextureTransition {
                texture: texture_view.texture(),
                selector: None,
                state: TextureUses::RESOURCE,
            }
        }

        [
            Some(resource_barrier(self.color)),
            Some(resource_barrier(self.depth)),
            Some(resource_barrier(self.motion_vectors)),
            match &self.exposure {
                DlssSuperResolutionExposure::Manual { exposure, .. } => {
                    Some(resource_barrier(exposure))
                }
                DlssSuperResolutionExposure::Automatic => None,
            },
            self.bias.map(resource_barrier),
            Some(TextureTransition {
                texture: self.dlss_output.texture(),
                selector: None,
                state: TextureUses::STORAGE_READ_WRITE,
            }),
        ]
        .into_iter()
        .flatten()
    }
}
