//! BENILLA: the build without the NGX SDK (`build.rs`): the real API's shape, every constructor
//! answering [`DlssError::FeatureNotSupported`], so a host compiles one way and falls back at
//! run time.

use crate::types::*;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use wgpu::{Adapter, CommandBuffer, CommandEncoder, Device, Queue, TextureView};

/// Nothing to search: there is no NGX to load the runtime.
pub fn set_runtime_paths(_search: Vec<PathBuf>, _data: PathBuf) {}

/// Application-wide DLSS object; never constructed in the mock.
pub struct DlssSdk {
    _never: std::convert::Infallible,
}

impl DlssSdk {
    pub fn new(_project_id: Uuid, _device: Device) -> Result<Arc<Mutex<Self>>, DlssError> {
        Err(DlssError::FeatureNotSupported)
    }

    pub fn backend(&self) -> DlssBackend {
        match self._never {}
    }

    pub fn ray_reconstruction_available(&self) -> bool {
        match self._never {}
    }

    pub fn get_vram_allocated_bytes(&mut self) -> Result<u64, DlssError> {
        match self._never {}
    }
}

/// Instance extensions: none, and nothing supported.
pub fn register_instance_extensions(
    _project_id: Uuid,
    _args: &mut wgpu::hal::vulkan::CreateInstanceCallbackArgs,
    feature_support: &mut FeatureSupport,
) -> Result<(), DlssError> {
    feature_support.super_resolution_supported = false;
    feature_support.ray_reconstruction_supported = false;
    Ok(())
}

/// Device extensions: none, and nothing supported.
pub fn register_device_extensions(
    _project_id: Uuid,
    _args: &mut wgpu::hal::vulkan::CreateDeviceCallbackArgs,
    _raw_adapter: &wgpu::hal::vulkan::Adapter,
    feature_support: &mut FeatureSupport,
) -> Result<(), DlssError> {
    feature_support.super_resolution_supported = false;
    feature_support.ray_reconstruction_supported = false;
    Ok(())
}

pub mod super_resolution {
    use super::*;

    pub struct DlssSuperResolution {
        _never: std::convert::Infallible,
    }

    impl DlssSuperResolution {
        pub fn new(
            _upscaled_resolution: [u32; 2],
            _fixed_render_resolution: Option<[u32; 2]>,
            _perf_quality_mode: DlssPerfQualityMode,
            _feature_flags: DlssFeatureFlags,
            _sdk: Arc<Mutex<DlssSdk>>,
            _device: &Device,
            _queue: &Queue,
        ) -> Result<Self, DlssError> {
            Err(DlssError::FeatureNotSupported)
        }

        pub fn render(
            &mut self,
            _render_parameters: DlssSuperResolutionRenderParameters,
            _command_encoder: &mut CommandEncoder,
            _adapter: &Adapter,
        ) -> Result<CommandBuffer, DlssError> {
            match self._never {}
        }

        pub fn upscaled_resolution(&self) -> [u32; 2] {
            match self._never {}
        }

        pub fn render_resolution(&self) -> [u32; 2] {
            match self._never {}
        }

        pub fn render_resolution_range(&self) -> RangeInclusive<[u32; 2]> {
            match self._never {}
        }
    }

    pub struct DlssSuperResolutionRenderParameters<'a> {
        pub color: &'a TextureView,
        pub depth: &'a TextureView,
        pub motion_vectors: &'a TextureView,
        pub exposure: DlssSuperResolutionExposure<'a>,
        pub bias: Option<&'a TextureView>,
        pub dlss_output: &'a TextureView,
        pub reset: bool,
        pub jitter_offset: [f32; 2],
        pub partial_texture_size: Option<[u32; 2]>,
        pub motion_vector_scale: Option<[f32; 2]>,
    }

    pub enum DlssSuperResolutionExposure<'a> {
        Manual {
            exposure: &'a TextureView,
            exposure_scale: Option<f32>,
            pre_exposure: Option<f32>,
        },
        Automatic,
    }
}
