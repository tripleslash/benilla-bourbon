use crate::{feature_info::with_feature_info, nvsdk_ngx::*, types::*};
use std::{
    ffi::c_void,
    ptr,
    sync::{Arc, Mutex},
    thread,
};
use uuid::Uuid;
use wgpu::{Device, hal::api::Vulkan};

/// Application-wide DLSS object.
pub struct DlssSdk {
    pub(crate) parameters: *mut NVSDK_NGX_Parameter,
    pub(crate) device: Device,
    /// BENILLA: the API NGX was initialized on, read from the device's backend.
    pub(crate) backend: DlssBackend,
    /// BENILLA: the raw `ID3D12Device` NGX was initialized with, for its shutdown.
    #[cfg_attr(not(windows), allow(dead_code))]
    raw_d3d12: *mut c_void,
}

impl DlssSdk {
    /// Creates the DLSS SDK on the device's backend: Vulkan (upstream's path) or, BENILLA, D3D12.
    ///
    /// This should be done once per application.
    pub fn new(project_id: Uuid, device: Device) -> Result<Arc<Mutex<Self>>, DlssError> {
        let mut parameters = ptr::null_mut();
        let mut raw_d3d12 = ptr::null_mut();
        let backend = unsafe {
            if let Some(hal_device) = device.as_hal::<Vulkan>() {
                let shared_instance = hal_device.shared_instance();
                let raw_instance = shared_instance.raw_instance();
                with_feature_info(project_id, Default::default(), |feature_info| {
                    check_ngx_result(NVSDK_NGX_VULKAN_Init_with_ProjectID(
                        feature_info.Identifier.v.ProjectDesc.ProjectId,
                        NVSDK_NGX_EngineType_NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                        feature_info.Identifier.v.ProjectDesc.EngineVersion,
                        feature_info.ApplicationDataPath,
                        raw_instance.handle(),
                        hal_device.raw_physical_device(),
                        hal_device.raw_device().handle(),
                        shared_instance.entry().static_fn().get_instance_proc_addr,
                        raw_instance.fp_v1_0().get_device_proc_addr,
                        feature_info.FeatureInfo,
                        NVSDK_NGX_Version_NVSDK_NGX_Version_API,
                    ))
                })?;
                check_ngx_result(NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut parameters))?;
                DlssBackend::Vulkan
            } else {
                #[cfg(windows)]
                {
                    let raw = crate::d3d12::raw_device(&device)
                        .ok_or(DlssError::UnsupportedBackend)?;
                    parameters = with_feature_info(project_id, Default::default(), |info| {
                        crate::d3d12::init(raw, info)
                    })?;
                    raw_d3d12 = raw;
                    DlssBackend::D3D12
                }
                #[cfg(not(windows))]
                return Err(DlssError::UnsupportedBackend);
            }
        };
        // Constructed before the support check, so a refusal shuts NGX down through `Drop`.
        let sdk = Self {
            parameters,
            device,
            backend,
            raw_d3d12,
        };

        let mut dlss_supported = 0;
        unsafe {
            check_ngx_result(NVSDK_NGX_Parameter_GetI(
                parameters,
                NVSDK_NGX_Parameter_SuperSampling_Available.as_ptr().cast(),
                &mut dlss_supported,
            ))?;
        }
        if dlss_supported == 0 {
            return Err(DlssError::FeatureNotSupported);
        }
        // BENILLA: the OTA check runs only once NGX is up and DLSS is supported, so a machine
        // without it spawns nothing.
        check_for_updates(project_id);
        Ok(Arc::new(Mutex::new(sdk)))
    }

    /// The API this SDK drives NGX through.
    pub fn backend(&self) -> DlssBackend {
        self.backend
    }

    /// BENILLA: whether NGX reports DLSS Ray Reconstruction on this machine (the hook a
    /// ray-traced denoiser reads before creating `ray_reconstruction::DlssRayReconstruction`).
    pub fn ray_reconstruction_available(&self) -> bool {
        let mut available = 0;
        let ok = check_ngx_result(unsafe {
            NVSDK_NGX_Parameter_GetI(
                self.parameters,
                NVSDK_NGX_Parameter_SuperSamplingDenoising_Available
                    .as_ptr()
                    .cast(),
                &mut available,
            )
        });
        ok.is_ok() && available != 0
    }

    /// Returns the number of bytes of VRAM allocated by DLSS.
    pub fn get_vram_allocated_bytes(&mut self) -> Result<u64, DlssError> {
        let mut vram_allocated_bytes = 0;
        check_ngx_result(unsafe {
            NGX_DLSS_GET_STATS(self.parameters, &mut vram_allocated_bytes)
        })?;
        Ok(vram_allocated_bytes)
    }

    /// BENILLA: wait for the device to be idle, on either backend, before NGX frees anything.
    pub(crate) fn wait_idle(&self) {
        match self.backend {
            DlssBackend::Vulkan => unsafe {
                if let Some(hal_device) = self.device.as_hal::<Vulkan>() {
                    let _ = hal_device.raw_device().device_wait_idle();
                }
            },
            DlssBackend::D3D12 => {
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            }
        }
    }
}

fn check_for_updates(project_id: Uuid) {
    thread::spawn(move || {
        with_feature_info(
            project_id,
            NVSDK_NGX_Feature_NVSDK_NGX_Feature_SuperSampling,
            |feature_info| unsafe {
                NVSDK_NGX_UpdateFeature(&feature_info.Identifier, feature_info.FeatureID);
            },
        );
        with_feature_info(
            project_id,
            NVSDK_NGX_Feature_NVSDK_NGX_Feature_RayReconstruction,
            |feature_info| unsafe {
                NVSDK_NGX_UpdateFeature(&feature_info.Identifier, feature_info.FeatureID);
            },
        );
    });
}

impl Drop for DlssSdk {
    fn drop(&mut self) {
        self.wait_idle();
        unsafe {
            match self.backend {
                DlssBackend::Vulkan => {
                    let _ = check_ngx_result(NVSDK_NGX_VULKAN_DestroyParameters(self.parameters));
                    if let Some(hal_device) = self.device.as_hal::<Vulkan>() {
                        let _ = check_ngx_result(NVSDK_NGX_VULKAN_Shutdown1(
                            hal_device.raw_device().handle(),
                        ));
                    }
                }
                #[cfg(windows)]
                DlssBackend::D3D12 => crate::d3d12::destroy(self.parameters, self.raw_d3d12),
                #[cfg(not(windows))]
                DlssBackend::D3D12 => {}
            }
        }
    }
}

unsafe impl Send for DlssSdk {}
unsafe impl Sync for DlssSdk {}
