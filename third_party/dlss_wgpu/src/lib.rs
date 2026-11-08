//! # dlss_wgpu
//!
//! Rust bindings for integrating NVIDIA DLSS (Deep Learning Super Sampling) with `wgpu`.
//!
//! BENILLA: vendored from bevyengine/dlss_wgpu v4.0.0 and extended to wgpu's DX12 backend
//! (`d3d12.rs`). Without the NGX SDK at build time the crate is its mock (`mock.rs`). Every delta is
//! listed in `BENILLA.md`.
//!
//! For further info on how to integrate DLSS into your application, read
//! `$DLSS_SDK/doc/DLSS_Programming_Guide_Release.pdf`.

mod types;
pub use types::{
    DlssBackend, DlssError, DlssFeatureFlags, DlssFrameGenerationCamera,
    DlssFrameGenerationRenderParameters, DlssPerfQualityMode, FeatureSupport,
};

#[cfg(not(dlss_mock))]
#[cfg(windows)]
mod d3d12;
#[cfg(not(dlss_mock))]
mod feature_info;
#[cfg(not(dlss_mock))]
mod initialization;
#[cfg(not(dlss_mock))]
mod nvsdk_ngx;
#[cfg(not(dlss_mock))]
mod sdk;

/// BENILLA: DLSS Frame Generation through NGX directly.
#[cfg(not(dlss_mock))]
pub mod frame_generation;
/// DLSS Ray Reconstruction (Vulkan; the D3D12 path is the hook `BENILLA.md` describes).
#[cfg(not(dlss_mock))]
pub mod ray_reconstruction;
/// DLSS Super Resolution.
#[cfg(not(dlss_mock))]
pub mod super_resolution;

#[cfg(not(dlss_mock))]
pub use feature_info::set_runtime_paths;
#[cfg(not(dlss_mock))]
pub use initialization::{
    InitializationError, create_instance, register_device_extensions, register_instance_extensions,
    request_device,
};
#[cfg(not(dlss_mock))]
pub use sdk::DlssSdk;

#[cfg(dlss_mock)]
mod mock;
#[cfg(dlss_mock)]
pub use mock::{
    DlssSdk, frame_generation, register_device_extensions, register_instance_extensions,
    set_runtime_paths, super_resolution,
};

/// Whether this build carries NGX (`false`: the mock, built without the DLSS SDK).
pub const NGX_LINKED: bool = cfg!(not(dlss_mock));
