//! BENILLA: the public types, independent of the generated NGX bindings so the mock build has the
//! same API. The flag bits are NGX's own (`NVSDK_NGX_DLSS_Feature_Flags`); `nvsdk_ngx.rs` asserts
//! them against the generated constants in a real build.

/// How much DLSS should upscale by.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub enum DlssPerfQualityMode {
    /// Let DLSS decide.
    #[default]
    Auto,
    /// Anti-aliasing only, no upscaling.
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl DlssPerfQualityMode {
    /// BENILLA: the mode whose optimal render size is nearest `render / display` (per axis), for a
    /// host that fixes the render scale itself: NGX's ratios are 1.0, 0.667, 0.58, 0.5 and 0.333.
    pub fn for_scale(scale: f32) -> Self {
        const MODES: [(f32, DlssPerfQualityMode); 5] = [
            (1.0, DlssPerfQualityMode::Dlaa),
            (2.0 / 3.0, DlssPerfQualityMode::Quality),
            (0.58, DlssPerfQualityMode::Balanced),
            (0.5, DlssPerfQualityMode::Performance),
            (1.0 / 3.0, DlssPerfQualityMode::UltraPerformance),
        ];
        MODES
            .iter()
            .min_by(|a, b| (a.0 - scale).abs().total_cmp(&(b.0 - scale).abs()))
            .map_or(Self::Auto, |m| m.1)
    }
}

bitflags::bitflags! {
    /// Flags for creating a DLSS context.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DlssFeatureFlags: i32 {
        /// The colour input is HDR (linear, unbounded) rather than SDR.
        const HighDynamicRange = 1 << 0;
        /// Motion vectors are at the render resolution (not dilated to the display grid).
        const LowResolutionMotionVectors = 1 << 1;
        /// Motion vectors contain the jitter.
        const JitteredMotionVectors = 1 << 2;
        /// Reverse depth: near is 1, far is 0.
        const InvertedDepth = 1 << 3;
        /// DLSS meters its own exposure.
        const AutoExposure = 1 << 6;
        /// Upscale the colour's alpha too.
        const AlphaUpscaling = 1 << 7;
        /// Allow DLSS to write to a subrect of the output (not an NGX flag).
        const OutputSubrect = 1 << 8;
    }
}

impl DlssFeatureFlags {
    /// The NGX create flags: everything but [`Self::OutputSubrect`].
    pub fn as_flags(&self) -> i32 {
        let mut flags = *self;
        flags.remove(DlssFeatureFlags::OutputSubrect);
        flags.bits()
    }
}

/// Errors thrown by DLSS.
#[derive(thiserror::Error, Debug)]
pub enum DlssError {
    #[error(
        "The NGX SDK or a specific feature is not supported by the current system, hardware, and/or graphics API."
    )]
    FeatureNotSupported,
    #[error(
        "An error occurred within the underlying platform (graphics API, OS or system libraries). Consult the NGX logs and the graphics API's validation layers."
    )]
    PlatformError,
    #[error("A feature with identical parameters already exists.")]
    FeatureAlreadyExists,
    #[error("A feature associated with the provided handle could not be found.")]
    FeatureNotFound,
    #[error(
        "One or more provided parameters had an incorrect value or type, or a required one was missing."
    )]
    InvalidParameters,
    #[error("The feature requires a scratch buffer, but none was provided or it is too small.")]
    ScratchBufferTooSmall,
    #[error("The NGX SDK was used before it was initialized.")]
    NotInitialized,
    #[error("One or more input buffers had an unsupported format.")]
    UnsupportedInputFormat,
    #[error("An output buffer lacks read/write (UAV / storage) access.")]
    RWFlagMissing,
    #[error("A required input parameter was not provided.")]
    MissingInput,
    #[error("The feature could not be initialized, likely because its library could not be found.")]
    UnableToInitializeFeature,
    #[error("A newer NVIDIA driver or feature library is required.")]
    OutOfDate,
    #[error("Not enough GPU memory.")]
    OutOfGPUMemory,
    #[error("One or more buffers had an unsupported format.")]
    UnsupportedFormat,
    #[error("The SDK cannot write to the application data path.")]
    UnableToWriteToAppDataPath,
    #[error("A parameter is unsupported by this version or has an unsupported value.")]
    UnsupportedParameter,
    #[error("NVIDIA has restricted this feature in this application.")]
    Denied,
    #[error("Not implemented by this NGX SDK, driver or feature library.")]
    NotImplemented,
    /// BENILLA: the device is on a backend this crate has no path for (GL, Metal, WebGPU).
    #[error("The wgpu device is on a backend DLSS has no path for.")]
    UnsupportedBackend,
    #[error("The NGX SDK returned an unknown error code {0:#x}")]
    Unknown(i64),
}

/// Which DLSS features are supported on the current system.
pub struct FeatureSupport {
    /// DLSS Super Resolution (DLSS) is supported.
    pub super_resolution_supported: bool,
    /// DLSS Ray Reconstruction (DLSS-RR) is supported.
    pub ray_reconstruction_supported: bool,
    /// BENILLA: DLSS Frame Generation (DLSS-G) is supported.
    pub frame_generation_supported: bool,
}

impl Default for FeatureSupport {
    fn default() -> Self {
        Self {
            super_resolution_supported: true,
            ray_reconstruction_supported: true,
            frame_generation_supported: true,
        }
    }
}

/// BENILLA: the graphics API a [`crate::DlssSdk`] drives NGX through.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DlssBackend {
    Vulkan,
    D3D12,
}

/// BENILLA: the camera DLSS Frame Generation reads each frame. Every matrix is row-major
/// (`m[row][col]`, NGX's `float[4][4]` for a row vector times the matrix: glam's column-major
/// memory as is) and holds no jitter.
#[derive(Clone, Copy, Debug)]
pub struct DlssFrameGenerationCamera {
    pub view_to_clip: [[f32; 4]; 4],
    pub clip_to_view: [[f32; 4]; 4],
    pub clip_to_prev_clip: [[f32; 4]; 4],
    pub prev_clip_to_clip: [[f32; 4]; 4],
    pub position: [f32; 3],
    pub up: [f32; 3],
    pub right: [f32; 3],
    pub forward: [f32; 3],
    pub near: f32,
    pub far: f32,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    /// View width over height.
    pub aspect_ratio: f32,
}

/// BENILLA: what one DLSS Frame Generation evaluate reads and writes. NGX keeps the previous
/// backbuffer itself; the output is the frame halfway between that one and [`Self::backbuffer`].
pub struct DlssFrameGenerationRenderParameters<'a> {
    /// The finished frame without UI, at display size, read through a view in the format the
    /// feature was created with.
    pub backbuffer: &'a wgpu::TextureView,
    /// Depth at render size.
    pub depth: &'a wgpu::TextureView,
    /// Motion vectors at render size (previous − current, as DLSS Super Resolution takes them).
    pub motion_vectors: &'a wgpu::TextureView,
    /// The generated frame, display size, a storage texture of the backbuffer's format family.
    pub output: &'a wgpu::TextureView,
    /// This frame has no connection to the last evaluated one (a cut, a skipped frame).
    pub reset: bool,
    /// The jitter the frame was rendered with, in pixels.
    pub jitter_offset: [f32; 2],
    /// Scales the stored vectors to `[-1, 1]` screen units (`[1, 1]` for uv vectors).
    pub motion_vector_scale: [f32; 2],
    /// Depth is reverse-Z (near is 1).
    pub depth_inverted: bool,
    /// The backbuffer holds HDR values (not display-encoded bytes).
    pub hdr: bool,
    pub camera: DlssFrameGenerationCamera,
}
