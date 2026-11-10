//! fork: dlss (FORK.md) — NVIDIA Streamline 2.14.16, manual hooking
//! (`third_party/nvidia-dlss/streamline/docs/ProgrammingGuideManualHooking.md`), Windows only:
//! DLSS Frame Generation with its own present-side pacing and multi-frame generation
//! (`ProgrammingGuideDLSS_G.md`), and NVIDIA Reflex with PC Latency markers
//! (`ProgrammingGuideReflex.md`, `ProgrammingGuidePCL.md`) under either frame generation.
//!
//! **Boot** (`boot`, before `RenderPlugin`, on an NVIDIA adapter with the runtime in the DLSS
//! folder): the player's `sl.interposer.dll` is checked for a valid Authenticode signature
//! (Streamline guide 2.1.1), loaded, and `slInit` runs with `eUseManualHooking`, loading DLSS-G,
//! Reflex and PC Latency. Its hooks go into wgpu-hal (`third_party/wgpu-hal`, "The sixth patch"):
//! on DX12 `slUpgradeInterface` for the direct queue and the HWND swap chain; on Vulkan the
//! interposer is the Vulkan loader.
//!
//! **Device**: on DX12 `slSetD3DDevice` when wgpu-hal opens the adapter, before its first proxy;
//! then (`finish`) Reflex's and PC Latency's functions; then
//! `slIsFeatureSupported(DLSS_G)` on this adapter and, where it runs, `slDLSSGGetState`'s
//! `numFramesToGenerateMax`, the multiplier the card reaches ([`DlssSupport::frame_generation_max`]).
//!
//! **Each frame**: a frame token per main-world update; `slReflexSleep` and PC Latency's
//! simulation start in `First`, its end in `Last`; its submit and present markers from
//! `bevy_render`'s present hook (a held frame of FSR's cadence marks nothing: it is no game frame);
//! in the render world the common constants, the tags (depth, motion vectors, the HUD-less world)
//! and DLSS-G's options, on while [`super::DlssFrameGeneration`] holds, `numFramesToGenerate` from
//! [`super::DlssgFrames`]. Reflex's mode follows [`super::Reflex`], forced on while DLSS-G runs
//! (DLSS-G guide 8.0).
//!
//! The structs below restate Streamline's public headers (`third_party/nvidia-dlss/streamline/
//! include/sl*.h`, MIT, NVIDIA 2022–2023) field for field; the layouts are what
//! `sl.interposer.dll` reads.

#![allow(non_snake_case, non_upper_case_globals, dead_code)]

use std::ffi::{c_char, c_void, CString};
use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};
use std::sync::OnceLock;

use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::*;
use bevy::render::renderer::{
    render_system, RenderAdapter, RenderContext, RenderDevice, ViewQuery,
};
use bevy::render::sync_world::MainEntity;
use bevy::render::view::{ExtractedView, ViewDepthStencilTexture, ViewTarget};
use bevy::render::{Render, RenderApp, RenderSystems};
use windows_core::Interface;

use super::{DlssFrameGeneration, DlssRuntime, DlssSupport, DlssgFrames, Reflex, PROJECT_ID};
use crate::ffx_glow::FfxBackdrop;
use crate::fsr::{FsrHeld, FsrState};

// ── sl_struct.h / sl_core_types.h / sl_consts.h ──

#[repr(C)]
#[derive(Clone, Copy)]
struct StructType {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> StructType {
    StructType {
        data1,
        data2,
        data3,
        data4,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Base {
    next: *mut Base,
    struct_type: StructType,
    struct_version: usize,
}

impl Base {
    const fn new(struct_type: StructType, struct_version: usize) -> Self {
        Self {
            next: ptr::null_mut(),
            struct_type,
            struct_version,
        }
    }
}

type Feature = u32;
const kFeatureReflex: Feature = 3;
const kFeaturePCL: Feature = 4;
const kFeatureDLSS_G: Feature = 1000;

type BufferType = u32;
const kBufferTypeDepth: BufferType = 0;
const kBufferTypeMotionVectors: BufferType = 1;
const kBufferTypeHUDLessColor: BufferType = 2;

/// `slInit`'s version argument: `kSDKVersion` of the headers these structs restate, 2.14.16.
const SL_SDK_VERSION: u64 = (2 << 48) | (14 << 32) | (16 << 16) | 0xfedc;

const ePreferenceDisableCLStateTracking: u64 = 1 << 0;
const ePreferenceUseManualHooking: u64 = 1 << 2;

const eRenderD3D12: u32 = 1;
const eRenderVulkan: u32 = 2;

#[repr(C)]
struct Preferences {
    base: Base,
    show_console: bool,
    log_level: u32,
    paths_to_plugins: *const *const u16,
    num_paths_to_plugins: u32,
    path_to_logs_and_data: *const u16,
    allocate_callback: *const c_void,
    release_callback: *const c_void,
    log_message_callback: Option<unsafe extern "C" fn(u32, *const c_char)>,
    flags: u64,
    features_to_load: *const Feature,
    num_features_to_load: u32,
    application_id: u32,
    engine: u32,
    engine_version: *const c_char,
    project_id: *const c_char,
    render_api: u32,
}

#[repr(C)]
struct ViewportHandle {
    base: Base,
    value: u32,
}

impl ViewportHandle {
    fn zero() -> Self {
        Self {
            base: Base::new(
                guid(
                    0x171b6435,
                    0x9b3c,
                    0x4fc8,
                    [0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4],
                ),
                1,
            ),
            value: 0,
        }
    }
}

#[repr(C)]
struct AdapterInfo {
    base: Base,
    device_luid: *const u8,
    device_luid_size_in_bytes: u32,
    vk_physical_device: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Extent {
    top: u32,
    left: u32,
    width: u32,
    height: u32,
}

#[repr(C)]
struct Resource {
    base: Base,
    kind: i8,
    native: *mut c_void,
    memory: *mut c_void,
    view: *mut c_void,
    state: u32,
    width: u32,
    height: u32,
    native_format: u32,
    mip_levels: u32,
    array_layers: u32,
    gpu_virtual_address: u64,
    flags: u32,
    usage: u32,
    internal_flags: u16,
    reserved: u16,
}

const eResourceTex2d: i8 = 0;
const eValidUntilPresent: u32 = 1;

#[repr(C)]
struct ResourceTag {
    base: Base,
    resource: *mut Resource,
    kind: BufferType,
    lifecycle: u32,
    extent: Extent,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Float2 {
    x: f32,
    y: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Float3 {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Float4x4 {
    row: [[f32; 4]; 4],
}

impl Float4x4 {
    /// glam's column-major memory is Streamline's row-major, row-vector matrix (`v * M`).
    fn from(m: Mat4) -> Self {
        Self {
            row: m.to_cols_array_2d(),
        }
    }
}

const eFalse: i8 = 0;
const eTrue: i8 = 1;
const INVALID_FLOAT: f32 = 3.402_823_5e38;

#[repr(C)]
struct Constants {
    base: Base,
    camera_view_to_clip: Float4x4,
    clip_to_camera_view: Float4x4,
    clip_to_lens_clip: Float4x4,
    clip_to_prev_clip: Float4x4,
    prev_clip_to_clip: Float4x4,
    jitter_offset: Float2,
    mvec_scale: Float2,
    camera_pinhole_offset: Float2,
    camera_pos: Float3,
    camera_up: Float3,
    camera_right: Float3,
    camera_fwd: Float3,
    camera_near: f32,
    camera_far: f32,
    camera_fov: f32,
    camera_aspect_ratio: f32,
    motion_vectors_invalid_value: f32,
    depth_inverted: i8,
    camera_motion_included: i8,
    motion_vectors_3d: i8,
    reset: i8,
    orthographic_projection: i8,
    motion_vectors_dilated: i8,
    motion_vectors_jittered: i8,
    min_relative_linear_depth_object_separation: f32,
}

// ── sl_dlss_g.h / sl_reflex.h / sl_pcl.h ──

const eDLSSGModeOff: u32 = 0;
const eDLSSGModeOn: u32 = 1;

#[repr(C)]
struct DlssgOptions {
    base: Base,
    mode: u32,
    num_frames_to_generate: u32,
    flags: u32,
    dynamic_res_width: u32,
    dynamic_res_height: u32,
    num_back_buffers: u32,
    mvec_depth_width: u32,
    mvec_depth_height: u32,
    color_width: u32,
    color_height: u32,
    color_buffer_format: u32,
    mvec_buffer_format: u32,
    depth_buffer_format: u32,
    hud_less_buffer_format: u32,
    ui_buffer_format: u32,
    on_error_callback: *const c_void,
    b_reserved15: i8,
    queue_parallelism_mode: u32,
    enable_user_interface_recomposition: i8,
    dynamic_target_frame_rate: f32,
}

#[repr(C)]
struct ReflexOptions {
    base: Base,
    mode: u32,
    frame_limit_us: u32,
    use_markers_to_optimize: bool,
    virtual_key: u16,
    id_thread: u32,
}

const eReflexOff: u32 = 0;
const eReflexLowLatency: u32 = 1;
const eReflexLowLatencyWithBoost: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct ReflexReport {
    base: Base,
    frame_id: u64,
    input_sample_time: u64,
    sim_start_time: u64,
    sim_end_time: u64,
    render_submit_start_time: u64,
    render_submit_end_time: u64,
    present_start_time: u64,
    present_end_time: u64,
    driver_start_time: u64,
    driver_end_time: u64,
    os_render_queue_start_time: u64,
    os_render_queue_end_time: u64,
    gpu_render_start_time: u64,
    gpu_render_end_time: u64,
    gpu_active_render_time_us: u32,
    gpu_frame_time_us: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ReflexReport2 {
    base: Base,
    camera_constructed_time: u64,
    cross_adapter_copy_time_us: u32,
}

const kReflexFrameReportCount: usize = 64;

#[repr(C)]
struct ReflexState {
    base: Base,
    low_latency_available: bool,
    latency_report_available: bool,
    stats_window_message: u32,
    frame_report: [ReflexReport; kReflexFrameReportCount],
    flash_indicator_driver_controlled: bool,
    frame_report2: [ReflexReport2; kReflexFrameReportCount],
}

impl ReflexState {
    fn new() -> Box<Self> {
        let report = ReflexReport {
            base: Base::new(
                guid(
                    0x0d56_9b37,
                    0xa1c8,
                    0x4453,
                    [0xbe, 0x4d, 0x40, 0xf4, 0xde, 0x57, 0x95, 0x2b],
                ),
                1,
            ),
            frame_id: 0,
            input_sample_time: 0,
            sim_start_time: 0,
            sim_end_time: 0,
            render_submit_start_time: 0,
            render_submit_end_time: 0,
            present_start_time: 0,
            present_end_time: 0,
            driver_start_time: 0,
            driver_end_time: 0,
            os_render_queue_start_time: 0,
            os_render_queue_end_time: 0,
            gpu_render_start_time: 0,
            gpu_render_end_time: 0,
            gpu_active_render_time_us: 0,
            gpu_frame_time_us: 0,
        };
        let report2 = ReflexReport2 {
            base: Base::new(
                guid(
                    0x68bb_0632,
                    0x5e1c,
                    0x402b,
                    [0x89, 0x9d, 0xb4, 0x9f, 0x63, 0x3c, 0x56, 0xc2],
                ),
                1,
            ),
            camera_constructed_time: 0,
            cross_adapter_copy_time_us: 0,
        };
        Box::new(Self {
            base: Base::new(
                guid(
                    0xf0bb_5985,
                    0xdaf9,
                    0x4728,
                    [0xb2, 0xfd, 0xae, 0x80, 0xa2, 0xbd, 0x79, 0x89],
                ),
                2,
            ),
            low_latency_available: false,
            latency_report_available: false,
            stats_window_message: 0,
            frame_report: [report; kReflexFrameReportCount],
            flash_indicator_driver_controlled: false,
            frame_report2: [report2; kReflexFrameReportCount],
        })
    }
}

#[repr(C)]
struct DlssgState {
    base: Base,
    estimated_vram_usage_in_bytes: u64,
    status: u32,
    min_width_or_height: u32,
    num_frames_actually_presented: u32,
    num_frames_to_generate_max: u32,
    b_reserved4: i8,
    b_is_vsync_support_available: i8,
    inputs_processing_completion_fence: *mut c_void,
    last_present_inputs_processing_completion_fence_value: u64,
    b_is_dynamic_mfg_supported: i8,
}

impl DlssgState {
    fn new() -> Self {
        Self {
            base: Base::new(
                guid(
                    0xcc8a_c8e1,
                    0xa179,
                    0x44f5,
                    [0x97, 0xfa, 0xe7, 0x41, 0x12, 0xf9, 0xbc, 0x61],
                ),
                4,
            ),
            estimated_vram_usage_in_bytes: 0,
            status: 0,
            min_width_or_height: 0,
            num_frames_actually_presented: 0,
            num_frames_to_generate_max: 0,
            b_reserved4: 0,
            b_is_vsync_support_available: 0,
            inputs_processing_completion_fence: ptr::null_mut(),
            last_present_inputs_processing_completion_fence_value: 0,
            b_is_dynamic_mfg_supported: 0,
        }
    }
}

const eSimulationStart: u32 = 0;
const eSimulationEnd: u32 = 1;
const eRenderSubmitStart: u32 = 2;
const eRenderSubmitEnd: u32 = 3;
const ePresentStart: u32 = 4;
const ePresentEnd: u32 = 5;

/// `sl::FrameToken`: opaque, owned by Streamline.
type FrameToken = c_void;

// ── the interposer's exports and the feature functions ──

type SlResult = i32;
type PfnInit = unsafe extern "C" fn(*const Preferences, u64) -> SlResult;
type PfnUpgradeInterface = unsafe extern "C" fn(*mut *mut c_void) -> SlResult;
type PfnSetD3DDevice = unsafe extern "C" fn(*mut c_void) -> SlResult;
type PfnIsFeatureSupported = unsafe extern "C" fn(Feature, *const AdapterInfo) -> SlResult;
type PfnGetFeatureFunction =
    unsafe extern "C" fn(Feature, *const c_char, *mut *mut c_void) -> SlResult;
type PfnGetNewFrameToken = unsafe extern "C" fn(*mut *mut FrameToken, *const u32) -> SlResult;
type PfnSetConstants =
    unsafe extern "C" fn(*const Constants, *const FrameToken, *const ViewportHandle) -> SlResult;
type PfnSetTagForFrame = unsafe extern "C" fn(
    *const FrameToken,
    *const ViewportHandle,
    *const ResourceTag,
    u32,
    *mut c_void,
) -> SlResult;
type PfnDlssgSetOptions =
    unsafe extern "C" fn(*const ViewportHandle, *const DlssgOptions) -> SlResult;
type PfnDlssgGetState =
    unsafe extern "C" fn(*const ViewportHandle, *mut DlssgState, *const DlssgOptions) -> SlResult;
type PfnReflexSetOptions = unsafe extern "C" fn(*const ReflexOptions) -> SlResult;
type PfnReflexSleep = unsafe extern "C" fn(*const FrameToken) -> SlResult;
type PfnReflexGetState = unsafe extern "C" fn(*mut ReflexState) -> SlResult;
type PfnPclSetMarker = unsafe extern "C" fn(u32, *const FrameToken) -> SlResult;

struct Sl {
    _library: libloading::Library,
    render_api: u32,
    is_feature_supported: PfnIsFeatureSupported,
    set_d3d_device: PfnSetD3DDevice,
    get_feature_function: PfnGetFeatureFunction,
    get_new_frame_token: PfnGetNewFrameToken,
    set_constants: PfnSetConstants,
    set_tag_for_frame: PfnSetTagForFrame,
}

/// Loaded at boot, for the process.
static SL: OnceLock<Sl> = OnceLock::new();

/// The feature functions, once the device was given: Reflex's and PC Latency's always, DLSS-G's
/// where this adapter runs it.
struct Features {
    dlssg: Option<(PfnDlssgSetOptions, PfnDlssgGetState)>,
    reflex_set_options: PfnReflexSetOptions,
    reflex_sleep: PfnReflexSleep,
    reflex_get_state: PfnReflexGetState,
    pcl_set_marker: PfnPclSetMarker,
}

static FEATURES: OnceLock<Features> = OnceLock::new();

/// The main world's frame index, and the render world's current token for the present markers.
static FRAME: AtomicU32 = AtomicU32::new(0);
static RENDER_TOKEN: AtomicPtr<FrameToken> = AtomicPtr::new(ptr::null_mut());
/// DLSS-G's generated frames per rendered frame as last set (0: off), so the options call runs on
/// a change only.
static DLSSG_FRAMES: AtomicU32 = AtomicU32::new(0);
/// Reflex's mode as last set (`sl::ReflexMode`; `u32::MAX` before the first), which also says
/// whether `slReflexSleep` runs each frame.
static REFLEX_MODE: AtomicU32 = AtomicU32::new(u32::MAX);

/// Streamline's runtime, in the DLSS folder beside NGX's ([`super::RUNTIME_FILES`]):
/// `NvLowLatencyVk.dll` is Reflex's Vulkan library, `nvngx_dlssg.dll` DLSS-G's NGX snippet.
pub const RUNTIME_FILES: [&str; 7] = [
    "sl.interposer.dll",
    "sl.common.dll",
    "sl.dlss_g.dll",
    "sl.reflex.dll",
    "sl.pcl.dll",
    "NvLowLatencyVk.dll",
    "nvngx_dlssg.dll",
];

fn wide(path: &Path) -> Vec<u16> {
    path.as_os_str().encode_wide().chain([0]).collect()
}

unsafe extern "C" fn log_message(kind: u32, msg: *const c_char) {
    if msg.is_null() {
        return;
    }
    let text = unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy();
    let text = text.trim_end();
    match kind {
        2 => error!("streamline: {text}"),
        1 => warn!("streamline: {text}"),
        _ => debug!("streamline: {text}"),
    }
}

/// `WinVerifyTrust` with the generic Authenticode policy: the file carries a valid, trusted
/// signature (Streamline guide 2.1.1, "secure load").
fn signed(path: &Path) -> bool {
    #[repr(C)]
    struct Guid(u32, u16, u16, [u8; 8]);
    #[repr(C)]
    struct FileInfo {
        cb_struct: u32,
        path: *const u16,
        file: *mut c_void,
        known_subject: *const Guid,
    }
    #[repr(C)]
    struct TrustData {
        cb_struct: u32,
        policy_callback_data: *mut c_void,
        sip_client_data: *mut c_void,
        ui_choice: u32,
        revocation_checks: u32,
        union_choice: u32,
        file: *mut FileInfo,
        state_action: u32,
        state_data: *mut c_void,
        url_reference: *const u16,
        prov_flags: u32,
        ui_context: u32,
        signature_settings: *mut c_void,
    }
    type PfnVerify = unsafe extern "system" fn(*mut c_void, *const Guid, *mut TrustData) -> i32;
    const GENERIC_VERIFY_V2: Guid = Guid(
        0x00aa_c56b,
        0xcd44,
        0x11d0,
        [0x8c, 0xc2, 0x00, 0xc0, 0x4f, 0xc2, 0x95, 0xee],
    );
    let Ok(lib) = (unsafe { libloading::Library::new("wintrust.dll") }) else {
        return false;
    };
    let Ok(verify) = (unsafe { lib.get::<PfnVerify>(b"WinVerifyTrust\0") }) else {
        return false;
    };
    let wpath = wide(path);
    let mut file = FileInfo {
        cb_struct: std::mem::size_of::<FileInfo>() as u32,
        path: wpath.as_ptr(),
        file: ptr::null_mut(),
        known_subject: ptr::null(),
    };
    let mut data = TrustData {
        cb_struct: std::mem::size_of::<TrustData>() as u32,
        policy_callback_data: ptr::null_mut(),
        sip_client_data: ptr::null_mut(),
        ui_choice: 2,         // WTD_UI_NONE
        revocation_checks: 0, // WTD_REVOKE_NONE
        union_choice: 1,      // WTD_CHOICE_FILE
        file: &mut file,
        state_action: 1, // WTD_STATEACTION_VERIFY
        state_data: ptr::null_mut(),
        url_reference: ptr::null(),
        prov_flags: 0,
        ui_context: 0,
        signature_settings: ptr::null_mut(),
    };
    let trusted = unsafe { verify(ptr::null_mut(), &GENERIC_VERIFY_V2, &mut data) } == 0;
    data.state_action = 2; // WTD_STATEACTION_CLOSE
    unsafe { verify(ptr::null_mut(), &GENERIC_VERIFY_V2, &mut data) };
    trusted
}

/// Before `RenderPlugin`: secure-load Streamline, `slInit` in manual-hooking mode, and hand its
/// hooks to wgpu-hal. On any refusal, one log line and nothing installed: DLSS-G stays unsupported
/// and FSR frame generation stands in.
pub(super) fn boot(_app: &mut App, runtime: &DlssRuntime, backends: wgpu::Backends) {
    let render_api = if backends == wgpu::Backends::DX12 {
        eRenderD3D12
    } else if backends == wgpu::Backends::VULKAN {
        eRenderVulkan
    } else {
        info!("dlss: frame generation needs Direct3D 12 or Vulkan; FSR frame generation stands in");
        return;
    };
    if let Some(missing) = RUNTIME_FILES
        .iter()
        .find(|f| !runtime.dir.join(f).is_file())
    {
        info!(
            "dlss: no Streamline runtime ({missing}) in {}; FSR frame generation stands in",
            runtime.dir.display()
        );
        return;
    }
    let interposer = runtime.dir.join("sl.interposer.dll");
    if !signed(&interposer) {
        info!("dlss: sl.interposer.dll is not validly signed; FSR frame generation stands in");
        return;
    }
    let Ok(library) = (unsafe { libloading::Library::new(&interposer) }) else {
        info!("dlss: sl.interposer.dll did not load; FSR frame generation stands in");
        return;
    };
    macro_rules! sym {
        ($name:literal, $ty:ty) => {
            match unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) } {
                Ok(f) => *f,
                Err(_) => {
                    info!(
                        "dlss: sl.interposer.dll lacks {}; FSR frame generation stands in",
                        $name
                    );
                    return;
                }
            }
        };
    }
    let init = sym!("slInit", PfnInit);
    let upgrade = sym!("slUpgradeInterface", PfnUpgradeInterface);
    let sl = Sl {
        is_feature_supported: sym!("slIsFeatureSupported", PfnIsFeatureSupported),
        set_d3d_device: sym!("slSetD3DDevice", PfnSetD3DDevice),
        get_feature_function: sym!("slGetFeatureFunction", PfnGetFeatureFunction),
        get_new_frame_token: sym!("slGetNewFrameToken", PfnGetNewFrameToken),
        set_constants: sym!("slSetConstants", PfnSetConstants),
        set_tag_for_frame: sym!("slSetTagForFrame", PfnSetTagForFrame),
        render_api,
        _library: library,
    };

    let _ = std::fs::create_dir_all(&runtime.data);
    let plugin_dir = wide(&runtime.dir);
    let paths = [plugin_dir.as_ptr()];
    let logs = wide(&runtime.data);
    let features = [kFeatureDLSS_G, kFeatureReflex, kFeaturePCL];
    let project = CString::new(PROJECT_ID).unwrap();
    let engine_version = CString::new(env!("CARGO_PKG_VERSION")).unwrap();
    let prefs = Preferences {
        base: Base::new(
            guid(
                0x1ca10965,
                0xbf8e,
                0x432b,
                [0x8d, 0xa1, 0x67, 0x16, 0xd8, 0x79, 0xfb, 0x14],
            ),
            1,
        ),
        show_console: false,
        log_level: 1, // eDefault
        paths_to_plugins: paths.as_ptr(),
        num_paths_to_plugins: 1,
        path_to_logs_and_data: logs.as_ptr(),
        allocate_callback: ptr::null(),
        release_callback: ptr::null(),
        log_message_callback: Some(log_message),
        flags: ePreferenceDisableCLStateTracking | ePreferenceUseManualHooking,
        features_to_load: features.as_ptr(),
        num_features_to_load: features.len() as u32,
        application_id: 0,
        engine: 0, // eCustom
        engine_version: engine_version.as_ptr(),
        project_id: project.as_ptr(),
        render_api,
    };
    let result = unsafe { init(&prefs, SL_SDK_VERSION) };
    if result != 0 {
        info!("dlss: slInit refused ({result}); FSR frame generation stands in");
        return;
    }
    match render_api {
        eRenderD3D12 => {
            // The device first, then the proxies (manual-hooking guide 5.1): wgpu-hal calls
            // `slSetD3DDevice` when the adapter opens, before it upgrades the queue.
            wgpu::hal::dx12::interposer::set_device_hook(sl.set_d3d_device);
            wgpu::hal::dx12::interposer::set_upgrade_interface(upgrade);
        }
        _ => {
            wgpu::hal::vulkan::interposer::set_loader(interposer);
        }
    }
    bevy::render::renderer::present_hook::set(present_marker);
    let _ = SL.set(sl);
}

fn feature_fn<T: Copy>(sl: &Sl, feature: Feature, name: &str) -> Option<T> {
    let name = CString::new(name).ok()?;
    let mut f = ptr::null_mut();
    let ok = unsafe { (sl.get_feature_function)(feature, name.as_ptr(), &mut f) } == 0;
    (ok && !f.is_null()).then(|| unsafe { std::mem::transmute_copy::<*mut c_void, T>(&f) })
}

/// `DlssPlugin::finish`, before NGX opens: give Streamline the device, take Reflex's and PC
/// Latency's functions, and ask about DLSS-G on this adapter and the most frames it generates.
/// Sets [`DlssSupport::reflex`], [`DlssSupport::frame_generation`] and
/// [`DlssSupport::frame_generation_max`].
pub(super) fn finish(app: &mut App) {
    let Some(sl) = SL.get() else {
        return;
    };
    let Some(render_app) = app.get_sub_app(RenderApp) else {
        return;
    };
    let Some(device) = render_app.world().get_resource::<RenderDevice>() else {
        return;
    };
    let device = device.wgpu_device();
    let mut luid = [0u8; 8];
    let mut info = AdapterInfo {
        base: Base::new(
            guid(
                0x0677315f,
                0xa746,
                0x4492,
                [0x9f, 0x42, 0xcb, 0x61, 0x42, 0xc9, 0xc3, 0xd4],
            ),
            1,
        ),
        device_luid: ptr::null(),
        device_luid_size_in_bytes: 0,
        vk_physical_device: ptr::null_mut(),
    };
    unsafe {
        if sl.render_api == eRenderD3D12 {
            let Some(hal) = device.as_hal::<wgpu::hal::api::Dx12>() else {
                return;
            };
            let raw = hal.raw_device();
            // Handed over when the adapter opened (`boot`'s device hook).
            if wgpu::hal::dx12::interposer::device_set() != Some(true) {
                info!("dlss: slSetD3DDevice refused; no Reflex, FSR frame generation stands in");
                return;
            }
            let id = raw.GetAdapterLuid();
            luid[..4].copy_from_slice(&id.LowPart.to_le_bytes());
            luid[4..].copy_from_slice(&id.HighPart.to_le_bytes());
            info.device_luid = luid.as_ptr();
            info.device_luid_size_in_bytes = 8;
        } else if let Some(hal) = device.as_hal::<wgpu::hal::api::Vulkan>() {
            use ash::vk::Handle;
            info.vk_physical_device = hal.raw_physical_device().as_raw() as *mut c_void;
        }
    }
    let reflex_supported = unsafe { (sl.is_feature_supported)(kFeatureReflex, &info) } == 0;
    let (
        true,
        Some(reflex_set_options),
        Some(reflex_sleep),
        Some(reflex_get_state),
        Some(pcl_set_marker),
    ) = (
        reflex_supported,
        feature_fn::<PfnReflexSetOptions>(sl, kFeatureReflex, "slReflexSetOptions"),
        feature_fn::<PfnReflexSleep>(sl, kFeatureReflex, "slReflexSleep"),
        feature_fn::<PfnReflexGetState>(sl, kFeatureReflex, "slReflexGetState"),
        feature_fn::<PfnPclSetMarker>(sl, kFeaturePCL, "slPCLSetMarker"),
    )
    else {
        info!("dlss: Streamline offers no Reflex here; no Reflex, FSR frame generation stands in");
        return;
    };
    let dlssg_supported = unsafe { (sl.is_feature_supported)(kFeatureDLSS_G, &info) };
    let dlssg =
        if dlssg_supported == 0 {
            feature_fn::<PfnDlssgSetOptions>(sl, kFeatureDLSS_G, "slDLSSGSetOptions").zip(
                feature_fn::<PfnDlssgGetState>(sl, kFeatureDLSS_G, "slDLSSGGetState"),
            )
        } else {
            None
        };
    // `numFramesToGenerateMax`: 1 where the card reaches 2x (RTX 40), 3 for 4x (RTX 50).
    let max_generated = dlssg.map_or(0, |(_, get_state)| {
        let mut state = DlssgState::new();
        let viewport = ViewportHandle::zero();
        let ok = unsafe { get_state(&viewport, &mut state, ptr::null()) } == 0;
        if ok {
            state.num_frames_to_generate_max.max(1)
        } else {
            1
        }
    });
    let low_latency = {
        let mut state = ReflexState::new();
        unsafe { reflex_get_state(&mut *state) == 0 && state.low_latency_available }
    };
    info!(
        "dlss: Streamline up: Reflex {}, DLSS frame generation {}",
        if low_latency {
            "available"
        } else {
            "not available"
        },
        if dlssg.is_some() {
            format!("up to {}x", max_generated + 1)
        } else {
            format!("not supported here ({dlssg_supported}); FSR frame generation stands in")
        }
    );
    let _ = FEATURES.set(Features {
        dlssg,
        reflex_set_options,
        reflex_sleep,
        reflex_get_state,
        pcl_set_marker,
    });
    let mut support = app
        .world()
        .get_resource::<DlssSupport>()
        .copied()
        .unwrap_or_default();
    support.reflex = low_latency;
    support.frame_generation = dlssg.is_some();
    support.frame_generation_max = if dlssg.is_some() {
        max_generated + 1
    } else {
        0
    };
    app.insert_resource(support);
}

/// The token Streamline keys this frame index on (the same index answers the same token).
fn token(sl: &Sl, index: u32) -> *mut FrameToken {
    let mut token = ptr::null_mut();
    unsafe { (sl.get_new_frame_token)(&mut token, &index) };
    token
}

fn marker(marker: u32, token: *mut FrameToken) {
    if let (Some(f), false) = (FEATURES.get(), token.is_null()) {
        unsafe { (f.pcl_set_marker)(marker, token) };
    }
}

/// `bevy_render`'s present hook: PC Latency's submit and present markers on the render world's
/// current frame. A held frame of FSR's cadence (`fsr::cadence::run_held_frames`) is no game frame
/// of its own and marks nothing.
fn present_marker(phase: bevy::render::renderer::present_hook::Phase) {
    use bevy::render::renderer::present_hook::Phase;
    if crate::fsr::cadence::RENDERING_HELD.load(Ordering::Acquire) {
        return;
    }
    let m = match phase {
        Phase::SubmitStart => eRenderSubmitStart,
        Phase::SubmitEnd => eRenderSubmitEnd,
        Phase::PresentStart => ePresentStart,
        Phase::PresentEnd => ePresentEnd,
    };
    marker(m, RENDER_TOKEN.load(Ordering::Acquire));
}

/// The main world's frame index, handed to the render world.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
struct SlFrame(u32);

/// `First`: Reflex's mode when it moved ([`Reflex`], on while DLSS-G runs), then this frame's
/// token, Reflex's sleep before anything samples input, and the simulation's start.
fn simulation_start(
    mut frame: ResMut<SlFrame>,
    reflex: Option<Res<Reflex>>,
    dlss_fg: Option<Res<DlssFrameGeneration>>,
) {
    let (Some(sl), Some(f)) = (SL.get(), FEATURES.get()) else {
        return;
    };
    let wanted = reflex.map_or(Reflex::Off, |r| *r);
    let forced = dlss_fg.is_some_and(|d| d.0) && wanted == Reflex::Off;
    let mode = match (wanted, forced) {
        (_, true) | (Reflex::On, _) => eReflexLowLatency,
        (Reflex::Boost, _) => eReflexLowLatencyWithBoost,
        (Reflex::Off, false) => eReflexOff,
    };
    if REFLEX_MODE.swap(mode, Ordering::AcqRel) != mode {
        let options = ReflexOptions {
            base: Base::new(
                guid(
                    0xf03af81a,
                    0x6d0b,
                    0x4902,
                    [0xa6, 0x51, 0xc4, 0x96, 0x5e, 0x21, 0x54, 0x34],
                ),
                1,
            ),
            mode,
            frame_limit_us: 0,
            use_markers_to_optimize: false,
            virtual_key: 0,
            id_thread: 0,
        };
        let result = unsafe { (f.reflex_set_options)(&options) };
        info!(
            "dlss: Reflex {}{} ({result})",
            ["off", "on", "on + boost"][mode as usize],
            if forced {
                ", forced on by DLSS frame generation"
            } else {
                ""
            }
        );
    }
    frame.0 = FRAME.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    let token = token(sl, frame.0);
    if mode != eReflexOff {
        unsafe { (f.reflex_sleep)(token) };
    }
    marker(eSimulationStart, token);
}

fn simulation_end(frame: Res<SlFrame>) {
    let Some(sl) = SL.get() else { return };
    marker(eSimulationEnd, token(sl, frame.0));
}

/// Reflex's latency over its last reports: the mean milliseconds from the simulation's start to
/// the GPU's end of the frame (the PC latency less the input-to-simulation part, which Reflex
/// measures only with an input marker), and from the present's start to the GPU's end, with the
/// number of reports. `None` where Reflex reports nothing (off, or not yet).
pub fn latency_report() -> Option<(f32, f32, usize)> {
    let f = FEATURES.get()?;
    let mut state = ReflexState::new();
    if unsafe { (f.reflex_get_state)(&mut *state) } != 0 || !state.latency_report_available {
        return None;
    }
    let reports: Vec<&ReflexReport> = state
        .frame_report
        .iter()
        .filter(|r| {
            r.frame_id != 0
                && r.sim_start_time != 0
                && r.gpu_render_end_time > r.sim_start_time
                && r.gpu_render_end_time >= r.present_start_time
        })
        .collect();
    if reports.is_empty() {
        return None;
    }
    let n = reports.len() as f64;
    // Reflex's timestamps are microseconds.
    let total = reports
        .iter()
        .map(|r| (r.gpu_render_end_time - r.sim_start_time) as f64)
        .sum::<f64>();
    let tail = reports
        .iter()
        .map(|r| (r.gpu_render_end_time - r.present_start_time) as f64)
        .sum::<f64>();
    Some((
        (total / n / 1000.0) as f32,
        (tail / n / 1000.0) as f32,
        reports.len(),
    ))
}

/// Whether Streamline's Reflex runs on this device.
pub fn reflex_running() -> bool {
    FEATURES.get().is_some() && REFLEX_MODE.load(Ordering::Acquire) != eReflexOff
}

/// The world as DLSS-G's HUD-less input: the backdrop's own combine of the upscaled image, in the
/// claiming camera's format, without the interface (DLSS-G guide 5.0).
#[derive(Component)]
pub(crate) struct DlssHudless {
    texture: Texture,
    view: TextureView,
    pub(crate) format: TextureFormat,
    pub(crate) decode: bool,
    size: UVec2,
}

/// `PrepareResources`: a HUD-less target on each world view while DLSS-G runs, sized and formatted
/// like the claiming backdrop camera's target.
fn prepare_hudless(
    mut commands: Commands,
    device: Res<RenderDevice>,
    fg: Res<DlssFrameGeneration>,
    views: Query<(Entity, &MainEntity, &FsrState, Option<&DlssHudless>)>,
    backdrops: Query<(&FfxBackdrop, &ViewTarget)>,
) {
    for (entity, main, fsr, hudless) in &views {
        let claim = backdrops
            .iter()
            .find(|(b, _)| b.source == Some(main.id()))
            .map(|(b, target)| (target.main_texture_format(), b.decode));
        let dlssg = FEATURES.get().is_some_and(|f| f.dlssg.is_some());
        let (true, Some((format, decode))) = (fg.0 && dlssg, claim) else {
            if hudless.is_some() {
                commands
                    .entity(entity)
                    .remove::<(DlssHudless, super::HudlessCompose)>();
            }
            continue;
        };
        let size = fsr.display_size();
        if hudless.is_some_and(|h| h.size == size && h.format == format && h.decode == decode) {
            continue;
        }
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("dlss_hudless"),
            size: Extent3d {
                width: size.x.max(1),
                height: size.y.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        commands.entity(entity).insert((
            DlssHudless {
                texture,
                view,
                format,
                decode,
                size,
            },
            super::HudlessCompose { format, decode },
        ));
    }
}

/// `PostProcess`, after the FFXGlow filter: compose the HUD-less world, then leave every tagged
/// input in the read state the tags declare, which Streamline relies on at present.
fn hudless_pass(
    world: &World,
    view: ViewQuery<(&FsrState, &DlssHudless, &ViewDepthStencilTexture)>,
    held: Res<FsrHeld>,
    mut render_context: RenderContext,
) {
    if held.0 {
        return;
    }
    let entity = view.entity();
    let (fsr, hudless, depth) = view.into_inner();
    let Some(upscaled) = fsr.output() else {
        return;
    };
    crate::ffx_glow::compose_world_final(
        world,
        entity,
        upscaled,
        &hudless.view,
        &mut render_context,
    );
    fn read(t: &wgpu::Texture) -> wgpu::TextureTransition<&wgpu::Texture> {
        wgpu::TextureTransition {
            texture: t,
            selector: None,
            state: wgpu::TextureUses::RESOURCE,
        }
    }
    let mv = fsr.motion_vectors().texture();
    render_context.command_encoder().transition_resources(
        std::iter::empty(),
        [read(depth.texture()), read(mv), read(&hudless.texture)].into_iter(),
    );
}

/// One tagged texture, described the way the backend wants it (manual hooking 6.1, 6.2).
fn describe(
    texture: &wgpu::Texture,
    view: &wgpu::TextureView,
    adapter: &RenderAdapter,
    depth: bool,
) -> Resource {
    let mut r = Resource {
        base: Base::new(
            guid(
                0x3a9d70cf,
                0x2418,
                0x4b72,
                [0x83, 0x91, 0x13, 0xf8, 0x72, 0x1c, 0x72, 0x61],
            ),
            1,
        ),
        kind: eResourceTex2d,
        native: ptr::null_mut(),
        memory: ptr::null_mut(),
        view: ptr::null_mut(),
        state: u32::MAX,
        width: texture.width(),
        height: texture.height(),
        native_format: 0,
        mip_levels: 1,
        array_layers: 1,
        gpu_virtual_address: 0,
        flags: 0,
        usage: 0,
        internal_flags: 0,
        reserved: 0,
    };
    unsafe {
        if let Some(t) = texture.as_hal::<wgpu::hal::api::Dx12>() {
            r.native = t.raw_resource().as_raw();
            // `TextureUses::RESOURCE` on DX12: PIXEL | NON_PIXEL_SHADER_RESOURCE.
            r.state = 0x40 | 0x80;
        } else if let Some(t) = texture.as_hal::<wgpu::hal::api::Vulkan>() {
            use ash::vk::Handle;
            r.native = t.raw_handle().as_raw() as *mut c_void;
            if let Some(v) = view.as_hal::<wgpu::hal::api::Vulkan>() {
                r.view = v.raw_handle().as_raw() as *mut c_void;
            }
            // `TextureUses::RESOURCE`: SHADER_READ_ONLY_OPTIMAL, or DEPTH_STENCIL_READ_ONLY_OPTIMAL.
            r.state = if depth { 4 } else { 5 };
            if let Some(a) = adapter.as_hal::<wgpu::hal::api::Vulkan>() {
                r.native_format = a.texture_format_as_raw(texture.format()).as_raw() as u32;
            }
            let u = texture.usage();
            r.usage = [
                (TextureUsages::COPY_SRC, 0x1),
                (TextureUsages::COPY_DST, 0x2),
                (TextureUsages::TEXTURE_BINDING, 0x4),
                (TextureUsages::STORAGE_BINDING, 0x8),
                (
                    TextureUsages::RENDER_ATTACHMENT,
                    if depth { 0x20 } else { 0x10 },
                ),
            ]
            .into_iter()
            .filter(|(f, _)| u.contains(*f))
            .fold(0, |acc, (_, b)| acc | b);
        }
    }
    r
}

/// Just before the render graph: this frame's token, constants and tags for the world view, and
/// DLSS-G's mode when it changed.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn frame_inputs(
    frame: Res<SlFrame>,
    fg: Res<DlssFrameGeneration>,
    frames: Res<DlssgFrames>,
    held: Res<FsrHeld>,
    adapter: Res<RenderAdapter>,
    views: Query<(
        &ExtractedView,
        &ExtractedCamera,
        &FsrState,
        &crate::fsr::FsrView,
        &ViewDepthStencilTexture,
        Option<&DlssHudless>,
    )>,
) {
    let (Some(sl), Some(features)) = (SL.get(), FEATURES.get()) else {
        return;
    };
    let token = token(sl, frame.0);
    RENDER_TOKEN.store(token, Ordering::Release);
    let Some((dlssg_set_options, _)) = features.dlssg else {
        return;
    };
    let viewport = ViewportHandle::zero();
    let world = views.iter().find(|(.., hudless)| hudless.is_some());
    let on = fg.0 && world.is_some();
    // `numFramesToGenerate`: the multiplier less one, which the app clamped to the card's reach
    // (`numFramesToGenerateMax`); 0 here is off.
    let generate = if on { frames.0.max(1) } else { 0 };
    if DLSSG_FRAMES.swap(generate, Ordering::AcqRel) != generate {
        let options = DlssgOptions {
            base: Base::new(
                guid(
                    0xfac5f1cb,
                    0x2dfd,
                    0x4f36,
                    [0xa1, 0xe6, 0x3a, 0x9e, 0x86, 0x52, 0x56, 0xc5],
                ),
                5,
            ),
            mode: if on { eDLSSGModeOn } else { eDLSSGModeOff },
            num_frames_to_generate: generate.max(1),
            flags: 0,
            dynamic_res_width: 0,
            dynamic_res_height: 0,
            num_back_buffers: 0,
            mvec_depth_width: 0,
            mvec_depth_height: 0,
            color_width: 0,
            color_height: 0,
            color_buffer_format: 0,
            mvec_buffer_format: 0,
            depth_buffer_format: 0,
            hud_less_buffer_format: 0,
            ui_buffer_format: 0,
            on_error_callback: ptr::null(),
            b_reserved15: 2, // eInvalid
            queue_parallelism_mode: 0,
            enable_user_interface_recomposition: eFalse,
            dynamic_target_frame_rate: 0.0,
        };
        let result = unsafe { dlssg_set_options(&viewport, &options) };
        if on {
            info!(
                "dlss: DLSS frame generation on, {}x ({result})",
                generate + 1
            );
        } else {
            info!("dlss: DLSS frame generation off ({result})");
        }
    }
    let (true, false, Some((view, _camera, fsr, fsr_view, depth, Some(hudless)))) =
        (on, held.0, world)
    else {
        return;
    };

    let proj = view.clip_from_view;
    let [clip_from_rel, prev_clip_from_rel] = fsr.rel_matrices();
    let clip_to_prev = prev_clip_from_rel * clip_from_rel.inverse();
    let rot = view.world_from_view.rotation();
    let eye = view.world_from_view.translation();
    let v3 = |v: Vec3| Float3 {
        x: v.x,
        y: v.y,
        z: v.z,
    };
    let near = proj.w_axis.z;
    let constants = Constants {
        base: Base::new(
            guid(
                0xdcd35ad7,
                0x4e4a,
                0x4bad,
                [0xa9, 0x0c, 0xe0, 0xc4, 0x9e, 0xb2, 0x3a, 0xfe],
            ),
            2,
        ),
        camera_view_to_clip: Float4x4::from(proj),
        clip_to_camera_view: Float4x4::from(proj.inverse()),
        clip_to_lens_clip: Float4x4::from(Mat4::IDENTITY),
        clip_to_prev_clip: Float4x4::from(clip_to_prev),
        prev_clip_to_clip: Float4x4::from(clip_to_prev.inverse()),
        jitter_offset: Float2 {
            x: fsr_view.jitter.x,
            y: fsr_view.jitter.y,
        },
        // The adapter's vectors are already in uv, Streamline's normalized units.
        mvec_scale: Float2 { x: 1.0, y: 1.0 },
        camera_pinhole_offset: Float2 { x: 0.0, y: 0.0 },
        camera_pos: v3(eye),
        camera_up: v3(rot * Vec3::Y),
        camera_right: v3(rot * Vec3::X),
        camera_fwd: v3(rot * Vec3::NEG_Z),
        camera_near: near,
        // Reverse-Z with an infinite far plane; a far beyond any draw distance.
        camera_far: 100_000.0,
        camera_fov: 2.0 * (1.0 / proj.y_axis.y).atan(),
        camera_aspect_ratio: proj.y_axis.y / proj.x_axis.x,
        motion_vectors_invalid_value: INVALID_FLOAT,
        depth_inverted: eTrue,
        camera_motion_included: eTrue,
        motion_vectors_3d: eFalse,
        reset: if fsr.frame_index() == 0 {
            eTrue
        } else {
            eFalse
        },
        orthographic_projection: eFalse,
        motion_vectors_dilated: eFalse,
        motion_vectors_jittered: eFalse,
        min_relative_linear_depth_object_separation: 40.0,
    };
    unsafe { (sl.set_constants)(&constants, token, &viewport) };

    let render = fsr.render_size();
    let display = fsr.display_size();
    let full = |s: UVec2| Extent {
        top: 0,
        left: 0,
        width: s.x,
        height: s.y,
    };
    let mv_view = fsr.motion_vectors();
    let mut depth_res = describe(
        depth.texture(),
        depth.attachment.depth_stencil_views().attachment_view(),
        &adapter,
        true,
    );
    let mut mv_res = describe(mv_view.texture(), mv_view, &adapter, false);
    let mut hudless_res = describe(&hudless.texture, &hudless.view, &adapter, false);
    let tag = |resource: &mut Resource, kind, extent| ResourceTag {
        base: Base::new(
            guid(
                0x4c6a5aad,
                0xb445,
                0x496c,
                [0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53],
            ),
            1,
        ),
        resource,
        kind,
        lifecycle: eValidUntilPresent,
        extent,
    };
    let tags = [
        tag(&mut depth_res, kBufferTypeDepth, full(render)),
        tag(&mut mv_res, kBufferTypeMotionVectors, full(render)),
        tag(&mut hudless_res, kBufferTypeHUDLessColor, full(display)),
    ];
    // Every tag lives until present, so Streamline copies nothing and needs no command list.
    unsafe {
        (sl.set_tag_for_frame)(
            token,
            &viewport,
            tags.as_ptr(),
            tags.len() as u32,
            ptr::null_mut(),
        )
    };
}

pub(super) fn build(app: &mut App) {
    if SL.get().is_none() {
        return;
    }
    app.init_resource::<SlFrame>()
        .add_plugins(ExtractResourcePlugin::<SlFrame>::default())
        .add_systems(First, simulation_start)
        .add_systems(Last, simulation_end);
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .add_systems(
            Render,
            (
                prepare_hudless.in_set(RenderSystems::PrepareResources),
                frame_inputs
                    .in_set(RenderSystems::Render)
                    .before(render_system),
            ),
        )
        .add_systems(
            Core3d,
            hudless_pass
                .in_set(Core3dSystems::PostProcess)
                .after(crate::ffx_glow::ffx_glow_pass),
        );
}
