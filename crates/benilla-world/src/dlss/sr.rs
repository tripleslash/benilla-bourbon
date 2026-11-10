//! fork: dlss (FORK.md) — DLSS 4.5 Super Resolution in the FSR chain's slot. One NGX instance
//! ([`DlssSdkRes`]) on the render device; per world view a feature (`ViewDlss`, kept in
//! `FsrState`), created at the view's exact render and display sizes and recreated when either, or
//! the lighting model's HDR-ness, moves. `encode` is the evaluate `fsr_upscale_pass` calls in
//! place of FSR's passes 1–7.
//!
//! The inputs, all FSR's: the main texture (jittered, render size), the view depth (reverse-Z,
//! `InvertedDepth`), the `motion` adapter's vectors (uv, previous − current, jitter-free, at render
//! size: `LowResolutionMotionVectors`, scaled to pixels by the render size), FSR's jitter (DLSS and
//! FSR share the convention: pixel `p` sampled `p + 0.5 − jitter`), and the exposure the modern
//! lighting's meter drives, as the 1×1 texture DLSS reads (Programming Guide 3.9). The modern frame
//! is HDR (`HighDynamicRange`); a classic frame is display bytes and is not.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::ExtractedView;
use bevy::render::RenderApp;
use dlss_wgpu::super_resolution::{
    DlssSuperResolution, DlssSuperResolutionExposure, DlssSuperResolutionRenderParameters,
};
use dlss_wgpu::{DlssFeatureFlags, DlssPerfQualityMode, DlssSdk};

use super::{DlssRuntime, DlssSupport, PROJECT_ID};
use crate::fsr::{FsrState, FsrView, UpscalerKind};

fn project_id() -> uuid::Uuid {
    uuid::Uuid::parse_str(PROJECT_ID).expect("PROJECT_ID is a GUID")
}

/// The one NGX instance, on the render device.
#[derive(Resource, Clone)]
pub struct DlssSdkRes(pub Arc<Mutex<DlssSdk>>);

/// Vulkan only, before `RenderPlugin`: NGX's instance and device extensions through
/// `bevy_render`'s `raw_vulkan_init`, which the vendored fork runs only on a Vulkan-only boot.
pub(super) fn register_vulkan_extensions(app: &mut App) {
    use bevy::render::renderer::raw_vulkan_init::RawVulkanInitSettings;
    let id = project_id();
    let mut settings = app
        .world_mut()
        .get_resource_or_init::<RawVulkanInitSettings>();
    // SAFETY: the callbacks only add the extensions NGX asks for, and only when the instance or
    // device supports them (`register_*_extensions` check before extending).
    unsafe {
        settings.add_create_instance_callback(move |args, _features| {
            let mut support = dlss_wgpu::FeatureSupport::default();
            let _ = dlss_wgpu::register_instance_extensions(id, args, &mut support);
        });
        settings.add_create_device_callback(move |args, adapter, _features| {
            let mut support = dlss_wgpu::FeatureSupport::default();
            let _ = dlss_wgpu::register_device_extensions(id, args, adapter, &mut support);
        });
    }
}

/// `DlssPlugin::finish`: open NGX on the render device, or say in one line why DLSS will not run.
pub(super) fn init_sdk(app: &mut App) {
    let runtime = app.world().get_resource::<DlssRuntime>().cloned();
    let Some(render_app) = app.get_sub_app(RenderApp) else {
        return;
    };
    let Some(device) = render_app.world().get_resource::<RenderDevice>() else {
        return;
    };
    let device = device.wgpu_device().clone();
    let outcome = match runtime.as_ref() {
        _ if !dlss_wgpu::NGX_LINKED => Err("built without the DLSS SDK".to_string()),
        None => Err("no state folder to look for the runtime in".to_string()),
        Some(r) if r.super_resolution_library().is_none() => Err(format!(
            "no DLSS runtime (nvngx_dlss) in {}",
            r.dir.display()
        )),
        Some(_) => DlssSdk::new(project_id(), device).map_err(|e| e.to_string()),
    };
    // Streamline's half (frame generation, its multiplier, Reflex), set just before.
    let prior = app
        .world()
        .get_resource::<DlssSupport>()
        .copied()
        .unwrap_or_default();
    let fg = prior.frame_generation;
    match outcome {
        Ok(sdk) => {
            let (backend, rr) = {
                let locked = sdk.lock().unwrap();
                (locked.backend(), locked.ray_reconstruction_available())
            };
            info!(
                "dlss: DLSS 4.5 super resolution on {backend:?}{}",
                if fg {
                    ", frame generation through Streamline"
                } else {
                    ""
                }
            );
            let support = DlssSupport {
                super_resolution: true,
                ray_reconstruction: rr,
                ..prior
            };
            app.insert_resource(support);
            let render_app = app.sub_app_mut(RenderApp);
            render_app.insert_resource(DlssSdkRes(sdk));
            render_app.insert_resource(support);
        }
        Err(why) => {
            info!("dlss: unavailable ({why}); the upscaler falls back to FSR");
        }
    }
}

/// One view's DLSS feature and the sizes and flags it was created for.
pub struct ViewDlss {
    context: DlssSuperResolution,
    render: UVec2,
    display: UVec2,
    hdr: bool,
    exposure: Texture,
    exposure_view: TextureView,
}

/// `PrepareResources`, after `prepare_fsr`: give each DLSS view its feature, and write this
/// frame's exposure. A creation NGX refuses is logged once and the view runs FSR instead.
pub(crate) fn prepare_dlss(
    sdk: Option<Res<DlssSdkRes>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    held: Res<crate::fsr::FsrHeld>,
    mut views: Query<(&FsrView, &mut FsrState, &ExtractedView)>,
    mut refused: Local<bool>,
) {
    if held.0 {
        return;
    }
    for (fsr, mut state, _view) in &mut views {
        let want = fsr.upscaler == UpscalerKind::Dlss && sdk.is_some() && !*refused;
        if !want {
            if state.dlss.is_some() {
                state.dlss = None;
            }
            continue;
        }
        let Some(sdk) = sdk.as_deref() else { continue };
        let (render, display) = (state.render_size(), state.display_size());
        let hdr = state.is_modern();
        let keep = state
            .dlss
            .as_ref()
            .and_then(super::DlssViewState::inner)
            .is_some_and(|d| d.render == render && d.display == display && d.hdr == hdr);
        if !keep {
            // Drop the old feature before NGX creates the new one.
            state.dlss = None;
            let mut flags =
                DlssFeatureFlags::LowResolutionMotionVectors | DlssFeatureFlags::InvertedDepth;
            if hdr {
                flags |= DlssFeatureFlags::HighDynamicRange;
            }
            let mode = DlssPerfQualityMode::for_scale(render.x as f32 / display.x.max(1) as f32);
            match DlssSuperResolution::new(
                display.to_array(),
                Some(render.to_array()),
                mode,
                flags,
                sdk.0.clone(),
                device.wgpu_device(),
                &queue,
            ) {
                Ok(context) => {
                    let exposure = device.create_texture(&TextureDescriptor {
                        label: Some("dlss_exposure"),
                        size: Extent3d::default(),
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: TextureFormat::R32Float,
                        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    let exposure_view = exposure.create_view(&TextureViewDescriptor::default());
                    let shown = display;
                    info!("dlss: {mode:?} at {render} → {shown}, HDR {hdr}");
                    state.dlss = Some(super::DlssViewState::new(ViewDlss {
                        context,
                        render,
                        display,
                        hdr,
                        exposure,
                        exposure_view,
                    }));
                }
                Err(e) => {
                    warn!("dlss: feature creation failed ({e}); this view runs FSR");
                    *refused = true;
                    super::refuse();
                    continue;
                }
            }
        }
        // The exposure the FSR chain also takes: the modern look's, 1 on a classic frame.
        let exposure = if hdr {
            crate::modern::look::exposure().max(1e-6)
        } else {
            1.0
        };
        if let Some(d) = state.dlss.as_ref().and_then(super::DlssViewState::inner) {
            queue.write_texture(
                d.exposure.as_image_copy(),
                bytemuck::bytes_of(&exposure),
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: None,
                },
                Extent3d::default(),
            );
        }
    }
}

/// What one evaluate reads and writes.
pub(crate) struct Inputs<'a> {
    pub color: &'a TextureView,
    pub depth: &'a TextureView,
    pub motion: &'a TextureView,
    pub output: &'a TextureView,
    pub jitter: Vec2,
    pub reset: bool,
}

/// The evaluate, in its own command buffer queued right after everything recorded so far
/// (`dlss_wgpu`'s contract). `false` if NGX refused it; the caller then leaves the output unwritten.
pub(crate) fn encode(
    dlss: &mut super::DlssViewState,
    adapter: &RenderAdapter,
    render_context: &mut RenderContext,
    inputs: Inputs,
) -> bool {
    let Some(d) = dlss.inner_mut() else {
        return false;
    };
    let render = d.render;
    let params = DlssSuperResolutionRenderParameters {
        color: inputs.color,
        depth: inputs.depth,
        motion_vectors: inputs.motion,
        exposure: DlssSuperResolutionExposure::Manual {
            exposure: &d.exposure_view,
            exposure_scale: None,
            pre_exposure: None,
        },
        bias: None,
        dlss_output: inputs.output,
        reset: inputs.reset,
        jitter_offset: inputs.jitter.to_array(),
        partial_texture_size: Some(render.to_array()),
        // The adapter's vectors are uv; DLSS wants render pixels.
        motion_vector_scale: Some(render.as_vec2().to_array()),
    };
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    use bevy::render::diagnostic::RecordDiagnostics;
    let span = diagnostics.time_span(render_context.command_encoder(), "dlss_upscale");
    let result = d
        .context
        .render(params, render_context.command_encoder(), adapter);
    let ok = match result {
        Ok(buffer) => {
            render_context.add_command_buffer(buffer);
            true
        }
        Err(e) => {
            warn_once!("dlss: evaluate failed ({e})");
            false
        }
    };
    span.end(render_context.command_encoder());
    ok
}
