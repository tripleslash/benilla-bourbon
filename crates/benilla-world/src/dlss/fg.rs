//! fork: dlss (FORK.md) — DLSS Frame Generation through NGX, inside FSR frame generation's
//! cadence (`fsr::cadence`, `fsr::framegen`). The cadence, the `WorldFinal` pair and the present
//! are FSR's; only the generator differs. On each rendered world frame the compose writes the
//! finished world without UI into `WorldFinal[p]`, and [`dlss_framegen_pass`] hands NGX that
//! image, the render-size depth and the motion adapter's vectors: NGX keeps the previous
//! backbuffer itself and writes the frame halfway between, which the UI camera grounds on first;
//! the held frame after it grounds on `WorldFinal[p]`. So the pair costs one frame of latency, as
//! FSR's does, and the UI is drawn crisp over both.
//!
//! The feature (`DlssFgView`, on the world view) is created at the view's display and render sizes
//! in the backbuffer's linear format, and recreated when one moves. A creation NGX refuses is
//! logged once and frame generation falls back to FSR's chain for the session
//! ([`super::frame_generation_refused`]).

use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::{ExtractedView, ViewDepthTexture};
use dlss_wgpu::frame_generation::{
    DlssFrameGeneration, DlssFrameGenerationCamera, DlssFrameGenerationRenderParameters,
};

use super::sr::DlssSdkRes;
use crate::fsr::{FgSolo, FgState, FsrHeld, FsrState};

/// One world view's DLSS-G feature, the sizes and format it was created for, and whether the
/// last rendered world frame was evaluated (NGX's previous backbuffer is that frame's).
pub(crate) struct ViewFg {
    context: DlssFrameGeneration,
    display: UVec2,
    render: UVec2,
    format: wgpu::TextureFormat,
    evaluated_last: bool,
}

/// `PrepareResources`, after `prepare_framegen`: a feature on each view whose frame generation
/// DLSS-G makes ([`FgState::dlss_output`]), none elsewhere.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_dlss_fg(
    mut commands: Commands,
    sdk: Option<Res<DlssSdkRes>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    adapter: Res<RenderAdapter>,
    held: Res<FsrHeld>,
    mut views: Query<(Entity, &FgState, &FsrState, Option<&mut super::DlssFgView>)>,
    mut refused: Local<bool>,
) {
    if held.0 {
        return;
    }
    for (entity, fg, fsr, existing) in &mut views {
        let (Some(output), Some(sdk), false) = (fg.dlss_output(), sdk.as_deref(), *refused) else {
            if existing.is_some() {
                commands.entity(entity).remove::<super::DlssFgView>();
            }
            continue;
        };
        let (display, render) = (fsr.display_size(), fsr.render_size());
        let format = output.texture().format();
        if existing.is_some_and(|e| {
            e.inner.display == display && e.inner.render == render && e.inner.format == format
        }) {
            continue;
        }
        // Drop the old feature before NGX creates the new one.
        commands.entity(entity).remove::<super::DlssFgView>();
        match DlssFrameGeneration::new(
            display.to_array(),
            render.to_array(),
            format,
            sdk.0.clone(),
            device.wgpu_device(),
            &queue,
            &adapter,
        ) {
            Ok(context) => {
                // `display` would name `tracing::field::display` inside the macro.
                let out = display;
                info!("dlss: frame generation at {render} → {out}, {format:?}");
                commands.entity(entity).insert(super::DlssFgView {
                    inner: ViewFg {
                        context,
                        display,
                        render,
                        format,
                        evaluated_last: false,
                    },
                });
            }
            Err(e) => {
                warn!(
                    "dlss: frame generation creation failed ({e}); FSR frame generation stands in"
                );
                *refused = true;
                super::refuse_frame_generation();
            }
        }
    }
}

/// `PostProcess`, after `fsr_framegen_pass` composed `WorldFinal[p]`: the evaluate, in its own
/// command buffer queued right after everything recorded so far (`dlss_wgpu`'s contract). A solo
/// world frame (no held frame follows) presents the real frame and evaluates nothing; the next
/// evaluate then resets, its previous backbuffer being stale.
#[allow(clippy::type_complexity)]
pub(crate) fn dlss_framegen_pass(
    view: ViewQuery<(
        &FgState,
        &FsrState,
        &ExtractedView,
        &ViewDepthTexture,
        &mut super::DlssFgView,
    )>,
    held: Res<FsrHeld>,
    solo: Res<FgSolo>,
    adapter: Res<RenderAdapter>,
    mut render_context: RenderContext,
) {
    if held.0 {
        return;
    }
    let (fg, fsr, view, depth, mut dlss) = view.into_inner();
    let d = &mut dlss.inner;
    let Some(output) = fg.dlss_output() else {
        return;
    };
    if !fg.composed() || solo.0 {
        d.evaluated_last = false;
        return;
    }
    let params = DlssFrameGenerationRenderParameters {
        backbuffer: fg.current_final(),
        depth: depth.view(),
        motion_vectors: fsr.motion_vectors(),
        output,
        reset: !d.evaluated_last || fg.frame_index_since_reset() <= 1 || fsr.frame_index() == 0,
        jitter_offset: fsr.jitter().to_array(),
        // The adapter's vectors are uv: already DLSS-G's normalized units.
        motion_vector_scale: [1.0, 1.0],
        depth_inverted: true,
        // `WorldFinal` holds the UI target's display-encoded values.
        hdr: false,
        camera: camera(view, fsr),
    };
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(render_context.command_encoder(), "dlss_framegen");
    let result = d
        .context
        .render(params, render_context.command_encoder(), &adapter);
    match result {
        Ok(buffer) => {
            render_context.add_command_buffer(buffer);
            fg.mark_generated();
            d.evaluated_last = true;
        }
        Err(e) => {
            warn_once!("dlss: frame generation evaluate failed ({e})");
            d.evaluated_last = false;
        }
    }
    span.end(render_context.command_encoder());
}

/// glam's column-major memory is NGX's row-major matrix for a row vector (`v * M`).
fn rows(m: Mat4) -> [[f32; 4]; 4] {
    m.to_cols_array_2d()
}

/// The unjittered camera this frame rendered with: its projection, the camera-relative
/// current-to-previous clip transform the motion adapter also uses, and the eye's frame.
fn camera(view: &ExtractedView, fsr: &FsrState) -> DlssFrameGenerationCamera {
    let proj = view.clip_from_view;
    let [clip_from_rel, prev_clip_from_rel] = fsr.rel_matrices();
    let clip_to_prev = prev_clip_from_rel * clip_from_rel.inverse();
    let rot = view.world_from_view.rotation();
    DlssFrameGenerationCamera {
        view_to_clip: rows(proj),
        clip_to_view: rows(proj.inverse()),
        clip_to_prev_clip: rows(clip_to_prev),
        prev_clip_to_clip: rows(clip_to_prev.inverse()),
        position: view.world_from_view.translation().to_array(),
        up: (rot * Vec3::Y).to_array(),
        right: (rot * Vec3::X).to_array(),
        forward: (rot * Vec3::NEG_Z).to_array(),
        // Bevy's reverse-Z infinite projection keeps the near plane in `w_axis.z`.
        near: proj.w_axis.z,
        // No far plane; one beyond any draw distance.
        far: 100_000.0,
        fov_y: 2.0 * (1.0 / proj.y_axis.y).atan(),
        aspect_ratio: proj.y_axis.y / proj.x_axis.x,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NGX's row `r` is glam's column `r`: the translation sits in the last row.
    #[test]
    fn rows_put_the_translation_in_the_last_row() {
        let m = rows(Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(m[3], [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(m[0], [1.0, 0.0, 0.0, 0.0]);
    }
}
