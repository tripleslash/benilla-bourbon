//! The velocity target: moving content's own motion, which the upscaler's input adapter
//! (`shaders/fsr/motion.wgsl`) takes over its depth reprojection wherever it is written.
//!
//! `xy` is the uv motion, previous minus current and jitter-free; `w` is the "written" flag. The
//! target is cleared every rendered world frame, after the transparent pass and the lamp halos,
//! and the moving draws depth-test read-only against the view's depth.

use bevy::prelude::*;
use bevy::render::render_resource::{
    LoadOp, Operations, RenderPassColorAttachment, RenderPassDescriptor, StoreOp,
};
use bevy::render::renderer::{RenderContext, ViewQuery};

use super::{FsrHeld, FsrVelocityTarget};

/// `MainPass`, last: clear the velocity target to "nothing written".
pub(super) fn fsr_velocity_pass(
    view: ViewQuery<&FsrVelocityTarget>,
    held: Res<FsrHeld>,
    mut render_context: RenderContext,
) {
    if held.0 {
        return;
    }
    let target = view.into_inner();
    render_context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("fsr_velocity"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Default::default()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
}
