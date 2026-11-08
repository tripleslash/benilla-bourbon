//! fork: static_gx's mesh-shader lane, native only: the GPU cull's survivors drawn as meshlets
//! through a task and a mesh stage, the task stage culling each meshlet.

use super::bindless::{GxBindless, GxSlots};
use super::GxCellDraw;
use crate::modern::caps::HwCaps;
use crate::static_gx::meshlet::GxMeshlets;
use bevy::prelude::*;
use bevy::render::mesh::allocator::MeshAllocatorSettings;
use bevy::render::render_resource::{BufferUsages, CommandEncoder};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::RenderApp;
use bevy::shader::Shader;
use std::ops::Range;

/// The lane's shader: `static_gx.wgsl` under `GX_MESHLETS` with the mesh entries appended
/// (`static_gx_mesh.wgsl`), built from the sources at startup, so a hot reload of either file
/// does not reach it.
pub(super) const MESH_SHADER: Handle<Shader> =
    bevy::asset::uuid_handle!("4b7d2c91-e3a5-4f08-b6d1-9c2e5a7f3d18");

/// The composed source: the mesh-shader and draw-index enables must lead the module, and naga
/// refuses either on a device without it, so they are never in `static_gx.wgsl` itself.
const MESH_SOURCE: &str = concat!(
    "enable wgpu_mesh_shader, draw_index;\n",
    "#import benilla_modern::gx_occlusion\n",
    include_str!("../../shaders/static_gx.wgsl"),
    "\n",
    include_str!("../../shaders/modern/static_gx_mesh.wgsl"),
);

/// `WOW_GX_MESHLETS=0` turns the lane off: static_gx draws through the GPU cull's path.
fn switched_on() -> bool {
    crate::launch_env::var("WOW_GX_MESHLETS").as_deref() != Some("0")
}

/// Whether the lane can run this session: the device has mesh shaders as the lane needs them
/// ([`HwCaps::mesh_shader`]), the GPU cull's lane runs on native multi-draw, and the switch is
/// on. What widens group 1 and group 2 to the task and mesh stages and makes the meshlet arena.
pub(super) fn capable(caps: Option<&HwCaps>) -> bool {
    caps.is_some_and(|c| c.mesh_shader)
        && super::batch::gpu_cull_lane(caps)
        && crate::launch_env::var("WOW_STATIC_GX_MDI").as_deref() != Some("0")
        && switched_on()
}

/// A region's meshlet rows in the assembly's item order `order`, and per position `k` of
/// `order` its first meshlet (`starts[order.len()]` is the region's count).
fn arrange(m: &GxMeshlets, order: &[u32]) -> (Vec<[u32; 4]>, Vec<u32>) {
    let mut rows = Vec::with_capacity(m.rows.len());
    let mut starts = Vec::with_capacity(order.len() + 1);
    for &i in order {
        starts.push((rows.len() / 2) as u32);
        let i = i as usize;
        if let (Some(&a), Some(&b)) = (m.items.get(i), m.items.get(i + 1)) {
            rows.extend_from_slice(&m.rows[2 * a as usize..2 * b as usize]);
        }
    }
    starts.push((rows.len() / 2) as u32);
    (rows, starts)
}

/// `assemble_region`'s hook, after `place`: `draw`'s meshlets into the arena in `order` (the
/// index arena's item order), their blocks into `slots`. Returns the per-position meshlet starts
/// [`span`] maps a draw through; empty where the region has none (baked before the lane was
/// wanted, or no arena), and a batch holding such a region draws through the GPU cull's path.
pub(super) fn place(
    shared: &mut GxBindless,
    slots: &mut GxSlots,
    draw: &GxCellDraw,
    order: &[u32],
    device: &RenderDevice,
    queue: &RenderQueue,
    copies: &mut CommandEncoder,
) -> Vec<u32> {
    let Some(m) = draw.meshlets.as_deref() else {
        return Vec::new();
    };
    if m.items.len() != draw.draws.len() + 1 {
        return Vec::new();
    }
    let (rows, starts) = arrange(m, order);
    match shared.place_meshlets(&rows, &m.vertices, &m.triangles, device, queue, copies) {
        Some(placed) => {
            slots.meshlets = placed;
            starts
        }
        None => Vec::new(),
    }
}

/// The meshlets `(first, count)`, relative to the region's block, of a draw over the region's
/// arena indices `first..first + count` (relative to `index_base`): a draw is a run of whole items
/// in `order`, whose ranges (`ranges`, per item) rise along it.
pub(super) fn span(
    ranges: &[Range<u32>],
    order: &[u32],
    starts: &[u32],
    first: u32,
    count: u32,
) -> Option<(u32, u32)> {
    let at = |index: u32| order.partition_point(|&i| ranges[i as usize].start < index);
    let (a, b) = (*starts.get(at(first))?, *starts.get(at(first + count))?);
    Some((a, b.checked_sub(a)?))
}

/// Wire the lane (called from `render::build`, native only).
pub(super) fn build(app: &mut App) {
    if let Some(mut shaders) = app.world_mut().get_resource_mut::<Assets<Shader>>() {
        // A UUID id never fails to insert.
        let _ = shaders.insert(
            MESH_SHADER.id(),
            Shader::from_wgsl(MESH_SOURCE, "benilla_world/shaders/static_gx_mesh.wgsl"),
        );
    }
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    // The mesh stage reads the vertex slabs as storage. The allocator takes its usages when it is
    // made, in its plugin's `finish`, before the device is known, so every slab carries it
    // wherever the switch leaves the lane possible.
    if switched_on() {
        render_app
            .world_mut()
            .get_resource_or_init::<MeshAllocatorSettings>()
            .extra_buffer_usages |= BufferUsages::STORAGE;
    }
}
