//! fork: static_gx's mesh-shader lane, native only: the GPU cull's survivors drawn as meshlets
//! through a task and a mesh stage, the task stage culling each meshlet.

use bevy::prelude::*;
use bevy::render::mesh::allocator::MeshAllocatorSettings;
use bevy::render::render_resource::BufferUsages;
use bevy::render::RenderApp;
use bevy::shader::Shader;

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
