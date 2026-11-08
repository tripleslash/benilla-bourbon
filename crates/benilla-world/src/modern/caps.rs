//! fork: what the device can do for the native-only lanes, read once at render startup.
//!
//! [`HwCaps`] is the one place a lane asks whether a feature is there; it never asks
//! `RenderDevice::features()` itself. It is read in `RenderStartup` from the device's features and
//! limits and inserted into the render world, and the startup log carries one `caps:` line with
//! every field and the adapter it was read from.
//!
//! Every field is false on wasm32: the browser has none of these, and wgpu emulates a multi-draw
//! there as a loop of validated calls, dearer than the direct draws it replaces.
//!
//! A field is a capability, not a decision: a lane combines it with its own lever (static_gx's
//! bindless layout needs `binding_arrays && mdi_count` and honours `WOW_GX_BINDLESS=0`).
//!
//! [`HwCaps::mesh_shader`] is read from the features and limits, then held to the backends whose
//! shader translation emits task and mesh stages: naga 29 writes them to SPIR-V only (its HLSL
//! backend lists no `MESH_SHADER` capability), so DX12 and Metal never claim it.
//!
//! [`HwCaps::multiview_views`] and [`HwCaps::multiview_selective`] are what the multiview shadow
//! cascades need (`static_gx/render/modern_shadow/multiview.rs`): wgpu's `MULTIVIEW` with its view
//! limit, and `SELECTIVE_MULTIVIEW` for a mask that skips a layer. On Vulkan and DX12 a vertex
//! stage's `@builtin(view_index)` is the array layer, the mask's bit index (wgpu-hal DX12 lists a
//! view instance location for every layer up to the mask's highest bit).
//!
//! [`alloc_floor`] is the native allocation tier: the floor a native lane's self-growing buffer
//! opens at, sized against [`HwCaps::buffer_bytes`] instead of the floor the lane began with.

use bevy::prelude::*;
use bevy::render::render_resource::WgpuFeatures;
use bevy::render::renderer::{RenderAdapterInfo, RenderAsyncComputeQueue, RenderDevice};
use bevy::render::settings::WgpuLimits;
use bevy::render::{RenderApp, RenderStartup};

/// The binding-array elements per shader stage [`HwCaps::binding_arrays`] promises: static_gx's
/// texture-class array (1024, `static_gx/render/bindless.rs`) with room beside it for bevy's own view
/// arrays in the same fragment stage. Every device that has binding arrays reports far more
/// (wgpu's floor is 500 000).
pub const BINDING_ARRAY_ELEMENTS: u32 = 2048;

/// The device's native-only capabilities. Default is all false: what the browser has.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HwCaps {
    /// Texture and buffer binding arrays, indexed non-uniformly, with at least
    /// [`BINDING_ARRAY_ELEMENTS`] elements per stage.
    pub binding_arrays: bool,
    /// `multi_draw_indexed_indirect_count` in hardware, with each draw's own `first_instance`
    /// (`INDIRECT_FIRST_INSTANCE`). wgpu 29 has no plain multi-draw feature: without this it loops.
    pub mdi_count: bool,
    /// Immediate data (push constants) of some size.
    pub immediates: bool,
    /// Ray queries and acceleration structures (`EXPERIMENTAL_RAY_QUERY`; a buffer may carry
    /// `BLAS_INPUT`).
    pub ray_query: bool,
    /// Conservative rasterization.
    pub conservative_raster: bool,
    /// A compute queue of its own: `async_compute::init` made one (not on wasm32, not under
    /// `WOW_ASYNC_COMPUTE=0` or `WOW_GPU_STATS`, and only where the backend has a second queue).
    /// No feature bit says so, so [`Self::from_parts`] leaves it false and [`init`] reads the queue.
    pub async_compute: bool,
    /// The largest storage-buffer binding the device takes, and no larger than its largest
    /// buffer, in bytes. 0 on wasm32.
    pub buffer_bytes: u64,
    /// Task and mesh shaders that static_gx's meshlet lane can compile and draw from WGSL
    /// (`static_gx/render/meshlets.rs`): `EXPERIMENTAL_MESH_SHADER`, the task stage's
    /// `draw_index` (`SHADER_DRAW_INDEX`), the counted multi-draw and immediates, the output,
    /// workgroup and payload limits of the meshlet format (`static_gx::meshlet`), five bind
    /// groups, [`MESH_STORAGE_BUFFERS`] storage buffers a stage, and a backend naga writes mesh
    /// stages for (Vulkan; `init` clears it elsewhere).
    pub mesh_shader: bool,
    /// The most views a multiview render pass may span (`Limits::max_multiview_view_count`),
    /// 0 without `MULTIVIEW`. A pass over array layers `0..n` with the mask `2^n - 1` needs `n`.
    pub multiview_views: u32,
    /// `MULTIVIEW` and `SELECTIVE_MULTIVIEW`: a multiview mask may leave layers of the attachment
    /// out (Vulkan and DX12; wgpu does not expose it on Metal).
    pub multiview_selective: bool,
}

/// Storage buffers per shader stage the meshlet lane binds at most: the mesh-variant GPU cull's
/// eleven; the task stage's own group beside the region table.
pub const MESH_STORAGE_BUFFERS: u32 = 16;

/// Bind groups a mesh pipeline of the meshlet lane takes: the view, the region table, the light,
/// the ray-traced scene (colour, where it is declared) and its own.
pub const MESH_BIND_GROUPS: u32 = 5;

impl HwCaps {
    /// The capabilities in `features` and `limits`.
    pub fn from_parts(features: WgpuFeatures, limits: &WgpuLimits) -> Self {
        if cfg!(target_arch = "wasm32") {
            return Self::default();
        }
        let arrays = WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::BUFFER_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        Self {
            binding_arrays: features.contains(arrays)
                && limits.max_binding_array_elements_per_shader_stage >= BINDING_ARRAY_ELEMENTS,
            mdi_count: features.contains(
                WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT | WgpuFeatures::INDIRECT_FIRST_INSTANCE,
            ),
            immediates: features.contains(WgpuFeatures::IMMEDIATES)
                && limits.max_immediate_size > 0,
            ray_query: features.contains(WgpuFeatures::EXPERIMENTAL_RAY_QUERY),
            conservative_raster: features.contains(WgpuFeatures::CONSERVATIVE_RASTERIZATION),
            async_compute: false,
            buffer_bytes: limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
            mesh_shader: mesh_shader_capable(features, limits),
            multiview_views: if features.contains(WgpuFeatures::MULTIVIEW) {
                limits.max_multiview_view_count
            } else {
                0
            },
            multiview_selective: features
                .contains(WgpuFeatures::MULTIVIEW | WgpuFeatures::SELECTIVE_MULTIVIEW),
        }
    }
}

/// Whether `features` and `limits` hold everything [`HwCaps::mesh_shader`] names, the backend
/// aside.
#[cfg(not(target_arch = "wasm32"))]
fn mesh_shader_capable(features: WgpuFeatures, limits: &WgpuLimits) -> bool {
    use crate::static_gx::meshlet::{
        MAX_DRAW_TASKS, MESHLET_TRIANGLES, MESHLET_VERTICES, MESH_THREADS, TASK_MESHLETS,
        TASK_PAYLOAD_BYTES,
    };
    let needed = WgpuFeatures::EXPERIMENTAL_MESH_SHADER
        | WgpuFeatures::SHADER_DRAW_INDEX
        | WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT
        | WgpuFeatures::INDIRECT_FIRST_INSTANCE
        | WgpuFeatures::IMMEDIATES;
    features.contains(needed)
        && limits.max_immediate_size >= 8
        && limits.max_mesh_output_vertices >= MESHLET_VERTICES
        && limits.max_mesh_output_primitives >= MESHLET_TRIANGLES
        && limits.max_mesh_invocations_per_workgroup >= MESH_THREADS
        && limits.max_mesh_invocations_per_dimension >= MESH_THREADS
        && limits.max_task_invocations_per_workgroup >= TASK_MESHLETS
        && limits.max_task_invocations_per_dimension >= TASK_MESHLETS
        && limits.max_task_payload_size >= TASK_PAYLOAD_BYTES
        && limits.max_task_mesh_workgroups_per_dimension >= MAX_DRAW_TASKS
        && limits.max_task_mesh_workgroup_total_count >= MAX_DRAW_TASKS
        && limits.max_bind_groups >= MESH_BIND_GROUPS
        && limits.max_storage_buffers_per_shader_stage >= MESH_STORAGE_BUFFERS
}

#[cfg(target_arch = "wasm32")]
fn mesh_shader_capable(_: WgpuFeatures, _: &WgpuLimits) -> bool {
    false
}

/// Whether naga 29 writes task and mesh stages for `backend`'s shader language: SPIR-V alone.
fn emits_mesh_stages(backend: wgpu::Backend) -> bool {
    backend == wgpu::Backend::Vulkan
}

/// The floor, in units of `unit` bytes, a native lane's self-growing buffer opens at: `native`
/// where the device reported its limits ([`HwCaps::buffer_bytes`], never on wasm32) and
/// `WOW_ALLOC_TIER=0` did not turn the tier off, else `web`, the lane's own floor. A buffer still
/// grows past its floor; a larger one spares the first streaming burst the frames each growth
/// costs (a frame of direct draws, or a submit and a copy in the middle of prepare).
pub fn alloc_floor(caps: Option<&HwCaps>, web: u32, native: u32, unit: u64) -> u32 {
    let on = crate::launch_env::var("WOW_ALLOC_TIER").as_deref() != Some("0");
    floor_within(caps.map_or(0, |c| c.buffer_bytes), on, web, native, unit)
}

/// `native` held to an eighth of a `bytes` binding, never under `web`; `web` where `bytes` is 0
/// or the tier is off.
fn floor_within(bytes: u64, on: bool, web: u32, native: u32, unit: u64) -> u32 {
    if bytes == 0 || !on {
        return web;
    }
    let most = u32::try_from(bytes / 8 / unit.max(1)).unwrap_or(u32::MAX);
    native.min(most).max(web)
}

/// `RenderStartup`: read the caps and log them. Lanes that build on them order after this.
pub(crate) fn init(
    mut commands: Commands,
    device: Res<RenderDevice>,
    adapter: Res<RenderAdapterInfo>,
    async_queue: Option<Res<RenderAsyncComputeQueue>>,
) {
    let mut caps = HwCaps::from_parts(device.features(), &device.limits());
    caps.mesh_shader &= emits_mesh_stages(adapter.0.backend);
    // The queue `async_compute::init` made, which runs first (`plugin`).
    caps.async_compute = async_queue.is_some();
    info!(
        "caps: binding_arrays={} mdi_count={} immediates={} ray_query={} conservative_raster={} \
         async_compute={} mesh_shader={} multiview_views={} multiview_selective={} \
         ({} on {:?})",
        caps.binding_arrays,
        caps.mdi_count,
        caps.immediates,
        caps.ray_query,
        caps.conservative_raster,
        caps.async_compute,
        caps.mesh_shader,
        caps.multiview_views,
        caps.multiview_selective,
        adapter.0.name,
        adapter.0.backend,
    );
    commands.insert_resource(caps);
}

pub(super) fn plugin(app: &mut App) {
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(RenderStartup, init.after(super::async_compute::init));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No features, nothing claimed; binding arrays need all three features and the element floor.
    #[test]
    fn a_capability_needs_every_feature_it_names() {
        let mut limits = WgpuLimits::default();
        let buffer_bytes = if cfg!(target_arch = "wasm32") {
            0
        } else {
            limits.max_storage_buffer_binding_size
        };
        assert_eq!(
            HwCaps::from_parts(WgpuFeatures::empty(), &limits),
            HwCaps {
                buffer_bytes,
                ..HwCaps::default()
            }
        );
        let arrays = WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::BUFFER_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        limits.max_binding_array_elements_per_shader_stage = BINDING_ARRAY_ELEMENTS - 1;
        assert!(!HwCaps::from_parts(arrays, &limits).binding_arrays);
        limits.max_binding_array_elements_per_shader_stage = BINDING_ARRAY_ELEMENTS;
        let two = WgpuFeatures::TEXTURE_BINDING_ARRAY | WgpuFeatures::BUFFER_BINDING_ARRAY;
        assert!(!HwCaps::from_parts(two, &limits).binding_arrays);
        assert_eq!(
            HwCaps::from_parts(arrays, &limits).binding_arrays,
            !cfg!(target_arch = "wasm32")
        );
        let count_only = WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT;
        assert!(!HwCaps::from_parts(count_only, &limits).mdi_count);
    }

    /// Multiview counts views only with the feature; a selective mask needs both features.
    #[test]
    fn multiview_needs_its_feature() {
        let limits = WgpuLimits {
            max_multiview_view_count: 4,
            ..WgpuLimits::default()
        };
        let none = HwCaps::from_parts(WgpuFeatures::empty(), &limits);
        assert_eq!(none.multiview_views, 0);
        assert!(!none.multiview_selective);
        let plain = HwCaps::from_parts(WgpuFeatures::MULTIVIEW, &limits);
        let both = WgpuFeatures::MULTIVIEW | WgpuFeatures::SELECTIVE_MULTIVIEW;
        let selective = HwCaps::from_parts(both, &limits);
        if cfg!(target_arch = "wasm32") {
            assert_eq!(plain, HwCaps::default());
        } else {
            assert_eq!(plain.multiview_views, 4);
            assert!(!plain.multiview_selective);
            assert_eq!(selective.multiview_views, 4);
            assert!(selective.multiview_selective);
        }
    }

    /// The mesh-shader capability needs the feature set and every limit of the meshlet format;
    /// only Vulkan's translation emits the stages.
    #[test]
    fn mesh_shaders_need_the_features_the_limits_and_spirv() {
        let features = WgpuFeatures::EXPERIMENTAL_MESH_SHADER
            | WgpuFeatures::SHADER_DRAW_INDEX
            | WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT
            | WgpuFeatures::INDIRECT_FIRST_INSTANCE
            | WgpuFeatures::IMMEDIATES;
        let mut limits = WgpuLimits::default().using_recommended_minimum_mesh_shader_values();
        limits.max_immediate_size = 128;
        limits.max_storage_buffers_per_shader_stage = MESH_STORAGE_BUFFERS;
        limits.max_bind_groups = MESH_BIND_GROUPS;
        // wgpu's recommended minimum dispatch (256 a dimension, 1024 in all) is under the lane's.
        assert!(!HwCaps::from_parts(features, &limits).mesh_shader);
        limits.max_task_mesh_workgroups_per_dimension = 1 << 22;
        limits.max_task_mesh_workgroup_total_count = 1 << 22;
        assert_eq!(
            HwCaps::from_parts(features, &limits).mesh_shader,
            !cfg!(target_arch = "wasm32")
        );
        let no_draw_index = features - WgpuFeatures::SHADER_DRAW_INDEX;
        assert!(!HwCaps::from_parts(no_draw_index, &limits).mesh_shader);
        limits.max_mesh_output_primitives = 64;
        assert!(!HwCaps::from_parts(features, &limits).mesh_shader);
        assert!(emits_mesh_stages(wgpu::Backend::Vulkan));
        assert!(!emits_mesh_stages(wgpu::Backend::Dx12));
        assert!(!emits_mesh_stages(wgpu::Backend::Metal));
    }

    /// The native floor holds under an eighth of the binding, never under the web floor, and the
    /// web floor stands where the limits are unknown or the tier is off.
    #[test]
    fn a_native_floor_fits_the_binding() {
        assert_eq!(floor_within(0, true, 8, 64, 4), 8);
        assert_eq!(floor_within(1 << 30, false, 8, 64, 4), 8);
        assert_eq!(floor_within(1 << 30, true, 8, 64, 4), 64);
        assert_eq!(floor_within(8 * 4 * 16, true, 8, 64, 4), 16);
        assert_eq!(floor_within(8 * 4 * 4, true, 8, 64, 4), 8);
    }
}
