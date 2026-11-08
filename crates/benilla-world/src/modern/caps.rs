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

use bevy::prelude::*;
use bevy::render::render_resource::WgpuFeatures;
use bevy::render::renderer::{RenderAdapterInfo, RenderDevice};
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
    /// A compute queue of its own. False until the async compute lane reads it from here
    /// (`async_compute.rs` still probes the device itself).
    pub async_compute: bool,
}

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
            immediates: features.contains(WgpuFeatures::IMMEDIATES) && limits.max_immediate_size > 0,
            ray_query: features.contains(WgpuFeatures::EXPERIMENTAL_RAY_QUERY),
            conservative_raster: features.contains(WgpuFeatures::CONSERVATIVE_RASTERIZATION),
            async_compute: false,
        }
    }
}

/// `RenderStartup`: read the caps and log them. Lanes that build on them order after this.
pub(crate) fn init(
    mut commands: Commands,
    device: Res<RenderDevice>,
    adapter: Res<RenderAdapterInfo>,
) {
    let caps = HwCaps::from_parts(device.features(), &device.limits());
    info!(
        "caps: binding_arrays={} mdi_count={} immediates={} ray_query={} conservative_raster={} \
         async_compute={} ({} on {:?})",
        caps.binding_arrays,
        caps.mdi_count,
        caps.immediates,
        caps.ray_query,
        caps.conservative_raster,
        caps.async_compute,
        adapter.0.name,
        adapter.0.backend,
    );
    commands.insert_resource(caps);
}

pub(super) fn plugin(app: &mut App) {
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(RenderStartup, init);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No features, nothing claimed; binding arrays need all three features and the element floor.
    #[test]
    fn a_capability_needs_every_feature_it_names() {
        let mut limits = WgpuLimits::default();
        assert_eq!(
            HwCaps::from_parts(WgpuFeatures::empty(), &limits),
            HwCaps::default()
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
}
