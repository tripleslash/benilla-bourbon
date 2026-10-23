//! The celestial and star [`ExtendedMaterial`]s, drawn by `celestial.wgsl` and `star.wgsl`.

use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::sky_order::{sky_pipeline_state, SKY_VERTEX_SHADER};

/// Every disc and glare, blended in gamma space like the reference. A disc takes the horizon clip
/// and fade (`0x6d1960`) under a premultiplied blend; a glare skips it and adds gamma bytes, the
/// reference's SRC_ALPHA, ONE flare blend (`0x7e5a16`), which saturates the sun's core to white.
pub type CelestialMaterial = ExtendedMaterial<StandardMaterial, CelestialExt>;

/// The shader's controls. `fade`: `.x` the horizon ramp slope in sin-elevation
/// ([`DISC_HORIZON_FADE`]); `.y` a brightness on RGB, 1.0; `.z` 0 for a disc, 1 for a glare,
/// which never routes `0x6d1960`; `.w` the disc colour's alpha byte above the band, 1.0 from the
/// 0xFF broadcast and 0 for moon02, whose colour has no writer (`0xce98a4`). Under weather `.w`
/// takes the alpha seed `floor(255·(1−bcc))/255` (`0x6d2c74`), `bcc = min(1, density·4)`. The
/// reference's per-vertex fade is interpolated across [`CelestialExt::span`].
#[derive(Asset, AsBindGroup, Clone, TypePath, Default)]
pub struct CelestialExt {
    #[uniform(100)]
    pub(super) fade: Vec4,
    /// The disc quad's bottom (`.x`) and top (`.y`) edges in sin-elevation, written each frame.
    #[uniform(101)]
    pub(super) span: Vec4,
}

impl MaterialExtension for CelestialExt {
    /// The shared sky vertex stage, the far-depth pin ([`crate::sky_order`]).
    fn vertex_shader() -> ShaderRef {
        SKY_VERTEX_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_world/shaders/celestial.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        sky_pipeline_state(descriptor);
        Ok(())
    }
}

/// The discs' horizon ramp slope: `0x6d1960`'s `clamp(2.5·height, 0, 1)` at radius 12 is
/// `clamp(30·dir.y, 0, 1)`, a soft edge over the bottom ~1.9° of elevation; at or below 0 it clips.
pub(super) const DISC_HORIZON_FADE: f32 = 30.0;

/// The `Stars.m2` batches, blended premultiplied in gamma space by `star.wgsl`, like the reference.
pub type StarMaterial = ExtendedMaterial<StandardMaterial, StarExt>;

/// A star batch's second stage and blend. The base colour carries stage 0 and the vertex alpha,
/// the star-curve fade times the batch weight.
#[derive(Asset, AsBindGroup, Clone, TypePath, Default)]
pub struct StarExt {
    /// `.x` 1 for an additive batch (M2 blend 3/4), 0 for alpha blend; `.y` 1 when
    /// [`Self::stage2`] is bound; `.z` 1 when the frame is linear (modern lighting) and `.w` the
    /// reciprocal of the modern post's exposure, both written per frame.
    #[uniform(100)]
    pub(super) params: Vec4,
    /// The batch's second texture, on `UV_1`, modulated over stage 0: blend modes 1-4 set every
    /// stage's op to `GL_COMBINE` MODULATE on RGB and alpha (`0x70b7d7`, table `0x811f8c`, the GL
    /// rows at `0x85c254`).
    #[texture(101)]
    #[sampler(102)]
    pub(super) stage2: Option<Handle<Image>>,
    /// fork: the modern horizon fade (`follow::MODERN_STAR_HORIZON`), modern frames only: the dome is
    /// gone below the view elevation (sine) `.x` and whole above `.y`; `.y` 0 = no fade. The dome
    /// draws after the fog composite, so without it the nebulae ran unhazed down to the horizon.
    #[uniform(103)]
    pub(super) horizon: Vec4,
    /// fork: the modern sky's hue (`follow::star_tint`), modern frames only: `.rgb` the hue of the
    /// sky the dome draws over, at luma 1, and `.w` the share of the dome's own colour that gives way
    /// to it at the same luma; `.w` 0 = the authored colour. patch-Y's teal nebulae otherwise
    /// clashed with a zone's mauve or green night sky.
    #[uniform(104)]
    pub(super) tint: Vec4,
}

impl MaterialExtension for StarExt {
    /// The shared sky vertex stage, the far-depth pin ([`crate::sky_order`]).
    fn vertex_shader() -> ShaderRef {
        SKY_VERTEX_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_world/shaders/star.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        sky_pipeline_state(descriptor);
        Ok(())
    }
}
