//! The render materials: the four `ExtendedMaterial`s the world is drawn with (terrain splat,
//! M2/WMO model, WDL far band, liquid surface) and the WGSL each one binds.
//!
//! The WGSL is embedded (`embedded://benilla_assets/shaders/…`), not served: a relative path would
//! resolve against the host binary's `AssetPlugin::file_path` and render nothing elsewhere.

use bevy::image::Image;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
    MeshPipelineKey,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, ColorWrites,
    CompareFunction, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

// fork: modern lighting (FORK.md, P2) — the modern sun's cascade shadow map
// (`benilla_world::modern::shadow`) and its cascade block, as constant handles so every material can
// name them at construction whatever order the assets are created in.
pub const MODERN_SHADOW_MAP: Handle<Image> =
    bevy::asset::uuid_handle!("6d2f1e4a-3b7c-4e0a-9a51-0c7e8f2d9b31");
pub const MODERN_SHADOW_CASCADES: Handle<bevy::render::storage::ShaderBuffer> =
    bevy::asset::uuid_handle!("b3c94a7e-52d1-4f6b-8e2a-7d10c4e5f9a2");

/// fork: perf — `$WOW_MODERN_VERTEXWALKS=1` (an A/B lever, read once; the browser has no
/// environment and keeps the default): the modern pipelines of `wow_model.wgsl`, `terrain.wgsl`
/// and `static_gx.wgsl` are built with `MODERN_VERTEX_WALKS`, and every vertex walks the light
/// table for the Gouraud sum and the three-nearest selection whether its fragments read them or
/// not, as before.
pub fn modern_vertex_walks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("WOW_MODERN_VERTEXWALKS").as_deref() == Ok("1"))
}
// fork: modern lighting (FORK.md, P4) — the integrated froxel fog volume
// (`benilla_world::modern::volumetrics`), sampled by the transparent lanes.
pub const MODERN_FOG_VOLUME: Handle<Image> =
    bevy::asset::uuid_handle!("e8a1c6d2-4f37-4b9e-a0c5-93d17b2f6e48");

/// Compile the five WGSL files into the binary under `embedded://benilla_assets/shaders/…`. Call
/// after Bevy's `AssetPlugin`, whose registry this fills; [`crate::register_asset_loaders`] does.
pub fn register_shaders(app: &mut App) {
    bevy::asset::embedded_asset!(app, "shaders/terrain.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/wow_model.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/wdl.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/liquid.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/liquid_cata.wgsl");
    // fork: the importable libraries the world shaders call into, held loaded so the `#import`
    // resolves. A headless app with no `Shader` asset type (the loader tests) has nothing to
    // compile them for.
    if app
        .world()
        .contains_resource::<Assets<bevy::shader::Shader>>()
    {
        // `benilla::env_frame`, the block the world `WowLight` mirrors declare after the point
        // table (`benilla-world`'s `lighting/env_frame.rs`).
        bevy::shader::load_shader_library!(app, "shaders/env_frame.wgsl");
        // `benilla::wind_hook`, the shared vertex displacement for grass and foliage.
        bevy::shader::load_shader_library!(app, "shaders/wind_hook.wgsl");
        // `benilla::wet_hook`, the rain-on-surfaces look terrain, model and static_gx call.
        bevy::shader::load_shader_library!(app, "shaders/wet_hook.wgsl");
    }
}

/// The WDL far-band shader's source, for the tests that live beside the renderer (`wdl.rs`).
pub const WDL_WGSL: &str = include_str!("shaders/wdl.wgsl");

/// Alpha-test reference for blend mode 1 (`Blend_AlphaKey`): 224 in the reference's
/// per-blend-mode table at `0x85ad20`, `{0, 224, 1, 1, 1, 1, 1, 0, 0, 0, 0}`. Must stay in sync
/// with `VANILLA_ALPHA_KEY` in `shaders/wow_model.wgsl`.
pub const VANILLA_ALPHA_KEY_REF: f32 = 224.0 / 255.0;

/// `StandardMaterial` plus the per-tile layer-blend extension.
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExtension>;

/// World models (doodads, WMOs, creatures, GameObjects) lit in gamma space like terrain, not by
/// PBR; `StandardMaterial` carries only the texture, alpha mode and culling.
pub type WowModelMaterial = ExtendedMaterial<StandardMaterial, WowModelExt>;

/// Pipeline-specialization key for [`WowModelExt`]: the depth and blend states Bevy's `AlphaMode`
/// cannot express. `fade` (`model_flags.y`) is the M2 doodad fade twin and `clutter`
/// (`clutter_fade.w`) ground clutter; both fade opacity with depth-write on, as the reference does.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WowModelKey {
    fade: bool,
    clutter: bool,
    /// Additive glow card (`clutter_fade.z` bit 2).
    additive: bool,
    /// M2 render flag 0x10, no depth write (`clutter_fade.z` bit 0).
    no_depth_write: bool,
    /// M2 render flag 0x08, no depth test (`clutter_fade.z` bit 1).
    no_depth_test: bool,
    /// The multiply blends, Mod and Mod2x (`clutter_fade.z` bits 7/8).
    modulate: bool,
    modulate2x: bool,
    /// The depth-prime twin (`clutter_fade.z` bit 9, `model_render::zfill_material`).
    zfill: bool,
    /// Far side of the water plane (`clutter_fade.z` bit 11).
    far_side: bool,
    /// The WMO-skybox lane (`clutter_fade.z` bit 13, `model_render::SKY_DEPTH_MARKER`), a key axis
    /// because only this model's depth is pinned.
    sky_depth: bool,
    /// fork: modern lighting — the latched lighting model ([`MODERN_MODEL_PIPELINES`]). A classic
    /// pipeline does not compile the modern code at all (`WOW_MODERN`): carrying it dead behind the
    /// runtime flag turned Chrome/Metal's leaf-card holes black in classic frames.
    modern: bool,
    // The WMO batch order is deliberately not a key axis (a pipeline per batch index stalls a
    // city's first sight): the file-order layering (`0x6b4f10`/`0x6b5190`) rides `sun_scale.y`.
}

impl From<&WowModelExt> for WowModelKey {
    fn from(e: &WowModelExt) -> Self {
        let markers = e.clutter_fade.z as u32;
        Self {
            fade: e.model_flags.y > 0.5,
            clutter: e.clutter_fade.w > 0.5,
            additive: markers & 4 != 0,
            no_depth_write: markers & 1 != 0,
            no_depth_test: markers & 2 != 0,
            modulate: markers & 0x80 != 0,
            modulate2x: markers & 0x100 != 0,
            zfill: markers & 0x200 != 0,
            far_side: markers & 0x800 != 0,
            sky_depth: markers & 0x2000 != 0,
            modern: MODERN_MODEL_PIPELINES.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

/// fork: modern lighting — whether model pipelines compile the modern lighting code
/// (`wow_model.wgsl`'s `WOW_MODERN`). Set by `benilla_world::modern` at the lighting latch, which
/// then re-prepares every model material so the new key reaches the pipelines.
pub static MODERN_MODEL_PIPELINES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The model material's extension. Every `Vec4` uniform packs into binding 100, one buffer entry,
/// to stay under Metal's 16-buffer vertex-stage cap (Bevy 0.18 ignores `visibility` on uniforms);
/// `ModelParams` in `wow_model.wgsl` must match the field order.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
#[bind_group_data(WowModelKey)]
pub struct WowModelExt {
    /// Clutter fade: `x`, `y` = full-opacity and fully-faded view depth (yd; the ~70 yd horizon),
    /// `w` = clutter, all zero off clutter; `z` = the batch marker bits.
    #[uniform(100)]
    pub clutter_fade: Vec4,
    /// `x` = WMO, `y` = the fade blend twin, `z` = interior, `w` = unlit fullbright.
    #[uniform(100)]
    pub model_flags: Vec4,
    /// `x` = the terrain-shade selector, `y` = the WMO batch order (the clip-z layering nudge),
    /// `zw` = the UV-animation offset.
    #[uniform(100)]
    pub sun_scale: Vec4,
    /// `xyz` = the M2Color tint of a batch whose colour track animates, identity otherwise; `w` =
    /// the WMO interior batch class (`0x6b5190`): `0` exterior, `1` INT, `2` TRANS.
    #[uniform(100)]
    pub tint: Vec4,
    /// WMO glass, `0` on M2: `xyz` = the MOMT SIDN night emissive (ramp `0x6b4090`), `w` = the MOMT
    /// WINDOW flag, the brighter interior light (`0x6d37e0`, ambient +16/255).
    #[uniform(100)]
    pub sidn: Vec4,
    /// Rows of the shared light buffer's `matanim` region, `0` = identity: `x` = UV scroll, `y` =
    /// tint, `z` = the texture-transform affine, `w` = the UI tile's cell clip.
    #[uniform(100)]
    pub anim_slots: Vec4,
    /// The second texture stage: `x` = bound (1) or not (0), `y` = its combine preset 0-5
    /// (`benilla_formats::TextureStage::combine`); `z` = the stage-0 combine preset plus one
    /// (`benilla_formats::RenderSubmesh::combine`), 0 for a batch that has none; `w` = 1 for a stage
    /// on UV set B instead of the generated coordinate, `1 + slot` with a scroll row in the
    /// `matanim` table (tortoise).
    #[uniform(100)]
    pub stage1: Vec4,
    /// Stage 1's texture, sampled at the generated environment coordinate when `stage1.x` is set;
    /// `None` binds the fallback image, unread.
    #[texture(101, visibility(fragment))]
    #[sampler(102, visibility(fragment))]
    pub stage1_texture: Option<Handle<Image>>,
    /// The shared global light (`lighting::global_light`), updated in place once a frame; the
    /// vertex stage reads its point-light table, since Bevy's clusterable lights are fragment-only.
    #[storage(90, read_only, buffer, visibility(vertex, fragment))]
    pub light_buf: Buffer,
    // fork: modern lighting (FORK.md, P2) — sampled only when the light buffer's modern flag is set.
    #[texture(
        121,
        dimension = "2d_array",
        sample_type = "depth",
        visibility(fragment)
    )]
    #[sampler(122, sampler_type = "comparison", visibility(fragment))]
    pub modern_shadow: Handle<Image>,
    #[storage(123, read_only, visibility(fragment))]
    pub modern_cascades: Handle<bevy::render::storage::ShaderBuffer>,
    // fork: modern lighting (FORK.md, P4) — blended batches fog themselves from the froxel volume.
    #[texture(124, dimension = "3d", visibility(fragment))]
    #[sampler(125, visibility(fragment))]
    pub modern_fog: Handle<Image>,
}

impl MaterialExtension for WowModelExt {
    /// Bevy's mesh vertex plus the point-light term, per vertex like the reference's FFP.
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wow_model.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wow_model.wgsl".into()
    }

    /// The reference writes depth for every M2 batch, transparent ones too, and tests `LEQUAL`,
    /// unless render flag 0x10 (no write) or 0x08 (no test) clears it (`0x70c190`).
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // `WOW_RIG_SKIN` skins from the shared buffer's palette. The base layout lacks our joint
        // attributes: rebuild it with Bevy's conditionals (mesh.rs, locations 0-5) plus 10/11.
        if layout.0.contains(crate::ATTRIBUTE_WOW_JOINT_INDEX) {
            descriptor.vertex.shader_defs.push("WOW_RIG_SKIN".into());
            // The FRAGMENT stage keys on the same def: a skinned unit takes the whole-unit
            // ground-point shadow sample (one fetch at its rig origin) instead of the
            // per-fragment fetch, which would read the unit's own caster copy as an occluder.
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_RIG_SKIN".into());
            }
            let mut attrs = Vec::with_capacity(7);
            for (attr, loc) in [
                (Mesh::ATTRIBUTE_POSITION, 0),
                (Mesh::ATTRIBUTE_NORMAL, 1),
                (Mesh::ATTRIBUTE_UV_0, 2),
                (Mesh::ATTRIBUTE_UV_1, 3),
                (Mesh::ATTRIBUTE_TANGENT, 4),
                (Mesh::ATTRIBUTE_COLOR, 5),
            ] {
                if layout.0.contains(attr) {
                    attrs.push(attr.at_shader_location(loc));
                }
            }
            attrs.push(crate::ATTRIBUTE_WOW_JOINT_INDEX.at_shader_location(10));
            attrs.push(crate::ATTRIBUTE_WOW_JOINT_WEIGHT.at_shader_location(11));
            descriptor.vertex.buffers = vec![layout.0.get_layout(&attrs)?];
        }
        // A merged blob computes the doodad fade per vertex from its fade sphere; same rebuild,
        // and merged blobs never skin.
        if layout.0.contains(crate::ATTRIBUTE_WOW_FADE_SPHERE) {
            descriptor.vertex.shader_defs.push("WOW_MERGED_FADE".into());
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_MERGED_FADE".into());
            }
            let mut attrs = Vec::with_capacity(6);
            for (attr, loc) in [
                (Mesh::ATTRIBUTE_POSITION, 0),
                (Mesh::ATTRIBUTE_NORMAL, 1),
                (Mesh::ATTRIBUTE_UV_0, 2),
                (Mesh::ATTRIBUTE_UV_1, 3),
                (Mesh::ATTRIBUTE_TANGENT, 4),
                (Mesh::ATTRIBUTE_COLOR, 5),
            ] {
                if layout.0.contains(attr) {
                    attrs.push(attr.at_shader_location(loc));
                }
            }
            attrs.push(crate::ATTRIBUTE_WOW_FADE_SPHERE.at_shader_location(12));
            // An interior-prop blob's baked SH-probe slot replaces the per-entity MeshTag payload.
            if layout.0.contains(crate::ATTRIBUTE_WOW_MERGED_SLOT) {
                descriptor.vertex.shader_defs.push("WOW_MERGED_SLOT".into());
                if let Some(fragment) = descriptor.fragment.as_mut() {
                    fragment.shader_defs.push("WOW_MERGED_SLOT".into());
                }
                attrs.push(crate::ATTRIBUTE_WOW_MERGED_SLOT.at_shader_location(13));
            }
            descriptor.vertex.buffers = vec![layout.0.get_layout(&attrs)?];
        }
        if let Some(ds) = descriptor.depth_stencil.as_mut() {
            ds.depth_write_enabled = Some(!key.bind_group_data.no_depth_write);
            if key.bind_group_data.no_depth_test {
                ds.depth_compare = Some(CompareFunction::Always);
            }
        }
        // A sort bias is not a raster bias: the base `StandardMaterial::specialize` also packs it
        // into `depth_stencil.bias.constant`, where the far side's −4e4 would clip coplanar layers.
        if key.bind_group_data.far_side {
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.bias.constant = 0;
            }
        }
        if key.bind_group_data.modern {
            descriptor.vertex.shader_defs.push("WOW_MODERN".into());
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_MODERN".into());
            }
            if modern_vertex_walks() {
                descriptor
                    .vertex
                    .shader_defs
                    .push("MODERN_VERTEX_WALKS".into());
                if let Some(fragment) = descriptor.fragment.as_mut() {
                    fragment.shader_defs.push("MODERN_VERTEX_WALKS".into());
                }
            }
        }
        // The skybox pins clip z to 0, reverse-Z far, in the vertex stage; its bias is sort-only.
        if key.bind_group_data.sky_depth {
            descriptor.vertex.shader_defs.push("WOW_SKY_DEPTH".into());
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_SKY_DEPTH".into());
            }
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.bias.constant = 0;
            }
        }
        if key.bind_group_data.fade && !key.bind_group_data.sky_depth {
            // A fading model's near cards occlude its far ones; the sky never writes depth.
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.depth_write_enabled = Some(true);
            }
        }
        if key.bind_group_data.clutter {
            // Clutter blends in the alpha-mask pass, as the reference does, so the ~70 yd ramp
            // fades opacity; the shader keeps the 128/255 discard.
            if let Some(target) = descriptor
                .fragment
                .as_mut()
                .and_then(|f| f.targets.get_mut(0))
                .and_then(|t| t.as_mut())
            {
                target.blend = Some(BlendState::ALPHA_BLENDING);
            }
        }
        if key.bind_group_data.additive {
            // A pure (ONE, ONE) add: the shader weights by alpha in gamma space, where a `SrcAlpha`
            // factor or `AlphaMode::Add` would weight after linearising and fatten soft edges.
            if let Some(target) = descriptor
                .fragment
                .as_mut()
                .and_then(|f| f.targets.get_mut(0))
                .and_then(|t| t.as_mut())
            {
                target.blend = Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                });
            }
        }
        if key.bind_group_data.zfill {
            if let Some(frag) = descriptor.fragment.as_mut() {
                frag.shader_defs.push("WOW_ZFILL".into());
            }
            // The reference's `M2UseZFill` pre-pass (`0x707f7d`-`0x708072`): depth only, so a
            // translucent model shows one blended layer. The colour is computed and masked, not
            // skipped by an early return, which naga's MSL backend miscompiles.
            if let Some(frag) = descriptor.fragment.as_mut() {
                if let Some(target) = frag.targets.get_mut(0).and_then(|t| t.as_mut()) {
                    target.blend = None;
                    target.write_mask = ColorWrites::empty();
                }
            }
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.depth_write_enabled = Some(true);
            }
        }
        // The waterline clip (`benilla_world::straddle`) keys on the blend pass: no new pipelines.
        if key
            .mesh_key
            .intersection(MeshPipelineKey::BLEND_RESERVED_BITS)
            == MeshPipelineKey::BLEND_ALPHA
        {
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_WATER_CLIP".into());
            }
        }
        let key = &key.bind_group_data;
        if key.modulate || key.modulate2x {
            // `0x70c190`: Mod (M2 mode 5, WMO 4) is `DST_COLOR/ZERO`, Mod2x (M2 6, WMO 5)
            // `DST_COLOR/SRC_COLOR`; on a gamma framebuffer this is the reference's byte math.
            if let Some(target) = descriptor
                .fragment
                .as_mut()
                .and_then(|f| f.targets.get_mut(0))
                .and_then(|t| t.as_mut())
            {
                target.blend = Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Dst,
                        dst_factor: if key.modulate2x {
                            BlendFactor::Src
                        } else {
                            BlendFactor::Zero
                        },
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                });
            }
        }
        Ok(())
    }
}

/// Distant low-detail terrain (WDL), the horizon hills beyond the streamed tiles: opaque white
/// geometry under the scene fog colour at the hull pass's own start 0 and end 1.0.
pub type WdlMaterial = ExtendedMaterial<StandardMaterial, WdlExt>;

/// WDL has no per-material input: its whole colour is the scene fog, off the shared global light.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct WdlExt {
    /// The shared global light, rows 4/5 only (scene fog colour, farclip wall).
    #[storage(90, read_only, buffer)]
    pub light_buf: Buffer,
}

impl MaterialExtension for WdlExt {
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wdl.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wdl.wgsl".into()
    }
}

/// Liquid surfaces, one `liquid.wgsl` arm per reference liquid renderer: ADT MCLQ (the
/// `ocean0_s.bls` combine), WMO exterior and interior water, and magma and slime.
pub type LiquidMaterial = ExtendedMaterial<StandardMaterial, LiquidExt>;

/// Liquid inputs: the animated frames and the per-material lanes; all light comes off the shared
/// global light. The `Vec4` uniforms pack into binding 102, in the WGSL struct's field order.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct LiquidExt {
    /// The kind's animated frames (`lake_a`, `fast_a`, `ocean_h`): RGB near-black, alpha ripple.
    #[texture(100, dimension = "2d_array", visibility(fragment))]
    #[sampler(101, visibility(fragment))]
    pub frames: Handle<Image>,
    /// Which lanes of the shared light this surface reads:
    /// - `x` = fullbright (magma, slime): the sheet is the opaque body.
    /// - `y` = ocean: rows 15/16 (`Light.dbc` IntBand 14/15), else rows 13/14 (IntBand 16/17).
    /// - `z` = a WMO interior group's liquid, fogged with the interior block (rows 18/19): the
    ///   reference gates its fog (`0x6b6323`-`0x6b6342`) on the same `[0xca7f00]` as the group's
    ///   geometry (`0x6b51d9`/`0x6b51ea`).
    /// - `w` = the sun-sheen shininess (`lighting::WATER_SHININESS`).
    #[uniform(102)]
    pub kind: Vec4,
    /// `x` = the renderer (`liquid::surface::LiquidPath`): `0` ADT MCLQ, `1` WMO exterior, `2` WMO
    /// interior.
    #[uniform(102)]
    pub path: Vec4,
    /// `y` = frame count, `z` = scroll flag, `w` = clock enable.
    #[uniform(102)]
    pub anim: Vec4,
    /// The shared global light, read in both stages (the vertex stage evaluates the sun sheen).
    #[storage(90, read_only, buffer, visibility(vertex, fragment))]
    pub light_buf: Buffer,
    // fork: modern lighting (FORK.md, P4) — water draws after the fog composite and fogs
    // itself from the froxel volume (the block carries its parameters).
    #[storage(123, read_only, visibility(fragment))]
    pub modern_cascades: Handle<bevy::render::storage::ShaderBuffer>,
    #[texture(124, dimension = "3d", visibility(fragment))]
    #[sampler(125, visibility(fragment))]
    pub modern_fog: Handle<Image>,
}

/// **The Cataclysm liquid renderer** (`waterDetail 2`) — the second of benilla's two liquid paths.
///
/// A separate material type rather than a shader-def branch on [`LiquidExt`], and deliberately so:
///
/// * The two do not share a bind group. This one carries a bump map and three pictures of the frame
///   ([`benilla_world`'s `liquid::screen`]); the 1.12.1 one carries an animated sheet and nothing
///   else, and paying its cost on every machine that never turns cata water on would be a tax on
///   the default-off path for no reason.
/// * `MaterialExtension::vertex_shader` is an associated function with no key, so a *single*
///   extension type cannot answer with two different WGSL files. The alternative is one file
///   forked end to end by `#ifdef`, which would dilute `liquid.wgsl`'s header — 86 lines of
///   byte-level reverse engineering about a *different* renderer, five previously-wrong readings
///   and why each was reverted. That record earns its own file.
///
/// Swapping is therefore a component swap (`MeshMaterial3d<LiquidMaterial>` ↔
/// `MeshMaterial3d<LiquidCataMaterial>`), which `benilla_world`'s `liquid::surface` owns.
pub type LiquidCataMaterial = ExtendedMaterial<StandardMaterial, LiquidCataExt>;

/// Inputs for `liquid_cata.wgsl` — the ported `ProcWater` + `ProcWaterAbove`/`ProcWaterBelow`
/// (perm 4) program family.
///
/// The reference binds ten texture stages for this program; six of them belong to permutations this
/// client does not port (the dynamic displacement maps, the third material texture, the noise
/// volume, and the real reflection buffer). What is left is the four below plus the depth ramp,
/// which is evaluated from the shared light rather than sampled — see `liquid_cata.wgsl`.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct LiquidCataExt {
    /// The kind's animated frames, exactly as [`LiquidExt::frames`] carries them.
    ///
    /// **Water does not read this** — the cata water program has no animated-sheet stage at all,
    /// its surface detail coming from the bump map and its body from the depth ramp. It stays bound
    /// because magma and slime still take their sheet as the body colour, and one material type
    /// covering all five kinds is worth more than a binding the water lane ignores.
    #[texture(100, dimension = "2d_array", visibility(fragment))]
    #[sampler(101, visibility(fragment))]
    pub frames: Handle<Image>,

    /// Stage 5: the tiling wave normal map, sampled four times per fragment at two scales and two
    /// scroll phases. Synthesised at build time — `benilla_world`'s `liquid::art` is the whole
    /// account of why, and of why it is a full RGB normal rather than a two-channel derivative.
    #[texture(103, visibility(fragment))]
    #[sampler(104, visibility(fragment))]
    pub bump: Handle<Image>,

    /// Stage 0: the opaque frame, for refraction.
    #[texture(105, visibility(fragment))]
    #[sampler(106, visibility(fragment))]
    pub scene_color: Handle<Image>,
    /// Stage 3: a blurred quarter-size copy of the same, for the screen-space reflection. Shares
    /// sampler 106 — both are linear-clamp reads of the same picture at different scales, and
    /// Metal's per-stage sampler ceiling is close enough here to be worth not spending twice.
    #[texture(107, visibility(fragment))]
    pub scene_blur: Handle<Image>,
    /// Stage 1: the **planar reflection buffer** — a mirrored render of the world about the
    /// world plane `z = 0`, which `ProcWaterAbove` perm 0 samples as the BASE of its reflection
    /// and lays the screen-space trace over.
    ///
    /// `SetupRender` binds this on stage 1 as `a7 == 1 ? GetRealReflectionBuffer() :
    /// GetFakeReflectionBuffer()`, and the "fake" one is not a fake: `HasFakeReflection()` is
    /// `waterDetail != 0` outright, and it drives a real second scene render —
    /// `CWorldScene::RenderReflection(s_defaultReflectionView2, FakeReflectionFB, 0, 0.0)`,
    /// mirroring the camera with `C4Plane::Mirror` about `(0, 0, 1, −0)`. `benilla_world`'s
    /// `liquid::reflect` owns the camera that fills it.
    ///
    /// **No sampler of its own** — it takes 106 (`scene_samp`), the linear-clamp one the other two
    /// screen reads use. It is addressed by a projected screen coordinate exactly as they are, and
    /// the WebGPU 16-sampler cap is close enough here to be worth not spending a fourth on.
    #[texture(115, visibility(fragment))]
    pub reflect_map: Handle<Image>,

    /// Stage 7: the shoreline foam sheet, sampled twice at two scroll phases and ADDED.
    ///
    /// `ProcWaterAbove`'s `samp7` — `GetTexture(mat, 3, 0)`, the third of the reference's per-liquid
    /// material textures. Synthesised, like the bump map, and for the same reason: the column that
    /// names it does not exist in 1.12.1's `LiquidType.dbc`. See `benilla_world`'s `liquid::art`.
    ///
    /// **No sampler of its own** — it takes 104 (`art_samp`), which `liquid_cata.wgsl` explains at
    /// the declaration: this sheet, the bump map and the noise volume are built with identical
    /// descriptors, and WebGPU's 16-sampler-per-fragment-stage cap counts every one in the layout.
    #[texture(109, visibility(fragment))]
    pub foam: Handle<Image>,

    /// `ProcMagma`'s stage 0: the animated noise VOLUME whose three channels are three independent
    /// masks for three composited lava layers.
    ///
    /// The reference loads it from `XTextures\FX\noise.blob` as a 128 x 128 x 16 volume; this one
    /// is synthesised at half the face (`benilla_world`'s `liquid::art`). The third axis is a
    /// clock, not a third space dimension — the shader walks it — which is why the sampler repeats
    /// on `w` too.
    /// on `w` too — and 104, the shared `art_samp` it now takes instead of one of its own, repeats
    /// on all three axes for exactly that reason. A sampler carries no dimensionality, so the same
    /// binding serves this 3D volume and the two 2D sheets alike.
    #[texture(111, dimension = "3d", visibility(fragment))]
    pub noise_vol: Handle<Image>,

    /// `ProcWaterAbove`'s stage 4: the dynamic ripple displacement map — `.x` the current wave
    /// height, `.y` the previous one, packed so the water can interpolate between two fixed steps.
    /// `benilla_world`'s `liquid::ripple` owns the simulation.
    #[texture(113, visibility(fragment))]
    #[sampler(114, visibility(fragment))]
    pub ripple_map: Handle<Image>,

    /// Stage 6: linear view depth in yards, `R32Float`.
    ///
    /// `filterable = false` and **no sampler**: `R32Float` cannot take a filtering sampler without
    /// a device feature, and the shader reads it with `textureLoad`. That is not a concession —
    /// a depth used for an in-front-of/behind comparison wants a point fetch, and interpolating
    /// across a silhouette would invent a surface that is in neither place.
    #[texture(108, sample_type = "float", filterable = false, visibility(fragment))]
    pub scene_depth: Handle<Image>,

    // -------------------------------------------------------------------------------------------
    // Packed uniform buffer at binding 102 — the WGSL `LiquidParams` struct lists these fields in
    // the SAME order (each `Vec4` is 16 bytes, no padding). Reordering is a breaking change.
    // -------------------------------------------------------------------------------------------
    /// As [`LiquidExt::kind`]: `x` fullbright, `y` ocean swatch, `z` interior fog, `w` sheen
    /// shininess.
    #[uniform(102)]
    pub kind: Vec4,
    /// As [`LiquidExt::path`]: `x` = which of the reference's three liquid renderers owns this
    /// surface. `y` = **the below-water flag** — 1 when the camera eye is under this surface, which
    /// is what selects `ProcWaterBelow` over `ProcWaterAbove`. The reference makes the same choice
    /// per draw (`CMaterialWater::RenderAbove` / `RenderBelow`), not per fragment. `z` = **is
    /// [`Self::reflect_map`] live this frame** — the reference's own `HasFakeReflection()`, which
    /// stands the mirror down while the camera is submerged (`Liquid::CameraSubmerged`) and
    /// whenever `waterDetail` is 0; with it clear the shader falls back to the flat horizon colour
    /// perm 4 uses, rather than reading a frame-old mirror.
    #[uniform(102)]
    pub path: Vec4,
    /// As [`LiquidExt::anim`]: `x` reserved, `y` frame count, `z` scroll flag, `w` clock enable.
    #[uniform(102)]
    pub anim: Vec4,
    /// The bump lane, from the reference's `LiquidType.dbc` `Float[18]` block:
    /// `x`/`y` = the two tiling scales (`GetFloat(mat, 3)` / `GetFloat(mat, 4)`, each times
    /// `Liquid::WorldCoordScale`), `z` = the scroll rate, `w` = **bump strength** (`c23.w`,
    /// `GetFloat(mat, 7) × 0.35`).
    #[uniform(102)]
    pub bump_params: Vec4,
    /// The shading lane: `x` = the view-perturb amount (`c23.y`), `y` = specular power (`c4.w`),
    /// `z` = specular scale (`c27.w`), `w` = sun-glare scale (`c27.z`).
    #[uniform(102)]
    pub shade_params: Vec4,
    /// The shoreline foam lane: `x` = the sheet's tiling scale in repeats per yard, `y` = the
    /// **depth scale** in yards — how deep the water has to get before the surf fades out —
    /// `z` = strength (the reference's `c28.y`), `w` = how far the wave normal drags the second
    /// tap, so the foam churns with the swell instead of sliding over it.
    #[uniform(102)]
    pub foam_params: Vec4,
    /// `ProcMagma`'s tiling lane: `x`/`y` = the two noise-volume sample scales, `z`/`w` = the first
    /// two lava layer scales. All in repeats per yard.
    #[uniform(102)]
    pub magma_scales: Vec4,
    /// `ProcMagma`'s motion lane: `x` = the third layer's scale, `y` = the wobble frequency,
    /// `z` = how fast the clock walks the volume's third axis, `w` = how fast the two volume taps
    /// drift apart.
    #[uniform(102)]
    pub magma_params: Vec4,
    /// `ProcMagma`'s base colour — the reference's `c17.rgb`, what the surface is where every mask
    /// reads zero. The cold crust under the layers, not a tint over them.
    #[uniform(102)]
    pub magma_color: Vec4,
    /// fork: modern lighting — the water's optics on the modern lane: `xyz` = absorption per yard of
    /// water crossed (r, g, b), `w` = **the Modern (Experimental) water tier** (`waterDetail 3`),
    /// which alone runs the shader's `modern_water` combine; `liquid`'s `sync_water_tier` flips it
    /// with the setting. The classic lanes never read it.
    #[uniform(102)]
    pub optics: Vec4,
    /// fork: modern lighting — the Modern (Experimental) tier's stylised clear water, in the shader's
    /// `modern_water`: `x` = how far the bed is filtered toward the water's hue (the DEEP row), `y` =
    /// the depth in yards that filter takes to come in, `z` = the head-on reflectance Schlick starts
    /// from, `w` = how much of the reflection is pulled to the body. Per kind from `liquid`'s
    /// `cata_water_look`, overridable live through `$WOW_WATER_LOOK`. The classic lanes never read it.
    #[uniform(102)]
    pub look_a: Vec4,
    /// fork: modern lighting — the same tier's calm: `x` = the wind swell a river or lake keeps (the
    /// sea always keeps all of it), `y` = the swell left over water with no depth under it, `z` = the
    /// refraction's bend per yard of water crossed, `w` = the sun glitter spike's gain.
    #[uniform(102)]
    pub look_b: Vec4,
    /// fork: modern lighting — the same tier's Fresnel: `x` = the reflectance at grazing (the curve's
    /// cap; `look_a.z` is its head-on floor), `y` = the power of its climb along `1 − cos`, `z` = the
    /// share of the reference combine blended over the clear surface, `w` = the wind ripples' RMS
    /// slope, a multi-octave layer under the swell.
    #[uniform(102)]
    pub look_c: Vec4,

    /// **The shared global light**, as [`LiquidExt::light_buf`] — same buffer, same rows. The cata
    /// path reads the two water swatches for its depth ramp, the sun/ambient/diffuse for its
    /// lighting, and both fog blocks; it needs no light input the 1.12.1 path did not already have.
    #[storage(90, read_only, buffer, visibility(vertex, fragment))]
    pub light_buf: Buffer,

    /// **The ripple field's placement**, on its own shared buffer for the same reason the light is
    /// on one: it changes every frame, and a per-frame `#[uniform]` rewrite would re-prepare every
    /// liquid bind group every frame. Written in place by `liquid::ripple`'s render-world upload.
    ///
    /// A second buffer rather than more rows on the light one because the light buffer's tail is a
    /// variable-length point-light table — there is no free row to take, only spare lanes scattered
    /// through rows that mean something else.
    #[storage(91, read_only, buffer, visibility(fragment))]
    pub ripple_buf: Buffer,
    // fork: modern lighting (FORK.md, P4) — as `LiquidExt::modern_fog`, but with no
    // sampler of its own: this layout already sits at the 16-sampler stage limit, so the volume is read
    // through `scene_samp` (106, linear clamp).
    #[storage(123, read_only, visibility(fragment))]
    pub modern_cascades: Handle<bevy::render::storage::ShaderBuffer>,
    #[texture(124, dimension = "3d", visibility(fragment))]
    pub modern_fog: Handle<Image>,
    // fork: modern lighting — the sun cascades, so water in shade loses its sun glare and highlight.
    // Texture only, compared by hand with `textureLoad` (`sunshadow::sun_visibility_load`): the
    // layout has no sampler slot left.
    #[texture(
        121,
        dimension = "2d_array",
        sample_type = "depth",
        visibility(fragment)
    )]
    pub modern_shadow: Handle<Image>,
}

impl MaterialExtension for LiquidCataExt {
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/liquid_cata.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/liquid_cata.wgsl".into()
    }

    /// As [`LiquidExt`]'s: `sky_order::WATER_BIAS` is a sort rung, kept out of the rasterizer. As a
    /// depth-bias constant it sinks the surface in reversed-Z, and whatever stands just under the
    /// water wins the depth test and draws unshaded by it.
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(ds) = descriptor.depth_stencil.as_mut() {
            ds.bias.constant = 0;
        }
        Ok(())
    }
}

impl MaterialExtension for LiquidExt {
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/liquid.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/liquid.wgsl".into()
    }

    /// `sky_order::WATER_BIAS` (−2e4) is a sort rung, kept out of the rasterizer: as a depth-bias
    /// constant it would move the waterline, by an amount that doubles at every float exponent
    /// boundary, so neighbouring triangles' shorelines would disagree.
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(ds) = descriptor.depth_stencil.as_mut() {
            ds.bias.constant = 0;
        }
        Ok(())
    }
}

/// Per-tile terrain splat, one material per ADT tile: each merged-mesh vertex carries its chunk's
/// four layer indices (`COLOR`) and alpha layer (`UV1.x`), so a tile is one draw.
///
/// fork: modern lighting — the terrain lane's only pipeline key: the latched lighting model
/// ([`MODERN_MODEL_PIPELINES`]), exactly as [`WowModelKey::modern`] carries it for the model lane.
/// Without it both lighting models compile into every terrain pipeline, which is the fragment size
/// that turned Chrome/Metal's leaf-card holes black on the model lane, and the modern path pays a
/// discarded `shadow_array` sample on the lane already at the 16-texture cap. One value is live per
/// session (the latch is per world entry), so this mints no new variants at runtime.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TerrainKey {
    modern: bool,
}

impl From<&TerrainExtension> for TerrainKey {
    fn from(_: &TerrainExtension) -> Self {
        Self {
            modern: MODERN_MODEL_PIPELINES.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

/// One repeating sampler (105) serves every array, to stay under Metal's 16 fragment samplers;
/// `terrain.wgsl` insets the alpha UVs by half a texel so they do not wrap at a chunk edge. Every
/// `Vec4` uniform packs into binding 106, whose `TerrainParams` field order must match.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
#[bind_group_data(TerrainKey)]
pub struct TerrainExtension {
    #[texture(100, dimension = "2d_array", visibility(fragment))]
    #[sampler(105, visibility(fragment))]
    pub layer_array: Handle<Image>,
    #[texture(104, dimension = "2d_array", visibility(fragment))]
    pub alpha_array: Handle<Image>,
    /// Per-chunk MCSH shadow maps (`R` = shadowed); `UV1.y` is the layer, `-1` for none.
    #[texture(110, dimension = "2d_array", visibility(fragment))]
    pub shadow_array: Handle<Image>,

    // Binding 106's uniforms, in the field order of the WGSL `TerrainParams`.
    /// `x` = texture tiling factor (repeats per chunk); other lanes unused.
    #[uniform(106)]
    pub params: Vec4,

    /// The shared global light, rows 0-5 (light, fog, farclip).
    #[storage(90, read_only, buffer)]
    pub light_buf: Buffer,
    // fork: modern lighting (FORK.md, P2) — sampled only when the light buffer's modern flag is set.
    #[texture(
        121,
        dimension = "2d_array",
        sample_type = "depth",
        visibility(fragment)
    )]
    #[sampler(122, sampler_type = "comparison", visibility(fragment))]
    pub modern_shadow: Handle<Image>,
    #[storage(123, read_only, visibility(fragment))]
    pub modern_cascades: Handle<bevy::render::storage::ShaderBuffer>,
}

impl MaterialExtension for TerrainExtension {
    // The sun specular is per vertex (the reference's light flush `0x59c820`).
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/terrain.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/terrain.wgsl".into()
    }

    /// fork: modern lighting — compile one lighting model, not both ([`TerrainKey`]). The def
    /// reaches the vertex stage too: it decides which walk of the light table a vertex makes.
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if key.bind_group_data.modern {
            descriptor.vertex.shader_defs.push("WOW_MODERN".into());
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_MODERN".into());
            }
            if modern_vertex_walks() {
                descriptor
                    .vertex
                    .shader_defs
                    .push("MODERN_VERTEX_WALKS".into());
                if let Some(fragment) = descriptor.fragment.as_mut() {
                    fragment.shader_defs.push("MODERN_VERTEX_WALKS".into());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// The sky depth law for the WMO skybox, the one sky element on the model lane.
    #[test]
    fn the_sky_lane_pins_the_far_depth_at_the_vertex() {
        let src = include_str!("shaders/wow_model.wgsl");
        // The pin, matched behind its ifdef: ungated, every model draw sits at the far plane.
        let pin = src
            .find("#ifdef WOW_SKY_DEPTH")
            .expect("the sky-depth branch is gone");
        let branch = &src[pin..src[pin..].find("#endif").map_or(src.len(), |e| pin + e)];
        assert!(
            branch.contains("out.position.z = 0.0;"),
            "the model lane's sky branch no longer pins the far depth at the vertex — a WMO \
             skybox's shell radius is deciding occlusion again (benilla_world::sky_order, \"The \
             depth law\")"
        );
        assert!(
            !src.contains("@builtin(frag_depth)"),
            "the model lane writes a fragment depth again — the sky pin is the vertex stage's, \
             and a fragment write costs every draw on this lane its early-Z"
        );
    }

    /// A second texture stage samples `Model2.bls`'s generated coordinate: the program for a stage
    /// whose coordinate lookup is above 2 reflects the view-space position about the normal
    /// (`DP3`, `MUL 2`, `MAD −`), normalizes it (`RSQ`) and maps x, y by `MAD ·0.5 + 0.5`.
    #[test]
    fn stage1_samples_the_model2_environment_coordinate() {
        let src = include_str!("shaders/wow_model.wgsl");
        assert!(
            src.contains("let refl = normalize(p_view - 2.0 * dot(p_view, n_view) * n_view);")
                && src.contains("return refl.xy * 0.5 + vec2<f32>(0.5, 0.5);"),
            "the environment coordinate left `Model2.bls`'s reflection"
        );
        assert!(
            src.contains(
                "out.env_uv = wow_env_uv(view_rot * p_cam, normalize(view_rot * out.world_normal));"
            ),
            "stage 1 no longer generates its coordinate in view space"
        );
        assert!(
            src.contains("var st1_uv = in.env_uv;")
                && src.contains(
                    "tex1 = textureSampleBias(stage1_texture, stage1_sampler, st1_uv, view.mip_bias);"
                ),
            "stage 1 no longer samples its texture at the generated coordinate"
        );

        /// The ARB program's arithmetic, step for step.
        fn env_uv(p: [f32; 3], n: [f32; 3]) -> [f32; 2] {
            let pn = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
            let r: [f32; 3] = std::array::from_fn(|k| p[k] - 2.0 * pn * n[k]);
            let inv = 1.0 / (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            [r[0] * inv * 0.5 + 0.5, r[1] * inv * 0.5 + 0.5]
        }
        let close =
            |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6;
        let h = std::f32::consts::FRAC_1_SQRT_2;
        // Facing the eye straight on: R = (0, 0, 5), the texture's centre.
        assert!(close(env_uv([0.0, 0.0, -5.0], [0.0, 0.0, 1.0]), [0.5, 0.5]));
        // Off-axis on a facing normal: R = (3, 0, 4), u = 0.5 + 0.3.
        assert!(close(env_uv([3.0, 0.0, -4.0], [0.0, 0.0, 1.0]), [0.8, 0.5]));
        // A 45° normal turns the view ray to +x: the rim, u = 1.
        assert!(close(env_uv([0.0, 0.0, -4.0], [h, 0.0, h]), [1.0, 0.5]));
        // An upward normal above the eye: R = (0, −2, −2), v = 0.5 − 0.5·√½.
        assert!(close(
            env_uv([0.0, 2.0, -2.0], [0.0, 1.0, 0.0]),
            [0.5, 0.5 - 0.5 * h]
        ));
    }

    /// Stage 1 combines by the gx preset tables of the default device, Direct3D (`gxApi`,
    /// `0x63a230`): colour `0x80a25c` (4, 4, 13, 7, 5, 12 as `D3DTOP`) and alpha `0x80a274` (3, 4,
    /// 3, 7, 5, 3), args TEXTURE and CURRENT. The OpenGL records of `0x85c250` differ only in the
    /// weights of presets 2 and 5. Every stock two-texture unit is preset 1, colour and alpha
    /// modulated.
    #[test]
    fn stage1_combines_by_the_gx_presets() {
        let src = include_str!("shaders/wow_model.wgsl");
        for (line, why) in [
            (
                "var c = prev * tex.rgb;",
                "presets 0 and 1 modulate the colour",
            ),
            (
                "c = mix(prev, tex.rgb, tex.a);",
                "preset 2, BLENDTEXTUREALPHA, blends by the texel's alpha",
            ),
            ("c = prev + tex.rgb;", "preset 3 adds"),
            ("c = 2.0 * prev * tex.rgb;", "preset 4 modulates twice"),
            (
                "c = mix(prev, tex.rgb, diffuse_a);",
                "preset 5, BLENDDIFFUSEALPHA, blends by the diffuse alpha",
            ),
            (
                "return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));",
                "a stage saturates",
            ),
            ("a = prev * tex;", "preset 1 modulates the alpha"),
            ("a = prev + tex;", "preset 3 adds the alpha"),
            ("a = 2.0 * prev * tex;", "preset 4 doubles the alpha"),
            (
                "rgb = stage1_rgb(stage1_preset, rgb, tex1, obj_fade);",
                "stage 1 combines with the lit stage-0 result, preset 5 by the diffuse alpha",
            ),
            (
                "base_color.a = stage_alpha(stage1_preset, base_color.a, tex1.a);",
                "stage 1's alpha joins the alpha the cutout reads",
            ),
        ] {
            assert!(src.contains(line), "{why}: `{line}` is gone");
        }
        // Stage 1 combines after the light and before the fog, as the stages precede the fog.
        let combine = src.find("rgb = stage1_rgb(").expect("the combine");
        let fog = src
            .find("var fog_color = wow_light.fog_color;")
            .expect("the fog");
        let lit = src
            .find("var rgb = select(lit_rgb, unlit_rgb, is_emissive);")
            .expect("lit");
        assert!(
            lit < combine && combine < fog,
            "stage 1 sits between the light and the fog"
        );

        /// Preset 1 on one texel: the stock units' combine.
        fn preset1(prev: [f32; 4], tex: [f32; 4]) -> [f32; 4] {
            std::array::from_fn(|k| (prev[k] * tex[k]).clamp(0.0, 1.0))
        }
        let out = preset1([0.8, 0.6, 0.4, 1.0], [0.5, 0.25, 1.0, 0.5]);
        assert_eq!(out, [0.4, 0.15, 0.4, 0.5], "colour and alpha both modulate");
    }

    /// A fade twin blends by the alpha chain on the instance alpha, not the texel's: stage 0
    /// combines the diffuse alpha (`0x70c73a`, c28.w) with the texel by its preset's alpha op
    /// (`0x80a274` = 3, 4, 3, 7, 5, 3 as `D3DTOP`, args TEXTURE and CURRENT, `ALPHAARG` never set),
    /// so an opaque batch's preset 0 (`0x811f8c`) blends by the fade alone, and the 0x101 models'
    /// preset 1 by texel × fade.
    #[test]
    fn a_fade_twin_blends_by_the_stage_zero_alpha_op() {
        // As checked out: a Windows checkout under `core.autocrlf` ends its lines in CRLF.
        let src = include_str!("shaders/wow_model.wgsl").replace("\r\n", "\n");
        for (line, why) in [
            (
                "faded_alpha = stage_alpha(u32(m.stage1.z) - 1u, obj_fade, tex0_alpha);",
                "the blended alpha is stage 0's alpha op on the fade and the texel",
            ),
            (
                "faded_alpha = stage_alpha(stage1_preset, faded_alpha, tex1.a);",
                "stage 1 follows on the faded alpha",
            ),
            (
                "let tex0_alpha = base_color.a;",
                "stage 0's texel alpha is read before stage 1 joins it",
            ),
            (
                "fn stage_alpha(preset: u32, prev: f32, tex: f32) -> f32 {\n    var a = prev;",
                "an alpha op that selects ARG2 passes the previous alpha, the fade, whole",
            ),
            (
                "out.color = vec4<f32>(out_rgb, select(faded_alpha, 1.0, opaque_intent));",
                "the output alpha is the chain's",
            ),
        ] {
            assert!(src.contains(line), "{why}: `{line}` is gone");
        }
        // The old output multiplied the texel into every twin, whatever its combine.
        assert!(
            !src.contains("let faded_alpha = base_color.a * obj_fade;"),
            "the twin's alpha is the texel times the fade again, preset 0 included"
        );

        /// `stage_alpha`, step for step.
        fn stage_alpha(preset: u32, prev: f32, tex: f32) -> f32 {
            match preset {
                1 => prev * tex,
                3 => prev + tex,
                4 => 2.0 * prev * tex,
                _ => prev,
            }
            .clamp(0.0, 1.0)
        }
        // A texel at alpha 0 (the Wind Rider's `WyvernSkin02` batch) fading at 0.5.
        let (fade, texel) = (0.5, 0.0);
        assert_eq!(stage_alpha(0, fade, texel), 0.5, "preset 0: the fade alone");
        assert_eq!(stage_alpha(1, fade, texel), 0.0, "preset 1: texel × fade");
        // Preset 0's alpha does not move with the texel; preset 1's does.
        for t in [0.0, 0.25, 1.0] {
            assert_eq!(stage_alpha(0, fade, t), fade);
            assert_eq!(stage_alpha(1, fade, t), fade * t);
        }
        // The rest of the alpha ops: 2 and 5 select ARG2, 3 adds, 4 doubles the product.
        assert_eq!(stage_alpha(2, 0.5, 0.25), 0.5);
        assert_eq!(stage_alpha(5, 0.5, 0.25), 0.5);
        assert_eq!(stage_alpha(3, 0.5, 0.25), 0.75);
        assert_eq!(stage_alpha(4, 0.5, 0.25), 0.25);
    }

    /// The WGSL carries the reference's swatch row arithmetic, and a Rust mirror of it reproduces
    /// the row fill `0x68a830` on `Light.dbc` id 4, map 0, t = 1440, ocean `LightIntBand` 14 to 15.
    #[test]
    fn liquid_swatch_reproduces_the_reference_row_ramp() {
        let src = include_str!("shaders/liquid.wgsl");
        // The row accumulator `c0 + floor(i*(c1 - c0)/64)`, not a lerp to the deep endpoint.
        assert!(
            src.contains("let row = c0 + floor(i * (c1 - c0) / 64.0);"),
            "the swatch stopped building its rows the way `0x68a830` does: a plain lerp to the \
             deep endpoint runs a 64th of a ramp that does not exist"
        );
        // The ocean-only tail: floor(0.9*byte) on the last row, alpha forced opaque.
        assert!(
            src.contains("if ocean && i >= 63.0 {")
                && src.contains("vec4<f32>(floor(row.rgb * 0.9), 255.0)"),
            "the ocean's last-row darkening is gone — ~80% of the world's ocean vertices sample \
             that row, so this is the open sea's colour"
        );
        // Linear sampling across the two rows V falls between, at texel `V*64 - 0.5`.
        assert!(
            src.contains("let t = clamp(v * 64.0 - 0.5, 0.0, 63.0);"),
            "the swatch stopped sampling as an 8x64 LINEAR/CLAMP texture — the ocean darkening \
             would step instead of ramping across the final 1/64 of V"
        );

        /// `row(i)` for one channel, `swatch_row`'s byte arithmetic.
        fn row(c0: i32, c1: i32, i: i32) -> i32 {
            c0 + (i * (c1 - c0)).div_euclid(64)
        }
        /// The ocean tail's colour half.
        fn tail(b: i32) -> i32 {
            (f64::from(b) * 0.9).floor() as i32
        }

        // Ocean shallow (sub-14) 0x00457b63 and deep (sub-15) 0x000b2228, as R/G/B.
        let (shallow, deep) = ([69, 123, 99], [11, 34, 40]);
        let at = |i| [0, 1, 2].map(|k| row(shallow[k], deep[k], i));

        // Row 63 is one short of the endpoint on G: the ramp never arrives.
        assert_eq!(at(63), [11, 35, 40], "row 63 before the ocean tail");
        assert_ne!(
            at(63)[1],
            deep[1],
            "the ramp must NOT reach the deep endpoint"
        );
        assert_eq!(
            at(62),
            [12, 36, 41],
            "row 62, the other half of the last blend"
        );
        assert_eq!(
            at(63).map(tail),
            [9, 31, 36],
            "row 63 after the ocean tail (HSV V *= 0.9, which is floor(0.9*byte) per channel)"
        );
        // The tail and the shortfall are deep-end only.
        assert_eq!(at(0), shallow, "row 0 is the shallow endpoint verbatim");
    }
}
