//! **The modern sun's shadow pass** (fork-owned; `FORK.md` P2).
//!
//! A child of `static_gx::render` so it can reuse that module's GPU state as-is: the baked cell and
//! WMO vertex buffers, the per-cell bind groups (texture array, record table with kill bits) and the
//! draw runs. Each frame, for the primary world view, it:
//!
//! 1. uploads the cascade block (`modern::shadow::ModernCascades`) to its constant-handle buffer;
//! 2. clears every layer of the constant-handle shadow array (reverse-Z: to 0);
//! 3. draws every cached cell, WMO and prop whose world bounds overlap each active cascade,
//!    depth-only — opaque, or alpha-keyed for cutout runs, two-sided.
//!
//! Static casters (terrain, static_gx) are cached per cascade in a second depth array and copied
//! back while the cascade's matrix and the static caster set are unchanged; entity parts are drawn
//! on top every frame.

use bevy::app::SubApp;
use bevy::pbr::graph::NodePbr;
use bevy::render::render_resource::binding_types::uniform_buffer;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::storage::GpuShaderStorageBuffer;
use bevy::render::texture::GpuImage;
use bevy::shader::Shader;

use benilla_assets::materials::{MODERN_SHADOW_CASCADES, MODERN_SHADOW_MAP};

use super::*;
use crate::modern::lights::{ModernLights, ATLAS_TILES, PER_ATLAS_LAYER};
use crate::modern::shadow::{
    ModernCascades, ModernEntityCasters, ModernTerrainCasters, LOCAL_LAYER, SHADOW_LAYERS,
    SHADOW_MAP_SIZE, SKY_LAYER,
};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::binding_types::{storage_buffer_read_only_sized, texture_2d};
use bevy::platform::collections::HashMap as StrideMap;

#[derive(Clone, Copy, ShaderType)]
struct CasterView {
    clip_from_world: Mat4,
}

#[derive(Resource)]
struct GxShadowPipelines {
    view_layout: BindGroupLayoutDescriptor,
    opaque: CachedRenderPipelineId,
    cutout: CachedRenderPipelineId,
}

#[derive(Resource, Default)]
struct CasterViewBuffer(DynamicUniformBuffer<CasterView>);

#[derive(Clone, Copy, ShaderType)]
struct EntityDraw {
    world_from_local: Mat4,
    rig_slot: u32,
}

/// Entity caster pipelines by (mesh vertex layout, skinned, cutout).
#[derive(Resource)]
struct EntityShadowPipelines {
    view_layout: BindGroupLayoutDescriptor,
    draw_layout: BindGroupLayoutDescriptor,
    skin_layout: BindGroupLayoutDescriptor,
    cut_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
    by_key: StrideMap<(MeshVertexBufferLayoutRef, bool, bool), CachedRenderPipelineId>,
}

#[derive(Resource, Default)]
struct EntityDrawBuffer(DynamicUniformBuffer<EntityDraw>);

/// Cutout texture bind groups, rebuilt when the image's view changes.
#[derive(Resource, Default)]
struct CutoutBinds(StrideMap<AssetId<Image>, (TextureViewId, BindGroup)>);

struct EntityDrawCmd {
    mesh: AssetId<Mesh>,
    pipeline: CachedRenderPipelineId,
    offset: u32,
    skinned: bool,
    cutout: Option<BindGroup>,
    center: Vec3,
    radius: f32,
}

#[derive(Resource, Default)]
struct EntityShadowFrame {
    draws: Vec<EntityDrawCmd>,
    draw_bind: Option<BindGroup>,
    skin_bind: Option<BindGroup>,
}

/// Terrain caster pipelines, one per vertex stride (position is always attribute 0 at offset 0).
#[derive(Resource)]
struct TerrainShadowPipelines {
    view_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
    by_stride: StrideMap<u64, CachedRenderPipelineId>,
}

/// This frame's shadow targets: one depth view per array layer, the caster-view bind group and its
/// per-cascade offsets. `count == 0` = nothing to draw (classic, or not ready).
#[derive(Resource, Default)]
struct ModernShadowFrame {
    layers: Vec<TextureView>,
    /// The working array (the constant-handle image's texture) and the static-only cache.
    texture: Option<Texture>,
    cache: Option<Texture>,
    /// Per cascade: the key its cache layer was last written under (0 = invalid).
    cache_keys: [std::sync::atomic::AtomicU64; SHADOW_LAYERS],
    /// fork (perf): did the dynamic pass actually draw into this layer last frame?
    ///
    /// The whole point of the per-frame restore is to undo the previous frame's entity casters, so
    /// a layer nothing drew into still holds pure static depth and needs no restore at all. A
    /// `Depth32Float` copy is 2048² × 4 B = **16.8 MiB**, and five of them ran every settled frame
    /// — including the sky-occlusion layer, which has no dynamic pass at all (see the `is_sky`
    /// skip below) and so was restored every frame to undo nothing.
    ///
    /// Per LAYER and not per region, because that is the only granularity the API offers:
    /// wgpu-core's `validate_texture_copy_range` refuses a partial copy of a depth format on x/y
    /// ("Multisampled and depth-stencil formats do not support partial copies on x and y
    /// dimensions, but do support copying a subset of layers"), so the tempting
    /// copy-only-the-casters'-rectangle is not expressible. Skipping the layer wholesale is.
    layer_dirty: [std::sync::atomic::AtomicBool; SHADOW_LAYERS],
    offsets: Vec<u32>,
    /// fork: modern lighting (P6) — one caster view per cube face of every shadowed local light,
    /// and per atlas layer the key it currently holds (0 = redraw).
    local_offsets: Vec<u32>,
    local_keys: [std::sync::atomic::AtomicU64; crate::modern::shadow::MAX_LOCAL_SHADOW_LAYERS],
    bind: Option<BindGroup>,
    count: u32,
}

fn init_shadow_pipelines(
    mut commands: Commands,
    pipes: Res<GxPipelines>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/modern/static_gx_shadow.wgsl");
    let view_layout = BindGroupLayoutDescriptor::new(
        "modern_shadow_caster_view",
        &BindGroupLayoutEntries::single(ShaderStages::VERTEX, uniform_buffer::<CasterView>(true)),
    );
    let make = |label: &'static str, cutout: bool| RenderPipelineDescriptor {
        label: Some(label.into()),
        layout: vec![view_layout.clone(), pipes.cell_layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vertex".into()),
            buffers: vec![vertex_layout()],
        },
        // Opaque casters need no fragment stage; cutout ones alpha-key their texture.
        fragment: cutout.then(|| FragmentState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("fragment".into()),
            targets: vec![],
        }),
        // Two-sided: foliage cards and thin WMO walls must cast from either face.
        primitive: PrimitiveState {
            cull_mode: None,
            ..default()
        },
        depth_stencil: Some(DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: CompareFunction::GreaterEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        ..default()
    };
    commands.insert_resource(EntityShadowPipelines {
        view_layout: view_layout.clone(),
        draw_layout: BindGroupLayoutDescriptor::new(
            "modern_shadow_entity_draw",
            &BindGroupLayoutEntries::single(ShaderStages::VERTEX, uniform_buffer::<EntityDraw>(true)),
        ),
        skin_layout: BindGroupLayoutDescriptor::new(
            "modern_shadow_entity_skin",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (
                    uniform_buffer::<EntityDraw>(true),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        ),
        cut_layout: BindGroupLayoutDescriptor::new(
            "modern_shadow_entity_cutout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        ),
        shader: asset_server.load("embedded://benilla_world/shaders/modern/entity_shadow.wgsl"),
        by_key: StrideMap::default(),
    });
    commands.insert_resource(TerrainShadowPipelines {
        view_layout: view_layout.clone(),
        shader: asset_server.load("embedded://benilla_world/shaders/modern/terrain_shadow.wgsl"),
        by_stride: StrideMap::default(),
    });
    commands.insert_resource(GxShadowPipelines {
        opaque: pipeline_cache.queue_render_pipeline(make("modern_shadow_static", false)),
        cutout: pipeline_cache.queue_render_pipeline(make("modern_shadow_static_cutout", true)),
        view_layout,
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare_modern_shadow(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cascades: Option<Res<ModernCascades>>,
    lights: Option<Res<ModernLights>>,
    images: Res<RenderAssets<GpuImage>>,
    buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    pipes: Option<Res<GxShadowPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    mut uniforms: ResMut<CasterViewBuffer>,
    mut frame: ResMut<ModernShadowFrame>,
) {
    frame.count = 0;
    frame.bind = None;
    let (Some(cascades), Some(pipes)) = (cascades, pipes) else {
        return;
    };
    let Some(block) = buffers.get(MODERN_SHADOW_CASCADES.id()) else {
        return;
    };
    // Always uploaded, so receivers see `count == 0` in a classic frame.
    queue.write_buffer(&block.buffer, 0, &cascades.std430());
    if cascades.count == 0 {
        return;
    }
    let Some(image) = images.get(MODERN_SHADOW_MAP.id()) else {
        return;
    };
    // The array is re-created when the lighting latch re-sizes it (`modern::shadow::fit_shadow_map`):
    // every view, the static cache and every cached layer key belong to the old texture.
    if frame.texture.as_ref().is_some_and(|t| t.id() != image.texture.id()) {
        frame.layers.clear();
        frame.cache = None;
        for key in &frame.cache_keys {
            key.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        for key in &frame.local_keys {
            key.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    frame.texture = Some(image.texture.clone());
    if frame.cache.is_none() {
        frame.cache = Some(device.create_texture(&TextureDescriptor {
            label: Some("modern_shadow_static_cache"),
            size: Extent3d {
                width: image.texture.width(),
                height: image.texture.height(),
                depth_or_array_layers: image.texture.depth_or_array_layers(),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
            view_formats: &[],
        }));
    }
    if frame.layers.is_empty() {
        // **The texture's OWN depth, not [`SHADOW_LAYERS`].** P8 latched the array's depth to the
        // tier (`modern::shadow::fit_shadow_map`) so a session that shadows no local light stops
        // paying 16.8 MB a layer for an atlas it never samples — but this loop kept building views
        // for the largest depth any tier could ask for. On every tier below Cinematic, and on the
        // web for High and Ultra, that asked a five-layer texture for views of layers 5 and 6:
        //
        //   Texture view array layer range (baseArrayLayer: 5, arrayLayerCount: 1) exceeds the
        //   texture's array layer count (5).
        //
        // An invalid view invalidates the `modern_local_shadow` pass that attaches it, that pass's
        // command buffer, and the submit carrying it — so `static_gx c0 t0` and the pbr pipelines
        // all came up "invalid due to a previous error" and the screen went black. The readers
        // already index this through `.get()`, so a shorter list simply means "no atlas here",
        // which is what a tier with no local layers means.
        frame.layers = (0..image.texture.depth_or_array_layers())
            .map(|layer| {
                image.texture.create_view(&TextureViewDescriptor {
                    label: Some("modern_shadow_layer"),
                    dimension: Some(TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..default()
                })
            })
            .collect();
    }
    uniforms.0.clear();
    frame.offsets.clear();
    for i in 0..SHADOW_LAYERS {
        let clip_from_world = if i == SKY_LAYER {
            cascades.sky_clip
        } else if i < cascades.count as usize {
            cascades.clip_from_world[i]
        } else {
            Mat4::IDENTITY
        };
        let offset = uniforms.0.push(&CasterView { clip_from_world });
        frame.offsets.push(offset);
    }
    // fork: modern lighting (FORK.md, P6) — the shadowed local lights' cube
    // faces. Six 90° reverse-Z perspectives per light, whose convention `locallight::cube_face`
    // mirrors analytically (`modern::lights::cube_face_clip`).
    frame.local_offsets.clear();
    if let Some(lights) = lights.as_deref() {
        for light in &lights.shadowed {
            // A hole still gets its six offsets, so a slot's tiles are always at `slot * 6 + face`
            // however many lights are actually held — the identity the receivers' nibble depends on.
            let (pos, near) = light.map_or(
                (Vec3::ZERO, crate::modern::lights::CUBE_NEAR),
                |h| (h.light.pos, h.light.near),
            );
            for face in 0..6 {
                let clip_from_world = crate::modern::lights::cube_face_clip(pos, face, near);
                frame
                    .local_offsets
                    .push(uniforms.0.push(&CasterView { clip_from_world }));
            }
        }
    }
    uniforms.0.write_buffer(&device, &queue);
    let Some(binding) = uniforms.0.binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipes.view_layout);
    frame.bind = Some(device.create_bind_group(
        "modern_shadow_caster_view",
        &layout,
        &BindGroupEntries::single(binding),
    ));
    frame.count = cascades.count;
}

/// Queue a terrain caster pipeline for every vertex stride the resident cells use.
fn queue_terrain_shadow_pipelines(
    casters: Option<Res<ModernTerrainCasters>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    pipeline_cache: Res<PipelineCache>,
    pipes: Option<ResMut<TerrainShadowPipelines>>,
) {
    let (Some(casters), Some(mut pipes)) = (casters, pipes) else {
        return;
    };
    for (mesh_id, _, _) in &casters.cells {
        let Some(mesh) = meshes.get(*mesh_id) else {
            continue;
        };
        let stride = mesh.layout.0.layout().array_stride;
        if pipes.by_stride.contains_key(&stride) {
            continue;
        }
        let descriptor = RenderPipelineDescriptor {
            label: Some("modern_shadow_terrain".into()),
            layout: vec![pipes.view_layout.clone()],
            vertex: VertexState {
                shader: pipes.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("vertex".into()),
                buffers: vec![VertexBufferLayout {
                    array_stride: stride,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: None,
            primitive: PrimitiveState {
                cull_mode: None,
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            ..default()
        };
        let id = pipeline_cache.queue_render_pipeline(descriptor);
        pipes.by_stride.insert(stride, id);
    }
}

/// Build this frame's entity caster draws: pipelines (queued on first sight of a layout), one
/// uniform per part, cutout texture bind groups, and the two draw bind groups.
#[allow(clippy::too_many_arguments)]
fn prepare_entity_shadows(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    casters: Option<Res<ModernEntityCasters>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    images: Res<RenderAssets<GpuImage>>,
    light: Option<Res<crate::lighting::SharedLightBuffer>>,
    pipeline_cache: Res<PipelineCache>,
    pipes: Option<ResMut<EntityShadowPipelines>>,
    mut buffer: ResMut<EntityDrawBuffer>,
    mut cut_binds: ResMut<CutoutBinds>,
    mut frame: ResMut<EntityShadowFrame>,
) {
    frame.draws.clear();
    frame.draw_bind = None;
    frame.skin_bind = None;
    let (Some(casters), Some(light), Some(mut pipes)) = (casters, light, pipes) else {
        return;
    };
    if casters.parts.is_empty() {
        return;
    }
    buffer.0.clear();
    for part in &casters.parts {
        let Some(mesh) = meshes.get(part.mesh) else {
            continue;
        };
        let skinned = part.rig_slot.is_some();
        let cut_image = part.cutout.and_then(|id| images.get(id).map(|gpu| (id, gpu)));
        let mut cutout = cut_image.is_some();
        let key = (mesh.layout.clone(), skinned, cutout);
        let pipeline = match pipes.by_key.get(&key) {
            Some(id) => Some(*id),
            None => {
                let mut attrs = vec![Mesh::ATTRIBUTE_POSITION.at_shader_location(0)];
                if cutout {
                    attrs.push(Mesh::ATTRIBUTE_UV_0.at_shader_location(2));
                }
                if skinned {
                    attrs.push(benilla_assets::ATTRIBUTE_WOW_JOINT_INDEX.at_shader_location(10));
                    attrs.push(benilla_assets::ATTRIBUTE_WOW_JOINT_WEIGHT.at_shader_location(11));
                }
                match mesh.layout.0.get_layout(&attrs) {
                    Ok(vertex_layout) => {
                        let mut defs: Vec<ShaderDefVal> = Vec::new();
                        let mut layout = vec![
                            pipes.view_layout.clone(),
                            if skinned { pipes.skin_layout.clone() } else { pipes.draw_layout.clone() },
                        ];
                        if skinned {
                            defs.push("SKINNED".into());
                        }
                        if cutout {
                            defs.push("CUTOUT".into());
                            layout.push(pipes.cut_layout.clone());
                        }
                        let descriptor = RenderPipelineDescriptor {
                            label: Some("modern_shadow_entity".into()),
                            layout,
                            vertex: VertexState {
                                shader: pipes.shader.clone(),
                                shader_defs: defs.clone(),
                                entry_point: Some("vertex".into()),
                                buffers: vec![vertex_layout],
                            },
                            fragment: cutout.then(|| FragmentState {
                                shader: pipes.shader.clone(),
                                shader_defs: defs.clone(),
                                entry_point: Some("fragment".into()),
                                targets: vec![],
                            }),
                            primitive: PrimitiveState {
                                cull_mode: None,
                                ..default()
                            },
                            depth_stencil: Some(DepthStencilState {
                                format: CORE_3D_DEPTH_FORMAT,
                                depth_write_enabled: true,
                                depth_compare: CompareFunction::GreaterEqual,
                                stencil: StencilState::default(),
                                bias: DepthBiasState::default(),
                            }),
                            ..default()
                        };
                        let id = pipeline_cache.queue_render_pipeline(descriptor);
                        pipes.by_key.insert(key, id);
                        Some(id)
                    }
                    // A cutout part without UVs casts opaque; anything else missing is skipped.
                    Err(_) if cutout => {
                        cutout = false;
                        pipes.by_key.get(&(mesh.layout.clone(), skinned, false)).copied()
                    }
                    Err(_) => None,
                }
            }
        };
        let Some(pipeline) = pipeline else {
            continue;
        };
        let cutout_bind = match (cutout, cut_image) {
            (true, Some((id, gpu))) => {
                let view_id = gpu.texture_view.id();
                let stale = cut_binds.0.get(&id).is_none_or(|(v, _)| *v != view_id);
                if stale {
                    let layout = pipeline_cache.get_bind_group_layout(&pipes.cut_layout);
                    let bind = device.create_bind_group(
                        "modern_shadow_entity_cutout",
                        &layout,
                        &BindGroupEntries::sequential((&gpu.texture_view, &gpu.sampler)),
                    );
                    cut_binds.0.insert(id, (view_id, bind));
                }
                cut_binds.0.get(&id).map(|(_, b)| b.clone())
            }
            _ => None,
        };
        let offset = buffer.0.push(&EntityDraw {
            world_from_local: part.world_from_local,
            rig_slot: part.rig_slot.unwrap_or(0),
        });
        frame.draws.push(EntityDrawCmd {
            mesh: part.mesh,
            pipeline,
            offset,
            skinned,
            cutout: cutout_bind,
            center: part.center,
            radius: part.radius,
        });
    }
    // Drop the bind groups of cutout textures no caster uses any more: each holds its texture alive.
    if cut_binds.0.len() > 256 {
        let live: std::collections::HashSet<AssetId<Image>> =
            casters.parts.iter().filter_map(|p| p.cutout).collect();
        cut_binds.0.retain(|id, _| live.contains(id));
    }
    buffer.0.write_buffer(&device, &queue);
    let Some(binding) = buffer.0.binding() else {
        frame.draws.clear();
        return;
    };
    let draw_layout = pipeline_cache.get_bind_group_layout(&pipes.draw_layout);
    let skin_layout = pipeline_cache.get_bind_group_layout(&pipes.skin_layout);
    frame.draw_bind = Some(device.create_bind_group(
        "modern_shadow_entity_draw",
        &draw_layout,
        &BindGroupEntries::single(binding.clone()),
    ));
    frame.skin_bind = Some(device.create_bind_group(
        "modern_shadow_entity_skin",
        &skin_layout,
        &BindGroupEntries::sequential((binding, light.0.as_entire_binding())),
    ));
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct GxShadowLabel;

#[derive(Default)]
struct GxShadowNode;

impl ViewNode for GxShadowNode {
    type ViewQuery = &'static StaticGxView;

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        gx_view: QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        if !gx_view.primary {
            return Ok(());
        }
        let frame = world.resource::<ModernShadowFrame>();
        let Some(view_bind) = frame.bind.as_ref() else {
            return Ok(());
        };
        let cascades = world.resource::<ModernCascades>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let ready = world.get_resource::<GxShadowPipelines>().and_then(|pipes| {
            Some((
                pipeline_cache.get_render_pipeline(pipes.opaque)?,
                pipeline_cache.get_render_pipeline(pipes.cutout)?,
            ))
        });

        let gx = world.resource::<GxWorld>();
        let cache = world.resource::<GxGpuCache>();
        let meshes = world.resource::<RenderAssets<RenderMesh>>();
        let allocator = world.resource::<MeshAllocator>();
        let mut casters: Vec<(&GxCellGpu, &GxCellDraw)> = Vec::new();
        for (key, gpu) in &cache.cells {
            if let Some(draw) = gx.cells.get(key) {
                casters.push((gpu, draw));
            }
        }
        for (entity, gpu) in &cache.wmos {
            if let Some(draw) = gx.wmos.get(entity) {
                casters.push((gpu, draw));
            }
        }
        for (entity, gpu) in &cache.props {
            if let Some(draw) = gx.props.get(entity) {
                casters.push((gpu, draw));
            }
        }

        let terrain = world.get_resource::<ModernTerrainCasters>();
        let terrain_pipes = world.get_resource::<TerrainShadowPipelines>();
        // Order-independent fingerprint of the static caster set: cached layers are reused only
        // while it (and the cascade's matrix) is unchanged.
        let static_hash = {
            use std::hash::{Hash, Hasher};
            let item = |f: &dyn Fn(&mut std::collections::hash_map::DefaultHasher)| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                f(&mut h);
                h.finish()
            };
            let mut acc: u64 = 0x9e37_79b9_7f4a_7c15;
            for (gpu, draw) in &casters {
                acc = acc.wrapping_add(item(&|h| {
                    gpu.mesh.hash(h);
                    gpu.killed_applied.hash(h);
                    gpu.runs.len().hash(h);
                    draw.origin.to_array().map(f32::to_bits).hash(h);
                }));
            }
            if let Some(t) = terrain {
                for (mesh_id, center, _) in &t.cells {
                    acc = acc.wrapping_add(item(&|h| {
                        mesh_id.hash(h);
                        center.to_array().map(f32::to_bits).hash(h);
                    }));
                }
            }
            acc
        };
        let (Some(working), Some(static_cache)) = (frame.texture.as_ref(), frame.cache.as_ref()) else {
            return Ok(());
        };

        // The local-light atlas is the layer after the sky map and is drawn below, not here: it is
        // not a cascade, it is not cleared every frame, and no entity is drawn into it.
        for i in 0..frame.layers.len().min(SKY_LAYER + 1) {
            let layer = &frame.layers[i];
            // The last layer is the sky-occlusion map: static casters only, from straight above.
            let is_sky = i == SKY_LAYER;
            let active = if is_sky {
                cascades.count > 0
            } else {
                i < cascades.count as usize
            };
            let clip_from_world = if is_sky {
                cascades.sky_clip
            } else {
                cascades.clip_from_world[i]
            };
            let key = if active {
                cascade_key(&clip_from_world, static_hash)
            } else {
                0
            };
            let slot = &frame.cache_keys[i];
            let cached = active && key != 0 && slot.load(std::sync::atomic::Ordering::Relaxed) == key;

            if cached {
                // Only what the previous frame's entity casters dirtied needs putting back.
                if frame.layer_dirty[i].swap(false, std::sync::atomic::Ordering::Relaxed) {
                    copy_layer(render_context.command_encoder(), static_cache, working, i as u32);
                }
            } else {
                // Static pass: clear, then terrain and the static world.
                let mut complete = ready.is_some();
                {
                    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                        label: Some("modern_shadow_static"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                            view: layer,
                            depth_ops: Some(Operations {
                                load: LoadOp::Clear(0.0),
                                store: StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    if active {
                        pass.set_bind_group(0, view_bind, &[frame.offsets[i]]);
                        if let (Some(t), Some(tp)) = (terrain, terrain_pipes) {
                            for (mesh_id, center, half) in &t.cells {
                                if outside_box(&clip_from_world, *center, *half) {
                                    continue;
                                }
                                let Some(mesh) = meshes.get(*mesh_id) else {
                                    complete = false;
                                    continue;
                                };
                                let Some(pipeline) = tp
                                    .by_stride
                                    .get(&mesh.layout.0.layout().array_stride)
                                    .and_then(|id| pipeline_cache.get_render_pipeline(*id))
                                else {
                                    complete = false;
                                    continue;
                                };
                                let (Some(vslice), Some(islice)) = (
                                    allocator.mesh_vertex_slice(mesh_id),
                                    allocator.mesh_index_slice(mesh_id),
                                ) else {
                                    complete = false;
                                    continue;
                                };
                                let bevy::render::mesh::RenderMeshBufferInfo::Indexed { index_format, .. } =
                                    &mesh.buffer_info
                                else {
                                    continue;
                                };
                                pass.set_render_pipeline(pipeline);
                                pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                                pass.set_index_buffer(islice.buffer.slice(..), *index_format);
                                pass.draw_indexed(
                                    islice.range.clone(),
                                    i32::try_from(vslice.range.start).unwrap_or(0),
                                    0..1,
                                );
                            }
                        } else {
                            complete = false;
                        }
                        if let Some((opaque, cutout)) = ready {
                            for (gpu, draw) in &casters {
                                if outside_cascade(&clip_from_world, draw) {
                                    continue;
                                }
                                let Some(mesh) = meshes.get(draw.mesh.id()) else {
                                    complete = false;
                                    continue;
                                };
                                let (Some(vslice), Some(islice)) = (
                                    allocator.mesh_vertex_slice(&draw.mesh.id()),
                                    allocator.mesh_index_slice(&draw.mesh.id()),
                                ) else {
                                    complete = false;
                                    continue;
                                };
                                let index_format = match &mesh.buffer_info {
                                    bevy::render::mesh::RenderMeshBufferInfo::Indexed { index_format, .. } => {
                                        *index_format
                                    }
                                    bevy::render::mesh::RenderMeshBufferInfo::NonIndexed => continue,
                                };
                                pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                                pass.set_index_buffer(islice.buffer.slice(..), index_format);
                                for run in &gpu.runs {
                                    pass.set_render_pipeline(if run.cutout { cutout } else { opaque });
                                    pass.set_bind_group(1, &gpu.bind_groups[run.slot].1, &[]);
                                    pass.draw_indexed(
                                        (islice.range.start + run.index_range.start)
                                            ..(islice.range.start + run.index_range.end),
                                        i32::try_from(vslice.range.start).unwrap_or(0),
                                        0..1,
                                    );
                                }
                            }
                        }
                    }
                }
                // Either way the layer was just cleared and redrawn, so nothing of the previous
                // frame's dynamic casters survives in it.
                frame.layer_dirty[i].store(false, std::sync::atomic::Ordering::Relaxed);
                if active && complete {
                    copy_layer(render_context.command_encoder(), working, static_cache, i as u32);
                    slot.store(key, std::sync::atomic::Ordering::Relaxed);
                } else {
                    slot.store(0, std::sync::atomic::Ordering::Relaxed);
                }
            }

            // Dynamic pass, every frame: entity parts on top of the (cached or fresh) static layer.
            if !active || is_sky {
                continue;
            }
            let Some(entities) = world.get_resource::<EntityShadowFrame>() else {
                continue;
            };
            let (Some(draw_bind), Some(skin_bind)) = (&entities.draw_bind, &entities.skin_bind) else {
                continue;
            };
            if entities.draws.is_empty() {
                continue;
            }
            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("modern_shadow_dynamic"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: layer,
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, view_bind, &[frame.offsets[i]]);
            // Whether anything is actually ISSUED, not whether the list was non-empty: every
            // command is still cascade-culled below, so a far cascade can hold a full draw list
            // and take none of it — and a layer nothing wrote needs no restore next frame.
            let mut drew = false;
            for cmd in &entities.draws {
                if outside_box(&clip_from_world, cmd.center, Vec3::splat(cmd.radius)) {
                    continue;
                }
                let Some(pipeline) = pipeline_cache.get_render_pipeline(cmd.pipeline) else {
                    continue;
                };
                let Some(mesh) = meshes.get(cmd.mesh) else {
                    continue;
                };
                let Some(vslice) = allocator.mesh_vertex_slice(&cmd.mesh) else {
                    continue;
                };
                pass.set_render_pipeline(pipeline);
                pass.set_bind_group(1, if cmd.skinned { skin_bind } else { draw_bind }, &[cmd.offset]);
                if let Some(cut) = &cmd.cutout {
                    pass.set_bind_group(2, cut, &[]);
                }
                pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                match &mesh.buffer_info {
                    bevy::render::mesh::RenderMeshBufferInfo::Indexed { index_format, .. } => {
                        let Some(islice) = allocator.mesh_index_slice(&cmd.mesh) else {
                            continue;
                        };
                        pass.set_index_buffer(islice.buffer.slice(..), *index_format);
                        pass.draw_indexed(
                            islice.range.clone(),
                            i32::try_from(vslice.range.start).unwrap_or(0),
                            0..1,
                        );
                        drew = true;
                    }
                    bevy::render::mesh::RenderMeshBufferInfo::NonIndexed => {
                        pass.draw(vslice.range.clone(), 0..1);
                        drew = true;
                    }
                }
            }
            if drew {
                frame.layer_dirty[i].store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }

        // fork: modern lighting (FORK.md, P6) — the shadowed local lights' cube
        // atlas. Every face is static content lit by a light that does not move, so the whole layer
        // is redrawn only when the chosen set or the static world changes; a settled room pays
        // nothing at all after the first frame.
        self.draw_local_atlas(
            render_context,
            world,
            frame,
            &casters,
            static_hash,
            ready,
            pipeline_cache,
            meshes,
            allocator,
        );
        Ok(())
    }
}

impl GxShadowNode {
    /// fork: modern lighting (P6, extended by P8) — the shadowed local lights' cube atlas.
    ///
    /// **P6** drew only the retained static world here, once, and cached the whole layer against the
    /// chosen set and the static fingerprint: a torch does not move, a wall does not move, and a
    /// settled room paid nothing after its first frame.
    ///
    /// **P8** keeps exactly that for the static half and adds two things the tier is named for:
    /// - up to **two layers** of six lights, so twelve lamps can shadow at once;
    /// - an optional **dynamic pass** (`LightTier::dynamic_casters`) that draws entity casters on top
    ///   each frame, so a character beside a lamp blocks it and a glowing weapon is occluded by the
    ///   arm holding it. That is what costs the layer its free ride: like a cascade, it is restored
    ///   from the static cache before the entities are redrawn — and, like a cascade, only when the
    ///   previous frame actually drew something into it.
    ///
    /// **The culling is per FACE, not per light.** A cube's six faces are six render passes' worth
    /// of state, and most of them look at nothing: a sconce against a wall has three faces that see
    /// only the wall, and a lamp on open ground has one that sees only sky. Each face's caster list
    /// is gathered first; a face with no casters is cleared and skipped entirely, which is cheaper
    /// than the viewport, scissor and bind-group churn of drawing nothing.
    #[allow(clippy::too_many_arguments)]
    fn draw_local_atlas(
        &self,
        render_context: &mut RenderContext<'_>,
        world: &World,
        frame: &ModernShadowFrame,
        casters: &[(&GxCellGpu, &GxCellDraw)],
        static_hash: u64,
        ready: Option<(&RenderPipeline, &RenderPipeline)>,
        pipeline_cache: &PipelineCache,
        meshes: &RenderAssets<RenderMesh>,
        allocator: &MeshAllocator,
    ) {
        let (Some(lights), Some(view_bind)) = (world.get_resource::<ModernLights>(), frame.bind.as_ref())
        else {
            return;
        };
        let Some((opaque, cutout)) = ready else {
            return;
        };
        if lights.shadowed.is_empty() || frame.local_offsets.len() < lights.shadowed.len() * 6 {
            return;
        }
        let Some(cache) = frame.cache.as_ref() else {
            return;
        };
        let Some(working) = frame.texture.as_ref() else {
            return;
        };
        // The tile's stride in the atlas, and the square of it a face is actually drawn into
        // (`LightTier::tile_use` — Cinematic draws half the edge, a quarter of the depth).
        let tile = (
            SHADOW_MAP_SIZE / ATLAS_TILES,
            crate::modern::lights::face_px(&lights.tier, SHADOW_MAP_SIZE),
        );
        let dynamic = lights.tier.dynamic_casters;
        let entities = dynamic
            .then(|| world.get_resource::<EntityShadowFrame>())
            .flatten();

        for layer_index in 0..lights.tier.atlas_layers() {
            let slots = layer_index * PER_ATLAS_LAYER
                ..((layer_index + 1) * PER_ATLAS_LAYER).min(lights.shadowed.len());
            if slots.is_empty() {
                continue;
            }
            let array_layer = LOCAL_LAYER + layer_index;
            let Some(view) = frame.layers.get(array_layer) else {
                continue;
            };
            let key = local_atlas_key(&lights.shadowed[slots.clone()], static_hash);
            let slot_key = &frame.local_keys[layer_index];
            let dirty_slot = &frame.layer_dirty[array_layer];
            let cached = slot_key.load(std::sync::atomic::Ordering::Relaxed) == key;

            if cached {
                // Nothing static changed. Put back only what the previous frame's entities dirtied;
                // a layer nothing drew into still holds pure static depth.
                if !dynamic {
                    continue;
                }
                if dirty_slot.swap(false, std::sync::atomic::Ordering::Relaxed) {
                    copy_layer(
                        render_context.command_encoder(),
                        cache,
                        working,
                        array_layer as u32,
                    );
                }
            } else {
                let complete = self.draw_atlas_statics(
                    render_context,
                    frame,
                    view,
                    view_bind,
                    &lights.shadowed[slots.clone()],
                    slots.start,
                    tile,
                    casters,
                    (opaque, cutout),
                    meshes,
                    allocator,
                );
                // The layer was just cleared and redrawn, so no entity of the previous frame
                // survives in it.
                dirty_slot.store(false, std::sync::atomic::Ordering::Relaxed);
                if complete {
                    copy_layer(
                        render_context.command_encoder(),
                        working,
                        cache,
                        array_layer as u32,
                    );
                    slot_key.store(key, std::sync::atomic::Ordering::Relaxed);
                } else {
                    slot_key.store(0, std::sync::atomic::Ordering::Relaxed);
                    // An incomplete static layer must not be restored from a cache that was never
                    // written; the next frame redraws it whole anyway.
                    continue;
                }
            }

            if let Some(entities) = entities {
                if self.draw_atlas_entities(
                    render_context,
                    frame,
                    view,
                    view_bind,
                    entities,
                    &lights.shadowed[slots.clone()],
                    slots.start,
                    tile,
                    pipeline_cache,
                    meshes,
                    allocator,
                ) {
                    dirty_slot.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }

    /// The static half of one atlas layer: clear it, then replay the retained world into each face
    /// that has anything to see. Returns whether every caster it wanted was resident — an
    /// incomplete layer is not cached, so it is redrawn until it is whole.
    #[allow(clippy::too_many_arguments)]
    fn draw_atlas_statics(
        &self,
        render_context: &mut RenderContext<'_>,
        frame: &ModernShadowFrame,
        view: &TextureView,
        view_bind: &BindGroup,
        shadowed: &[Option<crate::modern::lights::HeldSlot>],
        slot_base: usize,
        tile: (u32, u32),
        casters: &[(&GxCellGpu, &GxCellDraw)],
        pipes: (&RenderPipeline, &RenderPipeline),
        meshes: &RenderAssets<RenderMesh>,
        allocator: &MeshAllocator,
    ) -> bool {
        let (opaque, cutout) = pipes;
        let mut complete = true;
        let mut near: Vec<&(&GxCellGpu, &GxCellDraw)> = Vec::new();
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("modern_local_shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        for (i, light) in shadowed.iter().enumerate() {
            // A hole's six tiles keep the clear the pass just laid down, which reads as "nothing
            // occludes" — exactly what an unheld slot should mean.
            let Some(light) = light.map(|h| h.light) else { continue };
            let slot = slot_base + i;
            // Sphere, not frustum: a cube face's clip is a perspective, and the planar corner test
            // the cascades use flips sign behind the light. A torch reaches a handful of cells, so
            // the six faces share one cheap list, which is then split by face below.
            let reach = light.range;
            // One sphere test per caster, reused by all six faces — the P6 list, kept as a borrowed
            // slice rather than collected: this path allocated nothing before P8 and still does not.
            near.clear();
            near.extend(casters.iter().filter(|(_, draw)| {
                let center = draw.origin + Vec3::from(draw.aabb.center);
                let radius = Vec3::from(draw.aabb.half_extents).length();
                center.distance(light.pos) <= reach + radius
            }));
            for face in 0..6usize {
                let t = slot * 6 + face;
                let Some(&offset) = frame.local_offsets.get(t) else {
                    complete = false;
                    continue;
                };
                // fork: P8 — the per-face cull, and the reason the tile is bound LAZILY: a caster
                // reaches this face only if its box touches the face's 90° cone, and a face nothing
                // touches must not pay for a viewport, a scissor and a bind group to draw nothing.
                // A sconce flat against a wall has three such faces.
                let mut set_up = false;
                for (gpu, draw) in near.iter() {
                    if !face_sees(
                        face,
                        light.pos,
                        draw.origin + Vec3::from(draw.aabb.center),
                        Vec3::from(draw.aabb.half_extents).length(),
                    ) {
                        continue;
                    }
                    let Some(mesh) = meshes.get(draw.mesh.id()) else {
                        complete = false;
                        continue;
                    };
                    let (Some(vslice), Some(islice)) = (
                        allocator.mesh_vertex_slice(&draw.mesh.id()),
                        allocator.mesh_index_slice(&draw.mesh.id()),
                    ) else {
                        complete = false;
                        continue;
                    };
                    let index_format = match &mesh.buffer_info {
                        bevy::render::mesh::RenderMeshBufferInfo::Indexed { index_format, .. } => {
                            *index_format
                        }
                        bevy::render::mesh::RenderMeshBufferInfo::NonIndexed => continue,
                    };
                    if !set_up {
                        set_atlas_tile(&mut pass, slot, face, tile);
                        pass.set_bind_group(0, view_bind, &[offset]);
                        set_up = true;
                    }
                    pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                    pass.set_index_buffer(islice.buffer.slice(..), index_format);
                    for run in &gpu.runs {
                        pass.set_render_pipeline(if run.cutout { cutout } else { opaque });
                        pass.set_bind_group(1, &gpu.bind_groups[run.slot].1, &[]);
                        pass.draw_indexed(
                            (islice.range.start + run.index_range.start)
                                ..(islice.range.start + run.index_range.end),
                            i32::try_from(vslice.range.start).unwrap_or(0),
                            0..1,
                        );
                    }
                }
            }
        }
        complete
    }

    /// fork: P8 — the dynamic half: the same entity caster list the cascades draw, on top of the
    /// restored static layer. Returns whether anything was actually issued, which is what decides
    /// whether the layer needs restoring next frame.
    #[allow(clippy::too_many_arguments)]
    fn draw_atlas_entities(
        &self,
        render_context: &mut RenderContext<'_>,
        frame: &ModernShadowFrame,
        view: &TextureView,
        view_bind: &BindGroup,
        entities: &EntityShadowFrame,
        shadowed: &[Option<crate::modern::lights::HeldSlot>],
        slot_base: usize,
        tile: (u32, u32),
        pipeline_cache: &PipelineCache,
        meshes: &RenderAssets<RenderMesh>,
        allocator: &MeshAllocator,
    ) -> bool {
        let (Some(draw_bind), Some(skin_bind)) = (&entities.draw_bind, &entities.skin_bind) else {
            return false;
        };
        if entities.draws.is_empty() {
            return false;
        }
        let mut drew = false;
        let mut near: Vec<&EntityDrawCmd> = Vec::new();
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("modern_local_shadow_dynamic"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        for (i, light) in shadowed.iter().enumerate() {
            let Some(light) = light.map(|h| h.light) else { continue };
            let slot = slot_base + i;
            // The lamp's own reach first — a creature across the room casts nothing into this cube —
            // and the face cone second, which is where most of the list goes.
            near.clear();
            near.extend(
                entities
                    .draws
                    .iter()
                    .filter(|cmd| cmd.center.distance(light.pos) <= light.range + cmd.radius),
            );
            if near.is_empty() {
                continue;
            }
            for face in 0..6usize {
                let t = slot * 6 + face;
                let Some(&offset) = frame.local_offsets.get(t) else {
                    continue;
                };
                let mut set_up = false;
                for cmd in &near {
                    if !face_sees(face, light.pos, cmd.center, cmd.radius) {
                        continue;
                    }
                    let Some(pipeline) = pipeline_cache.get_render_pipeline(cmd.pipeline) else {
                        continue;
                    };
                    let Some(mesh) = meshes.get(cmd.mesh) else {
                        continue;
                    };
                    let Some(vslice) = allocator.mesh_vertex_slice(&cmd.mesh) else {
                        continue;
                    };
                    if !set_up {
                        set_atlas_tile(&mut pass, slot, face, tile);
                        pass.set_bind_group(0, view_bind, &[offset]);
                        set_up = true;
                    }
                    pass.set_render_pipeline(pipeline);
                    pass.set_bind_group(
                        1,
                        if cmd.skinned { skin_bind } else { draw_bind },
                        &[cmd.offset],
                    );
                    if let Some(cut) = &cmd.cutout {
                        pass.set_bind_group(2, cut, &[]);
                    }
                    pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                    match &mesh.buffer_info {
                        bevy::render::mesh::RenderMeshBufferInfo::Indexed { index_format, .. } => {
                            let Some(islice) = allocator.mesh_index_slice(&cmd.mesh) else {
                                continue;
                            };
                            pass.set_index_buffer(islice.buffer.slice(..), *index_format);
                            pass.draw_indexed(
                                islice.range.clone(),
                                i32::try_from(vslice.range.start).unwrap_or(0),
                                0..1,
                            );
                            drew = true;
                        }
                        bevy::render::mesh::RenderMeshBufferInfo::NonIndexed => {
                            pass.draw(vslice.range.clone(), 0..1);
                            drew = true;
                        }
                    }
                }
            }
        }
        drew
    }
}

/// Point one render pass at one atlas tile — the viewport and the scissor together, because the
/// scissor is what keeps a face's filter taps out of its neighbour.
fn set_atlas_tile(
    pass: &mut bevy::render::render_phase::TrackedRenderPass<'_>,
    slot: usize,
    face: usize,
    (tile, drawn): (u32, u32),
) {
    let t = (slot % PER_ATLAS_LAYER) * 6 + face;
    let (tx, ty) = (t as u32 % ATLAS_TILES, t as u32 / ATLAS_TILES);
    // The face is drawn into the tile's top-left `drawn` texels (the whole tile on every tier but
    // Cinematic); the lookup scales its uv by the same fraction (`locallight::tile_use`).
    pass.set_viewport(
        (tx * tile) as f32,
        (ty * tile) as f32,
        drawn as f32,
        drawn as f32,
        0.0,
        1.0,
    );
    pass.set_scissor_rect(tx * tile, ty * tile, drawn, drawn);
}

/// fork: P8 — **does one cube face see this sphere at all?**
///
/// A face is the 90° cone about its major axis, so a sphere is visible from it iff the sphere
/// reaches past the two diagonal planes that bound that cone. Testing the sphere's centre against
/// the plane `axis·d >= |other|` — widened by the radius over √2, the planes' own normalisation —
/// is exact for a sphere and costs three multiplies and a compare.
///
/// This is the cheap half of the aggressive culling: a sconce flat against a wall has three faces
/// that see nothing, and a face that sees nothing is a pass that never binds anything.
fn face_sees(face: usize, light: Vec3, center: Vec3, radius: f32) -> bool {
    let d = center - light;
    let (along, a, b) = match face {
        0 => (d.x, d.y, d.z),
        1 => (-d.x, d.y, d.z),
        2 => (d.y, d.x, d.z),
        3 => (-d.y, d.x, d.z),
        4 => (d.z, d.x, d.y),
        _ => (-d.z, d.x, d.y),
    };
    // The cone's two bounding planes have normals `(axis ∓ other)/√2`; a sphere clears both iff
    // `along + radius·√2 >= |a|` and the same for `b`.
    let slack = radius * std::f32::consts::SQRT_2;
    along + slack >= a.abs() && along + slack >= b.abs()
}

/// One atlas LAYER's key: the shadowed lights it holds, as chosen, plus the static caster
/// fingerprint. Never 0, which is the "redraw me" value. Per layer and not per atlas since P8, so a
/// light entering the second layer does not invalidate the first layer's six.
fn local_atlas_key(
    shadowed: &[Option<crate::modern::lights::HeldSlot>],
    static_hash: u64,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for light in shadowed {
        // The hole is hashed too: a slot emptying has to invalidate the layer, or its tiles keep
        // the departed light's depth and the next light to take that slot inherits it.
        match light {
            // The FADE is deliberately not hashed: it changes every frame while a slot ramps and
            // the tiles do not depend on it, so hashing it would redraw six faces a frame for every
            // cross-fade — exactly the cost this cache exists to avoid.
            Some(l) => {
                1u8.hash(&mut h);
                l.light.pos.to_array().map(f32::to_bits).hash(&mut h);
                l.light.range.to_bits().hash(&mut h);
                l.light.near.to_bits().hash(&mut h);
            }
            None => 0u8.hash(&mut h),
        }
    }
    static_hash.hash(&mut h);
    h.finish().max(1)
}

/// The cache key of one cascade: its light clip and the static caster fingerprint (never 0).
fn cascade_key(clip_from_world: &Mat4, static_hash: u64) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    clip_from_world.to_cols_array().map(f32::to_bits).hash(&mut h);
    static_hash.hash(&mut h);
    h.finish().max(1)
}

/// Copy one whole array layer between two same-sized `Depth32Float` arrays.
fn copy_layer(encoder: &mut CommandEncoder, src: &Texture, dst: &Texture, layer: u32) {
    let size = Extent3d {
        width: src.width(),
        height: src.height(),
        depth_or_array_layers: 1,
    };
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: src,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: layer },
            aspect: TextureAspect::All,
        },
        TexelCopyTextureInfo {
            texture: dst,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: layer },
            aspect: TextureAspect::All,
        },
        size,
    );
}

/// Does a cell's world box miss the cascade's light-space rectangle? Depth is not tested: a caster
/// between the light and the near plane still casts (the shader pancakes it).
fn outside_cascade(clip_from_world: &Mat4, draw: &GxCellDraw) -> bool {
    // `aabb` is relative to the bake's recentring origin (`bake.rs`).
    outside_box(
        clip_from_world,
        draw.origin + Vec3::from(draw.aabb.center),
        Vec3::from(draw.aabb.half_extents),
    )
}

/// Does a world box miss the cascade's light-space rectangle? (Depth untested, as above.)
fn outside_box(clip_from_world: &Mat4, center: Vec3, half: Vec3) -> bool {
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for i in 0..8u8 {
        let sign = Vec3::new(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        );
        let p = clip_from_world.project_point3(center + half * sign);
        lo = lo.min(p.truncate());
        hi = hi.max(p.truncate());
    }
    hi.x < -1.0 || lo.x > 1.0 || hi.y < -1.0 || lo.y > 1.0
}

/// Wire the pass before the main pass (called from `render::build`).
pub(super) fn build(render_app: &mut SubApp) {
    render_app
        .init_resource::<CasterViewBuffer>()
        .init_resource::<EntityDrawBuffer>()
        .init_resource::<CutoutBinds>()
        .init_resource::<EntityShadowFrame>()
        .init_resource::<ModernShadowFrame>()
        .add_systems(
            bevy::render::RenderStartup,
            init_shadow_pipelines.after(init_pipelines),
        )
        .add_systems(
            Render,
            (prepare_modern_shadow, queue_terrain_shadow_pipelines, prepare_entity_shadows)
                .in_set(RenderSystems::PrepareResources),
        )
        .add_render_graph_node::<ViewNodeRunner<GxShadowNode>>(Core3d, GxShadowLabel)
        .add_render_graph_edges(
            Core3d,
            (NodePbr::LateShadowPass, GxShadowLabel, Node3d::StartMainPass),
        );
}
