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
use crate::modern::lights::{ModernLights, ATLAS_TILES};
use crate::modern::shadow::{
    ModernCascades, ModernEntityCasters, ModernTerrainCasters, LOCAL_LAYER, LOCAL_SHADOW_LAYERS,
    SHADOW_LAYERS, SHADOW_MAP_SIZE, SKY_LAYER,
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
    offsets: Vec<u32>,
    /// fork: modern lighting (P6) — one caster view per cube face of every shadowed local light,
    /// and the key the atlas layer currently holds (0 = redraw).
    local_offsets: Vec<u32>,
    local_key: std::sync::atomic::AtomicU64,
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
        frame.layers = (0..SHADOW_LAYERS as u32)
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
    if LOCAL_SHADOW_LAYERS > 0 {
        if let Some(lights) = lights.as_deref() {
            for light in &lights.shadowed {
                for face in 0..6 {
                    let clip_from_world = crate::modern::lights::cube_face_clip(light.pos, face);
                    frame
                        .local_offsets
                        .push(uniforms.0.push(&CasterView { clip_from_world }));
                }
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
                copy_layer(render_context.command_encoder(), static_cache, working, i as u32);
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
                    }
                    bevy::render::mesh::RenderMeshBufferInfo::NonIndexed => {
                        pass.draw(vslice.range.clone(), 0..1);
                    }
                }
            }
        }

        // fork: modern lighting (FORK.md, P6) — the shadowed local lights' cube
        // atlas. Every face is static content lit by a light that does not move, so the whole layer
        // is redrawn only when the chosen set or the static world changes; a settled room pays
        // nothing at all after the first frame.
        self.draw_local_atlas(render_context, world, frame, &casters, static_hash, ready, meshes, allocator);
        Ok(())
    }
}

impl GxShadowNode {
    #[allow(clippy::too_many_arguments)]
    fn draw_local_atlas(
        &self,
        render_context: &mut RenderContext<'_>,
        world: &World,
        frame: &ModernShadowFrame,
        casters: &[(&GxCellGpu, &GxCellDraw)],
        static_hash: u64,
        ready: Option<(&RenderPipeline, &RenderPipeline)>,
        meshes: &RenderAssets<RenderMesh>,
        allocator: &MeshAllocator,
    ) {
        if LOCAL_SHADOW_LAYERS == 0 {
            return;
        }
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
        let Some(layer) = frame.layers.get(LOCAL_LAYER) else {
            return;
        };
        let key = local_atlas_key(&lights.shadowed, static_hash);
        if frame.local_key.load(std::sync::atomic::Ordering::Relaxed) == key {
            return;
        }
        let tile = SHADOW_MAP_SIZE / ATLAS_TILES;
        let mut complete = true;
        {
            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("modern_local_shadow"),
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
            for (slot, light) in lights.shadowed.iter().enumerate() {
                // Sphere, not frustum: a cube face's clip is a perspective, and the planar corner
                // test the cascades use flips sign behind the light. A torch reaches a handful of
                // cells, so the six faces share one cheap list.
                let reach = light.range;
                for face in 0..6usize {
                    let t = slot * 6 + face;
                    let (tx, ty) = (t as u32 % ATLAS_TILES, t as u32 / ATLAS_TILES);
                    pass.set_viewport(
                        (tx * tile) as f32,
                        (ty * tile) as f32,
                        tile as f32,
                        tile as f32,
                        0.0,
                        1.0,
                    );
                    pass.set_scissor_rect(tx * tile, ty * tile, tile, tile);
                    pass.set_bind_group(0, view_bind, &[frame.local_offsets[t]]);
                    for (gpu, draw) in casters {
                        let center = draw.origin + Vec3::from(draw.aabb.center);
                        let radius = Vec3::from(draw.aabb.half_extents).length();
                        if center.distance(light.pos) > reach + radius {
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
        frame.local_key.store(
            if complete { key } else { 0 },
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// The atlas layer's key: the shadowed lights as chosen plus the static caster fingerprint. Never 0,
/// which is the "redraw me" value.
fn local_atlas_key(shadowed: &[crate::modern::lights::ShadowLight], static_hash: u64) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for light in shadowed {
        light.pos.to_array().map(f32::to_bits).hash(&mut h);
        light.range.to_bits().hash(&mut h);
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
