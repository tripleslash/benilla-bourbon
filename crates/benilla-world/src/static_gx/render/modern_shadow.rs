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
//! Terrain and entity casters, and static caching, follow in later P2 steps.

use bevy::app::SubApp;
use bevy::pbr::graph::NodePbr;
use bevy::render::render_resource::binding_types::uniform_buffer;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::storage::GpuShaderStorageBuffer;
use bevy::render::texture::GpuImage;
use bevy::shader::Shader;

use benilla_assets::materials::{MODERN_SHADOW_CASCADES, MODERN_SHADOW_MAP};

use super::*;
use crate::modern::shadow::{ModernCascades, ModernTerrainCasters, MAX_CASCADES};
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
    offsets: Vec<u32>,
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
    if frame.layers.is_empty() {
        frame.layers = (0..MAX_CASCADES as u32)
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
    for i in 0..MAX_CASCADES {
        let clip_from_world = if i < cascades.count as usize {
            cascades.clip_from_world[i]
        } else {
            Mat4::IDENTITY
        };
        let offset = uniforms.0.push(&CasterView { clip_from_world });
        frame.offsets.push(offset);
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

        for (i, layer) in frame.layers.iter().enumerate() {
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
            if i >= cascades.count as usize {
                continue; // cleared, unused this tier
            }
            pass.set_bind_group(0, view_bind, &[frame.offsets[i]]);
            let clip_from_world = cascades.clip_from_world[i];
            // Terrain first: hills and cliffs cast onto everything below them.
            if let (Some(terrain), Some(tp)) = (
                world.get_resource::<ModernTerrainCasters>(),
                world.get_resource::<TerrainShadowPipelines>(),
            ) {
                for (mesh_id, center, half) in &terrain.cells {
                    if outside_box(&clip_from_world, *center, *half) {
                        continue;
                    }
                    let Some(mesh) = meshes.get(*mesh_id) else {
                        continue;
                    };
                    let Some(pipeline) = tp
                        .by_stride
                        .get(&mesh.layout.0.layout().array_stride)
                        .and_then(|id| pipeline_cache.get_render_pipeline(*id))
                    else {
                        continue;
                    };
                    let (Some(vslice), Some(islice)) = (
                        allocator.mesh_vertex_slice(mesh_id),
                        allocator.mesh_index_slice(mesh_id),
                    ) else {
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
            }
            let Some((opaque, cutout)) = ready else {
                continue;
            };
            for (gpu, draw) in &casters {
                if outside_cascade(&clip_from_world, draw) {
                    continue;
                }
                let Some(mesh) = meshes.get(draw.mesh.id()) else {
                    continue;
                };
                let (Some(vslice), Some(islice)) = (
                    allocator.mesh_vertex_slice(&draw.mesh.id()),
                    allocator.mesh_index_slice(&draw.mesh.id()),
                ) else {
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
        Ok(())
    }
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
        .init_resource::<ModernShadowFrame>()
        .add_systems(
            bevy::render::RenderStartup,
            init_shadow_pipelines.after(init_pipelines),
        )
        .add_systems(
            Render,
            (prepare_modern_shadow, queue_terrain_shadow_pipelines)
                .in_set(RenderSystems::PrepareResources),
        )
        .add_render_graph_node::<ViewNodeRunner<GxShadowNode>>(Core3d, GxShadowLabel)
        .add_render_graph_edges(
            Core3d,
            (NodePbr::LateShadowPass, GxShadowLabel, Node3d::StartMainPass),
        );
}
