//! **The retained static world as shadow casters** (fork-owned; `FORK.md` P2a).
//!
//! A child of `static_gx::render` so it can reuse that module's GPU state as-is: the baked cell and
//! WMO vertex buffers, the per-cell bind groups (texture array, record table with kill bits) and the
//! draw runs. For every cascade view of the modern sun (`modern::shadow`) it draws every cached
//! cell, WMO and prop whose world bounds overlap the cascade, depth-only, after Bevy's own shadow
//! pass and into the same depth attachment — which the existing receivers already sample.
//!
//! Deliberately P2a-simple: every cascade is re-drawn each frame, and a cell casts only once it
//! has been visible and cached. P2b replaces this with cached static cascades.

use bevy::app::SubApp;
use bevy::pbr::graph::NodePbr;
use bevy::pbr::{LightEntity, ShadowView, ViewLightEntities};
use bevy::render::render_resource::binding_types::uniform_buffer;
use bevy::render::view::{ViewUniform, ViewUniforms};
use bevy::shader::Shader;

use super::*;

#[derive(Resource)]
struct GxShadowPipelines {
    view_layout: BindGroupLayoutDescriptor,
    opaque: CachedRenderPipelineId,
    cutout: CachedRenderPipelineId,
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
        "static_gx_shadow_view",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
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
    commands.insert_resource(GxShadowPipelines {
        opaque: pipeline_cache.queue_render_pipeline(make("static_gx_shadow", false)),
        cutout: pipeline_cache.queue_render_pipeline(make("static_gx_shadow_cutout", true)),
        view_layout,
    });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct GxShadowLabel;

#[derive(Default)]
struct GxShadowNode;

impl ViewNode for GxShadowNode {
    type ViewQuery = (&'static ViewLightEntities, &'static StaticGxView);

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (lights, gx_view): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        if !gx_view.primary || !crate::modern::frame_is_modern() || lights.lights.is_empty() {
            return Ok(());
        }
        let Some(pipes) = world.get_resource::<GxShadowPipelines>() else {
            return Ok(());
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let (Some(opaque), Some(cutout)) = (
            pipeline_cache.get_render_pipeline(pipes.opaque),
            pipeline_cache.get_render_pipeline(pipes.cutout),
        ) else {
            return Ok(());
        };
        let Some(view_binding) = world.resource::<ViewUniforms>().uniforms.binding() else {
            return Ok(());
        };
        let layout = pipeline_cache.get_bind_group_layout(&pipes.view_layout);
        let view_bind = render_context.render_device().create_bind_group(
            "static_gx_shadow_view",
            &layout,
            &BindGroupEntries::single(view_binding),
        );

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
        if casters.is_empty() {
            return Ok(());
        }

        for &light_view in &lights.lights {
            let Ok(entity) = world.get_entity(light_view) else {
                continue;
            };
            if !matches!(
                entity.get::<LightEntity>(),
                Some(LightEntity::Directional { .. })
            ) {
                continue;
            }
            let (Some(shadow_view), Some(offset), Some(extracted)) = (
                entity.get::<ShadowView>(),
                entity.get::<ViewUniformOffset>(),
                entity.get::<ExtractedView>(),
            ) else {
                continue;
            };
            let clip_from_world = extracted.clip_from_world.unwrap_or_else(|| {
                extracted.clip_from_view * extracted.world_from_view.to_matrix().inverse()
            });
            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("static_gx_shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(
                    shadow_view.depth_attachment.get_attachment(StoreOp::Store),
                ),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &view_bind, &[offset.offset]);
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
    let center = draw.origin + Vec3::from(draw.aabb.center);
    let half = Vec3::from(draw.aabb.half_extents);
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

/// Wire the node after Bevy's shadow passes (called from `render::build`).
pub(super) fn build(render_app: &mut SubApp) {
    render_app
        .add_systems(
            bevy::render::RenderStartup,
            init_shadow_pipelines.after(init_pipelines),
        )
        .add_render_graph_node::<ViewNodeRunner<GxShadowNode>>(Core3d, GxShadowLabel)
        .add_render_graph_edges(
            Core3d,
            (NodePbr::LateShadowPass, GxShadowLabel, Node3d::StartMainPass),
        );
}
