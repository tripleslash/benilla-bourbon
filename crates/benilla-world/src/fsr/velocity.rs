//! The velocity target: moving content's own motion, which the upscaler's input adapter
//! (`shaders/fsr/motion.wgsl`) takes over its depth reprojection wherever it is written.
//!
//! `xy` is the uv motion, previous minus current and jitter-free; `w` is the "written" flag. The
//! target is cleared every rendered world frame after the transparent pass and the lamp halos, and
//! the moving draws then depth-test read-only against the view's depth:
//!
//! - **Model parts** ([`collect_motion_parts`]): a `WowModelMaterial` part that is skinned or whose
//!   transform moved since the last world frame, and a moved `StandardMaterial` mesh (overhead
//!   names, raid marks). Skinned parts skin twice (`velocity.wgsl`), from the live palettes and from
//!   last world frame's copy of the shared light buffer ([`RigPalettePrev`]).
//! - **The effect lane** (`particles/render.rs`): the world view's merged effect draws, each vertex
//!   displaced by the motion its writer declares ([`crate::particles::buffer::EffectMotion`]),
//!   drawn with the lane's own raster and depth state (`effect_velocity.wgsl`).
//!
//! Nothing else moves geometry: the sky sits at depth 0 and reprojects by rotation, and terrain,
//! WMOs, `static_gx` and water are static, their animated shading left to the shading-change
//! detector and the reactive mask.

use bevy::camera::visibility::ViewVisibility;
use bevy::core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT;
use bevy::ecs::entity::EntityHashMap;
use bevy::mesh::{MeshVertexBufferLayoutRef, VertexBufferLayout};
use bevy::pbr::MeshMaterial3d;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::mesh::{RenderMesh, RenderMeshBufferInfo};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::TrackedRenderPass;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, texture_2d, uniform_buffer, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::sync_world::MainEntity;
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, ViewDepthTexture, ViewUniformOffset};
use bevy::shader::ShaderDefVal;
use bytemuck::{Pod, Zeroable};

use super::upscaler::FsrState;
use super::{FsrHeld, FsrVelocityTarget, FsrView};
use crate::lighting::SharedLightBuffer;
use crate::particles::buffer::EffectBlend;
use crate::particles::render::{EffectMeta, EffectParams, EffectPipeline, PointRun};

/// How a part covers the pixels it draws, as the main pass's own test: opaque everywhere, alpha
/// test at the vanilla 224/255, a blend where alpha is at least one half, an add where the colour
/// is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Coverage {
    Opaque = 0,
    AlphaTest = 1,
    Blend = 2,
    Add = 3,
}

/// One moving part, as the main world saw it this world frame and the last.
#[derive(Clone, Debug)]
pub struct MotionPart {
    pub mesh: AssetId<Mesh>,
    pub world_from_local: Mat4,
    pub prev_world_from_local: Mat4,
    /// The rig's palette slot; both frames' palettes are read at it.
    pub rig_slot: Option<u32>,
    pub texture: Option<AssetId<Image>>,
    pub coverage: Coverage,
}

/// This world frame's moving parts, extracted.
#[derive(Resource, Default, Clone, ExtractResource)]
pub struct FsrMotionParts {
    pub parts: Vec<MotionPart>,
    /// A skinned part was seen this world frame, fresh ones included: the next world frame may
    /// skin from this one's pose, so the velocity pass keeps the previous-pose copy.
    pub skinned_seen: bool,
}

/// `Last`, on world frames only (a held frame runs no main world): gather this frame's moving
/// parts. A part is moving when it is skinned or its transform changed since the last world frame;
/// a part seen for the first time, or whose rig slot was reassigned, is fresh and draws no motion
/// this frame (its reprojection is the camera's).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn collect_motion_parts(
    fsr: Query<(), With<FsrView>>,
    parts: Query<(
        Entity,
        &Mesh3d,
        &MeshMaterial3d<benilla_assets::materials::WowModelMaterial>,
        &GlobalTransform,
        &crate::model_render::ModelPart,
        &ViewVisibility,
        Option<&crate::rig_palette::RigPart>,
    )>,
    plain: Query<(
        Entity,
        &Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
        &GlobalTransform,
        &ViewVisibility,
    )>,
    rigs: Query<&crate::rig_palette::RigSkin>,
    model_materials: Res<Assets<benilla_assets::materials::WowModelMaterial>>,
    std_materials: Res<Assets<StandardMaterial>>,
    mut out: ResMut<FsrMotionParts>,
    mut last: Local<EntityHashMap<(Mat4, Option<u32>)>>,
    mut seen: Local<EntityHashMap<(Mat4, Option<u32>)>>,
) {
    if fsr.is_empty() {
        // Cleared through `&` first, so an idle list is not re-extracted every frame.
        if !out.parts.is_empty() || out.skinned_seen {
            out.parts.clear();
            out.skinned_seen = false;
        }
        last.clear();
        return;
    }
    out.parts.clear();
    out.skinned_seen = false;
    seen.clear();
    let coverage_of = |mode: AlphaMode, alpha_test: bool| match (alpha_test, mode) {
        (true, _) => Coverage::AlphaTest,
        (_, AlphaMode::Add) => Coverage::Add,
        (_, AlphaMode::Blend | AlphaMode::Premultiplied) => Coverage::Blend,
        _ => Coverage::Opaque,
    };
    let mut skinned_seen = false;
    let mut push = |entity: Entity,
                    mesh: AssetId<Mesh>,
                    world_from_local: Mat4,
                    rig_slot: Option<u32>,
                    texture: Option<AssetId<Image>>,
                    coverage: Coverage| {
        seen.insert(entity, (world_from_local, rig_slot));
        let Some(&(prev, prev_slot)) = last.get(&entity) else {
            return; // fresh: no previous pose to difference
        };
        if prev_slot != rig_slot {
            return; // a reassigned palette slot holds another rig's last pose
        }
        if rig_slot.is_none() && prev == world_from_local {
            return; // not moving: the camera's reprojection is exact
        }
        out.parts.push(MotionPart {
            mesh,
            world_from_local,
            prev_world_from_local: prev,
            rig_slot,
            texture,
            coverage,
        });
    };
    for (entity, mesh, material, gt, part, visible, rig) in &parts {
        if !visible.get() {
            continue;
        }
        use benilla_formats::ModelBlend;
        let alpha_test = match part.blend {
            ModelBlend::Opaque => false,
            ModelBlend::AlphaTest => true,
            ModelBlend::Blend => false,
            // A modulate multiplies what is already there; it has no surface of its own to move.
            ModelBlend::Mod | ModelBlend::Mod2x => continue,
        };
        let Some(m) = model_materials.get(&material.0) else {
            continue;
        };
        let coverage = if matches!(part.blend, ModelBlend::Opaque) {
            Coverage::Opaque
        } else {
            coverage_of(m.base.alpha_mode, alpha_test)
        };
        let texture = (coverage != Coverage::Opaque)
            .then(|| m.base.base_color_texture.as_ref().map(|h| h.id()))
            .flatten();
        let rig_slot = rig
            .and_then(|r| rigs.get(r.0).ok())
            .map(|skin| u32::from(skin.slot));
        skinned_seen |= rig_slot.is_some();
        push(
            entity,
            mesh.0.id(),
            gt.to_matrix(),
            rig_slot,
            texture,
            coverage,
        );
    }
    for (entity, mesh, material, gt, visible) in &plain {
        if !visible.get() {
            continue;
        }
        let Some(m) = std_materials.get(&material.0) else {
            continue;
        };
        let texture = m.base_color_texture.as_ref().map(|h| h.id());
        let coverage = match texture {
            None => Coverage::Opaque,
            Some(_) => coverage_of(m.alpha_mode, matches!(m.alpha_mode, AlphaMode::Mask(_))),
        };
        push(entity, mesh.0.id(), gt.to_matrix(), None, texture, coverage);
    }
    out.skinned_seen = skinned_seen;
    std::mem::swap(&mut *last, &mut *seen);
}

// ---------------------------------------------------------------- render world

/// Last world frame's copy of the shared light buffer, whose rig regions (`rig_table`,
/// `rig_origin`, the palettes) are the previous pose of every skinned part. Refreshed by a
/// `copy_buffer_to_buffer` after the velocity draws, on world frames only.
#[derive(Resource, Default)]
pub struct RigPalettePrev {
    buffer: Option<Buffer>,
    /// The copy holds a whole frame of the current buffer: false after a (re)allocation until the
    /// first copy lands, while skinned parts draw no motion.
    valid: bool,
}

/// The per-view matrices `velocity.wgsl` projects with: the jittered raster matrix and the
/// unjittered current and previous ones.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VelocityViewUniform {
    raster_clip_from_world: [[f32; 4]; 4],
    cur_clip_from_world: [[f32; 4]; 4],
    prev_clip_from_world: [[f32; 4]; 4],
}

const VIEW_UNIFORM_SIZE: u64 = std::mem::size_of::<VelocityViewUniform>() as u64;

/// One part's uniform; `VelocityPart` in `velocity.wgsl`.
#[derive(Clone, Copy, ShaderType)]
struct VelocityPartUniform {
    world_from_local: Mat4,
    prev_world_from_local: Mat4,
    rig_slot: u32,
    coverage: u32,
    _pad0: u32,
    _pad1: u32,
}

/// The pipelines, keyed by what varies a part's vertex layout and bindings.
#[derive(Resource)]
pub struct VelocityPipelines {
    shader: Handle<Shader>,
    view_layout: BindGroupLayoutDescriptor,
    part_layout: BindGroupLayoutDescriptor,
    skin_layout: BindGroupLayoutDescriptor,
    tex_layout: BindGroupLayoutDescriptor,
    by_key: HashMap<(MeshVertexBufferLayoutRef, bool, bool), CachedRenderPipelineId>,
}

/// One queued draw.
struct VelocityDraw {
    mesh: AssetId<Mesh>,
    pipeline: CachedRenderPipelineId,
    offset: u32,
    skinned: bool,
    texture: Option<BindGroup>,
}

/// This frame's part draws and their bind groups.
#[derive(Resource, Default)]
pub struct VelocityFrame {
    draws: Vec<VelocityDraw>,
    parts: DynamicUniformBuffer<VelocityPartUniform>,
    view_buffer: Option<Buffer>,
    view: Option<BindGroup>,
    part: Option<BindGroup>,
    skin: Option<BindGroup>,
    tex_binds: HashMap<AssetId<Image>, (TextureViewId, BindGroup)>,
    /// The bytes of the live light buffer the pass copies as the next frame's previous pose:
    /// `None` when no skinned part was seen, so no next frame can skin from this one's.
    prev_copy_bytes: Option<u64>,
}

pub(super) fn init_pipelines(mut commands: Commands, shaders: Option<Res<super::FsrShaders>>) {
    let Some(shaders) = shaders else {
        return;
    };
    let c = ShaderStages::VERTEX_FRAGMENT;
    commands.insert_resource(VelocityPipelines {
        shader: shaders.velocity.clone(),
        view_layout: BindGroupLayoutDescriptor::new(
            "fsr_velocity_view_layout",
            &BindGroupLayoutEntries::single(
                c,
                uniform_buffer_sized(false, std::num::NonZero::new(VIEW_UNIFORM_SIZE)),
            ),
        ),
        part_layout: BindGroupLayoutDescriptor::new(
            "fsr_velocity_part_layout",
            &BindGroupLayoutEntries::single(
                c,
                uniform_buffer_sized(true, Some(VelocityPartUniform::min_size())),
            ),
        ),
        skin_layout: BindGroupLayoutDescriptor::new(
            "fsr_velocity_skin_layout",
            &BindGroupLayoutEntries::sequential(
                c,
                (
                    uniform_buffer_sized(true, Some(VelocityPartUniform::min_size())),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        ),
        tex_layout: BindGroupLayoutDescriptor::new(
            "fsr_velocity_tex_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        ),
        by_key: HashMap::default(),
    });
}

/// `PrepareResources`, after [`super::upscaler::prepare_fsr`]: this frame's part draws —
/// pipelines queued on first sight of a layout, one uniform per part, texture bind groups — and
/// the view uniform from the FSR view's matrices.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_velocity(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    held: Res<FsrHeld>,
    parts: Option<Res<FsrMotionParts>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    images: Res<RenderAssets<GpuImage>>,
    light: Option<Res<SharedLightBuffer>>,
    pipeline_cache: Res<PipelineCache>,
    pipes: Option<ResMut<VelocityPipelines>>,
    palette: Option<Res<crate::rig_palette::RigPaletteExtract>>,
    mut prev: ResMut<RigPalettePrev>,
    mut frame: ResMut<VelocityFrame>,
    views: Query<&FsrState>,
) {
    if held.0 {
        return;
    }
    frame.draws.clear();
    frame.view = None;
    frame.prev_copy_bytes = None;
    let Some(state) = views.iter().next() else {
        // No view runs FSR: the previous-pose copy and the texture groups have no reader.
        if prev.buffer.is_some() {
            *prev = RigPalettePrev::default();
        }
        frame.tex_binds.clear();
        return;
    };
    let (Some(parts), Some(mut pipes)) = (parts, pipes) else {
        return;
    };
    // The previous-pose copy follows the live buffer's size; a new one holds nothing yet.
    let live_size = light.as_ref().map(|l| l.0.size());
    if live_size.is_some_and(|size| prev.buffer.as_ref().is_none_or(|b| b.size() != size)) {
        prev.buffer = live_size.map(|size| {
            device.create_buffer(&BufferDescriptor {
                label: Some("fsr_rig_palette_prev"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        prev.valid = false;
    }
    let skin_ready = prev.valid && light.is_some();
    // Up to the last allocated bone, everything below it included (`rig_palette.rs`); the whole
    // buffer while the palette has not published.
    frame.prev_copy_bytes = parts.skinned_seen.then(|| {
        palette
            .as_ref()
            .map_or(u64::MAX, |p| p.live_bytes().next_multiple_of(4))
    });

    let [raster, cur, prev_clip] = state.clip_from_world();
    let view_uniform = VelocityViewUniform {
        raster_clip_from_world: raster.to_cols_array_2d(),
        cur_clip_from_world: cur.to_cols_array_2d(),
        prev_clip_from_world: prev_clip.to_cols_array_2d(),
    };
    let view_buffer = frame
        .view_buffer
        .get_or_insert_with(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some("fsr_velocity_view"),
                size: VIEW_UNIFORM_SIZE,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .clone();
    queue.write_buffer(&view_buffer, 0, bytemuck::bytes_of(&view_uniform));
    frame.view = Some(device.create_bind_group(
        "fsr_velocity_view",
        &pipeline_cache.get_bind_group_layout(&pipes.view_layout),
        &BindGroupEntries::single(view_buffer.as_entire_binding()),
    ));

    frame.parts.clear();
    let mut draws = Vec::new();
    for part in &parts.parts {
        let skinned = part.rig_slot.is_some();
        if skinned && !skin_ready {
            continue;
        }
        let Some(mesh) = meshes.get(part.mesh) else {
            continue;
        };
        let tex = part
            .texture
            .and_then(|id| images.get(id).map(|gpu| (id, gpu)));
        let textured = tex.is_some() && part.coverage != Coverage::Opaque;
        let key = (mesh.layout.clone(), skinned, textured);
        let pipeline = match pipes.by_key.get(&key) {
            Some(id) => *id,
            None => {
                let mut attrs = vec![Mesh::ATTRIBUTE_POSITION.at_shader_location(0)];
                if textured {
                    attrs.push(Mesh::ATTRIBUTE_UV_0.at_shader_location(2));
                }
                if skinned {
                    attrs.push(benilla_assets::ATTRIBUTE_WOW_JOINT_INDEX.at_shader_location(10));
                    attrs.push(benilla_assets::ATTRIBUTE_WOW_JOINT_WEIGHT.at_shader_location(11));
                }
                let Ok(vertex_layout) = mesh.layout.0.get_layout(&attrs) else {
                    continue;
                };
                let mut defs: Vec<ShaderDefVal> = Vec::new();
                let mut layout = vec![
                    pipes.view_layout.clone(),
                    if skinned {
                        pipes.skin_layout.clone()
                    } else {
                        pipes.part_layout.clone()
                    },
                ];
                if skinned {
                    defs.push("SKINNED".into());
                }
                if textured {
                    defs.push("TEXTURED".into());
                    layout.push(pipes.tex_layout.clone());
                }
                let id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("fsr_velocity".into()),
                    layout,
                    vertex: VertexState {
                        shader: pipes.shader.clone(),
                        shader_defs: defs.clone(),
                        entry_point: Some("vertex".into()),
                        buffers: vec![vertex_layout],
                    },
                    fragment: Some(FragmentState {
                        shader: pipes.shader.clone(),
                        shader_defs: defs,
                        entry_point: Some("fragment".into()),
                        targets: vec![Some(ColorTargetState {
                            format: TextureFormat::Rgba16Float,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })],
                    }),
                    primitive: PrimitiveState {
                        cull_mode: None,
                        ..default()
                    },
                    // Read-only against the main pass's own depth, a small bias toward the eye so a
                    // re-rasterized surface passes where it drew (reverse-Z: nearer is larger).
                    depth_stencil: Some(DepthStencilState {
                        format: CORE_3D_DEPTH_FORMAT,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(CompareFunction::GreaterEqual),
                        stencil: StencilState::default(),
                        bias: DepthBiasState {
                            constant: 4,
                            slope_scale: 1.0,
                            clamp: 0.0,
                        },
                    }),
                    ..default()
                });
                pipes.by_key.insert(key, id);
                id
            }
        };
        let texture = match (textured, tex) {
            (true, Some((id, gpu))) => {
                let view_id = gpu.texture_view.id();
                if frame.tex_binds.get(&id).is_none_or(|(v, _)| *v != view_id) {
                    let bind = device.create_bind_group(
                        "fsr_velocity_texture",
                        &pipeline_cache.get_bind_group_layout(&pipes.tex_layout),
                        &BindGroupEntries::sequential((&gpu.texture_view, &gpu.sampler)),
                    );
                    frame.tex_binds.insert(id, (view_id, bind));
                }
                frame.tex_binds.get(&id).map(|(_, b)| b.clone())
            }
            _ => None,
        };
        let offset = frame.parts.push(&VelocityPartUniform {
            world_from_local: part.world_from_local,
            prev_world_from_local: part.prev_world_from_local,
            rig_slot: part.rig_slot.unwrap_or(0),
            coverage: part.coverage as u32,
            _pad0: 0,
            _pad1: 0,
        });
        draws.push(VelocityDraw {
            mesh: part.mesh,
            pipeline,
            offset,
            skinned,
            texture,
        });
    }
    if frame.tex_binds.len() > 512 {
        let live: std::collections::HashSet<AssetId<Image>> =
            parts.parts.iter().filter_map(|p| p.texture).collect();
        frame.tex_binds.retain(|id, _| live.contains(id));
    }
    frame.parts.write_buffer(&device, &queue);
    // Field-disjoint from here: the binding borrows `parts` while `part` and `skin` are written.
    let frame = &mut *frame;
    let Some(binding) = frame.parts.binding() else {
        return;
    };
    frame.part = Some(device.create_bind_group(
        "fsr_velocity_part",
        &pipeline_cache.get_bind_group_layout(&pipes.part_layout),
        &BindGroupEntries::single(binding.clone()),
    ));
    frame.skin = match (&light, &prev.buffer) {
        (Some(light), Some(prev)) => Some(device.create_bind_group(
            "fsr_velocity_skin",
            &pipeline_cache.get_bind_group_layout(&pipes.skin_layout),
            &BindGroupEntries::sequential((
                binding,
                light.0.as_entire_binding(),
                prev.as_entire_binding(),
            )),
        )),
        _ => None,
    };
    frame.draws = draws;
}

/// `MainPass`, last: clear the velocity target to "nothing written", draw every moving part into
/// it, and refresh the previous-pose copy of the rig palettes.
#[allow(clippy::too_many_arguments)]
pub(super) fn fsr_velocity_pass(
    view: ViewQuery<(
        &FsrVelocityTarget,
        &ViewDepthTexture,
        &ExtractedView,
        &ViewUniformOffset,
    )>,
    held: Res<FsrHeld>,
    frame: Res<VelocityFrame>,
    mut prev: ResMut<RigPalettePrev>,
    light: Option<Res<SharedLightBuffer>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    allocator: Res<MeshAllocator>,
    pipeline_cache: Res<PipelineCache>,
    effects: Res<EffectVelocityDraws>,
    effect_meta: Res<EffectMeta>,
    mut render_context: RenderContext,
) {
    if held.0 {
        return;
    }
    let (target, depth, extracted, view_offset) = view.into_inner();
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    {
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("fsr_velocity"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Default::default()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth.view(),
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        // The journal's `gpu_velocity` column.
        let span = diagnostics.pass_span(&mut pass, "fsr_velocity");
        if let (Some(view_bind), Some(part_bind)) = (&frame.view, &frame.part) {
            pass.set_bind_group(0, view_bind, &[]);
            for draw in &frame.draws {
                let Some(pipeline) = pipeline_cache.get_render_pipeline(draw.pipeline) else {
                    continue;
                };
                let group = if draw.skinned {
                    let Some(skin) = &frame.skin else {
                        continue;
                    };
                    skin
                } else {
                    part_bind
                };
                let Some(mesh) = meshes.get(draw.mesh) else {
                    continue;
                };
                let Some(vslice) = allocator.mesh_vertex_slice(&draw.mesh) else {
                    continue;
                };
                pass.set_render_pipeline(pipeline);
                pass.set_bind_group(1, group, &[draw.offset]);
                if let Some(texture) = &draw.texture {
                    pass.set_bind_group(2, texture, &[]);
                }
                pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                match &mesh.buffer_info {
                    RenderMeshBufferInfo::Indexed { index_format, .. } => {
                        let Some(islice) = allocator.mesh_index_slice(&draw.mesh) else {
                            continue;
                        };
                        pass.set_index_buffer(islice.buffer.slice(..), *index_format);
                        pass.draw_indexed(
                            islice.range.clone(),
                            i32::try_from(vslice.range.start).unwrap_or(0),
                            0..1,
                        );
                    }
                    RenderMeshBufferInfo::NonIndexed => {
                        pass.draw(vslice.range.clone(), 0..1);
                    }
                }
            }
        }
        // The effect lane's draws of this view, in the same pass.
        effects.draw(
            &mut pass,
            &pipeline_cache,
            &effect_meta,
            extracted.retained_view_entity.main_entity,
            view_offset.offset,
        );
        span.end(&mut pass);
    }
    // This frame's pose becomes the next world frame's previous one, while a skinned part is in
    // view; with none, the next frame's skinned parts are all fresh and the copy stands invalid.
    match (light, &prev.buffer, frame.prev_copy_bytes) {
        (Some(light), Some(copy), Some(bytes)) if copy.size() == light.0.size() => {
            render_context.command_encoder().copy_buffer_to_buffer(
                &light.0,
                0,
                copy,
                0,
                bytes.min(copy.size()),
            );
            prev.valid = true;
        }
        _ => prev.valid = false,
    }
}

// ---------------------------------------------------------------- the effect lane

/// The effect lane's velocity pipeline: `wow_effect.wgsl`'s raster and depth state, the lane's
/// coverage, and the motion from each vertex's delta. Group 0 is the lane's own view group.
#[derive(Resource)]
pub struct EffectVelocityPipeline {
    view_layout: BindGroupLayoutDescriptor,
    tex_layout: BindGroupLayoutDescriptor,
    motion_layout: BindGroupLayoutDescriptor,
    /// A point draw's row of the lane's params, by dynamic offset: its shape.
    params_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

/// What varies an effect draw's velocity pipeline: its coverage rule, its depth state and whether
/// it is a point draw (`particles::buffer::EffectTopology::Points`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectVelocityKey {
    blend: EffectBlend,
    raster_bias: i32,
    raster_slope_bits: u32,
    no_depth_test: bool,
    points: bool,
}

impl SpecializedRenderPipeline for EffectVelocityPipeline {
    type Key = EffectVelocityKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![
                VertexFormat::Float32x3,
                VertexFormat::Float32x2,
                VertexFormat::Float32x4,
            ],
        );
        let mut motion = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![VertexFormat::Float32x3],
        );
        motion.attributes[0].shader_location = 3;
        let mut defs: Vec<ShaderDefVal> = vec![match key.blend {
            EffectBlend::Add => "BLEND_ADD",
            EffectBlend::Alpha => "BLEND_ALPHA",
            EffectBlend::AlphaKey => "BLEND_ALPHAKEY",
            _ => "BLEND_OPAQUE",
        }
        .into()];
        if key.raster_bias != 0 {
            defs.push("DECAL_WORLD_CLIP".into());
        }
        let mut layouts = vec![
            self.view_layout.clone(),
            self.tex_layout.clone(),
            self.motion_layout.clone(),
        ];
        // A point draw: the lane's instance layout, no motion stream (it is still in the world),
        // and its shape row.
        let buffers = if key.points {
            defs.push("EFFECT_POINTS".into());
            layouts.push(self.params_layout.clone());
            vec![crate::particles::render::point_layout()]
        } else {
            vec![layout, motion]
        };
        RenderPipelineDescriptor {
            label: Some("fsr_effect_velocity".into()),
            layout: layouts,
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: defs.clone(),
                entry_point: Some("vertex".into()),
                buffers,
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: defs,
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                cull_mode: None,
                ..default()
            },
            // The lane's own depth state (`particles/render.rs`), never writing.
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(if key.no_depth_test {
                    CompareFunction::Always
                } else {
                    CompareFunction::GreaterEqual
                }),
                stencil: StencilState::default(),
                bias: DepthBiasState {
                    constant: key.raster_bias,
                    slope_scale: f32::from_bits(key.raster_slope_bits),
                    clamp: 0.0,
                },
            }),
            ..default()
        }
    }
}

pub(super) fn init_effect_pipeline(
    mut commands: Commands,
    shaders: Option<Res<super::FsrShaders>>,
    effect: Res<EffectPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<EffectVelocityPipeline>>,
) {
    let Some(shaders) = shaders else {
        return;
    };
    let pipeline = EffectVelocityPipeline {
        view_layout: effect.view_layout().clone(),
        tex_layout: BindGroupLayoutDescriptor::new(
            "fsr_effect_velocity_tex_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        ),
        motion_layout: BindGroupLayoutDescriptor::new(
            "fsr_effect_velocity_motion_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::VERTEX,
                uniform_buffer_sized(false, std::num::NonZero::new(128)),
            ),
        ),
        params_layout: BindGroupLayoutDescriptor::new(
            "fsr_effect_velocity_params_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::VERTEX,
                uniform_buffer::<EffectParams>(true),
            ),
        ),
        shader: shaders.effect_velocity.clone(),
    };
    // Warm: every coverage rule at the free-floating depth state, what particles and ribbons
    // draw with, and the point draw the snow and the mist take, compile at startup behind the
    // entry cover; a decal's biased variant compiles on first sight.
    for (blend, points) in [
        (EffectBlend::Add, false),
        (EffectBlend::Alpha, false),
        (EffectBlend::AlphaKey, false),
        (EffectBlend::Opaque, false),
        (EffectBlend::Alpha, true),
    ] {
        pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            EffectVelocityKey {
                blend,
                raster_bias: 0,
                raster_slope_bits: 0.0f32.to_bits(),
                no_depth_test: false,
                points,
            },
        );
    }
    commands.insert_resource(pipeline);
}

/// One effect draw of the velocity pass.
struct EffectVelocityDraw {
    pipeline: CachedRenderPipelineId,
    index_range: std::ops::Range<u32>,
    texture: AssetId<Image>,
    /// A point draw's instances and its params row; `None` draws `index_range`.
    points: Option<(PointRun, u32)>,
}

/// This frame's effect draws for the world view, their texture groups and the motion matrices.
#[derive(Resource, Default)]
pub struct EffectVelocityDraws {
    view: Option<MainEntity>,
    draws: Vec<EffectVelocityDraw>,
    motion_buffer: Option<Buffer>,
    motion: Option<BindGroup>,
    /// The lane's params buffer for the point draws' shape rows, while this frame has any.
    params: Option<BindGroup>,
    tex_binds: HashMap<AssetId<Image>, (TextureViewId, BindGroup)>,
}

impl EffectVelocityDraws {
    /// Draw this frame's effect velocity into the open pass, if `view` is the one it was built for.
    fn draw<'w>(
        &'w self,
        pass: &mut TrackedRenderPass<'w>,
        pipeline_cache: &'w PipelineCache,
        meta: &'w EffectMeta,
        view: MainEntity,
        view_offset: u32,
    ) {
        if self.view != Some(view) || self.draws.is_empty() {
            return;
        }
        let (Some(motion), Some(view_group)) = (&self.motion, meta.view_bind_group()) else {
            return;
        };
        pass.set_bind_group(0, view_group, &[view_offset]);
        pass.set_bind_group(2, motion, &[]);
        // The vertex streams exist only in a frame with a vertex draw, the points only with a
        // point draw: each draw takes its own, and a frame may hold either alone.
        let streams = meta.velocity_buffers();
        let points = meta.point_buffers().map(|(buffer, _)| buffer);
        for draw in &self.draws {
            let (Some(pipeline), Some((_, texture))) = (
                pipeline_cache.get_render_pipeline(draw.pipeline),
                self.tex_binds.get(&draw.texture),
            ) else {
                continue;
            };
            match &draw.points {
                Some((run, params_offset)) => {
                    let (Some(points), Some(params)) = (points, &self.params) else {
                        continue;
                    };
                    pass.set_render_pipeline(pipeline);
                    pass.set_bind_group(1, texture, &[]);
                    pass.set_bind_group(3, params, &[*params_offset]);
                    pass.set_vertex_buffer(0, points.slice(..));
                    pass.draw(0..run.vertices, run.instances.clone());
                }
                None => {
                    let Some((vertices, motion_vertices, indices)) = streams else {
                        continue;
                    };
                    pass.set_render_pipeline(pipeline);
                    pass.set_bind_group(1, texture, &[]);
                    pass.set_vertex_buffer(0, vertices.slice(..));
                    pass.set_vertex_buffer(1, motion_vertices.slice(..));
                    pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(draw.index_range.clone(), 0, 0..1);
                }
            }
        }
    }
}

/// `PrepareBindGroups`, after the lane's merge walk (`prepare_effects`): the world view's merged
/// draws that cover pixels as a surface (a modulate has none of its own), their pipelines and
/// texture groups, and this frame's camera-relative motion matrices.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_effect_velocity(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    held: Res<FsrHeld>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<Res<EffectVelocityPipeline>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<EffectVelocityPipeline>>,
    meta: Res<EffectMeta>,
    images: Res<RenderAssets<GpuImage>>,
    mut out: ResMut<EffectVelocityDraws>,
    views: Query<(&ExtractedView, &FsrState)>,
) {
    if held.0 {
        return;
    }
    out.draws.clear();
    out.view = None;
    let (Some(pipeline), Some((view, state))) = (pipeline, views.iter().next()) else {
        return;
    };
    let main = view.retained_view_entity.main_entity;
    let [clip_from_cur_rel, prev_clip_from_cur_rel] = state.rel_matrices();
    let mut matrices = [0.0f32; 32];
    matrices[..16].copy_from_slice(&clip_from_cur_rel.to_cols_array());
    matrices[16..].copy_from_slice(&prev_clip_from_cur_rel.to_cols_array());
    let buffer = out
        .motion_buffer
        .get_or_insert_with(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some("fsr_effect_velocity_motion"),
                size: 128,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .clone();
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&matrices));
    out.motion = Some(device.create_bind_group(
        "fsr_effect_velocity_motion",
        &pipeline_cache.get_bind_group_layout(&pipeline.motion_layout),
        &BindGroupEntries::single(buffer.as_entire_binding()),
    ));
    let tex_layout = pipeline_cache.get_bind_group_layout(&pipeline.tex_layout);
    out.params = meta.point_buffers().map(|(_, rows)| {
        device.create_bind_group(
            "fsr_effect_velocity_params",
            &pipeline_cache.get_bind_group_layout(&pipeline.params_layout),
            &BindGroupEntries::single(rows),
        )
    });
    let mut draws = Vec::new();
    for merged in meta.merged_for(main) {
        if matches!(merged.blend, EffectBlend::Multiply | EffectBlend::Mod2x) {
            continue;
        }
        // fork: perf — a GPU-expanded pool has no vertices here; it exists only while no camera
        // runs FSR, so this pass never meets one in its own frame.
        if merged.particles.is_some() {
            continue;
        }
        let Some(gpu) = images.get(merged.texture) else {
            continue;
        };
        let view_id = gpu.texture_view.id();
        if out
            .tex_binds
            .get(&merged.texture)
            .is_none_or(|(v, _)| *v != view_id)
        {
            let bind = device.create_bind_group(
                "fsr_effect_velocity_texture",
                &tex_layout,
                &BindGroupEntries::sequential((&gpu.texture_view, &gpu.sampler)),
            );
            out.tex_binds.insert(merged.texture, (view_id, bind));
        }
        let id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            EffectVelocityKey {
                blend: merged.blend,
                raster_bias: merged.raster_bias,
                raster_slope_bits: merged.raster_slope.to_bits(),
                no_depth_test: merged.no_depth_test,
                points: merged.points.is_some(),
            },
        );
        draws.push(EffectVelocityDraw {
            pipeline: id,
            index_range: merged.index_range.clone(),
            texture: merged.texture,
            points: merged.points.clone().map(|run| (run, merged.params_offset)),
        });
    }
    if out.tex_binds.len() > 512 {
        let live: std::collections::HashSet<AssetId<Image>> =
            draws.iter().map(|d| d.texture).collect();
        out.tex_binds.retain(|id, _| live.contains(id));
    }
    out.draws = draws;
    out.view = Some(main);
}
