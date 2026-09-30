//! The render-world half of the froxel volumetrics: the block upload, the two compute passes, and the
//! fog composite. See the module docs one level up.

use bevy::app::SubApp;
use bevy::core_pipeline::core_3d::{main_opaque_pass_3d, main_transparent_pass_3d};
use bevy::core_pipeline::{Core3d, Core3dSystems, FullscreenShader};
use bevy::ecs::query::QueryItem;
use bevy::pbr::{per_view_shadow_pass, LATE_SHADOW_PASS};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, texture_2d_array, texture_3d, texture_depth_2d,
    texture_depth_2d_multisampled, texture_storage_3d, uniform_buffer,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::texture::GpuImage;
use bevy::render::view::{
    Msaa, ViewDepthTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms,
};
use bevy::render::{Render, RenderStartup, RenderSystems};
use bevy::shader::ShaderDefVal;

use benilla_assets::materials::{MODERN_FOG_VOLUME, MODERN_SHADOW_CASCADES, MODERN_SHADOW_MAP};

use super::{FogMode, ModernFogView, ModernVolume, MODE_OFFSET};
use crate::lighting::SharedLightBuffer;
use crate::modern::shadow::CASCADES_BYTES;
use crate::modern::ActiveLighting;

/// The density noise texture's edge (texels); it tiles.
const NOISE_SIZE: u32 = 32;

#[derive(Resource)]
struct FroxelPipelines {
    scatter_layout: BindGroupLayoutDescriptor,
    integrate_layout: BindGroupLayoutDescriptor,
    scatter: CachedComputePipelineId,
    integrate: CachedComputePipelineId,
    composite_shader: Handle<Shader>,
    /// Per MSAA sample count: the depth binding's type and the target's sample count differ. The
    /// layout, the composite and the lamp-halo pipeline (`fs_halo`), which share it.
    composites: HashMap<
        u32,
        (
            BindGroupLayoutDescriptor,
            CachedRenderPipelineId,
            CachedRenderPipelineId,
        ),
    >,
}

/// The scatter ping-pong, the noise, and this frame's compute bind groups. `ready` = the compute
/// passes can run this frame.
#[derive(Resource, Default)]
struct FroxelTargets {
    grid: UVec3,
    scatter: Vec<(Texture, TextureView)>,
    /// The scatter texture this frame writes; the other one is its history.
    front: usize,
    noise: Option<(Texture, TextureView)>,
    repeat: Option<Sampler>,
    scatter_bind: Option<BindGroup>,
    integrate_bind: Option<BindGroup>,
    ready: bool,
    /// fork: a one-texel depth map the ground-fog banks read when the GI heightfield is absent.
    ground_stub: Option<(Texture, TextureView)>,
}

fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let scatter_layout = BindGroupLayoutDescriptor::new(
        "modern_froxel_scatter",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None),
                texture_2d_array(TextureSampleType::Depth),
                sampler(SamplerBindingType::Comparison),
                storage_buffer_read_only_sized(false, None),
                texture_3d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_storage_3d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_depth_2d(),
            ),
        ),
    );
    let integrate_layout = BindGroupLayoutDescriptor::new(
        "modern_froxel_integrate",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None),
                texture_3d(TextureSampleType::Float { filterable: false }),
                texture_storage_3d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let compute = |label: &'static str,
                   layout: &BindGroupLayoutDescriptor,
                   path: &'static str,
                   entry: &'static str| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: asset_server.load(path),
            shader_defs: vec![],
            entry_point: Some(entry.into()),
            zero_initialize_workgroup_memory: false,
        })
    };
    let scatter = compute(
        "modern_froxel_scatter",
        &scatter_layout,
        "embedded://benilla_world/shaders/modern/froxel_scatter.wgsl",
        "scatter",
    );
    let integrate = compute(
        "modern_froxel_integrate",
        &integrate_layout,
        "embedded://benilla_world/shaders/modern/froxel_integrate.wgsl",
        "integrate",
    );
    commands.insert_resource(FroxelPipelines {
        scatter_layout,
        integrate_layout,
        scatter,
        integrate,
        composite_shader: asset_server
            .load("embedded://benilla_world/shaders/modern/fog_composite.wgsl"),
        composites: HashMap::default(),
    });
}

/// A tileable 3D value noise, two octaves, as R8 bytes.
pub(super) fn tileable_noise(n: u32) -> Vec<u8> {
    fn lattice(x: i32, y: i32, z: i32, period: i32) -> f32 {
        let (x, y, z) = (
            x.rem_euclid(period),
            y.rem_euclid(period),
            z.rem_euclid(period),
        );
        let mut h = (x as u32).wrapping_mul(0x9e37_79b1)
            ^ (y as u32).wrapping_mul(0x85eb_ca77)
            ^ (z as u32).wrapping_mul(0xc2b2_ae3d);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2c1b_3c6d);
        h ^= h >> 12;
        (h & 0xffff) as f32 / 65535.0
    }
    let octave = |p: Vec3, cells: i32| {
        let g = p * cells as f32 / n as f32;
        let base = g.floor();
        let t = g - base;
        let t = t * t * (Vec3::splat(3.0) - 2.0 * t);
        let (bx, by, bz) = (base.x as i32, base.y as i32, base.z as i32);
        let mut v = 0.0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let w = (if dx == 0 { 1.0 - t.x } else { t.x })
                        * (if dy == 0 { 1.0 - t.y } else { t.y })
                        * (if dz == 0 { 1.0 - t.z } else { t.z });
                    v += w * lattice(bx + dx, by + dy, bz + dz, cells);
                }
            }
        }
        v
    };
    let mut out = Vec::with_capacity((n * n * n) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let p = Vec3::new(x as f32, y as f32, z as f32);
                let v = 0.65 * octave(p, 4) + 0.35 * octave(p, 8);
                out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    out
}

fn volume_texture(
    device: &RenderDevice,
    label: &'static str,
    grid: UVec3,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: grid.x,
            height: grid.y,
            depth_or_array_layers: grid.z,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

/// Upload the block tail, and get the compute passes' textures and bind groups ready. While they are
/// not (pipelines compiling, the volume not yet re-allocated), a froxel frame is uploaded as analytic,
/// so no lane ever samples an unwritten volume.
#[allow(clippy::too_many_arguments)]
fn prepare_froxels(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    active: Option<Res<ActiveLighting>>,
    volume: Option<Res<ModernVolume>>,
    images: Res<RenderAssets<GpuImage>>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    light: Option<Res<SharedLightBuffer>>,
    pipes: Option<Res<FroxelPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    scene: Option<Res<crate::modern::gi::ModernGiScene>>,
    mut targets: ResMut<FroxelTargets>,
    held: Option<Res<crate::fsr::FsrHeld>>,
) {
    // fork: fsr (FORK.md) — a held frame runs no froxel pass, so it must not flip `front`:
    // the next world frame would read the never-written slot as its history.
    if held.is_some_and(|h| h.0) {
        return;
    }
    targets.ready = false;
    targets.scatter_bind = None;
    targets.integrate_bind = None;
    let (Some(volume), Some(block)) = (volume, buffers.get(MODERN_SHADOW_CASCADES.id())) else {
        return;
    };
    let modern = active.is_some_and(|a| a.is_modern());
    let mut tail = volume.block.clone();
    // fork: the ground-fog banks lie on the GI terrain heightfield, whose addressing only this world
    // knows: its rows go into the fog block here, and its map into the scatter bind group.
    let ground = scene
        .as_deref()
        .filter(|s| s.clip.hf.x > 0.5)
        .and_then(|s| s.height.as_ref().map(|(_, view)| (s.clip, view.clone())));
    if let Some((clip, _)) = &ground {
        for (i, row) in [clip.hf_row_x, clip.hf_row_y, clip.hf].iter().enumerate() {
            let at = super::GROUND_OFFSET + i * 16;
            tail[at..at + 16].copy_from_slice(bytemuck::cast_slice(&row.to_array()));
        }
    }
    if targets.ground_stub.is_none() {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("modern_froxel_ground_stub"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        targets.ground_stub = Some((texture, view));
    }

    if modern && volume.mode == FogMode::Froxel {
        let inputs = (
            pipes.as_deref(),
            light.as_deref(),
            images.get(MODERN_SHADOW_MAP.id()),
            images.get(MODERN_FOG_VOLUME.id()),
        );
        if let (Some(pipes), Some(light), Some(shadow), Some(integrated)) = inputs {
            let grid = volume.grid;
            let allocated = integrated.texture_descriptor.size.width >= grid.x
                && integrated.texture_descriptor.size.height >= grid.y
                && integrated.texture_descriptor.size.depth_or_array_layers >= grid.z;
            let compiled = pipeline_cache.get_compute_pipeline(pipes.scatter).is_some()
                && pipeline_cache
                    .get_compute_pipeline(pipes.integrate)
                    .is_some();
            if allocated && compiled && grid.min_element() > 0 {
                if targets.grid != grid || targets.scatter.len() != 2 {
                    targets.scatter = vec![
                        volume_texture(&device, "modern_froxel_scatter_a", grid),
                        volume_texture(&device, "modern_froxel_scatter_b", grid),
                    ];
                    targets.grid = grid;
                }
                if targets.noise.is_none() {
                    let texture = device.create_texture(&TextureDescriptor {
                        label: Some("modern_froxel_noise"),
                        size: Extent3d {
                            width: NOISE_SIZE,
                            height: NOISE_SIZE,
                            depth_or_array_layers: NOISE_SIZE,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D3,
                        format: TextureFormat::R8Unorm,
                        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    queue.write_texture(
                        texture.as_image_copy(),
                        &tileable_noise(NOISE_SIZE),
                        TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(NOISE_SIZE),
                            rows_per_image: Some(NOISE_SIZE),
                        },
                        Extent3d {
                            width: NOISE_SIZE,
                            height: NOISE_SIZE,
                            depth_or_array_layers: NOISE_SIZE,
                        },
                    );
                    let view = texture.create_view(&TextureViewDescriptor::default());
                    targets.noise = Some((texture, view));
                }
                if targets.repeat.is_none() {
                    targets.repeat = Some(device.create_sampler(&SamplerDescriptor {
                        label: Some("modern_froxel_repeat"),
                        address_mode_u: AddressMode::Repeat,
                        address_mode_v: AddressMode::Repeat,
                        address_mode_w: AddressMode::Repeat,
                        mag_filter: FilterMode::Linear,
                        min_filter: FilterMode::Linear,
                        ..default()
                    }));
                }
                let front = targets.front;
                let write = &targets.scatter[front].1;
                let history = &targets.scatter[1 - front].1;
                let (Some((_, noise)), Some(repeat), Some((_, stub))) =
                    (&targets.noise, &targets.repeat, &targets.ground_stub)
                else {
                    return;
                };
                let ground_view = ground.as_ref().map_or(stub, |(_, view)| view);
                let scatter_bind = device.create_bind_group(
                    "modern_froxel_scatter",
                    &pipeline_cache.get_bind_group_layout(&pipes.scatter_layout),
                    &BindGroupEntries::sequential((
                        block.buffer.as_entire_binding(),
                        &shadow.texture_view,
                        &shadow.sampler,
                        light.0.as_entire_binding(),
                        noise,
                        repeat,
                        history,
                        write,
                        ground_view,
                    )),
                );
                let integrate_bind = device.create_bind_group(
                    "modern_froxel_integrate",
                    &pipeline_cache.get_bind_group_layout(&pipes.integrate_layout),
                    &BindGroupEntries::sequential((
                        block.buffer.as_entire_binding(),
                        write,
                        &integrated.texture_view,
                    )),
                );
                targets.scatter_bind = Some(scatter_bind);
                targets.integrate_bind = Some(integrate_bind);
                targets.front = 1 - front;
                targets.ready = true;
            }
        }
        if !targets.ready {
            tail[MODE_OFFSET..MODE_OFFSET + 4].copy_from_slice(bytemuck::bytes_of(&1.0f32));
        }
    }
    let end = (CASCADES_BYTES + tail.len()) as u64;
    if block.buffer.size() >= end {
        queue.write_buffer(&block.buffer, CASCADES_BYTES as u64, &tail);
    }
}

/// Build the composite pipeline for each fog view's sample count, once each.
fn specialize_composites(
    pipes: Option<ResMut<FroxelPipelines>>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
    active: Option<Res<ActiveLighting>>,
    views: Query<&Msaa, With<ModernFogView>>,
) {
    let Some(mut pipes) = pipes else {
        return;
    };
    if !active.is_some_and(|a| a.is_modern()) {
        return;
    }
    for msaa in &views {
        let samples = msaa.samples();
        if pipes.composites.contains_key(&samples) {
            continue;
        }
        let ms = samples > 1;
        let layout = BindGroupLayoutDescriptor::new(
            "modern_fog_composite",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    uniform_buffer::<ViewUniform>(true),
                    if ms {
                        texture_depth_2d_multisampled()
                    } else {
                        texture_depth_2d()
                    },
                    texture_3d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    storage_buffer_read_only_sized(false, None),
                    // The point table, for the per-pixel lamp halos (`fog_composite.wgsl`).
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        );
        let shader_defs: Vec<ShaderDefVal> = if ms { vec!["FOG_MSAA".into()] } else { vec![] };
        // `result = in-scatter + frame · transmittance`; the frame's alpha rides through.
        let blend = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::SrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
        };
        // The lamp halos add over the finished frame: `result = halo + frame`.
        let additive = BlendState {
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
        };
        let queue = |label: &str, entry: &str, blend: BlendState| {
            pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some(label.to_owned().into()),
                layout: vec![layout.clone()],
                vertex: fullscreen_shader.to_vertex_state(),
                fragment: Some(FragmentState {
                    shader: pipes.composite_shader.clone(),
                    shader_defs: shader_defs.clone(),
                    entry_point: Some(entry.to_owned().into()),
                    targets: vec![Some(ColorTargetState {
                        format: TextureFormat::Rgba16Float,
                        blend: Some(blend),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multisample: MultisampleState {
                    count: samples,
                    ..default()
                },
                ..default()
            })
        };
        let composite = queue("modern_fog_composite", "fs_composite", blend);
        let halo = queue("modern_lamp_halo", "fs_halo", additive);
        pipes.composites.insert(samples, (layout, composite, halo));
    }
}

pub(crate) fn froxel_pass(
    world: &World,
    view: ViewQuery<&ModernFogView>,
    mut render_context: RenderContext,
) {
    let _fog_view = view.into_inner();
    {
        let targets = world.resource::<FroxelTargets>();
        let (true, Some(scatter_bind), Some(integrate_bind)) = (
            targets.ready,
            &targets.scatter_bind,
            &targets.integrate_bind,
        ) else {
            return;
        };
        let pipes = world.resource::<FroxelPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let (Some(scatter), Some(integrate)) = (
            pipeline_cache.get_compute_pipeline(pipes.scatter),
            pipeline_cache.get_compute_pipeline(pipes.integrate),
        ) else {
            return;
        };
        let grid = targets.grid;
        // The journal's `gpu_modern` column, where the device times passes.
        let diagnostics = render_context.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();
        let encoder = render_context.command_encoder();
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("modern_froxel_scatter"),
                timestamp_writes: None,
            });
            let span = diagnostics.pass_span(&mut pass, "modern_froxel_scatter");
            pass.set_pipeline(scatter);
            pass.set_bind_group(0, scatter_bind, &[]);
            pass.dispatch_workgroups(grid.x.div_ceil(4), grid.y.div_ceil(4), grid.z.div_ceil(4));
            span.end(&mut pass);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("modern_froxel_integrate"),
                timestamp_writes: None,
            });
            let span = diagnostics.pass_span(&mut pass, "modern_froxel_integrate");
            pass.set_pipeline(integrate);
            pass.set_bind_group(0, integrate_bind, &[]);
            pass.dispatch_workgroups(grid.x.div_ceil(8), grid.y.div_ceil(8), 1);
            span.end(&mut pass);
        }
    }
}

/// The fullscreen fog passes' view query: both draw over the view target with the same bindings.
type FogPassQuery = (
    &'static ViewTarget,
    &'static ViewDepthTexture,
    &'static ViewUniformOffset,
    &'static Msaa,
    &'static ModernFogView,
);

pub(crate) fn fog_composite_pass(
    world: &World,
    view: ViewQuery<FogPassQuery>,
    mut render_context: RenderContext,
) {
    let view = view.into_inner();
    {
        run_fog_pass(FogPass::Composite, &mut render_context, view, world)
    }
}

/// fork: the nearest lamps' halos (`fog_composite.wgsl`'s `fs_halo`), added after the transparent
/// pass so an alpha-blended tree behind a lamp cannot paint over its glow.
pub(crate) fn lamp_halo_pass(
    world: &World,
    view: ViewQuery<FogPassQuery>,
    mut render_context: RenderContext,
) {
    let view = view.into_inner();
    {
        run_fog_pass(FogPass::Halo, &mut render_context, view, world)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum FogPass {
    Composite,
    Halo,
}

fn run_fog_pass(
    which: FogPass,
    render_context: &mut RenderContext,
    (view_target, depth, view_offset, msaa, _fog_view): QueryItem<'_, '_, FogPassQuery>,
    world: &World,
) {
    if !world
        .get_resource::<ActiveLighting>()
        .is_some_and(|a| a.is_modern())
    {
        return;
    }
    if world
        .get_resource::<ModernVolume>()
        .is_none_or(|v| v.mode == FogMode::Off)
    {
        return;
    }
    // Armed a frame after the latch (a reallocation); binding it earlier is a validation error.
    if !depth
        .texture
        .usage()
        .contains(TextureUsages::TEXTURE_BINDING)
    {
        return;
    }
    let pipes = world.resource::<FroxelPipelines>();
    let Some((layout, composite, halo)) = pipes.composites.get(&msaa.samples()) else {
        return;
    };
    let (label, id) = match which {
        FogPass::Composite => ("modern_fog_composite", composite),
        FogPass::Halo => ("modern_lamp_halo", halo),
    };
    let pipeline_cache = world.resource::<PipelineCache>();
    let Some(pipeline) = pipeline_cache.get_render_pipeline(*id) else {
        return;
    };
    let Some(view_binding) = world.resource::<ViewUniforms>().uniforms.binding() else {
        return;
    };
    let Some(volume) = world
        .resource::<RenderAssets<GpuImage>>()
        .get(MODERN_FOG_VOLUME.id())
    else {
        return;
    };
    let Some(block) = world
        .resource::<RenderAssets<GpuShaderBuffer>>()
        .get(MODERN_SHADOW_CASCADES.id())
    else {
        return;
    };
    let Some(light) = world.get_resource::<SharedLightBuffer>() else {
        return;
    };
    let bind = render_context.render_device().create_bind_group(
        label,
        &pipeline_cache.get_bind_group_layout(layout),
        &BindGroupEntries::sequential((
            view_binding,
            depth.view(),
            &volume.texture_view,
            &volume.sampler,
            block.buffer.as_entire_binding(),
            light.0.as_entire_binding(),
        )),
    );
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = render_context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    let span = diagnostics.pass_span(&mut pass, label);
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[view_offset.offset]);
    pass.draw(0..3, 0..1);
    span.end(&mut pass);
}

pub(super) fn build(render_app: &mut SubApp) {
    render_app
        .init_resource::<FroxelTargets>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(
            Render,
            specialize_composites.in_set(RenderSystems::PrepareResources),
        )
        // fork: after `PrepareResources`, where the GI pass leaves this frame's terrain heightfield
        // (`ModernGiScene`) for the ground-fog banks.
        .add_systems(
            Render,
            prepare_froxels.in_set(RenderSystems::PrepareBindGroups),
        )
        // The volume samples this frame's cascades, so it runs after the modern shadow pass.
        .add_systems(
            Core3d,
            (
                froxel_pass
                    .after(per_view_shadow_pass::<LATE_SHADOW_PASS>)
                    .after(crate::static_gx::gx_shadow_pass)
                    .before(Core3dSystems::MainPass),
                // After everything opaque (the retained cells and the SSILVB multiply included),
                // before the transparent pass — and before the water's scene copy, so a riverbed
                // seen through the water is fogged like the one beside it.
                fog_composite_pass
                    .in_set(Core3dSystems::MainPass)
                    .after(main_opaque_pass_3d)
                    .after(crate::static_gx::static_gx_pass)
                    .after(crate::ssilvb::ssilvb_pass)
                    .before(crate::liquid::liquid_screen_pass)
                    .before(main_transparent_pass_3d),
                // fork: the lamp halos, over everything the main pass drew, transparents included.
                lamp_halo_pass
                    .in_set(Core3dSystems::MainPass)
                    .after(main_transparent_pass_3d),
            ),
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_noise_tiles_and_spans_its_range() {
        let n = NOISE_SIZE;
        let noise = tileable_noise(n);
        assert_eq!(noise.len(), (n * n * n) as usize);
        let (lo, hi) = noise
            .iter()
            .fold((255u8, 0u8), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        assert!(hi - lo > 100, "range {lo}..{hi}");
        // The last texel of a row and the first of the next period differ like any neighbours do.
        let at = |x: u32, y: u32, z: u32| i32::from(noise[((z * n + y) * n + x) as usize]);
        let worst = (0..n)
            .map(|y| (at(n - 1, y, 3) - at(0, y, 3)).abs())
            .max()
            .unwrap();
        let typical = (0..n)
            .map(|y| (at(5, y, 3) - at(6, y, 3)).abs())
            .max()
            .unwrap();
        assert!(worst <= typical.max(8) * 2, "seam {worst} vs {typical}");
    }
}
