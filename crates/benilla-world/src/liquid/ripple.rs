//! **The dynamic ripple displacement map** — the height field a wading body pushes into the water,
//! and the wave that spreads out from it.
//!
//! `liquid_ripple.wgsl` carries the integration and the reprojection; this file carries the wiring
//! and the three decisions the wiring exists for.
//!
//! ## One square of world, following the camera, snapped to texels
//!
//! The reference keeps several `proceduralDisplMap%d` targets at different extents and blends
//! between them ([`GetDynamicDisplMapCnt`] > 1 binds a second on stage 8); this ports **one**, which
//! is the `samp4`-only permutation family and a shipped configuration rather than a reduction of
//! one. Its world rect is recomputed every frame from the camera and then **floored to a whole
//! texel** — `Liquid::SetDisplMapInfo` does the same `floorf(x · size/extent) · extent/size`, and it
//! is not a nicety: a sub-texel origin resamples the entire field every frame, and a wave that is
//! resampled every frame diffuses into nothing within about a second of walking.
//!
//! ## Two textures, and why the material still binds one
//!
//! The wave step reads five taps and writes one texel, so it cannot read and write the same texture.
//! It ping-pongs into a scratch target and the node then **copies the result back** into the
//! texture the material binds. That copy is a `copy_texture_to_texture` on a 256² surface — cheaper
//! than the alternative, which is swapping which handle the material points at, and that would
//! rebuild every liquid bind group every frame (`world_backdrop`'s decision 1647 is the account of
//! what that costs and how it goes wrong).
//!
//! ## A fixed timestep
//!
//! The wave equation is second-order and unconditionally unstable at a varying `dt`. It runs at a
//! fixed [`STEP_HZ`] off an accumulator, and the water shader interpolates between the two heights
//! the map already packs — which is what the reference's `GetDynamicDisplMapInterpolation` is, and
//! why its `ProcWaterAbove` lerps `.x` against `.y` before taking the gradient.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::FullscreenShader;
use bevy::ecs::query::QueryItem;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::asset::RenderAssetUsages;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{sampler, storage_buffer_read_only_sized, texture_2d};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::screen::LiquidScreenView;
use super::{water_surface_at, LiquidClaim, WaterChunkInfo};
use crate::view::WorldCamera;
use crate::world_unit::{ViewerUnit, WorldUnit};
use benilla_assets::coords::bevy_to_wow;

/// Edge length of the displacement map, in texels — **the reference's own 1024**
/// (`Liquid::CreateDisplMaps`, `dword_13F1410[0] = 1024`).
///
/// This number is not a quality knob, it is the wave's SPEED. The propagation step moves a
/// disturbance exactly one texel, so the field's resolution and its step rate together fix how fast
/// a ripple travels — and the first cut of this file, at 256 over 48 yards and stepping at 60 Hz,
/// sent waves out at 11 yd/s instead of 1. They crossed the whole map before the next stamp landed,
/// which reads as ripples that "spread very far and only trigger every few seconds": the rings were
/// so large and so short-lived that each one looked like a separate event.
pub const RIPPLE_SIZE: u32 = 1024;

/// How much world the map covers, in yards.
///
/// The reference derives this from the view size times a per-map factor, which is data this
/// client cannot read — but its other two numbers pin it: a 1024-texel map and a **0.3-yard**
/// perturbation radius ([`STAMP_RADIUS`], read off the rain emitter) only make sense together if a
/// texel is around a twentieth of a yard, and 50 yards over 1024 texels is 0.049. That also puts the
/// map's half-width at the 25 yards the reference's own rain scatters within, which is the shape of
/// a thing sized to hold its own content.
pub const RIPPLE_EXTENT: f32 = 50.0;

/// Milliseconds between simulation steps — the reference's `dword_13F141C = 50`, i.e. **20 Hz**.
///
/// Read as a period rather than a rate because that is what the reference stores and how it uses
/// it: `(now − last) % period` is its interpolation factor and `(now − last) / period` its step
/// count. At `rippleDetail 2` it runs two maps at 25 ms and 50 ms; this ports the single-map
/// configuration, which is the 50.
const STEP_MS: f32 = 50.0;

/// How many steps one frame may catch up — the reference's own `min(elapsed / period, 2)`.
///
/// A cap rather than a loop to convergence, and for a reason worth keeping: the wave equation has
/// no `dt`, so twenty steps in one frame is not a slow-motion catch-up, it is twenty times the wave
/// speed for a frame. After a hitch the field should keep its pace and lose the time.
const MAX_CATCHUP: u32 = 2;

/// The reference's own damping (`PropagateDisplMap`'s `c0.z`).
const DAMPING: f32 = 0.97;

/// The most disturbances integrated in one step — must match `liquid_ripple.wgsl`'s `MAX_STAMPS`.
///
/// Sized for RAIN, which is the reference's own source and by far the heaviest: it scatters twenty
/// drops per tick. Bodies contribute one each and never come close.
const MAX_STAMPS: usize = 32;

/// A perturbation's radius in yards — **the reference's 0.3**, read as the float literal
/// `1050253722` (`0x3E99999A`) in its rain emitter's `Liquid::AddPerturbation` call.
const STAMP_RADIUS: f32 = 0.3;

/// A raindrop's amplitude range — the reference's `−(rand()%200/1000 + 0.2)`, i.e. −0.2 to −0.4.
///
/// **Negative**, and that is the physics rather than a sign convention: a drop presses the surface
/// DOWN and the wave equation does the rest. A positive stamp would model a thing rising out of the
/// water.
const RAIN_AMPLITUDE: (f32, f32) = (-0.4, -0.2);

/// Rain drops scattered per step, and how far from the camera — the reference's twenty per tick,
/// spread over `(rand/500 − 1) · 50` in each axis.
const RAIN_DROPS: usize = 20;
const RAIN_SPREAD: f32 = 50.0;

/// The intensity below which rain makes no ripples — the reference's `> 0.2` gate.
const RAIN_MIN_DENSITY: f32 = 0.2;

/// A wading body's radius and amplitude.
///
/// **The reference has no such source.** Its dynamic displacement map is driven by rain and by
/// mesh-shaped disturbances only; a walking character's ripple is the separate `CWater0Ripple`
/// decal system, which benilla already carries in `water_fx`. These two numbers are therefore a
/// deliberate addition, scaled off the raindrop the reference does specify: half again its radius,
/// and an amplitude in the same band, because a leg entering water is a bigger version of the same
/// event and should not be a different order of magnitude.
const WADE_RADIUS: f32 = 0.45;
const WADE_AMPLITUDE: f32 = -0.30;

/// The gradient scale the water shader applies to the field.
///
/// The one number here with no reference value to take: it is `ProcWaterAbove`'s `c28.w`, a uniform
/// the client uploads from data this port has no copy of. Chosen against the terms it competes on —
/// the bump gradient it is added to runs around 0.3, and a wake that overwhelms the swell reads as a
/// defect rather than as a wake.
const RIPPLE_SLOPE: f32 = 2.0;

/// The packed uniform both `liquid_ripple.wgsl` and `liquid_cata.wgsl` read. Field order is the
/// WGSL `Ripple` struct's; reordering is a breaking change.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RippleStd430 {
    placement: [f32; 4],
    params: [f32; 4],
    prev: [f32; 4],
    stamp_count: [f32; 4],
    stamps: [[f32; 4]; MAX_STAMPS],
}

impl Default for RippleStd430 {
    fn default() -> Self {
        Self {
            placement: [0.0; 4],
            params: [1.0 / RIPPLE_SIZE as f32, RIPPLE_SLOPE, 0.0, DAMPING],
            prev: [0.0; 4],
            stamp_count: [0.0; 4],
            stamps: [[0.0; 4]; MAX_STAMPS],
        }
    }
}

/// The per-frame blob, extracted to the render world for upload.
#[derive(Resource, Clone, Copy, ExtractResource)]
pub(super) struct RippleData(RippleStd430);

impl Default for RippleData {
    fn default() -> Self {
        Self(RippleStd430::default())
    }
}

/// The buffer every liquid material binds, written in place each frame.
///
/// A second shared buffer rather than more rows on the light one, and deliberately: the light
/// buffer's rows 20+ are a variable-length point-light table, so there is no free row to take —
/// only scattered spare lanes in unrelated rows, which is how a layout becomes unreadable.
#[derive(Resource, Clone, ExtractResource)]
pub struct RippleBuffer(pub Buffer);

/// Create the shared ripple buffer. Called from the same place the shared light buffer is built, so
/// material construction can clone it into its `#[storage(91, …)]` field.
pub fn new_ripple_buffer(device: &RenderDevice) -> RippleBuffer {
    RippleBuffer(device.create_buffer(&BufferDescriptor {
        label: Some("wow_liquid_ripple"),
        size: std::mem::size_of::<RippleStd430>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

/// The two displacement targets and where they are.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct RippleField {
    /// What the water material binds and what the wave step reads.
    pub(crate) map: Handle<Image>,
    /// The scratch the wave step writes, copied back into `map` in the same node.
    pub(crate) scratch: Handle<Image>,
    /// Min corner of the map's world square, Bevy XZ, snapped to a texel.
    origin: Vec2,
    /// Where it was last step, for the shader's reprojection.
    prev_origin: Vec2,
    /// Milliseconds owed to the simulation.
    accumulator: f32,
    /// How many steps the node should integrate this frame (0, 1 or [`MAX_CATCHUP`]).
    steps: u32,
    stamps: Vec<Vec4>,
}

impl RippleField {
    fn new(images: &mut Assets<Image>) -> Self {
        RippleField {
            map: images.add(target("liquid_ripple_map")),
            scratch: images.add(target("liquid_ripple_scratch")),
            origin: Vec2::ZERO,
            prev_origin: Vec2::ZERO,
            accumulator: 0.0,
            steps: 0,
            stamps: Vec::new(),
        }
    }
}

/// One displacement target.
///
/// `Rgba16Float` because the field is SIGNED — a wave is a trough as much as a crest, and a `Unorm`
/// format would clip every trough to zero and turn the ripple into a series of bumps. Nearest-free
/// filtering is wanted here (the water shader takes a gradient across texels), and `ClampToEdge`
/// keeps a bump-offset lookup at the rim from wrapping to the far side of the map.
fn target(label: &'static str) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: RIPPLE_SIZE,
            height: RIPPLE_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::RENDER_ATTACHMENT
        | TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        label: Some(label.into()),
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Nearest,
        ..default()
    });
    image
}

/// Build the field's images. Called from `surface::setup_liquid`, so the material can never be
/// built before its bindings exist.
pub(super) fn setup(images: &mut Assets<Image>) -> RippleField {
    RippleField::new(images)
}

/// Advance the fixed-step clock and move the map with the camera.
///
/// Split from the packing below so that [`collect_stamps`] can run BETWEEN them and know whether a
/// step is due. Without that split the stamps gathered on a non-step frame were packed and then
/// thrown away when the node declined to integrate — at 144 fps against a 20 Hz sim that is six
/// frames of disturbances discarded for every one kept, which is what turned a continuous wake into
/// an occasional splash.
pub(super) fn advance_clock(
    time: Res<Time>,
    mut field: ResMut<RippleField>,
    camera: Query<&Transform, With<WorldCamera>>,
) {
    let Ok(cam) = camera.single() else {
        return;
    };
    field.accumulator += (time.delta_secs() * 1000.0).min(500.0);
    field.steps = ((field.accumulator / STEP_MS) as u32).min(MAX_CATCHUP);
    if field.steps == 0 {
        return;
    }
    field.accumulator -= field.steps as f32 * STEP_MS;
    // Keep the remainder inside one step: it is the shader's interpolation numerator, and a value
    // above the period would extrapolate past the frame it is meant to sit between.
    field.accumulator = field.accumulator.clamp(0.0, STEP_MS);

    let texel = RIPPLE_EXTENT / RIPPLE_SIZE as f32;
    // Centred on the camera and floored to a whole texel — `Liquid::SetDisplMapInfo` does the same
    // `floorf(x · size/extent) · extent/size`. A sub-texel origin resamples the entire field every
    // frame, and a wave that is resampled every frame diffuses into nothing.
    let want = ((cam.translation.xz() - Vec2::splat(RIPPLE_EXTENT * 0.5)) / texel).floor() * texel;
    field.prev_origin = field.origin;
    field.origin = want;
}

/// Pack the blob the shader reads, and clear the stamps the step consumed.
pub(super) fn pack_ripple_data(mut field: ResMut<RippleField>, mut data: ResMut<RippleData>) {
    let inv_extent = 1.0 / RIPPLE_EXTENT;
    data.0.placement = [
        field.origin.x,
        field.origin.y,
        inv_extent,
        // How far between the two packed heights the water should read — the reference's
        // `GetDynamicDisplMapInterpolation`, its `(now − last) % period / period`, and the reason a
        // 20 Hz simulation is smooth at any frame rate.
        (field.accumulator / STEP_MS).clamp(0.0, 1.0),
    ];
    data.0.prev = [field.prev_origin.x, field.prev_origin.y, inv_extent, 0.0];
    data.0.params = [1.0 / RIPPLE_SIZE as f32, RIPPLE_SLOPE, 1.0, DAMPING];
    let n = field.stamps.len().min(MAX_STAMPS);
    data.0.stamp_count = [n as f32, 0.0, 0.0, 0.0];
    data.0.stamps = [[0.0; 4]; MAX_STAMPS];
    for (slot, s) in data.0.stamps.iter_mut().zip(field.stamps.iter()).take(n) {
        *slot = s.to_array();
    }
    if field.steps > 0 {
        field.stamps.clear();
    }
}

/// Collect this step's disturbances: the rain the reference itself simulates, and the wading bodies
/// this port adds.
///
/// **Only on step frames.** The field integrates at 20 Hz; a stamp gathered between steps has
/// nowhere to go, and gathering one per frame regardless simply meant most of them were packed and
/// then discarded.
///
/// **Rain is the reference's own source, and its only one.** `Liquid::AddPerturbation` has exactly
/// one caller in the whole binary — `CWorldMap`'s weather tick, scattering twenty drops per tick
/// within fifty yards whenever the type is rain and the intensity is over 0.2. The wading half is
/// this port's addition; `water_fx` carries the reference's own answer for a body in water, which
/// is a decal rather than a wave.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_stamps(
    time: Res<Time>,
    mut field: ResMut<RippleField>,
    mut last: Local<bevy::platform::collections::HashMap<Entity, Vec3>>,
    mut seed: Local<u32>,
    weather: Option<Res<crate::weather::WeatherState>>,
    camera: Query<&Transform, With<WorldCamera>>,
    units: Query<(Entity, &Transform, &WorldUnit), Without<ViewerUnit>>,
    water: Query<&WaterChunkInfo>,
) {
    let dt = time.delta_secs().max(1.0e-4);
    // Track every body every frame — velocity has to be differenced against the LAST frame, not the
    // last step, or a body would appear to teleport between steps.
    let mut seen = bevy::platform::collections::HashMap::new();
    let mut moving: Vec<(Vec3, f32, f32)> = Vec::new();
    for (entity, transform, unit) in &units {
        if !unit.wades {
            continue;
        }
        let pos = transform.translation;
        seen.insert(entity, pos);
        let Some(prev) = last.get(&entity) else {
            continue; // first sight — no velocity yet
        };
        let speed = (pos - *prev).length() / dt;
        if speed < 0.15 {
            continue; // standing still pushes nothing; the wave already made carries on by itself
        }
        moving.push((pos, speed, unit.height * unit.scale.max(0.1)));
    }
    *last = seen;

    if field.steps == 0 {
        return;
    }

    for (pos, speed, height) in moving {
        // Only where there IS water under it, and only while the body is actually in it.
        let wow = bevy_to_wow(pos);
        let Some(surface) = water_surface_at(water.iter(), wow, LiquidClaim::Outdoors) else {
            continue;
        };
        let depth = surface - wow[2];
        if !(0.0..=height * 2.0).contains(&depth) {
            continue;
        }
        // Falls off as the body submerges — a swimmer's shoulders push far less water than a
        // runner's legs — and rises a little with speed.
        let submerge = 1.0 - (depth / (height * 2.0)).clamp(0.0, 1.0);
        let amp = WADE_AMPLITUDE * (speed / 4.0).clamp(0.3, 1.6) * submerge;
        field
            .stamps
            .push(Vec4::new(pos.x, pos.z, WADE_RADIUS, amp));
        if field.stamps.len() >= MAX_STAMPS {
            return;
        }
    }

    // ---- Rain --------------------------------------------------------------------------------
    let Some(weather) = weather else {
        return;
    };
    if weather.effect_kind != crate::weather::WeatherKind::Rain
        || weather.effect_density <= RAIN_MIN_DENSITY
    {
        return;
    }
    let Ok(cam) = camera.single() else {
        return;
    };
    let budget = MAX_STAMPS.saturating_sub(field.stamps.len()).min(RAIN_DROPS);
    for _ in 0..budget {
        // A cheap LCG rather than `rand`: this runs every step and wants no allocation, no thread
        // local and no reproducibility contract beyond "different every drop".
        let mut next = || {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let (x, z) = (next(), next());
        let a = next();
        field.stamps.push(Vec4::new(
            cam.translation.x + (x * 2.0 - 1.0) * RAIN_SPREAD,
            cam.translation.z + (z * 2.0 - 1.0) * RAIN_SPREAD,
            STAMP_RADIUS,
            RAIN_AMPLITUDE.0 + a * (RAIN_AMPLITUDE.1 - RAIN_AMPLITUDE.0),
        ));
    }
}

// ---------------------------------------------------------------- render world

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct RippleLabel;

#[derive(Resource)]
struct RipplePipeline {
    layout: BindGroupLayoutDescriptor,
    propagate: CachedRenderPipelineId,
}

fn init_pipeline(
    mut commands: Commands,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/liquid_ripple.wgsl");
    let layout = BindGroupLayoutDescriptor::new(
        "liquid_ripple_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (
                    0,
                    storage_buffer_read_only_sized(
                        false,
                        Some(
                            std::num::NonZero::new(std::mem::size_of::<RippleStd430>() as u64)
                                .unwrap(),
                        ),
                    ),
                ),
                (1, texture_2d(TextureSampleType::Float { filterable: true })),
                (2, sampler(SamplerBindingType::Filtering)),
            ),
        ),
    );
    commands.insert_resource(RipplePipeline {
        propagate: pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("liquid_ripple_propagate".into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader,
                shader_defs: vec![],
                entry_point: Some("fs_propagate".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            ..default()
        }),
        layout,
    });
}

/// Write the packed blob into the shared buffer, before anything reads it.
fn upload_ripple(
    queue: Res<RenderQueue>,
    buffer: Option<Res<RippleBuffer>>,
    data: Option<Res<RippleData>>,
) {
    let (Some(buffer), Some(data)) = (buffer, data) else {
        return;
    };
    queue.write_buffer(&buffer.0, 0, bytemuck::bytes_of(&data.0));
}

#[derive(Default)]
struct RippleNode;

impl ViewNode for RippleNode {
    // The world camera only. A booth bake has no wading bodies and no business integrating a wave.
    type ViewQuery = &'static LiquidScreenView;

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        _marker: QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let (Some(field), Some(buffer)) = (
            world.get_resource::<RippleField>(),
            world.get_resource::<RippleBuffer>(),
        ) else {
            return Ok(());
        };
        if field.steps == 0 {
            return Ok(()); // between fixed steps — the water interpolates across the gap
        }
        let pipelines = world.resource::<RipplePipeline>();
        let cache = world.resource::<PipelineCache>();
        let Some(propagate) = cache.get_render_pipeline(pipelines.propagate) else {
            return Ok(());
        };
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let (Some(map), Some(scratch)) = (
            gpu_images.get(&field.map),
            gpu_images.get(&field.scratch),
        ) else {
            return Ok(());
        };

        let device = render_context.render_device().clone();
        let bind = device.create_bind_group(
            "liquid_ripple_bind",
            &cache.get_bind_group_layout(&pipelines.layout),
            &BindGroupEntries::with_indices((
                (0, buffer.0.as_entire_binding()),
                (1, &map.texture_view),
                (2, &map.sampler),
            )),
        );
        // `steps` integrations, each one map -> scratch -> map. The reference catches up the same
        // way and with the same cap; the stamps are applied on every one of them, which is right,
        // because a catch-up step stands for a slice of time the disturbance was also present for.
        for _ in 0..field.steps {
            {
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_render_pass(&RenderPassDescriptor {
                            label: Some("liquid_ripple_propagate"),
                            color_attachments: &[Some(RenderPassColorAttachment {
                                view: &scratch.texture_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: Operations::default(),
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                pass.set_pipeline(propagate);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..3, 0..1);
            }
            // Back into the texture the materials bind. See the module header: a GPU copy is
            // cheaper than making every liquid bind group chase a handle that swaps every frame.
            render_context.command_encoder().copy_texture_to_texture(
                scratch.texture.as_image_copy(),
                map.texture.as_image_copy(),
                Extent3d {
                    width: RIPPLE_SIZE,
                    height: RIPPLE_SIZE,
                    depth_or_array_layers: 1,
                },
            );
        }
        Ok(())
    }
}

/// Wire the ripple simulation into the app. Called from [`super::LiquidPlugin`].
pub(super) fn register(app: &mut App) {
    app.init_resource::<RippleData>()
        .add_plugins(ExtractResourcePlugin::<RippleData>::default())
        .add_plugins(ExtractResourcePlugin::<RippleField>::default())
        .add_plugins(ExtractResourcePlugin::<RippleBuffer>::default())
        // Strictly ordered: the clock decides whether a step is due, the collector fills only when
        // one is, and the packer publishes what it collected. See `collect_stamps` for what the
        // wrong order costs.
        .add_systems(
            Update,
            (advance_clock, collect_stamps, pack_ripple_data).chain(),
        );

    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .add_systems(RenderStartup, init_pipeline)
        .add_systems(Render, upload_ripple.in_set(RenderSystems::PrepareResources))
        .add_render_graph_node::<ViewNodeRunner<RippleNode>>(Core3d, RippleLabel)
        // Before the transparent pass, where the water reads the field. It has no dependency on
        // anything the opaque pass produced — the sim is world-space and self-contained — so this
        // edge is about being finished in time, not about ordering against a producer.
        .add_render_graph_edges(
            Core3d,
            (Node3d::MainOpaquePass, RippleLabel, Node3d::MainTransparentPass),
        );
}
