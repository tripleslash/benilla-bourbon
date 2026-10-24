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

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_3d::{main_opaque_pass_3d, main_transparent_pass_3d};
use bevy::core_pipeline::{Core3d, Core3dSystems, FullscreenShader};
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, texture_2d,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::screen::LiquidScreenView;
use crate::view::WorldCamera;
use crate::world_point::{Subject, WorldPoint};
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

/// How much world the map covers, in yards — **the reference's own 102.4**, and it is a read
/// number rather than a fitted one.
///
/// `CWorldScene`'s render entry (`0xD488F0`) calls `Liquid::SetDisplMapInfo(s_playerPos, {128,
/// 128})` — the two `1124073472` immediates, which are `128.0f` — and `SetDisplMapInfo`
/// (`0x52C2A0`) multiplies that by the
/// per-map factor `dword_FF7A38[i]` before centring it on the player and flooring the corner to a
/// texel. That factor is data, not code, so it is read from the binary the decomp came from
/// (the 4.1.0 build 13850 Mac client, Mach-O i386, `__TEXT,__const` at VA `0xFF7A38`): **{0.8, 4.0}**. The single-map
/// configuration this file ports is map 0, so `128 × 0.8 = 102.4`.
///
/// Three numbers agree on it and are the reason to trust it over the 50 this constant used to
/// carry:
/// - a texel is **exactly 0.1 yd** (102.4 / 1024), which is a designed number, not a fitted one;
/// - `PerturbDisplMap` sizes a stamp `floor(radius / extent · size)` texels, so the 0.3-yd raindrop
///   is **exactly 3 texels** — at 50 it was 6, and at map 1's 512 it floors to 0 and vanishes;
/// - the rain emitter scatters ±50 yd ([`RAIN_SPREAD`]), which fits inside the ±51.2 half-width
///   with room to spare. At 50 the map's half-width was 25, so **three of every four raindrops fell
///   outside the field and did nothing** — which, with the halved wave speed and the missing
///   [`PERTURB_SCALE`], is why rain made no visible ripples at all.
///
/// The 50 it replaces came from reading the rain scatter as ±25 (it is ±50: `((rand%1000)/500 − 1)
/// · 50`, `CWorldMap.c`) and fitting the extent to that.
pub const RIPPLE_EXTENT: f32 = 102.4;

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

/// The most disturbances integrated in one step — **must match `liquid_ripple.wgsl`'s
/// `MAX_STAMPS`**, and this file is where the number lives.
///
/// Sized for RAIN, which is the reference's own source and by far the heaviest: [`RAIN_DROPS`] a
/// step, plus room for the bodies, which contribute one each and never come close. The shader's
/// copy sat at 16 while this said 32 — the two are a single ABI and nothing checks them, so the
/// mismatch was silent: WGSL clamps an out-of-range constant index, so every rain drop past the
/// sixteenth was quietly stamped on top of the sixteenth instead of where it fell.
const MAX_STAMPS: usize = 48;

/// A perturbation's radius in yards — **the reference's 0.3**, read as the float literal
/// `1050253722` (`0x3E99999A`) in its rain emitter's `Liquid::AddPerturbation` call.
const STAMP_RADIUS: f32 = 0.3;

/// A raindrop's amplitude range — the reference's `−(rand()%200/1000 + 0.2)`, i.e. −0.2 to −0.4.
///
/// **Negative**, and that is the physics rather than a sign convention: a drop presses the surface
/// DOWN and the wave equation does the rest. A positive stamp would model a thing rising out of the
/// water.
const RAIN_AMPLITUDE: (f32, f32) = (-0.4, -0.2);

/// Rain drops per simulation step, and how far from the camera they scatter.
///
/// The reference emits **20 per 33 ms** — its weather tick's own cadence (`CWorldMap`, `elapsed /
/// 0x21` ticks, capped at four, `20 × ticks` drops), which is the render loop's rate and NOT the
/// displacement map's 50 ms step. Its emission is decoupled from the simulation because a
/// perturbation added between steps is still stamped that frame and only propagates at the next
/// one. This port collects on step frames only (see [`collect_stamps`]), so the rate is carried
/// into the step instead: `20 × 50/33.3` = **30 a step** is the same 600 drops a second.
///
/// The spread is `((rand%1000)/500 − 1) · 50` in each axis — ±50 yards, inside the map's ±51.2.
const RAIN_DROPS: usize = 30;
const RAIN_SPREAD: f32 = 50.0;

/// The density from which rain rings the water at its full rate — the reference's `> 0.2` gate.
///
/// Deviation: the reference emits nothing at or below it, which leaves a light rain (a server's
/// intensity under 0.4, a density under 0.2) falling into still water. Below it the drop count
/// here falls off with the density instead, so every rain that draws drops rings in proportion.
const RAIN_MIN_DENSITY: f32 = 0.2;

/// Rain drops this step at `density`: [`RAIN_DROPS`] past [`RAIN_MIN_DENSITY`], a linear share of
/// it below, none without rain.
fn rain_drops(density: f32) -> usize {
    if density <= 0.0 {
        return 0;
    }
    let share = (density / RAIN_MIN_DENSITY).min(1.0);
    (RAIN_DROPS as f32 * share).ceil() as usize
}

/// A wading body's radius and amplitude.
///
/// **The reference has no such source.** Its dynamic displacement map is driven by rain and by
/// mesh-shaped disturbances only; a walking character's ripple is the separate `CWater0Ripple`
/// decal system, which benilla carries in `water_fx`. These two numbers are therefore a
/// deliberate addition, scaled off the raindrop the reference does specify: half again its radius,
/// and an amplitude in the same band, because a leg entering water is a bigger version of the same
/// event and should not be a different order of magnitude.
///
/// Adding it made the wading body a thing benilla draws **twice**, and the decal is the half that
/// gives way: `water_fx`'s emitter stands down on the cata path (its `emit_water_foam` carries the
/// reasoning) and keeps the whole of the legacy one, so each renderer has exactly one wake.
const WADE_RADIUS: f32 = 0.45;
const WADE_AMPLITUDE: f32 = -0.30;

/// Below this planar speed (yd/s) a body stamps nothing — standing in water pushes none of it, and
/// the wave already made carries on by itself.
///
/// Named rather than repeated because [`collect_stamps`] applies it in TWO places: the avatar,
/// whose position comes off [`crate::view::Viewer`], and the streamed units, whose comes off their
/// `Transform`. Two branches with one threshold between them is exactly the shape that drifts.
const WADE_MIN_SPEED: f32 = 0.15;

/// **The amplitude scale every perturbation is multiplied by**, and the clamp that follows it —
/// `PerturbDisplMap`'s `clamp(amp · dword_FF7A30[i], −1, 1)`.
///
/// Read from the same binary as [`RIPPLE_EXTENT`]'s factor, four bytes below it (VA `0xFF7A30`):
/// **{1.25, 0.25}**, map 0's being the 1.25. It is per-map rather than per-perturbation, which is
/// what makes it a property of the field and not of the raindrop: map 1 covers five times the world
/// at a fifth of the amplitude.
const PERTURB_SCALE: f32 = 1.25;

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
    /// fork: modern lighting — the sky dome's five gradient stops (zenith → 1.8°) and its horizon
    /// colour ([`crate::sky::dome_stops`]), for the water's sky reflection wherever the mirror has
    /// no picture. On this buffer because it is the water's own and already rewritten every frame:
    /// the tail costs 96 bytes of an upload that happens anyway, and no binding. `liquid_ripple.wgsl`
    /// declares the struct only up to `stamps`, which a larger buffer binds against.
    sky: [[f32; 4]; SKY_ROWS],
}

/// The sky rows at the blob's tail: the dome's five stops and its fog/horizon colour.
const SKY_ROWS: usize = 6;

impl Default for RippleStd430 {
    fn default() -> Self {
        Self {
            placement: [0.0; 4],
            params: [1.0 / RIPPLE_SIZE as f32, RIPPLE_SLOPE, 0.0, DAMPING],
            prev: [0.0; 4],
            stamp_count: [0.0; 4],
            stamps: [[0.0; 4]; MAX_STAMPS],
            sky: [[0.0; 4]; SKY_ROWS],
        }
    }
}

/// fork: modern lighting — copy the sky dome's stops into the blob's sky rows, so the water can
/// reflect the sky that is drawn when the mirror has nothing (not rendered yet, or about another
/// plane). One small copy a frame into a blob that is uploaded every frame regardless.
pub(super) fn pack_sky_rows(
    light: Option<Res<crate::lighting::WowLighting>>,
    mut data: ResMut<RippleData>,
) {
    let Some(light) = light else {
        return;
    };
    let (sky, fog) = crate::sky::dome_stops(&light);
    for (row, stop) in data
        .0
        .sky
        .iter_mut()
        .zip(sky.iter().chain(std::iter::once(&fog)))
    {
        *row = stop.to_array();
    }
}

/// The per-frame blob, extracted to the render world for upload.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub(super) struct RippleData(RippleStd430);

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

/// One disturbance, in the packed form the shader reads: `xy` world XZ, `z` radius in yards, `w`
/// amplitude — with [`PERTURB_SCALE`] and the reference's clamp already applied.
///
/// The scale belongs on the way IN rather than in the shader because it is a property of the map,
/// not of the event: every perturbation of a given map takes the same factor, and folding it in here
/// means the stamp array carries final amplitudes and the kernel stays the kernel.
fn perturbation(x: f32, z: f32, radius: f32, amplitude: f32) -> Vec4 {
    Vec4::new(x, z, radius, (amplitude * PERTURB_SCALE).clamp(-1.0, 1.0))
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
///
/// **The wet test goes through [`WorldPoint`], with each body's own [`Subject`].** It used to hand
/// [`LiquidClaim::Outdoors`] to the raw surface walk, which is not "wherever this body is" — it is
/// the ADT's MCLQ and *nothing else* (`WaterChunkInfo::answers`: an `Outdoors` claim rejects every
/// `LiquidSource::WmoGroup`). So a body wading in a WMO pool — Stormwind's canals, the fountains,
/// every interior pool in the game — pushed no wave at all, and the water it was standing in was
/// the one water in the world with no dynamic displacement on it. The claim is per-subject for the
/// same reason every other liquid query's is (0634/0696/0701), and asking for it is one call.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_stamps(
    time: Res<Time>,
    mut field: ResMut<RippleField>,
    mut last: Local<bevy::platform::collections::HashMap<Entity, Vec3>>,
    mut seed: Local<u32>,
    weather: Option<Res<crate::weather::WeatherState>>,
    viewer: Res<crate::view::Viewer>,
    self_unit: Query<&WorldUnit, With<ViewerUnit>>,
    camera: Query<&Transform, With<WorldCamera>>,
    units: Query<(Entity, &Transform, &WorldUnit), Without<ViewerUnit>>,
    world: WorldPoint,
) {
    let dt = time.delta_secs().max(1.0e-4);
    // Track every body every frame — velocity has to be differenced against the LAST frame, not the
    // last step, or a body would appear to teleport between steps.
    let mut seen = bevy::platform::collections::HashMap::new();
    // The subject rides along with the body: the wet test below is scoped by whose liquid answers,
    // and by the time it runs the entity the claim is looked up from is out of scope.
    let mut moving: Vec<(Subject, Vec3, f32, f32)> = Vec::new();

    // **The avatar, and it needs its own branch** — `Without<ViewerUnit>` below is not a way of
    // saying "everyone", it is half of a pair. The viewer's body is not read off a `Transform` like
    // a streamed unit's: the eye's own position is authoritative and lives on [`crate::view::Viewer`]
    // (`emit_water_foam` splits the same two cases for the same reason, and its avatar half is the
    // one this was missing). Taking only the `Without` half meant the one body the player actually
    // controls was the one body that never stamped — and because the decal emitter now stands down
    // on the cata path, its wake had no second source to fall back to. It went unseen because the
    // wade probe it was verified with is a streamed unit, not the viewer.
    if let Some(body) = viewer.at {
        seen.insert(Entity::PLACEHOLDER, body);
        if let Some(prev) = last.get(&Entity::PLACEHOLDER) {
            let speed = (body - *prev).length() / dt;
            if speed >= WADE_MIN_SPEED {
                // `Viewer::height` is the collision cylinder in yards, unscaled — the same input
                // the decal ring takes — so the unit's scale is applied here exactly as the
                // streamed branch applies its own.
                let scale = self_unit.single().map(|u| u.scale).unwrap_or(1.0);
                moving.push((Subject::Player, body, speed, viewer.height * scale.max(0.1)));
            }
        }
    }

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
        if speed < WADE_MIN_SPEED {
            continue; // standing still pushes nothing; the wave already made carries on by itself
        }
        moving.push((
            Subject::Unit(entity),
            pos,
            speed,
            unit.height * unit.scale.max(0.1),
        ));
    }
    *last = seen;

    if field.steps == 0 {
        return;
    }

    for (who, pos, speed, height) in moving {
        // Only where there IS water under it, and only while the body is actually in it — and it is
        // *this body's* water, not the open world's: outdoors that is the ADT's MCLQ, inside a
        // building it is that placement's own MLIQ pool (see the header note).
        let wow = bevy_to_wow(pos);
        let Some(surface) = world.water_surface_at(who, wow) else {
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
            .push(perturbation(pos.x, pos.z, WADE_RADIUS, amp));
        if field.stamps.len() >= MAX_STAMPS {
            return;
        }
    }

    // ---- Rain --------------------------------------------------------------------------------
    //
    // The field's only weather source: a point stamp spreads as a ring, which reads as a raindrop
    // whatever made it, so nothing stamps the open water in clear weather. A modern frame's wind
    // is in the shader's normal (`sunshadow::wind_swell`, `liquid_cata.wgsl`'s ripple trains).
    let Some(weather) = weather else {
        return;
    };
    if weather.effect_kind != crate::weather::WeatherKind::Rain {
        return;
    }
    let Ok(cam) = camera.single() else {
        return;
    };
    let budget = MAX_STAMPS
        .saturating_sub(field.stamps.len())
        .min(rain_drops(weather.effect_density));
    for _ in 0..budget {
        // A cheap LCG rather than `rand`: this runs every step and wants no allocation, no thread
        // local and no reproducibility contract beyond "different every drop".
        let mut next = || {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let (x, z) = (next(), next());
        let a = next();
        field.stamps.push(perturbation(
            cam.translation.x + (x * 2.0 - 1.0) * RAIN_SPREAD,
            cam.translation.z + (z * 2.0 - 1.0) * RAIN_SPREAD,
            STAMP_RADIUS,
            RAIN_AMPLITUDE.0 + a * (RAIN_AMPLITUDE.1 - RAIN_AMPLITUDE.0),
        ));
    }
}

// ---------------------------------------------------------------- render world

#[derive(Resource)]
struct RipplePipeline {
    layout: BindGroupLayoutDescriptor,
    propagate: CachedRenderPipelineId,
    /// The blob the CATCH-UP steps read: this frame's, with `prev` set to `placement`.
    ///
    /// The reprojection moves the field from last step's square to this one, which is a thing that
    /// happens ONCE per frame. A second step in the same frame reads a map that is already over this
    /// frame's square, so reprojecting it again shifts every wave by the camera's movement a second
    /// time — below 20 fps (or on any hitch) the ripples dragged along with a turning camera instead
    /// of staying where they were made. The first step binds the shared buffer; the rest bind this.
    settled: Buffer,
}

fn init_pipeline(
    mut commands: Commands,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
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
        settled: device.create_buffer(&BufferDescriptor {
            label: Some("wow_liquid_ripple_settled"),
            size: std::mem::size_of::<RippleStd430>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
    });
}

/// Write the packed blob into the shared buffer, before anything reads it.
fn upload_ripple(
    queue: Res<RenderQueue>,
    buffer: Option<Res<RippleBuffer>>,
    data: Option<Res<RippleData>>,
    field: Option<Res<RippleField>>,
    pipelines: Option<Res<RipplePipeline>>,
) {
    let (Some(buffer), Some(data)) = (buffer, data) else {
        return;
    };
    queue.write_buffer(&buffer.0, 0, bytemuck::bytes_of(&data.0));
    // Only a catch-up frame reads the settled copy.
    if let (Some(field), Some(pipelines)) = (field, pipelines) {
        if field.steps > 1 {
            let mut settled = data.0;
            settled.prev = settled.placement;
            queue.write_buffer(&pipelines.settled, 0, bytemuck::bytes_of(&settled));
        }
    }
}

fn ripple_pass(
    world: &World,
    view: ViewQuery<&LiquidScreenView>,
    mut render_context: RenderContext,
) {
    let _marker = view.into_inner();
    {
        let (Some(field), Some(buffer)) = (
            world.get_resource::<RippleField>(),
            world.get_resource::<RippleBuffer>(),
        ) else {
            return;
        };
        if field.steps == 0 {
            return; // between fixed steps — the water interpolates across the gap
        }
        let pipelines = world.resource::<RipplePipeline>();
        let cache = world.resource::<PipelineCache>();
        let Some(propagate) = cache.get_render_pipeline(pipelines.propagate) else {
            return;
        };
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let (Some(map), Some(scratch)) =
            (gpu_images.get(&field.map), gpu_images.get(&field.scratch))
        else {
            return;
        };

        let device = render_context.render_device().clone();
        let layout = cache.get_bind_group_layout(&pipelines.layout);
        let bind = device.create_bind_group(
            "liquid_ripple_bind",
            &layout,
            &BindGroupEntries::with_indices((
                (0, buffer.0.as_entire_binding()),
                (1, &map.texture_view),
                (2, &map.sampler),
            )),
        );
        let settled = (field.steps > 1).then(|| {
            device.create_bind_group(
                "liquid_ripple_bind_settled",
                &layout,
                &BindGroupEntries::with_indices((
                    (0, pipelines.settled.as_entire_binding()),
                    (1, &map.texture_view),
                    (2, &map.sampler),
                )),
            )
        });
        // `steps` integrations, each one map -> scratch -> map. The reference catches up the same
        // way and with the same cap; the stamps are applied on every one of them, which is right,
        // because a catch-up step stands for a slice of time the disturbance was also present for.
        for step in 0..field.steps {
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
                            multiview_mask: None,
                        });
                pass.set_pipeline(propagate);
                // Reproject on the first step only — see `RipplePipeline::settled`.
                pass.set_bind_group(
                    0,
                    settled.as_ref().filter(|_| step > 0).unwrap_or(&bind),
                    &[],
                );
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
        // wrong order costs. Only once `surface::setup_liquid` has built the field: with no client
        // data it bails before that, and there is no water to ripple.
        .add_systems(
            Update,
            (advance_clock, collect_stamps, pack_ripple_data)
                .chain()
                .run_if(resource_exists::<RippleField>),
        )
        .add_systems(
            Update,
            pack_sky_rows.in_set(crate::lighting::LightingConsumeSet),
        );

    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .add_systems(RenderStartup, init_pipeline)
        .add_systems(
            Render,
            upload_ripple.in_set(RenderSystems::PrepareResources),
        )
        // Before the transparent pass, where the water reads the field. It has no dependency on
        // anything the opaque pass produced — the sim is world-space and self-contained — so this
        // order is about being finished in time, not about ordering against a producer.
        .add_systems(
            Core3d,
            ripple_pass
                .in_set(Core3dSystems::MainPass)
                .after(main_opaque_pass_3d)
                .before(main_transparent_pass_3d),
        );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liquid::{LiquidSource, WaterChunkInfo, WmoPool};
    use crate::view::Viewer;
    use crate::wmo_portal::{PlayerWmoRoom, WmoRoom};
    use crate::world_point::init_world_point_resources;
    use benilla_assets::coords::wow_to_bevy;
    use benilla_formats::LiquidKind;
    use bevy::ecs::system::RunSystemOnce;
    use std::time::Duration;

    /// The blob's layout is copied by hand into both shaders that bind it, and `liquid_cata.wgsl`
    /// reads the sky rows PAST the stamp array — so a stamp count that drifts in one copy shifts the
    /// water's sky reflection onto stamp data with no error anywhere. Pin all three copies.
    #[test]
    fn both_shaders_declare_the_blob_this_packs() {
        let cata = include_str!("../../../benilla-assets/src/shaders/liquid_cata.wgsl");
        let step = include_str!("../shaders/liquid_ripple.wgsl");
        let stamps = format!("stamps: array<vec4<f32>, {MAX_STAMPS}>");
        assert!(
            cata.contains(&stamps),
            "liquid_cata.wgsl must declare `{stamps}`"
        );
        let sky = format!("sky: array<vec4<f32>, {SKY_ROWS}>");
        assert!(cata.contains(&sky), "liquid_cata.wgsl must declare `{sky}`");
        let count = format!("const MAX_STAMPS: u32 = {MAX_STAMPS}u;");
        assert!(
            step.contains(&count),
            "liquid_ripple.wgsl must declare `{count}`"
        );
    }

    /// One flat 20-yd-square wet cell at `z`, WoW `x`/`y` in `[0, 20]`.
    fn pool(source: LiquidSource, z: f32) -> WaterChunkInfo {
        let positions = vec![
            [0.0, 0.0, z],
            [20.0, 0.0, z],
            [0.0, 20.0, z],
            [20.0, 20.0, z],
        ];
        WaterChunkInfo::new(source, LiquidKind::Still, [2, 2], positions, vec![true])
    }

    /// The pool of the placement `owner`'s group 0, with no floor — the shape a WMO fountain or
    /// canal spawns with.
    fn wmo_source(owner: Entity) -> LiquidSource {
        LiquidSource::WmoGroup(WmoPool {
            owner: Some(WmoRoom {
                instance: owner,
                group: 0,
            }),
            floor: f32::NEG_INFINITY,
        })
    }

    /// How many stamps a viewer wading across `source`'s pool leaves, with the player's interior
    /// claim set to `room`.
    ///
    /// Two runs of one REGISTERED system, not two `run_system_once`s: the wade speed is differenced
    /// against a `Local`, and a fresh system has no last frame to difference against.
    fn stamps_from_a_wading_viewer(source: LiquidSource, room: Option<WmoRoom>) -> usize {
        let mut world = World::new();
        init_world_point_resources(&mut world);
        world.init_resource::<Assets<Image>>();
        world.init_resource::<Time>();
        world.resource_mut::<PlayerWmoRoom>().0 = room;

        let mut field = {
            let mut images = world.resource_mut::<Assets<Image>>();
            setup(&mut images)
        };
        // The step the collector only fills on. `advance_clock` owns this in the live schedule;
        // there is no clock to advance here, only the wade to test.
        field.steps = 1;
        world.insert_resource(field);

        world.spawn(pool(source, 5.0));
        world
            .run_system_once(super::super::maintain_water_index)
            .expect("index");

        // WoW (10, 10, 4) — inside the cell, one yard under the surface.
        world.insert_resource(Viewer {
            at: Some(wow_to_bevy([10.0, 10.0, 4.0])),
            height: 2.0,
            ..default()
        });
        let collect = world.register_system(collect_stamps);
        world.run_system(collect).expect("first frame");

        // Half a yard along WoW x in 50 ms — 10 yd/s, well over `WADE_MIN_SPEED`.
        world
            .resource_mut::<Time>()
            .advance_by(Duration::from_millis(50));
        world.resource_mut::<Viewer>().at = Some(wow_to_bevy([10.5, 10.0, 4.0]));
        world.run_system(collect).expect("second frame");

        world.resource::<RippleField>().stamps.len()
    }

    /// **A body wading in a WMO pool pushes a wave.** The stamp gate used to hand the surface walk
    /// a hardcoded [`LiquidClaim::Outdoors`], which is not "wherever this body is" — it is the ADT's
    /// MCLQ and nothing else, so every WMO pool in the game (Stormwind's canals and fountains, every
    /// interior pool) was the one water with no dynamic displacement on it.
    #[test]
    fn a_wmo_pool_takes_a_wading_bodys_wave() {
        let owner = Entity::from_raw_u32(7).expect("valid entity id");
        let room = WmoRoom {
            instance: owner,
            group: 0,
        };
        assert_eq!(
            stamps_from_a_wading_viewer(wmo_source(owner), Some(room)),
            1,
            "a viewer wading in the pool of the room it is standing in leaves a stamp"
        );
    }

    /// The other half of the same law: the claim still SCOPES. A pool the player is not inside of
    /// answers for nobody, so widening the gate must not turn every pool on the map into a wake.
    #[test]
    fn a_pool_in_another_building_takes_nothing() {
        let owner = Entity::from_raw_u32(7).expect("valid entity id");
        assert_eq!(
            stamps_from_a_wading_viewer(wmo_source(owner), None),
            0,
            "outdoors, a WMO pool does not answer"
        );
    }

    /// And the ADT path it never lost.
    #[test]
    fn map_liquid_still_takes_the_wave() {
        assert_eq!(
            stamps_from_a_wading_viewer(LiquidSource::AdtChunk, None),
            1,
            "outdoors, the ADT's own MCLQ answers"
        );
    }
}
