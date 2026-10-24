//! **Froxel volumetrics** (fork-owned; `FORK.md` P4).
//!
//! Hillaire 2015 on a camera-aligned froxel grid, driven by the zone's own fog:
//!
//! - [`media`] solves the authored linear GL fog into extinction, height falloff and in-scatter.
//! - `update_volume` (main world, `Last`) packs this frame's medium, the main camera's froxel
//!   projection, last frame's for the history, and the cloud-shadow grid into [`ModernVolume`].
//! - `node` (render world) writes that block behind the cascades in the shared modern storage buffer
//!   (`sunshadow::ModernBlock`), runs the scatter and integrate compute passes into the
//!   constant-handle volume ([`MODERN_FOG_VOLUME`]) before the main pass, and composites the fog over
//!   the opaque frame and the sky between the opaque and transparent passes — on the world camera and
//!   on the water mirror, which looks the volume up through the main camera's projection.
//! - Transparent lanes (blended models, water, particles) sample the same volume themselves.
//!
//! The Low tier runs no compute: the composite and the transparents take the analytic height fog,
//! and the composite adds 16-tap screen-space shafts.

use bevy::asset::RenderAssetUsages;
use bevy::camera::Projection;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::RenderApp;

use benilla_assets::materials::{
    LiquidCataMaterial, LiquidMaterial, WowModelMaterial, MODERN_FOG_VOLUME,
};

use super::{smoothstep, ActiveLighting, LightingModel, LightingQuality, LightingSettings};
use crate::lighting::WowLighting;
use crate::liquid::LiquidReflectionCamera;
use crate::view::WorldCamera;

pub mod media;
mod node;

// fork: modern lighting (FORK.md, P5) — the probe trace orders before this pass.
pub(crate) use node::froxel_pass;
// fork: fsr (FORK.md) — FSR's opaque copy and velocity pass order against these two.
pub(crate) use node::{fog_composite_pass, lamp_halo_pass};

/// Size of [`FroxelFog::std430`] — `froxelfog.wgsl`'s `FroxelFog`.
pub const FOG_BYTES: usize = 3 * 64 + 23 * 16;
/// Offset of the `ground_x` row inside the tail: the render world writes the terrain heightfield's
/// addressing there (`node::prepare_froxels`), which only it knows.
pub(crate) const GROUND_OFFSET: usize = 3 * 64 + 19 * 16;
/// The cloud-shadow grid's cells per side.
pub const CLOUD_CELLS: usize = 32;
/// Everything this module writes behind the cascades in the modern storage block.
pub const BLOCK_TAIL_BYTES: usize = FOG_BYTES + CLOUD_CELLS * CLOUD_CELLS * 4;
/// Offset of `alloc.w` (the mode) inside the tail — the render world downgrades a froxel frame to
/// analytic there while the compute passes are not ready.
pub(crate) const MODE_OFFSET: usize = 3 * 64 + 3 * 16 + 12;

/// The integrated volume's allocation: the largest grid the platform's tiers use. A tier writes and
/// reads its own corner of it, so a quality change never re-creates the texture the materials bind.
pub const VOLUME_ALLOC: UVec3 = if cfg!(target_arch = "wasm32") {
    UVec3::new(160, 90, 64)
} else {
    UVec3::new(192, 108, 96)
};

/// One cloud-shadow cell (yd); the grid covers 768 yd around the camera.
const CLOUD_CELL_YD: f32 = 24.0;
/// The cloud layer's height above the camera (yd), for projecting cloud cover down the key light.
const CLOUD_HEIGHT: f32 = 350.0;
/// Scatter history weight on a steady frame (was the plan's 0.9: beams shimmered with the jitter).
const HISTORY_WEIGHT: f32 = 0.95;
/// fork: **the night air** — haze extinction (1/yd) added in every zone overnight (scaled by the
/// SIDN night fraction), so a night city layers into a lit haze with distance, as the Forever
/// Orgrimmar night shot does. `froxel_scatter.wgsl` keeps `NIGHT_NEAR_SHARE` of it at the camera and
/// all of it past `NIGHT_CLEAR_FAR`, so the street stays readable and the lamps around the player
/// still have air to light. By day it is zero. It rides `room_media.z`, apart from the zone's own
/// shaft haze.
const NIGHT_HAZE: f32 = 0.004;
/// fork: the night air's own glow: the zone's fog colour (decoded) times this, times the night
/// fraction and the zone look's `night_glow`, is the radiance it scatters on top of the haze's. The
/// night fog alone solves to a near black veil (Orgrimmar's is 0.011 linear), where the reference's
/// night haze is a lit purple.
const NIGHT_GLOW: f32 = 5.0;
/// fork: **the air is as faint by day as it is at midnight.** The zone rows' shaft haze was fitted
/// to reference shots, and in play the daylight lit it into a veil — Goldshire at dusk sat in an
/// orange wash, while the same air under the moon read as the faint haze the user wants at every
/// hour. Every row's haze extinction is scaled by this, so the zones keep their
/// character against each other (Ashenvale still hazier than Durotar) but none of them fogs the
/// street the player stands in. Halfway to the thinnest tried (0.25 / 0.5), which cleared noon but
/// lost the dusk glow. The live `WOW_MODERN_LOOK` override is scaled too.
const HAZE_SCALE: f32 = 0.67;
/// fork: **ground fog** — the banks' extinction at the ground (1/yd) at the full SIDN night, which
/// falls off within a few yards of the terrain under each froxel (`froxel_scatter.wgsl`'s
/// `BANK_INV_FALLOFF`; the GI heightfield, past it the ground under the camera).
/// `froxel_scatter.wgsl` breaks them into patches and lights them with the shaft haze's key.
/// Overnight a zone look's `night_banks` takes its share (none in an ordinary zone, the whole bank
/// in a dark one), and its `ground_fog` keeps a share by day. Anchored to one plane at the camera's
/// ground, 0.035 filled Hillsbrad's low ground and left Gilneas's hills bare.
const BANK_EXTINCTION: f32 = 0.022;
/// fork: under a risen moon the night air's uniform glow drops to this share: the moonlight and
/// the banks carry the night instead of a veil over everything.
const MOON_VEIL: f32 = 0.4;
/// fork: **a hazy zone stays as hazy as it is.** The zone's own air by day (1/yd: its shaft haze,
/// its day banks and any sea fog) up to which the night air and rain's humid veil come in whole
/// (Elwynn's haze), and from which a zone counts as hazy already (Darkshore's coast, the eerie
/// zones' banks) and takes only [`HAZY_SHARE`] of them: the night and a downpour add a touch, never
/// a white-out over air that was thick at noon.
const CLEAR_AIR: f32 = 0.0017;
const HAZY_AIR: f32 = 0.005;
const HAZY_SHARE: f32 = 0.15;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FogMode {
    #[default]
    Off,
    /// Analytic height fog with screen-space shafts (Low).
    Analytic,
    /// The froxel volume (Medium and up).
    Froxel,
}

impl FogMode {
    fn as_f32(self) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::Analytic => 1.0,
            Self::Froxel => 2.0,
        }
    }
}

/// What a quality tier runs — the plan's tier table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FroxelTier {
    pub mode: FogMode,
    pub grid: UVec3,
    /// The grid's far plane (planar yd); the analytic tail takes over beyond it.
    pub range: f32,
    /// Nearest point lights scattered per froxel.
    pub point_lights: u32,
    /// Cloud shadow strength on the key light (0 = no cookie).
    pub cloud_shadow: f32,
    /// Screen-space shaft strength (the Low tier's stand-in for froxel shafts).
    pub shafts: f32,
}

pub fn froxel_tier(quality: LightingQuality) -> FroxelTier {
    let (mode, grid, range, point_lights, cloud_shadow, shafts) = match quality {
        LightingQuality::Low => (FogMode::Analytic, UVec3::ONE, 200.0, 0, 0.0, 0.9),
        // No cloud shadows on any tier: the cloud layer is a dome that travels with the camera, so its
        // shadows slid over the ground with the player — dark patches drifting ahead of every step.
        LightingQuality::Medium => (FogMode::Froxel, UVec3::new(96, 54, 48), 350.0, 8, 0.0, 0.0),
        LightingQuality::High => (
            FogMode::Froxel,
            UVec3::new(160, 90, 64),
            450.0,
            16,
            0.0,
            0.0,
        ),
        // fork: P8b — the point-light count is only the fallback walk's now (`?modern_lights=0`): with
        // the light grid live a froxel reads the lights of its own cell (`froxel_scatter.wgsl`).
        // Ultra lights from glowing things as Cinematic does, so it takes Cinematic's count.
        LightingQuality::Ultra => (
            FogMode::Froxel,
            UVec3::new(192, 108, 96),
            600.0,
            64,
            0.0,
            0.0,
        ),
        // fork: P8 — the same grid as Ultra (the volume is allocated once at the platform's largest,
        // so a finer one would need a bigger allocation on every tier).
        LightingQuality::Cinematic => (
            FogMode::Froxel,
            UVec3::new(192, 108, 96),
            600.0,
            64,
            0.0,
            0.0,
        ),
    };
    FroxelTier {
        mode,
        grid: grid.min(VOLUME_ALLOC),
        range,
        point_lights,
        cloud_shadow,
        shafts,
    }
}

/// The CPU mirror of `froxelfog.wgsl`'s `FroxelFog` (field order and meaning documented there).
#[derive(Clone, Copy, Debug, Default)]
pub struct FroxelFog {
    pub clip_from_world: Mat4,
    pub world_from_clip: Mat4,
    pub prev_clip_from_world: Mat4,
    pub camera: Vec4,
    pub view_row: Vec4,
    pub grid: Vec4,
    pub alloc: Vec4,
    pub media: Vec4,
    pub medium: Vec4,
    pub ambient: Vec4,
    pub sun: Vec4,
    pub sun_dir: Vec4,
    pub phase: Vec4,
    pub cloud: Vec4,
    pub proj: Vec4,
    pub screen: Vec4,
    pub wind: Vec4,
    pub room: Vec4,
    pub room_media: Vec4,
    pub room_ambient: Vec4,
    pub room_sun: Vec4,
    pub night: Vec4,
    /// The terrain heightfield's world-to-uv rows and its `hf` row (`probegi::GiClip`), written in
    /// the render world ([`GROUND_OFFSET`]); zero (off) as the main world packs it.
    pub ground_x: Vec4,
    pub ground_y: Vec4,
    pub ground: Vec4,
    /// fork: the lamp halos' clear-weather air: x = fog extinction, y = near-clear range, z = haze;
    /// w = the key's share on the shaft haze's and banks' isotropic half (`look::key_air_share`).
    pub halo_air: Vec4,
}

impl FroxelFog {
    pub fn std430(&self, out: &mut Vec<u8>) {
        for m in [
            self.clip_from_world,
            self.world_from_clip,
            self.prev_clip_from_world,
        ] {
            out.extend_from_slice(bytemuck::cast_slice(&m.to_cols_array()));
        }
        for v in [
            self.camera,
            self.view_row,
            self.grid,
            self.alloc,
            self.media,
            self.medium,
            self.ambient,
            self.sun,
            self.sun_dir,
            self.phase,
            self.cloud,
            self.proj,
            self.screen,
            self.wind,
            self.room,
            self.room_media,
            self.room_ambient,
            self.room_sun,
            self.night,
            self.ground_x,
            self.ground_y,
            self.ground,
            self.halo_air,
        ] {
            out.extend_from_slice(bytemuck::cast_slice(&v.to_array()));
        }
    }
}

/// This frame's volumetrics, extracted for the render world: the mode, the grid, and the packed
/// block tail (`FroxelFog` then the cloud grid).
#[derive(Resource, Clone, ExtractResource)]
pub struct ModernVolume {
    pub mode: FogMode,
    pub grid: UVec3,
    pub block: Vec<u8>,
}

impl Default for ModernVolume {
    /// Mode off, and a clear sky over the cloud grid.
    fn default() -> Self {
        let mut block = Vec::with_capacity(BLOCK_TAIL_BYTES);
        FroxelFog::default().std430(&mut block);
        block.extend_from_slice(bytemuck::cast_slice(&[1.0f32; CLOUD_CELLS * CLOUD_CELLS]));
        Self {
            mode: FogMode::Off,
            grid: UVec3::ZERO,
            block,
        }
    }
}

/// The view the fog composite runs on: the world camera. **Not the water mirror** — it renders the
/// plain lighting and nothing of the modern frame (`sunshadow::mirror_view`): a sky pixel under its
/// oblique near plane does not read as sky to the composite, which fogged the mirror's whole sky to
/// the horizon haze, and the composite was a full-screen pass over a buffer the waves then blur.
#[derive(Component, Clone, Copy, ExtractComponent)]
pub struct ModernFogView;

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<ModernVolume>()
        .add_plugins((
            ExtractResourcePlugin::<ModernVolume>::default(),
            ExtractComponentPlugin::<ModernFogView>::default(),
        ))
        .add_systems(Startup, create_volume_image)
        .add_systems(Update, (mark_fog_views, arm_fog_depth))
        // `Last`, never `PostUpdate` (see `HOOKS.md`, "Scheduling rule").
        .add_systems(Last, (update_volume, resize_volume_image));
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        node::build(render_app);
    }
}

/// The volume a session with this setting needs allocated.
fn volume_alloc(model: LightingModel, quality: LightingQuality) -> UVec3 {
    if model == LightingModel::Modern && froxel_tier(quality).mode == FogMode::Froxel {
        VOLUME_ALLOC
    } else {
        UVec3::ONE
    }
}

fn volume_image(size: UVec3) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: size.z,
        },
        TextureDimension::D3,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    // The integrate pass writes it as storage; every lane samples it filtered.
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        label: Some("modern_fog_volume".into()),
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

/// Insert the constant-handle volume. A session launched modern (`$WOW_LIGHTING`, `?lighting=`) gets
/// the full allocation up front; everyone else a 1³ placeholder the materials can bind.
fn create_volume_image(settings: Res<LightingSettings>, mut images: ResMut<Assets<Image>>) {
    let size = volume_alloc(settings.model, settings.quality);
    let _ = images.insert(MODERN_FOG_VOLUME.id(), volume_image(size));
}

/// Grow the volume when a world entry latches a froxel tier it was not allocated for. Material bind
/// groups hold the old texture's view until they are prepared again, so every material that binds
/// the volume is touched — once, behind the loading cover.
fn resize_volume_image(
    active: Res<ActiveLighting>,
    mut images: ResMut<Assets<Image>>,
    models: Option<ResMut<Assets<WowModelMaterial>>>,
    liquids: Option<ResMut<Assets<LiquidMaterial>>>,
    cata: Option<ResMut<Assets<LiquidCataMaterial>>>,
) {
    if !active.is_changed() || !active.is_modern() {
        return;
    }
    let want = volume_alloc(active.model, active.quality);
    let fits = images.get(MODERN_FOG_VOLUME.id()).is_some_and(|image| {
        let s = image.texture_descriptor.size;
        s.width >= want.x && s.height >= want.y && s.depth_or_array_layers >= want.z
    });
    if fits {
        return;
    }
    let _ = images.insert(MODERN_FOG_VOLUME.id(), volume_image(want));
    let mut touched = 0;
    if let Some(mut m) = models {
        touched += m.iter_mut().count();
    }
    if let Some(mut m) = liquids {
        touched += m.iter_mut().count();
    }
    if let Some(mut m) = cata {
        touched += m.iter_mut().count();
    }
    info!("modern fog: volume {want} allocated, {touched} materials rebound");
}

#[allow(clippy::type_complexity)]
fn mark_fog_views(
    mut commands: Commands,
    cams: Query<Entity, (With<WorldCamera>, With<Camera3d>, Without<ModernFogView>)>,
) {
    for entity in &cams {
        commands.entity(entity).insert(ModernFogView);
    }
}

/// The composite reads the depth buffer: opt the fog views' depth into `TEXTURE_BINDING` (set once,
/// never cleared — the SSILVB module's reasoning).
fn arm_fog_depth(active: Res<ActiveLighting>, mut cams: Query<&mut Camera3d, With<ModernFogView>>) {
    if !active.is_modern() {
        return;
    }
    for mut cam in &mut cams {
        let usage = TextureUsages::from(cam.depth_texture_usages);
        if usage.contains(TextureUsages::TEXTURE_BINDING) {
            continue;
        }
        cam.depth_texture_usages = (usage | TextureUsages::TEXTURE_BINDING).into();
    }
}

#[derive(Default)]
struct VolumeHistory {
    prev_clip: Option<Mat4>,
    prev_camera: Vec3,
    prev_forward: Vec3,
    prev_grid: UVec3,
    frame: u32,
    reference_height: Option<f32>,
    /// The ground under the camera the night's banks lie on, eased like the reference.
    bank_ground: Option<f32>,
    log_timer: f32,
    /// The room whose air is still fading ([`fading_room`]).
    room: Option<(Vec3, f32)>,
}

/// The `i`-th element of the Halton sequence in `base`.
fn halton(mut i: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_volume(
    active: Res<ActiveLighting>,
    lighting: Res<WowLighting>,
    look: Res<super::look::ActiveLook>,
    cameras: Query<
        (&GlobalTransform, &Projection),
        (With<WorldCamera>, Without<LiquidReflectionCamera>),
    >,
    time: Res<Time>,
    underwater: Option<Res<crate::liquid::Underwater>>,
    crossfade: Option<Res<crate::lighting::WmoCrossfade>>,
    room_extent: Option<Res<crate::wmo_portal::CameraRoomExtent>>,
    clouds: Option<Res<crate::clouds::CloudCoverage>>,
    debug: Option<Res<crate::dev_state::DebugState>>,
    wdl: Option<Res<crate::wdl::WdlStreamer>>,
    view_distance: Option<Res<crate::view::ViewDistance>>,
    weather: Option<Res<crate::weather::WeatherState>>,
    mut out: ResMut<ModernVolume>,
    mut hist: Local<VolumeHistory>,
) {
    // The debug panel's fog toggle turns the volume off like it turns the classic fog off.
    if !active.is_modern() || debug.is_some_and(|d| d.lighting.disable_fog) {
        if out.mode != FogMode::Off {
            *out = ModernVolume::default();
        }
        hist.prev_clip = None;
        hist.room = None;
        return;
    }
    let Some((camera, projection)) = cameras.iter().next() else {
        return;
    };
    let Projection::Perspective(persp) = projection else {
        return;
    };
    let tier = froxel_tier(active.quality);
    let l = &*lighting;
    // fork: modern weather — heavy rain's humid air (`modern::weather::heavy_rain`).
    let heavy = weather.as_deref().map_or(0.0, super::weather::heavy_rain);
    let added = added_air(&look.0);
    let humid = heavy * added;
    // By day only: the night air already swells a lamp's glow.
    let halo_humid = 1.0 + super::weather::HEAVY_HALO_GAIN * humid * (1.0 - l.sidn_night);
    let view_from_world = camera.to_matrix().inverse();
    let clip_from_view = projection.get_clip_from_view();
    let clip_from_world = clip_from_view * view_from_world;
    let cam = camera.translation();
    let forward = camera.forward().as_vec3();
    let to_light = (-l.sun_dir).normalize_or(Vec3::Y);

    // **The open air and the room, solved apart.** The scene's fog is the air outside; the
    // building's MFOG is the air in the room the camera stands in, and `room_w` (the classic 4 s
    // crossfade) says how much of one the camera is breathing. They stay apart all the way to the
    // shader because a room's medium reaches only as far as the room does ([`CameraRoomExtent`]):
    // one blended medium is what wrapped the city seen from Orgrimmar's gate hall in 292 yd of the
    // building's teal.
    let shared =
        |fog_color: [f32; 3], fog_start: f32, fog_end: f32, interior: f32| media::MediaInputs {
            fog_color,
            fog_start,
            fog_end,
            zone_fog_end: l.zone_fog_end,
            // The same moonlight floor the surfaces take, so night haze is not black under a lit ground.
            ambient: super::night_floor(l.ambient, l.sidn_night),
            // The key's horizon fade (a set moon scatters nothing into the haze).
            diffuse: l.diffuse.map(|c| c * super::sun_arc::key_horizon()),
            storm: l.storm_bcc,
            underwater: underwater.as_deref().is_some_and(|u| u.0.any()),
            interior,
            haze: look.0.haze * HAZE_SCALE + super::weather::HEAVY_HAZE * humid,
            // fork: under the moon the look keeps `moon_glow` of its forward lobe.
            forward: look.0.forward * (1.0 + (look.0.moon_glow - 1.0) * super::sun_arc::moon_key()),
            key_scatter: look.0.key_scatter,
            fog_reach: look.0.fog_reach,
            fog_tint: look.0.fog_tint,
            fog_falloff: look.0.fog_falloff,
            fog_clear: look.0.fog_clear,
            room_reach: look.0.room_reach,
            added_air: added,
        };
    let media = media::solve(&shared(l.fog_color, l.fog_start, l.fog_end, 0.0));
    // fork: the lamp halos' air — the same zone solved in clear weather, so rain and storms neither
    // swell nor fade a lamp's glow (`froxelfog::halo_fog`), but for heavy rain, whose wet air the
    // lamps bloom into (`modern::weather::HEAVY_HALO_GAIN`).
    let halo_air = media::solve(&media::MediaInputs {
        storm: 0.0,
        haze: look.0.haze * HAZE_SCALE,
        zone_fog_end: l.clear_fog[2],
        ..shared(l.fog_color, l.clear_fog[0], l.clear_fog[1], 0.0)
    });
    let room_w = crossfade.map_or(0.0, |c| c.t());
    // The room's sphere (`None` = the camera is not in one): its medium is solved whole, and the
    // weight alone rides the crossfade.
    let room = fading_room(
        &mut hist.room,
        room_extent
            .and_then(|r| r.0)
            .filter(|(_, radius)| *radius > 0.0),
        room_w,
    );
    let room_media = (room_w > 1e-3).then(|| {
        media::solve(&shared(
            l.wmo_fog_color,
            l.wmo_fog_start,
            l.wmo_fog_end,
            1.0,
        ))
    });

    // The height falloff's reference is the regional ground — the coarse WDL heights around the
    // camera, never above the camera itself — eased over ~2 s, so the fog lies in the valleys and a
    // flying or climbing camera does not drag the layer up with it; a teleport snaps it.
    let dt = time.delta_secs();
    let ground = wdl
        .as_deref()
        .and_then(|w| {
            let regional = regional_ground(|p| w.height_under(p), cam)?;
            Some(held_to_local_ground(regional, w.height_under(cam)))
        })
        .map_or(cam.y, |g| g.min(cam.y));
    let reference_height = match hist.reference_height {
        Some(y) if (y - ground).abs() < 400.0 => y + (ground - y) * (dt / 2.0).min(1.0),
        _ => ground,
    };
    hist.reference_height = Some(reference_height);
    let here = wdl
        .as_deref()
        .and_then(|w| w.height_under(cam))
        .map_or(ground, |h| h.min(cam.y));
    let bank_ground = match hist.bank_ground {
        Some(y) if (y - here).abs() < 400.0 => y + (here - y) * (dt / 2.0).min(1.0),
        _ => here,
    };
    hist.bank_ground = Some(bank_ground);

    // Dev (`?modern_foglog=1`): the solved medium every two seconds.
    if crate::launch_env::var("WOW_MODERN_FOGLOG").as_deref() == Some("1") {
        hist.log_timer -= dt;
        if hist.log_timer <= 0.0 {
            hist.log_timer = 2.0;
            info!(
                "modern fog: night {:.2} moon_key {:.2} cam_y {:.0} bank_ground {:.0} density {:.5} clear {:.0} ref {:.0} key_in {:?} amb_in {:?} diffuse {:?} ambient {:?} key_horizon {:.2} to_light {:?} zone_end {:.0} fog {:?} room {:?} room_w {:.2} room_density {:.5} wmo_fog {:.0}..{:.0}",
                l.sidn_night, super::sun_arc::moon_key(), cam.y, bank_ground,
                media.density, media.near_clear, reference_height, media.key_in, media.ambient_in,
                l.diffuse, l.ambient, super::sun_arc::key_horizon(), to_light, l.zone_fog_end, l.fog_color,
                room, room_w, room_media.map_or(0.0, |m| m.density), l.wmo_fog_start, l.wmo_fog_end
            );
        }
    }

    let shaft_key = shaft_key(l.diffuse, &look.0);
    let night = l.sidn_night.clamp(0.0, 1.0);
    let veil = 1.0 + (MOON_VEIL - 1.0) * super::sun_arc::moon_key();
    let night_glow = Vec3::from(l.fog_color.map(super::srgb_to_linear))
        * (NIGHT_GLOW * night * veil * look.0.night_glow);
    let grid = tier.grid;
    let cut = hist.prev_clip.is_none()
        || hist.prev_grid != grid
        || hist.prev_camera.distance(cam) > 12.0
        || hist.prev_forward.dot(forward) < 0.9;
    let prev_clip = hist.prev_clip.unwrap_or(clip_from_world);
    hist.frame = hist.frame.wrapping_add(1);
    let jitter = halton(hist.frame % 16 + 1, 2);

    // The key light on screen (the Low tier's shaft origin).
    let light_clip = clip_from_world * (cam + to_light * 1000.0).extend(1.0);
    let (light_uv, on_screen) = if light_clip.w > 1e-3 {
        let ndc = light_clip.truncate().truncate() / light_clip.w;
        let outside = (ndc.abs().max_element() - 1.0).max(0.0);
        (
            Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5),
            (1.0 - outside * 2.0).clamp(0.0, 1.0) * smoothstep(0.0, 0.15, to_light.y),
        )
    } else {
        (Vec2::splat(0.5), 0.0)
    };

    let mut cloud = [1.0f32; CLOUD_CELLS * CLOUD_CELLS];
    let cloud_place = cloud_grid(
        clouds.as_deref(),
        cam,
        to_light,
        tier.cloud_shadow,
        &mut cloud,
    );
    let seconds = time.elapsed_secs_wrapped();
    // The noise tiles every 48 yd, so its scroll wraps there seamlessly.
    let wind = (Vec3::new(0.9, 0.04, 0.35) * seconds).map(|v| v.rem_euclid(48.0));

    let fog = FroxelFog {
        clip_from_world,
        world_from_clip: clip_from_world.inverse(),
        prev_clip_from_world: prev_clip,
        camera: cam.extend(seconds),
        view_row: view_from_world.row(2),
        grid: grid.as_vec3().extend(tier.range),
        alloc: VOLUME_ALLOC.as_vec3().extend(tier.mode.as_f32()),
        media: Vec4::new(
            media.density,
            media.inv_height,
            reference_height,
            media.near_clear,
        ),
        medium: Vec4::new(
            media.haze,
            media.noise,
            shaft_key.x,
            if cut { 0.0 } else { HISTORY_WEIGHT },
        ),
        // The zone look's colour dials (`look::ZoneLook::glow` / `haze_tint`).
        ambient: (Vec3::from(media.ambient_in) * Vec3::from(look.0.haze_tint)).extend(shaft_key.y),
        sun: (Vec3::from(media.key_in) * Vec3::from(look.0.key_hue(look.0.glow)))
            .extend(media.forward_share),
        sun_dir: to_light.extend(look.0.silhouette),
        phase: Vec4::new(
            media.g_backward,
            // The view distance: where the detailed world ends and the distant (WDL) band begins.
            view_distance.map_or(0.0, |v| v.farclip),
            jitter,
            hist.frame as f32,
        ),
        cloud: cloud_place,
        proj: Vec4::new(
            persp.near,
            1.0 / clip_from_view.x_axis.x,
            1.0 / clip_from_view.y_axis.y,
            tier.shafts,
        ),
        screen: Vec4::new(light_uv.x, light_uv.y, on_screen, tier.point_lights as f32),
        wind: wind.extend(shaft_key.z),
        // The room the camera stands in, and its own medium — `room.w` 0 (no room, or none solved)
        // switches the whole lane off in the shader.
        room: match (room, &room_media) {
            (Some((centre, radius)), Some(_)) => centre.extend(radius),
            _ => Vec4::ZERO,
        },
        // fork: the night air's extinction and glow (see `NIGHT_HAZE`, `NIGHT_GLOW`); the glow's rgb
        // rides the three spare `.w` lanes of the room rows.
        room_media: Vec4::new(
            room_media.map_or(0.0, |m| m.density),
            room_w,
            NIGHT_HAZE * night * added,
            night_glow.x,
        ),
        room_ambient: room_media
            .map(|m| Vec3::from(m.ambient_in) * Vec3::from(look.0.haze_tint))
            .unwrap_or_default()
            .extend(night_glow.y),
        room_sun: room_media
            .map(|m| Vec3::from(m.key_in) * Vec3::from(look.0.key_hue(look.0.glow)))
            .unwrap_or_default()
            .extend(night_glow.z),
        night: Vec4::new(
            if super::banks_enabled() {
                BANK_EXTINCTION * night_bank_share(&look.0, night, added)
            } else {
                0.0
            },
            bank_ground,
            bank_air_share(&look.0),
            // fork: the zone look's sea fog (`froxel_scatter.wgsl`'s `sea_fog`).
            if super::banks_enabled() {
                look.0.sea_fog
            } else {
                0.0
            },
        ),
        ground_x: Vec4::ZERO,
        ground_y: Vec4::ZERO,
        ground: Vec4::ZERO,
        halo_air: Vec4::new(
            halo_air.density,
            halo_air.near_clear,
            halo_air.haze * halo_humid,
            super::look::key_air_share(),
        ),
    };
    hist.prev_clip = Some(clip_from_world);
    hist.prev_camera = cam;
    hist.prev_forward = forward;
    hist.prev_grid = grid;

    out.mode = tier.mode;
    out.grid = grid;
    out.block.clear();
    fog.std430(&mut out.block);
    out.block.extend_from_slice(bytemuck::cast_slice(&cloud));
}

/// fork: **the room the camera just left keeps its air until the crossfade has spent it.**
/// [`CameraRoomExtent`](crate::wmo_portal::CameraRoomExtent) goes `None` at the doorway while the
/// classic 4 s crossfade still holds `room_w` of that room's fog, and dropping the sphere there
/// snapped the tunnel's teal out of the air in front of Orgrimmar's gate in one frame. The last
/// room is latched instead — `WmoCrossfade`'s staged triple latches on exit for the same reason —
/// so the fog of a hall fades inside the hall. A room the camera stands in always replaces it.
fn fading_room(
    latched: &mut Option<(Vec3, f32)>,
    now: Option<(Vec3, f32)>,
    room_w: f32,
) -> Option<(Vec3, f32)> {
    if now.is_some() {
        *latched = now;
    } else if room_w <= 1e-3 {
        *latched = None;
    }
    *latched
}

/// The share of the night air and rain's humid veil a zone takes on top of its own air (see
/// [`CLEAR_AIR`]): 1 in a clear zone, [`HAZY_SHARE`] in one that is hazy all day.
pub(crate) fn added_air(look: &super::look::ZoneLook) -> f32 {
    let own = look.haze * HAZE_SCALE + BANK_EXTINCTION * look.ground_fog + look.sea_fog;
    let hazy = ((own - CLEAR_AIR) / (HAZY_AIR - CLEAR_AIR)).clamp(0.0, 1.0);
    1.0 + (HAZY_SHARE - 1.0) * hazy
}

/// The ground banks' share of [`BANK_EXTINCTION`] at night fraction `night`: the day's banks, and
/// `added` of what the night lays over them, all of it where the look's banks are ground fog
/// (`ZoneLook::bank_moon` 1).
fn night_bank_share(look: &super::look::ZoneLook, night: f32, added: f32) -> f32 {
    let day = look.ground_fog.max(0.0);
    let added = added + (1.0 - added) * look.bank_moon.clamp(0.0, 1.0);
    day + added * (night * look.night_banks - day).max(0.0)
}

/// The key's share on the banks' isotropic half: the shaft haze's ([`super::look::key_air_share`]),
/// up to the whole moon where the look's banks are ground fog (`ZoneLook::bank_moon` 1).
fn bank_air_share(look: &super::look::ZoneLook) -> f32 {
    let haze = super::look::key_air_share();
    haze + (1.0 - haze) * look.bank_moon.clamp(0.0, 1.0)
}

/// The shaft haze's key radiance: the key light's own colour (decoded and horizon-faded) times the
/// zone's key scattering, glow and shaft gain — not the medium's key term, which is normalized to
/// the fog colour and would tint every beam with the zone's fog hue. Under the moon the isotropic
/// half takes `look::key_air_share` of it (`halo_air.w`).
fn shaft_key(diffuse: [f32; 3], look: &super::look::ZoneLook) -> Vec3 {
    let horizon = super::sun_arc::key_horizon();
    let scale = look.key_scatter.max(0.0) * look.shafts.max(0.0) * horizon;
    Vec3::from(diffuse.map(super::srgb_to_linear)) * Vec3::from(look.key_hue(look.glow)) * scale
}

/// How far the regional reference may sit below the ground under the camera (yd) — most of a
/// height falloff, so the air the player stands in keeps about `e^-0.85` (~0.43) of the zone's
/// density. Half a falloff (35) was tried first and read too milky in Darnassus.
const LOCAL_FLOOR_YD: f32 = 60.0;

/// The regional reference, never more than [`LOCAL_FLOOR_YD`] below the ground under the camera.
///
/// The regional mean is right in a valley and wrong on any high ground with low ground within
/// 600 yd: the sea around Teldrassil put Darnassus's reference at 543 with its streets at 1316,
/// eleven falloffs up, and the haze there (shafts included) was `e^-11` of the zone's — none at
/// all. Stormwind (the harbour) and Orgrimmar (the canyon) lost most of theirs the same way.
/// Held to the local ground the layer still does not follow a flying camera (the ground under it
/// does not rise), and a valley seen from a ridge still reads denser, just not the ridge itself.
fn held_to_local_ground(regional: f32, here: Option<f32>) -> f32 {
    here.map_or(regional, |h| regional.max(h - LOCAL_FLOOR_YD))
}

/// Spacing (yd) of the regional ground samples around the camera.
const GROUND_SAMPLE_YD: f32 = 300.0;

/// The regional ground height (Bevy y) around `cam`: the mean of the lower two thirds of a 5×5 grid of
/// WDL heights, so a ridge beside the camera does not lift the fog layer out of the valley it fills.
/// Open ocean (no WDL tile) counts as sea level. `None` when no sample is authored at all.
fn regional_ground(height: impl Fn(Vec3) -> Option<f32>, cam: Vec3) -> Option<f32> {
    let mut samples = Vec::with_capacity(25);
    let mut any = false;
    for j in -2..=2 {
        for i in -2..=2 {
            let p = cam + Vec3::new(i as f32, 0.0, j as f32) * GROUND_SAMPLE_YD;
            let h = height(p);
            any |= h.is_some();
            samples.push(h.unwrap_or(0.0));
        }
    }
    if !any {
        return None;
    }
    samples.sort_by(f32::total_cmp);
    let low = &samples[..samples.len() * 2 / 3];
    Some(low.iter().sum::<f32>() / low.len() as f32)
}

/// Fill the cloud-shadow grid: for each cell's ground point, the cloud cover where the key light's
/// ray crosses the cloud layer, read from the same coverage field the visible clouds and the glare
/// gate use — so a cloud's shadow is that cloud. Returns the grid's placement (`FroxelFog::cloud`).
fn cloud_grid(
    clouds: Option<&crate::clouds::CloudCoverage>,
    cam: Vec3,
    to_light: Vec3,
    strength: f32,
    out: &mut [f32; CLOUD_CELLS * CLOUD_CELLS],
) -> Vec4 {
    let half = CLOUD_CELL_YD * (CLOUD_CELLS / 2) as f32;
    let origin =
        (Vec2::new(cam.x, cam.z) / CLOUD_CELL_YD).floor() * CLOUD_CELL_YD - Vec2::splat(half);
    let day = smoothstep(0.02, 0.2, to_light.y);
    let Some(clouds) = clouds.filter(|_| strength > 0.0 && day > 0.0) else {
        out.fill(1.0);
        return Vec4::new(origin.x, origin.y, CLOUD_CELL_YD, 0.0);
    };
    let lift = to_light * (CLOUD_HEIGHT / to_light.y.max(0.2));
    for j in 0..CLOUD_CELLS {
        for i in 0..CLOUD_CELLS {
            let ground = Vec3::new(
                origin.x + (i as f32 + 0.5) * CLOUD_CELL_YD,
                cam.y,
                origin.y + (j as f32 + 0.5) * CLOUD_CELL_YD,
            );
            // `coverage` reads a camera-relative sky direction (the glare's 12-unit convention).
            let toward = (ground + lift - cam).normalize_or(Vec3::Y) * 12.0;
            let cover = smoothstep(0.2, 0.8, clouds.coverage(toward));
            out[j * CLOUD_CELLS + i] = 1.0 - strength * day * cover;
        }
    }
    Vec4::new(origin.x, origin.y, CLOUD_CELL_YD, CLOUD_CELLS as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clear zone takes the night air and rain whole, banks included; a zone hazy all day, only a
    /// touch of them. An ordinary zone lays no banks at night.
    #[test]
    fn a_hazy_zone_takes_only_a_touch_of_night_and_rain() {
        use crate::modern::look::ZoneLook;
        assert_eq!(added_air(&ZoneLook::NEUTRAL), 1.0, "Elwynn");
        for hazy in [ZoneLook::DARKSHORE, ZoneLook::EERIE, ZoneLook::TIRISFAL] {
            assert!((added_air(&hazy) - HAZY_SHARE).abs() < 1e-6);
        }
        let darkshore = ZoneLook::DARKSHORE;
        let (day, night) = (
            night_bank_share(&darkshore, 0.0, HAZY_SHARE),
            night_bank_share(&darkshore, 1.0, HAZY_SHARE),
        );
        assert!(night >= day && night - day <= HAZY_SHARE * darkshore.night_banks);
        // The dark zones' banks are ground fog: the whole night bank, whatever their haze.
        for eerie in [ZoneLook::EERIE, ZoneLook::TIRISFAL] {
            assert_eq!(night_bank_share(&eerie, 1.0, HAZY_SHARE), eerie.night_banks);
        }
        assert_eq!(night_bank_share(&ZoneLook::NEUTRAL, 1.0, 1.0), 0.0);
        let banked = ZoneLook {
            night_banks: ZoneLook::EERIE.night_banks,
            ..ZoneLook::NEUTRAL
        };
        assert_eq!(added_air(&banked), 1.0);
        assert_eq!(
            night_bank_share(&banked, 1.0, added_air(&banked)),
            banked.night_banks
        );
    }

    #[test]
    fn the_tail_matches_the_wgsl_layout() {
        let v = ModernVolume::default();
        assert_eq!(v.block.len(), BLOCK_TAIL_BYTES);
        assert_eq!(FOG_BYTES, 560);
        // `ModernBlock` = cascades (400) + fog (416) + grid (4096): a multiple of its 16-byte alignment.
        assert_eq!(
            (crate::modern::shadow::CASCADES_BYTES + BLOCK_TAIL_BYTES) % 16,
            0
        );
        // The mode lane the render world patches is `alloc.w`.
        let mut fog = FroxelFog::default();
        fog.alloc.w = 2.0;
        let mut bytes = Vec::new();
        fog.std430(&mut bytes);
        assert_eq!(
            bytemuck::pod_read_unaligned::<f32>(&bytes[MODE_OFFSET..MODE_OFFSET + 4]),
            2.0
        );
    }

    #[test]
    fn tiers_fit_the_allocation_and_low_runs_no_compute() {
        for q in LightingQuality::ALL {
            let t = froxel_tier(q);
            assert!(t.grid.cmple(VOLUME_ALLOC).all(), "{q:?}");
        }
        assert_eq!(froxel_tier(LightingQuality::Low).mode, FogMode::Analytic);
        assert_eq!(
            froxel_tier(LightingQuality::Medium).grid,
            UVec3::new(96, 54, 48)
        );
    }

    #[test]
    fn halton_base_two_is_the_van_der_corput_sequence() {
        assert_eq!(halton(1, 2), 0.5);
        assert_eq!(halton(2, 2), 0.25);
        assert_eq!(halton(3, 2), 0.75);
    }

    #[test]
    fn a_clear_or_night_sky_casts_no_cloud_shadow() {
        let mut grid = [0.0f32; CLOUD_CELLS * CLOUD_CELLS];
        let place = cloud_grid(None, Vec3::new(10.0, 50.0, -20.0), Vec3::Y, 0.5, &mut grid);
        assert!(grid.iter().all(|v| *v == 1.0));
        assert_eq!(place.w, 0.0);
        let night = Vec3::new(0.3, -0.5, 0.1).normalize();
        let _ = cloud_grid(None, Vec3::ZERO, night, 0.5, &mut grid);
        assert!(grid.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn a_room_left_behind_keeps_its_air_while_it_fades() {
        let (a, b) = (
            (Vec3::new(1.0, 2.0, 3.0), 31.0),
            (Vec3::new(-5.0, 0.0, 9.0), 12.0),
        );
        let mut latched = None;
        assert_eq!(fading_room(&mut latched, Some(a), 0.2), Some(a), "entering");
        assert_eq!(
            fading_room(&mut latched, None, 0.75),
            Some(a),
            "leaving: the air stays in the room"
        );
        assert_eq!(
            fading_room(&mut latched, Some(b), 0.75),
            Some(b),
            "a new room replaces it"
        );
        assert_eq!(fading_room(&mut latched, None, 0.0), None, "spent");
        assert_eq!(
            fading_room(&mut latched, None, 0.5),
            None,
            "nothing to hold"
        );
    }

    #[test]
    fn high_ground_beside_the_sea_keeps_its_haze() {
        // Darnassus: the sea around Teldrassil drags the regional mean to 543, the streets are at 1316.
        let r = held_to_local_ground(543.0, Some(1316.0));
        assert_eq!(r, 1316.0 - LOCAL_FLOOR_YD);
        // A valley keeps its regional reference, and no local height changes nothing.
        assert_eq!(held_to_local_ground(40.0, Some(60.0)), 40.0);
        assert_eq!(held_to_local_ground(543.0, None), 543.0);
    }
}
