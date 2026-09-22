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
pub(crate) use node::FroxelLabel;

/// Size of [`FroxelFog::std430`] — `froxelfog.wgsl`'s `FroxelFog`.
pub const FOG_BYTES: usize = 3 * 64 + 18 * 16;
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
        LightingQuality::High => (FogMode::Froxel, UVec3::new(160, 90, 64), 450.0, 16, 0.0, 0.0),
        // fork: P8b — the point-light count is only the fallback walk's now (`?modern_lights=0`): with
        // the light grid live a froxel reads the lights of its own cell (`froxel_scatter.wgsl`).
        // Ultra lights from glowing things as Cinematic does, so it takes Cinematic's count.
        LightingQuality::Ultra => (FogMode::Froxel, UVec3::new(192, 108, 96), 600.0, 64, 0.0, 0.0),
        // fork: P8 — the same grid as Ultra (the volume is allocated once at the platform's largest,
        // so a finer one would need a bigger allocation on every tier).
        LightingQuality::Cinematic => (FogMode::Froxel, UVec3::new(192, 108, 96), 600.0, 64, 0.0, 0.0),
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
}

impl FroxelFog {
    pub fn std430(&self, out: &mut Vec<u8>) {
        for m in [self.clip_from_world, self.world_from_clip, self.prev_clip_from_world] {
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

/// The views the fog composite runs on: the world camera, and the water mirror (`mirror`), which
/// runs no compute of its own.
#[derive(Component, Clone, Copy, ExtractComponent)]
pub struct ModernFogView {
    pub mirror: bool,
}

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
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
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
    cams: Query<
        (Entity, Has<LiquidReflectionCamera>),
        (
            Or<(With<WorldCamera>, With<LiquidReflectionCamera>)>,
            With<Camera3d>,
            Without<ModernFogView>,
        ),
    >,
) {
    for (entity, mirror) in &cams {
        commands.entity(entity).insert(ModernFogView { mirror });
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
    log_timer: f32,
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

#[allow(clippy::too_many_arguments)]
fn update_volume(
    active: Res<ActiveLighting>,
    lighting: Res<WowLighting>,
    look: Res<super::look::ActiveLook>,
    cameras: Query<(&GlobalTransform, &Projection), (With<WorldCamera>, Without<LiquidReflectionCamera>)>,
    time: Res<Time>,
    underwater: Option<Res<crate::liquid::Underwater>>,
    crossfade: Option<Res<crate::lighting::WmoCrossfade>>,
    room_extent: Option<Res<crate::wmo_portal::CameraRoomExtent>>,
    clouds: Option<Res<crate::clouds::CloudCoverage>>,
    debug: Option<Res<crate::dev_state::DebugState>>,
    wdl: Option<Res<crate::wdl::WdlStreamer>>,
    view_distance: Option<Res<crate::view::ViewDistance>>,
    mut out: ResMut<ModernVolume>,
    mut hist: Local<VolumeHistory>,
) {
    // The debug panel's fog toggle turns the volume off like it turns the classic fog off.
    if !active.is_modern() || debug.is_some_and(|d| d.lighting.disable_fog) {
        if out.mode != FogMode::Off {
            *out = ModernVolume::default();
        }
        hist.prev_clip = None;
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
    let shared = |fog_color: [f32; 3], fog_start: f32, fog_end: f32, interior: f32| media::MediaInputs {
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
        haze: look.0.haze,
        forward: look.0.forward,
        key_scatter: look.0.key_scatter,
        fog_reach: look.0.fog_reach,
        fog_tint: look.0.fog_tint,
        fog_falloff: look.0.fog_falloff,
        fog_clear: look.0.fog_clear,
    };
    let media = media::solve(&shared(l.fog_color, l.fog_start, l.fog_end, 0.0));
    let room_w = crossfade.map_or(0.0, |c| c.t());
    // The room's sphere (`None` = the camera is not in one): its medium is solved whole, and the
    // weight alone rides the crossfade.
    let room = room_extent.and_then(|r| r.0).filter(|(_, radius)| *radius > 0.0);
    let room_media = (room_w > 1e-3)
        .then(|| media::solve(&shared(l.wmo_fog_color, l.wmo_fog_start, l.wmo_fog_end, 1.0)));

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

    // Dev (`?modern_foglog=1`): the solved medium every two seconds.
    if crate::launch_env::var("WOW_MODERN_FOGLOG").as_deref() == Some("1") {
        hist.log_timer -= dt;
        if hist.log_timer <= 0.0 {
            hist.log_timer = 2.0;
            info!(
                "modern fog: density {:.5} clear {:.0} ref {:.0} key_in {:?} amb_in {:?} diffuse {:?} ambient {:?} key_horizon {:.2} to_light {:?} zone_end {:.0} fog {:?}",
                media.density, media.near_clear, reference_height, media.key_in, media.ambient_in,
                l.diffuse, l.ambient, super::sun_arc::key_horizon(), to_light, l.zone_fog_end, l.fog_color
            );
        }
    }

    let shaft_key = shaft_key(l.diffuse, &look.0);
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
    let cloud_place = cloud_grid(clouds.as_deref(), cam, to_light, tier.cloud_shadow, &mut cloud);
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
        media: Vec4::new(media.density, media.inv_height, reference_height, media.near_clear),
        medium: Vec4::new(
            media.haze,
            media.noise,
            shaft_key.x,
            if cut { 0.0 } else { HISTORY_WEIGHT },
        ),
        // The zone look's colour dials (`look::ZoneLook::glow` / `haze_tint`).
        ambient: (Vec3::from(media.ambient_in) * Vec3::from(look.0.haze_tint)).extend(shaft_key.y),
        sun: (Vec3::from(media.key_in) * Vec3::from(look.0.glow)).extend(media.forward_share),
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
        room_media: Vec4::new(
            room_media.map_or(0.0, |m| m.density),
            room_w,
            0.0,
            0.0,
        ),
        room_ambient: room_media
            .map(|m| Vec3::from(m.ambient_in) * Vec3::from(look.0.haze_tint))
            .unwrap_or_default()
            .extend(0.0),
        room_sun: room_media
            .map(|m| Vec3::from(m.key_in) * Vec3::from(look.0.glow))
            .unwrap_or_default()
            .extend(0.0),
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

/// The shaft haze's key radiance: the key light's own colour (decoded, horizon-faded) times the zone's
/// key scattering, glow and shaft gain — not the medium's key term, which is normalized to the fog
/// colour and would tint every beam with the zone's fog hue.
fn shaft_key(diffuse: [f32; 3], look: &super::look::ZoneLook) -> Vec3 {
    let horizon = super::sun_arc::key_horizon();
    let scale = look.key_scatter.max(0.0) * look.shafts.max(0.0) * horizon;
    Vec3::from(diffuse.map(super::srgb_to_linear)) * Vec3::from(look.glow) * scale
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
    let origin = (Vec2::new(cam.x, cam.z) / CLOUD_CELL_YD).floor() * CLOUD_CELL_YD - Vec2::splat(half);
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

    #[test]
    fn the_tail_matches_the_wgsl_layout() {
        let v = ModernVolume::default();
        assert_eq!(v.block.len(), BLOCK_TAIL_BYTES);
        assert_eq!(FOG_BYTES, 480);
        // `ModernBlock` = cascades (400) + fog (416) + grid (4096): a multiple of its 16-byte alignment.
        assert_eq!((crate::modern::shadow::CASCADES_BYTES + BLOCK_TAIL_BYTES) % 16, 0);
        // The mode lane the render world patches is `alloc.w`.
        let mut fog = FroxelFog::default();
        fog.alloc.w = 2.0;
        let mut bytes = Vec::new();
        fog.std430(&mut bytes);
        assert_eq!(bytemuck::pod_read_unaligned::<f32>(&bytes[MODE_OFFSET..MODE_OFFSET + 4]), 2.0);
    }

    #[test]
    fn tiers_fit_the_allocation_and_low_runs_no_compute() {
        for q in LightingQuality::ALL {
            let t = froxel_tier(q);
            assert!(t.grid.cmple(VOLUME_ALLOC).all(), "{q:?}");
        }
        assert_eq!(froxel_tier(LightingQuality::Low).mode, FogMode::Analytic);
        assert_eq!(froxel_tier(LightingQuality::Medium).grid, UVec3::new(96, 54, 48));
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
    fn high_ground_beside_the_sea_keeps_its_haze() {
        // Darnassus: the sea around Teldrassil drags the regional mean to 543, the streets are at 1316.
        let r = held_to_local_ground(543.0, Some(1316.0));
        assert_eq!(r, 1316.0 - LOCAL_FLOOR_YD);
        // A valley keeps its regional reference, and no local height changes nothing.
        assert_eq!(held_to_local_ground(40.0, Some(60.0)), 40.0);
        assert_eq!(held_to_local_ground(543.0, None), 543.0);
    }
}
