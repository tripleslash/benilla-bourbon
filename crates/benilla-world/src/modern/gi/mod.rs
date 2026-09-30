//! **Runtime probe GI** (fork-owned; `FORK.md` P5).
//!
//! Nothing is baked: every frame a slice of a cascaded probe field is re-traced against a voxel
//! model of the world that the GPU builds itself, and the result replaces the ambient term in the
//! modern surface lanes and modulates the froxels' ambient in-scatter.
//!
//! The three pieces:
//!
//! - **The scene** ([`crate::static_gx::render`]'s `modern_gi` child): the retained static world
//!   rasterized into a toroidal voxel clipmap (albedo + normal), and the terrain as a top-down
//!   heightfield. Both are incremental — a level is re-voxelized only where the camera scrolled it
//!   or where content streamed in or out.
//! - **The probes** ([`node`]): a compute trace of Fibonacci rays per probe (voxel DDA, coarsening
//!   by clipmap level, with the heightfield beyond it), lit by the key light under the modern
//!   cascades plus the previous frame's probe field, and a blend pass with hysteresis.
//! - **The sampling** (`benilla_modern::probegi`, and one small helper in each hooked shader): the
//!   eight probes around a fragment, weighted by validity and by facing, finest cascade first.
//!
//! **Why the plan's per-model proxy bake is not what landed.** P5 specified a worker bake per model
//! into a proxy pool stamped into the clipmap. The data that bake needs does not exist on the CPU in
//! this port: BLP images are `RENDER_WORLD`-only (the note at the top of `static_gx::render`), and
//! the retained lane's geometry lives in GPU buffers by the time anything could voxelize it. The GPU
//! has all of it already — the same cell, WMO and prop draws the shadow pass replays — so the
//! voxelization is a raster pass over those, three axes per box, and the pool becomes the clipmap
//! itself. The plan's `gi/heightfield.rs` survives as written, and is what terrain hits use.
//!
//! **Where the probes live.** In the shared modern storage block, behind the cascades, the froxel
//! parameters and the cloud grid (`sunshadow::ModernBlock`), at a 256-aligned offset so the blend
//! pass can bind that region alone as read-write. No lane needed a new binding for P5 either.

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::{Texture, TextureView};
use bevy::render::RenderApp;

use super::{ActiveLighting, LightingQuality};
use crate::lighting::WowLighting;
use crate::view::WorldCamera;

pub mod node;

/// Probe cascades — capped by `probegi.wgsl`'s fixed arrays. The fifth is P6's interior sub-grid,
/// which the exterior ladder makes room for while the camera stands in a WMO.
pub const MAX_CASCADES: usize = 5;
/// Clipmap levels — `probegi.wgsl`'s `GiClip` arrays. Unrelated to the cascade count since P6.
pub const MAX_LEVELS: usize = 4;
/// Size of [`GiParams::std430`] — `probegi.wgsl`'s `GiParams`. Chosen so that the light grid behind
/// it starts at a 256-byte offset inside the block, which is what lets the binning pass bind it (and
/// the probe array behind THAT stays 256-aligned for the blend pass).
pub const GI_PARAMS_BYTES: usize = 20 * 16;
/// Size of one `probegi::GiProbe`.
pub const PROBE_BYTES: usize = 96;

/// P6's interior probe spacing (yd) — the plan's 1.5 yd sub-grid.
pub const INTERIOR_SPACING: f32 = 1.5;

/// Probes allocated in the block, whatever the session's tier: the largest tier this platform is
/// expected to run, interior sub-grid included — High in the browser, Ultra natively. (Forcing Ultra
/// on the web overflows this and the field simply stops being written, with a warning, exactly as it
/// did before P6.) Held at one size so a quality change never re-creates the buffer every material
/// binds.
pub const PROBE_ALLOC: usize = if cfg!(target_arch = "wasm32") {
    4 * 32 * 16 * 32
} else {
    5 * 32 * 16 * 32
};

/// The probe field's own bytes in the modern storage block — its last, runtime-sized member.
pub const BLOCK_PROBE_BYTES: usize = PROBE_ALLOC * PROBE_BYTES;

/// What a quality tier traces — the plan's tier table, with the voxel clipmap in place of its proxy
/// pool and probe atlas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiTier {
    pub on: bool,
    /// Probes per axis in every cascade.
    pub dims: UVec3,
    pub cascades: usize,
    /// Probe spacing (yd) per cascade.
    pub spacing: [f32; MAX_CASCADES],
    pub rays: u32,
    /// Frames between a probe's traces. The trace's cost is `rays / period` of the field a frame,
    /// and `probegi::listed_cap` holds a moving camera's frames to the same share. fork: every tier
    /// runs 9x its P6 period (Medium 12, High 8, Ultra 6, Cinematic 4), a ninth of the trace for a
    /// field that settles over 9x as long (`SETTLE_SECS`). Keep it coprime with 7 —
    /// `gi_classify.wgsl`'s schedule is `(idx * 7 + frame) % period`.
    pub period: u32,
    pub levels: usize,
    /// Voxel size (yd) per clipmap level.
    pub voxel: [f32; MAX_LEVELS],
    /// Voxels per side in every level.
    pub voxels: u32,
    /// How far a probe ray reaches (yd), and how many DDA steps it may spend.
    pub reach: f32,
    pub steps: u32,
    /// P6: cascade 0 is the 1.5 yd interior sub-grid (the camera stands in a WMO).
    pub interior: bool,
    /// P6: Chebyshev probe visibility (High and above).
    pub chebyshev: bool,
}

impl GiTier {
    pub const OFF: Self = Self {
        on: false,
        dims: UVec3::ONE,
        cascades: 1,
        spacing: [1.0; MAX_CASCADES],
        rays: 1,
        period: 1,
        levels: 1,
        voxel: [1.0; MAX_LEVELS],
        voxels: 1,
        reach: 1.0,
        steps: 1,
        interior: false,
        chebyshev: false,
    };

    /// P6: the same tier with the interior sub-grid pushed in front of the exterior ladder. Every
    /// consumer downstream — the parameters, the trace, the sampling — then reads one uniform
    /// cascade list and never learns that one of them is special.
    pub fn with_interior(mut self) -> Self {
        if self.interior || !self.on || self.cascades + 1 > MAX_CASCADES {
            return self;
        }
        let mut spacing = [0.0; MAX_CASCADES];
        spacing[0] = INTERIOR_SPACING;
        spacing[1..=self.cascades].copy_from_slice(&self.spacing[..self.cascades]);
        self.spacing = spacing;
        self.cascades += 1;
        self.interior = true;
        self
    }

    pub fn probes(&self) -> usize {
        (self.dims.x * self.dims.y * self.dims.z) as usize * self.cascades
    }

    /// The world extent (yd) a cascade covers.
    pub fn cascade_extent(&self, c: usize) -> Vec3 {
        self.dims.as_vec3() * self.spacing[c]
    }
}

pub fn gi_tier(quality: LightingQuality) -> GiTier {
    match quality {
        // Low keeps P3's sky-occlusion map as its only ambient term: no voxels, no probes.
        LightingQuality::Low => GiTier::OFF,
        LightingQuality::Medium => GiTier {
            on: true,
            dims: UVec3::new(24, 12, 24),
            cascades: 3,
            spacing: [4.0, 12.0, 36.0, 0.0, 0.0],
            rays: 24,
            period: 108,
            levels: 3,
            voxel: [1.5, 4.0, 12.0, 0.0],
            voxels: 64,
            reach: 220.0,
            steps: 96,
            interior: false,
            chebyshev: false,
        },
        LightingQuality::High => GiTier {
            on: true,
            dims: UVec3::new(32, 16, 32),
            cascades: 3,
            spacing: [3.0, 9.0, 27.0, 0.0, 0.0],
            rays: 32,
            period: 72,
            levels: 4,
            voxel: [1.0, 3.0, 8.0, 16.0],
            voxels: 96,
            reach: 300.0,
            steps: 128,
            interior: false,
            chebyshev: true,
        },
        LightingQuality::Ultra => GiTier {
            on: true,
            dims: UVec3::new(32, 16, 32),
            cascades: 4,
            spacing: [2.0, 6.0, 18.0, 54.0, 0.0],
            rays: 48,
            period: 54,
            levels: 4,
            voxel: [0.75, 2.25, 6.0, 12.0],
            voxels: 96,
            reach: 380.0,
            steps: 160,
            interior: false,
            chebyshev: true,
        },
        // fork: P8 — Ultra's grids (`MAX_CASCADES` is 5 and Ultra's 4 + the interior sub-grid
        // already fills it, and the probe pool is allocated at one size for every tier), traced
        // harder: 64 rays, on the shortest period. The bounce matters more here than on any other tier
        // because Cinematic is the tier where a room's only light may be the hearth it bounces.
        LightingQuality::Cinematic => GiTier {
            on: true,
            dims: UVec3::new(32, 16, 32),
            cascades: 4,
            spacing: [2.0, 6.0, 18.0, 54.0, 0.0],
            rays: 64,
            period: 36,
            levels: 4,
            voxel: [0.75, 2.25, 6.0, 12.0],
            voxels: 96,
            reach: 380.0,
            steps: 192,
            interior: false,
            chebyshev: true,
        },
    }
}

/// The ground's albedo in the lanes' own value space (authored gamma, like a texel): what the
/// terrain heightfield bounces, standing in for the splat set a voxel cannot hold.
pub const GROUND_ALBEDO: Vec3 = Vec3::new(0.40, 0.37, 0.31);
/// The sky gradient's span, zenith over horizon. Its cosine-weighted mean is 1 by construction
/// (`probegi::sky_radiance`), so an open-field probe reads the classic ambient exactly.
pub const SKY_GRADIENT: f32 = 0.3;
/// A probe most of whose rays leave surfaces by their back sits inside geometry.
const BACKFACE_LIMIT: f32 = 0.35;
/// **How long a steady probe takes to settle** (seconds, one time constant) — the ease `gi_blend`
/// applies on a quiet frame, expressed as a TIME rather than a per-update weight (a large change
/// still eases faster, and a reseed replaces outright).
///
/// A probe's update is a few dozen rays under a rotation that turns every frame, so its estimate is
/// noisy by construction and the ease is what averages that noise away. The weight has to be read
/// against the tier's update PERIOD to mean anything: at a flat 0.9, Cinematic's probes (period 4)
/// settled in 40 frames and Medium's (period 12) in 120, and what the eye got either way was the
/// residue — a few percent of the noise, live, on every surface. Parked in Orgrimmar's gate hall the
/// distant city measured 0.81% of pixels moving frame to frame with `benilla-visual flicker`, in
/// runs the hotspot verdict called ONE SURFACE, re-lit; at this time constant it is 0.16%, against
/// a GI-off floor of 0.07% (the residue there is the torch flames, which are supposed to move).
///
/// It is long on purpose, and it is only the QUIET path: a probe whose estimate really moves still
/// eases in a few updates (`gi_blend`'s change term), a scrolled or cold one is replaced outright,
/// and a teleport cuts the whole field. What settles slowly is the last few percent of the noise,
/// which is exactly what was glimmering.
///
/// fork: 9x the 2.2 s the flicker numbers above were measured at, because every tier's period went
/// 9x with it (twice 3x, the second for a GI pass a third as costly). What averages the noise away
/// is the number of updates per time constant, so the two move together: the same residue for a
/// ninth of the trace.
const SETTLE_SECS: f32 = 19.8;

/// The steady-frame history weight for a tier updating every `period` frames at `dt`: the ease that
/// settles in [`SETTLE_SECS`] whatever the tier and the frame rate. Clamped so a stalled frame (a
/// streaming hitch, a breakpoint) cannot throw the whole field away.
fn history_weight(dt: f32, period: u32) -> f32 {
    let update = (dt * period.max(1) as f32).clamp(1e-3, SETTLE_SECS);
    (-update / SETTLE_SECS).exp().clamp(0.0, 0.99)
}
/// Sampling bias along the receiver's normal, in probe spacings.
const NORMAL_BIAS: f32 = 0.35;

/// Is probe GI on in this session (`$WOW_MODERN_GI`, browser `?modern_gi=0`)?
pub fn gi_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::launch_env::var("WOW_MODERN_GI").as_deref() != Some("0"))
}

/// The GI's gain over the classic ambient (`$WOW_MODERN_GI_GAIN`, browser `?modern_gi_gain=`), the
/// look-dev lever for how far the probes are allowed to move the frame from P3's balance.
pub fn gi_gain() -> f32 {
    static CELL: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        crate::launch_env::var("WOW_MODERN_GI_GAIN")
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(1.0)
            .clamp(0.0, 4.0)
    })
}

/// This frame's GI inputs, extracted for the render world, which owns every grid: the probe bases
/// follow the camera deterministically, and the clipmap's follow it with hysteresis, so both are
/// derived where the data they address lives.
#[derive(Resource, Clone, Copy, ExtractResource)]
pub struct ModernGi {
    pub on: bool,
    pub tier: GiTier,
    pub camera: Vec3,
    pub forward: Vec3,
    /// The view's half-diagonal field of view (radians): the cone `gi_classify.wgsl` holds a probe
    /// to its full update rate inside. Pi (everything in view) without a perspective camera.
    pub view_half: f32,
    pub frame: u32,
    /// A teleport or a model switch: every probe is traced and replaced this frame.
    pub cut: bool,
    pub gain: f32,
    /// The mean radiance an unoccluded probe reads — the froxels' reference for "open sky here".
    pub open_mean: f32,
    /// This frame's ray rotation.
    pub rot: Mat3,
    /// This frame's steady-probe history weight ([`history_weight`]).
    pub history: f32,
    /// P6: the claiming building's `MOHD.ambColor`, decoded, scaled into the shared light's units —
    /// what an interior probe's escaped rays see instead of the sky.
    pub interior_ambient: Vec3,
}

impl Default for ModernGi {
    fn default() -> Self {
        Self {
            on: false,
            tier: GiTier::OFF,
            camera: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            view_half: std::f32::consts::PI,
            frame: 0,
            cut: true,
            gain: 1.0,
            history: 0.9,
            open_mean: 0.0,
            rot: Mat3::IDENTITY,
            interior_ambient: Vec3::ZERO,
        }
    }
}

/// The CPU mirror of `probegi.wgsl`'s `GiParams` (field order and meaning documented there).
#[derive(Clone, Copy, Debug, Default)]
pub struct GiParams {
    pub params: Vec4,
    pub grid: [Vec4; MAX_CASCADES],
    pub dims: Vec4,
    pub sky: Vec4,
    pub ground: Vec4,
    pub misc: Vec4,
    pub interior: Vec4,
}

impl GiParams {
    pub fn std430(&self, out: &mut Vec<u8>) {
        let mut row = |v: Vec4| out.extend_from_slice(bytemuck::cast_slice(&v.to_array()));
        row(self.params);
        for g in &self.grid {
            row(*g);
        }
        row(self.dims);
        row(self.sky);
        row(self.ground);
        row(self.misc);
        row(self.interior);
        // `spare`: P7's grading, and the padding that aligns the light grid behind this block —
        // 9 rows since the froxel block's room fog, night banks, ground rows and the lamp halos'
        // air pushed it 144 bytes off the grid.
        for _ in 0..9 {
            row(Vec4::ZERO);
        }
    }
}

/// The CPU mirror of `probegi.wgsl`'s `GiClip` — the voxel clipmap and the terrain heightfield, as
/// the compute passes address them.
#[derive(Clone, Copy, Debug)]
pub struct GiClip {
    pub level: [Vec4; MAX_LEVELS],
    pub info: Vec4,
    pub hf_row_x: Vec4,
    pub hf_row_y: Vec4,
    pub hf_row_z: Vec4,
    pub hf: Vec4,
    /// fork: P8b — `x` = the highest ground the heightfield holds (world Y): the top of every
    /// resident terrain cell's box. `probegi::march` skips the map above it, which is exact because
    /// no texel can stand higher. [`NO_TERRAIN`] (the default) where there is no terrain at all.
    pub bounds: Vec4,
}

/// [`GiClip::bounds`]' `x` with no terrain: below everything, so every ray point is above it and the
/// (empty) map is never read.
pub const NO_TERRAIN: f32 = -1.0e9;

impl Default for GiClip {
    fn default() -> Self {
        Self {
            level: [Vec4::ZERO; MAX_LEVELS],
            info: Vec4::ZERO,
            hf_row_x: Vec4::ZERO,
            hf_row_y: Vec4::ZERO,
            hf_row_z: Vec4::ZERO,
            hf: Vec4::ZERO,
            bounds: Vec4::new(NO_TERRAIN, 0.0, 0.0, 0.0),
        }
    }
}

impl GiClip {
    pub fn std430(&self, out: &mut Vec<u8>) {
        let mut row = |v: Vec4| out.extend_from_slice(bytemuck::cast_slice(&v.to_array()));
        for l in &self.level {
            row(*l);
        }
        row(self.info);
        row(self.hf_row_x);
        row(self.hf_row_y);
        row(self.hf_row_z);
        row(self.hf);
        row(self.bounds);
    }
}

/// The voxel clipmap and the terrain heightfield, as the voxel pass leaves them for the trace.
/// Written by `static_gx::render::modern_gi` in `PrepareResources`, read by [`node`] in
/// `PrepareBindGroups`.
#[derive(Resource, Default)]
pub struct ModernGiScene {
    pub albedo: Option<(Texture, TextureView)>,
    pub normal: Option<(Texture, TextureView)>,
    pub height: Option<(Texture, TextureView)>,
    /// The clipmap's live addressing (bases as this frame's passes wrote them).
    pub clip: GiClip,
    /// Is there a voxel model to trace against this frame?
    pub ready: bool,
}

#[derive(Default)]
struct GiHistory {
    frame: u32,
    camera: Option<Vec3>,
    tier: Option<GiTier>,
    /// The debounced interior verdict: the building the sub-grid is live for, and how long the raw
    /// claim has disagreed with it (s).
    room: Option<Entity>,
    disagree: f32,
    /// fork: frames the current cut is still held for — a tier's `period`, so every probe's
    /// scheduled turn inside it replaces the probe rather than blending (the trace lists at most
    /// `probegi::listed_cap` probes a frame, so a cut can no longer replace the field in one).
    cut_frames: u32,
}

/// How long the camera's interior claim must hold before the 1.5 yd sub-grid appears (s)…
const INTERIOR_ENTER_SECS: f32 = 0.25;
/// …and how long it must stay gone before the sub-grid is dropped again. A doorway, or a claim that
/// flickers at a portal, must not toggle the grid every few frames.
const INTERIOR_LEAVE_SECS: f32 = 1.0;

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<ModernGi>()
        .add_plugins(ExtractResourcePlugin::<ModernGi>::default())
        // `Last`, never `PostUpdate` (see `HOOKS.md`, "Scheduling rule").
        .add_systems(Last, update_gi);
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.init_resource::<ModernGiScene>();
        node::build(render_app);
    }
}

#[allow(clippy::too_many_arguments)]
fn update_gi(
    active: Res<ActiveLighting>,
    lighting: Res<WowLighting>,
    cameras: Query<(&GlobalTransform, Option<&bevy::camera::Projection>), With<WorldCamera>>,
    // fork: modern lighting (P6) — the camera's room, and the building it belongs to.
    claim: Option<Res<crate::wmo_portal::CameraInteriorClaim>>,
    instances: Query<&crate::wmo_portal::WmoPortalInstance>,
    wmos: Res<Assets<benilla_assets::WmoModel>>,
    time: Res<Time>,
    mut out: ResMut<ModernGi>,
    mut hist: Local<GiHistory>,
) {
    let base_tier = gi_tier(active.quality);
    // P6: the 1.5 yd interior sub-grid exists only while the camera stands in a WMO — a grid that
    // fine covers 36 yd, which is a room and not a zone, so it is the room's presence that pays for
    // it. The exterior ladder behind it is untouched, so stepping outside is a cascade dropping off
    // the front rather than a whole field being re-fitted.
    let raw = claim.as_ref().and_then(|c| c.0).map(|c| c.room.instance);
    if raw.is_some() != hist.room.is_some() {
        hist.disagree += time.delta_secs();
        let limit = if raw.is_some() {
            INTERIOR_ENTER_SECS
        } else {
            INTERIOR_LEAVE_SECS
        };
        if hist.disagree >= limit {
            hist.room = raw;
            hist.disagree = 0.0;
        }
    } else {
        // Same verdict; walking from one building straight into another just follows it.
        hist.room = raw;
        hist.disagree = 0.0;
    }
    let inside = hist.room;
    let tier = if inside.is_some() {
        base_tier.with_interior()
    } else {
        base_tier
    };
    // The building's own `MOHD.ambColor`, in the shared light's units: the classic interior lanes
    // never read it (their MOCV bake IS the room's light), and P6 retires that bake, so this is what
    // a ray leaving the room through an unvoxelized wall is allowed to see.
    let interior_ambient = inside
        .and_then(|e| instances.get(e).ok())
        .and_then(|inst| wmos.get(&inst.handle))
        .map(|model| {
            let [r, g, b] = model.ambient;
            let d = crate::modern::srgb_to_linear;
            Vec3::new(d(r), d(g), d(b))
        })
        .unwrap_or(Vec3::ZERO);
    if !active.is_modern() || !tier.on || !gi_enabled() {
        if out.on {
            *out = ModernGi::default();
        }
        hist.camera = None;
        return;
    }
    let Some((camera, projection)) = cameras.iter().next() else {
        return;
    };
    // The cone a probe keeps its full update rate in: the frustum's half-diagonal.
    let view_half = match projection {
        Some(bevy::camera::Projection::Perspective(p)) => {
            let v = (p.fov * 0.5).tan();
            (v * v * (1.0 + p.aspect_ratio * p.aspect_ratio))
                .sqrt()
                .atan()
        }
        _ => std::f32::consts::PI,
    };
    let cam = camera.translation();
    // The interior sub-grid coming or going is not a cut: it is stored behind the exterior ladder
    // (`probegi::storage_cascade`), so every exterior probe keeps its slot and its history.
    if hist.camera.is_none_or(|p| p.distance(cam) > 60.0) || hist.tier != Some(base_tier) {
        hist.cut_frames = tier.period.max(1);
    }
    let cut = hist.cut_frames > 0;
    hist.cut_frames = hist.cut_frames.saturating_sub(1);
    hist.camera = Some(cam);
    hist.tier = Some(base_tier);
    hist.frame = hist.frame.wrapping_add(1);

    // The ray set turns a little every frame, so the estimate's own bias averages out over the
    // probe's history rather than baking a fixed 24-direction pattern into the field.
    let a = hist.frame as f32 * 0.618_034;
    let rot = Mat3::from_euler(
        EulerRot::YXZ,
        a * 6.2831855,
        (a * 3.883222).sin() * 0.8,
        a * 2.3999632,
    );

    // What a probe under nothing reads: half the sky's own mean, half what the ground bounces back.
    let ambient = Vec3::from(lighting.ambient);
    let diffuse = Vec3::from(lighting.diffuse) * Vec3::from(super::look::key_tint());
    let (key, fill) = super::look::key_fill();
    let ambient = ambient * fill;
    let sun_up = (-lighting.sun_dir.normalize_or(Vec3::Y)).y.max(0.0);
    let ground = GROUND_ALBEDO * (ambient + diffuse * key * (sun_up * 0.7));
    let lum = |c: Vec3| c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    let open_mean = 0.5 * lum(ambient) + 0.5 * lum(ground);

    *out = ModernGi {
        on: true,
        tier,
        camera: cam,
        forward: camera.forward().as_vec3(),
        view_half,
        frame: hist.frame,
        cut,
        gain: gi_gain(),
        open_mean,
        rot,
        history: history_weight(time.delta_secs(), tier.period),
        interior_ambient,
    };
}

/// The probe grid's base coord for one cascade — the corner of the toroidal window, in probe units.
/// Deterministic in the camera: a probe that scrolls out is one that scrolls back in unchanged.
pub fn probe_base(tier: &GiTier, cascade: usize, camera: Vec3) -> IVec3 {
    let spacing = tier.spacing[cascade];
    let centre = (camera / spacing).floor().as_ivec3();
    centre - (tier.dims.as_ivec3() / 2)
}

/// The GI block's parameters for this frame, given the render world's own grids.
pub fn params(gi: &ModernGi, camera: Vec3, sky: Vec3, debug: f32) -> GiParams {
    let tier = &gi.tier;
    let mut grid = [Vec4::ZERO; MAX_CASCADES];
    for (c, g) in grid.iter_mut().enumerate().take(tier.cascades) {
        let base = probe_base(tier, c, camera).as_vec3();
        *g = base.extend(tier.spacing[c]);
    }
    GiParams {
        params: Vec4::new(
            if gi.on { 1.0 } else { 0.0 },
            (tier.dims.x * tier.dims.y * tier.dims.z) as f32,
            gi.gain,
            NORMAL_BIAS,
        ),
        grid,
        dims: tier.dims.as_vec3().extend(tier.cascades as f32),
        sky: sky.extend(gi.open_mean),
        ground: GROUND_ALBEDO.extend(SKY_GRADIENT),
        misc: Vec4::new(tier.rays as f32, tier.period as f32, gi.frame as f32, debug),
        // fork: modern lighting (P6) — the interior sub-grid's ambient and the two sampling bits.
        interior: gi.interior_ambient.extend(f32::from(
            u8::from(tier.interior) | (u8::from(tier.chebyshev) << 1),
        )),
    }
}

/// The trace and blend passes' uniform, in one place because both bind it.
pub fn trace_bytes(gi: &ModernGi, params: &GiParams, clip: &GiClip) -> Vec<u8> {
    let mut out = Vec::with_capacity(GI_PARAMS_BYTES + 10 * 16 + 6 * 16);
    params.std430(&mut out);
    clip.std430(&mut out);
    let mut row = |v: Vec4| out.extend_from_slice(bytemuck::cast_slice(&v.to_array()));
    row(gi.camera.extend(if gi.cut { 1.0 } else { 0.0 }));
    row(gi.forward.extend(gi.history));
    row(gi.rot.row(0).extend(gi.view_half));
    row(gi.rot.row(1).extend(0.0));
    row(gi.rot.row(2).extend(0.0));
    // The sun bias is a voxel of the finest level; the ray's own start offset keeps a probe sitting
    // on a surface from hitting it at zero distance. The interior sub-grid's probes sit closer
    // together than a voxel, so the offset never grows with them.
    let bias = gi.tier.voxel[0];
    row(Vec4::new(BACKFACE_LIMIT, bias, bias * 0.5, 1.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_blocks_match_the_wgsl_layout() {
        let mut bytes = Vec::new();
        GiParams::default().std430(&mut bytes);
        assert_eq!(bytes.len(), GI_PARAMS_BYTES);
        // The light grid has to start 256-aligned (the binning pass binds that region alone), and
        // so does the probe array behind it (the blend pass binds that one).
        let lights = crate::modern::shadow::CASCADES_BYTES
            + crate::modern::volumetrics::BLOCK_TAIL_BYTES
            + GI_PARAMS_BYTES;
        assert_eq!(lights % 256, 0, "light offset {lights}");
        assert_eq!(lights, 5376);
        let probes = lights + crate::modern::lights::BLOCK_LIGHT_BYTES;
        assert_eq!(probes % 256, 0, "probe offset {probes}");
        let mut clip = Vec::new();
        GiClip::default().std430(&mut clip);
        assert_eq!(clip.len(), 10 * 16);
        let gi = ModernGi::default();
        assert_eq!(
            trace_bytes(&gi, &GiParams::default(), &GiClip::default()).len(),
            GI_PARAMS_BYTES + 10 * 16 + 6 * 16
        );
    }

    #[test]
    fn every_tier_fits_the_allocation_and_its_arrays() {
        for q in [
            LightingQuality::Low,
            LightingQuality::Medium,
            LightingQuality::High,
            LightingQuality::Ultra,
            LightingQuality::Cinematic,
        ] {
            let t = gi_tier(q);
            assert!(t.cascades <= MAX_CASCADES, "{q:?}");
            assert!(t.levels <= MAX_LEVELS, "{q:?}");
            assert!(t.probes() <= PROBE_ALLOC, "{q:?}: {}", t.probes());
            // P6: the interior sub-grid has to fit too — it appears the moment the camera steps
            // through a door, and a field that did not fit would simply stop being written.
            let inside = t.with_interior();
            assert!(inside.cascades <= MAX_CASCADES, "{q:?}");
            assert!(inside.probes() <= PROBE_ALLOC, "{q:?}: {}", inside.probes());
            if t.on {
                assert!(
                    inside.interior && inside.spacing[0] == INTERIOR_SPACING,
                    "{q:?}"
                );
                // The exterior ladder is intact behind it.
                assert_eq!(
                    inside.spacing[1..=t.cascades],
                    t.spacing[..t.cascades],
                    "{q:?}"
                );
            }
            if t.on {
                // Every cascade reaches past the one inside it, and the coarsest clipmap level
                // covers the coarsest cascade's own probes.
                for c in 1..t.cascades {
                    assert!(t.spacing[c] > t.spacing[c - 1], "{q:?}");
                }
                let coarsest = t.voxel[t.levels - 1] * t.voxels as f32;
                assert!(
                    coarsest >= t.cascade_extent(t.cascades - 1).x * 0.3,
                    "{q:?}"
                );
                // `gi_classify.wgsl` schedules `(idx * 7 + frame) % period`: a period sharing a
                // factor with 7 would leave some probes never due.
                assert!(!t.period.is_multiple_of(7), "{q:?}: period {}", t.period);
            }
        }
        assert!(!gi_tier(LightingQuality::Low).on);
    }

    /// The CPU twin of `gi_trace.wgsl`'s ray set and SH projection — the numbers the calibration
    /// below depends on live in the shader, and this is what pins them.
    fn open_field_probe(rays: u32, gradient: f32) -> (Vec3, [Vec3; 3]) {
        let mut dc = Vec3::ZERO;
        let mut sh = [Vec3::ZERO; 3];
        for i in 0..rays {
            let k = i as f32 + 0.5;
            let z = 1.0 - 2.0 * k / rays as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = k * 2.399963;
            let dir = Vec3::new(r * phi.cos(), z, r * phi.sin());
            // `probegi::sky_radiance` above the horizon; nothing bounces back from below.
            let radiance = if dir.y > 0.0 {
                Vec3::splat(1.0 - gradient * (2.0 / 3.0) + gradient * dir.y)
            } else {
                Vec3::ZERO
            };
            dc += radiance;
            sh[0] += radiance * dir.x;
            sh[1] += radiance * dir.y;
            sh[2] += radiance * dir.z;
        }
        let inv = 1.0 / rays as f32;
        (dc * inv, sh.map(|v| v * 2.0 * inv))
    }

    #[test]
    fn a_probe_under_open_sky_reads_back_the_classic_ambient() {
        // The whole calibration rests on this: with the sky at the ambient row's own value, a probe
        // that sees nothing but sky hands an up-facing surface that row back, so P1's balance
        // survives the switch from a flat ambient to a traced one.
        for rays in [24u32, 32, 48] {
            let (dc, sh) = open_field_probe(rays, SKY_GRADIENT);
            let up = dc + sh[1];
            assert!(
                (up.x - 1.0).abs() < 0.05,
                "{rays} rays: an open probe reads {up} of the ambient row"
            );
            // A wall sees half the sky and (here) nothing bouncing back: dimmer, which is the point.
            let side = dc + sh[0];
            assert!(
                side.x > 0.3 && side.x < 0.75,
                "{rays} rays: a wall reads {side}"
            );
        }
        // A uniform field round-trips exactly, whatever the ray count: no direction is favoured.
        let (dc, sh) = open_field_probe(32, 0.0);
        let uniform = dc + sh[1] * 0.0;
        assert!((uniform.x - 0.5).abs() < 0.02, "{uniform}");
    }

    #[test]
    fn a_probe_grid_follows_the_camera_by_whole_probes() {
        let tier = gi_tier(LightingQuality::Medium);
        let base = probe_base(&tier, 0, Vec3::new(100.0, 20.0, -40.0));
        // The camera sits in the middle cell of its cascade.
        assert_eq!(base, IVec3::new(25 - 12, 5 - 6, -10 - 12));
        // Half a spacing of movement never moves the window by more than one probe.
        let moved = probe_base(&tier, 0, Vec3::new(102.0, 20.0, -40.0));
        assert!((moved - base).abs().max_element() <= 1);
    }
}
