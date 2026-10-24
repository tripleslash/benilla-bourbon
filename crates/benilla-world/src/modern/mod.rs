//! **Modern lighting** — the fork's parallel lighting model (`FORK.md`).
//!
//! Everything under this module is fork-owned and runs *beside* upstream's classic lanes, never in
//! place of them: `lightingModel = 0` (classic) leaves every frame byte-identical to a
//! build without this module. The hook points it needs inside upstream-owned files are listed in
//! `HOOKS.md` next to this file, so an upstream merge can be checked against them.
//!
//! P0 scope: the settings, their per-world latch, and the shared WGSL libraries
//! (`benilla_modern::{color, frame}`). Nothing here draws yet.

use std::sync::atomic::{AtomicBool, Ordering};

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};

use crate::view::WorldCamera;

mod cloud_bake;
pub mod gi;
pub mod lights;
pub mod look;
mod meter;
pub mod shadow;
mod sky;
pub mod sun_arc;
pub mod volumetrics;
pub mod weather;

/// The moonlight floor a modern night keeps (gamma, the ambient row's units), ramped in by the SIDN
/// night fraction. The authored night ambient plus a key light that fades out with the moon's own
/// elevation leaves a moonless Darkshore midnight at almost nothing; the eye still reads a night
/// exterior as blue-grey, not as black.
pub const NIGHT_AMBIENT: [f32; 3] = [0.10, 0.13, 0.20];

/// The ambient row a modern frame commits: the authored row, never below [`NIGHT_AMBIENT`] scaled by
/// the night fraction (1 overnight, 0 by day).
pub fn night_floor(ambient: [f32; 3], night: f32) -> [f32; 3] {
    let n = night.clamp(0.0, 1.0);
    [0, 1, 2].map(|c| ambient[c].max(NIGHT_AMBIENT[c] * n))
}

/// P1b's fixed exposure (the FFXGlow combine's `wave.w`); auto-exposure lands in P7.
pub const MODERN_EXPOSURE: f32 = 1.0;

/// Which lighting model the world renders with. CVar `lightingModel`: `0` classic, `1` modern.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightingModel {
    /// Upstream's 1.12.1 reproduction: gamma lane, baked shadows, linear fog.
    Classic,
    /// Linear HDR, all-caster shadows, probe GI, froxel volumetrics. The fork's shipped model.
    #[default]
    Modern,
}

impl LightingModel {
    /// The CVar's numeric value space. Anything at or above one half reads as modern.
    pub fn from_cvar(v: f32) -> Self {
        if v >= 0.5 {
            Self::Modern
        } else {
            Self::Classic
        }
    }

    pub fn as_cvar(self) -> u8 {
        match self {
            Self::Classic => 0,
            Self::Modern => 1,
        }
    }

    /// `$WOW_LIGHTING` accepts the names as well as the numbers.
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "0" | "classic" => Some(Self::Classic),
            "1" | "modern" => Some(Self::Modern),
            _ => None,
        }
    }
}

/// The modern model's quality tier (the plan's tier table). CVar `lightingQuality`: `0`–`4`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LightingQuality {
    Low = 0,
    Medium = 1,
    High = 2,
    /// fork: P8b — lights from things that only glowed, as Cinematic does (every lamp that ships no
    /// light, every lit window, every glowing weapon), on the two-level light grid that lets them
    /// light a street at any distance — and no local-light shadows at all: that is the whole of the
    /// difference to Cinematic, and it is what keeps the extra lights cheap.
    Ultra = 3,
    /// fork: modern lighting (P8) — Ultra's sun, fog, GI and lights, plus a dozen cube shadows for
    /// the strongest local lights. Since P8b those are the cheap kind: one hardware compare each,
    /// faint (a shade floor) rather than soft, static casters only.
    Cinematic = 4,
}

impl LightingQuality {
    /// The shipped tier, on every platform: Cinematic. A browser that cannot carry it steps the
    /// pixel budget down through `web/gpu.js`'s recovery ladder, or the player picks a lower tier.
    pub const DEFAULT: Self = Self::Cinematic;

    /// Rounded and clamped into the tier range; a non-number reads as the default.
    pub fn from_cvar(v: f32) -> Self {
        if !v.is_finite() {
            return Self::DEFAULT;
        }
        match v.round().clamp(0.0, 4.0) as u8 {
            0 => Self::Low,
            1 => Self::Medium,
            2 => Self::High,
            3 => Self::Ultra,
            _ => Self::Cinematic,
        }
    }

    /// Every tier, in order — the one list the tier-table tests sweep, so a new row can never be
    /// added to the enum and forgotten by a table.
    pub const ALL: [Self; 5] = [
        Self::Low,
        Self::Medium,
        Self::High,
        Self::Ultra,
        Self::Cinematic,
    ];

    /// Does this tier light from authored *emissive* sources — a weapon's enchant glow, a brazier
    /// that ships with no M2 light, a lit window — rather than only from the lights the reference
    /// commits? One predicate, read by the table packer, the shadow-slot ranking and the light grid
    /// alike, so they can never disagree about what a frame contains (`modern::lights::emissive`).
    /// Ultra and up since P8b; Cinematic alone before it.
    pub fn lights_emissive_sources(self) -> bool {
        self >= Self::Ultra
    }

    pub fn as_cvar(self) -> u8 {
        self as u8
    }
}

/// What the player asked for. Written by `benilla-app`'s CVar host. In the world, both the MODEL
/// and the QUALITY tier take effect on the frame they are written (see [`ActiveLighting`]).
#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub struct LightingSettings {
    pub model: LightingModel,
    pub quality: LightingQuality,
}

impl Default for LightingSettings {
    /// `$WOW_LIGHTING` / `$WOW_LIGHTING_QUALITY` override the shipped defaults, session-only — the
    /// A/B levers, and the only way to drive this from `benilla-worldview`, which has no CVar host.
    fn default() -> Self {
        let model = crate::launch_env::var("WOW_LIGHTING")
            .and_then(|v| LightingModel::parse(&v))
            .unwrap_or_default();
        let quality = crate::launch_env::var("WOW_LIGHTING_QUALITY")
            .and_then(|v| v.trim().parse::<f32>().ok())
            .map_or(LightingQuality::DEFAULT, LightingQuality::from_cvar);
        Self { model, quality }
    }
}

/// What this world session actually renders with.
///
/// fork: the MODEL is **live** in the world, like the tier: a flip re-keys every model and terrain
/// material ([`sync_model_pipelines`]), refits the shadow array and fog volume (both key off this
/// resource changing), re-scatters the loaded clutter (`terrain_stream::rescatter_clutter`) and
/// re-uploads the cloud texture in the other format. The first frames after a flip hitch while the
/// other model's pipelines compile. Out of the world — login, character select — it is Classic.
///
/// The TIER is **live**: it follows [`LightingSettings::quality`] on the frame the player moves the
/// dropdown, because nothing about a tier change needs the loading cover. Every consumer already
/// reads the tier per frame (`shadow::cascade_splits`, `gi::gi_tier`, `lights::light_tier`,
/// `volumetrics::froxel_tier`), the two allocations a tier sizes already refit on a change to this
/// resource (`shadow::fit_shadow_map`, `volumetrics::resize_volume_image`, and the render world
/// drops its static caches when the shadow array's texture id moves), and the Cinematic gate is
/// read where a light is READ rather than where it was spawned (`global_light`,
/// `SynthesizedLight`), so a lamp that streamed in on Ultra lights up on Cinematic without
/// re-streaming. The tier costs a one-frame hitch when it crosses a boundary that re-sizes the
/// shadow array, and nothing at all when it does not.
#[derive(Resource, Clone, Copy, PartialEq, Debug, ExtractResource)]
pub struct ActiveLighting {
    pub model: LightingModel,
    pub quality: LightingQuality,
}

impl Default for ActiveLighting {
    fn default() -> Self {
        Self {
            model: LightingModel::Classic,
            quality: LightingQuality::DEFAULT,
        }
    }
}

impl ActiveLighting {
    pub fn is_modern(self) -> bool {
        self.model == LightingModel::Modern
    }
}

/// The latched model as a process-wide bit, for the Rust-side hooks inside upstream-owned systems
/// (sky colours, celestial and cloud materials, the FFXGlow combine) whose signatures stay as
/// upstream wrote them. Written only by [`resolve_active_lighting`].
static MODERN_FRAME: AtomicBool = AtomicBool::new(false);

/// fork: P8 — the same device for the same reason, one tier further in. The two spawners that would
/// have to ask whether this session lights emissive sources — `terrain_stream::spawn`'s placement
/// walk and `benilla-app`'s equipment attach — are upstream-owned systems already nested against
/// Bevy's 16-parameter limit, and threading a resource into either is a merge hazard for a
/// question whose answer is one bit. It moves with the live tier, so a lamp that streamed in on
/// Ultra is lit by the Cinematic the player just picked — which is why the readers ask it per
/// frame rather than at spawn. Written only by [`resolve_active_lighting`].
static EMISSIVE_LIGHTS: AtomicBool = AtomicBool::new(false);

/// Is the world frame linear (a modern session)?
pub fn frame_is_modern() -> bool {
    MODERN_FRAME.load(Ordering::Relaxed)
}

/// Does this session light from authored *emissive* sources — a glowing weapon, a brazier that
/// ships no M2 light (`modern::lights::emissive`)? True only in a modern session on a tier that
/// does ([`LightingQuality::lights_emissive_sources`]: Ultra and up), and only while the bisect
/// lever allows it (`$WOW_MODERN_EMISSIVE=0`, browser `?modern_emissive=0`).
pub fn emissive_lights_on() -> bool {
    EMISSIVE_LIGHTS.load(Ordering::Relaxed) && emissive_enabled()
}

/// The emissive-source bisect lever.
fn emissive_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    switch_on(&CELL, "WOW_MODERN_EMISSIVE")
}

/// The ground-fog banks' bisect lever (`$WOW_MODERN_BANKS=0`, browser `?modern_banks=0`).
pub(crate) fn banks_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    switch_on(&CELL, "WOW_MODERN_BANKS")
}

/// Bisect switches (`$WOW_MODERN_SUNARC=0`, `$WOW_MODERN_CPUHOOKS=0`; browser `?modern_sunarc=0`,
/// `?modern_cpuhooks=0`): turn one modern main-world behaviour off without a rebuild. Read once.
fn switch_on(cell: &'static std::sync::OnceLock<bool>, key: &str) -> bool {
    *cell.get_or_init(|| crate::launch_env::var(key).as_deref() != Some("0"))
}

/// The sun arc's direction overrides (`sun_arc`).
pub fn sun_arc_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    switch_on(&CELL, "WOW_MODERN_SUNARC")
}

/// The shadow debug view (`$WOW_MODERN_SHADOWDEBUG`, browser `?modern_shadowdebug=`): `1` = the modern
/// receivers output raw sun visibility as grey, `2` = sky visibility, `3` = the probe field's
/// irradiance (P5), `4` = how much of it the fragment's probe cell answered, `5` = what the local
/// light grid put on the fragment, cube shadows included (P6), `6` = **cell occupancy** (P8): green
/// where the cell kept every light that reached it, yellow where it is exactly full, red where more
/// lights reached it than `PER_CELL` can name — the probe for "why is that lamp dark?", which is a
/// budget question and not a falloff one. `0` (default) = off.
pub fn shadow_debug_mode() -> u8 {
    static CELL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        crate::launch_env::var("WOW_MODERN_SHADOWDEBUG")
            .and_then(|v| v.trim().parse::<u8>().ok())
            .unwrap_or(0)
    })
}

/// Dev A/B (`$WOW_MODERN_MIRRORTIER=1`, browser `?modern_mirrortier=1`): every view shades as the water
/// mirror does (`sunshadow::mirror_view`) — the plain ambient and N·L, none of the modern lanes — so
/// what the mirror costs and shows compares on the main frame.
pub fn mirror_tier_everywhere() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::launch_env::var("WOW_MODERN_MIRRORTIER").as_deref() == Some("1"))
}

/// Should a world-space compute pass (the probe trace, the light binning) run on this view?
/// `world_camera` is whether the view is the world camera's (`volumetrics::ModernFogView`). They
/// ran once per `Core3d` view: twice a frame with the water mirror up, and once more per booth.
/// Dev A/B (`$WOW_MODERN_PERVIEW=1`, read once): every view again.
pub(crate) fn world_view_only(world_camera: bool) -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    world_camera
        || *CELL
            .get_or_init(|| crate::launch_env::var("WOW_MODERN_PERVIEW").as_deref() == Some("1"))
}

/// Dev A/B (`$WOW_MODERN_VERTEXHINTS=0`, read once): the light buffer's vertex hints stay unwritten
/// (`shadow::VERTEX_HINT_OFFSET`), so every `wow_model.wgsl` and `terrain.wgsl` vertex walks the
/// light table whether or not the light grid will answer its fragments.
pub(crate) fn vertex_hints_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    switch_on(&CELL, "WOW_MODERN_VERTEXHINTS")
}

/// The CPU-side modern hooks: sky stop decode, the cloud texture's sRGB view, the disc `span.z` flag.
pub fn cpu_hooks_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    switch_on(&CELL, "WOW_MODERN_CPUHOOKS")
}

/// The exact sRGB transfer, CPU side — the twin of `benilla_modern::color::srgb_to_linear_channel`.
pub fn srgb_to_linear(c: f32) -> f32 {
    let x = c.max(0.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// Hermite smoothstep, as WGSL defines it.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// An authored gamma colour as the current frame wants it: untouched in a classic frame, decoded
/// (rgb only; `w` rides through) in a modern one.
pub fn decode_if_modern(v: Vec4) -> Vec4 {
    if frame_is_modern() && cpu_hooks_enabled() {
        Vec4::new(
            srgb_to_linear(v.x),
            srgb_to_linear(v.y),
            srgb_to_linear(v.z),
            v.w,
        )
    } else {
        v
    }
}

/// Registers the settings, the latch, and the shared WGSL libraries.
pub struct ModernLightingPlugin;

impl Plugin for ModernLightingPlugin {
    fn build(&self, app: &mut App) {
        crate::shaders::modern_libraries(app);
        app.init_resource::<LightingSettings>()
            .init_resource::<ActiveLighting>()
            .add_plugins(ExtractResourcePlugin::<ActiveLighting>::default())
            .add_systems(
                PreUpdate,
                (resolve_active_lighting, sync_model_pipelines).chain(),
            )
            .add_plugins((
                sun_arc::plugin,
                shadow::plugin,
                look::plugin,
                meter::MeterPlugin,
                sky::plugin,
                cloud_bake::plugin,
                volumetrics::plugin,
                gi::plugin,
                lights::plugin,
                weather::plugin,
            ));
    }
}

/// [`ActiveLighting`] for this frame: the model and tier as the player has them set right now, the
/// model forced Classic out of the world.
fn resolve_active_lighting(
    settings: Res<LightingSettings>,
    mut active: ResMut<ActiveLighting>,
    cameras: Query<&Camera, With<WorldCamera>>,
) {
    // The client spawns its world camera at startup and only switches it on for the world (and its
    // loading cover), so "in the world" is an ACTIVE world camera. Out of the world — login,
    // character select and creation — everything renders classic.
    let in_world = cameras.iter().any(|c| c.is_active);
    // Both ride through every frame, not just the world edge — see [`ActiveLighting`]. Built and
    // compared as a whole so the resource is only written when something actually moved: the
    // readers that refit an allocation key off `is_changed()`, and a deref-mut every frame would
    // re-create the fog volume forever.
    let next = ActiveLighting {
        model: if in_world {
            settings.model
        } else {
            LightingModel::Classic
        },
        quality: settings.quality,
    };
    if *active == next {
        return;
    }
    info!("lighting active: {:?} / {:?}", next.model, next.quality);
    *active = next;
    MODERN_FRAME.store(active.is_modern(), Ordering::Relaxed);
    EMISSIVE_LIGHTS.store(
        active.is_modern() && active.quality.lights_emissive_sources(),
        Ordering::Relaxed,
    );
}

/// Model pipelines compile the modern code only in a modern session
/// (`benilla_assets::materials::MODERN_MODEL_PIPELINES`). On a latch edge every model material is
/// touched, so it re-prepares under the new pipeline key; the runtime flag in the shader stays, so
/// a material not yet re-prepared still renders the right model.
///
/// **Terrain is touched here too, and for terrain it is load-bearing rather than tidy.** The model
/// lane keeps its runtime flag inside `WOW_MODERN`, so a stale model material still picks the right
/// path; the terrain lane's def now *decides* the path (it is what lets a modern terrain pipeline
/// stop sampling the retired MCSH map at all — `materials::TerrainKey`), so a terrain material left
/// on the old key would render the other lighting model rather than merely carry dead code. A
/// material created after the latch reads the atomic itself and is born correct; this sweep is for
/// the ones that already existed.
fn sync_model_pipelines(
    active: Res<ActiveLighting>,
    mut materials: ResMut<Assets<benilla_assets::materials::WowModelMaterial>>,
    mut terrain: ResMut<Assets<benilla_assets::materials::TerrainMaterial>>,
    mut last: Local<Option<bool>>,
) {
    let modern = active.is_modern();
    if *last == Some(modern) {
        return;
    }
    let first = last.is_none();
    *last = Some(modern);
    benilla_assets::materials::MODERN_MODEL_PIPELINES.store(modern, Ordering::Relaxed);
    if first && !modern {
        return;
    }
    // `into_inner`: a bare `AssetMut` emits no `Modified`, so nothing would re-specialize.
    let ids: Vec<_> = materials.ids().collect();
    for id in ids {
        if let Some(material) = materials.get_mut(id) {
            material.into_inner();
        }
    }
    let terrain_ids: Vec<_> = terrain.ids().collect();
    for id in terrain_ids {
        if let Some(material) = terrain.get_mut(id) {
            material.into_inner();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two latch tests drive the process-wide `MODERN_FRAME` / `EMISSIVE_LIGHTS` atomics, so
    /// they take turns rather than racing each other inside one test binary.
    static LATCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// How many frames a consumer saw [`ActiveLighting`] change — what `fit_shadow_map` and
    /// `resize_volume_image` key their refits off.
    #[derive(Resource, Default)]
    struct Refits(u32);

    fn count_refits(active: Res<ActiveLighting>, mut refits: ResMut<Refits>) {
        if active.is_changed() {
            refits.0 += 1;
        }
    }

    /// The world, with one world camera whose `is_active` the test drives.
    fn world_app(settings: LightingSettings) -> App {
        let mut app = App::new();
        app.insert_resource(settings)
            .init_resource::<ActiveLighting>()
            .init_resource::<Refits>()
            .add_systems(PreUpdate, (resolve_active_lighting, count_refits).chain());
        app.world_mut().spawn((
            WorldCamera,
            Camera {
                is_active: false,
                ..default()
            },
        ));
        app
    }

    fn enter_world(app: &mut App, enter: bool) {
        let mut cams = app.world_mut().query::<&mut Camera>();
        for mut c in cams.iter_mut(app.world_mut()) {
            c.is_active = enter;
        }
    }

    fn active(app: &App) -> ActiveLighting {
        *app.world().resource::<ActiveLighting>()
    }

    fn refits(app: &App) -> u32 {
        app.world().resource::<Refits>().0
    }

    /// Neither the TIER nor the MODEL waits for the loading cover. A player who moves either
    /// Lighting dropdown standing in Northshire sees it on the next frame, without reloading.
    #[test]
    fn the_tier_and_the_model_apply_live_in_the_world() {
        let _guard = LATCH.lock().unwrap_or_else(|e| e.into_inner());
        let mut app = world_app(LightingSettings {
            model: LightingModel::Modern,
            quality: LightingQuality::High,
        });

        // Out of the world — login and character select — everything renders classic.
        app.update();
        assert_eq!(active(&app).model, LightingModel::Classic);

        enter_world(&mut app, true);
        app.update();
        assert_eq!(active(&app).model, LightingModel::Modern);
        assert_eq!(active(&app).quality, LightingQuality::High);

        // The dropdown moves with the world already up: the tier follows on the next frame.
        app.world_mut().resource_mut::<LightingSettings>().quality = LightingQuality::Cinematic;
        app.update();
        assert_eq!(active(&app).quality, LightingQuality::Cinematic);
        assert!(
            emissive_lights_on(),
            "the Cinematic gate follows the live tier"
        );

        // And back down, the frame after.
        app.world_mut().resource_mut::<LightingSettings>().quality = LightingQuality::Low;
        app.update();
        assert_eq!(active(&app).quality, LightingQuality::Low);
        assert!(!emissive_lights_on());

        // The model follows on the next frame too, both ways.
        app.world_mut().resource_mut::<LightingSettings>().model = LightingModel::Classic;
        app.update();
        assert_eq!(active(&app).model, LightingModel::Classic);
        assert!(!frame_is_modern());
        app.world_mut().resource_mut::<LightingSettings>().model = LightingModel::Modern;
        app.update();
        assert_eq!(active(&app).model, LightingModel::Modern);
        assert!(frame_is_modern());

        // Leaving the world renders classic whatever is selected, and re-entering restores it.
        enter_world(&mut app, false);
        app.update();
        assert_eq!(active(&app).model, LightingModel::Classic);
        assert!(!frame_is_modern());
        enter_world(&mut app, true);
        app.update();
        assert_eq!(active(&app).model, LightingModel::Modern);
    }

    /// The readers that refit an allocation — the shadow array's depth, the froxel volume — key off
    /// `ActiveLighting::is_changed()`, and each refit re-prepares every material that binds the
    /// thing. A latch that wrote the resource every frame would re-create both of them forever, so
    /// a frame where neither the tier nor the model moved must leave the resource alone.
    #[test]
    fn a_still_frame_leaves_the_resource_untouched() {
        let _guard = LATCH.lock().unwrap_or_else(|e| e.into_inner());
        let mut app = world_app(LightingSettings {
            model: LightingModel::Modern,
            quality: LightingQuality::High,
        });
        enter_world(&mut app, true);
        for _ in 0..8 {
            app.update();
        }
        // Exactly one: the world entry itself.
        assert_eq!(refits(&app), 1);

        app.world_mut().resource_mut::<LightingSettings>().quality = LightingQuality::Ultra;
        for _ in 0..8 {
            app.update();
        }
        assert_eq!(refits(&app), 2, "one tier move is one refit, not eight");
    }

    #[test]
    fn model_cvar_round_trips_and_names_parse() {
        for m in [LightingModel::Classic, LightingModel::Modern] {
            assert_eq!(LightingModel::from_cvar(m.as_cvar() as f32), m);
        }
        assert_eq!(
            LightingModel::parse(" Modern "),
            Some(LightingModel::Modern)
        );
        assert_eq!(
            LightingModel::parse("classic"),
            Some(LightingModel::Classic)
        );
        assert_eq!(LightingModel::parse("2x"), None);
    }

    #[test]
    fn a_modern_night_keeps_a_moonlight_floor() {
        // Midnight (night = 1) over a nearly black authored row: the floor.
        assert_eq!(night_floor([0.01, 0.01, 0.02], 1.0), NIGHT_AMBIENT);
        // Daylight keeps the authored row exactly, whatever it is.
        assert_eq!(night_floor([0.4, 0.45, 0.5], 0.0), [0.4, 0.45, 0.5]);
        // A brighter authored night (a lit zone) is never dimmed to the floor.
        assert_eq!(night_floor([0.3, 0.3, 0.3], 1.0), [0.3, 0.3, 0.3]);
    }

    #[test]
    fn srgb_decode_matches_the_curve() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert!((srgb_to_linear(0.5) - 0.214_041).abs() < 1e-5);
        // Mod2x's neutral grey stays neutral after the shader's 2.295 source scale.
        assert!((2.0 * srgb_to_linear(0.5) * 2.295 - 0.982).abs() < 0.01);
    }

    #[test]
    fn quality_rounds_and_clamps() {
        assert_eq!(LightingQuality::from_cvar(-4.0), LightingQuality::Low);
        assert_eq!(LightingQuality::from_cvar(1.4), LightingQuality::Medium);
        assert_eq!(LightingQuality::from_cvar(2.6), LightingQuality::Ultra);
        assert_eq!(LightingQuality::from_cvar(4.0), LightingQuality::Cinematic);
        assert_eq!(LightingQuality::from_cvar(99.0), LightingQuality::Cinematic);
        assert_eq!(
            LightingQuality::from_cvar(f32::NAN),
            LightingQuality::DEFAULT
        );
        // `ALL` is what every tier table sweeps, so it has to hold every tier and hold them in
        // order — a row added to the enum and left out of here would silently stop being checked.
        assert_eq!(LightingQuality::ALL.len(), 5);
        for (i, q) in LightingQuality::ALL.iter().enumerate() {
            assert_eq!(LightingQuality::from_cvar(i as f32), *q);
            assert_eq!(q.as_cvar() as usize, i);
        }
        // Ultra and Cinematic light from things that merely glow (P8b); the tiers below still light
        // from exactly the sources the reference commits.
        assert!(LightingQuality::Cinematic.lights_emissive_sources());
        assert!(LightingQuality::Ultra.lights_emissive_sources());
        assert!(!LightingQuality::High.lights_emissive_sources());
    }
}
