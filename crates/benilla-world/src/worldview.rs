//! `benilla-worldview`: the engine with no game attached, and the wall that keeps it that way. It
//! boots the engine plugins with a free-fly camera over Elwynn and no server, login, UI or player,
//! so a game concept wired back into the engine breaks it. A crate boundary alone cannot hold that
//! line: two Bevy systems couple through a runtime resource with no symbol crossing between them.

use bevy::camera::Hdr;
use bevy::camera::{PerspectiveProjection, Projection};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, PrimaryWindow};

use benilla_assets::coords::wow_to_bevy;

// fork: fsr (FORK.md) — `WOW_RENDER_SCALE` / `WOW_FRAME_GENERATION`: FSR without a login.
mod fly_path;
mod fsr_view;
mod wade_probe;

use crate::boot;
use crate::build_id::BuildId;
use crate::terrain_stream::SPAWN_XY;
use crate::thread_qos;

/// Where the viewer opens, in WoW world coords: Northshire, the client's own boot anchor
/// ([`crate::terrain_stream::SPAWN_XY`]), so both binaries stream the same tiles.
const VIEW_START: (f32, f32) = SPAWN_XY;

/// Height above the spawn point the camera opens at, in yards.
const VIEW_START_HEIGHT: f32 = 60.0;

/// `WOW_WORLDVIEW_AT=<x>,<y>[,<z>]` opens the viewer elsewhere (raw WoW coords, `z` defaulting to
/// [`VIEW_OPEN_Z`]); `None`, unparseable or absent, stays at [`VIEW_START`] in its opening pose
/// ([`start_pose`]), which must remain the default.
fn view_start() -> Option<[f32; 3]> {
    let v = crate::launch_env::var("WOW_WORLDVIEW_AT")?;
    let c: Vec<f32> = v.split(',').filter_map(|p| p.trim().parse().ok()).collect();
    match c.len() {
        2 => Some([c[0], c[1], VIEW_OPEN_Z]),
        3 => Some([c[0], c[1], c[2]]),
        _ => {
            warn!("worldview: WOW_WORLDVIEW_AT wants `x,y[,z]` — staying at the default anchor");
            None
        }
    }
}

/// The look-at height when `WOW_WORLDVIEW_AT` names no `z`, clear of the Northshire valley floor.
const VIEW_OPEN_Z: f32 = 100.0;

/// Near plane in yards: the `nearclip` CVar's default, since the viewer has no CVar table.
const NEAR: f32 = crate::view::NEARCLIP_DEFAULT;

/// Vertical FOV in radians (70°), wider than the client camera's [`crate::view::CAM_FOVY`].
const FOVY: f32 = 1.221_730_5;

/// Builds and runs the world viewer; `build` is the launcher shim's compile-time git stamp.
pub fn run(build: BuildId) -> AppExit {
    // The install resolves from the launcher's folder when its stamp names one, as the client's
    // does; this shim has no `dev` feature, so it stamps none and the viewer keeps benilla's.
    benilla_formats::set_project_folder(build.project_dir);
    // fork: perf — the profiler's subscriber before any system exists, so every system's span
    // (made when the system is built) records.
    #[cfg(feature = "trace_systems")]
    crate::system_profile::install();
    let mut app = App::new();
    app.insert_resource(build);

    // `WOW_WORLDVIEW_SURVEY=1` turns Bevy's panic on unvalidated system parameters into a
    // warning, so one run names every missing resource.
    if std::env::var("WOW_WORLDVIEW_SURVEY").as_deref() == Ok("1") {
        warn!("worldview: SURVEY mode — unmet dependencies are warnings, not panics");
        app.set_error_handler(bevy::ecs::error::warn);
    }

    // `WOW_WORLDVIEW_CHECK[=seconds]`, the survey as a gate: it records every distinct fault, runs
    // for a bounded time, prints the set and exits non-zero if any; `scripts/gates.sh` runs it.
    let check = check_seconds();
    if let Some(secs) = check {
        app.set_error_handler(record_fault);
        app.insert_resource(CheckDeadline(secs))
            .add_systems(Update, end_check);
    }

    // Optional content layers, before either chain opens. `$WOW_HD_MODELS` is the whole of the
    // ladder here: this probe has no CVar host and no `config.toml`, so the environment is the only
    // thing that could name a layer — which is exactly what the env arm exists for.
    benilla_formats::set_hd_character_models(
        benilla_formats::layers::hd_models_env().unwrap_or(false),
    );

    // The `mpq://` source must be registered before `AssetPlugin` (in `DefaultPlugins`) builds.
    match benilla_formats::wow_data() {
        Some(data_dir) => {
            if let Err(e) = benilla_assets::register_mpq_source(&mut app, &data_dir) {
                eprintln!("benilla-assets: mpq:// source unavailable ({e:#})");
            }
        }
        None => eprintln!(
            "benilla-worldview: no WoW install found — looked in {:?}",
            benilla_formats::candidates()
        ),
    }

    let background = crate::bgwin::background_run();
    app.add_plugins(boot::tuned_default_plugins(Window {
        title: "benilla worldview".into(),
        resolution: std::env::var("WOW_WIN")
            .ok()
            .and_then(|v| {
                let (w, h) = v.split_once('x')?;
                Some(UVec2::new(w.parse().ok()?, h.parse().ok()?))
            })
            // Small for a run that reads no pixels, as in the client.
            .unwrap_or(if crate::bgwin::no_pixel_run() {
                UVec2::new(640, 360)
            } else {
                UVec2::new(1600, 900)
            })
            .into(),
        present_mode: if std::env::var("WOW_NOVSYNC").as_deref() == Ok("1") {
            bevy::window::PresentMode::AutoNoVsync
        } else {
            bevy::window::PresentMode::default()
        },
        // As in the client: an instrumented run opens unfocused, below normal windows.
        focused: !background,
        window_level: if background {
            bevy::window::WindowLevel::AlwaysOnBottom
        } else {
            bevy::window::WindowLevel::Normal
        },
        ..default()
    }))
    .add_plugins(thread_qos::ThreadQosPlugin)
    .add_plugins(crate::bgwin::BgWinPlugin)
    // macOS `Cmd+Q` is wired to `terminate:`, which leaves the event loop with no `AppExit`, and
    // `report_check` needs one for the exit code.
    .add_plugins(crate::mac_quit::MacQuitPlugin);

    // The cut line: the engine's whole plugin group, the same one the client adds.
    app.add_plugins(crate::world_plugins::WorldPlugins);
    // `WOW_GPU_PASSES=1` — bevy's per-pass GPU spans, so the perf probe can print what each pass
    // cost beside its wall-time line (`perf_probe`'s `PERF_PASSES`). The client registers the same
    // plugin for its journal; the viewer has no journal, so this is the only way it times a pass.
    // Needs a device that times inside passes (Vulkan, DX12); elsewhere the line never prints.
    if crate::launch_env::var("WOW_GPU_PASSES").is_some() {
        app.add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin);
    }
    stubs(&mut app);

    app.add_plugins(plugin);

    // After `AssetPlugin`: the loaders go into the live `AssetServer`, as in the client.
    benilla_assets::register_asset_loaders(&mut app);

    let exit = app.run();
    match check {
        Some(_) => report_check(exit),
        None => exit,
    }
}

/// `WOW_WORLDVIEW_CHECK` in seconds: unset is off, bare or unparseable is [`CHECK_SECS_DEFAULT`].
fn check_seconds() -> Option<f32> {
    let v = std::env::var("WOW_WORLDVIEW_CHECK").ok()?;
    Some(v.trim().parse().unwrap_or(CHECK_SECS_DEFAULT))
}

/// Long enough on a warm cache for every engine system to run and validate its parameters.
const CHECK_SECS_DEFAULT: f32 = 10.0;

/// Wall-clock seconds the check runs for.
#[derive(Resource)]
struct CheckDeadline(f32);

/// Every distinct fault the check saw, by the ECS construct that raised it. A `static` because
/// Bevy's error handler is a bare `fn` pointer and cannot capture.
static FAULTS: std::sync::Mutex<std::collections::BTreeSet<String>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// The check's error handler: records and warns, so one run names every fault.
fn record_fault(error: bevy::ecs::error::BevyError, ctx: bevy::ecs::error::ErrorContext) {
    let entry = format!("{} `{}`: {error}", ctx.kind(), ctx.name());
    warn!("worldview: {entry}");
    FAULTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(entry);
}

fn end_check(
    time: Res<Time<bevy::time::Real>>,
    deadline: Res<CheckDeadline>,
    mut exit: MessageWriter<AppExit>,
) {
    if time.elapsed_secs() >= deadline.0 {
        exit.write(AppExit::Success);
    }
}

/// Prints the check's verdict and makes it the process exit code.
fn report_check(exit: AppExit) -> AppExit {
    let faults = FAULTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if faults.is_empty() {
        println!("WORLDVIEW_CHECK ok — the engine booted and ran with no game attached");
        return exit;
    }
    for f in faults.iter() {
        println!("WORLDVIEW_CHECK fault {f}");
    }
    println!(
        "WORLDVIEW_CHECK {} fault(s) — a game concept is wired into the engine, or an engine \
         fact is parked on the game side.",
        faults.len()
    );
    // A run with no install (`WOW_DATA=`) faults for a different reason, so it says which.
    if benilla_formats::wow_data().is_none() {
        println!(
            "WORLDVIEW_CHECK ran with NO INSTALL: a fault here is a system taking a resource that \
             only exists when there is client data as a hard `Res`/`ResMut`. Take it as `Option` \
             and return, or insert it ahead of the no-data bail."
        );
    }
    AppExit::error()
}
/// What the engine still needs told: only whether there is a world. Coupling that never panics is
/// invisible to this binary: an ordering edge onto an unregistered system is dropped, an
/// `Option<Res<…>>` sees `None`, and a query on a component nobody spawns matches nothing.
fn stubs(app: &mut App) {
    // The viewer's world is always live; whatever composes the engine writes `WorldLive`.
    app.insert_resource(crate::schedule::WorldLive(true));
}

/// The viewer itself: the free-fly camera and its controller.
fn plugin(app: &mut App) {
    app.add_systems(
        Startup,
        spawn_view_camera.after(benilla_assets::AssetSet::Open),
    )
    .add_systems(
        Update,
        (
            fly,
            live_pose.after(fly),
            toggle_indirect,
            cycle_lighting_tier,
            wade_probe::toggle_wade_probe,
        )
            .in_set(crate::schedule::WorldStage::Input),
    )
    // The wade probe MOVES in the ordinary update, not the input stage: `water_fx`'s emitter reads
    // its `Transform` and differentiates it for the wake heading, so it has to be walking by the
    // time the emitter looks.
    .add_systems(
        Update,
        (wade_probe::auto_spawn_probe, wade_probe::drive_probe),
    );

    // fork: fsr (FORK.md) — the FSR mode, when its knobs are set.
    fsr_view::plugin(app);
    // fork: perf — a scripted flight (`WOW_WORLDVIEW_FLY`), when its waypoints are set.
    fly_path::plugin(app);
    // fork: perf — the per-system CPU profile, in a `trace_systems` build.
    #[cfg(feature = "trace_systems")]
    crate::system_profile::plugin(app);

    // `WOW_WORLDVIEW_SHOT=<png>` (at `WOW_WORLDVIEW_SHOT_AT` seconds, default 20) writes one frame
    // and exits, with no subject gate: the client's live shot needs a player the engine lacks.
    if std::env::var("WOW_WORLDVIEW_SHOT").is_ok() {
        app.add_systems(Update, shoot_and_exit);
    }
}

/// **`G` flips screen-space indirect lighting between full strength and off** — the viewer's A/B
/// lever, and the reason it is a hotkey rather than another `$WOW_*` variable: the two states have
/// to be seen in the SAME frame, over the same trunk or the same doorway, or the eye simply agrees
/// with whichever it saw last. Relaunching for each half cannot do that.
///
/// It flips between the SHIPPED strengths and off, not between a maximum and off. Since both halves
/// carry a gain that puts the intended look at the sliders' midpoint, the shipped 0.5 already is
/// the gather's own term applied whole; 1.0 is a deliberate over-drive, and A/B-ing against an
/// over-drive tells you what the exaggeration looks like rather than what the game looks like.
///
/// Toggling is safe mid-run. The first enable re-allocates the depth buffer to be sampleable, and
/// the render node stands that one frame down rather than bind a texture that is not armed yet;
/// the flag is never cleared again, so every later toggle is free.
fn toggle_indirect(
    keys: Res<ButtonInput<KeyCode>>,
    mut lighting: ResMut<crate::ssilvb::IndirectLighting>,
) {
    if !keys.just_pressed(KeyCode::KeyG) {
        return; // never touch the resource on a quiet frame — that would re-extract it every tick
    }
    let shipped = crate::ssilvb::IndirectLighting {
        ao: crate::ssilvb::AO_DEFAULT,
        gi: crate::ssilvb::GI_DEFAULT,
    };
    let off = crate::ssilvb::IndirectLighting { ao: 0.0, gi: 0.0 };
    *lighting = if lighting.enabled() { off } else { shipped };
    info!(
        "worldview: indirect lighting {} (ao {:.2}, gi {:.2})",
        if lighting.enabled() { "ON" } else { "OFF" },
        lighting.ao,
        lighting.gi,
    );
}

/// **`T` cycles the lighting tier** — Low → Medium → High → Ultra → Cinematic → Low.
///
/// The worldview has no CVar host, so `?lighting_quality=` is the only tier knob it has and it is
/// read once, at startup. That was fine while the tier latched at world entry and a change meant a
/// reload anyway; now that the tier is live (`modern::ActiveLighting`), a relaunch is exactly the
/// wrong instrument — it re-streams the zone, re-fits the probe field and re-seeds the fly camera,
/// so the one thing the change is about, *the same frame rendered at two tiers*, is the one thing
/// it cannot show. This is `toggle_indirect`'s reasoning, one knob over.
///
/// It writes [`LightingSettings`], not `ActiveLighting` — the same door the options dropdown pushes
/// on, so what the key exercises is the real path and not a shortcut past it.
fn cycle_lighting_tier(
    keys: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<crate::modern::LightingSettings>,
) {
    if !keys.just_pressed(KeyCode::KeyT) {
        return; // never touch the resource on a quiet frame — the latch keys off a real change
    }
    let all = crate::modern::LightingQuality::ALL;
    let next = all[(settings.quality as usize + 1) % all.len()];
    settings.quality = next;
    info!("worldview: lighting tier -> {next:?}");
}

/// The frame writer behind `WOW_WORLDVIEW_SHOT` — fires, then gives the screenshot observer a
/// couple of seconds to reach the disk before exiting (the write is asynchronous; an immediate exit
/// is how a "clean run" ends with no PNG, decision 0743).
///
/// **`WOW_WORLDVIEW_SHOT_COUNT=<n>` makes it a burst** — `n` frames written as `<stem>-000.png`,
/// `<stem>-001.png`, … , spaced by `WOW_WORLDVIEW_SHOT_EVERY` seconds (**default 0 = one per
/// frame**, i.e. adjacent frames). One shot still writes the bare path, so every existing
/// invocation is unchanged. This is the client burst's twin (`capture::live_shot`,
/// `WOW_LIVE_SHOT_COUNT`) for the engine binary, and it exists for the same reason: **a flicker is
/// a temporal artefact and a single frame cannot show one** (decision 0653). The viewer parks its
/// own camera, so a burst from it is already the still subject the arithmetic wants —
/// `benilla-visual flicker <dir>` collapses it to the envelope of what would not hold still, and
/// the lit pixels are the defect.
fn shoot_and_exit(
    time: Res<Time>,
    mut commands: Commands,
    mut shots: Local<(u32, Option<f32>)>,
    mut exit: MessageWriter<AppExit>,
) {
    let knob = |k: &str, d: f32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let at = knob("WOW_WORLDVIEW_SHOT_AT", 20.0);
    let every = knob("WOW_WORLDVIEW_SHOT_EVERY", 0.0);
    let count = knob("WOW_WORLDVIEW_SHOT_COUNT", 1.0).max(1.0) as u32;
    let now = time.elapsed_secs();
    let (taken, last) = *shots;
    // Every frame of the burst is written the same way; only the name changes with `count`.
    if taken < count && last.is_none_or(|t| now >= t + every) && now >= at {
        let stem = std::env::var("WOW_WORLDVIEW_SHOT").unwrap_or_default();
        let path = if count > 1 {
            let (base, ext) = stem.rsplit_once('.').unwrap_or((stem.as_str(), "png"));
            format!("{base}-{taken:03}.{ext}")
        } else {
            stem
        };
        info!("worldview: writing {path}");
        commands
            .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
            .observe(bevy::render::view::screenshot::save_to_disk(path));
        *shots = (taken + 1, Some(now));
        return;
    }
    // The observer writes asynchronously, so the last shot gets the same grace the single one had.
    if taken >= count && last.is_some_and(|t| now >= t + 2.0) {
        exit.write(AppExit::Success);
    }
}

/// The viewer's minimal free-fly camera, not a twin of the client's `FlyCam`.
#[derive(Component)]
struct ViewCam {
    yaw: f32,
    pitch: f32,
    speed: f32,
}

/// The eye [`view_start`] names, aimed by `WOW_WORLDVIEW_FACE` (heading degrees, 0 = +X,
/// counter-clockwise) and `WOW_WORLDVIEW_PITCH` (degrees, + = up). The client's
/// `WOW_CAPTURE=vista` twin, for the engine binary: `WOW_MAP` picks the continent, so one launch
/// frames any doodad in the world.
///
/// With no position named the viewer keeps the pose it has always opened in — above the anchor,
/// looking down at it — rather than the horizontal one the aim knobs default to. The two knobs
/// only make sense against a point someone chose, and defaulting them over the anchor would move
/// the enforcer's own opening frame.
fn start_pose() -> (Vec3, f32, f32) {
    let Some(at) = view_start() else {
        let start = wow_to_bevy([VIEW_START.0, VIEW_START.1, VIEW_OPEN_Z]);
        return (
            start + Vec3::new(0.0, VIEW_START_HEIGHT, VIEW_START_HEIGHT),
            0.0,
            -0.5,
        );
    };
    let deg = |k: &str| {
        // fork: modern lighting — reachable from the browser (?worldview_face= / ?worldview_pitch=).
        crate::launch_env::var(k)
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(0.0)
            .to_radians()
    };
    // WoW heading 0 = +X, which `wow_to_bevy` lands on −Z; the fly controller's yaw is measured
    // the same way its own look-drag writes it, so the two agree on the first frame.
    (
        wow_to_bevy(at),
        deg("WOW_WORLDVIEW_FACE"),
        deg("WOW_WORLDVIEW_PITCH"),
    )
}

fn spawn_view_camera(mut commands: Commands, msaa: Res<crate::view::MsaaSetting>) {
    let (eye, yaw, pitch) = start_pose();
    let far = crate::view::CAM_FAR;
    let look = eye + Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) * Vec3::NEG_Z;
    commands.spawn((
        Camera3d::default(),
        crate::view::WorldCamera,
        msaa.level(),
        Projection::from(PerspectiveProjection {
            far,
            near: NEAR,
            fov: FOVY,
            ..default()
        }),
        Hdr,
        Tonemapping::None,
        crate::ffx_glow::FfxGlow::WORLD,
        // One cluster, as the client's world camera (`view::cpu_light_clusters`).
        bevy::light::cluster::ClusterConfig::Single,
        Transform::from_translation(eye).looking_at(look, Vec3::Y),
        ViewCam {
            yaw,
            pitch,
            speed: 100.0,
        },
    ));
}

/// fork: modern lighting — `?worldview_pose=x,y,z,face,pitch[,ground]` (raw WoW coords, degrees),
/// read LIVE (`launch_env::live_var`), so a reference framing can be dialled in from the page with
/// `history.replaceState` while the streamed world stays loaded. With `ground`, `z` is ignored once
/// the terrain under the eye has streamed and the eye sits that many yards above it.
#[allow(clippy::type_complexity)]
fn live_pose(
    streamer: Option<Res<crate::terrain_stream::TerrainStreamer>>,
    tiles: Option<Res<Assets<benilla_assets::AdtTile>>>,
    mut cam: Query<(&mut Transform, &mut ViewCam)>,
    mut last: Local<Option<(String, bool)>>,
    debug: Option<ResMut<crate::dev_state::DebugState>>,
    time: Res<Time>,
    mut reported: Local<(f32, Vec3, f32, f32)>,
) {
    // Report where the viewer stands when it has settled somewhere new, in `?worldview_pose=` form.
    if let Ok((xf, view)) = cam.single() {
        reported.0 += time.delta_secs();
        let moved = xf.translation.distance(reported.1) > 1.0
            || (view.yaw - reported.2).abs() > 0.02
            || (view.pitch - reported.3).abs() > 0.02;
        if moved && reported.0 > 2.0 {
            reported.0 = 0.0;
            reported.1 = xf.translation;
            reported.2 = view.yaw;
            reported.3 = view.pitch;
            let w = benilla_assets::coords::bevy_to_wow(xf.translation);
            info!(
                "worldview pose: {:.1},{:.1},{:.1},{:.1},{:.1}",
                w[0],
                w[1],
                w[2],
                view.yaw.to_degrees(),
                view.pitch.to_degrees()
            );
        }
    }

    // The clock follows the page too (`?clock=`).
    if let (Some(mut d), Some(m)) = (
        debug,
        crate::launch_env::live_var("WOW_CLOCK").and_then(|c| c.trim().parse::<u32>().ok()),
    ) {
        if d.lighting.follow_server_time || d.lighting.manual_minute != m % 1440 {
            d.lighting.follow_server_time = false;
            d.lighting.manual_minute = m % 1440;
        }
    }
    let Some(v) = crate::launch_env::live_var("WOW_WORLDVIEW_POSE") else {
        return;
    };
    let c: Vec<f32> = v.split(',').filter_map(|p| p.trim().parse().ok()).collect();
    if c.len() < 5 {
        return;
    }
    let wants_ground = c.len() >= 6;
    if last
        .as_ref()
        .is_some_and(|(s, grounded)| *s == v && (*grounded || !wants_ground))
    {
        return;
    }
    let Ok((mut xf, mut view)) = cam.single_mut() else {
        return;
    };
    let mut eye = wow_to_bevy([c[0], c[1], c[2]]);
    let mut grounded = false;
    if wants_ground {
        if let (Some(s), Some(t)) = (streamer.as_deref(), tiles.as_deref()) {
            if let Some(h) = crate::terrain_stream::terrain_height_under(s, t, eye) {
                eye.y = h + c[5];
                grounded = true;
            }
        }
    }
    view.yaw = c[3].to_radians();
    view.pitch = c[4].to_radians();
    xf.translation = eye;
    xf.rotation = Quat::from_euler(EulerRot::YXZ, view.yaw, view.pitch, 0.0);
    *last = Some((v, grounded));
}

/// WASD, Space and C fly, a right or left drag looks, Ctrl boosts, the wheel sets speed; `G`
/// toggles indirect lighting ([`toggle_indirect`]).
fn fly(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    mut wheel: MessageReader<bevy::input::mouse::MouseWheel>,
    mut window: Query<&mut bevy::window::CursorOptions, With<PrimaryWindow>>,
    mut cam: Query<(&mut Transform, &mut ViewCam)>,
) {
    let Ok((mut xf, mut cam)) = cam.single_mut() else {
        return;
    };

    let looking = buttons.pressed(MouseButton::Right) || buttons.pressed(MouseButton::Left);
    if let Ok(mut cursor) = window.single_mut() {
        let want = if looking {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
        if cursor.grab_mode != want {
            cursor.grab_mode = want;
            cursor.visible = !looking;
        }
    }
    if looking {
        const LOOK: f32 = 0.003;
        cam.yaw -= motion.delta.x * LOOK;
        cam.pitch = (cam.pitch - motion.delta.y * LOOK).clamp(-1.54, 1.54);
    }
    xf.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);

    for ev in wheel.read() {
        cam.speed = (cam.speed * (1.0 + ev.y * 0.1)).clamp(1.0, 2000.0);
    }

    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        dir += *xf.forward();
    }
    if keys.pressed(KeyCode::KeyS) {
        dir += *xf.back();
    }
    if keys.pressed(KeyCode::KeyA) {
        dir += *xf.left();
    }
    if keys.pressed(KeyCode::KeyD) {
        dir += *xf.right();
    }
    if keys.pressed(KeyCode::Space) {
        dir += Vec3::Y;
    }
    if keys.pressed(KeyCode::KeyC) {
        dir -= Vec3::Y;
    }
    if dir != Vec3::ZERO {
        let boost = if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
            5.0
        } else {
            1.0
        };
        xf.translation += dir.normalize() * cam.speed * boost * time.delta_secs();
    }
}
