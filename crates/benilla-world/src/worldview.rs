//! `benilla-worldview` — **the engine with no game attached**, and the wall that keeps it that way.
//!
//! Decision 1160. `benilla-app` is being split: the world renderer becomes `benilla-world`, and the
//! game stands on it. A crate boundary alone cannot hold that line — in Bevy two systems couple
//! with no symbol crossing between them (a `Res<player::Player>` read needs that resource to exist
//! at *runtime*; move the type somewhere neutral and the crate graph is satisfied while the
//! coupling is fully intact). So the enforcer is a **second binary**: this one. It boots the engine
//! plugin set, spawns a free-fly camera, and flies over Elwynn — with no server, no login, no UI,
//! no player. Wire a game concept back into the engine and this stops working *that day, loudly*.
//!
//! It is also the world editor's first milestone: a window that loads a map and lets you fly
//! around is where `benilla-editor` starts.
//!
//! ## Reading this file
//!
//! The plugin list below is **the proposed cut line**, written down. Everything registered here is
//! claimed for `benilla-world`; everything in [`crate::run`] and not here is claimed for the game.
//! The `STUB` block is the spike's finding: the gameplay resources the engine reads *as data* and
//! which a stub therefore satisfies. Each stub is a line item on the work order — the engine should
//! end up not needing it (decision 1160, "the nine wires"), and until it doesn't, the stub says so
//! out loud rather than the coupling hiding inside a working game.

use bevy::camera::{PerspectiveProjection, Projection};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::render::view::Hdr;
use bevy::window::{CursorGrabMode, PrimaryWindow};

use benilla_assets::coords::wow_to_bevy;

mod wade_probe;

use crate::boot;
use crate::build_id::BuildId;
use crate::terrain_stream::SPAWN_XY;
use crate::thread_qos;

/// Where the viewer opens, in WoW world coords. Northshire, Elwynn — the same anchor the client
/// boots on ([`crate::terrain_stream::SPAWN_XY`]), so the two binaries stream the same tiles and a difference
/// between them is a difference in the *engine*, not in where they are standing.
const VIEW_START: (f32, f32) = SPAWN_XY;

/// Height above the spawn point the camera opens at, in yards.
const VIEW_START_HEIGHT: f32 = 60.0;

/// `WOW_WORLDVIEW_AT=<x>,<y>[,<z>]` — **open the viewer somewhere else** (raw WoW coords; `z`
/// defaults to [`VIEW_OPEN_Z`]). The viewer is the only **live, server-less** run we have — a real
/// clock, real streaming, no login — and it was pinned to Northshire, so every question that needs
/// time to pass at a named spot had to go through the capture harness, which freezes the clock by
/// design. That is the gap decision 2038 was diagnosed across: the registration half was
/// answerable in a capture, the per-frame tick was not, at any position but one.
///
/// `None` — unparseable or absent — leaves the viewer at [`VIEW_START`] in its historical opening
/// pose ([`start_pose`]): the enforcer's own anchor, which must stay the default (the two binaries
/// stream the same tiles, and a difference between them is a difference in the engine).
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

/// The look-at point's height when `WOW_WORLDVIEW_AT` names no `z` — high enough over Northshire
/// to clear the valley floor, which is what the viewer has always opened above.
const VIEW_OPEN_Z: f32 = 100.0;

/// Near plane, in yards. The viewer has no CVar table, so it opens at the `nearclip` CVar's
/// registered default and stays there — **read from the const now, not copied** (2163): "kept in
/// step by hand" is what this line used to say while holding `0.1` against a `CAM_NEAR` of `1/9`,
/// and the drift is what showed the 1/9 was wrong in the first place.
const NEAR: f32 = crate::view::NEARCLIP_DEFAULT;

/// Vertical FOV in radians — vanilla's 90° horizontal at 4:3.
const FOVY: f32 = 1.221_730_5;

/// Build and run the world viewer. `build` is the launcher shim's compile-time git stamp, exactly
/// as for [`crate::run`].
pub fn run(build: BuildId) -> AppExit {
    let mut app = App::new();
    app.insert_resource(build);

    // **The survey switch** (`WOW_WORLDVIEW_SURVEY=1`) — the split's instrument, and the reason the
    // spike is one run instead of one rebuild per finding. Bevy's default error handler panics on
    // the first system whose parameters don't validate, so a viewer missing N gameplay resources
    // reports exactly one of them; downgraded to `warn`, a single run names all N and the log IS
    // the work order (decision 1160). Off by default, because for the finished enforcer the panic
    // is the point: wire a game concept back into the engine and this binary must stop, loudly.
    if std::env::var("WOW_WORLDVIEW_SURVEY").as_deref() == Ok("1") {
        warn!("worldview: SURVEY mode — unmet dependencies are warnings, not panics");
        app.set_error_handler(bevy::ecs::error::warn);
    }

    // **The check** (`WOW_WORLDVIEW_CHECK[=seconds]`) — the survey, made into a gate. It collects
    // every distinct fault instead of dying on the first, runs for a bounded time, prints the set
    // and exits non-zero if it is non-empty. `scripts/gates.sh` runs this on every commit.
    //
    // This exists because the enforcer rotted. It is a binary whose entire purpose is to fail
    // loudly the day a game concept is wired back into the engine — and it had been failing on
    // frame one, unnoticed, because nothing ran it (decision 1164). A tripwire nobody trips is not
    // a tripwire. The wall test measures the doorway on every `cargo test`; this measures the
    // other half, the coupling that crosses no symbol at all.
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

    // The `mpq://` asset source must be registered BEFORE `AssetPlugin` (inside `DefaultPlugins`)
    // builds — same order, same reason, same install resolver (1175) as the client's boot.
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
            // Small + cornered for a run nothing photographs (decision 1148, the client's own
            // rule): the check reads the error log, not the framebuffer.
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
        // Same rule as the client (decision 0703): an instrumented run never fights the
        // director's screen.
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
    // The third launch-time platform correction, and the client's own (decision 1528): macOS's
    // `Cmd+Q` is wired to `terminate:`, which leaves the event loop without ever running another
    // frame. Here that costs the check its verdict — `report_check` turns the `AppExit` into the
    // process exit code, and there is no `AppExit` — so the viewer wants it for the same reason
    // the client does, one layer of consequence down.
    .add_plugins(crate::mac_quit::MacQuitPlugin);

    // **The cut line**, and the whole of it: everything `benilla-world` will own, in one name
    // (decision 1164, `crate::world_plugins`). This binary and the client add the identical group,
    // so a divergence between them is no longer possible to write by accident — which is what the
    // hand-kept twin list here was for.
    app.add_plugins(crate::world_plugins::WorldPlugins);
    stubs(&mut app);

    app.add_plugins(plugin);

    // Registered AFTER `AssetPlugin` (they go into the live `AssetServer`) — the client's order.
    benilla_assets::register_asset_loaders(&mut app);

    let exit = app.run();
    match check {
        Some(_) => report_check(exit),
        None => exit,
    }
}

/// `WOW_WORLDVIEW_CHECK` — unset is off; bare (or unparseable) is [`CHECK_SECS_DEFAULT`]; a number
/// is that many seconds. Long enough that the asset chain opens and the first tiles stream, which
/// is when the streaming half of the engine first runs.
fn check_seconds() -> Option<f32> {
    let v = std::env::var("WOW_WORLDVIEW_CHECK").ok()?;
    Some(v.trim().parse().unwrap_or(CHECK_SECS_DEFAULT))
}

/// How long the check runs when it is not given a number. Ten seconds is past the asset chain,
/// past `Startup`, and into streamed tiles on a warm cache — every system in the engine set has
/// been offered to the executor by then, which is what validates its parameters.
const CHECK_SECS_DEFAULT: f32 = 10.0;

/// Wall-clock seconds the check runs for.
#[derive(Resource)]
struct CheckDeadline(f32);

/// Every distinct fault the check saw, by the ECS construct that raised it. A `static` because
/// Bevy's error handler is a bare `fn` pointer and cannot capture.
static FAULTS: std::sync::Mutex<std::collections::BTreeSet<String>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// The check's error handler: record, warn, carry on — so one run names every fault rather than
/// the first one.
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

/// Print the check's verdict and turn it into a process exit code — the whole point of the mode.
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
         fact is parked on the game side. Decision 1160.",
        faults.len()
    );
    // The no-install run (`WOW_DATA=`, decision 1451) fails for a different reason than 1160's,
    // and the run that trips it is a gate line nobody is watching — so it says which reason.
    if benilla_formats::wow_data().is_none() {
        println!(
            "WORLDVIEW_CHECK ran with NO INSTALL: a fault here is a system taking a resource that \
             only exists when there is client data as a hard `Res`/`ResMut`. Take it as `Option` \
             and return, or insert it ahead of the no-data bail. Decision 1451."
        );
    }
    AppExit::error()
}
/// **What the engine still needs told, and nothing more.**
///
/// This used to be five stubs — the 1160 spike's whole finding: a fake `Player` mirrored from the
/// camera every frame, the game's session enum asserted, `ServerTime`, the loading screen, the
/// UI's pointer bool. Every one of them was the engine reaching across the line for a fact, and
/// each retired by the edit its record named rather than by growing a nicer stub:
///
/// - "where is the viewer" → `terrain_stream::ViewFocus` + `view::Viewer`, which the engine
///   defaults, so a program with no avatar follows its camera.
/// - "what time is it" → `lighting::WorldTime`, defaulting to noon.
/// - "is the cover up" → `view::Viewer::world_covered`, defaulting to no.
/// - "is the pointer over the UI" → the engine stopped asking; `cursor` and `bindings` own it.
///
/// What is left is one bit, and it is configuration rather than a stub: **is there a world**.
///
/// ## What this function CANNOT prove — read this before trusting it
///
/// A stub existed here because something *panicked*. Coupling that never panics is invisible to
/// this binary: an ordering edge onto an unregistered system is silently dropped by Bevy, an
/// `Option<Res<…>>` read just sees `None`, and a query filtered on a component nobody spawns
/// simply matches nothing. All three classes were found by *static* sweeps, not by running this —
/// the panic list is a floor on the work order, never a ceiling (decision 1163).
fn stubs(app: &mut App) {
    // ── The world-existence gate ──────────────────────────────────────────────────────────────
    // The viewer's world is permanently live. This used to assert the *game's* session state
    // (`char_select::ClientState::InWorld`) because the engine read it directly; since 1160's
    // wire (b) the engine owns a one-bit `WorldLive` that whatever composes it writes, so the
    // viewer just says yes.
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
        (fly, live_pose.after(fly), toggle_indirect, wade_probe::toggle_wade_probe)
            .in_set(crate::schedule::WorldStage::Input),
    )
    // The wade probe MOVES in the ordinary update, not the input stage: `water_fx`'s emitter reads
    // its `Transform` and differentiates it for the wake heading, so it has to be walking by the
    // time the emitter looks.
    .add_systems(
        Update,
        (wade_probe::auto_spawn_probe, wade_probe::drive_probe),
    );

    // `WOW_WORLDVIEW_SHOT=<png>` (at `WOW_WORLDVIEW_SHOT_AT` seconds, default 20) writes one frame
    // and exits. The client's own live shot (`capture::LiveShotPlugin`) can't serve here: its
    // subject gate reads `SelfPlayer`, the name cache and the net writer, none of which the engine
    // has. This is the viewer's own — deliberately gateless, because a viewer has no subject to be
    // aimed at, only a scene. Post-split it is how `benilla-world` gets a visual regression
    // baseline of its own, with no game in the frame to move underneath it.
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

/// The one-shot frame writer behind `WOW_WORLDVIEW_SHOT` — fires once, then gives the screenshot
/// observer a couple of seconds to reach the disk before exiting (the write is asynchronous; an
/// immediate exit is how a "clean run" ends with no PNG, decision 0743).
fn shoot_and_exit(
    time: Res<Time>,
    mut commands: Commands,
    mut fired_at: Local<Option<f32>>,
    mut exit: MessageWriter<AppExit>,
) {
    let at = std::env::var("WOW_WORLDVIEW_SHOT_AT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20.0);
    match *fired_at {
        None if time.elapsed_secs() >= at => {
            let path = std::env::var("WOW_WORLDVIEW_SHOT").unwrap_or_default();
            info!("worldview: writing {path}");
            commands
                .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
                .observe(bevy::render::view::screenshot::save_to_disk(path));
            *fired_at = Some(time.elapsed_secs());
        }
        Some(t) if time.elapsed_secs() >= t + 2.0 => {
            exit.write(AppExit::Success);
        }
        _ => {}
    }
}

/// The free-fly camera. The client's own `FlyCam` is gameplay-side today and comes over in stage
/// zero (decision 1160); until it does, this is the viewer's own — deliberately the *minimum* that
/// proves the engine renders, not a second implementation to keep in step.
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
) {
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
    if last.as_ref().is_some_and(|(s, grounded)| *s == v && (*grounded || !wants_ground)) {
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

/// WASD + Space/C fly, right-drag look, Ctrl boost, scroll to change speed (`G` toggles indirect
/// lighting, see [`toggle_indirect`]). The editor's
/// navigation, not the game's.
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
