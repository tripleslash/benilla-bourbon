//! **The perf probe** (`WOW_PERF=<frames>` / `?perf=<frames>`) — one greppable line per
//! configuration, in **every** build, including the browser's.
//!
//! ## Why this exists rather than one of the four instruments that already do this
//!
//! `benilla-app::perf` owns the frame-cost measurement layer and it is excellent — the cost pill,
//! the frame-phase split, the whole-frame GPU meter, `FPS_PROBE`'s percentiles. None of it can be
//! used to answer the question this module was built for, for two independent reasons:
//!
//! 1. **It is `#[cfg(feature = "dev")]`, and the web build is not.** `wenilla` takes `benilla-app`
//!    with `default-features = false` (`crates/wenilla/Cargo.toml`) and the page asks for
//!    `data-cargo-no-default-features` (`web/index.html`), deliberately, so the bundle everyone
//!    visits carries no instruments. The consequence nobody wrote down: **the one platform
//!    `FORK.md` calls the budget-setter ("Browser WebGPU sets the budget") is
//!    the one platform with no instrument at all**, which is why every phase note from P2 onward
//!    ends in "Measured GPU cost — deferred".
//! 2. **It lives a crate up.** `benilla-worldview` links `benilla-world` and nothing else
//!    (decision 1160), so an instrument in `benilla-app` cannot run in the world viewer — and the
//!    viewer is exactly where a *rendering* measurement wants to be taken, because it is the
//!    engine with no game attached: no net, no units, no UI, no glue.
//!
//! So this is the journal's own precedent (decision 2008) applied one crate lower: a player-lane
//! instrument, outside the dev seam, because the build whose frames we need to read is the build
//! that compiles the dev seam out. It registers **nothing** when `WOW_PERF` is absent.
//!
//! ## What it measures, and the trap it is designed around
//!
//! **Decision 0717's law: while synced, wall frame time measures the display's present grant, not
//! our cost.** A browser tab cannot opt out — WebGPU presents on `requestAnimationFrame` and there
//! is no `AutoNoVsync` to ask for, so on the web this probe's wall series is railed at the
//! refresh interval for any frame cheaper than it, and a pass ablated out of a 9 ms frame moves
//! nothing at all.
//!
//! Two answers, and the probe supports both:
//!
//! - **`WOW_SUPERSCALE=<f>`** — render the same frame at `f`× the linear resolution, which is the
//!   tree's own documented instrument for this ("supersampling is also the instrument — the only
//!   way to price a pixel on a machine whose present is railed at the display's grant",
//!   `benilla_app::world_backdrop::RENDER_SCALE_RANGE`). `benilla-app`'s `renderScale` is a client
//!   lane the viewer does not have, so this reaches the same place through the window's scale
//!   factor instead. Push it until `railed=no`, then ablate.
//! - **`railed=`** on every line, so a reading taken inside the rail is *labelled* as one rather
//!   than quietly averaged into a table. It is a hint, not a proof: p50 sitting within
//!   [`RAIL_SLACK_MS`] of the run's own fastest frame, at or above [`RAIL_FLOOR_MS`], is what a
//!   vsync-locked cadence looks like and what a genuinely flat CPU-bound frame also looks like.
//!   Treat it as "do not read the ablation deltas off this row".
//!
//! Native has the easier road and already had it: the viewer's `WOW_NOVSYNC=1` sets
//! `AutoNoVsync`, and then the wall series is cost.
//!
//! ## It waits for the world, and says whether it got it
//!
//! Sampling starts only once the terrain streamer has done nothing for [`SETTLE_QUIET_FRAMES`]
//! consecutive frames — not after a fixed warm-up, which the first run of this probe disproved:
//! against remote packs the Elwynn pin was still streaming a minute in, and the opening reading
//! (`min 7.52 / p50 21.49 / p99 40.82`) measured the network. [`SETTLE_MAX_FRAMES`] is the
//! backstop, and a reading taken through it says `settled=no` rather than passing itself off as
//! steady state.
//!
//! **Quiet is not arrived.** Before the streamer's first event its counters read zero, which is
//! quiet: against remote packs the probe settled at frame 240 on a black screen and printed
//! `settled=yes` over the network wait. So the scene must also be presentable with no load in
//! flight ([`world_is_here`]), the same term the loading cover keys on.
//!
//! So two columns decide whether a row is a measurement at all — `settled=yes` and `railed=no` —
//! and a row failing either is an instruction to take it again, not a number to put in a table.
//!
//! ## Per pass, where the device can say
//!
//! With `WOW_GPU_PASSES=1` (native Vulkan or DX12, which time inside passes) a second line,
//! `PERF_PASSES`, follows each `PERF_PROBE`: the GPU milliseconds per frame of every top-level
//! render-diagnostics span over the same sample window, costliest first, with the `modern_*`
//! passes summed as `modern=`. Wall time says a tier got cheaper; this says which pass paid for it.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

/// Frames discarded before the settle test even begins — the pipeline warm and the first-touch
/// uploads. Deliberately the same order as `benilla_app::capture::PROBE_WARMUP_FRAMES`, whose
/// value was chosen against the same warm-up.
const WARMUP_FRAMES: u32 = 120;

/// The window, in frames, the settle test measures the streamer's work rate over.
///
/// A fixed frame warm-up is not enough here and the first run proved it: against remote packs the
/// Elwynn pin was still streaming a minute in, and the probe's opening reading
/// (`min 7.52 / p50 21.49 / p99 40.82`) was a measurement of the network, not of the renderer.
/// What settles a reading is the streamer going quiet, which is a state, not a frame count.
const SETTLE_WINDOW_FRAMES: usize = 120;

/// How much work the streamer may still do across that window and still count as settled.
///
/// **Rate, not zero, and the first settle gate got this wrong.** Requiring the counters to be
/// perfectly still never fired: at the Elwynn pin the world was visibly complete — buildings,
/// trees, clutter grass, a static camera — and the probe still waited, because something in the
/// streamer drips a handful of events forever (a retried read against remote packs, a paced
/// furnisher working a backlog). Streaming a zone is thousands of events; a drip is single
/// digits. The threshold is what separates them, and it has to be a rate or the gate is a
/// promise the engine never keeps.
const SETTLE_TOLERANCE: u64 = 10;

/// …and the frame at which the probe stops waiting and samples anyway, saying so (`settled=no`).
/// A rig that hangs forever on a zone that never fully settles reports nothing, which is worse
/// than reporting a labelled-unsettled number. ~4 minutes at the 45 fps this build runs at.
const SETTLE_MAX_FRAMES: u32 = 12_000;

/// The streamer's event counters, snapshotted — so the backstop can name **which** one is still
/// moving instead of saying only that something is. Diagnosing that from a browser tab otherwise
/// costs a fifteen-minute wasm rebuild, which is how this module learned the lesson.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct StreamSnap([u64; 9]);

impl StreamSnap {
    const NAMES: [&'static str; 9] = [
        "tiles_dropped",
        "placements_dropped",
        "placement_entities_dropped",
        "tiles_requested",
        "tiles_spawned",
        "cells_spawned",
        "model_meshes_built",
        "placements_spawned",
        "colliders_attached",
    ];

    fn of(a: &crate::terrain_stream::StreamActivity) -> Self {
        Self([
            u64::from(a.tiles_dropped),
            u64::from(a.placements_dropped),
            u64::from(a.placement_entities_dropped),
            u64::from(a.tiles_requested),
            u64::from(a.tiles_spawned),
            u64::from(a.cells_spawned),
            u64::from(a.model_meshes_built),
            u64::from(a.placements_spawned),
            u64::from(a.colliders_attached),
        ])
    }

    fn total(self) -> u64 {
        self.0.iter().sum()
    }

    /// `name=delta` for every counter that moved between two snapshots — the backstop's evidence.
    fn moved_since(self, older: Self) -> String {
        let mut out = Vec::new();
        for (i, name) in Self::NAMES.iter().enumerate() {
            let d = self.0[i].saturating_sub(older.0[i]);
            if d > 0 {
                out.push(format!("{name}={d}"));
            }
        }
        if out.is_empty() {
            "nothing".to_string()
        } else {
            out.join(" ")
        }
    }
}

/// How close p50 must sit to the run's fastest frame to read as a vsync rail (ms).
const RAIL_SLACK_MS: f32 = 0.6;

/// …and the floor below which a flat cadence is just a fast frame, not a rail (ms). 15.5 sits
/// under a 60 Hz interval (16.67) and well over a 120 Hz one (8.33), so a 120 Hz panel reads as
/// not railed — which is correct: at 8.33 ms there is room to measure.
const RAIL_FLOOR_MS: f32 = 15.5;

/// `WOW_PERF=<frames>` — how many frames to sample. Absent or unparseable ⇒ the probe is off and
/// this plugin registers nothing.
fn armed() -> Option<u32> {
    crate::launch_env::var("WOW_PERF")
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|n| *n > 0)
}

/// `WOW_SUPERSCALE=<f>` — the resolution multiplier (module doc, "the trap"). Clamped to a sane
/// band: below 0.25 there is nothing left to shade, and above 4 a 1600×900 viewer asks for a
/// 6400×3600 target, which is past what some adapters will allocate.
fn superscale() -> Option<f32> {
    crate::launch_env::var("WOW_SUPERSCALE")
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|f| f.is_finite() && *f > 0.0)
        .map(|f| f.clamp(0.25, 4.0))
}

/// `WOW_PERF_WARMUP=<frames>` — frames discarded before the settle test starts. The default is
/// [`WARMUP_FRAMES`]; a zone whose first-touch uploads run long wants more.
fn warmup() -> u32 {
    crate::launch_env::var("WOW_PERF_WARMUP")
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(WARMUP_FRAMES)
}

/// `WOW_PERF_LOOP=1` — keep sampling: after each window's lines the probe starts the next window,
/// numbering them in the `cfg=` column (`base#3`). With the `shader_reload` feature a saved shader
/// discards the window in flight and waits [`RELOAD_SKIP_FRAMES`] for the pipelines to recompile,
/// so each printed window is wholly before or wholly after an edit.
fn looping() -> bool {
    crate::launch_env::var("WOW_PERF_LOOP").as_deref() == Some("1")
}

/// Frames a looping probe skips after a shader reload: the recompiles land over the next few.
const RELOAD_SKIP_FRAMES: u32 = 30;

/// `WOW_PERF_LABEL=<text>` — what to call this configuration in the line's `cfg=` column. The
/// sweep sets it per leg; by hand it is whatever names the ablation ("`ssao0`", "`base`").
fn label() -> String {
    crate::launch_env::var("WOW_PERF_LABEL").unwrap_or_else(|| "base".to_string())
}

/// The `cfg=` column for window `round` (numbered only when looping).
fn window_label(round: u32) -> String {
    if looping() {
        format!("{}#{round}", label())
    } else {
        label()
    }
}

/// The probe's sample window.
#[derive(Resource)]
struct ProbeRun {
    /// Frames to sample once the world has settled.
    want: u32,
    /// Frames seen at all, for the [`SETTLE_MAX_FRAMES`] backstop.
    seen: u32,
    /// Frames of pure warm-up still owed before the settle test starts.
    warmup_left: u32,
    /// The last [`SETTLE_WINDOW_FRAMES`] snapshots, oldest first — the settle test measures the
    /// work done across this window rather than comparing against zero, because
    /// `StreamActivity`'s counters are "zeroed by whoever reads them" and in the world viewer
    /// nothing does: its only reader (`benilla_app::perf`'s stream trace) is behind the dev seam
    /// the browser build compiles out. A difference across the window is the one test that is
    /// right whether or not someone else is clearing them.
    window: std::collections::VecDeque<StreamSnap>,
    /// Has the settle phase been left? Separate from [`Self::settled`] on purpose: leaving it
    /// because the backstop fired is still leaving it, and a single flag doing both jobs put the
    /// probe in a loop that re-fired the backstop — and its warning — every frame without ever
    /// sampling.
    sampling: bool,
    /// Did sampling start because the world settled, or because the backstop fired? This is the
    /// `settled=` column, and nothing else reads it.
    settled: bool,
    /// Wall frame intervals, ms.
    samples: Vec<f32>,
    /// Per-pass GPU time over the sample window (`WOW_GPU_PASSES`): ms summed per top-level
    /// span, the frames those readings covered, and the newest reading already folded.
    passes: PassSums,
    done: bool,
    /// The window being sampled, counted from 0 (`WOW_PERF_LOOP`).
    round: u32,
    /// A shader was saved mid-window (`shader_reload`): drop the window and start over.
    restart: bool,
    /// Frames still skipped after a restart.
    skip: u32,
}

/// The sample window's per-pass GPU sums, read out of bevy's render diagnostics — the same
/// `render/<pass>/elapsed_gpu` measurements the client's journal folds, and folded the same way:
/// only readings newer than the last fold, divided by the frames whose readback actually landed
/// (the diagnostics resolve a few frames late and drop some when readbacks bunch up).
#[derive(Default)]
struct PassSums {
    ms: std::collections::BTreeMap<String, f64>,
    /// Fragment-shader invocations summed per top-level pass (`render/<pass>/fragment_shader_invocations`,
    /// a pass span's pipeline-statistics query; absent on a device without them).
    frags: std::collections::BTreeMap<String, f64>,
    /// Vertex-shader invocations summed per top-level pass (`…/vertex_shader_invocations`).
    verts: std::collections::BTreeMap<String, f64>,
    /// `ms` and `frags` again, split by the span's order within its frame: a pass that runs once
    /// per view (the water mirror, then the world camera) reads as one figure per view.
    view_ms: std::collections::BTreeMap<String, Vec<f64>>,
    view_frags: std::collections::BTreeMap<String, Vec<f64>>,
    frames: u32,
    seen: Option<bevy::platform::time::Instant>,
}

impl PassSums {
    fn fold(&mut self, store: &bevy::diagnostic::DiagnosticsStore) {
        let mut newest = self.seen;
        let mut times: Vec<bevy::platform::time::Instant> = Vec::new();
        for diagnostic in store.iter() {
            let Some((pass, stat)) = diagnostic
                .path()
                .as_str()
                .strip_prefix("render/")
                .and_then(|p| p.rsplit_once('/'))
                .filter(|(p, _)| !p.contains('/'))
            else {
                continue;
            };
            let timed = stat == "elapsed_gpu";
            let (sums, views) = match stat {
                "elapsed_gpu" => (&mut self.ms, Some(&mut self.view_ms)),
                "fragment_shader_invocations" => (&mut self.frags, Some(&mut self.view_frags)),
                "vertex_shader_invocations" => (&mut self.verts, None),
                _ => continue,
            };
            let mut views = views.map(|v| v.entry(pass.to_string()).or_default());
            // One frame's readings share an instant (`sync_diagnostics`), in span order.
            let mut at = None;
            let mut nth = 0;
            for m in diagnostic
                .measurements()
                .filter(|m| self.seen.is_none_or(|s| m.time > s))
            {
                *sums.entry(pass.to_string()).or_default() += m.value;
                nth = if at == Some(m.time) { nth + 1 } else { 0 };
                at = Some(m.time);
                if let Some(views) = views.as_deref_mut() {
                    if views.len() <= nth {
                        views.resize(nth + 1, 0.0);
                    }
                    views[nth] += m.value;
                }
                if !timed {
                    continue;
                }
                if !times.contains(&m.time) {
                    times.push(m.time);
                }
                if newest.is_none_or(|n| m.time > n) {
                    newest = Some(m.time);
                }
            }
        }
        self.frames += times.len() as u32;
        self.seen = newest;
    }

    /// Skip whatever arrived before sampling began, so the window's sums start clean.
    fn mark(&mut self, store: &bevy::diagnostic::DiagnosticsStore) {
        self.fold(store);
        self.ms.clear();
        self.frags.clear();
        self.verts.clear();
        self.view_ms.clear();
        self.view_frags.clear();
        self.frames = 0;
    }

    /// `pass=ms` per frame, costliest first, and the `modern_*` passes summed as `modern=` —
    /// the one number the lighting tiers are priced by.
    fn line(&self) -> Option<String> {
        if self.frames == 0 {
            return None;
        }
        let n = f64::from(self.frames);
        let mut rows: Vec<(&String, f64)> = self.ms.iter().map(|(k, v)| (k, *v / n)).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        let total: f64 = rows.iter().map(|r| r.1).sum();
        let modern: f64 = rows
            .iter()
            .filter(|r| r.0.starts_with("modern_"))
            .map(|r| r.1)
            .sum();
        let each: Vec<String> = rows.iter().map(|(k, v)| format!("{k}={v:.3}")).collect();
        Some(format!(
            "gpu_frames={} gpu_ms={total:.3} modern={modern:.3} {}",
            self.frames,
            each.join(" ")
        ))
    }

    /// Fragment-shader invocations per frame per screen pixel, per pass, highest first: how many
    /// times the pass shaded each pixel of a `px`-pixel view. A pass that shades every pixel once
    /// reads 1.0; above it is overdraw (an offscreen pass's figure is its own size over the
    /// screen's).
    fn overdraw_line(&self, px: u64) -> Option<String> {
        if self.frames == 0 || px == 0 || self.frags.is_empty() {
            return None;
        }
        let per = f64::from(self.frames) * px as f64;
        let mut rows: Vec<(&String, f64)> = self.frags.iter().map(|(k, v)| (k, *v / per)).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        let each: Vec<String> = rows.iter().map(|(k, v)| format!("{k}={v:.3}")).collect();
        Some(format!("px={px} {}", each.join(" ")))
    }

    /// The passes that ran more than once a frame, each run on its own in span order:
    /// `pass=ms+ms(frag/px+frag/px)`, costliest first.
    fn views_line(&self, px: u64) -> Option<String> {
        if self.frames == 0 {
            return None;
        }
        let n = f64::from(self.frames);
        let mut rows: Vec<(&String, &Vec<f64>)> =
            self.view_ms.iter().filter(|(_, v)| v.len() > 1).collect();
        rows.sort_by(|a, b| b.1.iter().sum::<f64>().total_cmp(&a.1.iter().sum::<f64>()));
        let join = |v: &[f64], per: f64, digits: usize| {
            let each: Vec<String> = v.iter().map(|x| format!("{:.digits$}", x / per)).collect();
            each.join("+")
        };
        let each: Vec<String> = rows
            .iter()
            .map(|(k, v)| match self.view_frags.get(*k).filter(|_| px > 0) {
                Some(f) => format!("{k}={}({})", join(v, n, 2), join(f, n * px as f64, 3)),
                None => format!("{k}={}", join(v, n, 2)),
            })
            .collect();
        (!each.is_empty()).then(|| each.join(" "))
    }

    /// Vertex-shader invocations per frame in thousands, per pass, highest first.
    fn verts_line(&self) -> Option<String> {
        if self.frames == 0 || self.verts.is_empty() {
            return None;
        }
        let per = f64::from(self.frames) * 1000.0;
        let mut rows: Vec<(&String, f64)> = self.verts.iter().map(|(k, v)| (k, *v / per)).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        rows.retain(|r| r.1 >= 0.5);
        let each: Vec<String> = rows.iter().map(|(k, v)| format!("{k}={v:.0}k")).collect();
        Some(each.join(" "))
    }
}

pub struct PerfProbePlugin;

impl Plugin for PerfProbePlugin {
    fn build(&self, app: &mut App) {
        let Some(want) = armed() else {
            return;
        };
        app.insert_resource(ProbeRun {
            want,
            seen: 0,
            warmup_left: warmup(),
            window: std::collections::VecDeque::with_capacity(SETTLE_WINDOW_FRAMES + 1),
            sampling: false,
            settled: false,
            samples: Vec::with_capacity(want as usize),
            passes: PassSums::default(),
            done: false,
            round: 0,
            restart: false,
            skip: 0,
        })
        // `Startup`, not the plugin body: the primary window does not exist until `WindowPlugin`
        // has run, and on the web its physical size is the canvas's, which winit fills in later
        // still.
        .add_systems(Startup, apply_superscale)
        // `Last`, so the sample covers the whole frame the way `Time<Real>`'s own delta does.
        .add_systems(Last, sample);
        #[cfg(feature = "shader_reload")]
        app.add_systems(Last, note_shader_reload.before(sample));
    }
}

/// A saved shader restarts the window in flight (`WOW_PERF_LOOP`).
#[cfg(feature = "shader_reload")]
fn note_shader_reload(
    mut events: MessageReader<AssetEvent<Shader>>,
    assets: Res<AssetServer>,
    mut run: ResMut<ProbeRun>,
) {
    for event in events.read() {
        if let AssetEvent::Modified { id } = event {
            let path = assets.get_path(*id).map(|p| p.to_string());
            info!(
                "perf probe: shader reloaded ({}) — restarting the window",
                path.as_deref().unwrap_or("?")
            );
            run.restart = true;
        }
    }
}

/// Multiply the window's physical resolution (module doc, "the trap").
///
/// A scale-factor override rather than a render-target scale, because it is the one lever that
/// exists in this crate: the viewer renders straight to the surface and has no backdrop lane to
/// resize. Everything downstream — the depth buffer, SSILVB's full-res gather, the fog composite,
/// the glow chain — follows the viewport, which is the point.
fn apply_superscale(mut windows: Query<&mut Window, With<PrimaryWindow>>) {
    let Some(f) = superscale() else {
        return;
    };
    let Ok(mut win) = windows.single_mut() else {
        return;
    };
    win.resolution.set_scale_factor_override(Some(f));
    info!("perf probe: superscale {f}x — the frame is rendered at {f}x linear resolution");
}

fn sample(
    time: Res<Time<Real>>,
    mut run: ResMut<ProbeRun>,
    activity: Option<Res<crate::terrain_stream::StreamActivity>>,
    progress: Option<Res<crate::terrain_stream::WorldLoadProgress>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    adapter: Option<Res<bevy::render::renderer::RenderAdapterInfo>>,
    // Present only where something registered bevy's render diagnostics (`WOW_GPU_PASSES`).
    diagnostics: Option<Res<bevy::diagnostic::DiagnosticsStore>>,
) {
    if run.done {
        return;
    }
    run.seen += 1;

    // Phase 1 — the flat warm-up, before the settle test is even asked.
    if run.warmup_left > 0 {
        run.warmup_left -= 1;
        return;
    }

    // Phase 2 — wait for the streamer's work RATE to fall away (module doc, `SETTLE_TOLERANCE`).
    if !run.sampling {
        let now = activity.map_or_else(StreamSnap::default, |a| StreamSnap::of(&a));
        run.window.push_back(now);
        if run.window.len() > SETTLE_WINDOW_FRAMES {
            run.window.pop_front();
        }
        // A full window is required before any verdict: a short one reads as quiet simply for
        // having seen nothing yet, which would settle the probe during the loading screen.
        let full = run.window.len() >= SETTLE_WINDOW_FRAMES;
        let oldest = run.window.front().copied().unwrap_or_default();
        let done_in_window = now.total().saturating_sub(oldest.total());
        // No streamer at all (a test app) has nothing to wait for.
        let here = progress.as_deref().is_none_or(world_is_here);
        let quiet_enough = full && done_in_window <= SETTLE_TOLERANCE && here;
        let out_of_patience = run.seen >= SETTLE_MAX_FRAMES;
        if !quiet_enough && !out_of_patience {
            return;
        }
        run.sampling = true;
        run.settled = quiet_enough;
        if quiet_enough {
            info!(
                "perf probe: settled after {} frames ({done_in_window} streamer events in the \
                 last {SETTLE_WINDOW_FRAMES}) — sampling {} frames",
                run.seen, run.want,
            );
        } else {
            warn!(
                "perf probe: the streamer never went quiet in {} frames ({done_in_window} events \
                 in the last {SETTLE_WINDOW_FRAMES}: {}; scene presentable: {here}) — sampling \
                 anyway, and the line says settled=no. The number includes streaming work and is \
                 not a measurement of the renderer.",
                run.seen,
                now.moved_since(oldest),
            );
        }
        // The settle frame itself carries the last of the streaming work; never sample it.
        if let Some(store) = diagnostics.as_deref() {
            run.passes.mark(store);
        }
        return;
    }

    // Phase 3 — the measurement.
    if std::mem::take(&mut run.restart) {
        run.samples.clear();
        run.skip = RELOAD_SKIP_FRAMES;
        return;
    }
    if run.skip > 0 {
        run.skip -= 1;
        if run.skip == 0 {
            if let Some(store) = diagnostics.as_deref() {
                run.passes.mark(store);
            }
        }
        return;
    }
    let dt = time.delta_secs() * 1000.0;
    // A zero or absurd delta is a tab that was backgrounded or a frame the browser skipped —
    // neither is a measurement of this configuration.
    if dt > 0.0 && dt < 1000.0 {
        run.samples.push(dt);
    }
    if let Some(store) = diagnostics.as_deref() {
        run.passes.fold(store);
    }
    if (run.samples.len() as u32) < run.want {
        return;
    }
    run.done = true;
    let settled = run.settled;
    let cfg = window_label(run.round);
    emit(
        &cfg,
        &run.samples,
        settled,
        windows.single().ok(),
        adapter.as_deref(),
    );
    if let Some(line) = run.passes.line() {
        info!("PERF_PASSES cfg={cfg} {line}");
    }
    let px = windows.single().map_or(0, |w| {
        u64::from(w.resolution.physical_width()) * u64::from(w.resolution.physical_height())
    });
    if let Some(line) = run.passes.overdraw_line(px) {
        info!("PERF_OVERDRAW cfg={cfg} {line}");
    }
    if let Some(line) = run.passes.views_line(px) {
        info!("PERF_VIEWS cfg={cfg} {line}");
    }
    if let Some(line) = run.passes.verts_line() {
        info!("PERF_VERTS cfg={cfg} {line}");
    }
    if looping() {
        run.done = false;
        run.round += 1;
        run.samples.clear();
        if let Some(store) = diagnostics.as_deref() {
            run.passes.mark(store);
        }
    }
}

/// Has the world around the focus arrived? The scene presentable, as the loading cover reads it,
/// and nothing still on the wire.
fn world_is_here(p: &crate::terrain_stream::WorldLoadProgress) -> bool {
    p.presentable() && p.loads_in_flight == 0
}

/// The percentile a sorted sample lands on — nearest-rank, the same convention the journal and the
/// cost pill use, so a number from this probe is comparable with one from those.
fn pct(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p * sorted.len() as f32).ceil() as usize).saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

fn emit(
    cfg: &str,
    samples: &[f32],
    settled: bool,
    window: Option<&Window>,
    adapter: Option<&bevy::render::renderer::RenderAdapterInfo>,
) {
    let mut s = samples.to_vec();
    s.sort_by(f32::total_cmp);
    let min = s.first().copied().unwrap_or(0.0);
    let p50 = pct(&s, 0.50);
    let p95 = pct(&s, 0.95);
    let p99 = pct(&s, 0.99);
    let max = s.last().copied().unwrap_or(0.0);
    let mean = s.iter().sum::<f32>() / s.len().max(1) as f32;
    let railed = p50 >= RAIL_FLOOR_MS && (p50 - min) < RAIL_SLACK_MS;

    let (px, scale) = window.map_or((UVec2::ZERO, 1.0), |w| {
        (
            UVec2::new(
                w.resolution.physical_width(),
                w.resolution.physical_height(),
            ),
            w.resolution.scale_factor(),
        )
    });
    let gpu = adapter.map_or_else(
        || "?".to_string(),
        |a| format!("{} [{:?}]", a.name, a.backend),
    );

    // One line, one configuration, fixed column order — a census whose columns move cannot be
    // diffed across runs (`world_census`'s own rule).
    info!(
        "PERF_PROBE cfg={} frames={} px={}x{} scale={:.2} \
         min_ms={:.2} mean_ms={:.2} p50_ms={:.2} p95_ms={:.2} p99_ms={:.2} max_ms={:.2} \
         fps={:.1} railed={} settled={} gpu={}",
        cfg,
        s.len(),
        px.x,
        px.y,
        scale,
        min,
        mean,
        p50,
        p95,
        p99,
        max,
        1000.0 / p50.max(0.001),
        if railed { "yes" } else { "no" },
        if settled { "yes" } else { "no" },
        gpu,
    );
    if railed {
        warn!(
            "PERF_PROBE cfg={} is RAILED at the present interval — p50 {:.2} ms sits within \
             {RAIL_SLACK_MS} ms of the run's fastest frame. Ablation deltas read off this row are \
             meaningless (decision 0717). Raise WOW_SUPERSCALE until railed=no, or on native run \
             with WOW_NOVSYNC=1.",
            cfg, p50,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two frames of a 100-pixel view: a pass that shaded 200 fragments shaded each pixel once.
    #[test]
    fn overdraw_is_fragments_per_frame_per_pixel() {
        let mut sums = PassSums {
            frames: 2,
            ..default()
        };
        sums.frags.insert("main_opaque_pass_3d".into(), 500.0);
        sums.frags.insert("static_gx".into(), 200.0);
        assert_eq!(
            sums.overdraw_line(100).as_deref(),
            Some("px=100 main_opaque_pass_3d=2.500 static_gx=1.000")
        );
        assert_eq!(sums.overdraw_line(0), None);
    }

    /// A pass that ran for two views reads once per view; a pass that ran once is left out.
    #[test]
    fn a_pass_run_per_view_reads_per_view() {
        let mut sums = PassSums {
            frames: 2,
            ..default()
        };
        sums.view_ms.insert("static_gx".into(), vec![5.0, 2.0]);
        sums.view_frags
            .insert("static_gx".into(), vec![20.0, 200.0]);
        sums.view_ms.insert("upscaling".into(), vec![1.0]);
        assert_eq!(
            sums.views_line(100).as_deref(),
            Some("static_gx=2.50+1.00(0.100+1.000)")
        );
        sums.verts.insert("static_gx".into(), 6000.0);
        assert_eq!(sums.verts_line().as_deref(), Some("static_gx=3k"));
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let s: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        assert_eq!(pct(&s, 0.50), 50.0);
        assert_eq!(pct(&s, 0.95), 95.0);
        assert_eq!(pct(&s, 0.99), 99.0);
        // The tails are clamped rather than panicking on an off-by-one at either end.
        assert_eq!(pct(&s, 1.0), 100.0);
        assert_eq!(pct(&s, 0.0), 1.0);
        assert_eq!(pct(&[], 0.5), 0.0);
    }

    /// The settle test must ignore the two field families that never go quiet: `provisional_live`
    /// is a LEVEL (non-zero in a settled world) and the `_ms` self-timers are floats that move
    /// every frame. A settle test that counted either would never fire, and the probe would always
    /// fall through to its backstop and report `settled=no`.
    #[test]
    fn the_snapshot_counts_events_and_ignores_levels_and_timers() {
        use crate::terrain_stream::StreamActivity;
        let mut a = StreamActivity::default();
        assert_eq!(StreamSnap::of(&a).total(), 0);

        a.provisional_live = 7;
        a.stream_ms = 1.25;
        a.furnish_ms = 0.5;
        a.spawn_ms = 2.0;
        assert_eq!(
            StreamSnap::of(&a).total(),
            0,
            "a settled world still holds provisional colliders and still spends time in the \
             stream stage — neither is streaming WORK"
        );

        a.cells_spawned = 3;
        assert_eq!(StreamSnap::of(&a).total(), 3, "an event counter is work");
    }

    /// The backstop's whole value is naming the counter that would not stop, so that a drip can be
    /// diagnosed from a browser console instead of a fifteen-minute wasm rebuild.
    #[test]
    fn the_backstop_names_which_counter_is_still_moving() {
        let older = StreamSnap([1, 0, 0, 5, 2, 9, 0, 0, 3]);
        let newer = StreamSnap([1, 0, 0, 8, 2, 9, 0, 0, 4]);
        let moved = newer.moved_since(older);
        assert!(moved.contains("tiles_requested=3"), "{moved}");
        assert!(moved.contains("colliders_attached=1"), "{moved}");
        assert!(
            !moved.contains("cells_spawned"),
            "a still counter is not evidence: {moved}"
        );
        assert_eq!(newer.moved_since(newer), "nothing");
    }

    /// A streamer that has not started is quiet but not here: the black-screen settle.
    #[test]
    fn a_world_that_has_not_arrived_is_not_settled() {
        use crate::terrain_stream::WorldLoadProgress;
        assert!(!world_is_here(&WorldLoadProgress::default()));
        let here = WorldLoadProgress {
            scene_ready: true,
            focus_collidable: true,
            ..default()
        };
        assert!(world_is_here(&here));
        let fetching = WorldLoadProgress {
            loads_in_flight: 3,
            ..here
        };
        assert!(!world_is_here(&fetching));
    }

    /// The rate test, which is the fix for the gate that never fired: a drip must settle and a
    /// streaming burst must not.
    #[test]
    fn a_drip_settles_and_a_burst_does_not() {
        let settled = |done: u64| done <= SETTLE_TOLERANCE;
        assert!(settled(0), "a truly still world settles");
        assert!(
            settled(4),
            "a handful of retried reads across 120 frames is a drip"
        );
        assert!(!settled(1_500), "streaming a zone is not a settled world");
    }

    /// The rail test is the whole reason a web reading can be trusted or not, so it is pinned:
    /// a 60 Hz cadence reads as railed, a 120 Hz one does not (there is room to measure at
    /// 8.33 ms), and a frame with real spread does not whatever its median.
    #[test]
    fn rail_detection_separates_a_vsync_cadence_from_a_measurable_frame() {
        let railed = |min: f32, p50: f32| p50 >= RAIL_FLOOR_MS && (p50 - min) < RAIL_SLACK_MS;
        assert!(railed(16.6, 16.7), "a flat 60 Hz cadence is a rail");
        assert!(!railed(8.3, 8.34), "120 Hz has headroom below the floor");
        assert!(!railed(11.0, 17.2), "spread that wide is cost, not a grant");
    }
}
