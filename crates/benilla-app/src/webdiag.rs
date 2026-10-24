//! `webdiag` — the browser-build stutter probe.
//!
//! One instrument for one report: *the run is smooth for ~2 minutes and then micro-stutters, worse
//! the longer it runs.* That shape is a **growth** shape — something the frame walks gets bigger
//! every second — so the probe measures growth beside cost, in the same window, and prints both as
//! one line a viewer can paste back.
//!
//! Three questions, one per column of the report:
//!
//! - **Is the frame actually going long, and by how much?** A rolling window of wall frame times,
//!   reported as p50/p95/max plus a count of frames over [`SLOW_MS`]. p50 is the honest baseline
//!   here: on a browser's `requestAnimationFrame` the mean is dragged by every tab-throttle gap.
//! - **Which phase spends it?** The same marker-schedule technique `perf::phases` uses (a stamp
//!   at each main-schedule boundary is a hard sync point, so it cannot float), but **aggregated
//!   over the slow frames only** rather than printed one line per frame — a stutter storm would
//!   otherwise emit hundreds of lines a second and bury the growth signal it is meant to sit next
//!   to. The residue between the end of `Last` and the next frame's opening stamp is reported as
//!   its own span for exactly the reason the phases module gives: the first cut of that instrument
//!   closed its tape at the end of `Last` and reported a 506 ms frame as fitting inside 20 ms of
//!   `Main`.
//!
//!   **That residue is two things, and it used to be reported as one.** It is the render sub-app
//!   and the present — and it is also every asset decode, because on this target a loader is a
//!   chain of microtasks and the browser drains those only once the frame's animation-frame
//!   callback has returned (`benilla_assets::frame_budget`, whose own header says so). So a
//!   window whose worst frames were mostly BLP and ADT decode read `render+present=47.0` and
//!   pointed at the renderer, which is the one part of this client that cannot be moved off the
//!   frame thread at all (`WgpuWrapper` is a `SendWrapper` under `+atomics`). The two are now
//!   split: `decode` is what `frame_budget` charged the frame, `render+present` is the rest. The
//!   difference is the whole question of what is worth offloading, so it is worth two lines of
//!   arithmetic to stop guessing at it.
//! - **What is growing?** Entity count by **component**, summed across archetypes — a leak splits
//!   its entities over several archetypes as unrelated components come and go, so an archetype
//!   table hides the very population it is meant to name, while `Mesh3d: +3800` does not. Beside
//!   it: the `Assets<T>` populations (an asset leak needs no entity at all), wasm linear memory,
//!   and the JS heap — three answers that separate an ECS leak from a GPU-resource leak from a
//!   page-side one.
//! - **What did the page's own state cost this frame?** `localStorage.setItem` is **synchronous**
//!   and runs on the thread drawing the frame, so a resident of `benilla-config/` that is rewritten
//!   on a timer and whose contents *grow* with the session is a stutter that gets worse the longer
//!   the tab is open — and it is invisible to every frame meter we have, which can say a frame went
//!   long but never that a storage write is why. `crate::local_state::write_meter` times each write
//!   and names the file; a line here whose ms and KB both climb is the answer on its own.
//!
//! Deltas are printed twice over, because the two readings answer different questions: `d` on the
//! per-window line is the **rate** right now, and the growers block every [`GROWTH_SECS`] is the
//! **cumulative** move since the baseline, which is what a slow leak shows up in first. The
//! baseline is taken at the first report after [`BASELINE_SKIP_SECS`], so that login, world entry
//! and the first streaming burst — all of which legitimately add tens of thousands of entities —
//! do not sit in the number as a permanent, meaningless `+`.
//!
//! **Web-only, and opt-in on a shipped build.** This is a diagnostic build's instrument: the
//! reporter reaches for `console.log` directly rather than the `tracing` subscriber, so it still
//! speaks after `LogPlugin` is taken off the web build (`crate::quiet_console_on_web`) to leave the
//! console carrying nothing but these lines.
//!
//! In a **debug** build it is on unless `?diag=0`. In a **release** one it is off unless `?diag=1`,
//! because every line is a wasm→JS bridge call and the probe takes an `Instant::now()` — which is
//! `performance.now()` across that boundary — at each of thirteen schedule marks *every frame*
//! before it prints anything. A player who never opens the console should pay for none of it, and
//! [`armed`] false takes the marks, the stamps and the report out of the schedule entirely.
//!
//! It lives here rather than under `perf` for a build reason worth stating: `perf` is behind the
//! `dev` feature, and the web build is `--no-default-features --features webgpu` (see `Trunk.toml`)
//! — so a probe for the web build cannot be inside the module the web build compiles out.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use benilla_assets::materials::WowModelMaterial;
use bevy::app::{First, Last, MainScheduleOrder, PostUpdate, PreUpdate, Update};
use bevy::ecs::component::ComponentId;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::time::Real;

use benilla_world::schedule::WorldStage;

/// A frame at or over this is "slow" — the frames the phase aggregate is taken over. Twice a
/// 60 Hz interval: high enough that ordinary jitter and a vsync beat do not qualify, low enough to
/// catch the ~30-60 ms hitches a player reads as a micro-stutter (the pill's own hitch warn sits
/// at 250 ms and would see none of these).
const SLOW_MS: f32 = 33.0;
/// Seconds between report lines.
const REPORT_SECS: f32 = 5.0;
/// Seconds between the cumulative growers blocks.
const GROWTH_SECS: f32 = 30.0;
/// The baseline for cumulative growth is taken at the first report after this many seconds of
/// uptime — see the module header on why login and world entry must not be inside it.
const BASELINE_SKIP_SECS: f32 = 20.0;
/// Components/assets listed in a growers block.
const TOP_N: usize = 12;
/// Phase spans listed in a report line.
const TOP_PHASES: usize = 6;
/// Live-pipeline lines printed per session before the rest are silenced — see
/// [`note_live_pipeline`].
const LIVE_PIPE_CAP: usize = 24;

/// The marker schedules spliced between the main ones — the index is the position, so the splice
/// table below reads as the frame's shape.
#[derive(ScheduleLabel, Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DiagMark(u8);

/// Where each mark is spliced, and what the *preceding* span is called. The first names no span:
/// it is the frame's opening stamp.
const MARKS: &[(u8, &str)] = &[
    (0, ""),
    (1, "First"),
    (2, "PreUpdate"),
    (3, "StateTransition"),
    (4, "Update/post-Present"),
    (5, "PostUpdate"),
    (6, "Last"),
];

#[derive(Resource)]
struct Diag {
    /// Wall frame times this window (ms), reset each report.
    frames: Vec<f32>,
    /// Frames this window at or over [`SLOW_MS`].
    slow: u32,
    /// Total ms across those slow frames, and per-span sums over the same frames — so the report
    /// can say what a slow frame is *made of* rather than what an average frame is.
    slow_total_ms: f32,
    slow_spans: HashMap<&'static str, f32>,
    /// Asset-decode ms over EVERY frame this window, not just the slow ones — the steady-state
    /// cost of the work that could leave the thread, which is a different question from what a
    /// stutter is made of and is answered on the other line.
    decode_ms: f32,
    /// `(span name, instant at its END)` for the frame in flight; the first entry is its t0.
    marks: Vec<(&'static str, Instant)>,

    next_report: f32,
    next_growth: f32,
    /// `PipeWatch::created` at the previous report, so `pipes` is a per-window delta like every
    /// other growth number on the line.
    prev_pipes: usize,
    /// Counts at the previous report — the `d` on each line.
    prev: Counts,
    /// Counts at the baseline, once taken — the growers block's cumulative move.
    baseline: Option<Counts>,
}

/// One census: entities per component, plus the populations no component can see.
///
/// Keyed by [`ComponentId`], not by name: a census runs every [`REPORT_SECS`] and is cloned into
/// the previous-reading slot each time, so naming here would allocate a `String` per component per
/// census and clone the lot again — the instrument's own cost, paid on the thread it is measuring.
/// Ids are `Copy`; names are resolved for the dozen rows that actually print.
#[derive(Default, Clone)]
struct Counts {
    entities: u32,
    archetypes: usize,
    /// Archetypes ever made: bevy never removes one, and each new one costs every query a rescan.
    archetypes_made: usize,
    /// Component -> entities carrying it.
    components: HashMap<ComponentId, u32>,
    meshes: usize,
    images: usize,
    materials: usize,
    /// Entities in the **render** world — see [`RenderCensus`].
    render_entities: u64,
    /// wasm linear memory, bytes.
    wasm_bytes: u64,
    /// `performance.memory.usedJSHeapSize`, bytes — Chromium only, 0 elsewhere.
    js_heap: u64,
}

/// The render world's entity count, published to the main world by one atomic.
///
/// Bevy 0.18's render world is **retained**: entities are synced across from the main world and
/// live between frames rather than being cleared and re-extracted. So it is a second population
/// that can grow on its own, and the main-world census above cannot see a single row of it —
/// which would make an extract/sync leak read as "nothing is growing" on every line of this
/// report. One `AtomicU64`, stored in the render app's `Cleanup` and read in the main world's
/// `Last`: cheap enough to leave on, and it rules a whole world in or out.
#[derive(Resource, Clone)]
struct RenderCensus(Arc<AtomicU64>);

pub(super) fn plugin(app: &mut App) {
    if !armed() {
        return;
    }
    out("[diag] armed — one line every 5 s, growers every 30 s. `?diag=0` silences it.");
    // The `thr=` on every line below is this pair, said once in full. `0/0` is a page with no
    // workers, where every offload in this client is back on the frame thread.
    out(&format!(
        "[diag] worker threads: {} async-compute/compute.",
        worker_threads()
    ));
    app.insert_resource(Diag {
        frames: Vec::with_capacity(512),
        slow: 0,
        slow_total_ms: 0.0,
        slow_spans: HashMap::new(),
        decode_ms: 0.0,
        marks: Vec::with_capacity(16),
        prev_pipes: 0,
        next_report: REPORT_SECS,
        next_growth: BASELINE_SKIP_SECS + GROWTH_SECS,
        prev: Counts::default(),
        baseline: None,
    });

    // Splice the marker schedules. `insert_before(First, …)` opens the frame; every other mark
    // closes the schedule it is inserted after. `StateTransition` is seeded by `StatesPlugin`
    // after `PreUpdate`, so it is in the order by the time this builds.
    {
        let mut order = app.world_mut().resource_mut::<MainScheduleOrder>();
        order.insert_before(First, DiagMark(0));
        order.insert_after(First, DiagMark(1));
        order.insert_after(PreUpdate, DiagMark(2));
        order.insert_after(bevy::state::state::StateTransition, DiagMark(3));
        order.insert_after(Update, DiagMark(4));
        order.insert_after(PostUpdate, DiagMark(5));
        order.insert_after(Last, DiagMark(6));
    }
    for (i, name) in MARKS {
        let (i, name) = (*i, *name);
        app.add_systems(DiagMark(i), move |mut d: ResMut<Diag>| {
            if i == 0 {
                close_frame(&mut d);
            }
            d.marks.push((name, Instant::now()));
        });
    }

    // The four `Update` stages, stamped as EXCLUSIVE systems so each stamp is a sync point and the
    // span either side of it is that stage's own. On native this would be a real perturbation of
    // `Update`'s parallelism; the web build has one thread, so there is no parallelism to perturb.
    app.add_systems(
        Update,
        (
            stamp("Update/pre-Net").before(WorldStage::Net),
            stamp("Update/Net")
                .after(WorldStage::Net)
                .before(WorldStage::Input),
            stamp("Update/Input")
                .after(WorldStage::Input)
                .before(WorldStage::Stream),
            stamp("Update/Stream")
                .after(WorldStage::Stream)
                .before(WorldStage::Present),
            stamp("Update/Present").after(WorldStage::Present),
        ),
    );
    app.add_systems(Last, report);

    // The render world's own headcount (see [`RenderCensus`]). Absent sub-app = a headless run;
    // the column then reads 0 rather than the probe failing to build.
    let shared = Arc::new(AtomicU64::new(0));
    app.insert_resource(RenderCensus(shared.clone()));
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.insert_resource(RenderCensus(shared));
        render_app.add_systems(Render, render_census.in_set(RenderSystems::Cleanup));
    }
}

/// Render app, `Cleanup`: publish this frame's render-world headcount.
fn render_census(world: &mut World) {
    let n = u64::from(world.entities().len());
    if let Some(c) = world.get_resource::<RenderCensus>() {
        c.0.store(n, Ordering::Relaxed);
    }
}

/// Is the probe on? Web only, and on there unless the page says `?diag=0`.
///
/// Public to the crate because it is also the console-quieting switch: `benilla_app::run` takes
/// `LogPlugin` off the web build when this answers true, so the console carries these lines and
/// nothing else. One flag, because the two are one intent — *diagnostic mode* is a console with
/// only the diagnostic on it, and `?diag=0` restores both the probe's silence and the ordinary log.
pub(crate) fn armed() -> bool {
    if !cfg!(target_arch = "wasm32") {
        return false;
    }
    match crate::webenv::var("WOW_DIAG").as_deref() {
        Some("1") => true,
        Some("0") => false,
        // **Release is silent by default; a debug build still talks.** Every line this probe emits
        // is a `console.log`, which is a wasm→JS bridge call — and the probe is not one line a
        // frame, it is a `performance.now()` at each of thirteen schedule marks *per frame* before
        // it prints anything. That is the right price for a diagnostic and the wrong one for a
        // player who never opens the console, so a shipped build pays neither: `armed()` false
        // takes the marks, the stamps and the report out of the schedule entirely (see `plugin`,
        // which returns before adding any of them).
        //
        // `?diag=1` is the way back, and it is the same URL the last three stutter reports came
        // from — nothing about the instrument changed except who has to ask for it.
        _ => cfg!(debug_assertions),
    }
}

/// One exclusive stamp closing the named span.
fn stamp(name: &'static str) -> impl IntoSystem<(), (), ()> {
    IntoSystem::into_system(move |world: &mut World| {
        if let Some(mut d) = world.get_resource_mut::<Diag>() {
            d.marks.push((name, Instant::now()));
        }
    })
}

/// Close the previous frame's tape at the NEXT frame's opening stamp — everything between the end
/// of `Last` and here is the render sub-app and the present, and it is charged as such.
fn close_frame(d: &mut Diag) {
    let Some((_, t0)) = d.marks.first().copied() else {
        d.marks.clear();
        return;
    };
    let end = Instant::now();
    let total = (end - t0).as_secs_f32() * 1000.0;
    d.frames.push(total);
    // The frame that is closing here is the one `frame_budget` has been charging, and this runs
    // ahead of `First` — so it is the last moment the counter still names that frame rather than
    // the one beginning. See `frame_budget::spent_ms`, which documents the same ordering from its
    // side; a reader in `Last` would sample a frame whose decodes had not happened yet and get
    // zero every time.
    let decode = (benilla_assets::frame_budget::spent_ms() as f32).min(total);
    d.decode_ms += decode;
    if total >= SLOW_MS {
        d.slow += 1;
        d.slow_total_ms += total;
        let mut prev = t0;
        // Collected first, then folded in: the loop borrows `d.marks` while the fold writes
        // `d.slow_spans`.
        let spans: Vec<(&'static str, f32)> = d
            .marks
            .iter()
            .skip(1)
            .map(|(name, at)| {
                let ms = (*at - prev).as_secs_f32() * 1000.0;
                prev = *at;
                (*name, ms)
            })
            .collect();
        for (name, ms) in spans {
            *d.slow_spans.entry(name).or_default() += ms;
        }
        // The residue, split. Decode is charged inside this window (microtasks drain after the
        // callback returns), so it comes OUT of the residue rather than being reported beside it —
        // two spans that sum to what the one span used to say, which is what makes the new number
        // comparable to an old paste. `max(0.0)` because the two clocks are not the same
        // instrument: `frame_budget` times each slot's own `Drop` while this times the whole gap,
        // and a rounding disagreement must not print a negative renderer.
        let residue = (end - prev).as_secs_f32() * 1000.0;
        let decode = decode.min(residue);
        *d.slow_spans.entry("decode").or_default() += decode;
        *d.slow_spans.entry("render+present").or_default() += (residue - decode).max(0.0);
    }
    d.marks.clear();
}

/// The GPU's share of a frame, from bevy's per-pass render diagnostics (`RenderDiagnosticsPlugin`,
/// registered by `perf::journal`): each top-level pass's smoothed time over the frames it ran,
/// summed, and the largest. With the render app on a worker the page's frame time no longer holds
/// the render, so a frame the GPU paces reads here and nowhere else. `None` without timestamps.
fn gpu_line(world: &World) -> Option<String> {
    let store = world.get_resource::<bevy::diagnostic::DiagnosticsStore>()?;
    let mut passes: Vec<(&str, f64)> = store
        .iter()
        .filter_map(|d| {
            let pass = d
                .path()
                .as_str()
                .strip_prefix("render/")?
                .strip_suffix("/elapsed_gpu")?;
            (!pass.contains('/')).then_some(())?;
            Some((pass, d.average()?))
        })
        .collect();
    if passes.is_empty() {
        return None;
    }
    let total: f64 = passes.iter().map(|p| p.1).sum();
    passes.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut line = format!("\n[diag]   gpu avg={total:.1}ms per world frame:");
    for (pass, ms) in passes.iter().take(TOP_PHASES) {
        line.push_str(&format!(" {pass}={ms:.1}"));
    }
    Some(line)
}

/// fork: the render worker's share of the window (`bevy::render::render_worker`). With the render
/// app on a worker, the page's residue after `Last` is no longer the render: it is the wait for the
/// worker's extract (wake, extract, the page's queued buffer writes) and whatever the browser does
/// before the next callback. The render itself is timed on the worker and reported here, beside
/// the updates that ended without an extract (`owed`), the updates the runner then held until it
/// was paid (`skipped`), and the extracts the worker did not claim in time (`withdrawn`).
fn render_worker_line() -> Option<String> {
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    {
        use bevy::render::render_worker;
        if !render_worker::active() {
            return None;
        }
        // `?gpucensus=1` takes the same counters for the same 5 s; two readers would each see
        // part of every window, so the census line carries them alone.
        if crate::webenv::var("WOW_GPUCENSUS").as_deref() == Some("1") {
            return Some("\n[diag]   render worker: see the [gpucensus] line".to_owned());
        }
        let s = render_worker::take_stats();
        let avg = |sum: u64, n: u64| {
            if n > 0 {
                sum as f32 / n as f32 / 1000.0
            } else {
                0.0
            }
        };
        Some(format!(
            "\n[diag]   render worker: renders={} avg={:.1}ms max={:.1}ms | extracts={} wait \
             avg={:.1}ms max={:.1}ms | owed={} skipped={} withdrawn={}",
            s.renders,
            avg(s.render_us, s.renders),
            s.render_max_us as f32 / 1000.0,
            s.extracts,
            avg(s.wait_us, s.extracts),
            s.wait_max_us as f32 / 1000.0,
            s.owed,
            s.skipped,
            s.withdrawn,
        ))
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
    None
}

/// The report, in `Last` so the census sees the frame's finished world.
fn report(world: &mut World) {
    let now = world.resource::<Time<Real>>().elapsed_secs();
    if now < world.resource::<Diag>().next_report {
        return;
    }

    let counts = census(world);
    // Taken here, before `Diag` is borrowed: `report` holds that borrow to the end of the line it
    // is building, and the streamer's counters live in a second resource.
    let activity = world
        .get_resource_mut::<benilla_world::terrain_stream::StreamActivity>()
        .map(|mut a| std::mem::take(&mut *a));
    // The pipeline cache's own pair. On this target a cold variant is a SYNCHRONOUS compile on the
    // thread that draws (`bevy_render`'s `create_pipeline_task` has a `target_arch = "wasm32"` arm
    // that `block_on`s the build), so a window whose `pipes` moved and whose worst frame sat in
    // `render+present` has already explained itself — and `pending` says whether more is coming.
    let pipes = world
        .get_resource::<crate::pipe_warm::PipeWatch>()
        .map(|w| {
            use std::sync::atomic::Ordering;
            // `settled` first: the pair is published unsynchronised, so reading it first can only
            // over-report pending, never invent a negative.
            let settled = w.0.settled.load(Ordering::Relaxed);
            (w.0.created.load(Ordering::Relaxed), settled)
        });
    let gpu = gpu_line(world);
    let fresh = new_archetypes(world, world.resource::<Diag>().prev.archetypes_made);
    let mut d = world.resource_mut::<Diag>();
    d.next_report = now + REPORT_SECS;

    // ---- cost ----------------------------------------------------------------------------
    let mut ms = std::mem::take(&mut d.frames);
    ms.sort_by(|a, b| a.total_cmp(b));
    let pick = |q: f32| -> f32 {
        if ms.is_empty() {
            return 0.0;
        }
        ms[((ms.len() as f32 - 1.0) * q).round() as usize]
    };
    let (p50, p95, max, n) = (pick(0.5), pick(0.95), pick(1.0), ms.len());

    // ---- growth --------------------------------------------------------------------------
    let prev = std::mem::replace(&mut d.prev, counts.clone());
    let de = counts.entities as i64 - prev.entities as i64;
    let dm = counts.meshes as i64 - prev.meshes as i64;
    let di = counts.images as i64 - prev.images as i64;
    let dt = counts.materials as i64 - prev.materials as i64;
    let dr = counts.render_entities as i64 - prev.render_entities as i64;
    let dw = counts.wasm_bytes as i64 - prev.wasm_bytes as i64;
    let dj = counts.js_heap as i64 - prev.js_heap as i64;
    let da = counts.archetypes_made as i64 - prev.archetypes_made as i64;

    // Per FRAME, not per window: the same unit as the p50 beside it, so "decode is 3 of my 14 ms"
    // reads straight off the line. Zero here with a non-zero mesh delta is the answer this probe
    // was extended to give — the decode left the frame thread and the stream cost did not.
    let decode = if n > 0 { d.decode_ms / n as f32 } else { 0.0 };
    // Hoisted for the same reason `decode` is: the `format!` below borrows `d` immutably for the
    // whole call, and this reading advances the previous-report cursor.
    let pipes = pipes_field(&mut d, pipes);
    let mut line = format!(
        "[diag] t={now:.0}s frames={n} | ms p50={p50:.1} p95={p95:.1} max={max:.1} decode={decode:.1} \
         | slow>{SLOW_MS:.0}ms={} | ents={} {} rents={} {} arch={} made={} {} | mesh={} {} img={} {} mat={} {} \
         | wasm={} {} js={} {} | thr={} pipes={} audio={} | grow={}",
        d.slow,
        counts.entities,
        signed(de),
        counts.render_entities,
        signed(dr),
        counts.archetypes,
        counts.archetypes_made,
        signed(da),
        counts.meshes,
        signed(dm),
        counts.images,
        signed(di),
        counts.materials,
        signed(dt),
        mb(counts.wasm_bytes),
        signed_mb(dw),
        mb(counts.js_heap),
        signed_mb(dj),
        worker_threads(),
        pipes,
        audio_field(),
        grow_field(),
    );
    line.push_str(&fresh);

    // What a slow frame was made of, averaged over the slow frames in this window only.
    if d.slow > 0 {
        let slow = d.slow as f32;
        let mut spans: Vec<(&str, f32)> =
            d.slow_spans.iter().map(|(k, v)| (*k, v / slow)).collect();
        spans.sort_by(|a, b| b.1.total_cmp(&a.1));
        line.push_str(&format!(
            "\n[diag]   slow frame avg={:.1}ms:",
            d.slow_total_ms / slow
        ));
        for (name, ms) in spans.iter().take(TOP_PHASES) {
            if *ms >= 0.05 {
                line.push_str(&format!(" {name}={ms:.1}"));
            }
        }
    }
    // Which STREAMING system spent the frame. The counters already existed and were already
    // written on every target — `benilla_world`'s streamer has self-timed its five systems since
    // 0832 — but the only consumer was `perf::trace`, which is behind the `dev` feature and so is
    // not in the web build at all. On this target they were therefore write-only: nothing read
    // them, and nothing cleared them either. Reading them here is most of the answer to "what is
    // left on the frame thread now that decode is off it", and it costs one `take` per report.
    if let Some(stream) = activity.and_then(|a| stream_line(a, n)) {
        line.push_str(&stream);
    }
    if let Some(worker) = render_worker_line() {
        line.push_str(&worker);
    }
    if let Some(gpu) = gpu {
        line.push_str(&gpu);
    }
    // fork: the browser frees a dropped GPU object only when its JS wrapper is collected, so the
    // renderer destroys the released ones itself (`bevy_render`'s ninth delta); a window that
    // creates and drops textures or buffers and reads 0 here is leaking them until a collection.
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    line.push_str(&format!(
        "\n[diag]   gpu objects destroyed={}",
        bevy::render::take_released_gpu_objects()
    ));
    // State writes are `localStorage.setItem` on this thread — synchronous, and charged to
    // whatever frame made them. A file here whose ms and bytes both climb with the session is a
    // stutter that grows, and nothing else in this report would name it (see
    // `local_state::write_meter`).
    for (name, calls, bytes, ms) in storage_writes() {
        line.push_str(&format!(
            "\n[diag]   localStorage {name}: {calls}x {ms:.1}ms, {:.0}KB",
            bytes as f64 / 1024.0
        ));
    }
    // Blocking chain reads: a synchronous XHR plus a pack decode on this thread, each one. Named,
    // because the caller is what needs moving to `Chain::try_read` (see
    // `benilla_formats::webpack::sync_meter`).
    let blocking = blocking_reads();
    if !blocking.is_empty() {
        let total: f32 = blocking.iter().map(|b| b.2).sum();
        let reads: u32 = blocking.iter().map(|b| b.1).sum();
        line.push_str(&format!(
            "
[diag]   blocking chain reads: {reads}x {total:.1}ms —"
        ));
        for (name, calls, ms) in blocking.iter().take(TOP_BLOCKING) {
            line.push_str(&format!(" {name} {calls}x {ms:.1}ms;"));
        }
    }
    out(&line);

    d.slow = 0;
    d.slow_total_ms = 0.0;
    d.decode_ms = 0.0;
    d.slow_spans.clear();
    ms.clear();
    d.frames = ms;

    // ---- the cumulative growers block ----------------------------------------------------
    if now < BASELINE_SKIP_SECS {
        return;
    }
    if d.baseline.is_none() {
        d.baseline = Some(counts.clone());
        out(&format!(
            "[diag] baseline taken at t={now:.0}s — growers below are cumulative from here."
        ));
        return;
    }
    if now < d.next_growth {
        return;
    }
    d.next_growth = now + GROWTH_SECS;
    let base = d.baseline.clone().expect("baseline set above");
    let since = now;
    growers(world, &base, &counts, since);
}

/// The cumulative block: what has grown since the baseline, largest move first. Names are resolved
/// here and only here — for the [`TOP_N`] rows that print, out of the several hundred counted.
fn growers(world: &World, base: &Counts, now: &Counts, t: f32) {
    let mut rows: Vec<(ComponentId, i64, u32)> = now
        .components
        .iter()
        .map(|(&id, &n)| {
            let was = base.components.get(&id).copied().unwrap_or(0);
            (id, n as i64 - was as i64, n)
        })
        .filter(|(_, d, _)| *d > 0)
        .collect();
    rows.sort_by_key(|(_, d, _)| std::cmp::Reverse(*d));

    out(&format!(
        "[diag] growers at t={t:.0}s (vs baseline): ents {} {}, rents {} {}, mesh {} {}, \
         img {} {}, mat {} {}, wasm {} {}, js {} {}",
        now.entities,
        signed(now.entities as i64 - base.entities as i64),
        now.render_entities,
        signed(now.render_entities as i64 - base.render_entities as i64),
        now.meshes,
        signed(now.meshes as i64 - base.meshes as i64),
        now.images,
        signed(now.images as i64 - base.images as i64),
        now.materials,
        signed(now.materials as i64 - base.materials as i64),
        mb(now.wasm_bytes),
        signed_mb(now.wasm_bytes as i64 - base.wasm_bytes as i64),
        mb(now.js_heap),
        signed_mb(now.js_heap as i64 - base.js_heap as i64),
    ));
    if rows.is_empty() {
        out("[diag]   no component population grew — the leak is not entities.");
        return;
    }
    for (id, delta, total) in rows.into_iter().take(TOP_N) {
        let name = world
            .components()
            .get_info(id)
            .map_or_else(|| format!("{id:?}"), |i| short(&i.name().to_string()));
        out(&format!("[diag]   {delta:>+8}  (now {total:>7})  {name}"));
    }
}

/// How many archetypes a window lists by their components, at most.
const NEW_ARCHETYPES_SHOWN: usize = 8;

/// The first archetypes made since `since`, by their components: whatever keeps making them is the
/// code to look at. Nothing before the first report, which would list the whole boot.
fn new_archetypes(world: &World, since: usize) -> String {
    let made = world.archetypes().len();
    if since == 0 || made <= since {
        return String::new();
    }
    let mut out = String::new();
    for arch in world
        .archetypes()
        .iter()
        .skip(since)
        .take(NEW_ARCHETYPES_SHOWN)
    {
        let names: Vec<String> = arch
            .components()
            .iter()
            .map(|&id| {
                world
                    .components()
                    .get_info(id)
                    .map_or_else(|| format!("{id:?}"), |i| short(&i.name().to_string()))
            })
            .collect();
        out.push_str(&format!(
            "\n[diag]   new archetype ({} ents): {}",
            arch.len(),
            names.join(", ")
        ));
    }
    if made - since > NEW_ARCHETYPES_SHOWN {
        out.push_str(&format!(
            "\n[diag]   and {} more new archetypes",
            made - since - NEW_ARCHETYPES_SHOWN
        ));
    }
    out
}

/// One census of the world: every non-empty archetype's population charged to each component it
/// carries, so a population split across archetypes still reads as one number.
fn census(world: &mut World) -> Counts {
    let mut components: HashMap<ComponentId, u32> = HashMap::new();
    let mut archetypes = 0usize;
    for arch in world.archetypes().iter() {
        if arch.is_empty() {
            continue;
        }
        archetypes += 1;
        let n = arch.len();
        for &c in arch.components().iter() {
            *components.entry(c).or_default() += n;
        }
    }

    Counts {
        entities: world.entities().len(),
        archetypes,
        archetypes_made: world.archetypes().len(),
        components,
        meshes: world.get_resource::<Assets<Mesh>>().map_or(0, Assets::len),
        images: world.get_resource::<Assets<Image>>().map_or(0, Assets::len),
        materials: world
            .get_resource::<Assets<WowModelMaterial>>()
            .map_or(0, Assets::len),
        render_entities: world
            .get_resource::<RenderCensus>()
            .map_or(0, |c| c.0.load(Ordering::Relaxed)),
        wasm_bytes: wasm_bytes(),
        js_heap: js_heap(),
    }
}

/// `bevy_render::mesh::components::Mesh3d` -> `components::Mesh3d`: the last two path segments,
/// generics dropped. Enough to name a population without a line per row being a paragraph.
fn short(full: &str) -> String {
    let base = full.split('<').next().unwrap_or(full);
    let segs: Vec<&str> = base.split("::").collect();
    segs[segs.len().saturating_sub(2)..].join("::")
}

/// The streamer's five self-timed systems, as **per-frame ms over the window**, plus what they
/// actually produced — or `None` on a window where the streamer did nothing at all.
///
/// The ms are the question ("which system is spending the frame"); the counts are what keeps the
/// answer honest, because a lane that is expensive *because it is doing a lot* reads completely
/// differently from one that is expensive per item. `mfurnish` at 3 ms with `mmesh=0` is a bug;
/// `mfurnish` at 3 ms with `mmesh=1400` is the paced budget working as designed.
///
/// The caller **takes the resource**, which is safe for a reason worth stating rather than assuming: the
/// counters are per-frame by contract and `perf::trace` is what clears them — and `perf` is `dev`,
/// so on the web nothing has ever cleared them and nothing has ever read them (`any_event` has no
/// caller outside that module). Taking them per report is therefore not stealing a reading from
/// another consumer; it is giving them their first one on this target.
fn stream_line(a: benilla_world::terrain_stream::StreamActivity, frames: usize) -> Option<String> {
    let per = |ms: f32| if frames > 0 { ms / frames as f32 } else { 0.0 };
    let busy = a.stream_ms + a.furnish_ms + a.mfurnish_ms + a.spawn_ms + a.collider_ms;
    // A settled world runs every one of these systems every frame and spends ~nothing in them;
    // a line of zeroes each report would bury the windows that matter.
    if busy < 0.05 && !a.any_event() {
        return None;
    }
    Some(format!(
        "
[diag]   stream ms/f: furnish={:.2} mfurnish={:.2} spawn={:.2} collider={:.2}          stream={:.2} prov={:.2} | cells={} mmesh={} placed={} attached={} tiles={} prov_live={}",
        per(a.furnish_ms),
        per(a.mfurnish_ms),
        per(a.spawn_ms),
        per(a.collider_ms),
        per(a.stream_ms),
        per(a.provisional_ms),
        a.cells_spawned,
        a.model_meshes_built,
        a.placements_spawned,
        a.colliders_attached,
        a.tiles_spawned,
        a.provisional_live,
    ))
}

/// Which output the mix is playing through, and whether it is keeping up.
///
/// Three facts, because they fail independently and sound identical:
///
/// - **`worklet` vs `cpal`** — did `sound::web_output` get the `AudioWorklet` (drain on the
///   browser's audio thread, fill on a Web Worker, neither of them the thread that draws), or did
///   it fall back to kira's cpal backend, whose `ScriptProcessorNode` callback runs on the main
///   thread with the mix inside it? On `cpal` every frame hitch is audible by construction, and a
///   report of stuttering sound means something completely different than it does on `worklet`.
/// - **`ctx=`** — `AudioContext.state`. `suspended` is a tab still waiting on a user gesture, which
///   is silence with no fault at all.
/// - **`under=`** — frames the worklet had to invent because the ring was empty, since boot. The
///   one audible failure this design leaves, counted rather than guessed. It should stay at 0; a
///   number that climbs means the render worker is not keeping ahead.
#[cfg(target_arch = "wasm32")]
fn audio_field() -> String {
    format!(
        "{} ctx={} under={}",
        if crate::sound::web_output::on_worklet() {
            "worklet"
        } else {
            "cpal"
        },
        crate::sound::web_output::context_state(),
        crate::sound::web_output::underruns(),
    )
}

/// Native has its own owned output layer and its own meters (`sound::output`); nothing here.
#[cfg(not(target_arch = "wasm32"))]
fn audio_field() -> String {
    String::new()
}

/// `<grows>+<ahead>/<MB>`: wasm memory grows this window, how many of them took the allocator's
/// grow step rather than only the need, and the MB added (`wasm_alloc::take_grow_stats`). A stall
/// in a window that grew may be the allocator's: each grow runs under its span pool's lock.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    not(debug_assertions)
))]
fn grow_field() -> String {
    let (grows, ahead, bytes) = crate::wasm_alloc::take_grow_stats();
    format!("{grows}+{ahead}/{}MB", bytes >> 20)
}

/// No sharded allocator here (native, or a debug or single-threaded web build).
#[cfg(not(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    not(debug_assertions)
)))]
fn grow_field() -> String {
    "-".to_owned()
}

/// `+new/pending` for the pipeline cache this window, or `-` where there is no cache to ask
/// (a headless run, or a render app that never came up).
///
/// A non-zero `+` here beside a `render+present`-dominated slow frame is the whole diagnosis: the
/// pipeline was compiled synchronously on the drawing thread because the warm pass did not already
/// own that variant. `note_live_pipeline` names which one.
fn pipes_field(d: &mut Diag, pipes: Option<(usize, usize)>) -> String {
    let Some((created, settled)) = pipes else {
        return "-".to_string();
    };
    let new = created.saturating_sub(d.prev_pipes);
    d.prev_pipes = created;
    format!("+{new}/{}", created.saturating_sub(settled))
}

/// The warm pass's outcome — drained, or timed out with pipelines pending — on the console the
/// web build has. Its `info!`/`warn!` are silenced there while this probe is armed, and a timeout
/// is what sends the rest of the warm set to compile live, in play.
pub(crate) fn note_warm(what: &str) {
    if armed() {
        out(&format!("[diag] pipeline warm: {what}"));
    }
}

/// A pipeline compiled **live** — say so on the console the web build actually has.
///
/// `pipe_warm`'s tripwire is a `warn!`, and on this target that goes nowhere: `quiet_console_on_web`
/// takes `LogPlugin` off whenever this probe is armed, so the console carries the diag and nothing
/// else. Which means the single most explanatory line the client can emit — the one that says a
/// frame just spent 300 ms compiling a shader variant the warm pass missed — was silenced on
/// exactly the build where the stall is worst.
///
/// **Worst there for a reason that is not this module's:** `bevy_render`'s `create_pipeline_task`
/// has a `target_arch = "wasm32"` arm that `block_on`s the build unconditionally, so on the web a
/// cold variant is a synchronous compile on the thread that draws. That is not a stall the app can
/// pace; it is only one it can move behind the loading cover, which is what the menagerie is for.
/// So a line here is not a curiosity — it names a variant the warm pass has to grow to cover.
///
/// Capped per session. A coverage hole is usually a whole family of variants, and the first
/// [`LIVE_PIPE_CAP`] name it as well as five hundred would.
pub(crate) fn note_live_pipeline(id: usize, what: &str) {
    if !armed() {
        return;
    }
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEEN: AtomicUsize = AtomicUsize::new(0);
    let n = SEEN.fetch_add(1, Ordering::Relaxed);
    if n < LIVE_PIPE_CAP {
        out(&format!("[diag] pipeline compiled LIVE [{id}] {what}"));
    } else if n == LIVE_PIPE_CAP {
        out("[diag] pipeline compiled LIVE — further ones silenced this session.");
    }
}

/// Worker threads on the two pools offloaded work reaches, as `async/compute`.
///
/// **The one fact about a threaded build a pasted report could not previously state.** `0/0` is
/// an ordinary, supported outcome — a page that ships no thread bridge, an origin that is not
/// cross-origin isolated, or `?nothreads=1` — and on it every offload in this client falls back to
/// the frame thread (`benilla_assets::frame_budget::pooled_work_runs`). So a frame time being
/// explained by "the browser has threads now" is only explicable next to this number, and it goes
/// on every line rather than on the armed line alone because a report is pasted five lines at a
/// time and a constant nobody can see is a constant nobody can check.
///
/// `AsyncComputeTaskPool` is where `spawn_pooled` work goes; `ComputeTaskPool` is what the
/// `par_iter_mut` sweeps run their scopes on. `IoTaskPool` is deliberately absent — it is sized to
/// zero workers on this target on purpose (`benilla_world::boot`), so printing it would only
/// invite the reading that something is broken.
///
/// `offl=` is the decodes on the pool right now (`benilla_assets::frame_budget::offloads_in_flight`),
/// capped at four per async worker — 32 at `8/…`. Pinned at the cap beside a climbing `wasm=`
/// is a burst being held back as designed; climbing `wasm=` with `offl=` low is a leak elsewhere.
fn worker_threads() -> String {
    let async_compute = bevy::tasks::AsyncComputeTaskPool::try_get().map_or(0, |p| p.thread_num());
    let compute = bevy::tasks::ComputeTaskPool::try_get().map_or(0, |p| p.thread_num());
    let offl = benilla_assets::frame_budget::offloads_in_flight();
    format!("{async_compute}/{compute} offl={offl}")
}

fn signed(v: i64) -> String {
    if v == 0 {
        String::new()
    } else {
        format!("({v:+})")
    }
}

fn mb(bytes: u64) -> String {
    format!("{:.0}MB", bytes as f64 / (1024.0 * 1024.0))
}

fn signed_mb(bytes: i64) -> String {
    if bytes.abs() < 512 * 1024 {
        String::new()
    } else {
        format!("({:+.0}MB)", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// How many blocking chain reads a report names.
const TOP_BLOCKING: usize = 6;

/// This window's blocking chain reads: `(name, reads, ms)`, costliest first.
#[cfg(target_arch = "wasm32")]
fn blocking_reads() -> Vec<(String, u32, f32)> {
    benilla_formats::webpack::sync_meter::take()
}

#[cfg(not(target_arch = "wasm32"))]
fn blocking_reads() -> Vec<(String, u32, f32)> {
    Vec::new()
}

/// This window's `localStorage` writes: `(file, writes, bytes, ms)`, costliest first.
#[cfg(target_arch = "wasm32")]
fn storage_writes() -> Vec<(String, u32, usize, f32)> {
    crate::local_state::write_meter::take()
}

#[cfg(not(target_arch = "wasm32"))]
fn storage_writes() -> Vec<(String, u32, usize, f32)> {
    Vec::new()
}

/// wasm linear memory in bytes — `memory.size` is in 64 KiB pages. This is the allocator's whole
/// arena, so it never shrinks; what matters is whether it keeps *climbing*.
#[cfg(target_arch = "wasm32")]
fn wasm_bytes() -> u64 {
    (core::arch::wasm32::memory_size(0) as u64) * 64 * 1024
}

#[cfg(not(target_arch = "wasm32"))]
fn wasm_bytes() -> u64 {
    0
}

/// `performance.memory.usedJSHeapSize` — non-standard and Chromium-only, so it is read through
/// `Reflect` and answers 0 where it does not exist rather than needing a `web-sys` feature for a
/// property half the browsers do not have.
#[cfg(target_arch = "wasm32")]
fn js_heap() -> u64 {
    fn get(v: &wasm_bindgen::JsValue, key: &str) -> Option<wasm_bindgen::JsValue> {
        js_sys::Reflect::get(v, &key.into())
            .ok()
            .filter(|v| !v.is_undefined() && !v.is_null())
    }
    fn read() -> Option<u64> {
        let window = web_sys::window()?;
        let perf = get(window.as_ref(), "performance")?;
        let mem = get(&perf, "memory")?;
        Some(get(&mem, "usedJSHeapSize")?.as_f64()? as u64)
    }
    read().unwrap_or(0)
}

#[cfg(not(target_arch = "wasm32"))]
fn js_heap() -> u64 {
    0
}

/// The report's only way out. `console.log` directly rather than `info!`, so the probe still
/// speaks on a build whose `LogPlugin` has been taken off to quiet the console.
#[cfg(target_arch = "wasm32")]
pub(crate) fn out(s: &str) {
    use wasm_bindgen::prelude::wasm_bindgen;
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        fn log(s: &str);
    }
    log(s);
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn out(s: &str) {
    println!("{s}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe is web-only: a native run must register nothing, or every native capture grows a
    /// set of exclusive sync points inside `Update` that its own numbers are measured without.
    #[test]
    fn the_probe_is_web_only() {
        assert!(!armed(), "webdiag must stay off on native");
    }

    /// A row names its population without printing a paragraph, and generics never widen it.
    #[test]
    fn component_names_shorten_to_their_last_two_segments() {
        assert_eq!(
            short("bevy_render::mesh::components::Mesh3d"),
            "components::Mesh3d"
        );
        assert_eq!(short("benilla_app::names::NameCache"), "names::NameCache");
        assert_eq!(short("bevy_asset::handle::Handle<Mesh>"), "handle::Handle");
        assert_eq!(short("Bare"), "Bare");
    }

    /// A delta of zero prints as nothing — a line of `(+0)` per column is noise that hides the one
    /// column that moved.
    #[test]
    fn only_real_moves_are_printed() {
        assert_eq!(signed(0), "");
        assert_eq!(signed(-4), "(-4)");
        assert_eq!(signed(4), "(+4)");
        // Sub-megabyte memory jitter is not a move worth a column either.
        assert_eq!(signed_mb(1024), "");
        assert_eq!(signed_mb(4 * 1024 * 1024), "(+4MB)");
    }
}
