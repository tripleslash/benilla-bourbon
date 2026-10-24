//! fork: perf — per-system CPU time from Bevy's own system spans (the `trace_systems` feature,
//! which turns on `bevy/trace` and `bevy/debug` for the names). A tracing layer sums each `system`
//! span's entered time by the system's name, and a `Last` system prints the top systems every three
//! seconds as `SYSTEM_PROFILE <ms per second> <calls per second> <max ms> <name>`, main and render
//! world alike, then every single run over `WOW_SYSTEM_PROFILE_SPIKE` ms (8) as `SYSTEM_SPIKE`,
//! with when it ended on the wall clock and whether it ran on the app's thread. Always on in a
//! build with the feature, and never in a shipped one: the feature is opt-in on every target
//! (`wenilla`'s `trace_systems` for the browser build).
//!
//! The browser's page thread may not block (`Atomics.wait` traps there), so the tallies are taken
//! with a spinning `try_lock`. The web build has no `LogPlugin`, and `tracing_subscriber`'s registry
//! keeps each span's data behind a `RwLock` that a system span entered from the page and a worker
//! at once contends, so there [`NonBlocking`] is the whole subscriber: span names interned into
//! ids, enter times on a thread-local stack, and the report through [`set_web_hooks`]' printer.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock, TryLockError};
use std::time::Duration;

use bevy::log::tracing::span::{Attributes, Id, Record};
use bevy::log::tracing::{field, Event, Metadata, Subscriber};
use bevy::log::tracing_subscriber::layer::Context;
use bevy::log::tracing_subscriber::registry::LookupSpan;
use bevy::log::tracing_subscriber::Layer;
use bevy::platform::time::Instant;
use bevy::prelude::*;

/// One name's runs since the last report.
#[derive(Default)]
struct Tally {
    total: Duration,
    runs: u32,
    longest: Duration,
}

/// One run over the spike threshold: when it ended (wall-clock ms, the one clock every thread
/// shares), how long, which span, and whether it ran on the thread that built the app (the page's).
struct Spike {
    at_ms: f64,
    took: Duration,
    name: String,
    page: bool,
}

static TOTALS: Mutex<Option<HashMap<String, Tally>>> = Mutex::new(None);
static SPIKES: Mutex<Vec<Spike>> = Mutex::new(Vec::new());
/// The spike threshold, set once by [`plugin`].
static SPIKE_AT: OnceLock<Duration> = OnceLock::new();
/// Spikes kept between two reports; a storm past it is counted, not stored.
const SPIKES_KEPT: usize = 400;

/// The thread [`plugin`] ran on: the app's own, the page's on the web.
static APP_THREAD: OnceLock<std::thread::ThreadId> = OnceLock::new();

/// fork: web — the printer (`console.log`) and the wall clock (`Date.now`), which live where
/// `web-sys` does. A worker's `performance.now()` counts from its own start, so only the wall clock
/// puts a worker's spike and a page frame on one axis.
static PRINT: OnceLock<fn(&str)> = OnceLock::new();
static WALL_MS: OnceLock<fn() -> f64> = OnceLock::new();

/// fork: web — install the printer and the wall clock; the first call wins.
pub fn set_web_hooks(print: fn(&str), wall_ms: fn() -> f64) {
    let _ = PRINT.set(print);
    let _ = WALL_MS.set(wall_ms);
}

/// Wall-clock ms: the hook where installed, else the system clock.
fn wall_ms() -> f64 {
    match WALL_MS.get() {
        Some(f) => f(),
        None => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64() * 1000.0),
    }
}

/// A lock that never parks: the page thread may not.
fn spin_lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    loop {
        match m.try_lock() {
            Ok(g) => return g,
            Err(TryLockError::Poisoned(p)) => return p.into_inner(),
            Err(TryLockError::WouldBlock) => std::hint::spin_loop(),
        }
    }
}

/// The layer: tags each `system` span with its name, times every entry.
pub struct SystemProfile;

struct Name(String);
struct Entered(Instant);

struct NameVisitor(Option<String>);

impl field::Visit for NameVisitor {
    fn record_str(&mut self, f: &field::Field, value: &str) {
        if matches!(f.name(), "name" | "camera" | "system") {
            self.0 = Some(value.to_owned());
        }
    }
    fn record_debug(&mut self, f: &field::Field, value: &dyn std::fmt::Debug) {
        if matches!(f.name(), "name" | "camera" | "system") {
            self.0 = Some(format!("{value:?}").trim_matches('"').to_owned());
        }
    }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for SystemProfile {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        // A system by its name; any other span (`camera_schedule`, `queue_submit`,
        // `present_frames`, …) as `[span]`, with its camera or system where it has one, so the time a
        // driver spends outside its systems is attributed too. Nested spans count in both.
        let mut v = NameVisitor(None);
        attrs.record(&mut v);
        let span_name = attrs.metadata().name();
        let name = if span_name == "system" {
            v.0
        } else {
            Some(match v.0 {
                Some(camera) => format!("[span] {span_name} {camera}"),
                None => format!("[span] {span_name}"),
            })
        };
        if let (Some(name), Some(span)) = (name, ctx.span(id)) {
            span.extensions_mut().insert(Name(name));
        }
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut ext = span.extensions_mut();
            if ext.get_mut::<Name>().is_some() {
                ext.replace(Entered(Instant::now()));
            }
        }
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        let ext = span.extensions();
        let (Some(name), Some(entered)) = (ext.get::<Name>(), ext.get::<Entered>()) else {
            return;
        };
        record(&name.0, entered.0, Instant::now());
    }
}

/// Tally one run of `name` from `entered` to `now`, and keep it as a spike if it is one.
fn record(name: &str, entered: Instant, now: Instant) {
    let took = now.saturating_duration_since(entered);
    {
        // A name seen before is looked up by `&str`: allocating it per run made the profiler a
        // heavy client of the allocator lock it was measuring.
        let mut totals = spin_lock(&TOTALS);
        let totals = totals.get_or_insert_with(HashMap::new);
        if !totals.contains_key(name) {
            totals.insert(name.to_owned(), Tally::default());
        }
        let t = totals.get_mut(name).expect("inserted above");
        t.total += took;
        t.runs += 1;
        t.longest = t.longest.max(took);
    }
    if SPIKE_AT.get().is_some_and(|at| took >= *at) {
        let mut spikes = spin_lock(&SPIKES);
        if spikes.len() < SPIKES_KEPT {
            spikes.push(Spike {
                at_ms: wall_ms(),
                took,
                name: name.to_owned(),
                page: APP_THREAD.get() == Some(&std::thread::current().id()),
            });
        }
    }
}

/// The names in id order, and each name's id.
type Interned = (Vec<String>, Option<HashMap<String, u64>>);

/// The span names [`NonBlocking`] has seen, in id order: a span's id is its name's index + 1, so
/// two live spans of one name share an id, which timing by name never has to tell apart.
static NAMES: Mutex<Interned> = Mutex::new((Vec::new(), None));

thread_local! {
    /// The spans entered on this thread and not yet exited, innermost last.
    static ENTERED: std::cell::RefCell<Vec<(u64, Instant)>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

/// fork: web — the whole subscriber, for a build with no `LogPlugin`: nothing in it parks.
pub struct NonBlocking;

impl Subscriber for NonBlocking {
    fn enabled(&self, meta: &Metadata<'_>) -> bool {
        meta.is_span() && *meta.level() <= bevy::log::Level::INFO
    }

    fn max_level_hint(&self) -> Option<bevy::log::tracing::level_filters::LevelFilter> {
        Some(bevy::log::tracing::level_filters::LevelFilter::INFO)
    }

    fn new_span(&self, attrs: &Attributes<'_>) -> Id {
        let mut v = NameVisitor(None);
        attrs.record(&mut v);
        let span_name = attrs.metadata().name();
        let name = match (span_name, v.0) {
            ("system", Some(system)) => system,
            (_, Some(camera)) => format!("[span] {span_name} {camera}"),
            (_, None) => format!("[span] {span_name}"),
        };
        let mut names = spin_lock(&NAMES);
        let (list, index) = &mut *names;
        let index = index.get_or_insert_with(HashMap::new);
        let id = *index.entry(name).or_insert_with_key(|name| {
            list.push(name.clone());
            list.len() as u64
        });
        Id::from_u64(id)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, _: &Event<'_>) {}

    fn enter(&self, id: &Id) {
        let now = Instant::now();
        ENTERED.with(|e| e.borrow_mut().push((id.into_u64(), now)));
    }

    fn exit(&self, id: &Id) {
        let now = Instant::now();
        let id = id.into_u64();
        let Some(entered) = ENTERED.with(|e| {
            let mut e = e.borrow_mut();
            let at = e.iter().rposition(|(i, _)| *i == id)?;
            Some(e.remove(at).1)
        }) else {
            return;
        };
        // Recorded under `NAMES` rather than with a cloned name; nothing takes `NAMES` while
        // holding `TOTALS` or `SPIKES`, so the order cannot invert.
        let names = spin_lock(&NAMES);
        record(
            names.0.get(id as usize - 1).map_or("", String::as_str),
            entered,
            now,
        );
    }
}

/// One line out: the web printer when installed, else the log.
fn emit(line: &str) {
    match PRINT.get() {
        Some(print) => print(line),
        None => info!("{line}"),
    }
}

/// Every three seconds, the top systems by CPU time, then the spikes since the last report.
fn report(mut last: Local<Option<Instant>>) {
    let now = Instant::now();
    let since = *last.get_or_insert(now);
    let secs = now.saturating_duration_since(since).as_secs_f32();
    if secs < 3.0 {
        return;
    }
    *last = Some(now);
    let totals = spin_lock(&TOTALS).take();
    let spikes = std::mem::take(&mut *spin_lock(&SPIKES));
    let mut out = format!("SYSTEM_PROFILE wall={:.0}ms over {secs:.1}s", wall_ms());
    if let Some(totals) = totals {
        let mut rows: Vec<(String, Tally)> = totals.into_iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1.total));
        let all: f32 = rows.iter().map(|r| r.1.total.as_secs_f32()).sum::<f32>() * 1000.0 / secs;
        out.push_str(&format!(" total={all:.1}ms/s (nested spans double count)"));
        // `WOW_SYSTEM_PROFILE_TOP=<n>`: how many rows (45); the leaves sit under their schedules'
        // spans.
        let top = crate::launch_env::var("WOW_SYSTEM_PROFILE_TOP")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(45);
        for (name, t) in rows.iter().take(top) {
            out.push_str(&format!(
                "\nSYSTEM_PROFILE {:8.2}ms/s {:6.1}/s max={:8.2}ms {name}",
                t.total.as_secs_f32() * 1000.0 / secs,
                t.runs as f32 / secs,
                t.longest.as_secs_f32() * 1000.0,
            ));
        }
    }
    for s in &spikes {
        out.push_str(&format!(
            "\nSYSTEM_SPIKE wall_end={:.0}ms took={:.1}ms {} {}",
            s.at_ms,
            s.took.as_secs_f32() * 1000.0,
            if s.page { "app" } else { "worker" },
            s.name
        ));
    }
    emit(&out);
}

/// The threshold, the app's thread and, on the web, the subscriber; the first call wins. Call it
/// before any system is added: a system's span is made once, when the system is built, and one
/// made before the subscriber exists records nothing.
pub fn install() {
    let spike_ms: u64 = crate::launch_env::var("WOW_SYSTEM_PROFILE_SPIKE")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(8);
    let _ = SPIKE_AT.set(Duration::from_millis(spike_ms));
    let _ = APP_THREAD.set(std::thread::current().id());
    // fork: web — no `LogPlugin` there to carry the layer (`boot`'s `custom_layer`), so it gets a
    // subscriber of its own. Where one is already installed this does nothing.
    #[cfg(target_arch = "wasm32")]
    let _ = bevy::log::tracing::subscriber::set_global_default(NonBlocking);
}

pub fn plugin(app: &mut App) {
    install();
    app.add_systems(Last, report);
}
