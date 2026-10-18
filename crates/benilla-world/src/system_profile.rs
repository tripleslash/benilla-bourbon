//! fork: perf — per-system CPU time from Bevy's own system spans (the `trace_systems` feature,
//! which turns on `bevy/trace`). A tracing layer sums each `system` span's entered time by the
//! system's name; with `WOW_SYSTEM_PROFILE=1` a `Last` system prints the top systems every three
//! seconds as `SYSTEM_PROFILE <ms per second> <calls per second> <name>`, main and render world
//! alike. Never in a shipped build.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use bevy::log::tracing::span::{Attributes, Id};
use bevy::log::tracing::{field, Subscriber};
use bevy::log::tracing_subscriber::layer::Context;
use bevy::log::tracing_subscriber::registry::LookupSpan;
use bevy::log::tracing_subscriber::Layer;
use bevy::prelude::*;

/// Total time and runs per system name since the last report.
static TOTALS: Mutex<Option<HashMap<String, (Duration, u32)>>> = Mutex::new(None);

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
        let took = entered.0.elapsed();
        let mut totals = TOTALS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let e = totals
            .get_or_insert_with(HashMap::new)
            .entry(name.0.clone())
            .or_default();
        e.0 += took;
        e.1 += 1;
    }
}

/// Every three seconds, the top systems by CPU time.
fn report(mut last: Local<Option<Instant>>) {
    let now = Instant::now();
    let since = *last.get_or_insert(now);
    let secs = (now - since).as_secs_f32();
    if secs < 3.0 {
        return;
    }
    *last = Some(now);
    let Some(totals) = TOTALS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
    else {
        return;
    };
    let mut rows: Vec<(String, Duration, u32)> =
        totals.into_iter().map(|(n, (d, c))| (n, d, c)).collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    let all: f32 = rows.iter().map(|r| r.1.as_secs_f32()).sum::<f32>() * 1000.0 / secs;
    info!("SYSTEM_PROFILE total={all:.1}ms/s (nested spans double count) over {secs:.1}s");
    // `WOW_SYSTEM_PROFILE_TOP=<n>`: how many rows (45); the leaves sit under their schedules' spans.
    let top = crate::launch_env::var("WOW_SYSTEM_PROFILE_TOP")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(45);
    for (name, d, c) in rows.iter().take(top) {
        info!(
            "SYSTEM_PROFILE {:8.2}ms/s {:6.1}/s {name}",
            d.as_secs_f32() * 1000.0 / secs,
            *c as f32 / secs
        );
    }
}

pub fn plugin(app: &mut App) {
    if crate::launch_env::var("WOW_SYSTEM_PROFILE").is_some_and(|v| v == "1") {
        app.add_systems(Last, report);
    }
}
