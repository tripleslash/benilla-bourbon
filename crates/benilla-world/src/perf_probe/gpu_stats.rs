//! `WOW_GPU_STATS=1`: per-pass pipeline statistics, the counts beside `WOW_GPU_PASSES`' times,
//! read from the same brackets.
//!
//! Every `pass_span` of bevy's render diagnostics, a render or compute pass, opens a
//! pipeline-statistics query beside its timestamps where the device has
//! `PIPELINE_STATISTICS_QUERY` (Vulkan; DX12 through `third_party/wgpu-hal`'s fourth patch). The
//! recorder resolves both into one buffer per frame and maps it with `map_async`, from a pool of
//! such buffers it recycles once a mapping lands, so nothing waits on the GPU; the readings arrive
//! in the [`DiagnosticsStore`] a few frames late as `render/<pass>/<counter>`. This module folds
//! them, the way the journal folds `elapsed_gpu`.
//!
//! Five counters a pass, each per frame read ([`COLUMNS`]):
//! - `ia_prims`: primitives assembled, read as `clipper_invocations`. wgpu has no geometry or
//!   tessellation stage, so every primitive the input assembler emits enters the clipper, and its
//!   types carry no input-assembly counter of their own.
//! - `vs_inv`: vertex-shader invocations (after the post-transform cache).
//! - `clip_prims`: primitives out of the clipper, the ones rasterized.
//! - `fs_inv`: fragment-shader invocations, helper lanes of a 2×2 quad included.
//! - `cs_inv`: compute-shader invocations, workgroups × workgroup size.
//!
//! A render pass reports the first four and a compute pass the last. Counts do not depend on the
//! queue or the clock, so they compare across DX12 and Vulkan at one camera.
//!
//! **Lines.** `GPU_STATS` every `WOW_GPU_STATS_EVERY` frames (default 300): the frame's totals
//! over every pass, then each pass, most work first. `GPU_STATS_RUNS` follows for the passes that
//! ran more than once a frame, one entry per run in the order its spans closed: graph order for a
//! pass recorded serially, so the per-view passes read the water mirror (camera order −1) as `#0`
//! and the world camera as the last; arbitrary for jobs recorded in parallel (the GI voxelizer's),
//! whose sum is the number to read. The FPS journal, `PERF_PROBE` and `FPS_PROBE` carry the five
//! totals as columns.
//!
//! **Cost, and what the lever changes.** Without `WOW_GPU_STATS` the device is created without
//! the feature ([`disabled_features`]), so no build and no other instrument pays for the queries.
//! With it the modern compute stays on the main queue (`modern::async_compute`): the recorder's
//! query set counts graphics stages, which a query on a compute-only Vulkan queue family may not
//! (`VUID-vkCmdBeginQuery-queryType-00805`). The counts are the same on either queue; the timings
//! of a stats run are not the ones to quote.

use std::collections::BTreeMap;

use bevy::diagnostic::DiagnosticsStore;
use bevy::platform::time::Instant;
use bevy::prelude::*;

/// The five counters in column order, as this module names them.
pub const COLUMNS: [&str; 5] = ["ia_prims", "vs_inv", "clip_prims", "fs_inv", "cs_inv"];

/// The diagnostics field each of [`COLUMNS`] is read from (bevy's `RenderDiagnosticsPlugin`).
const COUNTERS: [&str; 5] = [
    "clipper_invocations",
    "vertex_shader_invocations",
    "clipper_primitives_out",
    "fragment_shader_invocations",
    "compute_shader_invocations",
];

/// `WOW_GPU_STATS=1`, read once.
pub fn armed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::launch_env::var("WOW_GPU_STATS").as_deref() == Some("1"))
}

/// The features the device is created without: pipeline statistics, unless [`armed`]. Bevy asks
/// for every feature the adapter has, and its render diagnostics then query every pass.
pub fn disabled_features() -> Option<wgpu::Features> {
    (!armed()).then_some(wgpu::Features::PIPELINE_STATISTICS_QUERY)
}

/// `WOW_GPU_STATS_EVERY`: frames between `GPU_STATS` lines, default 300.
fn every() -> u32 {
    crate::launch_env::var("WOW_GPU_STATS_EVERY")
        .and_then(|v| v.trim().parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(300)
}

/// A window's pipeline statistics, summed per pass and divided by the frames whose readback
/// landed. Folds only readings newer than the last fold.
#[derive(Default)]
pub struct GpuStats {
    /// Per pass path (below `render/`), the five counters summed over the window.
    sums: BTreeMap<String, [f64; 5]>,
    /// Per pass path, the same split by the run's order within its frame.
    runs: BTreeMap<String, Vec<[f64; 5]>>,
    frames: u32,
    seen: Option<Instant>,
}

impl GpuStats {
    /// A window that reads only what lands after `at`: the store may hold a history.
    pub fn since(at: Instant) -> Self {
        Self {
            seen: Some(at),
            ..Self::default()
        }
    }

    /// Fold in every counter reading newer than the last fold.
    pub fn fold(&mut self, store: &DiagnosticsStore) {
        let seen = self.seen;
        let mut newest = seen;
        let mut times: Vec<Instant> = Vec::new();
        for diagnostic in store.iter() {
            let Some((pass, counter)) = diagnostic
                .path()
                .as_str()
                .strip_prefix("render/")
                .and_then(|p| p.rsplit_once('/'))
            else {
                continue;
            };
            let Some(col) = COUNTERS.iter().position(|c| *c == counter) else {
                continue;
            };
            let mut sum = [0.0; 5];
            let mut runs: Vec<f64> = Vec::new();
            // One frame's readings share an instant (`sync_diagnostics`), in span order.
            let (mut at, mut nth) = (None, 0);
            for m in diagnostic
                .measurements()
                .filter(|m| seen.is_none_or(|s| m.time > s))
            {
                sum[col] += m.value;
                nth = if at == Some(m.time) { nth + 1 } else { 0 };
                at = Some(m.time);
                if runs.len() <= nth {
                    runs.resize(nth + 1, 0.0);
                }
                runs[nth] += m.value;
                if !times.contains(&m.time) {
                    times.push(m.time);
                }
                if newest.is_none_or(|n| m.time > n) {
                    newest = Some(m.time);
                }
            }
            if at.is_none() {
                continue;
            }
            let row = self.sums.entry(pass.to_string()).or_default();
            row[col] += sum[col];
            let split = self.runs.entry(pass.to_string()).or_default();
            if split.len() < runs.len() {
                split.resize(runs.len(), [0.0; 5]);
            }
            for (slot, v) in split.iter_mut().zip(runs) {
                slot[col] += v;
            }
        }
        self.frames += times.len() as u32;
        self.seen = newest;
    }

    /// Start a new window at what has been folded so far.
    pub fn clear(&mut self) {
        self.sums.clear();
        self.runs.clear();
        self.frames = 0;
    }

    /// Fold, then drop what was folded: the next window starts clean.
    pub fn mark(&mut self, store: &DiagnosticsStore) {
        self.fold(store);
        self.clear();
    }

    /// Frames whose counters landed in this window.
    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// The five counters per frame, summed over every pass; `None` when nothing landed. Only a
    /// pass span carries counters, so a pass nested in a time span is not counted twice.
    pub fn totals(&self) -> Option<[f64; 5]> {
        if self.frames == 0 {
            return None;
        }
        let n = f64::from(self.frames);
        let mut t = [0.0; 5];
        for row in self.sums.values() {
            for (t, v) in t.iter_mut().zip(row) {
                *t += v / n;
            }
        }
        Some(t)
    }

    /// The totals as CSV cells, `,ia,vs,clip,fs,cs`; five empty cells when nothing landed.
    pub fn csv_cells(&self) -> String {
        match self.totals() {
            Some(t) => t.iter().map(|v| format!(",{v:.0}")).collect(),
            None => ",".repeat(COLUMNS.len()),
        }
    }

    /// The totals as probe-line columns, ` ia_prims=… cs_inv=…`; empty when nothing landed.
    pub fn probe_columns(&self) -> String {
        self.totals().map_or_else(String::new, |t| {
            COLUMNS
                .iter()
                .zip(t)
                .map(|(k, v)| format!(" {k}={v:.0}"))
                .collect()
        })
    }

    /// The `GPU_STATS` line's body: totals, then `pass=ia/vs/clip/fs/cs` per frame, most shader
    /// work first.
    pub fn line(&self) -> Option<String> {
        let totals = self.totals()?;
        let n = f64::from(self.frames);
        let mut rows: Vec<(&String, [f64; 5])> = self
            .sums
            .iter()
            .map(|(k, v)| (k, v.map(|x| x / n)))
            .collect();
        rows.sort_by(|a, b| work(&b.1).total_cmp(&work(&a.1)).then(a.0.cmp(b.0)));
        let mut s = format!(
            "frames={} fields={} total={}",
            self.frames,
            COLUMNS.join("/"),
            slashed(&totals)
        );
        for (k, v) in rows {
            s.push_str(&format!(" {k}={}", slashed(&v)));
        }
        Some(s)
    }

    /// The `GPU_STATS_RUNS` line's body: each pass that ran more than once a frame, one
    /// `pass#run=…` per run, per frame.
    pub fn runs_line(&self) -> Option<String> {
        if self.frames == 0 {
            return None;
        }
        let n = f64::from(self.frames);
        let each: Vec<String> = self
            .runs
            .iter()
            .filter(|(_, v)| v.len() > 1)
            .flat_map(|(k, v)| {
                v.iter()
                    .enumerate()
                    .map(move |(i, r)| format!("{k}#{i}={}", slashed(&r.map(|x| x / n))))
            })
            .collect();
        (!each.is_empty()).then(|| format!("frames={} {}", self.frames, each.join(" ")))
    }
}

/// The ordering key: the shader invocations a pass costs.
fn work(v: &[f64; 5]) -> f64 {
    v[1] + v[3] + v[4]
}

fn slashed(v: &[f64; 5]) -> String {
    v.map(|x| format!("{x:.0}")).join("/")
}

/// The `GPU_STATS` printer's state.
#[derive(Resource)]
struct StatsPrint {
    every: u32,
    frame: u32,
    stats: GpuStats,
    /// The device's answer is logged once, so a run without the feature says why it is silent.
    told: bool,
}

/// `Last`: fold the frame's readings; every [`every`] frames print and start a new window.
fn print_stats(
    mut print: ResMut<StatsPrint>,
    store: Option<Res<DiagnosticsStore>>,
    device: Option<Res<bevy::render::renderer::RenderDevice>>,
) {
    let print = &mut *print;
    if !print.told {
        if let Some(device) = device {
            print.told = true;
            if device
                .features()
                .contains(wgpu::Features::PIPELINE_STATISTICS_QUERY)
            {
                info!("GPU_STATS: armed, a line every {} frames", print.every);
            } else {
                warn!("GPU_STATS: the device has no PIPELINE_STATISTICS_QUERY; no line will print");
            }
        }
    }
    let Some(store) = store else {
        return;
    };
    print.stats.fold(&store);
    print.frame += 1;
    if !print.frame.is_multiple_of(print.every) {
        return;
    }
    if let Some(line) = print.stats.line() {
        info!("GPU_STATS {line}");
    }
    if let Some(line) = print.stats.runs_line() {
        info!("GPU_STATS_RUNS {line}");
    }
    print.stats.clear();
}

/// Registers the printer when [`armed`]. The render diagnostics themselves are registered by the
/// client's journal, and by the world viewer under this lever or `WOW_GPU_PASSES`.
pub(crate) fn plugin(app: &mut App) {
    if !armed() {
        return;
    }
    app.insert_resource(StatsPrint {
        every: every(),
        frame: 0,
        stats: GpuStats::since(Instant::now()),
        told: false,
    })
    .add_systems(Last, print_stats);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::diagnostic::{Diagnostic, DiagnosticMeasurement, DiagnosticPath};
    use std::time::Duration;

    fn push(store: &mut DiagnosticsStore, path: &str, time: Instant, value: f64) {
        let p = DiagnosticPath::new(path.to_string());
        if store.get(&p).is_none() {
            store.add(Diagnostic::new(p.clone()));
        }
        store
            .get_mut(&p)
            .unwrap()
            .add_measurement(DiagnosticMeasurement { time, value });
    }

    #[test]
    fn a_window_sums_each_pass_per_frame_and_splits_its_runs() {
        let mut store = DiagnosticsStore::default();
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_millis(10);
        // Two frames; the opaque pass runs twice a frame (the mirror, then the world camera).
        for t in [t0, t1] {
            let p = "render/main_opaque_pass_3d/";
            push(&mut store, &format!("{p}clipper_invocations"), t, 10.0);
            push(&mut store, &format!("{p}clipper_invocations"), t, 30.0);
            push(
                &mut store,
                &format!("{p}vertex_shader_invocations"),
                t,
                20.0,
            );
            push(
                &mut store,
                &format!("{p}vertex_shader_invocations"),
                t,
                60.0,
            );
            push(
                &mut store,
                &format!("{p}fragment_shader_invocations"),
                t,
                100.0,
            );
            push(
                &mut store,
                &format!("{p}fragment_shader_invocations"),
                t,
                300.0,
            );
            push(
                &mut store,
                "render/modern_gi_trace/compute_shader_invocations",
                t,
                64.0,
            );
            // A timing beside them, never a counter.
            push(&mut store, &format!("{p}elapsed_gpu"), t, 2.0);
        }
        let mut stats = GpuStats::default();
        stats.fold(&store);
        stats.fold(&store); // nothing new: a no-op
        assert_eq!(stats.frames(), 2);
        assert_eq!(stats.totals(), Some([40.0, 80.0, 0.0, 400.0, 64.0]));
        assert_eq!(stats.csv_cells(), ",40,80,0,400,64");
        assert_eq!(
            stats.probe_columns(),
            " ia_prims=40 vs_inv=80 clip_prims=0 fs_inv=400 cs_inv=64"
        );
        assert_eq!(
            stats.line().as_deref(),
            Some(
                "frames=2 fields=ia_prims/vs_inv/clip_prims/fs_inv/cs_inv total=40/80/0/400/64 \
                 main_opaque_pass_3d=40/80/0/400/0 modern_gi_trace=0/0/0/0/64"
            )
        );
        assert_eq!(
            stats.runs_line().as_deref(),
            Some(
                "frames=2 main_opaque_pass_3d#0=10/20/0/100/0 main_opaque_pass_3d#1=30/60/0/300/0"
            )
        );
        // A cleared window reads nothing until a later frame lands.
        stats.clear();
        assert_eq!(stats.totals(), None);
        assert_eq!(stats.csv_cells(), ",,,,,");
        assert_eq!(stats.probe_columns(), "");
        let t2 = t1 + Duration::from_millis(10);
        push(
            &mut store,
            "render/static_gx/fragment_shader_invocations",
            t2,
            7.0,
        );
        stats.fold(&store);
        assert_eq!(stats.totals(), Some([0.0, 0.0, 0.0, 7.0, 0.0]));
    }

    #[test]
    fn a_window_since_now_skips_the_history() {
        let mut store = DiagnosticsStore::default();
        let old = Instant::now();
        push(
            &mut store,
            "render/static_gx/fragment_shader_invocations",
            old,
            5.0,
        );
        let mut stats = GpuStats::since(old);
        stats.fold(&store);
        assert_eq!(stats.frames(), 0);
        assert_eq!(stats.line(), None);
    }

    #[test]
    fn the_counters_are_the_ones_bevy_reports() {
        // The recorder names each counter's diagnostics path; a rename there would leave a
        // column silently empty.
        let recorder =
            include_str!("../../../../third_party/bevy_render/src/diagnostic/internal.rs");
        for counter in COUNTERS {
            assert!(
                recorder.contains(&format!("\"{counter}\"")),
                "{counter} is not a field bevy's recorder reports"
            );
        }
    }
}
