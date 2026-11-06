//! The tuned Bevy boot: the `DefaultPlugins` set every benilla binary stands on. The engine tuning
//! is shared; the `Window` is the caller's.

use bevy::app::{PluginGroupBuilder, TaskPoolOptions, TaskPoolPlugin};
use bevy::prelude::*;

use crate::thread_qos;

/// `DefaultPlugins` with benilla's engine tuning applied, around the caller's primary window.
#[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))]
pub fn tuned_default_plugins(mut primary_window: Window) -> PluginGroupBuilder {
    // The web build's canvas binding: with no `canvas` selector winit creates its own `<canvas>`
    // and appends it to `<body>` at a fixed size, ignoring the full-viewport element
    // `web/index.html` already gives it — so bind to that one by id instead and let it own the
    // surface. `prevent_default_event_handling` stops the page from scrolling/zooming under our
    // own keyboard and touch/wheel input (the browser's defaults for those events).
    //
    // **`fit_canvas_to_parent` is OFF, and that is a decision, not an omission.** All the flag
    // does is write `width: 100%; height: 100%` onto the canvas's inline style once, at window
    // creation — after which every frame of a window drag is a `ResizeObserver` notification, a
    // `Resized` event, and a full re-allocation of the frame's memory: the swapchain, both
    // `Rgba16Float` view targets, the depth and prepass targets, and
    // `benilla_app::world_backdrop`'s backdrop image, which is retired and re-published by
    // design (decision 1647). At a 2560×1600 canvas that is hundreds of megabytes of
    // allocate-and-free per frame, and on an integrated GPU sharing system memory it is what the
    // browser reports back as **"Device Hung"** — a driver TDR, taking the WebGPU device and
    // every resource on it with it.
    //
    // The page owns the box instead (`web/gpu.js`): it commits a size only once a resize has
    // settled and stretches the gap with a CSS transform, so one drag re-allocates once rather
    // than sixty times, and the same transform carries the pixel budget a recovered device drops
    // to. Leaving the flag on would put Bevy's one-shot write *after* the governor's first
    // commit and take the box straight back. A page that fails to install the governor is still
    // whole: `web/index.html`'s stylesheet carries `100vw × 100vh` as the fallback, which is what
    // the flag was producing anyway.
    #[cfg(target_arch = "wasm32")]
    {
        primary_window.canvas = Some("#benilla".into());
        primary_window.fit_canvas_to_parent = false;
        primary_window.prevent_default_event_handling = true;
    }
    DefaultPlugins
        .set(WindowPlugin {
            primary_window: Some(primary_window),
            ..default()
        })
        // Deliberately no `AssetPlugin::file_path`: a baked source path resolves only on the build
        // machine. Every shader is embedded (`embedded://<crate>/shaders/…`), so no root is read.
        // Quiet wgpu/naga; the ring keeps the last stderr lines for the crash report (`log_ring`).
        .set(bevy::log::LogPlugin {
            filter: "wgpu=error,naga=warn".into(),
            custom_layer: |_| {
                // fork: perf — the per-system CPU profile rides beside the ring when built in.
                #[cfg(feature = "trace_systems")]
                {
                    use bevy::log::tracing_subscriber::Layer;
                    Some(Box::new(
                        crate::log_ring::LogRing.and_then(crate::system_profile::SystemProfile),
                    ))
                }
                #[cfg(not(feature = "trace_systems"))]
                Some(Box::new(crate::log_ring::LogRing))
            },
            ..default()
        })
        // Asset loads parse synchronously on the IO pool, so a dense teleport waits on its width:
        // IO takes half the cores, floor 2 and ceiling 8. Compute, where every frame's parallel
        // passes run, keeps the size the quarter split left it ([`pool_split`]), so the IO threads
        // ride on top, idle outside a load. Workers spawn at default QoS, behind any background
        // build: compute is user-interactive, IO and async compute user-initiated, and
        // `ThreadQosPlugin` promotes the render thread from inside.
        .set(TaskPoolPlugin {
            task_pool_options: task_pool_options(),
        })
        // Sound is kira behind our own mixer; `bevy_audio` is off by feature (workspace
        // `Cargo.toml`). Kept though they look idle: gizmos (bowstring, fishing line), sprites (the
        // FrameXML quad pass), picking, TextPlugin (glue text), PostProcessPlugin (glow bloom) and
        // ScenePlugin (avian's collider backend reads `SceneSpawner`); the ForwardDecal family is
        // registered inside `PbrPlugin::build`, so it cannot be disabled on its own.
        //
        // M2/WMO/ADT load through our own `mpq://` loaders; there is no glTF. `PbrPlugin`'s glTF
        // material hook needs `GltfPlugin`'s handler registry, so it goes with it.
        .disable::<bevy::gltf::GltfPlugin>()
        .set(bevy::pbr::PbrPlugin {
            gltf_enable_standard_materials: false,
            ..default()
        })
        // No bevy AA: no Fxaa/TAA/SMAA/CAS component anywhere (MSAA is core render, unaffected).
        .disable::<bevy::anti_alias::AntiAliasPlugin>()
        // No gamepad input; 1.12's bindings are keyboard/mouse.
        .disable::<bevy::gilrs::GilrsPlugin>()
}

/// The task-pool sizing, split out because the browser wants a different shape and the native
/// numbers are too hard-won to restate.
///
/// See the call site for what each native policy is for. **On the web the split changes**, and the
/// reason is that only two of the three pools have anything to run on a thread there:
///
/// - **IO gets no workers at all.** `TaskPool::spawn` on wasm is the browser's microtask queue, not
///   a worker — it has to be, because an `AssetReader` read awaits `JsFuture`s and reaches
///   `window` (`benilla_formats::webpack::resident`), and a `JsValue` cannot leave the agent that
///   made it. `bevy_asset` spawns every load that way, so IO worker threads would be Web Workers
///   that never receive a task: a module instance and a stack each, for nothing. The fetch/decode
///   work that CAN go off-thread already does, in `web/packworker.js`.
/// - **Async-compute and compute get the machine**, because they are reached through
///   `spawn_pooled` and `scope`, which do run on workers: the collider builds
///   (`terrain_stream::collider`) and the `par_iter_mut` sweeps (`model_render::visibility`,
///   `rig_anim::pose`).
///
/// The QoS hooks are left as they are: bevy ignores `on_thread_spawn` under
/// `all(target_arch = "wasm32", feature = "web")`, and Darwin QoS classes mean nothing to a Web
/// Worker anyway.
fn task_pool_options() -> TaskPoolOptions {
    // `WOW_THREADS=1` serialises the frame's systems, a diagnostic: a defect that survives it is
    // not a race between two systems. `WOW_COMPUTE_THREADS=<n>` pins the pool's size, a
    // measurement lever for the per-spawn wake cost.
    let serial = std::env::var("WOW_THREADS").ok().as_deref() == Some("1");
    let split = pool_split(
        bevy::tasks::available_parallelism(),
        if serial { Some(1) } else { compute_threads() },
    );
    #[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))]
    let mut options = TaskPoolOptions {
        io: bevy::app::TaskPoolThreadAssignmentPolicy {
            min_threads: split.io,
            max_threads: split.io,
            percent: 1.0,
            on_thread_spawn: Some(std::sync::Arc::new(|| {
                thread_qos::promote_current_thread(thread_qos::QosClass::UserInitiated)
            })),
            on_thread_destroy: None,
        },
        async_compute: bevy::app::TaskPoolThreadAssignmentPolicy {
            on_thread_spawn: Some(std::sync::Arc::new(|| {
                thread_qos::promote_current_thread(thread_qos::QosClass::UserInitiated)
            })),
            ..TaskPoolOptions::default().async_compute
        },
        compute: bevy::app::TaskPoolThreadAssignmentPolicy {
            on_thread_spawn: Some(std::sync::Arc::new(|| {
                thread_qos::promote_current_thread(thread_qos::QosClass::UserInteractive)
            })),
            min_threads: split.compute,
            max_threads: split.compute,
            ..TaskPoolOptions::default().compute
        },
        ..default()
    };

    #[cfg(target_arch = "wasm32")]
    {
        options.io.min_threads = 0;
        options.io.max_threads = 0;
        options.io.percent = 0.0;
        // Half the workers each, and `min_threads: 1` on both so a two-core machine still gets a
        // thread for each rather than starving one pool to fill the other.
        options.async_compute.min_threads = 1;
        options.async_compute.max_threads = usize::MAX;
        options.async_compute.percent = 0.5;
        options.compute.min_threads = 1;
        options.compute.max_threads = usize::MAX;
        options.compute.percent = 1.0;
    }

    options
}

/// The IO and compute pools' thread counts on `cores`, with an optional compute override.
struct PoolSplit {
    io: usize,
    compute: usize,
}

/// IO gets half of `cores`, clamped to 2..=8. Compute gets what Bevy's quarter split for IO and
/// async compute (2..=8 and 1..=4) leaves, at least 1, unless `compute_override` names it.
fn pool_split(cores: usize, compute_override: Option<usize>) -> PoolSplit {
    let share = |percent: f32| (cores as f32 * percent).round() as usize;
    let quarter_io = share(0.25).min(cores).clamp(2, 8);
    let async_compute = share(0.25)
        .min(cores.saturating_sub(quarter_io))
        .clamp(1, 4);
    PoolSplit {
        io: share(0.5).min(cores).clamp(2, 8),
        compute: compute_override
            .unwrap_or_else(|| cores.saturating_sub(quarter_io + async_compute).max(1)),
    }
}

/// `WOW_COMPUTE_THREADS=<n>`: the compute pool's thread count, unset for Bevy's share.
fn compute_threads() -> Option<usize> {
    std::env::var("WOW_COMPUTE_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
}

#[cfg(test)]
mod tests {
    use super::pool_split;

    #[test]
    fn io_doubles_while_compute_keeps_the_quarter_split_remainder() {
        let s = pool_split(8, None);
        assert_eq!((s.io, s.compute), (4, 4));
        let s = pool_split(12, None);
        assert_eq!((s.io, s.compute), (6, 6));
        let s = pool_split(16, None);
        assert_eq!((s.io, s.compute), (8, 8));
        let s = pool_split(32, None);
        assert_eq!((s.io, s.compute), (8, 20));
    }

    #[test]
    fn small_machines_keep_their_floors() {
        let s = pool_split(2, None);
        assert_eq!((s.io, s.compute), (2, 1));
        let s = pool_split(4, None);
        assert_eq!((s.io, s.compute), (2, 1));
    }

    #[test]
    fn a_compute_override_wins() {
        let s = pool_split(8, Some(1));
        assert_eq!((s.io, s.compute), (4, 1));
    }
}
