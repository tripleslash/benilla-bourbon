// gpu.js — the GPU guard: keep the device from hanging, and survive it when it does anyway.
//
// Imported by the hosting page (web/index.html) BEFORE `boot(init)`, for three reasons that are
// each load-bearing:
//
//   1. it wraps `GPUAdapter.prototype.requestDevice`, and it has to be in place before wgpu asks
//      for the device — which happens inside `init()`, in `RenderPlugin::finish`;
//   2. it owns the canvas's CSS box, and winit reads that box when it creates the window;
//   3. it folds the recovery tier into `window.__wenilla_env`, which `benilla_app::webenv` reads
//      once, on the client's first look.
//
// ## The report this exists for
//
// Low-end machines — integrated GPUs, shared memory — hit **"Device Hung"** in the browser, most
// often while the window is being resized. That is a driver **TDR**: Windows' watchdog gives the
// GPU ~2 s to answer, kills and resets the device when it does not, and Dawn surfaces the reset as
// a lost `GPUDevice` (`DXGI_ERROR_DEVICE_HUNG` in the message). Everything wgpu holds — every
// texture, buffer and pipeline — dies with it, and there is no in-place recovery: `RenderDevice` is
// created once, in `RenderPlugin::finish`, and the whole render app is built around it.
//
// **Why a resize is the trigger.** The canvas's CSS box used to be `100vw × 100vh`, so a window
// drag made the browser's `ResizeObserver` fire on every frame of the drag, and winit turned each
// observation into a `Resized` event. Each one re-sizes the whole frame's memory:
//
//   * the swapchain (2-3 images) is reconfigured,
//   * every `ViewTarget` main texture is re-created (`Rgba16Float`, so 8 bytes a pixel, ×2),
//   * the depth and prepass targets go with them,
//   * and `benilla_app::world_backdrop` retires its full-resolution backdrop image and publishes a
//     new one — deliberately, because a mutated one is a texture-identity lie (decision 1647), but
//     it means a *new* allocation every step.
//
// At 2560×1600 that is a few hundred megabytes of allocate-and-free **per frame**, with the old set
// still alive for three more frames in Bevy's `TextureCache`. On a discrete card with its own VRAM
// that is merely wasteful. On an iGPU carving its heap out of system memory it is exactly the
// allocator thrash the watchdog was written to catch.
//
// ## So: two halves, prevention first
//
// **The governor (prevention, always on).** The page — not the stylesheet — owns the canvas's CSS
// box, and it only changes it when a resize has *settled*. While the drag is in flight the canvas
// keeps its size and a CSS `transform: scale()` stretches it over the viewport, which costs the
// compositor one blit and the GPU nothing at all. One resize gesture therefore reallocates once,
// at the end, instead of sixty times. This is what a native game does when you drag its window, and
// it is the half that runs on every machine, healthy or not.
//
// The same transform carries a **pixel budget**: the committed box can be made *smaller* than the
// viewport and scaled back up, which is "render at 1080p on a 4K panel" and cuts every full-screen
// pass in the frame proportionally. At the default tier the budget is off — nothing changes for a
// machine that has never had a problem — it is what a recovery steps down to.
//
// **The watchdog (recovery).** `GPUDevice.lost` is the only reliable signal: wgpu's browser backend
// never reads it, and the symptom the client shows is a frozen canvas plus a console full of
// "[Invalid Texture]" cascades. On a real loss the page steps the tier down, says what happened,
// and reloads — the packs are in the service worker's cache and the wasm in the HTTP cache, so a
// reload is seconds, not a fresh boot. Repeated losses stop auto-reloading and hand the player a
// button, because a reload loop is worse than an honest error.
//
// Everything here is best-effort and feature-detected, the same posture as `web/platform.js`: a
// browser that refuses any of it must still play exactly as it does today, and no failure in this
// module may ever be able to take the client down.

// The per-axis ceiling wgpu's default limits impose (`max_texture_dimension_2d`), mirrored from
// `benilla_app::world_backdrop::MAX_RENDER_AXIS`. A canvas wider than this is a texture the device
// refuses to create, so the governor clamps to it at every tier including the default one — that
// clamp is a correctness guard, not a quality setting.
const MAX_AXIS = 8192;

// The recovery ladder. `budget` is a ceiling on the canvas's backing store in DEVICE pixels;
// `renderScale` is the client's own `renderScale` CVar (`benilla_app::world_backdrop::RenderScale`),
// which shrinks the 3D and leaves the interface at full resolution — so the two compose: the budget
// takes the whole frame down a step, the render scale takes the world down a further one.
// `lightingQuality` is a ceiling on the client's `lightingQuality` CVar (0 low … 4 cinematic, the
// shipped default): Cinematic's cube shadows and Ultra's glow lights are the heaviest passes in a
// modern frame, and a smaller frame alone does not take them off a device that has already hung.
//
// **Tier 0 is genuinely untouched**, and that is deliberate. A budget applied to everyone would
// quietly soften a 4K display for a machine that has never had a problem, to fix something that
// machine does not have. The governor's coalescing is free and universal; the budget is the price
// of a device that has actually hung.
const TIERS = [
  { name: 'full', budget: Infinity, renderScale: null, lightingQuality: null },
  { name: 'reduced', budget: 2300000, renderScale: '0.85', lightingQuality: '2' }, // ~1080p, High
  { name: 'safe', budget: 1300000, renderScale: '0.7', lightingQuality: '1' }, //     ~900p, Medium
  { name: 'minimal', budget: 950000, renderScale: '0.55', lightingQuality: '0' }, //  ~720p, Low
];

const STORE_KEY = 'wenilla.gpu';
// A loss older than this is not evidence about this machine any more — a driver was updated, a
// dock was unplugged, the tab was left open for a week. Without a decay the very first bad day
// would pin the client at a lower tier for the life of the browser profile.
const TIER_DECAY_MS = 3 * 24 * 60 * 60 * 1000;
// Losses inside this window are "the same episode", and are what the auto-reload cap counts.
const EPISODE_MS = 10 * 60 * 1000;
// How many times an episode may reload itself before the page stops and asks. Three is enough to
// walk the ladder from `full` to `minimal`; a fourth would be a reload loop.
const MAX_AUTO_RELOADS = 3;
// The pause before each reload, by attempt. It grows because a GPU that has just been reset is
// still settling: reconnecting to it immediately is how the second hang gets earned.
const RELOAD_DELAY_MS = [1500, 4000, 9000];
// How long a resize must be quiet before its size is committed. Long enough to swallow a drag's
// frame-by-frame observations, short enough that a single deliberate resize does not feel stuck.
const SETTLE_MS = 180;
// The live step-down trigger: this many device-level errors (not validation — those are our bugs)
// inside this window means the device is in trouble but has not died yet.
const INTERNAL_ERRORS = 3;
const INTERNAL_WINDOW_MS = 10000;
// A live step-down reallocates once, so it must not be able to chatter.
const STEPDOWN_COOLDOWN_MS = 30000;

/// Install both halves. Call once, from the page, before `boot(init)`.
///
/// `canvas` is the element the client renders into (`#benilla`); everything else is optional and
/// exists for the dev page's query switches.
export function installGpuGuard(canvas, { query = new URLSearchParams(location.search) } = {}) {
  const state = loadState(query);
  publishTier(state, query);
  const governor = installResizeGovernor(canvas, state);
  installDeviceWatchdog(canvas, state, governor);
  // The diagnostic seam: a pasted console line is how a session on somebody else's machine gets
  // read, and nothing else on the page can imply which tier that session actually ran at. Its one
  // control is `reset()`, which the client calls when the player applies Graphics settings.
  window.__wenilla_gpu = {
    get tier() {
      return state.tier;
    },
    get tierName() {
      return TIERS[state.tier].name;
    },
    get losses() {
      return state.losses.slice();
    },
    get renderSize() {
      return governor.renderSize();
    },
    /// Drive the recovery path by hand, from the console. A real TDR cannot be provoked on
    /// demand — `device.destroy()` reports reason `"destroyed"`, which is teardown and correctly
    /// ignored — so without this the overlay, the ladder and the reload cap could only ever be
    /// seen on the machine that has the fault. `?gputier=N` covers the *result* of a recovery;
    /// this covers the recovery itself.
    simulateLoss(message = 'simulated device loss') {
      onDeviceLost(canvas, state, { reason: 'unknown', message });
    },
    /// The one call the client itself makes (`benilla_app::webgpu::settings_applied`): the player
    /// applied Graphics settings of their own, so the recovery tier is retired — persisted as
    /// tier 0 and the budget lifted now, without a reload. A tier pinned by `?gputier=` stays: it
    /// is somebody measuring on purpose. Returns whether a tier was cleared.
    reset() {
      return resetTier(state, governor);
    },
    /// The render worker's device (`web/renderworker.js`), reported through `web/threads.js`: a
    /// device made on a worker is not one the wrapper below sees, so the worker wraps its own
    /// `requestDevice` and forwards what it observes here.
    deviceLost(info) {
      onDeviceLost(canvas, state, info);
    },
    deviceError(kind) {
      onUncapturedError(state, governor, kind);
    },
  };
  // Silent at tier 0 — this page keeps the console for errors and `[diag]` lines. Above it,
  // something already went wrong on this machine and the reason the picture is softer than it was
  // must be readable without asking.
  if (state.tier > 0) {
    const why = state.losses.length
      ? `the GPU driver reset ${state.losses.length} time(s) recently`
      : 'set explicitly';
    console.info(
      `wenilla: graphics recovery tier ${state.tier} (${TIERS[state.tier].name}) — ${why}. ` +
        'Applying Graphics settings in the game clears it, as does ?gputier=0.'
    );
  }
  return window.__wenilla_gpu;
}

// ---------------------------------------------------------------------------------------------
// Tier state
// ---------------------------------------------------------------------------------------------

/// The persisted record: which tier this browser profile settled on, and when the device was lost.
///
/// `localStorage`, not `sessionStorage`: a machine that hangs hangs again tomorrow, and the whole
/// point of the ladder is that the *next* visit starts where the last one ended. It is read
/// through a try/catch because storage can be denied outright (a private window, a blocked
/// third-party context), and a client that refused to boot for want of a quality setting would be
/// a worse bug than the one this module is fixing.
function loadState(query) {
  const state = {
    tier: 0,
    losses: [],
    lastStepDown: 0,
    leaving: false,
    internal: [],
    pinned: false,
  };
  let stored = null;
  try {
    stored = JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
  } catch (_) {
    /* denied or corrupt — the defaults are the right answer */
  }
  if (stored && typeof stored === 'object') {
    state.tier = clampTier(stored.tier);
    state.losses = Array.isArray(stored.losses) ? stored.losses.filter(Number.isFinite) : [];
  }
  // Decay: a tier nobody has needed for days steps back up, one rung per visit, so a machine that
  // was merely having a bad afternoon climbs back to full on its own.
  const last = state.losses.length ? Math.max.apply(null, state.losses) : 0;
  if (state.tier > 0 && Date.now() - last > TIER_DECAY_MS) {
    state.tier -= 1;
    state.losses = [];
    saveState(state);
  }
  // `?gputier=N` pins the tier for this load and writes it through, which is how the ladder is
  // exercised on a machine that will not hang on demand. `?gpubudget=<megapixels>` overrides the
  // tier's budget alone, for pricing one specific resolution.
  const pinned = query.get('gputier');
  if (pinned !== null && pinned !== '') {
    state.pinned = true;
    state.tier = clampTier(Number(pinned));
    if (state.tier === 0) state.losses = [];
    saveState(state);
  }
  const budget = Number(query.get('gpubudget'));
  const override = Number.isFinite(budget) && budget > 0 ? budget * 1000000 : null;
  state.budget = () => (override === null ? TIERS[state.tier].budget : override);
  return state;
}

function saveState(state) {
  try {
    localStorage.setItem(
      STORE_KEY,
      JSON.stringify({ tier: state.tier, losses: state.losses, updated: Date.now() })
    );
  } catch (_) {
    /* storage denied; the tier then lasts one page load, which is still better than none */
  }
}

/// The player chose their own Graphics settings: the ladder's guess about this machine is spent.
/// Without this a single bad afternoon held the lower tier until three days passed, the browser's
/// storage was cleared, or somebody knew to type `?gputier=0`.
function resetTier(state, governor) {
  if (state.pinned || (state.tier === 0 && state.losses.length === 0)) return false;
  const was = state.tier;
  state.tier = 0;
  state.losses = [];
  state.internal = [];
  saveState(state);
  governor.recommit();
  if (was > 0) {
    console.info(
      `wenilla: graphics settings applied — recovery tier ${was} (${TIERS[was].name}) cleared.`
    );
  }
  return was > 0;
}

function clampTier(n) {
  return Number.isFinite(n) ? Math.min(TIERS.length - 1, Math.max(0, Math.trunc(n))) : 0;
}

/// Hand the tier's render scale to the client, through the same `window.__wenilla_env` object a
/// hosting page uses for credentials (`benilla_app::webenv`), under its own key
/// (`gpu_render_scale`, `cvars::apply_gpu_tier`). A fallback, not an override: it applies only
/// while the player has no render scale of their own saved, is never written to `config.toml`, and
/// the player's own Render Scale choice takes the setting back — so a recovery can neither pin
/// itself into saved settings nor lock the player out of the slider.
///
/// An explicit `?render_scale=` wins — it is somebody measuring on purpose, and a recovery tier
/// must not silently outrank a person at the keyboard.
///
/// The lighting quality rides beside it (`gpu_lighting_quality`) as a ceiling rather than a
/// fallback: it holds a saved quality down too, is never saved either, and the next Graphics Apply
/// lifts it. A `?lightingquality=` or `$WOW_LIGHTING_QUALITY` lever outranks it on the client side.
function publishTier(state, query) {
  const tier = TIERS[state.tier];
  const env = (window.__wenilla_env = window.__wenilla_env || {});
  if (tier.lightingQuality && env.gpu_lighting_quality === undefined) {
    env.gpu_lighting_quality = tier.lightingQuality;
  }
  if (!tier.renderScale || query.get('render_scale') !== null) return;
  if (env.render_scale === undefined && env.gpu_render_scale === undefined) {
    env.gpu_render_scale = tier.renderScale;
  }
}

// ---------------------------------------------------------------------------------------------
// The resize governor
// ---------------------------------------------------------------------------------------------

/// Own the canvas's CSS box: commit a size only when the viewport has stopped moving, and cover
/// the gap with a transform.
///
/// **The whole mechanism is one formula.** The canvas is laid out at `committed` CSS pixels and
/// carries `transform: scale(viewport / committed)` with `transform-origin: 0 0`, so it always
/// covers the viewport exactly, whatever `committed` is. Nothing but a commit allocates:
///
///   * *during a drag* `committed` is the pre-drag size and the scale is whatever the drag has
///     reached, so the picture stretches — no swapchain, no view targets, no backdrop;
///   * *at rest with no budget* `committed` is the viewport and the scale is exactly 1;
///   * *at rest under a budget* `committed` is the viewport divided by `k`, and the scale is `k` —
///     the client renders `k²` fewer pixels and the compositor scales them back up.
///
/// **Input is unaffected**, which is the property that makes the transform usable at all. winit
/// reads pointer positions from `offsetX`/`offsetY` (`platform_impl/web/web_sys/event.rs`), and
/// those are defined in the target element's own untransformed coordinate space — the browser
/// inverts the transform for us. The client sees a window `committed` CSS pixels across and a
/// cursor denominated in the same units, which is exactly the pairing every pick ray in
/// `benilla_app` is built on (`world_backdrop::retarget_world_camera`, B169).
///
/// **The backing store is unaffected too.** winit sizes the surface from the resize observer's
/// `devicePixelContentBoxSize`, which is the *layout* box in device pixels; CSS transforms do not
/// enter it. So `committed × devicePixelRatio` is the number of pixels the GPU is asked for, and
/// that is the number the budget is denominated in.
///
/// This is why `benilla_world::boot` turns `fit_canvas_to_parent` **off** on wasm: that flag makes
/// Bevy write `width: 100%; height: 100%` onto the canvas at window creation, which would land
/// after this module's first commit and take the box back.
function installResizeGovernor(canvas, state) {
  let committed = null;
  let timer = 0;

  // The layout viewport, which is what `100vw × 100vh` resolved to before this module existed —
  // `clientWidth`/`clientHeight` rather than `innerWidth`/`innerHeight` because they exclude a
  // classic scrollbar, and they follow the document into full screen.
  const viewport = () => ({
    w: Math.max(1, document.documentElement.clientWidth || window.innerWidth || 1),
    h: Math.max(1, document.documentElement.clientHeight || window.innerHeight || 1),
  });

  // How much smaller than the viewport the canvas has to be laid out to stay inside the tier's
  // pixel budget and wgpu's per-axis ceiling. Never below 1: this shrinks a frame, it never
  // enlarges one.
  const shrink = (v) => {
    const dpr = window.devicePixelRatio || 1;
    const budget = state.budget();
    const area = v.w * dpr * (v.h * dpr);
    const byArea = Number.isFinite(budget) && area > budget ? Math.sqrt(area / budget) : 1;
    const byAxis = Math.max((v.w * dpr) / MAX_AXIS, (v.h * dpr) / MAX_AXIS);
    return Math.max(1, byArea, byAxis);
  };

  // Follow the viewport with the transform alone. Runs on every observation during a drag, so it
  // must stay allocation-free and must not touch width/height.
  const stretch = () => {
    if (!committed) return;
    const v = viewport();
    const sx = v.w / committed.w;
    const sy = v.h / committed.h;
    canvas.style.transform = sx === 1 && sy === 1 ? '' : `scale(${sx}, ${sy})`;
  };

  // The one allocating step. Idempotent: a commit that lands on the size already committed writes
  // nothing, so a settle timer that fires on an unchanged viewport costs a surface reconfigure of
  // exactly zero.
  const commit = () => {
    clearTimeout(timer);
    timer = 0;
    const v = viewport();
    const k = shrink(v);
    const want = { w: v.w / k, h: v.h / k };
    if (!committed || !near(committed.w, want.w) || !near(committed.h, want.h)) {
      committed = want;
      canvas.style.width = `${want.w}px`;
      canvas.style.height = `${want.h}px`;
    }
    stretch();
  };

  const onViewportChange = () => {
    stretch();
    clearTimeout(timer);
    timer = setTimeout(commit, SETTLE_MS);
  };

  // `ResizeObserver` on the document element is the signal that actually tracks a window drag;
  // `resize` and `fullscreenchange` are belt and braces for the transitions some browsers do not
  // route through layout. All three land on the same debounce, so extra notifications are free.
  if (window.ResizeObserver) {
    try {
      new ResizeObserver(onViewportChange).observe(document.documentElement);
    } catch (_) {
      /* fall through to the window listeners below */
    }
  }
  window.addEventListener('resize', onViewportChange, { passive: true });
  document.addEventListener('fullscreenchange', onViewportChange);
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', onViewportChange, { passive: true });
  }
  watchDevicePixelRatio(onViewportChange);

  commit();

  return {
    /// Re-commit now — the live step-down's way of applying a smaller budget without a reload.
    recommit: commit,
    /// The backing store the client is currently being asked for, in device pixels. Diagnostic
    /// only; `world_backdrop`'s own log line is the authority on what the *world* renders at.
    renderSize() {
      const dpr = window.devicePixelRatio || 1;
      return committed
        ? { width: Math.round(committed.w * dpr), height: Math.round(committed.h * dpr) }
        : null;
    },
  };
}

/// A display-scale change (browser zoom, or the window crossing to a monitor with a different
/// ratio) moves the backing store without moving the layout box, so the budget has to be
/// recomputed even though the viewport reads the same numbers. `matchMedia` has to be re-armed
/// after each change: the query names the exact ratio it was created for, so it fires once.
function watchDevicePixelRatio(onChange) {
  if (!window.matchMedia) return;
  let mql;
  try {
    mql = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
  } catch (_) {
    return; // a browser without `resolution` queries; the layout signals still fire
  }
  const fire = () => {
    // Detach before re-arming on the legacy path: `addListener` has no `once`, and a listener per
    // zoom step would accumulate for the life of the tab.
    if (mql.removeListener) mql.removeListener(fire);
    onChange();
    watchDevicePixelRatio(onChange);
  };
  if (mql.addEventListener) mql.addEventListener('change', fire, { once: true });
  else if (mql.addListener) mql.addListener(fire);
}

/// Sub-pixel equality. The committed size is fractional whenever a budget divides the viewport, and
/// two sizes that round to the same device pixel must not be allowed to trigger a reconfigure.
function near(a, b) {
  return Math.abs(a - b) < 0.5;
}

// ---------------------------------------------------------------------------------------------
// The device watchdog
// ---------------------------------------------------------------------------------------------

/// Wrap `requestDevice` so every device the client creates is watched.
///
/// Wrapping the prototype rather than asking wgpu is the only option available: wgpu 27's browser
/// backend never reads `GPUDevice.lost`, and Bevy has no hook for it either — so from inside the
/// wasm module a hung device is indistinguishable from a slow one. The page can see it, and the
/// page is also the only layer that can do anything about it.
///
/// This composes with the two diagnostic wrappers `web/index.html` installs (the WGSL compile-error
/// reporter and the sampler census): each wraps whatever it found, so all three run.
function installDeviceWatchdog(canvas, state, governor) {
  if (!navigator.gpu || !window.GPUAdapter) return;

  // A device lost while the page is going away is not a fault — it is teardown. Without this the
  // watchdog would raise an overlay on top of a navigation that has already started.
  //
  // `pagehide` and not `beforeunload`: a `beforeunload` listener disqualifies the page from the
  // back/forward cache in every browser that has one, which would be a real cost paid for a flag.
  window.addEventListener('pagehide', () => {
    state.leaving = true;
  });

  const orig = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (desc) {
    const device = await orig.call(this, desc);
    try {
      device.lost.then(
        (info) => onDeviceLost(canvas, state, info),
        () => {}
      );
      device.addEventListener('uncapturederror', (e) =>
        onUncapturedError(state, governor, errorKind(e))
      );
    } catch (_) {
      /* an implementation without `lost` or events: nothing to watch, and nothing to break */
    }
    return device;
  };
}

/// The class name of an `uncapturederror` event's error, `''` when there is none.
function errorKind(e) {
  return (e && e.error && e.error.constructor && e.error.constructor.name) || '';
}

/// The device is gone. Step the ladder down, say so, and either restart or hand over.
function onDeviceLost(canvas, state, info) {
  const reason = (info && info.reason) || 'unknown';
  const message = (info && info.message) || '';
  // `"destroyed"` is the deliberate-teardown reason — the page navigating away, or something
  // calling `device.destroy()`. Only `"unknown"` is a driver reset, which is what a TDR looks like
  // from here (Dawn puts `DXGI_ERROR_DEVICE_HUNG` / `DEVICE_REMOVED` in the message).
  if (state.leaving || reason === 'destroyed') return;

  console.error(`wenilla: the GPU device was lost (${reason}) — ${message || 'no detail given'}`);

  const now = Date.now();
  state.losses = state.losses.filter((t) => now - t < TIER_DECAY_MS);
  state.losses.push(now);
  state.tier = clampTier(state.tier + 1);
  saveState(state);

  // The frame on the glass is whatever the device managed before it died, and the client will keep
  // drawing nothing over it. Take it away rather than leaving a frozen world under the message.
  try {
    canvas.style.visibility = 'hidden';
  } catch (_) {}

  const episode = state.losses.filter((t) => now - t < EPISODE_MS).length;
  const overlay = showOverlay();
  if (episode > MAX_AUTO_RELOADS) {
    overlay.giveUp(state, message);
    return;
  }
  overlay.restarting(RELOAD_DELAY_MS[Math.min(episode - 1, RELOAD_DELAY_MS.length - 1)]);
}

/// `GPUInternalError` and `GPUOutOfMemoryError` are the device saying it is in trouble while it is
/// still alive — a validation error, by contrast, is one of our own bugs and means nothing about
/// the machine. A burst of them is the last warning before a hang, so take the budget down a rung
/// *without* a reload: one reallocation now against a device reset later.
function onUncapturedError(state, governor, kind) {
  if (kind !== 'GPUInternalError' && kind !== 'GPUOutOfMemoryError') return;

  const now = Date.now();
  state.internal = state.internal.filter((t) => now - t < INTERNAL_WINDOW_MS);
  state.internal.push(now);
  if (state.internal.length < INTERNAL_ERRORS) return;
  if (now - state.lastStepDown < STEPDOWN_COOLDOWN_MS) return;
  // The last rung is reserved for an actual loss. Stepping all the way down on error counts alone
  // would let a noisy driver take the picture apart for something the device survived.
  if (state.tier >= TIERS.length - 2) return;

  state.internal = [];
  state.lastStepDown = now;
  state.tier = clampTier(state.tier + 1);
  saveState(state);
  console.warn(
    `wenilla: ${INTERNAL_ERRORS} device errors in ${INTERNAL_WINDOW_MS / 1000}s — ` +
      `dropping to graphics tier ${state.tier} (${TIERS[state.tier].name}) to keep the driver ` +
      'alive. The pixel budget applies now; the render scale and lighting cap follow on the next ' +
        'reload.'
  );
  governor.recommit();
}

// ---------------------------------------------------------------------------------------------
// The overlay
// ---------------------------------------------------------------------------------------------

/// The recovery overlay — deliberately its own DOM, not `boot.js`'s.
///
/// It has to be able to appear *over* the boot overlay (a device can be lost during the first
/// frames), it outlives the module that owns that one, and its content is a different thing: not
/// progress, but an explanation and a decision.
let overlayRoot = null;
function showOverlay() {
  // One overlay, ever. A device can be reported lost more than once (a second `GPUDevice` the
  // client had already asked for resolves its own `lost` promise), and two stacked explanations
  // with two countdowns racing each other would be worse than either.
  if (overlayRoot) {
    const box = overlayRoot.querySelector('.box');
    return overlayApi(box);
  }
  const root = document.createElement('div');
  root.id = 'wenilla-gpu';
  root.innerHTML = `
    <style>
      #wenilla-gpu { position: fixed; inset: 0; z-index: 30; display: grid; place-items: center;
        background: #000; color: #ccc; font: 15px system-ui, sans-serif; }
      #wenilla-gpu .box { width: min(520px, 86vw); text-align: center; }
      #wenilla-gpu h1 { font-size: 17px; font-weight: 600; color: #e8e8e8; margin: 0 0 .8rem; }
      #wenilla-gpu p { margin: 0 0 .8rem; line-height: 1.5; color: #9a9a9a; }
      #wenilla-gpu ul { margin: 0 0 1rem; padding-left: 1.2rem; text-align: left; color: #9a9a9a;
        line-height: 1.6; }
      #wenilla-gpu button { font: inherit; color: #111; background: #c9a227; border: 0;
        border-radius: 4px; padding: .5rem 1.1rem; cursor: pointer; }
      #wenilla-gpu code { color: #b9b9b9; word-break: break-word; }
    </style>
    <div class="box"></div>`;
  document.body.appendChild(root);
  overlayRoot = root;
  return overlayApi(root.querySelector('.box'));
}

function overlayApi(box) {
  return {
    /// The ordinary path: say what happened, count down, reload. The button is there because a
    /// countdown somebody is watching should never be the slowest way to act on it.
    restarting(delayMs) {
      box.innerHTML = `
        <h1>The graphics device was reset</h1>
        <p>Your graphics driver restarted, which the browser reports as a lost device. Restarting
           the client at a lighter setting.</p>
        <p class="in"></p>
        <button type="button">Restart now</button>`;
      const inEl = box.querySelector('.in');
      box.querySelector('button').addEventListener('click', () => location.reload());
      let left = Math.ceil(delayMs / 1000);
      const tick = () => {
        inEl.textContent = left > 0 ? `Restarting in ${left}…` : 'Restarting…';
        if (left-- <= 0) return;
        setTimeout(tick, 1000);
      };
      tick();
      setTimeout(() => location.reload(), delayMs);
    },

    /// The cap. Reloading again would be a loop, and a loop is worse than an honest message — so
    /// this one stops and says what is actually known, including the driver's own words.
    giveUp(state, message) {
      box.innerHTML = `
        <h1>The graphics driver keeps resetting</h1>
        <p>The client restarted ${MAX_AUTO_RELOADS} times and the device was lost again each time,
           so it has stopped trying. This is a driver-level reset (a TDR), not something the page
           can work around on its own.</p>
        <ul>
          <li>Update your graphics driver — this is the fix that works most often.</li>
          <li>Close other GPU-heavy tabs and applications; an integrated GPU shares its memory.</li>
          <li>Make the browser window smaller before loading, and avoid resizing it while playing.</li>
          <li>The client is already at its lowest graphics tier
              (<code>${TIERS[state.tier].name}</code>); <code>?gputier=0</code> puts it back.</li>
        </ul>
        <p class="detail"></p>
        <button type="button">Try again</button>`;
      if (message) box.querySelector('.detail').innerHTML = `<code>${escapeHtml(message)}</code>`;
      box.querySelector('button').addEventListener('click', () => location.reload());
    },
  };
}

/// The device-lost message comes from the driver, so it is not ours to trust as markup.
function escapeHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]
  );
}
