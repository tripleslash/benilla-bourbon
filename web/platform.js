// platform.js — browser-platform glue the client cannot ask for itself: full-screen key capture,
// holding the page against the mouse's side buttons, and (fork) the desktop client download.
// Imported by the hosting page (web/index.html), the same way boot.js is.
//
// Everything here is best-effort. Every entry point is feature-detected and every failure is
// swallowed: none of it is load-bearing for booting the client, and a browser that refuses it must
// still play exactly as it does today. The page imports this module *dynamically*, off the boot
// path, so even a missing platform.js only costs the button.
//
// Durable storage is deliberately NOT here. `navigator.storage.persist()` is asked for once per
// boot from `packs.js::bootstrap` — early and unawaited, because that call has to be ahead of the
// service worker's first write. The grant is origin-wide, so that one ask already covers the
// state folder a second call from here would have been asking for; see the Persistence note in
// web/README.md.

// The keys the game needs more than the browser does, as KeyboardEvent.code values. Keyboard
// Lock only takes effect in full screen, and only for the codes named here — locking everything
// (`lock()` with no argument) would take Ctrl+T, Ctrl+Tab and the rest hostage for no gain.
//
//   Escape — WoW's close-window / game-menu key. bindings/chord.rs maps it as "ESCAPE" and
//            login/mod.rs + char_select/ bind it directly. Unlocked, every press leaves full
//            screen and drops the pointer lock instead of ever reaching the client.
//   Tab    — target cycling. Unlocked, it walks browser focus out of the canvas.
//   KeyW   — forward. W already reaches the page on its own; locking the code is what keeps
//            Ctrl+W / Cmd+W from closing the tab mid-pull.
const LOCK_KEYS = ['Escape', 'Tab', 'KeyW'];

// Keep Keyboard Lock tied to the document's actual full-screen state, and optionally wire a
// button to toggle it.
//
// `button` may be null: the lock still follows F11 and every other way into full screen. When a
// button is given it also gets its label and aria-pressed kept in sync, and is hidden outright
// on a browser with no Fullscreen API.
//
// `focusTarget` (the canvas) gets focus back after every toggle: the click leaves focus on the
// button, and a focused button eats Space and Enter before winit's canvas listeners see them.
export function installFullscreenToggle(button, { target = document.documentElement, focusTarget = null } = {}) {
  const supported = !!(target.requestFullscreen && document.exitFullscreen);
  if (!supported) {
    if (button) button.hidden = true;
    return;
  }

  const lockKeys = async () => {
    if (!navigator.keyboard || !navigator.keyboard.lock) return;
    // Chrome shows its own "press and hold Esc to exit full screen" toast on the first lock.
    try { await navigator.keyboard.lock(LOCK_KEYS); } catch (_) { /* denied; keys stay browser-owned */ }
  };
  const unlockKeys = () => {
    if (!navigator.keyboard || !navigator.keyboard.unlock) return;
    try { navigator.keyboard.unlock(); } catch (_) {}
  };

  // Driven by `fullscreenchange`, not by the click: F11, the Esc long-press and the browser's
  // own exit affordances never go through our handler, and the lock has to follow all of them.
  // `initial` marks the install-time call, which only paints the button — stealing focus on
  // page load (before there is a client to type into) is not this module's business.
  const sync = (initial) => {
    const full = document.fullscreenElement !== null;
    if (button) {
      button.textContent = full ? 'exit full screen' : 'full screen';
      button.setAttribute('aria-pressed', String(full));
    }
    if (full) lockKeys(); else unlockKeys();
    if (!initial && focusTarget) focusTarget.focus?.();
  };

  if (button) {
    button.addEventListener('click', async () => {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await target.requestFullscreen({ navigationUI: 'hide' });
      } catch (e) {
        console.warn('fullscreen:', e);
      }
    });
  }
  document.addEventListener('fullscreenchange', () => sync(false));
  sync(true);
}

// Keep the mouse's side buttons from leaving the page. A browser reads them as history
// navigation (button 3 back, 4 forward) and the client binds them: 1.12's own
// `DefaultBindings.wtf` has `BUTTON4 TOGGLEAUTORUN`. A player reaching for autorun must not
// leave instead, which ends the session and loses the world load.
//
// `preventDefault()` on the mouse events is enough in some browsers and is kept for them, but on
// Windows Chrome decides the navigation before the page gets a say. So the history is pinned
// too: one state pushed, and put back whenever it is popped. Back and forward then move inside
// this same document, which never unloads. The listeners do NOT `stopPropagation()`: the events
// must still reach the canvas, where winit turns them into `MouseButton::Back`/`Forward` for the
// binding dispatch.
//
// This holds the browser's own Back button as well, deliberately: for a game client, leaving by
// accident is the failure. The tab's close button, a typed URL and a bookmark are unaffected.
export function holdMouseNavigationButtons(target = window) {
  const swallow = (e) => {
    if (e.button === 3 || e.button === 4) e.preventDefault();
  };
  // Capture phase, so the decision is made before anything downstream; never passive, since a
  // passive listener may not call `preventDefault`.
  for (const name of ['mousedown', 'mouseup', 'auxclick', 'click']) {
    target.addEventListener(name, swallow, { capture: true, passive: false });
  }
  // `pushState` throws on an opaque origin, and none of this is load-bearing for the boot.
  try {
    history.pushState(null, '', location.href);
    window.addEventListener('popstate', () => {
      history.pushState(null, '', location.href);
    });
  } catch (e) {
    console.warn('platform.js: history pin unavailable:', e);
  }
}

// fork: desktop — the "Desktop client" link: download `wow-reforged.exe`, which the site holds as
// 24 MiB parts named by `wow-reforged.exe.parts.json` (scripts/split-parts.sh), read back by
// `web/parts.js` as for the wasm module. The link stays hidden unless that manifest is found, so a
// site without the exe (or `trunk serve`) shows nothing. A click fetches the parts, checks the
// byte count against the manifest and saves the joined bytes as one file through a blob URL. The
// browser only sees that last save, so its own download progress never shows the transfer: the
// page's `progress` panel does (bar, MB so far of the total, percent, speed, time left, a cancel
// button); without the panel the percent goes in the link's label. `parts.js` is imported here,
// not at the top, so a page without it loses only this link.
//
// `focusTarget` (the canvas) gets focus back afterwards, for the reason `installFullscreenToggle`
// gives.
export async function installDesktopDownload(link, {
  manifest = 'wow-reforged.exe.parts.json',
  filename = 'wow-reforged.exe',
  progress = null,
  focusTarget = null,
} = {}) {
  if (!link) return;
  const { fetchPartsManifest, readParts } = await import('./parts.js');
  const label = link.textContent;
  const manifestUrl = new URL(manifest, document.baseURI);
  // `no-cache`: the parts are rewritten in place under the same names by every build, and a part
  // from one build joined to another's would be a corrupt exe that only fails when it is run.
  const noCache = { cache: 'no-cache' };
  if (await fetchPartsManifest(manifestUrl, noCache)) link.hidden = false;

  const panel = desktopProgressPanel(progress, link, label, filename);
  // The download in flight, or null. Cancel drops it at once: `fetchRetry` retries an aborted
  // fetch through its backoff before it gives up, so its rejection may come seconds later, and by
  // then a new click may own the panel.
  let run = null;
  panel.onCancel(() => {
    if (!run) return;
    run.abort();
    run = null;
    panel.hide();
    focusTarget?.focus?.();
  });

  link.addEventListener('click', async (e) => {
    e.preventDefault();
    if (run) return;
    const me = new AbortController();
    run = me;
    const init = { ...noCache, signal: me.signal };
    try {
      panel.start();
      const m = await fetchPartsManifest(manifestUrl, init);
      if (run !== me) return;
      if (!m) throw new Error(`${manifest}: not found`);
      panel.update(0, m.bytes);
      const r = await readParts(manifestUrl, m, {
        init,
        onBytes: (got, total) => { if (run === me) panel.update(got, total); },
      });
      if (!r.ok) throw new Error(`${r.url || manifest}: HTTP ${r.status}`);
      const blob = await r.blob();
      if (run !== me) return;
      if (blob.size !== m.bytes) throw new Error(`got ${blob.size} B, the manifest says ${m.bytes} B`);
      panel.done(blob.size);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.style.display = 'none';
      document.body.appendChild(a);
      a.click();
      a.remove();
      // The save dialog reads the URL asynchronously; a minute is far past it.
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (err) {
      if (run !== me) return; // cancelled: the panel is already gone
      console.warn('desktop client download:', err);
      panel.fail(err);
    } finally {
      if (run === me) {
        run = null;
        focusTarget?.focus?.();
      }
    }
  });
}

/// The download's progress UI: the page's `#desktop-progress` panel (index.html) when it has one,
/// else the link's own label, as a page from before the panel had it.
function desktopProgressPanel(root, link, label, filename) {
  const mb = (b) => (b / (1024 * 1024)).toFixed(1);
  const pctOf = (got, total) => (total > 0 ? Math.min(100, Math.floor((got * 100) / total)) : 0);
  let hideTimer = 0;

  if (!root) {
    return {
      onCancel() {},
      start() { clearTimeout(hideTimer); link.textContent = `${label} 0%`; },
      update(got, total) { link.textContent = `${label} ${pctOf(got, total)}%`; },
      done() { link.textContent = label; },
      fail() {
        link.textContent = `${label}: download failed`;
        hideTimer = setTimeout(() => { link.textContent = label; }, 5_000);
      },
      hide() { link.textContent = label; },
    };
  }

  const title = root.querySelector('.title');
  const bar = root.querySelector('.bar');
  const fill = root.querySelector('.bar > i');
  const data = root.querySelector('.data');
  const cancel = root.querySelector('.cancel');
  const titleText = title.textContent;
  // Speed over the last five seconds, not since the start, so a stall shows as one; one sample per
  // 250 ms. The DOM is touched at most every 100 ms: `onBytes` fires per network chunk, thousands
  // of times for one exe.
  let samples = [];
  let shownAt = 0;
  let shownBytes = 0;

  const rate = (now, got) => {
    const last = samples[samples.length - 1];
    if (!last || now - last.t >= 250) samples.push({ t: now, got });
    while (samples.length > 2 && now - samples[0].t > 5_000) samples.shift();
    const dt = (now - samples[0].t) / 1000;
    return dt >= 0.5 ? (got - samples[0].got) / dt : 0;
  };
  const left = (secs) => {
    if (!Number.isFinite(secs) || secs <= 0) return '';
    const s = Math.ceil(secs);
    return s < 60 ? `${s} s left` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s left`;
  };
  const setPct = (pct) => {
    fill.style.width = `${pct}%`;
    bar.setAttribute('aria-valuenow', String(pct));
  };
  const settle = (ms) => {
    if (cancel) cancel.hidden = true;
    link.removeAttribute('aria-busy');
    hideTimer = setTimeout(() => { root.hidden = true; }, ms);
  };

  return {
    onCancel(fn) { cancel?.addEventListener('click', fn); },
    start() {
      clearTimeout(hideTimer);
      samples = [];
      shownAt = 0;
      shownBytes = 0;
      root.classList.remove('err');
      title.textContent = titleText;
      if (cancel) cancel.hidden = false;
      setPct(0);
      data.textContent = 'Starting…';
      root.hidden = false;
      link.setAttribute('aria-busy', 'true');
    },
    update(got, total) {
      const now = performance.now();
      const speed = rate(now, got);
      if (got < total && now - shownAt < 100) return;
      shownAt = now;
      shownBytes = got;
      const pct = pctOf(got, total);
      setPct(pct);
      const line = [`${mb(got)} / ${mb(total)} MB`, `${pct}%`];
      if (speed > 0 && got < total) line.push(`${mb(speed)} MB/s`, left((total - got) / speed));
      data.textContent = line.filter(Boolean).join(' · ');
    },
    done(bytes) {
      setPct(100);
      title.textContent = 'Download complete';
      data.textContent = `${mb(bytes)} MB · saving ${filename}`;
      settle(4_000);
    },
    fail(err) {
      root.classList.add('err');
      title.textContent = 'Download failed';
      const at = shownBytes > 0 ? ` (at ${mb(shownBytes)} MB)` : '';
      data.textContent = `${err?.message || err}${at}`;
      settle(8_000);
    },
    hide() {
      clearTimeout(hideTimer);
      root.hidden = true;
      link.removeAttribute('aria-busy');
    },
  };
}
