// platform.js — browser-platform glue the client cannot ask for itself: full-screen key capture,
// and holding the page against the mouse's side buttons.
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
