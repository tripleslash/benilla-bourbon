// boot.js — the pre-login loading experience for web/index.html, the tree's one page (and for
// any hosting page that boots the client the same way).
//
// Why this exists: the client cannot show its own boot. Everything a fresh tab waits on — the
// 15 MB (brotli) wasm download, instantiation, and the first update's catalog pile (~100+
// synchronous chain reads in ONE rAF task, the "unresponsive tab" window) — happens before the
// first frame the app could draw. So the page owns a DOM overlay, and it does three jobs:
//
//   1. SHOW what is happening: a byte-counted wasm download bar (the counting stream is
//      re-wrapped in a Response so `WebAssembly.instantiateStreaming` still streams), a
//      "game data N/M MB" line for the container prefetch, then indeterminate "Starting…" states
//      driven by the client's own `window.__wenilla_progress(stage)` calls
//      (benilla-app/src/webprogress.rs) until "ready" — the first glue screen — fades it out.
//   2. HAND THE CLIENT ITS DATA. `packs.js::bootstrap` installs the service worker that makes the
//      container permanent (`web/sw.js`), fetches the index and listfile, and warms the packs a
//      boot actually touches — through `web/packworker.js`, so they arrive *decoded* and the
//      catalog frame that follows does no zstd at all. The index especially: `Chain::open` reads
//      it synchronously, and 1.6 MB through a sync XHR's per-character body decode is exactly what
//      that decode is worst at — fetched here it is a copy instead. Every miss degrades to a
//      slower read, never a failure. On a repeat visit none of it reaches the origin: the service
//      worker answers it all off disk, so the bar moves at cache speed and the only bytes on the
//      wire are `/config.json` and the ~10 KB `boot-packs.json` pointer.
//   3. CONFIGURE THE SESSION. `/config.json` carries the WebSocket gateway and **the realmlist**,
//      so the addresses a hosted client dials are a hundred-byte `no-cache` asset rather than
//      something baked into a 90 MB wasm module. It ships with the build (Trunk copies it into
//      `dist/`), not with the 3.3 GiB container, because it is the one file in the tree an
//      operator edits.
//
// Page contract: `const w = await boot(init); bind(w.memory);` — bind must stay immediately
// after (the Lua WASI shims need the module's linear memory). Query switches: `?boottrace=1`
// arms window.__wenilla_boottrace and skips the pack prefetch (a manifest-capture boot must show
// the true read order); `?noprefetch=1` skips the prefetch only (the A/B lever); `?nosw=1`
// unregisters the service worker, putting the page back on the pre-worker fetch path from the next
// load, which is how the Range/HTTP-cache fallback (and `wenilla-host --no-ranges`) stays
// exercisable; `?noworker=1` skips the fetch/decode worker, putting every pack decode back on the
// frame thread. None of them skips the index — without it nothing is readable at all.
//
// ## Regenerating `boot-manifest.json`
//
// Two ways, and the second is now the easy one.
//
//  1. `?boottrace=1` here: the arrays below fill in first-need order and a person copies the
//     deduplicated result out of the console. It is the only way that works against a CONTAINER,
//     because a container serves pack objects and a pack id does not say which name inside it was
//     wanted.
//  2. `wenilla-host --data "<WoW>/Data" --log-reads web/boot-manifest.json`: the per-name route
//     carries the name in the URL, so the HOST can record the read set with no console at all.
//     Boot, get to the character screen, Ctrl-C the host, and the manifest is written in the shape
//     `benilla-pack` parses. Add `WOW_HD_MODELS=1` to capture against the updated character models.
//
// Either way the names are then translated into each container's `boot-packs.json`:
// `benilla-pack sidecars --packs web/packs` and again `--packs web/packs-hd`.

import { bootstrap, prefetchWorld, fetchRetry } from './packs.js';

// The wasm travels brotli-compressed: content-length counts .br bytes while the counted stream
// yields decompressed ones. Dividing the two on a real build gives ~5.0; it drifts a little per
// build and that is cosmetic — the bar is clamped at 99% until the stream actually ends.
//
// Only the single-file path needs the estimate. The split site reads the module's true size out
// of `wenilla_bg.parts.json` (see fetchWasm) and this constant never enters it.
const BR_RATIO = 5.0;
const READY_TIMEOUT_MS = 20000;

export async function boot(init, opts = {}) {
  const q = new URLSearchParams(location.search);
  const ui = createBootUi();
  if (q.get('boottrace') === '1') {
    window.__wenilla_boottrace = [];
    // The layer half of the capture (`benilla_formats::web::trace_hd`). A separate array, so the
    // pasteable manifest stays a flat name list, and the thing it answers is the one a re-capture
    // has to: with the updated-character-model option ON, WHICH names the layer served — and
    // therefore whether it is being read at all. Empty with the option off, which is correct;
    // empty with it on means the layer is mounted and nothing wants it, which is a bug the base
    // trace cannot show you.
    window.__wenilla_boottrace_hd = [];
  }
  const skipPrefetch = q.get('boottrace') === '1' || q.get('noprefetch') === '1';
  installProgressHook(ui, skipPrefetch);

  // The container's root is the page's ORIGIN plus /data, not a path relative to this document —
  // the exact string `benilla_formats::web::data_base()` builds on the Rust side. The two must
  // agree or the client would fetch its packs from somewhere the page never warmed.
  const base = `${location.origin}/data`;

  const config = loadConfig(location.origin);
  const container = bootstrap(base, (done, total) => ui.setData(done, total), {
    prefetch: !skipPrefetch,
    sw: q.get('nosw') !== '1',
    worker: q.get('noworker') !== '1',
  }).catch((e) => {
    // A container that failed to bootstrap is very likely a fatal misconfiguration, but say so
    // through the client's own error path rather than guessing here: `Chain::open` will fetch the
    // index itself, and if that fails too its message names the URL and the cause.
    console.warn('wenilla: container bootstrap failed —', e);
    ui.setData(0, 0);
  });

  const wasm = fetchWasm(new URL('./wenilla_bg.wasm', import.meta.url), (got, total) =>
    ui.setWasmProgress(got, total)
  );
  // The prefetch is AWAITED before init(), and that is the whole trick. init() runs the app's
  // Startup soon after, and Startup is ~100+ *synchronous serial* chain reads in one rAF task:
  // on a 100 ms-RTT link that task measured 50 s uncached — the Mac "unresponsive tab" — and
  // still 25 s when the pool merely raced it (A/B, 2026-08-31). Waiting here costs the same
  // wall-clock the reads would cost anyway, but it is spent on a moving progress bar with the
  // page responsive, and the catalog task then runs against warm packs. The wasm download +
  // streaming compile still overlap it: fetchWasm() is already in flight, only `init()` —
  // instantiation + plugin build — moves after the data.
  await config;
  await container;
  let w;
  try {
    w = await init({ module_or_path: wasm });
  } catch (e) {
    // **Cross-origin isolation, checked here rather than up front.** The threaded build's linear
    // memory is a SharedArrayBuffer, and a page that is not cross-origin isolated cannot construct
    // one — so `init()` throws before anything else happens, with a message
    // (`WebAssembly.Memory(): shared memory is not enabled`) that names the symptom and not the
    // cause. Checking after the failure rather than before it keeps the PLAIN build unaffected:
    // that one has ordinary memory, needs no isolation, and must not be refused for lacking it.
    //
    // `web/_headers` sets COOP/COEP on the site, `wenilla-host::static_site` for `--www` and
    // `Trunk.toml`'s `[serve]` for `trunk serve`. If this fires, one of those three is not in play
    // — most likely the page is being served by something else entirely.
    if (typeof crossOriginIsolated !== 'undefined' && !crossOriginIsolated) {
      ui.fail(
        'This build needs cross-origin isolation (COOP/COEP) and this page does not have it, ' +
          'so its shared memory could not be created. Serve it with ' +
          'Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: require-corp.'
      );
      throw e;
    }
    ui.fail('The client failed to start: ' + (e && e.message ? e.message : e));
    throw e;
  }
  // init() resolved = module instantiated and the plugin graph built; the heavy catalog frame
  // is still ahead. From here the client drives the stages; the timeout is the drift backstop
  // (an old bundle without webprogress under a new page must not strand the overlay).
  ui.setStage('Starting…');
  ui.armTimeout(opts.readyTimeoutMs ?? READY_TIMEOUT_MS);
  return w;
}

/// Byte-counting fetch of the wasm, re-wrapped so streaming compilation survives. Returns a
/// Promise<Response> — wasm-bindgen's init feeds it straight to instantiateStreaming.
///
/// The module arrives in one of two layouts and this is the only place that knows it. On the
/// split layout it is N parts: a static host may refuse ANY static asset over 25 MiB and the module is
/// 77 MB with nothing left in it to strip, so `scripts/stage-packs.sh` splits it at 24 MiB and
/// writes `wenilla_bg.parts.json` beside the pieces. Everywhere else — `trunk serve` and
/// `wenilla-host --www`, both serving `web/dist`, where no such limit exists — it is the single
/// file and the manifest is simply not there.
///
/// Concatenating the parts into ONE stream is what keeps the split cheap: a wasm module is a byte
/// sequence, and nothing downstream can tell how many responses carried it, so the browser still
/// compiles while it downloads exactly as it did from one file. They are all requested at once for
/// the same reason — fetching them in turn would pay a round trip per part — and enqueued in order.
///
/// The manifest costs one ~200 B round trip ahead of the download, and buys back the number the
/// single-file path can only guess at: `bytes` is the module's true uncompressed size, where a lone
/// Content-Length counts compressed bytes and has to be scaled by BR_RATIO.
async function fetchWasm(url, onBytes) {
  const manifest = await fetchPartsManifest(url);
  const responses = manifest
    ? await Promise.all(manifest.parts.map((name) => fetchRetry(new URL(name, url))))
    : [await fetchRetry(url)];
  // Handed back as-is so init surfaces the real status — and still gets its non-streaming
  // fallback — rather than an error invented here.
  const failed = responses.find((r) => !r.ok || !r.body);
  if (failed) return failed;

  let total = manifest ? manifest.bytes : 0;
  if (!total) {
    const contentLength = Number(responses[0].headers.get('content-length')) || 0;
    const compressed = !!(responses[0].headers.get('content-encoding') || '').trim();
    total = contentLength ? contentLength * (compressed ? BR_RATIO : 1) : 0;
  }

  let got = 0;
  let i = 0;
  let reader = responses[0].body.getReader();
  const counted = new ReadableStream({
    async pull(c) {
      for (;;) {
        const { done, value } = await reader.read();
        if (!done) {
          got += value.byteLength;
          onBytes(got, total);
          c.enqueue(value);
          return;
        }
        // The end of a part is not the end of the module: step to the next reader and keep
        // pulling, so the consumer never sees the seam. The loop is why this is a `for (;;)` —
        // one `pull` must enqueue something or close, and an empty part must not do neither.
        if (++i >= responses.length) {
          onBytes(got, got); // the honest total, now that we know it
          c.close();
          return;
        }
        reader = responses[i].body.getReader();
      }
    },
    cancel(reason) {
      return Promise.all([
        reader.cancel(reason),
        // The parts still queued were never read from, so their bodies are unlocked and cancel
        // directly. Leaving them open would hold N connections open for a boot that has failed.
        ...responses.slice(i + 1).map((r) => r.body.cancel(reason)),
      ]);
    },
  });
  // Content-Type is SET, not copied from the response: instantiateStreaming refuses anything but
  // `application/wasm`, and this Response is ours — it should not inherit that from an origin's
  // extension mapping (nor carry the first part's Content-Length, which no longer describes the
  // body being returned).
  return new Response(counted, {
    status: responses[0].status,
    statusText: responses[0].statusText,
    headers: { 'content-type': 'application/wasm' },
  });
}

/// `wenilla_bg.parts.json` -> `{bytes, parts}`, or null when the module is not split.
///
/// A 404 IS an answer here and is not logged: unsplit is what every local run serves. Anything
/// else — a parse error, an offline fetch — also answers null, because the single-file path is
/// then tried and fails naming the real URL and the real cause, which is a better error than one
/// about a manifest the reader has never heard of.
async function fetchPartsManifest(url) {
  try {
    const r = await fetchRetry(new URL('./wenilla_bg.parts.json', url));
    if (!r.ok) return null;
    const m = await r.json();
    return Array.isArray(m.parts) && m.parts.length ? m : null;
  } catch {
    return null;
  }
}

/// `/config.json` -> `window.__wenilla_config`, read by `benilla_app::webenv::config`.
///
/// Two keys, `ws` and `realmlist`, and what each one is for is in `web/README.md` under
/// "the site's whole configuration" — the file itself is data an operator edits, so it carries
/// no commentary of its own (JSON has nowhere to put any).
///
/// **This is the only place the client learns its realmlist.** There is no login-screen editor any
/// more and no page-origin guess behind it: the address a hosted client dials is whatever this
/// file says, which is what makes it changeable by re-uploading one ~100-byte asset instead of a
/// 90 MB wasm module. `$WOW_HOST` / `?host=` still pins a session ahead of it — that is the
/// harness lever every probe and smoke run sets, and it never reaches a player's tree.
///
/// It sits at the ORIGIN ROOT rather than under `/data`, because it belongs to the build and not
/// to the container: `web/config.json` is checked in, Trunk copies it into `dist/`, and
/// `stage-packs.sh` hardlinks it into what a static host serves. The container is 3.3 GiB regenerated
/// from a game install, and a setting kept there would be a setting that lives in a directory
/// nobody edits and `.gitignore` cannot track.
///
/// Deliberately its OWN global rather than a merge into `window.__wenilla_env`. The env object
/// outranks the query string in `webenv::lookup`, so folding a hosted default into it would let
/// `config.json` beat a hand-written `?ws=…` — backwards. And `realmlist` must not travel as
/// `host` either: that key marks the session env-pinned, which is a statement about a harness run
/// and not about a site.
///
/// `no-cache`, and it is one of exactly two things a warm repeat visit still puts on the wire.
async function loadConfig(origin) {
  try {
    const r = await fetchRetry(`${origin}/config.json`, { cache: 'no-cache' });
    window.__wenilla_config = r.ok ? await r.json() : {};
  } catch (_) {
    window.__wenilla_config = {};
  }
}

/// `window.__wenilla_progress(stage)` — the client's calls land here (webprogress.rs).
///
/// `ready` also starts the SECOND prefetch: the world-entry pack set. Entering the world used
/// to be one frozen frame of ~200 synchronous sprite reads (184 `Interface\…\*.blp` on the
/// first UI draw, plus a few quest-marker models and zone sounds) — 206 s on a cold 100 ms-RTT
/// cache. That set is location-independent and, being almost all `Interface\`, collapses to a
/// handful of packs; it is warmed while the player is at the glue screens, with a corner line so
/// it is visible but never in the way. Whatever is still in flight when they click Enter World
/// just loads the slow way.
function installProgressHook(ui, skipPrefetch) {
  window.__wenilla_progress = (stage) => {
    if (stage === 'startup') ui.setStage('Loading interface…');
    else if (stage === 'ready') {
      ui.hide();
      // A trace capture ends at the first glue screen, which is exactly what the BOOT read set is.
      // Posting it here rather than at unload is deliberate: `beforeunload` is not guaranteed to
      // run a fetch to completion, and a capture that is sometimes lost is a capture nobody trusts.
      // Ask again later (`__wenilla_post_boottrace()` from the console) to fold in the world set.
      postBootTrace();
      if (!skipPrefetch) {
        const corner = createCornerLine();
        prefetchWorld((done, total) =>
          corner.set(total > 0 && done < total ? `preparing world data — ${mb(done)}/${mb(total)} MB` : '')
        )
          .catch(() => {})
          .finally(() => corner.remove());
      }
    }
  };
}

/// Hand the boot trace to `wenilla-host --log-reads`, which writes `web/boot-manifest.json` from it.
///
/// The alternative this replaces is a person opening devtools, deduplicating an array by eye and
/// pasting it into a file — which works, and which nobody does often enough to keep the manifest
/// fresh. The host has been able to record a read set from its per-name `/data/<name>` route for a
/// while; what it could never do is record one against a CONTAINER, where it sees pack objects and
/// a pack id names a few hundred files at once. This closes that: the client already knows every
/// name it asked for and in what order, so it sends the list.
///
/// A no-op unless `?boottrace=1` armed the array, and a failed POST is silent by design — a host
/// that was not started with `--log-reads` answers 404, which is the ordinary case for every run
/// that is not a capture.
///
/// Exposed as `window.__wenilla_post_boottrace()` so a session can post again after walking into
/// the world; `ReadLog` merges in first-need order, so a second post appends what is new.
function postBootTrace() {
  const names = window.__wenilla_boottrace;
  if (!Array.isArray(names) || !names.length) return Promise.resolve(false);
  // Deduplicated HERE, keeping first-need order, so the host receives the manifest's own shape and
  // the two sides cannot disagree about what "first need" meant.
  const seen = new Set();
  const ordered = [];
  for (const n of names) {
    const key = String(n).replace(/\//g, '\\').toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    ordered.push(n);
  }
  const hd = window.__wenilla_boottrace_hd;
  if (Array.isArray(hd)) {
    console.log(
      `wenilla: boot trace — ${ordered.length} names, ${new Set(hd).size} of them served by the ` +
        `updated-character-model layer`,
    );
  }
  return fetch('/__boottrace', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(ordered),
  })
    .then((r) => {
      if (r.ok) console.log(`wenilla: posted ${ordered.length} names to the host's read log`);
      return r.ok;
    })
    .catch(() => false);
}
window.__wenilla_post_boottrace = postBootTrace;

const mb = (n) => (n / (1024 * 1024)).toFixed(0);

/// The unobtrusive bottom-left status line the world prefetch reports on.
function createCornerLine() {
  const el = document.createElement('div');
  el.id = 'wenilla-corner';
  el.style.cssText =
    'position:fixed;left:.6rem;bottom:.5rem;z-index:15;font:12px system-ui,sans-serif;' +
    'color:#888;background:rgba(0,0,0,.55);padding:.2rem .5rem;border-radius:4px;' +
    'pointer-events:none;transition:opacity .4s ease';
  document.body.appendChild(el);
  return {
    set(text) {
      el.textContent = text;
      el.style.opacity = text ? '1' : '0';
    },
    remove() {
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 500);
    },
  };
}

/// The overlay itself. Same dark theme as the pages' own chrome; no dependencies, no images.
function createBootUi() {
  const root = document.createElement('div');
  root.id = 'wenilla-boot';
  root.innerHTML = `
    <style>
      #wenilla-boot { position: fixed; inset: 0; z-index: 20; display: grid; place-items: center;
        background: #000; color: #ccc; font: 15px system-ui, sans-serif;
        transition: opacity .4s ease; }
      #wenilla-boot.gone { opacity: 0; pointer-events: none; }
      #wenilla-boot .box { width: min(420px, 80vw); }
      #wenilla-boot .stage { margin-bottom: .6rem; text-align: center; }
      #wenilla-boot .bar { height: 6px; background: #222; border-radius: 3px; overflow: hidden; }
      #wenilla-boot .bar > i { display: block; height: 100%; width: 0; background: #c9a227;
        border-radius: 3px; transition: width .15s linear; }
      #wenilla-boot .bar.pulse > i { width: 30% !important; animation: wboot-slide 1.2s ease-in-out infinite; }
      @keyframes wboot-slide { 0% { margin-left: 0 } 50% { margin-left: 70% } 100% { margin-left: 0 } }
      #wenilla-boot .data { margin-top: .5rem; text-align: center; font-size: 12px; color: #888;
        min-height: 1.2em; }
      #wenilla-boot .err { color: #e07070; text-align: center; }
    </style>
    <div class="box">
      <div class="stage">Loading…</div>
      <div class="bar"><i></i></div>
      <div class="data"></div>
    </div>`;
  document.body.appendChild(root);
  const stageEl = root.querySelector('.stage');
  const barEl = root.querySelector('.bar');
  const fillEl = root.querySelector('.bar > i');
  const dataEl = root.querySelector('.data');
  let gone = false;
  let timeout = 0;
  return {
    setWasmProgress(got, total) {
      if (gone) return;
      if (total > 0) {
        const pct = Math.min(99, Math.floor((got / total) * 100));
        const done = got >= total;
        stageEl.textContent = done
          ? 'Downloading client — done'
          : `Downloading client — ${pct}% (${mb(got)} MB)`;
        fillEl.style.width = (done ? 100 : pct) + '%';
      } else {
        stageEl.textContent = `Downloading client — ${mb(got)} MB`;
        barEl.classList.add('pulse');
      }
    },
    /// Bytes, not files: a pack is ~0.6 MB, so "12 of 31" would say much less than "8 of 19 MB".
    setData(done, total) {
      if (gone) return;
      dataEl.textContent = total > 0 ? `game data — ${mb(done)}/${mb(total)} MB` : '';
    },
    setStage(text) {
      if (gone) return;
      stageEl.textContent = text;
      barEl.classList.add('pulse');
    },
    fail(text) {
      clearTimeout(timeout);
      barEl.remove();
      dataEl.remove();
      stageEl.className = 'err';
      stageEl.textContent = text;
    },
    armTimeout(ms) {
      timeout = setTimeout(() => this.hide(), ms);
    },
    hide() {
      if (gone) return;
      gone = true;
      clearTimeout(timeout);
      root.classList.add('gone');
      setTimeout(() => root.remove(), 500);
    },
  };
}
