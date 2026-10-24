// packworker.js — the pack fetch/decode Web Worker. Phase 6 of the streaming plan, and the whole
// of what "get the container off the frame thread" turned out to need.
//
// ## What was still on the frame thread
//
// `packs.js`'s prefetch already warmed the boot packs before `init()`, but it warmed them as WIRE
// bytes: into the service worker's cache and no further. The decode — `pack_body`'s header check
// and, for a SOLID pack, a zstd stream over ~655 KiB — happened later, in Rust, inside the first
// update's catalog task, once per pack, on the thread that draws. So the pack was "prefetched" and
// the frame still paid for it.
//
// Here that work moves. The worker fetches a pack, hands it to its own small wasm module
// (`crates/wenilla-packdecode`, the same `benilla_pack` reader the client links) and posts the
// DECODED body back as a transferable `ArrayBuffer` — zero copy, straight into the resident map.
// By the time the client's first synchronous `Chain::read` runs, its pack is decoded and resident
// and the read is a memcpy.
//
// **Why this stayed, once threads arrived.** The note here used to say `+atomics` "does not
// compile" — bevy_render enabled wgpu's `fragile-send-sync-non-atomic-wasm`, every wgpu handle went
// `!Send + !Sync` under `+atomics`, and `RenderDevice`/`RenderQueue` are `#[derive(Resource)]`. That
// stopped being true (bevy 0.18.1's `WgpuWrapper`), and the client now has a shared-memory build
// with real worker threads (`scripts/web-build-threads.sh`).
//
// This worker is untouched by that, and it is not redundant. Threads move work that is `Send`;
// fetching a pack is not. The read path awaits `JsFuture`s and reaches `window` — the resident map
// is `window.__wenilla_packs` on the main thread's JS heap — so `TaskPool::spawn` on wasm is still
// the microtask queue and `bevy_asset`'s loads still run on the thread that draws. A worker that
// owns its own `fetch` and its own decoder is how that work gets off the frame, with or without
// atomics. See `third_party/bevy_tasks/BENILLA.md` for where the `spawn`/`spawn_pooled` line falls.
//
// **Why a second wasm module.** The browser cannot decompress zstd — `DecompressionStream` speaks
// gzip and deflate only — and a JS re-implementation of the frame format would be a second copy of
// a law that lives in `benilla-pack`, whose failure mode when it drifts is silent corruption rather
// than an error. So the worker links the same reader, at ~106 KB against the client's 77 MB.
//
// ## The two modes, and why there are two
//
// `fetch`  — the worker fetches the object itself. Used when `sw.js` is in control, because then
//            caching is the service worker's job and the worker's `fetch()` goes through it like
//            any other: the main thread never sees the 655 KiB response at all.
// `decode` — the page fetches and posts the wire bytes here. Used when there is NO service worker
//            (`?nosw=1`, an insecure origin), where `packs.js`'s own Cache Storage is the only
//            persistence there is. Duplicating that cache in here would be a second implementation
//            of the same store racing the first one for the same quota; letting the page keep the
//            fetch costs one transfer and keeps one owner of the cache.
//
// Both modes put the decode — the expensive half — on this thread, which is the point.
//
// A worker that fails to start, fails to load its wasm, or dies mid-session is not an error the
// page has to handle specially: `packs.js` settles every pending warm as "not resident" and the
// read path does exactly what it did before this file existed.

importScripts('./wenilla-packdecode.js');

/// The wasm module, initialising in parallel with the page's own boot. Awaited by the first message
/// that needs it rather than gated behind a "ready" handshake: the page's `open` arrives after two
/// network round trips (`boot-packs.json`, `index.bpi`), so this is finished long before.
const wasmReady = wasm_bindgen({ module_or_path: './wenilla-packdecode_bg.wasm' });

/// The open container — `index.bpi` parsed, plus the base URL. Null until `open` succeeds; every
/// job before that is refused rather than queued, because the page does not send one until it has
/// been told the container is open.
let container = null;
/// The layers' containers, each opened by an `open` naming its `layer`: `hd`, the updated character
/// models (null whenever the player has the option off, the default — see `packs.js::bootstrapHd`),
/// `turtle`, the Turtle WoW 1.18.1 layer (open only while the client is on a 1.18.1 realm, and
/// dropped by a `close` when it leaves one), and `reforged`, the reforged art opened with it.
///
/// A **container per layer**, not more indices in the first: a `.bpk` does not restate its pack's
/// codec or decoded size, so decoding needs the index the pack came from, and every container
/// numbers its packs from 0 independently. `containerFor` is where that is resolved.
const layers = { hd: null, turtle: null, reforged: null };

/// **The key space.** Twins of `benilla_formats::layers::HD_PACK_KEY_BASE` and
/// `TURTLE_PACK_KEY_BASE` and of `packs.js`'s own constants; all must agree. Every id crossing this
/// file is a RESIDENT key, and the container it belongs to plus its id inside that container are
/// recovered here and nowhere else. `2 ** 31`, as a JavaScript `1 << 31` is negative.
const HD_PACK_KEY_BASE = 1 << 30;
const TURTLE_PACK_KEY_BASE = 2 ** 31;
/// Twin of `benilla_formats::layers::REFORGED_PACK_KEY_BASE`; `3 << 30` would be negative here.
const REFORGED_PACK_KEY_BASE = 3 * 2 ** 30;

/// Which container a resident key names, and what its id is inside it.
function containerFor(key) {
  if (key >= REFORGED_PACK_KEY_BASE) return { c: layers.reforged, id: key - REFORGED_PACK_KEY_BASE };
  if (key >= TURTLE_PACK_KEY_BASE) return { c: layers.turtle, id: key - TURTLE_PACK_KEY_BASE };
  if (key >= HD_PACK_KEY_BASE) return { c: layers.hd, id: key - HD_PACK_KEY_BASE };
  return { c: container, id: key };
}

self.onmessage = (event) => {
  const msg = event.data;
  if (!msg) return;
  switch (msg.type) {
    case 'open':
      open(msg);
      break;
    case 'close':
      if (msg.layer in layers) layers[msg.layer] = null;
      break;
    case 'fetch':
      run(msg.id, (id, c) => fetchPack(id, c));
      break;
    case 'decode':
      run(msg.id, () => Promise.resolve(new Uint8Array(msg.wire)));
      break;
  }
};

/// Parse the index the page already fetched. It is a COPY — the page keeps its own for the client's
/// `Chain::open` — because the pack-level codec and decoded size are index fields that a `.bpk`
/// does not restate, so a decoder without the index cannot know how to decode.
///
/// The reply carries the build id back. The page compares it against the one it read out of
/// `boot-packs.json`, which is how a worker that somehow opened a different container is caught at
/// boot rather than by a corrupt texture an hour in.
async function open(msg) {
  try {
    await wasmReady;
    const opened = new wasm_bindgen.Container(msg.base, new Uint8Array(msg.index));
    if (msg.layer in layers) layers[msg.layer] = opened;
    else container = opened;
    self.postMessage({
      type: 'opened',
      layer: msg.layer,
      buildId: opened.buildId(),
      packCount: opened.packCount(),
    });
  } catch (e) {
    // Only the corpus is fatal. A layer that will not open stays null and every warm of its keys
    // falls back to the page's own fetch — the same degradation as no worker at all, which is what
    // the "accelerator, never a dependency" law asks for.
    if (msg.layer in layers) layers[msg.layer] = null;
    else container = null;
    self.postMessage({
      type: 'failed',
      layer: msg.layer,
      error: String(e && e.message ? e.message : e),
    });
  }
}

/// One pack, start to finish: get the wire bytes however this message said to, decode them, and
/// post the body back with its buffer TRANSFERRED — the page adopts the bytes without a copy, and
/// this side loses them, which is exactly right since the resident map is the only owner there
/// should be.
///
/// A failure is reported as a `miss`, not thrown: the page's read path can still fetch that pack
/// the ordinary way, and one unreadable object must not take the worker down with it.
async function run(key, source) {
  const id = key;
  try {
    const { c, id: inner } = containerFor(key);
    if (!c) throw new Error('no container open');
    const wire = await source(inner, c);
    const body = c.decode(inner, wire);
    self.postMessage({ type: 'body', id, body: body.buffer }, [body.buffer]);
  } catch (e) {
    self.postMessage({ type: 'miss', id, error: String(e && e.message ? e.message : e) });
  }
}

/// A whole `.bpk` off the origin. `container.packUrl` rather than a path built here: the
/// "the path IS the id" law lives in `benilla_pack::pack_path` and this is one of the two places
/// allowed to spell it, through the same code rather than beside it.
///
/// A dedicated worker created by a controlled document is itself controlled, so this `fetch` goes
/// through `sw.js` and is answered off disk on every load after the first.
async function fetchPack(id, c) {
  const url = c.packUrl(id);
  if (!url) throw new Error(`pack ${id} is not in this container`);
  const r = await fetchRetry(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

// ---- transient-failure retry -------------------------------------------------------------
//
// A copy of `packs.js`'s helper; see the note there for why there are three. This worker fetches
// packs itself whenever `sw.js` is in control, so a rate-limited burst reaches it directly.
/// How many times a transient failure is re-tried before the caller is told it failed. Four
/// attempts total, so the worst case adds ~1.4 s of backoff before giving up.
const RETRY_ATTEMPTS = 4;
/// First backoff, doubled per attempt: 150, 300, 600 ms, plus jitter.
const RETRY_BASE_MS = 150;
/// The statuses worth asking again for. **429 is the one this exists for** — A static host rate-limits
/// a burst of object requests, and the boot prefetch is exactly a burst of object requests — with
/// the 5xx family alongside it because an edge under load answers those the same way, transiently.
///
/// **404 is deliberately absent, and so is every other 4xx.** A missing pack is a fact about the
/// hosted tree, not a hiccup: `Chain::read` turns it into the correct "file not in patch chain",
/// and retrying it three times would turn one honest error into four and delay it by a second.
const RETRY_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);

/// `fetch` with backoff on the failures a static object store actually produces under load.
///
/// The corpus is a few thousand objects fetched six-wide at boot and again at every zone edge, so
/// "the origin said 429 once" is a normal event in a normal session, not an outage. Before this,
/// one of those became a pack the resident map never got: the read path fell back to a blocking
/// XHR, or the prefetch simply dropped it and the first frame that wanted it paid a round trip —
/// and on a genuinely rate-limited burst, dozens did at once.
///
/// `Retry-After` is honoured when the origin sends one (seconds, or an HTTP date), because an
/// origin that tells you when to come back knows better than the schedule below. It is clamped, so
/// a header saying "an hour" costs one attempt rather than hanging the boot.
///
/// Jittered, because the whole point is that six workers hit the limit at the same instant; a fixed
/// backoff would have all six come back at the same instant too.
async function fetchRetry(url, init, attempts = RETRY_ATTEMPTS) {
  let lastErr;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    let response;
    try {
      response = await fetch(url, init);
    } catch (e) {
      // A rejected fetch is a network-layer failure (offline, DNS, a dropped connection). Those are
      // exactly the transient class, so they retry — but the last one is re-thrown, not swallowed,
      // because a caller that asked for bytes must not be handed `undefined`.
      lastErr = e;
      if (attempt + 1 >= attempts) throw e;
      await sleep(retryDelayMs(attempt, null));
      continue;
    }
    if (!RETRY_STATUS.has(response.status) || attempt + 1 >= attempts) return response;
    await sleep(retryDelayMs(attempt, response.headers.get('Retry-After')));
  }
  // Unreachable: the loop either returns a response or throws on its last attempt.
  throw lastErr || new Error(`fetch(${url}) exhausted ${attempts} attempts`);
}

/// The wait before attempt `n + 1` — `Retry-After` when the origin sent a usable one, else
/// exponential backoff with up to 50% jitter. Capped at 5 s so a hostile header cannot stall a boot.
function retryDelayMs(attempt, retryAfter) {
  const capped = (ms) => Math.max(0, Math.min(5000, ms));
  if (retryAfter) {
    const secs = Number(retryAfter);
    if (Number.isFinite(secs)) return capped(secs * 1000);
    const when = Date.parse(retryAfter);
    if (Number.isFinite(when)) return capped(when - Date.now());
  }
  const base = RETRY_BASE_MS * 2 ** attempt;
  return capped(base + Math.random() * base * 0.5);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
