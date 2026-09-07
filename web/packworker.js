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

self.onmessage = (event) => {
  const msg = event.data;
  if (!msg) return;
  switch (msg.type) {
    case 'open':
      open(msg);
      break;
    case 'fetch':
      run(msg.id, () => fetchPack(msg.id));
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
    container = new wasm_bindgen.Container(msg.base, new Uint8Array(msg.index));
    self.postMessage({
      type: 'opened',
      buildId: container.buildId(),
      packCount: container.packCount(),
    });
  } catch (e) {
    container = null;
    self.postMessage({ type: 'failed', error: String(e && e.message ? e.message : e) });
  }
}

/// One pack, start to finish: get the wire bytes however this message said to, decode them, and
/// post the body back with its buffer TRANSFERRED — the page adopts the bytes without a copy, and
/// this side loses them, which is exactly right since the resident map is the only owner there
/// should be.
///
/// A failure is reported as a `miss`, not thrown: the page's read path can still fetch that pack
/// the ordinary way, and one unreadable object must not take the worker down with it.
async function run(id, source) {
  try {
    if (!container) throw new Error('no container open');
    const wire = await source();
    const body = container.decode(id, wire);
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
async function fetchPack(id) {
  const url = container.packUrl(id);
  if (!url) throw new Error(`pack ${id} is not in this container`);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}
