// sw.js — the container's PERMANENCE layer: every `/data` object is fetched from the origin at
// most once, ever, and answered out of Cache Storage for the life of the install.
//
// ## Why a service worker and not more code in packs.js
//
// `packs.js` already had a Cache Storage path, and it closed half the problem: the async lane
// (`benilla-assets`' AssetReader) could read it. The other half could not, and the plan said so in
// as many words — "Cache Storage is async-only, so the sync miss path cannot read it. It is a
// source for the prefetcher, not for `Chain::read`." That asymmetry is the hole: the ~60
// synchronous `Chain::read` call sites go through `benilla_formats::web::fetch_sync_range`, a
// blocking `XMLHttpRequest`, and a blocking XHR cannot await a Promise. So every one of those reads
// went to the network on a cold resident map, and leaned on the HTTP cache — a store the browser
// evicts on its own schedule and revalidates when it feels like it — to not actually be a server
// hit.
//
// A service worker is the one place in the platform that closes it, because it sits BELOW the
// distinction: a synchronous XHR from a controlled document dispatches a `fetch` event exactly like
// `fetch()` does, and `respondWith` can answer it from an async source. The synchronous caller
// blocks; the worker awaits; the bytes come from disk. One interception point, both lanes, no
// change to the ~60 call sites and no change to the sync XHR that serves them.
//
// ## What that buys, beyond "faster"
//
// - **One origin hit per object, for the life of the install.** Not "usually cached" — cached, by a
//   store with an eviction policy we asked to opt out of (`navigator.storage.persist()`), keyed by
//   the container's build id so a re-pack invalidates cleanly instead of mixing two builds.
// - **Ranges stop involving the origin at all.** A `Range` request for a cached pack is sliced here
//   and answered 206. For an uncached one the worker fetches the WHOLE object, stores it, and
//   slices that — so a ranged read is still one origin hit per *pack*, never one per *blob*. Before
//   this, a 206 was the one response nothing kept: `packs.js::fetchFrame` cached only the full-body
//   case, so under `range` policy every read of an unheld pack went to the network, forever.
// - **The offline session.** Once a load's working set is here, the game data path needs no origin.
//
// ## What deliberately stays live
//
// `data/boot-packs.json` — the version pointer. It carries `build_id`, which is what tells the page
// whether the corpus on disk is still the corpus the origin is serving, so it is network-first with
// a cache fallback: ~10 KB per load when online, and a fully local session when not. Everything
// that is actually game data — `index.bpi`, `listfile.txt`, and all ~4,700 `p/<hi>/<id>.bpk` — is
// cache-first and permanent.
//
// ## Scope
//
// `/data/`, `/data-hd/`, `/data-turtle/` and `/data-reforged/` only. Everything else — the page, `wenilla.js`, the 77 MB wasm, and `/config.json`
// (the gateway address and the realmlist, which must never be held: re-uploading it is how a client
// gets repointed) — is not passed to `respondWith` at all, so it keeps its ordinary `no-cache`
// behaviour and a rebuilt bundle still lands on a reload. A service worker that made the dev loop
// sticky would be trading it for the wrong thing. `?nosw=1` on the page unregisters this entirely
// (see `packs.js`), which is how the pre-worker path — and `wenilla-host --no-ranges`, the
// Range-less host emulation — stays exercisable.

/// One cache for the container. Not namespaced by build id in its NAME: the id is not known when
/// the worker starts, and a fetch can arrive before the page has said. Instead the id is stored
/// inside as a marker and the whole cache is dropped when it changes — see `syncBuild`.
const CACHE = 'wenilla-data';
/// The marker's key. A URL under the origin so it is a legal Request, and one nothing ever fetches.
const BUILD_KEY = '/__wenilla_build_id';
/// The caches `packs.js` used to write before this worker existed. Dropped on activate: with the
/// worker in control the page stops writing its own, and leaving them would hold a second copy of
/// the corpus against the same quota.
const LEGACY_PREFIX = 'wenilla-packs-';

/// Stop storing at this fraction of the origin's quota. Going over does not fail loudly — the
/// browser evicts something, possibly the packs just written — so headroom is how the cache stays
/// useful rather than thrashing. Same number `packs.js` used, same reason.
const QUOTA_CEILING = 0.8;

const DATA_PREFIX = '/data/';
/// The optional updated-character-model layer's tree, laid out identically beside the corpus. Twin
/// of `benilla_formats::layers::HD_DATA_SEGMENT`.
///
/// Cached on the SAME terms and in the same cache, which is the answer to the question this prefix
/// raises: the layer is ~1.5 GiB of `immutable` objects whose names are their content, exactly like
/// the corpus, so nothing about the permanence argument changes for them. What differs is that a
/// player can turn the layer off — and the cache is then simply never asked for those names again.
/// Sweeping them at that moment would be the wrong instinct: turning it back on is one click, and
/// the alternative to a stale-but-unread 1.5 GiB is re-downloading it.
///
/// The layer's own `boot-packs.json` carries its own build id, and `syncBuild` is keyed on the
/// CORPUS's — so a re-pack of the layer alone does not purge here. That is safe for the same reason
/// the corpus's `p/*` rule is: a pack's URL names its content, so a rebuilt layer either produces
/// the same object at the same id or a different id nothing holds.
const HD_DATA_PREFIX = '/data-hd/';
/// The Turtle WoW 1.18.1 layer's tree, on the HD layer's terms and for the same reasons: immutable
/// objects whose names are their content, fetched only while the client is on a 1.18.1 realm and
/// kept for the next visit to one. Twin of `benilla_formats::layers::TURTLE_DATA_SEGMENT`.
const TURTLE_DATA_PREFIX = '/data-turtle/';
/// The reforged art layer's tree, on the Turtle layer's terms: fetched only while the client is on a
/// 1.18.1 realm with the setting on. Twin of `benilla_formats::layers::REFORGED_DATA_SEGMENT`.
const REFORGED_DATA_PREFIX = '/data-reforged/';
/// Network-first with a cache fallback: the build-id pointer.
const POINTER = 'boot-packs.json';

self.addEventListener('install', () => {
  // Take over on the first load rather than the second. Nothing here is version-sensitive across
  // workers — the cache is content, and `syncBuild` is what decides whether it is still valid — so
  // there is no half-updated state a wait would be protecting.
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      await self.clients.claim();
      const names = await caches.keys();
      await Promise.all(
        names.filter((n) => n.startsWith(LEGACY_PREFIX)).map((n) => caches.delete(n))
      );
    })()
  );
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  const prefix = [DATA_PREFIX, HD_DATA_PREFIX, TURTLE_DATA_PREFIX, REFORGED_DATA_PREFIX].find((p) =>
    url.pathname.startsWith(p)
  );
  if (!prefix) return;
  const name = url.pathname.slice(prefix.length);
  if (name === POINTER) {
    event.respondWith(pointer(url));
    return;
  }
  event.respondWith(permanent(request, url));
});

/// The page tells the worker which container it is talking to, and waits for the answer before it
/// reads `index.bpi` — see `packs.js`'s `syncBuild`. That ordering is the whole invalidation story:
/// a stale cache is dropped BEFORE the first read that could be answered out of it.
self.addEventListener('message', (event) => {
  const data = event.data;
  if (!data || data.type !== 'wenilla-build') return;
  const port = event.ports && event.ports[0];
  event.waitUntil(
    syncBuild(data.buildId).then(
      (purged) => port && port.postMessage({ ok: true, purged }),
      (e) => port && port.postMessage({ ok: false, error: String(e) })
    )
  );
});

/// Compare the container's build id against the one this cache was filled for, and drop the whole
/// cache if they differ. Returns whether it dropped one.
///
/// Whole-cache, not per-entry: a pack's id comes from a plan over the game files, so a re-pack can
/// reassign `p/00/0007.bpk` to different bytes at the same URL. There is no subset of the old cache
/// that is known-good, so keeping any of it would be keeping a hazard to save a download.
async function syncBuild(buildId) {
  if (!buildId) return false;
  const cache = await caches.open(CACHE);
  const marker = await cache.match(BUILD_KEY);
  const seen = marker ? await marker.text() : null;
  if (seen === buildId) return false;
  if (seen !== null) await caches.delete(CACHE);
  const fresh = await caches.open(CACHE);
  await fresh.put(BUILD_KEY, new Response(buildId));
  return seen !== null;
}

/// `boot-packs.json`: the origin's answer when there is one, the last one we saw when there is not.
async function pointer(url) {
  try {
    const response = await fetchRetry(url.href, { cache: 'no-cache' });
    if (response.ok) {
      const buf = await response.arrayBuffer();
      await store(url.pathname, buf, response.headers, 'no-cache');
      return body(buf, response.headers, 'no-cache');
    }
  } catch (_) {
    /* offline; the cached pointer below is exactly what this fallback is for */
  }
  const cached = await (await caches.open(CACHE)).match(url.pathname);
  return cached || new Response('boot-packs.json unavailable', { status: 504 });
}

/// Game data: the cache if it is there, one origin fetch if it is not, and never a second one.
///
/// The two shapes are kept apart on purpose. A plain GET returns the stored `Response` directly —
/// no `arrayBuffer()`, so a 9 MB pack is handed to the caller without a copy through the worker's
/// heap. Only a ranged GET pays to materialise the body, and a pack is 655 KiB at the container's
/// measured p50.
async function permanent(request, url) {
  const key = url.pathname;
  const range = request.headers.get('range');
  const cache = await caches.open(CACHE);
  const hit = await cache.match(key);

  if (!range) {
    if (hit) return hit;
    try {
      const { buf, headers } = await fetchOnce(key, url.href);
      return body(buf, headers);
    } catch (e) {
      return new Response(String(e), { status: 502 });
    }
  }

  let buf;
  if (hit) {
    buf = await hit.arrayBuffer();
  } else {
    // A range request for an object we do not have becomes a WHOLE-object fetch that is then
    // sliced. It costs the over-fetch once and buys every later frame in that pack for free —
    // the same bargain the whole-object read path makes, and the reason a ranged session is one
    // origin hit per pack rather than one per blob.
    try {
      buf = (await fetchOnce(key, url.href)).buf;
    } catch (e) {
      return new Response(String(e), { status: 502 });
    }
  }
  return slice(buf, range);
}

/// In-flight fetches, keyed by path. A boot is a burst of reads across a handful of packs, and
/// several of them land in the same object before the first response returns; without this each
/// would be its own download of the same bytes.
///
/// Resolves to the BYTES rather than to a `Response`, because a `Response` body can be read once
/// and the waiters need independent copies.
const inflight = new Map();

function fetchOnce(key, href) {
  let pending = inflight.get(key);
  if (pending) return pending;
  pending = (async () => {
    // The default cache mode on purpose: `/data/p/*` is served `immutable`, so a browser that still
    // holds the object in its HTTP cache answers this without a network round trip, and we get to
    // promote it into permanent storage for free.
    const response = await fetchRetry(href);
    if (!response.ok) throw new Error(`${key}: HTTP ${response.status}`);
    const buf = await response.arrayBuffer();
    await store(key, buf, response.headers);
    return { buf, headers: response.headers };
  })();
  inflight.set(key, pending);
  // Registered on the promise rather than written inside it: the entry must be dropped whether or
  // not the fetch threw, and only once every waiter already holds the promise.
  pending.catch(() => {}).finally(() => inflight.delete(key));
  return pending;
}

/// Put the bytes in the cache, quota permitting. Never throws: a full disk means the next load
/// re-fetches, which is the behaviour of a page with no worker at all.
async function store(key, buf, headers, cacheControl) {
  if (!(await roomFor(buf.byteLength))) return;
  try {
    const cache = await caches.open(CACHE);
    await cache.put(key, body(buf, headers, cacheControl));
  } catch (_) {
    /* quota, or a browser refusing to store; the network path is still correct */
  }
}

/// A `Response` over `buf` with headers safe to replay.
///
/// The origin's headers are NOT copied wholesale, and one of them is the reason: `Content-Encoding`.
/// `arrayBuffer()` hands back the DECODED body, so replaying `content-encoding: br` would tell the
/// consumer to brotli-decode bytes that are already plain — a silent corruption rather than an
/// error. `Content-Length` has the same problem in the other direction (it counts wire bytes), so
/// both are rebuilt from what is actually held.
function body(buf, headers, cacheControl) {
  const type = (headers && headers.get('content-type')) || 'application/octet-stream';
  return new Response(buf, {
    status: 200,
    headers: {
      'Content-Type': type,
      'Content-Length': String(buf.byteLength),
      'Cache-Control': cacheControl || 'public, max-age=31536000, immutable',
      'Accept-Ranges': 'bytes',
      'X-Wenilla-Source': 'sw',
    },
  });
}

/// Answer `Range` out of bytes already held. 206 with a `Content-Range`, or 416 for a range the
/// object cannot satisfy — the two answers a caller can act on.
///
/// This is also why the worker never stores a 206: `Cache.put` rejects one outright (the spec
/// refuses a partial response as a cache entry, and rightly — it is not the resource). So the
/// worker fetches whole objects and synthesises every partial itself.
function slice(buf, header) {
  const total = buf.byteLength;
  const parsed = parseRange(header, total);
  if (!parsed) {
    return new Response(null, { status: 416, headers: { 'Content-Range': `bytes */${total}` } });
  }
  const [start, end] = parsed;
  return new Response(buf.slice(start, end + 1), {
    status: 206,
    headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Length': String(end - start + 1),
      'Content-Range': `bytes ${start}-${end}/${total}`,
      'Cache-Control': 'public, max-age=31536000, immutable',
      'Accept-Ranges': 'bytes',
      'X-Wenilla-Source': 'sw',
    },
  });
}

/// `bytes=start-end`, `bytes=start-` and `bytes=-suffix`, inclusive as HTTP wants it. The client
/// only ever sends the first form (`benilla_pack::Located::wire_range`), but a half-implemented
/// Range parser is the kind of thing that is right until something else asks.
function parseRange(header, total) {
  const m = /^bytes=(\d*)-(\d*)$/.exec(String(header).trim());
  if (!m || (m[1] === '' && m[2] === '')) return null;
  let start;
  let end;
  if (m[1] === '') {
    const suffix = Number(m[2]);
    if (!suffix) return null;
    start = Math.max(0, total - suffix);
    end = total - 1;
  } else {
    start = Number(m[1]);
    end = m[2] === '' ? total - 1 : Number(m[2]);
  }
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  if (start > end || start >= total) return null;
  return [start, Math.min(end, total - 1)];
}

/// Sampled quota accounting. `navigator.storage.estimate()` per object would be a syscall per pack,
/// so it is re-read every 64 stores and accounted locally in between.
let quota = null;

async function roomFor(bytes) {
  if (!navigator.storage || !navigator.storage.estimate) return true;
  if (quota === null || quota.n > 64) {
    try {
      const e = await navigator.storage.estimate();
      quota = { usage: e.usage || 0, quota: e.quota || 0, n: 0 };
    } catch (_) {
      quota = { usage: 0, quota: 0, n: 0 };
    }
  }
  quota.n += 1;
  quota.usage += bytes;
  return !quota.quota || quota.usage < quota.quota * QUOTA_CEILING;
}

// ---- transient-failure retry -------------------------------------------------------------
//
// A copy of `packs.js`'s helper; see the note there for why there are three. With this worker in
// control EVERY `/data` request in the session passes through `fetchOnce`, so this is the single
// most load-bearing of the three copies.
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
