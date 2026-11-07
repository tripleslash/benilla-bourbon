// The site's split files, read back as one: the one reader of the `<stem>.parts.json` layout
// `scripts/split-parts.sh` writes, `{"bytes":N,"parts":["<stem>.part00<suffix>",...]}` (further
// keys skipped). A static host refuses any static asset over 25 MiB, so the wasm module
// (`boot.js`'s `fetchWasm`) and the desktop client (`platform.js`'s `installDesktopDownload`) are
// hosted as 24 MiB parts. The desktop client's own self-update reads the same manifest natively
// (`crates/benilla-desktop/src/update.rs`).

import { fetchRetry } from './packs.js';

/// The manifest at `url` -> `{bytes, parts, ...}`, or null when it is absent or not a manifest.
///
/// A 404 is an answer, not an error, and is not logged: an unsplit `web/dist` (`trunk serve`, the
/// native web host's `--www`) and a site without the desktop client both serve none. Anything
/// else that fails also answers null, so the caller's own path fails naming its real URL.
export async function fetchPartsManifest(url, init) {
  try {
    const r = await fetchRetry(url, init);
    if (!r.ok) return null;
    const m = await r.json();
    const parts = Array.isArray(m.parts) && m.parts.length > 0;
    return parts && Number.isSafeInteger(m.bytes) && m.bytes > 0 ? m : null;
  } catch {
    return null;
  }
}

/// Every part `manifest` names, resolved against `url` (the manifest's own), joined into ONE
/// streamed Response of `type`, with `onBytes(got, total)` per chunk. `init` goes to every fetch.
///
/// The parts are all requested at once (in turn would pay a round trip per part) and read in
/// order, so the consumer cannot tell how many responses carried the bytes: the browser still
/// compiles a wasm module while it downloads, exactly as from one file.
export async function readParts(url, manifest, { init, type, onBytes = () => {} } = {}) {
  const responses = await Promise.all(
    manifest.parts.map((name) => fetchRetry(new URL(name, url), init))
  );
  return joinResponses(responses, manifest.bytes, type, onBytes);
}

/// `responses`' bodies as one counted Response of `type`; `total` is what `onBytes` reports against
/// until the end, when it reports the true count. A failed response is handed back as it is, so
/// the caller surfaces its real status.
export function joinResponses(responses, total, type, onBytes = () => {}) {
  const failed = responses.find((r) => !r.ok || !r.body);
  if (failed) return failed;

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
        // The end of a part is not the end of the file: step to the next reader and keep pulling,
        // so the consumer never sees the seam. One `pull` must enqueue something or close, and an
        // empty part must not do neither, hence the loop.
        if (++i >= responses.length) {
          onBytes(got, got); // the honest total, now that it is known
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
        // directly; left open they would hold their connections for a read that has stopped.
        ...responses.slice(i + 1).map((r) => r.body.cancel(reason)),
      ]);
    },
  });
  // Content-Type is SET, not copied: instantiateStreaming refuses anything but `application/wasm`,
  // and this Response is ours, so it carries neither an origin's extension mapping nor the first
  // part's Content-Length, which no longer describes its body.
  return new Response(counted, {
    status: responses[0].status,
    statusText: responses[0].statusText,
    headers: { 'content-type': type || 'application/octet-stream' },
  });
}
