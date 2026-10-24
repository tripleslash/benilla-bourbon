//! The resident pack map — **held on the JS heap**, reached from here.
//!
//! Every byte of decoded pack body lives in `web/packs.js`'s `Map<packId, Uint8Array>`, outside
//! wasm linear memory, and this module is the whole of Rust's access to it. Two reasons it is not
//! a `HashMap` in a `OnceLock` like every other cache in this crate:
//!
//! 1. **32-bit address space.** The corpus is 3.3 GiB served. Even a modest working set of packs
//!    would sit inside wasm's 4 GiB linear memory next to the renderer's buffers, and linear memory
//!    only ever grows — a `Vec` freed by Rust is returned to *its* allocator, never to the browser.
//!    On the JS heap the same bytes are ordinary garbage the collector can take back.
//! 2. **Eviction is a policy, not a data structure.** The budget is bytes, not entries, and
//!    `packs.js` is also where the prefetcher and Cache Storage live, so the LRU belongs beside them
//!    rather than split across the wasm boundary.
//!
//! The page contract is `window.__wenilla_packs`, the same shape as every other page seam this
//! client uses (`__wenilla_env`, `__wenilla_progress`, `__wenilla_boottrace`): a plain global
//! object, no wasm-bindgen module import, so the page can ship it as an ordinary script. **A page
//! that ships none still runs** — the resident map is then always empty and the async lane falls
//! back to a plain `fetch` — because degrading to "correct but slow" is what keeps `wenilla-host`
//! serving a bare `web/dist` a usable way to bisect a rendering bug.

use js_sys::{Array, Function, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};

/// The page global everything here hangs off.
const GLOBAL: &str = "__wenilla_packs";

/// The bridge's functions, resolved once. wasm is single-threaded, so a `thread_local` is simply a
/// global with the right lifetime — and holding `JsValue`s in the `PackStore` itself is not an
/// option, since that would make `Chain` neither `Send` nor `Sync` and the Bevy `AssetReader`
/// requires both.
struct Bridge {
    slice: Option<Function>,
    has: Option<Function>,
    put: Option<Function>,
    warm_pack: Option<Function>,
    can_warm: Option<Function>,
    prefetch: Option<Function>,
    fetch_wire: Option<Function>,
    fetch_frame: Option<Function>,
    obj: JsValue,
}

thread_local! {
    static BRIDGE: std::cell::OnceCell<Option<Bridge>> = const { std::cell::OnceCell::new() };
}

fn with_bridge<T>(f: impl FnOnce(&Bridge) -> T) -> Option<T> {
    BRIDGE.with(|cell| {
        let bridge = cell.get_or_init(|| {
            let window = web_sys::window()?;
            let obj = Reflect::get(&window, &GLOBAL.into()).ok()?;
            if !obj.is_object() {
                return None;
            }
            let method = |name: &str| {
                Reflect::get(&obj, &name.into())
                    .ok()
                    .and_then(|v| v.dyn_into::<Function>().ok())
            };
            Some(Bridge {
                slice: method("slice"),
                has: method("has"),
                put: method("put"),
                warm_pack: method("warmPack"),
                can_warm: method("canWarm"),
                prefetch: method("prefetch"),
                fetch_wire: method("fetchWire"),
                fetch_frame: method("fetchFrame"),
                obj,
            })
        });
        bridge.as_ref().map(f)
    })
}

/// Read a property off the bridge object (the policy string, the prefetched index, the listfile).
pub(super) fn property(name: &str) -> Option<JsValue> {
    with_bridge(|b| Reflect::get(&b.obj, &name.into()).ok())
        .flatten()
        .filter(|v| !v.is_undefined() && !v.is_null())
}

/// Set a property on the bridge object — how a policy demotion reaches JS.
pub(super) fn set_property(name: &str, value: &JsValue) {
    with_bridge(|b| {
        let _ = Reflect::set(&b.obj, &name.into(), value);
    });
}

/// One blob's frame out of a resident pack, or `None` if the pack is not resident.
///
/// `off`/`len` address the pack's **decoded body**, which is what [`put`] stored — so for a SOLID
/// pack the zstd stream was undone once, at absorb time, and every later read of anything in that
/// pack is this memcpy. That asymmetry is the whole reason solid packs earn their inflexibility.
pub fn slice(pack_id: u32, off: u32, len: u32) -> Option<Vec<u8>> {
    with_bridge(|b| {
        let f = b.slice.as_ref()?;
        let got = f
            .call3(
                &b.obj,
                &JsValue::from_f64(pack_id as f64),
                &JsValue::from_f64(off as f64),
                &JsValue::from_f64(len as f64),
            )
            .ok()?;
        got.dyn_into::<Uint8Array>().ok().map(|a| a.to_vec())
    })
    .flatten()
}

/// Is this pack's decoded body already on the JS heap? One `Map.has`, no copy — which is what
/// separates it from [`slice`], the cheapest form of which still marshals a frame across the
/// boundary. The prefetch path asks this before spending anything at all.
///
/// `false` on a page with no bridge, which is correct there: nothing is ever resident.
pub fn is_resident(pack_id: u32) -> bool {
    with_bridge(|b| {
        let Some(f) = b.has.as_ref() else {
            return false;
        };
        f.call1(&b.obj, &JsValue::from_f64(pack_id as f64))
            .map(|v| v.is_truthy())
            .unwrap_or(false)
    })
    .unwrap_or(false)
}

/// Hand a decoded pack body to the resident map. Copies into the JS heap, which is the point: the
/// `Vec` this was decoded into is freed on return and the bytes survive on the collectable side.
pub fn put(pack_id: u32, body: &[u8]) {
    with_bridge(|b| {
        let Some(f) = b.put.as_ref() else { return };
        let view = Uint8Array::from(body);
        let _ = f.call2(&b.obj, &JsValue::from_f64(pack_id as f64), &view);
    });
}

/// Ask the page to have a pack **fetched, decoded and made resident** — `web/packworker.js`, the
/// phase-6 Web Worker — and report whether it is now there.
///
/// This is the whole of Rust's side of the worker, and deliberately so: the expensive halves are
/// the network wait and the zstd stream, both of which belong on a thread that is not drawing, and
/// neither needs anything from wasm to run. What comes back is a `Map` entry on the JS heap, which
/// [`slice`] already knows how to read.
///
/// `false` means "not resident — do it yourself", for every reason alike: a page with no worker, a
/// browser with no `Worker`, an object that 404ed, a worker that died. The caller falls back to the
/// path it took before this existed, so the worker is an accelerator and never a dependency.
pub async fn warm(pack_id: u32, url: &str, priority: u8) -> bool {
    let promise = with_bridge(|b| {
        let f = b.warm_pack.as_ref()?;
        let args = Array::of3(
            &JsValue::from_f64(pack_id as f64),
            &JsValue::from_str(url),
            &JsValue::from_f64(f64::from(priority)),
        );
        f.apply(&b.obj, &args)
            .ok()?
            .dyn_into::<js_sys::Promise>()
            .ok()
    })
    .flatten();
    let Some(p) = promise else { return false };
    wasm_bindgen_futures::JsFuture::from(p)
        .await
        .map(|v| v.is_truthy())
        .unwrap_or(false)
}

/// Could a warm of this pack be expected to land? The **precondition of the non-blocking lane**.
///
/// `false` means there is no accelerator here — no decode worker on this page, or this pack has
/// already failed one — and a caller that answered "not yet" on that basis would be telling its
/// own caller to retry something that will never arrive: an icon that never appears, a sound that
/// never plays. So [`super::PackStore::try_read_sync`] asks first and falls back to the blocking
/// read when the answer is no, which is the path it took before the lane existed.
///
/// This is the decode worker's "accelerator, never a dependency" law, stated for the one caller
/// that would otherwise have depended on it.
pub fn can_warm(pack_id: u32) -> bool {
    with_bridge(|b| {
        let Some(f) = b.can_warm.as_ref() else {
            return false;
        };
        f.call1(&b.obj, &JsValue::from_f64(pack_id as f64))
            .map(|v| v.is_truthy())
            .unwrap_or(false)
    })
    .unwrap_or(false)
}

/// Ask for a pack to be warmed and **do not wait** — the non-blocking lane's whole mechanism.
///
/// [`warm`] is the same request with an answer; this is the request without one, for the two
/// callers that have no await to spend: [`super::PackStore::try_read_sync`], which reports
/// `Pending` and lets its caller retry a frame later, and [`super::PackStore::prefetch`], which is
/// a caller saying it will want this before it does.
///
/// Idempotent and cheap to spam. `packs.js`'s `prefetch` returns immediately for a pack that is
/// resident or already in flight, so a UI system asking for the same not-yet-arrived icon on every
/// frame costs two `Map` lookups per frame and no second fetch.
///
/// `priority` is the wait order, not the work: see [`super::warm_priority`]. Re-asking a queued
/// pack at a better priority promotes it in place, which is why spamming this from a lane that has
/// become urgent is the right thing rather than a wasted call.
pub fn prefetch(pack_id: u32, url: &str, priority: u8) {
    with_bridge(|b| {
        let Some(f) = b.prefetch.as_ref() else { return };
        let args = Array::of3(
            &JsValue::from_f64(pack_id as f64),
            &JsValue::from_str(url),
            &JsValue::from_f64(f64::from(priority)),
        );
        let _ = f.apply(&b.obj, &args);
    });
}

/// Fetch a whole `.bpk` asynchronously. Goes through the page when it can, because that is where
/// **Cache Storage** lives — the cross-session pack cache a synchronous XHR structurally cannot
/// read. Falls back to a plain `fetch` of the same URL, so the async lane still works on a page
/// that ships no `packs.js`.
pub async fn fetch_wire(pack_id: u32, url: &str) -> Result<Vec<u8>, String> {
    let promise = with_bridge(|b| {
        let f = b.fetch_wire.as_ref()?;
        f.call2(
            &b.obj,
            &JsValue::from_f64(pack_id as f64),
            &JsValue::from_str(url),
        )
        .ok()?
        .dyn_into::<js_sys::Promise>()
        .ok()
    })
    .flatten();
    match promise {
        Some(p) => {
            let value = wasm_bindgen_futures::JsFuture::from(p)
                .await
                .map_err(js_msg)?;
            // `dyn_into`, not `Uint8Array::new`: the page already resolved a Uint8Array, and the
            // constructor would copy it a second time on the way past.
            let bytes: Uint8Array = value.dyn_into().map_err(|_| {
                "fetchWire resolved to something other than a Uint8Array".to_string()
            })?;
            Ok(bytes.to_vec())
        }
        None => plain_fetch(url).await,
    }
}

/// Fetch one frame's byte range asynchronously, reporting **whether the host actually honoured the
/// range**. A `false` here is not a failure: it means a 200 with the whole object came back, which
/// is what a static host does on a cache miss, and the caller absorbs that body rather than discarding
/// a response it has already paid for.
///
/// Unreachable without the bridge — the policy only leaves `whole` when `packs.js`'s probe says a
/// 206 came back — so this one has no fallback to write.
pub async fn fetch_frame(
    pack_id: u32,
    url: &str,
    start: u64,
    end: u64,
) -> Result<(bool, Vec<u8>), String> {
    let promise = with_bridge(|b| {
        let f = b.fetch_frame.as_ref()?;
        let args = Array::of4(
            &JsValue::from_f64(pack_id as f64),
            &JsValue::from_str(url),
            &JsValue::from_f64(start as f64),
            &JsValue::from_f64(end as f64),
        );
        f.apply(&b.obj, &args)
            .ok()?
            .dyn_into::<js_sys::Promise>()
            .ok()
    })
    .flatten();
    let Some(p) = promise else {
        return Err("no __wenilla_packs.fetchFrame on this page".into());
    };
    let value = wasm_bindgen_futures::JsFuture::from(p)
        .await
        .map_err(js_msg)?;
    let ranged = Reflect::get(&value, &"ranged".into())
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let bytes = Reflect::get(&value, &"bytes".into())
        .ok()
        .and_then(|v| v.dyn_into::<Uint8Array>().ok())
        .ok_or_else(|| "fetchFrame resolved without a bytes array".to_string())?;
    Ok((ranged, bytes.to_vec()))
}

/// The no-page-helper path: one `fetch`, no Cache Storage, no resident bookkeeping beyond what the
/// caller does itself.
async fn plain_fetch(url: &str) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or_else(|| "no window: fetch is browser-only".to_string())?;
    let value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(js_msg)?;
    let resp: web_sys::Response = value
        .dyn_into()
        .map_err(|_| "fetch() did not resolve to a Response".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let buffer = wasm_bindgen_futures::JsFuture::from(resp.array_buffer().map_err(js_msg)?)
        .await
        .map_err(js_msg)?;
    Ok(Uint8Array::new(&buffer).to_vec())
}

fn js_msg(e: JsValue) -> String {
    e.as_string().unwrap_or_else(|| format!("{e:?}"))
}
