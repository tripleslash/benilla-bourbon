//! Does this host honour HTTP `Range`? A **measurement**, held as a mutable policy, not a constant.
//!
//! The probe itself runs in `web/packs.js` before `init()` — it asks for `bytes=0-15` of a pack the
//! boot prefetch needs anyway, so it costs no extra round trip and the response is kept whichever
//! way it comes back. What lives here is Rust's side: reading the answer, and the two corrections
//! that keep it honest afterwards.
//!
//! The corrections matter because the answer is not stable. A static host's behaviour varies *by cache
//! state*: a cold `MISS` answers a range request with a full 200, a warm `HIT` on a stored object
//! can answer 206. So a single probe would be right about the object it probed and wrong about the
//! next one. Instead the policy degrades and recovers on evidence:
//!
//! - **Down**, here: a ranged request answered 200/full calls [`demote`], and the full body is
//!   *used* rather than discarded — it is a response the client has already paid for.
//! - **Up**, in `packs.js`: the probe re-runs once after the boot prefetch, because "the edge has
//!   now stored these objects" is precisely the state in which a static host starts answering 206.
//!   Promotion belongs there rather than here for a structural reason: a `whole`-policy read never
//!   issues a range request, so Rust never gets the chance to observe a 206 on its own.
//!
//! Both directions are safe because [`Policy::Whole`] is never *wrong*, only slower: it fetches an
//! object the client would otherwise have sliced, and warms that object's whole locality group in
//! the process. Which is why the container's design does not depend on the answer either way.

/// How the client asks for a blob it does not already hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Ask for exactly the frame's byte range. Only ever used on a `STORE`-bodied pack, where a
    /// frame is a contiguous run of the served object.
    Range,
    /// Fetch the whole object and keep its decoded body resident. The path that runs on a static host
    /// today.
    Whole,
}

/// The property the page publishes its probe result on.
const KEY: &str = "policy";

/// The page's current answer. Defaults to [`Policy::Whole`] whenever the page has not said —
/// including a page with no `packs.js` at all — because guessing `Range` costs a wasted round trip
/// per read on a host that does not do them, and guessing `Whole` costs nothing but bandwidth.
pub fn policy() -> Policy {
    match super::resident::property(KEY).and_then(|v| v.as_string()) {
        Some(s) if s == "range" => Policy::Range,
        _ => Policy::Whole,
    }
}

/// Step down to whole-object fetches: a range request just came back as a full 200. Written back to
/// the page, so the async lane and the prefetcher see the same demotion this read did.
pub fn demote() {
    super::resident::set_property(KEY, &wasm_bindgen::JsValue::from_str("whole"));
}
