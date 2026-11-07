//! The pack worker's wasm half — **the container reader, minus the client**.
//!
//! Phase 6 of the streaming plan is "pack fetch + decode off the frame thread". The fetch half a
//! Web Worker gets for free; the decode half needs a zstd decoder *inside* that worker, and the
//! browser has none — `DecompressionStream` speaks gzip and deflate only. So the worker gets its
//! own small wasm module: this crate, which is [`benilla_pack`] and nothing else, at a few hundred
//! kilobytes beside the client's 77 MB.
//!
//! **It is the same reader the client links, not a second one.** A JS re-implementation of the
//! frame format would be a copy of a law that lives in `benilla-pack`, and the failure mode of a
//! copy that drifts is silent corruption — both halves keep "working" on the packs that happen to
//! agree. Here the pack header check, the codec table and the length assertions are literally the
//! same code, so a container this module can read is one `benilla_formats::webpack` can read.
//!
//! ## What the worker asks of it
//!
//! The page hands over `index.bpi` once at boot (a copy — the client keeps its own), because the
//! *pack-level* codec and decoded size are index fields and a `.bpk` does not restate them. From
//! there the worker can name its own URLs and decode whatever it fetches, so the page's side of
//! the protocol is a pack id and nothing more.
//!
//! Note what is deliberately absent: no `locate`, no name hashing, no frame extraction. The worker
//! returns whole decoded pack **bodies**, which is what the resident map holds; picking one blob
//! out of a body is a slice the client already does on the main thread against bytes that are by
//! then in the JS heap.

use benilla_pack::{decode_pack_body, pack_body, Index};
use wasm_bindgen::prelude::*;

/// Route a panic to `console.error` instead of the browser's bare "unreachable executed" — inside
/// a worker there is no other channel for it, and a malformed pack reaching `ruzstd` is exactly
/// the kind of thing that would otherwise arrive as silence.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// One open container: the parsed index plus the base URL its objects hang off.
#[wasm_bindgen]
pub struct Container {
    /// `{origin}/data`, no trailing slash — the same string `benilla_formats::web::data_base`
    /// builds, handed over by the page so this module never has to guess an origin.
    base: String,
    index: Index,
}

#[wasm_bindgen]
impl Container {
    /// Parse `index.bpi`. Fails loudly: a worker holding an index it cannot read has nothing to do,
    /// and the page falls back to decoding on the main thread.
    #[wasm_bindgen(constructor)]
    pub fn new(base: String, index: Vec<u8>) -> Result<Container, JsError> {
        let index = Index::parse(index).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Container { base, index })
    }

    /// The container's build id as hex. The page compares it against the one it read out of
    /// `boot-packs.json`, so a worker that somehow opened a *different* container says so at boot
    /// rather than serving bodies from it.
    #[wasm_bindgen(js_name = buildId)]
    pub fn build_id(&self) -> String {
        format!("{:016x}", self.index.build_id())
    }

    #[wasm_bindgen(js_name = packCount)]
    pub fn pack_count(&self) -> u32 {
        self.index.pack_count()
    }

    /// Where one pack is served. Through [`benilla_pack::pack_path`], so the "the path IS the id"
    /// law stays in the one place that defines it.
    #[wasm_bindgen(js_name = packUrl)]
    pub fn pack_url(&self, pack_id: u32) -> Option<String> {
        self.index.pack(pack_id)?;
        Some(format!(
            "{}/{}",
            self.base,
            benilla_pack::pack_path(pack_id)
        ))
    }

    /// The served size of one pack, for a progress bar that counts bytes rather than objects.
    #[wasm_bindgen(js_name = sizeWire)]
    pub fn size_wire(&self, pack_id: u32) -> u32 {
        self.index.pack(pack_id).map(|p| p.size_wire).unwrap_or(0)
    }

    /// A served `.bpk` in, its decoded body out — the whole reason this module exists.
    ///
    /// Both halves matter. [`pack_body`] is the header check that turns a static host's
    /// 404-as-HTML into a legible error instead of feeding an HTML page to a zstd decoder, and
    /// [`decode_pack_body`] is the SOLID packs' zstd stream, the multi-millisecond piece this is
    /// here to keep off the frame thread. A STORE-coded pack costs one copy and no decode, which
    /// is most of them.
    pub fn decode(&self, pack_id: u32, wire: &[u8]) -> Result<Vec<u8>, JsError> {
        let meta = self
            .index
            .pack(pack_id)
            .ok_or_else(|| JsError::new(&format!("pack {pack_id} is not in this container")))?;
        let body = pack_body(wire, pack_id).map_err(|e| JsError::new(&e.to_string()))?;
        let body = decode_pack_body(body, &meta).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(body.into_owned())
    }
}
