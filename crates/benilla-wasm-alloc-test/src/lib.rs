//! fork: web — `benilla-app`'s `wasm_alloc` compiled alone, over its native model of linear memory,
//! so its tests build in seconds: `cargo test -p benilla-wasm-alloc-test`.

#[cfg(test)]
#[path = "../../benilla-app/src/wasm_alloc.rs"]
mod wasm_alloc;
