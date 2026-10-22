//! The browser's half of print screen: the tab has no writable folder (`std::fs` answers
//! "operation not supported on this platform"), so the PNG is handed to the page as a download
//! and lands wherever the browser saves files, by default the user's Downloads folder.

use wasm_bindgen::prelude::*;

#[wasm_bindgen(inline_js = r#"
export function download(name, bytes) {
    // `bytes` is a view into wasm memory, which the threaded build shares, and a Blob refuses a
    // shared view: `slice` copies it into an unshared buffer before anything can grow that memory.
    const url = URL.createObjectURL(new Blob([bytes.slice()], {type: 'image/png'}));
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
"#)]
extern "C" {
    pub(crate) fn download(name: &str, bytes: &[u8]);
}
