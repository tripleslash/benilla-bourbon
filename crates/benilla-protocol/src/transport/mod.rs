//! The one connection the protocol talks through — a byte pipe with a native and a browser body.
//!
//! Everything above this module (auth, the world handshake, the reader/writer halves) used to name
//! [`std::net::TcpStream`] directly, which is the single thing a browser cannot give us: a page has
//! no sockets, only WebSockets, and a WebSocket delivers its bytes through the JS event loop rather
//! than out of a blocking `read`. So the socket is behind [`Conn`], with one body per target
//! ([`native`] over `TcpStream`, `web` over `web_sys::WebSocket`), and the shape of the seam is set
//! by what the browser can honour:
//!
//! - **Reads are `async`** ([`ReadExactAsync`]). On the web a read cannot block — the frame it is
//!   waiting for arrives on the very event loop a block would be sitting on. Natively the future
//!   resolves immediately around a blocking `read_exact`, which is correct because the native
//!   sequencer runs on its own thread under `block_on`: same syscall, same thread, same behaviour
//!   as before this seam existed.
//! - **Writes stay synchronous** ([`std::io::Write`]). `WebSocket.send` buffers and returns; there
//!   is nothing to await. That keeps every one of the writer's ~200 verbs a plain sync call.
//!
//! `Conn` is pre-split (both directions on one value) for the handshake, and [`Conn::split`] hands
//! out the two halves the streaming phase needs: natively two `try_clone`d sockets, on the web two
//! handles onto the one shared `WebSocket`.

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{Conn, ConnReader, ConnWriter};

#[cfg(target_arch = "wasm32")]
pub mod web;
#[cfg(target_arch = "wasm32")]
pub use web::{Conn, ConnReader, ConnWriter};

/// Wait `d` without blocking the task: the sequencer's park races it against its channels.
/// Natively a helper thread sleeps and wakes the future (the sequencer runs under `block_on`, with
/// no reactor to time it); on the web a `setTimeout` resolves it on the event loop.
#[cfg(not(target_arch = "wasm32"))]
pub async fn sleep(d: std::time::Duration) {
    use std::sync::{Arc, Mutex, PoisonError};
    use std::task::{Poll, Waker};
    let state: Arc<Mutex<(bool, Option<Waker>)>> = Arc::default();
    let timer = Arc::clone(&state);
    std::thread::spawn(move || {
        std::thread::sleep(d);
        let mut s = timer.lock().unwrap_or_else(PoisonError::into_inner);
        s.0 = true;
        if let Some(w) = s.1.take() {
            w.wake();
        }
    });
    std::future::poll_fn(|cx| {
        let mut s = state.lock().unwrap_or_else(PoisonError::into_inner);
        if s.0 {
            Poll::Ready(())
        } else {
            s.1 = Some(cx.waker().clone());
            Poll::Pending
        }
    })
    .await;
}

/// [`sleep`] on the page: a `setTimeout` promise. With no window it never resolves, which leaves a
/// race to its other arms.
#[cfg(target_arch = "wasm32")]
pub async fn sleep(d: std::time::Duration) {
    let ms = d.as_millis().min(i32::MAX as u128) as i32;
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        if let Some(w) = web_sys::window() {
            let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// Read exactly `buf.len()` bytes, or fail — [`std::io::Read::read_exact`]'s contract, awaited.
///
/// `async fn` in a trait, so the browser body can suspend on the event loop. That makes the
/// returned futures un-nameable and un-`Send`, which the `async_fn_in_trait` lint warns about; both
/// are fine here and neither is negotiable on the web anyway: the only caller is the sequencer
/// future, which is polled on the thread (native) or the task (web) that owns the connection, and a
/// `web_sys::WebSocket` cannot cross threads to begin with.
#[allow(async_fn_in_trait)]
pub trait ReadExactAsync {
    /// Fill `buf` completely. On end of stream: [`std::io::ErrorKind::UnexpectedEof`]; on an
    /// expired read timeout: [`std::io::ErrorKind::TimedOut`] (or `WouldBlock`, which is what a
    /// native socket's own timeout raises).
    async fn read_exact_async(&mut self, buf: &mut [u8]) -> std::io::Result<()>;
}

/// A byte buffer reads as itself — `std::io::Read for &[u8]`, awaited.
///
/// For the parsers' own tests: a packet built in memory is the only way to pin a wire layout
/// without a server, and moving the reads behind this trait had taken that away (the parsers no
/// longer name `impl Read`). Nothing on either connection path goes through here; it exists so a
/// test can hand `&mut bytes.as_slice()` to the same function the socket path calls.
impl ReadExactAsync for &[u8] {
    async fn read_exact_async(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        std::io::Read::read_exact(self, buf)
    }
}
