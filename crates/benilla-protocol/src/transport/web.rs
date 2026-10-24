//! [`Conn`] over a browser `WebSocket` — the web body of the transport seam.
//!
//! A page cannot open a TCP socket, so the world and realmd connections ride a WebSocket to the
//! proxy `wenilla-host` runs, which relays the bytes to the real server. The scheme is fixed
//! (and shared with the host): `{ws-origin}/ws/{port}?host=…`, binary frames, one TCP read chunk
//! per frame, either side's close closing the other. `?host=` is carried for the proxy's log only —
//! the upstream it dials is its own `--upstream` flag, so a page cannot aim it at an arbitrary host.
//!
//! **The stream is reassembled here, not trusted frame-by-frame.** WebSocket delivers messages, TCP
//! delivers bytes, and the world protocol's headers straddle whatever chunking the proxy's reads
//! happened to produce — so every inbound frame is appended to one byte queue and
//! [`ReadExactAsync`] takes exactly what the parser asked for out of it.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::future::Future;
use std::io::{self, ErrorKind, Write};
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use send_wrapper::SendWrapper;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{BinaryType, MessageEvent, WebSocket};

use super::ReadExactAsync;

thread_local! {
    /// Overrides the origin the WebSocket URL is built from. Default: the page's own origin, which
    /// is right whenever the host serving the wasm is also the one proxying — the hosting this
    /// was written for. Set it to point a locally-served build at a remote host.
    static WS_BASE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Override the `ws(s)://host[:port]` the connection URL is built on (no trailing slash).
pub fn set_ws_base(base: String) {
    WS_BASE.with(|b| *b.borrow_mut() = Some(base));
}

/// `{ws-origin}/ws/{port}?host={host}` — `wss:` for a page served over `https:`, `ws:` otherwise,
/// because a secure page may not open an insecure socket.
fn connect_url(host: &str, port: u16) -> io::Result<String> {
    let base = match WS_BASE.with(|b| b.borrow().clone()) {
        Some(b) => b,
        None => {
            let location = web_sys::window()
                .ok_or_else(|| io::Error::new(ErrorKind::Unsupported, "no browser window"))?
                .location();
            let protocol = location.protocol().map_err(js_err)?;
            let authority = location.host().map_err(js_err)?;
            let scheme = if protocol == "https:" { "wss" } else { "ws" };
            format!("{scheme}://{authority}")
        }
    };
    let host = String::from(js_sys::encode_uri_component(host));
    Ok(format!("{base}/ws/{port}?host={host}"))
}

/// The bytes the socket has delivered but the protocol has not yet asked for.
#[derive(Default)]
struct Inbox {
    buf: VecDeque<u8>,
    /// The socket closed (cleanly or not). A read that cannot be satisfied ends as `UnexpectedEof`
    /// from here — the same shape a dead TCP socket gives the native body.
    closed: bool,
    /// The reader waiting on more bytes, if any. Woken by `onmessage`/`onclose` and by an expiring
    /// read timeout.
    waker: Option<Waker>,
}

impl Inbox {
    fn wake(&mut self) {
        if let Some(w) = self.waker.take() {
            w.wake();
        }
    }
}

/// The live socket, shared by [`Conn`] and both halves it splits into. The event handlers are kept
/// here because a `Closure` unregisters itself when dropped — dropping them would silently stop the
/// inbound stream.
struct Socket {
    ws: WebSocket,
    inbox: Rc<RefCell<Inbox>>,
    read_timeout: Cell<Option<Duration>>,
    _on_message: Closure<dyn FnMut(MessageEvent)>,
    _on_close: Closure<dyn FnMut(web_sys::CloseEvent)>,
    _on_error: Closure<dyn FnMut(web_sys::Event)>,
}

impl Drop for Socket {
    fn drop(&mut self) {
        // Both halves are gone, so nothing will read this connection again — close it rather than
        // leaving the proxy holding an upstream socket for a session no one is in. The handlers
        // come off FIRST: `close()` fires a `close` event afterwards, and a `Closure` that has been
        // dropped by then throws ("closure invoked after being dropped") instead of being ignored.
        self.ws.set_onmessage(None);
        self.ws.set_onclose(None);
        self.ws.set_onerror(None);
        let _ = self.ws.close();
    }
}

/// One connection, both directions — what the handshake holds before [`Conn::split`].
pub struct Conn {
    sock: SendWrapper<Rc<Socket>>,
}

/// The read half after [`Conn::split`] — the same socket, taking from the same inbound queue.
pub struct ConnReader {
    sock: SendWrapper<Rc<Socket>>,
}

/// The write half after [`Conn::split`].
///
/// `Send` — which a `web_sys::WebSocket` is not — via [`SendWrapper`], because the ECS holds the
/// writer in a Bevy `Resource` and Bevy 0.18 has no non-`Send` resources. The wrapper's contract is
/// exactly the truth of this build: the value may only be touched on the thread that made it, and a
/// wasm page has only that one thread.
pub struct ConnWriter {
    sock: SendWrapper<Rc<Socket>>,
}

impl Conn {
    /// Open the proxied connection and wait for the socket to come up.
    ///
    /// **The inbound handlers go on before the `open` await, not after it** — a frame that lands
    /// while this future is still parked is otherwise dropped by the browser, and this protocol has
    /// a peer that speaks first. A WebSocket with no `onmessage` does not buffer: the event fires,
    /// finds no listener, and the bytes are gone. Between the browser firing `open` and this
    /// `async fn` being polled again there is at least one task hop, and the world server sends
    /// `SMSG_AUTH_CHALLENGE` the instant it accepts — so the challenge landed inside that gap
    /// roughly half the time, the client sat waiting for a packet it had already been sent, and the
    /// handshake read timed out into a generic "Unable to connect" (decision 0065's bound doing its
    /// job on a self-inflicted silence). Logon never showed it: on 3724 the client speaks first, so
    /// nothing can arrive before we are listening.
    ///
    /// Installing them early costs the open-wait its easy rejection path — `onerror`/`onclose`
    /// cannot be the promise's `reject` when they are already the stream's own handlers — so the
    /// real closures take the `reject` half and call it themselves while a dial is still pending.
    /// After `open` it is dropped and they go back to being nothing but end-of-stream.
    pub async fn connect(host: &str, port: u16) -> io::Result<Conn> {
        let url = connect_url(host, port)?;
        let ws = WebSocket::new(&url).map_err(js_err)?;
        // Frames arrive as ArrayBuffers rather than Blobs, so `onmessage` can read the bytes
        // synchronously instead of awaiting a Blob reader mid-stream.
        ws.set_binary_type(BinaryType::Arraybuffer);

        let inbox = Rc::new(RefCell::new(Inbox::default()));

        let queue = Rc::clone(&inbox);
        let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
            let data = e.data();
            let Ok(buffer) = data.dyn_into::<js_sys::ArrayBuffer>() else {
                return; // not a binary frame — the proxy sends nothing else
            };
            let bytes = js_sys::Uint8Array::new(&buffer);
            let mut queue = queue.borrow_mut();
            queue.buf.reserve(bytes.length() as usize);
            queue.buf.extend(bytes.to_vec());
            queue.wake();
        });
        ws.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

        // The open promise's `reject`, held only while the dial is pending — `None` the moment the
        // socket is up, which is what turns the two closures below back into plain end-of-stream
        // handlers. Rejecting an already-resolved promise is a no-op, so a close that arrives after
        // `open` but before this future resumes ends the stream and lets the dial succeed; the
        // first read then reports the `UnexpectedEof` honestly instead of waiting out the timeout.
        let opening: Rc<RefCell<Option<js_sys::Function>>> = Rc::new(RefCell::new(None));

        // A close and an error both end the stream: the reader turns the flag into `UnexpectedEof`,
        // which the app already reads as a lost session.
        let queue = Rc::clone(&inbox);
        let pending = Rc::clone(&opening);
        let on_close =
            Closure::<dyn FnMut(web_sys::CloseEvent)>::new(move |e: web_sys::CloseEvent| {
                {
                    let mut queue = queue.borrow_mut();
                    queue.closed = true;
                    queue.wake();
                }
                if let Some(reject) = pending.borrow_mut().take() {
                    let _ = reject.call1(&JsValue::NULL, &e);
                }
            });
        ws.set_onclose(Some(on_close.as_ref().unchecked_ref()));

        let queue = Rc::clone(&inbox);
        let pending = Rc::clone(&opening);
        let on_error = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
            {
                let mut queue = queue.borrow_mut();
                queue.closed = true;
                queue.wake();
            }
            if let Some(reject) = pending.borrow_mut().take() {
                let _ = reject.call1(&JsValue::NULL, &e);
            }
        });
        ws.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        // Wait for `open`, failing on an `error`/`close` that beats it (a refused proxy, a
        // disallowed port) through the `reject` the handlers above are holding.
        {
            let socket = ws.clone();
            let pending = Rc::clone(&opening);
            let promise = js_sys::Promise::new(&mut |resolve, reject| {
                *pending.borrow_mut() = Some(reject);
                socket.set_onopen(Some(&resolve));
            });
            let opened = JsFuture::from(promise).await;
            ws.set_onopen(None);
            // The dial is over either way: the handlers stop speaking for it.
            opening.borrow_mut().take();
            if let Err(e) = opened {
                // **Say WHY.** The rejection value is the `close`/`error` event that beat `open`,
                // and a `CloseEvent` carries the code and reason the other end gave — which is the
                // difference between "cannot open <url>" (true of every failure, and actionable
                // for none) and "the gateway accepted the upgrade and then hung up 1011". A dial
                // that fails after a successful 101 looks identical to one that never connected
                // unless this is reported, and telling those apart by hand costs a packet capture.
                let detail = e
                    .dyn_ref::<web_sys::CloseEvent>()
                    .map(|c| {
                        let reason = c.reason();
                        if reason.is_empty() {
                            format!("closed with code {}", c.code())
                        } else {
                            format!("closed with code {} ({reason})", c.code())
                        }
                    })
                    .unwrap_or_else(|| "socket error before open".to_string());
                // The closures die with this scope, and a `Closure` that has been dropped throws
                // when invoked rather than being ignored — so take them off the socket first, the
                // same order `Socket::drop` uses.
                ws.set_onmessage(None);
                ws.set_onclose(None);
                ws.set_onerror(None);
                let _ = ws.close();
                return Err(io::Error::new(
                    ErrorKind::ConnectionRefused,
                    format!("cannot open {url}: {detail}"),
                ));
            }
        }

        Ok(Conn {
            sock: SendWrapper::new(Rc::new(Socket {
                ws,
                inbox,
                read_timeout: Cell::new(None),
                _on_message: on_message,
                _on_close: on_close,
                _on_error: on_error,
            })),
        })
    }

    /// No-op: a WebSocket has no Nagle knob (the browser and the proxy own the TCP options — the
    /// proxy sets `TCP_NODELAY` on the upstream socket for us).
    pub fn set_nodelay(&self, _on: bool) -> io::Result<()> {
        Ok(())
    }

    /// Without blocking: whether bytes wait (`true`), none yet (`false`), or the socket closed
    /// with nothing left to read (`UnexpectedEof`), the native body's answers off the inbox.
    pub fn poll_readable(&self) -> io::Result<bool> {
        let inbox = self.sock.inbox.borrow();
        if !inbox.buf.is_empty() {
            return Ok(true);
        }
        if inbox.closed {
            return Err(io::Error::new(
                ErrorKind::UnexpectedEof,
                "the server ended the connection",
            ));
        }
        Ok(false)
    }

    /// Bound how long a read may wait. Applied by the reader's future (there is no socket option to
    /// set), and an expiry raises [`ErrorKind::TimedOut`] — the kind a native socket's own timeout
    /// gives, so decision 0065's handshake bound behaves the same in a page.
    pub fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.sock.read_timeout.set(t);
        Ok(())
    }

    /// Split into a read and a write half. There is one socket underneath (a WebSocket cannot be
    /// duplicated), so both halves hold the same `Rc` — which is enough, because the two directions
    /// touch disjoint state: the reader drains the inbound queue, the writer only calls `send`.
    pub fn split(self) -> io::Result<(ConnReader, ConnWriter)> {
        let sock = self.sock.take();
        Ok((
            ConnReader {
                sock: SendWrapper::new(Rc::clone(&sock)),
            },
            ConnWriter {
                sock: SendWrapper::new(sock),
            },
        ))
    }

    /// The halves [`Self::split`] made, one connection again. Both hold the same socket, so the
    /// reader's `Rc` simply drops; the socket closes only when the last holder goes.
    pub fn join(reader: ConnReader, writer: ConnWriter) -> Conn {
        drop(reader);
        Conn { sock: writer.sock }
    }
}

impl ReadExactAsync for Conn {
    async fn read_exact_async(&mut self, buf: &mut [u8]) -> io::Result<()> {
        read_exact(&self.sock, buf).await
    }
}

impl ReadExactAsync for ConnReader {
    async fn read_exact_async(&mut self, buf: &mut [u8]) -> io::Result<()> {
        read_exact(&self.sock, buf).await
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        send(&self.sock, buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Write for ConnWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        send(&self.sock, buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// One `write_all` from the protocol is one binary frame — the proxy turns each frame back into one
/// `write_all` on the upstream socket, so a packet stays a packet across the relay. `send` buffers
/// and returns, which is why writes never had to become `async`.
fn send(sock: &Socket, buf: &[u8]) -> io::Result<usize> {
    // **Copy out of linear memory before sending, and that is not an optimisation to undo.**
    //
    // `send_with_u8_array` hands `send` a `Uint8Array` VIEW over wasm memory. In the shared-memory
    // build (`scripts/web-build-threads.sh`) that memory is a `SharedArrayBuffer`, and
    // `WebSocket.send` refuses a shared-backed view outright:
    //
    //     Failed to execute 'send' on 'WebSocket':
    //     The provided ArrayBufferView value must not be shared.
    //
    // Which is how a login could reach 101 and then send nothing at all: the socket was fine and
    // every packet bounced off `send`. The spec forbids it because `send` may read the buffer
    // asynchronously and another thread could write it underneath.
    //
    // So allocate on the JS heap, copy, and send that — an ordinary `ArrayBuffer`, which `send`
    // accepts. One code path rather than a `cfg`: `send_with_u8_array` copies inside the browser
    // anyway, the packets are tens to hundreds of bytes, and a transport that behaves the same on
    // both builds is worth more than an allocation this size.
    let array = js_sys::Uint8Array::new_with_length(buf.len() as u32);
    array.copy_from(buf);
    sock.ws
        .send_with_array_buffer(&array.buffer())
        .map_err(js_err)?;
    Ok(buf.len())
}

/// Take exactly `buf.len()` bytes off the inbound queue, waiting for the socket to deliver them.
fn read_exact<'a>(sock: &'a Socket, buf: &'a mut [u8]) -> ReadExact<'a> {
    let deadline = sock
        .read_timeout
        .get()
        .map(|t| js_sys::Date::now() + t.as_secs_f64() * 1000.0);
    ReadExact {
        sock,
        buf,
        deadline,
        timer: None,
    }
}

/// The awaited read. Polled by the sequencer task; woken by `onmessage`, by `onclose`, or by the
/// timeout's own `setTimeout`.
struct ReadExact<'a> {
    sock: &'a Socket,
    buf: &'a mut [u8],
    /// Wall-clock ms (`Date.now()`) the read gives up at — `None` when no read timeout is set.
    deadline: Option<f64>,
    /// Live only while parked with a deadline; dropping it clears the browser timer.
    timer: Option<Timer>,
}

impl Future for ReadExact<'_> {
    type Output = io::Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut inbox = this.sock.inbox.borrow_mut();
        let want = this.buf.len();
        if inbox.buf.len() >= want {
            for (slot, byte) in this.buf.iter_mut().zip(inbox.buf.drain(..want)) {
                *slot = byte;
            }
            this.timer = None;
            return Poll::Ready(Ok(()));
        }
        // Short of what was asked for and no more is coming.
        if inbox.closed {
            this.timer = None;
            return Poll::Ready(Err(io::Error::new(
                ErrorKind::UnexpectedEof,
                "websocket closed",
            )));
        }
        if let Some(deadline) = this.deadline {
            let remaining = deadline - js_sys::Date::now();
            if remaining <= 0.0 {
                this.timer = None;
                return Poll::Ready(Err(io::Error::new(ErrorKind::TimedOut, "read timed out")));
            }
            // Nothing else would wake us at the deadline: the socket is quiet, which is precisely
            // the case the timeout exists for.
            if this.timer.is_none() {
                this.timer = Timer::arm(&this.sock.inbox, remaining);
            }
        }
        inbox.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

/// A `setTimeout` that wakes the parked reader, cancelled when the read finishes first.
struct Timer {
    handle: i32,
    _callback: Closure<dyn FnMut()>,
}

impl Timer {
    fn arm(inbox: &Rc<RefCell<Inbox>>, after_ms: f64) -> Option<Timer> {
        let window = web_sys::window()?;
        let queue = Rc::clone(inbox);
        let callback = Closure::<dyn FnMut()>::new(move || queue.borrow_mut().wake());
        let handle = window
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                callback.as_ref().unchecked_ref(),
                after_ms.ceil() as i32,
            )
            .ok()?;
        Some(Timer {
            handle,
            _callback: callback,
        })
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some(window) = web_sys::window() {
            window.clear_timeout_with_handle(self.handle);
        }
    }
}

/// A DOM exception as an [`io::Error`] — the JS side's message, or its debug form when it has none.
fn js_err(e: JsValue) -> io::Error {
    let message = e
        .as_string()
        .or_else(|| {
            e.dyn_ref::<js_sys::Error>()
                .map(|e| String::from(e.message()))
        })
        .unwrap_or_else(|| format!("{e:?}"));
    io::Error::other(message)
}
