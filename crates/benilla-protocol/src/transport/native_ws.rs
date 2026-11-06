//! fork: desktop — the native transport over a WebSocket gateway: the browser build's wire
//! (`{ws-base}/ws/{port}?host={host}`, binary frames, the byte stream reassembled here), for a
//! client whose realm is reachable only through such a gateway (`wenilla-host`'s `/ws/{port}`, the
//! site's tunnel).
//!
//! One TLS session cannot be read and written from two threads at once, so the socket is one
//! [`tungstenite::WebSocket`] behind a lock: a pump thread waits for bytes **without the lock**
//! (a `peek` on a clone of the raw socket), then drains every complete message into the inbox under
//! it; a write takes the lock for one frame. The protocol's readers take from the inbox, with the
//! connection's read timeout, as the web body does.

use std::collections::VecDeque;
use std::io::{self, ErrorKind};
use std::net::{Shutdown, TcpStream};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

type Socket = WebSocket<MaybeTlsStream<TcpStream>>;

/// How long a drain pass waits on the socket for the rest of a frame before handing the lock back.
const DRAIN_WAIT: Duration = Duration::from_millis(1);

#[derive(Default)]
struct Inbox {
    buf: VecDeque<u8>,
    /// The socket closed: a read the buffer cannot satisfy ends as `UnexpectedEof`.
    closed: bool,
}

pub(super) struct Shared {
    ws: Mutex<Socket>,
    /// A clone of the socket under the TLS session, for the lock-free wait and the shutdown.
    raw: TcpStream,
    inbox: Mutex<Inbox>,
    arrived: Condvar,
    read_timeout: Mutex<Option<Duration>>,
}

/// What [`super::Conn`] and its halves hold: dropping the last one shuts the socket, which ends
/// the pump.
pub(super) struct Link(Arc<Shared>);

impl Drop for Link {
    fn drop(&mut self) {
        let _ = self.0.raw.shutdown(Shutdown::Both);
    }
}

/// `{base}/ws/{port}?host={host}` — the web transport's URL, byte for byte.
pub(super) fn connect_url(base: &str, host: &str, port: u16) -> String {
    let mut encoded = String::with_capacity(host.len());
    for b in host.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            encoded.push(b as char);
        } else {
            encoded.push_str(&format!("%{b:02X}"));
        }
    }
    format!("{}/ws/{port}?host={encoded}", base.trim_end_matches('/'))
}

/// Dial the gateway and start the pump.
pub(super) fn connect(base: &str, host: &str, port: u16) -> io::Result<Arc<Link>> {
    // rustls needs one process-wide crypto provider; a second install is a no-op error.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let url = connect_url(base, host, port);
    let (ws, _response) = tungstenite::connect(url.as_str()).map_err(|e| ws_err(&url, e))?;
    let inner = match ws.get_ref() {
        MaybeTlsStream::Plain(s) => s,
        MaybeTlsStream::Rustls(s) => s.get_ref(),
        _ => {
            return Err(io::Error::new(
                ErrorKind::Unsupported,
                format!("{url}: an unsupported TLS stream"),
            ))
        }
    };
    // The drain's short wait, on the descriptor tungstenite reads: Windows keeps `SO_RCVTIMEO` per
    // descriptor, so set on the clone it would never reach the session's reads. The clone only
    // waits (`peek`, no timeout) and shuts down.
    inner.set_read_timeout(Some(DRAIN_WAIT))?;
    let raw = inner.try_clone()?;
    raw.set_read_timeout(None)?;
    // Nagle off, as the TCP body sets it (decision 0617).
    let _ = raw.set_nodelay(true);
    let shared = Arc::new(Shared {
        ws: Mutex::new(ws),
        raw,
        inbox: Mutex::new(Inbox::default()),
        arrived: Condvar::new(),
        read_timeout: Mutex::new(None),
    });
    let pump = Arc::clone(&shared);
    std::thread::Builder::new()
        .name(format!("ws-pump-{port}"))
        .spawn(move || run_pump(&pump))?;
    Ok(Arc::new(Link(shared)))
}

fn ws_err(url: &str, e: tungstenite::Error) -> io::Error {
    match e {
        tungstenite::Error::Io(e) => io::Error::new(e.kind(), format!("{url}: {e}")),
        e => io::Error::new(ErrorKind::ConnectionRefused, format!("{url}: {e}")),
    }
}

fn run_pump(shared: &Shared) {
    let mut probe = [0u8; 1];
    // Drain first, then wait: the upgrade's read can already hold the server's first frame (the
    // world server speaks first), in tungstenite's buffer where no `peek` sees it.
    let mut drained = false;
    loop {
        if drained {
            // Wait for bytes with the lock free, so a write never queues behind an idle read.
            match shared.raw.peek(&mut probe) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
        drained = true;
        let mut ws = shared.ws.lock().unwrap_or_else(PoisonError::into_inner);
        let mut open = true;
        loop {
            match ws.read() {
                Ok(Message::Binary(bytes)) => {
                    let mut inbox = shared.inbox.lock().unwrap_or_else(PoisonError::into_inner);
                    inbox.buf.extend(bytes.iter());
                    shared.arrived.notify_all();
                }
                Ok(Message::Close(_)) => {
                    open = false;
                    break;
                }
                // Text, ping and pong: the gateway sends none the protocol reads; a ping's pong is
                // queued by tungstenite and goes out with the next flush.
                Ok(_) => {}
                Err(tungstenite::Error::Io(e))
                    if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) =>
                {
                    break
                }
                Err(_) => {
                    open = false;
                    break;
                }
            }
        }
        let _ = ws.flush();
        drop(ws);
        if !open {
            break;
        }
    }
    let mut inbox = shared.inbox.lock().unwrap_or_else(PoisonError::into_inner);
    inbox.closed = true;
    shared.arrived.notify_all();
}

impl Link {
    pub(super) fn set_read_timeout(&self, t: Option<Duration>) {
        *self
            .0
            .read_timeout
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = t;
    }

    /// Bytes waiting (`true`), none yet (`false`), or the peer closed (`UnexpectedEof`).
    pub(super) fn poll_readable(&self) -> io::Result<bool> {
        let inbox = self.0.inbox.lock().unwrap_or_else(PoisonError::into_inner);
        if !inbox.buf.is_empty() {
            Ok(true)
        } else if inbox.closed {
            Err(io::Error::new(
                ErrorKind::UnexpectedEof,
                "the server ended the connection",
            ))
        } else {
            Ok(false)
        }
    }

    /// Fill `buf` from the inbox, waiting up to the read timeout for the rest.
    pub(super) fn read_exact(&self, buf: &mut [u8]) -> io::Result<()> {
        let timeout = *self
            .0
            .read_timeout
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut inbox = self.0.inbox.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            let want = buf.len();
            if inbox.buf.len() >= want {
                for (dst, src) in buf.iter_mut().zip(inbox.buf.drain(..want)) {
                    *dst = src;
                }
                return Ok(());
            }
            if inbox.closed {
                return Err(io::Error::new(
                    ErrorKind::UnexpectedEof,
                    "the server ended the connection",
                ));
            }
            inbox = match deadline {
                None => self
                    .0
                    .arrived
                    .wait(inbox)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(at) => {
                    let left = at.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Err(io::Error::new(ErrorKind::TimedOut, "read timed out"));
                    }
                    self.0
                        .arrived
                        .wait_timeout(inbox, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
        }
    }

    /// Send `buf` as one binary frame.
    pub(super) fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let mut ws = self.0.ws.lock().unwrap_or_else(PoisonError::into_inner);
        let sent = ws.send(Message::binary(buf.to_vec()));
        sent.map(|()| buf.len()).map_err(|e| match e {
            tungstenite::Error::Io(e) => e,
            e => io::Error::new(ErrorKind::BrokenPipe, e.to_string()),
        })
    }

    pub(super) fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_url_is_the_web_transports() {
        assert_eq!(
            connect_url("wss://realm.example.org/", "realm.example.org", 3724),
            "wss://realm.example.org/ws/3724?host=realm.example.org"
        );
        assert_eq!(
            connect_url("ws://127.0.0.1:8080", "a b", 8085),
            "ws://127.0.0.1:8080/ws/8085?host=a%20b"
        );
    }

    /// A gateway on a loopback socket that echoes every frame back: the halves of a link carry a
    /// byte stream through it, split across frames and joined, and a close reads as end of stream.
    #[test]
    fn a_link_carries_a_byte_stream_through_a_gateway() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (sock, _) = listener.accept().unwrap();
            let mut ws = tungstenite::accept(sock).unwrap();
            // Greet first, as the world server does, then echo two frames and close.
            ws.send(Message::binary(vec![1u8, 2, 3])).unwrap();
            for _ in 0..2 {
                let msg = ws.read().unwrap();
                ws.send(msg).unwrap();
            }
            ws.close(None).unwrap();
            while ws.read().is_ok() {}
        });
        let link = connect(&format!("ws://{addr}"), "localhost", 3724).unwrap();
        link.set_read_timeout(Some(Duration::from_secs(5)));
        let mut hello = [0u8; 3];
        link.read_exact(&mut hello).unwrap();
        assert_eq!(hello, [1, 2, 3]);
        link.write(b"abc").unwrap();
        link.write(b"defg").unwrap();
        let mut echoed = [0u8; 7];
        link.read_exact(&mut echoed).unwrap();
        assert_eq!(&echoed, b"abcdefg");
        let err = link.read_exact(&mut [0u8; 1]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnexpectedEof);
        server.join().unwrap();
    }
}
