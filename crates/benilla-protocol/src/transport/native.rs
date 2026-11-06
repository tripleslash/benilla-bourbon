//! [`Conn`] over a real socket — the native body of the transport seam.
//!
//! Every TCP method is the `TcpStream` call it wraps, so nothing about the native wire changed when
//! the seam went in: the same `connect`, the same `TCP_NODELAY`, the same `SO_RCVTIMEO`, the same
//! `try_clone` split into a reader thread and a writer thread. The one new shape is that reads are
//! `async` — and here the future is ready the moment it is polled, because it *is* the blocking
//! `read_exact`, run (as before) on the network thread under
//! [`futures_lite::future::block_on`](https://docs.rs/futures-lite).
//!
//! fork: desktop — with a WebSocket gateway named ([`set_ws_base`]) the same connection rides the
//! gateway instead ([`super::native_ws`]), as the browser build's does.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use super::native_ws::{self, Link};
use super::ReadExactAsync;

/// The gateway every connection dials through, when one is named.
static WS_BASE: RwLock<Option<String>> = RwLock::new(None);

/// Dial every later connection through the WebSocket gateway at `base` (`wss://host[:port]`, no
/// trailing slash) — the site's `config.json` `ws`.
pub fn set_ws_base(base: String) {
    *WS_BASE.write().unwrap_or_else(|e| e.into_inner()) = Some(base);
}

/// The gateway named, if any.
pub fn ws_base() -> Option<String> {
    WS_BASE.read().unwrap_or_else(|e| e.into_inner()).clone()
}

enum Stream {
    Tcp(TcpStream),
    Ws(Arc<Link>),
}

/// One connection, both directions — what the handshake holds before [`Conn::split`].
pub struct Conn {
    stream: Stream,
}

/// The read half after [`Conn::split`] — its own `try_clone`d descriptor onto the same socket.
pub struct ConnReader {
    stream: Stream,
}

/// The write half after [`Conn::split`]. `Send`, so it can be handed to the write thread.
pub struct ConnWriter {
    stream: Stream,
}

impl Conn {
    /// Dial `host:port`. `host` may be a name or a literal address — `(&str, u16)` resolves either.
    pub async fn connect(host: &str, port: u16) -> io::Result<Conn> {
        let stream = match ws_base() {
            Some(base) => Stream::Ws(native_ws::connect(&base, host, port)?),
            None => Stream::Tcp(TcpStream::connect((host, port))?),
        };
        Ok(Conn { stream })
    }

    /// `TCP_NODELAY` (decision 0617 — Nagle is wrong for this protocol; see `world::send_packet`).
    /// A gateway link sets it on its own socket at connect.
    pub fn set_nodelay(&self, on: bool) -> io::Result<()> {
        match &self.stream {
            Stream::Tcp(s) => s.set_nodelay(on),
            Stream::Ws(_) => Ok(()),
        }
    }

    /// Without blocking: whether bytes wait (`true`), none yet (`false`), or the peer closed the
    /// connection (`UnexpectedEof`), for the character-select park's link check.
    pub fn poll_readable(&self) -> io::Result<bool> {
        let stream = match &self.stream {
            Stream::Tcp(s) => s,
            Stream::Ws(link) => return link.poll_readable(),
        };
        stream.set_nonblocking(true)?;
        let peeked = stream.peek(&mut [0u8; 1]);
        stream.set_nonblocking(false)?;
        match peeked {
            Ok(0) => Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the server ended the connection",
            )),
            Ok(_) => Ok(true),
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    /// Bound how long a read may wait; `None` blocks forever (decision 0065's handshake timeout).
    pub fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        match &self.stream {
            Stream::Tcp(s) => s.set_read_timeout(t),
            Stream::Ws(link) => {
                link.set_read_timeout(t);
                Ok(())
            }
        }
    }

    /// Split into independent halves so a reader can stream while another path writes.
    pub fn split(self) -> io::Result<(ConnReader, ConnWriter)> {
        let read = match &self.stream {
            Stream::Tcp(s) => Stream::Tcp(s.try_clone()?),
            Stream::Ws(link) => Stream::Ws(Arc::clone(link)),
        };
        Ok((
            ConnReader { stream: read },
            ConnWriter {
                stream: self.stream,
            },
        ))
    }

    /// The halves [`Self::split`] made, one connection again. The reader's descriptor is a clone
    /// of the writer's, so dropping it closes nothing.
    pub fn join(reader: ConnReader, writer: ConnWriter) -> Conn {
        drop(reader);
        Conn {
            stream: writer.stream,
        }
    }
}

impl Stream {
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        match self {
            Stream::Tcp(s) => s.read_exact(buf),
            Stream::Ws(link) => link.read_exact(buf),
        }
    }

    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Stream::Tcp(s) => s.write(buf),
            Stream::Ws(link) => link.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Stream::Tcp(s) => s.flush(),
            Stream::Ws(link) => link.flush(),
        }
    }
}

impl ReadExactAsync for Conn {
    async fn read_exact_async(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.stream.read_exact(buf)
    }
}

impl ReadExactAsync for ConnReader {
    async fn read_exact_async(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.stream.read_exact(buf)
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stream.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

impl Write for ConnWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stream.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}
