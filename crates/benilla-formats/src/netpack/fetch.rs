//! The network half of the native pack store: whole-object `GET`s against the static origin a
//! container is hosted to, blocking, on whatever thread asks.
//!
//! Behind a trait so the store's cache law (what is kept, what is thrown away, what a re-pack
//! invalidates) is testable against a folder standing in for the origin, with no network.

use std::io::{self, Read};
use std::path::PathBuf;
use std::time::Duration;

/// Where a container's objects come from: `get("index.bpi")`, `get("p/00/7.bpk")`.
pub trait Fetch: Send + Sync {
    /// The object at `rel` (relative to the container root, `/`-separated), whole. A missing
    /// object is [`io::ErrorKind::NotFound`], and never a page standing in for one.
    fn get(&self, rel: &str) -> io::Result<Vec<u8>>;

    /// The origin, for messages.
    fn describe(&self) -> String;
}

/// A container served over HTTP(S): `{base}/{rel}`.
pub struct Http {
    base: String,
    agent: ureq::Agent,
}

impl Http {
    /// `base` is the container root, no trailing slash (`https://wow.example.com/data`).
    pub fn new(base: &str) -> Self {
        let agent = ureq::Agent::config_builder()
            // A pack is at most 24 MiB; a stalled connection must fail rather than hang a read.
            .timeout_connect(Some(Duration::from_secs(15)))
            .timeout_recv_body(Some(Duration::from_secs(120)))
            .http_status_as_error(false)
            .user_agent(concat!("benilla-desktop/", env!("CARGO_PKG_VERSION")))
            .build()
            .new_agent();
        Self {
            base: base.trim_end_matches('/').to_string(),
            agent,
        }
    }
}

impl Fetch for Http {
    fn get(&self, rel: &str) -> io::Result<Vec<u8>> {
        let url = format!("{}/{rel}", self.base);
        let mut resp = self
            .agent
            .get(&url)
            .call()
            .map_err(|e| io::Error::other(format!("GET {url}: {e}")))?;
        let status = resp.status().as_u16();
        if status == 404 {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("{url}: 404"),
            ));
        }
        if status != 200 {
            return Err(io::Error::other(format!("GET {url}: HTTP {status}")));
        }
        let mut out = Vec::new();
        resp.body_mut()
            .with_config()
            // The packer's object cap is 24 MiB; the listfile is ~5 MB. Twice the cap is headroom.
            .limit(48 * 1024 * 1024)
            .reader()
            .read_to_end(&mut out)
            .map_err(|e| io::Error::other(format!("GET {url}: {e}")))?;
        Ok(out)
    }

    fn describe(&self) -> String {
        self.base.clone()
    }
}

/// A folder standing in for an origin: `{root}/{rel}`. For tests, and for pointing a client at a
/// container built on this machine through the same code path a remote one takes.
pub struct Folder(pub PathBuf);

impl Fetch for Folder {
    fn get(&self, rel: &str) -> io::Result<Vec<u8>> {
        std::fs::read(self.0.join(rel))
    }

    fn describe(&self) -> String {
        self.0.display().to_string()
    }
}
