//! The host OS pasteboard, the one place benilla talks to the platform clipboard. One handle is
//! held for the whole run because on X11 the owning connection is the clipboard: arboard drops the
//! selection with its last handle. A Wayland window goes through `smithay-clipboard`'s standard
//! `wl_data_device` on our own `wl_display`, since arboard here speaks only X11.

use std::ffi::c_void;

use bevy::prelude::*;
use bevy::window::RawHandleWrapper;

/// One platform pasteboard; a trait object so the Wayland half stays inside one `cfg` block.
trait Pasteboard {
    /// The pasteboard's text; `Ok(None)` when it holds no text, `Err` for a real failure.
    fn read_text(&mut self) -> Result<Option<String>, String>;

    fn write_text(&mut self, text: &str) -> Result<(), String>;

    /// The backend's name for log lines.
    fn name(&self) -> &'static str;
}

// `arboard` has no wasm32 backend at all (there is no OS pasteboard to open) — gated out so this
// impl only needs the crate on the platforms it ships for (see the Cargo.toml edge, wasm-only).
#[cfg(not(target_arch = "wasm32"))]
impl Pasteboard for arboard::Clipboard {
    fn read_text(&mut self) -> Result<Option<String>, String> {
        // `ContentNotAvailable` is arboard's empty clipboard, not a failure.
        match self.get_text() {
            Ok(text) if text.is_empty() => Ok(None),
            Ok(text) => Ok(Some(text)),
            Err(arboard::Error::ContentNotAvailable) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }

    fn write_text(&mut self, text: &str) -> Result<(), String> {
        self.set_text(text.to_string()).map_err(|e| e.to_string())
    }

    fn name(&self) -> &'static str {
        if cfg!(target_os = "macos") {
            "arboard/NSPasteboard"
        } else if cfg!(windows) {
            "arboard/win32"
        } else {
            "arboard/x11"
        }
    }
}

/// The browser's half: `navigator.clipboard`, whose reads and writes are promises. A read cannot
/// answer in the chord's own call, so it starts the read and answers "nothing yet"; the text lands
/// in [`web::take_pending`] when the promise resolves, and the UI feed types it into whatever box
/// has the focus on that frame ([`HostClipboard::take_pending_paste`]). The keydown that asked is
/// the user gesture the API wants; the browser may ask the player once for permission to read.
/// The DOM `paste` event is no way round it: the canvas prevents keydown's default action (so the
/// page does not scroll under the game), which is the action that would fire it.
#[cfg(target_arch = "wasm32")]
mod web {
    use std::cell::RefCell;

    use super::Pasteboard;

    thread_local! {
        static PENDING: RefCell<Option<String>> = const { RefCell::new(None) };
    }

    pub(super) struct BrowserClipboard;

    fn clipboard() -> Result<web_sys::Clipboard, String> {
        Ok(web_sys::window()
            .ok_or("no window")?
            .navigator()
            .clipboard())
    }

    impl Pasteboard for BrowserClipboard {
        fn read_text(&mut self) -> Result<Option<String>, String> {
            let promise = clipboard()?.read_text();
            wasm_bindgen_futures::spawn_local(async move {
                match wasm_bindgen_futures::JsFuture::from(promise).await {
                    Ok(value) => {
                        if let Some(text) = value.as_string().filter(|t| !t.is_empty()) {
                            PENDING.with(|p| *p.borrow_mut() = Some(text));
                        }
                    }
                    Err(e) => bevy::log::warn!("clipboard: readText refused — {e:?}"),
                }
            });
            Ok(None)
        }

        fn write_text(&mut self, text: &str) -> Result<(), String> {
            let promise = clipboard()?.write_text(text);
            wasm_bindgen_futures::spawn_local(async move {
                if let Err(e) = wasm_bindgen_futures::JsFuture::from(promise).await {
                    bevy::log::warn!("clipboard: writeText refused — {e:?}");
                }
            });
            Ok(())
        }

        fn name(&self) -> &'static str {
            "navigator.clipboard"
        }
    }

    /// The text a resolved read left, once.
    pub(super) fn take_pending() -> Option<String> {
        PENDING.with(|p| p.borrow_mut().take())
    }
}

/// The Wayland half; the gate must match `smithay-clipboard`'s target in `Cargo.toml`.
#[cfg(all(unix, not(any(target_os = "macos", target_os = "android"))))]
mod wl {
    use super::Pasteboard;

    impl Pasteboard for smithay_clipboard::Clipboard {
        fn read_text(&mut self) -> Result<Option<String>, String> {
            match self.load() {
                Ok(text) if text.is_empty() => Ok(None),
                Ok(text) => Ok(Some(text)),
                // Nothing offered for this seat: an empty clipboard.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(e.to_string()),
            }
        }

        fn write_text(&mut self, text: &str) -> Result<(), String> {
            // `store` hands the text to the clipboard thread, which serves it while we hold the
            // seat.
            self.store(text.to_string());
            Ok(())
        }

        fn name(&self) -> &'static str {
            "smithay/wl_data_device"
        }
    }
}

/// The primary window's `wl_display`, or `None` off Wayland, which selects [`arboard`]. Read from
/// the window handle, not `WAYLAND_DISPLAY`, because `smithay-clipboard` needs winit's own
/// connection.
#[cfg(all(unix, not(any(target_os = "macos", target_os = "android"))))]
pub(crate) fn wayland_display(handle: Option<&RawHandleWrapper>) -> Option<*mut c_void> {
    match handle?.get_display_handle() {
        raw_window_handle::RawDisplayHandle::Wayland(wl) => Some(wl.display.as_ptr()),
        _ => None,
    }
}

/// Off the Wayland-capable platforms the backend is always [`arboard`].
#[cfg(not(all(unix, not(any(target_os = "macos", target_os = "android")))))]
pub(crate) fn wayland_display(_handle: Option<&RawHandleWrapper>) -> Option<*mut c_void> {
    None
}

/// Build the Wayland backend for `display`.
///
/// # Safety
/// `display` must be a live `wl_display` that outlives the returned backend; the window handle's
/// `Arc` keeps the connection alive until app teardown drops the backend.
#[cfg(all(unix, not(any(target_os = "macos", target_os = "android"))))]
fn open_wayland(display: *mut c_void) -> Box<dyn Pasteboard> {
    // SAFETY: `display` is winit's own live `wl_display`, per the doc comment.
    Box::new(unsafe { smithay_clipboard::Clipboard::new(display) })
}

#[cfg(not(all(unix, not(any(target_os = "macos", target_os = "android")))))]
fn open_wayland(_display: *mut c_void) -> Box<dyn Pasteboard> {
    unreachable!("wayland_display() only yields Some on Wayland-capable platforms")
}

/// The session's display-server variables, appended to the clipboard log lines, since a Linux
/// paste failure depends on them.
fn session_note() -> String {
    if !cfg!(unix) || cfg!(target_os = "macos") {
        return String::new();
    }
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| "unset".to_string());
    format!(
        " [XDG_SESSION_TYPE={}, WAYLAND_DISPLAY={}, DISPLAY={}]",
        var("XDG_SESSION_TYPE"),
        var("WAYLAND_DISPLAY"),
        var("DISPLAY"),
    )
}

/// The process-wide handle on the OS pasteboard, a `NonSend` resource held for the whole run.
#[derive(Default)]
pub(crate) struct HostClipboard {
    backend: Option<Box<dyn Pasteboard>>,
    /// Opening is attempted once; a failure is not retried on every keystroke.
    opened: bool,
}

impl HostClipboard {
    /// Resolve the backend on first use, once the window gives the Wayland pointer.
    fn ensure(&mut self, wl_display: Option<*mut c_void>) {
        if self.opened {
            return;
        }
        self.opened = true;
        self.backend = match wl_display {
            Some(display) => Some(open_wayland(display)),
            #[cfg(not(target_arch = "wasm32"))]
            None => match arboard::Clipboard::new() {
                Ok(clipboard) => Some(Box::new(clipboard)),
                Err(e) => {
                    warn!("clipboard: unavailable — {e}{}", session_note());
                    None
                }
            },
            // The browser's clipboard, asynchronous: see [`web`].
            #[cfg(target_arch = "wasm32")]
            None => Some(Box::new(web::BrowserClipboard)),
        };
        if let Some(backend) = &self.backend {
            info!("clipboard: {}{}", backend.name(), session_note());
        }
    }

    /// The pasteboard's text, or `None` if it is empty, non-text or unavailable.
    pub(crate) fn read(&mut self, wl_display: Option<*mut c_void>) -> Option<String> {
        self.ensure(wl_display);
        let backend = self.backend.as_mut()?;
        match backend.read_text() {
            Ok(text) => text,
            Err(e) => {
                warn!(
                    "clipboard: read failed via {} — {e}{}",
                    backend.name(),
                    session_note()
                );
                None
            }
        }
    }

    /// The text a browser paste resolved to since the last frame, once; always `None` natively,
    /// where [`Self::read`] answers in the chord's own call.
    pub(crate) fn take_pending_paste(&mut self) -> Option<String> {
        #[cfg(target_arch = "wasm32")]
        return web::take_pending();
        #[cfg(not(target_arch = "wasm32"))]
        None
    }

    /// Put `text` on the pasteboard.
    pub(crate) fn write(&mut self, wl_display: Option<*mut c_void>, text: &str) {
        self.ensure(wl_display);
        let Some(backend) = self.backend.as_mut() else {
            return;
        };
        if let Err(e) = backend.write_text(text) {
            warn!(
                "clipboard: write failed via {} — {e}{}",
                backend.name(),
                session_note()
            );
        }
    }
}

/// A clipboard holding `text` in memory, so a test never touches the machine's pasteboard.
#[cfg(test)]
impl HostClipboard {
    pub(crate) fn holding(text: &str) -> Self {
        struct Memory(String);
        impl Pasteboard for Memory {
            fn read_text(&mut self) -> Result<Option<String>, String> {
                Ok((!self.0.is_empty()).then(|| self.0.clone()))
            }
            fn write_text(&mut self, text: &str) -> Result<(), String> {
                self.0 = text.to_string();
                Ok(())
            }
            fn name(&self) -> &'static str {
                "memory"
            }
        }
        HostClipboard {
            backend: Some(Box::new(Memory(text.to_string()))),
            opened: true,
        }
    }
}
