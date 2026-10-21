#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

//! `bevy_winit` provides utilities to handle window creation and the eventloop through [`winit`]
//!
//! Most commonly, the [`WinitPlugin`] is used as part of
//! [`DefaultPlugins`](https://docs.rs/bevy/latest/bevy/struct.DefaultPlugins.html).
//! The app's [runner](bevy_app::App::runner) is set by `WinitPlugin` and handles the `winit` [`EventLoop`].
//! See `winit_runner` for details.

extern crate alloc;

use bevy_derive::Deref;
use bevy_reflect::Reflect;
use bevy_window::{ExitSystems, RawHandleWrapperHolder, WindowEvent};
use core::cell::RefCell;
use winit::{event_loop::EventLoop, window::WindowId};

use bevy_a11y::AccessibilityRequested;
use bevy_app::{App, Last, Plugin};
use bevy_ecs::prelude::*;
use bevy_window::{CursorOptions, Window, WindowCreated};
use system::{changed_cursor_options, changed_windows, check_keyboard_focus_lost, despawn_windows};
pub use system::{create_monitors, create_windows};
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub use winit::platform::web::CustomCursorExtWebSys;
pub use winit::{
    event_loop::EventLoopProxy,
    window::{CustomCursor as WinitCustomCursor, CustomCursorSource},
};
pub use winit_config::*;
pub use winit_monitors::*;
pub use winit_windows::*;

use crate::{
    accessibility::{AccessKitPlugin, WinitActionRequestHandlers},
    state::winit_runner,
};

/// benilla fork — fork: fsr (FORK.md): called by the runner after each `App::update` it
/// drives, in the same task (on the web, the same `requestAnimationFrame` callback), once the
/// sub-apps have run and the main world's trackers are cleared. Frame generation's held frame runs
/// the next game update here, behind its present.
#[derive(Resource, Clone, Copy)]
pub struct AfterUpdate(pub fn(&mut World));

/// benilla fork — the render worker (`third_party/bevy_render/BENILLA.md`): who runs the plugins'
/// `finish` and `cleanup` once they are ready. Unset, the runner calls both itself, as upstream
/// does. The browser's render worker sets it so they run on the worker that owns the device.
static FINISH_PLUGINS: std::sync::OnceLock<fn(&mut App)> = std::sync::OnceLock::new();

/// Install [`FINISH_PLUGINS`]'s hook. The first call wins; call it before `App::run`.
pub fn set_finish_plugins(hook: fn(&mut App)) {
    let _ = FINISH_PLUGINS.set(hook);
}

/// benilla fork — the render worker: asked before each `App::update` the runner drives, with the
/// main world. `false` skips that update and asks for the next redraw: the browser's render worker
/// holds the page's next update until the last one's extract is paid, and the page must return to
/// the browser meanwhile rather than wait. Unset, every update runs, as upstream.
static MAY_UPDATE: std::sync::OnceLock<fn(&mut World) -> bool> = std::sync::OnceLock::new();

/// Install [`MAY_UPDATE`]'s hook. The first call wins; call it before `App::run`.
pub fn set_may_update(hook: fn(&mut World) -> bool) {
    let _ = MAY_UPDATE.set(hook);
}

/// Whether the runner may run the next update ([`MAY_UPDATE`]).
pub(crate) fn may_update(world: &mut World) -> bool {
    MAY_UPDATE.get().is_none_or(|hook| hook(world))
}

/// benilla fork — the render worker: the page's main world may stay with its render worker while
/// the page waits in the browser's event loop. `lend` runs after each runner callback, with the
/// world; `reclaim` before each, before the runner touches it. Unset, nothing is lent, as upstream.
static LEND_WORLD: std::sync::OnceLock<(fn(&mut World), fn())> = std::sync::OnceLock::new();

/// Install [`LEND_WORLD`]'s hooks. The first call wins; call it before `App::run`.
pub fn set_lend_world(lend: fn(&mut World), reclaim: fn()) {
    let _ = LEND_WORLD.set((lend, reclaim));
}

/// Lend the main world ([`LEND_WORLD`]): the runner returns to the browser.
#[cfg(target_arch = "wasm32")]
pub(crate) fn lend_world(world: &mut World) {
    if let Some((lend, _)) = LEND_WORLD.get() {
        lend(world);
    }
}

/// Take a lent main world back ([`LEND_WORLD`]): the runner is back from the browser.
#[cfg(target_arch = "wasm32")]
pub(crate) fn reclaim_world() {
    if let Some((_, reclaim)) = LEND_WORLD.get() {
        reclaim();
    }
}

/// benilla fork — the render worker: the proxy [`wake_event_loop`] sends through, kept from
/// [`WinitPlugin::build`].
static WAKE: std::sync::OnceLock<EventLoopProxy<WinitUserEvent>> = std::sync::OnceLock::new();

/// Wake the event loop from any thread (`WinitUserEvent::WakeUp`), so the runner checks its plugins'
/// readiness again: the browser's render worker publishes the GPU device off the page. A no-op
/// before the plugin is built.
pub fn wake_event_loop() {
    if let Some(proxy) = WAKE.get() {
        let _ = proxy.send_event(WinitUserEvent::WakeUp);
    }
}

/// `App::finish` then `App::cleanup`, through the hook when one is installed.
pub(crate) fn finish_plugins(app: &mut App) {
    match FINISH_PLUGINS.get() {
        Some(hook) => hook(app),
        None => {
            app.finish();
            app.cleanup();
        }
    }
}

pub mod accessibility;
pub mod converters;
mod cursor;
mod state;
mod system;
mod winit_config;
mod winit_monitors;
mod winit_windows;

thread_local! {
    /// Temporary storage of WinitWindows data to replace usage of `!Send` resources. This will be replaced with proper
    /// storage of `!Send` data after issue #17667 is complete.
    pub static WINIT_WINDOWS: RefCell<WinitWindows> = const { RefCell::new(WinitWindows::new()) };
}

/// A [`Plugin`] that uses `winit` to create and manage windows, and receive window and input
/// events.
///
/// This plugin will add systems and resources that sync with the `winit` backend and also
/// replace the existing [`App`] runner with one that constructs an [event loop](EventLoop) to
/// receive window and input events from the OS.
///
/// The `M` message type can be used to pass custom messages to the `winit`'s loop, and handled as messages
/// in systems.
///
/// When using eg. `MinimalPlugins` you can add this using `WinitPlugin::<WakeUp>::default()`, where
/// `WakeUp` is the default event that bevy uses.
#[derive(Default)]
pub struct WinitPlugin {
    /// Allows the window (and the event loop) to be created on any thread
    /// instead of only the main thread.
    ///
    /// See [`EventLoopBuilder::build`](winit::event_loop::EventLoopBuilder::build) for more information on this.
    ///
    /// # Supported platforms
    ///
    /// Only works on Linux (X11/Wayland) and Windows.
    /// This field is ignored on other platforms.
    pub run_on_any_thread: bool,
}

impl Plugin for WinitPlugin {
    fn name(&self) -> &str {
        "bevy_winit::WinitPlugin"
    }

    fn build(&self, app: &mut App) {
        let mut event_loop_builder = EventLoop::<WinitUserEvent>::with_user_event();

        // linux check is needed because x11 might be enabled on other platforms.
        #[cfg(all(target_os = "linux", feature = "x11"))]
        {
            use winit::platform::x11::EventLoopBuilderExtX11;

            // This allows a Bevy app to be started and ran outside the main thread.
            // A use case for this is to allow external applications to spawn a thread
            // which runs a Bevy app without requiring the Bevy app to need to reside on
            // the main thread, which can be problematic.
            event_loop_builder.with_any_thread(self.run_on_any_thread);
        }

        // linux check is needed because wayland might be enabled on other platforms.
        #[cfg(all(target_os = "linux", feature = "wayland"))]
        {
            use winit::platform::wayland::EventLoopBuilderExtWayland;
            event_loop_builder.with_any_thread(self.run_on_any_thread);
        }

        #[cfg(target_os = "windows")]
        {
            use winit::platform::windows::EventLoopBuilderExtWindows;
            event_loop_builder.with_any_thread(self.run_on_any_thread);
        }

        #[cfg(target_os = "android")]
        {
            use winit::platform::android::EventLoopBuilderExtAndroid;
            let msg = "Bevy must be setup with the #[bevy_main] macro on Android";
            event_loop_builder
                .with_android_app(bevy_android::ANDROID_APP.get().expect(msg).clone());
        }

        let event_loop = event_loop_builder
            .build()
            .expect("Failed to build event loop");
        // benilla fork: kept for `wake_event_loop`.
        let _ = WAKE.set(event_loop.create_proxy());

        app.init_resource::<WinitMonitors>()
            .init_resource::<WinitSettings>()
            .insert_resource(DisplayHandleWrapper(event_loop.owned_display_handle()))
            .insert_resource(EventLoopProxyWrapper(event_loop.create_proxy()))
            .add_message::<RawWinitWindowEvent>()
            .set_runner(|app| winit_runner(app, event_loop))
            .add_systems(
                Last,
                (
                    changed_windows,
                    changed_cursor_options,
                    despawn_windows.after(ExitSystems),
                    check_keyboard_focus_lost,
                )
                    .chain(),
            );

        app.add_plugins(AccessKitPlugin);
        app.add_plugins(cursor::WinitCursorPlugin);

        app.add_observer(
            |_window: On<Add, Window>, event_loop_proxy: Res<EventLoopProxyWrapper>| -> Result {
                event_loop_proxy.send_event(WinitUserEvent::WindowAdded)?;

                Ok(())
            },
        );
    }
}

/// Events that can be sent to perform actions inside the winit event loop.
///
/// Sent via the [`EventLoopProxyWrapper`] resource.
///
/// # Example
///
/// ```
/// # use bevy_ecs::prelude::*;
/// # use bevy_winit::{EventLoopProxyWrapper, WinitUserEvent};
/// fn wakeup_system(event_loop_proxy: Res<EventLoopProxyWrapper>) -> Result {
///     event_loop_proxy.send_event(WinitUserEvent::WakeUp)?;
///
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, Copy, Reflect)]
#[reflect(Debug, Clone)]
pub enum WinitUserEvent {
    /// Dummy event that just wakes up the winit event loop
    WakeUp,
    /// Tell winit that a window needs to be created
    WindowAdded,
}

/// The original window event as produced by Winit. This is meant as an escape
/// hatch for power users that wish to add custom Winit integrations.
/// If you want to process events for your app or game, you should instead use
/// `bevy::window::WindowEvent`, or one of its sub-events.
///
/// When you receive this event it has already been handled by Bevy's main loop.
/// Sending these events will NOT cause them to be processed by Bevy.
#[derive(Debug, Clone, Message)]
pub struct RawWinitWindowEvent {
    /// The window for which the event was fired.
    pub window_id: WindowId,
    /// The raw winit window event.
    pub event: winit::event::WindowEvent,
}

/// A wrapper type around [`winit::event_loop::EventLoopProxy`] with the specific
/// [`winit::event::Event::UserEvent`] used in the [`WinitPlugin`].
///
/// The `EventLoopProxy` can be used to request a redraw from outside bevy.
///
/// Use `Res<EventLoopProxyWrapper>` to retrieve this resource.
#[derive(Resource, Deref)]
pub struct EventLoopProxyWrapper(EventLoopProxy<WinitUserEvent>);

/// A wrapper around [`winit::event_loop::OwnedDisplayHandle`]
///
/// The `DisplayHandleWrapper` can be used to build integrations that rely on direct
/// access to the display handle
///
/// Use `Res<DisplayHandleWrapper>` to receive this resource.
#[derive(Resource, Deref)]
pub struct DisplayHandleWrapper(pub winit::event_loop::OwnedDisplayHandle);

trait AppSendEvent {
    fn send(&mut self, event: impl Into<WindowEvent>);
}

impl AppSendEvent for Vec<WindowEvent> {
    fn send(&mut self, event: impl Into<WindowEvent>) {
        self.push(Into::<WindowEvent>::into(event));
    }
}

/// The parameters of the [`create_windows`] system.
pub type CreateWindowParams<'w, 's> = (
    Commands<'w, 's>,
    Query<
        'w,
        's,
        (
            Entity,
            &'static mut Window,
            &'static CursorOptions,
            Option<&'static RawHandleWrapperHolder>,
        ),
        Added<Window>,
    >,
    MessageWriter<'w, WindowCreated>,
    ResMut<'w, WinitActionRequestHandlers>,
    Res<'w, AccessibilityRequested>,
    Res<'w, WinitMonitors>,
);

/// The parameters of the [`create_monitors`] system.
pub type CreateMonitorParams<'w, 's> = (Commands<'w, 's>, ResMut<'w, WinitMonitors>);
