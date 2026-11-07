//! The update's progress window: one small fixed window on the main thread, painted with GDI (a
//! title line, a bar and the megabytes so far), repainted on a 50 ms timer from the download
//! thread's counter, and destroyed once the download ends. Closing it cancels the update.

use std::cell::RefCell;
use std::ptr::{null, null_mut};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use windows_sys::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{
    BeginPaint, BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, CreateFontIndirectW,
    CreateSolidBrush, DeleteDC, DeleteObject, DrawTextW, EndPaint, FillRect, InvalidateRect,
    SelectObject, SetBkMode, SetTextColor, DT_END_ELLIPSIS, DT_LEFT, DT_NOPREFIX, DT_SINGLELINE,
    DT_VCENTER, HDC, HFONT, LOGFONTW, PAINTSTRUCT, SRCCOPY, TRANSPARENT,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::HiDpi::{
    GetDpiForSystem, SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRectEx, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW,
    GetMessageW, GetSystemMetrics, KillTimer, LoadCursorW, LoadIconW, PostQuitMessage,
    RegisterClassExW, SetTimer, ShowWindow, TranslateMessage, UnregisterClassW, IDC_ARROW, MSG,
    SM_CXSCREEN, SM_CYSCREEN, SW_SHOW, WM_CLOSE, WM_DESTROY, WM_ERASEBKGND, WM_PAINT, WM_TIMER,
    WNDCLASSEXW, WS_CAPTION, WS_MINIMIZEBOX, WS_OVERLAPPED, WS_SYSMENU,
};

use super::{Shared, RUNNING};

/// The client area at 96 dpi.
const WIDTH: i32 = 440;
const HEIGHT: i32 = 120;
const TIMER: usize = 1;

const BACKGROUND: COLORREF = rgb(24, 20, 16);
const TRACK: COLORREF = rgb(58, 49, 40);
const BAR: COLORREF = rgb(200, 160, 64);
const TEXT: COLORREF = rgb(232, 220, 192);
const DIM: COLORREF = rgb(168, 154, 128);

/// What the window procedure reads: it runs on this thread only, inside [`run`]'s message loop.
struct Ui {
    shared: Arc<Shared>,
    total: u64,
    scale: f32,
    font: HFONT,
    closed: bool,
}

thread_local! {
    static UI: RefCell<Option<Ui>> = const { RefCell::new(None) };
}

/// Show the window until the download ends or the player closes it; `Some(true)` when closed
/// first, `None` when no window could be made.
pub(super) fn run(shared: Arc<Shared>, total: u64) -> Option<bool> {
    let class = wide("benilla-desktop-update");
    let title = wide("WoW Reforged");
    // SAFETY: plain Win32 calls on this thread; every pointer is to a live local or null, and the
    // window, its class and its font are released before returning.
    unsafe {
        // The awareness the client's own window asks for, set first so this one is sharp too.
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        let scale = GetDpiForSystem().max(96) as f32 / 96.0;
        let instance = GetModuleHandleW(null());
        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: 0,
            lpfnWndProc: Some(window_proc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: instance,
            // The exe's icon, resource 1 (`build.rs`, winresource).
            hIcon: LoadIconW(instance, std::ptr::without_provenance(1)),
            hCursor: LoadCursorW(null_mut(), IDC_ARROW),
            hbrBackground: null_mut(),
            lpszMenuName: null(),
            lpszClassName: class.as_ptr(),
            hIconSm: null_mut(),
        };
        if RegisterClassExW(&wc) == 0 {
            return None;
        }

        let mut lf: LOGFONTW = std::mem::zeroed();
        lf.lfHeight = -scaled(15, scale);
        lf.lfWeight = 400;
        for (dst, src) in lf.lfFaceName.iter_mut().zip("Segoe UI".encode_utf16()) {
            *dst = src;
        }
        let font = CreateFontIndirectW(&lf);

        UI.with(|ui| {
            *ui.borrow_mut() = Some(Ui {
                shared,
                total,
                scale,
                font,
                closed: false,
            })
        });

        let style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        let mut frame = RECT {
            left: 0,
            top: 0,
            right: scaled(WIDTH, scale),
            bottom: scaled(HEIGHT, scale),
        };
        AdjustWindowRectEx(&mut frame, style, 0, 0);
        let (w, h) = (frame.right - frame.left, frame.bottom - frame.top);
        let x = (GetSystemMetrics(SM_CXSCREEN) - w) / 2;
        let y = (GetSystemMetrics(SM_CYSCREEN) - h) / 2;
        let hwnd = CreateWindowExW(
            0,
            class.as_ptr(),
            title.as_ptr(),
            style,
            x,
            y,
            w,
            h,
            null_mut(),
            null_mut(),
            instance,
            null(),
        );
        let closed = if hwnd.is_null() {
            None
        } else {
            ShowWindow(hwnd, SW_SHOW);
            SetTimer(hwnd, TIMER, 50, None);
            let mut msg: MSG = std::mem::zeroed();
            while GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            Some(())
        };
        let ui = UI.with(|ui| ui.borrow_mut().take());
        DeleteObject(font);
        UnregisterClassW(class.as_ptr(), instance);
        closed.and(ui.map(|ui| ui.closed))
    }
}

unsafe extern "system" fn window_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_TIMER => {
            let running = UI.with(|ui| {
                ui.borrow()
                    .as_ref()
                    .is_some_and(|ui| ui.shared.state.load(Ordering::Acquire) == RUNNING)
            });
            if running {
                InvalidateRect(hwnd, null(), 0);
            } else {
                KillTimer(hwnd, TIMER);
                DestroyWindow(hwnd);
            }
            0
        }
        WM_PAINT => {
            let mut ps: PAINTSTRUCT = std::mem::zeroed();
            let hdc = BeginPaint(hwnd, &mut ps);
            if let Some(view) = UI.with(|ui| ui.borrow().as_ref().map(View::of)) {
                paint(hdc, view);
            }
            EndPaint(hwnd, &ps);
            0
        }
        // The paint covers the whole client area; erasing first would only flicker.
        WM_ERASEBKGND => 1,
        WM_CLOSE => {
            UI.with(|ui| {
                if let Some(ui) = ui.borrow_mut().as_mut() {
                    ui.closed = true;
                }
            });
            DestroyWindow(hwnd);
            0
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            0
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

/// One frame's numbers, read out of [`Ui`] so no borrow is held while painting.
#[derive(Clone, Copy)]
struct View {
    done: u64,
    total: u64,
    scale: f32,
    font: HFONT,
}

impl View {
    fn of(ui: &Ui) -> View {
        View {
            done: ui.shared.done.load(Ordering::Relaxed).min(ui.total),
            total: ui.total,
            scale: ui.scale,
            font: ui.font,
        }
    }
}

/// Draw the whole client area off screen, then copy it in one blit.
unsafe fn paint(hdc: HDC, ui: View) {
    let s = |v: i32| scaled(v, ui.scale);
    let (w, h) = (s(WIDTH), s(HEIGHT));
    let mem = CreateCompatibleDC(hdc);
    let bitmap = CreateCompatibleBitmap(hdc, w, h);
    let old_bitmap = SelectObject(mem, bitmap);
    let old_font = SelectObject(mem, ui.font);

    fill(mem, rect(0, 0, w, h), BACKGROUND);
    let done = ui.done;
    let fraction = if ui.total == 0 {
        0.0
    } else {
        done as f64 / ui.total as f64
    };
    let track = rect(s(24), s(56), w - s(24), s(70));
    fill(mem, track, TRACK);
    let end = track.left + ((track.right - track.left) as f64 * fraction).round() as i32;
    if end > track.left {
        fill(mem, RECT { right: end, ..track }, BAR);
    }

    SetBkMode(mem, TRANSPARENT as _);
    SetTextColor(mem, TEXT);
    text(
        mem,
        "Downloading the new client\u{2026}",
        rect(s(24), s(16), w - s(24), s(46)),
    );
    SetTextColor(mem, DIM);
    let progress = format!(
        "{:.1} of {:.1} MB \u{b7} {:.0}%",
        mib(done),
        mib(ui.total),
        fraction * 100.0
    );
    text(mem, &progress, rect(s(24), s(78), w - s(24), s(104)));

    BitBlt(hdc, 0, 0, w, h, mem, 0, 0, SRCCOPY);
    SelectObject(mem, old_font);
    SelectObject(mem, old_bitmap);
    DeleteObject(bitmap);
    DeleteDC(mem);
}

unsafe fn fill(hdc: HDC, r: RECT, color: COLORREF) {
    let brush = CreateSolidBrush(color);
    FillRect(hdc, &r, brush);
    DeleteObject(brush);
}

unsafe fn text(hdc: HDC, s: &str, mut r: RECT) {
    let mut chars: Vec<u16> = s.encode_utf16().collect();
    DrawTextW(
        hdc,
        chars.as_mut_ptr(),
        chars.len() as i32,
        &mut r,
        DT_LEFT | DT_SINGLELINE | DT_VCENTER | DT_NOPREFIX | DT_END_ELLIPSIS,
    );
}

fn rect(left: i32, top: i32, right: i32, bottom: i32) -> RECT {
    RECT {
        left,
        top,
        right,
        bottom,
    }
}

fn scaled(v: i32, scale: f32) -> i32 {
    (v as f32 * scale).round() as i32
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// GDI's colour: red in the low byte.
const fn rgb(r: u8, g: u8, b: u8) -> COLORREF {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16)
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}
