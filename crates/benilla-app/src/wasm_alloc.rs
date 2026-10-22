//! fork: web — the threaded browser build's global allocator.
//!
//! std's allocator on `wasm32` with `+atomics` is one `dlmalloc` behind one spin lock
//! (`library/std/src/sys/alloc/wasm.rs`: `LOCKED.swap(1, Acquire)` in a loop, no queue, no backoff),
//! taken by every thread for every allocation: the page, the async and compute pools and the render
//! worker. In a crowd the page thread lost that race for seconds (measured: 5.9 s and 6.6 s frames,
//! ~75% of their samples in the allocator, every schedule stage slowed alike). So this allocator
//! owns linear memory itself, in mimalloc's shape: no thread waits for another to allocate or free a
//! block up to 64 KiB, and no thread ever waits to free anything.
//!
//! - Memory is cut into 64 KiB slices, wasm's page, so whatever `memory.grow` returns is whole
//!   slices. Runs of slices (spans) come from one pool under [`SPANS`]: a TLSF index by length, free
//!   neighbours merged. The pool grows memory by [`GROW_STEP`] at a time (`?alloc_grow_mb=`), so a
//!   load burst pays a few `memory.grow`s instead of one per allocation that outruns the free space.
//! - A block up to 64 KiB comes from a page: a span of one slice (classes up to 8 KiB) or four, cut
//!   into blocks of one size class. A page belongs to one heap, and a heap is held by one thread at a
//!   time through a flag nobody waits on: a thread takes the heap its stack address picks, or the
//!   next free one. Allocating from a page takes no lock.
//! - A block is freed into its page by taking the page's heap, when that is free; while another
//!   thread holds it, the block goes on the heap's inbox, a lock-free list its next holder takes in.
//!   A worker's mesh dropped on the page thread, or the page's buffers dropped on a worker, never
//!   wait (measured before: a 206 ms frame spinning in `dealloc` as unloaded tiles were dropped).
//! - A page whose blocks are all free goes back to the pool, so no thread or size class holds memory
//!   another needs. A span freed while [`SPANS`] is busy goes on a lock-free list ([`DEFERRED`]) for
//!   the lock's next holder, so no free waits for the pool either.
//! - A block over 64 KiB is a span of its own. Every span's metadata lives out of line, in tables by
//!   slice ([`SPAN`], [`HEAD`], [`OWNER`]), so a block's page is found from its address alone.
//!
//! The heap is picked from the stack address, not a thread-local: a new thread's stack and TLS block
//! are allocated with this allocator as wasm-bindgen starts it, possibly before that thread's TLS
//! exists, and a thread-local touched then would write through an unset TLS base.
//!
//! Memory below the first grow (static data, the main stack, the initial heap) is left alone. Should
//! anything still call std's `System`, its `dlmalloc` takes that initial chunk and grows memory by
//! itself, and the pool files only what its own `memory.grow`s returned.

use std::alloc::{GlobalAlloc, Layout};
use std::cell::UnsafeCell;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU8, AtomicUsize, Ordering};

/// A slice: wasm's 64 KiB page.
const SLICE_SHIFT: u32 = 16;
const SLICE: usize = 1 << SLICE_SHIFT;
/// Slices in wasm32's 4 GiB address space; the native model holds no more.
const MAX_SLICES: usize = 1 << 16;
/// No slice: the end of a list.
const NIL: u32 = u32::MAX;

/// The size classes, bytes: four per doubling, each a multiple of [`CLASS_ALIGN`], and every power
/// of two from 16 to 64 KiB among them.
const CLASSES: [usize; 44] = [
    16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
    1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336,
    16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536,
];
const N_CLASSES: usize = CLASSES.len();
const CLASS_ALIGN: usize = 16;
/// The largest size a class serves.
const MAX_SMALL: usize = CLASSES[N_CLASSES - 1];
/// The class index by `size.div_ceil(16)`, for sizes `0..=MAX_SMALL`.
const CLASS_OF: [u8; MAX_SMALL / 16 + 1] = class_table();
/// Classes up to this size are cut from one-slice pages, larger ones from [`MEDIUM_SLICES`].
const SMALL_PAGE_MAX: usize = 8 * 1024;
const MEDIUM_SLICES: u32 = 4;

const fn class_table() -> [u8; MAX_SMALL / 16 + 1] {
    let mut table = [0u8; MAX_SMALL / 16 + 1];
    let mut i = 1;
    let mut c = 0;
    while i < table.len() {
        while CLASSES[c] < i * 16 {
            c += 1;
        }
        table[i] = c as u8;
        i += 1;
    }
    table
}

/// The class serving `layout`, or `None` for a block that is a span of its own. An over-aligned
/// block takes a power-of-two class at least its alignment: a page is slice-aligned, and its blocks
/// sit at multiples of their size.
fn class_of(layout: Layout) -> Option<usize> {
    let size = if layout.align() <= CLASS_ALIGN {
        layout.size()
    } else {
        layout
            .size()
            .max(layout.align())
            .checked_next_power_of_two()?
    };
    (size <= MAX_SMALL).then(|| CLASS_OF[size.div_ceil(16)] as usize)
}

/// Slices in a page of class `c`.
const fn page_slices(c: usize) -> u32 {
    if CLASSES[c] <= SMALL_PAGE_MAX {
        1
    } else {
        MEDIUM_SLICES
    }
}

/// Slices a block of `size` bytes spans, if the address space holds that many.
fn slices_for(size: usize) -> Option<u32> {
    let n = size.div_ceil(SLICE).max(1);
    (n < MAX_SLICES).then_some(n as u32)
}

/// [`Span::kind`]s.
const FREE: u8 = 0;
const PAGE: u8 = 1;
const LARGE: u8 = 2;

/// A span's metadata, kept at its head slice. `kind` and `slices` change only under [`SPANS`]; a
/// page's other fields belong to whoever holds its heap, a free span's links to [`SPANS`]. Fields
/// are reached through raw pointers ([`sp`]), never a reference to the whole: a heap's holder and a
/// [`SPANS`] holder merging a neighbour touch different fields of one span at once.
struct Span {
    kind: u8,
    /// Page: its class.
    class: u8,
    /// Page: in its heap's list for its class, which holds every page with a block to give.
    listed: bool,
    /// Page: empty, and kept against its heap's [`EMPTY_BUDGET`].
    kept: bool,
    slices: u32,
    /// Page: blocks cut from it so far, blocks it holds, blocks out and not freed back to it.
    bump: u32,
    cap: u32,
    used: u32,
    /// A page's neighbours in its heap's list, a free span's in its bin, a deferred span's next.
    prev: u32,
    next: u32,
    /// Page: its free blocks, linked through their first word.
    free: *mut Node,
}

impl Span {
    const EMPTY: Self = Self {
        kind: FREE,
        class: 0,
        listed: false,
        kept: false,
        slices: 0,
        bump: 0,
        cap: 0,
        used: 0,
        prev: NIL,
        next: NIL,
        free: ptr::null_mut(),
    };
}

struct SpanCell(UnsafeCell<Span>);

// SAFETY: each field of a span has one owner at a time (see `Span`).
unsafe impl Sync for SpanCell {}

/// Per slice, the metadata of the span it heads.
static SPAN: [SpanCell; MAX_SLICES] =
    [const { SpanCell(UnsafeCell::new(Span::EMPTY)) }; MAX_SLICES];
/// Per slice, 1 + the head slice of its span, or 0 for memory the pool never had. Every slice of a
/// page or large block holds it, a free span at its first and last slice. Written under [`SPANS`];
/// read without it only for a live block's slice, which no one rewrites while the block lives.
static HEAD: [AtomicU32; MAX_SLICES] = [const { AtomicU32::new(0) }; MAX_SLICES];
/// Per page head, the heap the page belongs to.
static OWNER: [AtomicU8; MAX_SLICES] = [const { AtomicU8::new(0) }; MAX_SLICES];

fn sp(s: u32) -> *mut Span {
    SPAN[s as usize].0.get()
}

/// The head slice of the span `slice` is in; [`NIL`] for memory the pool never had.
fn head_of(slice: u32) -> u32 {
    HEAD[slice as usize].load(Ordering::Relaxed).wrapping_sub(1)
}

fn set_head(from: u32, n: u32, head: u32) {
    for slice in from..from + n {
        HEAD[slice as usize].store(head + 1, Ordering::Relaxed);
    }
}

fn slice_of(addr: usize) -> u32 {
    ((addr - heap::base()) >> SLICE_SHIFT) as u32
}

fn addr_of(s: u32) -> usize {
    heap::base() + ((s as usize) << SLICE_SHIFT)
}

/// The page or large span a live block is in.
fn page_of(block: *mut u8) -> u32 {
    head_of(slice_of(block as usize))
}

/// A free block, linked through its own first word.
struct Node {
    next: *mut Node,
}

/// Spins a waiter makes before it counts as starving.
const STARVE_AFTER: u32 = 1 << 12;

/// A spin lock that lets no waiter starve: it is taken by whoever gets there, until a waiter has
/// spun [`STARVE_AFTER`] times; from then on only starving waiters may take it, and the rest hold
/// back until none is left. Unlike a strict queue, a starving waiter the OS has descheduled holds
/// up nobody for longer than it takes the others to starve too. Everyone spins: the page thread
/// may not block, and what it guards is short.
struct FairLock {
    locked: AtomicBool,
    starving: AtomicU32,
}

impl FairLock {
    const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
            starving: AtomicU32::new(0),
        }
    }

    /// The lock if it is free and no waiter is starving; never waits.
    fn try_lock(&self) -> Option<FairGuard<'_>> {
        (self.starving.load(Ordering::Relaxed) == 0
            && !self.locked.load(Ordering::Relaxed)
            && !self.locked.swap(true, Ordering::Acquire))
        .then(|| FairGuard(self))
    }

    fn lock(&self) -> FairGuard<'_> {
        let mut spins = 0u32;
        let mut starving = false;
        loop {
            let may_take = starving || self.starving.load(Ordering::Relaxed) == 0;
            if may_take
                && !self.locked.load(Ordering::Relaxed)
                && !self.locked.swap(true, Ordering::Acquire)
            {
                if starving {
                    self.starving.fetch_sub(1, Ordering::Relaxed);
                }
                return FairGuard(self);
            }
            std::hint::spin_loop();
            if !starving {
                spins += 1;
                if spins == STARVE_AFTER {
                    starving = true;
                    self.starving.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

struct FairGuard<'a>(&'a FairLock);

impl Drop for FairGuard<'_> {
    fn drop(&mut self) {
        self.0.locked.store(false, Ordering::Release);
    }
}

/// The lock over the span pool: [`POOL`], every span's `kind` and `slices`, and [`HEAD`].
static SPANS: FairLock = FairLock::new();

/// Bins of free spans by length: exact below four slices, then four per doubling.
const BINS: usize = 64;

struct Pool {
    bins: [u32; BINS],
    /// Bit `b` set: bin `b` holds a span.
    map: u64,
}

struct PoolCell(UnsafeCell<Pool>);

// SAFETY: only a holder of `SPANS` touches the pool.
unsafe impl Sync for PoolCell {}

static POOL: PoolCell = PoolCell(UnsafeCell::new(Pool {
    bins: [NIL; BINS],
    map: 0,
}));

/// Slices filed free in the pool's bins, kept by [`bin_insert`] and [`bin_remove`] under [`SPANS`];
/// read without it by [`headroom`].
static POOLED: AtomicUsize = AtomicUsize::new(0);

fn pool(_turn: &FairGuard<'_>) -> *mut Pool {
    POOL.0.get()
}

/// The bin a free span of `n` slices is filed in.
fn bin_of(n: u32) -> usize {
    if n < 4 {
        (n - 1) as usize
    } else {
        let fl = 31 - n.leading_zeros();
        (3 + (fl - 2) * 4 + ((n >> (fl - 2)) & 3)) as usize
    }
}

/// The first bin every span of which holds `n` slices: `n` rounded up to the next bin's start.
fn bin_fit(n: u32) -> usize {
    if n < 4 {
        (n - 1) as usize
    } else {
        let fl = 31 - n.leading_zeros();
        bin_of(n + (1 << (fl - 2)) - 1)
    }
}

fn bin_insert(turn: &FairGuard<'_>, s: u32) {
    let pool = pool(turn);
    // SAFETY: the caller holds `SPANS`, which owns the pool and free spans' links.
    unsafe {
        let n = (*sp(s)).slices;
        POOLED.fetch_add(n as usize, Ordering::Relaxed);
        let b = bin_of(n);
        let head = (*pool).bins[b];
        (*sp(s)).prev = NIL;
        (*sp(s)).next = head;
        if head != NIL {
            (*sp(head)).prev = s;
        }
        (*pool).bins[b] = s;
        (*pool).map |= 1 << b;
    }
}

fn bin_remove(turn: &FairGuard<'_>, s: u32) {
    let pool = pool(turn);
    // SAFETY: as `bin_insert`.
    unsafe {
        let n = (*sp(s)).slices;
        POOLED.fetch_sub(n as usize, Ordering::Relaxed);
        let b = bin_of(n);
        let (prev, next) = ((*sp(s)).prev, (*sp(s)).next);
        if prev == NIL {
            (*pool).bins[b] = next;
            if next == NIL {
                (*pool).map &= !(1 << b);
            }
        } else {
            (*sp(prev)).next = next;
        }
        if next != NIL {
            (*sp(next)).prev = prev;
        }
    }
}

/// File slices `s..s + n` as one free span; its neighbours are not free.
fn file(turn: &FairGuard<'_>, s: u32, n: u32) {
    // SAFETY: the caller holds `SPANS`.
    unsafe {
        (*sp(s)).kind = FREE;
        (*sp(s)).slices = n;
    }
    HEAD[s as usize].store(s + 1, Ordering::Relaxed);
    HEAD[(s + n - 1) as usize].store(s + 1, Ordering::Relaxed);
    bin_insert(turn, s);
}

/// Return slices `s..s + n` to the pool, merged with the free spans on either side.
fn release(turn: &FairGuard<'_>, mut s: u32, mut n: u32) {
    // SAFETY: the caller holds `SPANS`, which owns `kind` and `slices` of every span.
    unsafe {
        if s > 0 {
            let left = head_of(s - 1);
            if left != NIL && (*sp(left)).kind == FREE {
                bin_remove(turn, left);
                n += (*sp(left)).slices;
                s = left;
            }
        }
        let right = s + n;
        if (right as usize) < MAX_SLICES && head_of(right) == right && (*sp(right)).kind == FREE {
            bin_remove(turn, right);
            n += (*sp(right)).slices;
        }
    }
    file(turn, s, n);
}

/// A span of exactly `n` slices from the pool, its tail filed back, or [`NIL`]. Its `kind` is the
/// caller's to set before letting [`SPANS`] go.
fn take(turn: &FairGuard<'_>, n: u32) -> u32 {
    let b = bin_fit(n);
    if b >= BINS {
        return NIL;
    }
    let pool = pool(turn);
    // SAFETY: the caller holds `SPANS`.
    unsafe {
        let fits = (*pool).map & (!0u64 << b);
        if fits == 0 {
            return NIL;
        }
        let s = (*pool).bins[fits.trailing_zeros() as usize];
        bin_remove(turn, s);
        let have = (*sp(s)).slices;
        if have > n {
            file(turn, s + n, have - n);
        }
        (*sp(s)).slices = n;
        s
    }
}

/// Grow memory for a span of `n` slices, by [`GROW_STEP`] when that is more, and file what it added.
/// Each grow runs under [`SPANS`], every other thread that needs a span waiting behind it: in a
/// crowd, 81% of the page thread's samples in frames of 250 ms or more sat behind or in dlmalloc's
/// grows, one per request that outran the free space. Near the maximum the step fails and the need
/// is grown alone.
fn grow(turn: &FairGuard<'_>, n: u32) -> bool {
    let need = n as usize;
    let step = (GROW_STEP.load(Ordering::Relaxed) >> SLICE_SHIFT).max(need);
    let (base, mut got) = match heap::grow(step) {
        Some(base) => (base, step),
        None if step > need => match heap::grow(need) {
            Some(base) => (base, need),
            None => return false,
        },
        None => return false,
    };
    GROWS.fetch_add(1, Ordering::Relaxed);
    if got > need {
        AHEAD.fetch_add(1, Ordering::Relaxed);
    }
    GROWN.fetch_add(got * SLICE, Ordering::Relaxed);
    let s = slice_of(base);
    // No block may end at the top of the address space: its end would wrap to 0.
    if s as usize + got >= MAX_SLICES {
        got = MAX_SLICES - 1 - s as usize;
    }
    if got > 0 {
        release(turn, s, got as u32);
    }
    got >= need
}

/// A span of `n` slices, growing memory when the pool has none; [`NIL`] when memory is exhausted.
fn alloc_span(turn: &FairGuard<'_>, n: u32) -> u32 {
    if n as usize >= MAX_SLICES {
        return NIL;
    }
    drain(turn, DRAIN);
    let mut s = take(turn, n);
    if s == NIL && DEFERRED.load(Ordering::Relaxed) != NIL {
        drain(turn, usize::MAX);
        s = take(turn, n);
    }
    if s == NIL && grow(turn, n) {
        s = take(turn, n);
    }
    s
}

/// Spans freed while [`SPANS`] was busy, linked through their `next`. Any thread pushes; only a
/// holder of [`SPANS`] pops, so a popped head cannot come back while it is read (no ABA).
static DEFERRED: AtomicU32 = AtomicU32::new(NIL);
/// Deferred spans one hold of [`SPANS`] releases first, at most: the hold stays short.
const DRAIN: usize = 2;

fn defer(s: u32) {
    let mut head = DEFERRED.load(Ordering::Relaxed);
    loop {
        // SAFETY: a span on its way to the pool is its freer's alone until listed.
        unsafe { (*sp(s)).next = head };
        match DEFERRED.compare_exchange_weak(head, s, Ordering::Release, Ordering::Relaxed) {
            Ok(_) => return,
            Err(now) => head = now,
        }
    }
}

/// Release up to `n` deferred spans; the caller holds [`SPANS`].
fn drain(turn: &FairGuard<'_>, n: usize) {
    for _ in 0..n {
        let s = DEFERRED.load(Ordering::Acquire);
        if s == NIL {
            return;
        }
        // SAFETY: a listed span's `next` was written before it was listed, and only a holder of
        // `SPANS` pops, so `s` is still listed while it is read.
        let next = unsafe { (*sp(s)).next };
        if DEFERRED
            .compare_exchange(s, next, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            // SAFETY: as above; `slices` is `SPANS`'s.
            release(turn, s, unsafe { (*sp(s)).slices });
        }
    }
}

/// Give a page or large span back to the pool, or to [`DEFERRED`] when [`SPANS`] is busy.
fn retire(s: u32) {
    match SPANS.try_lock() {
        // SAFETY: `slices` is `SPANS`'s, and the span is no longer in use.
        Some(turn) => release(&turn, s, unsafe { (*sp(s)).slices }),
        None => defer(s),
    }
}

/// Bytes memory grows by at least, each time the pool runs out (`set_grow_step`); 0 grows by need.
static GROW_STEP: AtomicUsize = AtomicUsize::new(64 << 20);
/// Since the last [`take_grow_stats`]: memory grows, those that took more than the need (the step),
/// and bytes grown.
static GROWS: AtomicU32 = AtomicU32::new(0);
static AHEAD: AtomicU32 = AtomicU32::new(0);
static GROWN: AtomicUsize = AtomicUsize::new(0);

/// Set the grow step, bytes; 0 grows by need alone.
#[cfg_attr(
    test,
    expect(
        dead_code,
        reason = "the web build's launch lever; the tests swap `GROW_STEP` directly"
    )
)]
pub(crate) fn set_grow_step(bytes: usize) {
    GROW_STEP.store(bytes, Ordering::Relaxed);
}

/// Bytes the allocator can still hand out: the free spans in the pool, and the memory not grown
/// yet. It rises again as memory is freed, which linear memory's size, only ever growing, cannot
/// show. Empty pages a heap keeps and spans waiting on [`DEFERRED`] count as used, so it errs low;
/// a request for one contiguous run can still fail on a fragmented pool.
pub(crate) fn headroom() -> usize {
    POOLED.load(Ordering::Relaxed) * SLICE + heap::max_bytes().saturating_sub(heap::bytes())
}

/// Memory growth since the last call: (grows, grows by the step, bytes grown).
pub(crate) fn take_grow_stats() -> (u32, u32, usize) {
    (
        GROWS.swap(0, Ordering::Relaxed),
        AHEAD.swap(0, Ordering::Relaxed),
        GROWN.swap(0, Ordering::Relaxed),
    )
}

/// Linear memory, which only grows: wasm's own, or natively (the tests and the stress app) a model
/// of it, one reserved region handed out in 64 KiB pages from its start, each grow optionally
/// costing time ([`heap::set_grow_cost`]), so the pool runs and can be measured off the browser.
pub(crate) mod heap {
    use super::SLICE;

    #[cfg(target_arch = "wasm32")]
    pub(super) fn base() -> usize {
        0
    }

    /// Bytes of linear memory.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn bytes() -> usize {
        core::arch::wasm32::memory_size(0) * SLICE
    }

    /// The most linear memory can grow to: the whole 32-bit address space, which the threaded
    /// build declares as its maximum (`scripts/web-build-threads.sh`, `--max-memory`). 4 GiB is one
    /// past what `usize` holds here; a byte short of it is all a gauge needs.
    #[cfg(target_arch = "wasm32")]
    pub(super) fn max_bytes() -> usize {
        usize::MAX
    }

    /// `memory.grow` by `slices`: the address of the first new one.
    #[cfg(target_arch = "wasm32")]
    pub(super) fn grow(slices: usize) -> Option<usize> {
        let prev = core::arch::wasm32::memory_grow(0, slices);
        (prev != usize::MAX).then(|| prev * SLICE)
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod model {
        use std::sync::atomic::{AtomicU64, AtomicUsize};

        pub(super) static BASE: AtomicUsize = AtomicUsize::new(0);
        pub(super) static PAGES: AtomicUsize = AtomicUsize::new(0);
        pub(super) static CAP_PAGES: AtomicUsize = AtomicUsize::new(1 << 14);
        pub(super) static GROW_COST_NS: AtomicU64 = AtomicU64::new(0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    use std::sync::atomic::Ordering::Relaxed;

    /// The modelled cost of one grow, nanoseconds (the stress app's lever).
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(dead_code, reason = "the stress app's lever")]
    pub(crate) fn set_grow_cost(ns: u64) {
        model::GROW_COST_NS.store(ns, Relaxed);
    }

    /// The modelled memory's maximum, bytes; before the first grow only.
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(dead_code, reason = "the stress app's lever")]
    pub(crate) fn set_model_bytes(bytes: usize) {
        model::CAP_PAGES.store((bytes / SLICE).min(super::MAX_SLICES), Relaxed);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn base() -> usize {
        model::BASE.load(Relaxed)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn bytes() -> usize {
        model::PAGES.load(Relaxed) * SLICE
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn max_bytes() -> usize {
        model::CAP_PAGES.load(Relaxed) * SLICE
    }

    /// The model's `memory.grow`, under `SPANS`: the region is reserved at the first one.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn grow(slices: usize) -> Option<usize> {
        use std::alloc::{GlobalAlloc, Layout, System};
        let cap = model::CAP_PAGES.load(Relaxed);
        if model::BASE.load(Relaxed) == 0 {
            let layout = Layout::from_size_align(cap * SLICE, SLICE).ok()?;
            // SAFETY: a non-zero-sized layout.
            let region = unsafe { System.alloc(layout) };
            if region.is_null() {
                return None;
            }
            model::BASE.store(region as usize, Relaxed);
        }
        let used = model::PAGES.load(Relaxed);
        if used + slices > cap {
            return None;
        }
        model::PAGES.store(used + slices, Relaxed);
        let cost = model::GROW_COST_NS.load(Relaxed);
        if cost > 0 {
            let until = std::time::Instant::now() + std::time::Duration::from_nanos(cost);
            while std::time::Instant::now() < until {
                std::hint::spin_loop();
            }
        }
        Some(base() + used * SLICE)
    }
}

/// Heaps, a power of two.
const HEAPS: usize = 64;
/// A heap takes in its inbox at every miss and every this many allocations.
const COLLECT_EVERY: u32 = 256;
/// Bytes of empty pages a heap keeps, each the only page of its class, so a class that empties and
/// fills again does not take a page from the pool each time. Every heap keeping one per class held
/// 88 MB of pages for 3 MB of blocks across 25 heaps in the stress app.
const EMPTY_BUDGET: u32 = 256 * 1024;
/// Tries at this thread's own heap before taking another: whoever holds it is likely freeing one
/// block. Taking another at once spread 16 threads over 25 heaps, each with its own pages.
const OWN_HEAP_TRIES: u32 = 64;

/// One heap: per class, its pages that have a block to give, and the flag that hands them to one
/// thread at a time.
struct Heap {
    busy: AtomicBool,
    /// Blocks of this heap's pages freed while another thread held it, newest first.
    inbox: AtomicPtr<Node>,
    lists: UnsafeCell<Lists>,
}

struct Lists {
    pages: [u32; N_CLASSES],
    allocs: u32,
    /// Bytes of the kept empty pages.
    kept: u32,
}

// SAFETY: `lists` is only touched by the thread that set `busy`.
unsafe impl Sync for Heap {}

impl Heap {
    const fn new() -> Self {
        Self {
            busy: AtomicBool::new(false),
            inbox: AtomicPtr::new(ptr::null_mut()),
            lists: UnsafeCell::new(Lists {
                pages: [NIL; N_CLASSES],
                allocs: 0,
                kept: 0,
            }),
        }
    }
}

static HEAP_TABLE: [Heap; HEAPS] = [const { Heap::new() }; HEAPS];

/// Heap `index`, if no other thread holds it; never waits. The guard is built only on success: a
/// guard dropped after a failed take would release another thread's heap.
fn try_take(index: usize) -> Option<HeapGuard> {
    let heap = &HEAP_TABLE[index];
    (!heap.busy.load(Ordering::Relaxed)
        && heap
            .busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok())
    .then(|| HeapGuard {
        heap,
        index: index as u8,
    })
}

/// This thread's heap, by the 64 KiB window its stack is in, or the next one free.
fn take_heap() -> HeapGuard {
    let probe = 0u8;
    let stack = std::hint::black_box(&probe) as *const u8 as usize;
    // Fibonacci hashing in 32 bits, the top bits picking the heap.
    let mixed = ((stack >> 16) as u32).wrapping_mul(0x9E37_79B1);
    let start = (mixed >> (32 - HEAPS.trailing_zeros())) as usize;
    for _ in 0..OWN_HEAP_TRIES {
        if let Some(heap) = try_take(start) {
            return heap;
        }
        std::hint::spin_loop();
    }
    let mut i = start;
    loop {
        if let Some(heap) = try_take(i) {
            return heap;
        }
        i = (i + 1) % HEAPS;
        if i == start {
            std::hint::spin_loop();
        }
    }
}

struct HeapGuard {
    heap: &'static Heap,
    index: u8,
}

impl Drop for HeapGuard {
    fn drop(&mut self) {
        self.heap.busy.store(false, Ordering::Release);
    }
}

impl HeapGuard {
    fn lists(&mut self) -> &mut Lists {
        // SAFETY: holding the guard is holding `busy`.
        unsafe { &mut *self.heap.lists.get() }
    }

    /// A block of class `c`: from the first listed page, from the inbox's returns, or from a new page.
    fn alloc(&mut self, c: usize) -> *mut u8 {
        let lists = self.lists();
        lists.allocs = lists.allocs.wrapping_add(1);
        if lists.allocs.is_multiple_of(COLLECT_EVERY) {
            self.collect();
        }
        let mut p = self.lists().pages[c];
        if p == NIL {
            self.collect();
            p = self.lists().pages[c];
            if p == NIL {
                p = self.new_page(c);
                if p == NIL {
                    return ptr::null_mut();
                }
            }
        }
        // SAFETY: a listed page is this heap's, and has a block to give.
        unsafe {
            let page = sp(p);
            if (*page).kept {
                (*page).kept = false;
                self.lists().kept -= (*page).slices << SLICE_SHIFT;
            }
            let block = if (*page).free.is_null() {
                let at = addr_of(p) + (*page).bump as usize * CLASSES[c];
                (*page).bump += 1;
                at as *mut u8
            } else {
                let node = (*page).free;
                (*page).free = (*node).next;
                node.cast()
            };
            (*page).used += 1;
            if (*page).free.is_null() && (*page).bump == (*page).cap {
                self.unlink(p);
            }
            block
        }
    }

    /// A new page of class `c` from the pool, listed in this heap; [`NIL`] when memory is exhausted.
    fn new_page(&mut self, c: usize) -> u32 {
        let n = page_slices(c);
        let p = {
            let turn = SPANS.lock();
            let p = alloc_span(&turn, n);
            if p == NIL {
                return NIL;
            }
            // SAFETY: `kind` is `SPANS`'s.
            unsafe { (*sp(p)).kind = PAGE };
            set_head(p, n, p);
            p
        };
        OWNER[p as usize].store(self.index, Ordering::Relaxed);
        // SAFETY: a page fresh from the pool is its heap's alone.
        unsafe {
            let page = sp(p);
            (*page).class = c as u8;
            (*page).bump = 0;
            (*page).cap = (n as usize * SLICE / CLASSES[c]) as u32;
            (*page).used = 0;
            (*page).kept = false;
            (*page).free = ptr::null_mut();
        }
        self.link(p);
        p
    }

    fn link(&mut self, p: u32) {
        // SAFETY: page `p` and the pages listed with it are this heap's.
        unsafe {
            let page = sp(p);
            let c = (*page).class as usize;
            let head = self.lists().pages[c];
            (*page).prev = NIL;
            (*page).next = head;
            if head != NIL {
                (*sp(head)).prev = p;
            }
            self.lists().pages[c] = p;
            (*page).listed = true;
        }
    }

    fn unlink(&mut self, p: u32) {
        // SAFETY: as `link`.
        unsafe {
            let page = sp(p);
            let (prev, next) = ((*page).prev, (*page).next);
            if prev == NIL {
                self.lists().pages[(*page).class as usize] = next;
            } else {
                (*sp(prev)).next = next;
            }
            if next != NIL {
                (*sp(next)).prev = prev;
            }
            (*page).listed = false;
        }
    }

    /// Free into their pages the blocks other threads left in this heap's inbox.
    fn collect(&mut self) {
        let mut node = self.heap.inbox.swap(ptr::null_mut(), Ordering::Acquire);
        while !node.is_null() {
            // SAFETY: an inbox block is an allocated block of one of this heap's pages, its next
            // link written before it was listed.
            unsafe {
                let next = (*node).next;
                self.free_local(page_of(node.cast()), node.cast());
                node = next;
            }
        }
    }

    /// Free `block` into page `p`, listing the page if it was full and giving it back to the pool
    /// once empty, unless it is its class's only page here and the heap's [`EMPTY_BUDGET`] has room.
    ///
    /// SAFETY: `block` is an allocated block of page `p`, which is this heap's.
    unsafe fn free_local(&mut self, p: u32, block: *mut u8) {
        unsafe {
            let page = sp(p);
            let node = block.cast::<Node>();
            (*node).next = (*page).free;
            (*page).free = node;
            (*page).used -= 1;
            if !(*page).listed {
                self.link(p);
            }
            if (*page).used == 0 {
                let bytes = (*page).slices << SLICE_SHIFT;
                let only = self.lists().pages[(*page).class as usize] == p && (*page).next == NIL;
                if only && self.lists().kept + bytes <= EMPTY_BUDGET {
                    (*page).kept = true;
                    self.lists().kept += bytes;
                } else {
                    self.unlink(p);
                    retire(p);
                }
            }
        }
    }
}

/// Free a class block: into its page if its heap is free, else onto that heap's inbox.
///
/// SAFETY: `block` is an allocated class block.
unsafe fn free_small(block: *mut u8) {
    let p = page_of(block);
    let index = OWNER[p as usize].load(Ordering::Relaxed) as usize;
    match try_take(index) {
        // SAFETY: the block is page `p`'s, and `p` is heap `index`'s.
        Some(mut heap) => unsafe { heap.free_local(p, block) },
        None => {
            let inbox = &HEAP_TABLE[index].inbox;
            let node = block.cast::<Node>();
            let mut head = inbox.load(Ordering::Relaxed);
            loop {
                // SAFETY: the block is free and ours until listed.
                unsafe { (*node).next = head };
                match inbox.compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
                {
                    Ok(_) => return,
                    Err(now) => head = now,
                }
            }
        }
    }
}

/// A block over the classes: a span of its own, at an alignment past a slice's cut from a longer
/// one with the ends given back.
fn alloc_large(layout: Layout) -> *mut u8 {
    let Some(n) = slices_for(layout.size()) else {
        return ptr::null_mut();
    };
    let extra = (layout.align() / SLICE).saturating_sub(1) as u32;
    let turn = SPANS.lock();
    let s = alloc_span(&turn, n.saturating_add(extra));
    if s == NIL {
        return ptr::null_mut();
    }
    let a = slice_of(addr_of(s).next_multiple_of(layout.align().max(SLICE)));
    // SAFETY: `kind` and `slices` are `SPANS`'s.
    unsafe {
        (*sp(a)).kind = LARGE;
        (*sp(a)).slices = n;
    }
    set_head(a, n, a);
    if a > s {
        release(&turn, s, a - s);
    }
    let end = s + n + extra;
    if a + n < end {
        release(&turn, a + n, end - a - n);
    }
    addr_of(a) as *mut u8
}

/// Resize a large block in place: shrink it, or grow it into a free span right after it.
fn resize_large(block: *mut u8, new_size: usize) -> bool {
    let Some(want) = slices_for(new_size) else {
        return false;
    };
    let s = page_of(block);
    // SAFETY: a live block's `slices` changes only by its owner's resize.
    let have = unsafe { (*sp(s)).slices };
    if want == have {
        return true;
    }
    let turn = SPANS.lock();
    // SAFETY: the caller holds `SPANS`.
    unsafe {
        if want < have {
            (*sp(s)).slices = want;
            release(&turn, s + want, have - want);
            return true;
        }
        let right = s + have;
        let need = want - have;
        if (right as usize) < MAX_SLICES
            && head_of(right) == right
            && (*sp(right)).kind == FREE
            && (*sp(right)).slices >= need
        {
            bin_remove(&turn, right);
            let spare = (*sp(right)).slices - need;
            if spare > 0 {
                file(&turn, right + need, spare);
            }
            set_head(right, need, s);
            (*sp(s)).slices = want;
            return true;
        }
    }
    false
}

/// Class blocks from pages in per-thread heaps, everything else a span of its own.
pub(crate) struct PageAlloc;

// SAFETY: a class block is at least the requested size at an alignment `class_of` admits; a large
// block is whole slices at its alignment; every block is handed out once until freed.
unsafe impl GlobalAlloc for PageAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match class_of(layout) {
            Some(c) => take_heap().alloc(c),
            None => alloc_large(layout),
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let block = unsafe { self.alloc(layout) };
        if !block.is_null() {
            unsafe { ptr::write_bytes(block, 0, layout.size()) };
        }
        block
    }

    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        match class_of(layout) {
            Some(_) => unsafe { free_small(block) },
            None => retire(page_of(block)),
        }
    }

    unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees `new_size` is a valid size at `layout.align()`.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        match (class_of(layout), class_of(new_layout)) {
            (Some(a), Some(b)) if a == b => return block,
            (None, None) if layout.align() <= SLICE && resize_large(block, new_size) => {
                return block;
            }
            _ => {}
        }
        let moved = unsafe { self.alloc(new_layout) };
        if !moved.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(block, moved, layout.size().min(new_size));
                self.dealloc(block, layout);
            }
        }
        moved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every size up to the largest class lands in the smallest class that holds it, and an
    /// over-aligned one in a power-of-two class at least its alignment.
    #[test]
    fn a_size_takes_the_smallest_class_that_holds_it() {
        for size in 1..=MAX_SMALL {
            let c = class_of(Layout::from_size_align(size, 8).unwrap()).unwrap();
            assert!(CLASSES[c] >= size, "{size} in class {}", CLASSES[c]);
            assert!(
                c == 0 || CLASSES[c - 1] < size,
                "{size} could take {}",
                CLASSES[c - 1]
            );
        }
        assert_eq!(
            class_of(Layout::from_size_align(MAX_SMALL + 1, 8).unwrap()),
            None
        );
        for (size, align, class) in [(64, 32, 64), (24, 4096, 4096), (5000, 64, 8192)] {
            let c = class_of(Layout::from_size_align(size, align).unwrap()).unwrap();
            assert_eq!(CLASSES[c], class, "{size} at {align}");
        }
        assert_eq!(
            class_of(Layout::from_size_align(16, 1 << 17).unwrap()),
            None
        );
    }

    /// A span is filed in the bin of its length, and the bin a fit searches from holds no span
    /// shorter than the request.
    #[test]
    fn a_fit_never_finds_a_short_span() {
        let mut low = [u32::MAX; BINS];
        let mut last = 0;
        for n in 1..=MAX_SLICES as u32 {
            let b = bin_of(n);
            assert!(b >= last && b < BINS, "{n} filed in {b}");
            last = b;
            low[b] = low[b].min(n);
        }
        for n in 1..MAX_SLICES as u32 {
            let b = bin_fit(n);
            assert!(b < BINS);
            assert!(
                low[b] == u32::MAX || low[b] >= n,
                "{n} searches bin {b} from {}",
                low[b]
            );
            assert!(b <= bin_of(n) + 1, "{n} skips past the next bin");
        }
    }

    /// Fill a block with a pattern its owner can check.
    fn fill(block: *mut u8, size: usize, seed: u8) {
        for i in 0..size {
            unsafe { *block.add(i) = seed.wrapping_add(i as u8) };
        }
    }

    fn check(block: *const u8, size: usize, seed: u8) {
        for i in 0..size {
            assert_eq!(unsafe { *block.add(i) }, seed.wrapping_add(i as u8));
        }
    }

    /// Eight threads allocate, fill, check and free mixed sizes and alignments, from the classes
    /// through large and over-aligned spans, half of them freeing blocks another thread allocated:
    /// no block is handed out twice or misaligned.
    #[test]
    fn threads_share_it_without_overlap() {
        let (tx, rx) = std::sync::mpsc::channel::<(usize, usize, usize, u8)>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        let threads: Vec<_> = (0..8u8)
            .map(|t| {
                let tx = tx.clone();
                let rx = rx.clone();
                std::thread::spawn(move || {
                    let mut live = Vec::new();
                    let mut x = 0x2545_F491u32.wrapping_mul(u32::from(t) + 1);
                    for i in 0..20_000usize {
                        x ^= x << 13;
                        x ^= x >> 17;
                        x ^= x << 5;
                        let size = if x.is_multiple_of(16) {
                            1 + (x as usize >> 4) % 300_000
                        } else {
                            1 + (x as usize % 4096)
                        };
                        let align = if x.is_multiple_of(97) {
                            1 << 17
                        } else {
                            1 << ((x as usize >> 20) % 6)
                        };
                        let layout = Layout::from_size_align(size, align).unwrap();
                        let block = unsafe { PageAlloc.alloc(layout) };
                        assert!(!block.is_null());
                        assert_eq!(block as usize % align, 0);
                        let seed = (i as u8).wrapping_add(t);
                        fill(block, size, seed);
                        if t % 2 == 0 && i % 3 == 0 {
                            tx.send((block as usize, size, align, seed)).unwrap();
                        } else {
                            live.push((block as usize, size, align, seed));
                        }
                        if live.len() > 64 {
                            let (b, s, a, sd) = live.swap_remove(x as usize % live.len());
                            check(b as *const u8, s, sd);
                            unsafe {
                                PageAlloc
                                    .dealloc(b as *mut u8, Layout::from_size_align(s, a).unwrap())
                            };
                        }
                        if t % 2 == 1 {
                            if let Ok((b, s, a, sd)) = rx.lock().unwrap().try_recv() {
                                check(b as *const u8, s, sd);
                                unsafe {
                                    PageAlloc.dealloc(
                                        b as *mut u8,
                                        Layout::from_size_align(s, a).unwrap(),
                                    )
                                };
                            }
                        }
                    }
                    for (b, s, a, sd) in live {
                        check(b as *const u8, s, sd);
                        unsafe {
                            PageAlloc.dealloc(b as *mut u8, Layout::from_size_align(s, a).unwrap())
                        };
                    }
                })
            })
            .collect();
        drop(tx);
        for t in threads {
            t.join().unwrap();
        }
        while let Ok((b, s, a, sd)) = rx.lock().unwrap().try_recv() {
            check(b as *const u8, s, sd);
            unsafe { PageAlloc.dealloc(b as *mut u8, Layout::from_size_align(s, a).unwrap()) };
        }
    }

    /// A free heap, held for the test.
    fn hold_a_heap() -> HeapGuard {
        (0..HEAPS).cycle().find_map(try_take).unwrap()
    }

    fn inbox_has(heap: &Heap, block: *mut u8) -> bool {
        let mut at = heap.inbox.load(Ordering::Acquire);
        while !at.is_null() {
            if at.cast() == block {
                return true;
            }
            at = unsafe { (*at).next };
        }
        false
    }

    /// A block freed while another thread holds its heap waits on that heap's inbox, and the
    /// holder's next collect frees it into its page.
    #[test]
    fn a_free_under_a_held_heap_goes_to_its_inbox() {
        let mut heap = hold_a_heap();
        let layout = Layout::from_size_align(64, 8).unwrap();
        let block = heap.alloc(class_of(layout).unwrap());
        let p = page_of(block);
        let used = unsafe { (*sp(p)).used };
        let freeing = block as usize;
        std::thread::spawn(move || unsafe { PageAlloc.dealloc(freeing as *mut u8, layout) })
            .join()
            .unwrap();
        assert!(
            inbox_has(heap.heap, block),
            "the heap was held, so it must be listed"
        );
        heap.collect();
        assert!(!inbox_has(heap.heap, block));
        assert!(
            unsafe { (*sp(p)).used } < used,
            "the collect freed it into its page"
        );
    }

    /// A page whose blocks are all freed goes back to the pool unless it is its class's only one.
    #[test]
    fn an_emptied_page_goes_back_to_the_pool() {
        let mut heap = hold_a_heap();
        let c = class_of(Layout::from_size_align(1024, 8).unwrap()).unwrap();
        let blocks: Vec<_> = (0..64 * 3 + 1).map(|_| heap.alloc(c)).collect();
        let mut pages: Vec<_> = blocks.iter().map(|b| page_of(*b)).collect();
        pages.dedup();
        assert!(pages.len() >= 3);
        for b in blocks {
            unsafe { heap.free_local(page_of(b), b) };
        }
        // Pages that still hold another test's blocks stay; of the emptied ones, one may.
        let kept = pages
            .iter()
            .filter(|p| unsafe {
                (*sp(**p)).kind == PAGE
                    && (*sp(**p)).listed
                    && (*sp(**p)).used == 0
                    && OWNER[**p as usize].load(Ordering::Relaxed) == heap.index
            })
            .count();
        assert!(kept <= 1, "{kept} of {} emptied pages kept", pages.len());
    }

    fn deferred_has(s: u32) -> bool {
        let mut at = DEFERRED.load(Ordering::Acquire);
        while at != NIL {
            if at == s {
                return true;
            }
            at = unsafe { (*sp(at)).next };
        }
        false
    }

    /// A large block freed while the pool is held elsewhere waits on the deferred list instead of
    /// the lock, and the next holder gives it back.
    #[test]
    fn a_large_free_under_a_busy_pool_is_deferred_and_released() {
        let layout = Layout::from_size_align(256 * 1024, 8).unwrap();
        let block = unsafe { PageAlloc.alloc(layout) };
        assert!(!block.is_null());
        let s = page_of(block);
        let turn = SPANS.lock();
        let freeing = block as usize;
        std::thread::spawn(move || unsafe { PageAlloc.dealloc(freeing as *mut u8, layout) })
            .join()
            .unwrap();
        assert!(
            deferred_has(s),
            "the free returned without the lock, so it must be listed"
        );
        drain(&turn, usize::MAX);
        assert!(!deferred_has(s));
        assert!(pooled(&turn, s).is_some(), "the holder's drain released it");
    }

    /// The free span filed in the pool that `slice` lies in: its head and length.
    fn pooled(turn: &FairGuard<'_>, slice: u32) -> Option<(u32, u32)> {
        let pool = pool(turn);
        (0..BINS).find_map(|b| {
            let mut s = unsafe { (*pool).bins[b] };
            while s != NIL {
                let n = unsafe { (*sp(s)).slices };
                if (s..s + n).contains(&slice) {
                    return Some((s, n));
                }
                s = unsafe { (*sp(s)).next };
            }
            None
        })
    }

    /// A span split off and given back merges with its free neighbour into one span again.
    #[test]
    fn freed_neighbours_merge() {
        let turn = SPANS.lock();
        let s = alloc_span(&turn, 8);
        assert_ne!(s, NIL);
        unsafe { (*sp(s)).kind = LARGE };
        set_head(s, 8, s);
        unsafe { (*sp(s)).slices = 4 };
        release(&turn, s + 4, 4);
        assert!(pooled(&turn, s + 4).is_some() && pooled(&turn, s).is_none());
        release(&turn, s, 4);
        let (head, slices) = pooled(&turn, s).unwrap();
        assert!(head + slices >= s + 8, "the halves merged into one span");
    }

    /// A grow takes the step, not the need, and files it: the next span that fits grows nothing.
    /// Under the pool's lock throughout, so no other test moves the counters meanwhile.
    #[test]
    fn a_grow_grows_ahead_and_the_next_span_fits() {
        let turn = SPANS.lock();
        let step = GROW_STEP.swap(8 << 20, Ordering::Relaxed);
        let _ = take_grow_stats();
        assert!(grow(&turn, 1));
        assert_eq!(take_grow_stats(), (1, 1, 8 << 20), "one grow, by the step");
        let s = take(&turn, 128);
        assert_ne!(s, NIL, "the step left a span of its size");
        release(&turn, s, 128);
        GROW_STEP.store(step, Ordering::Relaxed);
    }

    /// A span taken from the pool lowers the headroom by its size, and giving it back restores it;
    /// a grow on the way moves memory from not grown to pooled and changes nothing. Under the
    /// pool's lock throughout, so no other test moves the pool meanwhile.
    #[test]
    fn a_span_out_lowers_the_headroom_and_its_return_restores_it() {
        let turn = SPANS.lock();
        let before = headroom();
        let mut s = take(&turn, 16);
        if s == NIL {
            assert!(grow(&turn, 16));
            s = take(&turn, 16);
        }
        assert_ne!(s, NIL);
        assert_eq!(headroom(), before - 16 * SLICE);
        release(&turn, s, 16);
        assert_eq!(headroom(), before);
    }

    /// Growing and shrinking across the classes and through large spans, in place and moved, keeps
    /// the bytes, and a zeroed block reads zero even when it is a reused one.
    #[test]
    fn realloc_keeps_the_bytes_and_zeroed_reads_zero() {
        let layout = Layout::from_size_align(24, 8).unwrap();
        let mut block = unsafe { PageAlloc.alloc(layout) };
        fill(block, 24, 7);
        let mut size = 24;
        for next in [
            30, 100, 2048, 5000, 70_000, 300_000, 1_000_000, 200_000, 1500, 9,
        ] {
            block = unsafe {
                PageAlloc.realloc(block, Layout::from_size_align(size, 8).unwrap(), next)
            };
            assert!(!block.is_null());
            check(block, size.min(next).min(24), 7);
            size = next;
        }
        unsafe { PageAlloc.dealloc(block, Layout::from_size_align(size, 8).unwrap()) };

        for size in [64, 300_000] {
            let layout = Layout::from_size_align(size, 16).unwrap();
            let dirty = unsafe { PageAlloc.alloc(layout) };
            fill(dirty, size, 0xAA);
            unsafe { PageAlloc.dealloc(dirty, layout) };
            let zeroed = unsafe { PageAlloc.alloc_zeroed(layout) };
            assert!((0..size).all(|i| unsafe { *zeroed.add(i) } == 0));
            unsafe { PageAlloc.dealloc(zeroed, layout) };
        }
    }
}
