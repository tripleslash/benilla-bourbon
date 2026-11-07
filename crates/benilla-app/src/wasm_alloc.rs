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
//! - Pages are cut from the lowest free span, large blocks from the span that fits them best. A
//!   page a few long-lived blocks keep is kept for good, and pages placed by fit landed in every run
//!   the large blocks free, so memory with most of it free held no long run; at the bottom, they
//!   leave the runs above whole.
//! - A block up to 64 KiB comes from a page: a span of one slice (classes up to 8 KiB) or two, cut
//!   into blocks of one size class. A page belongs to one heap, and a heap is held by one thread at a
//!   time through a flag nobody waits on: a thread takes the heap its stack address picks, or the
//!   next free one. Allocating from a page takes no lock.
//! - One live block holds its whole page, and a live block cannot be moved, so pages are made to
//!   empty by themselves: a page that gets a block back goes last in its list, and sparse pages at
//!   the head are passed over while a fuller one has room, so new blocks stop landing in them. With
//!   none fuller, the first is filled, one page at a time, and the rest are left to empty.
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
use std::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering,
};

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
/// Classes up to this size are cut from one-slice pages, larger ones from [`MEDIUM_SLICES`]. Two
/// slices hold two blocks of the largest class: a survivor pins one neighbour at most there, and a
/// page takes [`SPANS`] once per two blocks, where a page per block would take it for each.
const SMALL_PAGE_MAX: usize = 8 * 1024;
const MEDIUM_SLICES: u32 = 2;

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
    /// Bit `s` set: a free span starts at slice `s`. A page is taken from the lowest one that holds
    /// it ([`take_low`]).
    heads: [u64; MAX_SLICES / 64],
    /// Bit `w` set: `heads[w]` is not 0.
    summary: [u64; MAX_SLICES / 64 / 64],
}

struct PoolCell(UnsafeCell<Pool>);

// SAFETY: only a holder of `SPANS` touches the pool.
unsafe impl Sync for PoolCell {}

static POOL: PoolCell = PoolCell(UnsafeCell::new(Pool {
    bins: [NIL; BINS],
    heads: [0; MAX_SLICES / 64],
    summary: [0; MAX_SLICES / 64 / 64],
}));

/// Slices filed free in the pool's bins, kept by [`bin_insert`] and [`bin_remove`] under [`SPANS`];
/// read without it by [`headroom`].
static POOLED: AtomicUsize = AtomicUsize::new(0);
/// Bit `b` set: bin `b` holds a span. Written under [`SPANS`]; read without it by [`largest_run`].
static MAP: AtomicU64 = AtomicU64::new(0);

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

/// The shortest span bin `b` files: [`bin_of`] inverted.
fn bin_low(b: usize) -> u32 {
    if b < 3 {
        b as u32 + 1
    } else {
        let (fl, sub) = (2 + (b as u32 - 3) / 4, (b as u32 - 3) % 4);
        (4 + sub) << (fl - 2)
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
        MAP.fetch_or(1 << b, Ordering::Relaxed);
        let w = s as usize / 64;
        (*pool).heads[w] |= 1 << (s % 64);
        (*pool).summary[w / 64] |= 1 << (w % 64);
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
                MAP.fetch_and(!(1 << b), Ordering::Relaxed);
            }
        } else {
            (*sp(prev)).next = next;
        }
        if next != NIL {
            (*sp(next)).prev = prev;
        }
        let w = s as usize / 64;
        (*pool).heads[w] &= !(1 << (s % 64));
        if (*pool).heads[w] == 0 {
            (*pool).summary[w / 64] &= !(1 << (w % 64));
        }
    }
}

/// The lowest free span head at or above slice `from`, or [`NIL`].
fn next_head(pool: &Pool, from: u32) -> u32 {
    let mut w = from as usize / 64;
    if w >= pool.heads.len() {
        return NIL;
    }
    let bits = pool.heads[w] & (!0u64 << (from % 64));
    if bits != 0 {
        return (w * 64) as u32 + bits.trailing_zeros();
    }
    w += 1;
    let mut sw = w / 64;
    while sw < pool.summary.len() {
        let words = if sw == w / 64 {
            pool.summary[sw] & (!0u64 << (w % 64))
        } else {
            pool.summary[sw]
        };
        if words != 0 {
            let w = sw * 64 + words.trailing_zeros() as usize;
            return (w * 64) as u32 + pool.heads[w].trailing_zeros();
        }
        sw += 1;
    }
    NIL
}

/// Free spans [`take_low`] looks at before leaving the request to [`take`]'s fit.
const LOW_LOOK: u32 = 64;

/// A span of exactly `n` slices cut from the lowest free span that holds it, its tail filed back,
/// or [`NIL`] when none of the lowest [`LOW_LOOK`] does. Pages, which a few long-lived blocks
/// hold for good, gather at the bottom of memory this way, and the runs above them stay whole
/// for the large blocks, where a fit by length scatters pages through every span it splits.
fn take_low(turn: &FairGuard<'_>, n: u32) -> u32 {
    // SAFETY: the caller holds `SPANS`.
    unsafe {
        let pool = &*pool(turn);
        let mut s = next_head(pool, 0);
        for _ in 0..LOW_LOOK {
            if s == NIL {
                return NIL;
            }
            if (*sp(s)).slices >= n {
                return cut(turn, s, n);
            }
            s = next_head(pool, s + 1);
        }
        NIL
    }
}

/// Take free span `s` out of the pool and cut its first `n` slices, filing the rest back.
fn cut(turn: &FairGuard<'_>, s: u32, n: u32) -> u32 {
    bin_remove(turn, s);
    // SAFETY: the caller holds `SPANS`.
    unsafe {
        let have = (*sp(s)).slices;
        if have > n {
            file(turn, s + n, have - n);
        }
        (*sp(s)).slices = n;
    }
    s
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

/// The first span holding `n` slices in the bin `n` is filed in, or [`NIL`]: only some of that bin's
/// spans hold it, so [`bin_fit`] searches past it, and this walk is for when no later bin has one.
fn fit_in_own_bin(turn: &FairGuard<'_>, n: u32) -> u32 {
    let pool = pool(turn);
    // SAFETY: the caller holds `SPANS`.
    unsafe {
        let mut s = (*pool).bins[bin_of(n)];
        while s != NIL && (*sp(s)).slices < n {
            s = (*sp(s)).next;
        }
        s
    }
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
        let fits = MAP.load(Ordering::Relaxed) & (!0u64 << b);
        let s = if fits == 0 {
            fit_in_own_bin(turn, n)
        } else {
            (*pool).bins[fits.trailing_zeros() as usize]
        };
        if s == NIL {
            return NIL;
        }
        cut(turn, s, n)
    }
}

/// Grow memory for a span of `n` slices, by [`GROW_STEP`] when that is more, and file what it added.
/// Each grow runs under [`SPANS`], every other thread that needs a span waiting behind it: in a
/// crowd, 81% of the page thread's samples in frames of 250 ms or more sat behind or in dlmalloc's
/// grows, one per request that outran the free space. Near the maximum the step is what is left, even
/// under the need, as it merges with a free span below it; a step the engine refuses is grown by the
/// need alone. False when memory did not grow. A span of `n` slices may still not fit when it did.
fn grow(turn: &FairGuard<'_>, n: u32) -> bool {
    let need = n as usize;
    // On wasm this is a slice short of the 4 GiB, a slice no block could use (below).
    let left = heap::max_bytes().saturating_sub(heap::bytes()) >> SLICE_SHIFT;
    if left == 0 {
        return false;
    }
    let step = (GROW_STEP.load(Ordering::Relaxed) >> SLICE_SHIFT)
        .max(need)
        .min(left);
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
    got > 0
}

/// A span of `n` slices, growing memory when the pool has none; [`NIL`] when memory is exhausted.
fn alloc_span(turn: &FairGuard<'_>, n: u32) -> u32 {
    if n as usize >= MAX_SLICES {
        return NIL;
    }
    drain(turn, DRAIN);
    let mut s = place(turn, n);
    if s == NIL && DEFERRED.load(Ordering::Relaxed) != NIL {
        drain(turn, usize::MAX);
        s = place(turn, n);
    }
    if s == NIL && grow(turn, n) {
        s = place(turn, n);
    }
    s
}

/// A span of `n` slices from the pool: a page's from the bottom of memory ([`take_low`]), a large
/// block's by fit.
fn place(turn: &FairGuard<'_>, n: u32) -> u32 {
    if n <= MEDIUM_SLICES {
        let s = take_low(turn, n);
        if s != NIL {
            return s;
        }
    }
    take(turn, n)
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
        Some(turn) => {
            release(&turn, s, unsafe { (*sp(s)).slices });
            // So a span deferred in a burst does not wait for the next allocation of a span.
            drain(&turn, DRAIN);
        }
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

/// Bytes the largest block can still get in one run: the shortest span the pool's highest filled
/// bin files, all a read without the lock can know, or the memory not grown yet, whichever is more. Near the top of the address space [`headroom`] is mostly scattered free slices, and a
/// block bigger than every run fails while their sum looks ample. Read without [`SPANS`], so a span
/// filed or taken meanwhile may or may not show; a grow's merge with the free span below it and
/// [`DEFERRED`] spans are not counted, so it errs low.
pub(crate) fn largest_run() -> usize {
    (pooled_run() as usize * SLICE).max(heap::max_bytes().saturating_sub(heap::bytes()))
}

/// Slices the pool can give in one span: its highest filled bin's floor, or 0 when it is empty.
fn pooled_run() -> u32 {
    match MAP.load(Ordering::Relaxed) {
        0 => 0,
        map => bin_low(63 - map.leading_zeros() as usize),
    }
}

/// The upper bound of each size bucket [`census_line`] splits large blocks into, bytes.
const LARGE_BUCKETS: [usize; 6] = [128 << 10, 512 << 10, 1 << 20, 4 << 20, 16 << 20, usize::MAX];
/// How many size classes [`census_line`] names, heaviest first.
const CENSUS_CLASSES: usize = 8;

/// Linear memory by what the allocator holds it for: live blocks against the pages they pin, large
/// blocks by size, free spans and spans waiting on [`DEFERRED`]. A walk of the span table under
/// [`SPANS`], so `kind` and `slices` are exact; a page's `used` is read racily and may be a free
/// behind, which a report 30 s apart does not feel.
#[cfg_attr(
    test,
    expect(
        dead_code,
        reason = "the web build's growth report reads it; the test reads `census`"
    )
)]
pub(crate) fn census_line() -> String {
    let c = census();
    let mb = |bytes: u64| bytes >> 20;
    let pages: u64 = c.pages.iter().map(|(span, _)| span).sum();
    let live: u64 = c.pages.iter().map(|(_, live)| live).sum();
    let large: u64 = c.large.iter().map(|(_, bytes)| bytes).sum();
    let mut line = format!(
        "\n[diag]   alloc: pages {}MB holding {}MB live, large {}MB, free {}MB, deferred {}MB\n\
         [diag]   alloc large:",
        mb(pages),
        mb(live),
        mb(large),
        mb(c.free),
        mb(c.deferred),
    );
    for (bound, (n, bytes)) in LARGE_BUCKETS.iter().zip(c.large) {
        if *bound == usize::MAX {
            line.push_str(&format!(" rest {n}/{}MB", mb(bytes)));
        } else {
            line.push_str(&format!(" <{}K {n}/{}MB", bound >> 10, mb(bytes)));
        }
    }
    line.push_str("\n[diag]   alloc classes (span/live):");
    let mut classes: Vec<usize> = (0..N_CLASSES).collect();
    classes.sort_unstable_by_key(|&i| std::cmp::Reverse(c.pages[i].0));
    for i in classes.into_iter().take(CENSUS_CLASSES) {
        let (span, live) = c.pages[i];
        line.push_str(&format!(" {}B {}/{}MB", CLASSES[i], mb(span), mb(live)));
    }
    line
}

struct Census {
    /// Per class: bytes of its pages, bytes of its live blocks.
    pages: [(u64, u64); N_CLASSES],
    /// Per [`LARGE_BUCKETS`] entry: large blocks, their bytes.
    large: [(u32, u64); LARGE_BUCKETS.len()],
    free: u64,
    deferred: u64,
}

fn census() -> Census {
    let mut c = Census {
        pages: [(0, 0); N_CLASSES],
        large: [(0, 0); LARGE_BUCKETS.len()],
        free: 0,
        deferred: 0,
    };
    let _turn = SPANS.lock();
    let top = (heap::bytes() >> SLICE_SHIFT).min(MAX_SLICES) as u32;
    let mut s = 0;
    while s < top {
        if head_of(s) != s {
            s += 1;
            continue;
        }
        // SAFETY: the walk holds `SPANS`, which owns `kind` and `slices`; the page fields are plain
        // integers read for a report, written only by their heap's holder.
        unsafe {
            let n = (*sp(s)).slices.max(1);
            let bytes = u64::from(n) << SLICE_SHIFT;
            match (*sp(s)).kind {
                FREE => c.free += bytes,
                PAGE => {
                    let class = usize::from((*sp(s)).class).min(N_CLASSES - 1);
                    let used = ptr::read_volatile(ptr::addr_of!((*sp(s)).used));
                    c.pages[class].0 += bytes;
                    c.pages[class].1 += u64::from(used) * CLASSES[class] as u64;
                }
                _ => {
                    let bucket = LARGE_BUCKETS
                        .iter()
                        .position(|&bound| bytes as usize <= bound)
                        .unwrap_or(LARGE_BUCKETS.len() - 1);
                    c.large[bucket].0 += 1;
                    c.large[bucket].1 += bytes;
                }
            }
            s += n;
        }
    }
    let mut at = DEFERRED.load(Ordering::Acquire);
    while at != NIL {
        // SAFETY: only a holder of `SPANS` pops, so a listed span stays listed during the walk.
        unsafe {
            c.deferred += u64::from((*sp(at)).slices) << SLICE_SHIFT;
            at = (*sp(at)).next;
        }
    }
    c
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

    /// Bytes of linear memory; a full 4 GiB is one past what `usize` holds, and must not read as 0.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn bytes() -> usize {
        core::arch::wasm32::memory_size(0).saturating_mul(SLICE)
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
/// A page with under 1/this of its blocks out is sparse, and passed over for a fuller one.
const SPARSE_BELOW: u32 = 4;
/// Pages a heap looks along a list for a fuller one, while its head is sparse.
const SPARSE_LOOK: u32 = 8;

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
    /// The last page of each list, where a page that gets a block back goes.
    tails: [u32; N_CLASSES],
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
                tails: [NIL; N_CLASSES],
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
        // The next holder may be a long way off when the thread that owns the heap goes idle, and
        // an inbox block holds its page until it is taken in.
        if !self.heap.inbox.load(Ordering::Relaxed).is_null() {
            self.collect();
        }
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
        p = self.page_to_fill(c, p);
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

    /// The page of class `c` to give a block from: `head` unless it is sparse, else the first fuller
    /// page within [`SPARSE_LOOK`], the sparse ones before it sent to the list's end. With none
    /// fuller, `head`: passing it for another sparse page would land a block in each of them in
    /// turn, and keep them all from emptying.
    fn page_to_fill(&mut self, c: usize, head: u32) -> u32 {
        // SAFETY: a listed page is this heap's, as are the pages listed with it.
        unsafe {
            let sparse = |p: u32| (*sp(p)).used * SPARSE_BELOW < (*sp(p)).cap;
            let mut fuller = head;
            let mut looked = 0;
            while sparse(fuller) {
                fuller = (*sp(fuller)).next;
                looked += 1;
                if fuller == NIL || looked == SPARSE_LOOK {
                    return head;
                }
            }
            while self.lists().pages[c] != fuller {
                let passed = self.lists().pages[c];
                self.unlink(passed);
                self.link(passed);
            }
            fuller
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
            let tail = self.lists().tails[c];
            (*page).prev = tail;
            (*page).next = NIL;
            if tail == NIL {
                self.lists().pages[c] = p;
            } else {
                (*sp(tail)).next = p;
            }
            self.lists().tails[c] = p;
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
            if next == NIL {
                self.lists().tails[(*page).class as usize] = prev;
            } else {
                (*sp(next)).prev = prev;
            }
            (*page).listed = false;
        }
    }

    /// Give the kept empty pages back to the pool. A kept page is its class's only one, so it
    /// heads its list.
    fn release_kept(&mut self) {
        if self.lists().kept == 0 {
            return;
        }
        for c in 0..N_CLASSES {
            let p = self.lists().pages[c];
            // SAFETY: a listed page is this heap's.
            if p != NIL && unsafe { (*sp(p)).kept } {
                unsafe { (*sp(p)).kept = false };
                self.lists().kept -= unsafe { (*sp(p)).slices } << SLICE_SHIFT;
                self.unlink(p);
                retire(p);
            }
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

/// Give back at once what the allocator holds as used and need not: every free heap's inbox and
/// kept empty pages, and every span on [`DEFERRED`], merged into the pool. A burst of frees under
/// load (a ring of tiles released) leaves spans deferred two at a time and pages kept per heap,
/// none of them in [`headroom`] or [`largest_run`] and none merged with their free neighbours; a
/// trim puts them back in one run where they lie side by side. A heap another thread holds is
/// passed over, so no thread waits but on [`SPANS`] for the drain.
pub(crate) fn trim() {
    for index in 0..HEAPS {
        if let Some(mut heap) = try_take(index) {
            heap.collect();
            heap.release_kept();
        }
    }
    drain(&SPANS.lock(), usize::MAX);
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
    BIGGEST.fetch_max(layout.size(), Ordering::Relaxed);
    let extra = (layout.align() / SLICE).saturating_sub(1) as u32;
    let turn = SPANS.lock();
    let s = alloc_span(&turn, n.saturating_add(extra));
    if s == NIL {
        drop(turn);
        out_of_memory(layout.size());
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

/// The largest block request since the last [`take_biggest`], bytes.
static BIGGEST: AtomicUsize = AtomicUsize::new(0);

/// The largest block requested since the last call, bytes: what [`largest_run`] must hold.
#[cfg_attr(
    test,
    expect(
        dead_code,
        reason = "the web build's memory gauge reads it; no test does"
    )
)]
pub(crate) fn take_biggest() -> usize {
    BIGGEST.swap(0, Ordering::Relaxed)
}

/// Name the block that found no run before the trap that follows: the out-of-memory abort reaches no
/// panic hook, and std's message goes to a stderr the page does not have.
#[cfg(target_arch = "wasm32")]
fn out_of_memory(size: usize) {
    crate::webdiag::out(&format!(
        "wow: no run for {} KiB: largest run {} KiB, headroom {} MiB",
        size >> 10,
        largest_run() >> 10,
        headroom() >> 20,
    ));
}

#[cfg(not(target_arch = "wasm32"))]
fn out_of_memory(_size: usize) {}

/// Resize a large block where it is: shrink it, or grow it into the free spans beside it, the one
/// after it first and the end of the one before it for the rest, its bytes moved down after
/// [`SPANS`] is let go. A buffer grown by doubling then needs no second run as long as itself,
/// beside the one it leaves. The block's address, or null when its neighbours are too short.
fn resize_large(block: *mut u8, old_size: usize, new_size: usize) -> *mut u8 {
    let Some(want) = slices_for(new_size) else {
        return ptr::null_mut();
    };
    let s = page_of(block);
    // SAFETY: a live block's `slices` changes only by its owner's resize.
    let have = unsafe { (*sp(s)).slices };
    if want == have {
        return block;
    }
    let turn = SPANS.lock();
    // SAFETY: the caller holds `SPANS`.
    unsafe {
        if want < have {
            (*sp(s)).slices = want;
            release(&turn, s + want, have - want);
            return block;
        }
        let need = want - have;
        let right = s + have;
        let rn = if (right as usize) < MAX_SLICES
            && head_of(right) == right
            && (*sp(right)).kind == FREE
        {
            (*sp(right)).slices
        } else {
            0
        };
        let left = if s > 0 { head_of(s - 1) } else { NIL };
        let ln = if left != NIL && (*sp(left)).kind == FREE {
            (*sp(left)).slices
        } else {
            0
        };
        if rn + ln < need {
            return ptr::null_mut();
        }
        let from_right = rn.min(need);
        let from_left = need - from_right;
        if from_right > 0 {
            bin_remove(&turn, right);
            if rn > from_right {
                file(&turn, right + from_right, rn - from_right);
            }
        }
        if from_left > 0 {
            bin_remove(&turn, left);
            if ln > from_left {
                file(&turn, left, ln - from_left);
            }
        }
        let head = s - from_left;
        (*sp(head)).kind = LARGE;
        (*sp(head)).slices = want;
        set_head(head, want, head);
        drop(turn);
        let moved = addr_of(head) as *mut u8;
        if moved != block {
            // The spans are this block's now; the copy overlaps, and runs without the lock.
            ptr::copy(block, moved, old_size);
        }
        moved
    }
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
            (None, None) if layout.align() <= SLICE => {
                let resized = resize_large(block, layout.size(), new_size);
                if !resized.is_null() {
                    return resized;
                }
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
#[path = "../../benilla-wasm-alloc-test/src/tests.rs"]
mod tests;
