//! fork: web — the threaded browser build's global allocator.
//!
//! std's allocator on `wasm32` with `+atomics` is one `dlmalloc` behind one spin lock
//! (`library/std/src/sys/alloc/wasm.rs`: `LOCKED.swap(1, Acquire)` in a loop, no queue, no backoff),
//! and every thread takes it for every allocation: the page, the async and compute pools and the
//! render worker. A spin lock grants no turn, so while the workers allocate hard — a crowd's
//! models, rigs and textures decoding at once — the page thread can lose the race for seconds,
//! and every system it runs stalls inside `malloc` (measured: 5.9 s and 6.6 s frames, ~75% of
//! their samples in the allocator, every schedule stage slowed alike).
//!
//! So small blocks (up to 16 KiB, alignment up to 16) are served from per-shard free lists, one
//! list per size class, taken with a `try_lock` that never waits: a busy shard falls through.
//! What reaches the shared allocator — a list's batched refill or flush, a block over 16 KiB, a
//! busy shard's request — goes through [`FairLock`], which lets no waiter starve: once one has
//! waited long, the others hold back until it is served, so the page thread waits a bounded time.
//!
//! The shard is picked from the stack address, not a thread-local: a new thread's stack and TLS
//! block are allocated with this allocator as wasm-bindgen starts it, possibly before that thread's
//! TLS exists, and a thread-local touched then would write through an unset TLS base. Thread stacks are disjoint, so
//! the stack address keeps a thread on its own shard; two threads that land on one shard only
//! share it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::UnsafeCell;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// The size classes, bytes; every class block is allocated at [`CLASS_ALIGN`].
const CLASSES: [usize; 36] = [
    16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
    1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336,
    16384,
];
const N_CLASSES: usize = CLASSES.len();
const CLASS_ALIGN: usize = 16;
/// The largest size a class serves.
const MAX_SMALL: usize = CLASSES[N_CLASSES - 1];
/// The class index by `size.div_ceil(16)`, for sizes `0..=MAX_SMALL`.
const CLASS_OF: [u8; MAX_SMALL / 16 + 1] = class_table();
/// What one list may hold, bytes: past it, a batch goes back to the shared allocator.
const LIST_BYTES: usize = 16 * 1024;
/// Shards, a power of two.
const SHARDS: usize = 32;
/// Blocks moved per refill or flush, at most: one hold of the shared lock stays short, which is
/// what bounds the wait of everyone queued behind it.
const BATCH: u32 = 32;

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

/// The class serving `layout`, or `None` for a block the shared allocator serves directly.
fn class_of(layout: Layout) -> Option<usize> {
    (layout.size() <= MAX_SMALL && layout.align() <= CLASS_ALIGN)
        .then(|| CLASS_OF[layout.size().div_ceil(16)] as usize)
}

/// The layout a class block has in the shared allocator.
fn class_layout(c: usize) -> Layout {
    // SAFETY: every class is a non-zero multiple of the power-of-two alignment.
    unsafe { Layout::from_size_align_unchecked(CLASSES[c], CLASS_ALIGN) }
}

/// Blocks a refill or flush moves for class `c`.
const fn batch(c: usize) -> u32 {
    let half = cap(c) / 2;
    if half < BATCH {
        half
    } else {
        BATCH
    }
}

/// How many blocks a class list holds before it flushes a batch.
const fn cap(c: usize) -> u32 {
    let n = LIST_BYTES / CLASSES[c];
    if n < 2 {
        2
    } else {
        n as u32
    }
}

/// A free block, linked through its own first word.
struct Node {
    next: *mut Node,
}

#[derive(Clone, Copy)]
struct FreeList {
    head: *mut Node,
    len: u32,
}

impl FreeList {
    const EMPTY: Self = Self {
        head: ptr::null_mut(),
        len: 0,
    };

    /// SAFETY: `block` is a free class block of this list's class.
    unsafe fn push(&mut self, block: *mut u8) {
        let node = block.cast::<Node>();
        unsafe { (*node).next = self.head };
        self.head = node;
        self.len += 1;
    }

    fn pop(&mut self) -> *mut u8 {
        let node = self.head;
        if !node.is_null() {
            // SAFETY: a listed node is a free block this list owns.
            self.head = unsafe { (*node).next };
            self.len -= 1;
        }
        node.cast()
    }
}

/// One shard: a list per class, and the flag that hands them to one thread at a time.
struct Shard {
    busy: AtomicBool,
    lists: UnsafeCell<[FreeList; N_CLASSES]>,
}

// SAFETY: `lists` is only touched by the thread that set `busy`.
unsafe impl Sync for Shard {}

impl Shard {
    const fn new() -> Self {
        Self {
            busy: AtomicBool::new(false),
            lists: UnsafeCell::new([FreeList::EMPTY; N_CLASSES]),
        }
    }

    /// The shard's lists, if no other thread holds them; never waits. The guard is built only on
    /// success: a guard dropped after a failed take would release another thread's shard.
    fn try_take(&self) -> Option<ShardGuard<'_>> {
        self.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
            .then(|| ShardGuard(self))
    }
}

struct ShardGuard<'a>(&'a Shard);

impl ShardGuard<'_> {
    fn list(&mut self, c: usize) -> &mut FreeList {
        // SAFETY: holding the guard is holding `busy`.
        unsafe { &mut (*self.0.lists.get())[c] }
    }
}

impl Drop for ShardGuard<'_> {
    fn drop(&mut self) {
        self.0.busy.store(false, Ordering::Release);
    }
}

static SHARD_TABLE: [Shard; SHARDS] = [const { Shard::new() }; SHARDS];

/// This thread's shard, by the 64 KiB window its stack is in.
#[inline]
fn shard() -> &'static Shard {
    let probe = 0u8;
    let sp = std::hint::black_box(&probe) as *const u8 as usize;
    // Fibonacci hashing in 32 bits, the top bits picking the shard.
    let mixed = ((sp >> 16) as u32).wrapping_mul(0x9E37_79B1);
    &SHARD_TABLE[(mixed >> (32 - SHARDS.trailing_zeros())) as usize]
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

/// The lock every call into the shared allocator takes.
static SHARED: FairLock = FairLock::new();

/// Small blocks from per-shard class lists, everything else from `System` in turn.
pub(crate) struct ShardedAlloc;

impl ShardedAlloc {
    /// A class block: from this shard's list (refilled in one batch when empty), or straight from
    /// the shared allocator when the shard is busy.
    unsafe fn alloc_small(&self, c: usize) -> *mut u8 {
        if let Some(mut shard) = shard().try_take() {
            let list = shard.list(c);
            if list.head.is_null() {
                let _turn = SHARED.lock();
                for _ in 0..batch(c) {
                    // SAFETY: a class layout is non-zero-sized.
                    let block = unsafe { System.alloc(class_layout(c)) };
                    if block.is_null() {
                        break;
                    }
                    // SAFETY: a fresh class block of class `c`.
                    unsafe { list.push(block) };
                }
            }
            let block = list.pop();
            if !block.is_null() {
                return block;
            }
        }
        let _turn = SHARED.lock();
        // SAFETY: a class layout is non-zero-sized.
        unsafe { System.alloc(class_layout(c)) }
    }

    /// Return a class block: onto this shard's list (flushing a batch when full), or straight to the
    /// shared allocator when the shard is busy.
    unsafe fn dealloc_small(&self, block: *mut u8, c: usize) {
        if let Some(mut shard) = shard().try_take() {
            let list = shard.list(c);
            if list.len >= cap(c) {
                let _turn = SHARED.lock();
                for _ in 0..batch(c) {
                    let spare = list.pop();
                    // SAFETY: every listed block came from `System` at the class layout.
                    unsafe { System.dealloc(spare, class_layout(c)) };
                }
            }
            // SAFETY: the caller frees a block of class `c`.
            unsafe { list.push(block) };
            return;
        }
        let _turn = SHARED.lock();
        // SAFETY: every class block came from `System` at the class layout.
        unsafe { System.dealloc(block, class_layout(c)) };
    }
}

// SAFETY: class blocks are at least the requested size at 16-byte alignment, which covers every
// layout `class_of` admits; everything else is `System`'s, called under `SHARED`.
unsafe impl GlobalAlloc for ShardedAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match class_of(layout) {
            Some(c) => unsafe { self.alloc_small(c) },
            None => {
                let _turn = SHARED.lock();
                unsafe { System.alloc(layout) }
            }
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        match class_of(layout) {
            Some(c) => {
                let block = unsafe { self.alloc_small(c) };
                if !block.is_null() {
                    unsafe { ptr::write_bytes(block, 0, layout.size()) };
                }
                block
            }
            None => {
                let _turn = SHARED.lock();
                unsafe { System.alloc_zeroed(layout) }
            }
        }
    }

    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        match class_of(layout) {
            Some(c) => unsafe { self.dealloc_small(block, c) },
            None => {
                let _turn = SHARED.lock();
                unsafe { System.dealloc(block, layout) };
            }
        }
    }

    unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees `new_size` is a valid size at `layout.align()`.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        match (class_of(layout), class_of(new_layout)) {
            (None, None) => {
                let _turn = SHARED.lock();
                unsafe { System.realloc(block, layout, new_size) }
            }
            (Some(a), Some(b)) if a == b => block,
            _ => {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every size up to the largest class lands in the smallest class that holds it.
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
        assert_eq!(class_of(Layout::from_size_align(64, 32).unwrap()), None);
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

    /// Eight threads allocate, fill, check and free mixed sizes and alignments, half of them
    /// freeing blocks another thread allocated: no block is handed out twice or misaligned.
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
                        let size = 1 + (x as usize % 4096);
                        let align = 1 << ((x as usize >> 20) % 6);
                        let layout = Layout::from_size_align(size, align).unwrap();
                        let block = unsafe { ShardedAlloc.alloc(layout) };
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
                                ShardedAlloc
                                    .dealloc(b as *mut u8, Layout::from_size_align(s, a).unwrap())
                            };
                        }
                        if t % 2 == 1 {
                            if let Ok((b, s, a, sd)) = rx.lock().unwrap().try_recv() {
                                check(b as *const u8, s, sd);
                                unsafe {
                                    ShardedAlloc.dealloc(
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
                            ShardedAlloc
                                .dealloc(b as *mut u8, Layout::from_size_align(s, a).unwrap())
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
            unsafe { ShardedAlloc.dealloc(b as *mut u8, Layout::from_size_align(s, a).unwrap()) };
        }
    }

    /// Growing and shrinking across the class boundary and into the shared allocator keeps the
    /// bytes, and a zeroed block reads zero even when it is a reused one.
    #[test]
    fn realloc_keeps_the_bytes_and_zeroed_reads_zero() {
        let layout = Layout::from_size_align(24, 8).unwrap();
        let mut block = unsafe { ShardedAlloc.alloc(layout) };
        fill(block, 24, 7);
        let mut size = 24;
        for next in [30, 100, 2048, 5000, 70_000, 1500, 9] {
            block = unsafe {
                ShardedAlloc.realloc(block, Layout::from_size_align(size, 8).unwrap(), next)
            };
            assert!(!block.is_null());
            check(block, size.min(next).min(24), 7);
            size = next;
        }
        unsafe { ShardedAlloc.dealloc(block, Layout::from_size_align(size, 8).unwrap()) };

        let dirty = unsafe { ShardedAlloc.alloc(Layout::from_size_align(64, 16).unwrap()) };
        fill(dirty, 64, 0xAA);
        unsafe { ShardedAlloc.dealloc(dirty, Layout::from_size_align(64, 16).unwrap()) };
        let zeroed = unsafe { ShardedAlloc.alloc_zeroed(Layout::from_size_align(64, 16).unwrap()) };
        assert!((0..64).all(|i| unsafe { *zeroed.add(i) } == 0));
        unsafe { ShardedAlloc.dealloc(zeroed, Layout::from_size_align(64, 16).unwrap()) };
    }
}
