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

/// Each bin's shortest span is filed there, the one shorter below it, and a fit from it searches
/// that bin: a span the highest bin holds serves a request of that length.
#[test]
fn a_bins_lower_bound_is_filed_there_and_fits_there() {
    for b in 0..BINS {
        let low = bin_low(b);
        if low as usize >= MAX_SLICES {
            break;
        }
        assert_eq!(bin_of(low), b, "{low}");
        assert_eq!(bin_fit(low), b, "{low}");
        if b > 0 {
            assert_eq!(bin_of(low - 1), b - 1, "{low}");
        }
    }
}

/// The pool gives a span as long as its largest run in one take, without a grow, and the run
/// is never more than the headroom. Under the pool's lock throughout.
#[test]
fn the_pools_largest_run_is_one_take() {
    let turn = SPANS.lock();
    if pooled_run() == 0 {
        assert!(grow(&turn, 64));
    }
    let n = pooled_run();
    assert!(n > 0);
    assert!(largest_run() <= headroom());
    let s = take(&turn, n);
    assert_ne!(s, NIL, "a run of {n} slices is filed");
    release(&turn, s, n);
}

/// The census finds a live large block in its size bucket and a live class block's page. Other
/// tests allocate meanwhile, so it reads a floor, and holds both blocks across the read.
#[test]
fn the_census_counts_live_blocks() {
    let large = Layout::from_size_align(300 * 1024, 8).unwrap();
    let small = Layout::from_size_align(1024, 8).unwrap();
    let (a, b) = unsafe { (PageAlloc.alloc(large), PageAlloc.alloc(small)) };
    let c = census();
    assert!(c.large[1].0 >= 1, "300 KiB is under the 512 KiB bound");
    let (span, live) = c.pages[class_of(small).unwrap()];
    assert!(span >= SLICE as u64 && live >= 1024);
    unsafe {
        PageAlloc.dealloc(a, large);
        PageAlloc.dealloc(b, small);
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
                            PageAlloc.dealloc(b as *mut u8, Layout::from_size_align(s, a).unwrap())
                        };
                    }
                    if t % 2 == 1 {
                        if let Ok((b, s, a, sd)) = rx.lock().unwrap().try_recv() {
                            check(b as *const u8, s, sd);
                            unsafe {
                                PageAlloc
                                    .dealloc(b as *mut u8, Layout::from_size_align(s, a).unwrap())
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

/// A held heap let go takes in its inbox, so no idle owner pins a page with freed blocks.
#[test]
fn letting_a_heap_go_takes_in_its_inbox() {
    let mut heap = hold_a_heap();
    let table = heap.heap;
    let layout = Layout::from_size_align(64, 8).unwrap();
    let block = heap.alloc(class_of(layout).unwrap());
    let freeing = block as usize;
    std::thread::spawn(move || unsafe { PageAlloc.dealloc(freeing as *mut u8, layout) })
        .join()
        .unwrap();
    assert!(inbox_has(table, block));
    drop(heap);
    assert!(!inbox_has(table, block));
}

/// A page drained to sparse is passed over while a fuller one has room, so it can empty.
#[test]
fn a_sparse_page_gets_no_new_blocks_while_a_fuller_one_has_room() {
    let mut heap = hold_a_heap();
    let c = class_of(Layout::from_size_align(12000, 8).unwrap()).unwrap();
    let mut blocks: Vec<*mut u8> = Vec::new();
    let (a, b) = loop {
        blocks.push(heap.alloc(c));
        let mut pages: Vec<u32> = blocks.iter().map(|b| page_of(*b)).collect();
        pages.dedup();
        let full = |p: &u32| unsafe { (*sp(*p)).used == (*sp(*p)).cap };
        if let [.., a, b] = pages.as_slice() {
            if full(a) && full(b) {
                break (*a, *b);
            }
        }
    };
    // `a` keeps one block, under a quarter of its cap; `b` gives one back and stays fuller.
    let mut kept_in_a = false;
    let mut freed_in_b = false;
    blocks.retain(|&block| {
        let p = page_of(block);
        let free = if p == a {
            std::mem::replace(&mut kept_in_a, true)
        } else if p == b {
            !std::mem::replace(&mut freed_in_b, true)
        } else {
            false
        };
        if free {
            unsafe { heap.free_local(p, block) };
        }
        !free
    });
    let next = heap.alloc(c);
    assert_ne!(page_of(next), a, "the sparse page was passed over");
    unsafe { heap.free_local(page_of(next), next) };
    for block in blocks {
        unsafe { heap.free_local(page_of(block), block) };
    }
}

/// No page spans more than two slices, and every page holds at least two blocks, so even the
/// largest class takes the pool's lock once per two blocks.
#[test]
fn a_page_is_two_slices_at_most_and_two_blocks_at_least() {
    for (c, &size) in CLASSES.iter().enumerate() {
        assert!(page_slices(c) <= 2, "{size}");
        assert!(page_slices(c) as usize * SLICE / size >= 2, "{size}");
    }
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

/// Full pages of class `c`, oldest first, and every block allocated on the way to them.
fn full_pages(heap: &mut HeapGuard, c: usize, want: usize) -> (Vec<u32>, Vec<*mut u8>) {
    let mut blocks = Vec::new();
    loop {
        let block = heap.alloc(c);
        blocks.push(block);
        let mut pages: Vec<u32> = blocks.iter().map(|b| page_of(*b)).collect();
        pages.dedup();
        pages.retain(|p| unsafe { (*sp(*p)).used == (*sp(*p)).cap });
        if pages.len() == want {
            return (pages, blocks);
        }
    }
}

/// Free `blocks` in order, keeping the first `keep(page)` of each page; returns those kept.
fn thin(heap: &mut HeapGuard, blocks: Vec<*mut u8>, keep: impl Fn(u32) -> usize) -> Vec<*mut u8> {
    let mut kept: Vec<*mut u8> = Vec::new();
    for block in blocks {
        let p = page_of(block);
        if kept.iter().filter(|b| page_of(**b) == p).count() < keep(p) {
            kept.push(block);
        } else {
            unsafe { heap.free_local(p, block) };
        }
    }
    kept
}

/// With every page sparse, new blocks fill one page instead of landing a block in each, which
/// would keep all of them from emptying.
#[test]
fn sparse_pages_with_none_fuller_fill_one_at_a_time() {
    let mut heap = hold_a_heap();
    let c = class_of(Layout::from_size_align(3000, 8).unwrap()).unwrap();
    let (pages, blocks) = full_pages(&mut heap, c, 5);
    let mut kept = thin(&mut heap, blocks, |p| usize::from(pages.contains(&p)));
    let next: Vec<*mut u8> = (0..4).map(|_| heap.alloc(c)).collect();
    let mut landed: Vec<u32> = next.iter().map(|b| page_of(*b)).collect();
    landed.sort_unstable();
    landed.dedup();
    kept.extend(next);
    for block in kept {
        unsafe { heap.free_local(page_of(block), block) };
    }
    // Two: a page another test left fuller may be first, and fill.
    assert!(landed.len() <= 2, "four blocks landed in {landed:?}");
}

/// A fuller page behind several sparse ones gets the block, and none of the sparse ones does.
#[test]
fn a_fuller_page_behind_several_sparse_ones_gets_the_block() {
    let mut heap = hold_a_heap();
    let c = class_of(Layout::from_size_align(3000, 8).unwrap()).unwrap();
    let (pages, blocks) = full_pages(&mut heap, c, 3);
    let cap = unsafe { (*sp(pages[2])).cap } as usize;
    let keep = |p: u32| match pages.iter().position(|q| *q == p) {
        Some(2) => cap - 1,
        Some(_) => 1,
        None => 0,
    };
    let kept = thin(&mut heap, blocks, keep);
    let next = heap.alloc(c);
    let landed = page_of(next);
    unsafe { heap.free_local(landed, next) };
    for block in kept {
        unsafe { heap.free_local(page_of(block), block) };
    }
    assert!(
        landed != pages[0] && landed != pages[1],
        "a sparse page got it"
    );
}

/// A span in the request's own bin that holds it is found once no longer span is left, though a
/// fit searches past that bin. Under the pool's lock throughout, with the longer spans set aside.
#[test]
fn a_span_in_the_requests_own_bin_is_found() {
    const WANT: u32 = 800;
    const HAVE: u32 = 850;
    assert_eq!(bin_of(WANT), bin_of(HAVE));
    assert_ne!(bin_of(WANT), bin_fit(WANT));
    let turn = SPANS.lock();
    let s = alloc_span(&turn, HAVE + 2);
    assert_ne!(s, NIL);
    let right = s + HAVE + 1;
    for guard in [s, right] {
        unsafe {
            (*sp(guard)).kind = LARGE;
            (*sp(guard)).slices = 1;
        }
        set_head(guard, 1, guard);
    }
    let mid = s + 1;
    release(&turn, mid, HAVE);
    let mut aside = Vec::new();
    for b in bin_of(WANT)..BINS {
        let mut at = unsafe { (*pool(&turn)).bins[b] };
        while at != NIL {
            let next = unsafe { (*sp(at)).next };
            if at != mid {
                bin_remove(&turn, at);
                aside.push(at);
            }
            at = next;
        }
    }
    let got = take(&turn, WANT);
    for at in aside {
        bin_insert(&turn, at);
    }
    if got != NIL {
        release(&turn, got, WANT);
    }
    release(&turn, s, 1);
    release(&turn, right, 1);
    assert_eq!(got, mid);
}

/// Near the maximum a grow takes what is left, and that merges with the free span below it, so
/// a request longer than either fits. Under the pool's lock throughout.
#[test]
fn a_grow_near_the_maximum_takes_what_is_left() {
    const LEFT: u32 = 16;
    let turn = SPANS.lock();
    assert!(grow(&turn, 1));
    let top = (heap::bytes() >> SLICE_SHIFT) as u32;
    let below = head_of(top - 1);
    assert_eq!(unsafe { (*sp(below)).kind }, FREE);
    let n = unsafe { (*sp(below)).slices } + LEFT / 2;
    let cap = heap::max_bytes();
    heap::set_model_bytes(heap::bytes() + LEFT as usize * SLICE);
    let _ = take_grow_stats();
    let grown = grow(&turn, n);
    let stats = take_grow_stats();
    let s = take(&turn, n);
    if s != NIL {
        release(&turn, s, n);
    }
    let more = grow(&turn, 1);
    let _ = take_grow_stats();
    heap::set_model_bytes(cap);
    assert!(grown, "what is left is worth growing");
    assert_eq!(stats, (1, 0, LEFT as usize * SLICE), "all of it, at once");
    assert_ne!(s, NIL, "it merged with the span below");
    assert!(!more, "nothing is left to grow");
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
        block =
            unsafe { PageAlloc.realloc(block, Layout::from_size_align(size, 8).unwrap(), next) };
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
