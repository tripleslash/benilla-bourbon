//! Writer/reader round-trip: everything the packer emits, the client can read back byte-identically.
//!
//! The container's whole job is to survive a trip through a dumb object store, so these tests
//! exercise the two shapes that trip actually takes — a whole-object fetch and a byte-range fetch —
//! and assert they agree with each other and with the bytes that went in.

#![cfg(feature = "write")]

use benilla_pack::{
    decode_pack_body, frame_bytes, frame_from_body, pack_body, pack_path, Codec, Index, PackClass,
    PackPlan, PackSetBuilder,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

/// A unique scratch dir, hand-rolled for the same reason `benilla-mpq`'s is: these leaf crates stay
/// dependency-light, so no `tempfile`.
fn scratch(tag: &str) -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let d = std::env::temp_dir().join(format!(
        "benilla_pack_{}_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed),
        tag
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

/// Highly compressible, so a Solid pack can actually beat per-frame framing — the branch under test.
fn textish(seed: u64, len: usize) -> Vec<u8> {
    let words = [
        "<Frame", "name=", "parent", "UIParent", "inherits", "Button",
    ];
    let mut out = Vec::with_capacity(len);
    let mut s = seed;
    while out.len() < len {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        out.extend_from_slice(words[(s >> 33) as usize % words.len()].as_bytes());
        out.push(b' ');
    }
    out.truncate(len);
    out
}

/// Incompressible, standing in for the MP3/DXT majority of the real corpus — the case where the
/// packer must decide to STORE rather than pay for nothing.
fn noise(seed: u64, len: usize) -> Vec<u8> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

fn read_pack(root: &std::path::Path, id: u32) -> Vec<u8> {
    std::fs::read(root.join(pack_path(id))).expect("pack file")
}

/// The whole-object path: fetch a `.bpk`, decode its body, slice each frame. This is what a
/// Range-less host (a static host today) forces, so it is the path that must always work.
#[test]
fn every_blob_reads_back_through_a_whole_pack_fetch() {
    let root = scratch("whole");
    let mut b = PackSetBuilder::new(&root, 10);

    let solid: Vec<(u64, Vec<u8>)> = (0..40).map(|i| (100 + i, textish(i, 3000))).collect();
    let mixed: Vec<(u64, Vec<u8>)> = (0..8)
        .map(|i| (200 + i, noise(i + 1, 5000 + i as usize * 111)))
        .collect();
    let big = vec![(900u64, noise(77, 300_000))];

    b.add(&PackPlan {
        blobs: solid.clone(),
        prefer: PackClass::Solid,
    })
    .unwrap();
    b.add(&PackPlan {
        blobs: mixed.clone(),
        prefer: PackClass::Rangeable,
    })
    .unwrap();
    b.add(&PackPlan {
        blobs: big.clone(),
        prefer: PackClass::Rangeable,
    })
    .unwrap();
    let index_path = b.finish().unwrap();

    let idx = Index::parse(std::fs::read(&index_path).unwrap()).unwrap();
    assert_eq!(idx.len(), solid.len() + mixed.len() + big.len());
    assert_eq!(idx.pack_count(), 3);

    for (key, want) in solid.iter().chain(&mixed).chain(&big) {
        let at = idx
            .locate(*key)
            .unwrap_or_else(|| panic!("key {key:#x} missing"));
        let meta = idx.pack(at.pack_id).unwrap();
        let wire = read_pack(&root, at.pack_id);
        assert_eq!(
            wire.len() as u32,
            meta.size_wire,
            "size_wire disagrees with the file"
        );
        let body = pack_body(&wire, at.pack_id).unwrap();
        let body = decode_pack_body(body, &meta).unwrap();
        let got = frame_from_body(&body, &at).unwrap();
        assert_eq!(&got, want, "key {key:#x} did not round-trip");
    }
}

/// The ranged path: fetch *only* `Located::wire_range()` out of the object and decode that alone,
/// with the rest of the pack never downloaded. Simulated by slicing the file, which is exactly what
/// a 206 delivers.
#[test]
fn a_ranged_fetch_of_one_frame_decodes_without_the_rest_of_the_pack() {
    let root = scratch("ranged");
    let mut b = PackSetBuilder::new(&root, 10);
    let blobs: Vec<(u64, Vec<u8>)> = (0..12)
        .map(|i| (500 + i, textish(i * 31, 2000 + i as usize * 97)))
        .collect();
    b.add(&PackPlan {
        blobs: blobs.clone(),
        prefer: PackClass::Rangeable,
    })
    .unwrap();
    let idx = Index::parse(std::fs::read(b.finish().unwrap()).unwrap()).unwrap();

    let wire = read_pack(&root, 0);
    for (key, want) in &blobs {
        let at = idx.locate(*key).unwrap();
        let meta = idx.pack(at.pack_id).unwrap();
        assert!(
            meta.is_range_extractable(),
            "a Rangeable pack must be sliceable"
        );
        let (start, end) = at.wire_range();
        let slice = &wire[start as usize..=end as usize];
        assert_eq!(slice.len() as u32, at.frame_len);
        assert_eq!(&frame_bytes(slice).unwrap(), want);
    }
}

/// A Solid pack must declare itself unsliceable. If it claimed otherwise the client would issue a
/// range request against compressed bytes and decode garbage — the one way this format could
/// corrupt data rather than error.
#[test]
fn a_solid_pack_refuses_to_claim_range_extractability() {
    let root = scratch("solid");
    let mut b = PackSetBuilder::new(&root, 10);
    let blobs: Vec<(u64, Vec<u8>)> = (0..64).map(|i| (i, textish(i, 4000))).collect();
    b.add(&PackPlan {
        blobs,
        prefer: PackClass::Solid,
    })
    .unwrap();
    let idx = Index::parse(std::fs::read(b.finish().unwrap()).unwrap()).unwrap();

    let meta = idx.pack(0).unwrap();
    assert_eq!(
        meta.class,
        PackClass::Solid,
        "highly compressible bundle should go solid"
    );
    assert_eq!(meta.codec_pack, Codec::Zstd);
    assert!(!meta.is_range_extractable());
    // And solid actually earned it.
    assert!(
        meta.size_wire < meta.size_raw,
        "solid pack {} wire vs {} raw",
        meta.size_wire,
        meta.size_raw
    );
}

/// Incompressible content must be STOREd rather than nominally "compressed" — otherwise every
/// sound file costs a pointless zstd decode on the client for no saved bytes.
#[test]
fn incompressible_frames_are_stored_not_pointlessly_compressed() {
    let root = scratch("store");
    let mut b = PackSetBuilder::new(&root, 10);
    let blobs = vec![(1u64, noise(5, 64_000))];
    b.add(&PackPlan {
        blobs: blobs.clone(),
        prefer: PackClass::Rangeable,
    })
    .unwrap();
    let idx = Index::parse(std::fs::read(b.finish().unwrap()).unwrap()).unwrap();

    let at = idx.locate(1).unwrap();
    // Frame header plus the payload verbatim: no expansion from a futile compress.
    assert_eq!(
        at.frame_len as usize,
        benilla_pack::FRAME_HEADER_LEN + 64_000
    );
    let meta = idx.pack(0).unwrap();
    let wire = read_pack(&root, 0);
    let body = pack_body(&wire, 0).unwrap();
    let body = decode_pack_body(body, &meta).unwrap();
    assert_eq!(frame_from_body(&body, &at).unwrap(), blobs[0].1);
}

/// The object cap is a property the output *has*: an over-cap pack is refused at write time, for
/// the caller to split and retry. It is never emitted and audited afterwards, because an audit
/// failure after a 3 GiB build is a wasted build.
#[test]
fn a_pack_over_the_object_cap_is_refused_at_write_time() {
    let root = scratch("cap");
    let mut b = PackSetBuilder::new(&root, 1).with_max_object_bytes(64 * 1024);
    let blobs = vec![(1u64, noise(9, 200_000))]; // incompressible, so it cannot squeeze under
    assert!(b
        .add(&PackPlan {
            blobs,
            prefer: PackClass::Rangeable
        })
        .is_err());
    // The rejected pack left no trace: no id consumed, no file, and the index still closes cleanly.
    assert_eq!(b.pack_count(), 0);
    assert!(!root.join(pack_path(0)).exists());
    let idx = Index::parse(std::fs::read(b.finish().unwrap()).unwrap()).unwrap();
    assert!(idx.is_empty());
}

/// Two names the source archive cannot tell apart must stop the build, not silently lose one.
#[test]
fn a_duplicate_key_fails_the_build_rather_than_dropping_a_blob() {
    let root = scratch("dup");
    let mut b = PackSetBuilder::new(&root, 1);
    b.add(&PackPlan {
        blobs: vec![(7, b"a".to_vec())],
        prefer: PackClass::Rangeable,
    })
    .unwrap();
    b.add(&PackPlan {
        blobs: vec![(7, b"b".to_vec())],
        prefer: PackClass::Rangeable,
    })
    .unwrap();
    match b.finish() {
        Err(e) => assert!(format!("{e}").contains("duplicate key"), "{e}"),
        Ok(_) => panic!("a duplicate key must fail the build"),
    }
}

/// Same corpus in, same bytes out — which is what makes a static host's content-hash upload actually
/// incremental, and what lets a re-pack after a patch archive touch only the packs that changed.
#[test]
fn two_builds_of_the_same_corpus_are_byte_identical() {
    let blobs: Vec<(u64, Vec<u8>)> = (0..24).map(|i| (i, textish(i, 1500))).collect();
    let build = |tag: &str| {
        let root = scratch(tag);
        let mut b = PackSetBuilder::new(&root, 10);
        b.add(&PackPlan {
            blobs: blobs.clone(),
            prefer: PackClass::Solid,
        })
        .unwrap();
        let idx = std::fs::read(b.finish().unwrap()).unwrap();
        (idx, std::fs::read(root.join(pack_path(0))).unwrap())
    };
    let (idx_a, pack_a) = build("det_a");
    let (idx_b, pack_b) = build("det_b");
    assert_eq!(idx_a, idx_b, "index.bpi must be reproducible");
    assert_eq!(pack_a, pack_b, "pack bytes must be reproducible");
    // And the build id, which namespaces the client's cache, moves with the layout.
    let id_a = Index::parse(idx_a).unwrap().build_id();
    assert_ne!(id_a, 0);
}

/// Absent keys must answer "no" from memory. This is the property that retires the 2,145 blocking
/// `HEAD` requests a world entry used to spend on `Chain::contains`.
#[test]
fn a_missing_key_is_answered_without_touching_a_pack() {
    let root = scratch("missing");
    let mut b = PackSetBuilder::new(&root, 1);
    b.add(&PackPlan {
        blobs: vec![(10, b"x".to_vec()), (20, b"y".to_vec())],
        prefer: PackClass::Rangeable,
    })
    .unwrap();
    let idx = Index::parse(std::fs::read(b.finish().unwrap()).unwrap()).unwrap();
    assert!(idx.locate(10).is_some());
    assert!(idx.locate(20).is_some());
    for absent in [0u64, 9, 11, 19, 21, u64::MAX] {
        assert!(idx.locate(absent).is_none(), "{absent} should be absent");
    }
}

/// The index is searched in place, so the key array must actually be sorted — an unsorted one would
/// make the binary search miss entries that are present, silently.
#[test]
fn keys_are_emitted_in_ascending_order() {
    let root = scratch("sorted");
    let mut b = PackSetBuilder::new(&root, 1);
    // Added deliberately out of order, across two packs.
    b.add(&PackPlan {
        blobs: vec![(u64::MAX, b"z".to_vec()), (5, b"a".to_vec())],
        prefer: PackClass::Rangeable,
    })
    .unwrap();
    b.add(&PackPlan {
        blobs: vec![(1000, b"m".to_vec()), (0, b"s".to_vec())],
        prefer: PackClass::Rangeable,
    })
    .unwrap();
    let idx = Index::parse(std::fs::read(b.finish().unwrap()).unwrap()).unwrap();
    let keys: Vec<u64> = idx.iter().map(|(k, _)| k).collect();
    assert_eq!(keys, vec![0, 5, 1000, u64::MAX]);
    for k in keys {
        assert!(idx.locate(k).is_some());
    }
}

/// Laying packs over a built container: the old packs keep their ids and objects, a key the new
/// pack also holds reads the new bytes, every other key still reads the old ones, and the build id
/// moves so a client's cache of the old index is dropped.
#[test]
fn an_appended_pack_supersedes_only_the_keys_it_holds() {
    let root = scratch("append");
    let mut b = PackSetBuilder::new(&root, 10);
    let old = vec![(1u64, noise(1, 4000)), (2u64, noise(2, 4000))];
    b.add(&PackPlan {
        blobs: old.clone(),
        prefer: PackClass::Rangeable,
    })
    .unwrap();
    let first = Index::parse(std::fs::read(b.finish().unwrap()).unwrap()).unwrap();
    let old_wire = read_pack(&root, 0);

    let mut b = PackSetBuilder::from_index(&root, 10, &first).unwrap();
    let new = vec![(2u64, noise(20, 3000)), (3u64, noise(3, 3000))];
    assert_eq!(
        b.add(&PackPlan {
            blobs: new.clone(),
            prefer: PackClass::Rangeable,
        })
        .unwrap(),
        1,
        "the new pack takes the next id"
    );
    let idx = Index::parse(std::fs::read(b.finish().unwrap()).unwrap()).unwrap();
    assert_eq!((idx.pack_count(), idx.len()), (2, 3));
    assert_eq!(read_pack(&root, 0), old_wire, "the old object is untouched");
    assert_ne!(idx.build_id(), first.build_id());
    for (key, want, pack) in [
        (1u64, &old[0].1, 0u32),
        (2, &new[0].1, 1),
        (3, &new[1].1, 1),
    ] {
        let at = idx.locate(key).unwrap();
        assert_eq!(at.pack_id, pack, "key {key}");
        let meta = idx.pack(at.pack_id).unwrap();
        let wire = read_pack(&root, at.pack_id);
        let body = decode_pack_body(pack_body(&wire, at.pack_id).unwrap(), &meta).unwrap();
        assert_eq!(&frame_from_body(&body, &at).unwrap(), want, "key {key}");
    }
}
