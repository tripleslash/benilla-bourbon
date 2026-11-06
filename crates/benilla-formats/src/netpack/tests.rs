use super::*;
use benilla_pack::{PackClass, PackPlan, PackSetBuilder};

/// A fresh empty folder under the system temp folder, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("benilla-netpack-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn key(name: &str) -> u64 {
    benilla_mpq::name_key(name)
}

/// `len` bytes no codec shrinks, so a change of `len` is a change of frame length.
fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut x = seed.wrapping_mul(2_654_435_761) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as u8
        })
        .collect()
}

/// Build a two-pack container at `origin` — a solid pack of small files and a rangeable one whose
/// second file is `second` — and its version pointer.
fn publish(origin: &Path, second: &[u8]) {
    let _ = std::fs::remove_dir_all(origin);
    std::fs::create_dir_all(origin).unwrap();
    let mut b = PackSetBuilder::new(origin, 3);
    b.add(&PackPlan {
        blobs: vec![
            (key("Interface\\a.lua"), b"print('a')".repeat(50)),
            (key("Interface\\b.lua"), b"print('b')".repeat(50)),
        ],
        prefer: PackClass::Solid,
    })
    .unwrap();
    b.add(&PackPlan {
        blobs: vec![
            (key("Sound\\one.wav"), vec![7u8; 4096]),
            (key("Sound\\two.wav"), second.to_vec()),
        ],
        prefer: PackClass::Rangeable,
    })
    .unwrap();
    b.finish().unwrap();
    let index = Index::parse(std::fs::read(origin.join(INDEX)).unwrap()).unwrap();
    std::fs::write(
        origin.join(POINTER),
        format!(
            "{{\"version\":1,\"build_id\":\"{:016x}\",\"boot\":{{\"packs\":[[1,10]]}}}}",
            index.build_id()
        ),
    )
    .unwrap();
    std::fs::write(origin.join(LISTFILE), "Interface\\a.lua\nSound\\two.wav\n").unwrap();
}

fn open(cache: &Path, origin: &Path) -> Arc<NetPackStore> {
    Arc::new(NetPackStore::open(cache, Some(Box::new(Folder(origin.to_path_buf())))).unwrap())
}

fn cached_packs(cache: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for fan in std::fs::read_dir(cache.join("p"))
        .into_iter()
        .flatten()
        .flatten()
    {
        for f in std::fs::read_dir(fan.path()).unwrap().flatten() {
            out.push(f.file_name().to_string_lossy().into_owned());
        }
    }
    out.sort();
    out
}

#[test]
fn a_read_fetches_its_pack_once_and_names_it_by_crc() {
    let (origin, cache) = (Scratch::new("origin"), Scratch::new("cache"));
    publish(&origin.0, &[9u8; 2048]);
    let store = open(&cache.0, &origin.0);
    assert!(
        cached_packs(&cache.0).is_empty(),
        "nothing fetched before a read"
    );

    assert_eq!(
        store.read("interface/A.LUA").unwrap(),
        b"print('a')".repeat(50)
    );
    assert_eq!(
        store.read("Interface\\b.lua").unwrap(),
        b"print('b')".repeat(50)
    );
    assert_eq!(store.read("Sound\\two.wav").unwrap(), vec![9u8; 2048]);
    let crc0 = store.index().pack(0).unwrap().crc32;
    let crc1 = store.index().pack(1).unwrap().crc32;
    assert_eq!(
        cached_packs(&cache.0),
        vec![
            format!("0000-{crc0:08x}.bpk"),
            format!("0001-{crc1:08x}.bpk")
        ]
    );
    let err = store
        .read("Interface\\missing.lua")
        .unwrap_err()
        .to_string();
    assert!(err.contains("file not in patch chain"), "{err}");
    assert_eq!(store.listfile().unwrap().len(), 2);
}

#[test]
fn the_cache_serves_reads_with_the_origin_gone() {
    let (origin, cache) = (Scratch::new("origin"), Scratch::new("cache"));
    publish(&origin.0, &[9u8; 2048]);
    open(&cache.0, &origin.0).read("Sound\\one.wav").unwrap();
    std::fs::remove_dir_all(&origin.0).unwrap();

    let store = open(&cache.0, &origin.0);
    assert_eq!(store.read("Sound\\one.wav").unwrap(), vec![7u8; 4096]);
    assert!(
        store.read("Interface\\a.lua").is_err(),
        "pack 0 was never cached"
    );
}

#[test]
fn a_repack_replaces_the_index_and_only_the_changed_packs() {
    let (origin, cache) = (Scratch::new("origin"), Scratch::new("cache"));
    publish(&origin.0, &[9u8; 2048]);
    let first = open(&cache.0, &origin.0);
    first.read("Interface\\a.lua").unwrap();
    first.read("Sound\\two.wav").unwrap();
    let before = cached_packs(&cache.0);
    let old_build = first.index().build_id();
    drop(first);

    publish(&origin.0, &noise(3000, 2));
    let second = open(&cache.0, &origin.0);
    assert_ne!(
        second.index().build_id(),
        old_build,
        "the re-pack is a new build"
    );
    let after = cached_packs(&cache.0);
    assert_eq!(
        after,
        vec![before[0].clone()],
        "the unchanged solid pack is kept"
    );
    assert_eq!(second.read("Sound\\two.wav").unwrap(), noise(3000, 2));
}

#[test]
fn a_download_that_fails_its_crc_is_not_kept() {
    let (origin, cache) = (Scratch::new("origin"), Scratch::new("cache"));
    publish(&origin.0, &[9u8; 2048]);
    let served = origin.0.join(benilla_pack::pack_path(1));
    let mut wire = std::fs::read(&served).unwrap();
    let last = wire.len() - 1;
    wire[last] ^= 0xff;
    std::fs::write(&served, wire).unwrap();

    let store = open(&cache.0, &origin.0);
    let err = store.read("Sound\\two.wav").unwrap_err();
    assert!(format!("{err:#}").contains("CRC32"), "{err:#}");
    assert!(cached_packs(&cache.0).is_empty());
    assert!(matches!(store.try_read("Sound\\two.wav"), TryRead::Missing));
}

#[test]
fn try_read_waits_on_the_warm_queue_then_answers() {
    let (origin, cache) = (Scratch::new("origin"), Scratch::new("cache"));
    publish(&origin.0, &[9u8; 2048]);
    let store = open(&cache.0, &origin.0);
    assert!(matches!(store.try_read("Sound\\one.wav"), TryRead::Pending));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match store.try_read("Sound\\one.wav") {
            TryRead::Ready(bytes) => {
                assert_eq!(bytes, vec![7u8; 4096]);
                break;
            }
            TryRead::Pending if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            _ => panic!("the warm never landed"),
        }
    }
    assert!(store.is_ready("Sound\\one.wav"));
    assert!(store.is_ready("Not\\held.blp"));
}

#[test]
fn the_boot_set_comes_off_the_pointer() {
    let json = r#"{"version":1,"build_id":"bd5a10ebdfb00cdf","boot":{"packs":[[0,4],[1,2]],"bytes":6},"world":{"packs":[[1,2],[72,3]]}}"#;
    assert_eq!(pointer_build_id(json), Some(0xbd5a_10eb_dfb0_0cdf));
    assert_eq!(boot_pack_ids(json), vec![0, 1, 72]);
    assert_eq!(pointer_build_id("{}"), None);
}

#[test]
fn a_local_tree_reads_in_place() {
    let tree = Scratch::new("tree");
    publish(&tree.0, &[9u8; 2048]);
    let store = NetPackStore::open(&tree.0, None).unwrap();
    assert_eq!(store.read("Sound\\one.wav").unwrap(), vec![7u8; 4096]);
}
