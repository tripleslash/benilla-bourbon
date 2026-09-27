//! [`benilla_formats::web::encode_name`] is pure string math (no browser dependency), so it is
//! exercised here natively rather than only under a wasm target — it is the client half of the Data
//! URL scheme (Lane A ↔ Lane H), and a mismatch with the host's decode would silently 404 every
//! chain name with a space, backslash, or other reserved byte in it.

use benilla_formats::web::encode_name;

#[test]
fn matches_the_data_url_schemes_worked_example() {
    // The plan's own worked example: a backslash-separated chain name with a space in it.
    assert_eq!(
        encode_name(r"Interface\Glues\a b.blp"),
        "Interface%5CGlues%5Ca%20b.blp"
    );
}

#[test]
fn unreserved_characters_pass_through_unescaped() {
    let unreserved = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.!~*'()";
    assert_eq!(encode_name(unreserved), unreserved);
}

#[test]
fn every_other_byte_is_percent_escaped_uppercase_hex() {
    // Backslash, forward slash, space, and a non-ASCII byte (encodeURIComponent works per UTF-8
    // byte, not per Unicode scalar) all leave the unreserved set.
    assert_eq!(encode_name("\\"), "%5C");
    assert_eq!(encode_name("/"), "%2F");
    assert_eq!(encode_name(" "), "%20");
    assert_eq!(encode_name("é"), "%C3%A9");
}

#[test]
fn empty_name_encodes_to_empty() {
    assert_eq!(encode_name(""), "");
}

/// **The layer's key-space constant is spelled three times and must be one number.**
///
/// `benilla_formats::layers::HD_PACK_KEY_BASE` here, `HD_PACK_KEY_BASE` in `web/packs.js`, and
/// `HD_PACK_KEY_BASE` in `web/packworker.js`. The Rust side offsets a resident-map key by it; the
/// page offsets the layer's `boot-packs.json` ids by it before warming them; the decode worker
/// subtracts it to pick which container a key belongs to.
///
/// Get one of the three wrong and nothing errors: the page warms a key nothing reads, or the worker
/// decodes a layer pack against the corpus's index and hands back a body that is either a zstd
/// failure or — the bad case — a frame sliced out of the wrong bytes. There is no runtime check that
/// could catch it, because every value involved is a plausible pack id. So it is caught here, by
/// reading the two JS files the way a human would.
///
/// The same reading pins `HD_DATA_SEGMENT`, for the same reason one step further out: the page
/// fetches `/{segment}/index.bpi` and Rust builds `{origin}/{segment}` for every pack URL, so a
/// disagreement is a container the client cannot find at an address the page already warmed.
#[test]
fn the_layer_key_space_is_the_same_number_in_rust_and_in_the_page() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let want = format!(
        "const HD_PACK_KEY_BASE = {};",
        // Spelled as the JS spells it — `1 << 30` — so the assertion reads as the line it is
        // looking for rather than as a magic number that happens to match.
        if benilla_formats::layers::HD_PACK_KEY_BASE == 1 << 30 {
            "1 << 30".to_string()
        } else {
            benilla_formats::layers::HD_PACK_KEY_BASE.to_string()
        }
    );
    for js in ["web/packs.js", "web/packworker.js"] {
        let path = root.join(js);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        assert!(
            text.contains(&want),
            "{js} does not spell `{want}` — the layer's key space has drifted from \
             benilla_formats::layers::HD_PACK_KEY_BASE"
        );
    }
    let packs = std::fs::read_to_string(root.join("web/packs.js")).expect("web/packs.js");
    assert!(
        packs.contains(&format!(
            "const HD_DATA_SEGMENT = '{}';",
            benilla_formats::layers::HD_DATA_SEGMENT
        )),
        "web/packs.js does not serve the layer from `{}`",
        benilla_formats::layers::HD_DATA_SEGMENT
    );
}

/// **The Turtle layer's key space and segment, spelled as the HD layer's are.** `TURTLE_PACK_KEY_BASE`
/// in both JS files as `2 ** 31` (a JavaScript `1 << 31` is negative, so the one spelling that would
/// look like the Rust is the one that is wrong), `TURTLE_DATA_SEGMENT` in the page, and the tree's
/// prefix in the service worker, which otherwise never caches the layer.
#[test]
fn the_turtle_layer_is_the_same_key_space_and_tree_in_rust_and_in_the_page() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let read = |js: &str| {
        std::fs::read_to_string(root.join(js)).unwrap_or_else(|e| panic!("reading {js}: {e}"))
    };
    assert_eq!(benilla_formats::layers::TURTLE_PACK_KEY_BASE, 1 << 31);
    for js in ["web/packs.js", "web/packworker.js"] {
        assert!(
            read(js).contains("const TURTLE_PACK_KEY_BASE = 2 ** 31;"),
            "{js} does not spell the Turtle layer's key space as `2 ** 31`"
        );
    }
    let segment = benilla_formats::layers::TURTLE_DATA_SEGMENT;
    assert!(read("web/packs.js").contains(&format!("const TURTLE_DATA_SEGMENT = '{segment}';")));
    assert!(read("web/sw.js").contains(&format!("const TURTLE_DATA_PREFIX = '/{segment}/';")));
    assert!(read("web/_headers").contains(&format!("/{segment}/p/*")));
}

/// **The reforged layer's key space and segment, spelled as the Turtle layer's are**: `3 * 2 ** 30`
/// in both JS files (a JavaScript `3 << 30` is negative), the segment in the page, the tree's prefix
/// in the service worker, and its packs' cache rule in `_headers`.
#[test]
fn the_reforged_layer_is_the_same_key_space_and_tree_in_rust_and_in_the_page() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let read = |js: &str| {
        std::fs::read_to_string(root.join(js)).unwrap_or_else(|e| panic!("reading {js}: {e}"))
    };
    assert_eq!(benilla_formats::layers::REFORGED_PACK_KEY_BASE, 3 << 30);
    for js in ["web/packs.js", "web/packworker.js"] {
        assert!(
            read(js).contains("const REFORGED_PACK_KEY_BASE = 3 * 2 ** 30;"),
            "{js} does not spell the reforged layer's key space as `3 * 2 ** 30`"
        );
    }
    let segment = benilla_formats::layers::REFORGED_DATA_SEGMENT;
    assert!(read("web/packs.js").contains(&format!("const REFORGED_DATA_SEGMENT = '{segment}';")));
    assert!(read("web/sw.js").contains(&format!("const REFORGED_DATA_PREFIX = '/{segment}/';")));
    assert!(read("web/_headers").contains(&format!("/{segment}/p/*")));
}

/// **The warm ladder is spelled twice and must be one order.**
///
/// `benilla_formats::web`'s five `PRIORITY_*` constants here, and five `const PRIORITY_* = n;`
/// lines in `web/packs.js` there. Rust classifies a name into a number; the page sorts its six
/// connections by that number and reserves slots against `priority <= PRIORITY_GEOMETRY`. Neither
/// side validates the other, and a disagreement does not error — it silently *reorders the
/// network*: swap two tiers and the tile a body is about to walk onto queues behind the art again,
/// which is a streaming regression nothing in a build catches and only a player standing at a tile
/// edge notices.
///
/// Read the way a human would, like the layer key space above.
#[test]
fn the_warm_ladder_is_the_same_order_in_rust_and_in_the_page() {
    use benilla_formats::web::{
        PRIORITY_GEOMETRY, PRIORITY_NORMAL, PRIORITY_TERRAIN, PRIORITY_TERRAIN_FOCUS,
        PRIORITY_TERRAIN_NEAR,
    };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let packs = std::fs::read_to_string(root.join("web/packs.js")).expect("web/packs.js");
    for (name, value) in [
        ("PRIORITY_TERRAIN_FOCUS", PRIORITY_TERRAIN_FOCUS),
        ("PRIORITY_TERRAIN_NEAR", PRIORITY_TERRAIN_NEAR),
        ("PRIORITY_TERRAIN", PRIORITY_TERRAIN),
        ("PRIORITY_GEOMETRY", PRIORITY_GEOMETRY),
        ("PRIORITY_NORMAL", PRIORITY_NORMAL),
    ] {
        let want = format!("const {name} = {value};");
        assert!(
            packs.contains(&want),
            "web/packs.js does not spell `{want}` — the warm ladder has drifted from              benilla_formats::web"
        );
    }
    // The property the page's slot reservation rests on: every terrain tier is inside the
    // collision lane, which `packs.js` tests as one `priority <= PRIORITY_GEOMETRY`.
    const {
        assert!(PRIORITY_TERRAIN_FOCUS < PRIORITY_TERRAIN_NEAR);
        assert!(PRIORITY_TERRAIN_NEAR < PRIORITY_TERRAIN);
        assert!(PRIORITY_TERRAIN < PRIORITY_GEOMETRY);
        assert!(PRIORITY_GEOMETRY < PRIORITY_NORMAL);
    }
}
