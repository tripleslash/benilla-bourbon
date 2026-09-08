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
