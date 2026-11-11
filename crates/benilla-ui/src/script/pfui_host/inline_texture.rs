//! The inline-texture escapes' byte spans, `|T…|t` and the atlas form `|A…|a`, for the callers
//! that must not draw one: an EditBox shows its markup raw, and text another player typed never
//! draws art. The stock decoder ([`crate::markup`]) has neither token, so the scan rides its
//! token walk: an escaped pipe (`||T`) and an escape with no closing tag are ordinary text. An
//! atlas escape draws as the `|T` escape of its atlas's file and rectangle ([`atlas_body`]).

use mlua::Lua;
use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;

/// Each escape's opening and closing tags.
const ESCAPES: [(&str, &str); 2] = [("|T", "|t"), ("|A", "|a")];

/// Each well-formed escape of `text`, `|T` through `|t` or `|A` through `|a`.
pub fn spans(text: &str) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    if !text.contains("|T") && !text.contains("|A") {
        return out;
    }
    let mut at = 0usize;
    while let Some(token) = crate::markup::token_at(text, at) {
        let rest = &text[at..];
        match ESCAPES
            .iter()
            .find(|(open, _)| rest.starts_with(open))
            .and_then(|(_, close)| rest[2..].find(close))
        {
            Some(end) if end > 0 => {
                out.push(at..at + end + 4);
                at += end + 4;
            }
            _ => at += token.byte_len,
        }
    }
    out
}

/// Whether the escape at `span` of `text` is the atlas form.
pub fn is_atlas(text: &str, span: &Range<usize>) -> bool {
    text.as_bytes().get(span.start + 1) == Some(&b'A')
}

/// An atlas as `C_Texture.RegisterAtlas` keeps it: its file, its size and its rectangle in the
/// file in texture coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasPiece {
    pub file: String,
    pub width: f32,
    pub height: f32,
    pub coords: [f32; 4],
}

/// The atlases by lower-cased name, and a count of the changes, which the renderer follows.
#[derive(Default)]
struct Atlases {
    generation: u64,
    pieces: HashMap<String, AtlasPiece>,
}

/// Record an atlas `|A…|a` escapes can name.
pub fn register_atlas(lua: &Lua, name: &str, piece: AtlasPiece) {
    if lua.app_data_ref::<Atlases>().is_none() {
        lua.set_app_data(Atlases::default());
    }
    let mut atlases = lua.app_data_mut::<Atlases>().expect("atlases");
    atlases.generation += 1;
    atlases.pieces.insert(name.to_lowercase(), piece);
}

/// The atlases with their generation, when they changed after `generation`.
pub fn atlases_since(lua: &Lua, generation: u64) -> Option<(u64, HashMap<String, AtlasPiece>)> {
    let atlases = lua.app_data_ref::<Atlases>()?;
    (atlases.generation != generation).then(|| (atlases.generation, atlases.pieces.clone()))
}

/// An atlas escape's body (`:atlas:height:width[:xoff:yoff[:r:g:b]]`) as the `|T` body that
/// draws it: the atlas's file cropped to its rectangle. A zero size follows the text height, the
/// other side keeping the atlas's aspect; `None` for an unknown atlas.
pub fn atlas_body(body: &str, atlases: &HashMap<String, AtlasPiece>) -> Option<String> {
    let mut fields = body.strip_prefix(':').unwrap_or(body).split(':');
    let piece = atlases.get(&fields.next()?.to_lowercase())?;
    let numbers: Vec<f32> = fields
        .map(|f| f.trim().parse::<f32>().unwrap_or(0.0))
        .collect();
    let at = |i: usize| numbers.get(i).copied().unwrap_or(0.0);
    let aspect = if piece.height > 0.0 {
        piece.width / piece.height
    } else {
        1.0
    };
    // `|T`'s sizes are height then width; a zero height with a width is the text height there.
    let (size1, size2) = match (at(0) > 0.0, at(1) > 0.0) {
        (true, true) => (at(0), at(1)),
        (true, false) => (at(0), at(0) * aspect),
        (false, true) => (at(1) / aspect, at(1)),
        (false, false) => (0.0, aspect),
    };
    let [left, right, top, bottom] = piece.coords;
    let mut out = format!(
        "{}:{size1}:{size2}:{}:{}:1:1:{left}:{right}:{top}:{bottom}",
        piece.file,
        at(2),
        at(3)
    );
    if numbers.len() >= 7 {
        out.push_str(&format!(":{}:{}:{}", at(4), at(5), at(6)));
    }
    Some(out)
}

/// `text` with each escape's opening pipe doubled, so it draws as the bytes it is written with.
pub fn raw(text: &str) -> Cow<'_, str> {
    let spans = spans(text);
    if spans.is_empty() {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len() + spans.len());
    let mut copied = 0usize;
    for span in spans {
        out.push_str(&text[copied..span.start]);
        out.push('|');
        copied = span.start;
    }
    out.push_str(&text[copied..]);
    Cow::Owned(out)
}

/// `text` without its escapes.
pub fn strip(text: &str) -> Cow<'_, str> {
    let spans = spans(text);
    if spans.is_empty() {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut copied = 0usize;
    for span in spans {
        out.push_str(&text[copied..span.start]);
        copied = span.end;
    }
    out.push_str(&text[copied..]);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_follow_the_token_walk() {
        assert_eq!(spans("a|Tfoo:16|tb"), vec![1..11]);
        assert_eq!(spans("|Ta|t|Tb|t"), vec![0..5, 5..10]);
        for text in ["a||Tfoo|t", "a|Tfoo:16", "a|T|tb", "plain"] {
            assert!(spans(text).is_empty(), "{text:?}");
        }
    }

    #[test]
    fn an_atlas_escape_spans_as_a_texture_escape_does() {
        let text = "a|A:Gem:16:0|ab|Tfoo|t";
        let found = spans(text);
        assert_eq!(found, vec![1..14, 15..22]);
        assert!(is_atlas(text, &found[0]) && !is_atlas(text, &found[1]));
        assert_eq!(strip(text), "ab");
        assert!(spans("a|A:Gem:16:0").is_empty());
    }

    #[test]
    fn an_atlas_body_is_its_files_texture_escape() {
        let atlases = HashMap::from([(
            "gem".to_string(),
            AtlasPiece {
                file: "Interface\\Gems".into(),
                width: 32.0,
                height: 16.0,
                coords: [0.0, 0.5, 0.25, 0.75],
            },
        )]);
        assert_eq!(
            atlas_body(":GEM:16:0", &atlases).as_deref(),
            Some("Interface\\Gems:16:32:0:0:1:1:0:0.5:0.25:0.75")
        );
        assert_eq!(
            atlas_body(":Gem:0:0:2:-1:255:0:0", &atlases).as_deref(),
            Some("Interface\\Gems:0:2:2:-1:1:1:0:0.5:0.25:0.75:255:0:0")
        );
        assert_eq!(atlas_body(":Missing:16:16", &atlases), None);
    }

    #[test]
    fn raw_text_draws_the_bytes_and_stripped_text_drops_them() {
        assert_eq!(raw("a|Tfoo:16|tb"), "a||Tfoo:16|tb");
        assert!(spans(&raw("a|Tfoo:16|tb|Tc|t")).is_empty());
        assert_eq!(strip("a|Tfoo:16|tb|Tc|t!"), "ab!");
        assert!(matches!(raw("a||Tfoo|t"), Cow::Borrowed(_)));
        assert!(matches!(strip("plain"), Cow::Borrowed(_)));
    }
}
