//! Inline `|T…|t` textures for the pfUI extension. Build 5875 has no such escape and the stock
//! decoder ([`benilla_ui::markup`]) stays as it is: an engine with the escape enabled swaps each
//! one for a private-use placeholder character before the text reaches the markup parser, and the
//! layout gives that character the texture's width and quad instead of a glyph.
//!
//! The grammar is the later clients'
//! `|Tpath:size1:size2:xoff:yoff:texW:texH:left:right:top:bottom:r:g:b|t`, and the behaviour is the
//! one ClassicAPI documents for the 1.12 client it patches: widths, heights and wraps count the
//! icon, a tall icon grows its line, an EditBox shows the markup raw, and text another player
//! typed draws no art.

use std::borrow::Cow;
use std::collections::HashMap;

use bevy::math::Rect;
use bevy::prelude::*;

use crate::ui_pass::{UiQuad, UvRect};

use super::engine::TextEngine;
use super::markup::ColorRun;

/// Supplementary Private Use Area-A: one code point per distinct escape.
const FIRST: u32 = 0xF_0000;
const LAST: u32 = 0xF_FFFD;

/// Where a font-height icon's centre sits in its line, as a fraction of the font height.
const CENTRE: f32 = 0.6;

/// Whether `ch` stands for an inline texture: it has no glyph in any face.
pub(super) fn is_placeholder(ch: char) -> bool {
    (FIRST..=LAST).contains(&(ch as u32))
}

struct Spec {
    path: String,
    size1: f32,
    size2: Option<f32>,
    offset: (f32, f32),
    /// `[left, right, top, bottom]` in 0..1; `None` samples the whole texture.
    crop: Option<[f32; 4]>,
    /// The `r:g:b` vertex tint in 0..1; white leaves the texture's own colours.
    tint: [f32; 3],
    /// `None` until resolved; `Some(None)` is a path with no texture.
    handle: Option<Option<Handle<Image>>>,
}

impl Spec {
    /// The later clients' size law over the text height, in device px.
    fn size_dev(&self, ppem: u16, units_to_dev: f32) -> (f32, f32) {
        let text_h = f32::from(ppem);
        let dev = |units: f32| (units * units_to_dev).round().max(1.0);
        match (self.size1 > 0.0, self.size2) {
            (false, None) => (text_h, text_h),
            (true, None) => (dev(self.size1), dev(self.size1)),
            (false, Some(aspect)) if aspect > 0.0 => ((aspect * text_h).round(), text_h),
            (false, Some(_)) => (text_h, text_h),
            (true, Some(width)) if width > 0.0 => (dev(width), dev(self.size1)),
            (true, Some(_)) => (text_h, dev(self.size1)),
        }
    }
}

/// Parse the text between `|T` and `|t`.
fn parse(body: &str) -> Option<Spec> {
    let mut fields = body.split(':');
    let path = fields.next().filter(|p| !p.is_empty())?.to_string();
    let numbers: Vec<f32> = fields
        .map(|f| f.trim().parse::<f32>().unwrap_or(0.0))
        .collect();
    let at = |i: usize| numbers.get(i).copied().unwrap_or(0.0);
    let crop = (at(4) > 0.0 && at(5) > 0.0 && numbers.len() >= 10)
        .then(|| [at(6) / at(4), at(7) / at(4), at(8) / at(5), at(9) / at(5)]);
    Some(Spec {
        path,
        size1: at(0),
        size2: numbers.get(1).copied(),
        offset: (at(2), at(3)),
        crop,
        tint: if numbers.len() >= 13 {
            [at(10) / 255.0, at(11) / 255.0, at(12) / 255.0]
        } else {
            [1.0; 3]
        },
        handle: None,
    })
}

/// The escapes one [`TextEngine`] has seen. Off until the extension enables it, so a stock client
/// and every stock test draw `|T…|t` literally.
#[derive(Default)]
pub(crate) struct Registry {
    enabled: bool,
    /// The VM's atlases for `|A…|a` escapes, as of their generation.
    atlases: HashMap<String, benilla_ui::script::pfui_host::inline_texture::AtlasPiece>,
    atlas_generation: u64,
    /// Device px per unit of an explicit size: the UI scale of the string being laid out times
    /// the DPI, set by the callers that know it.
    units_to_dev: f32,
    by_body: HashMap<String, char>,
    specs: Vec<Spec>,
}

impl Registry {
    fn spec(&self, ch: char) -> Option<&Spec> {
        if !is_placeholder(ch) {
            return None;
        }
        self.specs.get((ch as u32 - FIRST) as usize)
    }

    /// `text` with each well-formed escape replaced by its placeholder, an atlas escape by its
    /// atlas's texture escape's. An escaped pipe (`||T`), an escape with no closing tag and an
    /// atlas the VM never registered stay as they are.
    pub(super) fn substitute<'a>(&mut self, text: &'a str) -> Cow<'a, str> {
        use benilla_ui::script::pfui_host::inline_texture as escape;
        if !self.enabled || !(text.contains("|T") || text.contains("|A")) {
            return Cow::Borrowed(text);
        }
        let mut out = String::with_capacity(text.len());
        let mut copied = 0usize;
        for span in escape::spans(text) {
            let body = &text[span.start + 2..span.end - 2];
            let atlas = if escape::is_atlas(text, &span) {
                match escape::atlas_body(body, &self.atlases) {
                    Some(atlas) => Some(atlas),
                    None => continue,
                }
            } else {
                None
            };
            let Some(ch) = self.placeholder(atlas.as_deref().unwrap_or(body)) else {
                continue;
            };
            out.push_str(&text[copied..span.start]);
            out.push(ch);
            copied = span.end;
        }
        if copied == 0 {
            return Cow::Borrowed(text);
        }
        out.push_str(&text[copied..]);
        Cow::Owned(out)
    }

    fn placeholder(&mut self, body: &str) -> Option<char> {
        if let Some(ch) = self.by_body.get(body) {
            return Some(*ch);
        }
        let ch = char::from_u32(FIRST + u32::try_from(self.specs.len()).ok()?)
            .filter(|c| is_placeholder(*c))?;
        self.specs.push(parse(body)?);
        self.by_body.insert(body.to_string(), ch);
        Some(ch)
    }

    /// One character's step in device px: a placeholder's texture width with the outline's
    /// clearance on either side, else nothing.
    pub(super) fn advance_char(&self, ch: char, ppem: u16, radius: u8) -> f32 {
        if self.specs.is_empty() {
            return 0.0;
        }
        self.spec(ch).map_or(0.0, |s| {
            s.size_dev(ppem, self.units_to_dev).0 + 2.0 * f32::from(radius)
        })
    }

    #[cfg(test)]
    /// The device-px width the placeholders in `text` add to a line. An outlined font's ink
    /// reaches `radius` past each glyph, so an icon keeps that much clear on either side.
    pub(super) fn advance(&self, text: &str, ppem: u16, radius: u8) -> f32 {
        if self.specs.is_empty() {
            return 0.0;
        }
        text.chars()
            .filter_map(|c| self.spec(c))
            .map(|s| s.size_dev(ppem, self.units_to_dev).0 + 2.0 * f32::from(radius))
            .sum()
    }

    /// The whole rows of a message frame the grown lines of `lines` add: its rows are the font
    /// height each, so a tall icon takes the next whole row.
    pub(super) fn extra_rows(
        &self,
        lines: impl Iterator<Item = Vec<ColorRun>>,
        ppem: u16,
    ) -> usize {
        if self.specs.is_empty() {
            return 0;
        }
        let extra: f32 = lines.map(|line| self.extra_height(&line, ppem, 1.0)).sum();
        (extra / f32::from(ppem)).ceil() as usize
    }

    /// How far the tallest texture of `line` stands past the font height, in logical px: the line
    /// grows by it, half above its glyphs and half below, as the later clients grow a line.
    pub(super) fn extra_height(&self, line: &[ColorRun], ppem: u16, dpi: f32) -> f32 {
        if self.specs.is_empty() {
            return 0.0;
        }
        let tallest = line
            .iter()
            .flat_map(|run| run.text.chars())
            .filter_map(|c| self.spec(c))
            .map(|s| s.size_dev(ppem, self.units_to_dev).1)
            .fold(0.0, f32::max);
        // Whole device px on each side, so the glyph rows stay on the pixel grid.
        ((tallest - f32::from(ppem)).max(0.0) * 0.5).ceil() * 2.0 / dpi
    }

    /// Whether `quad` is an inline texture: a drop shadow repeats glyphs, not these.
    pub(crate) fn owns(&self, quad: &UiQuad) -> bool {
        quad.texture.as_ref().is_some_and(|t| {
            self.specs
                .iter()
                .any(|s| matches!(&s.handle, Some(Some(h)) if h == t))
        })
    }

    /// Load what has been seen and not yet looked up.
    fn resolve(&mut self, mut load: impl FnMut(&str) -> Option<Handle<Image>>) {
        for spec in self.specs.iter_mut().filter(|s| s.handle.is_none()) {
            spec.handle = Some(load(&spec.path));
        }
    }
}

/// Set the scale an explicit `|T` size is in: screen px per UI unit of the string's owner.
pub(crate) fn set_scale(e: &mut TextEngine, scale: f32) {
    e.inline.units_to_dev = scale * e.dpi();
}

/// One laid-out line's inline textures, pending the line's justify shift.
#[derive(Default)]
pub(super) struct Line {
    pending: Vec<Placed>,
}

struct Placed {
    x_dev: f32,
    w_dev: f32,
    h_dev: f32,
    offset_dev: (f32, f32),
    uv: UvRect,
    texture: Option<Handle<Image>>,
    color: [f32; 4],
    ppem: u16,
}

impl Line {
    /// Seat `ch` at the pen if it is a placeholder, and answer the pen's step in device px.
    pub(super) fn place(
        &mut self,
        e: &TextEngine,
        ppem: u16,
        radius: u8,
        ch: char,
        pen_dev: f32,
        alpha: f32,
    ) -> Option<f32> {
        let spec = e.inline.spec(ch)?;
        let scale = e.inline.units_to_dev;
        let (w_dev, h_dev) = spec.size_dev(ppem, scale);
        let ink = f32::from(radius);
        self.pending.push(Placed {
            x_dev: pen_dev + ink,
            w_dev,
            h_dev,
            offset_dev: (
                (spec.offset.0 * scale).round(),
                (spec.offset.1 * scale).round(),
            ),
            uv: spec.crop.map_or(UvRect::FULL, |[l, r, t, b]| {
                UvRect::from_rect(Rect::new(l, t, r, b))
            }),
            texture: spec.handle.clone().flatten(),
            color: [spec.tint[0], spec.tint[1], spec.tint[2], alpha],
            ppem,
        });
        Some(w_dev + 2.0 * ink)
    }

    /// Emit the line's textures against the glyph cell whose top is `cell_top_dev`. One no taller
    /// than the font centres 0.6 of the font height down, where the text's own weight sits; a
    /// taller one centres on the cell, and so in the line it grew. Only the string's alpha reaches
    /// them, never its colour.
    pub(super) fn emit(
        &mut self,
        quads: &mut Vec<UiQuad>,
        origin_dev_x: f32,
        cell_top_dev: f32,
        dpi: f32,
        z_key: u128,
    ) {
        for p in self.pending.drain(..) {
            let Some(texture) = p.texture else { continue };
            let x = (origin_dev_x + p.x_dev + p.offset_dev.0) / dpi;
            let em = f32::from(p.ppem);
            let centre = if p.h_dev > em { 0.5 } else { CENTRE };
            let top = cell_top_dev.round() + (em * centre - p.h_dev * 0.5).round() - p.offset_dev.1;
            let y = top / dpi;
            quads.push(UiQuad {
                rect: Rect::new(x, y, x + p.w_dev / dpi, y + p.h_dev / dpi),
                z_key,
                texture: Some(texture),
                uv: p.uv,
                color: p.color,
                ..default()
            });
        }
    }
}

/// Register and load the textures of a string about to be drawn, so an icon appears in the frame
/// its text does.
pub(crate) fn prepare(
    atlas: &super::UiFontAtlas,
    text: &str,
    assets: &mut Option<ResMut<benilla_assets::WorldAssets>>,
    images: &mut Assets<Image>,
) {
    if !text.contains("|T") && !text.contains("|A") {
        return;
    }
    let Some(assets) = assets.as_mut() else {
        return;
    };
    let mut engine = atlas.lock();
    engine.inline.substitute(text);
    engine
        .inline
        .resolve(|path| assets.sprite_texture(path, images));
}

/// Drop the escapes from a line another player typed: only the client's and its addons' own text
/// draws art. Addon messages are data and pass untouched.
pub(crate) fn strip_typed(sender_guid: u64, language: u32, text: &mut String) {
    const ADDON: u32 = u32::MAX;
    let from_player = sender_guid != 0 && sender_guid >> 48 == 0;
    if !from_player || language == ADDON {
        return;
    }
    if let Cow::Owned(stripped) = benilla_ui::script::pfui_host::inline_texture::strip(text) {
        *text = stripped;
    }
}

/// Enable the escape on the client's engine, take the VM's atlases when they changed, and load
/// the textures a measure met before any draw.
fn feed(
    atlas: Option<Res<super::UiFontAtlas>>,
    assets: Option<ResMut<benilla_assets::WorldAssets>>,
    mut images: ResMut<Assets<Image>>,
    script: Option<NonSend<benilla_ui::script::UiScript>>,
) {
    let (Some(atlas), Some(mut assets)) = (atlas, assets) else {
        return;
    };
    let mut engine = atlas.lock();
    engine.inline.enabled = true;
    if let Some((generation, atlases)) = script.as_ref().and_then(|script| {
        benilla_ui::script::pfui_host::inline_texture::atlases_since(
            script.lua(),
            engine.inline.atlas_generation,
        )
    }) {
        engine.inline.atlas_generation = generation;
        engine.inline.atlases = atlases;
    }
    engine
        .inline
        .resolve(|path| assets.sprite_texture(path, &mut images));
}

pub(crate) fn register(app: &mut App) {
    app.add_systems(Update, feed.in_set(crate::ui_script::UiFeed));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled() -> Registry {
        Registry {
            enabled: true,
            units_to_dev: 2.0,
            ..default()
        }
    }

    #[test]
    fn an_atlas_escape_draws_as_its_atlass_texture_escape() {
        let mut registry = enabled();
        registry.atlases.insert(
            "gem".into(),
            benilla_ui::script::pfui_host::inline_texture::AtlasPiece {
                file: "Interface\\Gems".into(),
                width: 16.0,
                height: 16.0,
                coords: [0.0, 0.5, 0.0, 0.5],
            },
        );
        let atlas = registry.substitute("a|A:Gem:14:14|ab").into_owned();
        let texture = registry
            .substitute("a|TInterface\\Gems:14:14:0:0:1:1:0:0.5:0:0.5|tb")
            .into_owned();
        assert_eq!(atlas, texture);
        assert_eq!(atlas.chars().count(), 3);
        assert_eq!(
            registry.substitute("a|A:Missing:14|ab"),
            "a|A:Missing:14|ab"
        );
    }

    #[test]
    fn a_disabled_registry_leaves_the_escape_literal() {
        let mut registry = Registry::default();
        let text = "a|TInterface\\Icons\\Foo:16:16|tb";
        assert!(matches!(registry.substitute(text), Cow::Borrowed(t) if t == text));
        assert_eq!(registry.advance(text, 12, 0), 0.0);
    }

    #[test]
    fn an_escape_becomes_one_placeholder_and_repeats_share_it() {
        let mut registry = enabled();
        let out = registry
            .substitute("a|TInterface\\Icons\\Foo:16|tb|TInterface\\Icons\\Foo:16|t")
            .into_owned();
        let chars: Vec<char> = out.chars().collect();
        assert_eq!(chars.len(), 4);
        assert_eq!((chars[0], chars[2]), ('a', 'b'));
        assert!(is_placeholder(chars[1]));
        assert_eq!(chars[1], chars[3]);
        assert_eq!(registry.specs.len(), 1);
        // A second pass over substituted text changes nothing.
        assert!(matches!(registry.substitute(&out), Cow::Borrowed(_)));
    }

    #[test]
    fn an_escaped_pipe_or_an_unclosed_escape_stays_text() {
        let mut registry = enabled();
        for text in ["a||Tfoo|t", "a|Tfoo:16", "a|T|tb"] {
            assert_eq!(registry.substitute(text), text, "{text:?}");
        }
        assert!(registry.specs.is_empty());
    }

    #[test]
    fn markup_around_an_escape_is_kept() {
        let mut registry = enabled();
        let out = registry
            .substitute("|cffff0000x|r|Tfoo:0|t|Hitem:1|h[y]|h")
            .into_owned();
        let ch = out.chars().find(|c| is_placeholder(*c)).expect("an icon");
        assert_eq!(out, format!("|cffff0000x|r{ch}|Hitem:1|h[y]|h"));
    }

    #[test]
    fn the_size_law_follows_the_text_height_and_the_unit_scale() {
        let size = |body: &str| parse(body).expect("a spec").size_dev(14, 2.0);
        assert_eq!(size("foo"), (14.0, 14.0));
        assert_eq!(size("foo:0"), (14.0, 14.0));
        assert_eq!(size("foo:0:0"), (14.0, 14.0));
        assert_eq!(size("foo:24"), (48.0, 48.0));
        assert_eq!(size("foo:24:12"), (24.0, 48.0));
        assert_eq!(size("foo:24:0"), (14.0, 48.0));
        assert_eq!(size("foo:0:2"), (28.0, 14.0));
    }

    #[test]
    fn a_crop_is_normalised_by_the_texture_size() {
        let spec = parse("foo:0:0:0:0:64:32:16:48:8:24").expect("a spec");
        assert_eq!(spec.crop, Some([0.25, 0.75, 0.25, 0.75]));
        assert_eq!(parse("foo:0:0:0:0").expect("a spec").crop, None);
    }

    /// `GetStringWidth`'s measure, which a tooltip sizes its cells by, counts an icon as the draw
    /// does and never as the bytes of its markup.
    #[test]
    fn the_string_width_counts_an_icon_and_not_its_markup() {
        use crate::ui_text::layout::{string_width, FontSpec};
        use benilla_ui::script::Outline;

        let Some(mut e) = crate::ui_text::engine::test_engine(1.0) else {
            return;
        };
        let spec = FontSpec {
            path: None,
            height: Some(12.0),
            outline: Outline::None,
            monochrome: false,
            spacing: 0.0,
            shadow_x: 0.0,
            alpha_gradient: None,
            non_space_wrap: false,
            max_lines: 0,
        };
        e.inline.enabled = true;
        set_scale(&mut e, 1.0);
        let plain = string_width(&mut e, "ab", spec, 0.0);
        let coin = "a|TInterface\\MoneyFrame\\UI-MoneyIcons:20:20:2:0:64:16:32:48:0:16|tb";
        let with_icon = string_width(&mut e, coin, spec, 0.0);
        assert!(
            with_icon > plain && with_icon < plain + 40.0,
            "one 20 px icon, got {with_icon} against {plain}"
        );
    }

    /// Measure and draw agree: the icon widens the string by its own width, the glyphs after it
    /// start past it, and no glyph is drawn for the escape's bytes.
    #[test]
    fn a_measured_icon_is_as_wide_as_it_draws() {
        use crate::ui_text::layout::{
            layout_text_quads, measure_text, FontSpec, Justify, TextSeat,
        };
        use benilla_ui::script::{JustifyH, JustifyV, Outline};

        let Some(mut e) = crate::ui_text::engine::test_engine(1.0) else {
            return;
        };
        let spec = FontSpec {
            path: None,
            height: Some(12.0),
            outline: Outline::None,
            monochrome: false,
            spacing: 0.0,
            shadow_x: 0.0,
            alpha_gradient: None,
            non_space_wrap: false,
            max_lines: 0,
        };
        let literal = measure_text(&mut e, "a|Tfoo:20|tb", None, spec).0;
        e.inline.enabled = true;
        set_scale(&mut e, 1.0);
        let plain = measure_text(&mut e, "ab", None, spec).0;
        let with_icon = measure_text(&mut e, "a|Tfoo:20|tb", None, spec).0;
        assert_eq!(with_icon, plain + 20.0);
        assert!(literal > with_icon, "the disabled engine drew the bytes");

        let handle = Handle::<Image>::default();
        e.inline.resolve(|_| Some(handle.clone()));
        let draw = |e: &mut TextEngine, text: &str| {
            layout_text_quads(
                e,
                text,
                Rect::new(0.0, 0.0, 0.0, 0.0),
                [1.0, 0.5, 0.5, 0.5],
                Justify {
                    h: JustifyH::Left,
                    v: JustifyV::Top,
                },
                7,
                spec,
                TextSeat::Exact,
            )
        };
        let plain_quads = draw(&mut e, "ab");
        let quads = draw(&mut e, "a|Tfoo:20|tb");
        assert_eq!(quads.len(), plain_quads.len() + 1);
        let icon = quads.iter().find(|q| e.inline.owns(q)).expect("the icon");
        assert_eq!((icon.rect.width(), icon.rect.height()), (20.0, 20.0));
        assert_eq!(icon.color, [1.0, 1.0, 1.0, 0.5], "untinted, at the alpha");
        let tint = "|Tfoo:0:0:0:0:0:0:0:0:0:0:255:0:51|t";
        e.inline.substitute(tint);
        e.inline.resolve(|_| Some(handle.clone()));
        let tinted = draw(&mut e, tint);
        assert_eq!(tinted[0].color, [1.0, 0.0, 0.2, 0.5]);
        assert_eq!(icon.z_key, 7);
        // The `b` sits one icon width right of where it sits without one.
        let last = |qs: &[UiQuad]| {
            qs.iter()
                .filter(|q| !e.inline.owns(q))
                .map(|q| q.rect.min.x)
                .fold(f32::MIN, f32::max)
        };
        assert_eq!(last(&quads), last(&plain_quads) + 20.0);

        // A texture taller than the font grows its line, and the block with it.
        let one = measure_text(&mut e, "ab", None, spec).1;
        assert_eq!(
            measure_text(&mut e, "a|Tfoo:20|tb", None, spec).1,
            one + 8.0
        );
        assert_eq!(measure_text(&mut e, "a|Tfoo:0|tb", None, spec).1, one);
        let rows = draw(&mut e, "a|Tfoo:20|tb\n|Tfoo:20|t");
        let tops: Vec<f32> = rows
            .iter()
            .filter(|q| e.inline.owns(q))
            .map(|q| q.rect.min.y)
            .collect();
        assert_eq!(tops, [0.0, 20.0], "two grown lines stack without overlap");
    }

    #[test]
    fn only_a_players_typed_line_loses_its_escapes() {
        let typed = |guid: u64, language: u32| {
            let mut text = "look |Tfoo:64|t here".to_string();
            strip_typed(guid, language, &mut text);
            text
        };
        assert_eq!(typed(5, 7), "look  here");
        assert_eq!(typed(0, 0), "look |Tfoo:64|t here", "the client's own line");
        assert_eq!(
            typed(0xF130_0000_0000_0001, 0),
            "look |Tfoo:64|t here",
            "a creature"
        );
        assert_eq!(typed(5, u32::MAX), "look |Tfoo:64|t here", "addon data");
    }

    #[test]
    fn a_tall_icon_takes_whole_message_rows() {
        let mut registry = enabled();
        let line = |registry: &mut Registry, text: &str| {
            vec![ColorRun {
                text: registry.substitute(text).into_owned(),
                color: [1.0; 4],
                link: None,
                marks: 0,
            }]
        };
        let tall = line(&mut registry, "a|Tfoo:12|t");
        let small = line(&mut registry, "a|Tfoo:0|t");
        // 24 device px on a 14 px font: 10 px over, one more row.
        assert_eq!(registry.extra_rows([tall].into_iter(), 14), 1);
        assert_eq!(registry.extra_rows([small].into_iter(), 14), 0);
    }

    #[test]
    fn the_advance_sums_the_placeholders_of_a_line() {
        let mut registry = enabled();
        let out = registry.substitute("x|Tfoo:0|ty|Tbar:10|t").into_owned();
        assert_eq!(registry.advance(&out, 12, 0), 12.0 + 20.0);
        assert_eq!(
            registry.advance(&out, 12, 2),
            12.0 + 20.0 + 8.0,
            "outline ink"
        );
        assert_eq!(registry.advance("xy", 12, 0), 0.0);
    }
}
