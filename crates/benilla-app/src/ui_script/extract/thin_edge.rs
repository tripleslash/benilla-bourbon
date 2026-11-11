//! A texture quad thinner than one device pixel, a backdrop's border piece above all, seated on a
//! pixel row or column.
//!
//! Deviation: (tortoise) a sub-pixel quad is seated on a pixel, because pfUI sizes its borders to
//! one device pixel of UIParent, so on a frame with
//! another effective scale (the world map at 0.7) a border is a fraction of a pixel thick. A quad
//! that thin covers a pixel centre only by luck, and where it covers none the edge is not drawn at
//! all: the world map's close button lost its bottom border. The Turtle WoW client shows all four
//! edges at the same sizes. This moves such a piece, by less than a pixel, onto the nearest
//! centre; a quad a pixel thick or more is never moved.
//!
//! Observed from screenshots of both clients, not read from the reference's rasteriser.

use bevy::math::Rect;

/// One axis: the shift, in device pixels, that makes `[lo, hi)` cover a pixel centre.
fn shift(lo: f32, hi: f32) -> f32 {
    let thick = hi - lo;
    if thick <= 0.0 || thick >= 1.0 {
        return 0.0;
    }
    // A centre sits at `k + 0.5`; the span covers one when the two ends fall in different cells
    // of the centre grid.
    if (lo - 0.5).ceil() < (hi - 0.5).ceil() {
        return 0.0;
    }
    let centre = ((lo + hi) * 0.5 - 0.5).round() + 0.5;
    if centre < lo {
        centre - lo
    } else {
        // The far end is exclusive, so it goes just past the centre.
        centre - hi + 1.0 / 64.0
    }
}

/// `rect`, in logical pixels, moved so that a side thinner than one device pixel is drawn.
pub(crate) fn seat(rect: Rect, dpi: f32) -> Rect {
    if dpi <= 0.0 {
        return rect;
    }
    let dx = shift(rect.min.x * dpi, rect.max.x * dpi) / dpi;
    let dy = shift(rect.min.y * dpi, rect.max.y * dpi) / dpi;
    if dx == 0.0 && dy == 0.0 {
        return rect;
    }
    Rect {
        min: rect.min + bevy::math::Vec2::new(dx, dy),
        max: rect.max + bevy::math::Vec2::new(dx, dy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn covered(lo: f32, hi: f32) -> bool {
        (lo - 0.5).ceil() < (hi - 0.5).ceil()
    }

    #[test]
    fn a_sub_pixel_edge_between_two_centres_moves_onto_the_nearer_one() {
        // The world map close button's bottom edge at 2133x1200: no centre inside.
        let (lo, hi) = (623.61_f32, 624.45_f32);
        assert!(!covered(lo, hi));
        let d = shift(lo, hi);
        assert!(d.abs() < 1.0, "less than a pixel");
        assert!(covered(lo + d, hi + d));
        assert!((lo + d..hi + d).contains(&624.5));
    }

    #[test]
    fn an_edge_that_is_drawn_or_a_pixel_thick_is_left_alone() {
        assert_eq!(shift(634.95, 635.79), 0.0, "covers 635.5 already");
        assert_eq!(shift(10.2, 11.2), 0.0, "a whole pixel always covers one");
        assert_eq!(shift(10.0, 26.0), 0.0);
        assert_eq!(shift(5.0, 5.0), 0.0, "an empty span has nothing to draw");
    }

    #[test]
    fn every_sub_pixel_span_ends_up_covering_a_centre() {
        for i in 0..200 {
            let lo = 100.0 + i as f32 * 0.013;
            for thick in [0.05_f32, 0.3, 0.54, 0.843, 0.99] {
                let d = shift(lo, lo + thick);
                assert!(d.abs() < 1.0);
                assert!(covered(lo + d, lo + thick + d), "lo={lo} thick={thick}");
            }
        }
    }

    #[test]
    fn the_seat_works_in_logical_pixels_and_moves_only_the_thin_axis() {
        let dpi = 2.0;
        // 0.4 logical = 0.8 device px tall, between centres; 20 px wide.
        let r = Rect::new(10.0, 311.82, 30.0, 312.2);
        let s = seat(r, dpi);
        assert_eq!((s.min.x, s.max.x), (10.0, 30.0));
        assert!(covered(s.min.y * dpi, s.max.y * dpi));
        assert!((s.height() - r.height()).abs() < 1e-4);
    }
}
