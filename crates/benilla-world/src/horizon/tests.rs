use super::*;

/// The eye at the origin looking down −Z (WoW north), a 90° square lens with no near offset, so
/// `x/z` and `y/z` are the plain slopes.
fn level_view() -> HorizonView {
    HorizonView::new(
        Vec3::ZERO,
        Vec3::NEG_Z,
        std::f32::consts::FRAC_PI_2,
        1.0,
        0.0,
        1000.0,
    )
    .expect("a level heading")
}

/// A flat chunk at `height` whose north-west corner is WoW `(nw_x, nw_y)`, 145 vertices in
/// stride-17 order.
fn flat_chunk(nw_x: f32, nw_y: f32, height: f32) -> Vec<[f32; 3]> {
    let u = benilla_formats::CHUNK_SIZE / 8.0;
    (0..145)
        .map(|i| {
            let (r, c) = (i / 17, i % 17);
            // Inner vertices (c ≥ 9) sit at the cell centres.
            let (r, c) = if c < 9 {
                (r as f32, c as f32)
            } else {
                (r as f32 + 0.5, (c - 9) as f32 + 0.5)
            };
            [nw_x - r * u, nw_y - c * u, height]
        })
        .collect()
}

fn chunk<'a>(cell: u8, positions: &'a [[f32; 3]], holes: u16) -> Chunk<'a> {
    let pts: Vec<Vec3> = positions.iter().map(|p| wow_to_bevy(*p)).collect();
    let (min, max) = pts[1..]
        .iter()
        .fold((pts[0], pts[0]), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    Chunk {
        cell: ((0, 0), cell),
        b: ChunkBox { min, max, holes },
        positions,
    }
}

fn horizon_of(view: HorizonView, w: Walk) -> Horizon {
    Horizon {
        walks: vec![WindowWalk {
            view,
            window: None,
            snapshots: w.snapshots,
        }],
        enforce: true,
        hidden_cells: w.hidden_cells,
        floor: 0.0,
        stats: w.stats,
    }
}

#[test]
fn fistp_rounds_half_to_even_and_stores_the_indefinite() {
    assert_eq!(fistp(0.5), 0);
    assert_eq!(fistp(1.5), 2);
    assert_eq!(fistp(2.5), 2);
    assert_eq!(fistp(-0.5), 0);
    assert_eq!(fistp(f64::NAN), i32::MIN);
    assert_eq!(fistp(1e12), i32::MIN);
}

#[test]
fn columns_are_64_per_unit_slope_about_160() {
    assert_eq!(col(0.0, COL_BIAS), 160);
    assert_eq!(col(1.0 / 64.0, COL_BIAS), 160);
    assert_eq!(col(2.0 / 64.0, COL_BIAS), 162);
    assert_eq!(col(-1.0, COL_BIAS), 96);
    assert_eq!(col(0.0, COL_BIAS + 1), 161);
}

#[test]
fn bands_are_33_yards_near_first_and_stop_at_32() {
    assert_eq!(band_of(-5.0), Some(0));
    assert_eq!(band_of(0.0), Some(0));
    assert_eq!(band_of(33.4), Some(1));
    assert_eq!(band_of(1050.0), Some(31));
    assert_eq!(band_of(1067.0), None);
}

#[test]
fn an_edge_raises_its_columns_to_its_lower_end() {
    let mut buf: Columns = [RESET; HORIZON_COLS];
    insert(&mut buf, &[[0.0, 0.2, 10.0], [0.5, 0.1, 10.0]]);
    assert_eq!(buf[159], RESET);
    assert_eq!(buf[160], 0.1);
    assert_eq!(buf[192], 0.1);
    assert_eq!(buf[193], RESET);
    // A higher edge raises; a lower one leaves the column.
    insert(&mut buf, &[[0.0, 0.3, 10.0], [0.0, 0.3, 10.0]]);
    assert_eq!(buf[160], 0.3);
    insert(&mut buf, &[[0.0, 0.05, 10.0], [0.0, 0.05, 10.0]]);
    assert_eq!(buf[160], 0.3);
}

#[test]
fn an_edge_with_an_end_nearer_than_the_gate_adds_nothing() {
    let mut buf: Columns = [RESET; HORIZON_COLS];
    insert(&mut buf, &[[0.0, 0.2, 2.7], [0.5, 0.1, 10.0]]);
    assert!(buf.iter().all(|&v| v == RESET));
}

#[test]
fn a_box_is_hidden_only_below_every_spanned_column_and_beyond_50() {
    let view = level_view();
    let buf: Columns = [0.1; HORIZON_COLS];
    let low = (Vec3::new(-5.0, -50.0, -110.0), Vec3::new(5.0, 5.0, -100.0));
    assert!(box_hidden(&view, &buf, low.0, low.1, 0));
    let tall = (Vec3::new(-5.0, -50.0, -110.0), Vec3::new(5.0, 20.0, -100.0));
    assert!(!box_hidden(&view, &buf, tall.0, tall.1, 0));
    // A corner nearer than 50 shows the box, unless flag 8 waives it.
    let near = (Vec3::new(-5.0, -50.0, -60.0), Vec3::new(5.0, 2.0, -40.0));
    assert!(!box_hidden(&view, &buf, near.0, near.1, 0));
    assert!(box_hidden(&view, &buf, near.0, near.1, FLAG_NEAR_OK));
    // One low column under the span shows it.
    let mut gap = buf;
    gap[160] = -1.0;
    assert!(!box_hidden(&view, &gap, low.0, low.1, 0));
}

#[test]
fn a_sphere_tops_out_at_its_centre_plus_its_radius() {
    let view = level_view();
    let buf: Columns = [0.1; HORIZON_COLS];
    let c = Vec3::new(0.0, 0.0, -100.0);
    assert!(sphere_hidden(&view, &buf, c, 1.0));
    assert!(!sphere_hidden(&view, &buf, c, 20.0));
    // Below the radius gate, and nearer than 50: both visible.
    assert!(!sphere_hidden(&view, &buf, c, 1e-8));
    assert!(!sphere_hidden(&view, &buf, Vec3::new(0.0, 0.0, -40.0), 1.0));
}

#[test]
fn an_inactive_horizon_hides_nothing() {
    let h = Horizon::default();
    assert!(!h.active());
    assert!(!h.hides_sphere(Vec3::new(0.0, -100.0, -500.0), 1.0));
    assert!(!h.hides_box(Vec3::splat(-1.0), Vec3::splat(1.0), 0));
    assert!(!h.hides_cell(&TerrainCell {
        tile: (0, 0),
        cell: 0
    }));
}

/// A 30 yd ridge chunk 100 to 133 yd north of a level eye: it hides what lies behind and below
/// its far edge, and nothing in front of it, however low.
#[test]
fn a_ridge_hides_what_lies_behind_it_and_nothing_before_it() {
    let view = level_view();
    let half = benilla_formats::CHUNK_SIZE / 2.0;
    let ridge = flat_chunk(100.0 + benilla_formats::CHUNK_SIZE, half, 30.0);
    let behind = flat_chunk(300.0 + benilla_formats::CHUNK_SIZE, half, 0.0);
    let w = walk(
        &view,
        1000.0,
        None,
        [chunk(0, &ridge, 0), chunk(1, &behind, 0)].into_iter(),
        |_, _| true,
    );
    assert_eq!(w.stats.chunks, 2);
    assert_eq!(w.stats.chunks_hidden, 1);
    assert_eq!(w.stats.occluders, 1);
    let h = horizon_of(view, w);
    assert!(!h.hides_cell(&TerrainCell {
        tile: (0, 0),
        cell: 0
    }));
    assert!(h.hides_cell(&TerrainCell {
        tile: (0, 0),
        cell: 1
    }));
    // Behind and below the far edge's 30/133 slope.
    assert!(h.hides_sphere(Vec3::new(0.0, 10.0, -300.0), 2.0));
    // Behind, but above it, or off to the side.
    assert!(!h.hides_sphere(Vec3::new(0.0, 120.0, -300.0), 2.0));
    assert!(!h.hides_sphere(Vec3::new(200.0, 0.0, -300.0), 2.0));
    // In front of the ridge and below its line: an earlier band, which it has not reached.
    assert!(!h.hides_sphere(Vec3::new(0.0, 5.0, -60.0), 1.0));
    // The far tiles test against the whole buffer.
    assert!(h.hides_far_box(Vec3::new(-5.0, 0.0, -900.0), Vec3::new(5.0, 10.0, -890.0)));
}

#[test]
fn a_holed_chunk_draws_but_never_occludes() {
    let view = level_view();
    let half = benilla_formats::CHUNK_SIZE / 2.0;
    let ridge = flat_chunk(100.0 + benilla_formats::CHUNK_SIZE, half, 30.0);
    let behind = flat_chunk(300.0 + benilla_formats::CHUNK_SIZE, half, 0.0);
    let w = walk(
        &view,
        1000.0,
        None,
        [chunk(0, &ridge, 1), chunk(1, &behind, 0)].into_iter(),
        |_, _| true,
    );
    // Only the chunk behind, now in the open, occludes.
    assert_eq!(w.stats.occluders, 1);
    assert_eq!(w.stats.chunks_hidden, 0);
    assert!(w.hidden_cells.is_empty());
    // All holes: dropped before the walk.
    let w = walk(
        &view,
        1000.0,
        None,
        [chunk(0, &ridge, ALL_HOLES)].into_iter(),
        |_, _| true,
    );
    assert_eq!(w.stats.chunks, 0);
}

#[test]
fn an_occluder_must_lie_inside_the_feed() {
    let view = level_view();
    let half = benilla_formats::CHUNK_SIZE / 2.0;
    let ridge = flat_chunk(100.0 + benilla_formats::CHUNK_SIZE, half, 30.0);
    // A far end nearer than the ridge's nearest point.
    let w = walk(
        &view,
        50.0,
        None,
        [chunk(0, &ridge, 0)].into_iter(),
        |_, _| true,
    );
    assert_eq!(w.stats.occluders, 0);
}

/// The level `0x6afb06` gives a chunk: 3, or 2 only while `lod` is on and the depth is strictly
/// above `lodDist`; a NaN depth stays at 3.
#[test]
fn a_chunk_drops_to_level_2_only_past_lod_dist_with_lod_on() {
    assert_eq!(level_of(150.0, None), LEVEL_NEAR);
    assert_eq!(level_of(150.0, Some(100.0)), LEVEL_FAR);
    assert_eq!(level_of(100.0, Some(100.0)), LEVEL_NEAR);
    assert_eq!(level_of(f32::NAN, Some(100.0)), LEVEL_NEAR);
    // Level 3 takes all nine outer vertices, level 2 every second one (`0x810f60`, `0x810f70`).
    let picks = |l: usize| -> Vec<usize> { (0..ROW_COUNT[l]).map(|k| k * ROW_STRIDE[l]).collect() };
    assert_eq!(picks(LEVEL_NEAR), (0..9).collect::<Vec<_>>());
    assert_eq!(picks(LEVEL_FAR), [0, 2, 4, 6, 8]);
    assert!((0..4).all(|l| *picks(l).last().unwrap() == 8));
}

/// A far chunk whose far row rises only between its odd and even vertex 1 and 2: at level 3 that
/// edge raises the buffer; at level 2, past `lodDist`, the row skips vertex 1 and nothing rises.
#[test]
fn a_far_chunk_past_lod_dist_adds_its_rows_at_level_2() {
    let view = level_view();
    let half = benilla_formats::CHUNK_SIZE / 2.0;
    let mut far = flat_chunk(300.0 + benilla_formats::CHUNK_SIZE, half, 0.0);
    // Row 0 is the far row looking north (WoW x rises along −Z).
    far[1][2] = 40.0;
    far[2][2] = 40.0;
    let top = |lod: Option<f32>| {
        let w = walk(
            &view,
            1000.0,
            lod,
            [chunk(0, &far, 0)].into_iter(),
            |_, _| true,
        );
        assert_eq!(w.stats.occluders, 1);
        w.snapshots
            .last()
            .unwrap()
            .iter()
            .copied()
            .fold(f32::MIN, f32::max)
    };
    assert!(top(None) > 0.05, "level 3 adds the raised edge");
    assert!(top(Some(LOD_DIST_DEFAULT)) < 0.01, "level 2 steps over it");
    // Within `lodDist` the chunk is level 3 either way.
    assert!(top(Some(1000.0)) > 0.05);
}

/// Inside a building the feed starts at the window's reach plus the margin (`0x6811a9`): the ridge
/// short of it hides what lies behind but never occludes through the doorway.
#[test]
fn a_window_walk_takes_no_occluder_short_of_its_reach() {
    let view = level_view();
    let half = benilla_formats::CHUNK_SIZE / 2.0;
    let ridge = flat_chunk(100.0 + benilla_formats::CHUNK_SIZE, half, 30.0);
    let w = walk_from(
        &view,
        150.0 + OCCLUDER_FAR_MARGIN,
        1000.0,
        None,
        [chunk(0, &ridge, 0)].into_iter(),
        |_, _| true,
    );
    assert_eq!(w.stats.chunks, 1);
    assert_eq!(w.stats.occluders, 0);
    assert_eq!(w.shown_cells.len(), 1, "the ridge itself still draws");
}

/// An object is hidden only when every walk whose window admits it hides it.
#[test]
fn one_window_that_sees_an_object_shows_it() {
    let view = level_view();
    let wall = |slope: f32| WindowWalk {
        view,
        window: None,
        snapshots: vec![[slope; HORIZON_COLS]; BANDS + 1],
    };
    let low = Vec3::new(0.0, 0.0, -100.0);
    let h = |walks: Vec<WindowWalk>| Horizon {
        walks,
        enforce: true,
        ..Horizon::default()
    };
    assert!(h(vec![wall(0.1)]).hides_sphere(low, 1.0));
    assert!(!h(vec![wall(0.1), wall(RESET)]).hides_sphere(low, 1.0));
    // A window that does not reach the object leaves the verdict to the others.
    let clip_from_world = Projection::Perspective(PerspectiveProjection {
        fov: std::f32::consts::FRAC_PI_2,
        aspect_ratio: 1.0,
        near: 0.1,
        far: 1000.0,
        ..default()
    })
    .get_clip_from_view();
    let rect = [0.5, -1.0, 1.0, 1.0];
    let right = WindowWalk {
        window: Some(WindowVolume {
            rect,
            clip_from_world,
            frustum: crate::exterior_cull::window_frustum(rect, &clip_from_world),
        }),
        ..wall(RESET)
    };
    assert!(h(vec![wall(0.1), right]).hides_sphere(low, 1.0));
}

/// The drawn level reads the same knob as the horizon rows: strict above `lodDist`, NaN near, and
/// near everywhere with `lod` off.
#[test]
fn the_drawn_chunk_level_is_the_horizons() {
    let on = HorizonCulling {
        enforce: true,
        lod: true,
        lod_dist: LOD_DIST_SHIPPED,
    };
    assert_eq!(on.chunk_level(LOD_DIST_SHIPPED), LEVEL_NEAR as u8);
    assert_eq!(on.chunk_level(LOD_DIST_SHIPPED + 0.5), LEVEL_FAR as u8);
    assert_eq!(on.chunk_level(f32::NAN), LEVEL_NEAR as u8);
    let off = HorizonCulling { lod: false, ..on };
    assert_eq!(off.chunk_level(1.0e6), LEVEL_NEAR as u8);
    assert!(LOD_DIST_RANGE.contains(&LOD_DIST_SHIPPED));
}
