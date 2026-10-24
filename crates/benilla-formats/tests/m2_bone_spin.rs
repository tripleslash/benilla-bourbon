//! The rigid bone spin collector (`m2_bone_spins`) and its sampler, which turn Caverns of Time's
//! asteroid belts without a skinning palette.

use benilla_formats::{load_m2_bone_spins, m2_bone_spins, BoneSpin, Chain};

fn about_z(deg: f32) -> [f32; 4] {
    let h = deg.to_radians() * 0.5;
    [0.0, 0.0, h.sin(), h.cos()]
}

/// A quaternion's angle about its axis in degrees, the same for `q` and `-q`.
fn angle_deg(q: [f32; 4]) -> f32 {
    let w = q[3].abs().clamp(0.0, 1.0);
    2.0 * w.acos().to_degrees()
}

fn spin(keys: Vec<(f32, [f32; 4])>, duration: f32, interp: bool) -> BoneSpin {
    BoneSpin {
        pivot: [0.0; 3],
        duration,
        interp,
        keys,
    }
}

/// Past the last key the sampler clamps: the reference interpolates only within the sequence band,
/// so a loop whose last key differs from its first snaps at the wrap.
#[test]
fn the_sampler_holds_slerps_clamps_and_wraps() {
    let s = spin(
        vec![(0.0, about_z(0.0)), (2.0, about_z(90.0))],
        4.0,
        /*interp=*/ true,
    );
    assert!(angle_deg(s.sample(0.0)) < 1e-3, "at the first key");
    assert!(
        (angle_deg(s.sample(-1.0)) - 90.0).abs() < 1e-3,
        "a negative cursor lands 1 s before the loop start, i.e. 3 s into the previous cycle — \
         `rem_euclid`, not `%`, which would mirror it onto the loop's opening pose instead"
    );
    assert!(
        (angle_deg(s.sample(1.0)) - 45.0).abs() < 1e-2,
        "halfway through the bracket: {}",
        angle_deg(s.sample(1.0))
    );
    assert!(
        (angle_deg(s.sample(3.0)) - 90.0).abs() < 1e-3,
        "past the last key: clamped, NOT interpolated back toward key 0"
    );
    assert!(
        (angle_deg(s.sample(5.0)) - 45.0).abs() < 1e-2,
        "one full period on from t=1: the loop wrapped"
    );
}

/// A step track (`interp_type == 0`) holds each key until the next.
#[test]
fn a_step_track_holds_its_key() {
    let s = spin(
        vec![(0.0, about_z(0.0)), (2.0, about_z(90.0))],
        4.0,
        /*interp=*/ false,
    );
    assert!(angle_deg(s.sample(1.9)) < 1e-3, "still on key 0 at 1.9s");
    assert!((angle_deg(s.sample(2.0)) - 90.0).abs() < 1e-3);
}

/// Keys whose quaternions dot negative slerp the short way round.
#[test]
fn the_slerp_takes_the_short_way_round() {
    // 350° and 0° are 10° apart, but their quaternions dot negative.
    let s = spin(
        vec![(0.0, about_z(350.0)), (1.0, about_z(360.0))],
        1.0,
        true,
    );
    let mid = angle_deg(s.sample(0.5));
    assert!(
        !(15.0..=345.0).contains(&mid),
        "midpoint took the long arc: {mid}°"
    );
}

/// `CavernsOfTimeSky.m2` spins three bones, the asteroid belts, over one 66.667 s loop (25°, 90°,
/// 360°); bone 0, which carries the other 17 batches, has no track.
#[test]
fn the_caverns_of_time_sky_spins_exactly_its_three_belt_bones() {
    let data = benilla_formats::wow_data_or_skip!();
    let mut chain = Chain::open(&data).expect("open vanilla patch chain");
    let spins = load_m2_bone_spins(&mut chain, "Environments\\Stars\\CavernsOfTimeSky.m2")
        .expect("read CavernsOfTimeSky.m2");

    let mut bones: Vec<u16> = spins.keys().copied().collect();
    bones.sort_unstable();
    assert_eq!(
        bones,
        vec![1, 2, 3],
        "the three asteroid-belt bones, and only those"
    );

    for (bone, turn) in [(1u16, 25.0f32), (2, 90.0), (3, 360.0)] {
        let s = &spins[&bone];
        assert!(
            (s.duration - 66.667).abs() < 0.01,
            "bone {bone}: one 66.667 s loop, got {}",
            s.duration
        );
        assert!(
            s.interp,
            "bone {bone}: the belts interpolate, they don't step"
        );
        assert!(
            angle_deg(s.keys[0].1) < 1e-2,
            "bone {bone}: the loop opens unrotated"
        );
        // The total turn at the last key; 360° reads back as 0°, the identity.
        let last = angle_deg(s.keys.last().expect("keyed").1);
        let expect = if turn >= 360.0 { 0.0 } else { turn };
        assert!(
            (last - expect).abs() < 0.5,
            "bone {bone}: expected {expect}° at the final key, got {last}°"
        );
        // The pivot sits yards off the origin, where the eye is, so the spin conjugates by it.
        let d = (s.pivot[0].powi(2) + s.pivot[1].powi(2) + s.pivot[2].powi(2)).sqrt();
        assert!(
            (2.0..5.0).contains(&d),
            "bone {bone}: pivot {d:.2} yd from the origin — the conjugation matters"
        );
    }
}

/// `DeathClouds.m2`, the ghost sky, turns its one bone on a 133.267 s global sequence, not on
/// anim 0 (`benilla-extract m2anim`).
#[test]
fn the_ghost_sky_spins_its_bone_on_the_global_sequence() {
    let data = benilla_formats::wow_data_or_skip!();
    let mut chain = Chain::open(&data).expect("open vanilla patch chain");
    let spins = load_m2_bone_spins(&mut chain, "Environments\\Stars\\DeathClouds.m2")
        .expect("read DeathClouds.m2");
    let s = spins.get(&0).expect("bone 0 spins");
    assert_eq!(spins.len(), 1, "its only bone");
    assert!(
        (s.duration - 133.267).abs() < 0.01,
        "one 133.267 s period, got {}",
        s.duration
    );
    // The turn from the first key keeps moving across the period.
    let from_start = |t: f32| {
        let (a, b) = (s.keys[0].1, s.sample(t));
        let dot: f32 = (0..4).map(|i| a[i] * b[i]).sum();
        2.0 * dot.abs().clamp(0.0, 1.0).acos().to_degrees()
    };
    let (early, later) = (from_start(s.duration * 0.05), from_start(s.duration * 0.25));
    assert!(early > 1.0, "already turning 5% in: {early}°");
    assert!(
        later > early,
        "and further on at 25%: {early}° then {later}°"
    );
}

/// `StratholmeSkybox` is three static opaque batches: nothing spins.
#[test]
fn the_stratholme_sky_spins_nothing() {
    let data = benilla_formats::wow_data_or_skip!();
    let mut chain = Chain::open(&data).expect("open vanilla patch chain");
    let bytes = chain
        .read_file("Environments\\Stars\\StratholmeSkybox.m2")
        .expect("read StratholmeSkybox.m2");
    assert!(
        m2_bone_spins(&bytes).is_empty(),
        "Stratholme's sky authors no bone animation"
    );
}
