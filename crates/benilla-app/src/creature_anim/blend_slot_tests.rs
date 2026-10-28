use super::*;

/// A looping clip on track `node`, its blend slot's twin on `node + 10`, fading in over 0.25 s.
fn clip(anim_id: u16, node: usize, twin: bool) -> AnimClip {
    AnimClip {
        anim_id,
        seq_index: 0,
        node: AnimationNodeIndex::new(node),
        blend_node: twin.then(|| AnimationNodeIndex::new(node + 10)),
        looping: true,
        duration: 1.0,
        move_speed: 0.0,
        blend_time: 0.25,
        bounds_center: Vec3::ZERO,
        bounds_radius: 0.0,
        bounds_min: Vec3::ZERO,
        bounds_max: Vec3::ZERO,
        events: Vec::new().into(),
        arm_nodes: None,
        upper_node: None,
        frequency: 0,
        replay: (0, 0),
        poses_bones: true,
    }
}

fn node(n: usize) -> AnimationNodeIndex {
    AnimationNodeIndex::new(n)
}

fn weight(player: &AnimationPlayer, n: usize) -> Option<f32> {
    player.animation(node(n)).map(|a| a.weight())
}

/// One frame at `now`: the arms, then the kernel's advance.
fn frame(slot: &mut BlendSlot, player: &mut AnimationPlayer, now: f64, arms: &[(&AnimClip, bool)]) {
    slot.set_clock(now);
    for &(c, blend) in arms {
        slot.arm(player, c, blend);
    }
    slot.advance(player);
}

/// Stand armed at t = 0 with nothing before it, then Walk blended at t = 1.
fn stand_then_walk() -> (BlendSlot, AnimationPlayer, AnimClip, AnimClip) {
    let (stand, walk) = (clip(0, 1, true), clip(4, 2, true));
    let mut slot = BlendSlot::default();
    let mut player = AnimationPlayer::default();
    frame(&mut slot, &mut player, 0.0, &[(&stand, true)]);
    frame(&mut slot, &mut player, 1.0, &[(&walk, true)]);
    (slot, player, stand, walk)
}

/// The old pose's weight is the kernel's smoothstep `(3 − 2x)·x²` of the fade left (`0x714880`),
/// not a linear ramp: 0.84 a quarter in and 0.16 three quarters in, against 0.75 and 0.25.
#[test]
fn the_old_pose_fades_by_the_kernels_smoothstep() {
    let (mut slot, mut player, ..) = stand_then_walk();
    assert_eq!(
        slot.fade(),
        Some(Fade {
            node: node(1),
            until: 1.25,
            span: 0.25,
        }),
        "Stand copied over Walk's blendTime (`0x7125d6`, `0x7125f2`)"
    );
    assert_eq!(weight(&player, 1), Some(1.0), "λ = 1 on the arm's frame");
    assert_eq!(weight(&player, 2), Some(0.0));
    for (now, old) in [(1.0625, 0.843_75), (1.125, 0.5), (1.1875, 0.156_25)] {
        frame(&mut slot, &mut player, now, &[]);
        let (out, inc) = (weight(&player, 1).unwrap(), weight(&player, 2).unwrap());
        assert!((out - old).abs() < 1e-4, "λ at {now}: {out}, want {old}");
        assert!((out + inc - 1.0).abs() < 1e-6, "the primary takes 1 − λ");
    }
}

/// At the fade's end (`0x7147b1`) the secondary alone stops and the primary is whole.
#[test]
fn the_slot_ends_on_its_own() {
    let (mut slot, mut player, ..) = stand_then_walk();
    frame(&mut slot, &mut player, 1.25, &[]);
    assert_eq!(slot.fade(), None);
    assert_eq!(weight(&player, 1), None, "Stand stopped");
    assert_eq!(weight(&player, 2), Some(1.0), "Walk runs whole");
}

/// An arm while the fade still weighs over half (`0x7125c9`/`0x7125d4`) writes the primary alone:
/// the secondary keeps its pose, clock and window, and the replaced primary is dropped, so no
/// second fade stacks.
#[test]
fn an_arm_inside_the_first_half_of_a_fade_is_refused_the_slot() {
    let (mut slot, mut player, _, _) = stand_then_walk();
    player.play(node(1)).seek_to(0.4);
    let run = clip(5, 3, true);
    frame(&mut slot, &mut player, 1.05, &[(&run, true)]);
    assert_eq!(
        slot.fade().map(|f| (f.node, f.until)),
        Some((node(1), 1.25)),
        "Stand still fades, on its own window"
    );
    assert_eq!(slot.primary(), Some(node(3)));
    assert_eq!(weight(&player, 2), None, "the refused copy drops Walk");
    assert_eq!(
        player.animation(node(1)).map(|a| a.seek_time()),
        Some(0.4),
        "the secondary is not seeked back"
    );
    assert_eq!(player.playing_animations().count(), 2, "one slot, no stack");
}

/// Past half (λ ≤ 0.5) the arm copies the primary over the secondary, which is stopped.
#[test]
fn an_arm_past_half_replaces_the_secondary() {
    let (mut slot, mut player, ..) = stand_then_walk();
    let run = clip(5, 3, true);
    frame(&mut slot, &mut player, 1.15, &[(&run, true)]);
    assert_eq!(
        slot.fade().map(|f| f.node),
        Some(node(2)),
        "Walk is the secondary now"
    );
    assert_eq!(weight(&player, 1), None, "Stand is stopped");
    assert_eq!(player.playing_animations().count(), 2);
}

/// A blended re-arm of the sequence the primary plays fades from the old clock: the outgoing play
/// moves to the secondary as it stands, and the new one starts at 0 on the clip's other track.
#[test]
fn a_same_sequence_rearm_fades_from_the_old_clock() {
    let (mut slot, mut player, _, walk) = stand_then_walk();
    frame(&mut slot, &mut player, 1.5, &[]);
    player.play(node(2)).seek_to(0.7);
    frame(&mut slot, &mut player, 1.5, &[(&walk, true)]);
    assert_eq!(slot.fade().map(|f| f.node), Some(node(2)));
    assert_eq!(slot.primary(), Some(node(12)), "the twin track");
    assert_eq!(
        player
            .animation(node(2))
            .map(|a| (a.seek_time(), a.weight())),
        Some((0.7, 1.0)),
        "the old play keeps its clock and, at λ = 1, the pose"
    );
    assert_eq!(
        player
            .animation(node(12))
            .map(|a| (a.seek_time(), a.weight())),
        Some((0.0, 0.0))
    );
    // Re-armed again past half, the play swaps back to the clip's first track.
    frame(&mut slot, &mut player, 1.65, &[(&walk, true)]);
    assert_eq!(slot.fade().map(|f| f.node), Some(node(12)));
    assert_eq!(slot.primary(), Some(node(2)));
}

/// A model with one track per clip cannot hold a sequence twice: re-armed while the refused
/// secondary still holds it, the arm takes the track and the fade ends.
#[test]
fn a_rearm_of_the_secondarys_sequence_without_a_twin_ends_the_fade() {
    let (stand, walk) = (clip(0, 1, false), clip(4, 2, false));
    let mut slot = BlendSlot::default();
    let mut player = AnimationPlayer::default();
    frame(&mut slot, &mut player, 0.0, &[(&stand, true)]);
    frame(&mut slot, &mut player, 1.0, &[(&walk, true)]);
    frame(&mut slot, &mut player, 1.05, &[(&stand, true)]);
    assert_eq!(slot.fade(), None);
    assert_eq!(slot.primary(), Some(node(1)));
    assert_eq!(weight(&player, 1), Some(1.0));
    assert_eq!(weight(&player, 2), None);
}

/// The shortcut (`0x712553` → `0x71262d`): the latched primary's own record re-armed is written
/// as the primary alone, restarting it with no fade.
#[test]
fn a_latched_same_record_rearm_writes_the_primary_alone() {
    let (mut slot, mut player, _, walk) = stand_then_walk();
    frame(&mut slot, &mut player, 2.0, &[]);
    player.play(node(2)).seek_to(0.9);
    slot.latch();
    frame(&mut slot, &mut player, 2.0, &[(&walk, true)]);
    assert_eq!(slot.fade(), None, "nothing copied");
    assert_eq!(slot.primary(), Some(node(2)), "on its own track");
    assert_eq!(
        player
            .animation(node(2))
            .map(|a| (a.seek_time(), a.weight())),
        Some((0.0, 1.0)),
        "restarted, whole"
    );
    // The arm clears the latch: the next re-arm blends again.
    frame(&mut slot, &mut player, 2.5, &[(&walk, true)]);
    assert_eq!(slot.fade().map(|f| f.node), Some(node(2)));
}

/// `blend` 0 clears the secondary (`0x712635`): the arm cuts.
#[test]
fn an_unblended_arm_clears_the_slot() {
    let (mut slot, mut player, ..) = stand_then_walk();
    let stand = clip(0, 1, true);
    frame(&mut slot, &mut player, 1.05, &[(&stand, false)]);
    assert_eq!(slot.fade(), None);
    assert_eq!(slot.primary(), Some(node(1)));
    assert_eq!(weight(&player, 1), Some(1.0));
    assert_eq!(weight(&player, 2), None);
}
