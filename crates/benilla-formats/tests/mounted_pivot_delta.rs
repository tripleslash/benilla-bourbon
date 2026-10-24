//! Pins the **mounted follow-camera correction** against real build-5875 assets — the third
//! consumer of the `M2Sequence` CAaBox the selection ring and the chat bubble already read.
//!
//! The reference's follow-camera preset (`0x50ca90`) is not the bind-pose attachment alone. It
//! reads attachment 17 for a base height, then corrects it by the difference between the box of
//! the animation the unit's POSTURE selects and the box of Stand:
//!
//! ```text
//! 0x50cba8  mov  eax, [unit->descriptors + 0x1fc]   ; UNIT_FIELD_MOUNTDISPLAYID
//! 0x50cbb9  jle  0x50cbc7                           ; not mounted -> the posture switch's pick
//! 0x50cbbb  mov  ebx, 0x5b                          ; mounted -> animation 91 = `Mount`
//! ...
//! 0x50cddf  cmp  ebx, 0x5b                          ; the selected animation IS Mount?
//! 0x50cde2  fld  [stand.max.z * scale]
//! 0x50cde5  fsub [anim.max.z  * scale]
//! 0x50cdeb  fld  [cam+0x11c] ; fsub st(1) ; fstp    ; posture preset -= (stand - anim)
//! 0x50cdf9  jne  0x50ce09
//! 0x50cdfb  fld  [cam+0x120] ; fsub st(1) ; fstp    ; …and, for Mount only, the standing one too
//! ```
//!
//! `0x5b` is `AnimationData.dbc` id 91, whose name in the shipped table is literally `Mount`; the
//! posture ids the switch above it picks are the seated/dead family (97 `SitGround`, 100 `Sleep`,
//! 102-104 `SitChairLow/Med/High`, 6 `Dead`, 115 `KneelLoop`), and `0x2a` = 42 `Swim` feeds the
//! third preset by the same subtraction. So the mount arm is one row of a general mechanism, and
//! `M2Bounds::mounted_pivot_delta` is that row read off the file.
//!
//! **Every value is negative, and that is the correction working**, not the camera sinking: the
//! Mount pose hangs the legs below the model origin and drops the neck with them, while the model
//! origin itself moves up to the mount's attachment-0 saddle. Pinning the sign is the point — a
//! reader that took the box's Z *extent* (what `stand_box_z` is, for the bubble) instead of its
//! `max.z` would answer +0.006 for a gnome where the reference answers −0.212, and the two are
//! indistinguishable on the one race somebody would eyeball.
//!
//! Skips when the gitignored client data isn't present.

use benilla_formats::{load_m2_bounds, open_chain};

#[test]
fn the_mounted_pivot_delta_is_the_mount_box_measured_against_stand() {
    let data = benilla_formats::wow_data_or_skip!();
    let mut chain = open_chain(&data).expect("open chain");

    // (model, Stand.max.z, Mount.max.z) — all three read straight out of the shipped M2s, through
    // `animationLookup[id]` rather than record order. Four races, deliberately: the spread runs
    // −0.21 to −1.19, so a constant fudge cannot pass here.
    let cases = [
        (
            "Character\\Human\\Male\\HumanMale.mdx",
            2.01092_f32,
            1.05913,
        ),
        (
            "Character\\Gnome\\Female\\GnomeFemale.mdx",
            1.14910,
            0.93701,
        ),
        ("Character\\Tauren\\Male\\TaurenMale.mdx", 2.24853, 1.56029),
        (
            "Character\\NightElf\\Female\\NightElfFemale.mdx",
            2.25393,
            1.06744,
        ),
    ];

    for (path, stand_z, mount_z) in cases {
        let b = load_m2_bounds(&mut chain, path).unwrap_or_else(|e| panic!("bounds {path}: {e}"));
        let want = mount_z - stand_z;
        assert!(
            (b.mounted_pivot_delta - want).abs() < 0.001,
            "{path}: mounted_pivot_delta {:.5} should be Mount.max.z − Stand.max.z = {want:.5}",
            b.mounted_pivot_delta
        );
        assert!(
            b.mounted_pivot_delta < 0.0,
            "{path}: the Mount pose sits BELOW the standing head in the model's own frame — a \
             positive answer means the seat height has been folded in where it does not belong"
        );
        // Not the Z *extent*: that is the bubble's number, off the same box, and on a gnome the
        // two even disagree in sign.
        let extent_delta = b.stand_box_z; // Stand's own extent, the value the bubble anchors on
        assert!(
            (b.mounted_pivot_delta - extent_delta).abs() > 0.02,
            "{path}: the delta must be read off max.z, never the box extent"
        );
    }

    // A model that authors no `Mount` sequence contributes nothing — `animationLookup[91]` misses,
    // and the reader must answer 0 rather than measuring whatever record 0 happens to be. A
    // chicken's record 0 is a flap; reading it as a mount pose would move the camera on a body
    // that can never be ridden.
    let chicken = load_m2_bounds(&mut chain, "Creature\\Chicken\\Chicken.mdx").expect("chicken");
    assert_eq!(
        chicken.mounted_pivot_delta, 0.0,
        "no Mount sequence → no correction"
    );
}

/// **The saddle is the mount model's authored attachment 0** — the other half of the mounted
/// preset, and the half whose *staticness* is the point.
///
/// `0x50ce1c`-`0x50ce56` reads it with `hasAttachment(mount, 0)` / `getAttachment(mount, &out, 0)`,
/// which is `0x712cf0` — the same call that answers the rider's own pivot, and one with no bone
/// matrix anywhere in its call tree. Reading the posed joint instead made the camera creep, wobble
/// with the idle and lift on a MountSpecial (reported 2026-09-09); this pins the source, not just
/// the number.
///
/// Resolved through `M2Model::attachment`, the reference's own `0x710310` — the AttachLookup with
/// its bounds check and `0xffff` sentinel — never a scan of the record table, whose ids repeat.
#[test]
fn the_mount_seat_is_attachment_zero_off_the_mount_model() {
    let data = benilla_formats::wow_data_or_skip!();
    let mut chain = open_chain(&data).expect("open chain");

    // The Black Qiraji Resonating Crystal's mount, and an ordinary riding horse for contrast —
    // both authored within 5 cm of each other, which is what a saddle height looks like.
    for (path, seat) in [
        (r"Creature\RidingSilithid\RidingSilithid.mdx", 1.817_f32),
        (r"Creature\RidingHorse\RidingHorse.mdx", 1.866),
    ] {
        let b = load_m2_bounds(&mut chain, path).unwrap_or_else(|e| panic!("bounds {path}: {e}"));
        assert!(
            (b.mount_seat_z - seat).abs() < 0.002,
            "{path}: mount_seat_z {:.4} should be attachment 0's model-space Z {seat}",
            b.mount_seat_z
        );
    }

    // **Slot 0 is overloaded, and this is the trap the reader has to be pointed away from.** The
    // same id is the SHIELD anchor on a character — a human male authors it at 1.036, a gnome
    // female at 0.385 — so "authors attachment 0" is not "is a mount", and a reader that took this
    // field off the RIDER would hang every unmounted body 0.4-1.0 yd of phantom saddle.
    //
    // The reference reads it off the mount model specifically (`0x50ce1c` takes `[unit+0xdc]`, the
    // model it picked at the top, which is the mount when the unit is mounted), and
    // `player::mirror_mount_seat_lift` reads it off the MOUNT CHILD entity for the same reason.
    for (path, shield) in [
        (r"Character\Human\Male\HumanMale.mdx", 1.036_f32),
        (r"Character\Gnome\Female\GnomeFemale.mdx", 0.385),
    ] {
        let b = load_m2_bounds(&mut chain, path).expect("character bounds");
        assert!(
            (b.mount_seat_z - shield).abs() < 0.002,
            "{path}: slot 0 is the shield anchor here, not a saddle — got {}",
            b.mount_seat_z
        );
    }
}
