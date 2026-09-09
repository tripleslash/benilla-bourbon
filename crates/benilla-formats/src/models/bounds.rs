//! An M2's authored bounds ([`M2Bounds`]) — the header sphere/box, the vertex-derived radius our
//! renderer computes, and the selection-ring **footprint** read from the model's Stand-animation
//! bounding box (the exact model-local input the real client's living-unit selection ring uses).
//! No render-batch concern here — just the bounds/measurement reads [`load_m2_bounds`]/
//! [`parse_m2_bounds`] expose to placement + the selection ring.

use std::io::Cursor;

use anyhow::{Context, Result};
use benilla_bytes::ByteExt;
use benilla_m2::parse_m2;

use crate::Chain;

use super::model_path;

/// An M2's authored bounds (read straight from the model header) alongside the vertex-derived radius
/// our renderer currently computes — so the distance-fade size bucket can be checked against the value
/// the reference's placement builder actually uses (`FUN_00694bc0` → `rec+0x68`). `sphere_radius` and
/// `bbox_*` are in model-local yards (multiply by the placement scale for the world radius).
#[derive(Clone, Copy, Debug)]
pub struct M2Bounds {
    /// `M2Header.bounding_sphere_radius` — the authored bounding-sphere radius (the reference's source).
    pub sphere_radius: f32,
    /// `M2Header.bounding_box_min/max` — the authored AABB; its centre is the sphere centre.
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
    /// Max vertex distance from the model **origin** — what `build_model` currently uses as the radius.
    pub vert_max_from_origin: f32,
    /// The selection-ring **footprint** (model-local, pre-scale): `sqrt(0.5 · sqrt(dx² + dy²))`, where
    /// `dx,dy` are the horizontal (X,Y) extents of the unit's **Stand** animation bounding box (the M2
    /// sequence CAaBox). This is the exact model-local input the real client's living-unit selection ring
    /// uses — byte-verified + Unicorn-emulated to the reference pixels (wow-re selection-ring RE,
    /// `0x608e00`/`0x60aee0`): the ring's world radius = this × `OBJECT_FIELD_SCALE_X`. The nested sqrt is
    /// a range-compressor (which is why the render-sphere never fit). Falls back to the header render-box
    /// XY extents for a model with no animation sequences.
    ///
    /// A **degenerate** box — `max.x == min.x` *and* `max.y == min.y` — takes
    /// [`DEGENERATE_RING_FOOTPRINT`] instead of the formula, which is the writer's own first branch
    /// and not a floor of ours (decision 1658).
    pub ring_footprint: f32,
    /// Model-space **Z** of attachment id 17 — the reference's follow-camera pivot height (wow-re
    /// `follow-camera`: `feet + (attach17.z + 0.0972)·scale`). `None` for a model with no slot-17
    /// attachment; the camera then falls back to a fraction of the box. See [`M2Bounds::pivot_z`].
    pub pivot_z: Option<f32>,
    /// The **Stand** animation box's vertical extent, `max.z − min.z` (model-local, pre-scale) — the
    /// chat bubble's anchor height above the unit's feet (× model scale, + 0.7 yd).
    ///
    /// The same `0x711a20` query the selection ring above is sized from, taking the box's Z instead
    /// of its XY: wow-re's chat-bubble anchor cross-check (2026-08-17) followed `0x4b0e38 call
    /// 0x711a20` into the model layer and found it reading the **MD20 header image** — file bytes,
    /// no bone matrix anywhere in the call tree — and returning `out+0x20 − out+0x14`, which is that
    /// CAaBox's Z extent. Benilla anchored the bubble on the posed PlayerName attachment instead,
    /// on a recorded INFERRED claim that `0x608640` and `0x711a20` were "both the head-region
    /// attachment height"; that equivalence is **REFUTED** — they differ precisely on
    /// animated-vs-static, `0x608640` reading the live posed palette (it bobs) and this reading a
    /// file constant. The overhead *name* keeps `0x608640`; only the bubble takes this. See 1406.
    ///
    /// Falls back to the header render-box Z extent for a model with no sequences, like the ring.
    pub stand_box_z: f32,
    /// The **mounted** correction to the follow-camera pivot (model-local, pre-scale):
    /// `Mount.max.z − Stand.max.z` off the same sequence boxes, `0.0` for a model authoring no
    /// Mount sequence.
    ///
    /// The reference's follow-camera preset is not the bind-pose attachment alone — `0x50ca90`
    /// corrects [`Self::pivot_z`] by the difference between the box of the animation the unit's
    /// POSTURE selects and the box of Stand, all in the model's own frame. Its switch picks
    /// SitGround / Sleep / SitChairLow-Med-High / Dead / KneelLoop from the posture word, and then
    /// one branch overrides all of them: `0x50cba8` reads `UNIT_FIELD_MOUNTDISPLAYID` and, when it
    /// is non-zero, forces animation **91 (`Mount`)**. The correction itself is `0x50cddf`-
    /// `0x50ce09`: `preset -= (stand.max.z − anim.max.z) × scale`, applied to the posture preset
    /// and — under the `cmp ebx, 0x5b` that singles Mount out — to the standing preset as well.
    ///
    /// **It is negative on every character** (human −0.95, gnome −0.21, tauren −0.69, night elf
    /// −1.19), because the Mount pose hangs the legs below the model origin and drops the neck
    /// with them. That is not the camera going down: while mounted the character model's origin is
    /// the mount's attachment-0 **saddle**, so the world pivot is `saddle + (pivot_z + this)·scale`
    /// and the net move is up by the seat height less this. Adding the seat without this
    /// correction rides a head too high; applying this without the seat sinks the camera into the
    /// mount.
    ///
    /// Only the Mount arm is read here. The seated postures need the unit's posture word, which is
    /// not a model fact, and their preset would belong beside it.
    pub mounted_pivot_delta: f32,
    /// **Attachment id 0's model-space Z**, in model-local yards, pre-scale — where a rider sits,
    /// *on a mount model*. `0.0` for a model that authors no slot-0 attachment.
    ///
    /// **Slot 0 is overloaded and this field is only meaningful on a mount.** The same id is the
    /// SHIELD anchor on a character (a human male authors it at z 1.036, a gnome female at 0.385),
    /// so "does this model author attachment 0" is not "is this a mount". That is why the reference
    /// reads it off the mount model specifically — `0x50ce1c` loads `[ebp-8]`, the model the
    /// function picked at its top, which is `[unit+0xdc]` when the unit is mounted — and why
    /// benilla reads it off the MOUNT CHILD entity and never off the rider's own.
    ///
    /// The other half of the mounted camera preset, and the reason it is a FILE CONSTANT rather
    /// than a posed bone. `0x50ce1c`-`0x50ce56` reads it off the **mount's** model —
    /// `hasAttachment(mount, 0)` then `getAttachment(mount, &out, 0)`, the same `0x712cf0` that
    /// answers [`Self::pivot_z`] — scales it, and `0x50cf77`-`0x50cf9e` adds it to all three
    /// presets. `0x712cf0` returns the record's authored position; no bone matrix appears anywhere
    /// in that call tree.
    ///
    /// Reading the **live seated** position instead is a defect with three faces, all reported
    /// against the hosted page on 2026-09-09: the camera creeps up over seconds instead of
    /// rising once (the target moves every frame, so the pivot glide re-arms every frame and
    /// crawls toward it), it then wobbles with the mount's idle animation, and a MountSpecial
    /// flourish lifts it further. The reference has none of those because the preset is computed
    /// when the model event lands and is a constant until the next one.
    pub mount_seat_z: f32,
}

/// Read an M2's authored bounds + the vertex-derived radius (see [`M2Bounds`]). Uses the same path
/// normalisation as [`super::load_m2_mesh`] so `.mdx`/`.mdl` doodad paths resolve identically.
pub fn load_m2_bounds(chain: &mut Chain, raw_path: &str) -> Result<M2Bounds> {
    let path = model_path(raw_path);
    let bytes = chain
        .read_file(&path)
        .with_context(|| format!("reading M2 {path}"))?;
    parse_m2_bounds(&bytes).with_context(|| format!("parsing M2 bounds {path}"))
}

/// The ring footprint the real client stores for a **degenerate** box — one whose X *and* Y extents
/// are both exactly zero. Byte-read at `0x60af4f..0x60af67` (wow-re `selection-ring-scale.md`): the
/// writer `0x60aee0` compares `max.x==min.x` and `max.y==min.y` and, when both hold, stores the
/// literal `0x3f99999a` = **1.2** into `[unit+0xcf0]` without ever running the `sqrt(0.5·sqrt(dx²+dy²))`
/// formula. wow-re recorded it as a branch that "never fires for real creatures", which is true of
/// the four life-size units it measured and false of the whole trigger-creature family: an
/// `InvisibleStalker` body authors all 135 sequence boxes at zero, so this **is** its ring — and the
/// Naxxramas weapon mobs, whose visible self is the axe in that body's hand, are exactly where a
/// player sees it. Ours read 0 and drew a ring the width of a coin (decision 1658).
///
/// It is the model-less fallback too: "no box to measure" and "a box that measures zero" are the
/// same question, and this is the reference's answer to it.
pub const DEGENERATE_RING_FOOTPRINT: f32 = 1.2;

/// Read the horizontal (X,Y) extents of the **Stand** animation's bounding box from a raw M2 — the input
/// the real client's living-unit selection ring is sized from (wow-re selection-ring RE, `0x60aee0`).
/// The animation `M2Sequence` array is at MD20 `0x1c`(count)/`0x20`(offset), stride `0x44`, with the
/// sequence's `CAaBox` at record `+0x24` (min C3 `+0x24`, max C3 `+0x30`). Stand is animation **id 0**,
/// whose *sequence index* is `animationLookup[0]` (array at `0x24`/`0x28`, `u16` each) — NOT necessarily
/// record 0 (a chicken's record 0 is a flap, its Stand sits at index 2). Returns `(dx, dy, dz)`, or
/// `None` when the model has no sequences / lookup or the indices are out of range (caller falls back
/// to the header box). VERIFIED: reproduces the reference ring radii to ~1 mm.
///
/// The Z extent joins the pair the ring needs because the chat bubble's anchor is the **same box**
/// read on the other axis (1406) — one parse, one Stand-sequence resolution, two consumers, so the
/// ring and the bubble can never drift onto different animations.
fn stand_box(bytes: &[u8]) -> Option<(f32, f32, f32)> {
    let rec = sequence_record(bytes, STAND_ANIM, true)?;
    // CAaBox @ rec+0x24: min C3 (+0x24), max C3 (+0x30). Horizontal (X,Y) extents; squared in the ring
    // formula so an unsorted box is harmless.
    let dx = bytes.f32_at(rec + 0x30)? - bytes.f32_at(rec + 0x24)?;
    let dy = bytes.f32_at(rec + 0x34)? - bytes.f32_at(rec + 0x28)?;
    let dz = bytes.f32_at(rec + 0x38)? - bytes.f32_at(rec + 0x2c)?;
    Some((dx, dy, dz))
}

/// `AnimationData.dbc` id **0** — Stand, the box every other sequence's box is read against.
const STAND_ANIM: usize = 0;
/// `AnimationData.dbc` id **91** — `Mount`, the pose `0x50cbbb` forces on a unit whose
/// `UNIT_FIELD_MOUNTDISPLAYID` is non-zero (`mov ebx, 0x5b`). Named against the shipped DBC.
const MOUNT_ANIM: usize = 91;

/// The file offset of one `M2Sequence` record, resolved the way the reference resolves an animation
/// id: through `animationLookup` (array at MD20 `0x24`/`0x28`, `u16` each), **not** by record order.
///
/// `fall_back_to_first` is the difference between the two callers. The ring/bubble ask for Stand,
/// which every model has, and take record 0 when the lookup cannot answer — the reference's own
/// static-model leg. The mount correction asks for a sequence most models genuinely lack, and
/// there "no Mount" must stay `None` rather than silently measuring whatever record 0 is: reading a
/// chicken's flap as its mount pose would move the camera on a model that can never be ridden.
fn sequence_record(bytes: &[u8], anim: usize, fall_back_to_first: bool) -> Option<usize> {
    let anim_count = bytes.u32_at(0x1c)? as usize;
    let anim_ofs = bytes.u32_at(0x20)? as usize;
    let lookup_count = bytes.u32_at(0x24)? as usize;
    let lookup_ofs = bytes.u32_at(0x28)? as usize;
    if anim_count == 0 {
        return None;
    }
    let looked_up = (anim < lookup_count)
        .then(|| bytes.u16_at(lookup_ofs + anim * 2))
        .flatten()
        .map(usize::from)
        .filter(|&i| i != 0xffff && i < anim_count);
    let idx = match looked_up {
        Some(i) => i,
        None if fall_back_to_first => 0,
        None => return None,
    };
    anim_ofs.checked_add(idx * 0x44)
}

/// `Mount.max.z − Stand.max.z` — the reference's mounted pivot correction (see
/// [`M2Bounds::mounted_pivot_delta`]). `None` for a model with no sequences or no `Mount`.
fn mounted_pivot_delta(bytes: &[u8]) -> Option<f32> {
    let stand = sequence_record(bytes, STAND_ANIM, true)?;
    let mount = sequence_record(bytes, MOUNT_ANIM, false)?;
    // `max.z` alone, not the extent: the preset is a height above the model origin, and the
    // reference reads `out+0x20` (the box's max C3's Z) on both sides of its subtraction.
    Some(bytes.f32_at(mount + 0x38)? - bytes.f32_at(stand + 0x38)?)
}

/// Read an **in-memory** M2's authored bounds + vertex-derived radius (see [`M2Bounds`]). The bytes-in
/// entry point for the Bevy `AssetLoader`; [`load_m2_bounds`] is the chain-reading wrapper.
pub fn parse_m2_bounds(bytes: &[u8]) -> Result<M2Bounds> {
    let format =
        parse_m2(&mut Cursor::new(bytes)).map_err(|e| anyhow::anyhow!("parsing M2: {e}"))?;
    let model = format.model();
    let h = &model.header;
    let vert_max_from_origin = model
        .vertices
        .iter()
        .map(|v| {
            let p = v.position;
            (p.x * p.x + p.y * p.y + p.z * p.z).sqrt()
        })
        .fold(0.0_f32, f32::max);
    // Selection-ring footprint: `sqrt(0.5 · sqrt(dx² + dy²))` of the Stand animation box's horizontal
    // extents (the real client's living-unit ring input), falling back to the header render-box XY for a
    // model with no animation sequences (the client's static-model path).
    let (rx, ry, rz) = stand_box(bytes).unwrap_or((
        h.bounding_box_max[0] - h.bounding_box_min[0],
        h.bounding_box_max[1] - h.bounding_box_min[1],
        h.bounding_box_max[2] - h.bounding_box_min[2],
    ));
    let ring_footprint = if rx == 0.0 && ry == 0.0 {
        DEGENERATE_RING_FOOTPRINT
    } else {
        (0.5 * (rx * rx + ry * ry).sqrt()).sqrt()
    };
    Ok(M2Bounds {
        sphere_radius: h.bounding_sphere_radius,
        bbox_min: h.bounding_box_min,
        bbox_max: h.bounding_box_max,
        vert_max_from_origin,
        ring_footprint,
        pivot_z: model.pivot_attach_z,
        stand_box_z: rz.max(0.0),
        mounted_pivot_delta: mounted_pivot_delta(bytes).unwrap_or(0.0),
        // Through `M2Model::attachment`, which is the reference's own id→record resolve
        // (`0x710310`: the AttachLookup, its bounds check and its `0xffff` sentinel) — never a
        // scan of the record table, whose ids are not unique.
        mount_seat_z: model.attachment(0).map_or(0.0, |a| a.position[2]),
    })
}
