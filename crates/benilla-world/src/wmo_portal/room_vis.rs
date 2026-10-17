//! Is a claimed WMO room visible this frame? The reference elects an object standing in a WMO
//! group with its building (`0x6834e0`, in the WMO-group render path): when the group is not
//! rendered, the object is neither drawn nor ticked. The anim-LOD gate (`creature_anim::lod`) and
//! the body draw election (`crate::exterior_cull`) both ask it of a body's [`UnitWmoRoom`], and one
//! function answers both so they cannot disagree.

use bevy::prelude::*;

use benilla_assets::coords::bevy_to_wow;
use benilla_assets::{WmoGroupNav, WmoModel, WmoPortalInfo, WmoPortalRef};

use super::{UnitWmoRoom, WmoPortalInstance, WmoRoom};

/// Whether a body's claimed room is visible this frame. Every seam (no claim, a despawned
/// placement, a loading model) fails open: a lookup miss must never hide or park a body.
///
/// `at` is the body's world-space sphere `(centre, radius)`. With it, the doorway-straddle guard
/// (see [`room_visible`]) admits a neighbour's PVS bit only for a body that reaches the portal the
/// two rooms share; `None` admits any neighbour, as before.
pub fn room_pvs_visible(
    room: Option<&UnitWmoRoom>,
    at: Option<(Vec3, f32)>,
    instances: &Query<&WmoPortalInstance>,
    wmos: &Assets<WmoModel>,
) -> bool {
    let Some(WmoRoom { instance, group }) = room.and_then(|r| r.room()) else {
        return true;
    };
    let Ok(inst) = instances.get(instance) else {
        return true;
    };
    let Some(model) = wmos.get(&inst.handle) else {
        return true;
    };
    // Into the portals' own space: WoW model coordinates. Placement scale is uniform, so the
    // radius scales by the inverse's x axis.
    let at = at.map(|(centre, radius)| {
        let local_from_world = inst.world_from_local.inverse();
        let p = local_from_world.transform_point3(centre);
        (bevy_to_wow(p), radius * local_from_world.matrix3.x_axis.length())
    });
    room_visible(
        &inst.visible,
        &model.group_nav,
        &model.portal_refs,
        &model.portal_infos,
        &model.portal_vertices,
        group,
        at,
    )
}

/// Is the claimed group in the PVS, or a group one portal hop from it while the body stands in
/// that portal? The hop is the doorway-straddle guard: a body extends past its room only through
/// a portal opening, so only a body within `radius` of the shared portal's box can show in the
/// neighbour.
///
/// **The hop used to admit every body of every neighbouring room** - wenilla carry. The camera's
/// own room is always in the PVS, so standing inside a WMO facing a wall still drew and animated
/// everyone in the rooms next door: the owner saw the frame drop with nobody on screen. The
/// reference has no hop at all (`0x6834e0` submits a group's members only while the group
/// renders); this keeps the guard for the body that needs it. An index out of range reads
/// visible, as in [`super::WmoGroupVis::drawn_by`]; a portal with no polygon reads reached.
fn room_visible(
    visible: &[bool],
    nav: &[WmoGroupNav],
    refs: &[WmoPortalRef],
    infos: &[WmoPortalInfo],
    portal_vertices: &[[f32; 3]],
    group: u16,
    at: Option<([f32; 3], f32)>,
) -> bool {
    let vis = |g: usize| visible.get(g).copied().unwrap_or(true);
    if vis(group as usize) {
        return true;
    }
    let Some(n) = nav.get(group as usize) else {
        return true;
    };
    let Some(hops) = refs.get(n.ref_start as usize..(n.ref_start as usize + n.ref_count as usize))
    else {
        return true;
    };
    let reaches = |r: &WmoPortalRef| {
        let Some((p, radius)) = at else {
            return true;
        };
        let Some(info) = infos.get(r.portal as usize) else {
            return true;
        };
        let start = info.start_vertex as usize;
        let Some(verts) = portal_vertices.get(start..start + info.count as usize) else {
            return true;
        };
        if verts.is_empty() {
            return true;
        }
        (0..3).all(|axis| {
            let lo = verts.iter().map(|v| v[axis]).fold(f32::INFINITY, f32::min);
            let hi = verts.iter().map(|v| v[axis]).fold(f32::NEG_INFINITY, f32::max);
            p[axis] >= lo - radius && p[axis] <= hi + radius
        })
    };
    hops.iter().any(|r| vis(r.group as usize) && reaches(r))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A [`WmoGroupNav`] whose only meaningful fields are its portal-ref slice.
    fn nav(ref_start: u16, ref_count: u16) -> WmoGroupNav {
        WmoGroupNav {
            flags: 0,
            bbox_min: [0.0; 3],
            bbox_max: [0.0; 3],
            ref_start,
            ref_count,
            area_table_id: 0,
            fog_indices: [0; 4],
            group_liquid: benilla_formats::NO_GROUP_LIQUID,
        }
    }

    fn pref(group: u16) -> WmoPortalRef {
        WmoPortalRef {
            portal: 0,
            group,
            side: 1,
        }
    }

    /// Direct bit, the straddle guard, all dark, the sealed room, and both fail-open seams.
    #[test]
    fn room_visible_covers_the_hop_guard_and_fails_open() {
        // Groups 0 ↔ 1 share one portal; group 2 is sealed (no refs).
        let navs = vec![nav(0, 1), nav(1, 1), nav(2, 0)];
        let refs = vec![pref(1), pref(0)];
        assert!(
            room_visible(&[true, false, false], &navs, &refs, &[], &[], 0, None),
            "direct PVS bit"
        );
        assert!(
            room_visible(&[true, false, false], &navs, &refs, &[], &[], 1, None),
            "own bit dark but the neighbour lit — the doorway-straddle guard keeps it live"
        );
        assert!(
            !room_visible(&[false, false, true], &navs, &refs, &[], &[], 0, None),
            "own room and every hop dark ⇒ not visible"
        );
        assert!(
            !room_visible(&[true, true, false], &navs, &refs, &[], &[], 2, None),
            "a sealed room's own bit decides — no hops exist to save it"
        );
        assert!(
            room_visible(&[false], &navs, &refs, &[], &[], 9, None),
            "a group past the PVS table reads visible (fail-open)"
        );
        assert!(
            room_visible(&[false], &[nav(7, 2)], &refs, &[], &[], 0, None),
            "a ref slice past the refs vec reads visible (fail-open)"
        );
    }

    /// Facing a wall in room 0, the PVS is room 0 alone: a body in room 1 shows only while it
    /// stands in the doorway they share, not anywhere in the room.
    #[test]
    fn a_neighbours_body_counts_only_in_the_doorway() {
        let navs = vec![nav(0, 1), nav(1, 1)];
        let refs = vec![pref(1), pref(0)];
        // Portal 0: a 2 x 2 opening in the x = 0 plane.
        let infos = vec![WmoPortalInfo {
            start_vertex: 0,
            count: 4,
            plane: [1.0, 0.0, 0.0, 0.0],
        }];
        let verts = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.0, -1.0, 2.0]];
        let pvs = [true, false];
        assert!(
            room_visible(&pvs, &navs, &refs, &infos, &verts, 1, Some(([0.5, 0.0, 1.0], 1.0))),
            "in the doorway: still drawn"
        );
        assert!(
            !room_visible(&pvs, &navs, &refs, &infos, &verts, 1, Some(([20.0, 0.0, 1.0], 1.0))),
            "deep in the next room behind the wall: not drawn"
        );
    }
}
