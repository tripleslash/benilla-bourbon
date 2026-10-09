//! The data behind the transport loading screen, the world map with a dotted route that
//! `SMSG_TRANSFER_PENDING`'s transport block raises (`0x4072c0`): the legs of a transport's
//! `TaxiPathNode.dbc` path (`0x408000`) and `LoadingScreenTaxiSplines.dbc`, each leg's bend points
//! on the world sheet (`0x407da0`).

use std::collections::HashMap;

use anyhow::{Context, Result};
use benilla_dbc::{FieldType, Schema, SchemaField};

use crate::dbc::{f32_at, parse, u32_at};
use crate::{Chain, TaxiPathNode};

const SPLINES: &str = "DBFilesClient\\LoadingScreenTaxiSplines.dbc";

/// A row's `LocX`/`LocY` arrays hold this many points (record +0x08 and +0x28).
const SPLINE_POINTS: usize = 8;

/// One leg of a transport path, a run of nodes on one map: its first and last stop (`flags & 2`,
/// the docks). A route leaves a leg's last stop and arrives at the next leg's first.
#[derive(Clone, Copy, Debug)]
pub struct ShipLeg {
    pub first_stop: TaxiPathNode,
    pub last_stop: TaxiPathNode,
}

/// A path's legs in order (`0x408000`). A teleport node (`flags & 1`) or a change of map closes
/// the leg; a stop opens one or extends it. `None` is a closed leg that held no stop, which the
/// reference stores with a zero node id and its lookups reject (`0x407395`, `0x407619`).
///
/// The reference walks the whole table once, so it never closes the last path's last leg; a path
/// is walked on its own here and its last leg closes with it.
pub fn ship_legs(path: &[TaxiPathNode]) -> Vec<Option<ShipLeg>> {
    let mut legs = Vec::new();
    let mut open: Option<ShipLeg> = None;
    for (i, node) in path.iter().enumerate() {
        let changed_map = i > 0 && path[i - 1].map_id != node.map_id;
        if node.flags & 1 != 0 || changed_map {
            legs.push(open.take());
        } else if node.flags & 2 != 0 {
            match &mut open {
                Some(leg) => leg.last_stop = *node,
                None => {
                    open = Some(ShipLeg {
                        first_stop: *node,
                        last_stop: *node,
                    })
                }
            }
        }
    }
    if open.is_some() {
        legs.push(open);
    }
    legs
}

/// `LoadingScreenTaxiSplines.dbc`: `(PathID, LegIndex)` → the route's bend points between its two
/// docks, in world-sheet fractions with y up.
pub struct ShipSplines {
    rows: HashMap<(u32, u32), Vec<[f32; 2]>>,
}

impl ShipSplines {
    /// The bend points of `path_id`'s leg `leg`; `None` when the table has no row for it, and
    /// then the reference shows the plain loading screen.
    pub fn points(&self, path_id: u32, leg: u32) -> Option<&[[f32; 2]]> {
        self.rows.get(&(path_id, leg)).map(Vec::as_slice)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

fn schema() -> Schema {
    let mut s = Schema::new("LoadingScreenTaxiSplines");
    s.add_field(SchemaField::new("ID", FieldType::UInt32));
    s.add_field(SchemaField::new("PathID", FieldType::UInt32));
    for axis in ["LocX", "LocY"] {
        for i in 0..SPLINE_POINTS {
            s.add_field(SchemaField::new(format!("{axis}{i}"), FieldType::Float32));
        }
    }
    s.add_field(SchemaField::new("LegIndex", FieldType::UInt32));
    s
}

/// Read `LoadingScreenTaxiSplines.dbc` off the patch chain into a [`ShipSplines`].
pub fn load_ship_splines(chain: &mut Chain) -> Result<ShipSplines> {
    let bytes = chain
        .read_file(SPLINES)
        .with_context(|| format!("reading {SPLINES}"))?;
    let rs = parse(&bytes, schema(), "LoadingScreenTaxiSplines")?;
    let mut rows = HashMap::with_capacity(rs.records().len());
    for r in rs.records() {
        let (Some(path_id), Some(leg)) = (u32_at(r, 1), u32_at(r, 2 + 2 * SPLINE_POINTS)) else {
            continue;
        };
        // The list ends at the first point with neither coordinate above zero (`0x407750`).
        let points = (0..SPLINE_POINTS)
            .map_while(|i| {
                let x = f32_at(r, 2 + i)?;
                let y = f32_at(r, 2 + SPLINE_POINTS + i)?;
                (x > 0.0 || y > 0.0).then_some([x, y])
            })
            .collect();
        rows.insert((path_id, leg), points);
    }
    Ok(ShipSplines { rows })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_taxi_path_nodes;

    fn node(index: u32, map_id: u32, flags: u32) -> TaxiPathNode {
        TaxiPathNode {
            id: 100 + index,
            path_id: 1,
            node_index: index,
            map_id,
            pos: [0.0; 3],
            flags,
            delay: 0,
        }
    }

    fn stops(legs: &[Option<ShipLeg>]) -> Vec<Option<(u32, u32)>> {
        legs.iter()
            .map(|l| l.map(|l| (l.first_stop.node_index, l.last_stop.node_index)))
            .collect()
    }

    #[test]
    fn a_map_change_closes_a_leg_and_the_path_end_closes_the_last() {
        let path = [
            node(0, 0, 0),
            node(1, 0, 2),
            node(2, 0, 0),
            node(3, 1, 0),
            node(4, 1, 2),
            node(5, 1, 0),
        ];
        assert_eq!(stops(&ship_legs(&path)), [Some((1, 1)), Some((4, 4))]);
    }

    #[test]
    fn a_teleport_node_closes_a_leg_on_one_map() {
        let path = [node(0, 0, 2), node(1, 0, 1), node(2, 0, 2), node(3, 0, 2)];
        assert_eq!(stops(&ship_legs(&path)), [Some((0, 0)), Some((2, 3))]);
    }

    #[test]
    fn a_leg_closed_without_a_stop_is_kept_as_none() {
        let path = [node(0, 0, 0), node(1, 1, 0), node(2, 1, 2)];
        assert_eq!(stops(&ship_legs(&path)), [None, Some((2, 2))]);
        // A trailing run with no stop is not a leg (`0x4080d0` pushes only with a first stop).
        let path = [node(0, 0, 2), node(1, 1, 0)];
        assert_eq!(stops(&ship_legs(&path)), [Some((0, 0))]);
    }

    /// The six 1.12 routes: every spline row names a leg that exists, and each path's two legs are
    /// its two docks.
    #[test]
    fn real_ship_splines_cover_both_legs_of_six_paths() {
        let data = crate::wow_data_or_skip!();
        let mut chain = crate::open_chain(&data).expect("open chain");
        let splines = load_ship_splines(&mut chain).expect("load LoadingScreenTaxiSplines");
        let nodes = load_taxi_path_nodes(&mut chain).expect("load TaxiPathNode");
        assert_eq!(splines.len(), 12);

        // (path, each leg's dock node index and map): the two cross-map boats from Menethil, the
        // Ratchet boat, the three zeppelins.
        for (path_id, docks) in [
            (241, [(11, 1), (29, 0)]),
            (285, [(4, 0), (20, 1)]),
            (292, [(8, 0), (21, 1)]),
            (295, [(7, 0), (24, 1)]),
            (301, [(5, 0), (21, 0)]),
            (302, [(7, 1), (28, 0)]),
        ] {
            let legs = ship_legs(nodes.path(path_id).expect("path exists"));
            let got: Vec<_> = legs
                .iter()
                .map(|l| {
                    let l = l.expect("each leg has its dock");
                    assert_eq!(l.first_stop.id, l.last_stop.id, "one dock per leg");
                    (l.first_stop.node_index, l.first_stop.map_id)
                })
                .collect();
            assert_eq!(got, docks, "path {path_id}");
            for leg in 0..2 {
                let points = splines.points(path_id, leg).expect("a row per leg");
                assert!((2..=SPLINE_POINTS).contains(&points.len()));
                assert!(points
                    .iter()
                    .all(|p| p.iter().all(|c| (0.0..1.0).contains(c))));
            }
        }
        // Ratchet to Booty Bay bends twice on its way east.
        assert_eq!(splines.points(241, 0).map(<[_]>::len), Some(2));
        assert_eq!(splines.points(241, 2), None);
    }
}
