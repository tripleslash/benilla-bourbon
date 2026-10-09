//! Every transport path in the chain's `TaxiPathNode.dbc` (a path with a stop), its loading-screen
//! legs and whether `LoadingScreenTaxiSplines.dbc` draws a route for each: what a crossing aboard
//! it shows, the route map or the plain screen.
//! `WOW_DATA=<install>/Data cargo run -p benilla-formats --example ship_routes`

use benilla_formats::{load_ship_splines, load_taxi_path_nodes, ship_legs};

fn main() -> anyhow::Result<()> {
    let data = benilla_formats::wow_data().expect("no WoW install found (set $WOW_DATA)");
    let mut chain = benilla_formats::open_chain(&data)?;
    let nodes = load_taxi_path_nodes(&mut chain)?;
    let splines = load_ship_splines(&mut chain)?;
    println!("{}: {} spline rows", data.display(), splines.len());

    let mut paths: Vec<_> = nodes
        .paths()
        .filter(|(_, path)| path.iter().any(|n| n.flags & 2 != 0))
        .collect();
    paths.sort_by_key(|(id, _)| *id);
    for (path_id, path) in paths {
        let legs = ship_legs(path);
        println!("path {path_id}: {} nodes, {} legs", path.len(), legs.len());
        for (i, leg) in legs.iter().enumerate() {
            let Some(leg) = leg else {
                println!("  leg {i}: no stop, ends the lookup");
                continue;
            };
            let next = legs[(i + 1) % legs.len()].as_ref();
            let to = next.map(|n| (n.first_stop.map_id, n.first_stop.pos));
            let shows = match (splines.points(path_id, i as u32), next) {
                (Some(points), Some(_)) => format!("ROUTE through {} bend points", points.len()),
                _ => "plain screen (no spline row)".to_string(),
            };
            println!(
                "  leg {i}: leaves map {} at [{:.0}, {:.0}] for {:?} — {shows}",
                leg.last_stop.map_id, leg.last_stop.pos[0], leg.last_stop.pos[1], to
            );
        }
    }
    Ok(())
}
