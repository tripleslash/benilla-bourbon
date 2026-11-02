//! `benilla-pack wdl` — **the distant terrain checked against the ADTs it stands in for**, read
//! from built containers rather than an install, and regenerated where they disagree.
//!
//! A map's `.wdl` holds one height per MCNK corner (17×17 a tile) and per MCNK centre (16×16), as
//! `int16` yards. A layer that re-authors a map's ADTs without shipping its `.wdl` leaves the
//! horizon on the old ground: Turtle WoW's patches re-author Azeroth and Kalimdor (the Stormwind
//! harbour, Hyjal, its added islands and coasts) and carry the 1.12.1 continents' WDLs unchanged,
//! so the far band draws sea where the coast now is and land where it was dug out, up to 785 yd
//! off; several of its own maps' WDLs are another map's copy.
//!
//! The stack is the containers in mount order (`web/packs`, then `web/packs-turtle`), each read
//! through its own `index.bpi`, so the check sees exactly what a client on that layer streams and
//! needs no install. Every map with a `.wdt` and a `.wdl` gets measured per WDT tile. The corners
//! are exact in the 1.12.1 continents (all 701,981 are their ADT height truncated); the centres
//! are not a sample the ADT can reproduce (the chunk mean is the nearest rule, 61% exact, 1.35 yd
//! off on average), so a stock tile is kept rather than rebuilt. A tile is rebuilt from its ADT
//! when its ADT comes from a higher container than the `.wdl`, when it has no MARE, or when a
//! corner is off by more than `--tolerance`; a MARE tile the WDT no longer has is dropped. With
//! `--out`, each map with such a tile gets a corrected `.wdl` in a loose tree for
//! `append --loose`. `--grid` measures the drawn surface itself against the ADT's.
//!
//! Deviation: a layer's far band is rebuilt on that layer's ADTs, because the reference (Turtle
//! WoW's own 1.12 client) reads the same stale `.wdl` and draws the old coasts beyond the new ones.
//!
//! ```text
//! benilla-pack wdl --packs web/packs --packs web/packs-turtle --out target/wdl-turtle
//! benilla-pack append --packs web/packs-turtle --loose target/wdl-turtle
//! ```

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use benilla_formats::{adt_to_tile_mesh, terrain_height_at, ChunkMesh, MareTile, WdlFile};
use clap::Parser;
use rayon::prelude::*;

use crate::stack::Stack;

/// Yards a tile edge, and a chunk edge (`benilla_formats::TILE_SIZE / 16`).
const TILE: f64 = 1600.0 / 3.0;
const CHUNK: f64 = TILE / 16.0;
/// World coordinate of lattice index 0 on both axes.
const MAP_OFFSET: f64 = 32.0 * TILE;

#[derive(Parser)]
pub struct WdlArgs {
    /// The containers the maps are read through, lowest first (a later one wins): `web/packs`,
    /// then a layer such as `web/packs-turtle`.
    #[arg(long, required = true)]
    pub packs: Vec<PathBuf>,
    /// Where corrected `.wdl`s go, as a loose tree for `append --loose`. Without it, only report.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// The largest difference (yd) between a MARE sample and the ADT height at its point that a
    /// tile may keep. The MARE stores whole yards, so an exact sample is within 1.
    #[arg(long, default_value_t = 1.0)]
    pub tolerance: f32,
    /// Only these map directories (any case).
    #[arg(long, value_delimiter = ',')]
    pub map: Vec<String>,
    /// Print the WDL's drawn height against the ADT's at a world point: `<map>:<x>,<y>`.
    #[arg(long)]
    pub sample: Vec<String>,
    /// Compare the WDL's drawn height with the ADT's at 16×16 points a tile (each chunk's
    /// quarter point) over a tile rectangle: `<map>:<x0>_<y0>-<x1>_<y1>`.
    #[arg(long)]
    pub grid: Vec<String>,
    /// Print one tile's MARE beside its ADT samples, outer then inner rows: `<map>:<x>_<y>`.
    #[arg(long)]
    pub dump: Vec<String>,
}

/// The WDT's `MAIN` table: which of the 64×64 tiles have an ADT, `[tile_y * 64 + tile_x]`.
fn wdt_tiles(bytes: &[u8]) -> Result<Vec<bool>> {
    let mut at = 0usize;
    while at + 8 <= bytes.len() {
        let magic = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
        let body = bytes
            .get(at + 8..at + 8 + len)
            .context("WDT chunk runs past the file")?;
        if magic == b"NIAM" {
            if len < 4096 * 8 {
                bail!("MAIN is {len} B, expected {}", 4096 * 8);
            }
            return Ok((0..4096)
                .map(|i| body[i * 8] & 1 != 0)
                .collect::<Vec<bool>>());
        }
        at += 8 + len;
    }
    bail!("no MAIN chunk")
}

/// What a `.wdl` carries besides its heights: the chunks between `MVER` and `MAOF`, and each
/// tile's `MAHO` hole mask where it has one.
struct Layout {
    head: Vec<u8>,
    holes: Vec<Option<[u16; 16]>>,
}

fn wdl_layout(bytes: &[u8]) -> Result<Layout> {
    let mut head = Vec::new();
    let mut holes = vec![None; 4096];
    let mut tile_at = std::collections::HashMap::new();
    let mut last_tile = None;
    let mut at = 0usize;
    while at + 8 <= bytes.len() {
        let magic: [u8; 4] = bytes[at..at + 4].try_into().unwrap();
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
        let chunk = bytes
            .get(at..at + 8 + len)
            .context("WDL chunk runs past the file")?;
        match &magic {
            b"REVM" => {}
            b"FOAM" => {
                for i in 0..4096 {
                    let off = u32::from_le_bytes(chunk[8 + i * 4..12 + i * 4].try_into().unwrap());
                    if off != 0 {
                        tile_at.insert(off as usize, i);
                    }
                }
            }
            b"ERAM" => last_tile = tile_at.get(&at).copied(),
            b"OHAM" => {
                if let (Some(i), true) = (last_tile, len >= 32) {
                    let mut mask = [0u16; 16];
                    for (r, m) in mask.iter_mut().enumerate() {
                        *m = u16::from_le_bytes([chunk[8 + r * 2], chunk[9 + r * 2]]);
                    }
                    holes[i] = Some(mask);
                }
            }
            _ if tile_at.is_empty() => head.extend_from_slice(chunk),
            _ => {}
        }
        at += 8 + len;
    }
    Ok(Layout { head, holes })
}

/// One tile's heights as its ADT has them at the MARE's sample points, in absolute yards, and
/// which chunks are holed through.
struct AdtSamples {
    outer: [Option<f32>; 289],
    inner: [Option<f32>; 256],
    /// Bit `c` of row `r`: chunk `(r, c)` has every 2×2 block holed (`MCNK+0x3C` = `0xFFFF`).
    holes: [u16; 16],
    chunks: Vec<ChunkMesh>,
}

/// The ADT's heights for the MARE: each chunk's north-east corner (MCVT outer `(0, 0)`), and for
/// the tile's last row and column the far corners of the chunks beside them — the same lattice
/// points the neighbouring tile's first row and column hold; and each chunk's mean over its 145
/// MCVT heights for the cell centre. The 1.12.1 continents' WDLs bear both out: every outer sample
/// is its corner height truncated, and an inner one sits on the chunk's mean rather than on the
/// centre vertex (off by tens of yards on a cliff, where the mean is within a few).
fn adt_samples(tile_x: u32, tile_y: u32, bytes: &[u8]) -> Result<AdtSamples> {
    let tile = adt_to_tile_mesh(bytes)?;
    let mut s = AdtSamples {
        outer: [None; 289],
        inner: [None; 256],
        holes: [0; 16],
        chunks: Vec::new(),
    };
    for chunk in &tile.chunks {
        let [x, y, _] = chunk.positions[0];
        let r = ((MAP_OFFSET - f64::from(x)) / CHUNK).round() as i64 - i64::from(tile_y) * 16;
        let c = ((MAP_OFFSET - f64::from(y)) / CHUNK).round() as i64 - i64::from(tile_x) * 16;
        if !(0..16).contains(&r) || !(0..16).contains(&c) {
            bail!("chunk at ({x}, {y}) is not in tile ({tile_x}, {tile_y})");
        }
        let (r, c) = (r as usize, c as usize);
        let z = |vr: usize, vc: usize| chunk.positions[vr * 17 + vc][2];
        s.outer[r * 17 + c] = Some(z(0, 0));
        let sum: f64 = chunk.positions.iter().map(|p| f64::from(p[2])).sum();
        s.inner[r * 16 + c] = Some((sum / chunk.positions.len() as f64) as f32);
        if r == 15 {
            s.outer[16 * 17 + c] = Some(z(8, 0));
        }
        if c == 15 {
            s.outer[r * 17 + 16] = Some(z(0, 8));
        }
        if r == 15 && c == 15 {
            s.outer[16 * 17 + 16] = Some(z(8, 8));
        }
        if chunk.holes == 0xffff {
            s.holes[r] |= 1 << c;
        }
    }
    s.chunks = tile.chunks;
    Ok(s)
}

/// The MARE height for an ADT sample: whole yards, truncated toward zero — the rule every outer
/// sample of the 1.12.1 continents' own WDLs follows (`benilla-pack wdl --packs web/packs`).
fn quantise(h: f32) -> i16 {
    h.trunc().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

/// How one WDT tile's MARE compares with its ADT.
#[derive(Default, Clone, Copy)]
struct TileCheck {
    /// Largest |MARE − ADT| over the corners, and the corners that are their ADT height truncated.
    outer_err: f32,
    truncated: usize,
    outer: usize,
    /// Largest |MARE − ADT chunk mean| over the centres.
    inner_err: f32,
}

fn check_tile(mare: &MareTile, adt: &AdtSamples) -> TileCheck {
    let mut t = TileCheck::default();
    for (&m, a) in mare.outer.iter().zip(&adt.outer) {
        let Some(a) = *a else { continue };
        t.outer += 1;
        t.truncated += usize::from(m == quantise(a));
        t.outer_err = t.outer_err.max((f32::from(m) - a).abs());
    }
    for (&m, a) in mare.inner.iter().zip(&adt.inner) {
        let Some(a) = *a else { continue };
        t.inner_err = t.inner_err.max((f32::from(m) - a).abs());
    }
    t
}

/// Why a WDT tile's MARE is rebuilt from its ADT.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stale {
    /// The WDL has no MARE for it.
    Missing,
    /// Its ADT comes from a container above the one the WDL comes from: a layer re-authored the
    /// ground and left the horizon as it was.
    Newer,
    /// A corner is off its ADT height by more than the tolerance.
    Off,
}

/// A map's WDL against its ADTs, and the corrected file when it needs one.
struct MapReport {
    wdt_tiles: usize,
    mare_tiles: usize,
    /// Rebuilt tiles and why, with the tile's corner error before.
    stale: Vec<((u32, u32), Stale, f32)>,
    /// MARE tiles the WDT has no ADT for: dropped.
    orphans: Vec<(u32, u32)>,
    /// WDT tiles whose ADT is absent or does not parse: their MARE, if any, is kept.
    unreadable: Vec<(u32, u32)>,
    /// Over the tiles the WDL keeps (all of them, when nothing is written): the corners that are
    /// their ADT height truncated, of how many, the worst corner and the worst centre.
    truncated: usize,
    outer: usize,
    outer_err: f32,
    inner_err: f32,
    wdl_holder: Option<usize>,
    corrected: Option<Vec<u8>>,
}

fn check_map(stack: &Stack, dir: &str, tolerance: f32) -> Result<Option<MapReport>> {
    let wdt_name = format!(r"World\Maps\{dir}\{dir}.wdt");
    let wdl_name = format!(r"World\Maps\{dir}\{dir}.wdl");
    let Some(wdt) = stack.read(&wdt_name)? else {
        return Ok(None);
    };
    let present = wdt_tiles(&wdt).with_context(|| wdt_name.clone())?;
    let wdl_bytes = stack.read(&wdl_name)?;
    let wdl = match &wdl_bytes {
        Some(b) => Some(WdlFile::parse(b).with_context(|| wdl_name.clone())?),
        None => None,
    };
    let layout = match &wdl_bytes {
        Some(b) => wdl_layout(b)?,
        None => Layout {
            head: Vec::new(),
            holes: vec![None; 4096],
        },
    };
    let wdl_holder = stack.holder(&wdl_name);

    let tiles: Vec<(u32, u32)> = (0..4096u32)
        .filter(|&i| present[i as usize])
        .map(|i| (i % 64, i / 64))
        .collect();
    let derived: Vec<Option<(usize, AdtSamples)>> = tiles
        .par_iter()
        .map(|&(x, y)| {
            let name = format!(r"World\Maps\{dir}\{dir}_{x}_{y}.adt");
            let layer = stack.holder(&name)?;
            let bytes = stack.read(&name).ok().flatten()?;
            Some((layer, adt_samples(x, y, &bytes).ok()?))
        })
        .collect();

    let mut r = MapReport {
        wdt_tiles: tiles.len(),
        mare_tiles: wdl.as_ref().map_or(0, |w| w.present_count()),
        stale: Vec::new(),
        orphans: Vec::new(),
        unreadable: Vec::new(),
        truncated: 0,
        outer: 0,
        outer_err: 0.0,
        inner_err: 0.0,
        wdl_holder,
        corrected: None,
    };
    let mut out: Vec<Option<MareTile>> = vec![None; 4096];
    let mut holes = layout.holes.clone();
    for (&(x, y), adt) in tiles.iter().zip(&derived) {
        let i = (y * 64 + x) as usize;
        let mare = wdl.as_ref().and_then(|w| w.tile(x, y));
        let Some((adt_layer, adt)) = adt else {
            // A WDT tile whose ADT does not parse: keep whatever the WDL had.
            r.unreadable.push((x, y));
            out[i] = mare.cloned();
            continue;
        };
        let check = mare.map(|m| check_tile(m, adt));
        let stale = match (mare, check) {
            (None, _) | (_, None) => Some(Stale::Missing),
            (Some(_), Some(c)) if c.outer_err > tolerance => Some(Stale::Off),
            _ if wdl_holder.is_some_and(|w| *adt_layer > w) => Some(Stale::Newer),
            _ => None,
        };
        let Some(stale) = stale else {
            let c = check.unwrap_or_default();
            r.truncated += c.truncated;
            r.outer += c.outer;
            r.outer_err = r.outer_err.max(c.outer_err);
            r.inner_err = r.inner_err.max(c.inner_err);
            out[i] = mare.cloned();
            continue;
        };
        r.stale
            .push(((x, y), stale, check.map_or(f32::NAN, |c| c.outer_err)));
        let fill = |s: Option<f32>, old: Option<i16>| s.map(quantise).or(old).unwrap_or(0);
        let mut tile = MareTile {
            outer: [0; 289],
            inner: [0; 256],
        };
        for (k, h) in tile.outer.iter_mut().enumerate() {
            *h = fill(adt.outer[k], mare.map(|m| m.outer[k]));
        }
        for (k, h) in tile.inner.iter_mut().enumerate() {
            *h = fill(adt.inner[k], mare.map(|m| m.inner[k]));
        }
        out[i] = Some(tile);
        if layout.holes.iter().any(Option::is_some) {
            holes[i] = Some(adt.holes);
        }
    }
    if let Some(w) = &wdl {
        for (x, y) in w.present_tiles() {
            if !present[(y * 64 + x) as usize] {
                r.orphans.push((x, y));
                holes[(y * 64 + x) as usize] = None;
            }
        }
    }
    if wdl.is_some() && (!r.stale.is_empty() || !r.orphans.is_empty()) {
        let file = WdlFile::from_tiles(out)?;
        r.corrected = Some(file.encode(&layout.head, Some(&holes)));
    }
    Ok(Some(r))
}

/// `<map>:<x>,<y>` — the WDL's drawn height against the ADT's at one world point.
fn sample(stack: &Stack, spec: &str) -> Result<()> {
    let (map, at) = spec
        .split_once(':')
        .with_context(|| format!("--sample {spec}: want <map>:<x>,<y>"))?;
    let (x, y) = at
        .split_once(',')
        .with_context(|| format!("--sample {spec}: want <map>:<x>,<y>"))?;
    let (x, y): (f32, f32) = (x.trim().parse()?, y.trim().parse()?);
    let (tx, ty) = benilla_formats::world_to_tile(x, y);
    let wdl = stack
        .read(&format!("World\\Maps\\{map}\\{map}.wdl"))?
        .map(|b| WdlFile::parse(&b))
        .transpose()?;
    let adt = stack.read(&format!("World\\Maps\\{map}\\{map}_{tx}_{ty}.adt"))?;
    let ground = match adt {
        Some(b) => terrain_height_at(&adt_to_tile_mesh(&b)?.chunks, [x, y, 0.0]),
        None => None,
    };
    let far = wdl.and_then(|w| w.height_at(x, y));
    let fmt = |h: Option<f32>| h.map_or("none".to_string(), |h| format!("{h:.1}"));
    let diff = match (far, ground) {
        (Some(f), Some(g)) => format!("{:+.1}", f - g),
        _ => "-".to_string(),
    };
    println!(
        "sample {map} ({x}, {y}) tile ({tx}, {ty}): wdl {} adt {} diff {diff}",
        fmt(far),
        fmt(ground)
    );
    Ok(())
}

/// `<map>:<x0>_<y0>-<x1>_<y1>` — the drawn WDL against the ADT over a rectangle of tiles: per
/// WDT tile the mean and largest |WDL − ADT| over its 256 points, then the whole rectangle's.
fn grid(stack: &Stack, spec: &str) -> Result<()> {
    let bad = || format!("--grid {spec}: want <map>:<x0>_<y0>-<x1>_<y1>");
    let (map, rect) = spec.split_once(':').with_context(bad)?;
    let (a, b) = rect.split_once('-').with_context(bad)?;
    let corner = |t: &str| -> Result<(u32, u32)> {
        let (x, y) = t.split_once('_').with_context(bad)?;
        Ok((x.parse()?, y.parse()?))
    };
    let ((x0, y0), (x1, y1)) = (corner(a)?, corner(b)?);
    let wdl = stack
        .read(&format!(r"World\Maps\{map}\{map}.wdl"))?
        .map(|b| WdlFile::parse(&b))
        .transpose()?;
    // The tiles the WDT has: the client loads no other ADT, so no other ground is drawn.
    let present = stack
        .read(&format!(r"World\Maps\{map}\{map}.wdt"))?
        .map(|b| wdt_tiles(&b))
        .transpose()?
        .unwrap_or_else(|| vec![false; 4096]);
    let tiles: Vec<(u32, u32)> = (y0..=y1)
        .flat_map(|y| (x0..=x1).map(move |x| (x, y)))
        .filter(|&(x, y)| present[(y * 64 + x) as usize])
        .collect();
    let rows: Vec<Option<(f64, f32, usize, usize)>> = tiles
        .par_iter()
        .map(|&(x, y)| {
            let b = stack
                .read(&format!(r"World\Maps\{map}\{map}_{x}_{y}.adt"))
                .ok()??;
            let chunks = adt_to_tile_mesh(&b).ok()?.chunks;
            let (mut sum, mut worst, mut n, mut bare) = (0.0f64, 0.0f32, 0usize, 0usize);
            for r in 0..16 {
                for c in 0..16 {
                    let wx = (MAP_OFFSET - (f64::from(y) * 16.0 + r as f64 + 0.25) * CHUNK) as f32;
                    let wy = (MAP_OFFSET - (f64::from(x) * 16.0 + c as f64 + 0.25) * CHUNK) as f32;
                    let Some(g) = terrain_height_at(&chunks, [wx, wy, 0.0]) else {
                        continue;
                    };
                    match wdl.as_ref().and_then(|w| w.height_at(wx, wy)) {
                        Some(f) => {
                            let d = (f - g).abs();
                            sum += f64::from(d);
                            worst = worst.max(d);
                            n += 1;
                        }
                        None => bare += 1,
                    }
                }
            }
            Some((sum, worst, n, bare))
        })
        .collect();
    let (mut sum, mut worst, mut n, mut bare) = (0.0f64, 0.0f32, 0usize, 0usize);
    for (&(x, y), row) in tiles.iter().zip(&rows) {
        let Some((s, w, k, b)) = *row else { continue };
        if k > 0 || b > 0 {
            println!(
                "grid {map} {x}_{y}: mean {:.1} max {w:.1} yd over {k} points, {b} with ground and no WDL",
                s / k.max(1) as f64
            );
        }
        sum += s;
        worst = worst.max(w);
        n += k;
        bare += b;
    }
    println!(
        "grid {map} {x0}_{y0}-{x1}_{y1}: mean {:.2} max {worst:.1} yd over {n} points, {bare} with ground and no WDL",
        sum / n.max(1) as f64
    );
    Ok(())
}

/// `<map>:<x>_<y>` — one tile's MARE and ADT samples, row by row, as `mare/adt`.
fn dump(stack: &Stack, spec: &str) -> Result<()> {
    let (map, at) = spec
        .split_once(':')
        .with_context(|| format!("--dump {spec}: want <map>:<x>_<y>"))?;
    let (x, y) = at
        .split_once('_')
        .with_context(|| format!("--dump {spec}: want <map>:<x>_<y>"))?;
    let (x, y): (u32, u32) = (x.parse()?, y.parse()?);
    let wdl = stack
        .read(&format!("World\\Maps\\{map}\\{map}.wdl"))?
        .map(|b| WdlFile::parse(&b))
        .transpose()?;
    let adt = stack
        .read(&format!("World\\Maps\\{map}\\{map}_{x}_{y}.adt"))?
        .context("no such ADT")?;
    let adt = adt_samples(x, y, &adt)?;
    let mare = wdl.as_ref().and_then(|w| w.tile(x, y));
    let cell = |m: Option<i16>, a: Option<f32>| {
        format!(
            "{}/{}",
            m.map_or("-".into(), |m| m.to_string()),
            a.map_or("-".into(), |a| format!("{a:.1}"))
        )
    };
    println!("{map} {x}_{y} outer:");
    for r in 0..17 {
        let row: Vec<String> = (0..17)
            .map(|c| cell(mare.map(|m| m.outer[r * 17 + c]), adt.outer[r * 17 + c]))
            .collect();
        println!("  {}", row.join(" "));
    }
    println!("{map} {x}_{y} inner:");
    for r in 0..16 {
        let row: Vec<String> = (0..16)
            .map(|c| cell(mare.map(|m| m.inner[r * 16 + c]), adt.inner[r * 16 + c]))
            .collect();
        println!("  {}", row.join(" "));
    }
    Ok(())
}

pub fn wdl(a: WdlArgs) -> Result<()> {
    let stack = Stack::open(&a.packs)?;
    for spec in &a.sample {
        sample(&stack, spec)?;
    }
    for spec in &a.dump {
        dump(&stack, spec)?;
    }
    for spec in &a.grid {
        grid(&stack, spec)?;
    }
    let probes = a.sample.len() + a.dump.len() + a.grid.len();
    if probes > 0 && a.map.is_empty() && a.out.is_none() {
        return Ok(());
    }
    let only: Vec<String> = a.map.iter().map(|m| m.to_ascii_lowercase()).collect();
    let (mut written, mut truncated, mut outer) = (0, 0, 0);
    let mut without_wdl = Vec::new();
    for (lower, dir) in stack.maps() {
        if !only.is_empty() && !only.contains(&lower) {
            continue;
        }
        let Some(r) = check_map(&stack, &dir, a.tolerance)? else {
            continue;
        };
        if r.wdt_tiles == 0 {
            continue;
        }
        let Some(holder) = r.wdl_holder else {
            without_wdl.push(format!("{dir} ({})", r.wdt_tiles));
            continue;
        };
        truncated += r.truncated;
        outer += r.outer;
        let count = |k: Stale| r.stale.iter().filter(|s| s.1 == k).count();
        println!(
            "{dir}: {} WDT tiles, {} MARE tiles ({}); kept tiles: corners worst {:.2} yd, {}/{} truncated, centres worst {:.1} yd off the chunk mean; rebuilt: {} newer ADT, {} corners off > {} yd, {} without MARE; {} orphan MARE, {} unreadable ADT",
            r.wdt_tiles,
            r.mare_tiles,
            stack.dir(holder).display(),
            r.outer_err,
            r.truncated,
            r.outer,
            r.inner_err,
            count(Stale::Newer),
            count(Stale::Off),
            a.tolerance,
            count(Stale::Missing),
            r.orphans.len(),
            r.unreadable.len(),
        );
        if !r.unreadable.is_empty() && r.unreadable.len() <= 12 {
            let list: Vec<String> = r
                .unreadable
                .iter()
                .map(|(x, y)| format!("{x}_{y}"))
                .collect();
            println!("  unreadable ADT: {}", list.join(", "));
        }
        let mut off: Vec<_> = r.stale.iter().filter(|s| !s.2.is_nan()).collect();
        off.sort_by(|a, b| b.2.total_cmp(&a.2));
        if !off.is_empty() {
            let list: Vec<String> = off
                .iter()
                .take(12)
                .map(|((x, y), _, e)| format!("{x}_{y} {e:.0}"))
                .collect();
            println!("  worst corners before: {}", list.join(", "));
        }
        if !r.orphans.is_empty() {
            let list: Vec<String> = r.orphans.iter().map(|(x, y)| format!("{x}_{y}")).collect();
            println!("  orphan MARE: {}", list.join(", "));
        }
        if let (Some(out), Some(bytes)) = (&a.out, &r.corrected) {
            let path = out
                .join("World")
                .join("Maps")
                .join(&dir)
                .join(format!("{dir}.wdl"));
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
            println!("  wrote {} ({} B)", path.display(), bytes.len());
            written += 1;
        }
    }
    if !without_wdl.is_empty() {
        println!("maps with ADTs and no WDL: {}", without_wdl.join(", "));
    }
    eprintln!(
        "{written} corrected WDLs; {truncated} of {outer} kept MARE corners are the ADT height truncated"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MARE stores whole yards toward zero: Azeroth 30_26's corners hold −36 for an ADT −36.5
    /// and 29 for 29.8 (`--dump Azeroth:30_26` over `web/packs`).
    #[test]
    fn a_mare_height_is_the_adt_height_truncated() {
        assert_eq!(quantise(-36.5), -36);
        assert_eq!(quantise(-35.8), -35);
        assert_eq!(quantise(29.8), 29);
        assert_eq!(quantise(1e9), i16::MAX);
    }

    #[test]
    fn the_wdt_main_table_says_which_tiles_exist() {
        let mut wdt = Vec::new();
        wdt.extend(b"REVM");
        wdt.extend(4u32.to_le_bytes());
        wdt.extend(18u32.to_le_bytes());
        wdt.extend(b"NIAM");
        wdt.extend((4096u32 * 8).to_le_bytes());
        let mut main = vec![0u8; 4096 * 8];
        main[(48 * 64 + 31) * 8] = 1;
        wdt.extend(main);
        let tiles = wdt_tiles(&wdt).unwrap();
        assert_eq!(tiles.iter().filter(|t| **t).count(), 1);
        assert!(tiles[48 * 64 + 31]);
    }

    /// A corrected file keeps what its source carried besides the heights: the chunks before
    /// `MAOF`, and a `MAHO` behind each tile's `MARE` where the source had them.
    #[test]
    fn a_rewritten_wdl_keeps_its_head_and_hole_masks() {
        let mut tiles: Vec<Option<MareTile>> = vec![None; 4096];
        let mut holes = vec![None; 4096];
        for (i, at) in [(31usize, 48usize), (32, 48)]
            .iter()
            .map(|&(x, y)| (y * 64 + x, x))
        {
            tiles[i] = Some(MareTile {
                outer: [at as i16; 289],
                inner: [-(at as i16); 256],
            });
            holes[i] = Some([at as u16; 16]);
        }
        let mut head = Vec::new();
        head.extend(b"OMWM");
        head.extend(3u32.to_le_bytes());
        head.extend(b"ab\0");
        let bytes = WdlFile::from_tiles(tiles)
            .unwrap()
            .encode(&head, Some(&holes));
        let layout = wdl_layout(&bytes).unwrap();
        assert_eq!(layout.head, head);
        assert_eq!(layout.holes, holes);
        let back = WdlFile::parse(&bytes).unwrap();
        assert_eq!(back.present_count(), 2);
        assert_eq!(back.tile(32, 48).unwrap().outer[0], 32);
        assert_eq!(back.tile(31, 48).unwrap().inner[255], -31);
    }

    /// Against the built 1.12.1 container: every corner of the Stormwind tile is its ADT height
    /// truncated, the rule a corrected tile is written by.
    #[test]
    fn the_stock_wdl_corners_are_the_adt_heights_truncated() {
        let Some(dir) = benilla_pack::find::find() else {
            eprintln!(
                "skipping: no BENPACK container — looked in {:?}",
                benilla_pack::find::candidates()
            );
            return;
        };
        let stack = Stack::open(&[dir]).unwrap();
        let wdl = WdlFile::parse(
            &stack
                .read(r"World\Maps\Azeroth\Azeroth.wdl")
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let adt = stack
            .read(r"World\Maps\Azeroth\Azeroth_31_48.adt")
            .unwrap()
            .unwrap();
        let check = check_tile(
            wdl.tile(31, 48).unwrap(),
            &adt_samples(31, 48, &adt).unwrap(),
        );
        assert_eq!(check.outer, 289);
        assert_eq!(check.truncated, 289, "worst corner {} yd", check.outer_err);
    }
}
