//! `benilla-pack sidn` — **the SIDN flag written back into the WMO roots that lost it**, in the
//! layer containers, so the flag decides everywhere at runtime as the reference's does
//! (`0x6b4090` tests MOMT `0x10` before it lights a material's SIDN colour).
//!
//! Turtle WoW's copies of [`SIDN_FLAG_LOST`] zero every MOMT flag but keep the SIDN colours, which
//! stand on their window and lamp glass alone (and `kttownhall`'s clock face): left as they are,
//! that glass stays dark at night. `Stormwind.wmo`'s stock copy flags exactly the materials its
//! Turtle copy colours, 6 of 6. A census of the 1,335 roots on the reforged-over-Turtle-over-base
//! chain found 129 with an unflagged colour; the other 107 colour walls, roofs and props, and are
//! left as they are. For the 21 without a stock copy the list is a judgement from texture names:
//! `BalorCastleRuined.wmo`, whose one coloured material is a door, and `Dwarven_house.wmo`, whose
//! colours include non-glass, are not on it.
//!
//! Each copy of a listed root a `--patch` container holds is rewritten with bit `0x10` set on every
//! material whose SIDN colour is not black, and no other byte changed, into a loose tree for
//! `append --loose`: `<out>/<container folder>/<name>`.
//!
//! Deviation: the flag is set where Turtle WoW's data lost it, because its own client reads these
//! roots unflagged and leaves their glass dark while the colours, like the stock city's flags, say
//! that glass was meant to glow.
//!
//! ```text
//! benilla-pack sidn --packs web/packs --packs web/packs-turtle --packs web/packs-reforged \
//!     --patch web/packs-turtle --patch web/packs-reforged --out target/sidn
//! benilla-pack append --packs web/packs-turtle --loose target/sidn/packs-turtle
//! benilla-pack append --packs web/packs-reforged --loose target/sidn/packs-reforged
//! ```

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use benilla_mpq::name_key;
use clap::Parser;

use crate::stack::Stack;

/// The roots whose SIDN flag Turtle WoW's data lost, by file name.
pub const SIDN_FLAG_LOST: [&str; 22] = [
    "stormwind.wmo",
    "stormwindharbor.wmo",
    "al_barracks.wmo",
    "al_twostory.wmo",
    "albs.wmo",
    "alfarm.wmo",
    "alinn.wmo",
    "dal_hq.wmo",
    "highelf_smallhouse01.wmo",
    "kt_barn.wmo",
    "kt_barracks.wmo",
    "kt_hq.wmo",
    "kt_lo.wmo",
    "kt_lumbermill.wmo",
    "kt_magetower.wmo",
    "kt_twostory.wmo",
    "ktbs.wmo",
    "ktfarm.wmo",
    "ktinn.wmo",
    "kttownhall.wmo",
    "magetower.wmo",
    "prisonlonghouse_lordaeron.wmo",
];

/// MOMT's flag that lets a material's SIDN colour glow.
const SIDN_FLAG: u32 = 0x10;
/// One MOMT record in a 1.12 root, and where its SIDN colour (BGRA) sits.
const MOMT_LEN: usize = 64;
const SIDN_AT: usize = 0x10;

#[derive(Parser)]
pub struct SidnArgs {
    /// The containers the roots are read through, lowest first.
    #[arg(long, required = true)]
    pub packs: Vec<PathBuf>,
    /// A container among `--packs` whose copies of the listed roots are rewritten.
    #[arg(long)]
    pub patch: Vec<PathBuf>,
    /// Where the rewritten roots go. Without it, only report.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

/// One MOMT record: where it sits in the file, its flags and its SIDN colour as RGB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Material {
    pub at: usize,
    pub flags: u32,
    pub sidn: [u8; 3],
}

impl Material {
    /// A coloured material the flag does not light.
    pub fn glass_unflagged(&self) -> bool {
        self.sidn != [0; 3] && self.flags & SIDN_FLAG == 0
    }
}

/// The root's MOMT records, by walking its top-level chunks.
pub fn materials(root: &[u8]) -> Result<Vec<Material>> {
    let mut at = 0usize;
    while at + 8 <= root.len() {
        let len = u32::from_le_bytes(root[at + 4..at + 8].try_into().unwrap()) as usize;
        let body = at + 8;
        root.get(body..body + len)
            .context("WMO chunk runs past the file")?;
        if &root[at..at + 4] == b"TMOM" {
            return Ok((0..len / MOMT_LEN)
                .map(|i| {
                    let m = body + i * MOMT_LEN;
                    let [b, g, r] = [
                        root[m + SIDN_AT],
                        root[m + SIDN_AT + 1],
                        root[m + SIDN_AT + 2],
                    ];
                    Material {
                        at: m,
                        flags: u32::from_le_bytes(root[m..m + 4].try_into().unwrap()),
                        sidn: [r, g, b],
                    }
                })
                .collect());
        }
        at = body + len;
    }
    Ok(Vec::new())
}

/// The root with `0x10` set on every coloured material that lacks it, or `None` when none does.
pub fn restore_flag(root: &[u8]) -> Result<Option<Vec<u8>>> {
    let lost: Vec<Material> = materials(root)?
        .into_iter()
        .filter(Material::glass_unflagged)
        .collect();
    if lost.is_empty() {
        return Ok(None);
    }
    let mut out = root.to_vec();
    for m in lost {
        out[m.at..m.at + 4].copy_from_slice(&(m.flags | SIDN_FLAG).to_le_bytes());
    }
    Ok(Some(out))
}

/// Whether `name` is one of [`SIDN_FLAG_LOST`].
pub fn is_listed(name: &str) -> bool {
    let file = name.rsplit(['\\', '/']).next().unwrap_or(name);
    SIDN_FLAG_LOST
        .iter()
        .any(|lost| file.eq_ignore_ascii_case(lost))
}

/// Every listed root a container's listfile names, one spelling per key.
pub fn listed_roots(stack: &Stack, layer: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    stack
        .names(layer)
        .into_iter()
        .filter(|n| is_listed(n) && seen.insert(name_key(n)))
        .collect()
}

fn describe(materials: &[Material]) -> String {
    materials
        .iter()
        .map(|m| {
            let rgb = if m.sidn == [0; 3] {
                String::new()
            } else {
                format!(" sidn {:02x}{:02x}{:02x}", m.sidn[0], m.sidn[1], m.sidn[2])
            };
            format!("{:#x}{rgb}", m.flags)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn sidn(a: SidnArgs) -> Result<()> {
    let stack = Stack::open(&a.packs)?;
    let patch: Vec<usize> = a
        .patch
        .iter()
        .map(|p| {
            a.packs
                .iter()
                .position(|q| q == p)
                .with_context(|| format!("--patch {} is not one of --packs", p.display()))
        })
        .collect::<Result<_>>()?;
    let mut found = HashSet::new();
    let mut written = 0;
    for layer in 0..stack.layer_count() {
        let dir = stack.dir(layer);
        for name in listed_roots(&stack, layer) {
            let Some(bytes) = stack.read_in(layer, &name)? else {
                continue;
            };
            let file = name
                .rsplit('\\')
                .next()
                .unwrap_or(&name)
                .to_ascii_lowercase();
            found.insert(file);
            let mats = materials(&bytes).with_context(|| name.clone())?;
            let lost = mats.iter().filter(|m| m.glass_unflagged()).count();
            println!(
                "{} {name}: {} materials, {lost} coloured and unflagged: [{}]",
                dir.display(),
                mats.len(),
                describe(&mats)
            );
            let (Some(out), true) = (&a.out, patch.contains(&layer)) else {
                continue;
            };
            let Some(fixed) = restore_flag(&bytes)? else {
                continue;
            };
            let folder = dir
                .file_name()
                .context("a --patch container has no folder name")?;
            let path = name
                .split('\\')
                .fold(out.join(folder), |p, part| p.join(part));
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(&path, &fixed).with_context(|| format!("writing {}", path.display()))?;
            println!(
                "  wrote {}: [{}]",
                path.display(),
                describe(&materials(&fixed)?)
            );
            written += 1;
        }
    }
    let missing: Vec<&str> = SIDN_FLAG_LOST
        .iter()
        .copied()
        .filter(|f| !found.contains(*f))
        .collect();
    eprintln!(
        "{} of {} listed roots on the stack, {written} rewritten{}",
        found.len(),
        SIDN_FLAG_LOST.len(),
        if missing.is_empty() {
            String::new()
        } else {
            format!("; not found: {}", missing.join(", "))
        }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(materials: &[(u32, [u8; 4])]) -> Vec<u8> {
        let chunk = |magic: &[u8; 4], body: &[u8]| {
            let mut out = magic.to_vec();
            out.extend((body.len() as u32).to_le_bytes());
            out.extend(body);
            out
        };
        let mut momt = Vec::new();
        for (i, (flags, bgra)) in materials.iter().enumerate() {
            let mut m = [i as u8 + 1; MOMT_LEN];
            m[..4].copy_from_slice(&flags.to_le_bytes());
            m[SIDN_AT..SIDN_AT + 4].copy_from_slice(bgra);
            momt.extend(m);
        }
        let mut bytes = chunk(b"REVM", &17u32.to_le_bytes());
        bytes.extend(chunk(b"XTOM", b"a.blp\0\0\0"));
        bytes.extend(chunk(b"TMOM", &momt));
        bytes.extend(chunk(b"NGOM", b"g\0\0\0"));
        bytes
    }

    /// Only bit `0x10`, only on a coloured material, and no other byte.
    #[test]
    fn restoring_the_flag_sets_one_bit_on_the_coloured_glass() {
        let bytes = root(&[
            (0, [3, 2, 1, 255]),
            (0x4, [0, 0, 0, 255]),
            (0x10 | 0x1, [9, 9, 9, 255]),
            (0x40, [0, 0, 7, 0]),
        ]);
        let fixed = restore_flag(&bytes)
            .unwrap()
            .expect("two materials lost the flag");
        let before = materials(&bytes).unwrap();
        let after = materials(&fixed).unwrap();
        assert_eq!(
            after.iter().map(|m| m.flags).collect::<Vec<_>>(),
            [0x10, 0x4, 0x11, 0x50]
        );
        assert_eq!(after[0].sidn, [1, 2, 3]);
        let changed: Vec<usize> = (0..bytes.len()).filter(|&i| bytes[i] != fixed[i]).collect();
        assert_eq!(changed, [before[0].at, before[3].at]);
        assert_eq!(restore_flag(&fixed).unwrap(), None);
    }

    #[test]
    fn the_list_matches_a_root_by_file_name_in_any_case() {
        assert!(is_listed(
            r"World\wmo\Azeroth\Buildings\Stormwind\StormwindHarbor.wmo"
        ));
        assert!(is_listed(
            "world/wmo/azeroth/buildings/stormwind/stormwind.wmo"
        ));
        assert!(!is_listed(
            r"World\wmo\Azeroth\Buildings\Stormwind\Stormwind_000.wmo"
        ));
        assert!(!is_listed(
            r"World\wmo\Kalimdor\Buildings\DSNightElfHuntersLodge.wmo"
        ));
    }

    /// **The packed layers carry the flag.** Through the Turtle layer, and the reforged layer over
    /// it, every listed root resolves and none has a coloured material the flag does not light, so
    /// the client needs no list of its own.
    #[test]
    fn the_packed_roots_that_lost_their_sidn_flag_have_it_back() {
        let Some(base) = benilla_pack::find::find() else {
            eprintln!(
                "skipping: no BENPACK container — looked in {:?}",
                benilla_pack::find::candidates()
            );
            return;
        };
        let layer = |name: &str| {
            base.parent()
                .map(|p| p.join(name))
                .filter(|d| d.join("index.bpi").is_file())
        };
        let (Some(turtle), Some(reforged)) = (layer("packs-turtle"), layer("packs-reforged"))
        else {
            eprintln!(
                "skipping: no packs-turtle and packs-reforged beside {}",
                base.display()
            );
            return;
        };
        for dirs in [
            vec![base.clone(), turtle.clone()],
            vec![base.clone(), turtle, reforged],
        ] {
            let stack = Stack::open(&dirs).unwrap();
            let mut names: Vec<String> = (0..stack.layer_count())
                .flat_map(|l| listed_roots(&stack, l))
                .collect();
            let mut seen = HashSet::new();
            names.retain(|n| seen.insert(name_key(n)));
            let mut files = HashSet::new();
            for name in &names {
                let bytes = stack.read(name).unwrap().expect("a listed root reads");
                let lost: Vec<Material> = materials(&bytes)
                    .unwrap()
                    .into_iter()
                    .filter(Material::glass_unflagged)
                    .collect();
                assert!(
                    lost.is_empty(),
                    "{dirs:?}: {name} has unflagged glass {lost:?}"
                );
                files.insert(name.rsplit('\\').next().unwrap().to_ascii_lowercase());
            }
            assert_eq!(
                files.len(),
                SIDN_FLAG_LOST.len(),
                "{dirs:?}: listed roots found"
            );
        }
    }
}
