//! A stack of built BENPACK containers read the way a client mounts them, lowest first, each
//! through its own `index.bpi`: the data a layer serves, with no install to read it from.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use benilla_mpq::name_key;
use benilla_pack::{decode_pack_body, frame_from_body, pack_body, pack_path, Index};

/// The containers in mount order, each with its index.
pub struct Stack {
    layers: Vec<(PathBuf, Index)>,
}

impl Stack {
    pub fn open(dirs: &[PathBuf]) -> Result<Self> {
        let layers = dirs
            .iter()
            .map(|d| {
                let path = d.join("index.bpi");
                let bytes =
                    std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                let index = Index::parse(bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok((d.clone(), index))
            })
            .collect::<Result<_>>()?;
        Ok(Self { layers })
    }

    /// Which container answers `name`: the index of the highest that holds it.
    pub fn holder(&self, name: &str) -> Option<usize> {
        let key = name_key(name);
        self.layers
            .iter()
            .rposition(|(_, index)| index.locate(key).is_some())
    }

    pub fn dir(&self, layer: usize) -> &Path {
        &self.layers[layer].0
    }

    /// How many containers the stack mounts.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// `name`'s bytes from the highest container that holds it.
    pub fn read(&self, name: &str) -> Result<Option<Vec<u8>>> {
        for layer in (0..self.layers.len()).rev() {
            if let Some(bytes) = self.read_in(layer, name)? {
                return Ok(Some(bytes));
            }
        }
        Ok(None)
    }

    /// `name`'s bytes from container `layer` alone.
    pub fn read_in(&self, layer: usize, name: &str) -> Result<Option<Vec<u8>>> {
        let (dir, index) = &self.layers[layer];
        let Some(at) = index.locate(name_key(name)) else {
            return Ok(None);
        };
        let meta = index
            .pack(at.pack_id)
            .with_context(|| format!("{name}: pack {} not in the index", at.pack_id))?;
        let wire = std::fs::read(dir.join(pack_path(at.pack_id)))
            .with_context(|| format!("{name}: reading pack {}", at.pack_id))?;
        let body = pack_body(&wire, at.pack_id).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
        let body = decode_pack_body(body, &meta).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
        Ok(Some(
            frame_from_body(&body, &at).map_err(|e| anyhow::anyhow!("{name}: {e}"))?,
        ))
    }

    /// The names container `layer`'s `listfile.txt` holds.
    pub fn names(&self, layer: usize) -> Vec<String> {
        let text =
            std::fs::read_to_string(self.layers[layer].0.join("listfile.txt")).unwrap_or_default();
        text.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// Every map directory a container's listfile names a `<dir>\<dir>.wdt` in, by lowercase name,
    /// spelled as the highest listfile spells it.
    pub fn maps(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for layer in 0..self.layers.len() {
            for line in self.names(layer) {
                let lower = line.to_ascii_lowercase();
                let Some(rest) = lower.strip_prefix("world\\maps\\") else {
                    continue;
                };
                let Some((map, file)) = rest.split_once('\\') else {
                    continue;
                };
                if file.strip_suffix(".wdt") == Some(map) {
                    out.insert(map.to_string(), line[11..11 + map.len()].to_string());
                }
            }
        }
        out
    }
}
