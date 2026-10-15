//! fork (Everwood port): the Graphics Preset ladder, `graphicsQuality`.
//!
//! The row is not a knob: it is a name for a combination of the graphics rows
//! ([`GRAPHICS_PRESETS`]). Writing a rung's name writes that rung's column; writing anything else
//! is corrected on the next sync, which re-derives the name from the values the rows hold
//! ([`derive_graphics_quality`]). So the stored value is never stale: it is a rung every governed
//! row agrees with, or `Custom`.
//!
//! The last rung, Cinematic, is the registered defaults, so a config from before the ladder and a
//! new player both read Cinematic with nothing seeded. Classic is the reference client: the
//! classic lighting model and the reference's own value on every row that deviates from it.

use super::{Cvars, SetOutcome};

/// What the ladder shows when the rows match no rung. Not a rung: selecting it writes nothing,
/// and the next derivation puts back whatever the rows really say.
pub(crate) const GRAPHICS_CUSTOM: &str = "Custom";

/// The row's name.
pub(crate) const GRAPHICS_QUALITY: &str = "graphicsQuality";

/// The rungs, in the order [`derive_graphics_quality`] tries them and the column order of
/// [`GRAPHICS_PRESETS`].
pub(crate) const GRAPHICS_PRESET_NAMES: [&str; 6] =
    ["Classic", "Low", "Medium", "High", "Ultra", "Cinematic"];

/// The one place the ladder is written down: one line per governed row, one column per rung.
///
/// Left out: `spellEffectLevel` and `foliageWindStrength` (the same on every
/// rung) and the rows latched at boot (`anisotropic`, `gxMultisample`, `gxVSync`). With frame
/// generation on, render scale is capped at 1, which every column respects. A row that lands
/// later is one line here, once it is registered: an unregistered member never matches, and the
/// ladder would read Custom for ever.
#[rustfmt::skip]
pub(crate) const GRAPHICS_PRESETS: &[(&str, [&str; 6])] = &[
    //                     Classic  Low     Medium  High   Ultra   Cinematic
    ("lightingModel",     ["0",    "1",    "1",    "1",   "1",    "1"]),
    ("lightingQuality",   ["0",    "0",    "1",    "2",   "3",    "4"]),
    ("farclip",           ["350",  "350",  "477",  "777", "1000", "1000"]),
    ("WorldDetail",       ["1",    "1",    "2",    "2",   "2",    "2"]),
    ("weatherDensity",    ["2",    "3",    "3",    "3",   "3",    "3"]),
    ("worldShadows",      ["0",    "1",    "1",    "1",   "1",    "1"]),
    ("ssaoStrength",      ["0",    "0",    "0.4",  "0.4", "0.4",  "0.4"]),
    ("ssgiStrength",      ["0",    "0",    "0",    "0",   "0",    "0"]),
    ("waterDetail",       ["0",    "1",    "2",    "3",   "3",    "3"]),
    ("hdCharacterModels", ["0",    "1",    "1",    "1",   "1",    "1"]),
    ("reforgedModels",    ["0",    "1",    "1",    "1",   "1",    "1"]),
    ("renderScale",       ["1",    "0.67", "0.67", "1",   "1",    "1"]),
    ("frameGeneration",   ["0",    "1",    "1",    "1",   "1",    "1"]),
    ("foliageWind",       ["0",    "1",    "2",    "2",   "2",    "2"]),
    ("rainSurfaces",      ["0",    "0",    "1",    "1",   "1",    "1"]),
    ("lavaLight",         ["0",    "1",    "1",    "1",   "1",    "1"]),
];

/// Two registry values are the same setting when they are the same number: `"0"`, `"0.00"` and
/// `"0.0"` all mean a lane that is off, and a rung whose member reads `"1"` is not Custom because
/// a slider wrote `"1.00"`. Falls back to a case-insensitive compare for rows that hold words.
fn same_value(a: &str, b: &str) -> bool {
    match (a.trim().parse::<f32>(), b.trim().parse::<f32>()) {
        (Ok(x), Ok(y)) => (x - y).abs() < 1e-4,
        _ => a.eq_ignore_ascii_case(b),
    }
}

/// The column of a rung, matched case-insensitively; `None` for `Custom` or anything else.
fn graphics_column(name: &str) -> Option<usize> {
    GRAPHICS_PRESET_NAMES
        .iter()
        .position(|n| n.eq_ignore_ascii_case(name.trim()))
}

/// The rung every governed row currently spells, or [`GRAPHICS_CUSTOM`]. A session-owned row (an
/// env lever, the web GPU recovery tier's `lightingQuality` cap) is the session's, not the
/// player's, and is skipped.
pub(crate) fn derive_graphics_quality(cvars: &Cvars) -> &'static str {
    for (col, name) in GRAPHICS_PRESET_NAMES.iter().enumerate() {
        let matched = GRAPHICS_PRESETS
            .iter()
            .filter(|(k, _)| !cvars.is_session_owned(k))
            .all(|(k, v)| cvars.get(k).is_some_and(|live| same_value(live, v[col])));
        if matched {
            return name;
        }
    }
    GRAPHICS_CUSTOM
}

/// Write one rung's column as the player's own writes; `None` (and nothing written) for an
/// unknown name, `Custom` included, else how many rows moved.
pub(crate) fn apply_graphics_preset(cvars: &mut Cvars, name: &str) -> Option<usize> {
    let col = graphics_column(name)?;
    let mut moved = 0;
    for (k, values) in GRAPHICS_PRESETS {
        if cvars.set_as_player(k, values[col]) == SetOutcome::Changed {
            moved += 1;
        }
    }
    Some(moved)
}

impl Cvars {
    /// A host-side write that stands for the player's own: it takes a row the GPU recovery tier
    /// holds back from the session, as a write through the VM does, so a rung the player picked
    /// is applied in full and saved.
    fn set_as_player(&mut self, name: &str, value: &str) -> SetOutcome {
        let key = name.to_ascii_lowercase();
        if self.yielding.remove(&key).is_some() {
            self.session_owned.remove(&key);
            self.touch();
        }
        self.write(name, value, false)
    }

    /// The derived rung, when the row does not already say it. Read before
    /// [`Self::mirror_graphics_quality`], so a quiet frame never deref-muts the registry.
    pub(crate) fn stale_graphics_quality(&self) -> Option<&'static str> {
        let label = derive_graphics_quality(self);
        self.get(GRAPHICS_QUALITY)
            .is_some_and(|v| v != label)
            .then_some(label)
    }

    /// Mirror the derived rung into the row: a label, never another request to apply a rung.
    pub(crate) fn mirror_graphics_quality(&mut self) {
        if let Some(label) = self.stale_graphics_quality() {
            self.mirror(GRAPHICS_QUALITY, label);
        }
    }

    /// Re-apply the saved rung to the rows the file does not carry. A saved rung name means every
    /// governed row matched it when the file was written, and a row at its registered default is
    /// not saved; so a governed row absent from the file was registered after the save, and it
    /// takes the saved rung's value, not its own default. Rows the file carries and session-owned
    /// rows are never touched; `Custom`, or no saved name, writes nothing. Returns how many moved.
    pub(crate) fn reapply_saved_presets(&mut self) -> usize {
        let carried = |cvars: &Self, k: &str| {
            cvars.file.keys().any(|f| f.eq_ignore_ascii_case(k)) || cvars.is_session_owned(k)
        };
        if self.is_session_owned(GRAPHICS_QUALITY) {
            return 0;
        }
        let Some(col) = self
            .file
            .iter()
            .find(|(f, _)| f.eq_ignore_ascii_case(GRAPHICS_QUALITY))
            .and_then(|(_, v)| graphics_column(v))
        else {
            return 0;
        };
        let mut moved = 0;
        for (k, values) in GRAPHICS_PRESETS {
            if carried(self, k) {
                continue;
            }
            if self.write(k, values[col], false) == SetOutcome::Changed {
                moved += 1;
            }
        }
        moved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn column(name: &str) -> usize {
        graphics_column(name).expect("a rung")
    }

    /// Every governed row is registered, unlatched and on the list a Graphics Apply is known by
    /// (the web GPU recovery tier retires on it), and so is the ladder's own row.
    #[test]
    fn every_governed_row_is_a_registered_graphics_row() {
        let cvars = Cvars::default();
        for (k, values) in GRAPHICS_PRESETS {
            let row = cvars
                .row(k)
                .unwrap_or_else(|| panic!("{k}: not registered"));
            assert!(!row.latched, "{k}: a latched row cannot follow a rung");
            for v in values {
                assert!(
                    !row.numeric() || v.parse::<f32>().is_ok(),
                    "{k}: '{v}' would be refused"
                );
            }
            assert!(
                super::super::GRAPHICS_CVARS
                    .iter()
                    .any(|c| c.eq_ignore_ascii_case(k)),
                "{k}: governed, but not in GRAPHICS_CVARS"
            );
        }
        assert!(super::super::GRAPHICS_CVARS.contains(&GRAPHICS_QUALITY));
        assert!(!cvars.row(GRAPHICS_QUALITY).expect("registered").numeric());
    }

    /// Cinematic is the registered defaults, so an existing config and a new player both read it
    /// with nothing seeded.
    #[test]
    fn todays_defaults_read_cinematic() {
        let cvars = Cvars::default();
        let col = column("Cinematic");
        for (k, values) in GRAPHICS_PRESETS {
            assert!(
                same_value(cvars.default_of(k).expect("registered"), values[col]),
                "{k}: Cinematic is {} but the row registers {:?}",
                values[col],
                cvars.default_of(k)
            );
        }
        assert_eq!(derive_graphics_quality(&cvars), "Cinematic");
        assert_eq!(cvars.default_of(GRAPHICS_QUALITY), Some("Cinematic"));
    }

    /// No two rungs spell the same rows, or the earlier one would shadow the later.
    #[test]
    fn the_rungs_are_pairwise_distinct() {
        for a in 0..GRAPHICS_PRESET_NAMES.len() {
            for b in a + 1..GRAPHICS_PRESET_NAMES.len() {
                assert!(
                    GRAPHICS_PRESETS
                        .iter()
                        .any(|(_, v)| !same_value(v[a], v[b])),
                    "{} and {} are the same column",
                    GRAPHICS_PRESET_NAMES[a],
                    GRAPHICS_PRESET_NAMES[b]
                );
            }
        }
    }

    #[test]
    fn every_rung_round_trips_from_every_other() {
        for from in GRAPHICS_PRESET_NAMES {
            for name in GRAPHICS_PRESET_NAMES {
                let mut cvars = Cvars::default();
                cvars.set(GRAPHICS_QUALITY, from);
                assert_eq!(derive_graphics_quality(&cvars), from);
                cvars.set(GRAPHICS_QUALITY, name);
                assert_eq!(derive_graphics_quality(&cvars), name, "{from} -> {name}");
                cvars.mirror_graphics_quality();
                assert_eq!(cvars.get(GRAPHICS_QUALITY), Some(name));
                let col = column(name);
                for (k, values) in GRAPHICS_PRESETS {
                    assert!(
                        same_value(cvars.get(k).unwrap(), values[col]),
                        "{name}: {k}"
                    );
                }
            }
        }
    }

    /// A slider's spelling of a number is the same setting.
    #[test]
    fn a_number_spelled_another_way_still_matches() {
        let mut cvars = Cvars::default();
        cvars.set(GRAPHICS_QUALITY, "High");
        cvars.set("renderScale", "1.00");
        cvars.set("ssaoStrength", "0.40");
        assert_eq!(derive_graphics_quality(&cvars), "High");
    }

    #[test]
    fn one_member_off_reads_custom() {
        for (col, name) in GRAPHICS_PRESET_NAMES.iter().enumerate() {
            for (k, values) in GRAPHICS_PRESETS {
                let mut cvars = Cvars::default();
                cvars.set(GRAPHICS_QUALITY, name);
                // A value this column does not hold and the row's callback takes (`farclip`
                // refuses anything under 177).
                let off = ["0.5", "0.25", "600"]
                    .into_iter()
                    .filter(|v| !same_value(values[col], v))
                    .find(|v| cvars.set(k, v) == SetOutcome::Changed)
                    .unwrap_or_else(|| panic!("{k} takes no off value"));
                let derived = derive_graphics_quality(&cvars);
                assert_eq!(derived, GRAPHICS_CUSTOM, "{name} with {k} = {off}");
                cvars.mirror_graphics_quality();
                assert_eq!(cvars.get(GRAPHICS_QUALITY), Some(GRAPHICS_CUSTOM));
            }
        }
    }

    #[test]
    fn custom_and_unknown_names_write_nothing() {
        let mut cvars = Cvars::default();
        cvars.set(GRAPHICS_QUALITY, "Medium");
        let before: Vec<String> = GRAPHICS_PRESETS
            .iter()
            .map(|(k, _)| cvars.get(k).unwrap().to_string())
            .collect();
        assert_eq!(apply_graphics_preset(&mut cvars, GRAPHICS_CUSTOM), None);
        assert_eq!(apply_graphics_preset(&mut cvars, "Epic"), None);
        cvars.set(GRAPHICS_QUALITY, GRAPHICS_CUSTOM);
        let after: Vec<String> = GRAPHICS_PRESETS
            .iter()
            .map(|(k, _)| cvars.get(k).unwrap().to_string())
            .collect();
        assert_eq!(before, after);
        // The label is put back by the next derivation.
        cvars.mirror_graphics_quality();
        assert_eq!(cvars.get(GRAPHICS_QUALITY), Some("Medium"));
    }

    /// A write through the VM (the options dropdown) applies the rung at its place in the queue,
    /// so a member edit queued behind it wins, and counts as a Graphics Apply.
    #[test]
    fn a_vm_write_applies_the_rung_and_a_later_member_edit_wins() {
        let mut cvars = Cvars::default();
        cvars.set_from_vm(GRAPHICS_QUALITY, "Low");
        cvars.set_from_vm("waterDetail", "3");
        assert!(cvars.take_graphics_applied());
        assert_eq!(cvars.get("lightingQuality"), Some("0"));
        assert_eq!(cvars.get("waterDetail"), Some("3"));
        cvars.mirror_graphics_quality();
        assert_eq!(cvars.get(GRAPHICS_QUALITY), Some(GRAPHICS_CUSTOM));
    }

    /// The GPU recovery tier's cap is the session's: the derive skips it, and a rung the player
    /// applies takes the row back and saves it.
    #[test]
    fn a_rung_takes_a_capped_row_back_from_the_recovery_tier() {
        let mut cvars = Cvars::default();
        assert!(cvars.cap_for_session("lightingQuality", "1"));
        assert_eq!(
            derive_graphics_quality(&cvars),
            "Ultra",
            "the capped row is skipped, and Ultra is the first rung the rest spell"
        );
        cvars.set_from_vm(GRAPHICS_QUALITY, "High");
        assert!(!cvars.is_session_owned("lightingQuality"));
        assert_eq!(cvars.get("lightingQuality"), Some("2"));
        assert_eq!(
            cvars.compose().get("lightingQuality").map(String::as_str),
            Some("2")
        );
        // The tier's release has nothing left to restore over the player's rung.
        assert_eq!(cvars.release_fallbacks(), 0);
        assert_eq!(derive_graphics_quality(&cvars), "High");
    }

    /// A saved rung fills the governed rows the file does not carry (rows registered after the
    /// save); rows it carries stay the player's, and Custom fills nothing.
    #[test]
    fn reapply_fills_rows_missing_from_the_file() {
        let mut cvars = Cvars::default();
        cvars.load_file(BTreeMap::from([
            (GRAPHICS_QUALITY.to_string(), "Low".to_string()),
            ("waterDetail".to_string(), "3".to_string()),
        ]));
        assert!(cvars.reapply_saved_presets() > 0);
        let col = column("Low");
        for (k, values) in GRAPHICS_PRESETS {
            if *k == "waterDetail" {
                assert_eq!(cvars.get(k), Some("3"), "the file's own value stays");
            } else {
                assert!(same_value(cvars.get(k).unwrap(), values[col]), "{k}");
            }
        }

        let mut custom = Cvars::default();
        custom.load_file(BTreeMap::from([(
            GRAPHICS_QUALITY.to_string(),
            GRAPHICS_CUSTOM.to_string(),
        )]));
        assert_eq!(custom.reapply_saved_presets(), 0);

        let mut none = Cvars::default();
        none.load_file(BTreeMap::from([(
            "waterDetail".to_string(),
            "0".to_string(),
        )]));
        assert_eq!(none.reapply_saved_presets(), 0);
    }
}
