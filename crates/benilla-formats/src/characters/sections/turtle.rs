//! Turtle's appended player skin colours from the flagged CharSections skin rows.
//!
//! Deviation: (tortoise) the flagged skin rows are extra colours, because Turtle's CharSections.dbc
//! stores extra body skins as skin rows with flag bit 0 set and on-disk colour 0, and its server
//! sends players colours past the last normal row (a High Elf female's 17 against normal colours 0 to 10). Those colours are the flagged
//! rows in file order: race 10 female has 8, colours 11 to 18. 1.12.1 has no such bank and fails
//! the component build there (`0x476b90`). Inferred from the data and a player report, not from
//! Turtle's executable. After TuringKi/benilla 2c38e058.

use super::*;

impl CharSections {
    /// Read Turtle's extended player skin bank as well as the standard lookups. Its Scourge
    /// female colour 6 uses the flagged BloodWidow skin row with on-disk colour 0; the face
    /// rows already use colour 6. Keep standard colours and give extra body skins consecutive
    /// colours after them, without losing their original NPC lookup keys.
    pub fn load_turtle(chain: &mut Chain) -> Result<Self> {
        let mut sections = Self::load(chain)?;
        let bytes = chain.read(CHAR_SECTIONS)?;
        let rs = parse(&bytes, char_sections_schema(), "CharSections")?;
        let mut normal = HashMap::<(u8, u8), u16>::new();
        let mut extra = Vec::new();
        for r in rs.records() {
            if u32_at(r, 3) != Some(u32::from(SECTION_SKIN)) || u32_at(r, 4) != Some(0) {
                continue;
            }
            let (Some(race), Some(sex), Some(color)) = (u32_at(r, 1), u32_at(r, 2), u32_at(r, 5))
            else {
                continue;
            };
            let bank = (race as u8, sex as u8);
            let textures = std::array::from_fn(|i| str_at(&rs, r, i + 6).unwrap_or_default());
            if u32_at(r, 9).unwrap_or(0) & 1 == 0 {
                normal
                    .entry(bank)
                    .and_modify(|n| *n = (*n).max(color as u16 + 1))
                    .or_insert(color as u16 + 1);
            } else {
                extra.push((bank, textures));
            }
        }
        sections.append_turtle_skins(normal, extra);
        Ok(sections)
    }

    fn append_turtle_skins(
        &mut self,
        mut next: HashMap<(u8, u8), u16>,
        rows: impl IntoIterator<Item = ((u8, u8), [String; 3])>,
    ) {
        for ((race, sex), textures) in rows {
            if textures[0].is_empty() {
                continue;
            }
            let Some(color) = next.get_mut(&(race, sex)) else {
                continue;
            };
            // An extra row is a distinct texture, not a second spelling of a normal skin.
            if (0..*color).any(|c| {
                self.skin_texture(race, sex, c as u8)
                    .is_some_and(|p| p.eq_ignore_ascii_case(&textures[0]))
            }) {
                continue;
            }
            if u8::try_from(*color)
                .ok()
                .and_then(|c| self.skin_texture(race, sex, c))
                .is_some_and(|p| p.eq_ignore_ascii_case(&textures[0]))
            {
                *color += 1;
                continue;
            }
            while *color <= u16::from(u8::MAX)
                && self
                    .sections
                    .contains_key(&(race, sex, SECTION_SKIN, 0, *color as u8))
            {
                *color += 1;
            }
            if let Ok(index) = u8::try_from(*color) {
                self.sections
                    .insert((race, sex, SECTION_SKIN, 0, index), textures);
                *color += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textures(name: &str) -> [String; 3] {
        [name.into(), String::new(), String::new()]
    }

    #[test]
    fn an_extra_skin_does_not_replace_colour_zero_and_can_composite_gear() {
        let mut sections = CharSections {
            sections: HashMap::from([
                ((5, 1, SECTION_SKIN, 0, 0), textures("normal0.blp")),
                ((5, 1, SECTION_SKIN, 0, 1), textures("normal1.blp")),
                ((5, 1, SECTION_FACE, 0, 2), textures("extra-face.blp")),
            ]),
        };
        // Before the bank: colour 2 has no row and composites on colour 0 (the #2258 carry).
        assert_eq!(
            sections
                .composite_plan(5, 1, 2, 0, 0, 0, 0, [None; 8], None, false)
                .map(|p| p.base),
            Some("normal0.blp".into())
        );
        sections.append_turtle_skins(
            HashMap::from([((5, 1), 2)]),
            [((5, 1), textures("extra.blp"))],
        );
        assert_eq!(sections.skin_texture(5, 1, 0), Some("normal0.blp"));
        assert_eq!(sections.skin_texture(5, 1, 1), Some("normal1.blp"));
        assert_eq!(sections.skin_texture(5, 1, 2), Some("extra.blp"));
        let mut chest = ItemDisplay::default();
        chest.region_textures[3] = Some("chest".into());
        let mut worn = [None; 8];
        worn[1] = Some(&chest);
        let plan = sections
            .composite_plan(5, 1, 2, 0, 0, 0, 0, worn, None, false)
            .unwrap();
        assert_eq!(plan.base, "extra.blp");
        assert!(plan
            .layers
            .iter()
            .any(|(paths, _)| paths.iter().any(|p| p.contains("chest"))));
        assert!(plan
            .layers
            .iter()
            .any(|(paths, _)| paths.iter().any(|p| p == "extra-face.blp")));
    }

    #[test]
    fn skin_banks_are_independent_and_duplicate_textures_do_not_shift_colours() {
        let mut sections = CharSections {
            sections: HashMap::from([
                ((5, 1, SECTION_SKIN, 0, 0), textures("normal.blp")),
                ((1, 0, SECTION_SKIN, 0, 0), textures("human.blp")),
            ]),
        };
        sections.append_turtle_skins(
            HashMap::from([((5, 1), 1), ((1, 0), 1)]),
            [
                ((5, 1), textures("NORMAL.blp")),
                ((1, 0), textures("extra-human.blp")),
                ((5, 1), textures("widow.blp")),
                ((5, 1), textures("WIDOW.blp")),
                ((5, 1), textures("next.blp")),
            ],
        );
        assert_eq!(sections.skin_texture(1, 0, 1), Some("extra-human.blp"));
        assert_eq!(sections.skin_texture(5, 1, 1), Some("widow.blp"));
        assert_eq!(sections.skin_texture(5, 1, 2), Some("next.blp"));
    }

    #[test]
    fn an_already_numbered_extra_skin_does_not_take_two_colours() {
        let mut sections = CharSections {
            sections: HashMap::from([
                ((5, 1, SECTION_SKIN, 0, 0), textures("normal.blp")),
                ((5, 1, SECTION_SKIN, 0, 1), textures("widow.blp")),
            ]),
        };
        sections.append_turtle_skins(
            HashMap::from([((5, 1), 1)]),
            [
                ((5, 1), textures("widow.blp")),
                ((5, 1), textures("next.blp")),
            ],
        );
        assert_eq!(sections.skin_texture(5, 1, 1), Some("widow.blp"));
        assert_eq!(sections.skin_texture(5, 1, 2), Some("next.blp"));
        assert_eq!(sections.skin_texture(5, 1, 3), None);
    }
}
