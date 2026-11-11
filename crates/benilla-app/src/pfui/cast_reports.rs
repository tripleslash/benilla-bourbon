//! Nampower's attempt reads: the refusals the cast reports hold for a spell, which a call's
//! attempt left before its host signals them.

impl super::CastReports {
    /// How many refusals of `spell` wait to be signalled, and the newest one's reason.
    pub(crate) fn pfui_refusals(&self, spell: u32) -> (usize, Option<u8>) {
        self.0
            .iter()
            .filter_map(|(_, report)| match report {
                super::CastReport::Refused(f) if f.spell_id == spell => Some(f.reason),
                _ => None,
            })
            .fold((0, None), |(n, _), reason| (n + 1, Some(reason)))
    }
}
