//! The game clock's calendar time, which only the optional extension reads.

impl super::game_time::GameTime {
    /// The clock's date and minute as a local Unix time, once a server has given it one: the day
    /// count is the packed date's own serial (`year·372 + month·31 + day`).
    pub(crate) fn pfui_local_epoch(&self) -> Option<f64> {
        if !self.live() {
            return None;
        }
        let serial = u32::try_from(self.day_count()).ok()?;
        benilla_pfui::clock::epoch(serial, self.minute_of_day())
    }
}
