//! The optional Nampower clock reads the GCD leg, never the longest spell cooldown.

use bevy::platform::time::Instant;
use std::time::Duration;

impl super::Record {
    /// What is left of the record's GCD, which runs from its recovery's start.
    fn gcd_remaining(&self, now: Instant) -> Duration {
        (self.recovery.start + self.gcd).saturating_duration_since(now)
    }
}

impl super::Cooldowns {
    pub(crate) fn pfui_gcd_remaining_ms(&self, category: u32) -> f64 {
        let now = Instant::now();
        self.records
            .iter()
            .filter(|record| record.gcd_category == category)
            .map(|record| record.gcd_remaining(now).as_secs_f64() * 1000.0)
            .fold(0.0, f64::max)
    }

    pub(crate) fn pfui_gcd_end(&self, category: u32, clock: &crate::ui_script::UiClock) -> f64 {
        let now = Instant::now();
        self.records
            .iter()
            .filter(|record| {
                record.gcd_category == category && !record.gcd_remaining(now).is_zero()
            })
            .max_by_key(|record| record.gcd_remaining(now))
            .map_or(0.0, |record| {
                // The GCD runs from the recovery's start ([`super::Cooldowns`]' `Record::gcd`).
                let end = record.recovery.start + record.gcd;
                let offset = if end >= clock.anchor {
                    end.duration_since(clock.anchor).as_secs_f64()
                } else {
                    -clock.anchor.duration_since(end).as_secs_f64()
                };
                clock.ui_now + offset
            })
    }
}
