//! Aura duration packets use real time; their Lua expiration must keep a fixed GetTime epoch.

use bevy::prelude::{Real, Time};

/// This frame's `GetTime`, mapped through the frame's [`crate::ui_script::UiClock`] pair: the aura
/// feed runs before the VM tick, so `script.now()` alone is still the previous frame's value and
/// subtracting this frame's real time from it would shift every expiration by the frame delta.
pub(super) fn script_now(
    time: &Time<Real>,
    clock: Option<&crate::ui_script::UiClock>,
    fallback: f64,
) -> f64 {
    let Some((now, clock)) = time.last_update().zip(clock) else {
        return fallback;
    };
    let delta = if now >= clock.anchor {
        now.duration_since(clock.anchor).as_secs_f64()
    } else {
        -clock.anchor.duration_since(now).as_secs_f64()
    };
    clock.ui_now + delta
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::platform::time::Instant;
    use std::time::Duration;

    #[test]
    fn an_unchanged_duration_keeps_its_expiration_through_uneven_frames_and_hitches() {
        let start = Instant::now();
        let mut time = Time::<Real>::new(start);
        time.update_with_instant(start);
        let mut clock = crate::ui_script::UiClock {
            anchor: start,
            ui_now: 100.0,
        };
        let stamp = crate::ui_aura::DurationStamp {
            total: 15.0,
            received_at: 1.0,
            expires_at: 16.0,
        };
        for millis in [1000, 1200, 4500, 4550] {
            let next = start + Duration::from_millis(millis);
            time.update_with_instant(next);
            let now = script_now(&time, Some(&clock), clock.ui_now);
            let (duration, expiry) =
                crate::ui_aura::join_duration(Some(&stamp), 1.0, time.elapsed_secs_f64(), now);
            assert_eq!(duration, 15.0);
            assert!((expiry - 116.0).abs() < 1e-9, "frame {millis}: {expiry}");
            clock.anchor = next;
            clock.ui_now = now;
        }
        let refreshed = crate::ui_aura::DurationStamp {
            total: 15.0,
            received_at: 4.55,
            expires_at: 19.55,
        };
        let (_, expiry) = crate::ui_aura::join_duration(
            Some(&refreshed),
            1.0,
            time.elapsed_secs_f64(),
            script_now(&time, Some(&clock), clock.ui_now),
        );
        assert!((expiry - 119.55).abs() < 1e-9);
    }

    #[test]
    fn standalone_aura_feeds_can_run_before_a_clock_or_first_frame_exists() {
        let time = Time::<Real>::default();
        assert_eq!(script_now(&time, None, 12.0), 12.0);
        assert_eq!(
            script_now(&time, Some(&crate::ui_script::UiClock::default()), 13.0),
            13.0
        );
    }
}
