//! A bone's two tracks as op4 `0x7121a0` keeps them: the primary and its one secondary blend slot.
//! The world's units ride it on bone 0 and the character pane's copies on their key bone, so both
//! arm through [`BlendSlot::arm`] and fade through [`BlendSlot::advance`].

use benilla_assets::AnimClip;
use bevy::animation::graph::AnimationNodeIndex;
use bevy::animation::ActiveAnimation;
use bevy::prelude::*;

use super::select::blend_lambda;

/// The primary (`[blk+0x98]`) and the secondary blend slot (`[blk+0xc4]`), read on the scene clock
/// op4 reads, stamped once a frame by [`Self::set_clock`].
#[derive(Component, Clone, Copy, Default, Debug)]
pub(crate) struct BlendSlot {
    /// The node the primary runs, either of its clip's two tracks.
    primary: Option<AnimationNodeIndex>,
    fade: Option<Fade>,
    /// The latch `[+0xc0]`: the primary's window ran out (`0x7194bc`), cleared by the next arm.
    latched: bool,
    now: f64,
}

/// The secondary slot: the pose an arm replaced, on its own clock and a track the primary is not
/// using, faded out by λ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Fade {
    pub(crate) node: AnimationNodeIndex,
    /// When λ reaches 0 (`[blk+0x100]`).
    pub(crate) until: f64,
    /// The incoming clip's `M2Sequence.blendTime` (`0x7125f2`); the rate `[+0x104]` is its inverse.
    pub(crate) span: f32,
}

impl Fade {
    /// The fade still to run, 1 to 0: `(blendEnd − now) · blendRate` (`0x7148b3`).
    fn frac(&self, now: f64) -> f32 {
        ((self.until - now) / f64::from(self.span)) as f32
    }
}

impl BlendSlot {
    /// A slot whose primary already runs `node` with no arm behind it: a loader's idle seed
    /// (`0x70ebd0`), which the first arm fades out of.
    pub(crate) fn seeded(node: Option<AnimationNodeIndex>) -> Self {
        BlendSlot {
            primary: node,
            ..default()
        }
    }

    /// The node the primary runs.
    pub(crate) fn primary(&self) -> Option<AnimationNodeIndex> {
        self.primary
    }

    /// The secondary slot, as the tests read it.
    #[cfg(test)]
    pub(crate) fn fade(&self) -> Option<Fade> {
        self.fade
    }

    /// Stamps the scene clock the next arm and advance read, in seconds.
    pub(crate) fn set_clock(&mut self, now: f64) {
        self.now = now;
    }

    /// Sets the latch: the primary's window ran out, so a re-arm of its record writes the primary
    /// alone (`0x712553`).
    pub(crate) fn latch(&mut self) {
        self.latched = true;
    }

    /// The secondary's weight λ now, 0 with no fade: the kernel's `(3 − 2x)·x²` (`0x714880`).
    pub(crate) fn lambda(&self) -> f32 {
        self.fade.map_or(0.0, |f| blend_lambda(f.frac(self.now)))
    }

    /// One primary arm of `clip`, the slot half of `0x7121a0`; returns the primary restarted at 0,
    /// for the caller to set its repeat, rate and offset. `blend` 0 clears the secondary
    /// (`0x712635`). With `blend` the outgoing primary is copied into it on its own clock, over the
    /// incoming clip's blendTime (`0x7125d6`–`0x712623`), unless the latch is set and the record is
    /// the primary's (`0x712553` → `0x71262d`), or a fade still weighs over half (`0x7125c9`/
    /// `0x7125d4`): both keep the secondary and its window and write the primary alone.
    pub(crate) fn arm<'p>(
        &mut self,
        player: &'p mut AnimationPlayer,
        clip: &AnimClip,
        blend: bool,
    ) -> &'p mut ActiveAnimation {
        let outgoing = self.primary;
        let blend_time = clip.blend_time.max(0.0);
        // The primary's record (`[blk+0xa4]`), on either of its clip's tracks.
        let same = outgoing.is_some_and(|n| n == clip.node || Some(n) == clip.blend_node);
        if !blend {
            self.clear(player);
        } else if same && self.latched {
            // The shortcut: the sequence that ran to its end is written again as the primary.
        } else if self.lambda() <= 0.5 {
            // A NaN λ fails the test and keeps the slot, as `jp` does. A zero blendTime copies
            // nothing, so the arm cuts.
            self.clear(player);
            self.fade = outgoing.filter(|_| blend_time > 0.0).map(|node| Fade {
                node,
                until: self.now + f64::from(blend_time),
                span: blend_time,
            });
        }
        // The primary runs on the clip's track the secondary is not on, so a re-arm of the sequence
        // the secondary holds leaves that clock alone.
        let node = match (self.fade, clip.blend_node) {
            (Some(f), Some(twin)) if f.node == clip.node => twin,
            (Some(f), None) if f.node == clip.node => {
                // A model built with one track per clip: the arm takes it, and the secondary ends.
                self.fade = None;
                clip.node
            }
            _ => clip.node,
        };
        // The primary write replaces the outgoing primary (`0x712692`), unless the copy kept it.
        if let Some(out) = outgoing.filter(|&o| o != node && self.fade.is_none_or(|f| f.node != o))
        {
            player.stop(out);
        }
        self.primary = Some(node);
        self.latched = false;
        let fading = self.fade.is_some();
        let active = player.start(node);
        // λ = 1 on a blended arm's first frame, so the pose does not move on the frame it lands.
        active.set_weight(if fading { 0.0 } else { 1.0 });
        active
    }

    /// Empties the secondary slot, stopping its track.
    pub(crate) fn clear(&mut self, player: &mut AnimationPlayer) {
        if let Some(old) = self.fade.take() {
            player.stop(old.node);
        }
    }

    /// One frame of the kernel's blend: λ weights the secondary and the primary takes `1 − λ`, each
    /// bone `new + (old − new)·λ`. At the window's end (`0x7147b1`) the secondary alone stops. Run
    /// after the frame's arms, as the kernel runs after the frame's plays.
    pub(crate) fn advance(&mut self, player: &mut AnimationPlayer) {
        let lambda = match self.fade {
            Some(fade) if fade.frac(self.now) > 0.0 => {
                let lambda = blend_lambda(fade.frac(self.now));
                if let Some(out) = player.animation_mut(fade.node) {
                    out.set_weight(lambda);
                }
                lambda
            }
            Some(fade) => {
                self.fade = None;
                player.stop(fade.node);
                0.0
            }
            None => 0.0,
        };
        if let Some(cur) = self.primary.and_then(|n| player.animation_mut(n)) {
            cur.set_weight(1.0 - lambda);
        }
    }
}

#[cfg(test)]
#[path = "blend_slot_tests.rs"]
mod tests;
