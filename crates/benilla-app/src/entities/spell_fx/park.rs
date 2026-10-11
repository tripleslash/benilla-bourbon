//! fork: an effect model on a parked host stops writing its joints.
//!
//! An effect instance (a spell-kit visual, a weapon-enchant glow) poses through Bevy's
//! one-entity-per-bone lane: [`super::arm_effect_rig`] gives every bone an `AnimatedBy` target, and
//! `animate_targets` commits every joint's `Transform` each frame its player plays. Those writes
//! raise `Changed<Transform>`, so propagation walks the instance's subtree and the palette pass
//! rebuilds its rows. The rig gate (`creature_anim/lod.rs`) never sees the instance, which has no
//! `RigPose`, so an enchanted weapon or a standing aura behind the camera cost what one in view does.
//!
//! `animate_targets` skips a clip whose `ActiveAnimation` weight is exactly 0.0 for every target
//! (`bevy_animation-0.19.1/src/lib.rs`, "If the weight is zero ... stop here"), so nothing is
//! committed: the joints hold their last pose and raise no change. The player's clock still
//! advances, so the lifecycle's completions, the event scan and the emitters' sequence clock run
//! on, and a wake resumes in phase.
//!
//! An instance parks while the nearest rig at or above its host model is parked, the gate's own
//! verdict; a free-standing instance (a missile, a ground effect: no [`ParentModel`]) is left
//! alone. An effect player's clips carry weight 1.0 and nothing else writes one, so a wake restores
//! 1.0; a clip the lifecycle starts while parked arrives at 1.0 and is zeroed on the next pass.
//! An instance in view poses as before: only one whose host the gate has stopped posing is held.

use bevy::ecs::entity::EntityHashSet;
use bevy::prelude::*;

use benilla_world::model_fade::{ParentModel, MAX_MODEL_CHAIN};
use benilla_world::rig_anim::{AnimParked, RigPose};

use super::EffectPlayer;

/// How far up the transform tree a host's rig may sit: item root, anchor, joints root, body, and
/// one seat further for a rider.
const MAX_HOST_DEPTH: usize = 2 * MAX_MODEL_CHAIN;

/// An effect root with an armed player. It is never a doodad host; the filter states it, so these
/// player writes and the doodad gate's are disjoint.
type ArmedEffect = (
    With<EffectPlayer>,
    Without<benilla_world::doodad_anim::DoodadAnimHost>,
);

/// Zero or restore each hosted effect's clip weights from its host rig's park state.
pub(crate) fn park_effect_rigs(
    mut effects: Query<(Entity, &ParentModel, &mut AnimationPlayer), ArmedEffect>,
    parents: Query<&ChildOf>,
    rigs: Query<Has<AnimParked>, With<RigPose>>,
    mut held: Local<EntityHashSet>,
) {
    for (root, host, mut player) in &mut effects {
        if host_parked(host.0, &parents, &rigs) {
            // Read through `Deref` first: a weight already 0 costs no `Mut` write.
            if player.playing_animations().any(|(_, a)| a.weight() != 0.0) {
                for (_, a) in player.playing_animations_mut() {
                    a.set_weight(0.0);
                }
            }
            held.insert(root);
        } else if held.remove(&root) {
            for (_, a) in player.playing_animations_mut() {
                a.set_weight(1.0);
            }
        }
    }
    // A despawned instance has nothing to restore.
    if !held.is_empty() {
        held.retain(|e| effects.contains(*e));
    }
}

/// Whether the nearest `RigPose` holder at or above `host` is parked; `false` when there is none.
fn host_parked(
    host: Entity,
    parents: &Query<&ChildOf>,
    rigs: &Query<Has<AnimParked>, With<RigPose>>,
) -> bool {
    let mut e = host;
    for _ in 0..MAX_HOST_DEPTH {
        if let Ok(parked) = rigs.get(e) {
            return parked;
        }
        match parents.get(e) {
            Ok(c) => e = c.parent(),
            Err(_) => return false,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::animation::graph::AnimationNodeIndex;

    fn weight(app: &App, fx: Entity) -> f32 {
        app.world()
            .get::<AnimationPlayer>(fx)
            .unwrap()
            .playing_animations()
            .map(|(_, a)| a.weight())
            .next()
            .unwrap()
    }

    fn playing() -> AnimationPlayer {
        let mut player = AnimationPlayer::default();
        player.play(AnimationNodeIndex::new(1));
        player
    }

    #[test]
    fn a_parked_host_zeroes_its_effect_and_a_wake_restores_it() {
        let mut app = App::new();
        app.add_systems(Update, park_effect_rigs);
        let pose = RigPose::new(
            Entity::PLACEHOLDER,
            &benilla_assets::ModelSkeleton::default(),
        );
        let body = app.world_mut().spawn((pose, AnimParked)).id();
        let item = app.world_mut().spawn(ChildOf(body)).id();
        let fx = app
            .world_mut()
            .spawn((EffectPlayer, ParentModel(item), playing()))
            .id();
        app.update();
        assert_eq!(weight(&app, fx), 0.0);
        app.world_mut().entity_mut(body).remove::<AnimParked>();
        app.update();
        assert_eq!(weight(&app, fx), 1.0);
    }

    #[test]
    fn a_live_host_and_a_free_standing_instance_keep_their_weight() {
        let mut app = App::new();
        app.add_systems(Update, park_effect_rigs);
        let pose = RigPose::new(
            Entity::PLACEHOLDER,
            &benilla_assets::ModelSkeleton::default(),
        );
        let body = app.world_mut().spawn(pose).id();
        let hosted = app
            .world_mut()
            .spawn((EffectPlayer, ParentModel(body), playing()))
            .id();
        // A missile: no host model, so no verdict to follow.
        let free = app.world_mut().spawn((EffectPlayer, playing())).id();
        app.update();
        assert_eq!(weight(&app, hosted), 1.0);
        assert_eq!(weight(&app, free), 1.0);
    }
}
