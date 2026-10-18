//! Carried M2 lights: the point lights an entity's models bring, beside the placed ones
//! (`benilla_world::terrain_stream`'s `spawn_lights_for`). The reference makes no distinction:
//! `0x718960` gathers every drawn model's `type == 1` lights each frame, attached children included
//! (`0x7191b9`, `0x719286`), at the live bone matrix into the one scene light DB (`0x71b650`,
//! `0x71bb60`), so a held torch lights the ground. Each light here rides its host bone's joint.

use benilla_assets::coords::wow_to_bevy;
use benilla_assets::ModelLight;
use bevy::prelude::*;

use benilla_world::terrain_stream::point_light;

/// Spawn a point light child for each casting M2 light of an entity's model. A light whose bone
/// `joint` resolves rides that joint at `position − bone_pivot`; one without (a held item has no
/// skeleton, a `-1` bone) hangs off `frame` in model space. As children, the lights leave with a
/// gear change, a despawn or a mount transition. `placed`: the host is a GameObject that stays
/// where the server put it, so its lights may group with their neighbours
/// ([`benilla_world::lighting::PlacedLight`]).
pub(super) fn spawn_carried_lights(
    commands: &mut Commands,
    lights: &[ModelLight],
    frame: Entity,
    placed: bool,
    joint: impl Fn(i16) -> Option<Entity>,
) {
    for l in lights {
        if !l.def.casts() {
            continue; // directionals feed an ambient term; a static `0` visibility key is dark
        }
        let (parent, local) = match joint(l.def.bone) {
            Some(j) => (
                j,
                [
                    l.def.position[0] - l.bone_pivot[0],
                    l.def.position[1] - l.bone_pivot[1],
                    l.def.position[2] - l.bone_pivot[2],
                ],
            ),
            None => (frame, l.def.position),
        };
        let glow = commands
            .spawn((
                point_light(l.def.diffuse_color, l.def.diffuse_intensity),
                // fork: P8 — **which body's room this light is in.** Without it the light names no
                // rooms, and a light that names no rooms may not light an INTERIOR surface at all
                // (`static_gx.wgsl`'s `interior && w < 0.5` gate, the rule that stops a street lamp
                // lighting the chapel beside it through the wall). That rule reads "no rooms" as
                // "outdoors", which is true of a placed ADT lamp and false of every light on this
                // lane — so until this was attached, a torch indoors, a fire elemental in a
                // dungeon and a GameObject brazier in Orgrimmar all lit exactly nothing.
                // `frame`, not `parent`: the room belongs to the BODY, and `parent` may be a bone.
                benilla_world::lighting::LightOwner(frame),
                Transform::from_translation(wow_to_bevy(local)),
                Visibility::default(),
            ))
            .id();
        // fork: a held torch burns like a staked one. The seed is the light's model-local spot plus
        // its body, so two torch-bearers never breathe in step.
        let seed_at = Vec3::from(local) + Vec3::splat((frame.to_bits() % 4096) as f32);
        if let Some(flicker) = benilla_world::lighting::flame_flicker(
            None,
            false,
            l.def.diffuse_color,
            l.def.diffuse_intensity,
            seed_at,
        ) {
            commands.entity(glow).insert(flicker);
        }
        if placed {
            commands
                .entity(glow)
                .insert(benilla_world::lighting::PlacedLight);
        }
        commands.entity(parent).add_child(glow);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_formats::M2Light;

    fn light(light_type: u16, bone: i16, position: [f32; 3], visibility_off: bool) -> ModelLight {
        ModelLight {
            def: M2Light {
                light_type,
                bone,
                position,
                bone_z: [0.0, 0.0, 1.0],
                ambient_color: [1.0; 3],
                ambient_intensity: 0.0,
                diffuse_color: [0.466_666_7, 0.290_196_1, 0.133_333_34], // the real torch's warm orange
                diffuse_intensity: 3.0,
                attenuation_start: 1.388_889,
                attenuation_end: 2.222_222_3,
                visibility_off,
            },
            bone_pivot: [1.0, 0.0, 0.5],
        }
    }

    /// Only casting lights spawn (the gather's gate in `0x718960`), each a child of its bone's
    /// joint at `position − bone_pivot`; colour × intensity survives the packer's `intensity / 4π`.
    #[test]
    fn only_casting_lights_spawn_and_they_ride_their_bone() {
        let mut app = App::new();
        let frame = app.world_mut().spawn(Transform::IDENTITY).id();
        let joint = app.world_mut().spawn(Transform::IDENTITY).id();
        let lights = [
            light(1, 3, [2.0, 0.0, 1.5], false),  // casts, on bone 3
            light(0, 3, [2.0, 0.0, 1.5], false),  // directional → ambient term, never a GL light
            light(1, 3, [2.0, 0.0, 1.5], true),   // point but authored dark
            light(1, -1, [0.0, 0.0, 4.0], false), // casts, boneless → the frame itself
        ];
        app.world_mut().commands().queue(move |world: &mut World| {
            let mut q = world.commands();
            spawn_carried_lights(&mut q, &lights, frame, false, move |bone| {
                (bone == 3).then_some(joint)
            });
        });
        app.world_mut().flush();

        let mut spawned: Vec<(Entity, Vec3, Entity)> = app
            .world_mut()
            .query::<(
                Entity,
                &benilla_world::lighting::WorldPointLight,
                &Transform,
                &ChildOf,
            )>()
            .iter(app.world())
            .map(|(e, _, t, c)| (e, t.translation, c.parent()))
            .collect();
        // Spawn order, i.e. light-table order: `Entity`'s own `Ord` is not index-ascending.
        spawned.sort_by_key(|(e, ..)| e.index());
        assert_eq!(
            spawned.len(),
            2,
            "the directional and the dark one stay out"
        );

        assert_eq!(spawned[0].2, joint);
        assert_eq!(spawned[0].1, wow_to_bevy([1.0, 0.0, 1.0]));
        // Boneless: off the frame in model space, as a held item's light always is.
        assert_eq!(spawned[1].2, frame);
        assert_eq!(spawned[1].1, wow_to_bevy([0.0, 0.0, 4.0]));

        let pl = app
            .world()
            .entity(spawned[0].0)
            .get::<benilla_world::lighting::WorldPointLight>()
            .unwrap();
        let recovered = pl.intensity / (4.0 * std::f32::consts::PI);
        assert!(
            (pl.color[0] * recovered - 1.4).abs() < 1e-3,
            "colour × intensity survives the packing"
        );
    }
}
