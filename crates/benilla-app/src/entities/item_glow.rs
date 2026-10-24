//! Item glows: the permanent glow an item display authors and the glow a weapon enchant carries,
//! `Spells\Enchantments` models hung on the item's own model, not the wearer's body.
//!
//! The chain is `ItemDisplayInfo` column 22, then `ItemVisuals`, then five `ItemVisualEffects`
//! models ([`benilla_formats::ItemVisualCatalog`]). The held-item attach passes the display's own
//! id (`0x47a200` into `0x4798c0`, attaching at `0x479700` when it is above 0); the helm, shoulder
//! and quiver sites pass a literal 0, so on a body only hand items glow, and the ammo lane
//! (`0x479f40`) passes its own. `0x479700` hangs effect i on the item model's attachment id i
//! (`0x712f70`). An item model rests at bind pose, so each glow is a plain child of the item root:
//! it goes with a gear change and moves with a sheath swap.

use std::collections::HashMap;

use benilla_formats::{EnchantCatalog, ItemVisualCatalog, ITEM_VISUAL_SLOTS};
use bevy::prelude::*;

use benilla_assets::m2_url;
use benilla_assets::materials::WowModelMaterial;

use super::equipment::{ItemDisplays, ItemModelKind};
use super::spell_fx::{attach_effect_visuals, EffectHost, FxMaterials, FxTintAnims};
use super::{DisplayModel, ModelHandle};

/// The joined `ItemVisuals`/`ItemVisualEffects` catalog and a path-keyed [`DisplayModel`] per glow
/// model. Without it nothing glows; without [`crate::items::Enchants`] only enchant glows go quiet.
#[derive(Resource)]
pub(crate) struct ItemGlows {
    pub(super) visuals: ItemVisualCatalog,
    pub(super) models: HashMap<String, DisplayModel>,
}

impl ItemGlows {
    pub(super) fn new(visuals: ItemVisualCatalog) -> Self {
        ItemGlows {
            visuals,
            models: HashMap::new(),
        }
    }

    /// The glow models by attach slot; `None` when the id names no row (0, the shipped `-1`s).
    pub(super) fn effects(&self, visual: i32) -> Option<&[Option<String>; ITEM_VISUAL_SLOTS]> {
        self.visuals.effects(visual)
    }
}

/// The ItemVisuals id an item glows with, the fork of `0x62ec70` (per weapon slot, `0x5eed50`):
/// the base display's id when it names a row, which suppresses the enchant's, else the first
/// enchant slot with a nonzero visual, else 0.
pub(in crate::entities) fn effective_visual(
    glows: &ItemGlows,
    enchants: Option<&EnchantCatalog>,
    base: i32,
    item_enchants: impl IntoIterator<Item = u32>,
) -> i32 {
    if glows.visuals.effects(base).is_some() {
        return base;
    }
    let Some(enchants) = enchants else { return 0 };
    for enchant in item_enchants {
        if let Some(visual) = enchants.visual(enchant) {
            return visual;
        }
    }
    0
}

/// Create a visual's glow-model cache entries, so `super::update_display_models` builds them the
/// frame the equipment resolve asks.
pub(in crate::entities) fn ensure_glow_models(
    glows: &mut ItemGlows,
    visual: i32,
    asset_server: &AssetServer,
) {
    let Some(paths) = glows.visuals.effects(visual) else {
        return;
    };
    // Collected first: `effects` borrows the catalog, and the insert below borrows the cache.
    let wanted: Vec<String> = paths
        .iter()
        .flatten()
        .filter(|p| !glows.models.contains_key(*p))
        .cloned()
        .collect();
    for path in wanted {
        let handle = ModelHandle::M2(asset_server.load(m2_url(&path)));
        glows.models.insert(
            path,
            DisplayModel {
                handle,
                ..super::empty_shell()
            },
        );
    }
}

/// A held-item root that should carry a glow, written by `super::equipment`'s attach and consumed
/// once by [`attach_item_glows`].
#[derive(Component)]
pub(in crate::entities) struct ItemGlow {
    pub(in crate::entities) display: u32,
    pub(in crate::entities) kind: ItemModelKind,
    pub(in crate::entities) visual: i32,
    /// The item's seat on the body ([`super::BoneAttach`]), kept to place the portrait mirrors:
    /// the attach path that knew it is gone by the time the glow models load.
    pub(in crate::entities) bone: u16,
    pub(in crate::entities) offset: bevy::prelude::Vec3,
    /// The item's M2 attachment id on the body: a glow chains under its item (`0x712f70`), so a
    /// widget's attach reset takes both together.
    pub(in crate::entities) attach: u16,
}

/// The item's glow instances are spawned, or there were none. As children of the root they go
/// with a gear change and move, live particles and all, with a sheath swap.
#[derive(Component)]
pub(crate) struct ItemGlowAttached;

/// Spawn each pending item's glow instances, one per authored slot at that slot's attachment point
/// on the item model; all or nothing, so the set waits while any of its models loads.
pub(super) fn attach_item_glows(
    mut commands: Commands,
    pending: Query<(Entity, &ItemGlow), Without<ItemGlowAttached>>,
    glows: Option<Res<ItemGlows>>,
    items: Option<Res<ItemDisplays>>,
    time: Res<Time>,
    mut wow_materials: ResMut<Assets<WowModelMaterial>>,
    mut tint_reg: ResMut<FxTintAnims>,
    mut uv_reg: ResMut<benilla_world::doodad_anim::UvAnimMaterials>,
    mut anim_table: ResMut<benilla_world::mat_anim_table::MatAnimTable>,
    ibps: Res<Assets<bevy::mesh::skinning::SkinnedMeshInverseBindposes>>,
    mut palettes: ResMut<benilla_world::rig_palette::RigPalettes>,
    // The session's first spawned glow is logged once at info level.
    mut logged: Local<bool>,
) {
    let (Some(glows), Some(items)) = (glows, items) else {
        return;
    };
    let now = time.elapsed_secs();
    for (root, glow) in &pending {
        let Some(paths) = glows.effects(glow.visual) else {
            commands.entity(root).insert(ItemGlowAttached); // nothing to hang
            continue;
        };
        let Some(item) = items.models.get(&(glow.display, glow.kind)) else {
            continue; // the item's display entry was evicted: retry
        };
        // Ready: every model has built its parts; an emitter-only model's empty list is `Some`.
        let ready = paths
            .iter()
            .flatten()
            .all(|p| glows.models.get(p).is_some_and(|dm| dm.parts.is_some()));
        if !ready {
            continue;
        }
        let mut spawned = 0usize;
        // fork: modern lighting (`FORK.md` P8) — the glow's own light, gathered
        // across the slots and spawned once below. Per ITEM and not per slot: the common visual
        // repeats one effect along the whole blade (BlueGlow_Med ×5), and five lights for one sword
        // would be five table slots and five cube-shadow candidates for one glow. Every slot's
        // emitters go into ONE `synthesize` call, which is the same rule it applies to the several
        // emitters of a single model — five sprites along a blade *are* brighter than one, and the
        // recipe's own cap is what stops that running away.
        let mut glow_sources = Vec::new();
        let mut glow_cards = Vec::new();
        let mut glow_authored = false;
        let mut glow_at = (Vec3::ZERO, 0usize);
        for (slot, path) in paths
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.as_ref().map(|p| (i, p)))
        {
            let Some(dm) = glows.models.get(path) else {
                continue;
            };
            // A model without this slot's attachment hangs nothing there, as in the reference:
            // `0x712f70` stores `0x710310`'s miss as `0xffff`, whose child `0x7140aa`, `0x718668`
            // and `0x719266` skip. Common: Ironfoe's hammer authors only ids 2 to 4.
            let Some(at) =
                crate::portrait::attachment_point(&item.skeleton, &item.attachments, slot as u16)
            else {
                debug!(
                    "item glow: display {} has no attachment {slot} for {path}",
                    glow.display
                );
                continue;
            };
            let instance = commands
                .spawn((Transform::from_translation(at), Visibility::default()))
                .id();
            commands.entity(root).add_child(instance);
            // The portrait mirrors, at the item's seat plus this slot's point. The shared spell-fx
            // lane stamps none, as a spell's effect must never reach a portrait, so the glow's
            // come from here; 32 of the 35 shipped glow models are pure emitters.
            let seat = glow.offset + at;
            for p in dm.parts.iter().flatten() {
                match &p.billboard {
                    Some(info) => {
                        commands.entity(instance).with_child((
                            Transform::default(),
                            Visibility::default(),
                            crate::portrait::PortraitBillboard {
                                mesh: p.mesh.clone(),
                                material: p.material.clone(),
                                bone: glow.bone,
                                seat: crate::portrait::PortraitSeat::Rider(seat + info.pivot),
                                kind: info.kind,
                                attach: Some(glow.attach),
                            },
                        ));
                    }
                    None => {
                        commands.entity(instance).with_child((
                            Transform::default(),
                            Visibility::default(),
                            crate::portrait::PortraitRider {
                                static_mesh: p.mesh.clone(),
                                material: p.material.clone(),
                                bone: glow.bone,
                                offset: seat,
                                attach: Some(glow.attach),
                                idle: Transform::IDENTITY,
                            },
                        ));
                    }
                }
            }
            if !dm.emitters.is_empty() {
                commands
                    .entity(instance)
                    .insert(crate::portrait::PortraitEffects {
                        bone: glow.bone,
                        offset: seat,
                        attach: Some(glow.attach),
                        emitters: dm.emitters.clone(),
                        rig: None,
                    });
            }
            attach_effect_visuals(
                &mut commands,
                instance,
                dm,
                now,
                false, // a weapon glow is never ground-anchored
                // Chained to the item root (`0x712f70`) and through it to the wearer, so it fades
                // in with the body and dies with the weapon.
                EffectHost { parent: Some(root) },
                // Not armed by `PlaySpellVisualKit`: no kit stage, the plain single-clip arm.
                None,
                &mut FxMaterials {
                    store: &mut wow_materials,
                    tint: &mut tint_reg,
                    uv: &mut uv_reg,
                    table: &mut anim_table,
                },
                &ibps,
                &mut palettes,
                None, // the glow models author one looping sequence
            );
            spawned += 1;
            // fork: P8 — what this slot contributes to the weapon's light, and where it sits. A
            // glow model that authors a real M2 light of its own suppresses the synthesis outright
            // (there are none today, but a torch-like effect model would, and then its own light
            // wins) — the same law the doodad path takes, spelled by the same function.
            {
                glow_authored |= dm.lights.iter().any(|l| l.def.casts());
                glow_sources.extend(
                    dm.emitters
                        .iter()
                        .map(|e| (e.def.blend, e.def.lit, &e.def.over_life)),
                );
                // A glow model's additive CARDS count too, not only its emitters: 32 of the 35
                // shipped ones are pure emitters, but `Sparkle_A.m2` is a lone camera-facing quad
                // and would otherwise light nothing at all.
                glow_cards.extend(dm.parts.iter().flatten().filter_map(|p| {
                    benilla_world::modern::lights::emissive::glow_of_geometry(&p.geometry)
                }));
                glow_at = (glow_at.0 + at, glow_at.1 + 1);
            }
        }
        // fork: P8 — **the glowing sword.** One light for the item, at the mean of the points its
        // glow actually landed on (not of the points the visual *authors*: a model missing an
        // attachment hangs nothing there, and averaging a point nothing was hung on would drag the
        // light off the blade). A child of the item root, so it rides the hand through the
        // animation, survives a sheath swap the way the glow instances do, and dies with the weapon
        // — the same lifecycle rule `carried_light` states for an authored M2 light.
        //
        // The intensity is the recipe's, unscaled: an enchant glow is a faint source and the recipe
        // already says so (its ramp is a small additive sprite). What makes it read is the tier's
        // soft, faint cube shadow, not its brightness.
        if glow_at.1 > 0 {
            if let Some(e) = benilla_world::modern::lights::emissive::combine(
                glow_authored,
                glow_sources,
                glow_cards,
            ) {
                // **No tier gate at attach time**, and that is the whole of why Teebu's Blazing
                // Longsword did not glow. This runs once per weapon and then `ItemGlowAttached`
                // seals it, so reading `emissive_lights_on()` here asked a question whose answer
                // had not been decided yet — the lighting model latches after the session is up —
                // and never asked again. The light is spawned unconditionally and tagged; the tier
                // gate lives in `global_light`, per frame, exactly as it does for a lamppost.
                let light = commands
                    .spawn((
                        benilla_world::terrain_stream::point_light(e.color, e.intensity_scale),
                        benilla_world::modern::lights::SynthesizedLight,
                        // The wielder's room (see `LightOwner`): a glowing sword indoors is the
                        // same refusal every other entity light was getting.
                        benilla_world::lighting::LightOwner(root),
                        Transform::from_translation(glow_at.0 / glow_at.1 as f32),
                        Visibility::default(),
                    ))
                    .id();
                commands.entity(root).add_child(light);
            }
        }
        commands.entity(root).insert(ItemGlowAttached);
        // What spawned, not what the row authors: they differ where the model lacks a point.
        let authored = paths.iter().flatten().count();
        if !*logged && spawned > 0 {
            *logged = true;
            info!(
                "item glow: display {} visual {} → {spawned} of {authored} model(s) attached \
                 (the first this session)",
                glow.display, glow.visual,
            );
        }
        debug!(
            "item glow: display {} visual {} → {spawned} of {authored} model(s)",
            glow.display, glow.visual,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog(visual_rows: &[u32], enchant_rows: &[(u32, i32)]) -> (ItemGlows, EnchantCatalog) {
        let visuals = visual_rows
            .iter()
            .map(|id| {
                (
                    *id,
                    std::array::from_fn(|i| {
                        (i == 3).then(|| format!("Spells\\Enchantments\\Row{id}.mdx"))
                    }),
                )
            })
            .collect();
        (
            ItemGlows::new(ItemVisualCatalog::from_visuals(visuals)),
            EnchantCatalog::from_rows(
                enchant_rows.iter().copied().collect(),
                HashMap::new(),
                Default::default(),
            ),
        )
    }

    #[test]
    fn base_visual_wins_over_enchant_and_suppresses_it() {
        let (glows, ench) = catalog(&[25, 61], &[(1, 61), (7, 25)]);
        let ench = Some(&ench);
        assert_eq!(effective_visual(&glows, ench, 25, [1]), 25);
        assert_eq!(effective_visual(&glows, ench, 0, [1]), 61);
        // An enchant without a visual is walked past, not the end of the scan.
        assert_eq!(effective_visual(&glows, ench, 0, [999, 7]), 25);
        assert_eq!(effective_visual(&glows, ench, 0, [999]), 0);
        // The shipped `-1` base is not a row: it neither glows nor suppresses the enchant.
        assert_eq!(effective_visual(&glows, ench, -1, [1]), 61);
        assert_eq!(effective_visual(&glows, ench, -1, []), 0);
    }

    #[test]
    fn without_the_enchant_catalog_only_the_base_leg_answers() {
        let (glows, _) = catalog(&[25, 61], &[(1, 61)]);
        assert_eq!(effective_visual(&glows, None, 25, [1]), 25);
        assert_eq!(effective_visual(&glows, None, 0, [1]), 0);
    }

    /// A weapon standing in for **Teebu's Blazing Longsword**: a bind-pose item skeleton whose
    /// bones telescope up the blade, with the five glow attachment ids (0..4, the module doc's
    /// `BlueGlow_Med ×5` shape) seated along it.
    ///
    /// Pure translations, as a vanilla M2's rest pose is by construction — no inverse-bind array,
    /// identity rest TRS, the pivot encoding bind position
    /// (`benilla_assets::model::build_skeleton`). Bevy space, so the blade runs up **+Y**.
    fn teebu() -> (
        benilla_assets::ModelSkeleton,
        Vec<benilla_assets::ModelAttachment>,
    ) {
        let joints = (0..5)
            .map(|i| benilla_assets::ModelJoint {
                parent: i as i16 - 1, // -1 on the grip bone = root
                local_translation: Vec3::new(0.0, if i == 0 { 0.0 } else { 0.25 }, 0.0),
                billboard: None,
                parent_arm: None,
            })
            .collect();
        let attachments = (0..5)
            .map(|i| benilla_assets::ModelAttachment {
                id: i,
                bone: i,
                offset: Vec3::ZERO,
            })
            .collect();
        (
            benilla_assets::ModelSkeleton {
                joints,
                spine_bone: None,
                head_bone: None,
            },
            attachments,
        )
    }

    /// **Teebu's blade glow, swung** — where its sprites actually travel, measured through the
    /// real chain rather than reasoned about.
    ///
    /// A glow sprite's direction composes through four links, and each one either carries the
    /// host's rotation or drops it:
    ///
    /// | link | built by | carries |
    /// |---|---|---|
    /// | hand anchor | [`benilla_world::rig_anim::RigPose::anchor_for`] | the hand bone's composed pose, **rotation included** |
    /// | item root | `equipment::spawn_slot` | the body attach point's offset — translation only |
    /// | glow instance | [`attach_item_glows`] | the ITEM's own attachment point — translation only |
    /// | the birth fold | [`benilla_world::particles::birth_world`] | `emit_place.rotation ×` the kernel's local +Z |
    ///
    /// The two translation-only links are **not** a dropped rotation, which is the thing this
    /// test exists to settle: an item model poses at REST (no rig is spawned for it — module
    /// doc), and a vanilla M2's rest pose is identity TRS, so there is no bone rotation at either
    /// seat for `attachment_point` to have lost. Every rotation a sprite needs therefore arrives
    /// through the hand anchor and nowhere else, and the law is that it must arrive intact: break
    /// the parenting, seat an instance at a world position instead of under the item, or let the
    /// sim read a placement stripped of its host's rotation, and the spray falls back to the
    /// world axes — straight up out of the blade whatever the sword is doing, which is the
    /// reported symptom exactly.
    ///
    /// Asserted per pose **through a swing**, and that is the point: at rest, with the blade
    /// upright, "along the blade" and "straight up" are the same vector and a dropped rotation is
    /// invisible. Only a turned wrist tells them apart.
    #[test]
    fn teebu_s_blade_sprites_travel_along_the_blade_through_a_swing() {
        use benilla_world::particles::birth_world;
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

        let (skeleton, attachments) = teebu();
        // The body attach point's own offset (`BoneAttach`) — the item root's only transform.
        let body_offset = Vec3::new(0.05, -0.1, 0.02);
        // The kernel's output for `BlueGlow_Med`, a Plane emitter on a narrow cone about local
        // +Z ([`emit_local`]): dead-centre is +Z, and the emitter-frame R(+Z, 90°) that
        // `emit_local` applies leaves that axis alone, so this IS what the sim hands the fold.
        let kernel_dir = Vec3::Z;

        // A swing, from the ready pose through a full horizontal cleave: the hand bone's
        // composed rotation at each step. Identity is deliberately NOT one of them — it is the
        // one pose that cannot discriminate.
        let swing = [
            Quat::from_rotation_x(FRAC_PI_2),  // blade levelled forward
            Quat::from_rotation_z(-FRAC_PI_2), // blade out to the side
            Quat::from_rotation_x(FRAC_PI_4) * Quat::from_rotation_y(FRAC_PI_4),
            Quat::from_rotation_z(PI), // blade straight down — the inverted guard
        ];

        for (step, hand) in swing.into_iter().enumerate() {
            let mut app = App::new();
            app.add_plugins((MinimalPlugins, bevy::transform::TransformPlugin));

            // The wearer's hand anchor, as `anchor_for` seats it: the bone's composed pose.
            let anchor = app
                .world_mut()
                .spawn(Transform {
                    translation: Vec3::new(12.0, 1.4, -40.0),
                    rotation: hand,
                    scale: Vec3::ONE,
                })
                .id();
            // The item root, as `spawn_slot` builds it: the attach offset, translation only.
            let root = app
                .world_mut()
                .spawn(Transform::from_translation(body_offset))
                .id();
            app.world_mut().entity_mut(anchor).add_child(root);

            // One glow instance per authored slot, as `attach_item_glows` builds them: a plain
            // child of the item root at that slot's bind-pose attachment point.
            let instances: Vec<Entity> = (0..5)
                .map(|slot| {
                    let at = crate::portrait::attachment_point(&skeleton, &attachments, slot)
                        .expect("Teebu authors all five glow attachment points");
                    let e = app.world_mut().spawn(Transform::from_translation(at)).id();
                    app.world_mut().entity_mut(root).add_child(e);
                    e
                })
                .collect();

            app.update(); // propagate

            // The blade axis in world: the item rests at bind pose, so its model up IS the
            // hand's up. This is the direction the sprites are supposed to run.
            let blade = hand * Vec3::Y;
            let mut seats = Vec::new();

            for (slot, &instance) in instances.iter().enumerate() {
                // Exactly what the sim reads for this emitter's `placement` — the owner's live
                // `GlobalTransform` (a `BlueGlow_Med` has no bones, so the owner IS the
                // instance; `attach_effect_visuals` falls back to the instance root).
                let emit_place = app
                    .world()
                    .entity(instance)
                    .get::<GlobalTransform>()
                    .expect("a propagated instance")
                    .compute_transform();
                let (pos, vel) = birth_world(&emit_place, Vec3::ZERO, kernel_dir, 3.0);
                seats.push(pos);

                let travel = vel.normalize();
                assert!(
                    travel.distance(blade) < 1e-5,
                    "swing step {step}, glow slot {slot}: sprites travel {travel:?}, but the \
                     blade runs {blade:?} — a rotation was dropped between the hand and the \
                     emitter's birth frame",
                );
                // The failure this actually guards, stated as itself: a stripped placement
                // sprays along world up no matter what the wrist is doing.
                assert!(
                    travel.distance(Vec3::Y) > 1e-3,
                    "swing step {step}, glow slot {slot}: sprites went straight up out of a \
                     blade that is not upright",
                );
            }

            // …and the five seats are strung UP the blade, not piled at one point: the glow is
            // supposed to read as a lit edge, and it does that only if each slot lands on its own
            // attachment. Consecutive seats step one bone apart, along the blade.
            for (slot, pair) in seats.windows(2).enumerate() {
                let step_v = pair[1] - pair[0];
                assert!(
                    (step_v.length() - 0.25).abs() < 1e-5
                        && step_v.normalize().distance(blade) < 1e-5,
                    "swing step {step}: glow slots {slot}→{} are {step_v:?} apart, which is not \
                     one bone along the blade",
                    slot + 1,
                );
            }
        }
    }
}
