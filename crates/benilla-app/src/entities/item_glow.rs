//! **Item / enchant glow effects** (decision 0805): the permanent weapon glow a display authors,
//! and the enchant glow a weapon's enchant carries — `Spells\Enchantments\*.mdx` effect models
//! hung on the ITEM's own model, not on the wearer's body.
//!
//! ## The chain, and where each half is verified
//!
//! `ItemDisplayInfo` col 22 → `ItemVisuals` → 5 × `ItemVisualEffects` → a glow `.mdx` per slot
//! ([`benilla_formats::ItemVisualCatalog`], whose module doc owns the table shapes and the
//! reference's per-slot skip rules). Two sources feed the id, and the reference resolves exactly
//! one of them per item (wow-re `object-layer/scratch/item-visual-enchant.md`, byte-verified;
//! the call sites below were read at the bytes for decision 0805):
//!
//! - **Intrinsic** — the display's own id, passed by the character's held-item attach:
//!   `0x47a200: mov edx,[esi+0x58]` → `0x4798c0(…, itemVisualsId)`, which calls the attach
//!   `0x479700` iff the id is **`> 0`** (`jle` skip). The same primitive attaches the helm (id
//!   11), the shoulders (5/6) and the quiver (26) — and **every one of those call sites pushes a
//!   literal `0`**, so on a character body only the WEAPON/SHIELD hand attach can glow. The
//!   ranged-ammo lane (`0x479f40`) passes `[ebx+0x58]` at its tail, so a thrown/ammo model glows
//!   from its own display too.
//! - **Enchant** — `0x5eed50` per weapon slot → `0x62ec70(item)`: if the base display's
//!   ItemVisuals row **exists**, it returns `-1` and the enchant glow is suppressed (the base
//!   already draws one); otherwise the first of the item's 7 enchant slots whose
//!   `SpellItemEnchantment` row carries a nonzero visual wins. [`effective_visual`] is that fork.
//!
//! ## Where the glow hangs
//!
//! `0x479700` walks the 5 effect columns and calls `0x712f70(glow, itemModel, attachId = the
//! column index)` — so **slot i is M2 attachment id i on the item's own model**, and a row that
//! repeats one effect across all five slots (BlueGlow_Med ×5, the common shape) reads as a glow
//! along the whole blade.
//!
//! **How many of the five actually land is the model's business, and it varies**: of the 160
//! distinct weapon models the 360 glowing displays resolve to, 61 author the full ids 0..4 and 44
//! author only 2/3/4 (Ironfoe's hammer is one — three of its five copies draw). An id the model
//! doesn't author hangs nothing at all, byte-verified: `0x712f70` stores `0x710310`'s `0xffff`
//! miss and every consumer of that field (`0x7140aa`, `0x718668`, `0x719266`) tests
//! `cmp 0xffff; je` and skips the child — there is no fallback to the model origin. Which record
//! an id resolves to is the model's **AttachLookup**, never a scan of its table
//! (`benilla_m2::M2Model::attachment`): ~5% of weapon models author two records under one id.
//!
//! An item model poses at rest (no rig is spawned for it — `super::equipment`), so the attach
//! point is its bind-pose position ([`crate::portrait::attachment_point`]) and the glow instance
//! is a plain child of the item root: it rides the hand/hip/back bone with the weapon, and dies
//! with it on a gear or sheath change. Everything past that — the effect model's own rig,
//! emitters, ribbons, material animation — is the shared effect body
//! ([`super::spell_fx::attach_effect_visuals`]); these models are almost all pure particle
//! emitters (BlueGlow_Med: zero render batches, one Plane/Add emitter on `Flare.blp`).

use std::collections::HashMap;

use benilla_formats::{EnchantCatalog, ItemVisualCatalog, ITEM_VISUAL_SLOTS};
use bevy::prelude::*;

use benilla_assets::m2_url;
use benilla_assets::materials::WowModelMaterial;

use super::equipment::{ItemDisplays, ItemModelKind};
use super::spell_fx::{attach_effect_visuals, EffectHost, FxMaterials, FxTintAnims};
use super::{DisplayModel, ModelHandle};

/// The glow chain's data + the effect-model cache: the joined `ItemVisuals`/`ItemVisualEffects`
/// catalog, plus a path-keyed [`DisplayModel`] per glow model (the [`super::spell_fx::SpellFx`]
/// pattern — entries are created by [`ensure_glow_models`] at equipment-resolve time and built by
/// `super::update_display_models` the same frame). Optional resource: without the DBC nothing
/// glows, exactly as before this lane existed.
///
/// The enchant half of the fork lives in [`crate::items::Enchants`] — the same
/// `SpellItemEnchantment` load the tooltip's enchant line reads (decision 0915). Without it a
/// weapon still shows its INTRINSIC glow; only the enchant leg goes quiet.
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

    /// The glow models an ItemVisuals id resolves to, by attach slot — `None` when the id names
    /// no row (0, the shipped `-1`s, out of range).
    pub(super) fn effects(&self, visual: i32) -> Option<&[Option<String>; ITEM_VISUAL_SLOTS]> {
        self.visuals.effects(visual)
    }
}

/// The ItemVisuals id an item actually glows with — the reference's base-or-enchant fork
/// (`0x62ec70`): the **base** display's id wins whenever it names a real ItemVisuals row (the
/// client returns `-1` from the enchant resolver precisely so the two can't double up); otherwise
/// the **first** enchant slot whose `SpellItemEnchantment` row carries a nonzero visual. `0` when
/// neither does.
/// `enchants` absent (the DBC failed to load) means only the base leg can answer.
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

/// Create the [`DisplayModel`] cache entries for a visual's glow models, so
/// `super::update_display_models` builds them the same frame the equipment resolve asked for
/// them (the held-item pattern).
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

/// A spawned held-item root that should carry a glow: the item's display-model key (for its
/// attachment table) and the resolved ItemVisuals id. Written by `super::equipment`'s attach;
/// consumed once by [`attach_item_glows`], which then marks it [`ItemGlowAttached`].
#[derive(Component)]
pub(in crate::entities) struct ItemGlow {
    pub(in crate::entities) display: u32,
    pub(in crate::entities) kind: ItemModelKind,
    pub(in crate::entities) visual: i32,
    /// The ITEM's seat on the wearer's body — the attach point's bone and offset
    /// ([`super::BoneAttach`]). Carried purely so this lane can publish the booth mirrors of what it
    /// spawns (decision 0822): a glow's seat is this offset plus the slot's point on the item model,
    /// and by the time the glow models load the attach path that knew the body bone is long gone.
    pub(in crate::entities) bone: u16,
    pub(in crate::entities) offset: bevy::prelude::Vec3,
    /// The ITEM's **M2 attachment id** on the body, for the same reason: a glow is chained under
    /// the item (`0x712f70`), so a widget's attach reset takes the glow exactly when it takes the
    /// item it hangs on ([`crate::portrait::attach_reset`]).
    pub(in crate::entities) attach: u16,
}

/// Set once an item root's glow instances are spawned (or resolved to nothing) — the once-only
/// gate. The instances are children of the root, so they need no lifetime bookkeeping of their
/// own: a gear change or the unit despawning takes the root and them with it. A **sheath swap**
/// keeps them: it moves the root to the new attach point (decision 0826), so a glowing weapon
/// keeps its glow — and its live particles — across the draw.
#[derive(Component)]
pub(crate) struct ItemGlowAttached;

/// Spawn each pending item root's glow instances: one effect-model instance per authored slot, at
/// that slot's attachment point on the item's own model.
///
/// All-or-nothing per item — while any of the item's glow models is still loading the whole set
/// waits, so a two-model visual can't spawn half of itself and then re-enter here. A model that
/// never loads simply never glows (the item is still perfectly drawn); the retry is a handful of
/// roots per frame.
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
    // One breadcrumb per session the first time a glow actually spawns — the machine-readable
    // "this lane is live" signal for a subsystem whose whole symptom is *absence* (the same
    // idiom `apply_unit_mat_alpha` uses for its cull).
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
            continue; // the item's own display entry is gone (a cache evict) — retry
        };
        // Ready = every authored slot's model has built its parts. Emitter-only models build an
        // EMPTY part list, which is `Some` — the readiness test is the option, not the length.
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
            // Slot index == M2 attachment id on the ITEM model (module doc). **A model missing
            // that point hangs nothing there**, and that is the reference's behaviour, not a
            // shortcut: `0x712f70` stores the attach-lookup miss as `0xffff`
            // (`0x710310`'s bounds return), and every consumer of that field —
            // `0x7140aa`/`0x718668`/`0x719266` — tests `cmp 0xffff; je` and skips the child
            // outright. It is a common case, not a corner: Ironfoe's hammer authors only ids
            // 2/3/4, so three of its visual's five copies draw and two do not.
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
            // The booth mirrors for this instance (decision 0822), at its seat on the BODY: the item's
            // attach point plus this slot's point on the item model. `attach_effect_visuals` below
            // spawns the effect's real geometry and emitters — the meshes as instance children, the
            // emitters as free owner-followed entities — and the shared spell-fx lane is deliberately
            // not the place to stamp portrait markers (a *spell's* fx must never reach a booth), so
            // this lane publishes its own from the same `dm`. Without them a permanently-glowing
            // weapon glowed nothing in the character window: 32 of the 35 shipped glow models are pure
            // emitters, and `Sparkle_A.m2` is a lone camera-facing quad.
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
                    });
            }
            attach_effect_visuals(
                &mut commands,
                instance,
                dm,
                now,
                false, // a weapon glow is never ground-anchored
                // An attached model in the client's sense — it rides the ITEM (`0x712f70`), so it
                // is chained to that item root and through it to the wearer: the glow fades in
                // with the body instead of blazing over a character that has not appeared yet,
                // and it dies with the weapon instead of hanging in the air behind it (0833).
                EffectHost { parent: Some(root) },
                // An `ItemVisuals` glow is armed by a different leg than `PlaySpellVisualKit`
                // (0805) and carries no kit stage — the plain single-clip arm, as before.
                None,
                &mut FxMaterials {
                    store: &mut wow_materials,
                    tint: &mut tint_reg,
                    uv: &mut uv_reg,
                    table: &mut anim_table,
                },
                &ibps,
                &mut palettes,
                None, // the glow models author one looping sequence — the default pick
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
                glow_cards.extend(
                    dm.parts
                        .iter()
                        .flatten()
                        .filter_map(|p| {
                            benilla_world::modern::lights::emissive::glow_of_geometry(&p.geometry)
                        }),
                );
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
        // Counted from what actually spawned, never from what the row *authors* — the two differ
        // whenever the item model lacks a point (above), and a count of authored paths reads as a
        // glow that isn't there.
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

    /// The base-or-enchant fork (`0x62ec70`): an intrinsic visual wins and **suppresses** the
    /// enchant's, an item without one takes the first enchant slot that carries a visual, and
    /// enchant slots with no visual are walked past rather than ending the scan.
    #[test]
    fn base_visual_wins_over_enchant_and_suppresses_it() {
        let (glows, ench) = catalog(&[25, 61], &[(1, 61), (7, 25)]);
        let ench = Some(&ench);
        // Base 25 exists → base, even with an enchant that also carries one.
        assert_eq!(effective_visual(&glows, ench, 25, [1]), 25);
        // No base → the enchant's.
        assert_eq!(effective_visual(&glows, ench, 0, [1]), 61);
        // A visual-less enchant in the way: the scan continues to the one that has it.
        assert_eq!(effective_visual(&glows, ench, 0, [999, 7]), 25);
        // Nothing anywhere.
        assert_eq!(effective_visual(&glows, ench, 0, [999]), 0);
        // The shipped `-1` base is not a row: it neither glows nor suppresses the enchant.
        assert_eq!(effective_visual(&glows, ench, -1, [1]), 61);
        assert_eq!(effective_visual(&glows, ench, -1, []), 0);
    }

    /// No enchant catalog (the DBC failed to load): the INTRINSIC leg still answers in full, and
    /// only the enchant leg goes quiet — the split's whole safety claim.
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
    fn teebu() -> (benilla_assets::ModelSkeleton, Vec<benilla_assets::ModelAttachment>) {
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
                    let e = app
                        .world_mut()
                        .spawn(Transform::from_translation(at))
                        .id();
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
