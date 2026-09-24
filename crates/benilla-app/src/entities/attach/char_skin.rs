//! Character appearance → per-look skin materials — the *who wears what* half of [`super`].
//!
//! [`super`] (attach) spawns a streamed entity's visual; this module resolves the character-specific
//! inputs it swaps in: the entity's [`CharLook`] (race/sex/customization — wire fields for a player,
//! CreatureDisplayInfoExtra for a character-model NPC, decision 0041), its worn-equipment display ids
//! ([`WornEquip`], decision 0074), and the per-appearance material quints over the composited body
//! atlas / hair / cape textures ([`build_char_skin_materials`], decisions 0044 / 0045).
//!
//! ## The body composite does not run on the browser's frame thread
//!
//! One look's atlas is a base skin plus up to fifteen overlays, each of them a BLP decode, blended
//! per texel — and on the web the thread that would run it is the thread that draws. So the work is
//! cut where the platform cuts it: [`benilla_formats::CompositePlan::read`] pulls the blobs off the
//! chain (which only this thread can reach — see its doc for why the type system does not say so),
//! and [`benilla_formats::CompositeBlobs::blend`] is plain bytes in linear memory and goes to a
//! worker through `AsyncComputeTaskPool::spawn_pooled`. The look is deferred while its blend is in
//! flight, through the same [`SKIN_DEFERRALS`] latch a not-yet-arrived read already used, so no
//! caller learned a new state — a body that is not finished is simply not stamped, and comes back.
//!
//! **Native is unchanged and deliberately so.** There the blend runs inline in the frame it is
//! asked for, exactly as it always has: a native read is a file read, the decode is native-speed,
//! and buying a frame of a naked body to move work off a thread that has cores to spare is not a
//! trade anything has asked for. The gate is a `cfg`, not a runtime dial, so the native arm cannot
//! be reached by a mis-set flag.

use benilla_formats::{CharSkinSlot, ModelBlend};
use benilla_protocol::EntityKind;
use bevy::prelude::*;

use crate::net::{NetEntity, ObjectStore};
use benilla_assets::materials::WowModelMaterial;
use benilla_assets::{repeat_texture_authored, LockRecover, WorldAssets};

use super::super::{DisplayModel, EntityPart, SkinKey, SkinSections};

/// The resolved character appearance for one entity that renders as a character-model body — either a
/// **player** (appearance decoded from the wire) or a **character-model NPC** (from its display's
/// CreatureDisplayInfoExtra, decision 0041). It drives both the geoset selection and the per-appearance
/// skin/hair materials, so the two cases share one code path. `None` for a beast NPC / GameObject / a
/// unit with no appearance data — those render whole, with their built textures, as before.
pub(super) struct CharLook {
    pub(super) race: u8,
    pub(super) sex: u8,
    /// skinColor — keys the composited base skin AND the standalone extra-skin BLP (the tauren fur,
    /// M2 type 8), which even a pre-baked NPC atlas doesn't cover, so it lives here, not in `body`.
    pub(super) skin: u8,
    pub(super) hair_style: u8,
    pub(super) hair_color: u8,
    pub(super) facial_hair: u8,
    /// Where the body-skin atlas comes from.
    pub(super) body: BodySkin,
}

/// The body-skin atlas source for a [`CharLook`].
pub(super) enum BodySkin {
    /// A player (or a character-model NPC row that carries no bake name): composite the atlas live from
    /// CharSections — needs the face (skinColor is on [`CharLook`]) — cached per appearance in
    /// [`SkinComposites`].
    Composite { face: u8 },
    /// A character-model NPC: the pre-baked body atlas the client ships under `Textures\BakedNpcTextures\`
    /// (CreatureDisplayInfoExtra field 18). Loaded directly through the async `mpq://` pipeline — no
    /// compositing (the faithful path: the file ships and the client loads it, it does not re-bake).
    Baked(String),
}

/// Resolve the character look for a net entity, or `None` if it isn't a character-model body. A player
/// takes race/sex + customization from the wire (its [`ObjectStore`] descriptor fields); a
/// character-model NPC takes them from its display's [`NpcAppearance`](benilla_formats::NpcAppearance)
/// — a baked body when the row names one, else a live composite. A beast NPC / GameObject has no look.
pub(super) fn resolve_char_look(
    net: &NetEntity,
    dm: Option<&DisplayModel>,
    entity: Entity,
    stores: &Query<&ObjectStore>,
) -> Option<CharLook> {
    // The look follows the DISPLAY, not the entity kind (decision 0695): with live display-id
    // swaps a Player-kind entity can wear any display — a druid's bear form is a plain creature
    // model (no look; Monster skins instead), and a GM-morphed player wearing a humanoid NPC
    // display wears ITS CreatureDisplayInfoExtra appearance. The reference's own race/gender
    // getters answer from the display's cached row with the descriptor as fallback (wow-re
    // `w2d2.md`, the `0x60c690` getter family) — exactly this order: display appearance first,
    // wire appearance only for a character body that carries none.
    let d = dm?;
    if let Some(npc) = d.npc_appearance.as_ref() {
        // A character-model NPC display (whoever wears it): its CreatureDisplayInfoExtra
        // appearance — a baked body when the row names one, else a live composite.
        return Some(CharLook {
            race: npc.race,
            sex: npc.sex,
            skin: npc.skin,
            hair_style: npc.hair_style,
            hair_color: npc.hair_color,
            facial_hair: npc.facial_hair,
            body: match &npc.bake_name {
                Some(name) => BodySkin::Baked(name.clone()),
                None => BodySkin::Composite { face: npc.face },
            },
        });
    }
    // A **corpse** that is not a bone pile (decision 1706): its look is its own
    // `CORPSE_FIELD_BYTES_1/_2` snapshot, taken at death — not the owner's live `PLAYER_BYTES`,
    // which the corpse cannot see and which may belong to a player who has since logged out. This
    // is the reference's own source: `0x5d6260` reads `[[corpse+0x110]+0x69..+0x6f]` into the
    // corpse's own `CCharacterComponent`. A bone pile builds no component and reads `None` here,
    // which is exactly `0x5d6291`'s early skip.
    if net.kind == EntityKind::Corpse && d.is_character_body {
        let look = super::super::corpse::corpse_char_look(stores.get(entity).ok())?;
        return Some(CharLook {
            race: look.race,
            sex: look.sex,
            skin: look.skin,
            hair_style: look.hair_style,
            hair_color: look.hair_color,
            facial_hair: look.facial_hair,
            body: BodySkin::Composite { face: look.face },
        });
    }
    if net.kind == EntityKind::Player && d.is_character_body {
        // A player wearing a character body. Race/sex come from `UNIT_FIELD_BYTES_0`; the
        // `PLAYER_BYTES` / `PLAYER_BYTES_2` customization is *optional on the wire* — vmangos omits an
        // all-zero field from the create mask, so a fully-default character (like our own "One") sends
        // no `PLAYER_BYTES` at all. Per UpdateFields semantics an absent field means its default (0),
        // not "no data": default each byte to 0 rather than gate the whole look on its presence (an
        // over-strict `?` here left a default-customized avatar unskinned — the pre-0061 component
        // path defaulted these implicitly).
        let s = &stores.get(entity).ok()?.0;
        return Some(CharLook {
            race: s.unit_race()?,
            sex: s.unit_gender()?,
            skin: s.player_skin().unwrap_or(0),
            hair_style: s.player_hair_style().unwrap_or(0),
            hair_color: s.player_hair_color().unwrap_or(0),
            facial_hair: s.player_facial_hair().unwrap_or(0),
            body: BodySkin::Composite {
                face: s.player_face().unwrap_or(0),
            },
        });
    }
    // A beast display (whoever wears it), a GameObject, a model-less display: no look.
    None
}

/// How many body composites have been put off — because their textures had not arrived, or because
/// the blend they need is still running on a worker. The signal `attach_entity_visuals` reads to
/// know it must come back to a unit.
///
/// A counter rather than a return value because the answer has to cross
/// [`build_char_skin_materials`], whose four-part return is already the whole of a character's
/// materials and has no room for a status. The caller brackets the call: unequal before and after
/// means "this unit was not finished", and it simply does not stamp `VisualAttached`.
///
/// Permanently zero on native, where `Chain::ready` never says no and the blend is never deferred.
static SKIN_DEFERRALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Note one deferral, and answer `None` — the shape every "come back next frame" arm below wants.
fn defer<T>() -> Option<T> {
    SKIN_DEFERRALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    None
}

/// Read [`SKIN_DEFERRALS`]. Snapshot around [`build_char_skin_materials`]; a change means the unit
/// must be attached again next frame.
pub(crate) fn skin_deferrals() -> u64 {
    SKIN_DEFERRALS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Upload a finished atlas and cache it under the look it answers.
///
/// Through the upload gate like every other texture: a composite is layered texel-by-texel on the
/// CPU so it is already RGBA8, and `for_upload` is a no-op on it — but going through it is what
/// makes the format and the bytes provably agree (decision 1626).
fn upload_composite(
    ready: &mut benilla_assets::SpatialCache<SkinKey, Handle<Image>>,
    images: &mut Assets<Image>,
    key: SkinKey,
    composed: benilla_formats::BlpMipChain,
) -> Option<Handle<Image>> {
    let handle = images.add(repeat_texture_authored(
        benilla_assets::for_upload(composed),
        (true, true),
    ));
    ready.insert(key, handle.clone());
    Some(handle)
}

/// A finished blend, or `None` while it is still running. One place, because the composite arm and
/// [`drain_skin_composites`] must not disagree about what "done" means.
fn poll_blend(task: &mut BlendTask) -> Option<Option<benilla_formats::BlpMipChain>> {
    bevy::tasks::block_on(bevy::tasks::futures_lite::future::poll_once(task))
}

/// One look's blend, in flight.
type BlendTask = bevy::tasks::Task<Option<benilla_formats::BlpMipChain>>;

/// Hand the CPU half of a composite to a worker — `Ok(task)` — or give the blobs back when there is
/// no worker to take it, in which case the caller blends inline exactly as this always did.
///
/// **Native always gets the blobs back, by `cfg` rather than by a runtime answer.** There the blend
/// runs in the frame it was asked for and always has: a native read is a file read, the decode is
/// native-speed, and buying a frame of an undressed body to move work off a thread with cores to
/// spare is a trade nothing has asked for. Keeping it a `cfg` means the native arm cannot be
/// reached by a mis-set flag, and that native behaviour does not move — the rule the rest of the
/// offload work in this tree holds to.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_blend(
    blobs: benilla_formats::CompositeBlobs,
) -> Result<BlendTask, benilla_formats::CompositeBlobs> {
    Err(blobs)
}

/// See the native definition above for the contract; this is the browser half.
///
/// The `Send` bound on `spawn_pooled` is what makes this safe rather than merely intended:
/// [`benilla_formats::CompositeBlobs`] is `Vec<u8>`s and `Copy` rects, and a payload that could
/// reach a `JsValue`, the DOM or the pack map would not compile here. That is exactly the wall
/// `CompositePlan::read` exists to put the chain on the other side of.
#[cfg(target_arch = "wasm32")]
fn spawn_blend(
    blobs: benilla_formats::CompositeBlobs,
) -> Result<BlendTask, benilla_formats::CompositeBlobs> {
    // `cfg`, not just the runtime gate: `spawn_pooled` does not exist on a wasm build without
    // atomics (`bevy_tasks` swaps in `single_threaded_task_pool.rs`), so this has to be compiled
    // away rather than merely skipped. `pooled_work_runs` answers `false` there too, so the two
    // agree; the attribute is what keeps the agreement from having to compile.
    #[cfg(target_feature = "atomics")]
    if benilla_assets::frame_budget::pooled_work_runs() {
        return Ok(bevy::tasks::AsyncComputeTaskPool::get()
            .spawn_pooled(async move { blobs.blend().ok() }));
    }
    // No worker to take it. Inline on the frame thread, which is what this build always did — and
    // deliberately NOT under a `frame_budget` slot: the caller is a system, not an `async` loader,
    // so there is nothing here that can await one.
    Err(blobs)
}

/// Land every body composite a worker has finished, and reap the ones nobody is waiting for.
///
/// The composite arm polls its own key, so this is not what makes a waiting body dress — it is what
/// stops a look from leaking. A job outlives its asker whenever the unit that wanted it walks out
/// of the streamer's window (or the booth closes) between the spawn and the finish, and a finished
/// `BlpMipChain` nobody drains is a megabyte and a half held for the session. Draining it into the
/// cache also means the look is *ready* if that body comes back, which is the same bet the cache
/// itself makes.
///
/// Compiles to a walk over an always-empty map on native, where nothing ever queues a blend.
pub(crate) fn drain_skin_composites(
    mut composites: ResMut<super::super::SkinComposites>,
    mut images: ResMut<Assets<Image>>,
) {
    if composites.blending.is_empty() {
        return;
    }
    // Destructured, so the two halves are disjoint borrows and a finished atlas can be uploaded
    // in place instead of being collected and walked a second time.
    let super::super::SkinComposites { ready, blending } = composites.as_mut();
    blending.retain(|key, task| {
        let Some(blended) = poll_blend(task) else {
            return true; // still running
        };
        // A blend that failed drops out of the map with nothing cached — the next asker plans and
        // reads it again, which is what the inline version did too.
        if let Some(chain) = blended {
            upload_composite(ready, &mut images, *key, chain);
        }
        false
    });
}

/// The `mpq://` URL for a pre-baked NPC body atlas (a CreatureDisplayInfoExtra bake name) under
/// `Textures\BakedNpcTextures\`. Lowercased forward-slash, like the other `mpq://` loads.
fn baked_npc_url(bake_name: &str) -> String {
    format!(
        "mpq://textures/bakednpctextures/{}",
        bake_name.replace('\\', "/").to_ascii_lowercase()
    )
}

/// The worn-equipment display ids that drive a character body's geoset selection (the equipment
/// branches in [`benilla_formats::CharacterGeosets::visible_geosets`]) and — for a player — the
/// region-texture composite: the 8 armor bodyslots (shirt..tabard, the [`EquipGeosets`] index order),
/// the cloak, and the helm.
///
/// A **player** takes them from its resolved [`super::Equipment`] (the wire's visible-item entries →
/// item template → display id, decision 0074). A **character-model NPC** takes them from its display's
/// [`NpcAppearance`](benilla_formats::NpcAppearance) equipment columns (CreatureDisplayInfoExtra —
/// decision 0060 named the gap; this is the fill): its bodyslots 2..9 (shirt..tabard) map straight onto
/// the 8 slots, field 0 (head) is the helm, and there is no NPC cloak column (the row stops at bodyslot
/// 9). A beast NPC / GameObject / a unit with no appearance row yields the all-zero naked default.
#[derive(Default)]
pub(super) struct WornEquip {
    /// shirt · chest · belt · pants · boots · wrist · gloves · tabard (bodyslot 2–9).
    pub(super) bodyslots: [u32; 8],
    pub(super) cloak: u32,
    pub(super) helm: u32,
    /// The wearer's guild tabard (decision 1704) — a **player's** only, off their resolved
    /// [`super::Equipment`]. A character-model NPC has no guild: CreatureDisplayInfoExtra carries
    /// no guild column, and a display-driven body never joins one, so this stays `None` there and a
    /// tabard in an NPC's bodyslot-9 column keeps its own art.
    pub(super) emblem: Option<benilla_formats::GuildEmblem>,
    pub(super) tabard_preview: bool,
}

pub(super) fn resolve_worn_equip(
    net: &NetEntity,
    equipment: Option<&super::super::Equipment>,
    dm: Option<&DisplayModel>,
) -> WornEquip {
    match net.kind {
        // A **corpse** rides the player arm (decision 1706): its `Equipment` is resolved from the
        // 19 `CORPSE_FIELD_ITEM` slots, which are already ItemDisplayInfo ids — the same values a
        // player's items resolve to, so the geoset + region-composite law downstream is literally
        // the same law, on a different source.
        EntityKind::Player | EntityKind::Corpse => equipment
            .map(|e| WornEquip {
                bodyslots: e.bodyslots,
                cloak: e.cloak,
                helm: e.helm,
                emblem: e.emblem,
                tabard_preview: e.tabard_preview,
            })
            .unwrap_or_default(),
        // A character-model NPC's worn gear ships in its display's CreatureDisplayInfoExtra columns
        // (bodyslot-indexed) — the same ItemDisplayInfo ids a player's items resolve to. bodyslots
        // 2..9 map directly onto the 8 armor slots; field 0 is the helm; there is no cloak column.
        EntityKind::Unit => dm
            .and_then(|d| d.npc_appearance.as_ref())
            .map(|npc| WornEquip {
                bodyslots: std::array::from_fn(|i| npc.equipment[i + 2]),
                cloak: 0,
                helm: npc.equipment[0],
                emblem: None,
                tabard_preview: false,
            })
            .unwrap_or_default(),
        _ => WornEquip::default(),
    }
}

/// The worn geoset selectors for a set of equipment display ids (decisions 0074/1864's B1–B8
/// branches + the cloak group + the helm's RF-0083 hide-mask row pair): each non-zero display
/// resolves its ItemDisplayInfo row's geoset columns, and B3's forearm gate comes off the same
/// composite plan the atlas blits. One helper for the world attach path and the glue-preview
/// builder — the selection law can't fork (decision 0465).
///
/// [`EquipGeosets::tabard_preview`](benilla_formats::EquipGeosets::tabard_preview) (B6) is the
/// tabard designer's flag, up on the local player's body while that window is open (decision
/// 1977) — its only setter in the reference too.
pub(in crate::entities) fn equip_geosets(
    displays: Option<&super::super::ItemDisplays>,
    bodyslots: &[u32; 8],
    cloak: u32,
    helm: u32,
    tabard_preview: bool,
) -> benilla_formats::EquipGeosets {
    let mut eg = benilla_formats::EquipGeosets {
        tabard_preview,
        ..Default::default()
    };
    if let Some(d) = displays {
        let mut worn: [Option<&benilla_formats::ItemDisplay>; 8] = [None; 8];
        for (i, id) in bodyslots.iter().enumerate() {
            if *id != 0 {
                worn[i] = d.catalog.get(*id);
                eg.bodyslots[i] = worn[i].map(|row| row.geoset_groups);
            }
        }
        // B3's gate is the ArmLower tile's own occupancy, not "is a chest equipped" — the same
        // plan the composite blits (decision 1864).
        eg.forearm_dressed = benilla_formats::forearm_dressed(&worn);
        if cloak != 0 {
            eg.cloak = d.catalog.get(cloak).map(|row| row.geoset_groups[0]);
        }
        if helm != 0 {
            // Only a display that names a head MODEL is a worn helm, and only a worn helm tucks
            // hair/facial/ears away (B93: `CreatureDisplayInfoExtra`'s head column points 126
            // character-model NPC displays at model-less jewellery rows that still carry a full
            // hide mask — see [`benilla_formats::ItemDisplay::worn_helm_vis`]).
            eg.helm_vis = d
                .catalog
                .get(helm)
                .and_then(benilla_formats::ItemDisplay::worn_helm_vis);
        }
    }
    eg
}

/// One per-appearance material set for a character runtime texture slot: (steady, interior-matte,
/// appear-fade-blend, interior-bake, interior-bake-blend, depth-prime twin). `model_material` /
/// `zfill_material` dedup by texture+blend, so all players of one look share them.
pub(super) type MatQuint = (
    Handle<WowModelMaterial>,
    Handle<WowModelMaterial>,
    Handle<WowModelMaterial>,
    Handle<WowModelMaterial>,
    Handle<WowModelMaterial>,
    Option<Handle<WowModelMaterial>>,
);

/// The full character material set [`build_char_skin_materials`] returns: `(body, hair, object,
/// skin_extra)`, the body + skin-extra each a (single-sided, two-sided) pair. Each per-slot
/// [`MatQuint`] is `None` for an absent row (a bald style, a non-fur race) or missing tables. Named
/// so the create-preview builder (`super::create_preview`) can select the steady variant per part.
pub(super) type CharSkinMaterials = (
    Option<(MatQuint, MatQuint)>,
    Option<MatQuint>,
    Option<MatQuint>,
    (Option<MatQuint>, Option<MatQuint>),
);

/// The `WOW_PROBE_SHARED_SKIN` pricing lever (see its use in [`build_char_skin_materials`]).
fn shared_skin_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("WOW_PROBE_SHARED_SKIN").is_some())
}

/// Build a character body's per-appearance materials — the **body** atlas (as a (single-sided,
/// two-sided) pair — a body batch keeps its own M2 0x04, e.g. the robe skirt) and the **hair**-mesh
/// texture (a single CharSections BLP, decision 0045) — each as a (steady, interior-matte, fade,
/// interior-bake) quad. Works for a
/// player (atlas composited live from CharSections + overlays, decisions 0041 / 0044) *and* a
/// character-model NPC (atlas = the shipped pre-baked BLP), selected by [`CharLook::body`]. Returns
/// `(body, hair, object, skin_extra)`; each is `None` for an absent row (e.g. a bald style has no hair
/// texture, only tauren author an extra skin) or when the tables / world chain / lighting aren't
/// available (those parts then keep their built, untextured material). A composited atlas is uploaded once per look ([`super::super::SkinComposites`] cache); a baked or hair
/// BLP loads through the async `mpq://` pipeline (which dedups by path). `parts` supplies the hair
/// batches' blend (hair is alpha-cut, so it can't be forced opaque like the body).
#[allow(clippy::type_complexity)]
pub(super) fn build_char_skin_materials(
    look: &CharLook,
    // The worn armor display ids (bodyslot 2–9) + the cloak's, and the ItemDisplayInfo catalog to
    // resolve their region textures (decision 0074). Consumed **only** on the live-composite path
    // (a player, or the rare bake-less NPC row); a baked NPC atlas already owns the skin, so its
    // equip ids don't paint here — they drive the geosets only. `[0; 8]`/`0`/`None` = the naked body.
    equip: [u32; 8],
    cloak: u32,
    // The wearer's guild tabard (decision 1704), painted over the torso layers of a tabard whose
    // display asks for it. `None` = no guild, or its identity has not arrived; both leave the
    // tabard garment showing its own art.
    emblem: Option<benilla_formats::GuildEmblem>,
    tabard_preview: bool,
    displays: Option<&super::super::ItemDisplays>,
    sections: Option<&SkinSections>,
    world_assets: Option<&WorldAssets>,
    parts: &[EntityPart],
    images: &mut Assets<Image>,
    composites: &mut super::super::SkinComposites,
    asset_server: &AssetServer,
    mats: &mut benilla_world::model_render::M2BatchMaterials,
) -> CharSkinMaterials {
    let (Some(sections), true) = (sections, mats.ready()) else {
        return (None, None, None, (None, None));
    };
    // The quint IS the engine's variant set, in the order the character swap reads it: steady,
    // interior-matte, appear-fade-blend, interior-bake, interior-bake-blend, depth-prime twin.
    let quint = |v: benilla_world::model_render::BatchVariants| {
        (
            v.steady,
            v.interior,
            v.fade_blend,
            v.interior_bake,
            v.interior_bake_blend,
            v.zfill,
        )
    };

    // Body-skin atlas. A character-model NPC loads its shipped, pre-baked atlas directly; a player (or an
    // NPC row without a bake name) composites it live from CharSections + overlays, uploaded + cached
    // once per appearance ([`SkinComposites`]).
    //
    // **This is the last blocking chain read on a gameplay path, and on the web it is a real one.**
    // A stranger of a race nobody has worn yet composites from `Character\<Race>\<Sex>\` and
    // `Item\TextureComponents\`, and a pack that is not resident costs a fetch on the frame thread.
    // It is bounded rather than continuous — the reads are locality groups, so it is roughly one
    // fetch per race/gear family for the session, not one per stranger — which is why it did not
    // take `Chain::try_read` with the sprite caches and the sound kits. Deferring here needs a look
    // to be *revisited* (this runs on an appearance edge, and the skin cache is only written on
    // success), so the honest fix is a retry latch on the entity, not a `None` that would leave a
    // player permanently untextured. That latch is `SKIN_DEFERRALS`, and it is what the off-thread
    // blend below rides on too: "still coming" and "still blending" are the same answer to the
    // caller, so moving the CPU half to a worker needed no new state anywhere above this function.
    let body_tex: Option<Handle<Image>> = match &look.body {
        BodySkin::Baked(name) => Some(asset_server.load::<Image>(baked_npc_url(name))),
        BodySkin::Composite { face } => world_assets.and_then(|world| {
            let mut key = SkinKey {
                race: look.race,
                sex: look.sex,
                skin: look.skin,
                face: *face,
                facial_hair: look.facial_hair,
                hair_style: look.hair_style,
                hair_color: look.hair_color,
                equip,
                emblem,
                tabard_preview,
            };
            // `WOW_PROBE_SHARED_SKIN=1` — a PRICING lever, never a look: every body composites
            // the same key, so every body part of one mesh shares one material and bevy's
            // batcher can instance them. What it measures is the ceiling of the shared-body-
            // material lane (draw count → render-thread CPU) before that lane is built.
            if shared_skin_probe() {
                key = SkinKey {
                    race: 1,
                    sex: 0,
                    skin: 0,
                    face: 0,
                    facial_hair: 0,
                    hair_style: 0,
                    hair_color: 0,
                    equip: [0; 8],
                    emblem: None,
                    tabard_preview: false,
                };
            }
            if let Some(handle) = composites.ready.fetch(&key) {
                return Some(handle);
            }
            // Already blending on a worker: poll it here rather than only in the drain pass, so a
            // look that finished since the last frame dresses this body now instead of next frame.
            // Not finished ⇒ the same deferral a not-yet-arrived read takes.
            if let Some(task) = composites.blending.get_mut(&key) {
                let Some(done) = poll_blend(task) else {
                    return defer();
                };
                composites.blending.remove(&key);
                // A blend that failed answers like the fused version's `.ok()?` always did: no
                // atlas, nothing cached, and this body keeps its built material.
                return upload_composite(&mut composites.ready, images, key, done?);
            }
            // The worn ItemDisplayInfo rows whose region textures dress the atlas
            // (decision 0074); an unknown/zero display id contributes nothing.
            let catalog = displays.map(|d| &d.catalog);
            let mut worn: [Option<&benilla_formats::ItemDisplay>; 8] = [None; 8];
            if let Some(catalog) = catalog {
                for (i, id) in equip.iter().enumerate() {
                    if *id != 0 {
                        worn[i] = catalog.get(*id);
                    }
                }
            }
            // The chain guard is scoped: nothing past the read needs it, and holding a lock
            // across a task spawn (or an `Assets<Image>` insert) is a habit worth not forming.
            let blobs = {
                let chain = &mut world.chain.lock_recover();
                // What this appearance needs, worked out before a byte is read
                // (`CharSections::composite_plan`). The split exists for the browser: the atlas
                // below is blended once and cached per appearance for the session, so a layer
                // that has not arrived yet cannot be skipped and picked up later — it would be
                // a tabard missing for as long as the player wears it. `ready` warms every path
                // and says whether they are all in memory; until they are, this whole unit's
                // attach is deferred (see the caller) rather than the frame blocked on a fetch.
                // On native `ready` is always true and none of this runs.
                let plan = sections.0.composite_plan(
                    key.race,
                    key.sex,
                    key.skin,
                    key.face,
                    key.facial_hair,
                    key.hair_style,
                    key.hair_color,
                    worn,
                    key.emblem,
                    key.tabard_preview,
                )?;
                // A full pass, not `all`: every path must be ASKED, because asking is what starts
                // the warm. A short-circuit would warm the first missing one and leave the rest
                // for the next frame to discover, one round trip at a time.
                let ready = plan.paths().filter(|p| !chain.ready(p)).count() == 0;
                if !ready {
                    return defer();
                }
                // The chain half, and the only half that has to be here
                // (`CompositePlan::read`). Every path is resident by the line above, so this is
                // a memcpy out of the pack map and a frame decode each — not a fetch.
                plan.read(chain).ok()?
            };
            // …and the CPU half, which is plain bytes and goes to a worker where there is one.
            match spawn_blend(blobs) {
                Ok(task) => {
                    composites.blending.insert(key, task);
                    defer()
                }
                Err(blobs) => {
                    upload_composite(&mut composites.ready, images, key, blobs.blend().ok()?)
                }
            }
        }),
    };
    // Body atlas: a (single-sided, two-sided) quint pair, chosen per body batch at the swap. The
    // naked body is a closed single-sided mesh (the approved decision-0044 look), but body-slot
    // batches carry their own M2 0x04: every race's robe skirt (geoset 1302) — and the undead
    // ragged-trouser batch — is authored two-sided, and flattening it to the closed-body default
    // culled the robe's inner faces (see-through from below). `model_material` dedups by key, so
    // the second quint is a handful of cache entries per look, shared like the first.
    let body = body_tex.map(|tex| {
        (
            quint(
                mats.char_variants(tex.clone(), ModelBlend::Opaque, false)
                    .expect("light buffer checked at entry"),
            ),
            quint(
                mats.char_variants(tex, ModelBlend::Opaque, true)
                    .expect("light buffer checked at entry"),
            ),
        )
    });

    // Hair-mesh texture — a single CharSections BLP loaded through the async pipeline (dedups by path).
    // Alpha-cut, so it takes the hair batches' own blend. `None` for a model with no hair part. Same
    // for a player + an NPC: the baked/composited body atlas covers the head *skin*, but the 3D hair
    // geometry is a separate geoset with its own texture, keyed here — and that same type-6 unit also
    // dresses the *facial* hair on the races whose beards are geometry, so this resolves through
    // `hair_mesh_texture`'s bald fallback rather than the raw row (a bald orc/gnome still has a beard
    // to texture; taking the blank row left it flat white).
    let hair = sections
        .0
        .hair_mesh_texture(look.race, look.sex, look.hair_style, look.hair_color)
        .and_then(|path| {
            // Hair cards are alpha-cut + two-sided (M2 `0x04`); carry both onto the swapped material.
            let hair_part = parts
                .iter()
                .find(|p| p.char_slot == Some(CharSkinSlot::Hair))?;
            let tex = asset_server.load::<Image>(format!(
                "mpq://{}",
                path.replace('\\', "/").to_ascii_lowercase()
            ));
            mats.char_variants(tex, hair_part.blend, hair_part.two_sided)
                .map(&quint)
        });

    // Cape texture (decision 0074's empirical pin): the worn cloak's ItemDisplayInfo
    // `model_texture[0]` is the Cape BLP basename under `Item\ObjectComponents\Cape\`, bound to the
    // body's runtime type-2 batches ([`CharSkinSlot::Object`]) — the cloak-geoset skin. `None`
    // (cloak-less, unknown display, no Object part) keeps those parts' built material — moot when
    // the cloak geoset branch has them hidden anyway.
    let object = (cloak != 0)
        .then_some(())
        .and_then(|()| displays?.catalog.get(cloak)?.model_texture[0].as_deref())
        .and_then(|tex_name| {
            let part = parts
                .iter()
                .find(|p| p.char_slot == Some(CharSkinSlot::Object))?;
            let tex = asset_server.load::<Image>(format!(
                "mpq://item/objectcomponents/cape/{}.blp",
                tex_name.to_ascii_lowercase()
            ));
            mats.char_variants(tex, part.blend, part.two_sided)
                .map(&quint)
        });

    // Extra-skin texture (the tauren fur, M2 type 8) — CharSections `sectionType 0` `TextureName[1]`
    // keyed by skinColor, loaded plain through the async pipeline (the client's extra loader is a bare
    // TextureCreate — no compositing; a pre-baked NPC atlas doesn't cover it either, so this applies to
    // players and NPCs alike). The batches come in exactly two authored flavors — the opaque
    // single-sided fur core and the alpha-cut two-sided fringe cards — so build one quint per flavor
    // from a representative part (a flavor no part carries is never selected). Empty column (every
    // non-fur race) ⇒ `(None, None)`; those models carry no type-8 batch anyway.
    let skin_extra = sections
        .0
        .skin_extra_texture(look.race, look.sex, look.skin)
        .map_or((None, None), |path| {
            let tex = asset_server.load::<Image>(format!(
                "mpq://{}",
                path.replace('\\', "/").to_ascii_lowercase()
            ));
            let mut quint_for = |two_sided: bool| {
                let part = parts.iter().find(|p| {
                    p.char_slot == Some(CharSkinSlot::SkinExtra) && p.two_sided == two_sided
                })?;
                mats.char_variants(tex.clone(), part.blend, part.two_sided)
                    .map(&quint)
            };
            (quint_for(false), quint_for(true))
        });

    (body, hair, object, skin_extra)
}
