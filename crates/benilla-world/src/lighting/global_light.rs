//! The one shared global light, as the reference has a single scene light every draw reads.
//!
//! One persistent storage buffer, which every material binds at `storage(90)`: [`build_light_data`]
//! packs the resolved light each frame and [`upload_light`] writes it in place. Material assets
//! are never mutated after creation, since Bevy rebuilds the bind group of a mutated material.

use benilla_formats::LiquidKind;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::{Buffer, BufferDescriptor, BufferUsages};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{Render, RenderApp, RenderSystems};

use super::prop_probes::MAX_PROP_PROBES;
use super::{sh, WowLighting};
use crate::dev_state::DebugState;
use crate::view::ViewDistance;
use crate::view::WorldCamera;

/// The shared light, std430-packed as `vec4<f32>` rows (all `vec4`, so std430 equals std140). Every
/// shader that binds the buffer (`wow_model`, `terrain`, `liquid`, `wdl`, `wow_effect`,
/// `static_gx`) mirrors this row order as a prefix; keep them in sync.
///   0 light_ambient (w=Mod2x 1.0) · 1 light_diffuse (w=clamp on) · 2 light_sun (w=dir/SH enable) ·
///   3 light_spec (w=terrain shininess 20) · 4 fog_color (w=enable) ·
///   5 fog_params (x=start y=end w=farclip) · 6-8 sh_c10_{r,g,b} · 9-11 sh_c13_{r,g,b} ·
///   12 sh_c16 · 13-14 water river {shallow,deep} (w=alpha) · 15-16 water ocean (the same) ·
///   17 grade: `.x` the SIDN night fraction, `.yzw` the sun's SH DC at intensity 1 ·
///   18 wmo_fog_color · 19 wmo_fog_params (x=start y=end): the interior fog triple, read by
///      interior-tagged content in place of 4-5 ·
///   20 point_count (x = live entries) · 21+ the point-light table, two rows per light,
///      `[pos.xyz, range]` and `[rgb, 0]`, here because Bevy's clusterable buffer is
///      fragment-only in the view layout and the point term is per vertex.
///
/// The GPU buffer is larger: the interior-prop probes and the skin-palette regions follow this
/// prefix. They stay out of this struct because the extract clones it by value every frame, and
/// ~900 KB overflowed a render thread's stack.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct LightStd430 {
    rows: [[f32; 4]; LIGHT_HEADER_ROWS],
    points: [[f32; 4]; 2 * MAX_POINT_LIGHTS],
}

/// Header row count of the layout above (rows 0..=20), which every light blob sizes against.
pub const LIGHT_HEADER_ROWS: usize = 21;

/// Packs the model-lighting core from `(ambient, diffuse, sun_dir)` and leaves every other lane
/// alone: rows 0-2, the SH block (6-11 and 12 `.xyz`, ambient in the DC `.w` lanes) and the sun's
/// SH DC on 17 `.yzw`, the exterior M2 lane's light. The sun's bands are `Model2.bls`'s
/// `E(n) = D·(3 + 16μ + 15μ²)/34`, μ = n·u toward the light, the interior lane's
/// [`sh::prop_probe_coeffs`] fold: all linear in `D`, so a consumer scales each by the intensity
/// (packed at 1). Producers call this and never copy the layout.
pub fn pack_model_core_rows(
    rows: &mut [[f32; 4]; LIGHT_HEADER_ROWS],
    ambient: [f32; 3],
    diffuse: [f32; 3],
    sun_dir: Vec3,
) {
    rows[0] = [ambient[0], ambient[1], ambient[2], 1.0]; // 0 light_ambient (w=Mod2x 1.0)
    rows[1] = [diffuse[0], diffuse[1], diffuse[2], 1.0]; // 1 light_diffuse (w=clamp on)
    rows[2] = [sun_dir.x, sun_dir.y, sun_dir.z, 1.0]; // 2 light_sun (w=dir/SH enable 1.0)
    let sun = sh::prop_probe_coeffs([0.0; 3], &[(-sun_dir, diffuse)]);
    // The sun folds at intensity 1 without ambient: ambient takes the DC lanes unscaled, and the
    // fold's own DC moves to row 17 `.yzw`, where the shader scales it by the intensity.
    for (i, row) in sun.iter().enumerate().take(6) {
        rows[6 + i] = row.to_array(); // 6-8 sh_c10_{r,g,b} · 9-11 sh_c13_{r,g,b}
    }
    rows[6][3] = ambient[0]; // the DC lanes carry ambient alone
    rows[7][3] = ambient[1];
    rows[8][3] = ambient[2];
    rows[12][0] = sun[6].x; // 12 sh_c16 xyz; .w is free
    rows[12][1] = sun[6].y;
    rows[12][2] = sun[6].z;
    // 17 `.yzw`: the sun's SH DC at intensity 1; `.x` (SIDN) is the scene's.
    rows[17][1] = sun[0].w;
    rows[17][2] = sun[1].w;
    rows[17][3] = sun[2].w;
}

/// The reference commits a point light's diffuse raw, `colour × intensity × modelFade` (composed
/// at `0x716a67`), over-gamut included: `0x71ca80` stores a peak-normalized byte colour and the
/// peak `max(1, r, g, b)`, and `0x593040` decodes them back before the GL light is set. Saturation
/// comes from the per-vertex clamp of the summed lighting, never from the commit. Deviation: the
/// round trip's 8-bit quantization is skipped, because it moves a channel by under 0.5 %.
pub fn commit_raw(rgb: [f32; 3]) -> [f32; 3] {
    rgb.map(|c| c.max(0.0))
}

/// Capacity of the point-light table, fixed in the WGSL mirror structs (keep in sync); past it
/// [`build_light_data`] keeps the lights nearest the camera.
pub(super) const MAX_POINT_LIGHTS: usize = 256;

/// Lights are packed only within this camera distance (yd): a light's effect ends at its 48 yd
/// range, a pool past 300 yd is sub-pixel and fogged, and the cap bounds the per-vertex walk.
const POINT_PACK_RADIUS: f32 = 300.0;
/// fork: P8 — **the modern frame's pack radius**, and the far edge of a smooth distance fade
/// rather than a hard cut. A modern frame lights every fragment from a grid, not from a unit's three
/// nearest, so whatever this admits is visible wherever the grid reaches — and an edge here is a
/// lamp switching on as the player walks toward it. Lights fade out over the last
/// [`MODERN_PACK_FADE`] yards of it instead, and the emissive tiers' far grid level covers this plus a
/// light's own range (`modern::lights`' test pins that).
pub const MODERN_PACK_RADIUS: f32 = 270.0;
/// fork: P8 — the width of that fade, in yards inside the radius in force.
const MODERN_PACK_FADE: f32 = 60.0;
/// fork: P8 — how fast the radius in force closes in when more lights want the table than it holds
/// (yd/s), and how fast it opens back out when they no longer do. Shrinking is the urgent
/// direction: until it has, the cap below is truncating nearest-first again.
const PACK_SHRINK_RATE: f32 = 150.0;
const PACK_GROW_RATE: f32 = 40.0;

/// fork: the modern lane's gain on every local light. Its surfaces are lit in linear HDR under a
/// tonemap, a probe field and a night haze, where the reference's committed intensities read a little
/// weak: a brazier lit its own bowl and not the square around it. The classic lane never takes it.
const MODERN_LOCAL_GAIN: f32 = 1.3;

/// fork: P8 — **a light that only shines at night**: a WMO window's glow, derived from its MOMT
/// SIDN material (`terrain_stream::spawn`). The packer multiplies its colour by the SIDN night
/// fraction — the same ramp the window's own emissive texture follows on screen — so the street
/// under a lit window brightens as the window does at dusk and goes dark with it at dawn.
#[derive(Component, Clone, Copy, Debug)]
pub struct NightGlow;

/// fork: **the light a luminous spell throws** — a Fireball's core, a Fire Nova's flash, a holy
/// blessing (`benilla_formats::spell_light`, spawned and enveloped by the app's spell-effect lanes).
/// Packed only on the modern tiers that bin lights into a grid (Medium and up): the classic
/// three-nearest lane has three slots per vertex, and a volley of spells would push the torches
/// out of them. Never a shadow caster — it moves and it is gone in a second.
#[derive(Component, Clone, Copy, Debug)]
pub struct SpellFxLight;

/// fork: P8 — how fast a light ramps into and out of the committed table, in units of fade per
/// second. ~4 frames at 60 Hz: long enough that no edge reads as a switch, short enough that a
/// lamp is at full strength by the time the player has finished stepping through the door.
const ADMIT_FADE_RATE: f32 = 15.0;

/// The rooms a point light belongs to: a WMO's MOLT fixture (the groups whose MOLR names it) or a
/// prop's M2 light (the groups whose MODR names the prop). The reference admits a WMO's props only
/// in frames the portal walk visits their group (`0x6838f0`, from `0x685d70`), so a culled room's
/// torch registers no light; [`build_light_data`] drops the light while its rooms are culled. A
/// newtype, since a bare [`crate::wmo_portal::WmoGroupVis`] would enlist the light in
/// `apply_model_visibility`.
#[derive(Component)]
pub struct LightRooms(pub(crate) crate::wmo_portal::WmoGroupVis);

/// fork: P8 — **the body an ENTITY-lane light belongs to**, so its [`LightRooms`] can follow that
/// body from room to room.
///
/// A placed light's rooms are authored and static: a WMO prop's MOLR list names them once at spawn
/// and they never change ([`crate::terrain_stream`]'s spawner). An entity's are neither. A
/// GameObject brazier, a torch-bearing NPC and a glowing sword are all standing somewhere, that
/// somewhere is a room or it is the open world, and it changes when they walk through a door — so
/// the claim has to be re-read, and it has to be re-read from the BODY rather than from the light,
/// which may hang three levels down a bone hierarchy from it.
///
/// **Without this every entity light was refused by every interior surface in the game.** The
/// receiver-side gate is `static_gx.wgsl`'s `if (interior && …w < 0.5) { continue; }` — a light
/// that names no rooms may not light an interior one, because a local light has no occluder unless
/// it holds a cube slot and a street lamp would otherwise light the inside of the chapel beside it
/// through the wall. That rule reads "names no rooms" as "is outdoors", which is true of an ADT
/// doodad's lamp and false of everything on this lane: nothing in `benilla-app` ever attached
/// [`LightRooms`], so `w` was 0 for every one of them, and a brazier standing in Orgrimmar lit
/// nothing at all. Reported against a "Mighty Blaze" that cast no light on the floor it stood on.
#[derive(Component, Clone, Copy)]
pub struct LightOwner(pub Entity);

/// fork: P8 — keep an entity light's [`LightRooms`] equal to the room its [`LightOwner`] stands in.
///
/// Cheap by construction: it walks the lights, not the bodies, and the world holds a handful of
/// entity lights against thousands of bodies. The owner's own claim
/// ([`crate::wmo_portal::UnitWmoRoom`]) is what it reads, so the expensive part — the down-ray that
/// decides which room a body is in — is already paid for by the tracker that publishes it, once per
/// body per MOVE rather than once per frame.
///
/// One group, not a set: a prop is named by every group whose MODR lists it and any of them being
/// drawn is a chance to be drawn, but a body *stands in* exactly one room. The portal gate over it
/// is then the same one a prop's light gets, and it is the behaviour we want — a brazier in a room
/// the flood has culled lights nothing, which is what `room_admits_light` already says for props.
fn track_entity_light_rooms(
    mut commands: Commands,
    lights: Query<(Entity, &LightOwner, Option<&LightRooms>), With<WorldPointLight>>,
    owners: Query<&crate::wmo_portal::UnitWmoRoom>,
    parents: Query<&ChildOf>,
) {
    for (light, owner, had) in &lights {
        // **Up to the nearest BODY, not just to the named entity.** `LightOwner` names what the
        // spawn site had in hand, and that is not always the thing standing in the room: a weapon
        // enchant's light is owned by the ITEM root, which is itself a child of the wielder, and
        // only the wielder is a wire body with a claim. Walking up covers all three lanes with one
        // rule instead of asking each spawn site to know how deep it sits.
        //
        // Bounded, because an unbounded parent walk over a malformed hierarchy is a hang rather
        // than a bug: a light is a child of a bone of a rig of a body at the very worst, and the
        // chain that misses at eight is a chain that has nothing to find.
        let mut at = owner.0;
        let mut want = None;
        for _ in 0..8 {
            if let Ok(claim) = owners.get(at) {
                want = claim.room();
                break;
            }
            match parents.get(at) {
                Ok(parent) => at = parent.parent(),
                Err(_) => break,
            }
        }
        // Only on a real change: `LightRooms` holds an `Arc`, and rebuilding one every frame for
        // every torch in a city is an allocation per light per frame for an answer that moves at
        // walking pace.
        let same = match (&want, had) {
            (Some(r), Some(LightRooms(vis))) => {
                vis.instance == r.instance && vis.groups.len() == 1 && vis.groups[0] == r.group
            }
            (None, None) => true,
            _ => false,
        };
        if same {
            continue;
        }
        match want {
            Some(room) => {
                commands
                    .entity(light)
                    .insert(LightRooms(crate::wmo_portal::WmoGroupVis {
                        instance: room.instance,
                        groups: std::sync::Arc::from([room.group]),
                    }));
            }
            // Out in the open world again: the rooms come OFF, which is what puts the light back
            // under the room-less law rather than leaving it claiming a building it has left.
            None => {
                commands.entity(light).remove::<LightRooms>();
            }
        }
    }
}

/// An authored point light (an M2 light, a WMO MOLT omni, a carried torch) as the packed table
/// reads it. Not a Bevy `PointLight`: nothing here reads clustered lights, and Bevy would run its
/// light passes over every one each frame.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct WorldPointLight {
    /// Linear RGB, hue preserved.
    pub color: [f32; 3],
    /// `4π × authored intensity`, the `PointLight` convention, which the packer's `/(4π)` undoes.
    pub intensity: f32,
    /// The ≤3-nearest selection radius (yd).
    pub range: f32,
}

/// The packed light for this frame, extracted for [`upload_light`].
#[derive(Resource, Clone, Copy, ExtractResource)]
struct WowLightData(LightStd430);

impl Default for WowLightData {
    fn default() -> Self {
        Self(LightStd430 {
            rows: [[0.0; 4]; 21],
            points: [[0.0; 4]; 2 * MAX_POINT_LIGHTS],
        })
    }
}

/// The persistent storage buffer every material binds, created by [`new_shared_light_buffer`] and
/// extracted to the render world; a `Buffer` clone shares the GPU resource.
#[derive(Resource, Clone, ExtractResource)]
pub struct SharedLightBuffer(pub Buffer);

/// Registers the light pack, the probe publish, their extracts and the render-world uploads.
pub(super) fn register(app: &mut App) {
    app.init_resource::<WowLightData>()
        .init_resource::<super::prop_probes::PropProbeExtract>()
        .add_plugins(ExtractResourcePlugin::<WowLightData>::default())
        .add_plugins(ExtractResourcePlugin::<SharedLightBuffer>::default())
        .add_plugins(ExtractResourcePlugin::<super::prop_probes::PropProbeExtract>::default())
        // After transform propagation: a carried light (a torch in a hand) is a child of a moving
        // joint, so its `GlobalTransform` is this frame's only once `Propagate` has run.
        .add_systems(
            PostUpdate,
            build_light_data
                .after(bevy::transform::TransformSystems::Propagate)
                .after(super::update_time_lighting)
                // …and after the rooms it reads are current for this frame. `build_light_data`
                // packs `indoors` straight off `LightRooms`, so a claim applied after it would
                // spend a frame as "outdoors" — which for an entity light means a frame refused by
                // every interior surface, i.e. a visible flicker at every doorway.
                .after(track_entity_light_rooms),
        )
        .add_systems(
            PostUpdate,
            track_entity_light_rooms.after(bevy::transform::TransformSystems::Propagate),
        )
        // After the spawners (PostUpdate): publish the probe table for extraction on change.
        .add_systems(PostUpdate, super::prop_probes::publish_prop_probes);
    // A headless build (no GPU, `backends: None`), as the schedule tests use, has no render app.
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(
            Render,
            (upload_light, super::prop_probes::upload_prop_probes)
                .in_set(RenderSystems::PrepareResources),
        );
    }
}

/// Creates the shared light buffer, sized by [`light_blob_bytes`], from the main-world
/// `RenderDevice`; the assets foundation builds it at startup.
pub fn new_shared_light_buffer(device: &RenderDevice) -> SharedLightBuffer {
    SharedLightBuffer(device.create_buffer(&BufferDescriptor {
        label: Some("wow_shared_light"),
        size: light_blob_bytes(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

/// The shared light buffer's full size: the per-frame blob ([`LIGHT_HEADER_ROWS`] rows and the
/// point table), the interior-prop probe region and the skin-palette regions. Every buffer bound
/// as `wow_light` must be this big: wgpu validates the bound size against `wow_model.wgsl`'s
/// whole layout at each draw.
pub fn light_blob_bytes() -> u64 {
    per_frame_blob_bytes()
        + (7 * MAX_PROP_PROBES * 16) as u64
        + crate::rig_palette::palette_regions_bytes()
}

/// The per-frame prefix's size, which is also the probe region's offset.
pub(super) fn per_frame_blob_bytes() -> u64 {
    std::mem::size_of::<LightStd430>() as u64
}

/// Packs the resolved [`WowLighting`], the fog toggle, the farclip and the nearest point lights
/// into the std430 blob.
#[allow(clippy::type_complexity)]
fn build_light_data(
    light: Res<WowLighting>,
    debug: Res<DebugState>,
    view: Res<ViewDistance>,
    cam: Query<&GlobalTransform, With<WorldCamera>>,
    lights_q: Query<(
        Entity,
        &WorldPointLight,
        &GlobalTransform,
        Option<&LightRooms>,
        Option<&crate::modern::lights::SynthesizedLight>,
        Option<&NightGlow>,
        // fork: the flame flicker ([`super::FlameFlicker`]) — folded into the committed colour
        // here and nowhere else, so no other reader of the light's intensity sees it.
        Option<&super::FlameFlicker>,
        Has<SpellFxLight>,
    )>,
    // fork: P8 — each light's admission fade, keyed by entity. See the gather below.
    mut admit: Local<bevy::platform::collections::HashMap<Entity, f32>>,
    // fork: P8 — the modern pack radius in force this frame (0 = not yet set). See the gather.
    mut pack_radius: Local<f32>,
    mut wanting: Local<Vec<f32>>,
    // The per-frame portal PVS, for the rooms term.
    portals: Query<&crate::wmo_portal::WmoPortalInstance>,
    // …and the buildings themselves, for the `w` channel's INDOOR test — a light's rooms are group
    // indices, and whether those groups are interiors is a property of the model
    // ([`crate::wmo_portal::rooms_are_indoor`]).
    // `Option`, because a lane can run before the asset collection exists (and the goldens below
    // build a bare `App`). Absent ⇒ no light is classified indoors, which is the safe direction:
    // wrongly "outdoor" leaves a light doing exactly what it did before this test existed, wrongly
    // "indoor" switches a working lamp off (the 1276 lopsided-error rule again).
    wmos: Option<Res<bevy::asset::Assets<benilla_assets::WmoModel>>>,
    mut data: ResMut<WowLightData>,
    time: Res<Time>,
    // fork: modern lighting — row 20 `.y` tells every world shader the frame is linear.
    active: Option<Res<crate::modern::ActiveLighting>>,
    mut last_dump: Local<f64>,
    mut last_rows_dump: Local<f64>,
) {
    let l = &*light;
    let fog_enable = if debug.lighting.disable_fog { 0.0 } else { 1.0 };
    let farclip = view.farclip;
    // River and lake share the non-ocean swatch.
    let (rs, rd, rsa, rda) = l.water_colors(LiquidKind::Still);
    let (os, od, osa, oda) = l.water_colors(LiquidKind::Ocean);
    // Built in a scratch copy and written through `ResMut` only when a row moved: the extract
    // clones this 8.5 KB blob every frame it reads as changed.
    let mut fresh = data.0;
    fresh.rows = [[0.0; 4]; LIGHT_HEADER_ROWS];
    // fork: modern lighting (FORK.md) — the zone look's fog tint (1 in classic).
    let (fog_tint, silhouette) = if active.as_ref().is_some_and(|a| a.is_modern()) {
        (
            crate::modern::look::fog_tint(),
            crate::modern::look::silhouette(),
        )
    } else {
        ([1.0; 3], 0.0)
    };
    let rows = &mut fresh.rows;
    rows[3] = [l.spec[0], l.spec[1], l.spec[2], 20.0]; // 3 light_spec (w=terrain shininess 20)
    rows[4] = [
        l.fog_color[0] * fog_tint[0],
        l.fog_color[1] * fog_tint[1],
        l.fog_color[2] * fog_tint[2],
        fog_enable,
    ]; // 4 fog_color (w=enable)
    rows[5] = [l.fog_start, l.fog_end, silhouette, farclip]; // 5 fog_params (z = modern WDL silhouette; w=farclip)
    rows[13] = [rs[0], rs[1], rs[2], rsa]; // 13 water river shallow (w=alpha)
    rows[14] = [rd[0], rd[1], rd[2], rda]; // 14 water river deep
    rows[15] = [os[0], os[1], os[2], osa]; // 15 water ocean shallow
    rows[16] = [od[0], od[1], od[2], oda]; // 16 water ocean deep
    rows[17][0] = l.sidn_night;
    // Row 17 `.x` is the SIDN night fraction and `.yzw` the core packer's; 18/19 are the interior
    // fog triple, and 19.zw and 12.w are free.
    rows[18] = [
        l.wmo_fog_color[0],
        l.wmo_fog_color[1],
        l.wmo_fog_color[2],
        fog_enable,
    ];
    rows[19] = [l.wmo_fog_start, l.wmo_fog_end, 0.0, 0.0];
    // Rows 0-2, 6-12.xyz and 17.yzw: the model-light core.
    // fork: modern lighting (FORK.md, P4) — the zone look's key and fill gains (A3).
    let (key, fill, key_tint) = if active.as_ref().is_some_and(|a| a.is_modern()) {
        let (key, fill) = crate::modern::look::key_fill();
        (key, fill, crate::modern::look::key_tint())
    } else {
        (1.0, 1.0, [1.0; 3])
    };
    // fork: modern lighting — a modern night keeps a moonlight floor (`modern::night_floor`): the
    // authored night row plus a key that fades with the moon's elevation left a moonless midnight
    // black.
    let ambient_row = if active.as_ref().is_some_and(|a| a.is_modern()) {
        crate::modern::night_floor(l.ambient, l.sidn_night)
    } else {
        l.ambient
    };
    let diffuse_row = [0, 1, 2].map(|i| l.diffuse[i] * key * key_tint[i]);
    pack_model_core_rows(rows, ambient_row.map(|c| c * fill), diffuse_row, l.sun_dir);
    // The point table: lights within [`POINT_PACK_RADIUS`], nearest first past capacity. Dividing
    // by 4π undoes the spawn's premultiply, so the colour is the authored `colour × intensity`,
    // committed raw. Entries past the count stay stale; the count row guards every reader.
    let cam_pos = cam.single().map(|t| t.translation()).unwrap_or(Vec3::ZERO);
    // fork: P8 — **every light fades in and out of the table; none of them ever switches.**
    //
    // Admission is a per-frame verdict made of hard edges: a room gate that flips as the player
    // crosses a doorway, a pack radius, a tier switch. Committing that verdict directly meant a
    // lamp appearing or vanishing between one frame and the next, which is the flicker reported
    // walking around Northshire Abbey and in and out of its door. Nothing about a lamp changes
    // when the player steps over a threshold, so nothing about its light may either.
    //
    // So the verdict drives a ramp instead, and the ramp scales the colour this light COMMITS. A
    // light on its way out is still in the table, dimming; at zero it leaves. Every edge above is
    // smoothed by this one mechanism, including ones not yet thought of.
    let dt = time.delta_secs().clamp(0.0, 0.1);
    let step = (ADMIT_FADE_RATE * dt).clamp(0.0, 1.0);
    // fork: P8 — **a modern frame fades lights out with distance; it never cuts them off.** The
    // classic table keeps the reference's shape: nearest-first within `POINT_PACK_RADIUS`, which
    // is all a unit's three-nearest selection ever sees. A modern frame reads the table through a
    // grid that covers the view, so the table's edge is on screen — and past its capacity, the
    // lights it dropped were whichever came 257th this frame, blinking in and out as the camera
    // moved. So in a modern frame the radius is a soft edge ([`MODERN_PACK_FADE`] wide), and when
    // more lights want in than the table holds, the radius itself closes in (smoothly) until they
    // fit: the far lights fade out together instead of one at a time.
    let modern_pack = active.as_ref().is_some_and(|a| a.is_modern());
    if *pack_radius <= 0.0 {
        *pack_radius = MODERN_PACK_RADIUS;
    }
    let radius = if modern_pack {
        *pack_radius
    } else {
        POINT_PACK_RADIUS
    };
    let night = l.sidn_night.clamp(0.0, 1.0);
    // fork: the flicker's clock and gain — absolute time, so the waveform is frame-rate independent.
    let flicker_t = time.elapsed_secs_wrapped();
    let flicker_gain = super::flicker::flicker_gain();
    // fork: spell lights ride only a frame whose lights are binned into the modern grid.
    let spell_tier = active.as_ref().is_some_and(|a| {
        a.is_modern()
            && crate::modern::lights::light_tier(a.quality).on
            && crate::modern::lights::lights_enabled()
    });
    wanting.clear();
    let mut pts: Vec<(f32, Vec3, f32, [f32; 3], bool)> = lights_q
        .iter()
        .filter_map(
            |(entity, pl, gt, rooms, synthesized, night_glow, flame, spell)| {
                let p = gt.translation();
                let d2 = p.distance_squared(cam_pos);
                // The tier gate for DERIVED lights, read where the light is READ rather than where it
                // was spawned: a lamppost's light entity always exists, and whether it reaches the
                // table is this frame's question — so a doodad that streamed in before the lighting
                // latched is not left permanently dark.
                let tier_ok = (synthesized.is_none() || crate::modern::emissive_lights_on())
                    && (!spell || spell_tier);
                // A culled room's light registers nothing in the reference ([`LightRooms`]); a
                // light that names no rooms always passes.
                let room_ok = crate::wmo_portal::room_admits_light(
                    rooms.map(|r| &r.0),
                    rooms.and_then(|r| portals.get(r.0.instance).ok()),
                );
                // A window's glow is a light only at night; by day it is not in the table at all
                // (rather than in it at zero, holding a slot a lamp could use).
                let lit = night_glow.is_none() || night > 0.0;
                let wants = tier_ok && room_ok && lit;
                if wants && modern_pack && d2 < MODERN_PACK_RADIUS * MODERN_PACK_RADIUS {
                    wanting.push(d2);
                }
                let admitted = wants && d2 < radius * radius;
                let fade = admit.entry(entity).or_insert(0.0);
                *fade = if admitted {
                    (*fade + step).min(1.0)
                } else {
                    (*fade - step).max(0.0)
                };
                if *fade <= 0.0 {
                    return None;
                }
                let c = pl.color;
                // The modern soft edge: full strength inside `radius - MODERN_PACK_FADE`, nothing at
                // `radius`, smoothstepped between — so reaching the edge is a fade over sixty yards of
                // walking, not a switch.
                let reach = if modern_pack {
                    let t = ((radius - d2.sqrt()) / MODERN_PACK_FADE).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                } else {
                    1.0
                };
                let glow = if night_glow.is_some() { night } else { 1.0 };
                let gain = if modern_pack { MODERN_LOCAL_GAIN } else { 1.0 };
                let flick = flame.map_or(super::flicker::FlickerMod::STEADY, |f| {
                    f.at(flicker_t, flicker_gain)
                });
                let s = pl.intensity / (4.0 * std::f32::consts::PI)
                    * *fade
                    * reach
                    * glow
                    * gain
                    * flick.intensity;
                let rgb = commit_raw([c[0] * s, c[1] * s, c[2] * s * flick.blue]);
                // The `w` channel (see the pack below): INDOORS, not merely "names rooms".
                let indoors = rooms.is_some_and(|r| {
                    portals
                        .get(r.0.instance)
                        .ok()
                        .and_then(|inst| wmos.as_ref()?.get(&inst.handle))
                        .is_some_and(|model| crate::wmo_portal::rooms_are_indoor(&r.0, model))
                });
                Some((d2, p, pl.range, rgb, indoors))
            },
        )
        .collect();
    // A despawned light must not keep its ramp: the next entity to reuse that id would inherit it.
    admit.retain(|e, _| lights_q.contains(*e));
    // fork: P8 — steer the modern radius for next frame: close in until the lights inside it fit
    // the table, open back out when they do.
    if modern_pack {
        let target = if wanting.len() > MAX_POINT_LIGHTS {
            let (_, nth, _) = wanting.select_nth_unstable_by(MAX_POINT_LIGHTS - 1, f32::total_cmp);
            nth.sqrt()
        } else {
            MODERN_PACK_RADIUS
        };
        *pack_radius = if target < *pack_radius {
            (*pack_radius - PACK_SHRINK_RATE * dt).max(target)
        } else {
            (*pack_radius + PACK_GROW_RATE * dt).min(target)
        }
        .max(MODERN_PACK_FADE);
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    // fork: P8 — how many lights actually wanted in. Deriving a light from every glow card made
    // this a real number rather than a formality: past the cap the table keeps the nearest and
    // DROPS the rest, and the set it drops changes as the camera moves — which is a light blinking
    // on and off for no reason the player can see.
    let in_range = pts.len();
    pts.truncate(MAX_POINT_LIGHTS);
    if in_range > MAX_POINT_LIGHTS {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            bevy::log::warn!(
                "modern lights: {in_range} lights within {POINT_PACK_RADIUS} yd but the table holds                  {MAX_POINT_LIGHTS} — the {} past the cap drop in and out as the camera moves",
                in_range - MAX_POINT_LIGHTS,
            );
        }
    }
    let modern = if active.is_some_and(|a| a.is_modern()) {
        1.0
    } else {
        0.0
    };
    fresh.rows[20] = [pts.len() as f32, modern, 0.0, 0.0];
    for (i, (_, p, range, rgb, indoors)) in pts.iter().enumerate() {
        fresh.points[2 * i] = [p.x, p.y, p.z, *range];
        // fork: modern lighting (P8) — **`w` is 1 for a light that is INDOORS**, and the surface
        // lanes' class test runs both ways off it: an interior fragment takes only indoor lights,
        // an exterior fragment takes only outdoor ones.
        //
        // A local light has no occluder unless it holds one of the twelve cube slots, so nothing
        // else stops one crossing a wall. Both directions of that were reported on the same
        // building: a street lamp lighting the inside of the chapel beside it, and then Northshire
        // Abbey's own candles lighting the road outside and costing the frame to do it. They are
        // one bug seen from two sides, and the side that matters is the RECEIVER's — which is why
        // the answer lives here and in the lanes, and NOT in the portal gate above. A per-frame
        // admitted set cannot tell the two apart: anything narrow enough to keep the abbey's
        // candles off the road also darkens them when the player looks in through the door, which
        // is exactly what a hop-capped `lighting_reach` did before it was backed out.
        //
        // `w` is derived from the naming groups' own `0x8`/`0x40` flags
        // ([`crate::wmo_portal::rooms_are_indoor`]), NOT from "does it name rooms". The two differ
        // on every porch lamp and courtyard brazier in the game: those name a building's shell, so
        // the old reading called them indoor and a two-way test would have switched them off.
        fresh.points[2 * i + 1] = [rgb[0], rgb[1], rgb[2], f32::from(u8::from(*indoors))];
    }
    // `WOW_POINTS_DUMP=1` prints the nearest 8 packed lights once a second; `=frame` every frame,
    // which a pool that changes frame to frame needs.
    // Read through `launch_env`, not `std::env`: on wasm the process environment is empty, so the
    // browser could never reach this probe — which is exactly where the light pool needed measuring.
    // `benilla_app::webenv` seeds the page's `WOW_*` query params into it before the app is built.
    static POINTS_DUMP: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    if let Some(mode) = POINTS_DUMP.get_or_init(|| crate::launch_env::var("WOW_POINTS_DUMP")) {
        let every = if mode == "frame" { 0.0 } else { 1.0 };
        let now = time.elapsed_secs_f64();
        if now - *last_dump >= every {
            *last_dump = now;
            // Candidates for the camera chunk's three slots: the Chebyshev box of `terrain.wgsl`'s
            // `TERRAIN_REACH` (keep in sync), with the 48 yd sphere's count beside it.
            let cell = 533.333_3 / 16.0;
            let half = 32.0 * 533.333_3;
            let snap = |v: f32| (((half + v) / cell).floor() + 0.5) * cell - half;
            let anchor = Vec3::new(snap(cam_pos.x), cam_pos.y, snap(cam_pos.z));
            let (mut boxed, mut sphere) = (0usize, 0usize);
            for (_, p, _, _, _) in &pts {
                let dv = *p - anchor;
                boxed += usize::from(dv.x.abs().max(dv.z.abs()) <= 33.570_166);
                sphere += usize::from(dv.length() <= 48.0);
            }
            // `bevy::log::info!`, not `eprintln!` — and that is the whole difference between a
            // probe and a probe you can USE. On wasm `eprintln!` goes to a stderr nothing is
            // reading, so the browser (the one build whose GPU sets the project's budget, and the
            // only place the light count was ever in question) printed nothing at all and read
            // exactly like the dump not running. The emission dump was made reachable the same way
            // in `d7f4b594`. One record, built and logged once, rather than nine lines the console
            // interleaves with everything else.
            let mut out = format!(
                "[points] {} packed, cam {cam_pos:.1?} — this chunk's candidates: {boxed} (was {sphere} at the 48 yd sphere), 3 slots",
                pts.len()
            );
            for (d2, p, _, rgb, _) in pts.iter().take(8) {
                out.push_str(&format!(
                    "\n  d {:6.2}  at [{:8.2},{:7.2},{:8.2}]  rgb [{:.3},{:.3},{:.3}]",
                    d2.sqrt(),
                    p.x,
                    p.y,
                    p.z,
                    rgb[0],
                    rgb[1],
                    rgb[2]
                ));
            }
            bevy::log::info!("{out}");
        }
    }
    // `WOW_LIGHT_DUMP=frame` (or `=1` for once a second) prints every packed header row as raw f32
    // bits, so a change below a printed decimal still shows.
    static LIGHT_DUMP: std::sync::OnceLock<Option<std::ffi::OsString>> = std::sync::OnceLock::new();
    if let Some(mode) = LIGHT_DUMP.get_or_init(|| std::env::var_os("WOW_LIGHT_DUMP")) {
        let every = if mode.as_os_str() == "frame" {
            0.0
        } else {
            1.0
        };
        let now = time.elapsed_secs_f64();
        if now - *last_rows_dump >= every {
            *last_rows_dump = now;
            let hash = data
                .0
                .rows
                .iter()
                .flatten()
                .fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
                    (h ^ u64::from(v.to_bits())).wrapping_mul(0x1000_0000_01b3)
                });
            eprintln!("[light] rows {hash:#018x}");
            for (i, r) in fresh.rows.iter().enumerate() {
                eprintln!(
                    "  {i:2} {:08x} {:08x} {:08x} {:08x}   {:9.5} {:9.5} {:9.5} {:9.5}",
                    r[0].to_bits(),
                    r[1].to_bits(),
                    r[2].to_bits(),
                    r[3].to_bits(),
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                );
            }
        }
    }
    if data.0 != fresh {
        data.0 = fresh;
    }
}

/// Render world, in `PrepareResources`: writes the packed light into the shared buffer before any
/// draw reads it.
fn upload_light(
    queue: Res<RenderQueue>,
    buffer: Option<Res<SharedLightBuffer>>,
    data: Option<Res<WowLightData>>,
) {
    let (Some(buffer), Some(data)) = (buffer, data) else {
        return;
    };
    queue.write_buffer(&buffer.0, 0, bytemuck::bytes_of(&data.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GOLDEN — **an entity light inherits the room its body stands in**, and loses it again when
    /// the body walks out.
    ///
    /// The gate this feeds is `static_gx.wgsl`'s `interior && w < 0.5 -> skip`, and `w` is just
    /// "does this light name rooms". That rule reads "names no rooms" as "is outdoors", which is
    /// true of a placed ADT lamp and false of every light on the entity lane — and nothing in
    /// `benilla-app` attached [`LightRooms`], so a GameObject brazier standing inside Orgrimmar
    /// was refused by the floor it stood on, by its own stand, and by every wall around it. A
    /// "Mighty Blaze" lighting nothing at all is what this test is written from.
    ///
    /// The walk up matters as much as the claim: a weapon enchant's light is owned by the ITEM
    /// root, which has no claim of its own — only the wielder underneath it does.
    #[test]
    fn an_entity_light_takes_the_room_its_body_is_standing_in() {
        use crate::wmo_portal::{UnitWmoRoom, WmoRoom};

        let mut app = App::new();
        app.add_systems(Update, track_entity_light_rooms);

        let instance = app.world_mut().spawn_empty().id();
        // A body standing in group 7 of that building, and its light hanging off a bone of it.
        let body = app
            .world_mut()
            .spawn(UnitWmoRoom::claimed(WmoRoom { instance, group: 7 }))
            .id();
        let bone = app.world_mut().spawn(ChildOf(body)).id();
        let light = app
            .world_mut()
            .spawn((
                WorldPointLight {
                    color: [1.0; 3],
                    intensity: 1.0,
                    range: 10.0,
                },
                LightOwner(bone),
                ChildOf(bone),
            ))
            .id();
        app.update();

        let rooms = app.world().get::<LightRooms>(light);
        let vis = rooms.map(|LightRooms(v)| (v.instance, v.groups.to_vec()));
        assert_eq!(
            vis,
            Some((instance, vec![7u16])),
            "the light must name the one room its body is in — without it the packer writes w = 0 \
             and every interior surface refuses it",
        );

        // …and the body walks out into the open world.
        *app.world_mut().get_mut::<UnitWmoRoom>(body).unwrap() = UnitWmoRoom::default();
        app.update();
        assert!(
            app.world().get::<LightRooms>(light).is_none(),
            "outdoors again, the rooms come OFF — a light left claiming a building it has left is \
             one the portal gate can cull while it stands in the street",
        );
    }

    /// `wow_model.wgsl`'s exterior doodad and entity lane over the packed rows gives
    /// `E = A + I·D·(4/17)(0.375 + 2μ + 1.875μ²)`: ambient never scales by the intensity I, and
    /// every sun band scales by it once. `eval_sh_lane` mirrors the WGSL line for line, so a swap
    /// in the shader shows only when the two are read side by side.
    #[test]
    fn the_sh_response_lane_matches_the_closed_form_at_every_intensity() {
        // A committed Stormwind pair at minute ≈1185, from the reference's uploaded constants.
        let ambient = [102.0 / 255.0, 97.0 / 255.0, 123.0 / 255.0];
        let diffuse = [255.0 / 255.0, 112.0 / 255.0, 0.0];
        let sun_dir = Vec3::new(0.31, -0.82, 0.48).normalize(); // travel dir; to-light = −this
        let mut rows = [[0.0f32; 4]; LIGHT_HEADER_ROWS];
        pack_model_core_rows(&mut rows, ambient, diffuse, sun_dir);

        /// The SH branch of `wow_model.wgsl`'s exterior lane, verbatim.
        fn eval_sh_lane(rows: &[[f32; 4]; LIGHT_HEADER_ROWS], n: Vec3, intensity: f32) -> [f32; 3] {
            let quad = [n.x * n.y, n.y * n.z, n.z * n.z, n.x * n.z];
            let x2y2 = n.x * n.x - n.y * n.y;
            let dot3 = |r: [f32; 4]| r[0] * n.x + r[1] * n.y + r[2] * n.z;
            let dot4 =
                |r: [f32; 4]| r[0] * quad[0] + r[1] * quad[1] + r[2] * quad[2] + r[3] * quad[3];
            [0usize, 1, 2].map(|ch| {
                // sh_c10_{r,g,b} = rows[6+ch] (.w = ambient) · sh_c13_{r,g,b} = rows[9+ch]
                // sh_c16.xyz = rows[12][ch] · grade.yzw = rows[17][1+ch] (the sun's DC, at I=1)
                rows[6 + ch][3]
                    + rows[17][1 + ch] * intensity
                    + intensity * (dot3(rows[6 + ch]) + dot4(rows[9 + ch]) + rows[12][ch] * x2y2)
            })
        }

        let u = -sun_dir; // toward-light unit
        let f = |mu: f32| (4.0 / 17.0) * (0.375 + 2.0 * mu + 1.875 * mu * mu);
        // A side-on normal (μ = 0) and a mid-back one (μ ≈ −0.53, the lobe's negative dip).
        let side = u.cross(Vec3::Y).normalize();
        let mid_back = (u * -0.5333 + side * (1.0f32 - 0.5333 * 0.5333).sqrt()).normalize();
        for (label, n) in [
            ("facing", u),
            ("away", -u),
            ("side-on", side),
            ("mid-back", mid_back),
        ] {
            for intensity in [0.5f32, 1.0, 2.5] {
                let mu = n.dot(u);
                let got = eval_sh_lane(&rows, n, intensity);
                for ch in 0..3 {
                    let want = ambient[ch] + intensity * diffuse[ch] * f(mu);
                    assert!(
                        (got[ch] - want).abs() < 1e-5,
                        "{label} I={intensity} ch{ch}: got {} want {want}",
                        got[ch]
                    );
                }
            }
        }
        // At μ = 1 the lobe equals the FFP peak `A + D`, by the 16/17 accumulate scale.
        let peak = eval_sh_lane(&rows, u, 1.0);
        for ch in 0..3 {
            let ffp_peak = ambient[ch] + diffuse[ch]; // ambient + D·max(N·L,0) at N·L = 1
            assert!(
                (peak[ch] - ffp_peak).abs() < 1e-5,
                "peak ch{ch}: SH {} vs FFP {}",
                peak[ch],
                ffp_peak
            );
        }
        // The mid-back dip goes below ambient, the reference's SH ringing; clamping per term
        // instead of the sum would erase it.
        let dip = eval_sh_lane(&rows, mid_back, 1.0);
        assert!(
            dip[0] < ambient[0],
            "mid-back should dip below ambient: {} vs {}",
            dip[0],
            ambient[0]
        );
    }

    /// A held torch, authored `(0.467, 0.290, 0.133) × 3.0`, commits its raw product through the
    /// real packer, red 40% past white (`0x71ca80` encodes, `0x593040` decodes it back).
    #[test]
    fn the_torch_commits_the_raw_authored_product() {
        let mut app = App::new();
        app.init_resource::<WowLighting>()
            .init_resource::<crate::dev_state::DebugState>()
            .init_resource::<crate::view::ViewDistance>()
            .init_resource::<WowLightData>()
            .init_resource::<Time>()
            .add_systems(Update, build_light_data);
        app.world_mut()
            .spawn((crate::view::WorldCamera, GlobalTransform::IDENTITY));
        // The authored torch light, through the spawn recipe.
        app.world_mut().spawn((
            crate::terrain_stream::point_light([0.466_666_7, 0.290_196_1, 0.133_333_34], 3.0),
            GlobalTransform::from_translation(Vec3::new(0.0, 1.5, 0.0)),
        ));
        // Past the admission ramp before reading: every light now fades INTO the table rather
        // than switching on (`ADMIT_FADE_RATE`), and the ramp scales the colour it commits — so
        // a torch sampled mid-fade is a dimmed torch, and this golden is about the commit
        // expression, not the ramp. One frame of the ramp's own dt clamp (0.1 s ⇒ step 1.5,
        // saturated) puts it at full strength, which is where the authored product is legible.
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_millis(100));
        app.update();

        let rows = &app.world().resource::<WowLightData>().0;
        assert_eq!(rows.rows[20][0], 1.0, "the light packed");
        let rgb = rows.points[1];
        // Over-white is kept: saturation is the receiving vertex's clamp, not the commit's.
        assert!(
            (rgb[0] - 1.400_000_1).abs() < 1e-4,
            "red commits raw past white: {rgb:?}"
        );
        assert!(
            (rgb[1] - 0.870_588_3).abs() < 1e-4,
            "green commits raw: {rgb:?}"
        );
        assert!((rgb[2] - 0.4).abs() < 1e-4, "blue commits raw: {rgb:?}");
    }

    /// The packed SH block gives `Model2.bls`'s `clamp01(A + D·I·(3 + 16μ + 15μ²)/34)` at each
    /// intensity rung (2.5 lit, 1.0 mid-band, 0.5 MCSH-shadowed), with ambient in the DC lanes and
    /// the sun's DC on `grade.yzw`.
    #[test]
    fn exterior_lane_reproduces_the_closed_form_at_every_intensity_rung() {
        let ambient = [0.30, 0.32, 0.38];
        let diffuse = [0.85, 0.70, 0.45];
        let sun_dir = Vec3::new(0.3, -0.8, 0.52).normalize(); // travel dir; to-light = −sun_dir
        let mut rows = [[0.0f32; 4]; LIGHT_HEADER_ROWS];
        pack_model_core_rows(&mut rows, ambient, diffuse, sun_dir);
        // The shader's exterior eval over the packed rows, per channel, at intensity `i`.
        let eval = |n: Vec3, i: f32| -> [f32; 3] {
            let quad = [n.x * n.y, n.y * n.z, n.z * n.z, n.x * n.z];
            [0usize, 1, 2].map(|ch| {
                let c10 = rows[6 + ch];
                let c13 = rows[9 + ch];
                let lin = c10[0] * n.x + c10[1] * n.y + c10[2] * n.z;
                let q: f32 = (0..4).map(|k| c13[k] * quad[k]).sum::<f32>()
                    + rows[12][ch] * (n.x * n.x - n.y * n.y);
                (c10[3] + rows[17][1 + ch] * i + i * (lin + q)).clamp(0.0, 1.0)
            })
        };
        let u = -sun_dir; // toward-light
        let side = u.cross(Vec3::Y).normalize();
        for i in [2.5f32, 1.0, 0.5] {
            for (n, mu, who) in [
                (u, 1.0f32, "facing"),
                (-u, -1.0, "away"),
                (side, 0.0, "side"),
            ] {
                let b = (3.0 + 16.0 * mu + 15.0 * mu * mu) / 34.0;
                let got = eval(n, i);
                for ch in 0..3 {
                    let want = (ambient[ch] + diffuse[ch] * i * b).clamp(0.0, 1.0);
                    assert!(
                        (got[ch] - want).abs() < 1e-5,
                        "I={i} {who}: ch{ch} got {} want {want}",
                        got[ch]
                    );
                }
            }
        }
        // Over the back hemisphere the sun term never dips below the closed form's minimum,
        // −0.0373·C·I at μ ≈ −0.53.
        let zero_amb = {
            let mut r = [[0.0f32; 4]; LIGHT_HEADER_ROWS];
            pack_model_core_rows(&mut r, [0.0; 3], diffuse, sun_dir);
            r
        };
        let eval0 = |n: Vec3, i: f32| -> f32 {
            let quad = [n.x * n.y, n.y * n.z, n.z * n.z, n.x * n.z];
            let c10 = zero_amb[6];
            let c13 = zero_amb[9];
            let lin = c10[0] * n.x + c10[1] * n.y + c10[2] * n.z;
            let q: f32 = (0..4).map(|k| c13[k] * quad[k]).sum::<f32>()
                + zero_amb[12][0] * (n.x * n.x - n.y * n.y);
            zero_amb[6][3] + zero_amb[17][1] * i + i * (lin + q)
        };
        for k in 0..=20 {
            let mu = -1.0 + k as f32 / 20.0;
            let n = (u * mu + side * (1.0 - mu * mu).sqrt()).normalize();
            let floor = -0.0374 * diffuse[0] * 2.5;
            assert!(
                eval0(n, 2.5) >= floor,
                "μ={mu}: ringing {} below the closed-form floor {floor}",
                eval0(n, 2.5)
            );
        }
    }
}
