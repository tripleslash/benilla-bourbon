//! What the player is pointing at: the identity of every pickable world thing ([`WorldObject`]),
//! the world click gestures, and the ray caster ([`pick`]), which casts against resident geometry
//! so colliderless props that a physics ray misses still pick.
//!
//! The consolidated render lanes draw many placements with no entity per placement, so a pick
//! declaration takes three shapes: [`PickMesh`] or [`PickBox`] on an entity, [`PickBlob`] members
//! on a blob entity, and a [`PickSource`] lane that answers the ray itself.

use bevy::platform::collections::HashSet;
use bevy::prelude::*;

use crate::model_render::ModelKind;

mod pick;
mod posed_once;

pub use posed_once::PosedPickScratch;

pub use pick::{
    cast_nearest_accepted, cast_object_ray, cast_pick_ray, cast_pick_ray_inflated, pick_at_cursor,
    pick_object_at_cursor, ObjectHit, PickAccept, PickBlob, PickBox, PickMember, PickMesh,
    PickParts, PickSource,
};

pub use pick::{ray_member, ray_mesh_bounds, ray_posed_mesh, RayHit};

/// The identity of a pickable world thing, attached to its renderable mesh entities at spawn.
#[derive(Component, Clone, Debug)]
pub struct WorldObject {
    pub kind: ModelKind,
    /// A model path (doodads, WMOs, GameObjects) or a unit name.
    pub label: String,
    /// Placement `uniqueId`, server guid or display id; `0` if none.
    pub id: u32,
    /// An optional kind-specific second line, shown when non-empty.
    pub detail: String,
}

/// Marks every pickable GameObject part, so the GameObject hover's pick set is an archetype query
/// rather than a scan of every `WorldObject`.
#[derive(Component, Clone, Copy)]
pub struct GoPickPart;

/// Marks every pickable unit or player part, the model-less fallback cube included: the unit
/// pick's skinless-fallback set. Doodad and WMO parts carry neither marker.
#[derive(Component, Clone, Copy)]
pub struct CreaturePickPart;

/// A left select gesture, sent on release when the press met the reference's click predicate:
/// under 200 ms, or under 800 ms having turned the camera less than 2.25° of yaw and 2.0° of
/// pitch. A drag that meets it selects too, as the reference arms the click beside the camera look.
/// It carries no position: the pick is latched at the press (`target::PressPick`).
#[derive(Message, Clone, Copy)]
pub struct WorldClick;

/// The right button's context gesture, on the same predicate as [`WorldClick`].
#[derive(Message, Clone, Copy)]
pub struct WorldRightClick;

/// The right button's press in the world (off the UI, or any press while a look holds the hidden
/// cursor), sent before the click test starts, as the reference's `CGWorldFrame::OnMouseDown`
/// (`0x483c40`), where ground targeting's right-click cancel and the repair-mode reset hang
/// (`0x492c20`). It consumes nothing, so the turn and the release's context click still run.
#[derive(Message, Clone, Copy)]
pub struct WorldRightPress;

/// An entity that carries pick geometry, on itself or as a blob's members.
type CarriesPickGeometry = Or<(With<PickMesh>, With<PickBlob>)>;

/// The world's pick sources as one `SystemParam`: the entity pick geometry ([`PickParts`]) and
/// every lane that draws without entities ([`PickSource`]). Use this, not [`PickParts`] alone,
/// which still returns hits but misses most of the static world.
#[derive(bevy::ecs::system::SystemParam)]
pub struct WorldPick<'w, 's> {
    parts: PickParts<'w, 's>,
    /// The identity every entity hit resolves through.
    objects: Query<'w, 's, &'static WorldObject>,
    /// Every entity that carries pick geometry, for the casts that walk the whole drawn world.
    geometry: Query<'w, 's, Entity, CarriesPickGeometry>,
    /// The retained static pass; `None` when `WOW_STATIC_GX=0` keeps every placement on the entity
    /// path. The static merge's blobs are entities and answer through [`PickBlob`].
    gx: Option<Res<'w, crate::static_gx::StaticGx>>,
}

impl WorldPick<'_, '_> {
    /// This frame's entity-less sources, in the shape [`cast_object_ray`] takes.
    fn sources(&self) -> Vec<&dyn PickSource> {
        self.gx
            .as_deref()
            .map(|gx| gx as &dyn PickSource)
            .into_iter()
            .collect()
    }

    /// The identified cast against everything drawn ([`cast_object_ray`]).
    pub fn cast(&self, ray: Ray3d, pickable: &HashSet<Entity>, all_hits: bool) -> Vec<ObjectHit> {
        cast_object_ray(
            ray,
            pickable,
            &self.parts,
            &self.objects,
            &self.sources(),
            all_hits,
        )
    }

    /// The doodad leg of the mouse pick's world trace, nearest-first: a static doodad occludes the
    /// pick by its drawn opaque batches, never by its collision hull. The reference's `0x69cdb0`
    /// registers every doodad for the trace with hit-test mode `1 + (flags & 0x1000000 != 0)`, 2
    /// for the mouse's mask `0x1000114` (`0x69ce4a`-`0x69ce5c`); the M2 hit test (`0x7089c0`) sends
    /// only mode 3 to the collision triangles (`0x708ff2` → `0x70967c`) and, in mode 2, skips any
    /// render batch whose material blend is not 0 (`0x70903b`-`0x709054`). Deviation: admission is
    /// drawn-this-frame where the reference walks loaded defs, and the texunit-`0x08`, textureless
    /// and zero-alpha skips (`0x70902f`, `0x709071`, `0x7090c2`) are not modelled; neither can stop
    /// the trace sooner than the reference.
    pub fn doodad_occluder(&self, ray: Ray3d) -> Option<RayHit> {
        cast_nearest_accepted(
            ray,
            self.geometry.iter(),
            &self.parts,
            &self.sources(),
            &|kind, blend| {
                kind == ModelKind::Doodad && blend == benilla_formats::ModelBlend::Opaque
            },
        )
    }

    /// The identified cast through a screen pixel.
    pub fn at_cursor(
        &self,
        cursor: Vec2,
        camera: &Camera,
        cam_tf: &GlobalTransform,
        pickable: &HashSet<Entity>,
    ) -> Option<ObjectHit> {
        pick_object_at_cursor(
            cursor,
            camera,
            cam_tf,
            pickable,
            &self.parts,
            &self.objects,
            &self.sources(),
        )
    }
}

/// Registers the three world click messages.
pub struct InteractPlugin;

impl Plugin for InteractPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<WorldClick>()
            .add_message::<WorldRightClick>()
            .add_message::<WorldRightPress>();
    }
}
