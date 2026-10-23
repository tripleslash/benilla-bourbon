//! Which client the realm being played is for: a 1.12.1 realm (**Vanilla**) or a Turtle WoW
//! 1.18.1 realm (**Turtle**), read off the version tag the realm list carries
//! (`benilla_protocol::RealmVersion`). Everything Turtle — its data layer, its races, its wire
//! additions, its interface rows — applies only while [`ClientFlavor`] is `Turtle`.
//!
//! The flavor changes at one edge, entering a realm ([`enter_realm`]), and only when the realm's
//! flavor differs from the one in force; an entry whose world dial fails changes it back. The switch mounts or unmounts the Turtle layer, reruns
//! [`benilla_assets::DataLoad`] so every catalog is rebuilt from the new data, drops every cache of
//! chain-derived art ([`benilla_assets::ChainDataChanged`]), reloads the server-held assets the
//! layer names and reinstalls the boot VM, and only then answers the realm park. Nothing here runs
//! per frame: a session that never changes flavor pays nothing for it.
//!
//! The Turtle layer exists on the web build only (`benilla_formats::layers::turtle_available`); a
//! Turtle realm is refused there when the site serves no `/data-turtle`, and on native always.

use bevy::prelude::*;

use benilla_protocol::RealmInfo;

use crate::net::{RealmChoice, RealmRequest};

/// The client the realm being played is for.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ClientFlavor {
    /// World of Warcraft 1.12.1: the reference client, and every realm without a 1.18 tag.
    #[default]
    Vanilla,
    /// Turtle WoW 1.18.1: the 1.12.1 client plus Turtle's data and additions.
    Turtle,
}

impl ClientFlavor {
    /// The flavor a realm is for: Turtle iff its tag says 1.18.
    pub(crate) fn of(realm: &RealmInfo) -> Self {
        if realm.version.is_some_and(|v| v.is_turtle()) {
            Self::Turtle
        } else {
            Self::Vanilla
        }
    }

    pub(crate) fn is_turtle(self) -> bool {
        self == Self::Turtle
    }

    /// The interface flavor a VM built under this client flavor runs.
    pub(crate) fn ui(self) -> benilla_ui::script::UiFlavor {
        match self {
            Self::Vanilla => benilla_ui::script::UiFlavor::Vanilla,
            Self::Turtle => benilla_ui::script::UiFlavor::Turtle,
        }
    }
}

/// The flavor in force, for a table with no `World` at hand (a race's name, side, helm prefix):
/// the Turtle layer is mounted exactly while [`ClientFlavor`] is `Turtle`, since the switch
/// changes both, so this is one relaxed atomic load. Always false on native, which has no layer.
pub(crate) fn turtle() -> bool {
    benilla_formats::layers::turtle_mounted()
}

/// Whether this build can play `realm` at all: a Turtle realm needs the Turtle layer, which only
/// the web build mounts. The auto-answer skips a realm that fails this; a click on one is refused
/// with a reason.
pub(crate) fn playable(realm: &RealmInfo) -> bool {
    !ClientFlavor::of(realm).is_turtle() || benilla_formats::layers::turtle_available()
}

/// The flavor of the last realm whose world handshake held (its roster arrived): the one a failed
/// entry falls back to. Vanilla until a first roster.
#[derive(Resource, Default)]
pub(crate) struct Committed(pub(crate) ClientFlavor);

pub(crate) struct FlavorPlugin;

impl Plugin for FlavorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClientFlavor>()
            .init_resource::<Committed>()
            .add_systems(
                Update,
                settle_entry.after(benilla_world::schedule::WorldStage::Net),
            )
            // After a flavor switch, the glue VM off the new chain, once the manager is idle.
            .add_systems(
                Update,
                crate::ui_script::rebuild_stale_glue_vm
                    .after(crate::ui_script::glue_mgr::drive_glue_mgr)
                    .after(settle_entry),
            );
    }
}

/// A roster commits the flavor its realm was entered under. A login failure after an entry that
/// switched it (the Turtle realm's world server unreachable) switches back, so the realm list and
/// the next realm are never served the failed realm's data.
pub(crate) fn settle_entry(
    mut rosters: MessageReader<crate::net::CharListMessage>,
    mut failures: MessageReader<crate::net::LoginFailedMessage>,
    flavor: Res<ClientFlavor>,
    mut committed: ResMut<Committed>,
    mut commands: Commands,
) {
    if rosters.read().last().is_some() {
        committed.0 = *flavor;
    }
    if failures.read().last().is_some() && *flavor != committed.0 {
        let to = committed.0;
        commands.queue(move |world: &mut World| {
            let from = *world.resource::<ClientFlavor>();
            if from == to {
                return;
            }
            match switch(world, to) {
                Ok(()) => info!("flavor: {from:?} → {to:?}, the realm entry failed"),
                Err(e) => warn!("flavor: no way back to {to:?} after a failed entry — {e:#}"),
            }
        });
    }
}

/// Answer the realm park with `realm`, the record as the list held it, switching the flavor first
/// when the realm's differs from the one in force. Every [`RealmRequest::Enter`] is sent from
/// here, the native list's and the stock glue's alike, so no realm is dialled on the other
/// client's data. `generation` is the abandon generation the app held when it asked
/// ([`crate::net::LoginAbandon::now`]). Queued, so the switch has the whole world; it runs at
/// this frame's command flush, before the world dial it precedes can return anything.
pub(crate) fn enter_realm(commands: &mut Commands, realm: RealmInfo, generation: u64) {
    commands.queue(move |world: &mut World| {
        let to = ClientFlavor::of(&realm);
        let from = world
            .get_resource::<ClientFlavor>()
            .copied()
            .unwrap_or_default();
        if from != to {
            if let Err(e) = switch(world, to) {
                refuse(world, &realm.name, e);
                return;
            }
            info!("flavor: {from:?} → {to:?} for realm {:?}", realm.name);
        }
        if let Some(choice) = world.get_resource::<RealmChoice>() {
            let _ = choice.0.send(RealmRequest::Enter { realm, generation });
        }
    });
}

/// Change the data the client answers from to `to`'s, and everything built from it. The mount is
/// the first step and the only one that can fail, so on an error nothing has changed.
fn switch(world: &mut World, to: ClientFlavor) -> anyhow::Result<()> {
    // The layer, and which names it holds: those are the names whose bytes change.
    let layer = mount(to)?;
    // In force before the catalogs rerun: a loader whose checks are 1.12.1's (the create catalog's
    // race and class table) reads the flavor to know which data it is looking at.
    world.insert_resource(to);
    // The server-answered templates kept across sessions (the reference's WDB caches) belong to
    // the other client; the creature cache is per realm on disk and reloads with the realm.
    if let Some(mut items) = world.get_resource_mut::<crate::items::Items>() {
        items.forget_templates();
    }
    if let Some(mut gos) = world.get_resource_mut::<crate::go_templates::GameObjectTemplates>() {
        gos.forget();
    }
    if let Some(mut pages) = world.get_resource_mut::<crate::ui_item_text::PageTexts>() {
        pages.forget();
    }
    // Before the catalogs rerun: a loader that reads art through the sprite cache must not be
    // handed the other flavor's.
    if let Some(mut assets) = world.get_resource_mut::<benilla_assets::WorldAssets>() {
        assets.forget_chain_art();
    }
    benilla_assets::run_data_load(world);
    world.trigger(benilla_assets::ChainDataChanged);
    let reloaded = layer.map_or(0, |in_layer| {
        benilla_assets::reload_chain_assets(world, in_layer).len()
    });
    // The boot VM holds GlobalStrings off the chain it was built on.
    crate::ui_script::reinstall_boot_vm(world);
    info!("flavor: catalogs rebuilt, chain art forgotten, {reloaded} loaded asset(s) reloaded");
    Ok(())
}

/// Mount or unmount the Turtle layer for `to`, and the reforged art over it while the player's
/// setting has that on ([`benilla_formats::layers::reforged_models`], read as the page loaded),
/// returning a test for the names they hold.
///
/// The reforged art is Turtle's own add-on, so it goes wherever the Turtle layer goes and never
/// onto a 1.12.1 realm. An origin that serves no `/data-reforged` plays on Turtle's art; only the
/// Turtle layer's own mount can refuse the realm.
#[cfg(target_arch = "wasm32")]
fn mount(to: ClientFlavor) -> anyhow::Result<Option<impl Fn(&str) -> bool>> {
    let (store, reforged) = match to {
        ClientFlavor::Turtle => {
            benilla_formats::web::mount_turtle_layer()?;
            if benilla_formats::layers::reforged_models() {
                if let Err(e) = benilla_formats::web::mount_reforged_layer() {
                    warn!("flavor: the reforged models and textures could not be mounted — {e:#}");
                }
            }
            (
                benilla_formats::layers::turtle(),
                benilla_formats::layers::reforged(),
            )
        }
        ClientFlavor::Vanilla => {
            // Taken before the unmount: their names are the ones going back to the 1.12.1 bytes.
            let stores = (
                benilla_formats::layers::turtle(),
                benilla_formats::layers::reforged(),
            );
            benilla_formats::web::close_reforged_layer();
            benilla_formats::web::close_turtle_layer();
            stores
        }
    };
    if store.is_none() && reforged.is_none() {
        return Ok(None);
    }
    Ok(Some(move |name: &str| {
        store.as_ref().is_some_and(|s| s.contains(name))
            || reforged.as_ref().is_some_and(|r| r.contains(name))
    }))
}

/// A mounted layer's membership test.
#[cfg(not(target_arch = "wasm32"))]
type LayerContains = fn(&str) -> bool;

/// Native has no Turtle layer: a Turtle realm cannot be played, and Vanilla is all there is.
#[cfg(not(target_arch = "wasm32"))]
fn mount(to: ClientFlavor) -> anyhow::Result<Option<LayerContains>> {
    match to {
        ClientFlavor::Turtle => anyhow::bail!("this build has no Turtle WoW layer (web only)"),
        ClientFlavor::Vanilla => Ok(None),
    }
}

/// A realm the client cannot play: say why, and leave the park unanswered so another realm can be
/// picked. The words go out as a login failure, which ends the connect the stock glue screens'
/// manager is waiting on ([`crate::ui_script::glue_mgr`]) and shows over the realm list. The run's
/// own pick never takes such a realm ([`crate::glue_session::realm`]), so nothing resubmits it.
fn refuse(world: &mut World, name: &str, e: anyhow::Error) {
    warn!("flavor: realm {name:?} refused — {e:#}");
    let why = format!(
        "{name} is a Turtle WoW 1.18.1 realm, and this client could not load the Turtle WoW data \
         it needs."
    );
    if let Some(mut failures) =
        world.get_resource_mut::<bevy::ecs::message::Messages<crate::net::LoginFailedMessage>>()
    {
        failures.write(crate::net::LoginFailedMessage {
            refusal: None,
            reason: why,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_protocol::RealmVersion;

    fn realm(version: Option<RealmVersion>) -> RealmInfo {
        named("R", version)
    }

    fn named(name: &str, version: Option<RealmVersion>) -> RealmInfo {
        RealmInfo {
            name: name.into(),
            address: "127.0.0.1:8085".into(),
            population: 1.0,
            characters: 0,
            realm_type: 0,
            flags: 0x04,
            category: 1,
            id: 1,
            version,
        }
    }

    fn v(major: u8, minor: u8, patch: u8) -> Option<RealmVersion> {
        Some(RealmVersion {
            major,
            minor,
            patch,
            build: None,
        })
    }

    #[test]
    fn only_a_1_18_tag_is_turtle() {
        assert_eq!(ClientFlavor::of(&realm(v(1, 18, 1))), ClientFlavor::Turtle);
        assert_eq!(ClientFlavor::of(&realm(v(1, 12, 1))), ClientFlavor::Vanilla);
        assert_eq!(ClientFlavor::of(&realm(None)), ClientFlavor::Vanilla);
        assert_eq!(ClientFlavor::of(&realm(v(2, 18, 0))), ClientFlavor::Vanilla);
    }

    /// Native has no Turtle layer, so a Turtle realm is not auto-entered there; a 1.12.1 realm
    /// always is.
    #[test]
    fn playable_follows_the_layer() {
        assert!(playable(&realm(v(1, 12, 1))));
        assert!(playable(&realm(None)));
        assert_eq!(
            playable(&realm(v(1, 18, 1))),
            benilla_formats::layers::turtle_available()
        );
    }

    /// Entering a realm of the flavor already in force answers the park and touches nothing else:
    /// no layer, no catalog rerun, no cache flush.
    #[test]
    fn same_flavor_entry_only_answers_the_park() {
        let mut app = App::new();
        app.add_plugins(FlavorPlugin);
        let (tx, rx) = async_channel::unbounded();
        app.insert_resource(RealmChoice(tx.into()));
        app.add_schedule(bevy::ecs::schedule::Schedule::new(benilla_assets::DataLoad));
        let runs = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter = runs.clone();
        app.add_systems(benilla_assets::DataLoad, move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        let mut commands = app.world_mut().commands();
        enter_realm(
            &mut commands,
            named("Classic (1.12.1.5875)", v(1, 12, 1)),
            7,
        );
        app.world_mut().flush();
        assert!(
            matches!(
                rx.try_recv(),
                Ok(RealmRequest::Enter { realm, generation: 7 })
                    if realm.name == "Classic (1.12.1.5875)"
            ),
            "the record and the generation it was asked under"
        );
        assert_eq!(runs.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(
            *app.world().resource::<ClientFlavor>(),
            ClientFlavor::Vanilla
        );
    }

    /// Native refuses a Turtle realm and leaves the park unanswered and the flavor as it was.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_refuses_a_turtle_realm_and_changes_nothing() {
        let mut app = App::new();
        app.add_plugins(FlavorPlugin);
        let (tx, rx) = async_channel::unbounded();
        app.insert_resource(RealmChoice(tx.into()));
        let mut commands = app.world_mut().commands();
        enter_realm(
            &mut commands,
            named("TurtleWoW (1.18.1.7272)", v(1, 18, 1)),
            0,
        );
        app.world_mut().flush();
        assert!(rx.try_recv().is_err(), "the park stays unanswered");
        assert_eq!(
            *app.world().resource::<ClientFlavor>(),
            ClientFlavor::Vanilla
        );
    }
}
