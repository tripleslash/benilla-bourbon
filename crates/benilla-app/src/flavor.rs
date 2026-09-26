//! Which client the realm being played is for: a 1.12.1 realm (**Vanilla**) or a Turtle WoW
//! 1.18.1 realm (**Turtle**), read off the version tag the realm list carries
//! (`benilla_protocol::RealmVersion`). Everything Turtle — its data layer, its races, its wire
//! additions, its interface rows — applies only while [`ClientFlavor`] is `Turtle`.
//!
//! The flavor changes at one edge, entering a realm ([`enter_realm`]), and only when the realm's
//! flavor differs from the one in force. The switch mounts or unmounts the Turtle layer, reruns
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

pub(crate) struct FlavorPlugin;

impl Plugin for FlavorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClientFlavor>();
    }
}

/// Answer the realm park with `name`, switching the flavor first when the realm's differs from
/// the one in force. Queued, so the switch has the whole world; it runs at this frame's command
/// flush, before the world dial it precedes can return anything.
pub(crate) fn enter_realm(commands: &mut Commands, name: String, to: ClientFlavor) {
    commands.queue(move |world: &mut World| {
        let from = *world.resource::<ClientFlavor>();
        if from != to {
            if let Err(e) = switch(world, to) {
                refuse(world, &name, e);
                return;
            }
            info!("flavor: {from:?} → {to:?} for realm {name:?}");
        }
        if let Some(choice) = world.get_resource::<RealmChoice>() {
            let _ = choice.0.try_send(RealmRequest::Enter(name));
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
    // Before the catalogs rerun: a loader that reads art through the sprite cache must not be
    // handed the other flavor's.
    if let Some(mut assets) = world.get_resource_mut::<benilla_assets::WorldAssets>() {
        assets.forget_chain_art();
    }
    benilla_assets::run_data_load(world);
    world.trigger(benilla_assets::ChainDataChanged);
    let reloaded = layer.map_or(0, |in_layer| {
        benilla_assets::reload_chain_assets(world, in_layer)
    });
    // The boot VM holds GlobalStrings off the chain it was built on.
    crate::ui_script::reinstall_boot_vm(world);
    info!("flavor: catalogs rebuilt, chain art forgotten, {reloaded} loaded asset(s) reloaded");
    Ok(())
}

/// Mount or unmount the Turtle layer for `to`, returning a test for the names it holds.
#[cfg(target_arch = "wasm32")]
fn mount(to: ClientFlavor) -> anyhow::Result<Option<impl Fn(&str) -> bool>> {
    let store = match to {
        ClientFlavor::Turtle => {
            benilla_formats::web::mount_turtle_layer()?;
            benilla_formats::layers::turtle()
        }
        ClientFlavor::Vanilla => {
            // Taken before the unmount: its names are the ones going back to the 1.12.1 bytes.
            let store = benilla_formats::layers::turtle();
            benilla_formats::web::close_turtle_layer();
            store
        }
    };
    Ok(store.map(|s| move |name: &str| s.contains(name)))
}

/// Native has no Turtle layer: a Turtle realm cannot be played, and Vanilla is all there is.
#[cfg(not(target_arch = "wasm32"))]
fn mount(to: ClientFlavor) -> anyhow::Result<Option<fn(&str) -> bool>> {
    match to {
        ClientFlavor::Turtle => anyhow::bail!("this build has no Turtle WoW layer (web only)"),
        ClientFlavor::Vanilla => Ok(None),
    }
}

/// A realm the client cannot play: say why over the realm list, which stays up, and leave the
/// park unanswered so another realm can be picked.
fn refuse(world: &mut World, name: &str, e: anyhow::Error) {
    warn!("flavor: realm {name:?} refused — {e:#}");
    if let Some(mut dialog) = world.get_resource_mut::<crate::glue::dialog::GlueDialog>() {
        dialog.open_error(&format!(
            "{name} is a Turtle WoW 1.18.1 realm, and this client could not load the Turtle WoW \
             data it needs."
        ));
    }
    if let Some(mut realms) = world.get_resource_mut::<crate::realm_select::Realms>() {
        realms.reopen();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_protocol::RealmVersion;

    fn realm(version: Option<RealmVersion>) -> RealmInfo {
        RealmInfo {
            name: "R".into(),
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

    /// Goblin and High Elf exist only under Turtle, on the sides their templates' group masks give;
    /// 1.12.1's tables answer exactly as before.
    #[test]
    fn turtle_races_are_turtle_only() {
        use crate::char_create::{alliance_races, horde_races};
        use crate::ui_unit::{race_faction_group_in, race_names_in, race_pvp_team_in};
        assert_eq!(race_names_in(9, false), None);
        assert_eq!(race_names_in(10, false), None);
        assert_eq!(race_names_in(9, true), Some(("Goblin", "Goblin")));
        assert_eq!(race_names_in(10, true), Some(("High Elf", "BloodElf")));
        assert_eq!(race_names_in(5, true), race_names_in(5, false));
        assert_eq!(race_faction_group_in(9, false), None);
        assert_eq!(race_faction_group_in(9, true), Some("Horde"));
        assert_eq!(race_faction_group_in(10, true), Some("Alliance"));
        assert_eq!(
            race_pvp_team_in(9, false),
            1,
            "1.12.1's Goblin row answers Alliance"
        );
        assert_eq!(race_pvp_team_in(10, false), -1);
        assert_eq!(race_pvp_team_in(9, true), 0);
        assert_eq!(race_pvp_team_in(10, true), 1);
        assert_eq!(alliance_races(false), [1, 3, 4, 7]);
        assert_eq!(horde_races(false), [2, 5, 6, 8]);
        assert_eq!(alliance_races(true), [1, 3, 4, 7, 10]);
        assert_eq!(horde_races(true), [2, 5, 6, 8, 9]);
    }

    /// Entering a realm of the flavor already in force answers the park and touches nothing else:
    /// no layer, no catalog rerun, no cache flush.
    #[test]
    fn same_flavor_entry_only_answers_the_park() {
        let mut app = App::new();
        app.add_plugins(FlavorPlugin);
        let (tx, rx) = async_channel::unbounded();
        app.insert_resource(RealmChoice(tx));
        app.add_schedule(bevy::ecs::schedule::Schedule::new(benilla_assets::DataLoad));
        let runs = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter = runs.clone();
        app.add_systems(benilla_assets::DataLoad, move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        let mut commands = app.world_mut().commands();
        enter_realm(
            &mut commands,
            "Classic (1.12.1.5875)".into(),
            ClientFlavor::Vanilla,
        );
        app.world_mut().flush();
        assert!(
            matches!(rx.try_recv(), Ok(RealmRequest::Enter(n)) if n == "Classic (1.12.1.5875)")
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
        app.insert_resource(RealmChoice(tx));
        let mut commands = app.world_mut().commands();
        enter_realm(
            &mut commands,
            "TurtleWoW (1.18.1.7272)".into(),
            ClientFlavor::Turtle,
        );
        app.world_mut().flush();
        assert!(rx.try_recv().is_err(), "the park stays unanswered");
        assert_eq!(
            *app.world().resource::<ClientFlavor>(),
            ClientFlavor::Vanilla
        );
    }
}
