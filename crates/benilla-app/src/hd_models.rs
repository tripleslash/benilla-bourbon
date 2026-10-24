//! The updated character models (`hdCharacterModels`) switched under a running client, on the Apply
//! that commits the options row.
//!
//! On the web build the layer is a live mount ([`benilla_formats::layers`]), so a change of the
//! CVar changes the art at that edge, the Turtle WoW switch's steps ([`crate::flavor`]) over the
//! one layer: mount or unmount it, drop the chain art the sprite cache holds, rerun
//! [`benilla_assets::DataLoad`] (the layer carries seven character tables), drop every display and
//! skin cache ([`benilla_assets::ChainDataChanged`]) and reload the loaded assets the layer holds.
//! The character bodies on screen keep the old model until the reloaded models land, then rebuild
//! as a display swap does ([`crate::entities::rebuild_character_bodies`]). A body waiting on the
//! old model stays whole because its rig holds the bindposes it was built with
//! (`benilla_world::rig_palette::RigSkin::bindposes`): the model's own are a sub-asset the reload
//! replaces in place the frame the new model lands. Textures are not waited for: a material holds
//! its image by handle and shows the new bytes as they land, and the skin composite waits for its
//! own inputs. Natively a chain decides its archives as it opens, and the change takes effect at
//! the next start ([`crate::cvars::apply_boot_layers`]).
//!
//! Nothing here runs per frame: the observer fires on a change of the CVar, and the rebuild wait
//! runs only while a switch is waiting.

use std::collections::HashSet;

use bevy::asset::{AssetEvent, UntypedAssetId};
use bevy::prelude::*;

use crate::cvars::CvarChanged;

/// How long the bodies wait for the reloaded models before they rebuild on whatever arrived: the
/// web build fetches the layer's packs on first use, and a stalled fetch must not keep the old art
/// up forever.
const REBUILD_WAIT_SECS: f64 = 60.0;

pub(crate) struct HdModelsPlugin;

impl Plugin for HdModelsPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(on_cvar).add_systems(
            Update,
            rebuild_when_reloaded.run_if(resource_exists::<PendingRebuild>),
        );
    }
}

/// A switch waiting for its reloaded models: the ones that have not landed yet, and when the
/// bodies rebuild regardless.
#[derive(Resource)]
struct PendingRebuild {
    reloading: HashSet<UntypedAssetId>,
    deadline: f64,
}

/// `hdCharacterModels` moved: switch the layer at this frame's command flush, with the whole
/// world. The parse is [`benilla_formats::layers::layer_switch`]'s, as the boot read's is.
fn on_cvar(ev: On<CvarChanged>, mut commands: Commands) {
    if !ev.is("hdCharacterModels") {
        return;
    }
    let on = benilla_formats::layers::layer_switch(&ev.new).unwrap_or(false);
    commands.queue(move |world: &mut World| switch(world, on));
}

/// Mount or unmount the layer and rebuild everything built from it; a no-op when it already is
/// as asked (the stored setting read at boot fires the CVar's change too).
fn switch(world: &mut World, on: bool) {
    if benilla_formats::layers::hd_mounted() == on {
        return;
    }
    if !benilla_formats::layers::hd_live() {
        info!(
            "hdCharacterModels: {} at the next start (a native chain decides its archives as it \
             opens)",
            if on { "on" } else { "off" }
        );
        return;
    }
    let layer = match mount(on) {
        Ok(layer) => layer,
        Err(e) => {
            warn!("hdCharacterModels: the updated character models could not be mounted — {e:#}");
            return;
        }
    };
    // Before the catalogs rerun: a loader that reads art through the sprite cache must not be
    // handed the other layer's.
    if let Some(mut assets) = world.get_resource_mut::<benilla_assets::WorldAssets>() {
        assets.forget_chain_art();
    }
    benilla_assets::run_data_load(world);
    world.trigger(benilla_assets::ChainDataChanged);
    let reloading: Vec<UntypedAssetId> = layer.map_or_else(Vec::new, |in_layer| {
        benilla_assets::reload_chain_assets(world, in_layer)
    });
    // The models are what the bodies are built from; everything else lands in place.
    let models: Vec<UntypedAssetId> = reloading
        .iter()
        .copied()
        .filter(|id| id.type_id() == std::any::TypeId::of::<benilla_assets::M2Model>())
        .collect();
    info!(
        "hdCharacterModels: {} — catalogs rebuilt, chain art forgotten, {} loaded asset(s) \
         reloading; the character bodies rebuild once its {} model(s) land",
        if on { "on" } else { "off" },
        reloading.len(),
        models.len()
    );
    let now = world
        .get_resource::<Time<Real>>()
        .map_or(0.0, |t| t.elapsed_secs_f64());
    // A switch back before the last one's reloads landed waits for both.
    let mut waiting: HashSet<UntypedAssetId> = models.into_iter().collect();
    if let Some(earlier) = world.remove_resource::<PendingRebuild>() {
        waiting.extend(earlier.reloading);
    }
    world.insert_resource(PendingRebuild {
        reloading: waiting,
        deadline: now + REBUILD_WAIT_SECS,
    });
}

/// Mount or unmount the layer, returning a test for the names it holds: those are the names whose
/// bytes change. Unmounting takes the layer before it goes.
#[cfg(target_arch = "wasm32")]
fn mount(on: bool) -> anyhow::Result<Option<impl Fn(&str) -> bool>> {
    let store = if on {
        benilla_formats::web::mount_hd_layer()?;
        benilla_formats::layers::hd()
    } else {
        let store = benilla_formats::layers::hd();
        benilla_formats::web::close_hd_layer();
        store
    };
    Ok(store.map(|s| move |name: &str| s.contains(name)))
}

/// A mounted layer's membership test.
#[cfg(not(target_arch = "wasm32"))]
type LayerContains = fn(&str) -> bool;

/// Native has no live layer; [`switch`] returns before it asks.
#[cfg(not(target_arch = "wasm32"))]
fn mount(_on: bool) -> anyhow::Result<Option<LayerContains>> {
    anyhow::bail!("the updated character models are not a live mount on this build")
}

/// While a switch waits: strike each reloaded model as its new bytes land or its load fails, and
/// once none is left, or the wait runs out, rebuild the character bodies on the new models.
fn rebuild_when_reloaded(
    mut pending: ResMut<PendingRebuild>,
    mut models: MessageReader<AssetEvent<benilla_assets::M2Model>>,
    server: Res<AssetServer>,
    time: Res<Time<Real>>,
    mut commands: Commands,
) {
    for event in models.read() {
        if let AssetEvent::Modified { id } = event {
            pending.reloading.remove(&id.untyped());
        }
    }
    pending
        .reloading
        .retain(|&id| !server.load_state(id).is_failed());
    let timed_out = time.elapsed_secs_f64() >= pending.deadline;
    if !pending.reloading.is_empty() && !timed_out {
        return;
    }
    if timed_out {
        warn!(
            "hdCharacterModels: {} reloaded model(s) had not landed after {REBUILD_WAIT_SECS} s — \
             rebuilding the character bodies on what arrived",
            pending.reloading.len()
        );
    }
    commands.remove_resource::<PendingRebuild>();
    commands.queue(|world: &mut World| {
        let rebuilt = crate::entities::rebuild_character_bodies(world);
        info!("hdCharacterModels: {rebuilt} character bod(ies) rebuilt on the new art");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Image>()
            .init_asset::<benilla_assets::M2Model>()
            .add_plugins(HdModelsPlugin);
        app
    }

    /// A model the switch reloads, as the store holds it before its new bytes land.
    fn reloading_model(app: &mut App) -> bevy::asset::AssetId<benilla_assets::M2Model> {
        app.world()
            .resource::<Assets<benilla_assets::M2Model>>()
            .reserve_handle()
            .id()
    }

    /// The bodies wait for every reloaded model's new bytes, and rebuild the frame the last one
    /// lands; a texture landing does not end the wait.
    #[test]
    fn the_rebuild_waits_for_the_reloaded_models_to_land() {
        let mut app = app();
        let model = reloading_model(&mut app);
        app.update();
        app.world_mut().insert_resource(PendingRebuild {
            reloading: HashSet::from([model.untyped()]),
            deadline: f64::MAX,
        });
        app.update();
        let image = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(Image::default());
        app.world_mut()
            .write_message(AssetEvent::<Image>::Modified { id: image.id() });
        app.update();
        assert!(
            app.world().contains_resource::<PendingRebuild>(),
            "no model has landed: the old bodies stay up"
        );
        // The reload landing, as the asset server announces it.
        app.world_mut()
            .write_message(AssetEvent::<benilla_assets::M2Model>::Modified { id: model });
        app.update();
        assert!(!app.world().contains_resource::<PendingRebuild>());
    }

    /// A stalled reload does not keep the old art up past the wait.
    #[test]
    fn a_stalled_reload_rebuilds_at_the_deadline() {
        let mut app = app();
        let model = reloading_model(&mut app);
        app.update();
        app.world_mut().insert_resource(PendingRebuild {
            reloading: HashSet::from([model.untyped()]),
            deadline: 0.0,
        });
        app.update();
        assert!(!app.world().contains_resource::<PendingRebuild>());
    }

    /// Natively the layer is no live mount: a change of the row waits for the next start and
    /// arms nothing, whichever way it goes.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_native_change_waits_for_the_next_start() {
        let mut app = app();
        for (old, new) in [("1", "0"), ("0", "1")] {
            app.world_mut().trigger(CvarChanged {
                name: "hdCharacterModels".into(),
                old: old.into(),
                new: new.into(),
            });
            app.update();
            assert!(!app.world().contains_resource::<PendingRebuild>());
        }
    }
}
