//! `ChrClasses.dbc`, loaded once for its unrelated readers: the pet book (field 4, pet name token),
//! `UnitHasRelicSlot` (field 16) and `crate::spell::mods` (field 15, spell family). Absent when the
//! load fails; each reader then takes the reference's degraded answer: `"PET"`, no relic slot,
//! family 0.

use bevy::prelude::*;

use benilla_assets::DataLoadAppExt;
use benilla_assets::{LockRecover, WorldAssets};
use benilla_formats::{ChrClasses, ChrRaces};

/// The parsed table; [`benilla_formats::ChrClasses`] documents the columns.
#[derive(Resource)]
pub(crate) struct ChrClassTable(pub(crate) ChrClasses);

/// `ChrRaces.dbc`'s names and tokens, at the install's locale column; `UnitRace` and the race words
/// the chat and social lists print come from it. Absent when the load fails, and no race has a name.
#[derive(Resource)]
pub(crate) struct ChrRaceTable(pub(crate) ChrRaces);

pub(crate) struct ChrClassesPlugin;

impl Plugin for ChrClassesPlugin {
    fn build(&self, app: &mut App) {
        app.add_data_loaders((load_chr_classes, load_chr_races));
    }
}

fn load_chr_classes(mut commands: Commands, assets: Option<Res<WorldAssets>>) {
    let Some(assets) = assets else { return };
    let loaded = {
        let mut chain = assets.chain.lock_recover();
        benilla_formats::load_chr_classes(&mut chain)
    };
    match loaded {
        Ok(table) => commands.insert_resource(ChrClassTable(table)),
        // A rerun on a data change that fails drops the other client's table.
        Err(e) => {
            commands.remove_resource::<ChrClassTable>();
            warn!(
            "chr_classes: ChrClasses.dbc failed to load — every pet book tab reads the client's \
             own \"PET\" fallback, so a warlock's says Pet rather than Demon, no class reads \
             as having a relic slot, and no talent spell modifier can apply (the gate has no \
             class family to match): {e:#}"
            )
        }
    }
}

fn load_chr_races(mut commands: Commands, assets: Option<Res<WorldAssets>>) {
    let Some(assets) = assets else { return };
    let loaded = {
        let mut chain = assets.chain.lock_recover();
        benilla_formats::load_chr_races(&mut chain)
    };
    match loaded {
        Ok(table) => commands.insert_resource(ChrRaceTable(table)),
        Err(e) => {
            commands.remove_resource::<ChrRaceTable>();
            warn!(
                "chr_races: ChrRaces.dbc failed to load — no unit has a race name or token, so \
             `UnitRace` answers nil and the social lists print no race: {e:#}"
            )
        }
    }
}
