//! The engine's WGSL, embedded in the binary; the game's shaders are embedded by `benilla-app`.
//! These calls must stay in a file directly under `src/`: `embedded_asset!` derives the served
//! path from `file!()` without normalizing it, so a call from a submodule would serve the shader
//! under that module's directory instead of `embedded://benilla_world/shaders/<name>.wgsl`.

use bevy::prelude::*;

/// Embed the engine's WGSL files under `embedded://benilla_world/shaders/`. The first member of
/// [`crate::world_plugins::WorldPlugins`]; it needs `DefaultPlugins`, which creates the asset
/// registry, added first.
pub(crate) fn plugin(app: &mut App) {
    bevy::asset::embedded_asset!(app, "shaders/sky_vertex.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/sky.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/star.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/cloud.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/celestial.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/ffx_glow.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/wow_effect.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/static_gx.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/ssilvb.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/liquid_screen.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/liquid_ripple.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/wet_ssr.wgsl");
    // fork: perf — a library (`benilla_world::effect_points`): registered and held loaded, so
    // the effect lane's and FSR's `#import` of it resolves.
    bevy::shader::load_shader_library!(app, "shaders/effect_points.wgsl");
}

/// The modern lighting model's shared WGSL libraries (`benilla_modern::*`), registered AND held
/// loaded so an `#import` of them resolves (`FORK.md`). Called by
/// [`crate::modern::ModernLightingPlugin`]; lives here for the same path reason as [`plugin`].
pub(crate) fn modern_libraries(app: &mut App) {
    // fork: perf — `$WOW_MODERN_ARRAYINDEX=1` (an A/B lever, read once) builds the two libraries
    // that hold an array inside a by-value struct with the computed index they had before
    // (`sunshadow::cascade_clip`, `probegi::grid_row`).
    let array_index = crate::launch_env::var("WOW_MODERN_ARRAYINDEX").as_deref() == Some("1");
    let block_defs = move |settings: &mut bevy::shader::ShaderSettings| {
        if array_index {
            settings.shader_defs.push("MODERN_ARRAY_INDEX".into());
        }
    };
    bevy::shader::load_shader_library!(app, "shaders/modern/color.wgsl");
    bevy::shader::load_shader_library!(app, "shaders/modern/frame.wgsl");
    bevy::shader::load_shader_library!(app, "shaders/modern/sunshadow.wgsl", block_defs);
    bevy::shader::load_shader_library!(app, "shaders/modern/froxelfog.wgsl");
    bevy::shader::load_shader_library!(app, "shaders/modern/probegi.wgsl", block_defs);
    bevy::shader::load_shader_library!(app, "shaders/modern/locallight.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/froxel_scatter.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/froxel_integrate.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/fog_composite.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/static_gx_shadow.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/terrain_shadow.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/entity_shadow.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/depth_fill.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/static_gx_voxel.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/gi_voxel_clear.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/gi_classify.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/gi_trace.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/gi_blend.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/light_bin.wgsl");
}

#[cfg(test)]
mod tests {
    use bevy::asset::io::AssetSourceId;
    use bevy::prelude::*;

    /// Reads each file on disk back through the embedded source a `ShaderRef` uses, so a shader
    /// that is unregistered, or registered from a submodule, fails here instead of drawing nothing.
    #[test]
    fn every_engine_shader_answers_at_its_embedded_path() {
        let mut app = App::new();
        app.add_plugins(bevy::asset::AssetPlugin::default());
        // The library line loads its shader, which wants the io pool and the asset type.
        bevy::tasks::IoTaskPool::get_or_init(bevy::tasks::TaskPool::new);
        app.init_asset::<bevy::shader::Shader>();
        app.add_plugins(super::plugin);
        let server = app.world().resource::<AssetServer>();
        let source = server.get_source(AssetSourceId::from("embedded")).unwrap();

        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders");
        let mut found = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            if !name.ends_with(".wgsl") {
                continue;
            }
            found += 1;
            let path = format!("benilla_world/shaders/{name}");
            let read = bevy::tasks::block_on(async {
                source.reader().read(std::path::Path::new(&path)).await
            });
            assert!(
                read.is_ok(),
                "embedded://{path} does not resolve — {name} is on disk but not registered in \
                 `shaders::plugin`, or was registered from a file that is not directly under src/"
            );
        }
        // One per `embedded_asset!` or `load_shader_library!` line in `plugin`.
        assert_eq!(found, 13, "the engine's shader set changed size");
    }
}
