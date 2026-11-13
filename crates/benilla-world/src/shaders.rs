//! The engine's WGSL, embedded in the binary; the game's shaders are embedded by `benilla-app`.
//! These calls must stay in a file directly under `src/`: `embedded_asset!` derives the served
//! path from `file!()` without normalizing it, so a call from a submodule would serve the shader
//! under that module's directory instead of `embedded://benilla_world/shaders/<name>`.

use bevy::prelude::*;

/// Embed the engine's WGSL files under `embedded://benilla_world/shaders/`. The first member of
/// [`crate::world_plugins::WorldPlugins`]; it needs `DefaultPlugins`, which creates the asset
/// registry, added first.
pub(crate) fn plugin(app: &mut App) {
    bevy::asset::embedded_asset!(app, "shaders/sky_vertex.wesl");
    bevy::asset::embedded_asset!(app, "shaders/sky.wesl");
    bevy::asset::embedded_asset!(app, "shaders/star.wesl");
    bevy::asset::embedded_asset!(app, "shaders/cloud.wesl");
    bevy::asset::embedded_asset!(app, "shaders/celestial.wesl");
    bevy::asset::embedded_asset!(app, "shaders/ffx_glow.wesl");
    bevy::asset::embedded_asset!(app, "shaders/wow_effect.wesl");
    bevy::asset::embedded_asset!(app, "shaders/static_gx.wesl");
    bevy::asset::embedded_asset!(app, "shaders/ssilvb.wesl");
    bevy::asset::embedded_asset!(app, "shaders/depth_of_field.wesl");
    bevy::asset::embedded_asset!(app, "shaders/liquid_screen.wesl");
    bevy::asset::embedded_asset!(app, "shaders/liquid_ripple.wesl");
    bevy::asset::embedded_asset!(app, "shaders/wet_ssr.wesl");
    // fork: perf — a library (`benilla_world::shaders::effect_points`): registered and held
    // loaded, so the effect lane's and FSR's import of it resolves.
    bevy::shader::load_shader_library!(app, "shaders/effect_points.wesl");
    // fork: a library (`modern::skydome`): the dome's dawn/dusk warp, shared by the dome
    // and the water's reflection of it. Here rather than in `modern_libraries` because the dome
    // draws under both lighting models.
    bevy::shader::load_shader_library!(app, "shaders/modern/skydome.wesl");
}

/// The modern lighting model's shared WESL libraries (`benilla_world::shaders::modern::*`),
/// registered AND held loaded so an import of them resolves (`FORK.md`). Called by
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
    bevy::shader::load_shader_library!(app, "shaders/modern/color.wesl");
    bevy::shader::load_shader_library!(app, "shaders/modern/frame.wesl");
    bevy::shader::load_shader_library!(app, "shaders/modern/sunshadow.wesl", block_defs);
    bevy::shader::load_shader_library!(app, "shaders/modern/froxelfog.wesl");
    bevy::shader::load_shader_library!(app, "shaders/modern/probegi.wesl", block_defs);
    bevy::shader::load_shader_library!(app, "shaders/modern/locallight.wesl");
    // fork: the bindless static_gx group 1, imported under `GX_BINDLESS` only.
    bevy::shader::load_shader_library!(app, "shaders/modern/gx_bindless.wesl");
    // fork: the ray-traced scene's hit library and attribute repack (`rt_scene.rs`), native only:
    // nothing composes the library until a ray-traced lane imports it.
    #[cfg(not(target_arch = "wasm32"))]
    {
        bevy::shader::load_shader_library!(app, "shaders/modern/rt_scene.wesl");
        bevy::asset::embedded_asset!(app, "shaders/modern/rt_repack.wgsl");
        // fork: static_gx's Hi-Z pyramid and two-phase GPU cull (`gpu_cull.rs`), native only; the
        // Hi-Z tests the draw cull and the meshlet task stage share.
        bevy::asset::embedded_asset!(app, "shaders/modern/gx_hiz.wesl");
        bevy::asset::embedded_asset!(app, "shaders/modern/gx_cull.wesl");
        bevy::shader::load_shader_library!(app, "shaders/modern/gx_occlusion.wesl");
        // fork: the probe trace's ray-query stage (`modern::gi::rt_gi`), native only.
        bevy::asset::embedded_asset!(app, "shaders/modern/gi_trace_rt.wesl");
        // fork: the ray-traced shadows' library (`modern::rt_shadow`), native only: imported under
        // `RT_SHADOWS` / `RT_FROXEL`, which only the ray-traced tier's pipelines carry.
        bevy::shader::load_shader_library!(app, "shaders/modern/rt_shadow.wesl");
        // fork: G — the water's ray-traced reflection (`modern::rt_reflect`), native only: imported
        // under `RT_REFLECT`, which only the water's pipelines carry where the lane may run.
        bevy::shader::load_shader_library!(app, "shaders/modern/rt_reflect.wesl");
    }
    // The browser registers no ray-traced library: a receiver imports one under `@if(RT_…)`
    // alone, which the web never sets, and wesl resolves no import its conditions remove.
    bevy::asset::embedded_asset!(app, "shaders/modern/froxel_scatter.wesl");
    bevy::asset::embedded_asset!(app, "shaders/modern/froxel_integrate.wesl");
    bevy::asset::embedded_asset!(app, "shaders/modern/fog_composite.wesl");
    bevy::asset::embedded_asset!(app, "shaders/modern/static_gx_shadow.wesl");
    bevy::asset::embedded_asset!(app, "shaders/modern/terrain_shadow.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/entity_shadow.wesl");
    bevy::asset::embedded_asset!(app, "shaders/modern/depth_fill.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/static_gx_voxel.wesl");
    bevy::asset::embedded_asset!(app, "shaders/modern/gi_voxel_clear.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/modern/gi_classify.wesl");
    bevy::asset::embedded_asset!(app, "shaders/modern/gi_trace.wesl");
    bevy::asset::embedded_asset!(app, "shaders/modern/gi_blend.wesl");
    bevy::asset::embedded_asset!(app, "shaders/modern/light_bin.wesl");
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
            if !(name.ends_with(".wgsl") || name.ends_with(".wesl")) {
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
    /// The lines of `src` that take an implicit derivative (`textureSample`, `textureSampleBias`,
    /// `textureSampleCompare`, and the explicit builtins `dpdx`, `dpdy`, `fwidth` with their
    /// `Coarse` and `Fine` variants) below a function body's top level. WGSL allows them only in
    /// uniform control flow, and a branch on a fragment input is not. This reads text: it does not
    /// follow helper calls, a sample after a `return` inside a branch, or `/* */` comments. A WESL
    /// `@if(…) {` / `@elif(…) {` / `@else {` block is no level: wesl flattens it into its parent.
    fn derivatives_in_nested_blocks(src: &str) -> Vec<usize> {
        const IMPLICIT: [&str; 12] = [
            "textureSample",
            "textureSampleBias",
            "textureSampleCompare",
            "dpdx",
            "dpdxCoarse",
            "dpdxFine",
            "dpdy",
            "dpdyCoarse",
            "dpdyFine",
            "fwidth",
            "fwidthCoarse",
            "fwidthFine",
        ];
        /// Whether `head`, the text before a `{`, is a bare conditional-translation attribute.
        fn flattens(head: &str) -> bool {
            let head = head.trim();
            if head == "@else" {
                return true;
            }
            let Some(rest) = head
                .strip_prefix("@if")
                .or_else(|| head.strip_prefix("@elif"))
            else {
                return false;
            };
            let rest = rest.trim_start();
            let mut open = 0usize;
            for (i, c) in rest.char_indices() {
                match c {
                    '(' => open += 1,
                    ')' => {
                        open = open.saturating_sub(1);
                        if open == 0 {
                            return rest[i + 1..].trim().is_empty();
                        }
                    }
                    _ if open == 0 => return false,
                    _ => {}
                }
            }
            false
        }
        // Per open brace: whether it is a real block (true) or a flattened `@if` one.
        let mut open: Vec<bool> = Vec::new();
        let mut head = String::new();
        let mut hits = Vec::new();
        for (n, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            let mut word = String::new();
            for c in code.chars().chain(std::iter::once(' ')) {
                if c.is_alphanumeric() || c == '_' {
                    word.push(c);
                    head.push(c);
                    continue;
                }
                let depth = open.iter().filter(|real| **real).count();
                if depth > 1 && IMPLICIT.contains(&word.as_str()) {
                    hits.push(n + 1);
                }
                word.clear();
                match c {
                    '{' => {
                        open.push(!flattens(&head));
                        head.clear();
                    }
                    '}' | ';' => {
                        if c == '}' {
                            open.pop();
                        }
                        head.clear();
                    }
                    _ => head.push(c),
                }
            }
            head.push(' ');
        }
        hits
    }

    /// naga's check for this is compiled out of the fragment stage
    /// (`DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE`, naga 27 and 30), so these shaders validate
    /// natively and fail in a browser's compiler. This reads the text instead: no sample that
    /// takes implicit derivatives sits in a nested block, where a branch on a fragment input
    /// would make it non-uniform. Hoist the sample and `select` after.
    #[test]
    fn derivative_samples_stay_in_uniform_control_flow() {
        let bad = "fn fragment(in: V) -> vec4<f32> {\n    if (in.flag != 0u) {\n        let c = textureSampleBias(t, s, in.uv, 0.0);\n    }\n    let d = textureSample(t, s, in.uv);\n}\n";
        assert_eq!(
            derivatives_in_nested_blocks(bad),
            [3],
            "the scanner finds the nested sample"
        );
        let flat = "@if(A) {
fn a(in: V) {
    let d = dpdx(in.uv);
}
} @else {
fn b(in: V) {
    @if(B && (C || D)) {
    let e = fwidth(in.uv);
    }
    if (in.flag != 0u) {
        let f = dpdy(in.uv);
    }
}
}
";
        assert_eq!(
            derivatives_in_nested_blocks(flat),
            [11],
            "an `@if` block is no level; a real branch inside one still is"
        );

        let root = env!("CARGO_MANIFEST_DIR");
        for path in [
            "/src/shaders/static_gx.wesl",
            "/../benilla-assets/src/shaders/terrain.wesl",
            // fork: the water, whose footprint and shore derivatives sit at its body's top.
            "/../benilla-assets/src/shaders/liquid_cata.wesl",
        ] {
            let src = std::fs::read_to_string(format!("{root}{path}")).expect(path);
            assert_eq!(
                derivatives_in_nested_blocks(&src),
                Vec::<usize>::new(),
                "{path}"
            );
        }
    }
}
