//! The game's five WGSL shaders, embedded in the binary. Registered from the root of `src/`: the
//! macro does not normalize paths, so a call from a submodule would bake its directory into the
//! served path.

use bevy::prelude::*;

/// Register the five shaders under `embedded://benilla_app/shaders/…`; added after `DefaultPlugins`
/// creates the registry it fills.
pub(crate) fn plugin(app: &mut App) {
    bevy::asset::embedded_asset!(app, "shaders/ui_quad.wesl");
    bevy::asset::embedded_asset!(app, "shaders/ui_add.wesl");
    bevy::asset::embedded_asset!(app, "shaders/ui_gamma.wesl");
    bevy::asset::embedded_asset!(app, "shaders/ui_node_gamma.wesl");
    bevy::asset::embedded_asset!(app, "shaders/ui_slice_gamma.wesl");
    bevy::asset::embedded_asset!(app, "shaders/shadow_caster.wgsl");
}

#[cfg(test)]
mod tests {
    use bevy::asset::io::AssetSourceId;
    use bevy::prelude::*;

    /// Reads every file in the directory back from its embedded path.
    #[test]
    fn every_game_shader_answers_at_its_embedded_path() {
        let mut app = App::new();
        app.add_plugins(bevy::asset::AssetPlugin::default());
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
            let path = format!("benilla_app/shaders/{name}");
            let read = bevy::tasks::block_on(async {
                source.reader().read(std::path::Path::new(&path)).await
            });
            assert!(
                read.is_ok(),
                "embedded://{path} does not resolve — {name} is on disk but not registered in \
                 `shaders::plugin`, or was registered from a file that is not directly under src/"
            );
        }
        assert_eq!(found, 6, "the game's shader set changed size");
    }

    /// Every `.wgsl` and `.wesl` file under `dir` of the embedded source, as the asset server would serve it.
    /// The tests' `ShaderCache` loader: bevy 0.20 composes to WGSL text, which naga parses.
    fn naga_module(
        _: &(),
        source: bevy::shader::ShaderCacheSource,
        _: &bevy::shader::ValidateShader,
    ) -> Result<naga::Module, bevy::shader::ShaderCacheError> {
        match source {
            bevy::shader::ShaderCacheSource::Wgsl(text) => naga::front::wgsl::parse_str(&text)
                .map_err(|e| {
                    bevy::shader::ShaderCacheError::CreateShaderModule(e.emit_to_string(&text))
                }),
            bevy::shader::ShaderCacheSource::SpirV(_) => {
                panic!("a composed shader is WGSL text")
            }
        }
    }

    /// An embedded shader as the asset loader makes it: WESL by its extension, else WGSL.
    fn shader_of(text: String, path: &str) -> bevy::shader::Shader {
        let path_url = format!("embedded://{path}");
        if path.ends_with(".wesl") {
            bevy::shader::Shader::from_wesl(text, path_url)
        } else {
            bevy::shader::Shader::from_wgsl(text, path_url)
        }
    }

    fn embedded_shaders(
        source: &bevy::asset::io::AssetSource,
        dir: &std::path::Path,
        out: &mut Vec<(String, String)>,
    ) {
        use bevy::tasks::futures_lite::StreamExt;
        let reader = source.reader();
        let entries: Vec<std::path::PathBuf> = bevy::tasks::block_on(async {
            match reader.read_directory(dir).await {
                Ok(stream) => stream.collect().await,
                Err(_) => Vec::new(),
            }
        });
        for path in entries {
            if bevy::tasks::block_on(reader.is_directory(&path)).unwrap_or(false) {
                embedded_shaders(source, &path, out);
            } else if path.extension().is_some_and(|e| e == "wgsl" || e == "wesl") {
                let text = bevy::tasks::block_on(async {
                    let mut bytes = Vec::new();
                    let mut r = reader.read(&path).await.ok()?;
                    r.read_to_end(&mut bytes).await.ok()?;
                    String::from_utf8(bytes).ok()
                });
                // An asset path is `/`-separated on every platform, but the in-memory reader joins
                // a child with the platform's separator (`PathBuf::join`), `\` on Windows.
                let asset_path = path
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                if let Some(text) = text {
                    out.push((asset_path, text));
                }
            }
        }
    }

    /// No gate draws a frame, so `ui_quad.wesl` is composed here as the pipeline cache composes it:
    /// bevy's `ShaderCache` over every embedded shader library, with the defs the 2D mesh pipeline
    /// gives each of the two vertex layouts it draws (the batch mesh and the minimap's tile quad).
    /// The module must validate and translate for the backend this platform renders with.
    #[test]
    fn the_ui_quad_shader_composes_and_translates() {
        use bevy::shader::{Shader, ShaderCache, ShaderDefVal};
        let app = crate::game_plugins::schedule_tests::headless_client_unrun();
        let server = app.world().resource::<AssetServer>();
        let source = server.get_source(AssetSourceId::from("embedded")).unwrap();
        let mut files = Vec::new();
        embedded_shaders(source, std::path::Path::new(""), &mut files);

        // The cache loads into naga alone: no device.
        let mut cache: ShaderCache<naga::Module, ()> = ShaderCache::new((), naga_module);
        let mut ui_quad = None;
        for (i, (path, text)) in files.into_iter().enumerate() {
            let id = AssetId::<Shader>::Uuid {
                uuid: bevy::asset::uuid::Uuid::from_u128(i as u128 + 1),
            };
            if path == "benilla_app/shaders/ui_quad.wesl" {
                ui_quad = Some(id);
            }
            cache.set_shader(id, shader_of(text, &path));
        }
        let ui_quad = ui_quad.expect("ui_quad.wesl is embedded");

        let storage =
            bevy::render::settings::WgpuLimits::default().max_storage_buffers_per_shader_stage;
        for layout in [
            ["VERTEX_POSITIONS", "VERTEX_UVS", "VERTEX_COLORS"],
            ["VERTEX_POSITIONS", "VERTEX_NORMALS", "VERTEX_UVS"],
        ] {
            let mut defs: Vec<ShaderDefVal> = layout.iter().map(|&d| d.into()).collect();
            defs.push(ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 2));
            defs.push(ShaderDefVal::UInt(
                "AVAILABLE_STORAGE_BUFFER_BINDINGS".into(),
                storage,
            ));
            let module = cache
                .get(0, ui_quad, &defs)
                .unwrap_or_else(|e| panic!("{layout:?}: ui_quad.wesl does not compose: {e:?}"));
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{layout:?}: ui_quad.wesl does not validate: {e:?}"));
            #[cfg(target_os = "macos")]
            naga::back::msl::write_string(
                &module,
                &info,
                // wgpu's Metal backend writes at the device's MSL 2.x; 1.0, the default, lacks
                // the instance index every 2D mesh shader reads.
                &naga::back::msl::Options {
                    lang_version: (2, 4),
                    ..Default::default()
                },
                &naga::back::msl::PipelineOptions::default(),
            )
            .unwrap_or_else(|e| {
                panic!("{layout:?}: ui_quad.wesl does not translate to MSL: {e:?}")
            });
            #[cfg(not(target_os = "macos"))]
            naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
                .unwrap_or_else(|e| {
                    panic!("{layout:?}: ui_quad.wesl does not translate to SPIR-V: {e:?}")
                });
        }
    }

    /// Which shader libraries a [`cata_water_cache`] holds.
    #[derive(Clone, Copy, PartialEq)]
    enum Libraries {
        /// Every library this platform's client registers.
        Here,
        /// The browser's set: the ray-query libraries out (`benilla_world::shaders`' native-only
        /// block).
        Web,
    }

    /// A `ShaderCache` over `libraries` plus the cata water (`benilla_assets`'s
    /// `liquid_cata.wesl`), and the water's id in it. Bevy 0.20's cache composes to WGSL text
    /// whatever the device; each test validates against the capabilities it names.
    fn cata_water_cache(
        libraries: Libraries,
    ) -> (
        bevy::shader::ShaderCache<naga::Module, ()>,
        AssetId<bevy::shader::Shader>,
    ) {
        embedded_cache(libraries, "benilla_assets/shaders/liquid_cata.wesl")
    }

    /// A `ShaderCache` over `libraries`, and the id in it of the embedded shader at `path`.
    fn embedded_cache(
        libraries: Libraries,
        path: &str,
    ) -> (
        bevy::shader::ShaderCache<naga::Module, ()>,
        AssetId<bevy::shader::Shader>,
    ) {
        use bevy::shader::{Shader, ShaderCache};
        let app = crate::game_plugins::schedule_tests::headless_client_unrun();
        let server = app.world().resource::<AssetServer>();
        let source = server.get_source(AssetSourceId::from("embedded")).unwrap();
        let mut files = Vec::new();
        embedded_shaders(source, std::path::Path::new(""), &mut files);
        if libraries != Libraries::Here {
            // The web registers no library that needs ray queries.
            files.retain(|(_, text)| !text.contains("enable wgpu_ray_query;"));
        }

        let mut cache: ShaderCache<naga::Module, ()> = ShaderCache::new((), naga_module);
        // The world's plugins embed the material's shader with the libraries it imports.
        let target = files
            .iter()
            .position(|(p, _)| p == path)
            .unwrap_or_else(|| panic!("{path} is embedded"));
        let target = AssetId::<Shader>::Uuid {
            uuid: bevy::asset::uuid::Uuid::from_u128(target as u128 + 1),
        };
        for (i, (path, text)) in files.into_iter().enumerate() {
            let id = AssetId::<Shader>::Uuid {
                uuid: bevy::asset::uuid::Uuid::from_u128(i as u128 + 1),
            };
            cache.set_shader(id, shader_of(text, &path));
        }
        (cache, target)
    }

    /// The material pipeline's defs for the cata water over a liquid mesh's vertex layout: plain,
    /// or a WMO interior pool's with its vertex colour.
    fn cata_water_defs(colors: bool) -> Vec<bevy::shader::ShaderDefVal> {
        use bevy::shader::ShaderDefVal;
        let storage =
            bevy::render::settings::WgpuLimits::default().max_storage_buffers_per_shader_stage;
        let mut defs: Vec<ShaderDefVal> = [
            "VERTEX_POSITIONS",
            "VERTEX_NORMALS",
            "VERTEX_UVS",
            "VERTEX_UVS_A",
            "VERTEX_UVS_B",
        ]
        .iter()
        .map(|&d| d.into())
        .collect();
        if colors {
            defs.push("VERTEX_COLORS".into());
        }
        defs.push(ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 3));
        defs.push(ShaderDefVal::UInt(
            "AVAILABLE_STORAGE_BUFFER_BINDINGS".into(),
            storage,
        ));
        defs.push(ShaderDefVal::UInt("MAX_DIRECTIONAL_LIGHTS".into(), 10));
        defs.push(ShaderDefVal::UInt("MAX_CASCADES_PER_LIGHT".into(), 4));
        defs.push(ShaderDefVal::UInt("MAX_RECT_LIGHTS".into(), 8));
        defs
    }

    /// The cata water composed as the material pipeline composes it, over the two vertex layouts
    /// the liquid meshes draw, validated and translated for this platform's backend: the variant
    /// the web and the classic path draw. Every other check that would compile it needs an install
    /// and a frame; this one needs neither.
    #[test]
    fn the_cata_water_shader_composes_and_translates() {
        let (mut cache, water) = cata_water_cache(Libraries::Here);
        for colors in [false, true] {
            let defs = cata_water_defs(colors);
            let module = cache.get(0, water, &defs).unwrap_or_else(|e| {
                panic!("colors {colors}: liquid_cata.wesl does not compose: {e:?}")
            });
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default(),
            )
            .validate(&module)
            .unwrap_or_else(|e| {
                panic!("colors {colors}: liquid_cata.wesl does not validate: {e:?}")
            });
            #[cfg(target_os = "macos")]
            naga::back::msl::write_string(
                &module,
                &info,
                &naga::back::msl::Options {
                    lang_version: (2, 4),
                    ..Default::default()
                },
                &naga::back::msl::PipelineOptions::default(),
            )
            .unwrap_or_else(|e| {
                panic!("colors {colors}: liquid_cata.wesl does not translate to MSL: {e:?}")
            });
            #[cfg(not(target_os = "macos"))]
            naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
                .unwrap_or_else(|e| {
                    panic!("colors {colors}: liquid_cata.wesl does not translate to SPIR-V: {e:?}")
                });
        }
    }

    /// The cata water composes over the browser's libraries, which hold no `rtshadow` and no
    /// `rtreflect`: it imports both under `RT_SHADOWS` / `RT_REFLECT` alone, which the web never
    /// sets, and wesl resolves no import its conditions remove. Were one resolved, the cache
    /// would answer "import not yet available", which the pipeline cache retries for ever: every
    /// M2, terrain and water pipeline on the web would stay queued and draw nothing.
    #[test]
    fn the_cata_water_shader_composes_over_the_web_libraries() {
        let (mut cache, water) = cata_water_cache(Libraries::Web);
        for colors in [false, true] {
            let module = cache
                .get(0, water, &cata_water_defs(colors))
                .unwrap_or_else(|e| {
                    panic!("colors {colors}: liquid_cata.wesl does not compose on the web: {e:?}")
                });
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default(),
            )
            .validate(&module)
            .unwrap_or_else(|e| {
                panic!("colors {colors}: liquid_cata.wesl does not validate on the web: {e:?}")
            });
        }
    }

    /// fork: modern weather — the effect lane's rain point draws (`benilla_world`'s
    /// `wow_effect.wesl`), as `EffectPipeline::specialize` builds them: `EFFECT_POINTS` under the
    /// reference's `BLEND_MOD2X` and a modern frame's `BLEND_RAIN_LIGHT`, over this platform's
    /// libraries and the browser's.
    #[test]
    fn the_rain_point_draws_compose() {
        use bevy::shader::ShaderDefVal;
        for libraries in [Libraries::Here, Libraries::Web] {
            let (mut cache, effect) =
                embedded_cache(libraries, "benilla_world/shaders/wow_effect.wesl");
            for blend in ["BLEND_MOD2X", "BLEND_RAIN_LIGHT"] {
                let defs: Vec<ShaderDefVal> = vec![blend.into(), "EFFECT_POINTS".into()];
                let module = cache
                    .get(0, effect, &defs)
                    .unwrap_or_else(|e| panic!("{blend}: wow_effect.wesl does not compose: {e:?}"));
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::default(),
                )
                .validate(&module)
                .unwrap_or_else(|e| panic!("{blend}: wow_effect.wesl does not validate: {e:?}"));
            }
        }
    }

    /// fork: F and G — the cata water's native ray-traced variant, as `benilla_assets`'
    /// `push_rt_water` builds its colour pipeline where both lanes may run: `RT_SHADOWS` and
    /// `RT_REFLECT`, the scene at `RT_GROUP` 4 (the group after the material's), composed with
    /// the ray-query and binding-array capabilities the native device has (`rt_scene.wesl`'s
    /// `binding_array` of the texture pool) and the native libraries (`rt_scene`, `rtshadow`,
    /// `rtreflect`), then validated and written as SPIR-V. Vulkan and DX12 only: neither the web
    /// nor Metal compiles it.
    #[cfg(not(any(target_arch = "wasm32", target_os = "macos")))]
    #[test]
    fn the_cata_water_shader_composes_with_the_ray_traced_lanes() {
        use bevy::shader::ShaderDefVal;
        use naga::valid::Capabilities;
        let (mut cache, water) = cata_water_cache(Libraries::Here);
        // What `wgpu_naga_bridge::features_to_naga_capabilities` makes of those features and of
        // `DownlevelFlags::all()`, which the pipeline cache validates against.
        let capabilities = Capabilities::RAY_QUERY
            | Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
            | Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING
            | Capabilities::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING
            | Capabilities::SHADER_FLOAT16_IN_FLOAT32
            | Capabilities::MULTISAMPLED_SHADING
            | Capabilities::CUBE_ARRAY_TEXTURES;
        for colors in [false, true] {
            let mut defs = cata_water_defs(colors);
            defs.push("RT_SHADOWS".into());
            defs.push("RT_REFLECT".into());
            defs.push(ShaderDefVal::UInt("RT_GROUP".into(), 4));
            let module = cache.get(0, water, &defs).unwrap_or_else(|e| {
                panic!("colors {colors}: the ray-traced liquid_cata.wesl does not compose: {e:?}")
            });
            let info =
                naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
                    .validate(&module)
                    .unwrap_or_else(|e| {
                        panic!(
                        "colors {colors}: the ray-traced liquid_cata.wesl does not validate: {e:?}"
                    )
                    });
            naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
                .unwrap_or_else(|e| {
                    panic!(
                        "colors {colors}: the ray-traced liquid_cata.wesl does not translate to \
                         SPIR-V: {e:?}"
                    )
                });
        }
    }
    /// fork: N — the modern entity shadow caster (`benilla_world`'s `modern/entity_shadow.wesl`)
    /// in every variant `EntityShadowPipelines::descriptor` builds: `SKINNED`, `CUTOUT` and
    /// `FIXTURE_CLIP` (with `CUTOUT_OR_CLIP`) in each combination, and the sun parts' `MULTIVIEW`
    /// variants (no fixture clip) under the `MULTIVIEW` capability the multiview lane requires,
    /// whose vertex stage reads `@builtin(view_index)`. Composed, validated and written as SPIR-V:
    /// the multiview lane runs on Vulkan and DX12 alone, so not on wasm32 or macOS.
    #[cfg(not(any(target_arch = "wasm32", target_os = "macos")))]
    #[test]
    fn the_entity_shadow_caster_composes_in_every_variant() {
        use bevy::shader::{Shader, ShaderCache, ShaderDefVal};
        use naga::valid::Capabilities;
        for multiview in [false, true] {
            let mut cache: ShaderCache<naga::Module, ()> = ShaderCache::new((), naga_module);
            let id = AssetId::<Shader>::Uuid {
                uuid: bevy::asset::uuid::Uuid::from_u128(1),
            };
            cache.set_shader(
                id,
                Shader::from_wesl(
                    include_str!("../../benilla-world/src/shaders/modern/entity_shadow.wesl"),
                    "embedded://benilla_world/shaders/modern/entity_shadow.wesl",
                ),
            );
            // `DownlevelFlags::all()`'s share of what the pipeline cache validates against.
            let mut capabilities = Capabilities::SHADER_FLOAT16_IN_FLOAT32
                | Capabilities::MULTISAMPLED_SHADING
                | Capabilities::CUBE_ARRAY_TEXTURES;
            if multiview {
                capabilities |= Capabilities::MULTIVIEW;
            }
            for bits in 0..8u32 {
                let (skinned, cutout, clip) = (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
                // The multiview pass draws the sun cascades' parts, never a lamp's fixture clip.
                if multiview && clip {
                    continue;
                }
                let mut defs: Vec<ShaderDefVal> = Vec::new();
                for (on, def) in [
                    (multiview, "MULTIVIEW"),
                    (skinned, "SKINNED"),
                    (cutout, "CUTOUT"),
                    (clip, "FIXTURE_CLIP"),
                    (cutout || clip, "CUTOUT_OR_CLIP"),
                ] {
                    if on {
                        defs.push(def.into());
                    }
                }
                let module = cache.get(0, id, &defs).unwrap_or_else(|e| {
                    panic!("{defs:?}: entity_shadow.wesl does not compose: {e:?}")
                });
                let info =
                    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
                        .validate(&module)
                        .unwrap_or_else(|e| {
                            panic!("{defs:?}: entity_shadow.wesl does not validate: {e:?}")
                        });
                naga::back::spv::write_vec(
                    &module,
                    &info,
                    &naga::back::spv::Options::default(),
                    None,
                )
                .unwrap_or_else(|e| {
                    panic!("{defs:?}: entity_shadow.wesl does not translate to SPIR-V: {e:?}")
                });
            }
        }
    }

    /// fork: meshlets — static_gx's mesh-shader lane (`benilla_world`'s `static_gx_mesh.wesl`), as
    /// `meshlets::queue_pipeline` builds it over each of static_gx's prepassed pipelines: the task
    /// and both mesh entries under the vertex defs, the two fragment entries under the fragment
    /// defs (the colour pass's with the ray-traced shadows at `RT_GROUP` 3 and the mesh group at
    /// 4), over every library the client registers plus `static_gx.wesl`, which the module
    /// imports. Composed, validated with the mesh-shader and draw-index capabilities the lane
    /// requires, written as SPIR-V and its task and mesh entries as HLSL: the lane runs on Vulkan
    /// and DX12, so not on wasm32 or macOS.
    #[cfg(not(any(target_arch = "wasm32", target_os = "macos")))]
    #[test]
    fn the_meshlet_lane_composes_over_static_gx() {
        use bevy::shader::{Shader, ShaderCache, ShaderDefVal};
        use naga::valid::Capabilities;
        let app = crate::game_plugins::schedule_tests::headless_client_unrun();
        let server = app.world().resource::<AssetServer>();
        let source = server.get_source(AssetSourceId::from("embedded")).unwrap();
        let mut files = Vec::new();
        embedded_shaders(source, std::path::Path::new(""), &mut files);
        // static_gx and the lane's module, and `benilla_assets`' libraries static_gx imports, each
        // put in or replaced by its source here.
        for (path, text) in [
            (
                "benilla_assets/shaders/env_frame.wesl",
                include_str!("../../benilla-assets/src/shaders/env_frame.wesl"),
            ),
            (
                "benilla_assets/shaders/wind_hook.wesl",
                include_str!("../../benilla-assets/src/shaders/wind_hook.wesl"),
            ),
            (
                "benilla_assets/shaders/wet_hook.wesl",
                include_str!("../../benilla-assets/src/shaders/wet_hook.wesl"),
            ),
            (
                "benilla_world/shaders/static_gx.wesl",
                include_str!("../../benilla-world/src/shaders/static_gx.wesl"),
            ),
            (
                "benilla_world/shaders/modern/static_gx_mesh.wesl",
                include_str!("../../benilla-world/src/shaders/modern/static_gx_mesh.wesl"),
            ),
        ] {
            files.retain(|(p, _)| p != path);
            files.push((path.into(), text.into()));
        }
        let mut cache: ShaderCache<naga::Module, ()> = ShaderCache::new((), naga_module);
        let mut mesh = None;
        for (i, (path, text)) in files.into_iter().enumerate() {
            let id = AssetId::<Shader>::Uuid {
                uuid: bevy::asset::uuid::Uuid::from_u128(i as u128 + 1),
            };
            if path == "benilla_world/shaders/modern/static_gx_mesh.wesl" {
                mesh = Some(id);
            }
            cache.set_shader(id, shader_of(text, &path));
        }
        let mesh = mesh.expect("the lane's module is in the cache");

        // What `wgpu_naga_bridge::features_to_naga_capabilities` makes of the lane's features
        // (mesh shaders, draw index, ray queries for the colour pass) and `DownlevelFlags::all()`.
        let capabilities = Capabilities::MESH_SHADER
            | Capabilities::DRAW_INDEX
            | Capabilities::IMMEDIATES
            | Capabilities::RAY_QUERY
            | Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
            | Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING
            | Capabilities::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING
            | Capabilities::SHADER_FLOAT16_IN_FLOAT32
            | Capabilities::MULTISAMPLED_SHADING
            | Capabilities::CUBE_ARRAY_TEXTURES;
        let storage =
            bevy::render::settings::WgpuLimits::default().max_storage_buffers_per_shader_stage;
        for bits in 0..8u32 {
            let (cutout, two_sided, modern) = (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
            // `static_gx::render`'s defs for a prepassed bindless frame.
            let mut defs: Vec<ShaderDefVal> = vec![
                "GX_BINDLESS".into(),
                "SHADOW_FILTER_METHOD_GAUSSIAN".into(),
                "GX_PREPASSED".into(),
                ShaderDefVal::UInt("AVAILABLE_STORAGE_BUFFER_BINDINGS".into(), storage),
                ShaderDefVal::UInt("MAX_DIRECTIONAL_LIGHTS".into(), 10),
                ShaderDefVal::UInt("MAX_CASCADES_PER_LIGHT".into(), 4),
                ShaderDefVal::UInt("MAX_RECT_LIGHTS".into(), 8),
            ];
            for (on, def) in [
                (cutout, "GX_CUTOUT"),
                (two_sided, "GX_TWO_SIDED"),
                (modern, "WOW_MODERN"),
            ] {
                if on {
                    defs.push(def.into());
                }
            }
            // The prepass (three groups, the mesh group at 3) and the colour pass, which traces
            // its shadows in a modern frame (the scene at 3, the mesh group at 4).
            let prepass = [
                defs.clone(),
                vec![ShaderDefVal::UInt("GX_MESH_GROUP".into(), 3)],
            ]
            .concat();
            let colour = if modern {
                [
                    defs.clone(),
                    vec![
                        "RT_SHADOWS".into(),
                        ShaderDefVal::UInt("RT_GROUP".into(), 3),
                        ShaderDefVal::UInt("GX_MESH_GROUP".into(), 4),
                    ],
                ]
                .concat()
            } else {
                prepass.clone()
            };
            let mesh_defs = [
                defs.clone(),
                vec![ShaderDefVal::UInt(
                    "GX_MESH_GROUP".into(),
                    if modern { 4 } else { 3 },
                )],
            ]
            .concat();
            for (what, defs, entries) in [
                (
                    "prepass",
                    &prepass,
                    &["meshlet_task", "meshlet_mesh_prepass", "prepass_fragment"][..],
                ),
                (
                    "colour mesh",
                    &mesh_defs,
                    &["meshlet_task", "meshlet_mesh"][..],
                ),
                ("colour fragment", &colour, &["fragment"][..]),
            ] {
                let module = cache.get(0, mesh, defs).unwrap_or_else(|e| {
                    panic!("{what} {defs:?}: static_gx_mesh.wesl does not compose: {e:?}")
                });
                for entry in entries {
                    assert!(
                        module.entry_points.iter().any(|e| e.name == *entry),
                        "{what} {defs:?}: `{entry}` is missing from the composed module"
                    );
                }
                let info =
                    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
                        .validate(&module)
                        .unwrap_or_else(|e| {
                            panic!("{what} {defs:?}: static_gx_mesh.wesl does not validate: {e:?}")
                        });
                // SPIR-V 1.4 at least: the mesh-shading extension asks it, and wgpu-hal writes
                // 1.6 on the Vulkan 1.3 devices the lane runs on.
                let spirv = naga::back::spv::Options {
                    lang_version: (1, 6),
                    ..Default::default()
                };
                naga::back::spv::write_vec(&module, &info, &spirv, None).unwrap_or_else(|e| {
                    panic!(
                        "{what} {defs:?}: static_gx_mesh.wesl does not translate to SPIR-V: {e:?}"
                    )
                });
                // The lane runs on DX12 too: the task and mesh entries as HLSL, the task stage's
                // `draw_index` read from the special constants (the vendored naga).
                for entry in entries.iter().filter(|e| e.starts_with("meshlet_")) {
                    let hlsl = hlsl_entry(&module, &info, entry).unwrap_or_else(|e| {
                        panic!("{what} {defs:?}: `{entry}` does not translate to HLSL: {e:?}")
                    });
                    // DXC compiles the wrapper naga writes around a task or mesh body: the group
                    // index it passes must be declared, and declared once (a second `SV_GroupIndex`
                    // input beside the input struct's is an overlapping semantic).
                    let problem = wrapper_group_index(&hlsl, entry);
                    assert!(
                        problem.is_none(),
                        "{what} {defs:?}: `{entry}`'s HLSL wrapper: {}",
                        problem.unwrap_or_default()
                    );
                    if *entry == "meshlet_task" {
                        assert!(
                            hlsl.contains("_NagaConstants.other"),
                            "{what} {defs:?}: the task stage's draw index is not the special \
                             constants' third word"
                        );
                    }
                }
            }
        }
    }

    /// What is wrong with the group index of `entry`'s HLSL wrapper (`void {entry}(…) { … }`), the
    /// function DXC compiles as the entry point: a `local_invocation_index` its body uses that its
    /// signature does not declare, or more than one `SV_GroupIndex` across its signature and the
    /// input struct it takes. `None` when neither.
    #[cfg(not(any(target_arch = "wasm32", target_os = "macos")))]
    fn wrapper_group_index(hlsl: &str, entry: &str) -> Option<String> {
        let head = format!("void {entry}(");
        let at = hlsl.find(&head)?;
        let rest = &hlsl[at..];
        let sig_end = rest.find('{').unwrap_or(rest.len());
        let (sig, body) = rest.split_at(sig_end);
        let body = &body[..body
            .find(
                "
}",
            )
            .unwrap_or(body.len())];
        // The input struct, where the signature opens with one: its type is the first word.
        let first = sig[head.len()..].split_whitespace().next().unwrap_or("");
        let in_struct = hlsl
            .find(&format!("struct {first} {{"))
            .map(|s| &hlsl[s..s + hlsl[s..].find("};").unwrap_or(0)])
            .unwrap_or("");
        let declared =
            sig.matches("SV_GroupIndex").count() + in_struct.matches("SV_GroupIndex").count();
        if declared > 1 {
            return Some(format!("{declared} `SV_GroupIndex` inputs: {sig}"));
        }
        for word in body.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
            if word.starts_with("local_invocation_index") && !sig.contains(word) {
                return Some(format!("`{word}` is used but not declared: {sig}"));
            }
        }
        None
    }

    /// One entry point of `module` as wgpu-hal's DX12 backend asks naga for it: Shader Model 6.7
    /// (the static DXC), the special constants bound (every DX12 pipeline layout has them), the
    /// immediates at a register of their own, every other binding where its group and index say.
    #[cfg(not(any(target_arch = "wasm32", target_os = "macos")))]
    fn hlsl_entry(
        module: &naga::Module,
        info: &naga::valid::ModuleInfo,
        entry: &str,
    ) -> Result<String, naga::back::hlsl::Error> {
        use naga::back::hlsl::{BindTarget, Options, PipelineOptions, ShaderModel, Writer};
        let stage = module
            .entry_points
            .iter()
            .find(|e| e.name == entry)
            .map(|e| e.stage)
            .expect("the entry point is in the module");
        let options = Options {
            shader_model: ShaderModel::V6_7,
            special_constants_binding: Some(BindTarget {
                space: 10,
                register: 0,
                ..Default::default()
            }),
            immediates_target: Some(BindTarget {
                space: 11,
                register: 0,
                ..Default::default()
            }),
            ..Default::default()
        };
        let pipeline = PipelineOptions {
            entry_point: Some((stage, entry.to_string())),
        };
        let mut hlsl = String::new();
        Writer::new(&mut hlsl, &options, &pipeline).write(module, info, None)?;
        Ok(hlsl)
    }

    /// fork: the vendored naga's `@builtin(draw_index)` in HLSL (`third_party/naga/BENILLA.md`): a
    /// vertex stage reads it as the special constants' third word and keeps its other built-ins'
    /// semantics, and a compute stage's `num_workgroups` still reads all three words.
    #[cfg(not(any(target_arch = "wasm32", target_os = "macos")))]
    #[test]
    fn draw_index_reads_the_special_constants_in_hlsl() {
        use naga::valid::Capabilities;
        let src = "enable draw_index;\n\
            @vertex fn vs(@builtin(draw_index) d: u32, @builtin(vertex_index) v: u32) \
            -> @builtin(position) vec4<f32> {\n\
                return vec4<f32>(f32(d), f32(v), 0.0, 1.0);\n\
            }\n\
            var<workgroup> w: u32;\n\
            @compute @workgroup_size(1) fn cs(@builtin(num_workgroups) n: vec3<u32>) {\n\
                w = n.x;\n\
            }\n";
        let module = naga::front::wgsl::parse_str(src).expect("the module parses");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            Capabilities::DRAW_INDEX,
        )
        .validate(&module)
        .expect("the module validates");
        let vs = hlsl_entry(&module, &info, "vs").expect("the vertex stage translates");
        assert!(vs.contains("_NagaConstants.other"), "{vs}");
        assert!(vs.contains("SV_VertexID"), "{vs}");
        let cs = hlsl_entry(&module, &info, "cs").expect("the compute stage translates");
        assert!(
            cs.contains(
                "uint3(_NagaConstants.first_vertex, _NagaConstants.first_instance, \
                 _NagaConstants.other)"
            ),
            "{cs}"
        );
    }

    /// The world's material and pass shaders no other gate composes, composed as the pipeline
    /// cache composes them, over every embedded library, with the defs of a mesh material and the
    /// water's screen pass, and validated. The headless client embeds them as the client does
    /// (`WorldPlugins`), so a shader the world's plugins leave unregistered is missing here too.
    #[test]
    fn the_world_shaders_compose_and_validate() {
        use bevy::shader::{Shader, ShaderCache, ShaderDefVal};
        let app = crate::game_plugins::schedule_tests::headless_client_unrun();
        let server = app.world().resource::<AssetServer>();
        let source = server.get_source(AssetSourceId::from("embedded")).unwrap();
        let mut files = Vec::new();
        embedded_shaders(source, std::path::Path::new(""), &mut files);
        let mut cache: ShaderCache<naga::Module, ()> = ShaderCache::new((), naga_module);
        let mut ids = std::collections::HashMap::new();
        for (i, (path, text)) in files.into_iter().enumerate() {
            let id = AssetId::<Shader>::Uuid {
                uuid: bevy::asset::uuid::Uuid::from_u128(i as u128 + 1),
            };
            cache.set_shader(id, shader_of(text, &path));
            ids.insert(path, id);
        }
        // Each with its mesh's vertex colour or not (terrain's chunks carry one), and the model
        // and terrain under the modern lighting too.
        for (path, colors, modern) in [
            ("benilla_assets/shaders/terrain.wesl", true, false),
            ("benilla_assets/shaders/terrain.wesl", true, true),
            ("benilla_assets/shaders/wow_model.wesl", false, false),
            ("benilla_assets/shaders/wow_model.wesl", true, false),
            ("benilla_assets/shaders/wow_model.wesl", false, true),
            ("benilla_assets/shaders/wdl.wesl", false, false),
            ("benilla_assets/shaders/liquid.wesl", false, false),
            ("benilla_world/shaders/celestial.wesl", false, false),
            ("benilla_world/shaders/liquid_screen.wesl", false, false),
        ] {
            let mut defs = cata_water_defs(colors);
            // What bevy's mesh pipeline gives every material it specializes.
            defs.push("MESH_PIPELINE".into());
            defs.push("VERTEX_OUTPUT_INSTANCE_INDEX".into());
            if modern {
                defs.push("WOW_MODERN".into());
            }
            defs.push(ShaderDefVal::UInt("LIQUID_REFLECT_SHIFT".into(), 2));
            let id = *ids
                .get(path)
                .unwrap_or_else(|| panic!("{path} is not embedded by the world's plugins"));
            let module = cache.get(0, id, &defs).unwrap_or_else(|e| {
                panic!("{path} (colors {colors}, modern {modern}) does not compose: {e:?}")
            });
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default(),
            )
            .validate(&module)
            .unwrap_or_else(|e| {
                panic!("{path} (colors {colors}, modern {modern}) does not validate: {e:?}")
            });
        }
    }
}
