# Modern lighting: hook points in upstream-owned files

The modern lighting model (`FORK.md`) lives in fork-owned files:
- `benilla-world/src/modern/`
- `benilla-world/src/shaders/modern/`

Every edit it needs inside a file that upstream benilla also edits is listed here. After an upstream merge, check each row is still present.

**Shader hook rule (from P1 on):** wrap upstream code, never rewrite it:

```
#ifndef LIGHTING_MODERN
<upstream code, byte-identical>
#else
<one call into benilla_modern::*>
#endif
```

## Scheduling rule: add no systems to `PostUpdate`

Bevy 0.18 scheduled `check_entities_needing_specialization::<M>` in `PostUpdate`, unordered against upstream's material writers (`model_fade`, `straddle`, `zfill`, `model_render` lazy spawns, `static_gx` cull…). Adding any system there reshuffled the single-threaded executor's order; a writer could then land after the check, and the render world panicked at `bevy_pbr` `material.rs:1061` (a missing `EntitySpecializationTicks` entry) in classic and modern alike (bisected 2026-09-15). `modern::tick_heal` inserted the missing tick.

Bevy 0.19 replaced the tick table with change lists: `specialize_material_meshes` skips an entity it has no entry for (`let … else continue`), and a writer that lands after the check is picked up by the next frame's `Changed<>` query. The panic path no longer exists, so `tick_heal` is deleted. The rule stands for the frame it would still cost: a late writer's mesh draws one frame late.

Fork systems go in `PreUpdate`, `Update` or `Last`.

## Live hooks

| File | Hook | Since |
|---|---|---|
| `benilla-world/src/lib.rs` | `pub mod modern;` | P0 |
| `benilla-world/src/world_plugins.rs` | `.add(crate::modern::ModernLightingPlugin)` after `SsilvbPlugin` | P0 |
| `benilla-world/src/shaders.rs` | `modern_libraries()`: `load_shader_library!` for `shaders/modern/*.wgsl` | P0 |
| `benilla-app/src/cvars/{mod,table}.rs` | the `lightingModel` / `lightingQuality` rows, the "fork-owned block" markers, the `lightingmodel`/`lightingquality` apply arms, the `KnobParams`/`Knobs` `lighting` field, the `GetCVar` seed rows, the `WOW_LIGHTING*` session-owned env keys | P0 |
| `benilla-world/src/lib.rs` | `pub mod launch_env;` (fork-owned module) | P0 |
| `benilla-world/src/worldview.rs` | `view_start` reads `WOW_WORLDVIEW_AT` through `crate::launch_env::var` | P0 |
| `benilla-app/src/lib.rs` | `run_worldview` wrapper (was a `pub use`) + `wants_worldview` | P0 |
| `wenilla/src/lib.rs` | `?worldview=1` switch in `start()` | P0 |
| `benilla-world/src/lighting/global_light.rs` | `build_light_data`: `active` param; row 20 `.y` = modern flag | P1 |
| `benilla-world/src/lighting/resolve.rs` | `apply_sky_backdrop`: decoded clear colour when modern | P1 |
| `benilla-world/src/lighting/mod.rs`, `lighting/resolve.rs` | `WowLighting::night_key` / `night_tint`: the midnight diffuse and celestial rows, sampled in `update_time_lighting` while the modern moon is up (`sun_arc::moonlit`) | P2 fix |
| `benilla-world/src/sky.rs` | `update_sky_colors`: sky stops + fog through `decode_if_modern` | P1 |
| `benilla-world/src/sun/follow.rs` | `disc_span` `.z` = modern flag | P1 |
| `benilla-world/src/ffx_glow.rs` | render-world `combine_uniform` call: `uniform[6]` (`wave.z`) = modern flag | P1 |
| `terrain.wgsl`, `static_gx.wgsl`, `wdl.wgsl` | `#import benilla_modern::color`; output through `color::modern_out(…, point_count.y)`; wdl's struct padded to row 20 | P1 |
| `wow_model.wgsl` | `#import`; decode block before `out.color` (additive premultiplied in gamma, then decoded; Mod/Mod2x identity, Mod2x ×2.295) | P1 |
| `wow_effect.wgsl` | `#import`; struct padded to row 20; decode before the blend `#ifdef` ladder (`BLEND_ADD` folds α in gamma first; `add_alpha` in its return) | P1 |
| `liquid.wgsl`, `liquid_cata.wgsl` | `#import`; struct row 20; `fragment` renamed `fragment_classic` + wrapper entry at EOF; cata re-encodes its scene-copy/blur/mirror reads via `color::modern_in` | P1 |
| `ffx_glow.wgsl` | `#import`; combine reads scene + blur through `color::modern_in(…, ffx.wave.z)` | P1 |
| `celestial.wgsl` | keeps Bevy's linear colour when `span.z > 0.5`; discs ×4 (HDR) in a modern frame | P1 |
| `ffx_glow.wgsl` | `fs_downsample_hdr` entry; combine exposes (`wave.w`), adds bloom above white, `modern_tonemap` shoulder, before the classic byte math | P1b |
| `ffx_glow.rs` | `FfxGlowPipelines::downsample_hdr` (+ queue + construction); node picks it and writes `uniform[7]` exposure when modern | P1b |
| `benilla-world/src/static_gx/render.rs` | `mod modern_shadow;` at EOF; `modern_shadow::build(render_app)` at the end of `build`; the light group layout gains the shadow array, compare sampler and cascade block (bindings 1-3) and `prepare_view_bind` binds the constant-handle assets | P2 |
| `benilla-assets/src/materials.rs` | `MODERN_SHADOW_MAP` / `MODERN_SHADOW_CASCADES` constant handles; `WowModelExt` and `TerrainExtension` gain `modern_shadow` (121/122) and `modern_cascades` (123) | P2 |
| `world_assets/mod.rs`, `model_render.rs` (×2), `terrain_stream.rs` | the two new fields at each material construction site | P2 |
| `terrain.wgsl` | `#import benilla_modern::sunshadow`; bindings 121-123; modern: sun visibility scales the primary's sun share, MCSH and the classic character term retire | P2 |
| `wow_model.wgsl` | same import and bindings; modern: `player_shadow` from the modern cascades, full occlusion (`shadow_term = visibility`) | P2 |
| `static_gx.wgsl` | same import; group 2 bindings 1-3; modern: `world_shadow` from the modern cascades (exterior), full occlusion | P2 |
| `benilla-app/src/player_shadow.rs` | `wanted &= !frame_is_modern()` — the creature proxy retires in a modern session | P2a |
| `benilla-world/src/shaders.rs` | `embedded_asset!` for `shaders/modern/static_gx_shadow.wgsl` | P2a |
| `benilla-world/src/dev_state.rs` | `WOW_CLOCK` read through `crate::launch_env::var` (browser `?clock=`); `benilla-app` seeds it for the worldview | P2a |
| `benilla-world/src/entity_shade.rs` | the ground-shade chain `.run_if(!frame_is_modern())` | P3 |
| `benilla-world/src/clutter.rs` | `MCSH_LIT` instead of `mcsh_tint_at` in a modern session | P3 |
| `benilla-app/src/blob_shadow.rs` | the decal hides when `frame_is_modern()` | P3 |
| `wow_model.wgsl`, `static_gx.wgsl` | `shade_t = 0` in a modern frame (baked doodad shade retires) | P3 |
| `terrain.wgsl` | modern: per-pixel Lambert sun under the cascades, point share recovered from the Gouraud primary | P3 |
| `wow_model.wgsl`, `static_gx.wgsl` | modern: exterior WMO drops its MOCV light multiplier | P3 |
| `wow_model.wgsl` | `lit_with_shadow` → `lit_with_shadow_base` + foliage transmission (double-sided + alpha-mask flags) | P3 |
| `static_gx.wgsl`, `static_gx/render.rs` | doodad lane `lit` is `var` + foliage transmission under `GX_TWO_SIDED`/`GX_CUTOUT`; `GX_TWO_SIDED` def pushed for two-sided pipelines | P3 |
| `benilla-world/src/worldview.rs` | `WOW_WORLDVIEW_FACE`/`_PITCH` via `launch_env` (browser `?worldview_face=`) | P3 |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | modern: ambient × `sunshadow::sky_ambient` (the top-down sky-occlusion map, shadow array layer 4); static_gx splits `lit_nl_shadowed` / `lit_doodad` into `_classic` + modern override; debug mode 2 shows sky visibility | P3 |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | `point_light_select` / `point_light_eval` after `point_light_sum`; a flat `point_sel` varying (terrain @8, wow_model @11, static_gx @6) set beside `point_lit`; modern fragments light points per pixel (terrain's recovered share, wow_model `point_diffuse`, static_gx `point_lit_px`) | P3 |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | binding 123 / group 2 binding 3 is `modern_block: sunshadow::ModernBlock` (was `modern_cascades: ModernCascades`); every helper call passes `modern_block.cascades` | P4 |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | modern: the sun visibility × `sunshadow::cloud_mix` (the cloud-shadow grid); the classic fog `if` gains a not-modern term (the composite fogs the frame) | P4 |
| `wow_model.wgsl` | `#import benilla_modern::froxelfog`; bindings 124/125; the modern output branch fogs non-opaque-intent batches with `froxelfog::fog_at` by fog policy | P4 |
| `liquid.wgsl`, `liquid_cata.wgsl` | `sunshadow`/`froxelfog` imports; bindings 123-125; `apply_fog` returns early in a modern frame; the `fragment` wrapper applies `fog_at` to transparent kinds | P4 |
| `wow_effect.wgsl`, `particles/render.rs` | image layout bindings 4-6 (block, volume, sampler; vertex); `VertexOutput.fog` from `fog_at` per vertex; the modern decode block applies it by fog policy and the classic scene fog is gated off; the bind cache also invalidates on the volume's view id | P4 |
| `wdl.wgsl` | `WDL_ALBEDO`; modern: the band is lit with a facet normal instead of the flat fog hull | P4 |
| `sky.rs`, `sky.wgsl` | `SkyExt.modern` / `.glow` (+ `setup_sky` zeros); modern: Mie halo + analytic disk core | P4 |
| `clouds/layer.rs`, `cloud.wgsl` | `CloudExt.modern` / `.light` at uniform 102 (+ construction zeros); modern: the silver lining (the elevation brightness moved into the bake, see the modern clouds rows) | P4 |
| `benilla-assets/src/materials.rs` | `MODERN_FOG_VOLUME`; `WowModelExt.modern_fog` (124/125); `LiquidExt` / `LiquidCataExt` gain `modern_cascades` (123) + `modern_fog` (124/125) | P4 |
| `world_assets/mod.rs`, `model_render.rs` (×2), `liquid/surface.rs` (×2) | the new fields at each construction site | P4 |
| `lighting/global_light.rs` | `build_light_data`: ambient × fill and diffuse × key from `modern::look::key_fill()` in a modern frame | P4 |
| `ffx_glow.rs` | `uniform[7]` = `modern::look::exposure()` | P4 |
| `static_gx/render.rs`, `static_gx/mod.rs` | `pub(crate) use … GxShadowLabel` (the froxel pass orders after the shadow pass) | P4 |
| `liquid/mod.rs` | `pub(crate) use reflect::LiquidReflectionCamera` and `screen::LiquidScreenLabel` (the composite runs on the mirror, before the scene copy) | P4 |
| `benilla-world/src/world_map.rs` | `map_id_from_env` reads `WOW_MAP` through `launch_env` (browser `?map=`) | P4 |
| `benilla-app/src/lib.rs` | `run_worldview` seeds `WOW_MAP` and `WOW_MODERN_ZONE` | P4 |

| `benilla-world/src/shaders.rs` | `load_shader_library!` for `shaders/modern/probegi.wgsl`; `embedded_asset!` for `static_gx_voxel`, `gi_voxel_clear`, `gi_trace`, `gi_blend` | P5 |
| `benilla-world/src/static_gx/render.rs` | `mod modern_gi;` + `pub(crate) use modern_gi::GxVoxelLabel;` at EOF; `modern_gi::build(render_app)` at the end of `build` | P5 |
| `benilla-world/src/static_gx/mod.rs` | `pub(crate) use render::GxVoxelLabel` (the probe trace orders after the voxel pass) | P5 |
| `benilla-world/src/modern/volumetrics/mod.rs` | `pub(crate) use node::FroxelLabel` (the probe trace orders before the froxels) | P5 |
| `sunshadow.wgsl` | `#import benilla_modern::probegi`; `ModernBlock` gains `gi: GiParams` + the runtime-sized `probes` array | P5 |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | `#import benilla_modern::probegi`; a fork-owned `modern_gi_ambient` helper beside the modern bindings (the eight probe reads a WGSL module cannot do for the caller); modern exterior ambient = `mix(ambient x sky map, probes, confidence)` — terrain's `primary`, wow_model's `lit_with_shadow_sky`, static_gx's `lit_nl_shadowed` and `lit_doodad`; debug modes 3 and 4 | P5 |
| `froxel_scatter.wgsl` | `#import benilla_modern::probegi`; the ambient in-scatter's sky term follows the nearest probe against the open-field reference | P5 |
| `benilla-world/src/modern/shadow.rs` | the storage block is sized with `gi::BLOCK_GI_BYTES` behind the froxel tail | P5 |

| `benilla-formats/src/models/wmo/root.rs`, `lib.rs` | `wmo_root_ambient` (`MOHD.ambColor` @ `0x1c`) + its re-export | P6 |
| `benilla-assets/src/wmo.rs` | `WmoModel.ambient`; the loader's root closure returns it beside `wmo_id` | P6 |
| `benilla-world/src/wmo_portal/{mod,interior,audit/mod}.rs` | the new field at each `WmoModel` literal (test scaffolds + the audit build) | P6 |
| `benilla-world/src/shaders.rs` | `load_shader_library!` for `shaders/modern/locallight.wgsl`; `embedded_asset!` for `light_bin` | P6 |
| `benilla-world/src/lighting/mod.rs` | `ordering_tests::ALLOWED` gains `modern/gi/mod.rs` (a `Last` reader, like `modern/volumetrics`) | P6 |
| `benilla-app/src/lib.rs` | `run_worldview`'s `SEEDED` gains `WOW_MODERN_GI`, `WOW_MODERN_GI_GAIN`, `WOW_MODERN_LIGHTS` | P6 |
| `sunshadow.wgsl` | `#import benilla_modern::locallight`; `ModernBlock` gains `lights: locallight::LightGrid` **between** `gi` and the runtime-sized `probes` (the order is what puts both writable sub-ranges on 256-byte offsets) | P6 |
| `probegi.wgsl` | `GiParams.grid` is 5 cascades (the interior sub-grid) + an `interior` row; `GiProbe` gains a `depth` row (distance moments + the interior class); `pick` takes the receiver's class; `accum` weights by an asymmetric class match and by Chebyshev visibility; `miss_radiance` | P6 |
| `gi_trace.wgsl` | the light table's `points` are bound too; a probe's class is the P3 sky map times its own last sky share; ray misses go through `miss_radiance`; hits take the nearby local lights; the distance moments are accumulated | P6 |
| `gi_blend.wgsl` | the `depth` row blends (the moments ease, the class does not) | P6 |
| `benilla-world/src/static_gx/render/modern_shadow.rs` | the cascade loop stops at `SKY_LAYER`; `local_offsets` (six cube-face caster views per shadowed light) and `draw_local_atlas` write the atlas layer behind it, keyed so a settled room redraws nothing | P6 |
| `benilla-world/src/modern/shadow.rs` | `LOCAL_SHADOW_LAYERS` / `LOCAL_LAYER` (native only); the block is sized `GI_PARAMS + BLOCK_LIGHT + BLOCK_PROBE` | P6 |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | `#import benilla_modern::locallight`; a fork-owned `modern_local_lights` helper beside `modern_gi_ambient` (the light table is the lane's own binding); the modern point term reads the binned grid where it is live; `modern_gi_ambient` takes the receiver's interior class; debug mode 5 | P6 |
| `wow_model.wgsl`, `static_gx.wgsl` | modern interiors (`gi.params.x > 0.5`, so Low keeps the bake): `lit_modern_room` from the interior probe cascade over the building's MOHD ambient; every WMO batch class takes it; MOCV rgb is dropped as lighting and MOCV.a becomes an unclamped emission on the INT class; SIDN is an emitter on every class with a daylight floor; the INT `tex·MOCV·(1+4a)` override is classic-only; an interior MODD prop reads the room instead of its `prop_probes` slot | P6 |
| `wow_model.wgsl` | `is_rig` hoisted beside `is_wmo`/`is_interior` (the modern interior light has to exclude a booth) | P6 |

| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | the modern exterior ambient (sky map / probe mix) goes through `sunshadow::shade_floor(…, ambient row)` — terrain's `ambient`, wow_model's `amb_gi`, static_gx's `lit_nl_shadowed` and `lit_doodad` | P6 fix |
| `benilla-world/src/liquid/mod.rs` | `WATER_DETAIL_SSR` (= 1, the reference's medium slot); `from_cvar` maps (0, 1.5) to it; `WaterDetail::has_planar_reflection`; `WOW_WATER_DETAIL` read through `launch_env` | P6 fix |
| `benilla-world/src/liquid/reflect.rs` | `sync_reflection_camera`: the mirror runs only when `has_planar_reflection()` | P6 fix |
| `benilla-app/assets/ui/OptionsFrame.xml` | Water Detail dropdown gains "Modern (SSR)" = `1`; the tooltip names it | P6 fix |
| `benilla-app/src/cvars/{mod,table}.rs`, `ui_script/options_tests.rs` | the `waterDetail 1` assertions (SSR tier, no mirror; three dropdown entries) | P6 fix |
| `benilla-app/src/lib.rs` | `seed_launch_env()` (the old `run_worldview` key list, plus `WOW_WATER_DETAIL`) is called from `run` as well as `run_worldview`, so `?lighting=modern` reaches the client | P6 fix |
| `benilla-app/src/cvars/{mod,table}.rs` | the `WOW_LIGHTING*` / `WOW_WATER_DETAIL` session-owned checks read `benilla_world::launch_env::var` (the browser's query) instead of `std::env` | P6 fix |
| `benilla-app/assets/ui/OptionsFrame.xml`, `ui_script/options_tests.rs` | Graphics page `RowLightingModel` (`lightingModel`, Classic/Modern) and `RowLightingQuality` (`lightingQuality`, Low…Ultra) dropdowns between Water Detail and Render Scale, their `BENILLA_TOOLTIP_LIGHTING_*` strings, and the tooltip census (82 → 84) | P6 fix |
| `benilla-assets/src/materials.rs` | `WowModelKey.modern` from `MODERN_MODEL_PIPELINES`; `specialize` pushes `WOW_MODERN` to both stages; `LiquidCataExt.modern_shadow` at 121 (texture only, no sampler) | P6 fix |
| `wow_model.wgsl` | every modern block (room light, cascades, foliage, sky/GI ambient, window emitter, per-pixel points, linear decode + froxel fog, debug views, the vertex `point_sel`) sits under `#ifdef WOW_MODERN` with a classic `#else` value: a classic pipeline compiling that code dead drew Chrome/Metal leaf-card holes black (bisected to P3 step 4's `wow_model` hunk) | P6 fix |
| `liquid_cata.wgsl` | binding 121 `modern_shadow_map`; `modern_glint` private; modern: `sunshadow::wind_swell` on the bump slope, sun visibility (`sun_visibility_load` × cloud) on the body light, glare and specular, an HDR glitter lobe added by the linear wrapper | P6 fix |
| `benilla-world/src/liquid/surface.rs` | `modern_shadow: MODERN_SHADOW_MAP` at the cata construction site | P6 fix |
| `wow_model.wgsl` | an explicit alpha-mask discard (`material.flags` MASK, `alpha_cutoff`) in the entry point before `alpha_discard`; `#ifdef WOW_ZFILL` cuts the colour lane out of the depth-prime twin | P6 fix 2 |
| `benilla-assets/src/materials.rs` | `specialize` pushes `WOW_ZFILL` to the depth-prime twin's fragment | P6 fix 2 |
| `benilla-world/src/ffx_glow.rs` | the modern decode / HDR downsample only for the `FfxState::Player` view (`FfxGlow::WORLD`); booths, UI panes and the glue scene stay classic | P6 fix 3 |
| `sunshadow.wgsl` | `ModernCascades.sky_params.w` = the water mirror's plane while it renders (`NO_MIRROR_PLANE` otherwise); `mirror_view(c, eye)`; `sun_visibility` / `cascade_visibility` / `sky_ambient` take `lite` (1 shadow tap, 1 sky tap) | P7 perf |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | `modern_gi_ambient` / `modern_local_lights` take `lite` (one probe cascade without Chebyshev; no cube shadows); every call passes `sunshadow::mirror_view(modern_block.cascades, view.world_position)` (debug views pass `false`); static_gx reads the sky map and probes ONCE for `lit_nl_shadowed` and `lit_doodad` (`modern_sky` / `modern_gi`) | P7 perf |
| `fog_composite.wgsl` | the Low tier's screen shafts skip the mirror view | P7 perf |
| `benilla-world/src/liquid/{mod,reflect}.rs` | `ReflectionPlane` re-exported; `ReflectionPlane::live_plane` | P7 perf |
| `benilla-app/src/lib.rs` | `SEEDED` gains `WOW_MODERN_MIRRORTIER` (dev A/B: every view shades the mirror tier) and `WOW_MODERN_CASTERREACH` (dev A/B: cascade caster reach, yd) | P7 perf |
| `benilla-world/src/liquid/ripple.rs` | fix (not modern-only): catch-up steps bind `RipplePipeline::settled` (prev = placement), so a two-step frame no longer reprojects the field twice and drags ripples with the camera | P7 perf |
| `benilla-world/src/terrain_stream/spawn/fx.rs` | `spawn_emissive_light_for` + `emitter_centroid` beside `spawn_lights_for`; a no-op unless `modern::emissive_lights_on()` | P8 |
| `benilla-world/src/terrain_stream/spawn/mod.rs` | the ADT-doodad and WMO-doodad placement walks each call `spawn_emissive_light_for` after `spawn_lights_for` | P8 |
| `benilla-app/src/entities/item_glow.rs` | `attach_item_glows` accumulates the glow models' emitters across the visual's slots and spawns one `WorldPointLight` child of the item root | P8 |
| `benilla-app/src/cvars/{mod,table}.rs`, `assets/ui/OptionsFrame.xml` | `lightingQuality` reaches 4; a "Cinematic" dropdown row and the tooltip paragraph for it | P8 |
| `benilla-app/src/lib.rs` | `SEEDED` gains `WOW_MODERN_EMISSIVE` (`?modern_emissive=0`) | P8 |
| `liquid_cata.wgsl` | `LiquidParams.optics` (xyz = absorption per yard, w = the experimental tier); `modern_water` (the bed through Beer–Lambert toward the DEEP row), the soft surf band, and the glare/sheen skip — all under `experimental` (above water, `optics.w`), so plain Modern keeps the reference's combine | water tier |
| `benilla-assets/src/materials.rs` | `LiquidCataExt.optics` | water tier |
| `benilla-world/src/liquid/{mod,surface}.rs` | `WATER_DETAIL_EXPERIMENTAL` (= 3); `from_cvar` maps [2.5, 3.5) to it; `WaterDetail::is_experimental`; `cata_optics`; `sync_water_tier` (`Update`) flips `optics.w` on the shared cata materials | water tier |
| `benilla-app/assets/ui/OptionsFrame.xml`, `src/cvars/{mod,table}.rs`, `ui_script/options_tests.rs` | Water Detail gains "Modern (Experimental)" = `3` (four entries) and a tooltip paragraph; the `waterDetail 3` assertions | water tier |
| `liquid_cata.wgsl` | the modern sun shadow on water is read down the unrefracted view ray at the bed (`WATER_SHADOW_BED_LIFT` short of it), handing over to the unshifted surface point as the column deepens from `WATER_SHADOW_PROBE` to twice that (the surface from below) instead of at the surface shifted by the wave slope — both modern water tiers | Forever water |
| `liquid_cata.wgsl` | the experimental tier (`waterDetail 3`) is Forever's stylised clear water and runs under either lighting model: `LiquidParams.look_a/b/c`; the water column is read before the normal; calm swell (`look_b.x` of the wind sea off the ocean, `look_b.y` over no depth); refraction proportional to the water crossed; the bed filtered toward the DEEP row's hue; a capped stylised Fresnel (`look_a.z` → `look_c.x`); the reference's mirror/trace lookup with `sky_dome(r)` in the horizon constant's place; a `look_c.z` blend with the reference combine over one shared bed read; the glitter spike scaled by `look_b.w`; a classic frame keeps the Blinn specular and decodes the scene copy for the linear combine; a multi-octave wind-ripple slope layer under the calm swell, faded on the pixel footprint like a mip level, `look_c.w` its RMS slope (`wind_ripples`) | Forever water |
| `benilla-assets/src/materials.rs` | `LiquidCataExt.look_a` / `look_b` / `look_c` | Forever water |
| `benilla-world/src/liquid/{mod,surface}.rs` | `cata_water_look` (per-kind lanes); `sync_water_look` — the `$WOW_WATER_LOOK` override, registered only when the session is launched with it | Forever water |
| `benilla-world/src/liquid/ripple.rs`, `sky.rs` | `RippleStd430.sky` (6 rows at the tail: the dome's stops + fog) packed each frame by `pack_sky_rows` from `sky::dome_stops` (factored out of `update_sky_colors`); a test pins the stamp count in both shaders | Forever water |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | the water mirror (`mirror_view`) takes none of the modern lanes — no cascades, cloud shadow, probe field, sky map, light grid or foliage transmission; terrain keeps the Gouraud primary, interiors keep the bake | Forever water |
| `benilla-world/src/modern/volumetrics/{mod,node}.rs` | `ModernFogView` is a unit marker on the world camera only: the fog composite no longer runs on the mirror (its oblique near plane hid the sky from the composite, which fogged the mirror's sky solid) | Forever water |
| `benilla-app/src/lib.rs`, `assets/ui/OptionsFrame.xml` | `SEEDED` gains `WOW_WATER_LOOK`; the Water Detail tooltip and comment describe the new tier under either lighting model | Forever water |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | `modern_cell_lights` looks a shadowed light up from its slot's anchor (`modern_block.light_shadow[slot].xyz`, the middle of the light's sway, where its cube was rendered) instead of the table's live position | lamp shadows |
| `benilla-world/src/static_gx/render/modern_shadow.rs` | the lamp atlas is two phases per slot: static faces drawn once into the static cache after `ATLAS_SETTLE_FRAMES` of stillness (`lights::publish_built` tells the main world), and a pruned dynamic phase (entity casters within `lights::DYNAMIC_REACH`, the light's host excluded) restored and redrawn only when `same_casters` fails; the cache is a render attachment; `EntityDrawCmd` carries `tight` and `pose` | lamp shadows |
| `benilla-world/src/modern/shadow.rs` | `EntityCaster::tight` (the part's bound without the rig margin) | lamp shadows |
| `benilla-world/src/lighting/global_light.rs` | `MODERN_LOCAL_GAIN` (1.3) on every packed local light in a modern frame | lamp shadows |
| `benilla-world/src/static_gx/render/modern_shadow.rs` | the lamp atlas's dynamic phase restores and redraws per FACE (`dirty_faces` / `settle_drawn`), and `DYN_MOVE_YD`/`DYN_TURN` drop to float noise (0.02) so a walking body's lamp shadow keeps up | pkuzic port |
| `benilla-world/src/lighting/{flicker.rs,global_light.rs,mod.rs}` | `FlameFlicker` folded into the packed colour in `build_light_data` (intensity + blue), `$WOW_FIRE_FLICKER`; `SpellFxLight` packed only when the modern grid is on | pkuzic port |
| `benilla-world/src/terrain_stream/spawn/fx.rs` | `spawn_point_light` takes the light's `FlameFlicker` (authored warm lights, emitter-derived flames) | pkuzic port |
| `benilla-app/src/entities/{carried_light,wmo_props}.rs`, `attach/mod.rs` | carried, transport and GameObject-emissive lights carry `FlameFlicker` | pkuzic port |
| `benilla-formats/src/spell_light.rs`, `benilla-app/src/entities/spell_fx/light.rs` | spell lights: the school rules, and `SpellLight`'s envelope/budget spawned from `spell_fx` (Kit), `missile.rs` (Missile) and `dest_fx.rs` (Burst); registered beside `advance_fx_anim` in `entities.rs` | pkuzic port |
| `benilla-app/src/lib.rs` | `SEEDED` gains `WOW_FIRE_FLICKER` (`?fire_flicker=`) and `WOW_SPELL_LIGHT` (`?spell_light=0`) | pkuzic port |
| `benilla-world/src/wmo_portal/fog.rs` | `CameraRoomExtent` (the room the camera stands in, as a world sphere), re-exported from `wmo_portal/mod.rs` | P4 fix |
| `benilla-world/src/wmo_portal/mod.rs` | `compute_wmo_pvs` writes `CameraRoomExtent` beside `CameraWmoFog`: the sphere of `fog_room` (the true interior whose MFOG engaged), `None` wherever the fog target is — never an exterior-lit courtyard's | P4 fix |
| `benilla-world/src/lighting/{global_light,prop_probes,mod}.rs` | `LightStd430.env` (12 rows after the point table, packed in `build_light_data` from `EnvFrame`); the rain-shelter region (65,568 B) between the per-frame blob and the prop probes, so `prop_probe_region_offset` and every region behind it move | Everwood port |
| `wow_model.wgsl`, `terrain.wgsl`, `static_gx.wgsl`, `fsr/velocity.wgsl`, `modern/entity_shadow.wgsl` | the light mirrors declare `env` and `shelter_hdr` / `shelter_cfg` / `shelter` after `points` (the last two as unread padding) | Everwood port |
| `benilla-assets/src/materials.rs` | `register_shaders` loads the `benilla::env_frame` / `wind_hook` / `wet_hook` libraries | Everwood port |
| `benilla-world/src/clutter.rs` | `build_chunk_clutter` writes `ATTRIBUTE_UV_1` = (height over the model's span, per-tuft phase) | Everwood port: wind |
| `wow_model.wgsl` | `vertex`: `p_cam` is `var` and takes `wind_hook::grass_offset` (clutter, under `VERTEX_UVS_B`) and `tree_offset` (material marker bit 14) | Everwood port: wind |
| `static_gx.wgsl`, `static_gx/{mod,bake}.rs` | `WORD_FOLIAGE_WIND` (bit 29) from `foliage_wind_batch`; `gx_clip_position` takes the anchor and adds `gx_wind_offset`; `GxPrepassVertex` gains `anchor` | Everwood port: wind |
| `benilla-world/src/model_render.rs`, `terrain_stream/spawn/assemble.rs` | `FOLIAGE_WIND_MARKER` (bit 14), `foliage_model_material`; a leaf batch's fade seed carries the marked pair | Everwood port: wind |
| `benilla-world/src/world_plugins.rs`, `lib.rs` | `pub mod wind;`, `.add(crate::wind::WindPlugin)` after `WeatherPlugin` | Everwood port: wind |
| `benilla-app/src/cvars/{mod,table,presets}.rs` | the `graphicsQuality` row; `set` / `set_from_vm` apply a rung's column; `sync_cvars` mirrors the derived label; `load_config` re-applies a saved rung to rows the file lacks; `GRAPHICS_CVARS` gains the row and `reforgedModels` / `lavaLight` | Everwood port: presets |
| `benilla-app/assets/ui/OptionsFrame.xml` | `RowGraphicsPreset` heads the Graphics page (`BenillaOptionsPresetRow_OnUpdate`, `BenillaOptionsPreset_Settle`); a `benillaPreset` row heads Turtle's first video category, committed first and followed by `BenillaPresetWatch` | Everwood port: presets |
| `sky.wgsl` | a modern frame replaces the piecewise-linear ring mix with a monotone cubic (`smooth_gradient`) through the same six stops; classic keeps the linear rings | Everwood port: sky |
| `benilla-world/src/lighting/mod.rs`, `modern/lights/mod.rs` | `mod lava_light;` and its `register`; `update_lights` refuses a `LavaLight` a cube shadow, beside `SpellFxLight` | Everwood port: lava |
| `benilla-app/src/cvars/{mod,table}.rs`, `video.rs` | the `lavaLight` row and its `on_cvar` arm | Everwood port: lava |
| `benilla-world/src/weather/mod.rs` | `pub mod wetness; pub mod shelter;` and their `register` calls in `WeatherPlugin` | Everwood port: rain |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | the wet hooks before the fog: the shelter grid's open share, `wet_hook::wet_surface` (and `wet_puddles` on terrain) on the colour, `wet_sheen` added and clamped | Everwood port: rain |
| `benilla-formats/src/terrain.rs`, `benilla-assets/src/adt.rs`, `terrain.wgsl` | `ChunkMesh::water_depths`; `chunks_to_mesh` adds `pack_water_depth` to `UV1.x` (the alpha index's fraction); the wet hook multiplies the open share by `wet_hook::above_water` | Everwood port: rain |
| `benilla-app/src/cvars/{mod,table}.rs`, `video.rs`, `assets/ui/OptionsFrame.xml` | the `rainSurfaces` row, its `on_cvar` arm, `GRAPHICS_CVARS`, the option rows in both windows | Everwood port: rain |
| `benilla-app/src/cvars/{mod,table}.rs`, `video.rs`, `pipe_warm/menagerie.rs`, `assets/ui/OptionsFrame.xml` | the `foliageWind` / `foliageWindStrength` rows, their `on_cvar` arms, `GRAPHICS_CVARS`, the option rows in both windows; the clutter pipelines warmed on the UV_1 layout | Everwood port: wind |
| `benilla-world/src/weather/precip/mist.rs` | `REFERENCE_PUFF`; `push_mist` takes a `modern::weather::PuffLook` (half-extent, distance fade, life ramps, alpha cap) in place of its constants | modern weather |
| `benilla-world/src/weather/precip/mod.rs` | `push_precip` passes `modern::weather::puff_look()` / `puff_color` in a modern frame, `REFERENCE_PUFF` and the fog colour otherwise | modern weather |
| `clouds/layer.rs` | `CloudExt.shape` / `.key` / `.haze` at uniform 102 (+ construction zeros); `CloudLayer` is `pub(crate)` | modern clouds |
| `clouds/mod.rs` | `pub(crate) use layer::CloudLayer`; `CloudCoverage::tile`; `tick_clouds` returns before the upload in a modern frame (the image is `modern::cloud_bake`'s 512² layer) and restores the tile's size and format on the first classic frame | modern clouds |
| `cloud.wgsl` | the modern branch: a hole returns early; the baked layer gets the silver lining (its thinness from the alpha against `shape.x`), the horizon haze's colour low in the sky, and the horizon fade | modern clouds |
| `benilla-world/src/shaders.rs` | `modern_libraries`: `$WOW_MODERN_ARRAYINDEX=1` adds `MODERN_ARRAY_INDEX` to the `sunshadow` and `probegi` libraries (the computed index into a by-value array, as before) | perf |
| `wow_model.wgsl`, `terrain.wgsl`, `static_gx.wgsl` | modern vertex stages: one walk of the light table at most, none where the light grid answers the fragment (`point_light_gouraud` in terrain; `GxVsOut.anchor` @7 and the mirror's selection in static_gx's fragment); `MODERN_VERTEX_WALKS` restores both walks | perf |
| `benilla-assets/src/materials.rs`, `static_gx/render.rs` | `modern_vertex_walks()` (`$WOW_MODERN_VERTEXWALKS=1`) pushes `MODERN_VERTEX_WALKS` beside `WOW_MODERN` | perf |
| the shared light buffer (`lighting::global_light`) | row 20 `.z` / `.w` are the modern vertex hints (`modern::shadow::VERTEX_HINT_OFFSET`): `build_light_data` still packs 0, `modern::lights`' `prepare_lights` writes them in the render world after `upload_light` | perf |

## Planned hooks (not yet landed)

- `wow_model.wgsl`, `terrain.wgsl`, `static_gx.wgsl`: the lighting and fog spans (P1–P4)
- `wdl`, `liquid`, `liquid_cata`, `sky`, `cloud`, `celestial`, `star`, `wow_effect` `.wgsl`: one output/fog region each (P1, P4)
- `static_gx/render.rs`: `pub(crate)` GPU cache types, plus an early return in `StaticGxNode` when modern is active (P1/P2)
- `ssilvb.wgsl` / `ssilvb.rs`: `SSILVB_LINEAR` composite branch (P1)
- `particles/render.rs`: a `modern` bit in `EffectPipelineKey` (P1)
- `benilla-app/src/player_shadow.rs`, `blob_shadow.rs`: classic-only gate (P2)
- `benilla-app/assets/ui/OptionsFrame.xml`: one lighting row block (P7)
- `ffx_glow.rs`: `FfxGlow`'s state and `ffx_glow_pass` read by `modern/meter.rs`, the auto-exposure's screen meter, which runs before the combine
