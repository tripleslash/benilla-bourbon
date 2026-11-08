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
| `wow/src/lib.rs` | `?worldview=1` switch in `start()` | P0 |
| `benilla-world/src/lighting/global_light.rs` | `build_light_data`: `active` param; row 20 `.y` = modern flag | P1 |
| `benilla-world/src/lighting/resolve.rs` | `apply_sky_backdrop`: decoded clear colour when modern | P1 |
| `benilla-world/src/lighting/mod.rs`, `lighting/resolve.rs` | `WowLighting::night_key` / `night_tint`: the midnight diffuse and celestial rows, sampled in `update_time_lighting` while the modern moon is up (`sun_arc::moonlit`) | P2 fix |
| `benilla-world/src/sky.rs` | `update_sky_colors`: sky stops + fog through `decode_if_modern` | P1 |
| `benilla-world/src/sun/follow.rs` | `disc_span` `.z` = modern flag | P1 |
| `benilla-world/src/sun/follow.rs` | `follow_moons`: moon02's quad at size 0 in a modern frame | P4 fix |
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
| `liquid_cata.wgsl` | `LiquidParams.look_d`; `look_b.w` is the highlight's GGX roughness (was the glitter spike's gain); `specular_aa`, `disk_highlight` (GGX + height-correlated Smith + Schlick at `WATER_F0`, lit by a disk through Karis's representative point); the wind ripples' unit layer is read once on the sea and shared, `look_c.w` on its calm normal and `look_d.z` on its highlight normal; the experimental tier's glint is `disk_highlight` on that normal (`spec_n`: each kind's own octaves — bump, wind sea at the calm's `look_b.x` share off the sea, the sea's ripples — without the shallow calm, × `look_d.y`), the plain Modern tier keeps the three Phong lobes | Forever water: PBR highlight |
| `benilla-assets/src/materials.rs`, `benilla-world/src/liquid/surface.rs` | `LiquidCataExt.look_d`; `cata_water_look` returns four lanes; `sync_water_look` takes sixteen values | Forever water: PBR highlight |
| `benilla-world/src/liquid/screen.rs` | `REFLECT_SHIFT` 1 → 2 (a `Deviation:` comment); `LiquidScreenTargets.reflect_blur` (allocated, retired and re-pointed with the rest; `reflect_map` binds it); `ScreenVariant.reflect_blur`; the `LIQUID_REFLECT_SHIFT` define; a fourth pass in `liquid_screen_pass` blurs `reflect` into `reflect_blur`; header table and note | Forever water: soft mirror |
| `benilla-world/src/shaders/liquid_screen.wgsl` | `fs_blur_reflect`: `fs_blur`'s four taps at three FRAME pixels out | Forever water: soft mirror |
| `benilla-world/src/liquid/{surface,reflect}.rs`, `benilla-assets/src/materials.rs` | the cata material's `reflect_map` takes `targets.reflect_blur`; the reflect test's `LiquidScreenTargets` literal gains `reflect_blur`; the `reflect_map` doc | Forever water: soft mirror |
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
| `benilla-world/src/static_gx/render.rs` | `mod bindless;` at EOF; `enum GxRegionBinds { Own{..}, Shared(GxSlots) }` replaces `GxCellGpu`'s `bind_groups` / `record_table` / `cell_uniform` / `indices`; `GxPipelines::bindless` + `layout_defs()`; `init_pipelines` takes `Option<Res<HwCaps>>`, picks `cell_layout` and inserts `GxBindless`; `prepare_static_gx` takes `Option<ResMut<GxBindless>>` (evict on retain and replace, `reset` on the map clear, `refresh_bind` before the copies submit, `GxCellGpu::write_records` in the kill and fog syncs); `assemble_region` takes `shared`, packs the class into record x's high half bindless, and its per-region half moved to `own_binds` | bindless |
| `benilla-world/src/static_gx/render.rs` | the colour and prepass pipelines start their defs from `pipes.layout_defs()` (the shadow casters and the voxelizer, fork-owned, do the same) | bindless |
| `benilla-world/src/static_gx/pool.rs` | `GxTexturePool::class_count()` | bindless |
| `benilla-world/src/shaders.rs` | `modern_libraries`: `load_shader_library!` for `shaders/modern/gx_bindless.wgsl` | bindless |
| `static_gx.wgsl` | `#ifdef GX_BINDLESS` `#import benilla_modern::gx_bindless`; group 1 under `#ifdef GX_BINDLESS` / `#else`: bindless, a `var<private> cell` filled by `gx_enter(row)`; both sides define `gx_rec(word)`, and every `recs[word & 0xffffu]` read is `gx_rec(word)`; `GxVertex` / `GxPrepassVertex` gain `@builtin(instance_index)` and `GxVsOut` @8 / `GxPrepassOut` @3 a flat `region`, entered first in `vertex`, `prepass_vertex`, `fragment` and `prepass_fragment`; `gx_sample` has a bindless twin | bindless |
| `benilla-world/src/static_gx/render.rs` | `pub(crate) mod rt_scene;` after `modern_gi` and `build` calling `rt_scene::build`, both `cfg(not(target_arch = "wasm32"))`; `prepare_static_gx` calls `rt_scene::assemble_unseen` after the visible regions and before `pool.drain_pending`, in a modern frame with `rt_scene::enabled()` (this frame's: capable and the `rayTracing` option on) | rt scene |
| `benilla-world/src/static_gx/mod.rs` | `pub(crate) use render::rt_scene;` (native only) | rt scene |
| `benilla-world/src/shaders.rs` | `modern_libraries`: native only, `load_shader_library!` for `shaders/modern/rt_scene.wgsl` and `embedded_asset!` for `shaders/modern/rt_repack.wgsl` | rt scene |
| `benilla-world/src/static_gx/render.rs` | `static_gx_pass`: the `ViewQuery` takes `&ExtractedView`; before `batch.finish()` `gpu_cull::GxCullPass::new` (only with the prepass pipelines) and `batch.set_culled`, after it `bind`; the prepass job records `cull.early` before its pass and, after it, `cull.pyramid`, `cull.late` and a `static_gx_prepass_late` pass drawing `batch.emit_late` (depth loaded). `pub(crate) mod gpu_cull;` at EOF and `gpu_cull::build(app)` at the end of `build`. All `cfg(not(target_arch = "wasm32"))` | gpu cull |
| `benilla-world/src/static_gx/render.rs` | `init_pipelines`: `GxBindless::new(caps)` (it reads `ray_query`, `batch::gpu_cull_lane` and the native allocation floors itself); `assemble_region` passes `&draw.groups` to `place` (the key bound rows) | gpu cull |
| `benilla-world/src/static_gx/mod.rs` | `pub(crate) use render::gpu_cull;` (native only) | gpu cull |
| `benilla-world/src/shaders.rs` | `modern_libraries`: native only, `embedded_asset!` for `shaders/modern/gx_hiz.wgsl` and `shaders/modern/gx_cull.wgsl` | gpu cull |
| `benilla-world/src/shaders.rs` | `modern_libraries`: native only, `embedded_asset!` for `shaders/modern/gi_trace_rt.wgsl` | rt gi |
| `benilla-world/src/static_gx/render/modern_gi.rs` (fork-owned, under upstream's `static_gx/render`) | `prepare_gi_voxels`: in a frame `modern::gi::rt_gi()` holds (per frame) no clipmap texture, caster diff, box or voxel raster pipeline (so `gi_voxel_pass` records no `modern_gi_voxel_clear` / `modern_gi_voxelize`); a standing clipmap is freed (textures, depth, store bind group, caster record) the frame it goes live, and the first frame off it is made fresh and refilled, its raster pipelines queued on first use; the heightfield is drawn as before; `ModernGiScene.albedo` / `.normal` are `None` | rt gi |
| `benilla-app/src/video.rs` | `with_render_api`'s `disabled_features` is `gpu_stats::disabled_features()` alone: the device keeps the hardware ray-tracing features whatever `rayTracing` says (the option switches at run time); `VideoKnobs`' fork tuple gains `Option<ResMut<benilla_world::modern::RayTracingSetting>>` (`fork.9`) and the `raytracing` arm writes it | ray tracing setting |
| `benilla-app/src/cvars/table.rs` | `ours("rayTracing", "1", …)` in its own fork-owned block after dlss's; its comment: applies at run time | ray tracing setting |
| `benilla-app/assets/ui/OptionsFrame.xml` | `BENILLA_TOOLTIP_RAY_TRACING`; `$parentRowRayTracing` (checkbox, hidden, shown under `BENILLA_NATIVE`) after Auto Exposure, inserted into `BENILLA_OPTIONS_PAGE_ROWS.Graphics` after `RowAutoExposure` on native, with `RowWeatherDensity` re-anchored under it; Turtle's window: `BENILLA_RAY_TRACING` label, a native-only `PIXEL_SHADERS` checkbutton row on `rayTracing`, last in the Lighting section. The tooltip says no restart is needed (the Graphics page still commits on Apply, as every row of it) | ray tracing setting |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | `#ifdef RT_SHADOWS` `#import benilla_modern::rtshadow` after `locallight`; at every modern `sunshadow::sun_visibility` site (static_gx's exterior and doorway sites) an `#ifdef RT_SHADOWS` twin calling `rtshadow::sun_visibility(…, to_sun)` (`-normalize(wow_light.light_sun.xyz)`, terrain's `ml`), and in `modern_cell_lights` an `#ifdef RT_SHADOWS` twin calling `rtshadow::local_visibility` with the light's live position and its fixture box (`light_spare` rows 9-11); the original calls byte for byte under `#else` | rt shadows |
| `liquid_cata.wgsl` | `#ifdef RT_SHADOWS` `#import benilla_modern::rtshadow` after `skydome`; the surface's and the bed probe's `sun_visibility_load` each times `rtshadow::sun_visibility_ray` under `#ifdef RT_SHADOWS`, the original under `#else` | rt shadows |
| `benilla-assets/src/materials.rs` | `RT_SHADOW_LAYOUT` (a `OnceLock<BindGroupLayoutDescriptor>`, set where `modern::shadow::rt_shadows_capable()`, for the whole session) and `push_rt_shadows`, called at the end of `WowModelExt::specialize`'s and `TerrainExtension::specialize`'s modern blocks and in `LiquidCataExt::specialize`: the scene's layout appended after the material's group, `RT_SHADOWS` and `RT_GROUP` on the fragment stage, never on a prepass pipeline | rt shadows |
| `benilla-world/src/static_gx/render.rs` | `GxPipelines::rt`, `rt_shadow_layout` (where `modern::shadow::rt_shadows_capable()`), `RT_SHADOW_GROUP` (3); `prepare_static_gx` appends the scene's layout to the colour pipelines' layout and `RT_SHADOWS` / `RT_GROUP` to their fragment defs only (`vertex_defs` keeps the vertex stage's); `static_gx_pass` binds `RtScene::lane_bind(rt_shadows())` at group 3 in the colour job (native only), and returns when `rt` holds but neither the scene nor the stand-in exists | rt shadows |
| `benilla-world/src/static_gx/render/modern_shadow.rs` (fork-owned, under upstream's `static_gx/render`) | `gx_shadow_pass`: with `modern::shadow::rt_shadows()` a cascade's static pass draws no terrain or static_gx caster (it only clears) and its `layer_key` takes no caster and hash 0; the sky map keeps its static casters; `draw_local_atlas` gets no caster and hash 0 and settles a slot after one frame. `prepare_modern_shadow`: on a frame `rt_shadows()` differs from the last (`ModernShadowFrame::rt_was`) every `cache_keys` entry is cleared (the cascades and the sky map redraw from their casters that frame) and `hurry_until` lets a lamp slot draw its static faces after one frame for `ATLAS_SETTLE_FRAMES` | rt shadows |
| `benilla-world/src/shaders.rs` | `modern_libraries`: native only, `load_shader_library!` for `shaders/modern/rt_shadow.wgsl` | rt shadows |
| `third_party/bevy_pbr/src/material.rs` (vendored, its `BENILLA.md` fourth delta) | `DrawMaterial` gains `SetMaterialExtraBindGroup<{ MATERIAL_BIND_GROUP_INDEX + 1 }>` before `DrawMesh`; `MaterialExtraBindGroup` (render-world resource) and the command are new. Re-apply on a bevy_pbr bump | rt shadows |
| `benilla-assets/src/materials.rs` | `RT_REFLECT_LAYOUT` (set where `modern::rt_reflect::rt_reflections_capable()`) and `push_rt_water`, which `LiquidCataExt::specialize` calls in place of `push_rt_shadows`: the scene's layout appended once where either layout is set, `RT_SHADOWS` and `RT_REFLECT` each pushed only where its own is, `RT_GROUP` naming the group; the `anim` field doc (`x` the lane's flag) | rt reflect |
| `liquid_cata.wgsl` | `#ifdef RT_REFLECT` `#import benilla_modern::rtreflect` and `probegi` after the `rtshadow` import; under `#ifdef RT_REFLECT` the `LANE_*` constants, `LaneMarch`, `LaneReflection`, `lane_march`, `lane_shade`, `lane_reflection` after `mirror_read_rough`; after the mirror read an `#ifdef RT_REFLECT` block that sets `base` from `lane_reflection` where `w.anim.x > 0.5`, and `#ifdef RT_REFLECT` / `#else` twins of the experimental tier's `ssr_share` and `x.refl_fogged` lines (`max(mirror_valid * w.path.z, lane_on / lane_fogged)`); `LiquidParams.anim` comment (`x` the lane) | rt reflect |
| `liquid_cata.wgsl` | inside F's `#ifdef RT_SHADOWS` blocks, the surface's and the bed's `rtshadow::sun_visibility_ray` run only where `modern_block.light_spare[12].x` (the shadows' own lane) holds | rt reflect |
| `benilla-world/src/liquid/reflect.rs` | `sync_reflection_camera` gains a native-only `rt_lane: Option<Res<RtReflectLane>>` and, after `wanted`, `wanted && !rt_lane.on` (native only); `sync_reflection_flag` watches `(live, plane, lane)` and writes `anim.x = RtReflectLane::shader_flag()` beside `path.z` / `path.w` (0 on wasm); `register` adds `rt_reflect::decide_water_lane` (native only) `.after(SubmersionVerdict).after(WorldStage::Present).before(sync_reflection_camera)`; the test's `LiquidScreenTargets` literal gains `depth_pyramid` | rt reflect |
| `benilla-world/src/liquid/screen.rs` | `LiquidScreenTargets::depth_pyramid` and `depth_pyramid()`; `allocate` gives `depth` a full mip chain and `STORAGE_BINDING` where it holds; `resize_targets` also republishes when it changes; `liquid_screen_pass` draws the depth copy through a level-0 view when it has levels | rt reflect |
| `benilla-world/src/shaders.rs` | `modern_libraries`: native only, `load_shader_library!` for `shaders/modern/rt_reflect.wgsl` | rt reflect |
| `Cargo.toml` (workspace root) | `[patch.crates-io] naga_oil` → `third_party/naga_oil` (vendored, its `BENILLA.md`: a composed mesh or task entry point's output and payload handles are remapped into the derived module). Re-apply on a naga_oil bump | meshlets |
| `third_party/bevy_render/src/render_resource/pipeline_cache.rs` (vendored, its `BENILLA.md` sixteenth delta) | `PipelineCache::queue_mesh_pipeline(descriptor, task_entry)` and `start_create_mesh_pipeline`, native only: a `RenderPipelineDescriptor` whose vertex stage stands for the mesh stage (no buffers), built with `create_mesh_pipeline`, read back with `get_render_pipeline`. Re-apply on a bevy_render bump | meshlets |
| `third_party/bevy_pbr/src/render/mesh_view_bindings.rs` (vendored, its `BENILLA.md` fifth delta) | `MeshPipelineViewLayoutParams::view_stages`: binding 0 (the view uniform) is `VERTEX_FRAGMENT \| TASK \| MESH` where the device has `EXPERIMENTAL_MESH_SHADER`. Re-apply on a bevy_pbr bump | meshlets |
| `benilla-world/src/static_gx/mod.rs` | `pub(crate) mod meshlet;` (native only) | meshlets |
| `benilla-world/src/static_gx/bake.rs` | `BakedCell::meshlets` (native only), built in `bake_cell` from the recentred positions and indices where `meshlet::wanted()` (one `ItemSpan` per draw: its index range, two-sided, foliage wind), handed on by `publish` | meshlets |
| `benilla-world/src/static_gx/render.rs` | `GxCellDraw::meshlets` and `GxCellGpu::meshlet_starts`; `init_pipelines`: the light layout's stages and `bindless::layout(mesh)` gain `TASK \| MESH` where `bindless::mesh_capable`; `GxPipelines::{mesh, mesh_prepass}`; `prepare_static_gx`: the prepass and colour descriptors bound to a variable first, their mesh twins queued through `meshlets::queue_pipeline` in a prepassed frame where `meshlets::lane_active()`; `assemble_region`: `slots` mutable and `meshlets::place` right after `place`; `static_gx_pass`: `meshlets::GxMeshletPass::new` after `cull.bind`, `begin` after `cull.early`, the early prepass, the colour pass and the late prepass drawing through `GxMeshletPass::emit` (EARLY, COLOUR, LATE; the mesh group at 3, at 4 in a ray-traced colour pass) in place of `batch.emit` / `emit_late` where it is `Some`, `end` after the late prepass; `mod meshlets;` at EOF and `meshlets::build(app)` at the end of `build`. The new fields and calls are native only | meshlets |
| `static_gx.wgsl` | the headers of `vertex` and `prepass_vertex` under `#ifdef GX_MESHLETS` (`fn gx_vertex` / `fn gx_prepass_vertex`, no `@vertex`) / `#else` (the originals) / `#endif`; the bodies untouched | meshlets |
| `benilla-world/src/shaders.rs` | `modern_libraries`: native only, `load_shader_library!` for `shaders/modern/gx_occlusion.wgsl` | meshlets |

## Interfaces for Hi-Z / RT (bindless static_gx)

Where `bindless::enabled` holds (`HwCaps.binding_arrays && mdi_count`, not `WOW_GX_BINDLESS=0`), every retained region lives in three shared GPU objects a GPU cull or a ray-traced lane can read without touching the CPU path. The contracts are the module docs of `static_gx/render/bindless.rs` and `static_gx/render/batch.rs`; this is their summary.

**The region table** (`GxBindless`, group 1 binding 0, `array<vec4<u32>>`, `STORAGE | COPY_DST | COPY_SRC`). A region's block starts at its `row`, which every draw passes as `first_instance`:

| row | x | y | z | w |
|---|---|---|---|---|
| `row + 0` | origin.x (f32 bits) | origin.y | origin.z | item count |
| `row + 1` | mesh-local bound centre.x (f32 bits) | centre.y | centre.z | `index_base` (first index in the arena) |
| `row + 2` | bound half-extent.x (f32 bits) | half.y | half.z | `index_len` |
| `row + 3 + i` | item `i`'s record: layer \| class << 16 | batch-order nudge | SIDN rgb | flags (bit 0 kill, 1..=13 probe slot, 14 interior fog) |
| `row + 3 + items + 2k` | selection key `k`'s mesh-local bound centre (f32 bits) | | | 0 |
| `row + 4 + items + 2k` | key `k`'s half-extent (f32 bits) | | | 0 |

World = mesh-local + origin; a region's world bound is `origin + centre ± half`. The key rows exist only where the GPU cull runs (`GxSlots::keys` pairs, keys `0..=` the region's largest WMO group or prop referrer-set index; a key without its own box takes the region's; cells have none). `items` is row 0's w.

**The bind group** (`static_gx_bindless_layout`): 0 the table (vertex, fragment); 1 `binding_array<texture_2d_array<f32>, 1024>`, pool class `c` at element `c` (fragment); 2 the repeat sampler; 3 the clamp sampler. No uniform, no dynamic offset. WGSL side: `benilla_modern::gx_bindless` (`enter`, `origin`, `rec`, `sample_grad`, `sample_level`, `dims`).

**The index arena** (one `u32` index buffer, `INDEX | STORAGE | COPY_DST | COPY_SRC`, `+ BLAS_INPUT` where `HwCaps.ray_query`). A region's indices are `arena[index_base..][..index_len]`, into the mesh allocator's vertex slab at the region's slab offset (the draw's `base_vertex`). A BLAS per region takes that range and that slab range; blocks keep their offsets when the arena grows (growth is a GPU copy), and move only when a region re-assembles or leaves.

**The buckets** (`GxIndirect`, rebuilt each frame by `GxBatch::finish`). Two buffers, both `INDIRECT | STORAGE | COPY_DST`: args, 20-byte `DrawIndexedIndirectArgs`, and counts, one `u32` per bucket. A bucket is one `(phase, pipeline, vertex slab)` state of one pass, drawn as one `multi_draw_indexed_indirect_count(args, args_at * 20, counts, count_at * 4, capacity)`; its `capacity` args sit contiguously from `args_at` in `(row, first index)` order, each `index_count`, `instance_count = 1`, `first_index` (`index_base` added), `base_vertex` (slab offset), `first_instance = row`. The CPU writes the count as `capacity`.

**The in-place cull recipe.** Before the pass, per bucket: zero `counts[count_at]`; per candidate draw, read its region's bound from table rows `row + 1` / `row + 2` (its `first_instance`), test it (frustum, Hi-Z), and for a survivor `let slot = atomicAdd(&counts[count_at], 1u)` and write its args at `args[args_at + slot]`. The draw call already reads the count, so the CPU side does not change. The candidates are the CPU-written args, read from a copy or a second range before the cull overwrites them. `gpu_cull` (below) is this recipe: the candidates go to a buffer of their own.

## Interfaces for the Hi-Z pyramid and the GPU cull (gpu_cull)

Where `batch::gpu_cull_lane` holds (the bindless lane, native multi-draw, not `WOW_GX_GPUCULL=0`), `static_gx/render/gpu_cull.rs` culls each static_gx view's colour pass and prepass on the GPU in two phases, in a modern frame with the prepass and a bindable depth (the module doc is the contract):

| step (span) | what |
|---|---|
| `static_gx_cull_early` | per candidate: current frustum; then last frame's pyramid under last frame's matrix. Visible: into `args`/`counts`. Occluded: deferred |
| `static_gx_prepass` | the early survivors' depth: the static occluders |
| `static_gx_hiz` | this frame's pyramid from that depth |
| `static_gx_cull_late` | the deferred against the new pyramid; survivors appended to `args`/`counts` and written to `late_args`/`late_counts` |
| `static_gx_prepass_late` | the late survivors' depth |
| `static_gx` | the colour pass: both phases' survivors |

**Buffers** (`batch::GxCullTwins`, at the args'/counts' offsets): candidates (`DrawIndexedIndirectArgs`), meta (`[count slot, bucket's first arg, selection key or 0xffff, phase-1 verdict]`), late args, late counts. A culled batch fuses draws only inside one selection key. **Bound**: the key's rows (above) or, for a cell or `0xffff`, the region's. The candidates are exactly the CPU walk's list for that pass, so a region `rt_scene::assemble_unseen` assembled off screen is never drawn.

**The pyramid** (`crate::static_gx::gpu_cull::GxHiZ`, render-world resource, present while the lane runs; `pyramid(&RetainedViewEntity) -> Option<HiZPyramid>`): one per static_gx view (world camera, water mirror). `R32Float`, `STORAGE_BINDING | TEXTURE_BINDING`; level 0 is the previous power of two of the view's depth texture per axis, a full chain to 1x1; each texel is the farthest reverse-Z depth (the minimum) over a conservative footprint; uv (0, 0) is the depth's top left. `HiZPyramid { view (all levels), size (level 0), mip_count, clip_from_world (the unjittered matrix of its contents, `None` before the first build) }`. Bind as `texture_2d<f32>` (unfilterable) and `textureLoad`. It holds **only the static world's early-phase depth** (no terrain, entities or late draws), valid after `crate::static_gx::static_gx_pass` for that view (`.after(...)`) until the next frame's. A lane that wants the full depth (Hi-Z SSR) builds its own pyramid after the opaque pass from the same `gx_hiz.wgsl` (`HIZ_FROM_DEPTH` for level 0, then one dispatch a level).

**Not culled on the GPU**: the shadow cascades. Their CPU list already tests each region's whole box against each cascade (`modern_shadow::outside_cascade`), the grain the GPU would have for a region drawn whole, and their static layers are cached between frames; occlusion does not apply.

**Switch**: `WOW_GX_GPUCULL=0` (default on where the lane can run). **Instruments**: the spans above in `WOW_GPU_PASSES` / `WOW_GPU_STATS`; with `WOW_GX_PERF`, a `GX_CULL <world|mirror> per frame: candidates= early= late= culled=` line every 64 frames (GPU counters read back a few frames late).

## Interfaces for the ray-traced lanes (rt_scene)

Where `rt_scene::enabled()` holds (capable: `HwCaps.ray_query`, static_gx bindless, not `WOW_RT=0`; and the `rayTracing` option on this frame), `static_gx/render/rt_scene.rs` keeps a BLAS per assembled static_gx region and per resident terrain cell and one TLAS over them, in modern frames only. Its module doc is the contract; this is what the probe GI, shadow and reflection lanes build on. The option switches at run time: see "The run-time switch" below.

**Rust** (`crate::static_gx::rt_scene`, native only):

| item | what |
|---|---|
| `capable() -> bool` | the session can ray-trace (fixed once render startup ran; false before): what decides which pipelines declare the scene |
| `enabled() -> bool` | capable and the `rayTracing` option on, this frame: the scene is (being) built |
| `live() -> bool` | enabled and the scene ready, this frame: the ray-traced lanes trace. Any world (outside the render world, its latest decision) |
| `RtTier { capable, enabled, live, frame }` | render-world resource, inserted at `RenderStartup` by `init_rt_scene` (a startup system that reads it orders `.after(crate::static_gx::rt_scene::init_rt_scene)`); `enabled` / `live` decided once a frame by `update_rt_tier` in `RenderSystems::PrepareAssets` |
| `RtScene` | render-world resource, present exactly while `RtTier.capable`; `bind_group() -> Option<&BindGroup>` (`None` until the TLAS holds an instance, in classic frames and while disabled), `lane_bind(live) -> Option<&BindGroup>` (the scene while `live`, else the stand-in), `pending()` (entries still waiting for a BLAS) |
| `layout() -> BindGroupLayoutDescriptor` | the bind group's layout, for a pipeline's `layout` vec |
| `shader_defs(group) -> Vec<ShaderDefVal>` | `RT_GROUP = group`: the group index the importer binds the scene at |
| `rt_scene_pass` | the `Core3d` system that records the frame's copies, repack, BLAS and TLAS builds (`.before(Core3dSystems::MainPass)`, once a frame in the first static_gx view); a pass that traces orders `.after(crate::static_gx::rt_scene::rt_scene_pass)` |

**The bind group** (`rt_scene_layout`, compute and fragment): 0 the TLAS; 1 the instance table `array<vec4<u32>>`, two rows a slot; 2 static_gx's region table; 3 static_gx's index arena; 4 the terrain index arena; 5 the attribute arena (5 `u32` a vertex: uv.x, uv.y, octahedral normal snorm16x2, static vertex word, colour unorm8x4); 6 `binding_array<texture_2d_array<f32>, 1024>` (the pool classes); 7 repeat sampler; 8 clamp sampler.

**WGSL** (`#import benilla_modern::rt_scene`): the importer's own first line is `enable wgpu_ray_query;` (naga_oil keeps enables per module), and its pipeline passes `rt_scene::shader_defs(group)`.

| entry | what |
|---|---|
| `trace(origin, dir, t_min, t_max, mask) -> RayIntersection` | nearest blocking hit; candidates on cutout geometry pass where the texel's alpha (mip 0) is under 224/255, items a record hides (bit 0) never block |
| `occluded(origin, dir, t_min, t_max, mask) -> bool` | the same test with `RAY_FLAG_TERMINATE_ON_FIRST_HIT`: shadow and visibility rays |
| `surface(origin, dir, hit) -> Surface` | a committed hit as `kind`, `row`, `word`, `t`, world `position`, interpolated world `normal` facing the ray, `uv`, vertex `color`, `front_face` |
| `surface_texel(s, level) -> vec4<f32>` | a static surface's texel through its wrap bits (white on terrain: the splat is not in the scene) |
| `texel(row, word, uv, level)`, `record(row, word)`, `instance(custom)`, `triangle(inst, geometry, primitive)` | the pieces, for a lane that needs more |
| `MASK_CELL`, `MASK_WMO`, `MASK_PROP`, `MASK_TERRAIN`, `MASK_STATIC`, `MASK_ALL`; `KIND_*` | instance masks (`1 << kind`) and kinds |

**What the scene holds.** Every region the static_gx GPU cache has assembled; while enabled, `prepare_static_gx` also assembles the regions the camera has not drawn, four a frame, so the cache (and with it the cascades' static casters and the GI voxelizer's casters) holds every streamed region. Builds go nearest the camera first, `WOW_RT_BUDGET` triangles a frame (default 262144); until `pending()` is 0 some resident geometry is missing from the TLAS. Terrain hits carry no texture: a lane that shades terrain takes its colour elsewhere.

## Interfaces for the ray-traced probe GI (rt gi)

Where `crate::modern::gi::rt_gi()` holds this frame (native, `rt_scene::live()`, not `WOW_RT_GI=0`; `WOW_RT=0` and the `rayTracing` option off imply off), each probe ray is a ray query against the TLAS instead of the voxel DDA, and the voxelizer does not run. Everything else in the probe GI (classify, setup, sum, blend, the field's layout and sampling) is unchanged.

| item | what |
|---|---|
| `modern::gi::rt_gi() -> bool` | the lane's decision, per frame (`rt_scene::live()` and `rt_gi_capable()`), any world; false before render startup |
| `modern::gi::rt_gi_capable() -> bool` | `rt_scene::capable()` and not `WOW_RT_GI=0`: fixed for the session; the RT pipeline is queued where it holds |
| `modern_gi_trace_rt` | pipeline (`gi_trace_rt.wgsl`, entry `trace_rays`): group 0 the `modern_gi_trace` layout, group 1 `rt_scene::layout()` with `rt_scene::shader_defs(1)`; queued in `gi::node::init_pipelines` (`.after(rt_scene::init_rt_scene)`) beside `modern_gi_trace`, and the pass picks one per frame |
| span `modern_gi_trace_rt` | the trace pass at this tier (setup, ray query, sum), in place of `modern_gi_trace`; the voxel path's `modern_gi_voxel_clear` / `modern_gi_voxelize` spans and voxel draws are absent; `modern_gi_heightfield` remains |
| `gi_trace_pass` | `.after(rt_scene::rt_scene_pass)` on native, besides its existing edges (after `gx_shadow_pass` and `gi_voxel_pass`, before `froxel_pass` and `MainPass`); same (main) queue as before |
| `ModernGiScene` | `albedo` / `normal` `None` at this tier; `height` and `clip` (`hf*`, `bounds.x`, `info.w` = the reach) as before. The field is off (`GiParams.params.x = 0`) until `RtScene::bind_group()` is `Some` |
| `GiFrame::{rt_was, cut_frames}` | a switch of `rt_gi()` either way traces the next probe period as a cut (`ModernGi::cut`, as after a teleport): every probe is traced on the new path and replaces its history |
| `GiFrame.vox_stub` | a 1x1x1 `Rgba8Unorm` 3D view fills the trace layout's bindings 4 and 5 (the clipmap's slots) at this tier |

**What a hit is.** `rt_scene::trace(probe, dir, tune.z, clip.info.w, MASK_ALL)`. Albedo: static items' texel at mip 3 through `rt_scene::texel`, `linear_to_srgb`'d (mid grey untextured); terrain the `GiParams.ground` albedo. Normal: the interpolated vertex normal through `object_to_world`, not turned to the ray, so a ray leaving a surface by its back counts toward the probe's backface share. Two-sided (lit through, never back-faced): the region's cutout geometry. No `FOLIAGE_GAP`: rays pass leaves where their texels do. Light: `sunshadow::sun_visibility_anywhere` (the cascades), the probe's four near lights, the previous field's nearest probe; a miss is `probegi::miss_radiance` with distance = the reach.

**For F (RT shadows).** The hit's sun term is the line `sunshadow::sun_visibility_anywhere(..., lift)` in `gi_trace_rt.wgsl`'s `trace_rays`; with ray-traced sun visibility it may become `rt_scene::occluded(lift, to_sun, 0.0, far, MASK_ALL)` (the shader already binds the scene at group 1). The heightfield is not F's to remove: the froxels' ground banks read it.
**For G (RT reflections).** Nothing of this lane is shared beyond `rt_scene`; `cast` in `gi_trace_rt.wgsl` is the pattern for shading a hit (vertex normal, texel at a mip, ground albedo on terrain).
**For H (async compute).** The trace pass at either tier reads the modern block, the cascades, the light table, the heightfield (setup) and, at this tier, the TLAS; it must stay after `rt_scene_pass`'s TLAS build and before `froxel_pass`. The voxel path's `gi_voxel_pass` dependency stays only when `rt_gi()` is false.

## Interfaces for the ray-traced shadows and the froxel rays (rt shadows)

Where `crate::modern::shadow::rt_shadows()` holds this frame (native, `rt_scene::live()`, not `WOW_RT_SHADOWS=0`; `WOW_RT=0` and the `rayTracing` option off imply off), every modern receiver takes the sun's and the shadowed lamps' visibility through the static world and the terrain as a ray query against the TLAS, and the maps keep only what the scene does not hold. Where `crate::modern::volumetrics::rt_froxel()` holds this frame (`rt_scene::live()`, not `WOW_RT_FROXEL=0`), the froxel scatter does the same per froxel. The two lanes are independent. `modern/rt_shadow.rs`'s module doc is the contract.

**What changes at the lane.** The cascades (layers 0-3) draw the entity casters alone; their `modern_shadow_static_c*` spans only clear. The sky map (layer 4) keeps its static casters. The lamp atlas draws no static face (`modern_local_shadow` only clears a slot's tiles). The receivers multiply the maps' term (entities) by the rays' term (static world, terrain), under the same shade floor. With `WOW_RT_FROXEL=0` and the shadows on, the froxels see the entity cascades alone inside the cascade range: an A/B lever for the scatter's cost, not a look.

**The penumbra.** The sun: four rays to a ring at 0.7 of a disc `SUN_CONE` = 0.0093 rad in radius (twice the sun's), the opposite pair first, the other two only where those disagree; the pattern turns per receiver by `locallight`'s wrapped world-position hash. Contact-hardening, in place of the cascades' fixed five-tap PCF; no temporal accumulation is assumed. The shade floor's cover read is three rays from a `COVER_RADIUS_YD` ring in the tangent plane, times the entity cascades' own ring. A lamp: four rays to a ring at 0.7 of the tier's emitter radius (`soft.y`) where the tier filters (`soft.x > 1`), else one. The water, the probe hits and the froxels take one ray.

**Rust:**

| item | what |
|---|---|
| `modern::shadow::rt_shadows() -> bool`, `modern::volumetrics::rt_froxel() -> bool` | the lanes' decisions, per frame (`rt_scene::live()` and the lane's `_capable()`), any world; false before render startup |
| `modern::shadow::rt_shadows_capable()`, `modern::volumetrics::rt_froxel_capable()` | `rt_scene::capable()` and the lane's `WOW_RT_*` not "0": fixed for the session; what decides which pipelines compile the lane |
| `RtScene::lane_bind(live) -> Option<&BindGroup>` | the scene's bind group while the lane is `live` (and it exists), else the stand-in over an empty TLAS that every ray misses, whose rows carry `STUB_MARK` so `rt_scene::live()` is false in WGSL; `None` only before the first `prepare_rt_scene`. For a pipeline that declares the scene for the whole session |
| `benilla_assets::materials::RT_SHADOW_LAYOUT` | `rt_scene::layout()`, set at `RenderStartup` by `modern::rt_shadow::init_rt_shadows` (`.after(init_rt_scene)`) where `rt_shadows_capable()` holds; the material lanes' `specialize` appends it as group 4 |
| `bevy::pbr::MaterialExtraBindGroup(Option<BindGroup>)` | render-world resource bound at group 4 by every `DrawMaterial` draw; inserted where `rt_shadows_capable()` holds, filled every frame in `PrepareBindGroups` with `lane_bind(rt_shadows())` (`bind_rt_shadows`) |
| static_gx group 3 | the colour pipelines' scene (`RT_SHADOW_GROUP`), `lane_bind(rt_shadows())` in `static_gx_pass`'s colour job |
| `GiPipelines::voxel_rt`, `GiFrame::shadow_bind` | the voxel probe trace compiles `RT_SHADOWS` and binds group 1 for its hits' sun wherever `rt_shadows_capable()` holds: `lane_bind(rt_shadows())`, so on the stand-in its hits read the cascades alone |
| `FroxelPipelines::scatter_rt`, `FroxelTargets::scatter_rt_bind` | the froxel rays' pipeline (`modern_froxel_scatter_rt`: the scatter layout, then `rt_scene::layout()` at 1, defs `rt_scene::shader_defs(1)` + `RT_FROXEL`), queued where `rt_froxel_capable()`; `scatter_rt_bind` (the live scene) is `Some` exactly in a frame `rt_froxel()` holds, and then the pass runs `scatter_rt` in place of `scatter`. A switch cuts the volume's history (`medium.w` = 0 at `volumetrics::HISTORY_OFFSET`) |
| `LightParams::slot_clear` | the twelve slots' fixture boxes (`ShadowLight::clear`) in the light block's spare rows 9-11 (native only; rows 12-13 stay free) |

**WGSL** (`#import benilla_modern::rtshadow` under `#ifdef RT_SHADOWS` / `RT_FROXEL`; the importer needs no `enable`, the module and `rt_scene` carry it). Every entry below first asks `rt_scene::live()`; on the stand-in it returns its `sunshadow` / `locallight` original's value (the `_ray` and `lamp_rays` entries 1), so a receiver's one pipeline draws the raster path whenever the lane is not live:

| entry | what |
|---|---|
| `sun_visibility(map, samp, c, p, n, view_distance, lite, to_sun)` | `sunshadow::sun_visibility`'s twin: entity cascades times `sun_rays`, under the shade floor |
| `sun_visibility_anywhere(map, samp, c, p, to_sun)` | the probe hits' twin (`p` already lifted) |
| `sun_visibility_fast(map, samp, c, p, view_distance, to_sun)` | the froxels' twin: the cascade tap times one ray |
| `sun_visibility_ray(c, p, to_sun)` | one ray alone, 1 with no cascades this frame (the water) |
| `sun_ray(origin, dir)`, `sun_rays(p, n, to_sun, view_distance, soft)`, `cover_rays(…)`, `entity_cascades(…)` | the pieces: one ray (`MASK_ALL`, `SUN_REACH` = 1200 yd), the soft ring, the cover ring, the cascades without their far term |
| `local_visibility(map, samp, g, slot, anchor, light, clear, p, n, near, fade)` | `locallight::local_visibility`'s twin: the atlas (entities) times `lamp_rays`, under the floor and the fade |
| `lamp_rays(p, n, light, clear, near, emitter, taps)`, `lamp_ray`, `fixture_entry` | rays to a lamp that stop at its fixture box and the cube's near plane; a zero `n` starts at `p` |
| `locallight::local_visibility_raw(…) -> f32` | the atlas lookup before the floor and the fade; -1 where the blocker search found nothing (wholly lit) |

**Spans and instruments.** The receivers' rays run inside the colour passes (`static_gx`, `main_opaque_pass_3d`, the transparent pass for the water), so their cost reads against a `WOW_RT_SHADOWS=0` run of the same spans. What the lane removes reads in the `modern_shadow_static_c0..c3` spans and in `WOW_GX_PERF`'s static shadow caster draws (only the sky map's remain), and in `modern_local_shadow`. The froxel rays: `modern_froxel_scatter_rt` in place of `modern_froxel_scatter`.

**Ordering.** `froxel_pass` `.after(rt_scene_pass)` on native, besides its existing edges; the volumetrics `init_pipelines` `.after(init_rt_scene)`. The receivers draw in `MainPass`, after `rt_scene_pass`.

**For G (RT reflections).** `liquid_cata`'s pipelines already declare the scene at group 4 and compile `RT_SHADOWS` where `rt_shadows_capable()` holds, and are bound `lane_bind(rt_shadows())` every frame. A reflection lane in the water reuses that one mechanism (`push_rt_shadows`, `MaterialExtraBindGroup`) under a capability condition of its own (shadows capable or reflections capable) rather than a second group, and binds the scene when either lane is live; a reflected hit's sun is `rtshadow::sun_visibility_anywhere` or `sun_ray` (both fall back to the cascades on the stand-in). The receivers' `lite` (mirror) paths take no ray; with the mirror camera off while the lane is live, nothing of this lane draws into it, and the mirror must come back the frame the lane is not live ("The run-time switch").
**For H (async compute).** A pass that binds the scene, the stand-in included, must be recorded after `rt_scene_pass`'s TLAS build in submission order (wgpu refuses an unbuilt TLAS); on another queue that is a fence after it. The voxel trace's group 1 (`voxel_rt`) moves with the trace. The froxels stay on the main queue.

## Interfaces for the ray-traced reflections and Hi-Z SSR (rt reflect)

Where `crate::modern::rt_reflect::rt_reflections()` holds this frame (native, `rt_scene::live()`, not `WOW_RT_REFLECT=0`; `WOW_RT=0` and the `rayTracing` option off imply off), the cata water's planar mirror camera (`liquid::reflect`, about 4 ms of render-thread CPU at a waterside town) stands down at the tiers that have one (`waterDetail` 2 and 3, `WaterDetail::has_planar_reflection`), above the water, and the water takes the reflection the mirror read would have shown along the same ray: a Hi-Z march through this frame's full-depth pyramid first, then one ray query against the TLAS for what the screen cannot answer. `modern/rt_reflect.rs`'s module doc is the contract.

**The tiers.** `waterDetail 1` ("Modern (SSR)") never had a mirror: its reflection is the reference's perm-4 trace, one point 128 yd along the flattened reflected ray projected into `scene_blur`, over the horizon colour. The lane leaves that tier alone and does not replace that trace: at tier 2 the perm-4 point is still lerped over the base by its validity, as the reference lays it over the mirror; at tier 3 (experimental) it was laid only where the mirror was invalid, and the lane's base is valid everywhere, so there the march replaces it. One screen-space march exists (the lane's); the perm-4 point is the reference's combine, not a second march.

**Rust** (native only):

| item | what |
|---|---|
| `modern::rt_reflect::rt_reflections() -> bool` | the lane's decision, per frame (`rt_scene::live()` and `rt_reflections_capable()`), any world; false before render startup |
| `rt_reflections_capable() -> bool` | `rt_scene::capable()` and not `WOW_RT_REFLECT=0` (read once): the water compiles `RT_REFLECT` and declares the scene |
| `ssr_enabled() -> bool` | not `WOW_SSR=0` (read once) |
| `RtReflectLane { on }` | main-world resource, extracted: the lane stands in for the mirror this frame. Set by `decide_water_lane` (`Update`, after `WorldStage::Present` and `SubmersionVerdict`, before `sync_reflection_camera`): `rt_reflections()`, the option and a modern frame read this frame, a tier with a mirror, the eye above the water. `shader_flag()` is the water's `anim.x` |
| `benilla_assets::materials::RT_REFLECT_LAYOUT`, `push_rt_water` | the water's one scene group: declared where shadows or reflections are capable, each define only for its own lane |
| `MaterialExtraBindGroup` | `bind_rt_shadows` binds `lane_bind(rt_shadows() \|\| rt_reflections())`; inserted where either lane is capable |
| `LightParams::lanes` | the modern block's `light_spare[12]`: `x` = `rt_shadows()`, `y` = `rt_reflections()` this frame, written by `lights::node::prepare_lights` (`PrepareBindGroups`). A lane's WGSL asks its own row besides `rt_scene::live()`, since the group is live while either lane is |
| `ReflectHiZ`, `reflect_hiz_pass` | the pyramid's reduction (`modern_reflect_hiz`) and its per-level bind groups, remade when the depth copy is republished |

**The pyramid.** The water's own linear depth copy (`LiquidScreenTargets::depth`, `liquid_cata.wgsl` binding 108 `scene_depth`) carries a full mip chain where `rt_reflections_capable() && ssr_enabled()` (`screen::depth_pyramid()`, republished once render startup has decided it). `R32Float`, `TEXTURE_BINDING | RENDER_ATTACHMENT | STORAGE_BINDING`, level 0 viewport-sized (not a power of two; wgpu's halving per level). Level 0 is the planar view depth in yards of the world camera after the opaque pass (terrain, entities, static_gx, the sky at 1e9), written by `liquid_screen_pass` (MSAA: sample 0); levels 1.. are the minimum over a conservative footprint (`gx_hiz.wgsl` `downsample` under `HIZ_LINEAR`): on linear depth the **nearest** surface, which is what a march skips cells by (a cell is skipped only while the ray stays in front of everything in it). C's reverse-Z farthest is conservative for occlusion culling and the wrong way round for a march. A level-L texel covers at least uv `[x/size_L, (x+1)/size_L)`. Valid from `modern_reflect_hiz` until the next frame's, in a frame `RtReflectLane::on` holds; bind as `texture_2d<f32>` (unfilterable) and `textureLoad`. The world camera's depth is already `TEXTURE_BINDING` (`screen::arm_depth_binding`).

**WGSL.** `liquid_cata.wgsl` under `RT_REFLECT`: `lane_reflection(surface, dir, sky)` (the march, then `rtreflect::trace` while `light_spare[12].y`, the march's hit fading into the ray's toward the screen edge by the trace's quartic), `lane_march` (64 steps, thickness 0.35 yd + 2 % of the depth, 600 yd reach; state hit / left the screen / went behind), `lane_shade` (texel or `gi.ground`, the nearest probe or the ambient row, key light under `hit.sun` and the cloud shadow, in the authored value space the mirror read is in). The ray: from the surface (lifted 0.05 yd) along the mirror-reflected direction from the eye through the point the mirror read projects, without the experimental tier's plane term (each surface about its own height), folded above the water. Library `benilla_modern::rtreflect` (`shaders/modern/rt_reflect.wgsl`): `trace(origin, dir, t_min, reach, to_sun) -> ReflectHit { kind (0 sky, 1 static, 2 terrain), t, position, normal, albedo (static texel at mip 2, `linear_to_srgb`'d), sun (`rtshadow::sun_ray`) }`; returns nothing on the stand-in. One ray, no recursion; reach the far clip (`fog_params.w`), else 1200 yd. A miss keeps the sky the water already used (`sky_dome_rough` at tier 3, the horizon constant at tier 2).

**What the lane drops.** The experimental tier's rough-mirror read (`mirror_read_rough`'s mip lobe and contrast grade) has no counterpart: the march reads `scene_blur` at mip 0 and a ray hit is one sample. A ray hit carries no froxel fog of its reflected leg (the surface's own fog applies); the march's screen reads were fogged where drawn (`lane_fogged`).

**Switches.** `WOW_RT_REFLECT=0`: the mirror, as before (no `RT_REFLECT`, no pyramid levels). `WOW_SSR=0`: the lane stays (mirror off) but every ray goes to the TLAS (`anim.x` 2; no pyramid levels allocated, no `modern_reflect_hiz`). `WOW_RT_SHADOWS=0` with reflections on: the water declares the group for the reflections alone and compiles no `RT_SHADOWS`.

**Spans and instruments.** `modern_reflect_hiz` (the pyramid) in `WOW_GPU_PASSES` / `WOW_GPU_STATS`. The march and the ray run in the water's fragment stage, inside the transparent pass's span (`main_transparent_pass_3d`): read them against a `WOW_RT_REFLECT=0` run (and `WOW_SSR=0` for the march's share). What the lane removes: the mirror camera's whole render (its `static_gx` mirror pass, `GX_CULL mirror` lines and the mirror's draw count in `WOW_GX_PERF`, its opaque and prepass spans) and the render-thread CPU it cost. One `rt reflect: switch at frame N: on …` / `off (why) …` line per switch of the lane (not per dive).

**Ordering.** `reflect_hiz_pass`: `Core3d`, `MainPass`, `.after(liquid_screen_pass).before(main_transparent_pass_3d)`; `prepare_reflect_hiz` in `PrepareBindGroups`; `init_reflect_hiz` / `init_rt_reflect` `.after(init_rt_scene)` (`init_rt_reflect` also after `init_rt_shadows`). The water's draws bind the scene in `MainPass`, after `rt_scene_pass`. Main world: `decide_water_lane` → `sync_reflection_camera` → `sync_reflection_flag`; the mirror camera's activity and the water's `anim.x` / `path.z` reach the same extraction. The one-frame lag (`decide_water_lane`'s doc): going live the mirror runs a frame longer; falling, the option and the model are read the same frame, and only a render-side fall (a teleport, a map change, an option callback that runs after the decision) shows one frame of the march over the stand-in with the sky for the rest. Neither path keeps history; nothing is cut on a switch.

**For M (meshlets / mesh shaders).** The pyramid is the water's, built from the opaque depth after every opaque draw, so a mesh-shader static path only has to keep writing that depth; nothing of the lane reads the region table except through `rt_scene`. A task-shader cull that wants a full-depth pyramid could take this one's levels (linear, nearest) only for the next frame and only with an inverted test; C's farthest reverse-Z pyramid stays the cull's.
**For N (multiview shadows).** Nothing: the lane reads the cascades nowhere (the hit's sun is a ray). The water's own sun shadows are F's (`light_spare[12].x`).
**For H (integration, async compute).** `modern_reflect_hiz` reads the depth copy `liquid_screen_pass` writes and is read by the transparent pass: on another queue it is a fence after the screen pass and before the transparent pass, so it gains little. Could the full-depth pyramid replace C's? Not as is: C culls static_gx against the early static depth before the opaque pass, which needs the farthest reverse-Z of last frame and this frame's early phase; this pyramid exists only after the opaque pass. A first-compile check: the RT variant of `liquid_cata` (`RT_SHADOWS`, `RT_REFLECT`, `RT_GROUP = 4`) is not in `benilla-app`'s compose test, which composes the web/classic variant only; add one with the native libraries and ray-query capabilities.

## The run-time switch (rt_scene: capable, enabled, live)

The `rayTracing` option (`benilla_world::modern::RayTracingSetting`, written by `benilla-app`'s `raytracing` CVar arm and extracted every frame) switches every ray-traced lane at run time, with no restart and no pipeline re-specialized. The device always has the ray-tracing features where the adapter does. The `rt_scene.rs` module doc is the contract.

| level | what | fixed? |
|---|---|---|
| capable (`rt_scene::capable()`) | `HwCaps.ray_query`, static_gx bindless, not `WOW_RT=0`. `RtScene` exists; every pipeline that can trace compiles its RT variant and declares the scene. A lane's `*_capable()` adds its own `WOW_RT_*` | for the session (render startup) |
| enabled (`rt_scene::enabled()`) | capable and the option on: the scene is built (nearest the camera first, `WOW_RT_BUDGET`) and `assemble_unseen` runs. Off: `RtScene::free_all` frees every BLAS, the TLAS, the instance table, the arenas and the scratch | per frame |
| live (`rt_scene::live()`) | enabled and the scene ready. The lanes (`rt_gi()`, `rt_shadows()`, `rt_froxel()`) are `live() && *_capable()` | per frame |

All three are decided once a frame by `rt_scene::update_rt_tier` in `RenderSystems::PrepareAssets`, before any reader, so every system and pass of a render frame sees one value. Outside the render world the functions return its latest decision.

**Ready** (`RtScene::update_ready`, after each frame's plan). Rising: no entry waits for its BLAS, no region static_gx streamed is still unassembled (`assemble_unseen`'s backlog; let go after `READY_GRACE` = 300 frames of nothing else missing), the TLAS holds an instance and its bind group exists, for `READY_FRAMES` = 8 frames in a row. Falling, only while ready: the scene is empty (a classic frame, a map change), or more entries wait than are built (a teleport). A cell streaming in while live waits its turn and the lanes stay on rays.

**What a switch does.**

| on → off (or a live scene falling back) | off → live |
|---|---|
| The receivers are bound the stand-in (`RtScene::lane_bind`): `rt_scene::live()` is false in WGSL and every `rtshadow` twin returns its raster original | The receivers are bound the scene; the twins trace |
| The cascades and the sky map redraw from their casters that frame (`ModernShadowFrame::rt_was` clears `cache_keys`); the lamp slots redraw their static faces after one frame (`hurry_until`) | The cascades and the lamp slots drop their static casters the same way |
| The probe trace runs `modern_gi_trace`; `prepare_gi_voxels` makes the clipmap fresh and refills it (raster pipelines queued on first use); the field is cut for one probe period | `modern_gi_trace_rt`; the clipmap is freed; the field is cut for one probe period |
| The froxels run `modern_froxel_scatter`; the volume's history is cut | `modern_froxel_scatter_rt`; the history is cut |
| (option off) the scene is freed; regions already assembled stay in static_gx's cache until they stream out, as a seen region always has | the scene is rebuilt from the cache, unseen regions assembled four a frame |
| The water's mirror camera renders again and the water reads it (`RtReflectLane` off, `anim.x` 0), the frame the main world reads the option off (one frame later where the render world falls first: that frame keeps the march over the stand-in) | The mirror stands down and the water marches, then traces (`RtReflectLane` on), the frame after the main world first reads `live()` |

**Instrument.** One `rt: switch at frame N: requested <on|off>, effective <on|off> …` line at each switch (the option, and the scene going live or falling back); the going-live line carries the milliseconds and frames since the lanes asked for the scene. `rt_scene: freed …` when the option frees it. A scripted run flips the option mid-session through the chat probe and the console: `WOW_PROBE_CHAT="/console rayTracing 0;/console rayTracing 1" WOW_PROBE_CHAT_AT=60 WOW_PROBE_CHAT_EVERY=60` (each line a `ConsoleExec` write, whose callback runs at once), so one session measures both with `WOW_GPU_PASSES` / `WOW_GPU_STATS` / the FPS journal.

**For G (RT water reflections, Hi-Z SSR).** Key the reflection lane on a per-frame decision, `rt_scene::live() && rt_reflections_capable()` (a `WOW_RT_*` of its own read once), never on a startup value. Compile the RT variant where the lane is capable, and pick the ray or the SSR path per frame from the group bound (`lane_bind(live)` and `rt_scene::live()` in WGSL), not by re-specializing. If the water mirror camera is turned off while the lane is live, its switch must follow `live()` every frame: on a fall-back to raster the mirror (or SSR) has to be back that same frame, and any temporal reflection history is cut on the switch.
**For M (meshlets / mesh shaders).** Nothing traces through meshlets, but a mesh-shader path that replaces static_gx's draws must keep the region table and the index arena the scene's BLASes read (`bindless.rs`), whatever the option says: the scene may be rebuilt from them at any frame.
**For H (integration, async compute).** A pass that binds the scene or the stand-in stays after `rt_scene_pass` in submission order. Read the lane decisions in the render world only after `PrepareAssets` (`update_rt_tier`); a system that caches one across frames must handle it changing every frame. Freeing happens in `prepare_rt_scene` (`PrepareResources`), before any pass of the frame records, and relies on wgpu keeping in-flight objects alive.

## Interfaces for meshlets and mesh shaders (meshlets)

Where `meshlets::capable` holds, `static_gx/render/meshlets.rs` draws each static_gx view's GPU-culled batch as meshlets through a task and a mesh stage, the task stage culling each meshlet; anything short of the whole batch draws as the GPU cull draws, from the same cull. Its module doc is the contract; `static_gx/meshlet.rs` is the meshlets' format.

**Capability** (decided at render startup, `meshlets::lane_active()`):

| level | what |
|---|---|
| `HwCaps.mesh_shader` (`modern/caps.rs`) | `EXPERIMENTAL_MESH_SHADER`, `SHADER_DRAW_INDEX`, `MULTI_DRAW_INDIRECT_COUNT`, `INDIRECT_FIRST_INSTANCE`, `IMMEDIATES`, the mesh and task limits the format needs, 5 bind groups and 16 storage buffers a stage, on Vulkan only: naga 29 writes task and mesh stages to SPIR-V alone. False on DX12 (benilla's default `gxApi`) and the web |
| `meshlets::capable(caps)` | `HwCaps.mesh_shader`, `batch::gpu_cull_lane` (bindless, not `WOW_GX_GPUCULL=0`), not `WOW_STATIC_GX_MDI=0`, not `WOW_GX_MESHLETS=0`: groups 1 and 2 are widened, the meshlet arena exists, the cull twins carry the mesh buffers |
| `meshlets::lane_active()` | `GxIndirect::meshlet_lane()` at `init_meshlets` (`RenderStartup`, after `batch::init_indirect`): `meshlet::set_wanted(true)`, every later bake builds meshlets, `GxMeshletLane` exists, the mesh pipelines are queued in prepassed frames |
| per view and frame | `GxMeshletPass::new`: the batch maps onto meshlets (`GxBatch::meshlets`), the cull ran its mesh variant (`GxCullPass::meshlets`), every slab of the batch has a mesh group, every mesh pipeline is compiled |

**Data** (native only; the region table and the index arena are as before, so the ray-traced scene and every vertex-path pass read them unchanged):

| item | what |
|---|---|
| `meshlet::GxMeshlets` | per region, items in bake order: `rows` (two a meshlet: `[vertex offset, triangle offset, vertex count \| triangle count << 8, cone]` and `[mesh-local centre xyz, radius]` in f32 bits; the cone a snorm8 axis in bytes 0..3 and cutoff in byte 3, 127 = none), `vertices` (an index into the region's mesh each), `triangles` (`a \| b << 8 \| c << 16`, meshlet-local), `items` (item `i`'s meshlets `items[i]..items[i + 1]`). 64 vertices and 124 triangles at most; a meshlet never spans items; two-sided and foliage-wind meshlets carry no cone, wind ones a sphere 1 yd wider |
| the meshlet arena (`GxBindless`) | three `STORAGE` `GrowBuffer`s suballocated first-fit like the index arena: the meshlet table (32 B a meshlet, the rows with both offsets made absolute), the meshlet vertices and the meshlet triangles (`u32` each). `GxSlots::meshlets: GxMeshletSlots { first, count, vertex, vertices, triangle, triangles }`; `meshlet_buffers() -> GxMeshletBuffers { table, vertices, triangles, capacity }` |
| `GxCellGpu::meshlet_starts` | a region's meshlets are in its index arena's item order; per position `k` of `order` its first meshlet in the region's block (empty: the region has none). `meshlets::span` maps a draw's arena range onto its meshlets |
| `GxCullTwins` mesh buffers | at the args' offsets: the mesh meta (`[first meshlet, count]` per candidate, 8 B, written by `GxBatch::finish`), the task dispatches (`DispatchIndirectArgs`, 12 B, `INDIRECT \| STORAGE`) and the candidate refs (`u32`, the candidate's index, bit 31 for a late survivor), both written by the cull's mesh variant beside `args` |
| per view (`GxMeshletLane`) | the drawn bits (one per meshlet of the table, this frame's only), the `GX_MESHLET` counters, a mesh group per vertex slab |
| mesh-allocator slabs | `STORAGE` added to `MeshAllocatorSettings::extra_buffer_usages` on native unless `WOW_GX_MESHLETS=0`, before the device is known: on DX12 too, where the lane stays off. A slab past `max_storage_buffer_binding_size` gets no mesh group |

**Layouts.** Bevy's view binding 0, static_gx group 1 (bindless: the table and samplers) and group 2 (the light layout, every binding) are visible to `TASK | MESH` where capable, so the vertex and mesh pipelines keep sharing them. The mesh group (`meshlets::mesh_layout()`, `TASK | MESH`, at the pipeline's group count: 3, or 4 in a colour pipeline that declares the ray-traced scene): 0 the cull's `CullView` uniform, 1 the meshlet table, 2 the meshlet vertices, 3 the meshlet triangles, 4 the draw's vertex slab (`array<u32>`, 16 words a vertex), 5 the candidates, 6 the mesh meta, 7 the candidate refs, 8 the drawn bits (atomic), 9 the view's pyramid, 10 the counters. The cull's mesh variant (`static_gx_cull_mesh_layout`) is the cull's ten bindings and 10 the mesh meta, 11 the task dispatches, 12 the candidate refs.

**WGSL.** `meshlets::MESH_SHADER`, a constant-handle asset built at startup from `enable wgpu_mesh_shader, draw_index;`, `#import benilla_modern::gx_occlusion`, `static_gx.wgsl` and `shaders/modern/static_gx_mesh.wgsl` (a tail, not a module), composed under `GX_MESHLETS` and `GX_MESH_GROUP`: `meshlet_task` (32 threads, one meshlet each; immediates `[bucket's first arg slot, mode]`; payload `{ row, base_vertex, mode, pad, meshlets: array<u32, 32> }`, 144 B), `meshlet_mesh` (64 threads through `gx_vertex`, the `vertex` body) and `meshlet_mesh_prepass` (through `gx_prepass_vertex`); the fragments are static_gx's own. Library `benilla_modern::gx_occlusion` (`shaders/modern/gx_occlusion.wgsl`): `CullView` (176 B: C's uniform plus the unjittered `eye`), `PREV_VALID`, `STATS`, `MIRRORED` (the cone test skipped), `Bound`, `outside`, `occluded(m, b, pyramid, size, mip_count)`, `backfacing(cone, centre, radius, eye)`; `gx_cull.wgsl` imports it.

**Passes and spans** (the GPU cull's, unchanged; `static_gx_cull_*` run their `_mesh` pipelines):

| span | mode | what the task stage draws |
|---|---|---|
| `static_gx_prepass` | EARLY | an early survivor's meshlets in the frustum, not back-facing, not occluded in last frame's pyramid under last frame's matrix; sets or clears each one's bit |
| `static_gx_prepass_late` | LATE | over every survivor: an early draw's meshlets with the bit clear, every meshlet of a late draw, against this frame's pyramid |
| `static_gx` | COLOUR | the meshlets whose bit is set: exactly what the prepasses laid, so `Equal` holds |

One `multi_draw_mesh_tasks_indirect_count` per bucket, with the bucket's count; `GX_PERF`'s `draws/frame gx=` counts them as it counted the indexed multi-draws.

**Switches.** `WOW_GX_MESHLETS=0`: the lane off, C's path, no `STORAGE` on the slabs (default on where capable). `WOW_GX_GPUCULL=0`, `WOW_GX_BINDLESS=0`, `WOW_STATIC_GX_MDI=0` imply off; `WOW_STATIC_GX_PREPASS=0` and classic frames queue no mesh pipelines. **Instruments**: the spans above in `WOW_GPU_PASSES` / `WOW_GPU_STATS` (pipeline statistics count task and mesh invocations there); with `WOW_GX_PERF` a `GX_MESHLET <world|mirror> per frame: meshlets= early= late= culled=` line every 64 frames beside `GX_CULL`; read against a `WOW_GX_MESHLETS=0` run.

**Not on the lane.** The shadow cascades, the lamp atlas and the GI voxelizer draw regions whole on the vertex path (no occlusion to gain, cached layers). DX12 would need passthrough task and mesh shaders. A hot reload of `static_gx.wgsl` does not reach the mesh module. A region baked before startup decided has no meshlets; a batch holding one draws on C's path that frame.

**For N (multiview shadows).** Nothing of the lane touches the cascades; widening groups 1 and 2 to `TASK | MESH` changes nothing for a vertex pipeline. A mesh-shader cascade path would bind the same meshlet arena through a group of its own (no pyramid, no drawn bits) after its own groups; whether naga 29's mesh stage takes `@builtin(view_index)` is unchecked.
**For H (integration, async compute).** All of the lane's GPU work records in the prepass job's encoder in order (cull, EARLY, pyramid, cull, LATE, the counters' copy) and in the colour job, which reads the drawn bits the prepass job wrote: both must stay on one queue in that submission order, as C's cull already requires. The lane adds no pass and no span of its own.

## Planned hooks (not yet landed)

- `wow_model.wgsl`, `terrain.wgsl`, `static_gx.wgsl`: the lighting and fog spans (P1–P4)
- `wdl`, `liquid`, `liquid_cata`, `sky`, `cloud`, `celestial`, `star`, `wow_effect` `.wgsl`: one output/fog region each (P1, P4)
- `static_gx/render.rs`: `pub(crate)` GPU cache types, plus an early return in `StaticGxNode` when modern is active (P1/P2)
- `ssilvb.wgsl` / `ssilvb.rs`: `SSILVB_LINEAR` composite branch (P1)
- `particles/render.rs`: a `modern` bit in `EffectPipelineKey` (P1)
- `benilla-app/src/player_shadow.rs`, `blob_shadow.rs`: classic-only gate (P2)
- `benilla-app/assets/ui/OptionsFrame.xml`: one lighting row block (P7)
- `ffx_glow.rs`: `FfxGlow`'s state and `ffx_glow_pass` read by `modern/meter.rs`, the auto-exposure's screen meter, which runs before the combine
