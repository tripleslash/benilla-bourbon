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

Bevy 0.18 schedules `check_entities_needing_specialization::<M>` in `PostUpdate`, unordered against upstream's material writers (`model_fade`, `straddle`, `zfill`, `model_render` lazy spawns, `static_gx` cull…). Adding any system there reshuffles the single-threaded executor's order. A writer can then land after the check, and the render world panics at `bevy_pbr` `material.rs:1061` in classic and modern alike (bisected 2026-09-15).

Fork systems go in `PreUpdate`, `Update` or `Last`.

The race also fires from data alone: the sun arc moving the sky was enough. So `modern::tick_heal` (always registered, classic too) inserts the missing tick just before `specialize_material_meshes`, and warns the first time it does.

## Live hooks

| File | Hook | Since |
|---|---|---|
| `benilla-world/src/lib.rs` | `pub mod modern;` | P0 |
| `benilla-world/src/world_plugins.rs` | `.add(crate::modern::ModernLightingPlugin)` after `SsilvbPlugin` | P0 |
| `benilla-world/src/shaders.rs` | `modern_libraries()`: `load_shader_library!` for `shaders/modern/*.wgsl` | P0 |
| `benilla-app/src/cvars.rs` | the `lightingModel` / `lightingQuality` rows, the "fork-owned block" markers, the `lightingmodel`/`lightingquality` apply arms, the `KnobParams`/`Knobs` `lighting` field, the `GetCVar` seed rows, the `WOW_LIGHTING*` session-owned env keys | P0 |
| `benilla-world/src/lib.rs` | `pub mod launch_env;` (fork-owned module) | P0 |
| `benilla-world/src/worldview.rs` | `view_start` reads `WOW_WORLDVIEW_AT` through `crate::launch_env::var` | P0 |
| `benilla-app/src/lib.rs` | `run_worldview` wrapper (was a `pub use`) + `wants_worldview` | P0 |
| `wenilla/src/lib.rs` | `?worldview=1` switch in `start()` | P0 |
| `benilla-world/src/lighting/global_light.rs` | `build_light_data`: `active` param; row 20 `.y` = modern flag | P1 |
| `benilla-world/src/lighting/resolve.rs` | `apply_sky_backdrop`: decoded clear colour when modern | P1 |
| `benilla-world/src/sky.rs` | `update_sky_colors`: sky stops + fog through `decode_if_modern` | P1 |
| `benilla-world/src/sun/follow.rs` | `disc_span` `.z` = modern flag | P1 |
| `benilla-world/src/clouds/mod.rs` | cloud image re-upload: `Rgba8UnormSrgb` when modern | P1 |
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
| `clouds/layer.rs`, `cloud.wgsl` | `CloudExt.modern` / `.light` at uniform 102 (+ construction zeros); modern: key-elevation brightness + silver lining | P4 |
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
| `benilla-app/src/cvars.rs`, `ui_script/options_tests.rs` | the `waterDetail 1` assertions (SSR tier, no mirror; three dropdown entries) | P6 fix |
| `benilla-app/src/lib.rs` | `seed_launch_env()` (the old `run_worldview` key list, plus `WOW_WATER_DETAIL`) is called from `run` as well as `run_worldview`, so `?lighting=modern` reaches the client | P6 fix |
| `benilla-app/src/cvars.rs` | the `WOW_LIGHTING*` / `WOW_WATER_DETAIL` session-owned checks read `benilla_world::launch_env::var` (the browser's query) instead of `std::env` | P6 fix |
| `benilla-app/assets/ui/OptionsFrame.xml`, `ui_script/options_tests.rs` | Graphics page `RowLightingModel` (`lightingModel`, Classic/Modern) and `RowLightingQuality` (`lightingQuality`, Low…Ultra) dropdowns between Water Detail and Render Scale, their `BENILLA_TOOLTIP_LIGHTING_*` strings, and the tooltip census (82 → 84) | P6 fix |
| `benilla-assets/src/materials.rs` | `WowModelKey.modern` from `MODERN_MODEL_PIPELINES`; `specialize` pushes `WOW_MODERN` to both stages; `LiquidCataExt.modern_shadow` at 121 (texture only, no sampler) | P6 fix |
| `wow_model.wgsl` | every modern block (room light, cascades, foliage, sky/GI ambient, window emitter, per-pixel points, linear decode + froxel fog, debug views, the vertex `point_sel`) sits under `#ifdef WOW_MODERN` with a classic `#else` value: a classic pipeline compiling that code dead drew Chrome/Metal leaf-card holes black (bisected to P3 step 4's `wow_model` hunk) | P6 fix |
| `liquid_cata.wgsl` | binding 121 `modern_shadow_map`; `modern_glint` private; modern: `sunshadow::wind_swell` on the bump slope, sun visibility (`sun_visibility_load` × cloud) on the body light, glare and specular, an HDR glitter lobe added by the linear wrapper | P6 fix |
| `benilla-world/src/liquid/surface.rs` | `modern_shadow: MODERN_SHADOW_MAP` at the cata construction site | P6 fix |
| `wow_model.wgsl` | an explicit alpha-mask discard (`material.flags` MASK, `alpha_cutoff`) in the entry point before `alpha_discard`; `#ifdef WOW_ZFILL` cuts the colour lane out of the depth-prime twin | P6 fix 2 |
| `benilla-assets/src/materials.rs` | `specialize` pushes `WOW_ZFILL` to the depth-prime twin's fragment | P6 fix 2 |
| `benilla-world/src/ffx_glow.rs` | the modern decode / HDR downsample only for the `FfxState::Player` view (`FfxGlow::WORLD`); booths, UI panes and the glue scene stay classic | P6 fix 3 |
| `benilla-world/src/liquid/ripple.rs` | `collect_stamps`: wind cat's-paw stamps (`WIND_*`) before the rain when `frame_is_modern()` | P6 fix |
| `sunshadow.wgsl` | `ModernCascades.sky_params.w` = the water mirror's plane while it renders (`NO_MIRROR_PLANE` otherwise); `mirror_view(c, eye)`; `sun_visibility` / `cascade_visibility` / `sky_ambient` take `lite` (1 shadow tap, 1 sky tap) | P7 perf |
| `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` | `modern_gi_ambient` / `modern_local_lights` take `lite` (one probe cascade without Chebyshev; no cube shadows); every call passes `sunshadow::mirror_view(modern_block.cascades, view.world_position)` (debug views pass `false`); static_gx reads the sky map and probes ONCE for `lit_nl_shadowed` and `lit_doodad` (`modern_sky` / `modern_gi`) | P7 perf |
| `fog_composite.wgsl` | the Low tier's screen shafts skip the mirror view | P7 perf |
| `benilla-world/src/liquid/{mod,reflect}.rs` | `ReflectionPlane` re-exported; `ReflectionPlane::live_plane` | P7 perf |
| `benilla-app/src/lib.rs` | `SEEDED` gains `WOW_MODERN_MIRRORTIER` (dev A/B: every view shades the mirror tier) and `WOW_MODERN_CASTERREACH` (dev A/B: cascade caster reach, yd) | P7 perf |
| `benilla-world/src/liquid/ripple.rs` | fix (not modern-only): catch-up steps bind `RipplePipeline::settled` (prev = placement), so a two-step frame no longer reprojects the field twice and drags ripples with the camera | P7 perf |

## Planned hooks (not yet landed)

- `wow_model.wgsl`, `terrain.wgsl`, `static_gx.wgsl`: the lighting and fog spans (P1–P4)
- `wdl`, `liquid`, `liquid_cata`, `sky`, `cloud`, `celestial`, `star`, `wow_effect` `.wgsl`: one output/fog region each (P1, P4)
- `static_gx/render.rs`: `pub(crate)` GPU cache types, plus an early return in `StaticGxNode` when modern is active (P1/P2)
- `ssilvb.wgsl` / `ssilvb.rs`: `SSILVB_LINEAR` composite branch (P1)
- `particles/render.rs`: a `modern` bit in `EffectPipelineKey` (P1)
- `benilla-app/src/player_shadow.rs`, `blob_shadow.rs`: classic-only gate (P2)
- `benilla-app/assets/ui/OptionsFrame.xml`: one lighting row block (P7)
