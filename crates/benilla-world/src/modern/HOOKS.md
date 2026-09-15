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
| `wow_model.wgsl` | `#import`; decode block before `out.color` (additive premultiply, Mod/Mod2x identity, Mod2x ×2.295) | P1 |
| `wow_effect.wgsl` | `#import`; struct padded to row 20; decode before the blend `#ifdef` ladder | P1 |
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

## Planned hooks (not yet landed)

- `wow_model.wgsl`, `terrain.wgsl`, `static_gx.wgsl`: the lighting and fog spans (P1–P4)
- `wdl`, `liquid`, `liquid_cata`, `sky`, `cloud`, `celestial`, `star`, `wow_effect` `.wgsl`: one output/fog region each (P1, P4)
- `static_gx/render.rs`: `pub(crate)` GPU cache types, plus an early return in `StaticGxNode` when modern is active (P1/P2)
- `ssilvb.wgsl` / `ssilvb.rs`: `SSILVB_LINEAR` composite branch (P1)
- `particles/render.rs`: a `modern` bit in `EffectPipelineKey` (P1)
- `benilla-app/src/player_shadow.rs`, `blob_shadow.rs`: classic-only gate (P2)
- `benilla-app/assets/ui/OptionsFrame.xml`: one lighting row block (P7)
