# FSR 3.1.5: hook points in upstream-owned files

FSR (`FORK.md`) lives in fork-owned files:
- `benilla-world/src/fsr/` (this module) and `benilla-world/src/shaders/fsr/`
- `benilla-world/src/worldview/fsr_view.rs`

Every edit it needs inside a file that upstream benilla also edits carries the marker
`fork: fsr (FORK.md)` and is listed here. After an upstream merge, check each row is still
present. `fsr::tests::every_hook_row_carries_its_marker` reads this table and fails on a file that
lost its marker.

**Scheduling rule:** as `modern/HOOKS.md` says, no fork system goes in `PostUpdate`; FSR's main-world
systems run in `Last`.

## Live hooks

| File | Hook |
|---|---|
| `benilla-world/src/lib.rs` | `pub mod fsr;` |
| `benilla-world/src/world_plugins.rs` | `.add(crate::fsr::FsrPlugin)` after `ModernLightingPlugin` |
| `benilla-world/Cargo.toml` | `wgpu.workspace = true` (the tier-1 flag's type); `naga` dev-dependency (shader validation) |
| `benilla-world/src/lighting/global_light.rs` | the shared light buffer gains `COPY_SRC`, for the previous rig pose |
| `benilla-world/src/modern/volumetrics/mod.rs` | re-exports `fog_composite_pass` and `lamp_halo_pass` for FSR's ordering |
| `benilla-world/src/modern/volumetrics/node.rs` | `prepare_froxels` stands down on `FsrHeld`, so the scatter ping-pong flips only with its pass |
| `benilla-world/src/ffx_glow.rs` | `FfxBackdrop::decode`; quarter targets sized from `FsrView::display`; the compose pair; `ffx_glow_pass` is `pub(crate)` and reads `FsrState::output`; the backdrop ground reads `FgPresent` then `FsrState::output`; `compose_world_final` |
| `benilla-world/src/particles/buffer.rs` | `EffectMotion`, `EffectDrawSpec::motion`, `EffectQuads::motion` (kept parallel by `begin`/`commit`, only while `begin_effect_frame` sees an `FsrView`: `tracks_motion`), `EffectBatch::moving`/`motion_mut`/`tracks_motion`, `WorldEffectDraw::tracks_motion` |
| `benilla-world/src/particles/render.rs` | the motion vertex stream, extracted and uploaded only while tracked (`EffectMeta::motion_valid`); `MergedDraw`'s view and pipeline key; `EffectMeta` and `EffectPipeline` accessors for the velocity pass |
| `benilla-world/src/particles/quads.rs` | `push_particle_motion` beside `expand_quads` |
| `benilla-world/src/particles/sim.rs` | each emitter's last placement, pruned to this frame's; `PerVertex` particle motion, both while tracked |
| `benilla-world/src/rig_palette.rs` | `RigPaletteExtract::bone_watermark`/`live_bytes`, the previous-pose copy's bound |
| `benilla-world/src/ribbons.rs` | the head's motion (`PerVertex`) |
| `benilla-world/src/ground_fx.rs` | the decal centre's motion (`Rigid`) |
| `benilla-world/src/water_fx/mod.rs` | `EffectMotion::Static` |
| `benilla-world/src/liquid/drift.rs` | `EffectMotion::Static` |
| `benilla-world/src/weather/precip/mod.rs` | `EffectMotion::Static` |
| `benilla-world/src/worldview.rs` | `mod fsr_view;` and `fsr_view::plugin(app)` |
| `benilla-app/src/world_backdrop.rs` | `effective_scale`, `fsr_active`, `fsr_mip_bias`; `sync_fsr_view`; the `frameGeneration` arm of `on_cvar` |
| `benilla-app/src/ui_pass.rs` | `PresentsHeldFrames` on the player-UI camera |
| `benilla-app/src/debug_panel/mod.rs` | `PresentsHeldFrames` on the egui overlay camera |
| `benilla-app/src/target/ring.rs` | the ring's motion (`Rigid`) |
| `benilla-app/src/weapon_trail.rs` | the blade edge's motion (`PerVertex`) |
| `benilla-app/src/entities/chain_beam.rs` | why the beam stays `Static` |
| `benilla-app/src/cvars/table.rs` | the `frameGeneration` row in its fork-owned block |
| `benilla-app/src/cvars/mod.rs` | `GRAPHICS_CVARS`; the default weld and the observer arm in the tests |
| `benilla-app/src/perf/journal.rs` | `gpu_fsr`, `gpu_fg`, `gpu_velocity`, `rendered`, `presented` |
| `benilla-app/src/pipe_warm/mod.rs` | the FSR lanes named as warmed at `RenderStartup` |
| `benilla-app/src/ui_script/options_tests.rs` | the frame-generation row's tests and counts |
| `benilla-app/assets/ui/OptionsFrame.xml` | `BENILLA_TOOLTIP_FRAME_GENERATION`, Render Scale's tooltip, `RowFrameGeneration` in both windows |
| `benilla-app/src/ui_script/mod.rs` | `feed_framerate` counts presented frames (`FgCadence::presents_per_update`) |
| `../third_party/bevy_winit/src/state.rs` | the `AfterUpdate` call after `App::update`, where the web's held frame runs the next `Main` |
