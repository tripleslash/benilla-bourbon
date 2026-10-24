# benilla-bourbon

benilla-bourbon is a fork of [samwhosung/benilla](https://github.com/samwhosung/benilla), a
from-scratch World of Warcraft 1.12.1 client in Rust and Bevy. It follows upstream's `main` by
merge, so every upstream commit is in its history, and on top of it it adds:

- **An optional modern renderer** (a *Lighting* setting: Classic or Modern, with Low, Medium,
  High, Ultra and Cinematic tiers):
  - linear HDR lighting, cascaded sun shadows and runtime probe GI;
  - froxel volumetrics, per-zone looks, a moon at night and modern weather;
  - modern water, and FSR 3.1.5 upscaling with frame generation.
- **Asset loading and streaming work:**
  - asset loading that is deferred and moved off the frame thread;
  - streaming that fetches geometry first and holds the body safely while the ground arrives;
  - optional HD and reforged art layers.
- **Turtle WoW 1.18.1 support:** a realm advertising 1.18.1 switches the whole client onto Turtle
  WoW's data, races, interface and protocol at the moment it is entered.
- **wenilla, a browser build:** the same client compiled to WebAssembly, played in a browser tab
  over WebGPU, with threads, a render worker and a streamed content-addressed game-data container.
- **Gameplay, UI and performance fixes** found along the way, many of them fidelity fixes against
  the 1.12.1 binary.

Everything is behind the stock 1.12.1 behaviour where that is possible:

- **Classic lighting** is upstream's renderer, and the modern hooks leave its values bit-exact.
- **A 1.12.1 realm** never reads a Turtle file.
- **The deviations** are written down where they live (see [Deviations](#deviations)).

The classic renderer, the protocol and the stock UI are upstream's. This file describes the
fork's own work as of the export.

## Contents

- [Merging upstream](#merging-upstream)
- [Rendering and graphics](#rendering-and-graphics)
  - [The modern lighting model (P0–P8)](#the-modern-lighting-model-p0p8)
  - [Water](#water)
  - [Sky, night and weather](#sky-night-and-weather)
  - [Culling, LOD and field of view](#culling-lod-and-field-of-view)
  - [Bevy 0.19.1 and FSR 3.1.5](#bevy-0191-and-fsr-315)
  - [Renderer performance](#renderer-performance)
- [Asset loading and streaming](#asset-loading-and-streaming)
- [Turtle WoW 1.18.1](#turtle-wow-1181)
- [The browser build (wenilla)](#the-browser-build-wenilla)
- [Gameplay and UI](#gameplay-and-ui)
- [Settings](#settings)
- [Vendored libraries](#vendored-libraries)
- [Instruments](#instruments)
- [Deviations](#deviations)
- [Credits](#credits)

## Merging upstream

```bash
git remote add upstream https://github.com/samwhosung/benilla.git
git fetch upstream
git merge upstream/main
```

The merge keeps one mechanism per problem. Where upstream and the fork both solve the same problem,
the merge takes the better approach or combines the two, and never leaves both standing
(`docs/METHOD.md`, "This fork"). Three tests are the usual merge tripwires:

- `benilla-assets`'s **`shader_slots`** fails on a WGSL location or binding given twice. An
  upstream shader can merge cleanly and still take a slot the fork's lighting uses.
- **The schedule ambiguity ceilings and the world API wall** in `benilla-app` count what the fork
  adds. Their notes say which rows are the fork's.
- **`HOOKS.md`** in `crates/benilla-world/src/modern/` and `crates/benilla-world/src/fsr/` lists
  every edit the fork makes inside an upstream-owned file. Code comments mark the same places with
  `fork:`.

Code comments in the fork cite this file: `fork: modern lighting (FORK.md, P4)` points at the
phase of the lighting model below, and `fork: fsr (FORK.md)` at the FSR section.

## Rendering and graphics

### The modern lighting model (P0–P8)

`lightingModel` 0 is the stock renderer and 1 is the modern one. `lightingQuality` picks the tier,
and both apply live on Apply. Portraits, UI model panes and the glue screens always render classic.
Modern code lives behind `#ifdef WOW_MODERN` in the shared material shaders and in the
`benilla_modern::*` WGSL libraries, so a classic pipeline compiles none of it.

| Phase | What it adds |
| --- | --- |
| P0 | `lightingModel`/`lightingQuality` CVars, the shared WGSL libraries, `launch_env` (pre-seedable environment for the browser), the engine-only world viewer in the browser (`?worldview=1`). |
| P1 | The world frame goes linear behind a runtime flag (classic stays bit-exact); HDR FFXGlow with bloom, a shoulder tonemap and a cinematic grade (contrast as an S-curve, saturation, toe, vignette). |
| P2 | The sun and moon on an east–south–west arc on the reference's elevation timing (`sun_arc`). Own sun cascades: a CPU-fitted, texel-snapped reverse-Z Depth32Float array with cached static layers; terrain cells, static_gx (WMOs, merged doodads, props) and entity parts (skinned, alpha-cutout) cast; 5-tap PCF receivers. |
| P3 | The baked MCSH shade, blob shadows and exterior MOCV retire in a modern frame. Per-pixel terrain sun, foliage translucency, a top-down sky-occlusion map, per-pixel point lights. |
| P4 | Froxel volumetrics: the DBC fog solved into a participating medium, compute scatter and integrate into a 3D volume with reprojected history. Long-range height fog, WDL silhouettes to the horizon (10 km far plane, a seam dissolve at the view distance), HDR sky with lit clouds, per-zone looks. |
| P5 | Runtime probe GI: a toroidal voxel clipmap of the static world plus a terrain heightfield, cascaded probe grids traced in compute, order-1 SH with hysteresis. |
| P6 | Interiors and local lights: a binned world-space light grid, shadowed local lights as cube faces in the shadow array, an interior probe sub-grid, MOHD ambient, MOCV retired indoors with a floor at 0.9 of the classic light. Indoor and outdoor light classes from WMO group flags, so a street lamp never lights the chapel beside it and the chapel's candles never light the road. |
| P8 | The **Cinematic** tier, about local light: lights derived from what glows (additive emitters and glow cards, weapon enchants, server-spawned braziers, lit windows, lamp glass), twelve contact-hardening cube shadows with stable, contestable, fading slots, a two-level light grid, per-pixel lamp halos, flame flicker and spell lights. |

Beyond the phases:

- **Lighting and exposure:** screen-space AO and indirect light (SSILVB, linear in a modern frame);
  a screen-metered auto-exposure (`autoExposure`); a moonlight floor; a golden-noon key ramp.
- **Zone looks:**
  - a default look calibrated against reference shots, plus EERIE for the dark zones;
  - Tirisfal's and Darkshore's own looks;
  - COLD and COLD_NIGHT for snow, and a hall row for Ironforge.
- **Additions from the Everwood fork:**
  - **foliage wind:** grass parts around the nearest units and leaves sway; the wind code is from
    WarcraftXL, and its attribution header travels with it;
  - **rain on surfaces:** wet ground with a shelter map and screen-space reflections on puddles;
  - **lava light**, and a monotone cubic sky gradient.

### Water

`waterDetail` keeps the reference CVar's value space:

- **0, Classic:** upstream's 1.12.1 combine.
- **2, Modern:** a port of the Cataclysm liquid programs (ProcWater, ProcWaterAbove,
  ProcWaterBelow, ProcMagma), read out of the shipped programs. It has:
  - a planar mirror reflection (a second scene render with an oblique near clip, armed only for
    visible water and thinned of small objects) under a screen-space trace;
  - shoreline foam, refraction and lava;
  - a ripple displacement map stamped by rain and wading bodies.
- **3, Modern (Experimental):** a clear-water look with Beer–Lambert absorption, Schlick Fresnel,
  wind ripple octaves on the sea and stable deep-water shade.

The bump, foam and magma-noise textures the Cataclysm programs sample exist in no 1.12.1 install.
They are synthesised (`crates/benilla-world/src/liquid/art.rs`), committed as bytes and held to
their generator by a test. Nothing Blizzard authored is bundled.

MCNK liquid layers draw as cells of 4×4 chunks instead of one draw per chunk.

### Sky, night and weather

- **Night:** a moon arc from sunset to sunrise that keys the scene with the zone's night rows, and
  ground-fog banks laid on the terrain heightfield.
  - A look's `moon_key` gain doubles the moon only where the night needs it (the snow night,
    Darkshore, Duskwood and the dark zones, Tirisfal); elsewhere the authored night diffuse
    carries it.
  - On the eerie looks (`bank_moon`) the night banks are ground fog: they take the whole night
    bank, lit by the whole moon.
- **Stars:** the night-sky `Stars.m2` batches draw as authored (second UV set and stage); a modern
  frame fades the star dome at the horizon and tints it to the zone's sky.
- **Clouds:** a worker bakes the modern cloud layer from the reference coverage tile; weather
  thickens it, and a thunderstorm closes the sky.
- **Modern weather:**
  - storms darken and cool without going grey; heavy rain sinks into a teal-blue;
  - lightning in heavy rain, and soft procedural mist;
  - rain streaks that shrink toward the eye;
  - precipitation drawn as points built in the vertex stage.
- **Fixes:** a BLP mip chain ends at the last complete level (black rain streaks); a zero-duration
  global sequence holds its first key; the ghost sky's DeathClouds spin.

### Culling, LOD and field of view

- **Terrain horizon occlusion,** ported from the client (`0x686000`, `0x686180`, `0x681b50`). It
  applies outdoors and per portal window inside WMOs; it parks hidden rigs and drops shadow casters
  below the horizon. `/console showCull` toggles it.
- **Terrain chunk LOD** (levels 2 and 3, from `0x6c51b0`, stitched across cell and tile edges), with
  the shadow and GI passes drawing the same level.
- **Animation LOD:** distant rigs are re-posed every 2nd, 3rd or 4th frame.
- **Field of view:** a 90° horizontal field at 16:9, Hor+.
- **Mouse pick:** doodad collision hulls no longer occlude it; drawn opaque batches do, as in the
  client (`0x480df0`).

### Bevy 0.19.1 and FSR 3.1.5

The fork moved to Bevy 0.19.1 (render graph as systems) on a vendored wgpu 29.0.4, and added
**FSR 3.1.5** (`crates/benilla-world/src/fsr/`):

- the upscaler, motion vectors for skinned models and the effect lane, and frame generation with
  its own held-frame cadence;
- an RCAS sharpening slider and Render Scale marks at 67% and 100%;
- instruments: `WOW_FSR_DEBUG=views|residual`.

AMD's licence for the FidelityFX shaders is `fsr/LICENSE-FIDELITYFX.txt`.

### Renderer performance

Each item was measured, and the commit that made it carries the numbers. The largest:

- **Depth and draws:**
  - a static_gx depth prepass with Equal-tested shading: 17.4 → 35.7 fps at Cinematic;
  - per-level voxel box merging and region index reordering;
  - far cascades redraw entity casters on alternate frames.
- **Shader work:**
  - constant-index cascade and probe rows (Cinematic GPU 28.6 → 14.4 ms);
  - one point-light walk per vertex at most, and none where the grid answers;
  - a probe trace with one warp per ray direction and throttled off-screen probes;
  - an AO-only SSILVB variant.
- **CPU and caching:**
  - shadow and GI caches keyed on revisions instead of per-frame hashes;
  - light clusters computed on the CPU.
- **Particles and billboards:**
  - emitters step in parallel on the compute pool, and their pools expand in parallel chunks;
  - a pool whose atlas columns are a power of two draws on the GPU while no camera runs FSR, from an
    instance buffer and a uniform (no storage buffer, so WebGL2 can draw it); `WOW_FX_CPU=1` keeps
    every pool on the CPU;
  - a rigged placement's billboard batches are skinned through its rig palette and faced by the
    animate kernel, as the reference does (`0x714260`), instead of CPU-faced cards;
    `WOW_BILLBOARD_CARDS=1` restores the cards.
- **Per-frame work:**
  - PreUpdate and the fixed-timestep schedules run single-threaded (`WOW_MT_PREUPDATE`,
    `WOW_MT_FIXED` restore them); Bevy's light-visibility pass runs only while a Bevy light exists;
    the IO pool takes a quarter of the threads, so compute keeps its own;
  - native release builds allocate through mimalloc;
  - rain and snow drift and creature ground clamps run in parallel chunks;
  - change marks only where something changed: the spyglass projection, card and marker re-seats,
    hidden cards, leaf rig anchors, physics sync of moved bodies only, probe rows as deltas;
  - the UI's extract walk skips while the UI model's mutation count stands (`ModelCell`);
    `WOW_UI_VERIFY=1` walks anyway and reports a skip that would have drawn stale.

## Asset loading and streaming

- **Deferred and off-thread loading:**
  - BLP, ADT, M2 (geometry, rig and every animation clip) and WMO decode run on workers
    (`frame_budget::offload`), metered where no worker exists;
  - the character skin composite blends on a worker;
  - collider builds are budgeted;
  - a deferred sprite reopens the FrameXML extract gate, so late art is picked up.
- **Streaming order and safety:**
  - fetches are ordered geometry first, on a five-tier ladder by distance from the focus tile;
  - the residency window requests the tile ahead before the body reaches it;
  - provisional colliders (the WDL heightfield, WMO bounding boxes) hold a body up while real
    collision streams;
  - a body walking onto a building still arriving is held under the loading cover.
- **Content layers:** each is its own container and a live mount on the web.
  - **HD:** the updated character models from patch-A, with patch-T's vetted 2× character art.
    Textures are upscaled to the compositor's 2× without premultiplying alpha. Geoset groups
    outside the 1.12 managed set (feet, waist, eye glow) are drawn, and the skin composite scales
    its tiles to the base skin.
  - **Turtle:** the Turtle WoW 1.18.1 patch layer.
  - **Reforged:** Project Reforged's letter patches.
- **Format fixes:**
  - MPQ files encrypted under their name read by name;
  - a BLP mip past the file's end ends the chain;
  - a texture name without an extension asks for its `.blp`;
  - a WMO group's MOGP runs to EOF;
  - a texture stage past the texture-unit table reads its own UV set.
- **`benilla-pack` / `benilla-packer`:** the BENPACK container.
  - Packs are flagged Rangeable or Solid; every frame carries its own codec; a 20-byte index entry
    is keyed on MPQ's own name hash; objects are capped at 24 MiB; output is deterministic.
  - The packer groups by path and by captured boot manifests. `overlay`, `merge`, `append` and
    `sidecars` cut the Turtle, HD and reforged layers and the boot sets.
  - `verify` compares the container byte for byte against the chain.

## Turtle WoW 1.18.1

A realm whose list entry carries the 1.18.1 version tag (`SPECIFYBUILD`, "(1.18.1.7272)") makes the
client switch flavor when the realm is entered (`crate::flavor`). The switch mounts the Turtle
layer, reruns every catalog loader (the `DataLoad` schedule), reloads every chain asset, evicts the
caches that outlived the data, and rebuilds the interface VMs. A failed entry falls back to the
last realm that held. A 1.12.1 realm pays one relaxed atomic load per read for all of this.

- **Protocol:** CMSG_CHAR_CREATE's trailing challenge mask; the language read from
  ChrRaces.BaseLanguage; a GameObject template keyed by OBJECT_FIELD_ENTRY; transports created
  without a pose; an unasked template miss evicts.
- **Characters:**
  - ten races (Goblin and High Elf) from Turtle's own tables, its CharBaseInfo pairs and its class
    list;
  - per-race glue scenes; booth batches drawn at their Stand alpha;
  - High Elf creature type, PvP sides and helm infixes.
- **Interface:**
  - Turtle's FrameXML rows spliced into `benilla.toc` off its FrameXML.toc, and Turtle's own
    GlueXML with challenge modes;
  - `GetBuildInfo` answers 1.18.1; Turtle_General and Turtle_GroupUI are always loaded;
  - Turtle's own options window, carrying the fork's Graphics rows behind an Apply button;
  - CameraZoomIn/Out, maxStack in GetAuctionSellItemInfo, macro icons by texture name, a rankless
    trainer subtext, and the arena leftovers its FrameXML names.
- **World:**
  - uniqueId reuse and exact-double placements resolved;
  - exterior WMO groups drawn without MOCV;
  - zero-scale creature displays, extensionless baked NPC textures, junk creature-skin records;
  - a map with no Map.dbc row does not hold the loading cover; the dungeon finder's teleport row.
- **Not loaded:**
  - **the donation shop:** `Turtle_ShopUI` rows are filtered out;
  - **the radio:** its http stream sound kits are dropped once, with a warning.

## The browser build (wenilla)

`crates/wenilla` is the browser entry crate. Build and host it as `web/README.md` describes.

- **Platform layer:**
  - `webenv` for page configuration; localStorage as the state folder;
  - the C Lua VM compiled with the wasi-sdk sysroot;
  - `web_time` clocks; full-screen keyboard lock; navigator.clipboard copy and paste;
  - screenshots that download; mouse side buttons kept on the page.
- **Network:**
  - an async `Conn` and protocol, a WebSocket transport, and the sequencer as a task;
  - `wenilla-host`: a WebSocket gateway with a port allowlist and loopback pinning, plus local
    `--www`/`--packs` serving with Range emulation.
- **Game data:**
  - the BENPACK container is fetched and decoded in a Web Worker, with Range support detected at
    runtime and retries on 429 and 5xx;
  - a service worker caches the data;
  - boot, world and glue read sets are prefetched behind a loading overlay;
  - `/config.json` names the gateway and the realmlist.
- **Threads** (nightly, `-Zbuild-std`, shared memory):
  - forks of `bevy_tasks`, `bevy_ecs`, `bevy_platform`, `bevy_render`, `async-executor`,
    `event-listener`, `parking` and `web-task`;
  - `may-park` (page-aware locks), and `spawn_pooled` for Send work.
- **Render worker:**
  - the render sub-app, the WebGPU device and an OffscreenCanvas live on a Web Worker;
  - the page lends the world for extract;
  - the GPU is kept at most two frames deep.
- **Memory:**
  - `PageAlloc`, a mimalloc-shaped global allocator with per-thread heaps, so no thread starves the
    page thread;
  - released GPU buffers, textures and query sets are destroyed explicitly;
  - a memory-pressure valve steps the view distance down near the 4 GiB limit.
- **Audio:** an AudioWorklet drains a ring that a worker running kira fills; `decodeAudioData`
  decodes the music.
- **GPU health:** a governor commits the canvas size once a resize settles; a `GPUDevice.lost`
  watchdog steps a recovery tier down.
- **Browser-specific fixes:** WGSL that Chrome and Firefox accept, a Firefox matrix-copy
  workaround, CPU light clusters.

## Gameplay and UI

- **Transports:**
  - riders stay aboard through step-ups, forced moves, logout and login;
  - every outbound movement word carries ON_TRANSPORT, and other passengers are dead-reckoned in
    the deck's frame;
  - the crossing's route screen is upstream's; the fork loads its tables as a data loader (again
    on a Turtle WoW chain change) and its textures through the pending sprite;
  - a worldport that carries the player on a transport not yet boarded seats them on arrival;
  - ship and zeppelin props play the travel phase the transport tick stores.
- **Macros (a fork addition):**
  - retail option syntax (`/cast [@mouseover,help,nodead][] …`, conditions, `known:`,
    `equipped:`, modifiers) and `#show`/`#showtooltip`;
  - `@mouseover` names a hovered unit frame's unit;
  - plain `/cast` and `CastSpellByName` stay 1.12's.
- **Fidelity fixes:**
  - auction list events; the mounted camera pivot; GetBattlefieldStatus;
  - unit reaction colour clamping; UnitCanAssist; aura durations at login;
  - retarget events; tooltips and fonts linked live to their inherited font; the quest log's
    event objective;
  - nameplates before a name answers; glue screen positions against both clients' GlueXML;
  - UI model panes loading their named file;
  - battleground exit (movement, scoreboard); world entry waiting for spell tooltip views.
- **Robustness:** a failed command logs an error instead of panicking; attach and teardown use
  `try_` commands; a player's body waits for its equipment for at most 3 s.
- **The era AddOns options page,** one checkbox per installed addon.

## Settings

The fork's own rows, all in both options windows (ours and Turtle WoW's):

- **Graphics Preset** (`graphicsQuality`), one ladder over fifteen rows: Classic is the reference
  client and Cinematic is the shipped default.
- **Lighting and post-processing:** Lighting, Lighting Quality, Ambient Occlusion, Indirect
  Lighting, Auto Exposure.
- **Water and world:** Water Detail; Foliage Wind and its strength; Wet Surfaces; Lava Light.
- **Models:** Use updated character models; Use reforged models (Turtle WoW only).
- **Display:** Render Scale; RCAS Sharpening; Frame Generation; Gamma and Brightness as two
  sliders.
- **Time:** a Server Time Offset slider.

Every Graphics row stages to Apply.

## Vendored libraries

Every crate under `third_party/` is either a crates.io release with a short, documented patch, or
(`may-park`) the fork's own. Each is wired in through `[patch.crates-io]` in the root `Cargo.toml`,
whose comments say why. Each has a `BENILLA.md` (wgpu: `PATCH.md`) with the hunk list and a
verifying diff against the registry crate.

| Crate | Upstream version | Why it is vendored |
| --- | --- | --- |
| `wgpu` | 29.0.4 | WebGPU backend only. Exposes `subgroups` and `texture-formats-tier1` (FSR 3.1.5 in the browser), rewrites WGSL Chrome rejects (an out-of-range `f32::MAX` literal, missing `enable subgroups;`), reads device limits and features once, creates pipelines asynchronously, labels per-frame objects only in debug builds, and destroys released buffers, textures and query sets (the browser frees them only on garbage collection). |
| `bevy_tasks` | 0.19.1 | Threads in the browser: the `multi_threaded` gate asked `target_arch` where it meant "has threads", threads come from a pluggable spawner, and `block_on` never parks the page's thread. |
| `bevy_ecs` | 0.19.1 | The same threading gates in `par_iter`/`QueryState`, and one logic change: a query's archetype update tests only the new archetypes when they are fewer than its rarest component's set (a crowd stall on the page thread). |
| `bevy_platform` | 0.19.1 | Its `Mutex`/`RwLock` spin on the page's thread and park elsewhere, through `may-park`, because the render worker shares Bevy's locks with the page. |
| `bevy_render` | 0.19.1 | Wgpu handles kept off worker scopes; Skip cameras keep their view attachments and the window's texture; per-task canvas textures on the web; mesh slabs destroyed when let go; a mesh slice as one lookup; `RenderAsset::DEFERRABLE` so the UI's meshes never wait on the upload budget; the render worker's hooks; explicit destruction of released GPU objects. |
| `bevy_winit` | 0.19.1 | The runner keeps its per-frame query and system states; an `AfterUpdate` hook (frame generation's held frame); wake, may-update and lend-world hooks for the render worker. |
| `async-executor` | 1.14.0 | Its state locks may not park the page's thread on `wasm32 + atomics`. |
| `event-listener` | 5.4.1 | Same rule: the `std`-path list lock is reached on every channel send/receive. |
| `parking` | 2.2.1 | Same rule: `Unparker::unpark` takes a mutex, usually from the page. |
| `web-task` | 1.1.3 | A finished `spawn_local` task left an `Atomics.waitAsync` waiter on freed memory. |
| `kira` | 0.12.4 | Upstream's spatial-gain patch, plus the fork's: the cpal backend stays on cpal 0.17 so the browser build resolves one `alsa-sys`. |
| `alsa-0.9` | 0.9.1 | Upstream's. |
| `may-park` | (own) | Which threads may park, and `std` locks that ask. |

`third_party/addons/` bundles the code and databases of two third-party addons by Shagu (MIT):

- **pfQuest 7.0.1 (enUS)** on every realm.
- **pfQuest-turtle** on Turtle WoW realms only.

pfQuest's images (`img/` and `compat/close.tga`) are left out, because some of its icons are Blizzard art and this repository
ships no game data. Without it the addon's map and tracking icons draw blank. A player who wants
them installs pfQuest's own release into their `AddOns` folder: a player's own copy of an addon
with the same name wins over the bundled one. Each addon is off until the player ticks it on the
AddOns screen. The reference ships no addon, so the bundle is a deviation, written at
`benilla_formats::bundled_addons`.

## Instruments

The fork's measuring tools, all off unless asked for:

- **World viewer and probes:**
  - the browser world viewer (`?worldview=1`, `?turtle=1`, `?worldview_pose=`, `?modern_look=`,
    live levers through `launch_env`);
  - the perf probe (`?perf=` / `WOW_PERF`, `PERF_VIEWS`, `PERF_VERTS`, `PERF_OVERDRAW`,
    `WOW_PERF_LOOP`);
  - scripted flights (`WOW_WORLDVIEW_FLY`), and a server scene with no server (`WOW_FX_SCENE`).
- **Profiling:**
  - the per-system profile (`--features trace_systems`, `WOW_SYSTEM_PROFILE=1`), in the browser
    too;
  - opt-in shader hot reload (`--features shader_reload`).
- **Browser diagnostics:** `webdiag` (`?diag=1`), a WebGPU object census (`?gpucensus=1`), a
  render-worker timeline (`?rwtrace=1`), and the shadow and light debug views
  (`?modern_shadowdebug=1..6`).
- **Script twins:** the build and gate scripts in bash, PowerShell 5.1 and Python, each tree in
  its own language.

## Deviations

Every place the fork differs from 1.12.1 is written where it lives: a CVar's `Deviates` row or a
`Deviation:` comment. `docs/DEVIATIONS.md` gathers them, regenerated from the code by
`scripts/genmap.sh`. The fork's headline deviations:

- **Shipped defaults:**
  - the modern renderer at Cinematic, Modern (Experimental) water, frame generation on;
  - farclip 1000 yd (a slider up to 1500), `lodDist` 500, anisotropic 16;
  - the updated and reforged character models on.
- **View:** a 90° Hor+ field of view; animation LOD for distant rigs; a long same-map teleport
  covered by the loading screen.
- **Login and glue:**
  - the login screen has no Realmlist editor or Quit button, and no web-link buttons;
  - the TOS and EULA stand accepted.
- **Browser build:** enforces no script memory limit (a tab cannot quit).
- **Fork additions:** retail macro options; bundled opt-in addons.
- **Turtle WoW:** no donation shop or radio; its options window stages to Apply.

## Credits

benilla is samwhosung's, and [benilla-re](https://github.com/samwhosung/benilla-re) is the reverse
engineering it is built on.

This fork carries work by:

- **Arnesen:** the first browser build. This covers the Lua VM on wasm, the async transport, the
  web host, localStorage state and the boot prefetch.
- **Thorsten42:** HD model texture stages, tooltip, font and quest-log fixes, and the
  character-pane render layers.
- **kalistoking:** TargetByName in the chunk, nameplate bounds, and the Turtle WMO MOGP fix.
- **chairmanpow:** the AddOns options page and battleground fixes, plus attach and teardown
  robustness.
- **The [manila](https://github.com/Adrinalin4ik/manila) fork (Adrinalin4ik) and its contributors:**
  many performance and fidelity fixes, ported and credited commit by commit.

It also takes from:

- **pkuzic's Everwood graphics fork:** foliage wind, rain on surfaces, lava light, the sky
  gradient, the preset ladder and the lamp-halo technique.
- **pkuzic's benilla-lighting:** flame flicker, spell lights and the spark floor.
- **WarcraftXL, by iThorgrim:** the wind code.
- **physics-rs:** the wgpu subgroups patch and FSR's widened formats.
- **The benilla-twow and benilla-octo forks:** Turtle WoW fixes.
- **pfQuest, by Shagu.**
