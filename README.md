<div align="center">
  <h1>benilla-bourbon</h1>
  <p><b>A fork of <a href="https://github.com/samwhosung/benilla">benilla</a>, the from-scratch World of Warcraft 1.12.1 client in Rust and Bevy, with a modern renderer, Turtle WoW support and a browser build</b></p>
</div>

## What this fork changes

benilla-bourbon follows upstream benilla's `main` by merge, so everything in upstream's README
below still holds. On top of it, the fork adds the following. [`FORK.md`](FORK.md) describes each
change in full, why each vendored library is vendored, and where the fork deviates from 1.12.1.

- **Modern lighting** (optional; Classic stays the reference client): linear HDR, sun shadow
  cascades, probe GI, froxel volumetrics, SSILVB ambient occlusion, auto-exposure, per-zone
  looks, point lights and lamps, a night sky with a moon, stars and clouds.
- **Water and weather:** a port of the Cataclysm client's water and lava with reflections,
  ripples and shoreline foam; rain, storms, lightning, mist and wet surfaces.
- **Culling and LOD:** the client's terrain horizon occlusion and chunk LOD, animation LOD, and a
  90° Hor+ field of view.
- **Engine:** Bevy 0.19.1 on a vendored wgpu 29.0.4, FSR 3.1.5 upscaling with frame generation
  and RCAS sharpening, GPU particle pools, and measured CPU and GPU performance work.
- **Asset loading:** BLP, ADT, M2 and WMO decode off the main thread, streaming ahead of the
  player under an upload budget, optional updated (HD) and reforged model layers, and a
  content-addressed container for the game data.
- **Turtle WoW 1.18.1:** its patch archives and data, realm detection with a live data switch,
  its races, its GlueXML and its options window.
- **Browser build (wenilla):** the whole client compiled to WebAssembly with worker threads and a
  render worker, streaming game data from a static host and reaching the realm through a
  WebSocket gateway.
- **Gameplay and UI:** transport riding fixes, retail macro options (`[@mouseover]`,
  `#showtooltip`), an AddOns options page, a server time offset, fidelity fixes, and an opt-in
  bundled pfQuest.
- **Settings:** a Graphics Preset ladder and the fork's own rows in both options windows; every
  default that differs from 1.12.1 is listed in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).
- **Vendored libraries:** patched copies of wgpu, five Bevy crates and a few support crates under
  `third_party/`, each with a note saying why.

### Running it in a browser

The browser build needs nothing installed on the player's machine: raw TCP becomes a WebSocket
through a small host (`wenilla-host`), the MPQ chain becomes a content-addressed container the
browser streams from any static file host, and the C Lua 5.1 VM is compiled for wasm with the
wasi-sdk sysroot. WebGPU is required for the world.

```bash
scripts/web-setup.sh                                   # once: wasm target, Trunk, wasi-sdk (no sudo)
WASI_SDK=$PWD/tools/wasi-sdk trunk build --release      # → web/dist/
scripts/stage-packs.sh                                 # → web/site/ (.ps1 and .py twins too)
scripts/web-build-threads.sh                           # …or the same build WITH worker threads, staged

# once per install: 6.3 GiB of MPQ → a 3.3 GiB container of ~4,700 static objects
cargo run --release -p benilla-packer --bin benilla-pack --     build --data /path/to/WoW/Data --out web/packs
# optional, if the install has patch-A.MPQ: the updated character models, own container
#   (extract -> scripts/upscale-hd-textures.py -> build --replace-from; see web/README.md)
cargo run --release -p benilla-packer --bin benilla-pack --     build --data /path/to/WoW/Data/patch-A.mpq --out web/packs-hd --replace-from /tmp/hd-char
cargo run --release -p wenilla-host -- --www web/dist --packs web/packs --packs-hd web/packs-hd
# open http://127.0.0.1:8090/
```

That command is the *local* shape: page, game data and net proxy from one process. A run that
serves files has no login, so keep it on loopback or a private network. Hosted for others it
splits in two: the static tree (`web/site/`) goes to any static file host, and the same binary
runs on the realm's machine as a **gateway and nothing else** (`wenilla-host --bind …`, no
`--www`/`--packs`), relaying `/ws/3724` and `/ws/8085` to realmd and mangosd on loopback.
[`web/README.md`](web/README.md) has the how and the porting notes.

---

*Everything below is upstream benilla's README.*

<div align="center">
  <h1>benilla</h1>
  <p><b>A complete World of Warcraft 1.12.1 client, written from scratch in Rust and <a href="https://bevy.org">Bevy</a></b></p>
  <p>
    <a href="https://discord.gg/wJSJx467G4"><img src="https://img.shields.io/discord/1529280129518538922?style=for-the-badge&logo=discord&logoColor=white&label=discord&color=5865F2" alt="Discord"></a>
    <a href="https://www.youtube.com/playlist?list=PLdCnpZNKxyb8"><img src="https://img.shields.io/badge/devlog-youtube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube devlog"></a>
    <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-blue?style=for-the-badge" alt="License"></a>
  </p>
</div>

benilla plays the whole game: character creation, questing and professions, dungeons and raids,
battlegrounds and honor, groups, guilds, trade, mail and the auction house, on the stock 1.12
interface and with your addons. It connects to a 1.12.1 server over the original protocol and
reads the game's data from your own 1.12.1 install. Every file format, the network protocol and
the interface engine are written from scratch, with no original client code, no third-party WoW
crates and no bundled game assets.

What the real client does is read at its bytes in
[benilla-re](https://github.com/samwhosung/benilla-re), the reverse engineering benilla is built
against, by the same client addresses the comments here cite.

## What's inside

- **Formats:** readers for the whole asset stack (the MPQ patch chain, BLP, DBC, ADT, WDT, WDL, M2
  and WMO), wired into Bevy as an asset source.
- **World:** terrain streamed out to the horizon, portal-culled buildings with interior lighting,
  doodads and ground clutter, swimmable water, sky and weather, and the client's own day/night
  lighting, fog and gamma.
- **Models:** GPU-skinned M2s with the full animation controller, particles, ribbons and the spell
  visuals, and characters end to end: customization, the armor composite, weapons with their
  enchant glows, forms, stealth and mounts.
- **Movement:** networked movement in both directions, the server-granted modes from slow fall to
  roots, a follow camera with collision, boats, zeppelins and taxi flights.
- **Networking:** SRP6 auth through world-session crypto and the object mirror into the ECS,
  covering the game from movement and chat through combat, spells, groups and raids, quests,
  trade, mail, the auction house and battlegrounds.
- **Interface:** a FrameXML and Lua engine that runs the stock 1.12 interface off your install's
  patch chain, and third-party 1.12 addons: Questie, pfUI, Bagnon, Bartender2 and most others
  run. By choice, the options window and the ESC menu follow the Classic Era client's rather than
  1.12's.
- **Audio:** music, ambience and effects under the client's own selection and crossfade rules,
  with interior and underwater transitions and zone reverb.

The format readers and the UI engine core are plain Rust with no Bevy in them, and the world
renderer runs with no game attached. [`docs/MAP.md`](docs/MAP.md) maps every crate and subsystem,
and [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) lists everywhere benilla differs from 1.12.1; both are
generated from the code.

## Status

Complete and fully playable. What is left:

- The long tail of small features that separates a working client from a finished one, tracked
  as [issues](https://github.com/samwhosung/benilla/issues).
- Addons, options and performance, ongoing.

benilla is a faithful 1.12.1 client and the foundation people build on. There are no prebuilt
downloads: a packaged build, a particular server's changes or anything 1.12.1 never had belongs
in a fork, and forks are welcome. GitHub lists
[every public fork](https://github.com/samwhosung/benilla/forks).

Not planned: other expansions or client versions, Warden (anticheat).

## Running it

benilla builds and runs on macOS, Linux and Windows. You need:

- **A 1.12.1 client (build 5875)** for the game data, in any locale. benilla only reads it, and
  shows its text in the client's language; development runs on English clients.
- **A 1.12.1 server.** [vmangos](https://github.com/vmangos/core) is what development runs
  against; cMaNGOS and the other 1.12.1 cores speak the same protocol. benilla does not answer
  Warden, so a server that enforces it disconnects the session after its timeout; vmangos ships
  with Warden off.
- **Stable Rust and a C compiler**, which some dependencies build with: on macOS the
  Xcode command line tools, on Linux the ALSA and udev development packages and pkg-config, on
  Windows the MSVC build tools that the Rust installer sets up.

```sh
WOW_DATA=/path/to/WoW/Data cargo run --release -p benilla
```

On Windows, in PowerShell:

```powershell
$env:WOW_DATA="C:\path\to\WoW\Data"; cargo run --release -p benilla
```

Each release is a tag on the [Releases page](https://github.com/samwhosung/benilla/releases):
`git checkout <tag>` first runs that release, and `main` is the development tip.

`WOW_DATA` names the install's `Data` folder; a link to the install named `WoW` at the repo root
does the same (`ln -s /path/to/WoW WoW`, or on Windows a junction, which needs no admin rights:
`New-Item -ItemType Junction -Path WoW -Target C:\path\to\WoW`). The server defaults to
`localhost:3724`, the stock auth port. Point `WOW_HOST` at another (`WOW_HOST=play.example.com`, or
`play.example.com:5000` for a remapped port), or set the `realmList` CVar in `config.toml`; this
fork has no Realmlist button, and the browser build takes the address from the site's
`/config.json`. Credentials go in at the login screen, or set `WOW_USER` and `WOW_PASS` to skip it.

Settings, screenshots and addons live in `benilla-config/` at the repo root: a 1.12 addon goes in
`benilla-config/AddOns/`. [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) has the rest, from the
player build to the tests.

## Contributing

Issues and pull requests are open. [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) says where to
start, what gets in and how a change is judged. Bugs, questions and ideas are welcome on the
[Discord](https://discord.gg/wJSJx467G4) too.

---

Early inspiration and file format guidance came from the
[wowemulation-dev](https://github.com/wowemulation-dev) community, and
[warcraft-rs](https://github.com/wowemulation-dev/warcraft-rs) in particular.

benilla is an independent fan project, not affiliated with or endorsed by Blizzard Entertainment.
It ships **no Blizzard content**: no art, models, sounds, maps, MPQ contents or FrameXML. You
provide your own legally obtained 1.12.1 client, and the stock interface runs off its FrameXML at
runtime. The few files under `crates/benilla-app/assets/ui/` are our own, not copies of it:
adapters over stock files, and the settings windows and script error log benilla draws itself.

World of Warcraft is a trademark of Blizzard Entertainment, Inc. Our own code is licensed under
[MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option. The kira audio engine under
`third_party/` is vendored under the same two licenses, with its author's copyright notices beside it.
