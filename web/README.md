# wenilla — benilla in a browser

The same client, compiled to WebAssembly: log in, pick a character, and play a 1.12.1
server from a browser tab — no install. Everything a browser cannot do natively is replaced
behind a seam the native build already had:

| native | browser |
|---|---|
| raw TCP to realmd/mangosd | a WebSocket to a gateway, which proxies it to the server (`benilla-protocol::transport::Conn`) |
| the MPQ patch chain on disk | a **BENPACK container** — the same corpus repackaged into ~4,700 static objects plus a 1.6 MB index — fetched from any dumb object store (`benilla-formats::webpack` behind `Chain` on wasm) |
| Lua 5.1 in C, `longjmp` for errors | the same C, compiled against the wasi-sdk sysroot with LLVM's wasm SJLJ; 64-bit `lua_Integer` kept (`third_party/mlua-sys`) |
| `WOW_USER`/`WOW_PASS`/`WOW_CHAR`/`WOW_HOST` env | a `window.__wenilla_env = {user, pass, host, …}` object the page sets before `init()`, else the query string `?user=…&pass=…&char=…&host=…` — but the three credential keys are read from the query only when the env object carries `dev_query_creds: "1"` (the dev `index.html` sets it; a login-gated host such as `wenilla-realm` fetches the credentials itself and never puts them in a URL) |
| the window | `<canvas id="benilla">` in `web/index.html` |

## Requirements

- **WebGPU** for the world (the shared light blob and the skin palettes are storage buffers,
  which WebGL2 has none of). Chrome/Edge on Windows and macOS, Safari 26+, Firefox 141+;
  Chrome on Linux behind `chrome://flags/#enable-unsafe-webgpu`. WebGPU also needs a secure
  context — `localhost`, or https. A WebGL2 build exists (`data-cargo-features="webgl2"`) and
  reaches the glue screens only.
- On the build machine: stable Rust, `rustup`, ~10 GB of `target/`; `scripts/web-setup.sh`
  fetches the rest (the `wasm32-unknown-unknown` target, Trunk, wasi-sdk into `tools/wasi-sdk/`).
  No sudo. Windows (Git Bash), Linux and macOS.
- To *build a container*: a 1.12.1 (build 5875) `Data/` directory, ~35 minutes and ~4 GB of disk.
- To *play*: a reachable realmd (3724) + mangosd (8085) behind a WebSocket gateway.

## Build and run

```bash
scripts/web-setup.sh                                    # once
WASI_SDK=$PWD/tools/wasi-sdk trunk build --release      # → web/dist, then web/site
```

Game data is a **separate, far slower, one-time job** — it reads 6.3 GiB of MPQ and writes 3.3 GiB
of container, so it is not part of the build and must not be re-run per code change:

```bash
cargo run --release -p benilla-packer --bin benilla-pack -- \
    build --data "/path/to/WoW 1.12.1/Data" --out web/packs
cargo run --release -p wenilla-host -- --www web/dist --packs web/packs
# then open http://127.0.0.1:8090/
```

`--packs` serves the container as plain files, exactly as a static host would. `--no-ranges`
additionally strips every request's `Range` header, which is what a static host measurably does — the
only way to exercise the client's whole-object path locally, and worth doing before any upload,
because a fallback that is never run is a fallback that does not work.

`wenilla-host --data <Data>` still serves the old per-name `GET /data/{name}` route out of a live
MPQ chain. **The current client does not use it**; it is how a session runs without spending half an
hour packing a container, and it goes away once the container path has run in anger.

**`wenilla-host` is for local testing only.** `/data` serves the game files you point it at to
anyone who can reach the socket, with no login, so it binds to `127.0.0.1` by default and must
not be exposed on the open internet — the files are Blizzard's, and whoever serves them is
distributing them. Hosting for other players is
[`wenilla-realm`](../crates/wenilla-realm/README.md), which puts the same routers behind a session
cookie (set `CLIENT_PACKS` to a container).

## Staging to a static host

The hosted tree is static assets and **no Worker script** — no invocations, no request billing.
`scripts/stage-packs.sh` (Trunk's post-build hook) assembles `web/site/` out of the two things
that must not be in the same directory:

```
web/dist/     Trunk's output           cleared on every build
web/packs/    benilla-pack's output    3.3 GiB, rebuilt only when the game files change
web/site/   what a static host serves    hardlinks to dist/, plus data -> ../packs
```

```bash
an upload to the static host      # reads the host's config
```

Then set the gateway in `web/packs/config.json` (`{"ws": "wss://gw.example:443", "realmlist": "…"}`)
and re-upload — it is ~100 bytes served `no-cache`, so it repoints the client without a wasm rebuild.
That is the whole reason the address is not compiled in.

**The data is public and unauthenticated on such a site**, unlike the login-gated `/data` route it
replaces. Whoever serves these files is distributing Blizzard's.

## Things to know

- **How a read works.** `index.bpi` maps a name's MPQ hash pair to `(pack, offset, length)`, so
  `Chain::contains` is a binary search over bytes already in memory rather than the blocking `HEAD`
  it used to be — measured at **2,145 of them in one world entry**, ~125 s of frozen tab at 100 ms
  RTT. A read that misses fetches its pack, keeps the decoded body on the **JS heap** (`web/packs.js`
  — 3.3 GiB of corpus must never touch a 32-bit wasm address space), and thereby warms the whole
  locality group. HTTP `Range` support is probed, not assumed; the whole-object path is never wrong,
  only bigger.
- **The boot prefetch.** `web/boot-manifest.json` and `web/world-manifest.json` are captured read
  sets (`?boottrace=1`) and are checked in. The packer translates them into `data/boot-packs.json`
  *and* groups those blobs together in the container, which is what keeps the prefetch honest:
  ungrouped, the 403 boot names landed in 141 packs — 91 MB fetched to reach 14.7 MB. Grouped, it is
  **22 packs and 12.3 MB**. `benilla-pack sidecars --packs web/packs` re-translates after a
  recapture without re-packing.
- **Hosting behind a login** — `crates/wenilla-realm` wraps this host's routers behind a session
  cookie (the wasm, `/data/*` and `/ws/*` all need it; same-origin requests carry the cookie by
  themselves), renders its own play page that fetches the player's hidden game credentials over
  the session and hands them to the client through `window.__wenilla_env`, and adds an admin
  panel for the server. The packaged one-VM hosting is github.com/Arnesen/wenilla-realm.
- **Sound** starts on the first click or key in the page — browsers suspend every AudioContext
  until a user gesture; `index.html` resumes it then.
- **Persistence** — the state folder (`config.toml`, macros, key bindings, layouts, chat
  settings, saved variables, the remembered account name) lives in the browser's `localStorage`,
  one entry per file keyed by its path (`benilla:/benilla-config/config.toml`, …). Per browser
  and per origin, like a `benilla-config/` beside each native install; the server never sees it.
  Clearing site data resets it. Pack bytes live in **Cache Storage** instead, namespaced by the
  container's build id so a re-pack invalidates rather than mixes.
- **Performance** — the client is CPU-bound on its single wasm thread (Bevy's task pools are
  single-threaded on `wasm32`): ~11 ms/frame in an outdoor scene on a laptop iGPU, scaling
  with visible objects, not pixels. The Trunk link runs `wasm-opt -O3` (+6 %). Any CVar
  can be pinned from the URL for that session (`?renderScale=0.75&farclip=200&worlddetail=0`) —
  `renderScale` helps a weak GPU, the others a weak CPU. On a high-refresh display an 11 ms
  frame alternates between 2 and 3 refresh intervals, which reads as judder; a 60 Hz mode is
  smoother until the web build is multithreaded.
- **Debugging a crash**: the panic hook prints to the console; drop `--strip-debug` from the Trunk
  link's `data-wasm-opt-params` to keep the wasm name section so the stack trace has symbols.
  `index.html` also prints the real WGSL compile diagnostics (wgpu's browser backend does not ask
  for them). `?noprefetch=1` skips the pack warm; `?boottrace=1` also arms the read trace.
- **Porting notes** — the things that differ from native and bit this port: std
  `Instant`/`SystemTime`/`env::vars_os` *panic* on wasm (use `bevy::platform::time::Instant` /
  `web-time`); the browser's WGSL compiler enforces the uniformity rule naga does not (no
  `textureSample` in a per-fragment branch — `textureSampleGrad`/`textureSampleLevel` instead);
  wasm-bindgen crate and CLI versions must match exactly (`Trunk.toml`'s `[tools] wasm_bindgen`
  against the `Cargo.lock` pin); wasi-libc's `fd_prestat_get` stub must return EBADF
  (`web/wasi_stubs.js`) or libc's constructor exits.
