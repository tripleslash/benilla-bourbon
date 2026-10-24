#!/usr/bin/env python3
"""The browser build WITH THREADS.

The Python twin of scripts/web-build-threads.sh; that header holds the why, and the twins change together.

    python scripts/web-build-threads.py

Overrides, as in the bash script: BENILLA_NIGHTLY, BENILLA_MAX_MEMORY, WASI_SDK, CARGO_TARGET_DIR,
RUSTFLAGS (prepended), BENILLA_KEEP_NAMES=1.
"""

import glob
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLOSING = """
built. Serve it somewhere cross-origin isolated - the page cannot construct a SharedArrayBuffer
without COOP/COEP, and without one this module does not instantiate at all:

  cargo run --release -p wenilla-host -- --www web/dist --packs web/packs

(`wenilla-host::static_site` and `Trunk.toml`'s `[serve] headers` both set the same pair
`web/_headers` sets on the site, so all three agree.)"""


def say(text=""):
    print(text, flush=True)


def err(text):
    sys.stdout.flush()
    print(text, file=sys.stderr, flush=True)


def env_or(name, default):
    value = os.environ.get(name)
    return value if value else default


def run(cmd, capture=False):
    """Run a program; return (exit code, captured stdout or "")."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None, universal_newlines=capture)
    except FileNotFoundError:
        err("%s: command not found" % cmd[0])
        return 127, ""
    return p.returncode, (p.stdout or "")


def du_h(path):
    """Roughly what `du -h` prints for one file: binary units, rounded up."""
    size = float(os.path.getsize(path))
    units = ["", "K", "M", "G", "T"]
    i = 0
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    if i == 0:
        return "%d" % size
    if size < 10:
        return "%.1f%s" % (math.ceil(size * 10) / 10, units[i])
    return "%d%s" % (math.ceil(size), units[i])


def rewrite_specifier(path):
    with open(path, "r+b") as f:
        data = f.read().replace(b'from "wasi_snapshot_preview1"', b'from "./wasi_stubs.js"')
        f.seek(0)
        f.write(data)
        f.truncate()


def main():
    os.chdir(str(ROOT))
    cwd = os.getcwd()

    os.environ["RUSTUP_TOOLCHAIN"] = env_or("BENILLA_NIGHTLY", "nightly")
    os.environ["CARGO_UNSTABLE_BUILD_STD"] = "std,panic_abort"
    os.environ["WASI_SDK"] = env_or("WASI_SDK", os.path.join(cwd, "tools", "wasi-sdk"))
    os.environ["CARGO_TARGET_DIR"] = env_or("CARGO_TARGET_DIR", os.path.join(cwd, "target", "threads"))

    max_memory = env_or("BENILLA_MAX_MEMORY", "4294967296")
    # Shared, imported and exported memory plus the five exports wasm-bindgen's threading transform reads.
    thread_flags = [
        "-Ctarget-feature=+atomics,+bulk-memory,+mutable-globals",
        "-Clink-arg=--shared-memory",
        "-Clink-arg=--import-memory",
        "-Clink-arg=--export-memory",
        "-Clink-arg=--max-memory=%s" % max_memory,
        "-Clink-arg=--export=__heap_base",
        "-Clink-arg=--export=__wasm_init_tls",
        "-Clink-arg=--export=__tls_base",
        "-Clink-arg=--export=__tls_size",
        "-Clink-arg=--export=__tls_align",
    ]
    os.environ["RUSTFLAGS"] = os.environ.get("RUSTFLAGS", "") + " " + " ".join(thread_flags)
    toolchain = os.environ["RUSTUP_TOOLCHAIN"]

    say("== toolchain")
    rc, text = run(["rustc", "-vV"], capture=True)
    for line in text.splitlines():
        say("   " + line)
    if rc:
        return rc
    say("   RUSTFLAGS=" + os.environ["RUSTFLAGS"])
    say("   CARGO_TARGET_DIR=" + os.environ["CARGO_TARGET_DIR"])

    # `-Zbuild-std` compiles std out of rust-src.
    _rc, text = run(["rustup", "component", "list", "--toolchain", toolchain, "--installed"], capture=True)
    if "rust-src" not in text.splitlines():
        say("installing rust-src for %s (-Zbuild-std needs std's sources)" % toolchain)
        rc, _ = run(["rustup", "component", "add", "rust-src", "--toolchain", toolchain])
        if rc:
            return rc

    say("== trunk build --release")
    rc, _ = run(["trunk", "build", "--release"])
    if rc:
        return rc

    say("== stage-packs")
    rc, _ = run([sys.executable, str(ROOT / "scripts" / "stage-packs.py")])
    if rc:
        return rc

    # Verify the module actually came out threaded, from the build's own output.
    wasms = sorted(glob.glob("web/dist/wenilla_bg*.wasm"))
    if not wasms:
        err("no wenilla_bg*.wasm in web/dist - the build did not get that far")
        return 1
    wasm = wasms[0]

    say("== verify: %s (%s)" % (os.path.basename(wasm), du_h(wasm)))
    rc, report = run([sys.executable, str(ROOT / "scripts" / "wasm-memory.py"), wasm], capture=True)
    if rc:
        return rc
    report = report.rstrip("\n")
    for line in report.split("\n"):
        say("   " + line)

    # All three have to hold. Each is a different way to end up single-threaded without an error.
    fail = False
    if "shared=YES" not in report:
        err("   FAIL: linear memory is not shared - no worker can attach to it.")
        fail = True
    if "IMPORTED memory" not in report:
        err("   FAIL: linear memory is DEFINED, not imported - every instance would make its own heap")
        err("         and the workers would share nothing with the page.")
        fail = True
    if "+atomics" not in report:
        err("   FAIL: no +atomics in target_features - RUSTFLAGS did not reach the cargo Trunk spawns.")
        fail = True
    if fail:
        return 1
    say("   OK: shared, imported, atomics - this module can take worker threads.")

    # BENILLA_KEEP_NAMES=1: re-emit the glue from the cargo artifact without wasm-opt, so stack traces
    # name Rust functions. For localhost and a console, never a site.
    if os.environ.get("BENILLA_KEEP_NAMES", "") == "1":
        say("== BENILLA_KEEP_NAMES=1: re-emitting unoptimized, named glue")
        artifact = os.environ["CARGO_TARGET_DIR"] + "/wasm32-unknown-unknown/release/wenilla.wasm"
        if not os.path.isfile(artifact):
            err("   no cargo artifact at " + artifact)
            return 1
        # The wasm-bindgen Trunk downloaded, whose version MUST match the crate pin (Trunk.toml).
        home = os.path.expanduser("~")
        local = env_or("LOCALAPPDATA", home + "/AppData/Local")
        candidates = [
            local + "/trunkrs/trunk/cache/wasm-bindgen-0.2.126/wasm-bindgen.exe",
            home + "/.cache/trunk/wasm-bindgen-0.2.126/wasm-bindgen",
            shutil.which("wasm-bindgen") or "",
        ]
        wbg = ""
        for cand in candidates:
            if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
                wbg = cand
                break
        if not wbg:
            err("   no wasm-bindgen 0.2.126 found (Trunk caches one; or cargo install wasm-bindgen-cli)")
            return 1
        rc, _ = run([wbg, "--target", "web", "--out-dir", "web/dist", "--out-name", "wenilla", "--no-demangle", artifact])
        if rc:
            return rc
        # Same rewrite stage-packs.py does: workers have no import map.
        rewrite_specifier("web/dist/wenilla.js")
        say("   web/dist now carries the NAMED module (%s); web/site still has the optimized one."
            % du_h("web/dist/wenilla_bg.wasm"))

    say(CLOSING)
    return 0


if __name__ == "__main__":
    sys.exit(main())
