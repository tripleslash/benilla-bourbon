#!/usr/bin/env python3
"""One-time toolchain for the browser build (no sudo, nothing system-wide).

The Python twin of scripts/web-setup.sh; that header holds the why, and the twins change together.

    python scripts/web-setup.py

Re-runnable; each step is skipped when already done. WASI_SDK_VERSION and BINARYEN_VERSION override
the releases.
"""

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLOSING = """
ready. The build is now:

  WASI_SDK=$PWD/tools/wasi-sdk trunk build --release      # -> web/dist
  python scripts/stage-packs.py                           # -> web/site

or, for the build with worker threads (nightly + -Zbuild-std + shared memory; same output dirs,
and it verifies the emitted module really came out threaded):

  python scripts/web-build-threads.py

and the game data is a separate, far slower, one-time job:

  cargo run --release -p benilla-packer --bin benilla-pack -- \\
      build --data "/path/to/WoW 1.12.1/Data" --out web/packs

Serve both locally with:

  cargo run --release -p wenilla-host -- --www web/dist --packs web/packs
  cargo run --release -p wenilla-host -- --www web/dist --packs web/packs --no-ranges  # as a static host behaves"""


def say(text=""):
    print(text, flush=True)


def err(text):
    sys.stdout.flush()
    print(text, file=sys.stderr, flush=True)


def run(cmd, capture=False):
    """Run a program; return (exit code, captured stdout or "")."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None, universal_newlines=capture)
    except FileNotFoundError:
        err("%s: command not found" % cmd[0])
        return 127, ""
    return p.returncode, (p.stdout or "")


def host():
    """(wasi-sdk platform, binaryen platform, host name), the platforms empty when unsupported."""
    system = platform.system()
    machine = platform.machine()
    name = "%s-%s" % (system, machine)
    if system == "Linux" and machine == "x86_64":
        return "x86_64-linux", "x86_64-linux", name
    if system == "Linux" and machine == "aarch64":
        return "arm64-linux", "aarch64-linux", name
    if system == "Darwin" and machine == "arm64":
        return "arm64-macos", "arm64-macos", name
    if system == "Darwin" and machine == "x86_64":
        return "x86_64-macos", "x86_64-macos", name
    if system == "Windows" or system.startswith(("MINGW", "MSYS", "CYGWIN")):
        return "x86_64-windows", "x86_64-windows", name
    return "", "", name


def fetch_untar(url, dest):
    """curl | tar xz --strip-components=1 -C dest"""
    os.makedirs(dest, exist_ok=True)
    with tempfile.TemporaryFile() as tmp:
        with urllib.request.urlopen(url) as resp:
            shutil.copyfileobj(resp, tmp)
        tmp.seek(0)
        with tarfile.open(fileobj=tmp, mode="r:gz") as tf:
            members = []
            for m in tf.getmembers():
                parts = [p for p in m.name.split("/") if p not in ("", ".")]
                # Nothing that would land outside dest.
                if len(parts) < 2 or ".." in parts or m.name.startswith("/"):
                    continue
                m.name = "/".join(parts[1:])
                if m.islnk():
                    link = [p for p in m.linkname.split("/") if p not in ("", ".")]
                    if len(link) < 2 or ".." in link:
                        continue
                    m.linkname = "/".join(link[1:])
                members.append(m)
            kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tf.extractall(dest, members=members, **kwargs)


def main():
    os.chdir(str(ROOT))

    wasi_version = os.environ.get("WASI_SDK_VERSION") or "27"
    wasi_dir = "tools/wasi-sdk"
    plat, bplat, host_name = host()

    say("== rustup target")
    rc, _ = run(["rustup", "target", "add", "wasm32-unknown-unknown"])
    if rc:
        return rc

    say("== trunk")
    if shutil.which("trunk"):
        rc, text = run(["trunk", "--version"], capture=True)
        if rc:
            return rc
        lines = text.splitlines()
        fields = lines[0].split() if lines else []
        say("trunk %s already installed" % (fields[1] if len(fields) > 1 else ""))
    else:
        rc, _ = run(["cargo", "install", "trunk", "--locked"])
        if rc:
            return rc

    say("== wasi-sdk %s" % wasi_version)
    if os.path.isfile(wasi_dir + "/share/wasi-sysroot/lib/wasm32-wasip1/libsetjmp.a"):
        say("already at %s" % wasi_dir)
    elif not plat:
        err("unsupported host %s; set WASI_SDK to an unpacked SDK instead" % host_name)
        return 1
    else:
        url = ("https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-%s/wasi-sdk-%s.0-%s.tar.gz"
               % (wasi_version, wasi_version, plat))
        os.makedirs(wasi_dir, exist_ok=True)
        say("downloading %s" % url)
        fetch_untar(url, wasi_dir)

    say("== binaryen (Trunk fetches its own wasm-opt; this is for invoking it by hand)")
    binaryen_version = os.environ.get("BINARYEN_VERSION") or "130"
    opts = ("tools/binaryen/bin/wasm-opt", "tools/binaryen/bin/wasm-opt.exe")
    if any(os.path.isfile(p) and os.access(p, os.X_OK) for p in opts):
        say("already at tools/binaryen")
    elif bplat:
        url = ("https://github.com/WebAssembly/binaryen/releases/download/version_%s/binaryen-version_%s-%s.tar.gz"
               % (binaryen_version, binaryen_version, bplat))
        fetch_untar(url, "tools/binaryen")
    else:
        say("  no binaryen release for this host - Trunk's own wasm-opt still runs")

    say(CLOSING)
    return 0


if __name__ == "__main__":
    sys.exit(main())
