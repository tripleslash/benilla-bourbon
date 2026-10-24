#!/usr/bin/env python3
"""Assemble web/site/, the directory a static host serves.

The Python twin of scripts/stage-packs.sh; that header holds the why, and the twins change together.

    python scripts/stage-packs.py

    STAGE_PACKS_LINK=hardlink   link every pack individually instead of linking the directory.
    LOOSE_DATA, LOOSE_DATA_TURTLE   installs whose loose *.html files are staged into the containers.
"""

import fnmatch
import os
import shutil
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PACKS = "web/packs"
PACKS_HD = "web/packs-hd"
PACKS_TURTLE = "web/packs-turtle"
PACKS_REFORGED = "web/packs-reforged"
SITE = "web/site"
ADDONS = "third_party/addons"
WASM_PART_MAX = 24 * 1024 * 1024
KEEP = ("data", "data-hd", "data-turtle", "data-reforged")


def out(text):
    print(text, flush=True)


def err(text):
    sys.stdout.flush()
    print(text, file=sys.stderr, flush=True)


def visible(name):
    """What a bash `*` glob would match: no dot-names."""
    return not name.startswith(".")


def is_link(path):
    st = os.lstat(path)
    reparse = getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    return stat.S_ISLNK(st.st_mode) or bool(reparse)


def remove(path):
    """rm -rf that removes a junction or symlink itself and never walks into its target."""
    if is_link(path):
        try:
            os.rmdir(path)
        except OSError:
            os.unlink(path)
    elif os.path.isdir(path):
        with os.scandir(path) as it:
            children = [e.path for e in it]
        for child in children:
            remove(child)
        os.rmdir(path)
    else:
        try:
            os.unlink(path)
        except PermissionError:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            os.unlink(path)


def link_tree(src, dst):
    """cp -lr src/. dst/"""
    os.makedirs(dst, exist_ok=True)
    with os.scandir(src) as it:
        entries = list(it)
    for e in entries:
        target = os.path.join(dst, e.name)
        if e.is_dir():
            link_tree(e.path, target)
        else:
            os.link(e.path, target)


def stage_loose(src, container):
    """The install's loose *.html files, lowercased into <container>/loose/; the install is only read."""
    if not src or not os.path.isdir(src) or not os.path.isdir(container):
        return
    loose = os.path.join(container, "loose")
    os.makedirs(loose, exist_ok=True)
    for name in sorted(os.listdir(src)):
        f = os.path.join(src, name)
        if visible(name) and name.endswith(".html") and os.path.isfile(f):
            shutil.copy(f, os.path.join(loose, name.lower()))
    count = len([n for n in os.listdir(loose) if visible(n)])
    out("  staged %d loose file(s) from %s into %s/loose" % (count, src, container))


def link_data(src, name):
    """src = the container directory, name = the name it takes under the site."""
    dst = os.path.join(SITE, name)
    if (os.environ.get("STAGE_PACKS_LINK") or "dir") == "hardlink":
        if os.path.lexists(dst):
            remove(dst)
        link_tree(src, dst)
        return
    if os.path.exists(dst):
        return
    if os.name == "nt":
        import _winapi

        # A junction needs neither administrator rights nor Developer Mode; a symlink does.
        _winapi.CreateJunction(os.path.abspath(src), os.path.abspath(dst))
    else:
        os.symlink("../" + os.path.basename(src), dst)


def list_dir(path):
    with os.scandir(path) as it:
        entries = sorted(it, key=lambda e: e.name)
    for e in entries:
        st = os.lstat(e.path)
        link = is_link(e.path)
        kind = "l" if link else ("d" if stat.S_ISDIR(st.st_mode) else "-")
        arrow = ""
        if link:
            try:
                arrow = " -> " + os.readlink(e.path)
            except OSError:
                pass
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
        out("%s %12d %s %s%s" % (kind, st.st_size, stamp, e.name, arrow))


def main():
    os.chdir(str(ROOT))

    # First candidate that exists and is non-empty; staging outranks dist.
    dist = ""
    for candidate in (os.environ.get("TRUNK_STAGING_DIR", ""), "web/dist/.stage", "web/dist"):
        if candidate and os.path.isdir(candidate) and os.listdir(candidate):
            dist = candidate
            break
    if not dist:
        err("no build found in web/dist or web/dist/.stage - run `trunk build --release` first")
        return 1
    out("staging from " + dist)

    # Workers have no import map: point the glue's bare specifier at the file, before anything links it.
    for name in sorted(os.listdir(dist)):
        js = os.path.join(dist, name)
        if not (visible(name) and name.endswith(".js") and os.path.isfile(js)):
            continue
        with open(js, "r+b") as f:
            data = f.read()
            if b"wasi_snapshot_preview1" not in data:
                continue
            data = data.replace(b'from "wasi_snapshot_preview1"', b'from "./wasi_stubs.js"')
            data = data.replace(b"from 'wasi_snapshot_preview1'", b"from './wasi_stubs.js'")
            f.seek(0)
            f.write(data)
            f.truncate()
        out("  rewrote the wasi_snapshot_preview1 specifier in %s (workers have no import map)" % name)

    if not os.path.isdir(PACKS):
        err("warning: no %s - staging the client with no game data (benilla-pack build)" % PACKS)
    if not os.path.isdir(PACKS_HD):
        err("note: no %s - staging without the updated-character-model layer" % PACKS_HD)
    if not os.path.isdir(PACKS_TURTLE):
        err("note: no %s - staging without the Turtle WoW 1.18.1 layer" % PACKS_TURTLE)
    if not os.path.isdir(PACKS_REFORGED):
        err("note: no %s - staging without the reforged art layer" % PACKS_REFORGED)

    stage_loose(os.environ.get("LOOSE_DATA", ""), PACKS)
    stage_loose(os.environ.get("LOOSE_DATA_TURTLE", ""), PACKS_TURTLE)

    os.makedirs(SITE, exist_ok=True)
    # Everything except the container links.
    for name in os.listdir(SITE):
        if name not in KEEP:
            remove(os.path.join(SITE, name))

    # Hardlink the build; copy where the filesystem refuses a link.
    for name in sorted(os.listdir(dist)):
        if not visible(name):
            continue
        src = os.path.join(dist, name)
        dst = os.path.join(SITE, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy(src, dst)

    # The bundled addons as loose files, with index.txt listing them in bytewise order.
    if os.path.isdir(ADDONS):
        addons_out = os.path.join(SITE, "addons")
        shutil.copytree(ADDONS, addons_out, dirs_exist_ok=True)
        paths = []
        for dirpath, _dirnames, filenames in os.walk(addons_out):
            for fn in filenames:
                if fn == "index.txt":
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), addons_out).replace(os.sep, "/")
                paths.append(os.fsencode(rel))
        paths.sort()
        with open(os.path.join(addons_out, "index.txt"), "wb") as f:
            f.write(b"".join(p + b"\n" for p in paths))
        out("staged %d bundled addon files" % len(paths))

    # Split the wasm module into 24 MiB *.wasm parts plus wenilla_bg.parts.json.
    wasm = os.path.join(SITE, "wenilla_bg.wasm")
    if os.path.isfile(wasm):
        size = os.path.getsize(wasm)
        stem = wasm[:-5]
        i = 0
        with open(wasm, "rb") as f:
            while True:
                buf = f.read(WASM_PART_MAX)
                if not buf:
                    break
                with open("%s.part%02d.wasm" % (stem, i), "wb") as part:
                    part.write(buf)
                i += 1
        # The link, not the build: web/dist keeps its inode.
        os.remove(wasm)
        parts = sorted(n for n in os.listdir(SITE) if fnmatch.fnmatchcase(n, "wenilla_bg.part*.wasm"))
        manifest = '{"bytes":%d,"parts":[%s]}\n' % (size, ",".join('"%s"' % p for p in parts))
        with open(os.path.join(SITE, "wenilla_bg.parts.json"), "wb") as f:
            f.write(manifest.encode("utf-8"))
        out("split wenilla_bg.wasm (%d B) into %d parts" % (size, len(parts)))

    # Nothing at the top of the site may exceed the cap; the packs are capped by the packer.
    with os.scandir(SITE) as it:
        oversize = [
            e.name for e in it
            if e.is_file(follow_symlinks=False) and e.stat(follow_symlinks=False).st_size > WASM_PART_MAX
        ]
    if oversize:
        err("error: asset over a static host's 25 MiB limit (cap here is 24 MiB):")
        for name in oversize:
            err("%s/%s" % (SITE, name))
        return 1

    if os.path.isdir(PACKS):
        link_data(PACKS, "data")
    if os.path.isdir(PACKS_HD):
        link_data(PACKS_HD, "data-hd")
    if os.path.isdir(PACKS_TURTLE):
        link_data(PACKS_TURTLE, "data-turtle")
    if os.path.isdir(PACKS_REFORGED):
        link_data(PACKS_REFORGED, "data-reforged")

    # Counted off the container trees, not through the links.
    with os.scandir(SITE) as it:
        objects = sum(1 for e in it if e.is_file(follow_symlinks=False))
    for tree in (PACKS, PACKS_HD, PACKS_TURTLE, PACKS_REFORGED):
        if os.path.isdir(tree):
            for _dirpath, _dirnames, filenames in os.walk(tree):
                objects += len(filenames)
    out("assets to upload: %d (a typical static host's limit: 20000)" % objects)
    if objects > 20000:
        err("error: over a static host's 20,000-asset limit - drop a layer or rebuild with fewer objects")
        return 1

    out("staged %s:" % SITE)
    list_dir(SITE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
