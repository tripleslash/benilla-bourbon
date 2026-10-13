#!/usr/bin/env python3
"""The in-loop verify: fmt everywhere, clippy and test on the changed crates and their dependents.

The Python twin of scripts/check.sh; that header holds the why, and the twins change together.
Escalates to scripts/gates.py.

    python scripts/check.py

CHECK_FULL=1 goes straight to gates.py.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED_LINE = re.compile(r"panicked at|^---- .* ----$|^test .* FAILED$|^error(\[E[0-9]+\])?:")


def say(text=""):
    print(text, flush=True)


def capture(cmd):
    """(exit code, stdout with trailing newlines cut, as `$(...)` does); stderr dropped."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return 127, ""
    return p.returncode, p.stdout.decode("utf-8", "replace").rstrip("\n")


def read_lines(path):
    try:
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def mktemp(prefix):
    fd, path = tempfile.mkstemp(prefix=prefix + ".", dir=os.environ.get("TMPDIR") or tempfile.gettempdir())
    os.close(fd)
    return path


def gates(root):
    """Hand over to the full chain, as `exec scripts/gates.sh` does."""
    sys.stdout.flush()
    sys.exit(subprocess.run([sys.executable, os.path.join(root, "scripts", "gates.py")]).returncode)


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    # Check the checkout you stand in when it is one of this repo, else this file's (as gates.py does).
    rc, root = capture(["git", "rev-parse", "--show-toplevel"])
    if rc != 0 or not root or not os.path.isfile(os.path.join(root, "scripts", "check.py")):
        root = HERE
    os.chdir(root)

    if os.environ.get("CHECK_FULL", "") == "1":
        gates(root)

    # The round's change set: the fork point against main, plus staged, unstaged and untracked files.
    rc, base = capture(["git", "merge-base", "HEAD", "main"])
    if rc != 0:
        base = capture(["git", "rev-parse", "HEAD"])[1]
    paths = set()
    for cmd in (["git", "diff", "--name-only", base, "HEAD"],
                ["git", "diff", "--name-only", "HEAD"],
                ["git", "ls-files", "--others", "--exclude-standard"]):
        paths.update(line for line in capture(cmd)[1].split("\n") if line)
    changed = sorted(paths, key=lambda p: p.encode("utf-8"))

    if not changed:
        say("check: no changes against main and a clean tree \u2014 nothing to verify")
        return 0

    # Map files to crate dirs; anything unmappable escalates to the full chain.
    dirs = set()
    full_reason = ""
    for f in changed:
        if f.startswith("docs/") or f.endswith(".md"):
            continue
        m = re.match(r"^crates/([^/]*)/.", f)
        if m:
            dirs.add("crates/" + m.group(1))
            continue
        full_reason = f

    if full_reason:
        say("check: '%s' is outside the crate map \u2014 the whole workspace could be affected." % full_reason)
        say("check: escalating to the full chain: scripts/gates.py")
        gates(root)

    dirlist = sorted(dirs, key=lambda d: d.encode("utf-8"))
    if not dirlist:
        say("check: docs-only round (%d file(s)) \u2014 no gate can change; fmt only" % len(changed))
        if subprocess.run(["cargo", "fmt", "--all", "--", "--check"]).returncode != 0:
            return 1
        say("check: green (docs only)")
        return 0

    # Changed crates and their reverse-dependency closure, off cargo metadata.
    try:
        meta = json.loads(subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=root, check=True).stdout)
    except (OSError, subprocess.CalledProcessError, ValueError):
        say("check: cargo metadata failed \u2014 falling back to the full chain")
        gates(root)
    names = {p["name"] for p in meta["packages"]}
    by_dir, deps = {}, {}
    for p in meta["packages"]:
        rel = os.path.relpath(os.path.dirname(p["manifest_path"]), root)
        by_dir[rel.replace(os.sep, "/")] = p["name"]
        deps[p["name"]] = {d["name"] for d in p["dependencies"] if d["name"] in names}
    missing = [d for d in dirlist if d not in by_dir]
    if missing:
        say("check: changed path(s) not in the workspace map (%s) \u2014 full chain" % " ".join(missing))
        gates(root)
    scope = {by_dir[d] for d in dirlist}
    grew = True
    while grew:
        grew = False
        for name, ds in deps.items():
            if name not in scope and ds & scope:
                scope.add(name)
                grew = True
    pkg_list = sorted(scope)
    pkgs = " ".join(pkg_list)

    pflags = []
    for p in pkg_list:
        pflags += ["-p", p]
    say("check: scope = " + pkgs)
    say("check:   (changed crates + everything that depends on them; the full chain runs once, at land)")

    log = mktemp("benilla-check")
    skips = mktemp("benilla-check-skips")
    try:
        def run(name, cmd, sets=None):
            """Each gate logs to a file read afterwards."""
            env = dict(os.environ, **(sets or {}))
            with open(log, "wb") as f:
                try:
                    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
                except OSError:
                    f.write(("%s: command not found\n" % cmd[0]).encode("utf-8"))
                    rc = 127
            if rc != 0:
                say("CHECK FAILED: " + name)
                # The failing test or error first: a test gate's tail is the doc-test noise after it.
                lines = read_lines(log)
                for line in [l for l in lines if FAILED_LINE.search(l)][:40]:
                    say(line)
                for line in lines[-30:]:
                    say(line)
                sys.exit(1)
            say("check ok: " + name)

        run("fmt", ["cargo", "fmt", "--all", "--", "--check"])
        run("clippy", ["cargo", "clippy"] + pflags + ["--all-targets", "--", "-D", "warnings"])
        # --no-fail-fast: one red test binary does not hide the failures in the ones after it.
        run("test", ["cargo", "test", "--no-fail-fast"] + pflags, {"BENILLA_SKIP_LOG": skips})
        # Data-gated tests skip without the install or the addon corpus and libtest swallows their line.
        lines = read_lines(skips)
        if lines:
            say("  test: %d data-gated tests SKIPPED on this machine \u2014" % len(lines))
            counts = {}
            for line in lines:
                counts[line] = counts.get(line, 0) + 1
            # sort | uniq -c | sort -rn: the most frequent first, ties in reverse byte order.
            rows = sorted(counts, key=lambda k: k.encode("utf-8"), reverse=True)
            rows.sort(key=lambda k: -counts[k])
            for row in rows:
                say("    %d \u00d7 %s" % (counts[row], row))
            say('    (they run where the data is: docs/CONTRIBUTING.md, "Setting up")')

        say("CHECK GREEN (scoped: %s)" % pkgs)
        say("  (the full chain, scripts/gates.py, runs once on the tree that lands; a run before it repeats it)")
        return 0
    finally:
        for path in (log, skips):
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    sys.exit(main())
