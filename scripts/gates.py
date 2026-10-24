#!/usr/bin/env python3
"""The land gate chain: every gate in order, stopping at the first failure.

The Python twin of scripts/gates.sh; that header holds the why, and the twins change together.
Runs doc-links.py and pass-span-lint.py, and memoizes on this file's own hash.

    python scripts/gates.py

GATES_FORCE=1 runs the chain anyway, WOW_DATA picks the install (set and empty: none) and TMPDIR
holds the scratch files. BENILLA_RE names the benilla-re checkout the reference gate compares
`reference/` with.
"""

import datetime
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = time.monotonic()
FAILED_LINE = re.compile(r"panicked at|^---- .* ----$|^test .* FAILED$|^error(\[E[0-9]+\])?:")
DOCS_ONLY = re.compile(r"^(docs/|[^/]*\.md$|\.github/(ISSUE_TEMPLATE/|pull_request_template\.md$))")


def seconds():
    """bash's $SECONDS."""
    return int(time.monotonic() - T0)


def say(text=""):
    print(text, flush=True)


def capture(cmd, env=None):
    """(exit code, stdout with trailing newlines cut, as `$(...)` does); stderr dropped."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    except OSError:
        return 127, ""
    return p.returncode, p.stdout.decode("utf-8", "replace").rstrip("\n")


def quiet(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    except OSError:
        return 127


def tmpdir():
    return os.environ.get("TMPDIR") or tempfile.gettempdir()


def mktemp(prefix):
    fd, path = tempfile.mkstemp(prefix=prefix + ".", dir=tmpdir())
    os.close(fd)
    return path


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


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def short_head():
    rc, out = capture(["git", "rev-parse", "--short", "HEAD"])
    return out if rc == 0 else "-"


def append_line(path, line):
    try:
        with open(path, "ab") as f:
            f.write((line + "\n").encode("utf-8"))
    except OSError:
        pass


def after_first(s):
    """${s#*|}"""
    return s.split("|", 1)[1] if "|" in s else s


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    # Gate the checkout the cwd is in, not this file's: outside a checkout of this repo, gate this file's own.
    rc, root = capture(["git", "rev-parse", "--show-toplevel"])
    if rc != 0 or not root or not os.path.isfile(os.path.join(root, "scripts", "gates.py")):
        root = HERE
    os.chdir(root)
    say("gating: " + root)

    stamp = "target/.gates-green"
    corpus = "+corpus" if os.path.isdir(os.path.join(root, "wow-addons-vanilla")) else ""

    def tree_key():
        tmpidx = mktemp("benilla-gates-idx")
        os.remove(tmpidx)
        env = dict(os.environ, GIT_INDEX_FILE=tmpidx)
        t = ""
        try:
            rc, out = capture(["git", "read-tree", "HEAD"], env)
            t += out
            if rc == 0:
                rc, out = capture(["git", "add", "-A", "."], env)
                t += out
                if rc == 0:
                    _rc, out = capture(["git", "write-tree"], env)
                    t += out
        finally:
            if os.path.exists(tmpidx):
                os.remove(tmpidx)
        if not t:
            return ""
        _rc, rustc = capture(["rustc", "-V"])
        try:
            with open(os.path.join(root, "scripts", "gates.py"), "rb") as f:
                sha = hashlib.sha1(f.read()).hexdigest()
        except OSError:
            sha = ""
        return "%s|%s|%s" % (t, rustc, sha)

    def resolve_wow():
        return capture(["cargo", "run", "-q", "-p", "benilla-formats", "--example", "where"])[1]

    def docs_only_delta(a, b):
        """A delta only under docs/, in top-level *.md files or in the issue and PR templates keeps the stamped verdict."""
        if quiet(["git", "cat-file", "-e", a + "^{tree}"]) or quiet(["git", "cat-file", "-e", b + "^{tree}"]):
            return False
        _rc, out = capture(["git", "diff-tree", "-r", "--name-only", a, b])
        return not any(line and not DOCS_ONLY.match(line) for line in out.split("\n"))

    smoke_line = "  (a clean run is the fourth gate: scripts/smoke.py \u2014 live logout/re-login round trip)"

    key_start = tree_key()
    if os.environ.get("GATES_FORCE", "") != "1" and key_start and os.path.isfile(stamp):
        with open(stamp, "rb") as f:
            old = f.read().decode("utf-8", "replace").rstrip("\n")
        old_key, _, old_wow = old.rpartition("|")
        if not _:
            old_key = old_wow = old
        memo = ""
        if old_key == key_start:
            memo = "this exact tree already passed"
        elif after_first(old_key) == after_first(key_start) \
                and docs_only_delta(old_key.split("|")[0], key_start.split("|")[0]):
            memo = "this tree differs from one that passed only under docs/ or in top-level *.md, which no gate reads"
        if memo and old_wow == resolve_wow() + corpus:
            say("ALL GATES GREEN (memoized \u2014 %s; GATES_FORCE=1 re-runs)" % memo)
            if os.path.isdir("target"):
                append_line("target/.gates-timing", "%s\t%s\tchain\t0\tmemo" % (utc_stamp(), short_head()))
            say(smoke_line)
            return 0

    log = mktemp("benilla-gates")
    skips = mktemp("benilla-gates-skips")
    try:
        timing = "target/.gates-timing"
        chain_t0 = seconds()
        tree_sha = short_head()

        def note_timing(gate, secs, verdict):
            if os.path.isdir("target"):
                append_line(timing, "%s\t%s\t%s\t%s\t%s" % (utc_stamp(), tree_sha, gate, secs, verdict))

        def report_skips(rung):
            """Each data-gated skip the tests logged, grouped, after a test rung; then the file is emptied."""
            lines = read_lines(skips)
            if not lines:
                return
            say("  %s: %d data-gated tests SKIPPED on this machine \u2014" % (rung, len(lines)))
            counts = {}
            for line in lines:
                counts[line] = counts.get(line, 0) + 1
            # sort | uniq -c | sort -rn: the most frequent first, ties in reverse byte order.
            rows = sorted(counts, key=lambda k: k.encode("utf-8"), reverse=True)
            rows.sort(key=lambda k: -counts[k])
            for row in rows:
                say("    %d \u00d7 %s" % (counts[row], row))
            say('    (they run where the data is: docs/CONTRIBUTING.md, "Setting up")')
            open(skips, "wb").close()

        def run(name, cmd, sets=None, unset=()):
            t0 = seconds()
            env = os.environ.copy()
            for k in unset:
                env.pop(k, None)
            env.update(sets or {})
            with open(log, "wb") as f:
                try:
                    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
                except OSError:
                    f.write(("%s: command not found\n" % cmd[0]).encode("utf-8"))
                    rc = 127
            secs = seconds() - t0
            if rc != 0:
                say("GATE FAILED: %s (%d s)" % (name, secs))
                note_timing(name, secs, "FAILED")
                # The failing test or error first: a test gate's tail is the doc-test noise after it.
                lines = read_lines(log)
                for line in [l for l in lines if FAILED_LINE.search(l)][:40]:
                    say(line)
                for line in lines[-30:]:
                    say(line)
                sys.exit(1)
            say("gate ok: %s (%d s)" % (name, secs))
            note_timing(name, secs, "ok")

        def script(name):
            return [sys.executable, os.path.join(root, "scripts", name)]

        run("fmt", ["cargo", "fmt", "--all", "--", "--check"])
        run("clippy", ["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"])

        wow_data = resolve_wow()
        require_data = ""
        if wow_data and corpus:
            require_data = "1"
        else:
            say("gates: NOTE \u2014 install or addon corpus not found here (install='%s', corpus=%s): "
                "data-gated tests skip silently on this machine"
                % (wow_data or "none", "yes" if os.path.isdir(os.path.join(root, "wow-addons-vanilla")) else "no"))

        re_env = os.environ.get("BENILLA_RE", "")
        if re_env and not os.path.isdir(os.path.join(re_env, "tables")):
            say("GATE FAILED: reference (BENILLA_RE=%s holds no benilla-re checkout)" % re_env)
            sys.exit(1)
        rc, common = capture(["git", "rev-parse", "--git-common-dir"])
        primary = os.path.dirname(os.path.abspath(common)) if rc == 0 and common else ""
        candidates = [re_env, os.path.join(root, "..", "benilla-re")]
        if primary:
            candidates.append(os.path.join(primary, "..", "benilla-re"))
        re_checkout = next((c for c in candidates if c and os.path.isdir(os.path.join(c, "tables"))), "")
        if re_checkout:
            run("reference", script("sync-reference.py") + [re_checkout, "--check"])
        elif require_data or os.environ.get("BENILLA_REQUIRE_DATA", "") == "1":
            say("GATE FAILED: reference (no benilla-re checkout: set BENILLA_RE, or clone it beside the repo)")
            sys.exit(1)
        else:
            say("gate SKIPPED: reference (no benilla-re checkout found: BENILLA_RE, ../benilla-re)")
            note_timing("reference", 0, "SKIPPED")

        sets = {"BENILLA_SKIP_LOG": skips}
        if require_data:
            sets["BENILLA_REQUIRE_DATA"] = "1"
        run("test", ["cargo", "test", "--no-fail-fast", "--workspace"], sets)
        report_skips("test")

        run("test-no-install", ["cargo", "test", "--no-fail-fast", "--workspace"],
            {"WOW_DATA": ""}, unset=("BENILLA_REQUIRE_DATA",))

        run("doc-links", script("doc-links.py"))
        run("message-catalog", script("gen-message-catalog.py") + ["--check"])
        run("pass-span-lint", script("pass-span-lint.py"))

        sets = {}
        if wow_data:
            sets["WOW_DATA"] = wow_data
        if corpus:
            sets["BENILLA_ADDON_CORPUS"] = os.path.join(root, "wow-addons-vanilla")
        if require_data:
            sets["BENILLA_REQUIRE_DATA"] = "1"
        sets["BENILLA_SKIP_LOG"] = skips
        run("player-tests",
            ["cargo", "test", "-p", "benilla-formats", "-p", "benilla-app", "--no-default-features", "--lib"], sets)
        report_skips("player-tests")

        if quiet(["cargo", "run", "-q", "-p", "benilla-formats", "--example", "where"]) == 0:
            run("enforcer", ["cargo", "run", "-q", "-p", "benilla-worldview"], {"WOW_WORLDVIEW_CHECK": "10"})
        else:
            say("gate SKIPPED: enforcer (no WoW install found \u2014 the engine boot check needs one;")
            say("             `cargo run -p benilla-formats --example where` says where it looked)")
            note_timing("enforcer", 0, "SKIPPED")

        run("enforcer-no-install", ["cargo", "run", "-q", "-p", "benilla-worldview"],
            {"WOW_DATA": "", "WOW_WORLDVIEW_CHECK": "5"})

        chain_secs = seconds() - chain_t0
        say("ALL GATES GREEN (%d s; per-gate seconds in %s)" % (chain_secs, timing))
        note_timing("chain", chain_secs, "ok")

        # Stamp only if the tree still hashes as it did at the start.
        key_end = tree_key()
        if key_end and key_end == key_start and os.path.isdir("target"):
            try:
                with open(stamp, "wb") as f:
                    f.write(("%s|%s%s\n" % (key_end, wow_data, corpus)).encode("utf-8"))
            except OSError:
                pass

        say(smoke_line)
        return 0
    finally:
        for path in (log, skips):
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    sys.exit(main())
