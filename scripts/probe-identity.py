#!/usr/bin/env python3
"""Who a scripted run logs in as; loaded by smoke.py, cine.py and summon-live.py, never run alone.

The Python twin of scripts/probe-identity.sh; that header holds the why, and the twins change together.

`probe_identity(tag, root)` returns an Identity (user, password, char, declared, host, auth_port)
and prints the identity it chose; a refusal prints why and returns None. `probe_server_or_skip(tag,
ident)` prints a skip and returns False when the auth port, or a local server's world port 8085,
does not answer.
"""

import os
import socket


class Identity:
    def __init__(self, user, password, char, declared):
        self.user = user
        self.password = password
        self.char = char
        self.declared = declared
        # The server: WOW_HOST is `host` or `host:port`, the client's own spelling (realmlist.rs).
        host = os.environ.get("WOW_HOST") or "localhost"
        self.host = host.split(":", 1)[0]
        self.auth_port = host.rsplit(":", 1)[1] if ":" in host else "3724"


def _first_value(lines, key):
    """sed -n 's/^KEY=//p' | head -1"""
    for line in lines:
        if line.startswith(key + "="):
            return line[len(key) + 1:]
    return ""


def probe_identity(tag, root):
    path = os.path.join(root, ".probe-identity")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            lines = f.read().decode("utf-8", "replace").split("\n")
        user = _first_value(lines, "WOW_USER")
        password = _first_value(lines, "WOW_PASS")
        char = _first_value(lines, "WOW_CHAR")
        if not user or not password or not char:
            print("%s: REFUSING \u2014 %s is missing one of WOW_USER, WOW_PASS, WOW_CHAR" % (tag, path), flush=True)
            return None
        ignored = "".join(" " + v for v in ("WOW_USER", "WOW_PASS", "WOW_CHAR") if os.environ.get(v))
        if ignored:
            print("%s: ignoring%s \u2014 this checkout declares its own account (.probe-identity)" % (tag, ignored),
                  flush=True)
        print("%s: %s / %s (declared by .probe-identity)" % (tag, user, char), flush=True)
        return Identity(user, password, char, True)
    user = os.environ.get("WOW_USER", "")
    password = os.environ.get("WOW_PASS", "")
    char = os.environ.get("WOW_CHAR", "")
    if not user or not password or not char:
        print("%s: REFUSING \u2014 no account to log in as. A scripted run takes WOW_USER, WOW_PASS" % tag)
        print("       and WOW_CHAR from the environment, all three, or from a .probe-identity file")
        print("       at the tree root, naming a test account on your server whose login kicks")
        print('       nobody (docs/CONTRIBUTING.md, "Setting up").', flush=True)
        return None
    print("%s: %s / %s (from the environment)" % (tag, user, char), flush=True)
    return Identity(user, password, char, False)


def probe_server_or_skip(tag, ident):
    ports = [ident.auth_port]
    if ident.host in ("localhost", "127.0.0.1"):
        ports.append("8085")
    for port in ports:
        try:
            socket.create_connection((ident.host, int(port))).close()
        except (OSError, ValueError):
            print("%s: SKIPPED \u2014 nothing listening on %s:%s (auth %s; a local" % (tag, ident.host, port, ident.auth_port))
            print("       vmangos also serves the world on 8085). Start your 1.12.1 server, or point")
            print("       WOW_HOST at it.", flush=True)
            return False
    return True
