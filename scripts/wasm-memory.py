#!/usr/bin/env python3
"""Say what a wasm module's linear memory actually is, and which target features it was built with.

The threaded browser build (`scripts/web-build-threads.sh`) has exactly one silent failure mode:
a module that lost `--shared-memory` or `--import-memory` still compiles, still links, still loads
and still runs — single-threaded, with every worker spawn failing in a way the client is designed
to survive. So "it works" is not evidence that threads are on, and this reads the answer out of the
bytes instead.

Three things have to be true and all three are printed:

  shared=YES     the memory is a SharedArrayBuffer. Without it a worker has nothing to attach to.
  IMPORTED       the PAGE supplies the memory, so every instance gets the SAME one. A *defined*
                 memory is created fresh per instantiation, so workers would share nothing — this
                 is the failure the 2026-09-06 spike actually produced.
  +atomics       ... in target_features, alongside +bulk-memory and +mutable-globals.

`wasm-dis` would answer the first question too, but it disassembles the whole module to do it, and
a debug build of this client is 2 GB. This walks the section table and reads three fields.

    python scripts/wasm-memory.py web/dist/wenilla_bg.wasm
"""

import sys

def uleb(f):
    r = 0; s = 0
    while True:
        b = f.read(1)[0]
        r |= (b & 0x7f) << s
        if not (b & 0x80): return r
        s += 7

def name(f):
    n = uleb(f)
    return f.read(n).decode('utf-8', 'replace')

def limits(f):
    flags = f.read(1)[0]
    mn = uleb(f)
    mx = uleb(f) if flags & 0x01 else None
    return flags, mn, mx

path = sys.argv[1]
f = open(path, 'rb')
assert f.read(4) == b'\0asm', 'not a wasm module'
f.read(4)
found = False
while True:
    b = f.read(1)
    if not b: break
    sid = b[0]
    size = uleb(f)
    start = f.tell()
    if sid == 2:  # imports
        for _ in range(uleb(f)):
            m, n = name(f), name(f)
            kind = f.read(1)[0]
            if kind == 0x00: uleb(f)
            elif kind == 0x01: f.read(1); limits(f)
            elif kind == 0x02:
                flags, mn, mx = limits(f)
                found = True
                print(f"IMPORTED memory  {m}.{n}  flags=0x{flags:02x} "
                      f"shared={'YES' if flags & 0x02 else 'no'} "
                      f"min={mn} pages ({mn*64//1024} MiB) "
                      f"max={mx if mx is None else f'{mx} pages ({mx*64//1024} MiB)'}")
            elif kind == 0x03: f.read(1); f.read(1)
            # A tag (the exception-handling proposal): an attribute byte and a type index. Unread,
            # the walk loses its place and misses a memory imported after it.
            elif kind == 0x04: f.read(1); uleb(f)
    elif sid == 5:  # memory
        for _ in range(uleb(f)):
            flags, mn, mx = limits(f)
            found = True
            print(f"DEFINED memory   flags=0x{flags:02x} "
                  f"shared={'YES' if flags & 0x02 else 'no'} "
                  f"min={mn} pages ({mn*64//1024} MiB) "
                  f"max={mx if mx is None else f'{mx} pages ({mx*64//1024} MiB)'}")
    elif sid == 0:  # custom
        here = f.tell()
        cn = name(f)
        if cn == 'target_features':
            cnt = uleb(f)
            feats = []
            for _ in range(cnt):
                pfx = f.read(1)[0]
                feats.append(('+' if pfx == 0x2b else '-') + name(f))
            print("target_features:", ' '.join(sorted(feats)))
        f.seek(here)
    f.seek(start + size)
if not found:
    print("no memory declaration found")
