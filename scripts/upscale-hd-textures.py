#!/usr/bin/env python3
"""upscale-hd-textures.py — resample the HD character layer's art to 2x, in place, BLP to BLP.

WHY THIS EXISTS
---------------
`benilla_formats::characters::sections` composites a character's body from a base skin plus up to
fifteen overlays, and its tile table is authored at `UPSCALE_FACTOR = 2` — a 512-square atlas over
the reference's 256-square partition. The compositor scales its tiles to the base skin and
nearest-samples an overlay that is not its tile's size, so mixed art composites in place; but a
1x overlay on a 2x skin is then sampled up blocky, which a Lanczos resample here avoids.

The vanilla corpus satisfies it because `patch-U`/`patch-V` resample the whole character tree to 2x.
`patch-A.MPQ` — the updated character models — does not: its skins are 256-square and its overlays
128x32 / 128x64, exactly 1x. So the layer has to be brought up to the same 2x the compositor and the
rest of the corpus already speak, and that is this script's whole job.

It is deliberately a *pre-pass over the art*, not a change to the compositor. Scaling at blend time
would mean resampling the same overlays on every appearance change, on the frame thread, forever;
doing it once at pack time costs nothing at runtime and keeps the composite a memcpy.

WHAT IT TOUCHES
---------------
`Character\\` only, by default — and that bound is load-bearing rather than conservative. The
composite reads its base skin and every overlay from there, so that is the whole set the atlas size
depends on. `Textures\\` is the models' own bound art, already authored at 512-1024, and doubling it
would take the container from ~1.5 GiB to something nobody should download.

WHAT IT DOES NOT DO
-------------------
This is a Lanczos resample, not a neural one. Real-ESRGAN needs an inference runtime and pretrained
weights, neither of which is in this tree (no ort/tract/candle/torch/onnxruntime — checked). The
seam is `upscale_rgba` below: give it a better function and nothing else here changes.

USAGE
-----
    benilla-pack extract --data "<Data>/patch-A.mpq" --prefix "Character\\" --out <dir>
    python scripts/upscale-hd-textures.py <dir>
    benilla-pack build --data "<Data>/patch-A.mpq" --out web/packs-hd --replace-from <dir>

Rewrites every `.blp` under <dir> in place. Idempotent only in the sense that re-running doubles
again — extract fresh before each run.

A layer cut by `benilla-pack overlay` lists its composite inputs with the exact 2x size each needs
(`compose.tsv`: some ship at 1x, some at half that, some above 2x), and `--targets` resamples each
listed file to that size instead of by `--factor`:

    python scripts/upscale-hd-textures.py <out>/compose --targets <out>/compose.tsv
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

# ---------------------------------------------------------------------------------------------
# BLP2, exactly as `benilla_blp::parse_header` reads it. Field offsets are that function's, so a
# change there is a change here; the round-trip check at the bottom of a run is what catches drift.
# ---------------------------------------------------------------------------------------------
HEADER_SIZE = 148          # magic .. mip_sizes[16]
PALETTE_SIZE = 256 * 4     # always present in BLP2 direct content, even when unused
PAYLOAD_BASE = HEADER_SIZE + PALETTE_SIZE
MAX_LEVELS = 16

COMPRESSION_PALETTIZED = 1
COMPRESSION_DXT = 2
COMPRESSION_BGRA8 = 3

# `Header::dxt_texels`: alpha_type 0 -> DXT1, 1 -> DXT3(BC2), 7 -> DXT5(BC3), anything else -> DXT1.
ALPHA_TYPE_DXT1 = 0
ALPHA_TYPE_DXT3 = 1
ALPHA_TYPE_DXT5 = 7


def mip_chain_count(width: int, height: int, has_mipmaps: bool) -> int:
    """`benilla_blp::mip_chain_count` — floor(log2) of the larger side, or 0 without mips.

    Kept bit-identical to the Rust (which is itself kept bit-identical to wow-blp) because the
    decoder reads exactly this many levels and a file that stores fewer decodes short.
    """
    if not has_mipmaps:
        return 0
    import math
    return max(int(math.log2(width)), int(math.log2(height)))


def level_size(width: int, height: int, level: int) -> tuple[int, int]:
    if level == 0:
        return width, height
    return max(width >> level, 1), max(height >> level, 1)


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------
def _dds_header(fourcc: bytes, width: int, height: int, payload_len: int) -> bytes:
    """A minimal DDS wrapper so PIL's own DXT decoder can read a BLP's block payload.

    Wrapping rather than hand-decoding DXT: PIL already has a correct, C-speed decoder for these
    exact block formats, and a second implementation here would be a second thing to be wrong.
    """
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000   # CAPS|HEIGHT|WIDTH|PIXELFORMAT|LINEARSIZE
    pf = struct.pack(
        "<II4sIIIII",
        32,          # dwSize
        0x4,         # DDPF_FOURCC
        fourcc,
        0, 0, 0, 0, 0,
    )
    return (
        b"DDS "
        + struct.pack("<IIIIIII", 124, flags, height, width, payload_len, 0, 0)
        + b"\0" * 44          # dwReserved1[11]
        + pf
        + struct.pack("<IIIII", 0x1000, 0, 0, 0, 0)   # caps..reserved2
    )


def _block_bytes(fourcc: bytes, width: int, height: int) -> int:
    blocks = ((width + 3) // 4) * ((height + 3) // 4)
    return blocks * (8 if fourcc == b"DXT1" else 16)


def _decode_palettized(payload: bytes, palette: bytes, w: int, h: int, alpha_bits: int) -> Image.Image:
    """`benilla_blp::decode_raw1` — 1-byte palette indices, then a packed alpha plane."""
    import numpy as np

    n = w * h
    idx = np.frombuffer(payload[:n], dtype=np.uint8)
    if idx.size < n:
        idx = np.pad(idx, (0, n - idx.size))
    pal = np.frombuffer(palette, dtype=np.uint8).reshape(256, 4)   # BGRA
    rgb = pal[idx][:, [2, 1, 0]]

    alpha_src = payload[n:]
    if alpha_bits == 8:
        a = np.frombuffer(alpha_src[:n], dtype=np.uint8)
        if a.size < n:
            a = np.pad(a, (0, n - a.size), constant_values=255)
    elif alpha_bits in (1, 4):
        per_byte = 8 // alpha_bits
        need = (n + per_byte - 1) // per_byte
        raw = np.frombuffer(alpha_src[:need], dtype=np.uint8)
        if raw.size < need:
            raw = np.pad(raw, (0, need - raw.size), constant_values=0xFF)
        bits = np.unpackbits(raw, bitorder="little").reshape(-1, 8)
        if alpha_bits == 1:
            a = (bits[:, :].reshape(-1)[:n] * 255).astype(np.uint8)
        else:
            nib = (bits[:, 0:4] << np.arange(4)).sum(axis=1).astype(np.uint8)
            hi = (bits[:, 4:8] << np.arange(4)).sum(axis=1).astype(np.uint8)
            a = np.stack([nib, hi], axis=1).reshape(-1)[:n] * 17
    else:
        a = np.full(n, 255, dtype=np.uint8)

    rgba = np.concatenate([rgb, a.reshape(-1, 1)], axis=1).astype(np.uint8)
    return Image.fromarray(rgba.reshape(h, w, 4), "RGBA")


def read_blp(data: bytes) -> tuple[Image.Image, dict]:
    """Level 0 as RGBA, plus what the file said about itself."""
    if len(data) < HEADER_SIZE or data[0:4] != b"BLP2":
        raise ValueError("not a BLP2 file")
    compression = data[8]
    alpha_bits = data[9]
    alpha_type = data[10]
    has_mips = data[11] != 0
    width, height = struct.unpack_from("<II", data, 12)
    offsets = struct.unpack_from("<16I", data, 20)
    sizes = struct.unpack_from("<16I", data, 84)
    palette = data[HEADER_SIZE:HEADER_SIZE + PALETTE_SIZE]
    meta = dict(compression=compression, alpha_bits=alpha_bits, alpha_type=alpha_type,
                has_mips=has_mips, width=width, height=height)

    off, size = offsets[0], sizes[0]
    payload = data[off:off + size]

    if compression == COMPRESSION_DXT:
        fourcc = {ALPHA_TYPE_DXT3: b"DXT3", ALPHA_TYPE_DXT5: b"DXT5"}.get(alpha_type, b"DXT1")
        # The stored size can under-run the block span on the sub-block tail; pad rather than fail,
        # which is what `dxt_level_span` does on the Rust side.
        want = _block_bytes(fourcc, width, height)
        if len(payload) < want:
            payload = payload + b"\0" * (want - len(payload))
        dds = _dds_header(fourcc, width, height, want) + payload[:want]
        return Image.open(io.BytesIO(dds)).convert("RGBA"), meta

    if compression == COMPRESSION_BGRA8:
        import numpy as np
        n = width * height * 4
        buf = payload[:n]
        if len(buf) < n:
            buf = buf + b"\0" * (n - len(buf))
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)[:, :, [2, 1, 0, 3]]
        return Image.fromarray(arr, "RGBA"), meta

    if compression == COMPRESSION_PALETTIZED:
        return _decode_palettized(payload, palette, width, height, alpha_bits), meta

    raise ValueError(f"unsupported BLP compression {compression}")


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------
def _encode_dxt(img: Image.Image, fourcc: bytes) -> bytes:
    """One level's block payload, via PIL's DDS encoder with its container stripped.

    PIL emits a DX10 extended header for BC3, so the payload offset is read from the file rather
    than assumed — 128 bytes normally, 148 when `dwFourCC` is `DX10`.
    """
    fmt = "DXT1" if fourcc == b"DXT1" else "BC3"
    buf = io.BytesIO()
    src = img if fourcc != b"DXT1" else img.convert("RGBA")
    src.save(buf, format="DDS", pixel_format=fmt)
    raw = buf.getvalue()
    body = 128
    if raw[84:88] == b"DX10":
        body += 20
    return raw[body:]


def write_blp(img: Image.Image, fourcc: bytes, alpha_type: int, alpha_bits: int) -> bytes:
    """A DXT-compressed BLP2 with a full mip chain, in the shape `benilla_blp::decode` reads."""
    width, height = img.size
    chain = mip_chain_count(width, height, True)
    levels = []
    for level in range(min(max(chain, 1), MAX_LEVELS)):
        lw, lh = level_size(width, height, level)
        # Through `resample`, not `Image.resize`: a premultiplying mip chain darkens every
        # transparent-edged texel toward black as the levels shrink, which is the same defect the
        # top level had, spread over the minified lanes.
        lvl = img if level == 0 else resample(img, (lw, lh))
        levels.append(_encode_dxt(lvl, fourcc))

    offsets = [0] * MAX_LEVELS
    sizes = [0] * MAX_LEVELS
    cursor = PAYLOAD_BASE
    for i, payload in enumerate(levels):
        offsets[i] = cursor
        sizes[i] = len(payload)
        cursor += len(payload)

    head = bytearray()
    head += b"BLP2"
    head += struct.pack("<I", 1)                                   # content: direct
    head += bytes([COMPRESSION_DXT, alpha_bits, alpha_type, 1])    # compression, bits, type, mips
    head += struct.pack("<II", width, height)
    head += struct.pack("<16I", *offsets)
    head += struct.pack("<16I", *sizes)
    assert len(head) == HEADER_SIZE, len(head)
    return bytes(head) + b"\0" * PALETTE_SIZE + b"".join(levels)


# ---------------------------------------------------------------------------------------------
# The resample itself — the one seam a better upscaler drops into.
# ---------------------------------------------------------------------------------------------
def resample(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Lanczos to `size`, with alpha resampled **separately from the colour**.

    The separation is the whole of it, and it is not a nicety. **Pillow premultiplies alpha when it
    resamples an RGBA image** (measured on 12.3.0: LANCZOS and BICUBIC both zero the colour under a
    fully transparent texel; NEAREST is the only filter that leaves it), so a plain
    `img.resize(...)` here destroys every texel this layer painted under alpha 0.

    That is the right answer for art whose transparent regions are empty, and the wrong one for
    this layer, which has such texels and DRAWS them: `Character\Gnome\Hair00_0*.blp` carries a
    fully painted right half at alpha 0, the gnome models' hair batches are `Opaque` (measured —
    vanilla's are `AlphaTest`), and an Opaque batch samples the colour and never looks at the
    alpha. Premultiplied through this function, that half encoded to black and every gnome FEMALE
    — whose hair geosets are the ones that UV into it — rendered with black hair on the hosted
    page while the same character was correct natively, off the un-upscaled archive (2026-09-08).

    Resampling the two independently is what the docstring below has always claimed this does:
    alpha is a fourth channel here, not a weight.
    """
    rgb = img.convert("RGB").resize(size, Image.LANCZOS)
    alpha = img.getchannel("A").resize(size, Image.LANCZOS)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def upscale_rgba(img: Image.Image, factor: int) -> Image.Image:
    """`factor`x, Lanczos, alpha carried through.

    RGB is resampled with alpha as a plain fourth channel rather than premultiplied, because BLP
    alpha here is a cutout/blend mask over art whose transparent texels still carry their authored
    colour — the composite's own blend reads them the same way, and a model batch that ignores
    alpha reads nothing else. [`resample`] is what makes that true against a Pillow that would
    otherwise premultiply.
    """
    return resample(img, (img.width * factor, img.height * factor))


def choose_format(img: Image.Image, meta: dict) -> tuple[bytes, int, int]:
    """Which DXT flavour the output takes: `(fourcc, alpha_type, alpha_bits)`.

    The source's own choice is kept where PIL can write it. DXT3 is the exception — PIL encodes
    DXT1 and BC3 only — and it promotes to DXT5, which is the same 16 bytes per block with
    interpolated rather than explicit alpha, so nothing is lost. A source that was not DXT at all
    picks by whether the pixels actually use alpha.
    """
    if meta["compression"] == COMPRESSION_DXT:
        if meta["alpha_type"] in (ALPHA_TYPE_DXT3, ALPHA_TYPE_DXT5):
            return b"DXT5", ALPHA_TYPE_DXT5, 8
        return b"DXT1", ALPHA_TYPE_DXT1, meta["alpha_bits"]
    alpha = img.getchannel("A").getextrema()
    if alpha[0] < 255:
        return b"DXT5", ALPHA_TYPE_DXT5, 8
    return b"DXT1", ALPHA_TYPE_DXT1, 0


def process(path: str, factor: int, size: tuple[int, int] | None = None) -> tuple[str, int, int, str | None]:
    """Resample one file in place: by `factor`, or to exactly `size` when one is given."""
    try:
        before = os.path.getsize(path)
        with open(path, "rb") as f:
            data = f.read()
        img, meta = read_blp(data)
        want = size or (img.width * factor, img.height * factor)
        if want[0] > 4096 or want[1] > 4096:
            return path, before, before, "skipped: would exceed the decoder's dimension cap"
        big = img if img.size == want else resample(img, want)
        fourcc, alpha_type, alpha_bits = choose_format(big, meta)
        out = write_blp(big, fourcc, alpha_type, alpha_bits)
        # Prove it before replacing the input: a file that will not read back is worse than one that
        # was never upscaled, because the packer would ship it and the client would fail at draw.
        check, _ = read_blp(out)
        if check.size != big.size:
            return path, before, before, f"round-trip size {check.size} != {big.size}"
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(out)
        os.replace(tmp, path)
        return path, before, len(out), None
    except Exception as e:  # noqa: BLE001 — one bad file must not take the run down
        return path, 0, 0, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="the directory `benilla-pack extract` wrote")
    ap.add_argument("--factor", type=int, default=2,
                    help="resample factor; 2 is what UPSCALE_FACTOR expects (default: 2)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=0, help="stop after N files (a smoke run)")
    ap.add_argument("--targets", metavar="TSV",
                    help="`benilla-pack overlay`'s compose.tsv: resample each listed file to the "
                         "exact size it names (`name width height`, the name a path under root) "
                         "instead of by --factor, up or down")
    args = ap.parse_args()

    # path -> the exact size it is resampled to, or None for `--factor`.
    sizes: dict[str, tuple[int, int] | None] = {}
    if args.targets:
        with open(args.targets, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line or line.startswith("#"):
                    continue
                name, w, h = line.split("\t")
                path = os.path.join(args.root, *[p for p in name.replace("/", "\\").split("\\") if p])
                sizes[path] = (int(w), int(h))
    else:
        for dirpath, _, names in os.walk(args.root):
            for n in names:
                if n.lower().endswith(".blp"):
                    sizes[os.path.join(dirpath, n)] = None
    files = sorted(sizes)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no .blp under {args.root}", file=sys.stderr)
        return 1

    how = "to their listed sizes" if args.targets else f"{args.factor}x"
    print(f"resampling {len(files)} textures {how} on {args.jobs} workers", flush=True)
    ok = failed = 0
    before_total = after_total = 0
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(process, f, args.factor, sizes[f]) for f in files]
        for i, fut in enumerate(as_completed(futures), 1):
            path, before, after, err = fut.result()
            if err:
                failed += 1
                if len(errors) < 20:
                    errors.append(f"{os.path.relpath(path, args.root)}: {err}")
            else:
                ok += 1
                before_total += before
                after_total += after
            if i % 500 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    mib = lambda b: b / (1024 * 1024)
    print(f"\n{ok} resampled, {failed} failed")
    print(f"{mib(before_total):.1f} MiB -> {mib(after_total):.1f} MiB "
          f"({after_total / max(before_total, 1):.2f}x)")
    for e in errors:
        print(f"  {e}", file=sys.stderr)
    if failed and len(errors) == 20:
        print(f"  … {failed - 20} more", file=sys.stderr)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
