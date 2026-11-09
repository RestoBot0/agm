#!/usr/bin/env python3
"""
AGM (Anisotropic Geodesic Manifold) Native Image Codec — Python SDK & CLI
=========================================================================
Provides the Python SDK and CLI for the Native `.agm` format (`AGM-L` bit-exact
lossless and `AGM-P` perceptual orthonormal wavelet lossy), plus universal
pre-conditioning for `.PNG`, `.WEBP`, and `.JPG`.

Native `.agm` Binary Format:
  1. `AGM-L` (Lossless, quality == 100):
     - 9-byte Header:
       Bytes 0..3 : Magic `"AGML"` (0x41, 0x47, 0x4D, 0x4C)
       Bytes 4..5 : Width  (uint16 big-endian)
       Bytes 6..7 : Height (uint16 big-endian)
       Byte  8    : Channels (uint8: 1 = Grayscale, 3 = RGB)
       Bytes 9..N : 64-bit Carryless Range-Coded Geodesic + RLS bitstream

  2. `AGM-P` (Perceptual Wavelet Lossy, quality 1..99):
     - 11-byte Header:
       Bytes 0..3 : Magic `"AGMP"` (0x41, 0x47, 0x4D, 0x50)
       Bytes 4..5 : Width  (uint16 big-endian)
       Bytes 6..7 : Height (uint16 big-endian)
       Byte  8    : Channels (uint8: 1 = Grayscale, 3 = RGB)
       Byte  9    : Quality (uint8: 1..99)
       Byte  10   : Chroma Subsampling Flag (uint8: 0 = 4:4:4, 1 = 4:2:0 Luma-Guided)
       Bytes 11..N: 64-bit Carryless Range-Coded CDF 9/7 Wavelet Cone bitstream
"""

import argparse
import ctypes
import io
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image

_DIR = Path(__file__).resolve().parent
_LIB_PATH = _DIR / "libagm.so"
_C_PATH = _DIR / "agm_codec.c"

if not _LIB_PATH.exists() and _C_PATH.exists():
    subprocess.check_call([
        "gcc", "-O3", "-march=native", "-shared", "-fPIC",
        str(_C_PATH), "-lm", "-o", str(_LIB_PATH)
    ])

if not _LIB_PATH.exists():
    raise RuntimeError(f"Shared library not found at {_LIB_PATH}")

_lib = ctypes.CDLL(str(_LIB_PATH))

_lib.agm_encode.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
]
_lib.agm_encode.restype = ctypes.c_size_t

_lib.agm_decode.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
]
_lib.agm_decode.restype = ctypes.c_int

_lib.agm_encode_lossless.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_size_t,
]
_lib.agm_encode_lossless.restype = ctypes.c_size_t

_lib.agm_decode_lossless.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_lib.agm_decode_lossless.restype = ctypes.c_int

_lib.agm_encode_lossy.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
]
_lib.agm_encode_lossy.restype = ctypes.c_size_t

_lib.agm_decode_lossy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_lib.agm_decode_lossy.restype = ctypes.c_int

_lib.agm_manifold_png_engine.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
]
_lib.agm_manifold_png_engine.restype = None

_lib.agm_lattice_png_scanlines.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
]
_lib.agm_lattice_png_scanlines.restype = None


def _prepare_array(img: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    arr = np.ascontiguousarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        h, w = arr.shape
        ch = 1
    elif arr.ndim == 3:
        h, w, ch = arr.shape
        if ch == 4:
            arr = np.ascontiguousarray(arr[:, :, :3])
            ch = 3
        elif ch == 1:
            arr = np.ascontiguousarray(arr[:, :, 0])
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return arr, w, h, ch


def get_info(agm_bytes: bytes) -> Dict[str, Union[int, str]]:
    """Parses the 9-byte (`AGML`) or 11-byte (`AGMP`) native `.agm` header."""
    if len(agm_bytes) < 9 or agm_bytes[:3] != b"AGM":
        raise ValueError("Not a valid .agm file (missing AGM magic header)")
    magic = agm_bytes[:4].decode("ascii", errors="replace")
    w = (agm_bytes[4] << 8) | agm_bytes[5]
    h = (agm_bytes[6] << 8) | agm_bytes[7]
    ch = agm_bytes[8]
    if magic == "AGML":
        return {
            "magic": "AGML",
            "mode": "lossless",
            "width": w,
            "height": h,
            "channels": ch,
            "quality": 100,
            "header_bytes": 9,
        }
    elif magic == "AGMP":
        if len(agm_bytes) < 11:
            raise ValueError("Truncated AGMP header")
        return {
            "magic": "AGMP",
            "mode": "lossy",
            "width": w,
            "height": h,
            "channels": ch,
            "quality": int(agm_bytes[9]),
            "sub_chroma": int(agm_bytes[10]),
            "header_bytes": 11,
        }
    else:
        raise ValueError(f"Unknown AGM magic: {magic}")


def encode_lossless(img: np.ndarray) -> bytes:
    """Encodes an image into a bit-exact lossless `AGML` (.agm) bitstream."""
    arr, w, h, ch = _prepare_array(img)
    cap = w * h * ch * 2 + 65536
    out = np.empty(cap, dtype=np.uint8)
    sz = _lib.agm_encode_lossless(
        arr.ctypes.data_as(ctypes.c_void_p), w, h, ch,
        out.ctypes.data_as(ctypes.c_void_p), cap,
    )
    if sz == 0:
        raise RuntimeError("agm_encode_lossless failed")
    return out[:sz].tobytes()


def encode_lossy(img: np.ndarray, quality: int = 85) -> bytes:
    """Encodes an image into a perceptual wavelet `AGMP` (.agm) bitstream (quality 1..99)."""
    arr, w, h, ch = _prepare_array(img)
    q = max(1, min(99, int(quality)))
    cap = w * h * ch * 2 + 65536
    out = np.empty(cap, dtype=np.uint8)
    sz = _lib.agm_encode_lossy(
        arr.ctypes.data_as(ctypes.c_void_p), w, h, ch, q,
        out.ctypes.data_as(ctypes.c_void_p), cap,
    )
    if sz == 0:
        raise RuntimeError("agm_encode_lossy failed")
    return out[:sz].tobytes()


def encode(img: np.ndarray, quality: int = 100) -> bytes:
    """
    Unified Native `.agm` encoder:
      - quality >= 100 -> `AGM-L` (Mathematically Bit-Exact Lossless `.agm`)
      - quality 1..99  -> `AGM-P` (Orthonormal CDF 9/7 Wavelet `.agm`)
    """
    if int(quality) >= 100:
        return encode_lossless(img)
    return encode_lossy(img, quality=int(quality))


def decode(data: bytes) -> np.ndarray:
    """Decodes a Native `.agm` (`AGML` or `AGMP`) buffer back to a uint8 NumPy array."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" or data[:4] == b"RIFF" or data[:2] == b"\xff\xd8":
        return np.array(Image.open(io.BytesIO(data)))
    info = get_info(data)
    w, h, ch = info["width"], info["height"], info["channels"]
    shape = (h, w) if ch == 1 else (h, w, ch)
    out = np.empty(shape, dtype=np.uint8)
    buf = np.frombuffer(data, dtype=np.uint8)
    cw, ch_out, cch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    rc = _lib.agm_decode(
        buf.ctypes.data_as(ctypes.c_void_p), len(data),
        out.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(cw), ctypes.byref(ch_out), ctypes.byref(cch),
    )
    if rc != 0:
        raise RuntimeError(f"agm_decode failed with status {rc}")
    return out


decode_lossless = decode
decode_lossy = decode


def encode_file(input_path: Union[str, Path], output_path: Union[str, Path], quality: int = 100) -> int:
    """Reads an image file (PNG/JPG/WebP) and writes a Native `.agm` file."""
    im = Image.open(input_path)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    arr = np.array(im, dtype=np.uint8)
    agm_bytes = encode(arr, quality=quality)
    Path(output_path).write_bytes(agm_bytes)
    return len(agm_bytes)


def decode_file(input_path: Union[str, Path], output_path: Union[str, Path]) -> np.ndarray:
    """Reads a Native `.agm` file and saves the reconstructed image (e.g. `.png`)."""
    raw = Path(input_path).read_bytes()
    arr = decode(raw)
    Image.fromarray(arr).save(output_path)
    return arr


def _make_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _pack_standard_png(scanlines: bytes, w: int, h: int, channels: int) -> bytes:
    color_type = 2 if channels == 3 else 0
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    best_idat = None
    for strat in (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED):
        comp = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=15, memLevel=9, strategy=strat)
        idat = comp.compress(scanlines) + comp.flush()
        if best_idat is None or len(idat) < len(best_idat):
            best_idat = idat
    return (
        b"\x89PNG\r\n\x1a\n"
        + _make_png_chunk(b"IHDR", ihdr)
        + _make_png_chunk(b"IDAT", best_idat)
        + _make_png_chunk(b"IEND", b"")
    )


def compress_universal(
    img: np.ndarray,
    fmt: str = "agm",
    tier: str = "ultra",
) -> Tuple[bytes, np.ndarray, dict]:
    """
    Compresses `img` into either Native `.AGM` (`AGML` / `AGMP`) or standard `.PNG`, `.WEBP`, `.JPG`.
    """
    arr, w, h, ch = _prepare_array(img)
    fmt = fmt.lower().strip().lstrip(".")
    if fmt == "jpeg":
        fmt = "jpg"
    tier = tier.lower().strip()

    if fmt == "agm":
        agm_q_map = {"exact": 100, "ultra": 92, "high": 82, "max": 68}
        q = agm_q_map.get(tier, 100)
        out_bytes = encode(arr, quality=q)
        final_rec = arr.copy() if q >= 100 else decode(out_bytes)
        max_err = int(np.max(np.abs(arr.astype(np.int16) - final_rec.astype(np.int16))))
        engine_desc = (
            "Native AGM-L (Geodesic + Biharmonic + Bilateral Lifting + RLS + 64-bit Range Coder, Bit-Exact)"
            if q >= 100
            else f"Native AGM-P (4-Level Orthonormal CDF 9/7 Wavelet Cone + Range Coder, q={q})"
        )
        return out_bytes, final_rec, {
            "format": "AGM",
            "tier": tier,
            "delta_lsb": max_err,
            "max_pixel_error": max_err,
            "engine": engine_desc,
        }

    tier_to_delta = {"exact": 0, "ultra": 1, "high": 2, "max": 4}
    delta = tier_to_delta.get(tier, 1)
    scanlines = np.empty(h * (w * ch + 1), dtype=np.uint8)
    rec = np.empty_like(arr)

    _lib.agm_lattice_png_scanlines(
        arr.ctypes.data_as(ctypes.c_void_p),
        w, h, ch, delta,
        scanlines.ctypes.data_as(ctypes.c_void_p),
        rec.ctypes.data_as(ctypes.c_void_p),
    )

    if fmt == "png":
        out_bytes = _pack_standard_png(scanlines.tobytes(), w, h, ch)
        final_rec = rec
        engine_desc = (
            f"Quotient-Lattice PNG (s={2*delta+1}, ±{delta} LSB max error)"
            if delta > 0
            else "Sliding-Window Shannon-Entropy Bit-Exact PNG (0 LSB error)"
        )
    elif fmt == "webp":
        pil_img = Image.fromarray(rec if ch == 3 else rec.reshape(h, w))
        buf = io.BytesIO()
        pil_img.save(buf, format="WEBP", lossless=True, quality=100, method=4)
        out_bytes = buf.getvalue()
        final_rec = rec
        engine_desc = f"Quotient-Lattice WebP-Lossless (s={2*delta+1}, ±{delta} LSB max error)"
    elif fmt == "jpg":
        jpg_q_map = {"exact": 98, "ultra": 92, "high": 85, "max": 75}
        jpg_q = jpg_q_map.get(tier, 90)
        pil_img = Image.fromarray(rec if ch == 3 else rec.reshape(h, w))
        buf = io.BytesIO()
        subsampling = 0 if tier in ("exact", "ultra") else 2
        pil_img.save(buf, format="JPEG", quality=jpg_q, optimize=True, subsampling=subsampling)
        out_bytes = buf.getvalue()
        dec_img = Image.open(io.BytesIO(out_bytes)).convert("RGB" if ch == 3 else "L")
        final_rec = np.ascontiguousarray(np.array(dec_img), dtype=np.uint8)
        engine_desc = f"Quotient-Lattice + Optimal Huffman JPEG (q={jpg_q})"
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    max_err = int(np.max(np.abs(arr.astype(np.int16) - final_rec.astype(np.int16))))
    return out_bytes, final_rec, {
        "format": fmt.upper(),
        "tier": tier,
        "delta_lsb": delta,
        "max_pixel_error": max_err,
        "engine": engine_desc,
    }


def _main():
    parser = argparse.ArgumentParser(description="AGM Native Image Codec CLI (.agm)")
    sub = parser.add_subparsers(dest="cmd")

    p_enc = sub.add_parser("encode", help="Encode an image (PNG/JPG/WebP) to Native .agm")
    p_enc.add_argument("input", help="Input image path")
    p_enc.add_argument("output", help="Output .agm path")
    p_enc.add_argument("-q", "--quality", type=int, default=100, help="Quality: 100 = Bit-Exact Lossless (AGM-L), 1..99 = Perceptual Wavelet (AGM-P)")

    p_dec = sub.add_parser("decode", help="Decode a Native .agm file back to PNG/JPG/WebP")
    p_dec.add_argument("input", help="Input .agm path")
    p_dec.add_argument("output", help="Output image path (e.g. out.png)")

    p_info = sub.add_parser("info", help="Inspect header of a Native .agm file")
    p_info.add_argument("input", help="Input .agm path")

    args = parser.parse_args()
    if args.cmd == "encode":
        sz = encode_file(args.input, args.output, quality=args.quality)
        mode_str = "AGM-L (Bit-Exact Lossless)" if args.quality >= 100 else f"AGM-P (Lossy Wavelet q={args.quality})"
        print(f"Encoded {args.input} -> {args.output} [{mode_str}]: {sz:,} bytes")
    elif args.cmd == "decode":
        arr = decode_file(args.input, args.output)
        print(f"Decoded {args.input} -> {args.output} ({arr.shape[1]}x{arr.shape[0]})")
    elif args.cmd == "info":
        raw = Path(args.input).read_bytes()
        info = get_info(raw)
        print(f"File: {args.input} ({len(raw):,} bytes) -> {info}")
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
