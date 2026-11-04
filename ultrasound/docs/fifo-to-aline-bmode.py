#!/usr/bin/env python3
"""
Offline (and optional live) A-line + 4-channel B-mode pipeline for
AD9670 / HSC-ADC-EVALC FIFO captures.

Decode (default):
  raw 16-bit word -> ((word >> 4) & 0xFFF) - 2048   # 12-bit offset-binary
Active RX FIFOs (default map): 2, 3, 6, 7  ->  B-mode scan lines 0..3

Dependencies: numpy required; matplotlib optional (PNG via pure writer fallback).
scipy is NOT used (32-bit lab Python friendly).

Examples:
  python fifo-to-aline-bmode.py --demo --out ./ultrasound_output
  python fifo-to-aline-bmode.py live.csv --out ./ultrasound_output
  python fifo-to-aline-bmode.py ./fifo_folder --out ./ultrasound_output
  python fifo-to-aline-bmode.py --live --out ./ultrasound_output   # Windows + DLL
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Defaults (documented for Immersive Team)
# ---------------------------------------------------------------------------

DEFAULT_FS_HZ = 40e6
DEFAULT_SOUND_SPEED_M_S = 1540.0
DEFAULT_DYNAMIC_RANGE_DB = 40.0
DEFAULT_FIFO_CHANNELS = (2, 3, 6, 7)
DEFAULT_SAMPLE_COUNT = 8192
DEFAULT_OUT_DIR = "ultrasound_output"

# Duplicate-pair policy:
#   "keep"  — keep every decoded sample (default; safest until layout is locked)
#   "drop"  — if consecutive raw words match, keep only the first of each pair
DEFAULT_DEDUPE = "keep"


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode_12bit_offset_binary(words: np.ndarray) -> np.ndarray:
    """Map 16-bit FIFO words to signed RF using bits 15:4, center at 2048."""
    raw = np.asarray(words, dtype=np.uint16)
    code12 = ((raw.astype(np.uint32) >> 4) & 0x0FFF).astype(np.int32)
    return (code12 - 2048).astype(np.float64)


def apply_dedupe_pairs(raw_words: np.ndarray, mode: str) -> np.ndarray:
    """
    Handle high duplicate-pair counts seen in lab captures.

    keep: return all samples.
    drop: collapse runs of identical consecutive words to a single sample
          (common when the FIFO appears to repeat each value).
    """
    raw = np.asarray(raw_words, dtype=np.uint16).ravel()
    if mode == "keep" or raw.size == 0:
        return raw
    if mode != "drop":
        raise ValueError("dedupe mode must be 'keep' or 'drop'")
    # Keep index i when i==0 or raw[i] != raw[i-1]
    mask = np.empty(raw.size, dtype=bool)
    mask[0] = True
    mask[1:] = raw[1:] != raw[:-1]
    return raw[mask]


def duplicate_pair_fraction(raw_words: np.ndarray) -> float:
    raw = np.asarray(raw_words).ravel()
    if raw.size < 2:
        return 0.0
    return float(np.mean(raw[1:] == raw[:-1]))


# ---------------------------------------------------------------------------
# Envelope / A-line (numpy-only Hilbert approx)
# ---------------------------------------------------------------------------

def analytic_signal(x: np.ndarray) -> np.ndarray:
    """FFT-based analytic signal (same idea as scipy.signal.hilbert)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x.astype(np.complex128)
    X = np.fft.fft(x)
    h = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


def envelope_abs_hilbert(rf: np.ndarray) -> np.ndarray:
    return np.abs(analytic_signal(rf))


def log_compress(envelope: np.ndarray, dynamic_range_db: float) -> np.ndarray:
    """Map envelope to [0, 1] with configurable dB dynamic range."""
    env = np.asarray(envelope, dtype=np.float64)
    peak = float(np.max(env)) if env.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(env)
    env_db = 20.0 * np.log10(np.maximum(env / peak, 1e-12))
    clipped = np.clip(env_db, -abs(dynamic_range_db), 0.0)
    return (clipped + abs(dynamic_range_db)) / abs(dynamic_range_db)


def depth_axis_m(n_samples: int, fs_hz: float, sound_speed_m_s: float) -> np.ndarray:
    """Round-trip depth. Provisional until TX time offset is calibrated."""
    t = np.arange(n_samples, dtype=np.float64) / fs_hz
    return 0.5 * sound_speed_m_s * t


# ---------------------------------------------------------------------------
# I/O — ingest
# ---------------------------------------------------------------------------

def _load_npy_column_file(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    return np.asarray(arr).ravel()


def _load_text_column(path: Path) -> np.ndarray:
    # Accept one integer/float per line, or a single column CSV.
    try:
        data = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
    except Exception:
        data = np.loadtxt(str(path), dtype=np.float64)
    return np.asarray(data).ravel()


def load_folder_of_fifos(folder: Path) -> Dict[int, np.ndarray]:
    """
    Load per-FIFO files named like fifo_2.csv, fifo2.npy, FIFO_3.txt, etc.
    """
    captures: Dict[int, np.ndarray] = {}
    patterns = (
        "fifo_{i}.csv", "fifo_{i}.npy", "fifo_{i}.txt", "fifo{i}.csv",
        "fifo{i}.npy", "FIFO_{i}.csv", "FIFO{i}.csv", "ch{i}.csv", "ch_{i}.csv",
    )
    for i in range(8):
        found = None
        for pat in patterns:
            p = folder / pat.format(i=i)
            if p.is_file():
                found = p
                break
        if found is None:
            continue
        if found.suffix.lower() == ".npy":
            captures[i] = _load_npy_column_file(found)
        else:
            captures[i] = _load_text_column(found)
    if not captures:
        raise FileNotFoundError(
            f"No per-FIFO files found in {folder} "
            "(expected fifo_2.csv / fifo2.npy / …)"
        )
    return captures


def load_tabular_capture(path: Path) -> Dict[int, np.ndarray]:
    """
    Load a CSV/NPY with 4–8 FIFO columns.

    CSV rules:
      - Optional header row containing 'fifo' / 'ch' labels, or bare integers.
      - Else assume columns are FIFO 0..N-1 in order.
      - Or header names like fifo_2, FIFO3, ch6.
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return {0: arr.ravel()}
        if arr.ndim != 2:
            raise ValueError(f"NPY must be 1-D or 2-D, got shape {arr.shape}")
        # Prefer (samples, channels)
        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 8:
            arr = arr.T
        return {i: arr[:, i].ravel() for i in range(arr.shape[1])}

    # CSV / text
    with path.open("r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty file: {path}")

    def _parse_header_fifo_index(name: str) -> Optional[int]:
        s = name.strip().lower().replace(" ", "")
        for prefix in ("fifo_", "fifo", "ch_", "ch", "channel_", "channel"):
            if s.startswith(prefix):
                rest = s[len(prefix) :]
                if rest.isdigit():
                    return int(rest)
        if s.isdigit():
            return int(s)
        return None

    header = rows[0]
    header_idxs = [_parse_header_fifo_index(c) for c in header]
    has_header = any(v is not None for v in header_idxs) or any(
        (not _is_number(c)) for c in header
    )

    if has_header:
        data_rows = rows[1:]
        col_map: List[Optional[int]] = header_idxs
        # Fill missing header labels as sequential FIFO indices
        next_i = 0
        for j, v in enumerate(col_map):
            if v is None:
                while next_i in [x for x in col_map if x is not None]:
                    next_i += 1
                col_map[j] = next_i
                next_i += 1
    else:
        data_rows = rows
        col_map = list(range(len(header)))

    n_cols = len(col_map)
    cols: Dict[int, List[float]] = {int(fi): [] for fi in col_map if fi is not None}
    for row in data_rows:
        if not row or all(not c.strip() for c in row):
            continue
        for j in range(min(n_cols, len(row))):
            fi = col_map[j]
            if fi is None:
                continue
            cols[int(fi)].append(float(row[j]))

    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def load_capture(path: Path) -> Dict[int, np.ndarray]:
    if path.is_dir():
        return load_folder_of_fifos(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_tabular_capture(path)


# ---------------------------------------------------------------------------
# Optional live HadBoard capture (Windows lab only)
# ---------------------------------------------------------------------------

def try_live_capture(
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    dll_path: Optional[str] = None,
    fpga_file: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """
    Attempt HadBoardApi FillFifos + CaptureFifoWords for FIFOs 0–7.
    Raises RuntimeError with a clear message if DLL/board unavailable.
    """
    try:
        import clr  # type: ignore  # pythonnet
    except Exception as exc:
        raise RuntimeError(
            "pythonnet/clr not available — live capture skipped "
            "(expected on Windows lab PC only)."
        ) from exc

    dll = dll_path or os.environ.get(
        "HADBOARD_DLL",
        r"C:\Program Files (x86)\Analog Devices\VisualAnalog\HadBoardApi.dll",
    )
    fpga = fpga_file or os.environ.get(
        "AD9670_FPGA",
        r"C:\Program Files (x86)\Analog Devices\VisualAnalog\Hardware"
        r"\HSC_ADC_EVALC\ad9670_lvds_mar2012.bin",
    )
    if not os.path.isfile(dll):
        raise RuntimeError(f"HadBoardApi.dll not found at {dll}")

    import time

    clr.AddReference(dll)
    from HadBoardApi import HadBoard  # type: ignore

    HadBoard.FindBoards()
    if HadBoard.BoardCount < 1:
        raise RuntimeError("HSC-ADC-EVALC not detected")

    HadBoard.SelectBoard(0)
    board = HadBoard.Board
    print("Configuring FPGA...")
    board.ConfigureFpga(fpga)
    board.SelectOutputWidth(16)
    print("Filling FIFOs...")
    board.FillFifos(sample_count, 1000)
    time.sleep(0.06)

    captures: Dict[int, np.ndarray] = {}
    for fifo in range(8):
        words = board.CaptureFifoWords(sample_count, fifo)
        captures[fifo] = np.asarray(
            [int(words[i]) & 0xFFFF for i in range(len(words))],
            dtype=np.uint16,
        )
    return captures


# ---------------------------------------------------------------------------
# Synthetic demo RF (smoke test without board)
# ---------------------------------------------------------------------------

def make_synthetic_capture(
    n_samples: int = DEFAULT_SAMPLE_COUNT,
    fs_hz: float = DEFAULT_FS_HZ,
    fifos: Sequence[int] = DEFAULT_FIFO_CHANNELS,
    seed: int = 7,
) -> Dict[int, np.ndarray]:
    """
    Build fake 16-bit FIFO words (12-bit in bits 15:4) with a few echo blobs
    at different depths per channel so B-mode shows structure.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs_hz
    captures: Dict[int, np.ndarray] = {}
    # Quiet FIFOs 0,1,4,5 as near-DC (matches lab "flat" FIFOs)
    for fifo in range(8):
        if fifo not in fifos:
            code = np.full(n_samples, 2048, dtype=np.int32)
            captures[fifo] = (code << 4).astype(np.uint16)
            continue
        # Carrier ~5 MHz pulsed echoes
        fc = 5.0e6
        rf = 0.05 * rng.standard_normal(n_samples)
        # Two or three Gaussian-enveloped bursts at channel-dependent depth
        centers = [0.12 + 0.02 * (fifo % 4), 0.28 + 0.015 * (fifo % 4)]
        if fifo in (6, 7):
            centers.append(0.40)
        for c_frac, amp in zip(centers, (0.9, 0.55, 0.35)):
            c = int(c_frac * n_samples)
            width = int(0.008 * n_samples)
            env = np.exp(-0.5 * ((np.arange(n_samples) - c) / max(width, 1)) ** 2)
            rf += amp * env * np.sin(2 * np.pi * fc * t)
        # Scale into ~12-bit offset-binary range and pack into bits 15:4
        peak = np.max(np.abs(rf)) or 1.0
        code = np.clip(np.round(2048 + (rf / peak) * 1800), 0, 4095).astype(np.int32)
        captures[fifo] = ((code << 4) & 0xFFF0).astype(np.uint16)
    return captures


def save_synthetic_csv(captures: Dict[int, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fifos = sorted(captures.keys())
    n = max(len(captures[f]) for f in fifos)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"fifo_{i}" for i in fifos])
        for r in range(n):
            w.writerow(
                [int(captures[i][r]) if r < len(captures[i]) else "" for i in fifos]
            )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def form_alines(
    captures: Dict[int, np.ndarray],
    fifo_channels: Sequence[int],
    dedupe: str,
    dynamic_range_db: float,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns
      rf_matrix:     (n_ch, n_samples) signed RF
      env_log:       (n_ch, n_samples) log-compressed [0,1]
      meta:          decode diagnostics
    """
    rf_list: List[np.ndarray] = []
    env_list: List[np.ndarray] = []
    meta = {"channels": [], "dedupe": dedupe, "dup_fraction": {}}

    for fifo in fifo_channels:
        if fifo not in captures:
            raise KeyError(
                f"FIFO {fifo} missing from capture "
                f"(have {sorted(captures.keys())})"
            )
        raw = np.asarray(captures[fifo]).ravel()
        # Accept already-decoded floats (range roughly [-2048, 2047]) or raw words
        if raw.dtype.kind == "f" and np.nanmax(np.abs(raw)) <= 4096 and (
            np.nanmax(raw) < 3000 and np.nanmin(raw) < 0
        ):
            # Heuristic: looks like already-centred RF
            rf = raw.astype(np.float64)
            dup = 0.0
            used_raw = raw
        else:
            raw_u = np.clip(raw, 0, 65535).astype(np.uint16)
            dup = duplicate_pair_fraction(raw_u)
            used_raw = apply_dedupe_pairs(raw_u, dedupe)
            rf = decode_12bit_offset_binary(used_raw)

        env = envelope_abs_hilbert(rf)
        env_c = log_compress(env, dynamic_range_db)
        rf_list.append(rf)
        env_list.append(env_c)
        meta["channels"].append(fifo)
        meta["dup_fraction"][fifo] = dup

    # Align lengths (dedupe may shorten)
    n = min(a.size for a in rf_list)
    rf_m = np.stack([a[:n] for a in rf_list], axis=0)
    env_m = np.stack([a[:n] for a in env_list], axis=0)
    return rf_m, env_m, meta


def save_png_grayscale(path: Path, image01: np.ndarray) -> None:
    """Write an 8-bit grayscale PNG. Prefer matplotlib; fallback to pure zlib."""
    img = np.clip(np.asarray(image01, dtype=np.float64), 0.0, 1.0)
    # Display: scan-line (channel) horizontal, depth vertical (row 0 = near)
    # Shape expected: (n_depth, n_lines) for imshow-like orientation
    if img.ndim != 2:
        raise ValueError("image must be 2-D")
    u8 = (img * 255.0).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 8))
        ax.imshow(u8, cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_xlabel("scan line (FIFO channel)")
        ax.set_ylabel("sample (depth ↑ near field at top)")
        ax.set_title("4-ch B-mode")
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        return
    except Exception:
        pass

    # Minimal PNG writer (grayscale, no filtering beyond None)
    height, width = u8.shape
    raw = b"".join(b"\x00" + u8[r].tobytes() for r in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big")
            + tag
            + data
            + zlib.crc32(tag + data).to_bytes(4, "big")
        )

    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([8, 0, 0, 0, 0])  # 8-bit grayscale
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def save_envelopes_csv(path: Path, env_log: np.ndarray, fifos: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample"] + [f"fifo_{i}_env" for i in fifos])
        for s in range(env_log.shape[1]):
            w.writerow([s] + [f"{env_log[c, s]:.6f}" for c in range(env_log.shape[0])])


def run_pipeline(
    captures: Dict[int, np.ndarray],
    out_dir: Path,
    fs_hz: float,
    sound_speed: float,
    dynamic_range_db: float,
    fifo_channels: Sequence[int],
    dedupe: str,
    save_csv: bool,
) -> dict:
    rf_m, env_m, meta = form_alines(
        captures, fifo_channels, dedupe, dynamic_range_db
    )
    # B-mode image: depth × scan-line
    bmode = env_m.T  # (n_samples, n_ch)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "bmode.png"
    save_png_grayscale(png_path, bmode)

    depth = depth_axis_m(env_m.shape[1], fs_hz, sound_speed)
    np.save(str(out_dir / "rf_signed.npy"), rf_m)
    np.save(str(out_dir / "envelope_log.npy"), env_m)
    np.save(str(out_dir / "depth_m.npy"), depth)

    csv_path = None
    if save_csv:
        csv_path = out_dir / "envelopes.csv"
        save_envelopes_csv(csv_path, env_m, fifo_channels)

    # Per-channel A-line PNGs (optional visual check)
    for i, fifo in enumerate(fifo_channels):
        # 1 × N looks odd; make a strip
        strip = np.tile(env_m[i : i + 1, :], (32, 1)).T  # depth × 32
        save_png_grayscale(out_dir / f"aline_fifo_{fifo}.png", strip)

    result = {
        "png": str(png_path),
        "csv": str(csv_path) if csv_path else None,
        "n_samples": int(env_m.shape[1]),
        "n_channels": int(env_m.shape[0]),
        "depth_max_m": float(depth[-1]) if depth.size else 0.0,
        "meta": meta,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AD9670 FIFO → A-line → 4-ch B-mode (offline / optional live)"
    )
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="CSV/NPY (4–8 FIFO columns) or folder of per-FIFO files",
    )
    p.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_FS_HZ,
        help=f"sample rate Hz (default {DEFAULT_FS_HZ:.0f})",
    )
    p.add_argument(
        "--sound-speed",
        type=float,
        default=DEFAULT_SOUND_SPEED_M_S,
        help=f"m/s (default {DEFAULT_SOUND_SPEED_M_S})",
    )
    p.add_argument(
        "--dynamic-range",
        type=float,
        default=DEFAULT_DYNAMIC_RANGE_DB,
        help=f"log-compress dB (default {DEFAULT_DYNAMIC_RANGE_DB})",
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--fifos",
        default=",".join(str(x) for x in DEFAULT_FIFO_CHANNELS),
        help="comma-separated FIFO indices for B-mode lines "
        f"(default {','.join(map(str, DEFAULT_FIFO_CHANNELS))})",
    )
    p.add_argument(
        "--dedupe",
        choices=("keep", "drop"),
        default=DEFAULT_DEDUPE,
        help="duplicate-pair policy: keep all samples, or drop consecutive "
        "duplicates (default keep)",
    )
    p.add_argument(
        "--save-csv",
        action="store_true",
        help="also write envelopes.csv",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="generate synthetic RF, save sample CSV, and run pipeline",
    )
    p.add_argument(
        "--demo-csv",
        default=None,
        help="path to write synthetic capture CSV when --demo "
        "(default: <out>/../synthetic_fifo_capture.csv under internal if possible)",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="attempt HadBoard live capture (Windows + DLL); graceful fail else",
    )
    p.add_argument("--dll", default=None, help="path to HadBoardApi.dll")
    p.add_argument("--fpga", default=None, help="path to ad9670 FPGA .bin")
    p.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"live/demo sample count (default {DEFAULT_SAMPLE_COUNT})",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    fifo_channels = tuple(int(x) for x in args.fifos.split(",") if x.strip() != "")
    out_dir = Path(args.out)

    captures: Optional[Dict[int, np.ndarray]] = None

    if args.demo:
        print("Generating synthetic 8-FIFO capture...")
        captures = make_synthetic_capture(
            n_samples=args.samples, fs_hz=args.fs, fifos=fifo_channels
        )
        demo_csv = Path(args.demo_csv) if args.demo_csv else out_dir / "synthetic_fifo_capture.csv"
        save_synthetic_csv(captures, demo_csv)
        print(f"Wrote synthetic CSV: {demo_csv}")
    elif args.live:
        try:
            captures = try_live_capture(
                sample_count=args.samples, dll_path=args.dll, fpga_file=args.fpga
            )
            print("Live capture OK")
        except Exception as exc:
            print(f"Live capture unavailable: {exc}", file=sys.stderr)
            return 2
    elif args.input:
        captures = load_capture(Path(args.input))
        print(f"Loaded FIFOs {sorted(captures.keys())} from {args.input}")
    else:
        build_argparser().print_help()
        print(
            "\nProvide an input path, or use --demo / --live.",
            file=sys.stderr,
        )
        return 1

    assert captures is not None
    result = run_pipeline(
        captures=captures,
        out_dir=out_dir,
        fs_hz=args.fs,
        sound_speed=args.sound_speed,
        dynamic_range_db=args.dynamic_range,
        fifo_channels=fifo_channels,
        dedupe=args.dedupe,
        save_csv=args.save_csv,
    )

    print("\n=== B-mode complete ===")
    print(f"Channels (FIFOs): {result['meta']['channels']}")
    print(f"Duplicate-pair fractions: {result['meta']['dup_fraction']}")
    print(f"Dedupe policy: {result['meta']['dedupe']}")
    print(f"Samples: {result['n_samples']}")
    print(f"Max depth (provisional): {result['depth_max_m']*1e3:.2f} mm")
    print(f"B-mode PNG: {result['png']}")
    if result["csv"]:
        print(f"Envelopes CSV: {result['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
