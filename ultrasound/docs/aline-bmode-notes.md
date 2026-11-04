# A-line / B-mode offline pipeline

Script: [fifo-to-aline-bmode.py](/cursor/stores/self/docs/fifo-to-aline-bmode.py)  
Companion capture diagnostics: [compare-fifo.py](/cursor/stores/self/docs/compare-fifo.py)

## Quick start

```bash
# Smoke test (no board) — writes PNG + synthetic CSV
python fifo-to-aline-bmode.py --demo --out ./ultrasound_output --save-csv

# Offline from lab export
python fifo-to-aline-bmode.py live.csv --out ./ultrasound_output --save-csv

# Folder of per-FIFO files (fifo_2.csv … fifo_7.csv)
python fifo-to-aline-bmode.py ./fifo_dump --out ./ultrasound_output

# Optional live (Windows lab PC with HadBoardApi.dll + board powered)
python fifo-to-aline-bmode.py --live --out ./ultrasound_output
```

### CLI defaults

| Flag | Default | Meaning |
|------|---------|---------|
| `--fs` | `40e6` | Sample rate (Hz) |
| `--sound-speed` | `1540` | Tissue sound speed (m/s) |
| `--dynamic-range` | `40` | Log-compress window (dB) |
| `--fifos` | `2,3,6,7` | FIFO → B-mode scan-line map |
| `--dedupe` | `keep` | Duplicate-pair policy |
| `--display` | `sector` | Primary `bmode.png`: `sector` (clinical fan) or `rect` (A-line stack) |
| `--sector-deg` | `80` | Sector sweep angle (degrees) |
| `--image-width` / `--image-height` | `640` | Sector image size (px) |
| `--out` | `ultrasound_output` | Output directory |

Outputs: `bmode.png` (primary), `bmode_sector.png`, `bmode_rect.png`, `aline_fifo_*.png`, `rf_signed.npy`, `envelope_log.npy`, `depth_m.npy`, `bmode_sector.npy`, optional `envelopes.csv`.

## Sector / clinical display

Default display is a **grayscale sector (fan)** via polar scan conversion: each log-compressed A-line is a ray from an apex at the top; rays span `--sector-deg` about vertical-down; bilinear interpolate in (line, depth); outside the sector is black.

Honest constraint: with only **4 channels** (and/or manual probe motion) this matches the clinical *format* of an apical/phased view, **not** a commercial A4C frame. Dense anatomy needs many more steering angles or channels later. Use `--display rect` when you want the raw rectangular A-line stack for debug.

## Decode assumptions

1. **Word format**: 16-bit FIFO words; AD9670 data in **bits 15:4** (lower 4 bits often zero in lab dumps).
2. **12-bit code**: `(word >> 4) & 0xFFF`.
3. **Offset-binary → signed RF**: `code - 2048` (mid-scale 2048 → 0).
4. **Active channels (for now)**: FIFOs **2, 3, 6, 7** become the four B-mode scan lines (left → right). Quiet FIFOs are ignored unless you change `--fifos`.
5. **Envelope**: numpy FFT analytic signal (`abs(hilbert)` equivalent). No scipy.
6. **Log compress**: peak-normalize each A-line, clip to `[-DR, 0]` dB, map to `[0, 1]`.
7. **Depth**: `z = 0.5 * c * n / fs`. **Provisional** — no TX start-time offset yet; do not treat mm scale as calibrated.
8. **Duplicate pairs**: Lab captures often show high consecutive-duplicate counts (possible paired/interleaved layout). Default **`--dedupe keep`** retains every sample. If images look “stretched” or duplicates dominate (~50%+), retry with `--dedupe drop` (collapse consecutive identical raw words). Document which setting you used when sharing results.

Note: `compare-fifo.py` still sets `ADC_BITS = 14` and comments bits 15:2, but its **active** diagnostic path is the same 12-bit `>> 4` rule used here. Lock alignment against a known reflector before changing the shift.

## How Immersive Team should export from the lab script

`compare-fifo.py` currently prints stats only; it does **not** write CSV. After a good capture on the Windows lab PC:

### Option A — extend the existing capture (recommended)

After the `captures[fifo] = words` loop in `compare-fifo.py`, add a short export:

```python
import csv
out = r"D:\Echo-cardiogram\live.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"fifo_{i}" for i in range(8)])
    for r in range(SAMPLE_COUNT):
        w.writerow([int(captures[i][r]) & 0xFFFF for i in range(8)])
print("Wrote", out)
```

Then on any machine:

```bash
python fifo-to-aline-bmode.py D:\Echo-cardiogram\live.csv --out D:\Echo-cardiogram\ultrasound_output --save-csv
```

### Option B — per-FIFO files

Write `fifo_0.csv` … `fifo_7.csv` (one integer per line, or a single column). Point the pipeline at the folder:

```bash
python fifo-to-aline-bmode.py D:\Echo-cardiogram\fifo_dump --out ...
```

Accepted names: `fifo_2.csv`, `fifo2.npy`, `FIFO_3.csv`, `ch6.csv`, etc.

### Option C — NPY

Shape `(8192, 8)` or `(8, 8192)` raw uint16 words; columns/rows map to FIFO index order 0…7 (or use a CSV header `fifo_2,...` if only active channels are exported).

**Export raw 16-bit words**, not the centred RF — let this pipeline apply `>> 4` and −2048 so decode stays in one place.

## Live path

`--live` loads `HadBoardApi.dll` via pythonnet, configures `ad9670_lvds_mar2012.bin`, `FillFifos(8192)`, then `CaptureFifoWords` for FIFOs 0–7. On Linux/cloud (no DLL), the script exits with a clear message and code 2 — offline/`--demo` still work.

## Smoke test artifact

Synthetic capture used for CI-less verification:

- `/cursor/stores/self/internal/synthetic_fifo_capture.csv`
- Outputs under `/cursor/stores/self/artifacts/ultrasound_output/`
