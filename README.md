# NullFox Steganography

A custom image steganography system that hides arbitrary text inside images with no visible colour distortion. Built from scratch — no borrowed encoding tech, no QR compatibility required.

> **Status: Active development / testing.** Core encoding is working and validated across all four modes. The web interface and resize-robustness under real-world conditions are still being evaluated.

---

## What it does

Takes an image and a message. Returns an image that looks pixel-perfect identical to the original. The message is invisible to the human eye and can be recovered by the decoder — and only the decoder.

It is not a QR code. It does not use DCT, JPEG coefficients, or any existing watermarking standard. The encoding scheme was designed from the ground up for this specific use case.

---

## How it works

### The four modes

#### `lsb` — Least Significant Bit (default)
The simplest and most invisible mode. Each bit of the message is written into the least significant bit of one pixel's blue channel. The maximum change to any pixel is **±1 out of 255** — mathematically undetectable to the human visual system.

Bits are distributed across the image using a deterministic PRNG permutation seeded with a fixed constant, so they're spread evenly rather than concentrated in a corner.

- Max per-pixel change: **1**
- PSNR vs original: **~82 dB** (effectively lossless)
- Capacity: scales with image resolution (~32,000 chars in a 512×512 image with default ECC)
- Constraint: **lossless format only** (PNG). One JPEG re-encode destroys all encoded bits.
- Resize robustness: **none** — pixel positions are absolute.

#### `zone` — Scale-Invariant Zone Encoding (custom)
Designed to survive image resizing while keeping the same ±1 per-pixel invisibility guarantee.

**Core insight:** bilinear/LANCZOS resampling is a locally linear operation. The arithmetic mean of a large enough pixel region is approximately preserved through any uniform resize, because interpolation kernel weights sum to 1. Individual pixel values get scrambled — the region mean does not.

**Protocol:**
1. The image is divided into an N×N grid of zones. Zone boundaries are defined as *fractions of image dimensions* — not fixed pixel offsets. A 32×32 grid on a 512px image uses 16×16px zones. The same 32×32 grid on a 1024px image uses 32×32px zones. Both decode identically.
2. Each zone encodes 1 bit via step-1 luminance quantisation: `_round(zone_mean_luminance) % 2`
3. To write a bit, random ±1 changes are dithered across zone pixels until the mean reaches the nearest integer with the correct parity. Step-1 math guarantees the required mean shift is always ≤ 1.0, so **no pixel ever changes by more than 1**.
4. Reed-Solomon ECC (40 parity bytes by default) corrects the small number of zones whose means drift enough after resampling to flip a bit.

`_round()` uses round-half-up (not Python banker's rounding) to stay consistent with JavaScript `Math.round()` — the web interface and CLI decode identically.

- Max per-pixel change: **1**
- PSNR vs original: **~53 dB** (imperceptible)
- Capacity: ~84 chars with n_zones=32, ecc=40 (independent of image resolution)
- Resize robustness: **±25–50%** depending on zone size and image resolution
- Constraint: PNG only. Aspect ratio must be preserved.

| Zone size | Image size     | Approximate resize tolerance |
|-----------|----------------|------------------------------|
| 32 × 32 px | 512 × 512    | ± 10–15 %                   |
| 32 × 32 px | 1024 × 1024  | ± 25 %                       |
| 64 × 64 px | 1024 × 1024  | ± 40 %                       |
| 64 × 64 px | 2048 × 2048  | ± 50 %                       |

#### `zone --tiled` — Crop-Resilient Tiled Zone Encoding
A variant of zone mode that stamps the same message redundantly into every `tile_px × tile_px` block across the image. Each tile is fully self-contained — zone boundaries are fractions of the tile, not the full image.

**Survives any crop that retains at least one complete tile.** The decoder scans all phase offsets automatically and returns the first valid tile it finds.

The trade-off is capacity: each tile must fit the full encoded payload, so practical message length is limited to what fits inside one tile (e.g. ~18 chars per tile with default settings: 256px tile, 16 zones, 10 ECC bytes).

- Max per-pixel change: **1**
- Capacity: ~18 chars with defaults (tile=256px, n_zones=16, ecc=10)
- Crop robustness: **any crop leaving one complete tile intact**
- Resize robustness: **none** — tile boundaries are fixed pixel offsets
- Constraint: PNG only. Image must be at least `tile_px × tile_px` pixels.

#### `grid` — Cell Luminance Quantisation
The original prototype mode. Divides the image into fixed-pixel cells and encodes bits by snapping each cell's mean luminance to an even/odd quantisation bin. More robust to light JPEG compression than LSB, but produces visible shifts (up to ±`strength` luminance units per pixel) in smooth image regions. Kept for situations where JPEG survival matters more than invisibility.

- Max per-pixel change: **up to ±strength** (default 12)
- JPEG survival: tolerates light JPEG (quality ≥ 75 approximately)
- Constraint: alignment markers in the four corners must not be cropped.

---

## Encoding layout

All modes share the same binary protocol:

```
[ 4-byte header ] [ RS-encoded payload ]
  └─ magic 0xABCD
  └─ payload length (uint16 big-endian)
```

The Reed-Solomon wrapper can correct up to `ecc_nsym / 2` byte errors before decoding fails.

---

## CLI usage

```bash
pip install -r requirements.txt
```

**Encode (invisible, PNG only):**
```bash
python -m steganography encode photo.png "secret message" output.png
```

**Decode:**
```bash
python -m steganography decode output.png
```

**Zone mode (resize-tolerant):**
```bash
python -m steganography encode photo.png "secret message" output.png --mode zone --n-zones 32
python -m steganography decode output.png --mode zone --n-zones 32
```

**Zone tiled mode (crop-resilient):**
```bash
python -m steganography encode photo.png "secret message" output.png --mode zone --tiled --tile-px 256
python -m steganography decode output.png --mode zone --tiled --tile-px 256
```

**Grid mode (light JPEG survival):**
```bash
python -m steganography encode photo.png "secret message" output.png --mode grid --strength 12
python -m steganography decode output.png --mode grid --strength 12
```

**Full option reference:**
```
encode  input  text  output  [--mode lsb|zone|grid]
                              [--n-zones N]       zone: grid size, default 16
                              [--tiled]           zone: enable crop-resilient tiling
                              [--tile-px N]       zone --tiled: tile size in px, default 256
                              [--cell-size N]     grid: pixels per cell, default 16
                              [--strength N]      grid: quantisation step, default 12
                              [--ecc-nsym N]      RS parity bytes, default 20 (lsb/grid) / 40 (zone) / 10 (zone --tiled)
                              [--debug]           save diff visualisation alongside output

decode  input               [same mode/layout flags as encode]
                              [--debug]
```

**Parameters that affect the bit layout (`--mode`, `--n-zones`, `--tiled`, `--tile-px`, `--cell-size`, `--ecc-nsym`) must match between encode and decode.**

---

## Web interface

A browser-based interface is available via GitHub Pages. It runs the full Python encoding engine in-browser using Pyodide (Python compiled to WebAssembly) — no server, no upload, fully client-side.

First load takes ~15 seconds (Pyodide + numpy + reedsolo download once and cache). After that it is instant.

All four modes are available and selectable from the header. Each mode surfaces its own parameter controls. The carrier image is stored after loading, so you can re-encode with different messages or settings without reloading the file. Drag-and-drop supported. Shows PSNR and usage statistics after encoding and downloads the result as a lossless PNG.

---

## Requirements

```
Pillow >= 10.0.0
numpy  >= 1.24.0
reedsolo >= 1.7.0
```

Python 3.10+.

---

## Project structure

```
steganography/
  utils.py       shared constants, ECC wrappers, bit I/O, luminance math
  lsb.py         LSB mode — encode / decode
  zone.py        Zone mode — encode / decode (resize-tolerant + tiled crop-resilient variant)
  encoder.py     Grid mode — encode
  decoder.py     Grid mode — decode
  cli.py         argparse CLI
  __init__.py    public API
  __main__.py    python -m steganography entry point
docs/
  index.html     GitHub Pages web interface (Pyodide, all four modes)
requirements.txt
```

---

## Known limitations

- **No camera / optical robustness.** This system is designed for digital file-to-file transfer. It is not compatible with phone cameras or physical printing.
- **JPEG destroys LSB and zone payloads.** Always output PNG. The grid mode tolerates light JPEG (quality ≥ 75 approximately).
- **Zone mode requires aspect ratio preservation.** Cropping, rotation, or non-uniform scaling will shift zone boundaries and cause decode failure. Use zone tiled mode if crop resilience is required.
- **Zone tiled mode has no resize tolerance.** Tile boundaries are fixed pixel offsets, so any resize will invalidate them.
- **Capacity scales with zone count, not image size.** In zone mode, the number of zones determines capacity regardless of resolution. With n_zones=32 and ecc=40, capacity is ~84 chars at any image size. Increase n_zones for more capacity.
- **This is a prototype.** The encoding scheme has not been subjected to formal cryptanalysis or steganalysis. It is not designed for adversarial environments where an attacker is actively looking for hidden data.

---

## License

Copyright (c) 2026 Silverfox0338. All rights reserved.

This software and all associated source code, algorithms, documentation, and assets are the original work of the author. No part of this project may be used, copied, modified, merged, published, distributed, sublicensed, sold, or incorporated into any other work — in whole or in part — without explicit written permission from the author.

Viewing the source for personal educational purposes is permitted. Everything else requires permission.

**This is not open source.**
