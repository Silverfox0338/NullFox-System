"""
Decoder: extracts an encoded UTF-8 string from a modified image.

The decoder mirrors the encoder exactly:
  1. Read the first HEADER_BITS data cells -> 4-byte header -> verify magic, get length.
  2. Read `length * 8` more data cells -> RS-encoded payload bits.
  3. Reed-Solomon decode -> original UTF-8 text.

Alignment marker cells (four 3x3 corners) are skipped in both passes, exactly
as they are during encoding.
"""

import numpy as np
from PIL import Image, ImageDraw

from .utils import (
    DEFAULT_CELL_SIZE, DEFAULT_STRENGTH, DEFAULT_ECC_NSYM,
    HEADER_BITS, MARKER_SIZE,
    rs_decode, bits_to_bytes,
    ensure_rgb, grid_dims,
    cell_mean_lum, bit_from_mean,
    unpack_header,
)
from .encoder import _data_cells, _MARKER_PATTERN


# ── Alignment marker verification ─────────────────────────────────────────────

def _verify_alignment_markers(arr: np.ndarray, cell_size: int,
                               strength: int, cols: int, rows: int) -> dict:
    """
    Read the four corner markers and report agreement with the expected pattern.
    Returns a dict with a 'score' (0.0-1.0) and per-corner details.
    """
    offsets = {
        'top-left':     (0,                0),
        'top-right':    (0,                cols - MARKER_SIZE),
        'bottom-left':  (rows - MARKER_SIZE, 0),
        'bottom-right': (rows - MARKER_SIZE, cols - MARKER_SIZE),
    }
    report  = {}
    total   = 0
    correct = 0
    for name, (r_off, c_off) in offsets.items():
        corner_ok = 0
        corner_total = MARKER_SIZE * MARKER_SIZE
        for dr in range(MARKER_SIZE):
            for dc in range(MARKER_SIZE):
                expected = _MARKER_PATTERN[dr][dc]
                mean     = cell_mean_lum(arr, r_off + dr, c_off + dc, cell_size)
                got      = bit_from_mean(mean, strength)
                if got == expected:
                    corner_ok += 1
        report[name] = f'{corner_ok}/{corner_total}'
        correct += corner_ok
        total   += corner_total

    report['score'] = correct / total if total else 0.0
    return report


# ── Public API ────────────────────────────────────────────────────────────────

def decode(
    image_path: str,
    cell_size: int = DEFAULT_CELL_SIZE,
    strength: int  = DEFAULT_STRENGTH,
    ecc_nsym: int  = DEFAULT_ECC_NSYM,
    debug: bool    = False,
) -> str:
    """
    Decode and return the text embedded in the image at `image_path`.

    Parameters
    ----------
    cell_size / strength / ecc_nsym must match the values used during encoding.
    debug : if True, save a '<image_path>.debug.png' showing sampled bit values.
    """
    image = ensure_rgb(Image.open(image_path))
    arr   = np.array(image)
    cols, rows = grid_dims(image, cell_size)

    # ── Optional marker check ─────────────────────────────────────────────────
    marker_report = _verify_alignment_markers(arr, cell_size, strength, cols, rows)
    score = marker_report.pop('score')
    if score < 0.7:
        print(f'[decode] Warning: alignment marker agreement is only {score:.0%}. '
              'Check cell_size / strength or image integrity.')
    else:
        print(f'[decode] Alignment markers OK ({score:.0%} agreement).')

    # ── Read header ───────────────────────────────────────────────────────────
    cell_iter  = _data_cells(cols, rows)
    header_bits = []
    for _ in range(HEADER_BITS):
        row, col = next(cell_iter)
        mean = cell_mean_lum(arr, row, col, cell_size)
        header_bits.append(bit_from_mean(mean, strength))

    header_bytes = bits_to_bytes(header_bits)
    payload_length = unpack_header(header_bytes)     # raises on bad magic
    print(f'[decode] Header OK - expecting {payload_length} RS-encoded bytes.')

    # ── Read payload ──────────────────────────────────────────────────────────
    payload_bits = []
    sampled_means = []          # kept for debug overlay
    for _ in range(payload_length * 8):
        row, col = next(cell_iter)
        mean = cell_mean_lum(arr, row, col, cell_size)
        sampled_means.append((row, col, mean))
        payload_bits.append(bit_from_mean(mean, strength))

    payload_bytes = bits_to_bytes(payload_bits)

    # ── Reed-Solomon error correction ─────────────────────────────────────────
    original_bytes = rs_decode(payload_bytes, ecc_nsym)
    text = original_bytes.decode('utf-8')

    print(f'[decode] Recovered {len(original_bytes)} bytes -> "{text}"')

    # ── Debug overlay ─────────────────────────────────────────────────────────
    if debug:
        debug_path = image_path.rsplit('.', 1)[0] + '.debug.png'
        _save_debug(image, sampled_means, payload_bits, cols, rows, cell_size, debug_path)
        print(f'[decode] Debug image saved to: {debug_path}')

    return text


# ── Debug helper ──────────────────────────────────────────────────────────────

def _save_debug(image: Image.Image, sampled: list, bits: list,
                cols: int, rows: int, cell_size: int, path: str) -> None:
    """
    Save a copy of the image with each sampled cell coloured green (1) or red (0),
    annotated with the decoded bit and the raw mean luminance.
    """
    overlay = image.copy().convert('RGBA')
    draw    = ImageDraw.Draw(overlay, 'RGBA')

    # Grid lines
    for c in range(cols + 1):
        draw.line([(c * cell_size, 0), (c * cell_size, rows * cell_size)],
                  fill=(255, 255, 0, 60), width=1)
    for r in range(rows + 1):
        draw.line([(0, r * cell_size), (cols * cell_size, r * cell_size)],
                  fill=(255, 255, 0, 60), width=1)

    for i, (row, col, mean) in enumerate(sampled):
        bit    = bits[i]
        x, y   = col * cell_size, row * cell_size
        colour = (0, 200, 0, 80) if bit else (200, 0, 0, 80)
        draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill=colour)
        draw.text((x + 1, y + 1), f'{bit}\n{mean:.0f}', fill=(255, 255, 255, 200))

    overlay.convert('RGB').save(path)
