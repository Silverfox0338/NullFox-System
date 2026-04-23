"""
Encoder: embeds a UTF-8 string into an image via grid-based luminance quantisation.

Protocol
--------
The image is divided into a (cols × rows) grid of (cell_size × cell_size) pixel blocks.
Each cell stores one bit by snapping the cell's mean luminance to a quantisation bin;
the parity of the bin index encodes 0 or 1.

Cell layout (reading left-to-right, top-to-bottom):
  cells [0 .. HEADER_BITS-1]   →  4-byte header  (magic + payload length)
  cells [HEADER_BITS .. end]   →  RS-encoded payload bits
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils import (
    DEFAULT_CELL_SIZE, DEFAULT_STRENGTH, DEFAULT_ECC_NSYM,
    HEADER_BITS, MARKER_SIZE,
    rs_encode, bytes_to_bits,
    ensure_rgb, grid_dims, cell_capacity,
    encode_bit_in_cell,
    pack_header,
)


# ── Alignment markers ─────────────────────────────────────────────────────────
# A 3×3 block of cells with a checkerboard pattern is stamped in each corner.
# The decoder uses the same pattern to verify orientation and calibrate the
# quantisation threshold.

_MARKER_PATTERN = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
]


def _stamp_alignment_markers(arr: np.ndarray, cell_size: int,
                              strength: int, cols: int, rows: int) -> None:
    """Overwrite the four corner 3×3 cell blocks with the fixed checkerboard."""
    offsets = [
        (0, 0),                           # top-left
        (0, cols - MARKER_SIZE),          # top-right
        (rows - MARKER_SIZE, 0),          # bottom-left
        (rows - MARKER_SIZE, cols - MARKER_SIZE),  # bottom-right
    ]
    for (r_off, c_off) in offsets:
        for dr in range(MARKER_SIZE):
            for dc in range(MARKER_SIZE):
                bit = _MARKER_PATTERN[dr][dc]
                encode_bit_in_cell(arr, r_off + dr, c_off + dc, cell_size, bit, strength)


# ── Data cell iterator ────────────────────────────────────────────────────────

def _data_cells(cols: int, rows: int):
    """
    Yield (row, col) for every non-marker cell, left-to-right, top-to-bottom.
    Cells occupied by corner alignment markers are skipped.
    """
    marker_cells = set()
    for r_off, c_off in [
        (0, 0), (0, cols - MARKER_SIZE),
        (rows - MARKER_SIZE, 0), (rows - MARKER_SIZE, cols - MARKER_SIZE),
    ]:
        for dr in range(MARKER_SIZE):
            for dc in range(MARKER_SIZE):
                marker_cells.add((r_off + dr, c_off + dc))

    for row in range(rows):
        for col in range(cols):
            if (row, col) not in marker_cells:
                yield row, col


# ── Public API ────────────────────────────────────────────────────────────────

def encode(
    image_path: str,
    text: str,
    output_path: str,
    cell_size: int  = DEFAULT_CELL_SIZE,
    strength: int   = DEFAULT_STRENGTH,
    ecc_nsym: int   = DEFAULT_ECC_NSYM,
    debug: bool     = False,
) -> None:
    """
    Encode `text` into the image at `image_path` and save to `output_path`.

    Parameters
    ----------
    cell_size : pixels per grid cell.  Must match the value used at decode time.
    strength  : luminance quantisation step.  Larger = more robust but more visible.
                Recommended range: 8 (invisible) – 30 (visible but reliable).
    ecc_nsym  : Reed-Solomon parity bytes.  More = can correct more errors, less capacity.
    debug     : if True, overlay the grid and bit values onto the saved image.
    """
    image = ensure_rgb(Image.open(image_path))
    W, H  = image.size
    cols, rows = grid_dims(image, cell_size)

    # ── Capacity check ────────────────────────────────────────────────────────
    available_cells = sum(1 for _ in _data_cells(cols, rows))
    available_bits  = available_cells - HEADER_BITS   # subtract header
    available_bytes = available_bits // 8

    rs_payload     = rs_encode(text.encode('utf-8'), ecc_nsym)
    payload_length = len(rs_payload)
    total_bits     = HEADER_BITS + payload_length * 8

    if total_bits > available_cells:
        raise ValueError(
            f'Message too long. '
            f'Image can hold {available_bytes - ecc_nsym} chars '
            f'(after ECC), but encoded payload needs {payload_length} bytes '
            f'({total_bits} bits) in a {cols}×{rows} cell grid. '
            f'Use a larger image, smaller cell_size, or shorter text.'
        )

    # ── Assemble bit stream ───────────────────────────────────────────────────
    header_bits  = bytes_to_bits(pack_header(payload_length))
    payload_bits = bytes_to_bits(rs_payload)
    all_bits     = header_bits + payload_bits

    # ── Write bits into image ─────────────────────────────────────────────────
    arr = np.array(image)

    cell_iter = _data_cells(cols, rows)
    for bit in all_bits:
        row, col = next(cell_iter)
        encode_bit_in_cell(arr, row, col, cell_size, bit, strength)

    # Stamp alignment markers last so they are not overwritten by data.
    _stamp_alignment_markers(arr, cell_size, strength, cols, rows)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_image = Image.fromarray(arr)
    out_image.save(output_path)           # always save the clean encoded image

    print(f'[encode] {len(text)} chars -> {payload_length} RS bytes -> {total_bits} bits')
    print(f'[encode] Grid: {cols}x{rows} cells ({cell_size}px each), '
          f'{available_cells} data cells available')
    print(f'[encode] Saved to: {output_path}')

    if debug:
        # Write debug overlay to a separate file so the encoded image is never corrupted.
        debug_path = output_path.rsplit('.', 1)[0] + '.debug.png'
        _draw_debug_overlay(out_image, all_bits, cols, rows, cell_size).save(debug_path)
        print(f'[encode] Debug grid saved to: {debug_path}')


# ── Debug overlay ─────────────────────────────────────────────────────────────

def _draw_debug_overlay(image: Image.Image, bits: list,
                         cols: int, rows: int, cell_size: int) -> Image.Image:
    """Draw grid lines and bit values over the image for inspection."""
    overlay = image.copy().convert('RGBA')
    draw    = ImageDraw.Draw(overlay, 'RGBA')

    # Semi-transparent grid lines
    for c in range(cols + 1):
        x = c * cell_size
        draw.line([(x, 0), (x, rows * cell_size)], fill=(255, 255, 0, 80), width=1)
    for r in range(rows + 1):
        y = r * cell_size
        draw.line([(0, y), (cols * cell_size, y)], fill=(255, 255, 0, 80), width=1)

    # Bit values (small text in each cell)
    bit_idx = 0
    for row in range(rows):
        for col in range(cols):
            if bit_idx >= len(bits):
                break
            x = col * cell_size + 2
            y = row * cell_size + 2
            colour = (0, 255, 0, 200) if bits[bit_idx] else (255, 0, 0, 200)
            draw.text((x, y), str(bits[bit_idx]), fill=colour)
            bit_idx += 1

    return overlay.convert('RGB')
