"""
Shared utilities: constants, ECC wrappers, bit I/O, cell sampling.
"""

import struct
import numpy as np
from PIL import Image
import reedsolo

# ── Protocol constants ────────────────────────────────────────────────────────

MAGIC = b'\xAB\xCD'          # 2-byte magic to identify encoded images
HEADER_BYTES = 4             # magic(2) + payload_length(2)
HEADER_BITS  = HEADER_BYTES * 8

DEFAULT_CELL_SIZE = 16       # pixels per grid cell (each cell = 1 bit)
DEFAULT_STRENGTH  = 12       # luminance quantisation step (larger = more robust, more visible)
DEFAULT_ECC_NSYM  = 20       # Reed-Solomon parity symbols (bytes)
MARKER_SIZE       = 3        # alignment marker block is MARKER_SIZE × MARKER_SIZE cells

# ── Reed-Solomon helpers ──────────────────────────────────────────────────────

def rs_encode(data: bytes, nsym: int = DEFAULT_ECC_NSYM) -> bytes:
    """Wrap data bytes in Reed-Solomon ECC."""
    rs = reedsolo.RSCodec(nsym)
    return bytes(rs.encode(data))


def rs_decode(payload: bytes, nsym: int = DEFAULT_ECC_NSYM) -> bytes:
    """Recover original bytes from RS-encoded payload (corrects up to nsym/2 byte errors)."""
    rs = reedsolo.RSCodec(nsym)
    decoded, _, _ = rs.decode(payload)
    return bytes(decoded)


# ── Bit-level helpers ─────────────────────────────────────────────────────────

def bytes_to_bits(data: bytes) -> list:
    """Convert bytes to a flat list of ints (MSB first)."""
    bits = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def bits_to_bytes(bits: list) -> bytes:
    """Convert a flat list of bit ints (MSB first) back to bytes."""
    result = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i : i + 8]
        chunk += [0] * (8 - len(chunk))          # pad last byte if needed
        result.append(sum(b << (7 - j) for j, b in enumerate(chunk)))
    return bytes(result)


# ── Grid helpers ──────────────────────────────────────────────────────────────

def grid_dims(image: Image.Image, cell_size: int) -> tuple:
    """Return (cols, rows) for the full grid covering the image."""
    W, H = image.size
    return W // cell_size, H // cell_size


def cell_capacity(image: Image.Image, cell_size: int) -> int:
    """Total number of cells (= bits) available in the grid."""
    cols, rows = grid_dims(image, cell_size)
    return cols * rows


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Return image as RGB (drop alpha, convert palette modes, etc.)."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


# ── Luminance helpers ─────────────────────────────────────────────────────────

def _cell_slice(row: int, col: int, cell_size: int):
    """Return numpy slice tuple for a cell."""
    y, x = row * cell_size, col * cell_size
    return slice(y, y + cell_size), slice(x, x + cell_size)


def cell_mean_lum(arr: np.ndarray, row: int, col: int, cell_size: int) -> float:
    """
    Mean luminance of a grid cell.
    arr is shape (H, W, 3) uint8, channels are RGB.
    Luminance = 0.299R + 0.587G + 0.114B  (Rec.601)
    """
    ys, xs = _cell_slice(row, col, cell_size)
    cell = arr[ys, xs].astype(np.float32)
    lum = 0.299 * cell[:, :, 0] + 0.587 * cell[:, :, 1] + 0.114 * cell[:, :, 2]
    return float(np.mean(lum))


# ── Quantisation codec ────────────────────────────────────────────────────────

def bit_from_mean(mean: float, strength: int) -> int:
    """
    Decode a bit from a cell's mean luminance.
    Parity of the quantisation bin index encodes the bit.
    """
    return int(round(mean / strength)) % 2


def target_mean_for_bit(mean: float, bit: int, strength: int) -> float:
    """
    Return the nearest quantisation-bin centre that encodes `bit`.
    Maximum delta from current mean is `strength` units.
    """
    q = round(mean / strength)
    if q % 2 == bit:
        return float(q * strength)
    # Flip to the adjacent bin that encodes the desired bit.
    q_up   = q + 1
    q_down = q - 1
    t_up   = q_up   * strength
    t_down = q_down * strength
    if abs(t_up - mean) <= abs(t_down - mean):
        return float(t_up)
    return float(t_down)


def encode_bit_in_cell(arr: np.ndarray, row: int, col: int,
                        cell_size: int, bit: int, strength: int) -> None:
    """
    Modify pixel values in a grid cell so the cell's mean luminance
    encodes `bit` under the quantisation scheme.

    Adding the same delta to R, G, B shifts luminance by exactly that
    delta (since 0.299+0.587+0.114 = 1.0), so the target is exact.
    """
    ys, xs = _cell_slice(row, col, cell_size)
    cell = arr[ys, xs].astype(np.float32)

    mean_lum = float(
        np.mean(0.299 * cell[:, :, 0] + 0.587 * cell[:, :, 1] + 0.114 * cell[:, :, 2])
    )
    target = target_mean_for_bit(mean_lum, bit, strength)
    delta  = target - mean_lum

    modified = np.clip(cell + delta, 0, 255).astype(np.uint8)
    arr[ys, xs] = modified


# ── Output path helpers ───────────────────────────────────────────────────────

def ensure_png_path(path: str) -> str:
    """
    Return `path` with the extension replaced by .png if it is a JPEG.
    JPEG output destroys LSB payloads, so we silently correct it rather than
    letting the caller write a broken file.
    """
    lower = path.lower()
    if lower.endswith('.jpg') or lower.endswith('.jpeg'):
        corrected = path.rsplit('.', 1)[0] + '.png'
        print(f'[info] Output changed from {path!r} to {corrected!r} '
              '(JPEG would destroy the payload).')
        return corrected
    return path


# ── Header pack / unpack ──────────────────────────────────────────────────────

def pack_header(payload_length: int) -> bytes:
    """Build 4-byte header: magic(2) + length(2, big-endian)."""
    return MAGIC + struct.pack('>H', payload_length)


def unpack_header(header_bytes: bytes) -> int:
    """
    Parse the 4-byte header.  Returns payload length.
    Raises ValueError if magic doesn't match.
    """
    if len(header_bytes) < HEADER_BYTES:
        raise ValueError('Header too short.')
    magic = header_bytes[:2]
    if magic != MAGIC:
        raise ValueError(
            f'Magic mismatch: expected {MAGIC!r}, got {magic!r}. '
            'Image may not contain encoded data.'
        )
    (length,) = struct.unpack('>H', header_bytes[2:4])
    return length
