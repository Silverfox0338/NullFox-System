"""
LSB (Least Significant Bit) steganography.

Each bit of the payload is hidden in the LSB of the blue channel of one pixel.
Maximum per-pixel change: ±1 out of 255.  Completely invisible.

Bits are distributed across pixels using a deterministic PRNG permutation so
they're spread evenly over the image rather than concentrated in one corner.
The same seed must be used for encode and decode (default is a fixed constant).

Constraint: the output MUST be saved as PNG (or another lossless format).
One JPEG re-encode will destroy all encoded bits.
"""

import numpy as np
from PIL import Image

from .utils import (
    DEFAULT_ECC_NSYM, HEADER_BITS,
    rs_encode, rs_decode,
    bytes_to_bits, bits_to_bytes,
    ensure_rgb, ensure_png_path,
    pack_header, unpack_header,
)

# Blue channel (index 2) is the least perceptible to human vision.
_CHANNEL = 2
_DEFAULT_SEED = 0x4E46_4F58  # "NFOX" in ASCII — fixed spread seed


def _spread_order(n_pixels: int, seed: int) -> np.ndarray:
    """Deterministic permutation of pixel indices for bit distribution."""
    return np.random.default_rng(seed).permutation(n_pixels)


# ── Public API ────────────────────────────────────────────────────────────────

def encode_lsb(
    image_path: str,
    text: str,
    output_path: str,
    ecc_nsym: int = DEFAULT_ECC_NSYM,
    seed: int     = _DEFAULT_SEED,
    debug: bool   = False,
) -> None:
    """
    Embed `text` into `image_path` via LSB substitution, save to `output_path`.
    Output must be a lossless format (PNG).  JPEG will destroy the payload.
    """
    output_path = ensure_png_path(output_path)
    image = ensure_rgb(Image.open(image_path))
    arr   = np.array(image)
    H, W  = arr.shape[:2]
    n_px  = H * W

    # ── Build full bit stream ─────────────────────────────────────────────────
    payload    = rs_encode(text.encode('utf-8'), ecc_nsym)
    header     = pack_header(len(payload))
    bits       = bytes_to_bits(header) + bytes_to_bits(payload)
    n_bits     = len(bits)

    if n_bits > n_px:
        raise ValueError(
            f'Message too long: payload needs {n_bits} pixels but image '
            f'only has {n_px}. Use a larger image or shorter text.'
        )

    # ── Write LSBs ────────────────────────────────────────────────────────────
    order = _spread_order(n_px, seed)
    flat  = arr.reshape(-1, 3)                    # (n_px, 3) view

    indices = order[:n_bits]
    bit_arr = np.array(bits, dtype=np.uint8)

    # Zero the LSB of the blue channel at chosen pixels, then OR in the bit.
    flat[indices, _CHANNEL] = (flat[indices, _CHANNEL] & 0xFE) | bit_arr

    Image.fromarray(arr).save(output_path)

    usage_pct = 100 * n_bits / n_px
    print(f'[encode-lsb] {len(text)} chars -> {len(payload)} RS bytes -> {n_bits} bits')
    print(f'[encode-lsb] Pixel usage: {n_bits}/{n_px} ({usage_pct:.1f}%)')
    print(f'[encode-lsb] Saved to: {output_path}')

    if debug:
        _save_debug(image, arr, order, n_bits, output_path)


def decode_lsb(
    image_path: str,
    ecc_nsym: int = DEFAULT_ECC_NSYM,
    seed: int     = _DEFAULT_SEED,
    debug: bool   = False,
) -> str:
    """Extract and return the text hidden in `image_path`."""
    image = ensure_rgb(Image.open(image_path))
    arr   = np.array(image)
    H, W  = arr.shape[:2]
    n_px  = H * W

    order = _spread_order(n_px, seed)
    flat  = arr.reshape(-1, 3)

    # ── Read header ───────────────────────────────────────────────────────────
    header_indices = order[:HEADER_BITS]
    header_bits    = (flat[header_indices, _CHANNEL] & 1).tolist()
    header_bytes   = bits_to_bytes(header_bits)
    payload_length = unpack_header(header_bytes)   # raises ValueError on bad magic
    print(f'[decode-lsb] Header OK - expecting {payload_length} RS bytes.')

    # ── Read payload ──────────────────────────────────────────────────────────
    n_payload_bits   = payload_length * 8
    payload_indices  = order[HEADER_BITS : HEADER_BITS + n_payload_bits]
    payload_bits     = (flat[payload_indices, _CHANNEL] & 1).tolist()
    payload_bytes    = bits_to_bytes(payload_bits)

    original = rs_decode(payload_bytes, ecc_nsym)
    text     = original.decode('utf-8')
    print(f'[decode-lsb] Recovered {len(original)} bytes.')

    if debug:
        _save_debug(image, arr, order, HEADER_BITS + n_payload_bits,
                    image_path, suffix='.lsb_debug.png')

    return text


# ── Debug helper ──────────────────────────────────────────────────────────────

def _save_debug(
    original: Image.Image,
    encoded_arr: np.ndarray,
    order: np.ndarray,
    n_bits: int,
    ref_path: str,
    suffix: str = '.lsb_debug.png',
) -> None:
    """
    Save a difference-amplified image showing which pixels were touched.
    The diff is scaled x50 so single-bit changes become visible for inspection.
    """
    orig_arr = np.array(original).astype(np.int16)
    enc_arr  = encoded_arr.astype(np.int16)
    diff     = np.abs(enc_arr - orig_arr)

    amplified = np.clip(diff * 50, 0, 255).astype(np.uint8)
    debug_path = ref_path.rsplit('.', 1)[0] + suffix
    Image.fromarray(amplified).save(debug_path)
    print(f'[lsb-debug] Diff map (x50) saved to: {debug_path}')
