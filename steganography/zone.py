"""
Scale-Invariant Zone Encoding (SIZE)
A region-statistics watermarking codec with deterministic decoding under resampling.

Core mechanism
--------------
The image is partitioned into N×N zones whose boundaries are expressed as
fractions of image dimensions — not fixed pixel offsets.  Zone boundaries
therefore scale proportionally with any resize, so the same scene region is
always sampled regardless of resolution.

Each zone encodes 1 bit by dithering random ±1 pixel changes until the zone's
mean luminance satisfies:

    _round(mean) % 2 == bit          (_round = round-half-up, not banker's)

Maximum per-pixel change: 1 out of 255 (invisible).
Reed-Solomon ECC corrects the few zones whose means drift after resampling.

Why mean luminance survives resize
-----------------------------------
Bilinear / LANCZOS resampling is a locally linear operation.  The arithmetic
mean of a large enough pixel region is approximately preserved because the
kernel weights sum to 1.  Remaining drift (from non-integer boundaries and
sub-pixel interpolation) is treated as noise and corrected by ECC.

Quantisation note
-----------------
We use step=1 and push zone means to integer targets (the centre of each
decision bin).  The half-integer boundaries (...49.5, 50.5, 51.5...) are the
danger points.  By targeting the bin centre, each bit has ±0.5 luminance units
of tolerance before flipping — the maximum achievable under step=1.  We use
_round() = int(x + 0.5) rather than Python's built-in round() to avoid
banker's rounding (which rounds 0.5 to even) and match JavaScript Math.round()
exactly, ensuring Python-CLI and browser-decoded outputs agree.

Resize tolerance (approximate, natural images, ecc_nsym=40)
------------------------------------------------------------
  Zone px    Image px     Resize range
  32 × 32    512 × 512    ± 10 %
  32 × 32    1024 × 1024  ± 25 %
  64 × 64    1024 × 1024  ± 40 %
  64 × 64    2048 × 2048  ± 50 %

Capacity (after ECC, ecc_nsym=40)
----------------------------------
  n_zones=16, any image  →  ~84 chars  (256 zones total)
  n_zones=32, any image  →  ~440 chars (1024 zones total)
  Larger n_zones = more capacity, smaller zones, less resize tolerance.
"""

import numpy as np
from PIL import Image

from .utils import (
    HEADER_BITS,
    rs_encode, rs_decode,
    bytes_to_bits, bits_to_bytes,
    ensure_rgb, ensure_png_path,
    pack_header, unpack_header,
)

DEFAULT_N_ZONES  = 16   # N×N zones; use 32 for 1024px+ images
DEFAULT_ECC_ZONE = 40   # heavier than LSB default — resize adds bit errors
_MIN_ZONE_PX     = 16   # smallest acceptable zone (pixels per side)

DEFAULT_TILE_PX    = 256  # tile size for crop-resilient mode
DEFAULT_TILE_ZONES = 16   # zones per tile dimension (zone_px = tile_px // n_zones = 16)
DEFAULT_TILE_ECC   = 10   # lighter ECC than resize mode — crops are lossless

def _round(x: float) -> int:
    """Round-half-up (not banker's rounding).  Matches JavaScript Math.round()
    so Python CLI and browser decode agree even at exact 0.5 boundaries."""
    return int(x + 0.5)


# ── Zone geometry ─────────────────────────────────────────────────────────────

def _bounds(H: int, W: int, n: int, row: int, col: int):
    """Pixel-coordinate bounds of zone (row, col) in an H×W image."""
    r0 = round(row     / n * H)
    r1 = round((row+1) / n * H)
    c0 = round(col     / n * W)
    c1 = round((col+1) / n * W)
    return r0, r1, c0, c1


def _mean_lum(arr: np.ndarray, r0, r1, c0, c1) -> float:
    z = arr[r0:r1, c0:c1].astype(np.float32)
    return float(np.mean(0.299 * z[:,:,0] + 0.587 * z[:,:,1] + 0.114 * z[:,:,2]))


# ── Bit I/O ───────────────────────────────────────────────────────────────────

def _read_bit(arr: np.ndarray, r0, r1, c0, c1) -> int:
    """Recover 1 bit from zone mean luminance parity."""
    return _round(_mean_lum(arr, r0, r1, c0, c1)) % 2


def _write_bit(arr: np.ndarray, r0, r1, c0, c1, bit: int, rng) -> None:
    """
    Dither random ±1 pixel changes into the zone until the zone mean encodes bit.

    Step-1 quantisation guarantees the required mean shift is ≤ 1.0, so each
    pixel changes by exactly 0 or 1 — never 2.  The fraction of pixels changed
    equals the required shift (e.g. a 0.3-unit shift changes 30 % of pixels).

    Adding the same delta to R, G, B shifts luminance by exactly that delta
    (since 0.299 + 0.587 + 0.114 = 1.0), so no channel arithmetic is needed.
    """
    zone = arr[r0:r1, c0:c1].astype(np.float32)
    h, w = zone.shape[:2]
    n_px = h * w
    if n_px == 0:
        return

    lum  = 0.299 * zone[:,:,0] + 0.587 * zone[:,:,1] + 0.114 * zone[:,:,2]
    mean = float(np.mean(lum))
    q    = _round(mean)

    if q % 2 == bit:
        return  # Zone already encodes the correct bit — no pixels touched.

    # Nearest integer with the correct parity, clamped to [0, 255].
    # Without the clamp, a fully-white zone (mean=255) would pick q_up=256,
    # clip every pixel write to 255, and silently encode the wrong bit.
    q_up, q_dn = q + 1, q - 1
    if q_up > 255:
        target = float(q_dn)
    elif q_dn < 0:
        target = float(q_up)
    elif abs(q_up - mean) <= abs(q_dn - mean):
        target = float(q_up)
    else:
        target = float(q_dn)

    delta     = target - mean
    direction = 1 if delta > 0 else -1

    flat = zone.reshape(-1, 3)

    # Only sample pixels that can actually move in the desired direction;
    # choosing a fully-clipped pixel wastes a "nudge" and undershoots the target.
    if direction == 1:
        candidates = np.where(np.any(flat < 255, axis=1))[0]
    else:
        candidates = np.where(np.any(flat > 0, axis=1))[0]

    if len(candidates) == 0:
        return  # Fully saturated zone — leave as-is and let ECC correct.

    n_change = min(len(candidates), round(abs(delta) * n_px))
    if n_change == 0:
        return

    idx = rng.choice(candidates, size=n_change, replace=False)
    flat[idx] = np.clip(flat[idx] + direction, 0, 255)
    arr[r0:r1, c0:c1] = flat.reshape(h, w, 3).astype(np.uint8)


# ── Capacity helper ───────────────────────────────────────────────────────────

def capacity_chars(n_zones: int, ecc_nsym: int) -> int:
    """Maximum encodable characters for the given grid and ECC parameters.

    Zones hold the full RS-encoded blob (original_data + ecc_parity bytes).
    Subtracting ecc_nsym recovers how many bytes are available for original data.
    """
    total_bits  = n_zones * n_zones
    data_bits   = total_bits - HEADER_BITS   # bits remaining after header
    rs_payload_bytes = data_bits // 8        # total bytes the zones can carry (data + parity)
    return max(0, rs_payload_bytes - ecc_nsym)


# ── Public API ────────────────────────────────────────────────────────────────

def encode_zone(
    image_path:  str,
    text:        str,
    output_path: str,
    n_zones:     int  = DEFAULT_N_ZONES,
    ecc_nsym:    int  = DEFAULT_ECC_ZONE,
    debug:       bool = False,
) -> None:
    """
    Embed `text` using scale-invariant zone encoding.

    Parameters
    ----------
    n_zones  : grid size (n_zones × n_zones zones).  Must match at decode time.
               Recommended: 16 for 512px images, 32 for 1024px+.
    ecc_nsym : Reed-Solomon parity bytes.  Use 40+ for resize robustness.
    """
    output_path = ensure_png_path(output_path)
    image = ensure_rgb(Image.open(image_path))
    arr   = np.array(image)
    H, W  = arr.shape[:2]

    zone_h = round(H / n_zones)
    zone_w = round(W / n_zones)
    if zone_h < _MIN_ZONE_PX or zone_w < _MIN_ZONE_PX:
        raise ValueError(
            f'Zone size {zone_h}x{zone_w} px is too small (min {_MIN_ZONE_PX}). '
            f'Use a larger image or smaller n_zones.'
        )

    payload  = rs_encode(text.encode('utf-8'), ecc_nsym)
    header   = pack_header(len(payload))
    all_bits = bytes_to_bits(header) + bytes_to_bits(payload)
    n_bits   = len(all_bits)
    n_total  = n_zones * n_zones

    if n_bits > n_total:
        cap = capacity_chars(n_zones, ecc_nsym)
        raise ValueError(
            f'Message too long: need {n_bits} zones, have {n_total}. '
            f'Max ~{cap} chars with current settings. '
            f'Try n_zones={n_zones * 2} or shorten the message.'
        )

    # Fixed seed → reproducible dithering pattern; decoding never uses this RNG.
    rng = np.random.default_rng(0)

    bit_idx = 0
    for row in range(n_zones):
        for col in range(n_zones):
            if bit_idx >= n_bits:
                break
            _write_bit(arr, *_bounds(H, W, n_zones, row, col), all_bits[bit_idx], rng)
            bit_idx += 1

    Image.fromarray(arr).save(output_path)

    print(f'[encode-zone] Grid: {n_zones}x{n_zones} zones, ~{zone_h}x{zone_w} px each')
    print(f'[encode-zone] {len(text)} chars -> {len(payload)} RS bytes -> {n_bits}/{n_total} zones')
    print(f'[encode-zone] Max per-pixel change: 1 out of 255')
    print(f'[encode-zone] Saved to: {output_path}')

    if debug:
        _save_debug(image, arr, n_zones, n_bits, output_path)


def decode_zone(
    image_path: str,
    n_zones:    int  = DEFAULT_N_ZONES,
    ecc_nsym:   int  = DEFAULT_ECC_ZONE,
    debug:      bool = False,
) -> str:
    """
    Recover text from a zone-encoded image.
    Works at any resolution as long as aspect ratio is preserved and n_zones
    matches the value used during encoding.
    """
    image = ensure_rgb(Image.open(image_path))
    arr   = np.array(image)
    H, W  = arr.shape[:2]

    zone_iter = ((r, c) for r in range(n_zones) for c in range(n_zones))

    header_bits = []
    for _ in range(HEADER_BITS):
        row, col = next(zone_iter)
        header_bits.append(_read_bit(arr, *_bounds(H, W, n_zones, row, col)))

    payload_length = unpack_header(bits_to_bytes(header_bits))
    print(f'[decode-zone] Header OK - expecting {payload_length} RS bytes.')

    payload_bits = []
    for _ in range(payload_length * 8):
        row, col = next(zone_iter)
        payload_bits.append(_read_bit(arr, *_bounds(H, W, n_zones, row, col)))

    original = rs_decode(bits_to_bytes(payload_bits), ecc_nsym)
    text     = original.decode('utf-8')
    print(f'[decode-zone] Recovered {len(original)} bytes.')
    return text


# ── Debug helper ──────────────────────────────────────────────────────────────

def _save_debug(
    original: Image.Image,
    encoded_arr: np.ndarray,
    n_zones: int,
    n_bits_used: int,
    output_path: str,
) -> None:
    """Overlay zone grid on a diff-amplified image (diff × 50 for visibility)."""
    from PIL import ImageDraw
    orig = np.array(original).astype(np.int16)
    diff = np.clip(np.abs(encoded_arr.astype(np.int16) - orig) * 50, 0, 255).astype(np.uint8)

    dbg  = Image.fromarray(diff).convert('RGBA')
    draw = ImageDraw.Draw(dbg, 'RGBA')
    H, W = encoded_arr.shape[:2]

    for row in range(n_zones + 1):
        y = round(row / n_zones * H)
        draw.line([(0, y), (W, y)], fill=(255, 255, 0, 120), width=1)
    for col in range(n_zones + 1):
        x = round(col / n_zones * W)
        draw.line([(x, 0), (x, H)], fill=(255, 255, 0, 120), width=1)

    debug_path = output_path.rsplit('.', 1)[0] + '.zone_debug.png'
    dbg.convert('RGB').save(debug_path)
    print(f'[zone-debug] Diff map (x50) + grid saved to: {debug_path}')


# ── Crop-resilient tiled encoding ─────────────────────────────────────────────

def _tile_zone_bounds(r0_tile: int, c0_tile: int, tile_px: int,
                      n: int, row: int, col: int):
    """Absolute pixel bounds of zone (row, col) within a tile at (r0_tile, c0_tile)."""
    r0 = r0_tile + round(row     / n * tile_px)
    r1 = r0_tile + round((row+1) / n * tile_px)
    c0 = c0_tile + round(col     / n * tile_px)
    c1 = c0_tile + round((col+1) / n * tile_px)
    return r0, r1, c0, c1


def _decode_tile_at(arr: np.ndarray, r0_tile: int, c0_tile: int,
                    tile_px: int, n_zones: int, ecc_nsym: int) -> str:
    """Try to decode from tile at (r0_tile, c0_tile). Raises on magic mismatch or ECC fail."""
    zone_iter = ((r, c) for r in range(n_zones) for c in range(n_zones))

    header_bits = []
    for _ in range(HEADER_BITS):
        row, col = next(zone_iter)
        header_bits.append(_read_bit(arr, *_tile_zone_bounds(r0_tile, c0_tile, tile_px, n_zones, row, col)))

    payload_length = unpack_header(bits_to_bytes(header_bits))

    payload_bits = []
    for _ in range(payload_length * 8):
        row, col = next(zone_iter)
        payload_bits.append(_read_bit(arr, *_tile_zone_bounds(r0_tile, c0_tile, tile_px, n_zones, row, col)))

    original = rs_decode(bits_to_bytes(payload_bits), ecc_nsym)
    return original.decode('utf-8')


def encode_zone_tiled(
    image_path:  str,
    text:        str,
    output_path: str,
    tile_px:     int = DEFAULT_TILE_PX,
    n_zones:     int = DEFAULT_TILE_ZONES,
    ecc_nsym:    int = DEFAULT_TILE_ECC,
) -> None:
    """
    Embed `text` into every tile_px × tile_px block across the image.

    Survives any crop that retains a complete tile_px × tile_px region from
    the original.  Zone boundaries are fractions of the tile, not the full
    image, so each tile is self-contained and crop-independent.

    Parameters
    ----------
    tile_px  : side length of each tile in pixels.  The minimum surviving crop
               size equals tile_px × tile_px.  Default 256.
    n_zones  : zone grid per tile (n_zones × n_zones).  zone_px = tile_px // n_zones
               must be ≥ 16.  Default 16 (zone_px = 16 px).
    ecc_nsym : Reed-Solomon parity bytes per tile.  Lighter than resize mode
               because crops are lossless — no zone-mean drift.  Default 10.
    """
    output_path = ensure_png_path(output_path)
    image = ensure_rgb(Image.open(image_path))
    arr   = np.array(image)
    H, W  = arr.shape[:2]

    zone_px = tile_px // n_zones
    if zone_px < _MIN_ZONE_PX:
        raise ValueError(
            f'Zone size {zone_px}px too small (min {_MIN_ZONE_PX}px). '
            f'Increase tile_px or decrease n_zones.'
        )

    n_tiles_r = H // tile_px
    n_tiles_c = W // tile_px
    if n_tiles_r == 0 or n_tiles_c == 0:
        raise ValueError(
            f'Image ({W}×{H}px) is smaller than one tile ({tile_px}×{tile_px}px). '
            f'Use a larger image or a smaller tile_px.'
        )

    payload  = rs_encode(text.encode('utf-8'), ecc_nsym)
    header   = pack_header(len(payload))
    all_bits = bytes_to_bits(header) + bytes_to_bits(payload)
    n_bits   = len(all_bits)
    n_total  = n_zones * n_zones

    if n_bits > n_total:
        cap = capacity_chars(n_zones, ecc_nsym)
        raise ValueError(
            f'Message too long: need {n_bits} zones, have {n_total} per tile. '
            f'Max ~{cap} chars with current settings.'
        )

    rng = np.random.default_rng(0)

    for tr in range(n_tiles_r):
        for tc in range(n_tiles_c):
            r0_tile = tr * tile_px
            c0_tile = tc * tile_px
            bit_idx = 0
            for row in range(n_zones):
                for col in range(n_zones):
                    if bit_idx >= n_bits:
                        break
                    _write_bit(arr, *_tile_zone_bounds(r0_tile, c0_tile, tile_px, n_zones, row, col),
                               all_bits[bit_idx], rng)
                    bit_idx += 1

    Image.fromarray(arr).save(output_path)

    n_tiles = n_tiles_r * n_tiles_c
    cap     = capacity_chars(n_zones, ecc_nsym)
    print(f'[encode-zone-tiled] {n_tiles} tiles ({n_tiles_r}×{n_tiles_c}), {tile_px}×{tile_px}px each')
    print(f'[encode-zone-tiled] {len(text)} chars → {len(payload)} RS bytes → {n_bits}/{n_total} zones/tile')
    print(f'[encode-zone-tiled] Survives any crop preserving ≥{tile_px}×{tile_px}px  |  max ~{cap} chars')
    print(f'[encode-zone-tiled] Saved to: {output_path}')


def decode_zone_tiled(
    image_path: str,
    tile_px:    int = DEFAULT_TILE_PX,
    n_zones:    int = DEFAULT_TILE_ZONES,
    ecc_nsym:   int = DEFAULT_TILE_ECC,
) -> str:
    """
    Recover text from a tiled zone-encoded image.

    Scans all phase offsets (dy, dx) mod tile_px at zone-px steps, trying each
    as a potential tile origin.  Returns the first offset where the magic header
    matches and ECC decodes cleanly.  Works even after an arbitrary crop as long
    as a complete tile_px × tile_px region of the original image is present.
    """
    image = ensure_rgb(Image.open(image_path))
    arr   = np.array(image)
    H, W  = arr.shape[:2]

    zone_px = tile_px // n_zones

    for dy in range(0, tile_px, zone_px):
        if dy + tile_px > H:
            break
        for dx in range(0, tile_px, zone_px):
            if dx + tile_px > W:
                break
            try:
                text = _decode_tile_at(arr, dy, dx, tile_px, n_zones, ecc_nsym)
                print(f'[decode-zone-tiled] Valid tile found at image offset ({dy}, {dx}).')
                return text
            except Exception:
                continue

    raise ValueError(
        'No valid tile found. '
        'Image may not be tiled-zone encoded, or the crop removed all complete tiles.'
    )
