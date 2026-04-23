"""
CLI entry point.

Modes
-----
  lsb  (default) — invisible (max ±1 per pixel), PNG only, no resize support.
  zone           — invisible (max ±1 per pixel), PNG only, survives moderate
                   resizing via scale-invariant percentage zones + heavy ECC.
  grid           — near-invisible, survives light JPEG compression, no resize.

Usage
-----
  Encode:
    python -m steganography encode input.png "secret text" output.png
    python -m steganography encode input.png "secret text" output.png --mode zone
    python -m steganography encode input.png "secret text" output.png --mode zone --n-zones 32
    python -m steganography encode input.png "secret text" output.png --mode grid --strength 16

  Decode:
    python -m steganography decode output.png
    python -m steganography decode output.png --mode zone
    python -m steganography decode output.png --mode zone --n-zones 32

Parameters that affect the bit layout (mode, n-zones, cell-size, ecc-nsym) MUST
match between encode and decode runs.
"""

import argparse
import sys

from .encoder import encode
from .decoder import decode
from .lsb     import encode_lsb,  decode_lsb
from .zone    import (encode_zone, decode_zone, DEFAULT_N_ZONES, DEFAULT_ECC_ZONE,
                      encode_zone_tiled, decode_zone_tiled,
                      DEFAULT_TILE_PX, DEFAULT_TILE_ECC)
from .utils   import DEFAULT_CELL_SIZE, DEFAULT_STRENGTH, DEFAULT_ECC_NSYM, ensure_png_path


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        '--mode', choices=['lsb', 'zone', 'grid'], default='lsb',
        help='lsb (default): invisible, no resize support.  '
             'zone: invisible, survives moderate resizing.  '
             'grid: near-invisible, survives light JPEG.',
    )
    p.add_argument(
        '--ecc-nsym', type=int, default=None, metavar='N',
        help='Reed-Solomon parity bytes (default: 20 for lsb/grid, 40 for zone). '
             'More = corrects more errors, less capacity.',
    )
    p.add_argument(
        '--debug', action='store_true',
        help='Save a debug visualisation alongside the output.',
    )


def _add_zone_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        '--n-zones', type=int, default=DEFAULT_N_ZONES, metavar='N',
        help=f'[zone mode] Grid size: N x N zones (default: {DEFAULT_N_ZONES}). '
             'Use 32 for 1024px+ images.  Must match at decode time.',
    )
    p.add_argument(
        '--tiled', action='store_true',
        help='[zone mode] Embed data in every tile_px×tile_px block for crop resilience. '
             'Survives any crop that retains a complete tile.',
    )
    p.add_argument(
        '--tile-px', type=int, default=DEFAULT_TILE_PX, metavar='N',
        help=f'[zone --tiled] Tile size in pixels (default: {DEFAULT_TILE_PX}). '
             'The image must be at least this size (in each dimension) after cropping.',
    )


def _add_grid_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        '--cell-size', type=int, default=DEFAULT_CELL_SIZE, metavar='N',
        help=f'[grid mode] Pixels per cell (default: {DEFAULT_CELL_SIZE}).',
    )
    p.add_argument(
        '--strength', type=int, default=DEFAULT_STRENGTH, metavar='N',
        help=f'[grid mode] Quantisation step (default: {DEFAULT_STRENGTH}). '
             'Larger = more JPEG-robust, slightly more visible.',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m steganography',
        description='Hide text invisibly inside images.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    enc = sub.add_parser('encode', help='Embed text into an image.')
    enc.add_argument('input',  help='Carrier image (PNG, JPEG, ...).')
    enc.add_argument('text',   help='Text to embed.')
    enc.add_argument('output', help='Output image path (PNG required for lsb/zone).')
    _add_common_args(enc)
    _add_zone_args(enc)
    _add_grid_args(enc)

    dec = sub.add_parser('decode', help='Extract text from an encoded image.')
    dec.add_argument('input', help='Encoded image path.')
    _add_common_args(dec)
    _add_zone_args(dec)
    _add_grid_args(dec)

    return parser


def _ecc(args, mode: str) -> int:
    """Return ecc_nsym: explicit flag wins, otherwise mode-appropriate default."""
    if args.ecc_nsym is not None:
        return args.ecc_nsym
    if mode == 'zone':
        return DEFAULT_TILE_ECC if getattr(args, 'tiled', False) else DEFAULT_ECC_ZONE
    return DEFAULT_ECC_NSYM


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    try:
        mode = args.mode

        if args.command == 'encode':
            ecc = _ecc(args, mode)
            if mode == 'lsb':
                encode_lsb(
                    image_path  = args.input,
                    text        = args.text,
                    output_path = ensure_png_path(args.output),
                    ecc_nsym    = ecc,
                    debug       = args.debug,
                )
            elif mode == 'zone':
                if args.tiled:
                    encode_zone_tiled(
                        image_path  = args.input,
                        text        = args.text,
                        output_path = ensure_png_path(args.output),
                        tile_px     = args.tile_px,
                        n_zones     = args.n_zones,
                        ecc_nsym    = ecc,
                    )
                else:
                    encode_zone(
                        image_path  = args.input,
                        text        = args.text,
                        output_path = ensure_png_path(args.output),
                        n_zones     = args.n_zones,
                        ecc_nsym    = ecc,
                        debug       = args.debug,
                    )
            else:  # grid
                encode(
                    image_path  = args.input,
                    text        = args.text,
                    output_path = ensure_png_path(args.output),
                    cell_size   = args.cell_size,
                    strength    = args.strength,
                    ecc_nsym    = ecc,
                    debug       = args.debug,
                )

        elif args.command == 'decode':
            ecc = _ecc(args, mode)
            if mode == 'lsb':
                text = decode_lsb(
                    image_path = args.input,
                    ecc_nsym   = ecc,
                    debug      = args.debug,
                )
            elif mode == 'zone':
                if args.tiled:
                    text = decode_zone_tiled(
                        image_path = args.input,
                        tile_px    = args.tile_px,
                        n_zones    = args.n_zones,
                        ecc_nsym   = ecc,
                    )
                else:
                    text = decode_zone(
                        image_path = args.input,
                        n_zones    = args.n_zones,
                        ecc_nsym   = ecc,
                        debug      = args.debug,
                    )
            else:  # grid
                text = decode(
                    image_path = args.input,
                    cell_size  = args.cell_size,
                    strength   = args.strength,
                    ecc_nsym   = ecc,
                    debug      = args.debug,
                )
            print(f'\nDecoded text:\n{text}')

    except ValueError as e:
        print(f'[{args.command}] Error: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'[{args.command}] Failed: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
