"""NullFox image steganography — hide text invisibly inside images."""

from .lsb     import encode_lsb, decode_lsb   # default: zero visible distortion
from .encoder import encode                     # grid mode: JPEG-tolerant
from .decoder import decode

__all__ = ['encode_lsb', 'decode_lsb', 'encode', 'decode']
