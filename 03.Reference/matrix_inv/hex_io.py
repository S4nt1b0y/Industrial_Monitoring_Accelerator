"""Shared .hex writer for matrix_inv testbench vectors (see 05.Sim/matrix_inv/README.md)."""

from pathlib import Path

import numpy as np

def write_hex(path, raw_array, width_bits=16):
    mask = (1 << width_bits) - 1
    nibbles = width_bits // 4
    lines = [f"{int(v) & mask:0{nibbles}x}" for v in np.asarray(raw_array).flatten()]
    Path(path).write_text("\n".join(lines) + "\n")
