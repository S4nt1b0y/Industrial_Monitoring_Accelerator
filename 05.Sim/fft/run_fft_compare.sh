#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="../../.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

rm -f sim_fft.vvp rtl_output.txt python_output.txt

iverilog -g2012 -o sim_fft.vvp \
    tb_fft.v \
    ../../04.RTL/fft/fftu_dif.v \
    ../../04.RTL/fft/butterfly_dif.v \
    ../../04.RTL/fft/addr_gen.v \
    ../../04.RTL/fft/twiddle_lut.v

vvp sim_fft.vvp
"$PYTHON_BIN" gen_fft_ref.py

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import sys

TOLERANCE = 2


def read_vectors(path):
    data = {}
    for line_no, line in enumerate(Path(path).read_text(encoding="ascii").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_no}: expected 3 fields, got {len(fields)}")
        idx, re, im = map(int, fields)
        data[idx] = (re, im)
    return data


rtl = read_vectors("rtl_output.txt")
ref = read_vectors("python_output.txt")

errors = []
for idx in range(64):
    if idx not in rtl or idx not in ref:
        errors.append(f"bin {idx:02d}: missing value")
        continue

    rtl_re, rtl_im = rtl[idx]
    ref_re, ref_im = ref[idx]
    diff_re = rtl_re - ref_re
    diff_im = rtl_im - ref_im

    if abs(diff_re) > TOLERANCE or abs(diff_im) > TOLERANCE:
        errors.append(
            f"bin {idx:02d}: rtl=({rtl_re:6d},{rtl_im:6d}) "
            f"python=({ref_re:6d},{ref_im:6d}) diff=({diff_re:5d},{diff_im:5d})"
        )

if errors:
    print("FAIL: FFT outputs differ")
    for error in errors:
        print(error)
    sys.exit(1)

print(f"PASS: all 64 FFT bins match within +/-{TOLERANCE} LSB")
PY
