#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="../../.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

rm -f sim_fft.vvp rtl_output.txt python_output.txt

"$PYTHON_BIN" gen_fft_ref.py
test -f python_output.txt

iverilog -g2012 -o sim_fft.vvp \
    tb_fft.v \
    ../../04.RTL/fft/fftu_dif.v \
    ../../04.RTL/fft/butterfly_dif.v \
    ../../04.RTL/fft/addr_gen.v \
    ../../04.RTL/fft/twiddle_lut.v

vvp sim_fft.vvp
test -f rtl_output.txt

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
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_no}: expected 4 fields, got {len(fields)}")
        window, idx, re, im = map(int, fields)
        data[(window, idx)] = (re, im)
    return data


rtl = read_vectors("rtl_output.txt")
ref = read_vectors("python_output.txt")

errors = []
for window in range(10):
    for idx in range(64):
        key = (window, idx)
        if key not in rtl or key not in ref:
            errors.append(f"window {window:02d} bin {idx:02d}: missing value")
            continue

        rtl_re, rtl_im = rtl[key]
        ref_re, ref_im = ref[key]
        diff_re = rtl_re - ref_re
        diff_im = rtl_im - ref_im

        if abs(diff_re) > TOLERANCE or abs(diff_im) > TOLERANCE:
            errors.append(
                f"window {window:02d} bin {idx:02d}: "
                f"rtl=({rtl_re:6d},{rtl_im:6d}) "
                f"python=({ref_re:6d},{ref_im:6d}) diff=({diff_re:5d},{diff_im:5d})"
            )

if len(rtl) != 640 or len(ref) != 640:
    errors.append(f"expected 640 output samples, got rtl={len(rtl)} python={len(ref)}")

if errors:
    print("FAIL: FFT outputs differ")
    for error in errors:
        print(error)
    sys.exit(1)

print(f"PASS: all 10 windows / 640 FFT bins match within +/-{TOLERANCE} LSB")
PY
