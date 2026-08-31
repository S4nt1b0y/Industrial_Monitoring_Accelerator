"""The matrix_inv "validador": generates .hex test vectors for 05.Sim/matrix_inv.

Usage (from 03.Reference):
    python -m tools.gen_matrix_inv_vectors --algo gj --dims 2 3 4
"""

import argparse
import json
from pathlib import Path

import numpy as np

from matrix_inv import reference
from matrix_inv.fixed_point import quantize
from matrix_inv.hex_io import write_hex

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "05.Sim" / "matrix_inv" / "vectors"


def random_well_conditioned(n, rng):
    a = rng.uniform(-4.0, 4.0, size=(n, n))
    for i in range(n):
        off_diag_sum = np.sum(np.abs(a[i])) - abs(a[i, i])
        a[i, i] = off_diag_sum + rng.uniform(2.0, 6.0)
    b = rng.uniform(-8.0, 8.0, size=n)
    return a, b


def random_near_singular(n, rng):
    a, b = random_well_conditioned(n, rng)
    # Perturbation well under the Q8.8 LSB (1/256 ~ 0.0039): float det is a
    # tiny nonzero number, but quantization collapses row 0/1 to
    # (near-)proportional, so the fixed-point pivot trips the epsilon.
    a[1, :] = a[0, :] + rng.normal(0.0, 0.0005, size=n)
    return a, b


def random_singular(n, rng):
    # Row 1 is forced equal to row 0 in the *quantized* domain (write_case),
    # not here: matching in float space and quantizing independently can
    # round row 0 and row 1 apart, silently producing a non-singular Q8.8
    # matrix.
    a = rng.uniform(-4.0, 4.0, size=(n, n))
    b = rng.uniform(-8.0, 8.0, size=n)
    return a, b


# Order/count relied upon by the Verilog testbenches, which build case
# folder names formulaically (case_00.._03 = well_conditioned, 04..07 =
# near_singular, 08..11 = singular for the default --cases-per-category=4)
# instead of parsing cases.txt. Keep 05.Sim/matrix_inv/tb_matrix_inv_gj.v
# in sync if this changes.
CATEGORY_GENERATORS = {
    "well_conditioned": random_well_conditioned,
    "near_singular": random_near_singular,
    "singular": random_singular,
}


def generate_cases(n, cases_per_category, rng):
    for category, gen in CATEGORY_GENERATORS.items():
        for _ in range(cases_per_category):
            yield category, *gen(n, rng)


def write_case(case_dir, algo, n, category, a_raw, b_raw):
    if category == "singular" and n > 1:
        # Exact raw-domain duplication (not float-domain proportionality,
        # see random_singular): guarantees the fixed-point pivot search
        # hits an exact zero, independent of quantization rounding.
        a_raw = a_raw.copy()
        a_raw[1, :] = a_raw[0, :]

    case_dir.mkdir(parents=True, exist_ok=True)
    x_raw, ainv_raw, singular = reference.solve(algo, a_raw, b_raw, n)

    write_hex(case_dir / "A.hex", a_raw)
    write_hex(case_dir / "b.hex", b_raw)
    if singular:
        write_hex(case_dir / "x_expected.hex", np.zeros(n, dtype=np.int64))
        write_hex(case_dir / "ainv_expected.hex", np.zeros(n * n, dtype=np.int64))
    else:
        write_hex(case_dir / "x_expected.hex", x_raw)
        write_hex(case_dir / "ainv_expected.hex", ainv_raw)

    meta = {"category": category, "singular_expected": bool(singular)}
    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Plain-text sidecar for the Verilog testbenches (JSON parsing isn't
    # worth it there): single "0" or "1" line.
    (case_dir / "singular.txt").write_text(("1" if singular else "0") + "\n")
    return singular


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", default="gj", choices=sorted(reference.SOLVERS))
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--cases-per-category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    total = 0
    for n in args.dims:
        cases = list(generate_cases(n, args.cases_per_category, rng))
        case_names = []
        for idx, (category, a, b) in enumerate(cases):
            a_raw = quantize(a)
            b_raw = quantize(b)
            case_name = f"case_{idx:02d}_{category}"
            case_dir = args.output_dir / args.algo / f"n{n}" / case_name
            singular = write_case(case_dir, args.algo, n, category, a_raw, b_raw)
            case_names.append(case_name)
            total += 1
            print(f"n={n} {case_name}: singular={singular} -> {case_dir}")

        # Manifest read by the Verilog testbenches ($fopen/$fgets) so the
        # case list/count isn't hardcoded on the RTL side.
        n_dir = args.output_dir / args.algo / f"n{n}"
        (n_dir / "cases.txt").write_text("\n".join(case_names) + "\n")

    print(f"Wrote {total} cases to {args.output_dir / args.algo}")


if __name__ == "__main__":
    main()
