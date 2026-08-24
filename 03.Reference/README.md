# 03.Reference

Bit-true reference models (Python/NumPy), one per module, used as golden
vectors for the testbenches in `05.Sim` and to measure fixed-point vs.
floating-point error. This is a Python project: `pyproject.toml` +
`Makefile` at this level, one subpackage per module (`matrix_inv/` is the
first), shared `tests/` and `tools/` (vector generators).

Run tests with `make test` (or `python -m pytest tests -q` from this
directory) — do not create a file named `pytest.py` anywhere under this
tree, it shadows the real `pytest` package when imported from here.

| Module        | Status                                                                                                                                             |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| mdc           | not implemented                                                                                                                                    |
| fft           | not implemented                                                                                                                                    |
| matrix_inv    | Gauss-Jordan implemented (`algorithms/gauss_jordan.py`, bit-true to RTL); cofactors/LU/QR not implemented — see `04.RTL/matrix_inv/README.md` |
| lms           | not implemented                                                                                                                                    |
| ml_classifier | not implemented                                                                                                                                    |
| cnn           | not implemented                                                                                                                                    |

## How to run

```
cd 03.Reference
# requer numpy + pytest

make test                      # or: python -m pytest tests -q
```

To (re)generate the `.hex` vectors consumed by `05.Sim/matrix_inv`'s
testbenches:

```
python -m tools.gen_matrix_inv_vectors --algo gj --dims 2 3 4
```
