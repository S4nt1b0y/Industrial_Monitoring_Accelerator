# 04.RTL

Synthesizable HDL, one folder per module. Each folder holds a placeholder
`.v` file with the module name only, so the empty tree is trackable from
the first commit.

| Module | Path | Status |
|---|---|---|
| MDC (GCD) | `mdc/mdc.v` | not implemented |
| FFT (64-pt) | `fft/fft.v` | not implemented |
| Matrix inversion (2x2/3x3/4x4, 4 algorithms) | `matrix_inv/` — see its `README.md` | Gauss-Jordan + cofactors implemented, N=2/3/4 passing; LU/QR pending |
| LMS adaptive filter | `lms/lms.v` | not implemented |
| ML classifier | `ml_classifier/ml_classifier.v` | not implemented |
| CNN accelerator | `cnn/cnn.v` | not implemented |
| Top-level integration | `top/top.v` | not implemented |

Comment style: block header `/* ... */`, no dashed dividers, English only.
