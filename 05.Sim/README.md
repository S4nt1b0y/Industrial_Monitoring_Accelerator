# 05.Sim

Testbenches, mirrors `04.RTL` one-to-one. Each folder holds a placeholder
`tb_<module>.v` so the empty tree is trackable from the first commit.

| Module | Testbench | Status |
|---|---|---|
| MDC (GCD) | `mdc/tb_mdc.v` | not implemented |
| FFT (64-pt) | `fft/tb_fft.v` | not implemented |
| Matrix inversion (2x2/3x3/4x4, 4 algorithms) | `matrix_inv/` — see its `README.md` | GJ + cofactors: 36/36 cases passing each (N=2/3/4); LU/QR pending |
| LMS adaptive filter | `lms/tb_lms.v` | not implemented |
| ML classifier | `ml_classifier/tb_ml_classifier.v` | not implemented |
| CNN accelerator | `cnn/tb_cnn.v` | not implemented |
| Top-level integration | `top/tb_top.v` | not implemented |

Golden reference vectors come from `03.Reference` (Python/NumPy models).
