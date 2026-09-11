# 04.RTL

Synthesizable HDL, one folder per module. Each folder holds a placeholder
`.v` file with the module name only, so the empty tree is trackable from
the first commit.

| Module | Path | Status |
|---|---|---|
| MDC (GCD) | `mdc/mdc.v` | not implemented |
| FFT (64-pt) | `fft/fft.v` | not implemented |
| Matrix inversion (4x4) | `matrix_inv/matrix_inv.v` | not implemented |
| LMS adaptive filter | `lms/lms.v` | not implemented |
| ML classifier | `ml_classifier/ml_classifier.v` | not implemented |
| CNN accelerator | `cnn/cnn.v` | not implemented |
| UART input | `uart/uart_rx.v`, `uart/uart_frame_buffer.v` | implemented |
| Top-level integration | `top/top.v` | not implemented |

Comment style: block header `/* ... */`, no dashed dividers, English only.
