"""matrix_inv as a motor-parameter estimator.

The enunciado (Secao 3.3, "Modulo de inversao de matriz") requires the
accelerator to "estimar parametros do motor a partir de um conjunto de
equacoes lineares obtidas com os dados dos sensores" via a matrix
inversion of dimension <=4x4, but leaves the specific linear system
open. This module proposes two concrete formulations, each building a
small (<=4x4) linear system from real sensor data and solving it with
the existing float reference (`matrix_inv.algorithms.gauss_jordan.
solve_float`) -- float first, fixed-point/RTL is a follow-up step, same
staging as every other module in this project.

1. Parabolic interpolation of the MDC/f0 spectral peak.
   `mdc.reference.estimate_fundamental_from_peaks` returns a
   bin-quantized f0 -- fitting a parabola through the winning bin and
   its two neighbors (a classic FFT peak-refinement technique) gives
   sub-bin resolution instead. The 3-point fit is a 3x3 linear system
   (Vandermonde matrix at x=-1,0,1), solved via matrix_inv rather than
   a closed-form vertex formula, so this genuinely exercises the
   module instead of just computing the same closed form directly.
   Adopted in `features/pipeline.py`: the raw bin-quantized f0 turned
   out to be a provably-inert constant across the whole dataset (zero
   variance -> zero gradient after normalization), while the refined
   version has real variance and measurably improves classification.

2. AR(4) coefficients via Yule-Walker, on the decimated low-frequency
   vibration segment -- a closed-form parametric alternative to the
   FFT spectrum, tested as a candidate feature. AR order 4 matches the
   enunciado's own <=4x4 ceiling for this module exactly. Adopted in a
   parallel pipeline v2 (`features/pipeline_v2.py`) rather than
   replacing v1: at 1-channel scale it was a genuine trade-off (helped
   most classes, hurt the rarest one), but at the official 4-channel
   scale it improves every class, including the rarest one.

A third formulation was also tried and rejected: ARX (AutoRegressive
with eXogenous input), modeling the vibration response as driven by
motor current (a proxy for load/torque ripple) via least squares on a
3x3 normal-equations system. Tested directly against both pipeline
versions: neutral on v1, consistently worse on v2. Not included here.
"""

import numpy as np

from matrix_inv.algorithms.gauss_jordan import solve_float

# x = -1, 0, 1 (bin offsets relative to the winning bin) -- rows are
# [x^2, x, 1] for the parabola y = A*x^2 + B*x + C.
_PARABOLA_VANDERMONDE = np.array([
    [1.0, -1.0, 1.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
])


def parabolic_peak_offset(mag_left, mag_center, mag_right):
    """Sub-bin peak offset (bins, clipped to [-0.5, 0.5]) from the 3
    magnitudes around a spectral peak, via matrix_inv.solve_float on the
    3x3 parabola-fit system. Returns 0.0 if the fit is singular or
    degenerate (a==0, no real curvature -- e.g. all 3 magnitudes equal)."""
    y = np.array([mag_left, mag_center, mag_right], dtype=np.float64)
    coeffs, _, singular = solve_float(_PARABOLA_VANDERMONDE, y)
    if singular:
        return 0.0
    a, b, _c = coeffs
    if a == 0:
        return 0.0
    return float(np.clip(-b / (2 * a), -0.5, 0.5))


def refine_f0_hz(spectrum, k0, fs_hz, n_fft):
    """Refined f0 (Hz) using the spectrum's neighbors of bin k0. Falls
    back to the raw bin-quantized value at the spectrum's edges (no
    neighbor available) -- same convention as bin_to_hz elsewhere."""
    if k0 <= 0 or k0 >= len(spectrum) - 1:
        return k0 * fs_hz / n_fft
    offset = parabolic_peak_offset(spectrum[k0 - 1], spectrum[k0], spectrum[k0 + 1])
    return (k0 + offset) * fs_hz / n_fft


def _autocorrelation(x, max_lag):
    """Biased autocorrelation estimate r(0..max_lag), mean-removed and
    scaled to unit variance (r(0)=1) -- the standard input to
    Yule-Walker. The unit-variance scaling matters here beyond
    numerical hygiene: matrix_inv.algorithms.gauss_jordan.solve_float's
    singularity check is an ABSOLUTE determinant threshold (1e-9),
    tuned for the O(1)-scale matrices this module normally sees. A raw
    vibration segment's autocorrelation lands around 1e-5 (r(0)~1e-4-
    1e-5 after ingestion normalization + decimation), and a 4x4
    Toeplitz built from that has a determinant around 1e-18 -- a false
    "singular" positive, not real rank deficiency (found by testing on
    real data: every single block was flagged singular before this
    fix). AR coefficients are exactly scale-invariant (scaling x by any
    constant c scales R and r in R*a=r by c^2 each, so a=R^-1*r is
    unchanged) -- normalizing here changes nothing about the answer,
    only fixes the false-positive threshold check downstream."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    r = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(max_lag + 1)])
    if r[0] == 0:
        return r
    return r / r[0]


def estimate_ar_coefficients(x, order=4):
    """AR(order) coefficients [a_1..a_order] via Yule-Walker
    (x(n) ~ sum_i a_i*x(n-i)), solved with matrix_inv on the order x
    order Toeplitz autocorrelation system -- order=4 is the enunciado's
    own <=4x4 ceiling for this module. Returns a zero vector for a
    silent/constant segment (r(0)==0) or a genuinely singular system,
    matching "no predictable structure found" rather than raising."""
    r = _autocorrelation(x, order)
    if r[0] == 0:
        return np.zeros(order)
    toeplitz = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
    rhs = r[1 : order + 1]
    coeffs, _, singular = solve_float(toeplitz, rhs)
    if singular:
        return np.zeros(order)
    return coeffs
