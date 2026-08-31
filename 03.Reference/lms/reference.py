"""LMS adaptive filter (8 taps) -- reference model, floating-point ("ideal").

Bit-true in *structure* to the planned RTL datapath: one MAC per call
site, the same multiplier the hardware would reuse serially.

Configured as an adaptive line enhancer (ALE, Widrow & Hoff 1960):
d(n) = x(n), predicted from a delay line of x(n-1)..x(n-8), never x(n)
itself -- feeding x(n) into both d(n) and tap 0 on the same step would
give the trivial degenerate solution w_0=1, rest=0, e(n)=0 always. The
1-sample gap falls out of run_lms's own sequential structure (`history`
is read before being updated with the current sample, so it already
holds x(n-1)..x(n-8) -- the same decorrelation a pipelined datapath
gets for free from not being able to read and write a register in the
same cycle). `delay` adds further decorrelation on top of that; the
default (0) is already sufficient. This is the standard way to extract
periodic content from broadband noise without a separate noise
reference channel.

y(n) is the enhanced periodic estimate; e(n) is the noise residual,
larger when the signal is less periodic.
"""

from dataclasses import dataclass

import numpy as np

N_TAPS = 8
DEFAULT_MU = 2.0**-6  # power of 2 -> a shift in hardware


@dataclass
class LMSResult:
    y: np.ndarray
    e: np.ndarray
    weights_final: np.ndarray
    weights_history: np.ndarray
    cycles_per_sample: int
    total_cycles: int


def run_lms(x, mu=DEFAULT_MU, n_taps=N_TAPS, delay=0, weights_init=None):
    """Streams x through the ALE-configured LMS filter.

    x: 1-D array, raw samples (time order).
    delay: extra decorrelation on top of the 1-sample gap the sequential
      loop already provides for free (see module docstring). delay=0
      already predicts x(n) from x(n-1)..x(n-8), which is sufficient.
    """
    x = np.asarray(x, dtype=np.float64)
    n_samples = len(x)
    weights = (
        np.zeros(n_taps, dtype=np.float64)
        if weights_init is None
        else np.array(weights_init, dtype=np.float64)
    )

    y = np.zeros(n_samples)
    e = np.zeros(n_samples)
    weights_history = np.zeros((n_samples, n_taps))

    # tap-delay-line of the DELAYED signal, index 0 = most recent tap
    history = np.zeros(n_taps)

    for n in range(n_samples):
        d_n = x[n]
        tap_sample = x[n - delay] if n - delay >= 0 else 0.0

        y_n = float(np.dot(weights, history))  # 8 MACs
        e_n = d_n - y_n
        weights = weights + mu * e_n * history  # 8 MACs

        y[n] = y_n
        e[n] = e_n
        weights_history[n] = weights

        history = np.roll(history, 1)
        history[0] = tap_sample

    cycles_per_sample = 2 * n_taps
    return LMSResult(
        y=y,
        e=e,
        weights_final=weights,
        weights_history=weights_history,
        cycles_per_sample=cycles_per_sample,
        total_cycles=cycles_per_sample * n_samples,
    )
