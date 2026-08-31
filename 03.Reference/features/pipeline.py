"""Assembles the MLP classifier's feature vector from raw vibration +
current blocks. Floating-point/"ideal" -- fixed-point quantization of
the feature vector is a hardware-side concern, not modeled here.

The vector is `f0_hz` (fundamental rotation frequency) plus the
low-frequency vibration spectrum of all 4 channels (2 biaxial
accelerometers, one per bearing housing) -- no RMS/energy-domain
scalars. Those were tried and dropped: with only 3 `operacao_normal`
recordings total (one per load), an RMS/energy feature correlates with
load more than with fault, so the classifier can learn "this load's
typical level = normal" instead of spectral shape. This includes the
residual error of the LMS adaptive filter -- `lms/reference.py` stays
implemented and tested as its own module regardless, it just isn't
wired into this vector.

All 4 vibration channels are used rather than one: a fault's vibration
signature can be anisotropic between sensor axes (`desalinhamento`
separates noticeably better on one accelerometer axis than the other),
so a sensor architecture covering both bearing housings and both radial
axes is the physically defensible choice.

`lowfreq_spectrum` is the usable low-frequency band (bins 1-32) of each
vibration channel, decimated by VIBRATION_DECIM_FACTOR=64
(Delta_f_dec=6.25Hz). At native resolution (Delta_f=400Hz) this whole
band falls inside the discarded DC bin. BLOCK_SAMPLES=4096 is the
minimum decimate() needs to produce one 64-sample decimated window at
that factor (n_fft * VIBRATION_DECIM_FACTOR), shared by every channel.

Every vibration/current channel passes through
dataset.signal_params.normalize_ingestion (its own per-channel scale)
before anything else touches it.

`f0_hz` comes from `matrix_inv.motor_parameters.refine_f0_hz`
(parabolic sub-bin interpolation, solved via matrix_inv) rather than
the raw bin-quantized value: at DECIM_FACTOR=32 (Delta_f_dec=12.5Hz),
the real rotation frequency always rounds to the same bin, so the raw
value is a constant with zero variance -- inert as a classifier input
after normalization. The refined value has real variance and
measurably improves classification.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import decimate

from dataset.signal_params import (
    CURRENT_INGESTION_SCALE,
    FS_HZ,
    N_FFT,
    VIBRATION_INGESTION_SCALE,
    VIBRATION_XB_INGESTION_SCALE,
    VIBRATION_Y_INGESTION_SCALE,
    VIBRATION_YB_INGESTION_SCALE,
    normalize_ingestion,
)
from fft.reference import (
    PeakDetectionException,
    fft_dif_radix2,
    magnitude_spectrum,
    top_3_local_maxima,
    unscramble,
)
from matrix_inv.motor_parameters import refine_f0_hz
from mdc.reference import MDCException, estimate_fundamental_from_peaks

DECIM_FACTOR = 32  # corrente/f0 path only
VIBRATION_DECIM_FACTOR = 64  # vibration low-freq spectrum
BLOCK_SAMPLES = N_FFT * VIBRATION_DECIM_FACTOR  # 4096

# 2 accelerometers (mancal A, mancal B), each biaxial (x, y) -- column
# order in the raw .mat matches build_dataset.py.
VIBRATION_CHANNEL_NAMES = ("x_A", "y_A", "x_B", "y_B")
VIBRATION_CHANNEL_SCALES = (
    VIBRATION_INGESTION_SCALE,
    VIBRATION_Y_INGESTION_SCALE,
    VIBRATION_XB_INGESTION_SCALE,
    VIBRATION_YB_INGESTION_SCALE,
)
N_VIBRATION_CHANNELS = len(VIBRATION_CHANNEL_NAMES)

# Usable band of the decimated-vibration spectrum (fs_dec=FS_HZ/VIBRATION_DECIM_FACTOR
# =400Hz, Delta_f_dec=6.25Hz, 0-200Hz) -- bin 0 excluded (DC), same policy as
# everywhere else.
LOWFREQ_BINS = tuple(range(1, 33))

FEATURE_NAMES = [
    "f0_hz",
    "f0_valido",
] + [f"lowfreq_{ch}_bin_{b:02d}" for ch in VIBRATION_CHANNEL_NAMES for b in LOWFREQ_BINS]


@dataclass(frozen=True)
class FeatureVector:
    f0_hz: float
    f0_valido: bool
    lowfreq_spectrum: tuple  # 128 floats: bins 1-32 of each of the 4 channels, channel-major

    def as_tuple(self):
        return (self.f0_hz, self.f0_valido) + tuple(self.lowfreq_spectrum)


def extract_features(vib_blocks, cur_block, fs_hz=FS_HZ, n_fft=N_FFT, decim_factor=DECIM_FACTOR):
    """vib_blocks: sequence of N_VIBRATION_CHANNELS arrays (x_A, y_A, x_B,
    y_B order), cur_block: 1 array -- all BLOCK_SAMPLES raw samples, same
    time span."""
    if len(vib_blocks) != N_VIBRATION_CHANNELS:
        raise ValueError(f"expected {N_VIBRATION_CHANNELS} vibration channels, got {len(vib_blocks)}")
    for block in (*vib_blocks, cur_block):
        if len(block) != BLOCK_SAMPLES:
            raise ValueError(f"expected {BLOCK_SAMPLES} samples per channel, got {len(block)}")

    cur_block = normalize_ingestion(cur_block, CURRENT_INGESTION_SCALE)
    cur_dec = decimate(cur_block, decim_factor, ftype="fir")
    fs_dec = fs_hz / decim_factor
    spectrum_cur = magnitude_spectrum(unscramble(fft_dif_radix2(cur_dec[:n_fft])))
    try:
        peaks = top_3_local_maxima(spectrum_cur)
        result = estimate_fundamental_from_peaks(peaks, fs_dec, n_fft)
        f0_hz, f0_valido = refine_f0_hz(spectrum_cur, result.k0, fs_dec, n_fft), True
    except (PeakDetectionException, MDCException):
        f0_hz, f0_valido = 0.0, False

    lowfreq_spectrum = []
    for block, scale in zip(vib_blocks, VIBRATION_CHANNEL_SCALES):
        normalized = normalize_ingestion(block, scale)
        vib_dec = decimate(normalized, VIBRATION_DECIM_FACTOR, ftype="fir")
        spectrum_vib_dec = magnitude_spectrum(unscramble(fft_dif_radix2(vib_dec[:n_fft])))
        lowfreq_spectrum.extend(float(v) for v in spectrum_vib_dec[list(LOWFREQ_BINS)])

    return FeatureVector(
        f0_hz=f0_hz,
        f0_valido=f0_valido,
        lowfreq_spectrum=tuple(lowfreq_spectrum),
    )
