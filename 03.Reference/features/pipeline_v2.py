"""Pipeline v2 -- v1's feature vector (features/pipeline.py) plus 4
AR(4) coefficients (matrix_inv.motor_parameters.estimate_ar_coefficients,
Yule-Walker via matrix_inv -- order 4 is the enunciado's own <=4x4
ceiling for that module).

Kept as a separate, parallel configuration rather than folded into v1:
the two are independently trained and compared, and which one (if
either) an eventual RTL implementation targets is a separate decision.
v1 stays untouched as the simpler baseline.

Reuses v1's extract_features unchanged (no duplicated FFT/decimation/
MDC/matrix_inv-peak-interpolation logic) and only adds the AR
coefficients, computed on the same x_mancal_a channel and decimated
segment (VIBRATION_DECIM_FACTOR=64, truncated to N_FFT samples).
"""

from dataclasses import dataclass

from scipy.signal import decimate

from dataset.signal_params import N_FFT, normalize_ingestion
from features.pipeline import (
    FEATURE_NAMES as FEATURE_NAMES_V1,
    VIBRATION_CHANNEL_SCALES,
    VIBRATION_DECIM_FACTOR,
    FeatureVector as FeatureVectorV1,
    extract_features as extract_features_v1,
)
from matrix_inv.motor_parameters import estimate_ar_coefficients

AR_ORDER = 4
AR_CHANNEL_INDEX = 0  # x_mancal_a

FEATURE_NAMES = FEATURE_NAMES_V1 + [f"ar_coef_{i}" for i in range(1, AR_ORDER + 1)]


@dataclass(frozen=True)
class FeatureVectorV2:
    v1: FeatureVectorV1
    ar_coefficients: tuple  # AR_ORDER floats, Yule-Walker via matrix_inv

    def as_tuple(self):
        return self.v1.as_tuple() + tuple(self.ar_coefficients)


def extract_features(vib_blocks, cur_block, **kwargs):
    v1_result = extract_features_v1(vib_blocks, cur_block, **kwargs)

    normalized = normalize_ingestion(vib_blocks[AR_CHANNEL_INDEX], VIBRATION_CHANNEL_SCALES[AR_CHANNEL_INDEX])
    vib_dec = decimate(normalized, VIBRATION_DECIM_FACTOR, ftype="fir")[:N_FFT]
    ar_coefficients = tuple(float(c) for c in estimate_ar_coefficients(vib_dec, order=AR_ORDER))

    return FeatureVectorV2(v1=v1_result, ar_coefficients=ar_coefficients)
