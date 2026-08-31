"""Sampling/window parameters, reconciled against the real dataset.

FS_HZ is read off the actual MAT/TDMS extraction (every one of the 45
sources shares the same increment, see
07.Datasets/processed/dataset_report.csv, mat_increment_s =
3.90625e-05 for all rows). N_FFT is fixed by the enunciado (Section
3.2), not a free design parameter.

At this fs, an N=64 window spans 2.5 ms, not the ~10 ms a naive reading
of the support material implies (that figure assumes fs ~= 6.4 kHz,
4x lower than this dataset's 25.6 kHz).
"""

import numpy as np

FS_HZ = 25_600.0
N_FFT = 64

SAMPLE_PERIOD_S = 1.0 / FS_HZ
WINDOW_DURATION_S = N_FFT / FS_HZ
FREQ_RESOLUTION_HZ = FS_HZ / N_FFT

# Ingestion normalization scales. Dividing a raw physical-unit sample by
# these maps it to roughly [-1, 1] -- desgaste_rolamento's vibration
# amplitude is an outlier and is excluded from the scale-setting
# statistic so it doesn't compress the usable resolution for the other
# 3 classes; current keeps the plain global max, since it doesn't have
# the same class-skew problem. Every signal-processing module (LMS,
# FFT, decimation) expects its input already in this normalized range --
# LMS's default step size is numerically unstable on raw amplitudes.
VIBRATION_INGESTION_SCALE = 18.95675413617375  # aceleracao_x_mancal_a, excl. desgaste_rolamento
VIBRATION_Y_INGESTION_SCALE = 28.53223997224331  # aceleracao_y_mancal_a, same recipe
VIBRATION_XB_INGESTION_SCALE = 8.100730171473325  # aceleracao_x_mancal_b, same recipe
VIBRATION_YB_INGESTION_SCALE = 7.016384318092012  # aceleracao_y_mancal_b, same recipe
CURRENT_INGESTION_SCALE = 5.520720347145732  # corrente_fase_u, global max (dataset_scale_report.csv)


def normalize_ingestion(raw_samples, scale):
    """float64 physical units -> ~[-1, 1] (saturating clip)."""
    return np.clip(np.asarray(raw_samples, dtype=np.float64) / scale, -1.0, 1.0)
