# 03.Reference

Bit-true reference models (Python/NumPy), one per module, used as golden
vectors for the testbenches in `05.Sim` and to measure fixed-point vs.
floating-point error.

| Module | Status |
|---|---|
| mdc | implemented (float reference + Euclidean-GCD dominance fallback) |
| fft | implemented (radix-2 DIF, 64-pt, float reference) |
| matrix_inv | implemented (Gauss-Jordan + cofactors, float and fixed-point) |
| lms | implemented (8-tap adaptive line enhancer, float reference) |
| ml_classifier | implemented, trained (MLP, two parallel feature configs -- see below) |
| cnn | implemented, trained (convolutional classifier over a vibration spectrogram) |

## Setup

```
pip install -r requirements.txt
pytest
```

`requirements.txt` installs this package in editable mode plus its dev
and notebook extras (pytest, Jupyter -- see the file itself). Running
`notebooks/analise_smma.ipynb` end to end installs the same thing from
its own first cell, so a fresh clone needs no manual setup beyond
opening the notebook.

## Dataset

Raw sensor data (`.mat`/`.tdms`) is not committed -- it's a public ~3GB
archive on [Mendeley Data](https://data.mendeley.com/datasets/ztmf3m7h5x).
Fetch and build everything the notebook needs in one step:

```
python -m tools.bootstrap_dataset   # or: make dataset
```

This downloads the archive, extracts every `.mat`/`.tdms` file flat into
`07.Datasets/` (see `dataset/paths.py`), and runs the processing chain
(split, k-fold groups, features, spectrograms) into `07.Datasets/processed/`.
Each step is skipped if its output already exists; pass `--force` to
rebuild from scratch. See `tools/bootstrap_dataset.py`'s module
docstring for the exact chain and what it deliberately leaves out.

## Layout

- `dataset/` -- filesystem/dataset-schema constants, ingestion
  normalization, train/val/test split and k-fold grouping (by
  recording, never by row -- see `dataset/split.py`), EDA utilities.
- `fft/`, `mdc/`, `lms/`, `matrix_inv/` -- one module per required
  building block (enunciado Section 3), each a direct floating-point
  reference of its planned RTL datapath.
- `matrix_inv/motor_parameters.py` -- the two concrete uses found for
  matrix inversion in this pipeline: sub-bin refinement of the
  fundamental rotation frequency, and autoregressive coefficients of
  the vibration signal (both solved as small linear systems via
  `matrix_inv.algorithms.gauss_jordan`).
- `features/` -- assembles the feature vector the MLP classifier
  consumes, from raw vibration + current sensor blocks. Two parallel,
  independently-trained configurations exist side by side:
  - `pipeline.py` ("v1"): fundamental frequency (refined via
    `matrix_inv`) plus the low-frequency vibration spectrum across all
    4 accelerometer channels -- 129 features.
  - `pipeline_v2.py` ("v2"): v1's vector plus 4 autoregressive
    coefficients from `matrix_inv` -- 133 features. Which of the two (if
    either) becomes the RTL target is a separate, later decision; both
    are kept trained and documented.
- `ml_classifier/` -- the MLP itself (`reference.py`/`reference_v2.py`,
  matching the two feature configs above). Inference is argmax-only
  over raw logits, no softmax, matching the planned hardware datapath.
- `cnn/` -- a convolutional classifier over a spectrogram built from
  vibration. Two spectrogram constructions exist: the direct,
  native-resolution one the enunciado's own wording suggests
  (`build_spectrogram`), and a decimated, low-frequency one
  (`build_lowfreq_spectrogram`) needed because the native resolution
  buries the rotation harmonics that separate imbalance from
  misalignment inside the discarded DC bin. The low-frequency,
  2-channel ("mancal A") config is the one actually trained and
  promoted -- see the classification results below.
- `tools/` -- CLI scripts that build the official datasets and train
  the official models end to end, from raw `.mat`/`.tdms` files in
  `07.Datasets/` through to the saved `.npz` weights in
  `07.Datasets/processed/`.

## Note on the LMS adaptive filter

`lms/reference.py` is implemented and tested as its own module
regardless of whether its output feeds the ML classifier -- it is a
required building block on its own (enunciado Section 3), independent
of that question. It was tried as a feature source (the residual
error's RMS) and excluded: on this dataset it measurably hurts the
classifier's ability to separate the rarest class (`operacao_normal`).
See `features/pipeline.py`'s module docstring for the detail.

## Classification results

Both classifiers are evaluated with grouped k-fold cross-validation (3
folds, grouped by recording so no window from the same session appears
in both train and test) and reported as precision/recall per class,
averaged over 5 random seeds. The official weights come from a final
training pass over every available recording, once the k-fold estimate
already validated the approach.

| Class | MLP v2 -- precision | MLP v2 -- recall | CNN -- precision | CNN -- recall |
|---|---|---|---|---|
| `operacao_normal` | 87.5% | 83.4% | 76.4% | 99.9% |
| `desbalanceamento` | 88.4% | 94.1% | 99.9% | 83.0% |
| `desalinhamento` | 79.5% | 78.0% | 100% | 66.7% |
| `desgaste_rolamento` | 99.6% | 99.8% | ~100% | ~99.8% |

`desalinhamento`'s recall on the CNN is a known, physical limitation,
not a training deficiency: the 2-channel CNN only sees one of the two
accelerometers (see `cnn/reference.py`'s module docstring), and the
information that would resolve this class fully lives in the other
one. Adding it back in (either as 4 combined channels, or as an
ensemble of two separately-trained models) was tried and rejected: it
destabilizes the other 3 classes -- particularly the already-scarce
`operacao_normal` -- more than it helps this one.
