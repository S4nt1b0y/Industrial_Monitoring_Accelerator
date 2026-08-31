# 03.Reference

Bit-true reference models (Python/NumPy), one per module, used as golden
vectors for the testbenches in `05.Sim` and to measure fixed-point vs.
floating-point error.

| Module | Status |
|---|---|
| mdc | not implemented |
| fft | not implemented |
| matrix_inv | not implemented |
| lms | not implemented |
| ml_classifier | implemented: see `artifacts/ml_classifier/README.md` |
| cnn | not implemented |

## ML classifier pipeline

The ML reference flow is split into three layers:

| File | Responsibility |
|---|---|
| `ml_classifier.py` | Pure reusable classifier API. Receives precomputed fixed-point feature matrices and labels. |
| `ml_pipeline.py` | Top processing chain. Receives four vibration channels, applies optional LMS, FFT, optional MDC, computes `n_features`, and calls `MLClassifier`. |
| `evaluate_datasets.py` | Dataset/logistics top. Reads Parquet datasets, balances windows by class, runs pipeline configurations, tests streaming classification, and writes comparison metrics. |

### `MLClassifier`

Use `MLClassifier` when features are already prepared:

```python
from ml_classifier import MLClassifier

clf = MLClassifier(n_features=N_FEATURES, data_width=16)
metrics = clf.train(features_train, labels_train)
classes = clf.classify(features_to_classify)
clf.save_artifacts("03.Reference/artifacts/ml_classifier/run_name")
```

Supported fixed-point formats:

| `data_width` | Format | Feature range |
|---:|---|---|
| 16 | Q1.15 | `0..32767` |
| 8 | Q1.7 | `0..127` |

### `MLPipeline`

Use `MLPipeline` when the input is four streams/channels of motor vibration samples:

```python
from ml_pipeline import MLPipeline

pipeline = MLPipeline(data_width=16, lms=True, mdc=True)
metrics = pipeline.train(ch0, ch1, ch2, ch3, labels)

valid, class_id = pipeline.classifier(sample_ch0, sample_ch1, sample_ch2, sample_ch3)
```

The pipeline uses fixed 64-sample windows. During streaming classification,
`classifier(...)` returns `(False, None)` until 64 samples have been buffered; on
the 64th sample it processes the window, clears the buffer, and returns
`(True, class_id)`.

Feature counts:

| Configuration | `n_features` |
|---|---:|
| FFT only, `mdc=False` | 132 |
| FFT + MDC `f0`/`valid`, `mdc=True` | 140 |

### Dataset evaluation top

Run all processed Parquet datasets through all pipeline configurations:

```bash
.venv/bin/python 03.Reference/evaluate_datasets.py
```

For a faster smoke run:

```bash
.venv/bin/python 03.Reference/evaluate_datasets.py --max-windows-per-class 200 --cv-folds 3
```

The script scans `07.Datasets/processed/*.parquet`, validates `label` plus the
four vibration channels, infers `data_width` from channel dtype (`int16` or
`int8`), balances complete 64-sample windows by the smallest class, and runs:

| Config name | LMS | MDC |
|---|---|---|
| `lms_on_mdc_off` | true | false |
| `lms_on_mdc_on` | true | true |
| `lms_off_mdc_off` | false | false |
| `lms_off_mdc_on` | false | true |

Outputs are written under:

```text
03.Reference/artifacts/dataset_evaluation/<dataset_stem>/<config_name>/
```

Each run writes classifier artifacts, `pipeline_config.json`, and `result.json`.
The top-level output directory also receives `comparison.json` and
`comparison.csv`, sorted by streaming accuracy and then internal test accuracy.

### Tests

Run all Python reference tests:

```bash
.venv/bin/python -m unittest discover -s 03.Reference -p 'test_*.py'
```
