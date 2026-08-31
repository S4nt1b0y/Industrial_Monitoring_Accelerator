# Open architectural decisions

Decisions made while building the Python reference model (`03.Reference`)
and the datasets it depends on. Each entry states the decision, the
alternatives considered, and why -- so a later change has the original
reasoning to argue against, not just the current code.

## Dataset

**Split by recording, grouped by (label, fault_detail), never by row.**
Rows from the same recording are temporally correlated (adjacent samples
of the same 60-300s run) -- a row-level split would leak near-duplicate
windows between train and test. Grouping by (label, fault_detail) rather
than by label alone matters too: with only 3 loads (0/2/4 Nm) per
severity, grouping by label alone can put a class's weakest severity
entirely in one split by chance, silently making the reported test
metric harder than the class actually is.

**Evaluation is grouped k-fold (3 folds), not a fixed train/val/test
split.** A fixed split permanently reserves 2/3 of the dataset's 45
recordings for validation/test and never trains on them. K-fold rotates
every recording through the test role exactly once, using the full
dataset far more efficiently -- important given how few recordings some
classes have (`operacao_normal`: 3 total).

**Ingestion normalization uses a fixed per-channel scale, chosen
deliberately per channel** (`dataset/signal_params.py`), not a
dynamically-computed one. For vibration channels, the scale-setting
statistic excludes `desgaste_rolamento`'s recordings: that class'
vibration amplitude is an outlier, and including it would compress the
usable numeric resolution for the other 3 classes. Every downstream
algorithm (LMS, FFT, decimation) expects its input already in this
normalized range -- LMS's default step size is numerically unstable on
raw physical-unit amplitudes otherwise.

**The rarest class (`operacao_normal`, 3 recordings total) gets extra,
overlapping sampling windows; every other class does not.**
`operacao_normal` is a stationary process -- sampling the same held-in
recording more densely (windows overlapping by 75%) sharpens the
model's estimate of what that class looks like without introducing
leakage (grouped k-fold still keeps a whole recording in one fold) or
synthetic data. Applying the same treatment to more than one class at
once was tried and rejected: classes then compete for decision-boundary
capacity, and the classes that already worked well get worse.

## Fundamental frequency (f0) and MDC

**f0 comes from the current signal, not vibration**, and uses a
dominance fallback around the GCD-based MDC algorithm rather than the
plain 3-peak GCD alone. On this dataset the current spectrum has one
clearly dominant peak with no reliable harmonic structure above the
noise floor -- forcing 3 peaks through GCD picks up noise-floor bumps at
essentially random bin positions. When the strongest peak dominates by
a wide, conservative margin, its own bin is used directly; the module
falls through to genuine GCD only when that dominance condition isn't
met (a signal that a real harmonic structure might be present).

**f0 is refined to sub-bin resolution via a small linear system solved
by `matrix_inv`** (parabolic interpolation through the 3 magnitudes
around the winning bin), rather than used as the raw bin-quantized
value. The raw bin-quantized value turned out to be an exact constant
across the entire dataset (every recording's rotation speed rounds to
the same bin) -- mathematically inert as a classifier input after
normalization. The refined value has real variance and measurably
improves classification.

## LMS adaptive filter

**Implemented and tested as its own module regardless of whether its
output feeds the ML classifier.** It is a required building block on
its own. It was tried as a feature source (RMS of the residual error)
and excluded from the feature vector: on this dataset it measurably
hurts the classifier's ability to separate `operacao_normal`, the
rarest and most important class to get right. This is a deliberate,
documented divergence from a literal reading of the spec's "classifier
receives the LMS filter's output" -- not an oversight.

## Feature vector for the MLP classifier

**All 4 vibration channels (2 accelerometers, one per bearing housing,
each biaxial) are used, not just one.** A fault's vibration signature
can be anisotropic between sensor axes -- already demonstrated for
`desalinhamento`, which separates noticeably better on one accelerometer
axis than the other. Physically, using only one channel would be a
shortcut specific to this dataset, not a generally defensible sensor
architecture.

**Two parallel feature configurations are kept trained side by side**
(`features/pipeline.py` / `pipeline_v2.py`, 129 vs. 133 features), rather
than picking one. The 4 extra features in v2 (autoregressive
coefficients via `matrix_inv`) are a genuine improvement at this
channel count -- every class improves, including the rarest one. Which
configuration (if either) an eventual RTL implementation targets is a
separate, later decision; both stay live and documented in the Python
reference model.

**`matrix_inv` was also tried as a system-identification feature (ARX:
vibration modeled as a response driven by motor current) and rejected.**
Tested directly against both feature configurations: neutral on v1,
consistently worse on v2. Not included in the shipped code -- the
supporting analysis is described in `matrix_inv/motor_parameters.py`'s
module docstring for anyone who wants to revisit it.

## CNN spectrogram

**The spectrogram is built from a decimated, low-frequency
representation of vibration, not the native-resolution FFT.** At native
resolution (Δf=400Hz), the harmonics of rotation that separate
`desbalanceamento`/`desalinhamento` from `operacao_normal` fall entirely
inside the discarded DC bin -- the same blind spot the MLP's feature
vector had before its own low-frequency spectrum was added. Both
constructions are kept in `cnn/reference.py`
(`build_spectrogram`/`build_lowfreq_spectrogram`) since the comparison
between them is itself informative, but only the low-frequency one is
trained and promoted.

**Only 2 channels (one accelerometer, "mancal A") are used, not 1 or
4, and not an ensemble of two models.** 1, 2, and 4 channels were
compared directly, along with a soft-voting ensemble of two
independently-trained models (one per accelerometer). 2 channels wins
on every class with no trade-off. 4 channels, and the ensemble, both
destabilize `operacao_normal` through overfitting (more convolution
parameters, same ~45 recordings) -- the ensemble in particular turned
out to be unreliable seed-to-seed rather than a genuine improvement (in
repeated runs, only 1 of 5 random seeds kept every class above the
target, and the average across seeds was worse than the simple 2-channel
model on 2 of the 4 classes). This does trade away some recall on
`desalinhamento`: the missing information lives in the other
accelerometer, and no combination method tried recovers it without
costing more elsewhere. See `03.Reference/README.md` for the numeric
comparison.

**The training loop normalizes its spectrogram input (z-score against
the training fold) before feeding the network.** This gap existed
silently until the low-frequency spectrogram was introduced: its
natural value scale is roughly 50x smaller than the native
spectrogram's, and the network's fixed-scale weight initialization
happened to only work by coincidence at the native scale. Training
without this normalization collapses to predicting a single majority
class regardless of the spectrogram source -- normalization is now
unconditional for both.

## Class imbalance in training

**Loss weighting is the inverse of each class' frequency in the
training fold** (`class_weights_from_counts`), the most aggressive
version of this kind of correction, not a softened one. Milder
alternatives (square-root of the inverse frequency, a capped weight
ratio, or no weighting at all) were all tried and were all worse on
every metric, for both classifiers -- for the CNN specifically, removing
the weight entirely collapses training to always predicting the
majority class. The weight is what keeps the rarest class learnable at
all; it is not the cause of that class' lower precision.

**Every training script reports precision and recall per class, not
only aggregate accuracy or a single recall-based "balanced accuracy".**
A recall-only metric can hide a classifier that simply over-predicts
the rare class whenever it's unsure -- high recall, low precision,
useless in practice. This was found directly: a config that looked
like a clean win under balanced accuracy (100% recall on
`operacao_normal`) turned out to have precision as low as 33% under
closer inspection. All reported results in `03.Reference/README.md` use
precision and recall together.
