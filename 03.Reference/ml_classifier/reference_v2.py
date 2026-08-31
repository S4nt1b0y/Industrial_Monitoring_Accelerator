"""MLP classifier v2: identical architecture/training math to
`ml_classifier/reference.py` (v1) -- same 1 hidden layer, ReLU,
argmax-only inference, same backprop -- just a wider input: v1's
features (features/pipeline.py) plus the 4 AR(4) Yule-Walker
coefficients from features/pipeline_v2.py.

Kept as its own file, not a parametrized variant of reference.py, so
v1 stays completely untouched while both configurations are trained
and compared independently.
"""

from dataclasses import dataclass

import numpy as np

from features.pipeline_v2 import FEATURE_NAMES as STORAGE_FEATURE_NAMES

# STORAGE_FEATURE_NAMES includes f0_valido (a boolean, for parquet
# storage/diagnostics) -- excluded here, same convention as v1
# (ml_classifier/reference.py never feeds the classifier a boolean flag).
FEATURE_NAMES = [name for name in STORAGE_FEATURE_NAMES if name != "f0_valido"]
N_FEATURES = len(FEATURE_NAMES)  # 133 = 129 (v1) + 4 (AR coefficients)
N_CLASSES = 4
CLASS_NAMES = ["operacao_normal", "desbalanceamento", "desalinhamento", "desgaste_rolamento"]


@dataclass
class MLPWeights:
    w1: np.ndarray  # (H, N_FEATURES)
    b1: np.ndarray  # (H,)
    w2: np.ndarray  # (N_CLASSES, H)
    b2: np.ndarray  # (N_CLASSES,)


def init_weights(n_hidden=16, seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, np.sqrt(2.0 / N_FEATURES), (n_hidden, N_FEATURES))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (N_CLASSES, n_hidden))
    b2 = np.zeros(N_CLASSES)
    return MLPWeights(w1=w1, b1=b1, w2=w2, b2=b2)


def relu(x):
    return np.maximum(x, 0.0)


def forward(x, weights):
    h = relu(weights.w1 @ x + weights.b1)
    logits = weights.w2 @ h + weights.b2
    return logits, int(np.argmax(logits))


def forward_train(x_batch, weights):
    z1 = x_batch @ weights.w1.T + weights.b1
    a1 = relu(z1)
    logits = a1 @ weights.w2.T + weights.b2
    return z1, a1, logits


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def compute_loss_and_grads(x_batch, y_batch, weights, sample_weights=None):
    n = len(y_batch)
    z1, a1, logits = forward_train(x_batch, weights)
    probs = softmax(logits)

    if sample_weights is None:
        sample_weights = np.ones(n)
    weight_sum = sample_weights.sum()

    sample_losses = -np.log(np.clip(probs[np.arange(n), y_batch], 1e-12, None))
    loss = float(np.sum(sample_losses * sample_weights) / weight_sum)

    d_logits = probs.copy()
    d_logits[np.arange(n), y_batch] -= 1.0
    d_logits *= sample_weights[:, None] / weight_sum

    d_w2 = d_logits.T @ a1
    d_b2 = d_logits.sum(axis=0)

    d_a1 = d_logits @ weights.w2
    d_z1 = d_a1 * (z1 > 0)

    d_w1 = d_z1.T @ x_batch
    d_b1 = d_z1.sum(axis=0)

    grads = MLPWeights(w1=d_w1, b1=d_b1, w2=d_w2, b2=d_b2)
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == y_batch))
    return loss, accuracy, grads


def apply_gradient_step(weights, grads, learning_rate):
    return MLPWeights(
        w1=weights.w1 - learning_rate * grads.w1,
        b1=weights.b1 - learning_rate * grads.b1,
        w2=weights.w2 - learning_rate * grads.w2,
        b2=weights.b2 - learning_rate * grads.b2,
    )
