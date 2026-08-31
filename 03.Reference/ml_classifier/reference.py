"""MLP classifier: N_FEATURES inputs -> H hidden (ReLU) -> 4 classes ->
argmax. No softmax at inference (only the winning class is needed, same
reasoning as cnn.reference.forward).

Training needs gradients, which the inference-only forward() doesn't
compute -- forward_train() keeps the intermediate activations backprop
needs, and compute_loss_and_grads() derives them by hand (cross-entropy
on a softmax taken only for the loss, matching how a real training
pipeline would do it even though the deployed/hardware model never
computes softmax). This is the only module in the project with a training
path, since it's the only one meant to learn weights from data rather
than implement a fixed algorithm.
"""

from dataclasses import dataclass

import numpy as np

N_FEATURES = 129
N_CLASSES = 4
FEATURE_NAMES = ["f0_hz"] + [
    f"lowfreq_{ch}_bin_{b:02d}" for ch in ("x_A", "y_A", "x_B", "y_B") for b in range(1, 33)
]
CLASS_NAMES = ["operacao_normal", "desbalanceamento", "desalinhamento", "desgaste_rolamento"]


@dataclass
class MLPWeights:
    w1: np.ndarray  # (H, N_FEATURES)
    b1: np.ndarray  # (H,)
    w2: np.ndarray  # (N_CLASSES, H)
    b2: np.ndarray  # (N_CLASSES,)


def init_weights(n_hidden=16, seed=0):
    rng = np.random.default_rng(seed)
    # He initialization (ReLU hidden layer): std = sqrt(2/fan_in)
    w1 = rng.normal(0, np.sqrt(2.0 / N_FEATURES), (n_hidden, N_FEATURES))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (N_CLASSES, n_hidden))
    b2 = np.zeros(N_CLASSES)
    return MLPWeights(w1=w1, b1=b1, w2=w2, b2=b2)


def relu(x):
    return np.maximum(x, 0.0)


def forward(x, weights):
    """x: (N_FEATURES,) single sample. Returns (logits, predicted_class) --
    matches the planned hardware inference path: argmax only, no softmax.
    """
    h = relu(weights.w1 @ x + weights.b1)
    logits = weights.w2 @ h + weights.b2
    return logits, int(np.argmax(logits))


def forward_train(x_batch, weights):
    """x_batch: (N, N_FEATURES). Returns (z1, a1, logits) -- the
    intermediates compute_loss_and_grads needs for backprop.
    """
    z1 = x_batch @ weights.w1.T + weights.b1  # (N, H)
    a1 = relu(z1)
    logits = a1 @ weights.w2.T + weights.b2  # (N, N_CLASSES)
    return z1, a1, logits


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def compute_loss_and_grads(x_batch, y_batch, weights, sample_weights=None):
    """Weighted mean cross-entropy loss + gradients w.r.t. every weight.

    sample_weights: optional (N,) per-sample loss weight, used to
    counter class imbalance (pass class_weight[y_i] per sample here,
    not a resampling scheme).
    """
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
