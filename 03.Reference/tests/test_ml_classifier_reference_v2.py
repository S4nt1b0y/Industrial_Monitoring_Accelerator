import numpy as np
import pytest

from ml_classifier.reference_v2 import (
    N_CLASSES,
    N_FEATURES,
    apply_gradient_step,
    compute_loss_and_grads,
    forward,
    init_weights,
)


def test_n_features_matches_v1_plus_ar_coefficients():
    from features.pipeline import FEATURE_NAMES as V1_FEATURE_NAMES

    # v1's FEATURE_NAMES includes f0_valido (excluded from the classifier
    # input, matching the pattern already established for v1 -- see
    # ml_classifier/reference.py's own N_FEATURES=129 vs. len(v1 FEATURE_NAMES)=130).
    assert N_FEATURES == len(V1_FEATURE_NAMES) - 1 + 4


def test_forward_returns_valid_class_index():
    weights = init_weights(n_hidden=8)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, N_FEATURES)
    logits, predicted = forward(x, weights)
    assert logits.shape == (N_CLASSES,)
    assert 0 <= predicted < N_CLASSES
    assert predicted == int(np.argmax(logits))


def _numerical_gradient(loss_fn, param, epsilon=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]
        param[idx] = original + epsilon
        loss_plus = loss_fn()
        param[idx] = original - epsilon
        loss_minus = loss_fn()
        param[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        it.iternext()
    return grad


@pytest.mark.parametrize("weight_name", ["w1", "b1", "w2", "b2"])
def test_backprop_matches_numerical_gradient(weight_name):
    rng = np.random.default_rng(0)
    n_samples, n_hidden = 5, 4
    x_batch = rng.uniform(-1, 1, (n_samples, N_FEATURES))
    y_batch = rng.integers(0, N_CLASSES, n_samples)
    weights = init_weights(n_hidden=n_hidden, seed=1)

    _, _, analytical_grads = compute_loss_and_grads(x_batch, y_batch, weights)
    analytical = getattr(analytical_grads, weight_name)

    def loss_only():
        loss, _, _ = compute_loss_and_grads(x_batch, y_batch, weights)
        return loss

    numerical = _numerical_gradient(loss_only, getattr(weights, weight_name))
    np.testing.assert_allclose(analytical, numerical, atol=1e-4, rtol=1e-3)


def test_gradient_step_reduces_loss_on_a_batch():
    rng = np.random.default_rng(4)
    x_batch = rng.uniform(-1, 1, (20, N_FEATURES))
    y_batch = rng.integers(0, N_CLASSES, 20)
    weights = init_weights(n_hidden=8, seed=5)

    loss_before, _, grads = compute_loss_and_grads(x_batch, y_batch, weights)
    weights = apply_gradient_step(weights, grads, learning_rate=0.5)
    loss_after, _, _ = compute_loss_and_grads(x_batch, y_batch, weights)
    assert loss_after < loss_before
