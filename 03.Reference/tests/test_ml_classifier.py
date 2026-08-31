import numpy as np
import pytest

from ml_classifier.reference import (
    N_CLASSES,
    N_FEATURES,
    MLPWeights,
    apply_gradient_step,
    compute_loss_and_grads,
    forward,
    init_weights,
    softmax,
)


def test_forward_returns_valid_class_index():
    weights = init_weights(n_hidden=8)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, N_FEATURES)
    logits, predicted = forward(x, weights)
    assert logits.shape == (N_CLASSES,)
    assert 0 <= predicted < N_CLASSES
    assert predicted == int(np.argmax(logits))


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0, 0.0], [-1.0, 0.0, 1.0, 2.0]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])
    assert np.all(probs > 0)


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


def test_sample_weights_change_the_gradient():
    rng = np.random.default_rng(2)
    x_batch = rng.uniform(-1, 1, (4, N_FEATURES))
    y_batch = np.array([0, 1, 2, 3])
    weights = init_weights(n_hidden=4, seed=3)

    _, _, grads_uniform = compute_loss_and_grads(x_batch, y_batch, weights)
    heavy_weights = np.array([10.0, 1.0, 1.0, 1.0])
    _, _, grads_weighted = compute_loss_and_grads(x_batch, y_batch, weights, sample_weights=heavy_weights)

    assert not np.allclose(grads_uniform.w1, grads_weighted.w1)


def test_gradient_step_reduces_loss_on_a_batch():
    rng = np.random.default_rng(4)
    x_batch = rng.uniform(-1, 1, (20, N_FEATURES))
    y_batch = rng.integers(0, N_CLASSES, 20)
    weights = init_weights(n_hidden=8, seed=5)

    loss_before, _, grads = compute_loss_and_grads(x_batch, y_batch, weights)
    weights = apply_gradient_step(weights, grads, learning_rate=0.5)
    loss_after, _, _ = compute_loss_and_grads(x_batch, y_batch, weights)
    assert loss_after < loss_before


def test_training_converges_on_a_trivially_separable_toy_problem():
    # 4 well-separated clusters, one per class -- a working training loop
    # should drive train accuracy close to 1.0 quickly.
    rng = np.random.default_rng(6)
    n_per_class = 30
    centers = np.eye(N_CLASSES, N_FEATURES) * 5.0
    x_batch, y_batch = [], []
    for c in range(N_CLASSES):
        x_batch.append(centers[c] + rng.normal(0, 0.1, (n_per_class, N_FEATURES)))
        y_batch.append(np.full(n_per_class, c))
    x_batch = np.concatenate(x_batch)
    y_batch = np.concatenate(y_batch)

    weights = init_weights(n_hidden=8, seed=7)
    for _ in range(300):
        _, _, grads = compute_loss_and_grads(x_batch, y_batch, weights)
        weights = apply_gradient_step(weights, grads, learning_rate=0.5)

    _, accuracy, _ = compute_loss_and_grads(x_batch, y_batch, weights)
    assert accuracy > 0.95
