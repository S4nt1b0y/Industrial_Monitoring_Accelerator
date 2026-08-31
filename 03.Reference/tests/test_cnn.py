import numpy as np
import pytest

from cnn.reference import (
    FLATTENED_SIZE,
    LOWFREQ_BLOCK_SAMPLES,
    N_CHANNELS,
    N_FILTERS_CONV1,
    SPECTROGRAM_SIZE,
    CNNWeights,
    build_lowfreq_spectrogram,
    build_lowfreq_spectrogram_multichannel,
    build_spectrogram,
    build_spectrogram_multichannel,
    compute_loss_and_grads,
    compute_loss_and_grads_batched,
    conv2d,
    conv2d_fast,
    forward,
    init_weights,
    max_pool2d,
    max_pool2d_fast,
    predict_batched,
    relu,
)


def test_conv2d_output_shape_matches_the_standard_formula():
    # (32 + 2*1 - 3)/1 + 1 = 32, 8 filters, N_CHANNELS input channels
    image = np.zeros((N_CHANNELS, 32, 32))
    kernels = np.zeros((8, N_CHANNELS, 3, 3))
    bias = np.zeros(8)
    out = conv2d(image, kernels, bias, stride=1, padding=1)
    assert out.shape == (8, 32, 32)


def test_conv2d_identity_kernel_passes_the_image_through():
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])  # (C=1, H=3, W=3)
    identity_kernel = np.array([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=float)  # (F=1, C=1, 3, 3)
    out = conv2d(image, identity_kernel, bias=np.zeros(1), stride=1, padding=1)
    np.testing.assert_allclose(out[0], image[0])


def test_conv2d_sum_kernel_matches_hand_computed_center_value():
    # all-ones 3x3 kernel on a 3x3 image, padding=1: center output = sum of
    # the whole image (all 9 elements land inside the kernel window).
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])  # (C=1, H=3, W=3)
    sum_kernel = np.ones((1, 1, 3, 3))
    out = conv2d(image, sum_kernel, bias=np.zeros(1), stride=1, padding=1)
    assert out[0, 1, 1] == pytest.approx(image.sum())


def test_conv2d_sums_across_every_input_channel():
    # Each output filter must sum its cross-correlation over ALL input
    # channels, not just the first -- a kernel that only reacts
    # to channel 1 (all-ones there, zero elsewhere) should ignore channel 0
    # entirely, no matter what's in it.
    channel_0 = np.full((3, 3), 100.0)  # would dominate the output if wrongly included
    channel_1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    image = np.stack([channel_0, channel_1])  # (C=2, H=3, W=3)

    kernel = np.zeros((1, 2, 3, 3))
    kernel[0, 1] = 1.0  # sum-kernel on channel 1 only, channel 0's kernel is all zero
    out = conv2d(image, kernel, bias=np.zeros(1), stride=1, padding=1)
    assert out[0, 1, 1] == pytest.approx(channel_1.sum())


def test_relu_zeroes_negatives_only():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    np.testing.assert_allclose(relu(x), [0, 0, 0, 0.5, 2.0])


def test_max_pool2d_picks_the_max_of_each_2x2_block():
    feature_map = np.array([[1.0, 3.0, 2.0, 0.0], [4.0, 2.0, 1.0, 5.0],
                             [0.0, 1.0, 9.0, 2.0], [3.0, 0.0, 4.0, 1.0]])
    out = max_pool2d(feature_map[np.newaxis, :, :])
    expected = np.array([[4.0, 5.0], [3.0, 9.0]])
    np.testing.assert_allclose(out[0], expected)


def test_build_spectrogram_shape():
    rng = np.random.default_rng(0)
    vib_block = rng.uniform(-1, 1, 2048)
    spectrogram = build_spectrogram(vib_block)
    assert spectrogram.shape == (SPECTROGRAM_SIZE, SPECTROGRAM_SIZE)


def test_build_spectrogram_requires_enough_samples():
    with pytest.raises(ValueError):
        build_spectrogram(np.zeros(64))  # only 1 native sub-window, need 32


def test_build_spectrogram_multichannel_stacks_per_channel_spectrograms():
    rng = np.random.default_rng(0)
    blocks = [rng.uniform(-1, 1, 2048) for _ in range(N_CHANNELS)]
    stacked = build_spectrogram_multichannel(blocks)
    assert stacked.shape == (N_CHANNELS, SPECTROGRAM_SIZE, SPECTROGRAM_SIZE)
    for c, block in enumerate(blocks):
        np.testing.assert_allclose(stacked[c], build_spectrogram(block))


def test_build_lowfreq_spectrogram_shape_and_rejects_wrong_length():
    rng = np.random.default_rng(0)
    block = rng.uniform(-1, 1, LOWFREQ_BLOCK_SAMPLES)
    spectrogram = build_lowfreq_spectrogram(block)
    assert spectrogram.shape == (SPECTROGRAM_SIZE, SPECTROGRAM_SIZE)
    with pytest.raises(ValueError):
        build_lowfreq_spectrogram(block[:-1])


def test_build_lowfreq_spectrogram_multichannel_stacks_per_channel_spectrograms():
    rng = np.random.default_rng(0)
    blocks = [rng.uniform(-1, 1, LOWFREQ_BLOCK_SAMPLES) for _ in range(N_CHANNELS)]
    stacked = build_lowfreq_spectrogram_multichannel(blocks)
    assert stacked.shape == (N_CHANNELS, SPECTROGRAM_SIZE, SPECTROGRAM_SIZE)
    for c, block in enumerate(blocks):
        np.testing.assert_allclose(stacked[c], build_lowfreq_spectrogram(block))


def test_build_lowfreq_spectrogram_detects_a_tone_at_the_expected_bin():
    # bin 8 -> 8*400/64 = 50Hz at the decimated rate (fs_dec=400Hz, N=64).
    fs_hz = 25_600.0
    t = np.arange(LOWFREQ_BLOCK_SAMPLES) / fs_hz
    quiet = np.random.default_rng(1).normal(0, 0.05, LOWFREQ_BLOCK_SAMPLES)
    loud = 5.0 * np.sin(2 * np.pi * 50.0 * t)

    quiet_spec = build_lowfreq_spectrogram(quiet)
    loud_spec = build_lowfreq_spectrogram(loud)
    bin_index = 8 - 1  # bin 1 is row 0
    assert loud_spec[bin_index].mean() > quiet_spec[bin_index].mean() * 10


def test_init_weights_shapes():
    weights = init_weights()
    assert weights.conv1_kernels.shape == (8, N_CHANNELS, 3, 3)
    assert weights.conv1_bias.shape == (8,)
    assert weights.dense_w.shape == (4, FLATTENED_SIZE)
    assert weights.dense_b.shape == (4,)


def test_forward_returns_a_valid_class_index():
    rng = np.random.default_rng(1)
    spectrogram = rng.uniform(0, 100, (N_CHANNELS, 32, 32))
    weights = init_weights()
    logits, predicted_class = forward(spectrogram, weights)
    assert logits.shape == (4,)
    assert 0 <= predicted_class < 4
    assert predicted_class == int(np.argmax(logits))


def test_forward_is_deterministic_for_fixed_weights_and_input():
    spectrogram = np.ones((N_CHANNELS, 32, 32)) * 5.0
    weights = init_weights(seed=42)
    logits_1, class_1 = forward(spectrogram, weights)
    logits_2, class_2 = forward(spectrogram, weights)
    np.testing.assert_array_equal(logits_1, logits_2)
    assert class_1 == class_2


def test_conv2d_fast_matches_the_slow_loop_version():
    rng = np.random.default_rng(3)
    image = rng.uniform(-1, 1, (N_CHANNELS, 32, 32))
    kernels = rng.uniform(-1, 1, (N_FILTERS_CONV1, N_CHANNELS, 3, 3))
    bias = rng.uniform(-1, 1, N_FILTERS_CONV1)
    slow = conv2d(image, kernels, bias)
    fast = conv2d_fast(image, kernels, bias)
    np.testing.assert_allclose(slow, fast, atol=1e-10)


def test_max_pool2d_fast_matches_the_slow_loop_version():
    rng = np.random.default_rng(4)
    feature_maps = rng.uniform(-1, 1, (N_FILTERS_CONV1, 32, 32))
    slow = max_pool2d(feature_maps)
    fast = max_pool2d_fast(feature_maps)
    np.testing.assert_allclose(slow, fast)


def _numerical_gradient_cnn(loss_fn, param, indices, epsilon=1e-5):
    # A larger epsilon (e.g. 1e-4) can land a sampled gradient component
    # right next to a ReLU/max-pool kink, where the central difference
    # needs a smaller step to stay inside the same linear region.
    grad = np.zeros(len(indices))
    for k, idx in enumerate(indices):
        original = param[idx]
        param[idx] = original + epsilon
        loss_plus = loss_fn()
        param[idx] = original - epsilon
        loss_minus = loss_fn()
        param[idx] = original
        grad[k] = (loss_plus - loss_minus) / (2 * epsilon)
    return grad


@pytest.mark.parametrize("weight_name,shape", [("conv1_kernels", (8, N_CHANNELS, 3, 3)), ("conv1_bias", (8,))])
def test_conv_backprop_matches_numerical_gradient(weight_name, shape):
    rng = np.random.default_rng(5)
    spectrograms = [rng.uniform(0, 10, (N_CHANNELS, 32, 32)) for _ in range(3)]
    labels = np.array([0, 1, 2])
    weights = init_weights(seed=6)

    _, _, analytical_grads = compute_loss_and_grads(spectrograms, labels, weights)
    analytical = getattr(analytical_grads, weight_name)

    def loss_only():
        loss, _, _ = compute_loss_and_grads(spectrograms, labels, weights)
        return loss

    param = getattr(weights, weight_name)
    # spot-check a handful of entries, not the whole tensor -- each numerical
    # gradient point costs 2 full forward passes over all samples.
    flat_size = int(np.prod(shape))
    sample_indices = [np.unravel_index(i, shape) for i in range(0, flat_size, max(1, flat_size // 6))]
    numerical = _numerical_gradient_cnn(loss_only, param, sample_indices)
    analytical_at_samples = np.array([analytical[idx] for idx in sample_indices])
    np.testing.assert_allclose(analytical_at_samples, numerical, atol=1e-3, rtol=1e-2)


def test_dense_backprop_matches_numerical_gradient():
    rng = np.random.default_rng(8)
    spectrograms = [rng.uniform(0, 10, (N_CHANNELS, 32, 32)) for _ in range(3)]
    labels = np.array([0, 1, 2])
    weights = init_weights(seed=9)

    _, _, analytical_grads = compute_loss_and_grads(spectrograms, labels, weights)

    def loss_only():
        loss, _, _ = compute_loss_and_grads(spectrograms, labels, weights)
        return loss

    sample_indices = [(0, 0), (1, 500), (2, 1000), (3, 2047)]
    numerical = _numerical_gradient_cnn(loss_only, weights.dense_w, sample_indices)
    analytical_at_samples = np.array([analytical_grads.dense_w[idx] for idx in sample_indices])
    np.testing.assert_allclose(analytical_at_samples, numerical, atol=1e-3, rtol=1e-2)


def test_gradient_step_reduces_cnn_loss():
    rng = np.random.default_rng(10)
    spectrograms = [rng.uniform(0, 10, (N_CHANNELS, 32, 32)) for _ in range(6)]
    labels = np.array([0, 1, 2, 3, 0, 1])
    weights = init_weights(seed=11)

    from cnn.reference import apply_gradient_step

    loss_before, _, grads = compute_loss_and_grads(spectrograms, labels, weights)
    weights = apply_gradient_step(weights, grads, learning_rate=1e-4)
    loss_after, _, _ = compute_loss_and_grads(spectrograms, labels, weights)
    assert loss_after < loss_before


def test_compute_loss_and_grads_batched_matches_the_per_sample_version():
    # compute_loss_and_grads_batched exists purely for training speed
    # (per-sample loop -> im2col + one matmul/einsum per epoch). It
    # must match the already-numerically-gradient-checked per-sample
    # version to machine precision, not just "close enough" -- same
    # cross-validation pattern as conv2d_fast vs conv2d.
    rng = np.random.default_rng(20)
    n = 7
    spectrograms_list = [rng.uniform(0, 10, (N_CHANNELS, SPECTROGRAM_SIZE, SPECTROGRAM_SIZE)) for _ in range(n)]
    spectrograms_batched = np.stack(spectrograms_list)
    labels = rng.integers(0, 4, n)
    sample_weights = rng.uniform(0.5, 2.0, n)
    weights = init_weights(seed=21)

    loss_p, acc_p, grads_p = compute_loss_and_grads(
        spectrograms_list, labels, weights, sample_weights=sample_weights
    )
    loss_b, acc_b, grads_b = compute_loss_and_grads_batched(
        spectrograms_batched, labels, weights, sample_weights=sample_weights
    )

    assert loss_b == pytest.approx(loss_p, abs=1e-10)
    assert acc_b == pytest.approx(acc_p, abs=1e-10)
    np.testing.assert_allclose(grads_b.conv1_kernels, grads_p.conv1_kernels, atol=1e-10)
    np.testing.assert_allclose(grads_b.conv1_bias, grads_p.conv1_bias, atol=1e-10)
    np.testing.assert_allclose(grads_b.dense_w, grads_p.dense_w, atol=1e-10)
    np.testing.assert_allclose(grads_b.dense_b, grads_p.dense_b, atol=1e-10)


def test_predict_batched_matches_forward_per_sample():
    # predict_batched exists purely so evaluating a trained model over a
    # whole test set doesn't pay for forward()'s per-sample
    # Python loop (that loop is fine for a handful of samples, not
    # thousands). Must agree with forward() exactly, same argmax rule.
    rng = np.random.default_rng(24)
    spectrograms = rng.uniform(0, 10, (9, N_CHANNELS, SPECTROGRAM_SIZE, SPECTROGRAM_SIZE))
    weights = init_weights(seed=25)

    predicted_batched = predict_batched(spectrograms, weights)
    predicted_per_sample = np.array([forward(s, weights)[1] for s in spectrograms])
    np.testing.assert_array_equal(predicted_batched, predicted_per_sample)


def test_batched_gradient_step_reduces_cnn_loss():
    rng = np.random.default_rng(22)
    spectrograms = rng.uniform(0, 10, (10, N_CHANNELS, SPECTROGRAM_SIZE, SPECTROGRAM_SIZE))
    labels = rng.integers(0, 4, 10)
    weights = init_weights(seed=23)

    from cnn.reference import apply_gradient_step

    loss_before, _, grads = compute_loss_and_grads_batched(spectrograms, labels, weights)
    weights = apply_gradient_step(weights, grads, learning_rate=1e-4)
    loss_after, _, _ = compute_loss_and_grads_batched(spectrograms, labels, weights)
    assert loss_after < loss_before


def test_forward_reacts_to_a_strong_localized_feature():
    # A spectrogram with one very hot corner should push the conv/pool
    # path to produce a clearly different logit vector than a flat input
    # -- sanity check that the architecture actually responds to spatial
    # structure, not just the mean.
    weights = init_weights(seed=7)
    flat = np.ones((N_CHANNELS, 32, 32)) * 1.0
    hot_corner = flat.copy()
    hot_corner[:, :4, :4] = 100.0

    logits_flat, _ = forward(flat, weights)
    logits_hot, _ = forward(hot_corner, weights)
    assert not np.allclose(logits_flat, logits_hot)
