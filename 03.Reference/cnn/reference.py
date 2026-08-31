"""CNN forward pass: 32x32xC spectrogram -> conv(8, 3x3, stride 1, pad 1)
+ ReLU -> max-pool 2x2 -> dense -> 4 classes.

Every array carries an explicit channel axis (image/kernels always
(C,H,W)/(F,C,kh,kw)); C=1 is just the generic case, not a special-cased
API, so the input can be 1 or several stacked vibration channels.

Floating-point/"ideal" only -- fixed-point quantization is a
hardware-side concern, not modeled here.

conv2d is a literal, unoptimized nested-loop implementation (not
vectorized) -- deliberately, so its operation count is easy to read off
directly instead of hiding behind a single vectorized call. Not meant
to run over a full dataset (that's what conv2d_fast, below, is for).
"""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import correlate2d

from scipy.signal import decimate

from dataset.signal_params import N_FFT
from fft.reference import fft_dif_radix2, magnitude_spectrum, unscramble

SPECTROGRAM_SIZE = 32
N_FILTERS_CONV1 = 8
KERNEL_SIZE = 3
POOL_SIZE = 2
N_CLASSES = 4
N_CHANNELS = 4  # x/y_mancal_a, x/y_mancal_b
FLATTENED_SIZE = (SPECTROGRAM_SIZE // POOL_SIZE) ** 2 * N_FILTERS_CONV1  # 16*16*8 = 2048

# Low-frequency (decimated) spectrogram -- an alternative to
# build_spectrogram's native-resolution one below. Native reads bins 1-32
# of the raw 64-point FFT (Delta_f=400Hz, 400-12800Hz), which buries the
# 1x/2x/3x rotational harmonics that separate imbalance from
# misalignment inside the discarded DC bin. Each of the 32 time-columns
# here is one VIBRATION_DECIM_FACTOR=64-decimated window (same
# decimation as features.pipeline's lowfreq_spectrum), stepped by
# LOWFREQ_HOP_DECIMATED=16 decimated samples between columns. Needs a
# much longer raw span per spectrogram (~1.4s vs. ~80ms native) -- a
# real cost, but the CNN already accumulates a full spectrogram before
# classifying, not per 10ms window, so a longer span is architecturally
# compatible, just slower to produce a new classification.
LOWFREQ_DECIM_FACTOR = 64
LOWFREQ_HOP_DECIMATED = 16
LOWFREQ_SPAN_SAMPLES = N_FFT * LOWFREQ_DECIM_FACTOR  # 4096 raw samples/decimated window
LOWFREQ_HOP_RAW_SAMPLES = LOWFREQ_HOP_DECIMATED * LOWFREQ_DECIM_FACTOR  # 1024
LOWFREQ_BLOCK_SAMPLES = (SPECTROGRAM_SIZE - 1) * LOWFREQ_HOP_RAW_SAMPLES + LOWFREQ_SPAN_SAMPLES  # 35,840


def build_spectrogram(vib_block, n_fft=N_FFT):
    """32x32 spectrogram from a 2048-sample vibration block (one channel):
    32 native FFT sub-windows (time, columns) x bins 1..32 (frequency,
    excludes DC bin 0).
    """
    n_sub = len(vib_block) // n_fft
    if n_sub < SPECTROGRAM_SIZE:
        raise ValueError(f"need >= {SPECTROGRAM_SIZE} native sub-windows, got {n_sub}")

    columns = []
    for i in range(SPECTROGRAM_SIZE):
        sub = vib_block[i * n_fft : (i + 1) * n_fft]
        spectrum = magnitude_spectrum(unscramble(fft_dif_radix2(sub)))
        columns.append(spectrum[1 : SPECTROGRAM_SIZE + 1])
    return np.stack(columns, axis=1)  # (freq=32, time=32)


def build_spectrogram_multichannel(vib_blocks, n_fft=N_FFT):
    """vib_blocks: sequence of C 2048-sample blocks, one per vibration
    channel, same time span. Returns (C, 32, 32) -- build_spectrogram
    applied per channel and stacked as input depth."""
    return np.stack([build_spectrogram(block, n_fft=n_fft) for block in vib_blocks], axis=0)


def build_lowfreq_spectrogram(vib_block, n_fft=N_FFT):
    """32x32 spectrogram from a LOWFREQ_BLOCK_SAMPLES-sample vibration
    block (one channel, already ingestion-normalized): 32 overlapping
    VIBRATION_DECIM_FACTOR=64-decimated windows (time, columns) x bins
    1..32 (frequency, Delta_f_dec=6.25Hz, excludes DC bin 0) -- the
    low-frequency alternative to build_spectrogram above."""
    if len(vib_block) != LOWFREQ_BLOCK_SAMPLES:
        raise ValueError(f"expected {LOWFREQ_BLOCK_SAMPLES} samples, got {len(vib_block)}")

    columns = []
    for i in range(SPECTROGRAM_SIZE):
        start = i * LOWFREQ_HOP_RAW_SAMPLES
        raw_window = vib_block[start : start + LOWFREQ_SPAN_SAMPLES]
        dec = decimate(raw_window, LOWFREQ_DECIM_FACTOR, ftype="fir")[:n_fft]
        spectrum = magnitude_spectrum(unscramble(fft_dif_radix2(dec)))
        columns.append(spectrum[1 : SPECTROGRAM_SIZE + 1])
    return np.stack(columns, axis=1)  # (freq=32, time=32)


def build_lowfreq_spectrogram_multichannel(vib_blocks, n_fft=N_FFT):
    """vib_blocks: sequence of C LOWFREQ_BLOCK_SAMPLES-sample blocks, one
    per vibration channel, same time span. Returns (C, 32, 32) --
    build_lowfreq_spectrogram applied per channel and stacked as input
    depth."""
    return np.stack([build_lowfreq_spectrogram(block, n_fft=n_fft) for block in vib_blocks], axis=0)


def conv2d(image, kernels, bias, stride=1, padding=1):
    """image: (C, H, W). kernels: (F, C, kh, kw). bias: (F,). Returns
    (F, H_out, W_out) -- each output filter sums its cross-correlation
    over every input channel (C=1 reduces to the single-channel case).

    Cross-correlation (no kernel flip), the ML-framework convention, not
    the textbook-signal-processing convolution.
    """
    n_channels, h, w = image.shape
    n_filters, kernel_channels, kh, kw = kernels.shape
    if kernel_channels != n_channels:
        raise ValueError(f"kernels expect {kernel_channels} channels, image has {n_channels}")
    padded = np.pad(image, ((0, 0), (padding, padding), (padding, padding)))
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1

    out = np.zeros((n_filters, h_out, w_out))
    for f in range(n_filters):
        for i in range(h_out):
            for j in range(w_out):
                region = padded[:, i * stride : i * stride + kh, j * stride : j * stride + kw]
                out[f, i, j] = np.sum(region * kernels[f]) + bias[f]
    return out


def relu(x):
    return np.maximum(x, 0.0)


def max_pool2d(feature_maps, size=POOL_SIZE, stride=POOL_SIZE):
    """feature_maps: (F, H, W) -> (F, H//stride, W//stride)."""
    n_filters, h, w = feature_maps.shape
    h_out, w_out = h // stride, w // stride
    out = np.zeros((n_filters, h_out, w_out))
    for f in range(n_filters):
        for i in range(h_out):
            for j in range(w_out):
                region = feature_maps[f, i * stride : i * stride + size, j * stride : j * stride + size]
                out[f, i, j] = region.max()
    return out


@dataclass
class CNNWeights:
    conv1_kernels: np.ndarray  # (8, C, 3, 3)
    conv1_bias: np.ndarray  # (8,)
    dense_w: np.ndarray  # (4, 2048)
    dense_b: np.ndarray  # (4,)


def init_weights(n_channels=N_CHANNELS, seed=0):
    """Small random weights -- untrained (real weights come from training
    a fresh model, e.g. tools/train_cnn_kfold.py).
    """
    rng = np.random.default_rng(seed)
    return CNNWeights(
        conv1_kernels=rng.normal(0, 0.1, (N_FILTERS_CONV1, n_channels, KERNEL_SIZE, KERNEL_SIZE)),
        conv1_bias=np.zeros(N_FILTERS_CONV1),
        dense_w=rng.normal(0, 0.01, (N_CLASSES, FLATTENED_SIZE)),
        dense_b=np.zeros(N_CLASSES),
    )


def forward(spectrogram, weights):
    """spectrogram: (C, 32, 32). Returns (logits, predicted_class). No
    softmax -- argmax only, same reasoning as ml_classifier: only the
    winning class is needed, not a probability.
    """
    conv_out = relu(conv2d(spectrogram, weights.conv1_kernels, weights.conv1_bias))
    pooled = max_pool2d(conv_out)
    flat = pooled.reshape(-1)
    logits = weights.dense_w @ flat + weights.dense_b
    return logits, int(np.argmax(logits))


# --- Training: forward()/conv2d()/max_pool2d() above are the unoptimized,
# validated reference; everything below is a fast, vectorized path used
# only for training speed, cross-checked against the slow path in
# tests/test_cnn.py before being trusted.


def relu_backward(d_out, pre_activation):
    return d_out * (pre_activation > 0)


def conv2d_fast(image, kernels, bias):
    """Same math as conv2d() above (validated there against hand-computed
    cases) via scipy's vectorized correlate2d -- fixed padding=1/stride=1/
    3x3, this architecture's only configuration, not a general conv2d
    replacement. image: (C,H,W), kernels: (F,C,kh,kw).
    """
    n_filters = len(kernels)
    n_channels, h, w = image.shape
    out = np.empty((n_filters, h, w))
    for f in range(n_filters):
        acc = np.zeros((h, w))
        for c in range(n_channels):
            acc += correlate2d(image[c], kernels[f, c], mode="same", boundary="fill", fillvalue=0)
        out[f] = acc + bias[f]
    return out


def conv2d_backward(image, d_out, n_filters):
    """Gradients w.r.t. kernels (F,C,kh,kw) and bias (F,) only -- the
    spectrogram input isn't a learnable parameter, so d_image is never
    needed. Standard CNN backward identity: d_kernel[f,c] = correlate(
    padded_input[c], d_out[f], 'valid') -- same per-channel accumulation
    as the forward pass sums over channels."""
    n_channels = image.shape[0]
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1)))
    d_kernels = np.empty((n_filters, n_channels, KERNEL_SIZE, KERNEL_SIZE))
    d_bias = np.empty(n_filters)
    for f in range(n_filters):
        for c in range(n_channels):
            d_kernels[f, c] = correlate2d(padded[c], d_out[f], mode="valid")
        d_bias[f] = d_out[f].sum()
    return d_kernels, d_bias


def max_pool2d_fast(feature_maps, size=POOL_SIZE, stride=POOL_SIZE):
    n_filters, h, w = feature_maps.shape
    reshaped = feature_maps.reshape(n_filters, h // stride, stride, w // stride, stride)
    return reshaped.max(axis=(2, 4))


def max_pool2d_backward(feature_maps, pooled, d_pooled, size=POOL_SIZE, stride=POOL_SIZE):
    """Routes gradient to whichever input(s) won the max in each block
    (ties split evenly -- rare with float inputs, but keeps the gradient
    check exact rather than picking an arbitrary winner).
    """
    n_filters, h, w = feature_maps.shape
    reshaped = feature_maps.reshape(n_filters, h // stride, stride, w // stride, stride)
    pooled_broadcast = pooled.reshape(n_filters, h // stride, 1, w // stride, 1)
    mask = (reshaped == pooled_broadcast).astype(np.float64)
    mask /= mask.sum(axis=(2, 4), keepdims=True)
    d_reshaped = mask * d_pooled.reshape(n_filters, h // stride, 1, w // stride, 1)
    return d_reshaped.reshape(n_filters, h, w)


def forward_train(spectrogram, weights):
    """Keeps every intermediate compute_loss_and_grads needs for backprop."""
    conv_pre = conv2d_fast(spectrogram, weights.conv1_kernels, weights.conv1_bias)
    conv_post = relu(conv_pre)
    pooled = max_pool2d_fast(conv_post)
    flat = pooled.reshape(-1)
    logits = weights.dense_w @ flat + weights.dense_b
    return conv_pre, conv_post, pooled, flat, logits


def softmax_single(logits):
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def compute_loss_and_grads(spectrograms, labels, weights, sample_weights=None):
    """spectrograms: sequence of (C,32,32) images. labels: (N,) int.

    Per-sample forward+backward, accumulated -- conv2d has no clean
    batched matrix form without a heavier im2col rewrite, and training set
    sizes here are a few thousand images, not millions. Interface matches
    ml_classifier.reference.compute_loss_and_grads (loss, accuracy, grads)
    so the two training loops read the same way.
    """
    n = len(labels)
    if sample_weights is None:
        sample_weights = np.ones(n)
    weight_sum = sample_weights.sum()

    d_conv1_kernels = np.zeros_like(weights.conv1_kernels)
    d_conv1_bias = np.zeros_like(weights.conv1_bias)
    d_dense_w = np.zeros_like(weights.dense_w)
    d_dense_b = np.zeros_like(weights.dense_b)

    total_loss = 0.0
    correct = 0

    for spectrogram, label, sw in zip(spectrograms, labels, sample_weights):
        conv_pre, conv_post, pooled, flat, logits = forward_train(spectrogram, weights)
        probs = softmax_single(logits)
        total_loss += -np.log(max(probs[label], 1e-12)) * sw
        if int(np.argmax(logits)) == label:
            correct += 1

        d_logits = probs.copy()
        d_logits[label] -= 1.0
        d_logits *= sw / weight_sum

        d_dense_w += np.outer(d_logits, flat)
        d_dense_b += d_logits

        d_flat = weights.dense_w.T @ d_logits
        d_pooled = d_flat.reshape(pooled.shape)

        d_conv_post = max_pool2d_backward(conv_post, pooled, d_pooled)
        d_conv_pre = relu_backward(d_conv_post, conv_pre)

        dk, db = conv2d_backward(spectrogram, d_conv_pre, N_FILTERS_CONV1)
        d_conv1_kernels += dk
        d_conv1_bias += db

    grads = CNNWeights(
        conv1_kernels=d_conv1_kernels,
        conv1_bias=d_conv1_bias,
        dense_w=d_dense_w,
        dense_b=d_dense_b,
    )
    return total_loss / weight_sum, correct / n, grads


def apply_gradient_step(weights, grads, learning_rate):
    return CNNWeights(
        conv1_kernels=weights.conv1_kernels - learning_rate * grads.conv1_kernels,
        conv1_bias=weights.conv1_bias - learning_rate * grads.conv1_bias,
        dense_w=weights.dense_w - learning_rate * grads.dense_w,
        dense_b=weights.dense_b - learning_rate * grads.dense_b,
    )


# --- Batched training: compute_loss_and_grads() above loops one sample
# at a time, which is too slow for a full training run over thousands of
# spectrograms and many epochs. Everything below does the SAME math over
# a whole batch at once via im2col (sliding_window_view + one
# matmul/einsum instead of nested loops) -- cross-validated against
# compute_loss_and_grads() to machine precision before being trusted,
# same "prove the fast path against the slow path" pattern as
# conv2d_fast/conv2d_backward above.


def conv2d_batched(images, kernels, bias, padding=1):
    """images: (N,C,H,W). kernels: (F,C,kh,kw). bias: (F,). Returns
    ((N,F,H,W), windows_flat) -- windows_flat is the im2col patch matrix,
    kept around because conv2d_batched_backward reuses it (forward and
    backward need the same input patches; recomputing them would be
    wasted work). Stride is fixed at 1, this architecture's only value.
    """
    n, c, h, w = images.shape
    f, kernel_channels, kh, kw = kernels.shape
    if kernel_channels != c:
        raise ValueError(f"kernels expect {kernel_channels} channels, images have {c}")
    padded = np.pad(images, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = sliding_window_view(padded, (kh, kw), axis=(2, 3))  # (N,C,H_out,W_out,kh,kw)
    h_out, w_out = windows.shape[2], windows.shape[3]
    # (N,C,H_out,W_out,kh,kw) -> (N, H_out*W_out, C*kh*kw): one row per
    # output position, one column per (channel, kernel-offset) -- the
    # standard im2col layout, matmul-compatible with a flattened kernel.
    windows_flat = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n, h_out * w_out, c * kh * kw)
    kernels_flat = kernels.reshape(f, c * kh * kw)
    out = windows_flat @ kernels_flat.T  # (N, H_out*W_out, F)
    out = out.transpose(0, 2, 1).reshape(n, f, h_out, w_out) + bias[None, :, None, None]
    return out, windows_flat


def conv2d_batched_backward(windows_flat, d_out, n_filters, n_channels):
    """d_kernel[f,c,di,dj] = sum over batch and output position of
    padded_input[c,i+di,j+dj] * d_out[f,i,j] -- the batched form of
    conv2d_backward's identity, expressed as one contraction over the
    same im2col patches the forward pass already built. optimize=True
    matters here: einsum's naive path was ~2.5x slower on this contraction
    shape (measured), not just a micro-optimization.
    """
    n, f, h_out, w_out = d_out.shape
    d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(n, h_out * w_out, f)
    d_kernels_flat = np.einsum("npk,npf->fk", windows_flat, d_out_flat, optimize=True)
    d_kernels = d_kernels_flat.reshape(n_filters, n_channels, KERNEL_SIZE, KERNEL_SIZE)
    d_bias = d_out.sum(axis=(0, 2, 3))
    return d_kernels, d_bias


def max_pool2d_batched(feature_maps, size=POOL_SIZE, stride=POOL_SIZE):
    """feature_maps: (N,F,H,W) -> (N,F,H//stride,W//stride)."""
    n, f, h, w = feature_maps.shape
    reshaped = feature_maps.reshape(n, f, h // stride, stride, w // stride, stride)
    return reshaped.max(axis=(3, 5))


def max_pool2d_backward_batched(feature_maps, pooled, d_pooled, size=POOL_SIZE, stride=POOL_SIZE):
    """Batched form of max_pool2d_backward -- same tie-splitting rule."""
    n, f, h, w = feature_maps.shape
    reshaped = feature_maps.reshape(n, f, h // stride, stride, w // stride, stride)
    pooled_broadcast = pooled.reshape(n, f, h // stride, 1, w // stride, 1)
    mask = (reshaped == pooled_broadcast).astype(np.float64)
    mask /= mask.sum(axis=(3, 5), keepdims=True)
    d_reshaped = mask * d_pooled.reshape(n, f, h // stride, 1, w // stride, 1)
    return d_reshaped.reshape(n, f, h, w)


def softmax_batched(logits):
    """logits: (N, N_CLASSES) -> (N, N_CLASSES), softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def predict_batched(spectrograms, weights):
    """spectrograms: (N,C,32,32). Returns (N,) predicted class indices --
    the batched-forward-only counterpart to calling forward() in a Python
    loop over samples. forward() itself stays on the naive conv2d() on
    purpose (it's the hardware-cycle-matching reference); this is for
    evaluating a trained model over a whole test set without paying for
    that loop -- same batched math as compute_loss_and_grads_batched,
    forward pass only, no backward/gradients computed.
    """
    conv_pre, _ = conv2d_batched(spectrograms, weights.conv1_kernels, weights.conv1_bias)
    pooled = max_pool2d_batched(relu(conv_pre))
    flat = pooled.reshape(len(spectrograms), -1)
    logits = flat @ weights.dense_w.T + weights.dense_b
    return np.argmax(logits, axis=1)


def compute_loss_and_grads_batched(spectrograms, labels, weights, sample_weights=None):
    """spectrograms: (N,C,32,32) array (NOT a list -- that's the whole
    point). labels: (N,) int. Same loss/metrics/return shape as
    compute_loss_and_grads, just computed over the whole batch in a
    handful of vectorized calls instead of a Python loop over samples.
    """
    n = len(labels)
    if sample_weights is None:
        sample_weights = np.ones(n)
    weight_sum = sample_weights.sum()

    conv_pre, windows_flat = conv2d_batched(spectrograms, weights.conv1_kernels, weights.conv1_bias)
    conv_post = relu(conv_pre)
    pooled = max_pool2d_batched(conv_post)
    flat = pooled.reshape(n, -1)
    logits = flat @ weights.dense_w.T + weights.dense_b
    probs = softmax_batched(logits)

    sample_losses = -np.log(np.clip(probs[np.arange(n), labels], 1e-12, None))
    loss = float(np.sum(sample_losses * sample_weights) / weight_sum)
    accuracy = float(np.mean(np.argmax(logits, axis=1) == labels))

    d_logits = probs.copy()
    d_logits[np.arange(n), labels] -= 1.0
    d_logits *= (sample_weights / weight_sum)[:, None]

    d_dense_w = d_logits.T @ flat
    d_dense_b = d_logits.sum(axis=0)

    d_flat = d_logits @ weights.dense_w
    d_pooled = d_flat.reshape(pooled.shape)

    d_conv_post = max_pool2d_backward_batched(conv_post, pooled, d_pooled)
    d_conv_pre = relu_backward(d_conv_post, conv_pre)

    d_conv1_kernels, d_conv1_bias = conv2d_batched_backward(
        windows_flat, d_conv_pre, N_FILTERS_CONV1, spectrograms.shape[1]
    )

    grads = CNNWeights(
        conv1_kernels=d_conv1_kernels,
        conv1_bias=d_conv1_bias,
        dense_w=d_dense_w,
        dense_b=d_dense_b,
    )
    return loss, accuracy, grads
