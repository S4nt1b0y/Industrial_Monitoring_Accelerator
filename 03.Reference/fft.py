import numpy as np
from functools import lru_cache

SUPPORTED_DATA_WIDTHS = (8, 16)


def q_format(data_width):
    if data_width not in SUPPORTED_DATA_WIDTHS:
        raise ValueError(f"data_width must be one of {SUPPORTED_DATA_WIDTHS}, got {data_width}")
    return f"q1_{data_width - 1}"


def fixed_point_params(data_width):
    q_format(data_width)
    scale = float(2 ** (data_width - 1))
    min_value = -(2 ** (data_width - 1))
    max_value = (2 ** (data_width - 1)) - 1
    return scale, min_value, max_value


def quantize_signed(values, data_width):
    scale, min_value, max_value = fixed_point_params(data_width)
    quantized = np.rint(np.asarray(values) * scale)
    return np.clip(quantized, min_value, max_value).astype(np.int32)


def fft_magnitude_q(signal, data_width, window_size=64, *, input_is_float=False):
    """
    Computes fixed-point FFT magnitude features for Q1.7/Q1.15 samples.

    The FFT output is divided by window_size before real/imag requantization,
    then magnitude is approximated as abs(real) + abs(imag) and saturated to
    the unsigned feature range of the selected Q format.
    """
    q_format(data_width)
    signal = np.asarray(signal)
    if signal.shape != (window_size,):
        raise ValueError(f"signal must have shape ({window_size},), got {signal.shape}")
    if window_size <= 0 or window_size & (window_size - 1):
        raise ValueError(f"window_size must be a power of 2, got {window_size}")
    if not input_is_float and not np.issubdtype(signal.dtype, np.integer):
        raise ValueError(f"signal must contain integer fixed-point values, got {signal.dtype}")

    if input_is_float:
        scale, _, max_value = fixed_point_params(data_width)
        signal_float = signal.astype(np.float64, copy=False)
    else:
        scale, min_value, max_value = fixed_point_params(data_width)
        min_sample = int(np.min(signal)) if signal.size else 0
        max_sample = int(np.max(signal)) if signal.size else 0
        if min_sample < min_value or max_sample > max_value:
            raise ValueError(
                "samples out of range for "
                f"{q_format(data_width)}: expected {min_value}..{max_value}, "
                f"got {min_sample}..{max_sample}"
            )
        return fft_magnitude_q_batch(signal.reshape(1, window_size), data_width, window_size)[0]

    fft_values = np.einsum("i,oi->o", signal_float, make_fft_matrix(window_size))

    real = quantize_signed(fft_values.real, data_width)
    imag = quantize_signed(fft_values.imag, data_width)
    magnitude = np.abs(real) + np.abs(imag)
    return np.clip(magnitude, 0, max_value).astype(np.int32)


def fft_magnitude_q_batch(signals, data_width, window_size=64):
    """
    Batch version of fft_magnitude_q for integer fixed-point windows.

    signals must have shape (window_count, window_size). The returned array has
    shape (window_count, window_size).
    """
    q_format(data_width)
    signals = np.asarray(signals)
    if signals.ndim != 2 or signals.shape[1] != window_size:
        raise ValueError(
            f"signals must have shape (window_count, {window_size}), got {signals.shape}"
        )
    if not np.issubdtype(signals.dtype, np.integer):
        raise ValueError(f"signals must contain integer fixed-point values, got {signals.dtype}")

    scale, min_value, max_value = fixed_point_params(data_width)
    if signals.size:
        min_sample = int(np.min(signals))
        max_sample = int(np.max(signals))
        if min_sample < min_value or max_sample > max_value:
            raise ValueError(
                "samples out of range for "
                f"{q_format(data_width)}: expected {min_value}..{max_value}, "
                f"got {min_sample}..{max_sample}"
            )

    signal_float = signals.astype(np.float64) / scale
    fft_values = np.einsum("ni,oi->no", signal_float, make_fft_matrix(window_size))
    real = quantize_signed(fft_values.real, data_width)
    imag = quantize_signed(fft_values.imag, data_width)
    magnitude = np.abs(real) + np.abs(imag)
    return np.clip(magnitude, 0, max_value).astype(np.int32)


@lru_cache(maxsize=None)
def make_fft_matrix(window_size):
    indices = bit_reverse_order(window_size)
    matrix = np.empty((window_size, window_size), dtype=np.complex128)
    for column in range(window_size):
        basis = np.zeros(window_size, dtype=np.float64)
        basis[column] = 1.0
        scrambled = fft_dif_radix2(basis)
        ordered = np.empty_like(scrambled)
        ordered[indices] = scrambled
        matrix[:, column] = ordered / window_size
    return matrix


def fft_dif_radix2(x):
    """
    Computes the 1D Radix-2 Decimation-in-Frequency (DIF) FFT.
    The input 'x' must have a length that is a power of 2.
    Returns the DFT coefficients in bit-reversed order.
    """
    x = np.array(x, dtype=complex)
    N = len(x)

    # Base case of the recursion
    if N == 1:
        return x

    # Verify the input size is a power of 2
    if N % 2 != 0:
        raise ValueError("The length of the input must be a power of 2.")

    half_N = N // 2

    # Calculate twiddle factors: W_N^k = exp(-2j * pi * k / N)
    twiddle = np.exp(-2j * np.pi * np.arange(half_N) / N)

    # DIF Butterfly Operations
    # 1. Top half: x[0] to x[N/2 - 1] -> addition without twiddle factors
    # 2. Bottom half: x[N/2] to x[N - 1] -> subtraction multiplied by twiddle factors
    x_top = x[:half_N] + x[half_N:]
    x_bottom = (x[:half_N] - x[half_N:]) * twiddle

    # Recursive steps on the split frequency halves
    X_even = fft_dif_radix2(x_top)     # Even indices: X[2k]
    X_odd = fft_dif_radix2(x_bottom)   # Odd indices: X[2k+1]

    # Concatenate results into a single array
    return np.concatenate([X_even, X_odd])

def bit_reverse_order(N):
    """Generates an array of indices rearranged in bit-reversed order."""
    indices = np.arange(N)
    bits = int(np.log2(N))
    # Standard bit reversal trick using bitwise operations
    reversed_indices = np.array([int('{:0{width}b}'.format(i, width=bits)[::-1], 2) for i in indices])
    return reversed_indices

# --- Verification Example ---
if __name__ == "__main__":
    # Define an array of size N = 2^3 = 8
    test_input = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]

    # Run the custom DIF FFT algorithm
    dif_output_scrambled = fft_dif_radix2(test_input)

    # Unscramble the bit-reversed output indices to get normal order
    rev_indices = bit_reverse_order(len(test_input))
    dif_output = np.zeros_like(dif_output_scrambled)
    dif_output[rev_indices] = dif_output_scrambled

    # Reference calculation using standard NumPy FFT
    numpy_output = np.fft.fft(test_input)

    print("DIF FFT Output:     ", np.round(dif_output, 2))
    print("NumPy FFT Output:   ", np.round(numpy_output, 2))
    print("Results Match:      ", np.allclose(dif_output, numpy_output))
