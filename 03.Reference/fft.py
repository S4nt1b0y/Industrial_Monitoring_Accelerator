import numpy as np

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