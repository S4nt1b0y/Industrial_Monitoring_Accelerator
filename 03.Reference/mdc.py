import numpy as np


SUPPORTED_DATA_WIDTHS = (8, 16)


def max_feature_value(data_width):
    if data_width not in SUPPORTED_DATA_WIDTHS:
        raise ValueError(f"data_width must be one of {SUPPORTED_DATA_WIDTHS}, got {data_width}")
    return (2 ** (data_width - 1)) - 1


def mdc(a, b):
    """
    Calcula o MDC de dois números usando o mesmo método do hardware:
    algoritmo de Euclides por subtrações sucessivas.
    """
    if a == 0:
        return b

    if b == 0:
        return a

    while a != b:
        if a > b:
            a = a - b
        else:
            b = b - a

    return a


def processar_tres_picos(picos, fs_hz, min_k, n_fft=64):
    """
    Modelo em Python equivalente ao comportamento funcional do mdc_tres_picos.v.

    Fluxo:
      1. Recebe os três picos em paralelo.
      2. Rejeita a janela se algum pico for zero.
      3. m = MDC(pico1, pico2)
      4. k = MDC(m, pico3)
      5. Rejeita se k < min_k
      6. k0 = k
      7. f0 = (k0 * fs_hz) // n_fft

    Retorna:
      k0, f0, result_valid
    """
    picos = np.asarray(picos)
    if picos.shape != (3,):
        raise ValueError(f"picos must have shape (3,), got {picos.shape}")

    pico1, pico2, pico3 = (int(picos[0]), int(picos[1]), int(picos[2]))

    if pico1 == 0 or pico2 == 0 or pico3 == 0:
        return 0, 0, False

    m = mdc(pico1, pico2)
    k = mdc(m, pico3)

    if k < min_k:
        return 0, 0, False

    k0 = k
    f0 = (k0 * fs_hz) // n_fft

    return k0, f0, True


def three_largest_peak_bins(magnitude_bins):
    magnitude_bins = np.asarray(magnitude_bins)
    if magnitude_bins.ndim != 1:
        raise ValueError(f"magnitude_bins must be a 1-D array, got shape {magnitude_bins.shape}")
    if magnitude_bins.shape[0] < 3:
        raise ValueError("at least three FFT bins are required for peak selection")
    peak_indices = np.argsort(-magnitude_bins, kind="stable")[:3]
    return np.sort(peak_indices.astype(np.int16))


def mdc_features_from_magnitude(magnitude_bins, data_width, fs_hz, min_k, n_fft=64):
    """
    Extracts classifier-ready MDC features from one channel FFT magnitude.

    Returns [f0, valid] in the feature range for Q1.7 or Q1.15.
    """
    max_value = max_feature_value(data_width)
    peak_bins = three_largest_peak_bins(magnitude_bins)
    _, f0, result_valid = processar_tres_picos(
        peak_bins,
        fs_hz=fs_hz,
        min_k=min_k,
        n_fft=n_fft,
    )
    return np.asarray(
        [
            np.clip(f0, 0, max_value),
            int(result_valid),
        ],
        dtype=np.int32,
    )


def mdc_features_from_magnitude_batch(magnitude_bins, data_width, fs_hz, min_k, n_fft=64):
    """
    Batch version of mdc_features_from_magnitude.

    magnitude_bins must have shape (window_count, bin_count). The returned array
    has shape (window_count, 2) with [f0, valid].
    """
    max_value = max_feature_value(data_width)
    magnitude_bins = np.asarray(magnitude_bins)
    if magnitude_bins.ndim != 2:
        raise ValueError(f"magnitude_bins must be a 2-D array, got shape {magnitude_bins.shape}")
    if magnitude_bins.shape[1] < 3:
        raise ValueError("at least three FFT bins are required for peak selection")

    peak_bins = np.argsort(-magnitude_bins, axis=1, kind="stable")[:, :3]
    peak_bins = np.sort(peak_bins.astype(np.int32), axis=1)
    k = np.gcd(np.gcd(peak_bins[:, 0], peak_bins[:, 1]), peak_bins[:, 2])
    valid = (
        (peak_bins[:, 0] != 0)
        & (peak_bins[:, 1] != 0)
        & (peak_bins[:, 2] != 0)
        & (k >= min_k)
    )
    f0 = (k.astype(np.int64) * int(fs_hz)) // int(n_fft)
    f0 = np.where(valid, np.clip(f0, 0, max_value), 0)
    return np.column_stack([f0, valid.astype(np.int32)]).astype(np.int32)



if __name__ == "__main__":
    # Exemplo
    FS_HZ = 6400
    MIN_K = 2
    N_FFT = 64

    k0, f0, result_valid = processar_tres_picos(
        [12, 18, 30],
        fs_hz=FS_HZ,
        min_k=MIN_K,
        n_fft=N_FFT
    )

    print("Exemplo:")
    print("k0 =", k0)
    print("f0 =", f0, "Hz")
    print("result_valid =", result_valid)
    print("--------------------------")

    # Casos de teste
    # Formato:
    # (pico1, pico2, pico3, k0_esperado, f0_esperado, valid_esperado)
    testes = [
        (12, 18, 30,  6,  600, True),
        (8,  16, 24,  8,  800, True),
        (5,  10, 20,  5,  500, True),
        (0,  18, 30,  0,    0, False),
        (12, 12, 12, 12, 1200, True),
        (12, 17, 31,  0,    0, False),
    ]

    for pico1, pico2, pico3, k0_esperado, f0_esperado, valid_esperado in testes:
        k0_obtido, f0_obtido, valid_obtido = processar_tres_picos(
            [pico1, pico2, pico3],
            fs_hz=FS_HZ,
            min_k=MIN_K,
            n_fft=N_FFT
        )

        print("Entradas:", pico1, pico2, pico3)

        print(
            "Esperado:",
            "k0 =", k0_esperado,
            "| f0 =", f0_esperado,
            "| valid =", valid_esperado
        )

        print(
            "Obtido:  ",
            "k0 =", k0_obtido,
            "| f0 =", f0_obtido,
            "| valid =", valid_obtido
        )

        if (
            k0_obtido == k0_esperado
            and f0_obtido == f0_esperado
            and valid_obtido == valid_esperado
        ):
            print("Teste aprovado!")
        else:
            print("Teste reprovado!")

        print("--------------------------")
