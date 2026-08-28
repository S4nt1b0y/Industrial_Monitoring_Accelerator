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


def processar_tres_picos(pico1, pico2, pico3, fs_hz, min_k, n_fft=64):
    """
    Modelo em Python equivalente ao comportamento funcional do mdc_corrigido.v.

    Fluxo:
      1. Rejeita a janela se algum pico for zero.
      2. m = MDC(pico1, pico2)
      3. k = MDC(m, pico3)
      4. Rejeita se k < min_k
      5. k0 = k
      6. f0 = (k0 * fs_hz) // n_fft

    Retorna:
      k0, f0, result_valid
    """

    # Mesma validação do estado CHECK do Verilog
    if pico1 == 0 or pico2 == 0 or pico3 == 0:
        return 0, 0, False

    # Primeiro MDC: m = MDC(pico1, pico2)
    m = mdc(pico1, pico2)

    # Segundo MDC: k = MDC(m, pico3)
    k = mdc(m, pico3)

    # Mesma validação do estado VALIDATE do Verilog
    if k < min_k:
        return 0, 0, False

    # Resultado válido
    k0 = k

    # Em hardware a divisão é inteira
    f0 = (k0 * fs_hz) // n_fft

    return k0, f0, True



# Exemplo


FS_HZ = 6400
MIN_K = 2
N_FFT = 64

k0, f0, result_valid = processar_tres_picos(
    12, 18, 30,
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
        pico1,
        pico2,
        pico3,
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
